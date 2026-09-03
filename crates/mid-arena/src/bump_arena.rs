// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-arena.md, section "bump_arena.rs"
// ============================================================================
//! Single-typed, chunk-linked bump allocator: [`BumpArena<T>`] allocates
//! `T` values by bumping a pointer through a growing chain of regions,
//! never freeing individual items, only the whole arena at once.
//!
//! Behind the `bump` feature. Real survey grounding, not assumed:
//! `docs/mid-arena.md`'s Rust benchmarks show `typed-arena`/`bumpalo`
//! winning insert by 2 to 12 times over every generation-checked
//! approach in this crate. This module follows `typed-arena`'s
//! single-typed shape specifically, not `bumpalo`'s mixed-type one,
//! because a single-typed arena can run `Drop` on every value it holds
//! when the arena itself drops, which `docs/mid-arena.md`'s own survey
//! marks as `bumpalo`'s one real tradeoff against `typed-arena`.
//!
//! # Second version, after the first one measured 3.2x slower than
//! bumpalo on real CI (see docs/mid-arena.md "Fixes and Problems" for
//! the actual numbers)
//!
//! The first version stored regions in `RefCell<Vec<Region<T>>>`.
//! Cloning and reading `bumpalo`'s and `slab`'s actual current source
//! (`fitzgen/bumpalo`, `tokio-rs/slab`, both at the versions this crate
//! benchmarks against) rather than continuing to guess turned up the
//! real reason: `bumpalo::Bump` holds a single
//! `Cell<NonNull<ChunkFooter>>` pointing directly at the current
//! chunk, an intrusive linked list where the bump pointer lives inside
//! the chunk itself. No `Vec` of chunks, no `RefCell`. The first
//! version's `RefCell<Vec<Region<T>>>` paid for three real things
//! `bumpalo` doesn't: a `RefCell` borrow check on every `alloc` call, a
//! `Vec` index to find the current region, and doing that lookup twice
//! per call (once before the growth check, once after). This version
//! replaces that with the same shape `bumpalo` actually uses: a
//! `Cell<NonNull<RegionNode<T>>>` intrusive linked list, each region
//! individually heap-allocated via `Box::into_raw` and linked backward
//! through its own `prev` pointer, walked and freed the same direction
//! `bumpalo::dealloc_chunks_until_stop` does in its own real `Drop`
//! impl (checked directly, not assumed to match).
//!
//! # Region growth, following tsoding/arena.h's real source
//!
//! Regions grow geometrically (each new region at least double the
//! previous one's capacity). Unlike `apr_pools.c`'s sorted-by-free-space
//! node list (`docs/mid-arena.md`'s C survey), regions here never get
//! revisited once bumped past. Simpler, and matches every Rust crate
//! surveyed (`typed-arena`, `bumpalo`) doing the same thing.
//!
//! # Verification note
//!
//! This is genuinely riskier unsafe code than this crate's other
//! modules: a hand-rolled intrusive linked list over raw pointers, not
//! just a union or a `Vec<MaybeUninit<T>>` bump within one buffer. No
//! Miri or AddressSanitizer in this sandbox (no rustup/nightly
//! component, same limitation noted in `mid-alloc`'s
//! `stack_allocator.rs` and this crate's `compact_slot_arena.rs`), so
//! this is checked by hand against `bumpalo`'s real, shipped structure
//! plus the tests below, not by a tool built for exactly this kind of
//! code. Said plainly, not left implied.

use alloc::boxed::Box;
use alloc::vec::Vec;
use core::cell::Cell;
use core::marker::PhantomData;
use core::mem::MaybeUninit;
use core::ptr::NonNull;

struct RegionNode<T> {
    data: Vec<MaybeUninit<T>>,
    len: Cell<usize>,
    prev: Option<NonNull<RegionNode<T>>>,
}

impl<T> RegionNode<T> {
    fn new_boxed(capacity: usize, prev: Option<NonNull<RegionNode<T>>>) -> Box<Self> {
        let mut data = Vec::with_capacity(capacity);
        // SAFETY: MaybeUninit<T> has no validity invariant, so treating
        // `capacity` freshly allocated, uninitialized slots as that many
        // MaybeUninit<T> elements is sound for any T. Standard
        // pre-Box::new_uninit_slice idiom (that method needs rustc
        // ~1.82, past this project's rustc 1.75 floor).
        unsafe {
            data.set_len(capacity);
        }
        Box::new(Self {
            data,
            len: Cell::new(0),
            prev,
        })
    }

    #[inline]
    fn capacity(&self) -> usize {
        self.data.len()
    }

    #[inline]
    fn remaining(&self) -> usize {
        self.capacity() - self.len.get()
    }

    /// Bump-allocates one slot, writes `value` into it, returns a
    /// `&mut T` borrowing from `self`, not from a `&mut self` call.
    /// Caller's job to check [`remaining`](Self::remaining) first --
    /// this returns `None` on a full region rather than panicking, but
    /// callers in this module never actually hit that path (they check
    /// first), same division of responsibility `bumpalo`'s own
    /// `try_alloc_layout_fast` has against its caller.
    fn alloc(&self, value: T) -> Option<&mut T> {
        let i = self.len.get();
        if i >= self.data.len() {
            return None;
        }
        self.len.set(i + 1);
        // SAFETY: index `i` was exclusively reserved by the `len.set`
        // above before this call returns -- every other call only ever
        // writes to whatever index `len` holds at its own call, and
        // `len` only advances, so no two calls can target the same
        // slot. `self.data.as_ptr()` yields `*const MaybeUninit<T>`
        // even though the buffer is genuinely being mutated here -- the
        // same interior-mutability cast `Cell<T>` performs internally,
        // needed because `alloc` takes `&self`, not `&mut self`.
        unsafe {
            let slot = self.data.as_ptr().add(i) as *mut MaybeUninit<T>;
            (*slot).write(value);
            Some((*slot).assume_init_mut())
        }
    }

    fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> {
        let len = self.len.get();
        self.data[..len].iter_mut().map(|slot| {
            // SAFETY: every slot below `len` was initialized by `alloc`
            // and never touched again outside that one write.
            unsafe { slot.assume_init_mut() }
        })
    }
}

impl<T> Drop for RegionNode<T> {
    fn drop(&mut self) {
        let len = self.len.get();
        // MaybeUninit<T>'s own Drop is a no-op by design -- without
        // this loop, every value this region ever allocated would leak
        // silently instead of running its destructor.
        for slot in &mut self.data[..len] {
            // SAFETY: every slot below `len` is a real, live,
            // not-yet-dropped `T` (same reasoning as `iter_mut` above).
            unsafe {
                slot.assume_init_drop();
            }
        }
    }
}

/// Single-typed, chunk-linked bump allocator. See this module's doc
/// comment for the full design and why it's shaped the way it is.
pub struct BumpArena<T> {
    current: Cell<NonNull<RegionNode<T>>>,
    // Asserts ownership of `T` for drop-check purposes. `current` is a
    // raw pointer (`NonNull`), which on its own does not tell the
    // compiler this type effectively owns and drops `T` values --
    // `Vec<MaybeUninit<T>>` inside `RegionNode` deliberately opts out
    // of that relationship too (that is the whole point of
    // `MaybeUninit`). Without this marker, dropck could accept code
    // that is not actually sound. Standard fix for a hand-rolled
    // container built on raw pointers and `MaybeUninit`, not specific
    // to this module.
    _marker: PhantomData<T>,
}

const DEFAULT_FIRST_REGION_CAPACITY: usize = 32;

impl<T> BumpArena<T> {
    /// Creates an arena whose first region holds a small default
    /// number of elements, growing geometrically from there. Use
    /// [`with_capacity`](Self::with_capacity) when the expected element
    /// count is known up front.
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_FIRST_REGION_CAPACITY)
    }

    /// Creates an arena whose first region holds at least `capacity`
    /// elements.
    pub fn with_capacity(capacity: usize) -> Self {
        let capacity = capacity.max(1);
        let first = RegionNode::new_boxed(capacity, None);
        let ptr = Box::into_raw(first);
        Self {
            // SAFETY: Box::into_raw never returns a null pointer.
            current: Cell::new(unsafe { NonNull::new_unchecked(ptr) }),
            _marker: PhantomData,
        }
    }

    /// Allocates `value`, returning a `&mut T` borrowing from `self`,
    /// not from a `&mut self` call -- see this module's doc comment for
    /// why that matters. Never fails; grows the region chain instead.
    pub fn alloc(&self, value: T) -> &mut T {
        // SAFETY: `current` always points at a region allocated by
        // `with_capacity` or `grow` below via `Box::into_raw`, and
        // never freed until this arena's own `Drop` runs (see that impl
        // below) -- so it is always valid to read here.
        let node = unsafe { self.current.get().as_ref() };

        if node.remaining() > 0 {
            // Common case: no growth needed. Returning directly here
            // (instead of falling through to a second `current.get()`
            // read below) matters for real, measured reasons, not just
            // tidiness -- an earlier version of this function always
            // re-read `current` after the growth check even when
            // nothing had changed, and that redundant read was part of
            // a real, measured 3.2x gap against `bumpalo` on real CI
            // (see this module's doc comment, "Second version").
            return node
                .alloc(value)
                .expect("just checked remaining() > 0 above");
        }

        self.grow(node.capacity().saturating_mul(2));

        // SAFETY: `grow` just set `current` to a freshly allocated,
        // empty region, so re-reading it here (only on the cold growth
        // path now, not on every call) is valid and guaranteed to have
        // room.
        let node = unsafe { self.current.get().as_ref() };
        node.alloc(value)
            .expect("a freshly grown region must have room for one more allocation")
    }

    fn grow(&self, next_capacity: usize) {
        let new_node = RegionNode::new_boxed(next_capacity.max(1), Some(self.current.get()));
        let ptr = Box::into_raw(new_node);
        // SAFETY: Box::into_raw never returns a null pointer.
        self.current.set(unsafe { NonNull::new_unchecked(ptr) });
    }

    /// Total number of regions currently in the chain. Mostly useful
    /// for tests and diagnostics.
    pub fn region_count(&self) -> usize {
        let mut count = 0usize;
        let mut cursor = Some(self.current.get());
        while let Some(ptr) = cursor {
            count += 1;
            // SAFETY: every pointer in this chain was allocated by
            // `with_capacity`/`grow` and stays valid until `Drop`
            // (same invariant `alloc` relies on above).
            cursor = unsafe { ptr.as_ref() }.prev;
        }
        count
    }

    /// Total elements allocated across every region.
    pub fn len(&self) -> usize {
        let mut total = 0usize;
        let mut cursor = Some(self.current.get());
        while let Some(ptr) = cursor {
            // SAFETY: same as `region_count` above.
            let node = unsafe { ptr.as_ref() };
            total += node.len.get();
            cursor = node.prev;
        }
        total
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Iterates every allocated value across every region, oldest
    /// region first. Takes `&mut self`, matching
    /// `typed_arena::Arena::iter_mut`'s own real constraint (checked
    /// directly while benchmarking it, not assumed) -- exclusive access
    /// to the arena rules out any outstanding `&mut T` from an earlier
    /// `alloc` call still being alive.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> {
        // Collect region pointers oldest-first, then hand out iterators
        // over each in that order.
        let mut ptrs = Vec::new();
        let mut cursor = Some(self.current.get());
        while let Some(ptr) = cursor {
            ptrs.push(ptr);
            // SAFETY: same invariant as `region_count`/`len` above.
            cursor = unsafe { ptr.as_ref() }.prev;
        }
        ptrs.reverse();

        ptrs.into_iter().flat_map(|mut ptr| {
            // SAFETY: `&mut self` on this method means no other
            // reference to this arena (or anything borrowed from it)
            // can be alive right now, so taking `&mut` through each
            // region pointer here is exclusive, matching the method's
            // own contract.
            unsafe { ptr.as_mut() }.iter_mut()
        })
    }
}

impl<T> Drop for BumpArena<T> {
    fn drop(&mut self) {
        // Walk backward through `prev` and free each region, same
        // direction `bumpalo::dealloc_chunks_until_stop` frees its own
        // chunk list (checked directly against that real function, not
        // assumed to match).
        let mut cursor = Some(self.current.get());
        while let Some(ptr) = cursor {
            // SAFETY: `ptr` came from a `Box::into_raw` call in
            // `with_capacity` or `grow`, and this loop is the only place
            // that ever calls `Box::from_raw` on a pointer from this
            // arena's chain -- each region is freed exactly once, right
            // here, walking the same links `alloc`/`len`/`region_count`
            // only ever read.
            let boxed = unsafe { Box::from_raw(ptr.as_ptr()) };
            cursor = boxed.prev;
            // `boxed` drops here: `RegionNode<T>`'s own `Drop` impl runs
            // first (dropping every live value in this region), then
            // the region's own heap allocation is freed.
        }
    }
}

impl<T> Default for BumpArena<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn starts_with_one_region_and_no_elements() {
        let a: BumpArena<u32> = BumpArena::new();
        assert_eq!(a.region_count(), 1);
        assert_eq!(a.len(), 0);
        assert!(a.is_empty());
    }

    #[test]
    fn alloc_stores_and_reads_back_the_value() {
        let a = BumpArena::new();
        let x = a.alloc(42u32);
        assert_eq!(*x, 42);
        *x = 43;
        assert_eq!(*x, 43);
    }

    #[test]
    fn multiple_simultaneous_allocations_stay_independent() {
        let a = BumpArena::new();
        let x = a.alloc(1u32);
        let y = a.alloc(2u32);
        let z = a.alloc(3u32);
        *x += 10;
        *y += 20;
        *z += 30;
        assert_eq!((*x, *y, *z), (11, 22, 33));
    }

    #[test]
    fn growing_past_the_first_region_adds_a_new_one() {
        let a = BumpArena::with_capacity(4);
        assert_eq!(a.region_count(), 1);
        for i in 0..4u32 {
            a.alloc(i);
        }
        assert_eq!(
            a.region_count(),
            1,
            "first region should still have exactly enough room"
        );
        a.alloc(99u32);
        assert_eq!(
            a.region_count(),
            2,
            "the 5th allocation should have grown a new region"
        );
        assert_eq!(a.len(), 5);
    }

    #[test]
    fn later_regions_hold_at_least_double_the_previous_capacity() {
        let a = BumpArena::with_capacity(2);
        for i in 0..2u32 {
            a.alloc(i);
        }
        a.alloc(2u32); // forces growth
        assert_eq!(a.region_count(), 2);
        for i in 0..3u32 {
            a.alloc(i);
        }
        a.alloc(99u32);
        assert_eq!(a.region_count(), 3);
    }

    #[test]
    fn many_allocations_across_several_regions_all_read_back_correctly() {
        let a = BumpArena::with_capacity(4);
        let mut refs = Vec::new();
        for i in 0..500u32 {
            refs.push(a.alloc(i));
        }
        assert!(
            a.region_count() > 1,
            "500 elements into a capacity-4 first region should have grown several times"
        );
        for (i, r) in refs.iter().enumerate() {
            assert_eq!(**r, i as u32);
        }
    }

    #[test]
    fn iter_mut_visits_every_value_in_allocation_order_and_writes_through() {
        let mut a = BumpArena::with_capacity(2);
        for i in 0..10u32 {
            a.alloc(i);
        }
        let seen: Vec<u32> = a.iter_mut().map(|v| *v).collect();
        assert_eq!(seen, (0..10).collect::<Vec<u32>>());

        for v in a.iter_mut() {
            *v *= 10;
        }
        let doubled: Vec<u32> = a.iter_mut().map(|v| *v).collect();
        assert_eq!(doubled, (0..10).map(|i| i * 10).collect::<Vec<u32>>());
    }

    #[test]
    fn iter_mut_visits_oldest_region_first_across_a_real_growth_boundary() {
        // growing_past_the_first_region_adds_a_new_one already checks
        // region count across this boundary; this checks iteration
        // order specifically, which that test doesn't.
        let mut a = BumpArena::with_capacity(2);
        for i in 0..7u32 {
            a.alloc(i);
        }
        assert!(a.region_count() >= 2);
        let seen: Vec<u32> = a.iter_mut().map(|v| *v).collect();
        assert_eq!(
            seen,
            (0..7).collect::<Vec<u32>>(),
            "must read back in allocation order, oldest region first"
        );
    }

    #[test]
    fn drop_runs_for_every_allocated_value_when_the_arena_itself_is_dropped() {
        use core::cell::Cell as StdCell;

        struct DropCounter<'a>(&'a StdCell<u32>);
        impl<'a> Drop for DropCounter<'a> {
            fn drop(&mut self) {
                self.0.set(self.0.get() + 1);
            }
        }

        let count = StdCell::new(0u32);
        {
            let a = BumpArena::with_capacity(2);
            for _ in 0..10 {
                a.alloc(DropCounter(&count));
            }
            assert_eq!(
                count.get(),
                0,
                "nothing should be dropped while the arena is still alive"
            );
        }
        assert_eq!(
            count.get(),
            10,
            "every value across every region must run Drop when the arena drops"
        );
    }

    #[test]
    fn drop_runs_across_many_regions_not_just_the_first() {
        // The Drop impl above walks `prev` across every region in the
        // chain -- this forces several real growth boundaries so that
        // walk is actually exercised, not just the single-region case
        // the previous test covers.
        use core::cell::Cell as StdCell;

        struct DropCounter<'a>(&'a StdCell<u32>);
        impl<'a> Drop for DropCounter<'a> {
            fn drop(&mut self) {
                self.0.set(self.0.get() + 1);
            }
        }

        let count = StdCell::new(0u32);
        {
            let a = BumpArena::with_capacity(2);
            for _ in 0..200 {
                a.alloc(DropCounter(&count));
            }
            assert!(
                a.region_count() > 4,
                "200 elements from a capacity-2 start should cross several region boundaries"
            );
        }
        assert_eq!(count.get(), 200);
    }

    #[test]
    fn zero_sized_types_do_not_panic_or_loop_forever() {
        let a: BumpArena<()> = BumpArena::with_capacity(4);
        for _ in 0..20 {
            a.alloc(());
        }
        assert_eq!(a.len(), 20);
    }
}
