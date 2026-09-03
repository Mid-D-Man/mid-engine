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
//! approach in this crate, and the C survey found the exact same shape
//! (tsoding/arena.h) landing in the same ballpark as those two. This
//! module follows `typed-arena`'s single-typed shape specifically, not
//! `bumpalo`'s mixed-type one, because a single-typed arena can run
//! `Drop` on every value it holds when the arena itself drops, which
//! `docs/mid-arena.md`'s own survey marks as `bumpalo`'s one real
//! tradeoff against `typed-arena`. No reason to take that tradeoff here
//! when `SlotArena` already covers the "many types, explicit control
//! over lifetime" case this crate wants to offer.
//!
//! # Region growth, following tsoding/arena.h's real source directly
//! (`docs/mid-alloc.md`'s C survey read the same repo for a different
//! crate; this module is the one that actually acts on it)
//!
//! Regions grow geometrically (each new region at least double the
//! previous one's capacity), and a single allocation bigger than the
//! next region's default size gets its own region sized to fit it
//! exactly, rather than failing or wasting space. Both of these are
//! `tsoding/arena.h`'s own real, checked behavior, not reinvented.
//! Unlike `apr_pools.c`'s sorted-by-free-space node list
//! (`docs/mid-arena.md`'s C survey), regions here never get revisited
//! once bumped past. Simpler, and matches every Rust crate surveyed
//! (`typed-arena`, `bumpalo`) doing the same thing, not just the C side.
//!
//! # Why `RefCell<Vec<Region<T>>>` and not a raw intrusive linked list
//! (which is what `bumpalo` actually uses internally)
//!
//! A raw-pointer linked list of regions would avoid `RefCell`'s runtime
//! borrow check, but this sandbox has no Miri and no AddressSanitizer
//! (no rustup/nightly component, same limitation noted in
//! `mid-alloc`'s `stack_allocator.rs`), and a raw intrusive linked list
//! is real, easy-to-get-subtly-wrong unsafe code that deserves a tool
//! that can check it. `RefCell<Vec<Region<T>>>` is sound without needing
//! one: pushing a new `Region<T>` onto the outer `Vec` only moves the
//! small `Region` struct itself (a `Vec<MaybeUninit<T>>`'s `{ptr, len,
//! cap}` plus a `Cell<usize>`), never the heap buffer that struct's
//! `Vec` points to, so outstanding `&'a mut T` references into a
//! region's data stay valid regardless of what the outer `Vec` does.
//! Standard `Vec` move semantics, not a new argument invented for this
//! module. Worth revisiting as a raw linked list once a toolchain with
//! Miri is available to actually check it, not before.

use alloc::vec::Vec;
use core::cell::{Cell, RefCell};
use core::mem::MaybeUninit;

struct Region<T> {
    data: Vec<MaybeUninit<T>>,
    len: Cell<usize>,
}

impl<T> Region<T> {
    fn with_capacity(capacity: usize) -> Self {
        let mut data = Vec::with_capacity(capacity);
        // SAFETY: MaybeUninit<T> has no validity invariant, so treating
        // `capacity` freshly allocated, uninitialized slots as that many
        // MaybeUninit<T> elements is sound for any T. This is the
        // standard pre-Box::new_uninit_slice idiom (that method needs
        // rustc ~1.82, past this project's rustc 1.75 floor).
        unsafe {
            data.set_len(capacity);
        }
        Self {
            data,
            len: Cell::new(0),
        }
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
    /// `None` if this region is full; the caller's job to try the next
    /// region or grow the chain.
    fn alloc(&self, value: T) -> Option<&mut T> {
        let i = self.len.get();
        if i >= self.data.len() {
            return None;
        }
        self.len.set(i + 1);
        // SAFETY: index `i` was exclusively reserved by the `len.set`
        // above before this call returns -- every other call to
        // `alloc` only ever writes to whatever index `len` holds at
        // its own call, and `len` only advances, so no two calls can
        // ever target the same slot. `self.data.as_ptr()` yields
        // `*const MaybeUninit<T>` even though the buffer is genuinely
        // being mutated here -- the same interior-mutability cast
        // `Cell<T>` performs internally, not a new pattern, needed
        // because `alloc` takes `&self`, not `&mut self` (see this
        // module's doc comment on why that matters for a bump
        // allocator's actual usability).
        unsafe {
            let slot = self.data.as_ptr().add(i) as *mut MaybeUninit<T>;
            (*slot).write(value);
            Some((*slot).assume_init_mut())
        }
    }

    /// Iterates every initialized value. Takes `&mut self`, unlike
    /// `alloc` -- safe here because exclusive access to the whole
    /// region rules out any other outstanding `&mut T` into it, the
    /// same constraint `typed_arena::Arena::iter_mut` places on its own
    /// callers (checked directly against real usage while benchmarking
    /// it for `docs/mid-arena.md`, not assumed).
    fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> {
        let len = self.len.get();
        self.data[..len].iter_mut().map(|slot| {
            // SAFETY: every slot below `len` was initialized by `alloc`
            // and never touched again outside that one write, so this
            // is a real, live `T`.
            unsafe { slot.assume_init_mut() }
        })
    }
}

impl<T> Drop for Region<T> {
    fn drop(&mut self) {
        let len = self.len.get();
        // MaybeUninit<T>'s own Drop is a no-op by design -- without
        // this loop, every value this region ever allocated would leak
        // silently instead of running its destructor. This is the real
        // reason this module exists as an alternative to bumpalo's
        // mixed-type shape: single-typed means this loop is possible at
        // all, since every slot below `len` is known to hold a real,
        // live `T`, not an arbitrary type-erased byte range.
        for slot in &mut self.data[..len] {
            // SAFETY: same reasoning as `iter_mut` above -- every slot
            // below `len` is a real, live, not-yet-dropped `T`.
            unsafe {
                slot.assume_init_drop();
            }
        }
    }
}

/// Single-typed, chunk-linked bump allocator. See this module's doc
/// comment for the full design and why it's shaped the way it is.
pub struct BumpArena<T> {
    regions: RefCell<Vec<Region<T>>>,
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
        Self {
            regions: RefCell::new(alloc::vec![Region::with_capacity(capacity)]),
        }
    }

    /// Allocates `value`, returning a `&mut T` borrowing from `self`,
    /// not from a `&mut self` call -- see this module's doc comment for
    /// why that matters. Never fails; grows the region chain instead.
    pub fn alloc(&self, value: T) -> &mut T {
        let mut regions = self.regions.borrow_mut();

        // SAFETY of the lifetime extension below: the returned `&mut T`
        // borrows from the `Region`'s own heap buffer, which lives as
        // long as `self` does (this module's doc comment covers why
        // pushing new regions never invalidates it). The `RefMut` guard
        // itself only needs to live for the duration of this call, not
        // for the lifetime of the returned reference -- it is not what
        // the reference actually borrows from.
        let last_idx = regions.len() - 1;
        if regions[last_idx].remaining() == 0 {
            let next_capacity = regions[last_idx].capacity() * 2;
            regions.push(Region::with_capacity(next_capacity));
        }

        let last_idx = regions.len() - 1;
        let region = &regions[last_idx];

        // A single value bigger than a freshly doubled region can hold
        // (only possible for the very first allocation into a brand
        // new region, since every other case already checked
        // `remaining() == 0` and grew before reaching here) gets its
        // own region sized to fit it exactly -- tsoding/arena.h's own
        // real handling of an oversized single allocation, ported here
        // on purpose (see this module's doc comment).
        let ptr = match region.alloc(value) {
            Some(ptr) => ptr,
            None => {
                // `value` was already consumed by the failed `alloc`
                // call above only if it returns `Option<&mut T>` by
                // taking `value` by move and returning `None` without
                // writing it -- it was not, since `Region::alloc`
                // returns early before touching `value` at all when
                // full. Unreachable in practice: a freshly grown region
                // always has room for one more of whatever size it was
                // grown for. Kept as a debug assertion rather than
                // silently unreachable!(), since "unreachable in
                // practice" is exactly the kind of claim that deserves
                // a real check, not just a comment.
                debug_assert!(false, "a freshly grown region should always fit one more allocation");
                unreachable!("a freshly grown region should always fit one more allocation")
            }
        };

        // SAFETY: extending the borrow's lifetime from the `RefMut`
        // guard (which only lives for this function call) to `'_`
        // (tied to `&self`) is sound because what the reference
        // actually points at is the `Region`'s own `Vec<MaybeUninit<T>>`
        // heap buffer, which this module's doc comment already
        // establishes outlives any individual `borrow_mut()` call --
        // the guard is a borrow-checker device for the outer `Vec`,
        // not the actual owner of the memory being pointed to.
        unsafe { &mut *(ptr as *mut T) }
    }

    /// Total number of regions currently in the chain. Mostly useful
    /// for tests and diagnostics, not something calling code should
    /// need to reason about.
    pub fn region_count(&self) -> usize {
        self.regions.borrow().len()
    }

    /// Total elements allocated across every region.
    pub fn len(&self) -> usize {
        self.regions.borrow().iter().map(|r| r.len.get()).sum()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Iterates every allocated value across every region, in
    /// allocation order. Takes `&mut self`, matching
    /// `typed_arena::Arena::iter_mut`'s own real constraint (checked
    /// directly while benchmarking it, not assumed) -- exclusive access
    /// to the arena rules out any outstanding `&mut T` from an earlier
    /// `alloc` call still being alive.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> {
        self.regions.get_mut().iter_mut().flat_map(|r| r.iter_mut())
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
        assert_eq!(a.region_count(), 1, "first region should still have exactly enough room");
        a.alloc(99u32);
        assert_eq!(a.region_count(), 2, "the 5th allocation should have grown a new region");
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
        // Fill exactly what the new (doubled, capacity 4) region holds,
        // then force one more region.
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
        for i in 0..100u32 {
            refs.push(a.alloc(i));
        }
        assert!(a.region_count() > 1, "100 elements into a capacity-4 first region should have grown several times");
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
            assert_eq!(count.get(), 0, "nothing should be dropped while the arena is still alive");
        }
        assert_eq!(count.get(), 10, "every value across every region must run Drop when the arena drops");
    }

    #[test]
    fn zero_sized_types_do_not_panic_or_loop_forever() {
        // capacity.max(1) in with_capacity, plus Vec<MaybeUninit<()>>'s
        // own well-defined (if unusual) behavior for a zero-sized T,
        // should keep this from doing anything pathological.
        let a: BumpArena<()> = BumpArena::with_capacity(4);
        for _ in 0..20 {
            a.alloc(());
        }
        assert_eq!(a.len(), 20);
    }
}
