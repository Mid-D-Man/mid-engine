// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-alloc.md, section "pool_allocator.rs"
// ============================================================================
//! Fixed-node-size, free-list allocator: [`PoolAllocator<T>`] hands out
//! and reclaims individual `T` slots in O(1), reusing freed slots
//! before growing. Modeled on foonathan/memory's `memory_pool` (real
//! source read, `include/foonathan/memory/memory_pool.hpp`:
//! `allocate_node()` pops the free list, or grows by one whole block
//! sized by `next_capacity()` and retries) for the grow-in-batches
//! mechanic, but follows Zig's `std.heap.MemoryPool` (real source
//! read, `lib/std/heap/memory_pool.zig`) for the public shape: a typed
//! `create()`/`destroy()` pair, not a generic byte-oriented allocator
//! interface.
//!
//! # Why typed `create`/`destroy`, not a generic allocator trait --
//! Zig's real choice, not invented here
//!
//! `foonathan::memory_pool` sits behind a `void* allocate_node()` and
//! a separate `allocator_traits<memory_pool<...>>` specialization that
//! adapts that to the generic interface every other `foonathan/memory`
//! type shares. `std.heap.MemoryPool` deliberately does not implement
//! Zig's own `Allocator` vtable at all: a fixed-single-type pool
//! already knows its item's size and alignment at compile time, so
//! routing every call through a byte-oriented, runtime-checked
//! interface throws that away for nothing. This module follows Zig's
//! call here: [`create`](PoolAllocator::create)/
//! [`destroy`](PoolAllocator::destroy) work directly in `T`, matching
//! this crate's own doc's stance on `stack_allocator`
//! (`docs/mid-alloc.md`: "mid-alloc's allocators hand out raw/typed
//! memory with no handle indirection at all").
//!
//! # Growth: an owned, chunk-linked region chain, not a raw block list
//!
//! `foonathan::memory_pool` grows by asking its `BlockOrRawAllocator`
//! for one new block sized `next_capacity()` and slicing it into fresh
//! free-list nodes. `std.heap.MemoryPool` grows the same batched way
//! but pulls that new memory from an *owned* `std.heap.ArenaAllocator`
//! rather than a raw block list it manages itself. This module follows
//! Zig's shape for that -- storage grows from an owned region chain,
//! not a separate block-tracking structure -- but the region chain
//! itself reuses `mid-arena`'s own `BumpArena` pattern
//! (`docs/mid-arena.md`: `Cell<NonNull<RegionNode<T>>>`, geometric
//! growth, already verified on real CI), reimplemented locally here
//! rather than taken as a dependency: `mid-alloc` stays at zero
//! mandatory dependencies, including on its sibling crate
//! (`docs/mid-alloc.md` "Relationship to mid-arena" already flags this
//! as a plausible future point of contact, not a present one).
//!
//! # The union trick: `mid-arena`'s own `CompactSlotArena`, not a new
//! invention
//!
//! A freed slot needs to hold a "next free" pointer instead of a `T`
//! until it's reused. Same problem `CompactSlotArena` solves
//! (`docs/mid-arena.md`: `union SlotUnion<T> { value: ManuallyDrop<T>,
//! next_free: u32 }`, ported from `slotmap`'s real layout) -- this
//! module reuses that exact shape, swapping `next_free: u32` for
//! `next: Option<NonNull<Slot<T>>>` since there's no generation index
//! here to piggyback on. `Vec<MaybeUninit<Slot<T>>>` storage (not a
//! raw byte buffer, unlike `stack_allocator.rs`) means the compiler
//! computes `Slot<T>`'s size and alignment for us -- no manual
//! alignment arithmetic needed here at all.
//!
//! # What this does NOT do: no per-item `Drop` on pool drop, matching
//! Zig's real behavior and this crate's own `StackAllocator` tradeoff,
//! not a new one
//!
//! [`destroy`](PoolAllocator::destroy) runs `T`'s destructor for that
//! one item -- has to, or every explicit destroy would leak. But an
//! item `create`d and never `destroy`ed before the pool itself drops
//! is leaked, not dropped: `std.heap.MemoryPool.deinit()` just frees
//! the arena's raw memory (Zig has no destructors to run in the first
//! place), and tracking which of this pool's non-contiguous slots are
//! still live -- unlike `BumpArena`'s simple "everything below `len`"
//! -- would mean a live/dead flag per slot, real memory and branching
//! cost a fixed-size pool exists specifically to avoid. Same tradeoff
//! `stack_allocator.rs` already makes for the same reason, said
//! plainly here too.
//!
//! # Growable or fixed-capacity, following Zig's real `Options.growable`
//!
//! [`new`](PoolAllocator::new)/[`with_capacity`](PoolAllocator::with_capacity)
//! grow the region chain forever, same default `std.heap.MemoryPool`
//! has. [`with_capacity_bounded`](PoolAllocator::with_capacity_bounded)
//! matches Zig's `Options{ .growable = false }` and this crate's own
//! `StackAllocator` precedent: a single fixed region,
//! [`create`](PoolAllocator::create) returns `Err(value)` once it and
//! the free list are both exhausted, rather than growing.
//!
//! # Verification note
//!
//! Same limitation as every other `unsafe`-carrying module in
//! `mid-arena`/`mid-alloc`: this sandbox's rustc 1.75 has no
//! rustup/nightly component, so no Miri or AddressSanitizer were
//! available. Checked by hand against `CompactSlotArena`'s and
//! `BumpArena`'s already-shipped, already-tested patterns -- this
//! module doesn't introduce a new unsafe shape, it recombines two that
//! are already proven in this workspace -- plus the tests below, and
//! `cargo test -p mid-alloc --features pool` was actually run on this
//! sandbox's rustc 1.75 (see docs/mid-alloc.md for the real result).
//! Worth a real Miri pass on a toolchain that has it before this ships
//! anywhere that isn't itself still under active development.

use alloc::boxed::Box;
use alloc::vec::Vec;
use core::cell::Cell;
use core::marker::PhantomData;
use core::mem::{ManuallyDrop, MaybeUninit};
use core::ptr::NonNull;

/// One pool slot: either a live `T` or a link to the next free slot.
/// Same shape as `mid-arena`'s `CompactSlotArena::SlotUnion<T>` -- see
/// this module's doc comment.
union Slot<T> {
    value: ManuallyDrop<T>,
    next: Option<NonNull<Slot<T>>>,
}

/// One chunk in the pool's owned region chain. Deliberately carries no
/// `Drop` impl of its own -- see this module's doc comment on why
/// pool slots don't get per-item drop tracking; letting
/// `Vec<MaybeUninit<Slot<T>>>` drop its raw storage with no per-element
/// destructor calls is exactly the behavior that tradeoff wants.
struct PoolRegion<T> {
    slots: Vec<MaybeUninit<Slot<T>>>,
    bumped: Cell<usize>,
    prev: Option<NonNull<PoolRegion<T>>>,
}

impl<T> PoolRegion<T> {
    fn new_boxed(capacity: usize, prev: Option<NonNull<PoolRegion<T>>>) -> Box<Self> {
        let mut slots = Vec::with_capacity(capacity);
        // SAFETY: MaybeUninit<Slot<T>> has no validity invariant, so
        // treating `capacity` freshly allocated, uninitialized slots
        // as that many MaybeUninit<Slot<T>> elements is sound for any
        // T -- same reasoning `BumpArena`'s `RegionNode::new_boxed`
        // uses for `MaybeUninit<T>` in `mid-arena`.
        unsafe {
            slots.set_len(capacity);
        }
        Box::new(Self {
            slots,
            bumped: Cell::new(0),
            prev,
        })
    }

    #[inline]
    fn capacity(&self) -> usize {
        self.slots.len()
    }

    /// Bump-allocates one never-yet-touched slot, returning a pointer
    /// to it uninitialized, or `None` if this region is full -- the
    /// caller (`PoolAllocator::bump_fresh_slot`) is the only place
    /// that calls this, and grows the region chain on `None` rather
    /// than needing a separate capacity check beforehand.
    fn bump(&self) -> Option<NonNull<Slot<T>>> {
        let i = self.bumped.get();
        if i >= self.slots.len() {
            return None;
        }
        self.bumped.set(i + 1);
        // SAFETY: index `i` was exclusively reserved by the
        // `bumped.set` above before this call returns -- same
        // reasoning `BumpArena`'s `RegionNode::alloc` uses.
        let ptr = unsafe { self.slots.as_ptr().add(i) } as *mut Slot<T>;
        Some(unsafe { NonNull::new_unchecked(ptr) })
    }
}

const DEFAULT_FIRST_REGION_CAPACITY: usize = 32;

/// Fixed-node-size, free-list allocator. See this module's doc comment
/// for the full design.
pub struct PoolAllocator<T> {
    current: Cell<NonNull<PoolRegion<T>>>,
    free_list: Cell<Option<NonNull<Slot<T>>>>,
    growable: bool,
    live_count: Cell<usize>,
    _marker: PhantomData<T>,
}

impl<T> PoolAllocator<T> {
    /// Creates a pool whose first region holds a small default number
    /// of slots, growing the region chain geometrically forever after
    /// that. Use [`with_capacity`](Self::with_capacity) when the
    /// expected slot count is known up front, or
    /// [`with_capacity_bounded`](Self::with_capacity_bounded) for a
    /// hard cap that never grows.
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_FIRST_REGION_CAPACITY)
    }

    /// Creates a pool whose first region holds at least `capacity`
    /// slots, growing the region chain geometrically forever after
    /// that.
    pub fn with_capacity(capacity: usize) -> Self {
        Self::new_with(capacity, true)
    }

    /// Creates a pool with exactly `capacity` slots that never grows:
    /// [`create`](Self::create) returns `Err(value)` once `capacity`
    /// slots are live and the free list is empty, matching Zig's
    /// `Options{ .growable = false }` -- see this module's doc
    /// comment.
    pub fn with_capacity_bounded(capacity: usize) -> Self {
        Self::new_with(capacity, false)
    }

    fn new_with(capacity: usize, growable: bool) -> Self {
        let capacity = capacity.max(1);
        let first = PoolRegion::new_boxed(capacity, None);
        let ptr = Box::into_raw(first);
        Self {
            // SAFETY: Box::into_raw never returns a null pointer.
            current: Cell::new(unsafe { NonNull::new_unchecked(ptr) }),
            free_list: Cell::new(None),
            growable,
            live_count: Cell::new(0),
            _marker: PhantomData,
        }
    }

    /// Number of slots currently live (created and not yet destroyed).
    #[inline]
    pub fn len(&self) -> usize {
        self.live_count.get()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.live_count.get() == 0
    }

    /// Whether this pool grows past its initial capacity. Set at
    /// construction -- see
    /// [`with_capacity_bounded`](Self::with_capacity_bounded).
    #[inline]
    pub fn is_growable(&self) -> bool {
        self.growable
    }

    /// Allocates one slot, moves `value` into it, and returns a `&mut
    /// T` borrowing from `self`, not from a `&mut self` call -- same
    /// `Cell`-based reasoning as `StackAllocator`/`BumpArena`, so
    /// multiple simultaneously-live allocations from the same pool
    /// stay possible. Reuses a freed slot if the free list has one;
    /// otherwise bump-allocates a fresh slot from the current region,
    /// growing the region chain first if that region is full and this
    /// pool [`is_growable`](Self::is_growable). Returns `value` back,
    /// unwritten, if a fixed-capacity pool is full and the free list
    /// is empty.
    pub fn create(&self, value: T) -> Result<&mut T, T> {
        let slot_ptr = match self.free_list.get() {
            Some(slot_ptr) => {
                // SAFETY: every pointer ever pushed onto `free_list`
                // came from `destroy` below, which just finished
                // writing a real `next` value into this exact slot's
                // union before linking it in -- `next` is this slot's
                // live field right now.
                let next = unsafe { (*slot_ptr.as_ptr()).next };
                self.free_list.set(next);
                slot_ptr
            }
            None => match self.bump_fresh_slot() {
                Some(slot_ptr) => slot_ptr,
                None => return Err(value),
            },
        };

        // SAFETY: `slot_ptr` is either a freed slot just unlinked
        // above (no live `T` in it right now -- `destroy` already ran
        // its destructor before linking it in) or a freshly bumped,
        // never-touched slot. Either way, writing a new `T` into
        // `value` here is a real initialization, not a store into a
        // possibly-live `T`.
        unsafe {
            (*slot_ptr.as_ptr()).value = ManuallyDrop::new(value);
        }
        self.live_count.set(self.live_count.get() + 1);
        let typed: NonNull<T> = slot_ptr.cast();
        // SAFETY: `typed` points at the union field just written
        // above, which shares `T`'s layout exactly
        // (`ManuallyDrop<T>` is `#[repr(transparent)]`) and is
        // exclusively ours until the next `destroy` call on this same
        // slot -- nothing else in this module hands out a reference
        // into it in between.
        Ok(unsafe { &mut *typed.as_ptr() })
    }

    fn bump_fresh_slot(&self) -> Option<NonNull<Slot<T>>> {
        // SAFETY: `current` always points at a region allocated by
        // `new_with` or `grow` below via `Box::into_raw`, never freed
        // until this pool's own `Drop` runs -- same reasoning
        // `BumpArena::alloc` uses for its own `current`.
        let region = unsafe { self.current.get().as_ref() };
        if let Some(slot_ptr) = region.bump() {
            return Some(slot_ptr);
        }
        if !self.growable {
            return None;
        }
        self.grow();
        // SAFETY: same as above -- `current` now points at the region
        // `grow` just linked in, which starts empty and therefore has
        // room for at least one slot.
        let grown = unsafe { self.current.get().as_ref() };
        Some(
            grown
                .bump()
                .expect("a freshly grown region must have room for at least one slot"),
        )
    }

    /// Links a new region in front of `current`, at least double the
    /// previous region's capacity -- same geometric growth
    /// `BumpArena`'s region chain already uses.
    fn grow(&self) {
        // SAFETY: same reasoning as `bump_fresh_slot` above.
        let old_capacity = unsafe { self.current.get().as_ref() }.capacity();
        let next = PoolRegion::new_boxed(old_capacity * 2, Some(self.current.get()));
        let ptr = Box::into_raw(next);
        // SAFETY: Box::into_raw never returns a null pointer.
        self.current.set(unsafe { NonNull::new_unchecked(ptr) });
    }

    /// Runs `T`'s destructor for `item` and returns its slot to the
    /// free list for reuse by a later [`create`](Self::create).
    ///
    /// # Safety
    ///
    /// `item` must be a reference this exact pool's `create` returned,
    /// not already passed to `destroy`, and not used again after this
    /// call -- using it afterward, or destroying it twice, is
    /// undefined behavior. Same class of contract
    /// [`StackMarker`](crate::StackMarker)'s doc comment already
    /// documents for `rewind`.
    pub unsafe fn destroy(&self, item: &mut T) {
        let slot_ptr: NonNull<Slot<T>> = NonNull::from(item).cast();
        // SAFETY: caller guarantees `item` came from this pool's
        // `create` and hasn't been destroyed yet, so `value` is this
        // slot's live field right now.
        unsafe {
            ManuallyDrop::drop(&mut (*slot_ptr.as_ptr()).value);
            (*slot_ptr.as_ptr()).next = self.free_list.get();
        }
        self.free_list.set(Some(slot_ptr));
        self.live_count.set(self.live_count.get() - 1);
    }
}

impl<T> Drop for PoolAllocator<T> {
    fn drop(&mut self) {
        // Frees every region's raw storage. Deliberately does not run
        // `T`'s destructor for slots that are still live (created and
        // never destroyed) -- see this module's doc comment for why
        // that's the accepted tradeoff here, matching
        // `std.heap.MemoryPool.deinit()`'s real behavior and this
        // crate's own `StackAllocator` precedent. `Vec<MaybeUninit<Slot<T>>>`
        // dropping is a no-op either way (`MaybeUninit` has no drop
        // glue), so nothing extra is needed to make that true --
        // unlike `BumpArena::RegionNode`, which has to drop its own
        // `T` range explicitly because it *does* run per-item `Drop`.
        let mut cursor = Some(self.current.get());
        while let Some(ptr) = cursor {
            // SAFETY: `ptr` came from a `Box::into_raw` call in
            // `new_with` or `grow`, and this loop is the only place
            // that ever calls `Box::from_raw` on a pointer from this
            // pool's region chain -- same reasoning `BumpArena::drop`
            // uses for its own chain.
            let boxed = unsafe { Box::from_raw(ptr.as_ptr()) };
            cursor = boxed.prev;
        }
    }
}

impl<T> Default for PoolAllocator<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn starts_empty() {
        let p: PoolAllocator<u32> = PoolAllocator::new();
        assert_eq!(p.len(), 0);
        assert!(p.is_empty());
        assert!(p.is_growable());
    }

    #[test]
    fn create_stores_and_reads_back_the_value() {
        let p = PoolAllocator::new();
        let x = p.create(42u32).unwrap();
        assert_eq!(*x, 42);
        *x = 43;
        assert_eq!(*x, 43);
        assert_eq!(p.len(), 1);
    }

    #[test]
    fn multiple_simultaneous_allocations_stay_independent() {
        let p = PoolAllocator::new();
        let x = p.create(1u32).unwrap();
        let y = p.create(2u32).unwrap();
        let z = p.create(3u32).unwrap();
        *x += 10;
        *y += 20;
        *z += 30;
        assert_eq!((*x, *y, *z), (11, 22, 33));
        assert_eq!(p.len(), 3);
    }

    #[test]
    fn destroy_runs_drop_and_frees_the_slot_for_reuse() {
        use core::cell::Cell as StdCell;
        #[derive(Debug)]
        struct DropCounter<'a>(&'a StdCell<u32>);
        impl<'a> Drop for DropCounter<'a> {
            fn drop(&mut self) {
                self.0.set(self.0.get() + 1);
            }
        }

        let count = StdCell::new(0u32);
        let p = PoolAllocator::new();
        let x = p.create(DropCounter(&count)).unwrap();
        assert_eq!(count.get(), 0);
        unsafe { p.destroy(x) };
        assert_eq!(count.get(), 1, "destroy must run T's destructor");
        assert_eq!(p.len(), 0);

        p.create(DropCounter(&count)).unwrap();
        assert_eq!(
            count.get(),
            1,
            "reusing the freed slot must not run the old destructor again"
        );
    }

    #[test]
    fn freed_slots_are_reused_before_growing() {
        let p = PoolAllocator::with_capacity(2);
        let a = p.create(1u32).unwrap();
        let a_addr = a as *mut u32 as usize;
        unsafe { p.destroy(a) };

        let b = p.create(2u32).unwrap();
        let b_addr = b as *mut u32 as usize;
        assert_eq!(
            a_addr, b_addr,
            "a freed slot should be reused before bump-allocating a fresh one"
        );
        assert_eq!(*b, 2);
    }

    #[test]
    fn free_list_reuse_order_is_lifo() {
        let p = PoolAllocator::with_capacity(4);
        let a = p.create(1u32).unwrap() as *mut u32;
        let b = p.create(2u32).unwrap() as *mut u32;
        let c = p.create(3u32).unwrap() as *mut u32;
        unsafe {
            p.destroy(&mut *a);
            p.destroy(&mut *b);
            p.destroy(&mut *c);
        }

        let r1 = p.create(10u32).unwrap() as *mut u32;
        let r2 = p.create(20u32).unwrap() as *mut u32;
        let r3 = p.create(30u32).unwrap() as *mut u32;
        assert_eq!(r1, c, "most recently freed slot should be reused first");
        assert_eq!(r2, b);
        assert_eq!(r3, a);
    }

    #[test]
    fn growable_pool_grows_past_first_region() {
        let p = PoolAllocator::with_capacity(4);
        for i in 0..4u32 {
            p.create(i).unwrap();
        }
        assert_eq!(p.len(), 4);
        // 5th allocation must grow a new region rather than failing.
        let x = p.create(99u32).unwrap();
        assert_eq!(*x, 99);
        assert_eq!(p.len(), 5);
    }

    #[test]
    fn bounded_pool_refuses_to_grow_past_capacity() {
        let p = PoolAllocator::with_capacity_bounded(2);
        assert!(!p.is_growable());
        p.create(1u32).unwrap();
        p.create(2u32).unwrap();
        match p.create(3u32) {
            Ok(_) => panic!("expected the bounded pool to refuse a 3rd live slot"),
            Err(v) => assert_eq!(v, 3),
        }
        assert_eq!(p.len(), 2);
    }

    #[test]
    fn bounded_pool_can_still_reuse_a_freed_slot() {
        let p = PoolAllocator::with_capacity_bounded(1);
        let a = p.create(1u32).unwrap();
        unsafe { p.destroy(a) };
        // The one and only slot was freed -- must be reusable even
        // though this pool can never grow.
        let b = p.create(2u32).unwrap();
        assert_eq!(*b, 2);
    }

    #[test]
    fn many_create_destroy_cycles_stay_consistent() {
        let p: PoolAllocator<u32> = PoolAllocator::with_capacity(4);
        let mut live: Vec<*mut u32> = Vec::new();

        for round in 0u32..200 {
            let ptr = p.create(round).unwrap() as *mut u32;
            live.push(ptr);
            if round % 3 == 0 && !live.is_empty() {
                let dead = live.remove(0);
                unsafe { p.destroy(&mut *dead) };
            }
        }
        assert_eq!(p.len(), live.len());
        for &ptr in &live {
            assert!(unsafe { *ptr } < 200);
        }
    }

    #[test]
    fn default_matches_new() {
        let p: PoolAllocator<u32> = PoolAllocator::default();
        assert!(p.is_empty());
        assert!(p.is_growable());
    }

    #[test]
    fn drop_of_still_live_items_does_not_run_their_destructor() {
        // Documents the real, deliberate tradeoff from this module's
        // doc comment: unlike BumpArena, PoolAllocator does NOT run
        // Drop for items that are still live when the pool itself
        // drops. This test exists so that tradeoff stays true on
        // purpose, not by accident -- if it starts failing, either the
        // doc comment or the implementation drifted.
        use core::cell::Cell as StdCell;
        #[derive(Debug)]
        struct DropCounter<'a>(&'a StdCell<u32>);
        impl<'a> Drop for DropCounter<'a> {
            fn drop(&mut self) {
                self.0.set(self.0.get() + 1);
            }
        }

        let count = StdCell::new(0u32);
        {
            let p = PoolAllocator::new();
            p.create(DropCounter(&count)).unwrap();
            p.create(DropCounter(&count)).unwrap();
        }
        assert_eq!(
            count.get(),
            0,
            "items never explicitly destroyed must not run Drop on pool drop"
        );
    }

    #[test]
    fn zero_sized_types_do_not_panic_or_loop_forever() {
        let p: PoolAllocator<()> = PoolAllocator::with_capacity(4);
        let mut refs = Vec::new();
        for _ in 0..20 {
            refs.push(p.create(()).unwrap() as *mut ());
        }
        assert_eq!(p.len(), 20);
    }
}
