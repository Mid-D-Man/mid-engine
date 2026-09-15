//! Fixed-capacity, marker/rewind bump allocator — Mid Engine's answer to
//! "per-frame scratch storage," directly modeled on foonathan/memory's
//! `memory_stack`/`stack_marker` (real source read, not assumed — see
//! docs/mid-alloc.md).
//!
//! # Design, following bumpalo's proven interior-mutability pattern
//! (already verified, benched, and shipped — see `mid-arena`'s own
//! survey, docs/mid-arena.md), not foonathan's C++ shape verbatim
//!
//! foonathan's `memory_stack::allocate()` takes `this` by ordinary
//! mutable reference because C++ has no borrow checker to fight —
//! callers are trusted not to alias. A naive Rust port taking `&mut
//! self` per `alloc()` call would make this allocator nearly useless:
//! you could never hold a reference to an earlier allocation while
//! making a new one, which is the entire point of a scratch allocator
//! (build up several temporary objects, use them together, then
//! rewind). `bumpalo`/`typed-arena` solve exactly this with
//! `Cell`-based interior mutability so `alloc(&self, ...)` can return
//! `&'a mut T` tied to the allocator's own lifetime, not to a per-call
//! borrow — same fix applied here, not independently invented.
//!
//! # What this does NOT do (matching bumpalo's own documented tradeoff,
//! confirmed directly in docs/mid-arena.md's real benchmark comparison,
//! not assumed)
//!
//! Stores raw bytes, mixed types, no per-value `Drop` tracking —
//! `rewind()`/`reset()` reclaim the bytes but do not run destructors
//! for whatever was written into them. Same tradeoff `bumpalo` makes
//! for the same reason: tracking per-allocation type information to run
//! `Drop` would mean every allocation carries a vtable/type-erased
//! drop-glue pointer, defeating the "just bump a pointer" performance
//! this allocator exists for. Put `Copy` types in it, or types where
//! leaking on rewind is acceptable — the typical case for per-frame
//! scratch data.
//!
//! # Fixed capacity, not chunk-linked (unlike mid-arena's planned `bump`
//! feature)
//!
//! Backed by one buffer allocated once at construction, sized up
//! front — not a growable chain of regions the way tsoding/arena.h or
//! bumpalo itself are (docs/mid-arena.md's survey). A scratch/frame
//! allocator's whole point is a known, fixed budget reused every frame;
//! growing on demand would mean either invalidating live markers
//! (unsound) or chunk-linking (real added complexity for a use case
//! that doesn't need it). `alloc()` returns `None`/`Err` on overflow
//! rather than growing — the caller's job is to size the buffer for its
//! real workload, the same contract `foonathan::memory_stack` makes
//! callers responsible for.
//!
//! # Verification note
//!
//! This module's `unsafe` blocks were checked by hand against a
//! well-precedented pattern (bumpalo's own, real, shipped design) and
//! exercised by the tests below, including alignment and
//! multiple-simultaneous-allocation cases — not run under Miri or
//! AddressSanitizer, neither of which this sandbox's stable rustc 1.75
//! (no rustup/nightly component) can run. Worth a real Miri pass on a
//! toolchain that has it before this ships in anything that isn't
//! itself still under active development.

use crate::raw_alloc::RawAlloc;
use alloc::vec::Vec;
use core::cell::Cell;
use core::mem;
use core::ptr::NonNull;

/// A saved position in a [`StackAllocator`], obtained from
/// [`StackAllocator::marker`] and later passed to
/// [`StackAllocator::rewind`] to reclaim everything allocated since.
///
/// # Safety contract (not statically enforced — see this module's doc
/// comment for why a stack allocator can't fully check this)
///
/// Rewinding to a marker invalidates every reference returned by
/// [`StackAllocator::alloc`]/[`alloc_raw`](StackAllocator::alloc_raw)
/// *after* that marker was taken. Using such a reference afterward is
/// undefined behavior — same contract `foonathan::memory_stack`'s own
/// marker makes, not a weaker one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StackMarker(usize);

/// Fixed-capacity marker/rewind bump allocator. See this module's doc
/// comment for the full design.
pub struct StackAllocator {
    buf: Vec<u8>,
    top: Cell<usize>,
}

impl StackAllocator {
    /// Allocates `capacity` bytes up front. This is the allocator's
    /// entire budget for its whole lifetime — see this module's doc
    /// comment on why it doesn't grow.
    pub fn with_capacity(capacity: usize) -> Self {
        let mut buf = Vec::with_capacity(capacity);
        buf.resize(capacity, 0u8);
        Self {
            buf,
            top: Cell::new(0),
        }
    }

    /// Total capacity in bytes, fixed at construction.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.buf.len()
    }

    /// Bytes currently in use (from the start of the buffer to the
    /// current bump position).
    #[inline]
    pub fn used(&self) -> usize {
        self.top.get()
    }

    /// Bytes still available before the next allocation returns
    /// `None`/`Err`.
    #[inline]
    pub fn remaining(&self) -> usize {
        self.buf.len() - self.top.get()
    }

    /// Saves the current position. Pass to [`rewind`](Self::rewind)
    /// later to reclaim everything allocated after this call.
    #[inline]
    pub fn marker(&self) -> StackMarker {
        StackMarker(self.top.get())
    }

    /// Reclaims every byte allocated since `marker` was taken. See
    /// [`StackMarker`]'s doc comment for the safety contract this
    /// relies on the caller to uphold.
    ///
    /// A `marker` from a *different* `StackAllocator`, or one further
    /// ahead than the current position, is a debug-asserted logic
    /// error, not something this method tries to guess its way around.
    pub fn rewind(&mut self, marker: StackMarker) {
        debug_assert!(
            marker.0 <= self.top.get(),
            "rewind() marker is ahead of the current position -- from a \
             different StackAllocator, or already rewound past?"
        );
        self.top.set(marker.0);
    }

    /// Reclaims everything, equivalent to rewinding to the marker taken
    /// at construction.
    #[inline]
    pub fn reset(&mut self) {
        self.top.set(0);
    }

    /// Allocates `size_bytes` aligned to `align`, returning a pointer to
    /// (zeroed, not truly uninitialized — see `with_capacity`) memory
    /// valid until the next `rewind`/`reset` that reclaims it, or
    /// `None` if the remaining capacity can't satisfy the request
    /// (including alignment padding). `align` must be a power of two —
    /// debug-asserted, not checked in release, matching `Layout`'s own
    /// contract. Exists for callers below [`alloc`](Self::alloc) who
    /// need to place a type with an unusual/runtime alignment.
    pub fn alloc_raw(&self, size_bytes: usize, align: usize) -> Option<NonNull<u8>> {
        debug_assert!(align.is_power_of_two(), "align must be a power of two");

        let base = self.buf.as_ptr() as usize;
        let current = base + self.top.get();
        // Round up to the next multiple of `align`. `align` is a power
        // of two, so this is the standard
        // `(x + align - 1) & !(align - 1)` trick — using `checked_add`,
        // not wrapping, so a pathological huge `align` can't silently
        // wrap `current` back into the buffer and report a false
        // success (a real, checkable failure mode APR's own
        // `apr_palloc` guards against too — docs/mid-arena.md's C
        // survey — ported here on purpose, not incidentally).
        let aligned = current.checked_add(align - 1)? & !(align - 1);
        let padding = aligned - current;
        let end = aligned.checked_add(size_bytes)?;

        if end > base + self.buf.len() {
            return None;
        }

        self.top.set(self.top.get() + padding + size_bytes);

        // SAFETY: `aligned` is inside `[base, base + buf.len())` by the
        // `end > base + self.buf.len()` check just above (and
        // `aligned >= current >= base` since `padding >= 0`), so this
        // is a valid, non-null pointer into `self.buf`'s live
        // allocation. Deriving it via `usize` arithmetic on
        // `self.buf.as_ptr()` rather than pointer-offset methods
        // throughout is what let the alignment math above use
        // `checked_add` (a real overflow check) instead of relying on
        // `<*const T>::add`'s narrower, UB-on-overflow contract.
        Some(unsafe { NonNull::new_unchecked(aligned as *mut u8) })
    }

    /// Safe, typed convenience over [`alloc_raw`](Self::alloc_raw):
    /// allocates space for a `T`, moves `value` into it, and returns a
    /// `&mut T` borrowing from `self` — not from a `&mut self` call,
    /// see this module's doc comment for why that's the point. Returns
    /// `value` back, unwritten, if there isn't enough remaining
    /// capacity, rather than losing it silently.
    pub fn alloc<T>(&self, value: T) -> Result<&mut T, T> {
        let ptr = match self.alloc_raw(mem::size_of::<T>(), mem::align_of::<T>()) {
            Some(ptr) => ptr,
            None => return Err(value),
        };
        let typed: NonNull<T> = ptr.cast();
        // SAFETY: `alloc_raw` reserved exactly `size_of::<T>()` bytes
        // starting at an address aligned to `align_of::<T>()`, and that
        // byte range is exclusively ours until the next
        // `rewind`/`reset` -- nothing else in this module hands out a
        // pointer into the same range without first bumping `top` past
        // it. `ptr::write` here is a real initialization, not a raw
        // store into possibly-live memory: the bytes were zeroed at
        // construction (`with_capacity`) and nothing has read them as
        // a `T` before this write, so this takes them from "zeroed
        // bytes" to "a live `T`" in one step, matching the usage
        // pattern this kind of placement requires.
        unsafe {
            typed.as_ptr().write(value);
            Ok(&mut *typed.as_ptr())
        }
    }
    /// Attempts to resize a previously-returned raw allocation in
    /// place, without moving it. `old_size` must be the size that was
    /// originally requested for `ptr` (this allocator hands back a bare
    /// pointer, not a sized slice, so the caller must supply it, same
    /// as `try_dealloc_raw`'s own contract elsewhere in this crate).
    /// Returns `true` if `ptr` is now valid to use up to `new_size`
    /// bytes without moving, `false` if it could not be done in place --
    /// the caller must then allocate fresh and copy, same as any other
    /// allocator's resize contract.
    ///
    /// Directly modeled on `std.heap.FixedBufferAllocator`'s real
    /// `resize`/`isLastAllocation` (source read,
    /// `lib/std/heap/FixedBufferAllocator.zig`, not assumed from the
    /// name): only the allocation that is genuinely the *most recent*
    /// one (its end exactly matches the current bump position) can grow
    /// or shrink in place, since nothing else sits between it and this
    /// allocator's free space. Every other allocation can still shrink
    /// *logically* -- report success without moving anything, since a
    /// smaller size is always a valid view of the same bytes -- but can
    /// never grow, since real live data may sit immediately after it.
    pub fn resize_raw(&self, ptr: NonNull<u8>, old_size: usize, new_size: usize) -> bool {
        let base = self.buf.as_ptr() as usize;
        let addr = ptr.as_ptr() as usize;
        let is_last_allocation = addr + old_size == base + self.top.get();

        if !is_last_allocation {
            return new_size <= old_size;
        }

        if new_size <= old_size {
            // Shrinking the last allocation for real reclaims the
            // freed tail immediately, rather than waiting for the
            // usual `rewind`/`reset` reclamation.
            self.top.set(self.top.get() - (old_size - new_size));
            return true;
        }

        let grow_by = new_size - old_size;
        if self.top.get() + grow_by > self.buf.len() {
            return false;
        }
        self.top.set(self.top.get() + grow_by);
        true
    }
}
/// `allocator_traits` specialization directly (checked against that
/// source, not assumed): its `try_deallocate_node` checks real pointer
/// ownership (`state.arena_.owns(ptr)`) and does nothing beyond that,
/// since actual reclamation only ever happens through `unwind()` --
/// this stack allocator's own `rewind()`/`reset()`. There is no way to
/// reclaim one arbitrary node out of bump order, and this allocator
/// was never meant to.
impl RawAlloc for StackAllocator {
    fn try_alloc_raw(&self, size: usize, align: usize) -> Option<NonNull<u8>> {
        self.alloc_raw(size, align)
    }

    unsafe fn try_dealloc_raw(&self, ptr: NonNull<u8>, _size: usize, _align: usize) -> bool {
        let start = self.buf.as_ptr() as usize;
        let end = start + self.buf.len();
        let addr = ptr.as_ptr() as usize;
        // `<= end`, not `< end`: a zero-sized allocation can validly
        // land exactly one byte past the last real byte (the same
        // "one past the end" pointer `Vec`'s own iterators rely on),
        // and this check only ever gates ownership routing, not an
        // actual memory access.
        addr >= start && addr <= end
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn starts_empty() {
        let a = StackAllocator::with_capacity(64);
        assert_eq!(a.used(), 0);
        assert_eq!(a.capacity(), 64);
        assert_eq!(a.remaining(), 64);
    }

    #[test]
    fn alloc_stores_and_reads_back_the_value() {
        let a = StackAllocator::with_capacity(64);
        let x = a.alloc(42u32).unwrap();
        assert_eq!(*x, 42);
        *x = 43;
        assert_eq!(*x, 43);
    }

    #[test]
    fn multiple_simultaneous_allocations_stay_independent() {
        // The actual property this whole Cell-based design exists for --
        // holding references to earlier allocations while making new
        // ones, which a naive &mut self API could never allow.
        let a = StackAllocator::with_capacity(256);
        let x = a.alloc(1u32).unwrap();
        let y = a.alloc(2u32).unwrap();
        let z = a.alloc(3u32).unwrap();
        assert_eq!(*x, 1);
        assert_eq!(*y, 2);
        assert_eq!(*z, 3);
        *x += 10;
        *y += 20;
        *z += 30;
        assert_eq!((*x, *y, *z), (11, 22, 33));
    }

    #[test]
    fn overflow_returns_the_value_back_unwritten() {
        let a = StackAllocator::with_capacity(4);
        // A u64 (8 bytes, needs 8-byte alignment) can't fit in a
        // 4-byte buffer no matter the alignment padding.
        match a.alloc(0xdead_beef_u64) {
            Ok(_) => panic!("expected overflow"),
            Err(v) => assert_eq!(v, 0xdead_beef_u64),
        }
        // The failed attempt must not have moved `top` at all.
        assert_eq!(a.used(), 0);
    }

    #[test]
    fn marker_and_rewind_reclaim_exactly_whats_after_the_marker() {
        let mut a = StackAllocator::with_capacity(256);
        a.alloc(1u32).unwrap();
        let mark = a.marker();
        let used_at_mark = a.used();

        a.alloc(2u32).unwrap();
        a.alloc(3u64).unwrap();
        assert!(a.used() > used_at_mark);

        a.rewind(mark);
        assert_eq!(a.used(), used_at_mark);

        // The reclaimed space is real: allocating again reuses it
        // rather than reporting overflow.
        let again = a.alloc(99u32).unwrap();
        assert_eq!(*again, 99);
    }

    #[test]
    fn reset_reclaims_everything() {
        let mut a = StackAllocator::with_capacity(64);
        a.alloc(1u32).unwrap();
        a.alloc(2u64).unwrap();
        assert!(a.used() > 0);
        a.reset();
        assert_eq!(a.used(), 0);
    }

    #[test]
    fn alignment_is_actually_respected_not_just_assumed() {
        // Force a misaligned starting position with a 1-byte alloc,
        // then check a type with real alignment requirements lands on
        // a correctly aligned address, not wherever the byte pointer
        // happened to be.
        #[repr(align(16))]
        #[derive(Debug)]
        struct Aligned16 {
            a: u64,
            b: u64,
        }

        let a = StackAllocator::with_capacity(128);
        let _byte = a.alloc(1u8).unwrap(); // pushes `top` to an odd offset
        let val = a.alloc(Aligned16 { a: 7, b: 8 }).unwrap();

        let addr = val as *mut Aligned16 as usize;
        assert_eq!(
            addr % mem::align_of::<Aligned16>(),
            0,
            "returned pointer must be aligned to the type's real requirement"
        );
        assert_eq!(val.a, 7);
        assert_eq!(val.b, 8);
    }

    #[test]
    fn rewind_to_the_very_start_then_realloc_reuses_the_same_bytes() {
        let mut a = StackAllocator::with_capacity(64);
        let start = a.marker();
        let x = a.alloc(111u32).unwrap();
        let x_addr = x as *mut u32 as usize;

        a.rewind(start);
        let y = a.alloc(222u32).unwrap();
        let y_addr = y as *mut u32 as usize;

        assert_eq!(x_addr, y_addr, "rewinding to the start should hand back the exact same bytes");
        assert_eq!(*y, 222);
    }

    #[test]
    fn alloc_raw_respects_capacity_boundary_exactly() {
        let a = StackAllocator::with_capacity(8);
        // Exactly fills the buffer -- must succeed.
        assert!(a.alloc_raw(8, 1).is_some());
        // Nothing left at all now.
        assert!(a.alloc_raw(1, 1).is_none());
    }

    #[test]
    fn raw_alloc_try_dealloc_recognizes_only_its_own_pointers() {
        use crate::raw_alloc::{HeapAlloc, RawAlloc};

        let s = StackAllocator::with_capacity(64);
        let mine = s.try_alloc_raw(8, 8).expect("plenty of room");
        let unrelated = HeapAlloc
            .try_alloc_raw(8, 8)
            .expect("a small heap allocation should not fail");

        unsafe {
            assert!(
                s.try_dealloc_raw(mine, 8, 8),
                "must recognize a pointer from its own buffer"
            );
            assert!(
                !s.try_dealloc_raw(unrelated, 8, 8),
                "must not falsely claim a pointer it never allocated"
            );
            // Real cleanup for the heap-backed pointer so this test
            // doesn't leak.
            assert!(HeapAlloc.try_dealloc_raw(unrelated, 8, 8));
        }
    }

    #[test]
    fn resize_raw_on_the_last_allocation_can_grow_in_place() {
        let a = StackAllocator::with_capacity(64);
        let ptr = a.alloc_raw(8, 1).unwrap();
        assert!(a.resize_raw(ptr, 8, 20));
        assert_eq!(a.used(), 20);
        // The 12 grown bytes are real, usable memory at the same address.
        unsafe {
            core::ptr::write_bytes(ptr.as_ptr(), 0x42, 20);
            assert_eq!(*ptr.as_ptr().add(19), 0x42);
        }
    }

    #[test]
    fn resize_raw_on_the_last_allocation_can_shrink_and_reclaims_the_tail() {
        let a = StackAllocator::with_capacity(64);
        let ptr = a.alloc_raw(20, 1).unwrap();
        assert!(a.resize_raw(ptr, 20, 8));
        assert_eq!(a.used(), 8, "the freed tail must be reclaimed immediately, not just logically");
        // The reclaimed space is real: a fresh allocation can reuse it.
        let next = a.alloc_raw(56, 1);
        assert!(next.is_some(), "the 12 reclaimed bytes plus remaining capacity should fit 56 more");
    }

    #[test]
    fn resize_raw_on_a_non_last_allocation_can_only_shrink_logically() {
        let a = StackAllocator::with_capacity(64);
        let first = a.alloc_raw(8, 1).unwrap();
        let _second = a.alloc_raw(8, 1).unwrap(); // now `first` is no longer the last allocation
        let used_before = a.used();

        assert!(
            a.resize_raw(first, 8, 4),
            "a non-last allocation can still shrink logically"
        );
        assert_eq!(
            a.used(),
            used_before,
            "shrinking a non-last allocation must not move the bump position at all"
        );
        assert!(
            !a.resize_raw(first, 8, 16),
            "a non-last allocation can never grow -- real live data sits right after it"
        );
    }

    #[test]
    fn resize_raw_growing_the_last_allocation_past_capacity_fails() {
        let a = StackAllocator::with_capacity(16);
        let ptr = a.alloc_raw(8, 1).unwrap();
        assert!(!a.resize_raw(ptr, 8, 32));
        assert_eq!(a.used(), 8, "a failed grow must not move the bump position");
    }
}
