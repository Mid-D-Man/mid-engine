// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-alloc.md, section "What's built: `StackAllocator`"
// ============================================================================
//! Fixed-capacity marker/rewind bump allocator for per-frame scratch
//! storage, modeled on foonathan/memory's `memory_stack`.
//!
//! Allocation takes `&self` (interior mutability through `Cell`), so
//! earlier allocations stay usable while new ones are made. Rewinding
//! to a [`StackMarker`] reclaims everything allocated after it.
//!
//! The buffer is one uninitialized `Vec<MaybeUninit<u8>>` sized at
//! construction. It never grows: `alloc_raw` returns `None`, and `alloc`
//! hands the value back, when the remaining capacity is too small.
//! Memory from `alloc_raw` is uninitialized, so write before reading.
//!
//! `rewind` and `reset` reclaim bytes without running destructors, the
//! same tradeoff `bumpalo` makes. Store `Copy` types, or types where
//! leaking on rewind is acceptable.
//!
//! The `unsafe` blocks in this file have not been run under Miri or
//! AddressSanitizer.

use crate::raw_alloc::{grow_via_alloc_copy_dealloc, RawAlloc};
use alloc::vec::Vec;
use core::cell::Cell;
use core::mem::{self, MaybeUninit};
use core::ptr::NonNull;

/// A saved position in a [`StackAllocator`], obtained from
/// [`StackAllocator::marker`] and later passed to
/// [`StackAllocator::rewind`] to reclaim everything allocated since.
///
/// # Safety contract (not statically enforced)
///
/// Rewinding to a marker invalidates every reference returned by
/// [`StackAllocator::alloc`]/[`alloc_raw`](StackAllocator::alloc_raw)
/// *after* that marker was taken. Using such a reference afterward is
/// undefined behavior, the same contract `foonathan::memory_stack`'s
/// marker makes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StackMarker(usize);

/// Fixed-capacity marker/rewind bump allocator. See this module's doc
/// comment for the full design.
pub struct StackAllocator {
    buf: Vec<MaybeUninit<u8>>,
    /// Bump position as an absolute address inside `buf`, so the hot
    /// path skips a base add. `buf` never reallocates, so the address
    /// stays valid when this struct moves.
    cur: Cell<usize>,
    /// One past the last byte of `buf`.
    end: usize,
}

impl StackAllocator {
    /// Allocates `capacity` bytes up front. This is the allocator's
    /// entire budget for its whole lifetime, since it never grows.
    pub fn with_capacity(capacity: usize) -> Self {
        let mut buf: Vec<MaybeUninit<u8>> = Vec::with_capacity(capacity);
        // SAFETY: `MaybeUninit<u8>` has no validity invariant, so any
        // `capacity` uninitialized slots are valid elements.
        unsafe { buf.set_len(capacity) };
        let base = buf.as_ptr() as usize;
        Self {
            end: base + buf.len(),
            cur: Cell::new(base),
            buf,
        }
    }

    #[inline]
    fn base(&self) -> usize {
        self.buf.as_ptr() as usize
    }

    /// Bump position as an offset from the start of the buffer.
    #[inline]
    fn top(&self) -> usize {
        self.cur.get() - self.base()
    }

    #[inline]
    fn set_top(&self, top: usize) {
        self.cur.set(self.base() + top);
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
        self.top()
    }

    /// Bytes still available before the next allocation returns
    /// `None`/`Err`.
    #[inline]
    pub fn remaining(&self) -> usize {
        self.end - self.cur.get()
    }

    /// Saves the current position. Pass to [`rewind`](Self::rewind)
    /// later to reclaim everything allocated after this call.
    #[inline]
    pub fn marker(&self) -> StackMarker {
        StackMarker(self.top())
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
            marker.0 <= self.top(),
            "rewind() marker is ahead of the current position -- from a \
             different StackAllocator, or already rewound past?"
        );
        self.set_top(marker.0);
    }

    /// Reclaims everything, equivalent to rewinding to the marker taken
    /// at construction.
    #[inline]
    pub fn reset(&mut self) {
        self.set_top(0);
    }

    /// Allocates `size_bytes` aligned to `align`, returning a pointer to
    /// uninitialized memory that stays valid until the next
    /// `rewind`/`reset` that reclaims it, or `None` if the
    /// remaining capacity can't satisfy the request, alignment padding
    /// included, or if `size_bytes` exceeds `isize::MAX`. `align` must be
    /// a power of two: debug-asserted, and release builds return `None`
    /// for anything else. Exists for callers below
    /// [`alloc`](Self::alloc) who need an unusual or runtime alignment.
    #[inline]
    pub fn alloc_raw(&self, size_bytes: usize, align: usize) -> Option<NonNull<u8>> {
        debug_assert!(align.is_power_of_two(), "align must be a power of two");
        // Release builds also refuse a bad `align` or a size no `Layout`
        // could hold, which keeps the arithmetic below from wrapping.
        if size_bytes > isize::MAX as usize || !align.is_power_of_two() {
            return None;
        }

        let cur = self.cur.get();
        // Bytes from `cur` up to the next multiple of `align`. Bit ops
        // only, so it cannot wrap.
        let padding = cur.wrapping_neg() & (align - 1);
        // `padding < align <= 2^(BITS-1)` and `size_bytes <= isize::MAX`,
        // so this sum cannot wrap.
        let needed = padding + size_bytes;
        // `cur <= end` always holds, so this cannot underflow.
        if needed > self.end - cur {
            return None;
        }

        self.cur.set(cur + needed);

        // SAFETY: `needed <= end - cur` puts `cur + padding` and the
        // `size_bytes` after it inside the buffer, and `cur >= base`
        // (a live allocation's non-null start), so the pointer is
        // non-null and valid for `size_bytes` bytes.
        Some(unsafe { NonNull::new_unchecked((cur + padding) as *mut u8) })
    }

    /// Safe, typed convenience over [`alloc_raw`](Self::alloc_raw):
    /// allocates space for a `T`, moves `value` into it, and returns a
    /// `&mut T` borrowing from `self`, not from a `&mut self` call, so
    /// several allocations can be live together. Returns `value` back,
    /// unwritten, if there isn't enough remaining capacity.
    #[inline]
    pub fn alloc<T>(&self, value: T) -> Result<&mut T, T> {
        let ptr = match self.alloc_raw(mem::size_of::<T>(), mem::align_of::<T>()) {
            Some(ptr) => ptr,
            None => return Err(value),
        };
        let typed: NonNull<T> = ptr.cast();
        // SAFETY: `alloc_raw` reserved exactly `size_of::<T>()` bytes
        // starting at an address aligned to `align_of::<T>()`, and that
        // byte range is exclusively ours until the next
        // `rewind`/`reset`: nothing else in this module hands out a
        // pointer into the same range without first bumping `top` past
        // it. The `write` initializes the slot before the reference is
        // formed.
        unsafe {
            typed.as_ptr().write(value);
            Ok(&mut *typed.as_ptr())
        }
    }

    /// Attempts to resize a previously returned raw allocation in
    /// place, without moving it. `old_size` must be the size originally
    /// requested for `ptr`, since this allocator returns a bare pointer.
    /// Returns `true` if `ptr` is now valid up to `new_size` bytes
    /// without moving, `false` if the caller must allocate fresh and
    /// copy.
    ///
    /// Only the most recent allocation (its end equals the bump
    /// position) can grow or shrink in place. Any other allocation can
    /// shrink logically, reporting success without moving anything, but
    /// can never grow, since live data may sit right after it.
    pub fn resize_raw(&self, ptr: NonNull<u8>, old_size: usize, new_size: usize) -> bool {
        let addr = ptr.as_ptr() as usize;
        let is_last_allocation = addr + old_size == self.cur.get();

        if !is_last_allocation {
            return new_size <= old_size;
        }

        if new_size <= old_size {
            // Shrinking the last allocation for real reclaims the
            // freed tail immediately, rather than waiting for the
            // usual `rewind`/`reset` reclamation.
            self.cur.set(self.cur.get() - (old_size - new_size));
            return true;
        }

        let grow_by = new_size - old_size;
        if grow_by > self.end - self.cur.get() {
            return false;
        }
        self.cur.set(self.cur.get() + grow_by);
        true
    }
}

/// `RawAlloc` view of the stack. `try_dealloc_raw` only checks that the
/// pointer lies inside this buffer and frees nothing, since space comes
/// back only through `rewind`/`reset`.
impl RawAlloc for StackAllocator {
    #[inline]
    fn try_alloc_raw(&self, size: usize, align: usize) -> Option<NonNull<u8>> {
        self.alloc_raw(size, align)
    }

    #[inline]
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

    /// Grows in place when `ptr` is the most recent allocation (the
    /// same check `resize_raw` makes), otherwise falls back to the
    /// shared alloc-copy-dealloc helper.
    #[inline]
    unsafe fn try_grow_raw(
        &self,
        ptr: NonNull<u8>,
        old_size: usize,
        new_size: usize,
        align: usize,
    ) -> Option<NonNull<u8>> {
        debug_assert!(
            new_size >= old_size,
            "try_grow_raw is for growing, not shrinking"
        );

        let addr = ptr.as_ptr() as usize;
        let is_last_allocation = addr + old_size == self.cur.get();

        if is_last_allocation {
            let grow_by = new_size - old_size;
            if grow_by <= self.end - self.cur.get() {
                self.cur.set(self.cur.get() + grow_by);
                return Some(ptr);
            }
        }

        // SAFETY: forwarding this call's own contract on
        // `ptr`/`old_size`/`align`/`new_size >= old_size` straight
        // through to the shared fallback.
        unsafe { grow_via_alloc_copy_dealloc(self, ptr, old_size, new_size, align) }
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

        assert_eq!(
            x_addr, y_addr,
            "rewinding to the start should hand back the exact same bytes"
        );
        assert_eq!(*y, 222);
    }

    #[test]
    fn alloc_raw_refuses_sizes_and_alignments_no_buffer_could_satisfy() {
        let a = StackAllocator::with_capacity(64);
        assert!(a.alloc_raw(usize::MAX, 1).is_none());
        assert!(a.alloc_raw(isize::MAX as usize + 1, 1).is_none());
        assert!(a.alloc_raw(isize::MAX as usize, 8).is_none());
        assert!(a.alloc_raw(8, 1 << (usize::BITS - 1)).is_none());
        assert_eq!(
            a.used(),
            0,
            "a refused request must not move the bump position"
        );
        assert!(a.alloc_raw(8, 8).is_some(), "still usable after refusals");
    }

    #[test]
    fn zero_capacity_stack_serves_only_zero_sized_requests() {
        let a = StackAllocator::with_capacity(0);
        assert!(a.alloc_raw(0, 1).is_some());
        assert!(a.alloc_raw(1, 1).is_none());
        assert_eq!(a.used(), 0);
        assert_eq!(a.remaining(), 0);
    }

    #[test]
    fn whole_capacity_is_usable_as_one_allocation() {
        let a = StackAllocator::with_capacity(4096);
        let p = a.alloc_raw(4096, 1).expect("full capacity fits in one allocation");
        assert_eq!(a.remaining(), 0);
        // SAFETY: `p` is valid for 4096 bytes, and every byte is written
        // before it is read.
        unsafe {
            for i in 0..4096usize {
                p.as_ptr().add(i).write((i % 251) as u8);
            }
            for i in 0..4096usize {
                assert_eq!(p.as_ptr().add(i).read(), (i % 251) as u8);
            }
        }
    }

    #[test]
    fn used_is_padding_plus_size_measured_from_the_buffer_start() {
        let a = StackAllocator::with_capacity(64);
        let _first = a.alloc_raw(1, 1).unwrap();
        assert_eq!(a.used(), 1);
        let p = a.alloc_raw(8, 8).unwrap();
        assert_eq!(p.as_ptr() as usize % 8, 0);
        let offset = p.as_ptr() as usize - a.buf.as_ptr() as usize;
        assert_eq!(a.used(), offset + 8);
        assert_eq!(a.remaining(), 64 - a.used());
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
        assert_eq!(
            a.used(),
            8,
            "the freed tail must be reclaimed immediately, not just logically"
        );
        // The reclaimed space is real: a fresh allocation can reuse it.
        let next = a.alloc_raw(56, 1);
        assert!(
            next.is_some(),
            "the 12 reclaimed bytes plus remaining capacity should fit 56 more"
        );
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

    #[test]
    fn try_grow_raw_on_the_last_allocation_grows_in_place() {
        use crate::raw_alloc::RawAlloc;
        let a = StackAllocator::with_capacity(64);
        let ptr = a.try_alloc_raw(8, 8).unwrap();
        unsafe {
            (ptr.as_ptr() as *mut u64).write(0x1122_3344_5566_7788);
        }
        let grown =
            unsafe { a.try_grow_raw(ptr, 8, 32, 8) }.expect("plenty of room to grow in place");
        assert_eq!(
            grown, ptr,
            "growing the most recent allocation in place must return the same address"
        );
        assert_eq!(a.used(), 32);
        assert_eq!(
            unsafe { (grown.as_ptr() as *const u64).read() },
            0x1122_3344_5566_7788,
            "in-place growth must not disturb the existing bytes"
        );
    }

    #[test]
    fn try_grow_raw_on_a_non_last_allocation_falls_back_to_alloc_copy_dealloc() {
        use crate::raw_alloc::RawAlloc;
        let a = StackAllocator::with_capacity(64);
        let first = a.try_alloc_raw(8, 8).unwrap();
        unsafe {
            (first.as_ptr() as *mut u64).write(0xAAAA_BBBB_CCCC_DDDD);
        }
        let _second = a.try_alloc_raw(8, 8).unwrap(); // now `first` is no longer last
        let used_before = a.used();

        let grown =
            unsafe { a.try_grow_raw(first, 8, 32, 8) }.expect("fallback path should still succeed");
        assert_ne!(
            grown, first,
            "a non-last allocation cannot grow in place -- must land at a new address"
        );
        assert_eq!(
            unsafe { (grown.as_ptr() as *const u64).read() },
            0xAAAA_BBBB_CCCC_DDDD,
            "the fallback path must still preserve the original bytes"
        );
        assert!(
            a.used() > used_before,
            "the fallback path allocates a real new block, using more of the buffer"
        );
    }
}
