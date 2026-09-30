// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-alloc.md, section "backed.rs"
// ============================================================================
//! [`BackedStack`], a `StackAllocator`-shaped bump allocator whose
//! backing buffer is provisioned by a parent [`RawAlloc`] instead of
//! the global allocator. The real gap this closes, stated directly
//! rather than left implied: every allocator in this crate up to this
//! point (`StackAllocator`'s own `Vec<u8>`, `PoolAllocator`'s
//! chunk-linked regions) reaches for the global heap independently the
//! moment it grows, with no way to have one allocator provision
//! another's backing memory instead. `BackedStack` is the first,
//! minimal, concrete instance of that pattern: allocate one big block
//! from a parent `RawAlloc` up front, then bump-allocate within it
//! exactly like `StackAllocator` does within its own `Vec<u8>`, and
//! give the whole block back to the parent on drop.
//!
//! Not modeled on a specific foonathan/memory or Zig type -- this
//! survey never found one real source with a matching "one allocator
//! carves its buffer from an arbitrary parent `RawAlloc`" shape to
//! port from (foonathan's own allocators are template-parameterized
//! over a `RawAllocator`, but none of the ones this survey read used
//! that parameterization to carve a *contiguous sub-region* out of
//! another allocator the way this does). The bump-allocation logic
//! itself is a direct copy of `StackAllocator::alloc_raw`'s own real
//! math (absolute-address cursor, wrap-free padding, `usize` pointer
//! arithmetic rather than `<*const T>::add`), not reinvented, since
//! that logic was already reviewed and tested there.

use crate::raw_alloc::RawAlloc;
use core::cell::Cell;
use core::mem;
use core::ptr::NonNull;

/// An opaque bump position from a [`BackedStack`]. See
/// [`StackAllocator`](crate::stack_allocator::StackAllocator)'s own
/// `StackMarker` for the safety contract this carries the same way --
/// kept as a distinct type here rather than reusing that one, since
/// its inner field is private to its own module and a marker from one
/// allocator must never be usable on the other.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BackedStackMarker(usize);

/// A fixed-capacity marker/rewind bump allocator, same shape as
/// [`StackAllocator`](crate::stack_allocator::StackAllocator), except
/// its one backing buffer comes from a parent `RawAlloc` (borrowed for
/// `'p`) instead of an owned `Vec<u8>`. See this module's doc comment
/// for what real gap this closes and what it doesn't.
pub struct BackedStack<'p, P: RawAlloc> {
    parent: &'p P,
    buf: NonNull<u8>,
    capacity: usize,
    align: usize,
    /// Bump position as an absolute address inside `buf`.
    cur: Cell<usize>,
    /// One past the last byte of `buf`.
    end: usize,
}

impl<'p, P: RawAlloc> BackedStack<'p, P> {
    /// Requests one block of `capacity` bytes aligned to `align` from
    /// `parent`. Returns `None` if `parent` cannot satisfy that
    /// request -- this allocator's entire budget for its whole
    /// lifetime, same as `StackAllocator::with_capacity`, it just asks
    /// somewhere other than the global allocator for it.
    pub fn new(parent: &'p P, capacity: usize, align: usize) -> Option<Self> {
        let capacity = capacity.max(1);
        let buf = parent.try_alloc_raw(capacity, align)?;
        let base = buf.as_ptr() as usize;
        Some(Self {
            parent,
            buf,
            capacity,
            align,
            cur: Cell::new(base),
            end: base + capacity,
        })
    }

    /// Bump position as an offset from the start of the buffer.
    #[inline]
    fn top(&self) -> usize {
        self.cur.get() - self.buf.as_ptr() as usize
    }

    #[inline]
    fn set_top(&self, top: usize) {
        self.cur.set(self.buf.as_ptr() as usize + top);
    }

    /// Total capacity in bytes, fixed at construction.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Bytes currently in use.
    #[inline]
    pub fn used(&self) -> usize {
        self.top()
    }

    /// Bytes still available before the next allocation returns `None`.
    #[inline]
    pub fn remaining(&self) -> usize {
        self.end - self.cur.get()
    }

    /// Saves the current position. Pass to [`rewind`](Self::rewind)
    /// later to reclaim everything allocated after this call.
    #[inline]
    pub fn marker(&self) -> BackedStackMarker {
        BackedStackMarker(self.top())
    }

    /// Reclaims every byte allocated since `marker` was taken. Takes
    /// `&mut self` for the same real reason
    /// `StackAllocator::rewind` does: rewinding retroactively
    /// invalidates any `&mut T` still borrowed from an allocation
    /// after `marker`, and `&mut self` here is what makes the borrow
    /// checker enforce that none are still alive when this is called.
    pub fn rewind(&mut self, marker: BackedStackMarker) {
        debug_assert!(
            marker.0 <= self.top(),
            "rewind() marker is ahead of the current position -- from a \
             different BackedStack, or already rewound past?"
        );
        self.set_top(marker.0);
    }

    /// Reclaims everything, equivalent to rewinding to the marker taken
    /// at construction.
    #[inline]
    pub fn reset(&mut self) {
        self.set_top(0);
    }

    /// Allocates `size_bytes` aligned to `align`. Same bump math as
    /// `StackAllocator::alloc_raw`, just inside this allocator's
    /// parent-provisioned `buf` instead of an owned `Vec<u8>`.
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

    /// Safe, typed convenience over [`alloc_raw`](Self::alloc_raw).
    /// Returns `value` back, unwritten, if there isn't enough
    /// remaining capacity.
    #[inline]
    pub fn alloc<T>(&self, value: T) -> Result<&mut T, T> {
        let ptr = match self.alloc_raw(mem::size_of::<T>(), mem::align_of::<T>()) {
            Some(ptr) => ptr,
            None => return Err(value),
        };
        let typed: NonNull<T> = ptr.cast();
        // SAFETY: same reasoning as `StackAllocator::alloc` -- these
        // `size_of::<T>()` bytes were exclusively reserved by
        // `alloc_raw` above and are correctly aligned for `T`.
        unsafe {
            typed.as_ptr().write(value);
            Ok(&mut *typed.as_ptr())
        }
    }
}

impl<'p, P: RawAlloc> Drop for BackedStack<'p, P> {
    fn drop(&mut self) {
        // SAFETY: `buf`/`capacity`/`align` are exactly the values
        // `parent.try_alloc_raw` returned and was called with in
        // `new` above, never mutated since, and this is the only
        // place that ever deallocates them.
        unsafe {
            self.parent
                .try_dealloc_raw(self.buf, self.capacity, self.align);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::raw_alloc::{HeapAlloc, NullAlloc};

    #[test]
    fn alloc_raw_refuses_sizes_and_alignments_no_buffer_could_satisfy() {
        let heap = HeapAlloc;
        let s = BackedStack::new(&heap, 64, 8).unwrap();
        assert!(s.alloc_raw(usize::MAX, 1).is_none());
        assert!(s.alloc_raw(isize::MAX as usize + 1, 1).is_none());
        assert!(s.alloc_raw(8, 1 << (usize::BITS - 1)).is_none());
        assert_eq!(
            s.used(),
            0,
            "a refused request must not move the bump position"
        );
        assert!(s.alloc_raw(8, 8).is_some(), "still usable after refusals");
    }

    #[test]
    fn new_fails_cleanly_when_the_parent_cannot_provide_the_block() {
        assert!(BackedStack::new(&NullAlloc, 64, 8).is_none());
    }

    #[test]
    fn allocates_and_reads_back_correctly() {
        let heap = HeapAlloc;
        let s = BackedStack::new(&heap, 64, 8).expect("heap should provide a small block");
        let a = s.alloc(11u32).unwrap();
        let b = s.alloc(22u32).unwrap();
        assert_eq!(*a, 11);
        assert_eq!(*b, 22);
    }

    #[test]
    fn marker_and_rewind_reclaim_exactly_whats_after_the_marker() {
        let heap = HeapAlloc;
        let mut s = BackedStack::new(&heap, 64, 8).expect("heap should provide a small block");
        s.alloc(1u32).unwrap();
        let m = s.marker();
        s.alloc(2u32).unwrap();
        s.alloc(3u32).unwrap();
        assert_eq!(s.used(), 12);
        s.rewind(m);
        assert_eq!(s.used(), 4);
    }

    #[test]
    fn respects_its_own_capacity_boundary() {
        let heap = HeapAlloc;
        let s = BackedStack::new(&heap, 8, 1).expect("heap should provide a small block");
        assert!(s.alloc_raw(8, 1).is_some());
        assert!(s.alloc_raw(1, 1).is_none());
    }

    #[test]
    fn drop_gives_the_whole_block_back_to_the_parent() {
        // Indirect check: allocate many backed stacks in a loop against
        // the real heap. If `drop` didn't correctly return each block,
        // a long enough loop would be a real, growing leak rather than
        // steady-state memory use -- this doesn't measure that
        // directly, but it does confirm `drop` runs without UB/double
        // free across many real alloc/dealloc round trips against the
        // actual global allocator, not a fake one.
        let heap = HeapAlloc;
        for _ in 0..10_000 {
            let s = BackedStack::new(&heap, 128, 8).unwrap();
            s.alloc(0u64).unwrap();
            drop(s);
        }
    }

    #[test]
    fn a_stack_can_be_backed_by_another_stack() {
        // The real, motivating case: carve a smaller stack out of a
        // bigger one instead of both reaching for the global heap
        // independently.
        use crate::stack_allocator::StackAllocator;
        let big = StackAllocator::with_capacity(256);
        let small = BackedStack::new(&big, 32, 8).expect("big has plenty of room for 32 bytes");
        assert_eq!(
            big.used(),
            32,
            "the small stack's whole block was carved out of big"
        );
        let v = small.alloc(7u32).unwrap();
        assert_eq!(*v, 7);
        drop(small);
        assert_eq!(
            big.used(),
            32,
            "StackAllocator's try_dealloc_raw is a real no-op (see its own RawAlloc impl) -- \
             the block stays reserved in big until big itself rewinds/resets, it is not \
             reclaimed just because the BackedStack carved from it dropped"
        );
    }
}
