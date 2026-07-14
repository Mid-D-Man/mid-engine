// crates/mid-math/src/mid_vec/iter.rs
//! Owned iterator over a `MidVec<T, N>`, yielding elements by value.

use core::mem::ManuallyDrop;

use super::raw::{dealloc_heap, RawMidVec};
use super::MidVec;

/// Owned, by-value iterator produced by `MidVec::into_iter()`.
///
/// Drains front-to-back (or back-to-front via `DoubleEndedIterator`) and,
/// on drop, disposes of any elements that were never yielded plus the
/// heap allocation, if any — same shape as `std::vec::IntoIter`.
pub struct IntoIter<T, const N: usize> {
    raw: RawMidVec<T, N>,
    on_heap: bool,
    start: usize,
    end: usize,
}

impl<T, const N: usize> IntoIter<T, N> {
    pub(super) fn new(vec: MidVec<T, N>) -> Self {
        let len = vec.len();
        let on_heap = vec.spilled();
        let mut this = ManuallyDrop::new(vec);
        // SAFETY: `this` is wrapped in `ManuallyDrop` and never touched
        // again, so bitwise-copying `raw` out of it here does not lead to
        // a double-free/drop of whatever it owns — ownership has simply
        // moved from `this.raw` to the `raw` field below.
        let raw = unsafe { core::ptr::read(&mut this.raw) };
        Self { raw, on_heap, start: 0, end: len }
    }

    #[inline]
    fn base_ptr(&self) -> *const T {
        if self.on_heap {
            // SAFETY: `heap` is the active variant.
            unsafe { self.raw.heap_parts().0.as_ptr() }
        } else {
            // SAFETY: `inline` is the active variant.
            unsafe { self.raw.as_ptr_inline() }
        }
    }

    #[inline]
    fn base_mut_ptr(&mut self) -> *mut T {
        if self.on_heap {
            // SAFETY: `heap` is the active variant.
            unsafe { self.raw.heap_parts().0.as_ptr() }
        } else {
            // SAFETY: `inline` is the active variant.
            unsafe { self.raw.as_mut_ptr_inline() }
        }
    }

    /// Remaining elements as a slice, without consuming them — handy for
    /// e.g. `Debug` or just inspecting what's left mid-iteration.
    pub fn as_slice(&self) -> &[T] {
        let len = self.end - self.start;
        // SAFETY: `[start, end)` is exactly the remaining, not-yet-yielded,
        // initialised range.
        unsafe { core::slice::from_raw_parts(self.base_ptr().add(self.start), len) }
    }
}

impl<T, const N: usize> Iterator for IntoIter<T, N> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<T> {
        if self.start == self.end {
            return None;
        }
        let index = self.start;
        self.start += 1;
        // SAFETY: `index` was within `[start, end)`, the remaining
        // initialised range, and is now excluded from it by the increment
        // above — so this is the one and only read of that slot.
        Some(unsafe { core::ptr::read(self.base_ptr().add(index)) })
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.end - self.start;
        (remaining, Some(remaining))
    }
}

impl<T, const N: usize> DoubleEndedIterator for IntoIter<T, N> {
    #[inline]
    fn next_back(&mut self) -> Option<T> {
        if self.start == self.end {
            return None;
        }
        self.end -= 1;
        // SAFETY: `end` (post-decrement) was within `[start, end)` before
        // this call and is now excluded from it — one and only read.
        Some(unsafe { core::ptr::read(self.base_ptr().add(self.end)) })
    }
}

impl<T, const N: usize> ExactSizeIterator for IntoIter<T, N> {
    #[inline]
    fn len(&self) -> usize {
        self.end - self.start
    }
}

impl<T, const N: usize> Drop for IntoIter<T, N> {
    fn drop(&mut self) {
        let remaining = self.end - self.start;
        let base = self.base_mut_ptr();
        let on_heap = self.on_heap;

        // Same panic-safety shape as `MidVec::drop`: build the
        // deallocation guard before dropping elements so the heap buffer
        // is still freed even if a `T::drop` panics partway through.
        struct DeallocGuard<T> {
            ptr: core::ptr::NonNull<T>,
            capacity: usize,
        }
        impl<T> Drop for DeallocGuard<T> {
            fn drop(&mut self) {
                // SAFETY: only constructed below from a live heap
                // allocation belonging to this `IntoIter`.
                unsafe { dealloc_heap::<T>(self.ptr, self.capacity) };
            }
        }

        let _guard = if on_heap {
            // SAFETY: `heap` is the active variant.
            let (ptr, capacity) = unsafe { self.raw.heap_parts() };
            Some(DeallocGuard { ptr, capacity })
        } else {
            None
        };

        // SAFETY: `[start, end)` is exactly the not-yet-yielded,
        // initialised range remaining in the buffer.
        unsafe {
            core::ptr::drop_in_place(core::slice::from_raw_parts_mut(base.add(self.start), remaining));
        }
        // `_guard` drops here, after the elements above.
    }
  }
