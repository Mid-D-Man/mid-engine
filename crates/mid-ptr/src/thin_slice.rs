// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-ptr.md, section "thin_slice.rs"
// ============================================================================

use core::cell::UnsafeCell;
use core::marker::PhantomData;
use core::ops::Range;
use core::ptr::NonNull;

use crate::debug_align::DebugEnsureAligned;

/// Conceptually equivalent to `&'a [T]`, but with the length information cut out for
/// performance reasons.
///
/// Because this type does not store the length of the slice, it cannot do any sort
/// of bounds checking on its own. As such, only [`Self::get_unchecked`] is available
/// for indexing — the caller is responsible for checking bounds itself.
///
/// When compiled in debug mode, this type does store the length of the slice and
/// performs bounds checking in [`Self::get_unchecked`] anyway, purely as a
/// debug-build safety net.
///
/// # Example
///
/// ```
/// # use core::mem::size_of;
/// # use mid_ptr::ThinSlicePtr;
/// #
/// let slice: &[u32] = &[2, 4, 8];
/// let thin_slice = ThinSlicePtr::from(slice);
///
/// assert_eq!(*unsafe { thin_slice.get_unchecked(0) }, 2);
/// assert_eq!(*unsafe { thin_slice.get_unchecked(1) }, 4);
/// assert_eq!(*unsafe { thin_slice.get_unchecked(2) }, 8);
/// ```
pub struct ThinSlicePtr<'a, T> {
    ptr: NonNull<T>,
    #[cfg(debug_assertions)]
    len: usize,
    _marker: PhantomData<&'a [T]>,
}

impl<'a, T> ThinSlicePtr<'a, T> {
    /// Indexes the slice without performing bounds checks.
    ///
    /// # Safety
    /// `index` must be in-bounds.
    #[inline]
    pub unsafe fn get_unchecked(&self, index: usize) -> &'a T {
        // `debug_assert!` can't be used here since `self.len` doesn't exist outside
        // debug builds.
        #[cfg(debug_assertions)]
        assert!(index < self.len, "tried to index out-of-bounds of a slice");

        // SAFETY: The caller guarantees `index` is in-bounds, so the resulting
        // pointer is valid to dereference.
        unsafe { &*self.ptr.add(index).as_ptr() }
    }

    /// Returns a slice without performing bounds checks.
    ///
    /// # Safety
    /// - There must be no mutable aliases for the lifetime `'a` to the slice.
    /// - `len` must be less than or equal to the slice's real length.
    pub unsafe fn as_slice_unchecked(&self, len: usize) -> &'a [T] {
        #[cfg(debug_assertions)]
        assert!(len <= self.len, "tried to create an out-of-bounds slice");

        // SAFETY:
        // - The caller guarantees `len` is not greater than the slice's real length.
        // - The caller upholds the aliasing rules.
        // - `self.ptr` is a valid pointer for type `T`.
        // - `len` is valid, so `len * size_of::<T>()` is less than `isize::MAX`.
        unsafe { core::slice::from_raw_parts(self.ptr.as_ptr(), len) }
    }

    /// Returns a subslice without performing bounds checks.
    ///
    /// # Safety
    /// - There must be no mutable aliases for the lifetime `'a` to the slice.
    /// - `range.start` and `range.end` must be less than or equal to the slice's
    ///   real length.
    /// - `range.start` must be less than or equal to `range.end`.
    pub unsafe fn slice_unchecked(&self, range: Range<usize>) -> &'a [T] {
        // SAFETY: The caller guarantees `range` is within bounds of the slice.
        unsafe {
            core::slice::from_raw_parts(self.ptr.as_ptr().add(range.start), range.end - range.start)
        }
    }
}

impl<'a, T> ThinSlicePtr<'a, UnsafeCell<T>> {
    /// Returns a mutable reference to the slice.
    ///
    /// # Safety
    /// - There must be no other aliases for the lifetime `'a` to the slice.
    /// - `len` must be less than or equal to the slice's real length.
    pub unsafe fn as_mut_slice_unchecked(&self, len: usize) -> &'a mut [T] {
        #[cfg(debug_assertions)]
        assert!(len <= self.len, "tried to create an out-of-bounds slice");

        // SAFETY:
        // - The caller guarantees no aliases exist and that `len` is in-bounds.
        // - `self.ptr` is a valid pointer for type `T`.
        // - `len` is valid, so `len * size_of::<T>()` is less than `isize::MAX`.
        unsafe { core::slice::from_raw_parts_mut(UnsafeCell::raw_get(self.ptr.as_ptr()), len) }
    }

    /// Returns a mutable subslice of the slice.
    ///
    /// # Safety
    /// - There must be no other aliases for the lifetime `'a` to the slice.
    /// - `range.start` and `range.end` must be less than or equal to the slice's
    ///   real length.
    /// - `range.start` must be less than or equal to `range.end`.
    pub unsafe fn slice_mut_unchecked(&self, range: Range<usize>) -> &'a mut [T] {
        // SAFETY: The caller guarantees `range` is within bounds of the slice.
        unsafe {
            core::slice::from_raw_parts_mut(
                UnsafeCell::raw_get(self.ptr.as_ptr().add(range.start)),
                range.end - range.start,
            )
        }
    }

    /// Returns a slice pointer to the underlying type `T`, dropping the
    /// `UnsafeCell` wrapper.
    pub fn cast(&self) -> ThinSlicePtr<'a, T> {
        ThinSlicePtr {
            // SAFETY: `self.ptr` is non-null, so `UnsafeCell::raw_get` always
            // returns a non-null pointer too.
            ptr: unsafe { NonNull::new_unchecked(UnsafeCell::raw_get(self.ptr.as_ptr())) },
            #[cfg(debug_assertions)]
            len: self.len,
            _marker: PhantomData,
        }
    }
}

impl<'a, T> Clone for ThinSlicePtr<'a, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, T> Copy for ThinSlicePtr<'a, T> {}

impl<'a, T> From<&'a [T]> for ThinSlicePtr<'a, T> {
    #[inline]
    fn from(slice: &'a [T]) -> Self {
        let ptr = slice.as_ptr().cast_mut().debug_ensure_aligned();

        Self {
            // SAFETY: A reference can never be null.
            ptr: unsafe { NonNull::new_unchecked(ptr) },
            #[cfg(debug_assertions)]
            len: slice.len(),
            _marker: PhantomData,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn get_unchecked_reads_every_element() {
        let slice: &[u32] = &[2, 4, 8];
        let thin = ThinSlicePtr::from(slice);
        // SAFETY: indices 0..3 are all in-bounds for a 3-element slice.
        unsafe {
            assert_eq!(*thin.get_unchecked(0), 2);
            assert_eq!(*thin.get_unchecked(1), 4);
            assert_eq!(*thin.get_unchecked(2), 8);
        }
    }

    #[test]
    fn as_slice_unchecked_reconstructs_the_original_slice() {
        let slice: &[u32] = &[10, 20, 30, 40];
        let thin = ThinSlicePtr::from(slice);
        // SAFETY: `4` is exactly the real length, and there are no mutable aliases.
        let back = unsafe { thin.as_slice_unchecked(4) };
        assert_eq!(back, slice);
    }

    #[test]
    fn slice_unchecked_returns_the_requested_subrange() {
        let slice: &[u32] = &[10, 20, 30, 40];
        let thin = ThinSlicePtr::from(slice);
        // SAFETY: `1..3` is within bounds and start <= end.
        let sub = unsafe { thin.slice_unchecked(1..3) };
        assert_eq!(sub, &[20, 30]);
    }

    #[test]
    fn unsafe_cell_slice_mut_and_cast_round_trip() {
        let cells: [UnsafeCell<u32>; 3] = [UnsafeCell::new(1), UnsafeCell::new(2), UnsafeCell::new(3)];
        let thin: ThinSlicePtr<'_, UnsafeCell<u32>> = ThinSlicePtr::from(cells.as_slice());

        // SAFETY: sole access to this `ThinSlicePtr` in this test, `3` is the real
        // length.
        let mutable = unsafe { thin.as_mut_slice_unchecked(3) };
        mutable[1] = 99;

        let plain = thin.cast();
        // SAFETY: no aliasing mutable access remains once `mutable` above went out
        // of scope, `3` is the real length.
        let read_back = unsafe { plain.as_slice_unchecked(3) };
        assert_eq!(read_back, &[1, 99, 3]);
    }
}
