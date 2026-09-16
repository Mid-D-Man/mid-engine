// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-ptr.md, section "aligned.rs"
// ============================================================================

use core::mem::size_of;
use core::ptr;

/// Used as a type argument to [`crate::Ptr`], [`crate::PtrMut`], [`crate::OwningPtr`],
/// and [`crate::MovingPtr`] to state that the pointer is guaranteed to be properly
/// aligned for its pointee type.
#[derive(Debug, Copy, Clone)]
pub struct Aligned;

/// Used as a type argument to [`crate::Ptr`], [`crate::PtrMut`], [`crate::OwningPtr`],
/// and [`crate::MovingPtr`] to state that the pointer may not be aligned for its
/// pointee type.
#[derive(Debug, Copy, Clone)]
pub struct Unaligned;

/// Sealed to [`Aligned`] and [`Unaligned`] only — a workaround for not being able to
/// write a const generic over an enum. Each read/copy/drop operation the erased
/// pointer types need is dispatched through here so the aligned/unaligned split only
/// has to be written once.
pub trait IsAligned: sealed::Sealed {
    /// Reads the value pointed to by `ptr`.
    ///
    /// # Safety
    /// - `ptr` must be valid for reads.
    /// - `ptr` must point to a valid instance of type `T`.
    /// - If this type is [`Aligned`], `ptr` must be properly aligned for type `T`.
    #[doc(hidden)]
    unsafe fn read_ptr<T>(ptr: *const T) -> T;

    /// Copies `count * size_of::<T>()` bytes from `src` to `dst`. The source and
    /// destination must not overlap.
    ///
    /// # Safety
    /// - `src` must be valid for reads of `count * size_of::<T>()` bytes.
    /// - `dst` must be valid for writes of `count * size_of::<T>()` bytes.
    /// - The `src` and `dst` regions must not overlap.
    /// - If this type is [`Aligned`], both `src` and `dst` must be properly aligned
    ///   for values of type `T`.
    #[doc(hidden)]
    unsafe fn copy_nonoverlapping<T>(src: *const T, dst: *mut T, count: usize);

    /// Drops the value pointed to by `ptr` in place.
    ///
    /// # Safety
    /// - `ptr` must be valid for reads and writes.
    /// - `ptr` must point to a valid instance of type `T`.
    /// - If this type is [`Aligned`], `ptr` must be properly aligned for type `T`.
    /// - The value behind `ptr` must be valid for dropping.
    /// - While this call is executing, the only way to touch the pointee is through
    ///   the `&mut Self` handed to `Drop::drop`.
    #[doc(hidden)]
    unsafe fn drop_in_place<T>(ptr: *mut T);
}

impl IsAligned for Aligned {
    #[inline]
    unsafe fn read_ptr<T>(ptr: *const T) -> T {
        // SAFETY: The caller upholds every precondition `IsAligned::read_ptr` documents;
        // `Self` being `Aligned` covers the alignment requirement `ptr::read` adds on
        // top of a plain unaligned read.
        unsafe { ptr.read() }
    }

    #[inline]
    unsafe fn copy_nonoverlapping<T>(src: *const T, dst: *mut T, count: usize) {
        // SAFETY: The caller upholds every precondition `IsAligned::copy_nonoverlapping`
        // documents, including alignment for both `src` and `dst` since `Self` is `Aligned`.
        unsafe {
            ptr::copy_nonoverlapping(src, dst, count);
        }
    }

    #[inline]
    unsafe fn drop_in_place<T>(ptr: *mut T) {
        // SAFETY: The caller upholds every precondition `IsAligned::drop_in_place`
        // documents, including alignment since `Self` is `Aligned`.
        unsafe {
            ptr::drop_in_place(ptr);
        }
    }
}

impl IsAligned for Unaligned {
    #[inline]
    unsafe fn read_ptr<T>(ptr: *const T) -> T {
        // SAFETY: The caller upholds every precondition except alignment, which
        // `read_unaligned` does not require.
        unsafe { ptr.read_unaligned() }
    }

    #[inline]
    unsafe fn copy_nonoverlapping<T>(src: *const T, dst: *mut T, count: usize) {
        // SAFETY: A byte-wise copy needs no alignment for either side, so the caller
        // only has to uphold validity and non-overlap, both required regardless of
        // whether `Self` is `Aligned` or `Unaligned`.
        unsafe {
            ptr::copy_nonoverlapping::<u8>(
                src.cast::<u8>(),
                dst.cast::<u8>(),
                count * size_of::<T>(),
            );
        }
    }

    #[inline]
    unsafe fn drop_in_place<T>(ptr: *mut T) {
        // SAFETY: `read_unaligned` needs no alignment; the caller still has to uphold
        // validity, and dropping the read-out value runs `T`'s own `Drop` impl exactly
        // once, matching `ptr::drop_in_place`'s contract.
        unsafe {
            drop(ptr.read_unaligned());
        }
    }
}

mod sealed {
    pub trait Sealed {}
    impl Sealed for super::Aligned {}
    impl Sealed for super::Unaligned {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aligned_read_ptr_round_trips() {
        let x = 42u32;
        // SAFETY: `&x` is valid, initialized, and aligned for `u32`.
        let y = unsafe { Aligned::read_ptr(&x as *const u32) };
        assert_eq!(x, y);
    }

    #[test]
    fn unaligned_read_ptr_round_trips_from_a_byte_buffer() {
        let mut buf = [0u8; 16];
        let value: u32 = 0xDEAD_BEEF;
        // Deliberately write at a one-byte offset so the field is misaligned for u32.
        buf[1..5].copy_from_slice(&value.to_ne_bytes());
        let misaligned = buf.as_ptr().wrapping_add(1).cast::<u32>();
        // SAFETY: `misaligned` points at 4 valid, initialized bytes within `buf`,
        // just not aligned for `u32` — exactly what `Unaligned` is for.
        let read_back = unsafe { Unaligned::read_ptr(misaligned) };
        assert_eq!(read_back, value);
    }

    #[test]
    fn aligned_copy_nonoverlapping_moves_the_right_bytes() {
        let src = [1u32, 2, 3, 4];
        let mut dst = [0u32; 4];
        // SAFETY: `src`/`dst` are both valid, aligned, non-overlapping `u32` buffers
        // of at least 4 elements.
        unsafe { Aligned::copy_nonoverlapping(src.as_ptr(), dst.as_mut_ptr(), 4) };
        assert_eq!(src, dst);
    }
}
