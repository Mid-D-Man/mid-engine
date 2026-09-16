// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-ptr.md, section "const_non_null.rs"
// ============================================================================

use core::ptr::NonNull;

/// A newtype around [`NonNull`] that only allows conversion to read-only borrows or
/// pointers. Think of it as the `*const T` counterpart to `NonNull<T>`'s `*mut T`.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct ConstNonNull<T: ?Sized>(NonNull<T>);

impl<T: ?Sized> ConstNonNull<T> {
    /// Creates a new `ConstNonNull` if `ptr` is non-null.
    pub fn new(ptr: *const T) -> Option<Self> {
        NonNull::new(ptr.cast_mut()).map(Self)
    }

    /// Creates a new `ConstNonNull` without checking that `ptr` is non-null.
    ///
    /// # Safety
    /// `ptr` must be non-null.
    pub const unsafe fn new_unchecked(ptr: *const T) -> Self {
        // SAFETY: This function's safety invariants are identical to
        // `NonNull::new_unchecked`'s; the caller upholds them.
        unsafe { Self(NonNull::new_unchecked(ptr.cast_mut())) }
    }

    /// Returns a shared reference to the value.
    ///
    /// # Safety
    /// When calling this method, the caller has to ensure that all of the following
    /// hold:
    /// - The pointer must be properly aligned.
    /// - It must be dereferenceable in the sense the `core::ptr` documentation
    ///   defines.
    /// - The pointer must point to an initialized instance of `T`.
    /// - The caller must enforce Rust's aliasing rules, since the returned lifetime
    ///   `'a` is chosen arbitrarily and does not necessarily reflect the data's real
    ///   lifetime. While this reference exists, the pointee must not be mutated
    ///   (except inside an `UnsafeCell`).
    ///
    /// This applies even if the result of this method goes unused.
    #[inline]
    pub unsafe fn as_ref<'a>(&self) -> &'a T {
        // SAFETY: This function's safety invariants are identical to
        // `NonNull::as_ref`'s; the caller upholds them.
        unsafe { self.0.as_ref() }
    }
}

impl<T: ?Sized> From<NonNull<T>> for ConstNonNull<T> {
    fn from(value: NonNull<T>) -> ConstNonNull<T> {
        ConstNonNull(value)
    }
}

impl<'a, T: ?Sized> From<&'a T> for ConstNonNull<T> {
    fn from(value: &'a T) -> ConstNonNull<T> {
        ConstNonNull(NonNull::from(value))
    }
}

impl<'a, T: ?Sized> From<&'a mut T> for ConstNonNull<T> {
    fn from(value: &'a mut T) -> ConstNonNull<T> {
        ConstNonNull(NonNull::from(value))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_rejects_null_and_accepts_non_null() {
        assert!(ConstNonNull::<u32>::new(core::ptr::null()).is_none());

        let x = 7u32;
        let ptr = ConstNonNull::<u32>::new(&x as *const u32);
        assert!(ptr.is_some());
    }

    #[test]
    fn as_ref_reads_through_to_the_original_value() {
        let x = 11u32;
        let ptr = ConstNonNull::new(&x as *const u32).unwrap();
        // SAFETY: `ptr` is aligned, dereferenceable, points at an initialized `u32`,
        // and `x` outlives this borrow.
        let r = unsafe { ptr.as_ref() };
        assert_eq!(*r, 11);
    }

    #[test]
    fn from_shared_and_mut_refs_round_trip() {
        let x = 5u32;
        let from_ref: ConstNonNull<u32> = (&x).into();
        // SAFETY: same as above.
        assert_eq!(unsafe { *from_ref.as_ref() }, 5);

        let mut y = 9u32;
        let from_mut: ConstNonNull<u32> = (&mut y).into();
        // SAFETY: same as above.
        assert_eq!(unsafe { *from_mut.as_ref() }, 9);
    }
}
