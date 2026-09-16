// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-ptr.md, section "unsafe_cell.rs"
// ============================================================================

use core::cell::UnsafeCell;

mod private {
    use core::cell::UnsafeCell;

    pub trait SealedUnsafeCell {}
    impl<'a, T> SealedUnsafeCell for &'a UnsafeCell<T> {}
}

/// Extension trait adding a few safe-looking helper methods on top of
/// [`UnsafeCell`], for call sites that have already established the aliasing
/// invariants some other way (typically because the surrounding erased-pointer type
/// already tracks them).
pub trait UnsafeCellDeref<'a, T>: private::SealedUnsafeCell {
    /// # Safety
    /// - The returned value must be unique and must not alias any mutable or
    ///   immutable reference to the `UnsafeCell`'s contents.
    /// - At all times, data races must be avoided: if multiple threads can reach the
    ///   same `UnsafeCell`, writes need a proper happens-before relation to every
    ///   other access, or need to go through atomics (see the [`UnsafeCell`] docs).
    unsafe fn deref_mut(self) -> &'a mut T;

    /// # Safety
    /// - For the returned value's lifetime `'a`, no mutable reference to the
    ///   `UnsafeCell`'s contents may be constructed.
    /// - At all times, data races must be avoided, as above.
    unsafe fn deref(self) -> &'a T;

    /// Returns a copy of the contained value.
    ///
    /// # Safety
    /// - The `UnsafeCell` must not currently have a live mutable reference to its
    ///   contents.
    /// - At all times, data races must be avoided, as above.
    unsafe fn read(self) -> T
    where
        T: Copy;
}

impl<'a, T> UnsafeCellDeref<'a, T> for &'a UnsafeCell<T> {
    #[inline]
    unsafe fn deref_mut(self) -> &'a mut T {
        // SAFETY: The caller upholds the aliasing rules documented on the trait.
        unsafe { &mut *self.get() }
    }

    #[inline]
    unsafe fn deref(self) -> &'a T {
        // SAFETY: The caller upholds the aliasing rules documented on the trait.
        unsafe { &*self.get() }
    }

    #[inline]
    unsafe fn read(self) -> T
    where
        T: Copy,
    {
        // SAFETY: The caller upholds the aliasing rules documented on the trait.
        unsafe { self.get().read() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deref_and_deref_mut_reach_the_same_cell() {
        let cell = UnsafeCell::new(5u32);
        // SAFETY: sole access to `cell` in this test; no other reference is alive.
        unsafe {
            *(&cell).deref_mut() = 6;
        }
        // SAFETY: the mutable access above has already ended.
        assert_eq!(*unsafe { (&cell).deref() }, 6);
    }

    #[test]
    fn read_returns_a_copy_without_disturbing_the_cell() {
        let cell = UnsafeCell::new(3u32);
        // SAFETY: no mutable reference to `cell` is live at this point.
        let copy = unsafe { (&cell).read() };
        assert_eq!(copy, 3);
        // SAFETY: same as above.
        assert_eq!(*unsafe { (&cell).deref() }, 3);
    }
}
