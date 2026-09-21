// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-platform.md, section "cell.rs"
// ============================================================================

//! Cell primitives with no `std`/`alloc` dependency at all — pure logic, no
//! locking involved.

use core::ptr;

/// A reimplementation of the currently unstable `std::sync::Exclusive`.
///
/// Provides a wrapper that allows making any type unconditionally [`Sync`] by
/// only providing mutable access.
#[repr(transparent)]
pub struct SyncCell<T: ?Sized> {
    inner: T,
}

impl<T: Sized> SyncCell<T> {
    /// Constructs a new instance of a `SyncCell` from the given value.
    pub fn new(inner: T) -> Self {
        Self { inner }
    }

    /// Deconstructs this `SyncCell` into its inner value.
    pub fn to_inner(Self { inner }: Self) -> T {
        inner
    }
}

impl<T: ?Sized> SyncCell<T> {
    /// Gets a reference to this `SyncCell`'s inner value.
    pub fn get(&mut self) -> &mut T {
        &mut self.inner
    }

    /// For types that implement [`Sync`], gets shared access to this
    /// `SyncCell`'s inner value.
    pub fn read(&self) -> &T
    where
        T: Sync,
    {
        &self.inner
    }

    /// Builds a mutable reference to a `SyncCell` from a mutable reference to
    /// its inner value, to skip constructing with [`new()`](SyncCell::new()).
    pub fn from_mut(r: &'_ mut T) -> &'_ mut SyncCell<T> {
        // SAFETY: repr is transparent, so refs have the same layout, and
        // `SyncCell`'s properties don't depend on how the `&mut` was obtained.
        unsafe { &mut *(ptr::from_mut(r) as *mut SyncCell<T>) }
    }
}

// SAFETY: `Sync` only allows multithreaded access via immutable reference. As
// `SyncCell` requires an exclusive reference to access the wrapped value for
// `!Sync` types, marking this type as `Sync` does not actually allow
// unsynchronized access to the inner value.
unsafe impl<T: ?Sized> Sync for SyncCell<T> {}

/// A reimplementation of the currently unstable `std::cell::SyncUnsafeCell`.
///
/// This is just an [`UnsafeCell`](core::cell::UnsafeCell), except it
/// implements [`Sync`] if `T` implements `Sync`.
///
/// `UnsafeCell` doesn't implement `Sync`, to prevent accidental misuse. Use
/// `SyncUnsafeCell` instead of `UnsafeCell` to allow it to be shared between
/// threads, if that's intentional. Providing proper synchronization is still
/// the caller's job — this type is just as unsafe to use as a plain
/// `UnsafeCell`.
#[repr(transparent)]
pub struct SyncUnsafeCell<T: ?Sized> {
    value: core::cell::UnsafeCell<T>,
}

// SAFETY: `T` is `Sync`; the caller is responsible for upholding Rust's
// aliasing rules when going through the raw pointer this type hands out.
unsafe impl<T: ?Sized + Sync> Sync for SyncUnsafeCell<T> {}

impl<T> SyncUnsafeCell<T> {
    /// Constructs a new instance of `SyncUnsafeCell` which will wrap the
    /// specified value.
    #[inline]
    pub const fn new(value: T) -> Self {
        Self {
            value: core::cell::UnsafeCell::new(value),
        }
    }

    /// Unwraps the value.
    #[inline]
    pub fn into_inner(self) -> T {
        self.value.into_inner()
    }
}

impl<T: ?Sized> SyncUnsafeCell<T> {
    /// Gets a mutable pointer to the wrapped value.
    ///
    /// This can be cast to a pointer of any kind. Ensure the access is unique
    /// (no active references, mutable or not) when casting to `&mut T`, and
    /// ensure there are no mutations or mutable aliases going on when casting
    /// to `&T`.
    #[inline]
    pub const fn get(&self) -> *mut T {
        self.value.get()
    }

    /// Returns a mutable reference to the underlying data.
    ///
    /// This call borrows the `SyncUnsafeCell` mutably (at compile time), which
    /// guarantees this is the only reference.
    #[inline]
    pub fn get_mut(&mut self) -> &mut T {
        self.value.get_mut()
    }

    /// Gets a mutable pointer to the wrapped value from a `*const Self`.
    ///
    /// See [`UnsafeCell::get`](core::cell::UnsafeCell::get) for details.
    #[inline]
    pub const fn raw_get(this: *const Self) -> *mut T {
        // Casting the pointer from `SyncUnsafeCell<T>` to `T` directly is
        // sound because of `#[repr(transparent)]` on both `SyncUnsafeCell`
        // and `UnsafeCell` — see `UnsafeCell::raw_get`.
        (this as *const T).cast_mut()
    }

    /// Returns a `&mut SyncUnsafeCell<T>` from a `&mut T`.
    #[inline]
    pub fn from_mut(t: &mut T) -> &mut SyncUnsafeCell<T> {
        let ptr = ptr::from_mut(t) as *mut SyncUnsafeCell<T>;
        // SAFETY: `ptr` is safe to mutably dereference, since it was obtained
        // from a mutable reference. `SyncUnsafeCell` has the same
        // representation as `T` itself, since it's `#[repr(transparent)]`.
        unsafe { &mut *ptr }
    }
}

impl<T> SyncUnsafeCell<[T]> {
    /// Returns a `&[SyncUnsafeCell<T>]` from a `&SyncUnsafeCell<[T]>`.
    ///
    /// ```
    /// # use mid_platform::cell::SyncUnsafeCell;
    /// let slice: &mut [i32] = &mut [1, 2, 3];
    /// let cell_slice: &SyncUnsafeCell<[i32]> = SyncUnsafeCell::from_mut(slice);
    /// let slice_cell: &[SyncUnsafeCell<i32>] = cell_slice.as_slice_of_cells();
    ///
    /// assert_eq!(slice_cell.len(), 3);
    /// ```
    pub fn as_slice_of_cells(&self) -> &[SyncUnsafeCell<T>] {
        let self_ptr: *const SyncUnsafeCell<[T]> = ptr::from_ref(self);
        let slice_ptr = self_ptr as *const [SyncUnsafeCell<T>];
        // SAFETY: both `UnsafeCell<T>` and `SyncUnsafeCell<T>` are
        // `#[repr(transparent)]`, so: `SyncUnsafeCell<T>` has the same layout
        // as `T`; `SyncUnsafeCell<[T]>` has the same layout as `[T]`; and
        // `SyncUnsafeCell<[T]>` has the same layout as `[SyncUnsafeCell<T>]`.
        unsafe { &*slice_ptr }
    }
}

impl<T: Default> Default for SyncUnsafeCell<T> {
    fn default() -> SyncUnsafeCell<T> {
        SyncUnsafeCell::new(Default::default())
    }
}

impl<T> From<T> for SyncUnsafeCell<T> {
    fn from(t: T) -> SyncUnsafeCell<T> {
        SyncUnsafeCell::new(t)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sync_cell_get_and_read_reach_the_same_value() {
        let mut cell = SyncCell::new(5u32);
        *cell.get() = 6;
        assert_eq!(*cell.read(), 6);
    }

    #[test]
    fn sync_cell_to_inner_returns_the_wrapped_value() {
        let cell = SyncCell::new(7u32);
        assert_eq!(SyncCell::to_inner(cell), 7);
    }

    #[test]
    fn sync_unsafe_cell_get_mut_and_raw_get_agree() {
        let mut cell = SyncUnsafeCell::new(3u32);
        *cell.get_mut() = 4;
        // SAFETY: sole access to `cell` in this test; no other reference is
        // alive at the point of this read.
        assert_eq!(unsafe { *SyncUnsafeCell::raw_get(&cell) }, 4);
    }

    #[test]
    fn sync_unsafe_cell_as_slice_of_cells_has_the_right_length() {
        let slice: &mut [i32] = &mut [1, 2, 3];
        let cell_slice: &SyncUnsafeCell<[i32]> = SyncUnsafeCell::from_mut(slice);
        let slice_cell: &[SyncUnsafeCell<i32>] = cell_slice.as_slice_of_cells();
        assert_eq!(slice_cell.len(), 3);
    }
}
