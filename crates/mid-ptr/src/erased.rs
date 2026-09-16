// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-ptr.md, section "erased.rs"
// ============================================================================

use core::fmt::{self, Debug, Formatter, Pointer};
use core::marker::PhantomData;
use core::mem::ManuallyDrop;
use core::ptr::NonNull;

use crate::aligned::{Aligned, IsAligned, Unaligned};
use crate::debug_align::DebugEnsureAligned;
use crate::moving::MovingPtr;

/// Type-erased borrow of some unknown type chosen when constructing this value.
///
/// Acts "borrow-like":
/// - Considered immutable: its target must not change while this pointer is alive.
/// - Must always point to a valid value of whatever the pointee type is.
/// - The lifetime `'a` accurately represents how long the pointer stays valid.
/// - If `A` is [`Aligned`], the pointer must always be properly aligned for the
///   unknown pointee type.
///
/// Similar in spirit to `&'a dyn Any` but without the vtable metadata, and able to
/// point at data with no corresponding Rust type at all.
#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct Ptr<'a, A: IsAligned = Aligned>(NonNull<u8>, PhantomData<(&'a u8, A)>);

/// Type-erased mutable borrow of some unknown type chosen when constructing this
/// value.
///
/// Acts "borrow-like":
/// - Considered exclusive and mutable; cannot be cloned, since that would alias a
///   mutable pointer.
/// - Must always point to a valid value of whatever the pointee type is.
/// - The lifetime `'a` accurately represents how long the pointer stays valid.
/// - If `A` is [`Aligned`], the pointer must always be properly aligned for the
///   unknown pointee type.
///
/// Similar in spirit to `&'a mut dyn Any` but without the vtable metadata.
#[repr(transparent)]
pub struct PtrMut<'a, A: IsAligned = Aligned>(NonNull<u8>, PhantomData<(&'a mut u8, A)>);

/// Type-erased, `Box`-like pointer to some unknown type chosen when constructing this
/// value.
///
/// Represents ownership of whatever data it points to and is responsible for calling
/// that data's `Drop` impl — but it is *not* responsible for freeing the memory it
/// points to, since that memory may belong to a slot in a larger allocation (a table
/// column, a `Vec` element, a stack local) rather than to this pointer alone.
///
/// Acts "borrow-like":
/// - Considered exclusive and mutable; cannot be cloned, since that risks aliased
///   mutability and use-after-free.
/// - Must always point to a valid value of whatever the pointee type is.
/// - The lifetime `'a` accurately represents how long the pointer stays valid.
/// - If `A` is [`Aligned`], the pointer must always be properly aligned for the
///   unknown pointee type.
#[repr(transparent)]
pub struct OwningPtr<'a, A: IsAligned = Aligned>(NonNull<u8>, PhantomData<(&'a mut u8, A)>);

macro_rules! impl_ptr {
    ($ptr:ident) => {
        impl<'a> $ptr<'a, Aligned> {
            /// Removes the alignment requirement of this pointer.
            pub fn to_unaligned(self) -> $ptr<'a, Unaligned> {
                $ptr(self.0, PhantomData)
            }
        }

        impl<'a, A: IsAligned> From<$ptr<'a, A>> for NonNull<u8> {
            fn from(ptr: $ptr<'a, A>) -> Self {
                ptr.0
            }
        }

        impl<A: IsAligned> $ptr<'_, A> {
            /// Calculates the offset from a pointer. Since the pointer is
            /// type-erased, no size information is available — `count` is always in
            /// raw bytes.
            ///
            /// *See also: [`ptr::offset`][ptr_offset]*
            ///
            /// # Safety
            /// - The offset cannot make the existing pointer null, or take it out of
            ///   bounds for its allocation.
            /// - If `A` is [`Aligned`], the offset must not make the resulting
            ///   pointer unaligned for the pointee type.
            /// - The value the resulting pointer points to must outlive this
            ///   pointer's own lifetime.
            ///
            /// [ptr_offset]: https://doc.rust-lang.org/std/primitive.pointer.html#method.offset
            #[inline]
            pub unsafe fn byte_offset(self, count: isize) -> Self {
                Self(
                    // SAFETY: The caller upholds safety for `offset` and ensures the
                    // result is not null.
                    unsafe { NonNull::new_unchecked(self.as_ptr().offset(count)) },
                    PhantomData,
                )
            }

            /// Calculates the offset from a pointer — convenience for
            /// `.byte_offset(count as isize)`. Since the pointer is type-erased, no
            /// size information is available — `count` is always in raw bytes.
            ///
            /// *See also: [`ptr::add`][ptr_add]*
            ///
            /// # Safety
            /// Same as [`byte_offset`](Self::byte_offset).
            ///
            /// [ptr_add]: https://doc.rust-lang.org/std/primitive.pointer.html#method.add
            #[inline]
            pub unsafe fn byte_add(self, count: usize) -> Self {
                Self(
                    // SAFETY: The caller upholds safety for `add` and ensures the
                    // result is not null.
                    unsafe { NonNull::new_unchecked(self.as_ptr().add(count)) },
                    PhantomData,
                )
            }
        }

        impl<A: IsAligned> Pointer for $ptr<'_, A> {
            #[inline]
            fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
                Pointer::fmt(&self.0, f)
            }
        }

        impl Debug for $ptr<'_, Aligned> {
            #[inline]
            fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
                write!(f, "{}<Aligned>({:?})", stringify!($ptr), self.0)
            }
        }

        impl Debug for $ptr<'_, Unaligned> {
            #[inline]
            fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
                write!(f, "{}<Unaligned>({:?})", stringify!($ptr), self.0)
            }
        }
    };
}

impl_ptr!(Ptr);
impl_ptr!(PtrMut);
impl_ptr!(OwningPtr);

impl<'a, A: IsAligned> Ptr<'a, A> {
    /// Creates a new instance from a raw pointer.
    ///
    /// # Safety
    /// - `inner` must point to a valid value of whatever the pointee type is.
    /// - If `A` is [`Aligned`], `inner` must be properly aligned for the pointee
    ///   type.
    /// - `inner` must have correct provenance to allow reads of the pointee type.
    /// - The lifetime `'a` must be constrained such that this `Ptr` stays valid and
    ///   nothing can mutate the pointee while it's live, except through an
    ///   `UnsafeCell`.
    #[inline]
    pub unsafe fn new(inner: NonNull<u8>) -> Self {
        Self(inner, PhantomData)
    }

    /// Transforms this `Ptr` into a [`PtrMut`].
    ///
    /// # Safety
    /// - The data this `Ptr` points to must be valid for writes.
    /// - There must be no active references (mutable or otherwise) to the underlying
    ///   data.
    /// - No other `PtrMut` for the same data may be created until the first is
    ///   dropped.
    #[inline]
    pub unsafe fn assert_unique(self) -> PtrMut<'a, A> {
        PtrMut(self.0, PhantomData)
    }

    /// Transforms this `Ptr<T>` into a `&T` with the same lifetime.
    ///
    /// # Safety
    /// - `T` must be the erased pointee type for this `Ptr`.
    /// - If `A` is [`Unaligned`], this pointer must still be properly aligned for `T`
    ///   at the point of this call.
    #[inline]
    pub unsafe fn deref<T>(self) -> &'a T {
        let ptr = self.as_ptr().cast::<T>().debug_ensure_aligned();
        // SAFETY: The caller ensures the pointee is `T` and that the pointer can be
        // dereferenced.
        unsafe { &*ptr }
    }

    /// Gets the underlying pointer, erasing the associated lifetime.
    ///
    /// Prefer [`deref`](Self::deref) where possible, since it retains the lifetime.
    #[inline]
    pub fn as_ptr(self) -> *mut u8 {
        self.0.as_ptr()
    }
}

impl<'a, T: ?Sized> From<&'a T> for Ptr<'a> {
    #[inline]
    fn from(val: &'a T) -> Self {
        // SAFETY: The returned pointer has the same lifetime as the reference passed
        // in, and access stays immutable.
        unsafe { Self::new(NonNull::from(val).cast()) }
    }
}

impl<'a, A: IsAligned> PtrMut<'a, A> {
    /// Creates a new instance from a raw pointer.
    ///
    /// # Safety
    /// - `inner` must point to a valid value of whatever the pointee type is.
    /// - If `A` is [`Aligned`], `inner` must be properly aligned for the pointee
    ///   type.
    /// - `inner` must have correct provenance to allow reads and writes of the
    ///   pointee type.
    /// - The lifetime `'a` must be constrained such that this `PtrMut` stays valid
    ///   and nothing else can read or mutate the pointee while it's live.
    #[inline]
    pub unsafe fn new(inner: NonNull<u8>) -> Self {
        Self(inner, PhantomData)
    }

    /// Transforms this `PtrMut` into an [`OwningPtr`].
    ///
    /// # Safety
    /// The caller must have the right to drop or move out of this `PtrMut`.
    #[inline]
    pub unsafe fn promote(self) -> OwningPtr<'a, A> {
        OwningPtr(self.0, PhantomData)
    }

    /// Transforms this `PtrMut<T>` into a `&mut T` with the same lifetime.
    ///
    /// # Safety
    /// - `T` must be the erased pointee type for this `PtrMut`.
    /// - If `A` is [`Unaligned`], this pointer must still be properly aligned for `T`
    ///   at the point of this call.
    #[inline]
    pub unsafe fn deref_mut<T>(self) -> &'a mut T {
        let ptr = self.as_ptr().cast::<T>().debug_ensure_aligned();
        // SAFETY: The caller ensures the pointee is `T` and that the pointer can be
        // dereferenced.
        unsafe { &mut *ptr }
    }

    /// Gets the underlying pointer, erasing the associated lifetime.
    ///
    /// Prefer [`deref_mut`](Self::deref_mut) where possible, since it retains the
    /// lifetime.
    #[inline]
    pub fn as_ptr(&self) -> *mut u8 {
        self.0.as_ptr()
    }

    /// Gets a `PtrMut` from this one with a shorter lifetime.
    #[inline]
    pub fn reborrow(&mut self) -> PtrMut<'_, A> {
        // SAFETY: The `PtrMut` being reborrowed from is assumed to already be valid.
        unsafe { PtrMut::new(self.0) }
    }

    /// Gets an immutable pointer from this mutable one.
    #[inline]
    pub fn as_ref(&self) -> Ptr<'_, A> {
        // SAFETY: `PtrMut`'s validity guarantees are a superset of `Ptr`'s.
        unsafe { Ptr::new(self.0) }
    }
}

impl<'a, T: ?Sized> From<&'a mut T> for PtrMut<'a> {
    #[inline]
    fn from(val: &'a mut T) -> Self {
        // SAFETY: The returned pointer has the same lifetime as the reference passed
        // in; the reference is mutable, so it cannot alias.
        unsafe { Self::new(NonNull::from(val).cast()) }
    }
}

impl<'a> OwningPtr<'a> {
    /// Exists mostly to cut compile times: the duplicated code below is generated
    /// once per pointee type rather than once per call site.
    ///
    /// # Safety
    /// The safety constraints of [`PtrMut::promote`] must be upheld.
    unsafe fn make_internal<T>(temp: &mut ManuallyDrop<T>) -> OwningPtr<'_> {
        // SAFETY: The constraints of `promote` are upheld by the caller.
        unsafe { PtrMut::from(&mut *temp).promote() }
    }

    /// Consumes a value and creates an `OwningPtr` to it, without ever risking a
    /// double drop.
    #[inline]
    pub fn make<T, F: FnOnce(OwningPtr<'_>) -> R, R>(val: T, f: F) -> R {
        let mut val = ManuallyDrop::new(val);
        // SAFETY: The value behind the pointer is never dropped or observed later
        // by this function, so promoting it to an owning pointer is sound.
        f(unsafe { Self::make_internal(&mut val) })
    }
}

impl<'a, A: IsAligned> OwningPtr<'a, A> {
    /// Creates a new instance from a raw pointer.
    ///
    /// # Safety
    /// - `inner` must point to a valid value of whatever the pointee type is.
    /// - If `A` is [`Aligned`], `inner` must be properly aligned for the pointee
    ///   type.
    /// - `inner` must have correct provenance to allow reads and writes of the
    ///   pointee type.
    /// - The lifetime `'a` must be constrained such that this `OwningPtr` stays
    ///   valid and nothing else can read or mutate the pointee while it's live.
    #[inline]
    pub unsafe fn new(inner: NonNull<u8>) -> Self {
        Self(inner, PhantomData)
    }

    /// Consumes this `OwningPtr` to obtain ownership of the underlying data of type
    /// `T`.
    ///
    /// # Safety
    /// - `T` must be the erased pointee type for this `OwningPtr`.
    /// - If `A` is [`Unaligned`], this pointer must still be properly aligned for `T`
    ///   at the point of this call.
    #[inline]
    pub unsafe fn read<T>(self) -> T {
        let ptr = self.as_ptr().cast::<T>().debug_ensure_aligned();
        // SAFETY: The caller ensures the pointee is `T` and upholds safety for
        // `read`.
        unsafe { ptr.read() }
    }

    /// Casts to a concrete type as a [`MovingPtr`].
    ///
    /// # Safety
    /// `T` must be the erased pointee type for this `OwningPtr`.
    #[inline]
    pub unsafe fn cast<T>(self) -> MovingPtr<'a, T, A> {
        // SAFETY: The caller guarantees `T` is the real pointee type, satisfying
        // `MovingPtr::new`'s own safety contract; the lifetime and alignment
        // parameters are carried over unchanged from `self`.
        unsafe { MovingPtr::new(self.0.cast::<T>()) }
    }

    /// Consumes this `OwningPtr` to drop the underlying data of type `T`.
    ///
    /// # Safety
    /// - `T` must be the erased pointee type for this `OwningPtr`.
    /// - If `A` is [`Unaligned`], this pointer must still be properly aligned for `T`
    ///   at the point of this call.
    #[inline]
    pub unsafe fn drop_as<T>(self) {
        let ptr = self.as_ptr().cast::<T>().debug_ensure_aligned();
        // SAFETY: The caller ensures the pointee is `T` and upholds safety for
        // `drop_in_place`.
        unsafe {
            ptr.drop_in_place();
        }
    }

    /// Gets the underlying pointer, erasing the associated lifetime.
    ///
    /// Prefer the other, more type-safe methods on this type where possible.
    #[inline]
    pub fn as_ptr(&self) -> *mut u8 {
        self.0.as_ptr()
    }

    /// Gets an immutable pointer from this owned pointer.
    #[inline]
    pub fn as_ref(&self) -> Ptr<'_, A> {
        // SAFETY: `OwningPtr`'s validity guarantees are a superset of `Ptr`'s.
        unsafe { Ptr::new(self.0) }
    }

    /// Gets a mutable pointer from this owned pointer.
    #[inline]
    pub fn as_mut(&mut self) -> PtrMut<'_, A> {
        // SAFETY: `OwningPtr`'s validity guarantees are a superset of `PtrMut`'s.
        unsafe { PtrMut::new(self.0) }
    }
}

impl<'a> OwningPtr<'a, Unaligned> {
    /// Consumes this `OwningPtr` to obtain ownership of the underlying data of type
    /// `T`, reading it unaligned.
    ///
    /// # Safety
    /// `T` must be the erased pointee type for this `OwningPtr`.
    pub unsafe fn read_unaligned<T>(self) -> T {
        let ptr = self.as_ptr().cast::<T>();
        // SAFETY: The caller ensures the pointee is `T` and upholds safety for
        // `read_unaligned`.
        unsafe { ptr.read_unaligned() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ptr_from_ref_derefs_back_to_the_same_value() {
        let x = 123u32;
        let ptr = Ptr::from(&x);
        // SAFETY: `ptr` erases a `&u32`, so casting back to `u32` is exactly the
        // real pointee type.
        let back: &u32 = unsafe { ptr.deref() };
        assert_eq!(*back, 123);
    }

    #[test]
    fn ptr_mut_deref_mut_allows_writing_through() {
        let mut x = 1u32;
        let ptr = PtrMut::from(&mut x);
        // SAFETY: same reasoning as above, mutable side.
        let back: &mut u32 = unsafe { ptr.deref_mut() };
        *back = 2;
        assert_eq!(x, 2);
    }

    #[test]
    fn ptr_mut_promote_to_owning_then_read() {
        let value = 7u64;
        OwningPtr::make(value, |ptr| {
            // SAFETY: `ptr` owns a `u64` it was just constructed from.
            let read_back: u64 = unsafe { ptr.read() };
            assert_eq!(read_back, 7);
        });
    }

    #[test]
    fn owning_ptr_drop_as_runs_drop_exactly_once() {
        struct DropCounter<'a>(&'a core::cell::Cell<u32>);
        impl<'a> Drop for DropCounter<'a> {
            fn drop(&mut self) {
                self.0.set(self.0.get() + 1);
            }
        }

        let count = core::cell::Cell::new(0u32);
        OwningPtr::make(DropCounter(&count), |ptr| {
            // SAFETY: `ptr` owns a `DropCounter` it was just constructed from.
            unsafe { ptr.drop_as::<DropCounter>() };
        });
        assert_eq!(count.get(), 1);
    }

    #[test]
    fn to_unaligned_then_read_unaligned_round_trips() {
        let value = 99u32;
        OwningPtr::make(value, |ptr| {
            let unaligned = ptr.to_unaligned();
            // SAFETY: `unaligned` still points at the `u32` `ptr` was constructed
            // from — dropping alignment tracking doesn't change the real layout.
            let back: u32 = unsafe { unaligned.read_unaligned() };
            assert_eq!(back, 99);
        });
    }
}
