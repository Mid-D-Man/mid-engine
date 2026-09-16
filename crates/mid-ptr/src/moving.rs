// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-ptr.md, section "moving.rs"
// ============================================================================

use core::fmt::{self, Debug, Formatter, Pointer};
use core::marker::PhantomData;
use core::mem::{self, ManuallyDrop, MaybeUninit};
use core::ops::{Deref, DerefMut};
use core::ptr::{self, NonNull};

use crate::aligned::{Aligned, IsAligned, Unaligned};
use crate::debug_align::DebugEnsureAligned;
use crate::erased::OwningPtr;

/// A `Box`-like pointer for moving a value to a new memory location without passing
/// it by value.
///
/// Represents ownership of whatever data it points to and runs that data's [`Drop`]
/// impl when dropped — but, like [`crate::OwningPtr`], is not responsible for freeing
/// the memory it points to. Acts "borrow-like":
/// - Considered exclusive and mutable; cannot be cloned, since that risks aliased
///   mutability and use-after-free.
/// - Must always point to a valid value of whatever the pointee type is.
/// - The lifetime `'a` accurately represents how long the pointer stays valid.
/// - Supports no pointer arithmetic at all.
/// - If `A` is [`Aligned`], the pointer must always be properly aligned for `T`.
///
/// A value can be deconstructed into its fields via
/// [`deconstruct_moving_ptr`](crate::deconstruct_moving_ptr) — see that macro's own
/// docs for a worked example.
#[repr(transparent)]
pub struct MovingPtr<'a, T, A: IsAligned = Aligned>(NonNull<T>, PhantomData<(&'a mut T, A)>);

impl<'a, T> MovingPtr<'a, T, Aligned> {
    /// Removes the alignment requirement of this pointer.
    #[inline]
    pub fn to_unaligned(self) -> MovingPtr<'a, T, Unaligned> {
        let value = MovingPtr(self.0, PhantomData);
        mem::forget(self);
        value
    }

    /// Creates a `MovingPtr` from a provided value of type `T`.
    ///
    /// For a safer alternative, prefer [`crate::move_as_ptr`] where possible.
    ///
    /// # Safety
    /// - `value` must store a properly initialized value of type `T`.
    /// - Once the returned `MovingPtr` has been used, `value` must be treated as
    ///   though it were uninitialized, unless it was explicitly leaked via
    ///   [`core::mem::forget`].
    #[inline]
    pub unsafe fn from_value(value: &'a mut MaybeUninit<T>) -> Self {
        // SAFETY:
        // - `MaybeUninit<T>` has the same memory layout as `T`.
        // - The caller guarantees `value` points to a valid instance of `T`.
        MovingPtr(NonNull::from(value).cast::<T>(), PhantomData)
    }
}

impl<'a, T, A: IsAligned> MovingPtr<'a, T, A> {
    /// Creates a new instance from a raw pointer.
    ///
    /// For a safer alternative, prefer [`crate::move_as_ptr`] where possible.
    ///
    /// # Safety
    /// - `inner` must point to a valid value of `T`.
    /// - If `A` is [`Aligned`], `inner` must be properly aligned for `T`.
    /// - `inner` must have correct provenance to allow reads and writes of the
    ///   pointee type.
    /// - The lifetime `'a` must be constrained such that this `MovingPtr` stays
    ///   valid and nothing else can read or mutate the pointee while it's live.
    #[inline]
    pub unsafe fn new(inner: NonNull<T>) -> Self {
        Self(inner, PhantomData)
    }

    /// Partially moves fields out of `self`.
    ///
    /// The partially-moved-from value is handed back pointing at `MaybeUninit<T>`.
    /// Calling this function is itself safe, but the returned `MovingPtr` needs care:
    /// it points at a value that may no longer be fully valid.
    #[inline]
    pub fn partial_move<R>(
        self,
        f: impl FnOnce(MovingPtr<'_, T, A>) -> R,
    ) -> (MovingPtr<'a, MaybeUninit<T>, A>, R) {
        let partial_ptr = self.0;
        let ret = f(self);
        (
            MovingPtr(partial_ptr.cast::<MaybeUninit<T>>(), PhantomData),
            ret,
        )
    }

    /// Reads the value pointed to by this pointer.
    #[inline]
    pub fn read(self) -> T {
        // SAFETY:
        // - `self.0` is valid for reads, since this type owns the value it points to.
        // - `self.0` always points to a valid instance of `T`.
        // - If `A` is `Aligned`, `self.0` is properly aligned for `T` by construction.
        let value = unsafe { A::read_ptr(self.0.as_ptr()) };
        mem::forget(self);
        value
    }

    /// Writes the value pointed to by this pointer into a provided location.
    ///
    /// Does *not* drop the value already stored at `dst` — that's the caller's
    /// responsibility.
    ///
    /// # Safety
    /// - `dst` must be valid for writes.
    /// - If `A` is [`Aligned`], `dst` must be properly aligned for `T`.
    /// - `dst` and the pointer `self` holds must not point at the same address.
    #[inline]
    pub unsafe fn write_to(self, dst: *mut T) {
        let src = self.0.as_ptr();
        mem::forget(self);
        // SAFETY:
        // - `src` is valid for reads, since this pointer owns the value it points to.
        // - The caller ensures `dst` is valid for writes.
        // - Since `A` mirrors `Aligned`/`Unaligned` on both sides, the caller's
        //   alignment obligation for `dst` matches `src`'s own alignment guarantee.
        // - The caller ensures `dst` and `src` are distinct addresses.
        // - `self` was taken by move and forgotten, so nothing else can observe
        //   `src` being moved out from under it.
        unsafe { A::copy_nonoverlapping(src, dst, 1) };
    }

    /// Writes the value pointed to by this pointer into `dst`, dropping whatever was
    /// previously stored there. Has the same semantics as a plain `*dst = ...`
    /// assignment.
    #[inline]
    pub fn assign_to(self, dst: &mut T) {
        // Equivalent in effect to:
        // ```
        // let src = self.0.as_ptr();
        // mem::forget(self);
        // *dst = unsafe { A::read_ptr(src) };
        // ```
        // but written through a drop guard instead, so it doesn't risk codegen-ing
        // into more than one memcpy the way the snippet above can.
        struct DropGuard<'a, 'b, T, A: IsAligned> {
            src: ManuallyDrop<MovingPtr<'a, T, A>>,
            dst: &'b mut T,
        }

        impl<'a, 'b, T, A: IsAligned> Drop for DropGuard<'a, 'b, T, A> {
            fn drop(&mut self) {
                // SAFETY: `self.src` is always initialized with a valid `MovingPtr`
                // and is only ever taken here, in `drop`; no other code can observe
                // the taken-out `self.src` afterward.
                let src = unsafe { ManuallyDrop::take(&mut self.src) };

                // SAFETY:
                // - `dst` is a mutable borrow, so it's valid for writes.
                // - `dst` is a mutable borrow, so it's always aligned.
                unsafe { src.write_to(self.dst) };
            }
        }

        let guard = DropGuard {
            src: ManuallyDrop::new(self),
            dst,
        };

        // SAFETY:
        // - `guard.dst` is a mutable borrow, so it points to a valid instance of `T`.
        // - `guard.dst` is a mutable borrow, so it points to a value valid for
        //   dropping.
        // - `guard.dst` is a mutable borrow, so it cannot alias any other access.
        // - `guard.dst` is overwritten when `guard` drops, so no other code can
        //   observe it mid-drop.
        unsafe {
            ptr::drop_in_place(guard.dst);
        }
    }

    /// Creates a `MovingPtr` for a specific field within `self`.
    ///
    /// Meant explicitly for deconstructive moves. The correct byte offset for a
    /// field can be obtained via [`core::mem::offset_of`].
    ///
    /// # Safety
    /// - `f` must return a non-null pointer to a valid field inside `T`.
    /// - If `A` is [`Aligned`], `T` must not be `repr(packed)`.
    /// - `self` must not be accessed or dropped as though it were a complete value
    ///   after this call — fields not yet moved out of may still be accessed or
    ///   dropped separately.
    /// - This call cannot alias the field with any other access, including other
    ///   calls to `move_field` for the same field, unless [`core::mem::forget`] is
    ///   called on the earlier one first.
    ///
    /// Together, these mean any operation that could drop `self` while pointers to
    /// its fields are still held is undefined behavior — including an early return
    /// from a panic. See [`crate::deconstruct_moving_ptr`] for the macro that wraps
    /// this safely.
    #[inline(always)]
    pub unsafe fn move_field<U>(&self, f: impl Fn(*mut T) -> *mut U) -> MovingPtr<'a, U, A> {
        MovingPtr(
            // SAFETY: The caller ensures `U` is the correct type for the field `f`
            // points at, so the result is non-null.
            unsafe { NonNull::new_unchecked(f(self.0.as_ptr())) },
            PhantomData,
        )
    }
}

impl<'a, T, A: IsAligned> MovingPtr<'a, MaybeUninit<T>, A> {
    /// Creates a `MovingPtr` for a specific field within `self`, projecting through
    /// the `MaybeUninit`.
    ///
    /// Meant explicitly for deconstructive moves. The correct byte offset for a
    /// field can be obtained via [`core::mem::offset_of`].
    ///
    /// # Safety
    /// Same as [`move_field`](MovingPtr::move_field).
    #[inline(always)]
    pub unsafe fn move_maybe_uninit_field<U>(
        &self,
        f: impl Fn(*mut T) -> *mut U,
    ) -> MovingPtr<'a, MaybeUninit<U>, A> {
        let self_ptr = self.0.as_ptr().cast::<T>();
        // SAFETY:
        // - The caller ensures `U` is the correct type for the field `f` points at,
        //   so the result is non-null.
        // - `MaybeUninit<T>` is `repr(transparent)`, so it shares `T`'s layout.
        let field_ptr = unsafe { NonNull::new_unchecked(f(self_ptr)) };
        MovingPtr(field_ptr.cast::<MaybeUninit<U>>(), PhantomData)
    }
}

impl<'a, T, A: IsAligned> MovingPtr<'a, MaybeUninit<T>, A> {
    /// Creates a `MovingPtr` pointing to a valid instance of `T`.
    ///
    /// See also [`MaybeUninit::assume_init`].
    ///
    /// # Safety
    /// The caller must ensure the value `self` points to is genuinely initialized.
    /// Calling this before that's true is immediate undefined behavior.
    #[inline]
    pub unsafe fn assume_init(self) -> MovingPtr<'a, T, A> {
        let value = MovingPtr(self.0.cast::<T>(), PhantomData);
        mem::forget(self);
        value
    }
}

impl<T, A: IsAligned> Pointer for MovingPtr<'_, T, A> {
    #[inline]
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Pointer::fmt(&self.0, f)
    }
}

impl<T> Debug for MovingPtr<'_, T, Aligned> {
    #[inline]
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "MovingPtr<Aligned>({:?})", self.0)
    }
}

impl<T> Debug for MovingPtr<'_, T, Unaligned> {
    #[inline]
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "MovingPtr<Unaligned>({:?})", self.0)
    }
}

impl<'a, T, A: IsAligned> From<MovingPtr<'a, T, A>> for OwningPtr<'a, A> {
    #[inline]
    fn from(value: MovingPtr<'a, T, A>) -> Self {
        // SAFETY:
        // - `value.0` always points to a valid value of `T`.
        // - `A` is mirrored from input to output, so alignment guarantees carry over.
        // - `value.0` has correct provenance for reads and writes of `T` by
        //   construction, which is a superset of what `OwningPtr` needs once erased.
        // - `'a` is mirrored from input to output, so lifetime guarantees carry
        //   over.
        // - `OwningPtr` upholds the same aliasing invariants `MovingPtr` does.
        let ptr = unsafe { OwningPtr::new(value.0.cast::<u8>()) };
        mem::forget(value);
        ptr
    }
}

impl<'a, T> TryFrom<MovingPtr<'a, T, Unaligned>> for MovingPtr<'a, T, Aligned> {
    type Error = MovingPtr<'a, T, Unaligned>;
    #[inline]
    fn try_from(value: MovingPtr<'a, T, Unaligned>) -> Result<Self, Self::Error> {
        let ptr = value.0;
        if ptr.as_ptr().is_aligned() {
            mem::forget(value);
            Ok(MovingPtr(ptr, PhantomData))
        } else {
            Err(value)
        }
    }
}

impl<T> Deref for MovingPtr<'_, T, Aligned> {
    type Target = T;
    #[inline]
    fn deref(&self) -> &Self::Target {
        let ptr = self.0.as_ptr().debug_ensure_aligned();
        // SAFETY: This type owns the value it points to, and `A = Aligned` means
        // this pointer is guaranteed aligned.
        unsafe { &*ptr }
    }
}

impl<T> DerefMut for MovingPtr<'_, T, Aligned> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        let ptr = self.0.as_ptr().debug_ensure_aligned();
        // SAFETY: This type owns the value it points to, and `A = Aligned` means
        // this pointer is guaranteed aligned.
        unsafe { &mut *ptr }
    }
}

impl<T, A: IsAligned> Drop for MovingPtr<'_, T, A> {
    fn drop(&mut self) {
        // SAFETY:
        // - `self.0` is valid for reads and writes, since this pointer type owns the
        //   value it points to.
        // - `self.0` always points to a valid instance of `T`.
        // - If `A` is `Aligned`, `self.0` is properly aligned for `T` by
        //   construction.
        // - This type owns the value it points to, so `self.0` is always valid for
        //   dropping until this pointer itself drops.
        // - This type owns the value it points to, so it cannot be mutably aliased
        //   elsewhere.
        unsafe { A::drop_in_place(self.0.as_ptr()) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_returns_the_value_and_skips_the_drop() {
        struct DropCounter<'a>(&'a core::cell::Cell<u32>);
        impl<'a> Drop for DropCounter<'a> {
            fn drop(&mut self) {
                self.0.set(self.0.get() + 1);
            }
        }

        let count = core::cell::Cell::new(0u32);
        let mut slot = MaybeUninit::new(DropCounter(&count));
        // SAFETY: `slot` is initialized and not touched again except through the
        // returned `MovingPtr`.
        let ptr = unsafe { MovingPtr::from_value(&mut slot) };
        let value = ptr.read();
        // `read` forgets the pointer's drop obligation and hands back the value by
        // move, so nothing should have run `Drop` yet.
        assert_eq!(count.get(), 0);
        drop(value);
        assert_eq!(count.get(), 1);
    }

    #[test]
    fn assign_to_drops_the_old_value_and_writes_the_new_one() {
        struct DropCounter<'a>(&'a core::cell::Cell<u32>, u32);
        impl<'a> Drop for DropCounter<'a> {
            fn drop(&mut self) {
                self.0.set(self.0.get() + 1);
            }
        }

        let drops = core::cell::Cell::new(0u32);
        let mut src_slot = MaybeUninit::new(DropCounter(&drops, 1));
        let mut dst = DropCounter(&drops, 0);

        // SAFETY: `src_slot` is initialized and not touched again except through
        // the returned `MovingPtr`.
        let src = unsafe { MovingPtr::from_value(&mut src_slot) };
        src.assign_to(&mut dst);

        assert_eq!(dst.1, 1, "dst now holds the moved-in value");
        assert_eq!(drops.get(), 1, "the old dst value was dropped exactly once");
    }

    #[test]
    fn to_unaligned_then_try_from_recovers_aligned() {
        let mut slot = MaybeUninit::new(5u32);
        // SAFETY: `slot` is initialized and not touched again except through the
        // returned `MovingPtr`.
        let ptr = unsafe { MovingPtr::from_value(&mut slot) };
        let unaligned = ptr.to_unaligned();
        let aligned = MovingPtr::try_from(unaligned)
            .unwrap_or_else(|_| panic!("a naturally aligned u32 stack slot must be aligned"));
        assert_eq!(aligned.read(), 5);
    }
}
