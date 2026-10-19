// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-ecs.md, section "`WorldCell`: shared access with checked
// disjointness (`world_cell.rs`)"
// ============================================================================
//! [`Interior`]: storage a [`crate::world_cell::WorldCell`] may hand out
//! mutable access to through a shared reference.
//!
//! Two queries of one system may hold, at the same moment, a `&mut` to the
//! `A` column of an archetype and a `&` to its `B` column. Rust's aliasing
//! rules allow that only if each column is reachable without a `&mut` to
//! (or a `&` that conflicts with) the structure holding the other, and
//! the only way to mutate through a shared path is interior mutability:
//! an [`UnsafeCell`]. So every column, every column of change ticks, and
//! every resource lives in an `Interior`.
//!
//! The wrapper is `repr(transparent)` over `UnsafeCell<T>`: no extra
//! indirection, no extra size, and the existing `&mut` paths are unchanged
//! in cost (`UnsafeCell::get_mut` is free).
//!
//! **The invariant, and who upholds it.** `Deref` hands out `&T` and
//! [`Interior::get_mut_unchecked`] hands out `&mut T`, both from `&self`,
//! so the type system alone cannot stop the two from overlapping. What
//! does: an `Interior` only ever sits inside a `World`, and the only code
//! that calls `get_mut_unchecked` is behind the `unsafe` functions of
//! `WorldCell`, which is created from a `&mut World` and carries that
//! borrow's lifetime. While a cell exists nothing else can reach the
//! world, so every overlap would have to come *through the cell's own
//! unsafe functions*, whose contract (stated on each) is that the caller
//! holds no conflicting access: the property the access sets
//! (`access.rs`) compute and a system checks when it is set up.

use std::cell::UnsafeCell;
use std::ops::{Deref, DerefMut};

/// A `T` that can be mutated through `&self` by `unsafe` code that has
/// proved exclusivity some other way. See the module docs.
#[repr(transparent)]
pub(crate) struct Interior<T>(UnsafeCell<T>);

impl<T> Interior<T> {
    #[inline]
    pub(crate) fn new(value: T) -> Self {
        Self(UnsafeCell::new(value))
    }

    /// Takes the value out.
    #[inline]
    pub(crate) fn into_inner(self) -> T {
        self.0.into_inner()
    }

    /// A `&mut T` from a shared reference.
    ///
    /// # Safety
    /// For as long as the returned reference (or anything derived from
    /// it) is live, no other reference to the same `Interior`'s value may
    /// be used: not through `Deref`, not through another call to this
    /// function, not through `DerefMut`. The reference may be given any
    /// lifetime up to that of the `&'a self` it came from.
    #[inline]
    #[allow(clippy::mut_from_ref)]
    pub(crate) unsafe fn get_mut_unchecked<'a>(&'a self) -> &'a mut T {
        // SAFETY: forwarded to the caller, per the contract above.
        unsafe { &mut *self.0.get() }
    }
}

impl<T> Deref for Interior<T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &T {
        // SAFETY: see the module docs. A `&mut` obtained from
        // `get_mut_unchecked` exists only while a `WorldCell` borrows the
        // whole world, which makes this call unreachable in safe code.
        unsafe { &*self.0.get() }
    }
}

impl<T> DerefMut for Interior<T> {
    /// The exclusive path: free, and safe because `&mut self` already is
    /// the proof.
    #[inline]
    fn deref_mut(&mut self) -> &mut T {
        self.0.get_mut()
    }
}

impl<T: Default> Default for Interior<T> {
    fn default() -> Self {
        Self::new(T::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_and_exclusive_writes_behave_like_the_plain_value() {
        let mut v = Interior::new(vec![1, 2, 3]);
        assert_eq!(v.len(), 3);
        v.push(4);
        assert_eq!(*v, vec![1, 2, 3, 4]);
        v.clear();
        assert!(v.is_empty());
        assert_eq!(Interior::new(7).into_inner(), 7);
        let d: Interior<Vec<u8>> = Interior::default();
        assert!(d.is_empty());
    }

    #[test]
    fn two_interiors_can_be_written_through_shared_references_at_once() {
        let a = Interior::new(1u32);
        let b = Interior::new(10u32);
        // SAFETY: two different interiors; each reference is the only
        // one used for its value.
        let (ra, rb) = unsafe { (a.get_mut_unchecked(), b.get_mut_unchecked()) };
        *ra += 1;
        *rb += 1;
        assert_eq!((*a, *b), (2, 11));
    }

    #[test]
    fn is_exactly_as_large_as_its_value() {
        assert_eq!(std::mem::size_of::<Interior<u64>>(), 8);
        assert_eq!(
            std::mem::size_of::<Interior<Vec<u32>>>(),
            std::mem::size_of::<Vec<u32>>()
        );
        assert_eq!(
            std::mem::size_of::<Interior<Box<dyn std::any::Any>>>(),
            std::mem::size_of::<Box<dyn std::any::Any>>()
        );
    }
}
