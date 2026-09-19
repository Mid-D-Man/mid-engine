// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-alloc.md, section "bump_vec.rs"
// ============================================================================
//! [`BumpVec`], a growable, contiguous, `Vec`-shaped collection backed
//! by any [`RawAlloc`] instead of the global allocator.
//!
//! # Why this lives in `mid-alloc`, not `mid-arena`
//!
//! This exists to answer a real question: does `mid-arena`'s
//! `BumpArena<T>` need something like `bumpalo::collections::Vec`?
//! Checked against `bumpalo::collections::vec.rs`/`raw_vec.rs` (source
//! read) directly: **no, and not because it was skipped, because it
//! cannot be built soundly on `BumpArena<T>` as designed.**
//! `bumpalo::Bump` can host a growable `Vec` because it is an untyped
//! byte arena that never runs destructors on its own -- that is
//! exactly why `bumpalo::boxed::Box` has to exist as a separate opt-in
//! for `Drop`. `BumpArena<T>` is the deliberate opposite: single-typed,
//! and its own `RegionNode<T>::drop` unconditionally runs
//! `assume_init_drop` on every slot from `0..len`, where `len` is the
//! *only* counter -- it does double duty as both the bump cursor and
//! the drop-tracked count. A growable Vec needs "reserved capacity
//! beyond what's actually been pushed," which is fundamentally
//! incompatible with that single counter: advancing it to reserve
//! space would make the arena believe uninitialized memory is a real,
//! live `T`, and it would try to drop it. `RawAlloc`, by contrast, is
//! untyped by design (`try_alloc_raw`/`try_dealloc_raw` deal in bytes,
//! not `T`), so it has no such invariant to violate -- the exact same
//! reason `bumpalo::Bump` can host one. `BumpVec` is a real, if
//! smaller, fork of `bumpalo::collections::vec`'s own design: grow via
//! [`RawAlloc::try_grow_raw`], which can skip copying entirely when
//! the parent supports growing in place (`StackAllocator`, when the
//! block being grown is genuinely the most recent allocation) or hand
//! the work to a real `realloc` (`HeapAlloc`), falling back to a
//! plain allocate-copy-deallocate sequence only when the parent
//! supports neither. See this file's own "Fixes and Problems" entry
//! for why the first version of this file did that copy by hand
//! instead, and what it cost at scale on real CI.

use crate::raw_alloc::RawAlloc;
use core::mem;
use core::ops::{Deref, DerefMut};
use core::ptr::{self, NonNull};

/// A growable, contiguous `[T]`-shaped collection, backed by a parent
/// [`RawAlloc`] instead of the global allocator. See this module's own
/// doc comment for why this exists and why it could not live on
/// `mid-arena`'s `BumpArena<T>` instead.
pub struct BumpVec<'p, T, P: RawAlloc> {
    parent: &'p P,
    ptr: NonNull<T>,
    cap: usize,
    len: usize,
}

impl<'p, T, P: RawAlloc> BumpVec<'p, T, P> {
    /// An empty vec that allocates nothing until the first `push`.
    pub fn new_in(parent: &'p P) -> Self {
        Self {
            parent,
            ptr: NonNull::dangling(),
            cap: 0,
            len: 0,
        }
    }

    /// A vec with room for `capacity` elements up front. Returns `None`
    /// if `parent` cannot provide that block.
    pub fn with_capacity_in(parent: &'p P, capacity: usize) -> Option<Self> {
        if capacity == 0 {
            return Some(Self::new_in(parent));
        }
        let ptr = Self::alloc_block(parent, capacity)?;
        Some(Self {
            parent,
            ptr,
            cap: capacity,
            len: 0,
        })
    }

    fn alloc_block(parent: &'p P, capacity: usize) -> Option<NonNull<T>> {
        let size = mem::size_of::<T>().checked_mul(capacity)?;
        let raw = parent.try_alloc_raw(size, mem::align_of::<T>())?;
        Some(raw.cast())
    }

    /// Number of elements currently held.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether this vec holds no elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Current backing capacity, in elements.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.cap
    }

    /// Grows to a new, bigger block (double, or 4 elements from empty,
    /// matching `bumpalo::RawVec`'s own real growth factor). Delegates
    /// the actual growth to `parent.try_grow_raw` rather than always
    /// allocating a fresh block and copying by hand -- see this
    /// module's own "Fixes and Problems" entry for why that hand-
    /// rolled version measured badly at scale on real CI, and why
    /// `try_grow_raw` (which can skip the copy entirely when the
    /// parent supports in-place growth, or hand it to a real
    /// `realloc`) fixes it. Returns `false`, leaving `self` completely
    /// unchanged, if `parent` cannot provide the new block.
    fn grow(&mut self) -> bool {
        let new_cap = if self.cap == 0 {
            4
        } else {
            self.cap.saturating_mul(2)
        };

        let new_ptr = if self.cap == 0 {
            match Self::alloc_block(self.parent, new_cap) {
                Some(p) => p,
                None => return false,
            }
        } else {
            let old_size = mem::size_of::<T>() * self.cap;
            let new_size = match mem::size_of::<T>().checked_mul(new_cap) {
                Some(s) => s,
                None => return false,
            };
            // SAFETY: `self.ptr` came from a live `try_alloc_raw`/
            // `try_grow_raw` call on `self.parent` for exactly
            // `old_size` bytes at `align_of::<T>()` -- `alloc_block`
            // and every previous `grow` call both go through
            // `self.parent` with exactly these size/align values --
            // and `new_size >= old_size` since `new_cap > self.cap`
            // whenever this branch runs (`saturating_mul(2)` on a
            // nonzero `cap`).
            let grown = unsafe {
                self.parent.try_grow_raw(
                    self.ptr.cast(),
                    old_size,
                    new_size,
                    mem::align_of::<T>(),
                )
            };
            match grown {
                Some(p) => p.cast(),
                None => return false,
            }
        };

        self.ptr = new_ptr;
        self.cap = new_cap;
        true
    }

    /// Appends `value`, growing first if necessary. Returns `false`,
    /// with `value` dropped exactly as if it had been pushed and
    /// immediately popped, only if growth itself fails (`parent` is
    /// out of room) -- real, exhausted-allocator failure, not
    /// something ordinary use is expected to hit.
    #[inline]
    pub fn push(&mut self, value: T) -> bool {
        if self.len == self.cap && !self.grow() {
            return false;
        }
        // SAFETY: `self.len < self.cap` now -- either already was, or
        // `grow` above just ensured it.
        unsafe {
            self.ptr.as_ptr().add(self.len).write(value);
        }
        self.len += 1;
        true
    }

    /// Removes and returns the last element, or `None` if empty.
    #[inline]
    pub fn pop(&mut self) -> Option<T> {
        if self.len == 0 {
            return None;
        }
        self.len -= 1;
        // SAFETY: slot `self.len` (post-decrement) was live and
        // initialized the moment before this call.
        Some(unsafe { self.ptr.as_ptr().add(self.len).read() })
    }
}

impl<'p, T, P: RawAlloc> Deref for BumpVec<'p, T, P> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        // SAFETY: `[self.ptr, self.ptr + self.len)` are exactly the
        // live, initialized elements this type maintains.
        unsafe { core::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}

impl<'p, T, P: RawAlloc> DerefMut for BumpVec<'p, T, P> {
    fn deref_mut(&mut self) -> &mut [T] {
        // SAFETY: same reasoning as `deref` above; `&mut self` rules
        // out any other live reference into this range.
        unsafe { core::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

impl<'p, T, P: RawAlloc> Drop for BumpVec<'p, T, P> {
    fn drop(&mut self) {
        // SAFETY: exactly the live elements this type maintains.
        unsafe {
            ptr::drop_in_place(core::slice::from_raw_parts_mut(
                self.ptr.as_ptr(),
                self.len,
            ));
        }
        if self.cap > 0 {
            let size = mem::size_of::<T>() * self.cap;
            // SAFETY: same reasoning as the deallocation in `grow`
            // above -- this is the block's one and only owner, and
            // every value it held was just dropped in place above.
            unsafe {
                self.parent
                    .try_dealloc_raw(self.ptr.cast(), size, mem::align_of::<T>());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::raw_alloc::{HeapAlloc, NullAlloc};
    use crate::stack_allocator::StackAllocator;

    #[test]
    fn push_pop_and_deref_work_like_a_real_vec() {
        let heap = HeapAlloc;
        let mut v: BumpVec<u32, _> = BumpVec::new_in(&heap);
        assert!(v.is_empty());
        for i in 0..10u32 {
            assert!(v.push(i));
        }
        assert_eq!(v.len(), 10);
        assert_eq!(&*v, &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
        assert_eq!(v.pop(), Some(9));
        assert_eq!(v.len(), 9);
    }

    #[test]
    fn grows_past_an_initial_with_capacity_and_keeps_earlier_values() {
        let heap = HeapAlloc;
        let mut v: BumpVec<u32, _> =
            BumpVec::with_capacity_in(&heap, 2).expect("heap should provide a small block");
        assert_eq!(v.capacity(), 2);
        v.push(1);
        v.push(2);
        v.push(3); // forces growth past the initial capacity of 2
        assert!(v.capacity() >= 3);
        assert_eq!(&*v, &[1, 2, 3]);
    }

    #[test]
    fn push_fails_cleanly_when_the_parent_is_exhausted() {
        let s = StackAllocator::with_capacity(4); // room for exactly one u32
        let mut v: BumpVec<u32, _> =
            BumpVec::with_capacity_in(&s, 1).expect("the 4-byte stack fits exactly one u32");
        assert!(v.push(1));
        assert!(
            !v.push(2),
            "growth needs a second, bigger block; the stack has nothing left"
        );
        assert_eq!(&*v, &[1], "the failed push must not have corrupted what was already there");
    }

    #[test]
    fn drop_runs_every_live_elements_destructor_exactly_once() {
        use core::cell::Cell as StdCell;

        struct DropCounter<'a>(&'a StdCell<u32>);
        impl<'a> Drop for DropCounter<'a> {
            fn drop(&mut self) {
                self.0.set(self.0.get() + 1);
            }
        }

        let count = StdCell::new(0u32);
        let heap = HeapAlloc;
        {
            let mut v: BumpVec<DropCounter, _> = BumpVec::new_in(&heap);
            for _ in 0..20 {
                v.push(DropCounter(&count));
            }
            // Force at least one real grow-and-move cycle before drop.
            assert!(v.capacity() >= 20);
        }
        assert_eq!(
            count.get(),
            20,
            "every element, including ones moved across a grow, must be dropped exactly once"
        );
    }

    #[test]
    fn pop_returns_ownership_without_double_dropping() {
        use core::cell::Cell as StdCell;

        struct DropCounter<'a>(&'a StdCell<u32>);
        impl<'a> Drop for DropCounter<'a> {
            fn drop(&mut self) {
                self.0.set(self.0.get() + 1);
            }
        }

        let count = StdCell::new(0u32);
        let heap = HeapAlloc;
        let mut v: BumpVec<DropCounter, _> = BumpVec::new_in(&heap);
        v.push(DropCounter(&count));
        v.push(DropCounter(&count));
        let popped = v.pop().unwrap();
        assert_eq!(count.get(), 0, "popping must not itself drop the value");
        drop(popped);
        assert_eq!(count.get(), 1);
        drop(v);
        assert_eq!(count.get(), 2, "the one remaining element must still be dropped");
    }

    #[test]
    fn new_in_never_touches_the_parent_until_the_first_push() {
        // NullAlloc always fails try_alloc_raw -- new_in must not call
        // it at all, only grow() (triggered by the first push) does.
        let n = NullAlloc;
        let v: BumpVec<u32, _> = BumpVec::new_in(&n);
        assert_eq!(v.capacity(), 0);
        assert!(v.is_empty());
    }

    #[test]
    fn zero_sized_types_do_not_panic_or_loop_forever() {
        let heap = HeapAlloc;
        let mut v: BumpVec<(), _> = BumpVec::new_in(&heap);
        for _ in 0..20 {
            assert!(v.push(()));
        }
        assert_eq!(v.len(), 20);
    }
}
