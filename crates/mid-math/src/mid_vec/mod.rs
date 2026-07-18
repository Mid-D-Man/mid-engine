// crates/mid-math/src/mid_vec/mod.rs
//! `MidVec<T, N>` — a small-size-optimised vector: `N` elements live
//! inline in the struct itself; pushing past `N` spills to a single heap
//! allocation, after which it behaves like a normal growable `Vec<T>`.
//!
//! Written from scratch (no `smallvec`/`tinyvec` dependency) to match this
//! crate's zero-runtime-dependency, `no_std`-friendly-core policy, and to
//! keep the memory layout under our own control for future FFI use.
//!
//! Design, compared to the two crates this was researched against:
//!   - Storage is a `union` (`inline` array *or* `heap` pointer+capacity),
//!     not an enum-of-two-variants — avoids paying for a discriminant on
//!     top of the union (`tinyvec` uses the enum approach).
//!   - No `T: Default` bound, and unused slots are never eagerly
//!     initialised — inline storage is genuinely `MaybeUninit` until
//!     written to (`tinyvec`'s `ArrayVec` always fully initialises its
//!     backing array up front, which costs one `Default::default()` write
//!     per slot on every construction).
//!   - No artificial length cap — `tinyvec::ArrayVec` stores its length in
//!     a `u16` (65,535-element ceiling); ours uses the spare high bits of
//!     a `usize`, so the practical ceiling is the same as `Vec`'s.
//!   - Zero-copy conversion to/from `Vec<T>` once already heap-allocated
//!     (steals/returns the raw allocation instead of copying element by
//!     element).
//!
//! Not supported (by design, not oversight): zero-sized `T`. Every real
//! consumer in this crate (`f32`, `Vec3`, curve keyframe types) has a
//! non-zero size, and supporting ZSTs correctly means an extra branch in
//! nearly every method for a case we will never hit. `new()` / `with_capacity()`
//! assert this at construction so a future ZST misuse fails loudly instead
//! of silently invoking UB in the allocator.

mod raw;
mod iter;

use core::cmp;
use core::fmt;
use core::mem::{size_of, ManuallyDrop};
use core::ops::{Deref, DerefMut};
use core::ptr::NonNull;

use raw::RawMidVec;

pub use iter::IntoIter;

/// A vector with `N` elements of inline storage before it spills to the heap.
///
/// Behaves like `Vec<T>` for almost all purposes — it derefs to `&[T]` /
/// `&mut [T]`, so indexing, slicing, iteration by reference, `.len()`,
/// `.is_empty()`, etc. all work exactly as they would on a `Vec<T>`.
pub struct MidVec<T, const N: usize> {
    /// Packed as `(len << 1) | on_heap`. See `len_val` / `on_heap` / `pack`.
    ///
    /// A non-ZST `Vec` is documented to never exceed `isize::MAX` elements,
    /// which leaves at least one spare high bit in `usize` to steal for the
    /// heap/inline tag — same trick `smallvec` uses.
    len: usize,
    raw: RawMidVec<T, N>,
}

// ── length/tag packing ─────────────────────────────────────────────────────

impl<T, const N: usize> MidVec<T, N> {
    #[inline]
    const fn pack(len: usize, on_heap: bool) -> usize {
        (len << 1) | (on_heap as usize)
    }

    #[inline]
    fn len_val(&self) -> usize {
        self.len >> 1
    }

    #[inline]
    fn on_heap(&self) -> bool {
        self.len & 1 == 1
    }

    #[inline]
    fn set_len_raw(&mut self, new_len: usize, on_heap: bool) {
        self.len = Self::pack(new_len, on_heap);
    }
}

// ── construction ────────────────────────────────────────────────────────────

impl<T, const N: usize> MidVec<T, N> {
    #[inline]
    pub fn new() -> Self {
        assert!(
            size_of::<T>() > 0,
            "MidVec<T, N> does not support zero-sized T"
        );
        Self { len: 0, raw: RawMidVec::new_inline() }
    }

    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        let mut v = Self::new();
        if capacity > N {
            v.grow_to(capacity);
        }
        v
    }

    /// Take ownership of an existing `Vec<T>` without copying its elements:
    /// if it already has a heap allocation, that allocation is adopted
    /// directly (pointer + capacity), matching `smallvec::from_vec`.
    pub fn from_vec(vec: Vec<T>) -> Self {
        assert!(
            size_of::<T>() > 0,
            "MidVec<T, N> does not support zero-sized T"
        );
        if vec.capacity() == 0 {
            return Self::new();
        }
        let mut vec = ManuallyDrop::new(vec);
        let len = vec.len();
        let cap = vec.capacity();
        // SAFETY: capacity() != 0 was just checked, so the pointer is
        // non-null (Vec never uses a null pointer for a real allocation).
        let ptr = unsafe { NonNull::new_unchecked(vec.as_mut_ptr()) };
        Self { len: Self::pack(len, true), raw: RawMidVec::new_heap(ptr, cap) }
    }

    /// Like `from_vec`, but actually uses inline storage when the incoming
    /// `Vec` is small enough to fit (`len() <= N`).
    ///
    /// `from_vec` always adopts the `Vec`'s existing heap allocation
    /// zero-copy — correct and cheap when the `Vec` is already large, but
    /// if it only has 2 elements and `N` is 8, `from_vec` leaves it
    /// permanently on the heap anyway, defeating the entire point of
    /// having inline storage. This is the right constructor for "ordinary
    /// application code handed us a `Vec`, probably small" call sites
    /// (e.g. spline control points); use plain `from_vec` when the source
    /// is itself another spill-aware container that already decided it
    /// needed the heap.
    ///
    /// Costs one extra per-element move for the small case (`vec.len() <=
    /// N`) since the elements have to be walked out of the old allocation
    /// into the new inline one before the old `Vec` is dropped. Identical
    /// zero-copy behaviour to `from_vec` once `vec.len() > N`.
    pub fn from_vec_or_inline(vec: Vec<T>) -> Self {
        if vec.len() <= N {
            let mut v = Self::new();
            for item in vec {
                v.push(item);
            }
            v
        } else {
            Self::from_vec(vec)
        }
    }

    /// Inverse of `from_vec`: zero-copy if already spilled to the heap,
    /// otherwise allocates once and copies the `N` (or fewer) inline
    /// elements out.
    pub fn into_vec(self) -> Vec<T> {
        let len = self.len_val();
        let on_heap = self.on_heap();
        // Suppress our own `Drop`: ownership of the elements (and, if
        // applicable, the heap allocation) is being transferred to the
        // `Vec` we return, so running our destructor too would double-free
        // or double-drop.
        let this = ManuallyDrop::new(self);
        if on_heap {
            // SAFETY: `heap` is the active variant, and the pointer/capacity
            // pair came from either `from_vec` or our own `grow_to`, both of
            // which allocate via the global allocator with `capacity_layout`
            // — exactly what `Vec::from_raw_parts` requires.
            unsafe {
                let (ptr, cap) = this.raw.heap_parts();
                Vec::from_raw_parts(ptr.as_ptr(), len, cap)
            }
        } else {
            let mut vec = Vec::with_capacity(len);
            // SAFETY: inline storage holds `len` valid, initialised `T`s;
            // `vec` was just allocated with room for exactly `len` of them.
            unsafe {
                core::ptr::copy_nonoverlapping(this.raw.as_ptr_inline(), vec.as_mut_ptr(), len);
                vec.set_len(len);
            }
            vec
        }
    }
}

impl<T, const N: usize> Default for MidVec<T, N> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

// ── growth ──────────────────────────────────────────────────────────────────

impl<T, const N: usize> MidVec<T, N> {
    /// Grow backing storage to hold at least `new_capacity` elements.
    /// Only ever grows — this type never moves a spilled heap allocation
    /// back inline, even if the vector shrinks below `N` again (same
    /// trade-off `Vec` itself makes; avoids churn from values oscillating
    /// around the inline/heap boundary).
    #[cold]
    fn grow_to(&mut self, new_capacity: usize) {
        debug_assert!(new_capacity > self.capacity());
        let len = self.len_val();
        let (new_ptr, new_cap) = if self.on_heap() {
            // SAFETY: `heap` is the active variant here.
            let (old_ptr, old_cap) = unsafe { self.raw.heap_parts() };
            // SAFETY: T is non-ZST (checked in `new`), old_ptr/old_cap
            // describe our own live allocation, new_capacity >= old_cap
            // (we only ever grow).
            unsafe { raw::grow_heap_in_place::<T>(old_ptr, old_cap, new_capacity) }
        } else {
            // SAFETY: T is non-ZST, inline is the active variant, `len`
            // inline elements are initialised and valid to read.
            unsafe { raw::alloc_and_copy_from::<T>(new_capacity, self.raw.as_ptr_inline(), len) }
        };
        self.raw = RawMidVec::new_heap(new_ptr, new_cap);
        self.set_len_raw(len, true);
    }

    pub fn reserve(&mut self, additional: usize) {
        let len = self.len_val();
        let needed = len
            .checked_add(additional)
            .unwrap_or_else(|| raw::capacity_overflow());
        if needed > self.capacity() {
            let new_capacity = needed
                .checked_next_power_of_two()
                .unwrap_or_else(|| raw::capacity_overflow());
            self.grow_to(cmp::max(new_capacity, needed));
        }
    }
}

// ── read-only accessors ─────────────────────────────────────────────────────

impl<T, const N: usize> MidVec<T, N> {
    #[inline]
    pub fn len(&self) -> usize {
        self.len_val()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len_val() == 0
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        if self.on_heap() {
            // SAFETY: `heap` is the active variant.
            unsafe { self.raw.heap_parts().1 }
        } else {
            N
        }
    }

    /// `true` once this vector has spilled from inline storage to the heap.
    #[inline]
    pub fn spilled(&self) -> bool {
        self.on_heap()
    }

    #[inline]
    pub fn as_ptr(&self) -> *const T {
        if self.on_heap() {
            // SAFETY: `heap` is the active variant.
            unsafe { self.raw.heap_parts().0.as_ptr() }
        } else {
            // SAFETY: `inline` is the active variant.
            unsafe { self.raw.as_ptr_inline() }
        }
    }

    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        if self.on_heap() {
            // SAFETY: `heap` is the active variant.
            unsafe { self.raw.heap_parts().0.as_ptr() }
        } else {
            // SAFETY: `inline` is the active variant.
            unsafe { self.raw.as_mut_ptr_inline() }
        }
    }

    #[inline]
    pub fn as_slice(&self) -> &[T] {
        // SAFETY: `as_ptr()` is valid for `len()` initialised elements,
        // whichever variant is active.
        unsafe { core::slice::from_raw_parts(self.as_ptr(), self.len_val()) }
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        let len = self.len_val();
        // SAFETY: `as_mut_ptr()` is valid for `len` initialised elements,
        // whichever variant is active.
        unsafe { core::slice::from_raw_parts_mut(self.as_mut_ptr(), len) }
    }

    #[inline]
    pub fn iter(&self) -> core::slice::Iter<'_, T> {
        self.as_slice().iter()
    }

    #[inline]
    pub fn iter_mut(&mut self) -> core::slice::IterMut<'_, T> {
        self.as_mut_slice().iter_mut()
    }
}

// ── mutation ────────────────────────────────────────────────────────────────

impl<T, const N: usize> MidVec<T, N> {
    #[inline]
    pub fn push(&mut self, value: T) {
        self.reserve(1);
        // SAFETY: `reserve(1)` just guaranteed capacity > len.
        unsafe { self.push_unchecked(value) };
    }

    /// Push without checking or growing capacity.
    ///
    /// # Safety
    /// The caller must ensure `self.len() < self.capacity()` — typically
    /// by having called `reserve` for the total number of elements about
    /// to be pushed, and not exceeding that count before the reserve is
    /// "used up." Used by `extend`/`extend_from_slice` to avoid redoing a
    /// capacity check (and its `on_heap()`/power-of-two arithmetic) on
    /// every single element when the caller already reserved once for the
    /// whole batch.
    #[inline]
    unsafe fn push_unchecked(&mut self, value: T) {
        let len = self.len_val();
        core::ptr::write(self.as_mut_ptr().add(len), value);
        self.set_len_raw(len + 1, self.on_heap());
    }

    #[inline]
    pub fn pop(&mut self) -> Option<T> {
        let len = self.len_val();
        if len == 0 {
            return None;
        }
        let new_len = len - 1;
        self.set_len_raw(new_len, self.on_heap());
        // SAFETY: index `new_len` was, until the line above, within the
        // initialised `0..len` range, and is now excluded from it — so
        // reading it out here does not double-expose it to `Drop`.
        Some(unsafe { core::ptr::read(self.as_ptr().add(new_len)) })
    }

    pub fn insert(&mut self, index: usize, value: T) {
        let len = self.len_val();
        assert!(index <= len, "MidVec::insert: index {index} > len {len}");
        self.reserve(1);
        // SAFETY: capacity > len after `reserve`, so writing at `index`
        // (after shifting the tail right by one) and at `len` (the new
        // last slot) both land in owned, allocated memory.
        unsafe {
            let p = self.as_mut_ptr();
            if index < len {
                core::ptr::copy(p.add(index), p.add(index + 1), len - index);
            }
            core::ptr::write(p.add(index), value);
        }
        self.set_len_raw(len + 1, self.on_heap());
    }

    pub fn remove(&mut self, index: usize) -> T {
        let len = self.len_val();
        assert!(index < len, "MidVec::remove: index {index} >= len {len}");
        // SAFETY: `index < len`, so `p.add(index)` is initialised; after
        // reading it out we shift the remaining tail left by one to close
        // the gap, then shrink `len` to match.
        unsafe {
            let p = self.as_mut_ptr();
            let result = core::ptr::read(p.add(index));
            core::ptr::copy(p.add(index + 1), p.add(index), len - index - 1);
            self.set_len_raw(len - 1, self.on_heap());
            result
        }
    }

    /// Remove and return the element at `index`, replacing it with the
    /// last element instead of shifting everything down. O(1) instead of
    /// `remove`'s O(n), at the cost of not preserving order.
    pub fn swap_remove(&mut self, index: usize) -> T {
        let len = self.len_val();
        assert!(index < len, "MidVec::swap_remove: index {index} >= len {len}");
        // SAFETY: both `index` and `len - 1` are within the initialised
        // range. `ptr::copy` (not `copy_nonoverlapping`) is required here:
        // when `index == len - 1` the source and destination are the same
        // address, and `copy` is defined for overlapping regions.
        unsafe {
            let p = self.as_mut_ptr();
            let result = core::ptr::read(p.add(index));
            core::ptr::copy(p.add(len - 1), p.add(index), 1);
            self.set_len_raw(len - 1, self.on_heap());
            result
        }
    }

    pub fn clear(&mut self) {
        let len = self.len_val();
        // Shrink the logical length first so that if a `T::drop` panics
        // partway through, we don't leave a stale length pointing at
        // partially-dropped memory.
        self.set_len_raw(0, self.on_heap());
        // SAFETY: these `len` elements were, until the line above, the
        // full initialised range.
        unsafe {
            core::ptr::drop_in_place(core::slice::from_raw_parts_mut(self.as_mut_ptr(), len));
        }
    }

    pub fn truncate(&mut self, new_len: usize) {
        let len = self.len_val();
        if new_len >= len {
            return;
        }
        let on_heap = self.on_heap();
        // Shrink first (panic safety), matching `clear`'s reasoning.
        self.set_len_raw(new_len, on_heap);
        // SAFETY: `new_len..len` was, until the line above, initialised
        // and is now excluded from the logical length.
        unsafe {
            let tail = core::slice::from_raw_parts_mut(self.as_mut_ptr().add(new_len), len - new_len);
            core::ptr::drop_in_place(tail);
        }
    }
}

impl<T: Clone, const N: usize> MidVec<T, N> {
    pub fn extend_from_slice(&mut self, other: &[T]) {
        self.reserve(other.len());
        for item in other {
            // SAFETY: reserved exactly `other.len()` additional capacity
            // above, and this loop pushes exactly that many elements,
            // once each.
            unsafe { self.push_unchecked(item.clone()) };
        }
    }

    pub fn from_slice(slice: &[T]) -> Self {
        let mut v = Self::with_capacity(slice.len());
        v.extend_from_slice(slice);
        v
    }
}

// ── Deref / DerefMut (this is what makes indexing, slicing, `.len()` on a
//    `&[T]`-shaped API, etc. all "just work") ────────────────────────────────

impl<T, const N: usize> Deref for MidVec<T, N> {
    type Target = [T];
    #[inline]
    fn deref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T, const N: usize> DerefMut for MidVec<T, N> {
    #[inline]
    fn deref_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

// ── Drop ─────────────────────────────────────────────────────────────────────

impl<T, const N: usize> Drop for MidVec<T, N> {
    fn drop(&mut self) {
        let len = self.len_val();
        let on_heap = self.on_heap();

        // If we're on the heap, the allocation must be freed even if one
        // of the element drops below panics. We arrange that by building
        // a guard *before* dropping elements; the guard's own `Drop` then
        // runs *after* (reverse declaration order), once `drop_in_place`
        // has either finished normally or unwound past this point.
        struct DeallocGuard<T> {
            ptr: NonNull<T>,
            capacity: usize,
        }
        impl<T> Drop for DeallocGuard<T> {
            fn drop(&mut self) {
                // SAFETY: only constructed with a live heap allocation
                // made via `capacity_layout`, from `MidVec::drop` below.
                unsafe { raw::dealloc_heap::<T>(self.ptr, self.capacity) };
            }
        }

        let _guard = if on_heap {
            // SAFETY: `heap` is the active variant.
            let (ptr, capacity) = unsafe { self.raw.heap_parts() };
            Some(DeallocGuard { ptr, capacity })
        } else {
            None
        };

        // SAFETY: these `len` elements are exactly the initialised range,
        // regardless of which variant is active (`as_mut_ptr` handles that).
        unsafe {
            core::ptr::drop_in_place(core::slice::from_raw_parts_mut(self.as_mut_ptr(), len));
        }
        // `_guard` drops here, after the elements above.
    }
}

// ── Clone / Debug / PartialEq / Eq ──────────────────────────────────────────

impl<T: Clone, const N: usize> Clone for MidVec<T, N> {
    fn clone(&self) -> Self {
        Self::from_slice(self.as_slice())
    }
}

impl<T: fmt::Debug, const N: usize> fmt::Debug for MidVec<T, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

impl<T: PartialEq, const N: usize, const M: usize> PartialEq<MidVec<T, M>> for MidVec<T, N> {
    fn eq(&self, other: &MidVec<T, M>) -> bool {
        self.as_slice() == other.as_slice()
    }
}
impl<T: Eq, const N: usize> Eq for MidVec<T, N> {}

impl<T: PartialEq, const N: usize> PartialEq<[T]> for MidVec<T, N> {
    fn eq(&self, other: &[T]) -> bool {
        self.as_slice() == other
    }
}
impl<T: PartialEq, const N: usize> PartialEq<Vec<T>> for MidVec<T, N> {
    fn eq(&self, other: &Vec<T>) -> bool {
        self.as_slice() == other.as_slice()
    }
}

// ── conversions ──────────────────────────────────────────────────────────────

impl<T, const N: usize> From<Vec<T>> for MidVec<T, N> {
    #[inline]
    fn from(vec: Vec<T>) -> Self {
        Self::from_vec(vec)
    }
}

impl<T, const N: usize> From<MidVec<T, N>> for Vec<T> {
    #[inline]
    fn from(v: MidVec<T, N>) -> Self {
        v.into_vec()
    }
}

impl<T: Clone, const N: usize> From<&[T]> for MidVec<T, N> {
    #[inline]
    fn from(slice: &[T]) -> Self {
        Self::from_slice(slice)
    }
}

impl<T, const N: usize, const M: usize> From<[T; M]> for MidVec<T, N> {
    fn from(array: [T; M]) -> Self {
        let mut v = Self::with_capacity(M);
        for item in array {
            v.push(item);
        }
        v
    }
}

// ── iteration ────────────────────────────────────────────────────────────────

impl<T, const N: usize> IntoIterator for MidVec<T, N> {
    type Item = T;
    type IntoIter = IntoIter<T, N>;
    #[inline]
    fn into_iter(self) -> IntoIter<T, N> {
        IntoIter::new(self)
    }
}

impl<'a, T, const N: usize> IntoIterator for &'a MidVec<T, N> {
    type Item = &'a T;
    type IntoIter = core::slice::Iter<'a, T>;
    #[inline]
    fn into_iter(self) -> core::slice::Iter<'a, T> {
        self.iter()
    }
}

impl<'a, T, const N: usize> IntoIterator for &'a mut MidVec<T, N> {
    type Item = &'a mut T;
    type IntoIter = core::slice::IterMut<'a, T>;
    #[inline]
    fn into_iter(self) -> core::slice::IterMut<'a, T> {
        self.iter_mut()
    }
}

impl<T, const N: usize> FromIterator<T> for MidVec<T, N> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut v = Self::new();
        v.extend(iter);
        v
    }
}

impl<T, const N: usize> Extend<T> for MidVec<T, N> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        let mut iter = iter.into_iter();
        let (lower, _) = iter.size_hint();
        self.reserve(lower);
        // `size_hint().0` is a guaranteed *minimum*, so we can push exactly
        // that many elements via the unchecked fast path without redoing
        // a capacity check each time — the single `reserve` above already
        // covers all of them.
        for _ in 0..lower {
            match iter.next() {
                // SAFETY: `reserve(lower)` covers this element; we've
                // pushed fewer than `lower` items so far this loop.
                Some(item) => unsafe { self.push_unchecked(item) },
                None => return, // iterator undershot its own lower bound
            }
        }
        // Anything beyond the promised lower bound goes through the
        // normal checked path, since we have no guarantee capacity
        // already covers it.
        for item in iter {
            self.push(item);
        }
    }
                          }
