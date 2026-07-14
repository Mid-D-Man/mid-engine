// crates/mid-math/src/mid_vec/raw.rs
//! Low-level backing storage for `MidVec<T, N>`: either `N` elements
//! living directly in the struct, or a heap allocation once the vector
//! outgrows them. This module owns all raw pointer / allocation logic;
//! `mod.rs` owns the safe public API built on top of it.

use core::alloc::Layout;
use core::mem::{size_of, ManuallyDrop, MaybeUninit};
use core::ptr::NonNull;

/// Union of "N elements stored inline" vs. "elements stored on the heap".
///
/// No discriminant is stored here — the owning `MidVec` tracks which
/// variant is active via the tag bit in its packed length (see `mod.rs`).
/// This keeps `RawMidVec` itself the size of `max(inline, heap)` rather
/// than paying for an enum discriminant on top.
pub(super) union RawMidVec<T, const N: usize> {
    inline: ManuallyDrop<MaybeUninit<[T; N]>>,
    /// `(pointer, capacity)`. Length is tracked separately by the owner.
    heap: (NonNull<T>, usize),
}

impl<T, const N: usize> RawMidVec<T, N> {
    #[inline]
    pub(super) const fn new_inline() -> Self {
        Self { inline: ManuallyDrop::new(MaybeUninit::uninit()) }
    }

    #[inline]
    pub(super) const fn new_heap(ptr: NonNull<T>, capacity: usize) -> Self {
        Self { heap: (ptr, capacity) }
    }

    /// # Safety
    /// The inline variant must be the currently active one.
    #[inline]
    pub(super) unsafe fn as_ptr_inline(&self) -> *const T {
        self.inline.as_ptr().cast::<T>()
    }

    /// # Safety
    /// The inline variant must be the currently active one.
    #[inline]
    pub(super) unsafe fn as_mut_ptr_inline(&mut self) -> *mut T {
        self.inline.as_mut_ptr().cast::<T>()
    }

    /// # Safety
    /// The heap variant must be the currently active one.
    #[inline]
    pub(super) unsafe fn heap_parts(&self) -> (NonNull<T>, usize) {
        self.heap
    }
}

/// Computes the `Layout` for `capacity` contiguous `T`s, panicking (rather
/// than silently overflowing or under-allocating) if `capacity` is so
/// large the allocation couldn't possibly be valid. None of the functions
/// below take an `N` type parameter — heap allocation doesn't depend on
/// how much inline capacity the vector happens to have.
#[inline]
pub(super) fn capacity_layout<T>(capacity: usize) -> Layout {
    match Layout::array::<T>(capacity) {
        Ok(layout) if layout.size() <= isize::MAX as usize => layout,
        _ => capacity_overflow(),
    }
}

#[cold]
#[inline(never)]
pub(super) fn capacity_overflow() -> ! {
    panic!("MidVec: requested capacity exceeds isize::MAX bytes")
}

/// Allocate fresh heap storage for `new_capacity` elements and copy `len`
/// elements in from `src` (which may point at either an inline array or an
/// existing, smaller heap block belonging to the caller).
///
/// # Safety
/// - `T` must not be a zero-sized type.
/// - `new_capacity >= len`.
/// - `src` must be valid for `len` reads of `T`.
pub(super) unsafe fn alloc_and_copy_from<T>(
    new_capacity: usize,
    src: *const T,
    len: usize,
) -> (NonNull<T>, usize) {
    debug_assert!(size_of::<T>() > 0);
    let layout = capacity_layout::<T>(new_capacity);
    let ptr = std::alloc::alloc(layout) as *mut T;
    let ptr = match NonNull::new(ptr) {
        Some(p) => p,
        None => std::alloc::handle_alloc_error(layout),
    };
    core::ptr::copy_nonoverlapping(src, ptr.as_ptr(), len);
    (ptr, new_capacity)
}

/// Grow an *existing* heap allocation of `old_capacity` elements at `ptr`
/// to `new_capacity` elements, reusing the allocation in place when the
/// allocator can manage it (no copy needed on our part).
///
/// # Safety
/// - `T` must not be a zero-sized type.
/// - `ptr` must have been allocated by `alloc_and_copy_from` /
///   `grow_heap_in_place` with exactly `old_capacity` elements of `T`,
///   using the global allocator.
/// - `new_capacity >= old_capacity`.
pub(super) unsafe fn grow_heap_in_place<T>(
    ptr: NonNull<T>,
    old_capacity: usize,
    new_capacity: usize,
) -> (NonNull<T>, usize) {
    debug_assert!(size_of::<T>() > 0);
    let old_layout = capacity_layout::<T>(old_capacity);
    let new_layout = capacity_layout::<T>(new_capacity);
    let new_ptr = std::alloc::realloc(ptr.as_ptr() as *mut u8, old_layout, new_layout.size());
    let new_ptr = match NonNull::new(new_ptr as *mut T) {
        Some(p) => p,
        None => std::alloc::handle_alloc_error(new_layout),
    };
    (new_ptr, new_capacity)
}

/// # Safety
/// - `T` must not be a zero-sized type.
/// - `ptr`/`capacity` must describe a live allocation made the same way as
///   `alloc_and_copy_from` (global allocator, `capacity_layout`).
pub(super) unsafe fn dealloc_heap<T>(ptr: NonNull<T>, capacity: usize) {
    debug_assert!(size_of::<T>() > 0);
    if capacity > 0 {
        let layout = capacity_layout::<T>(capacity);
        std::alloc::dealloc(ptr.as_ptr() as *mut u8, layout);
    }
      }
