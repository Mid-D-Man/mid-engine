// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-alloc.md, section "raw_alloc.rs"
// ============================================================================
//! Shared raw-allocation interface for combinators (`fallback`,
//! `segregator`, ...) to build against, plus [`HeapAlloc`], the
//! natural terminal allocator for a combinator chain.
//!
//! # `RawAlloc`, a real simplification of `foonathan::memory`'s
//! `RawAllocator` concept, stated directly rather than left implied
//!
//! `foonathan::memory::allocator_traits` distinguishes a "composable"
//! (`try_`-prefixed, fallible) allocation/deallocation function from a
//! plain one that is allowed to throw or abort on failure --
//! `fallback_allocator.hpp`'s real dispatch (checked against that
//! source, not assumed) only reaches for the fallback allocator
//! through that split. Every allocator in this crate is fallible by
//! construction (`Option`-returning is already the Rust idiom this
//! crate uses throughout -- see `StackAllocator::alloc_raw`,
//! `PoolAllocator::create`), so there is no second, throwing variant
//! to distinguish here. `RawAlloc` collapses that split into one
//! always-fallible interface.
//!
//! The deallocation side keeps the one real property that split was
//! actually protecting: `memory_stack`'s own composable
//! `try_deallocate_node` doesn't unconditionally claim every call, it
//! checks real pointer ownership (`state.arena_.owns(ptr)`, checked
//! against that source directly) and reports whether it handled the
//! request. `RawAlloc::try_dealloc_raw` mirrors that exactly, `bool`
//! return included -- it is what lets [`fallback::FallbackAllocator`]
//! route a deallocation to whichever side actually owns it instead of
//! guessing.

use alloc::alloc::{alloc, dealloc};
use core::alloc::Layout;
use core::ptr::NonNull;

/// A source of raw, untyped memory that reports failure instead of
/// panicking or aborting. See this module's doc comment for how this
/// simplifies `foonathan::memory`'s real `RawAllocator` concept, and
/// what it keeps.
pub trait RawAlloc {
    /// Attempts to allocate `size` bytes aligned to `align`. Returns
    /// `None` on failure for any reason (out of capacity, out of
    /// memory, an invalid `align`, or anything else this particular
    /// allocator can't satisfy) rather than panicking or aborting.
    fn try_alloc_raw(&self, size: usize, align: usize) -> Option<NonNull<u8>>;

    /// Attempts to deallocate `ptr` (originally returned by some
    /// `RawAlloc::try_alloc_raw(size, align)` call, not necessarily
    /// this allocator's own). Returns `true` if this allocator
    /// recognizes `ptr` as its own and has handled the deallocation --
    /// which, for an arena-shaped allocator, may correctly mean doing
    /// nothing beyond the ownership check, since its real reclamation
    /// happens elsewhere (see `StackAllocator`'s impl in
    /// `stack_allocator.rs`). Returns `false` if `ptr` does not belong
    /// to this allocator at all.
    ///
    /// # Safety
    /// If this returns `true`, `ptr`/`size`/`align` must genuinely
    /// have come from a live `try_alloc_raw` call on this allocator
    /// that has not already been deallocated. A `false` return must
    /// mean this allocator is certain `ptr` is not its own --
    /// reporting `false` for a pointer it actually owns is safe (the
    /// caller simply tries elsewhere or leaks it), but reporting
    /// `true` for a pointer it does not own is undefined behavior the
    /// moment the caller trusts that and moves on.
    unsafe fn try_dealloc_raw(&self, ptr: NonNull<u8>, size: usize, align: usize) -> bool;
}

/// A `RawAlloc` that always defers to the global allocator. The
/// natural terminal allocator for a `FallbackAllocator`/segregator
/// chain, matching `foonathan::heap_allocator`'s own real role (source
/// read: a thin wrapper over a `malloc`/`free`-equivalent pair,
/// nothing more).
///
/// # A real, stated limitation, not one this type can check for itself
///
/// The global heap exposes no "is this pointer mine" query, unlike
/// `StackAllocator`'s own contiguous, address-range-checkable buffer.
/// [`try_dealloc_raw`](RawAlloc::try_dealloc_raw) here always claims
/// the request and frees it -- composing `HeapAlloc` as anything
/// other than the *last* allocator in a chain is unsound, since it
/// will wrongly claim (and free) a pointer that actually belongs to
/// whatever comes after it. `foonathan::heap_allocator` carries the
/// same real property; it is used as the terminal allocator for
/// exactly this reason there too, not composed ahead of others.
#[derive(Debug, Default, Clone, Copy)]
pub struct HeapAlloc;

impl RawAlloc for HeapAlloc {
    fn try_alloc_raw(&self, size: usize, align: usize) -> Option<NonNull<u8>> {
        if !align.is_power_of_two() {
            return None;
        }
        // A zero-sized request has no well-defined live allocation to
        // hand the global allocator -- `GlobalAlloc::alloc` disallows
        // zero-sized layouts entirely. Return a dangling, correctly
        // aligned pointer instead of ever calling into the allocator,
        // the same convention `Vec`'s own zero-sized-type handling
        // uses internally.
        if size == 0 {
            // SAFETY: `align` was just checked non-zero and a power of
            // two above, so this cast is a valid, non-null, dangling
            // pointer aligned to `align`.
            return Some(unsafe { NonNull::new_unchecked(align as *mut u8) });
        }
        let layout = Layout::from_size_align(size, align).ok()?;
        // SAFETY: `layout` has a non-zero size, checked above.
        let ptr = unsafe { alloc(layout) };
        NonNull::new(ptr)
    }

    unsafe fn try_dealloc_raw(&self, ptr: NonNull<u8>, size: usize, align: usize) -> bool {
        if size != 0 {
            // SAFETY: caller's contract on `try_dealloc_raw` above
            // guarantees `ptr`/`size`/`align` match a live
            // `try_alloc_raw` call -- `align` was already validated a
            // power of two by that call, so `from_size_align_unchecked`
            // here reconstructs the exact same, already-valid layout.
            let layout = Layout::from_size_align_unchecked(size, align);
            dealloc(ptr.as_ptr(), layout);
        }
        true
    }
}

/// A `RawAlloc` that always fails. Matches
/// `foonathan::null_allocator`'s own real role directly (source read):
/// useful as an explicit terminal in a combinator chain when no real
/// fallback should exist -- for example, `Segregator<Pool,
/// NullAlloc>` to allow only small allocations through and refuse
/// everything else outright, rather than silently reaching for the
/// heap. Unlike the C++ original, which throws on the non-composable
/// path, this always reports failure through the same `Option`/`bool`
/// interface every other `RawAlloc` here uses -- there is no second,
/// throwing variant to distinguish (see this module's own doc comment).
#[derive(Debug, Default, Clone, Copy)]
pub struct NullAlloc;

impl RawAlloc for NullAlloc {
    fn try_alloc_raw(&self, _size: usize, _align: usize) -> Option<NonNull<u8>> {
        None
    }

    unsafe fn try_dealloc_raw(&self, _ptr: NonNull<u8>, _size: usize, _align: usize) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn heap_alloc_round_trips_a_value() {
        let h = HeapAlloc;
        let ptr = h
            .try_alloc_raw(8, 8)
            .expect("a small heap allocation should not fail");
        unsafe {
            (ptr.as_ptr() as *mut u64).write(0x1122_3344_5566_7788);
            assert_eq!((ptr.as_ptr() as *const u64).read(), 0x1122_3344_5566_7788);
            assert!(h.try_dealloc_raw(ptr, 8, 8));
        }
    }

    #[test]
    fn heap_alloc_zero_sized_request_returns_a_dangling_aligned_pointer() {
        let h = HeapAlloc;
        let ptr = h
            .try_alloc_raw(0, 4)
            .expect("zero-sized requests should always succeed");
        assert_eq!(
            ptr.as_ptr() as usize % 4,
            0,
            "dangling pointer must still respect the requested alignment"
        );
        unsafe {
            assert!(h.try_dealloc_raw(ptr, 0, 4));
        }
    }

    #[test]
    fn heap_alloc_rejects_a_non_power_of_two_alignment() {
        let h = HeapAlloc;
        assert!(h.try_alloc_raw(8, 3).is_none());
    }

    #[test]
    fn null_alloc_always_fails() {
        let n = NullAlloc;
        assert!(n.try_alloc_raw(1, 1).is_none());
        let real = HeapAlloc.try_alloc_raw(1, 1).unwrap();
        unsafe {
            assert!(!n.try_dealloc_raw(real, 1, 1));
            assert!(HeapAlloc.try_dealloc_raw(real, 1, 1));
        }
    }
}
