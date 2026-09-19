// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-alloc.md, section "segregator.rs"
// ============================================================================
//! [`Segregator`], routing an allocation to `Small` or `Large` based on
//! its size against a threshold. Directly modeled on
//! `foonathan::memory::binary_segregator<Segregatable, RawAllocator>`
//! (real source read, `segregator.hpp`) with `threshold_segregatable`'s
//! own `size <= max_size` check folded straight into this type's
//! `threshold` field, rather than kept as a separate wrapper type --
//! `RawAlloc`'s single always-fallible interface makes the extra layer
//! foonathan's own composable-vs-throwing split needed there
//! unnecessary here (see `crate::raw_alloc`'s doc comment for that same
//! simplification, applied consistently).
//!
//! # A real, load-bearing difference from `FallbackAllocator`, stated
//! directly rather than left for a reader to notice by comparing files
//!
//! `foonathan::binary_segregator::allocate_node` (checked against that
//! source, not assumed) does **not** fall through to the other
//! allocator if the size-selected one fails -- it calls that one
//! allocator's plain, non-composable allocation function and returns
//! whatever that gives, full stop. That is a real, deliberate
//! difference from `fallback_allocator`'s "try, then fall back"
//! behavior, not an oversight ported by accident: a segregator's whole
//! point is a deterministic size-based partition (this range always
//! goes here), not a resilience mechanism. `Segregator` keeps that
//! same hard partition. Deallocation routes the same way, by
//! re-checking the threshold against the given `size` rather than an
//! ownership probe -- sound as long as the caller passes the same
//! `size` it originally allocated with, the same contract any
//! `Layout`-based deallocation already requires.

use crate::raw_alloc::RawAlloc;
use core::ptr::NonNull;

/// Routes an allocation to `small` when `size <= threshold`, to
/// `large` otherwise -- no fallthrough either way. See this module's
/// doc comment for the real source this is modeled on and the one
/// real behavioral difference from [`FallbackAllocator`](crate::fallback::FallbackAllocator).
#[derive(Debug, Default)]
pub struct Segregator<Small, Large> {
    threshold: usize,
    small: Small,
    large: Large,
}

impl<Small, Large> Segregator<Small, Large> {
    /// Requests of `size <= threshold` route to `small`; everything
    /// larger routes to `large`.
    pub fn new(threshold: usize, small: Small, large: Large) -> Self {
        Self {
            threshold,
            small,
            large,
        }
    }

    /// The size threshold this segregator was built with.
    #[inline]
    pub fn threshold(&self) -> usize {
        self.threshold
    }

    /// The allocator handling requests at or below `threshold`.
    pub fn small(&self) -> &Small {
        &self.small
    }

    /// The allocator handling requests larger than `threshold`.
    pub fn large(&self) -> &Large {
        &self.large
    }

    #[inline]
    fn use_small(&self, size: usize) -> bool {
        size <= self.threshold
    }
}

impl<Small: RawAlloc, Large: RawAlloc> RawAlloc for Segregator<Small, Large> {
    #[inline]
    fn try_alloc_raw(&self, size: usize, align: usize) -> Option<NonNull<u8>> {
        if self.use_small(size) {
            self.small.try_alloc_raw(size, align)
        } else {
            self.large.try_alloc_raw(size, align)
        }
    }

    #[inline]
    unsafe fn try_dealloc_raw(&self, ptr: NonNull<u8>, size: usize, align: usize) -> bool {
        if self.use_small(size) {
            self.small.try_dealloc_raw(ptr, size, align)
        } else {
            self.large.try_dealloc_raw(ptr, size, align)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::raw_alloc::{HeapAlloc, NullAlloc};
    use crate::stack_allocator::StackAllocator;

    #[test]
    fn requests_at_or_below_threshold_use_small() {
        let s = Segregator::new(16, StackAllocator::with_capacity(64), HeapAlloc);
        let ptr = s.try_alloc_raw(16, 1).expect("exactly at threshold, should use small");
        // A pointer `small` (the `StackAllocator`) actually served must
        // fall inside its own buffer -- checked here through the
        // allocator itself rather than by trusting which branch ran.
        assert!(unsafe { s.small().try_dealloc_raw(ptr, 16, 1) });
    }

    #[test]
    fn requests_above_threshold_use_large() {
        let s = Segregator::new(16, StackAllocator::with_capacity(64), HeapAlloc);
        let ptr = s
            .try_alloc_raw(17, 1)
            .expect("one byte over threshold, should use large (heap)");
        unsafe {
            assert!(
                !s.small().try_dealloc_raw(ptr, 17, 1),
                "small (the 64-byte stack allocator) must not claim a pointer large served"
            );
            assert!(s.large().try_dealloc_raw(ptr, 17, 1));
        }
    }

    #[test]
    fn does_not_fall_through_when_the_selected_side_fails() {
        // `small` has room for only 4 bytes; a request of 4 (at or
        // under the 16-byte threshold) still routes to `small` and
        // must fail there, even though `large` (the heap) could have
        // served it easily -- the one real, deliberate difference from
        // `FallbackAllocator`'s try-then-fall-back behavior.
        let s = Segregator::new(16, StackAllocator::with_capacity(4), HeapAlloc);
        s.try_alloc_raw(4, 1).expect("first 4-byte request fills the tiny primary");
        assert!(
            s.try_alloc_raw(4, 1).is_none(),
            "small is now full; large must NOT be tried even though it easily could serve this"
        );
    }

    #[test]
    fn dealloc_routes_by_recomputing_the_threshold_not_by_probing_ownership() {
        let s = Segregator::new(8, NullAlloc, HeapAlloc);
        // NullAlloc never actually allocates, so this only exercises
        // the deallocation-routing decision itself, not a real pointer.
        let fake = core::ptr::NonNull::<u8>::dangling();
        unsafe {
            assert!(
                !s.try_dealloc_raw(fake, 8, 1),
                "size 8 is within the small threshold, routes to NullAlloc, which always refuses"
            );
        }
    }
}
