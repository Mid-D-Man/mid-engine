// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-alloc.md, section "fallback.rs"
// ============================================================================
//! [`FallbackAllocator`], a [`RawAlloc`](crate::raw_alloc::RawAlloc)
//! with a fallback: tries `Primary` first, falls back to `Secondary`
//! if that fails. Directly modeled on
//! `foonathan::memory::fallback_allocator<Default, Fallback>` (real
//! source read, `fallback_allocator.hpp`). Its real dispatch really is
//! about this short once ported: try the primary's fallible
//! allocation, and only reach for the fallback if that came back
//! empty. See `crate::raw_alloc`'s own doc comment for the one real
//! simplification carried over from the C++ source (no separate
//! composable-vs-throwing split, since every `RawAlloc` here is
//! fallible by construction) and the one real property kept
//! (deallocation routes to whichever side actually owns a given
//! pointer, checked rather than guessed).

use crate::raw_alloc::RawAlloc;
use core::ptr::NonNull;

/// A [`RawAlloc`] with a fallback. See this module's doc comment for
/// the real source this is modeled on and what changed in the port.
#[derive(Debug, Default)]
pub struct FallbackAllocator<Primary, Secondary> {
    primary: Primary,
    secondary: Secondary,
}

impl<Primary, Secondary> FallbackAllocator<Primary, Secondary> {
    /// Builds a fallback allocator that tries `primary` first, then
    /// `secondary`.
    pub fn new(primary: Primary, secondary: Secondary) -> Self {
        Self { primary, secondary }
    }

    /// The primary allocator, tried first on every request.
    pub fn primary(&self) -> &Primary {
        &self.primary
    }

    /// The secondary (fallback) allocator, tried only once `primary`
    /// reports failure.
    pub fn secondary(&self) -> &Secondary {
        &self.secondary
    }
}

impl<Primary: RawAlloc, Secondary: RawAlloc> RawAlloc for FallbackAllocator<Primary, Secondary> {
    fn try_alloc_raw(&self, size: usize, align: usize) -> Option<NonNull<u8>> {
        self.primary
            .try_alloc_raw(size, align)
            .or_else(|| self.secondary.try_alloc_raw(size, align))
    }

    /// Matches `foonathan::fallback_allocator::deallocate_node`'s real
    /// dispatch: try `Default`'s composable `try_deallocate_node`
    /// first, fall to `Fallback` only if that reports it wasn't the
    /// owner (checked against that source above, not assumed). Correct
    /// here specifically because both sides report real ownership
    /// (see `RawAlloc::try_dealloc_raw`'s own contract) rather than
    /// unconditionally claiming every call -- a combinator built on an
    /// allocator that couldn't tell would have no sound way to route
    /// this at all.
    unsafe fn try_dealloc_raw(&self, ptr: NonNull<u8>, size: usize, align: usize) -> bool {
        self.primary.try_dealloc_raw(ptr, size, align)
            || self.secondary.try_dealloc_raw(ptr, size, align)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::raw_alloc::HeapAlloc;
    use crate::stack_allocator::StackAllocator;

    #[test]
    fn fallback_allocator_uses_secondary_once_primary_is_full() {
        let a = FallbackAllocator::new(StackAllocator::with_capacity(8), HeapAlloc);

        let from_primary = a.try_alloc_raw(4, 1).expect("fits in the 8-byte primary");
        let from_secondary = a
            .try_alloc_raw(64, 8)
            .expect("only 4 bytes left in primary, must come from the heap fallback");

        unsafe {
            core::ptr::write_bytes(from_primary.as_ptr(), 0xAA, 4);
            core::ptr::write_bytes(from_secondary.as_ptr(), 0xBB, 64);
            assert_eq!(*from_primary.as_ptr(), 0xAA);
            assert_eq!(*from_secondary.as_ptr(), 0xBB);

            // SAFETY: both pointers genuinely came from `a.try_alloc_raw`
            // above with these exact size/align pairs.
            assert!(a.try_dealloc_raw(from_primary, 4, 1));
            assert!(a.try_dealloc_raw(from_secondary, 64, 8));
        }
    }

    #[test]
    fn fallback_allocator_routes_dealloc_to_whichever_side_really_owns_it() {
        let a = FallbackAllocator::new(StackAllocator::with_capacity(8), HeapAlloc);
        let from_primary = a.try_alloc_raw(4, 1).unwrap();
        let from_secondary = a.try_alloc_raw(64, 8).unwrap();

        // The one real property this whole design exists for: primary
        // must correctly refuse a pointer it never allocated, proving
        // this isn't just guessing which side to free through.
        unsafe {
            assert!(
                a.primary().try_dealloc_raw(from_primary, 4, 1),
                "primary should recognize its own pointer"
            );
            assert!(
                !a.primary().try_dealloc_raw(from_secondary, 64, 8),
                "primary must not falsely claim a pointer the heap fallback actually served"
            );
            // Free the heap-backed one for real so this test doesn't leak.
            assert!(a.secondary().try_dealloc_raw(from_secondary, 64, 8));
        }
    }

    #[test]
    fn fallback_allocator_falls_all_the_way_through_when_primary_never_succeeds() {
        // A zero-capacity primary can still satisfy a zero-sized
        // request (see `StackAllocator::alloc_raw`'s own boundary
        // math), so use a request that genuinely cannot fit at all.
        let a = FallbackAllocator::new(StackAllocator::with_capacity(0), HeapAlloc);
        let ptr = a
            .try_alloc_raw(16, 8)
            .expect("primary has zero capacity, this must come entirely from the fallback");
        unsafe {
            assert!(a.try_dealloc_raw(ptr, 16, 8));
        }
    }
}
