// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-ptr.md, section "debug_align.rs"
// ============================================================================

use core::mem::align_of;

/// Crate-internal helper: asserts a `*mut T` is properly aligned for `T` in debug
/// builds, and is a no-op in release builds. Every erased-pointer `deref`/`read`/
/// `drop_as`-style call routes through this before casting to a concrete `*mut T`,
/// so a caller that got the `A: IsAligned` contract wrong panics with a clear message
/// in debug/test builds instead of silently reading through a misaligned pointer.
///
/// `pub(crate)` rather than private: it's used from every module that casts a
/// type-erased pointer back to a concrete type ([`crate::erased`], [`crate::moving`],
/// [`crate::thin_slice`]), not just this file.
pub(crate) trait DebugEnsureAligned {
    fn debug_ensure_aligned(self) -> Self;
}

// Disabled under miri: miri already checks that pointer-to-reference casts are
// properly aligned, so this would just be a redundant, slower second check.
#[cfg(all(debug_assertions, not(miri)))]
impl<T: Sized> DebugEnsureAligned for *mut T {
    #[track_caller]
    fn debug_ensure_aligned(self) -> Self {
        assert!(
            self.is_aligned(),
            "pointer is not aligned. Address {:p} does not have alignment {} for type {}",
            self,
            align_of::<T>(),
            core::any::type_name::<T>()
        );
        self
    }
}

#[cfg(any(not(debug_assertions), miri))]
impl<T: Sized> DebugEnsureAligned for *mut T {
    #[inline(always)]
    fn debug_ensure_aligned(self) -> Self {
        self
    }
}
