// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-platform.md, section "sync/atomic.rs"
// ============================================================================

//! Atomic types, portable across every target this crate actually supports.
//!
//! Upstream `bevy_platform` falls back to the `portable-atomic` crate on
//! targets lacking a native atomic width (8/16/32/64/ptr), confirmed directly
//! against its own `[target.'cfg(not(all(target_has_atomic = "8", "16",
//! "32", "64", "ptr")))'.dependencies]` gate. None of mid-engine's real
//! targets (x86_64, aarch64, wasm32) lack any of those, so this module is a
//! plain re-export with no fallback dependency — see `docs/mid-platform.md`'s
//! dependency table for `portable-atomic`/`portable-atomic-util`.
//!
//! The `compile_error!` below exists so that if this crate is ever actually
//! built for a target that lacks full-width atomics, the build fails loudly
//! here with an explanation, instead of silently miscompiling or waiting to
//! be discovered as a runtime bug. Add a real `portable-atomic`-style
//! fallback then — not speculatively now, per this crate's own trigger-based
//! discipline (`docs/mid-platform.md`, "Build order").
#[cfg(not(all(
    target_has_atomic = "8",
    target_has_atomic = "16",
    target_has_atomic = "32",
    target_has_atomic = "64",
    target_has_atomic = "ptr"
)))]
compile_error!(
    "mid-platform::sync::atomic has no fallback for a target missing a native atomic width \
     (upstream bevy_platform uses the `portable-atomic` crate here). This target was never on \
     mid-engine's list when Phase 1 was built -- see docs/mid-platform.md's dependency table \
     before adding one."
);

pub use core::sync::atomic::{
    AtomicBool, AtomicI16, AtomicI32, AtomicI64, AtomicI8, AtomicIsize, AtomicPtr, AtomicU16,
    AtomicU32, AtomicU64, AtomicU8, AtomicUsize, Ordering,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn atomics_are_the_real_core_ones() {
        let x = AtomicU32::new(1);
        x.fetch_add(1, Ordering::SeqCst);
        assert_eq!(x.load(Ordering::SeqCst), 2);
    }
}
