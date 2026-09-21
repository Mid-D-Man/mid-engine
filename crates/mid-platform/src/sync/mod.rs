// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-platform.md, section "sync/mod.rs"
// ============================================================================

//! Synchronization alternatives to language/`std` primitives that work the
//! same way whether or not `std` is available.
//!
//! Phase 1 only: `Mutex`/`MutexGuard`, `Arc`/`Weak`, `atomic::*`, and the
//! `poison` error types they share. `RwLock`, `Once`/`OnceLock`, `LazyLock`,
//! and `Barrier` are Phase 2 — see `docs/mid-platform.md`, "Build order,"
//! before adding any of them here.

pub use mutex::{Mutex, MutexGuard};
pub use poison::{LockResult, PoisonError, TryLockError, TryLockResult};

pub mod atomic;

mod mutex;
mod poison;

#[cfg(feature = "alloc")]
pub use arc::{Arc, Weak};

#[cfg(feature = "alloc")]
mod arc {
    //! `Arc`/`Weak` need `portable-atomic-util` in upstream `bevy_platform`
    //! only on targets lacking pointer-width native atomics
    //! (`[target.'cfg(not(target_has_atomic = "ptr"))'.dependencies]`,
    //! confirmed directly against `bevy_platform`'s own `Cargo.toml`). None of
    //! mid-engine's real targets (x86_64, aarch64, wasm32) lack that, so this
    //! is a plain, zero-cost re-export.
    pub use alloc::sync::{Arc, Weak};
}
