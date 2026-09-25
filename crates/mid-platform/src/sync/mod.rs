// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-platform.md, section "sync/mod.rs"
// ============================================================================

//! Synchronization alternatives to language/`std` primitives that work the
//! same way whether or not `std` is available.
//!
//! Phase 1 and Phase 2 both done — `Mutex`/`MutexGuard`, `RwLock`/
//! `RwLockReadGuard`/`RwLockWriteGuard`, `Once`/`OnceLock`/`OnceState`,
//! `LazyLock`, `Barrier`/`BarrierWaitResult`, `Arc`/`Weak`, `atomic::*`, and
//! the `poison` error types they all share. See `docs/mid-platform.md`,
//! "Build order," for Phase 3 (fast hasher, `HashMap`/`HashSet`) — not
//! decided yet, don't add either here without that decision being made
//! first.

pub use mutex::{Mutex, MutexGuard};
pub use poison::{LockResult, PoisonError, TryLockError, TryLockResult};
pub use rwlock::{RwLock, RwLockReadGuard, RwLockWriteGuard};
pub use once::{Once, OnceLock, OnceState};
pub use lazy_lock::LazyLock;
pub use barrier::{Barrier, BarrierWaitResult};

pub mod atomic;

mod barrier;
mod lazy_lock;
mod mutex;
mod once;
mod poison;
mod rwlock;

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
