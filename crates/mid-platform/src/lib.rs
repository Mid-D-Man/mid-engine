// crates/mid-platform/src/lib.rs
//! Common platform-agnostic primitives for mid-engine, informed by Bevy's
//! `bevy_platform` crate (MIT/Apache-2.0) but **not** a port of it in the same
//! sense `mid-ptr` is a port of `bevy_ptr`. `bevy_platform` itself is not
//! zero-dependency (`spin`, `portable-atomic`, `foldhash`, `hashbrown`,
//! `critical-section`, and several target/feature-gated others); this crate
//! is `bevy_platform`-shaped (same names, same std-vs-fallback split) but
//! hand-rolled and dependency-free, since none of mid-engine's real targets
//! (native desktop, `wasm32-unknown-unknown`) actually need the platforms
//! those dependencies exist for. See `docs/mid-platform.md` for the full,
//! dependency-by-dependency reasoning and the phased build plan this crate
//! is following.
//!
//! **Phase 1 and Phase 2 both done** (`docs/mid-platform.md`, "Build
//! order"): [`sync::atomic`] (a plain re-export of [`core::sync::atomic`]),
//! [`cell::SyncCell`] / [`cell::SyncUnsafeCell`], [`sync::Mutex`] /
//! [`sync::MutexGuard`], [`sync::Arc`] / [`sync::Weak`], [`sync::RwLock`] /
//! [`sync::RwLockReadGuard`] / [`sync::RwLockWriteGuard`], [`sync::Once`] /
//! [`sync::OnceLock`] / [`sync::OnceState`], [`sync::LazyLock`], and
//! [`sync::Barrier`] / [`sync::BarrierWaitResult`]. A fast hasher and a
//! `HashMap`/`HashSet` are Phase 3 — separate, later decisions, not an
//! oversight.
//!
//! Every lock type here works identically whether the `std` feature (on by
//! default) is enabled or not: with `std`, it's a thin pass-through to
//! `std::sync`; without it, a self-contained spin-based fallback takes over,
//! with no change to the type or method names either way.

#![cfg_attr(not(feature = "std"), no_std)]
// This crate's own [lints] workspace = true (Cargo.toml) pulls in the root
// workspace's `unsafe_code = "deny"` — opted back out of here, explicitly, per
// docs/RUST_AND_CRATE_GUIDELINES.md §3. mid-math and mid-ptr are the two
// crates that had already opted into `[lints] workspace = true` before this
// one; every `unsafe` block below carries its own `// SAFETY:` comment or
// `# Safety` doc section, same model.
#![allow(unsafe_code)]

#[cfg(feature = "alloc")]
extern crate alloc;
#[cfg(feature = "std")]
extern crate std;

pub mod cell;
pub mod sync;

/// C-compatible FFI exports. `std`-only — see this module's own doc
/// comment for why (a linkable `cdylib`/`staticlib` needs a real OS
/// underneath it regardless). First pass covers `Once`/`Barrier` only;
/// `Mutex`/`RwLock`/`OnceLock`/`LazyLock` are open work — see
/// `docs/mid-platform.md`, "Fixes and Problems."
#[cfg(feature = "std")]
pub mod ffi;
