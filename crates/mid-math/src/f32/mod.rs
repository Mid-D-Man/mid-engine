// crates/mid-math/src/f32/mod.rs
//! f32 math types — architecture dispatch.
//!
//! On x86 / x86_64 : SSE2 implementations (Vec3, Vec4, Quat back __m128).
//! On aarch64       : NEON implementations (future — scalar stub for now).
//! On wasm32/64 + simd128 : WASM SIMD (future — scalar stub for now).
//! Everywhere else  : Pure scalar fallback.
//!
//! Consumers of this crate see only the public re-exports from lib.rs
//! and never need to know which backend is active.

pub(crate) mod math;

// ── Always-scalar types ───────────────────────────────────────────────────────

mod vec2;    // Vec2: 8 bytes, tight UV/2D physics — no SIMD benefit
mod mat3;    // Mat3: 36 bytes scalar — normal matrix, 2D transforms

pub use vec2::Vec2;
pub use mat3::Mat3;

// ── Scalar reference implementations (always compiled) ────────────────────────
//
// These are the authoritative correctness reference and the fallback for
// non-SIMD targets. The scalar Mat4 methods (`inverse_scalar`,
// `inverse_trs_scalar`) live on the concrete type and are available on all
// platforms for testing even when the SIMD dispatch is active.

pub(crate) mod scalar;

// ── Architecture dispatch ─────────────────────────────────────────────────────

// x86 and x86_64 — SSE2 is guaranteed on every x86_64 CPU (no runtime check
// needed when gating on target_arch alone).
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub(crate) mod sse2;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub use sse2::{Vec3, Vec4, Quat, Mat4};

// aarch64 — NEON is mandatory on all AArch64 targets (iOS, Android, Apple Silicon).
// Currently stubs to scalar; replace with float32x4_t implementation later.
#[cfg(target_arch = "aarch64")]
pub(crate) mod neon;

#[cfg(target_arch = "aarch64")]
pub use neon::{Vec3, Vec4, Quat, Mat4};

// wasm32 / wasm64 with explicit simd128 feature flag.
// Requires: RUSTFLAGS="-C target-feature=+simd128"
#[cfg(all(
    any(target_arch = "wasm32", target_arch = "wasm64"),
    target_feature = "simd128",
))]
pub(crate) mod wasm;

#[cfg(all(
    any(target_arch = "wasm32", target_arch = "wasm64"),
    target_feature = "simd128",
))]
pub use wasm::{Vec3, Vec4, Quat, Mat4};

// Pure scalar fallback for every other target (MIPS, RISC-V, PowerPC, etc.)
#[cfg(not(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(
        any(target_arch = "wasm32", target_arch = "wasm64"),
        target_feature = "simd128",
    ),
)))]
pub use scalar::{Vec3, Vec4, Quat, Mat4};
