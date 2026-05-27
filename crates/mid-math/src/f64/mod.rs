// crates/mid-math/src/f64/mod.rs
// (full replacement — adds ddual_quat submodule)

pub mod dvec2;
pub mod dvec3;
pub mod dvec4;
pub mod dquat;
pub mod dmat2;
pub mod dmat3;
pub mod dmat4;
pub mod daffine3;
pub mod ddual_quat;                   // ← new

// ── SSE2 — x86 / x86_64 ──────────────────────────────────────────────────────

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub(crate) mod sse2;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub use sse2::{DVec2, DVec4, DQuat};

// ── AVX2 ─────────────────────────────────────────────────────────────────────

#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    target_feature = "avx2",
))]
pub(crate) mod avx2;

// ── NEON — aarch64 ────────────────────────────────────────────────────────────

#[cfg(target_arch = "aarch64")]
pub(crate) mod neon;

#[cfg(target_arch = "aarch64")]
pub use neon::{DVec2, DVec4, DQuat};

// ── WASM SIMD128 ──────────────────────────────────────────────────────────────

#[cfg(all(
    any(target_arch = "wasm32", target_arch = "wasm64"),
    target_feature = "simd128",
))]
pub(crate) mod wasm;

#[cfg(all(
    any(target_arch = "wasm32", target_arch = "wasm64"),
    target_feature = "simd128",
))]
pub use wasm::{DVec2, DVec4, DQuat};

// ── Scalar fallback ───────────────────────────────────────────────────────────

#[cfg(not(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(
        any(target_arch = "wasm32", target_arch = "wasm64"),
        target_feature = "simd128",
    ),
)))]
pub use self::{dvec2::DVec2, dvec4::DVec4, dquat::DQuat};

// ── Always-scalar re-exports ──────────────────────────────────────────────────

pub use dvec3::DVec3;
pub use dmat2::DMat2;
pub use dmat3::DMat3;
pub use dmat4::DMat4;
pub use daffine3::DAffine3;
pub use ddual_quat::DDualQuat;        // ← new

/// f64 comparison epsilon. 1e-12.
pub const DEPSILON: f64 = dvec2::DEPSILON;
