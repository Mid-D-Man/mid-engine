// crates/mid-math/src/f32/mod.rs
// (full replacement — adds dual_quat submodule)

pub(crate) mod math;

mod vec2;
pub mod mat2;
pub mod mat3;
pub mod affine3;
pub mod dual_quat;                    // ← new: moved from helpers/

pub use vec2::Vec2;
pub use mat2::Mat2;
pub use mat3::Mat3;
pub use affine3::Affine3;
pub use dual_quat::DualQuat;          // ← new

pub(crate) mod scalar;

// ── SSE2 — x86 / x86_64 ──────────────────────────────────────────────────────

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub(crate) mod sse2;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub use sse2::{Vec3, Vec4, Quat, Mat4};

// ── AVX2 — x86 / x86_64 with target_feature = "avx2" ────────────────────────

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), target_feature = "avx2"))]
pub(crate) mod avx2;

// ── NEON — aarch64 ────────────────────────────────────────────────────────────

#[cfg(target_arch = "aarch64")]
pub(crate) mod neon;

#[cfg(target_arch = "aarch64")]
pub use neon::{Vec3, Vec4, Quat, Mat4};

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
pub use wasm::{Vec3, Vec4, Quat, Mat4};

// ── Portable SIMD (coresimd) ──────────────────────────────────────────────────

#[cfg(feature = "coresimd")]
pub(crate) mod coresimd;

#[cfg(all(
    feature = "coresimd",
    not(any(
        target_arch = "x86",
        target_arch = "x86_64",
        target_arch = "aarch64",
        all(any(target_arch = "wasm32", target_arch = "wasm64"), target_feature = "simd128"),
    )),
))]
pub use coresimd::{Vec3, Vec4, Quat, Mat4};

// ── Scalar fallback ───────────────────────────────────────────────────────────

#[cfg(not(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(any(target_arch = "wasm32", target_arch = "wasm64"), target_feature = "simd128"),
    feature = "coresimd",
)))]
pub use scalar::{Vec3, Vec4, Quat, Mat4};
