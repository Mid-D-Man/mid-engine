// crates/mid-math/src/f32/mod.rs  — full replacement
pub(crate) mod math;

mod vec2;
pub mod mat2;    // scalar fallback — always compiled, exported on non-x86 targets
pub mod mat3;
pub mod affine2;
pub mod affine3;
pub mod dual_quat;

pub use vec2::Vec2;
pub use mat3::Mat3;
pub use affine2::Affine2;
pub use affine3::Affine3;
pub use dual_quat::DualQuat;

pub(crate) mod scalar;

// ── x86 / x86_64 ─────────────────────────────────────────────────────────────

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub(crate) mod sse2;

/// On x86/x86_64: SSE2-backed Vec3, Vec4, Quat, Mat4, and Mat2.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub use sse2::{Vec3, Vec4, Quat, Mat4, Mat2};

// AVX + FMA fast paths — compiled when hardware and RUSTFLAGS support it.
// Currently provides: Mul<Mat4> for Mat4 (~3.2 ns vs 6.5 ns SSE2).
// Gating is symmetric: avx/mat4.rs has cfg(avx+fma), sse2/mat4.rs has cfg(not(avx+fma)).
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    target_feature = "avx",
    target_feature = "fma",
))]
pub(crate) mod avx;

// ── aarch64 ──────────────────────────────────────────────────────────────────

#[cfg(target_arch = "aarch64")]
pub(crate) mod neon;

#[cfg(target_arch = "aarch64")]
pub use neon::{Vec3, Vec4, Quat, Mat4};

/// aarch64 uses the scalar Mat2 until a NEON Mat2 is implemented.
#[cfg(target_arch = "aarch64")]
pub use mat2::Mat2;

// ── WASM SIMD128 ─────────────────────────────────────────────────────────────

#[cfg(all(any(target_arch = "wasm32", target_arch = "wasm64"), target_feature = "simd128"))]
pub(crate) mod wasm;

#[cfg(all(any(target_arch = "wasm32", target_arch = "wasm64"), target_feature = "simd128"))]
pub use wasm::{Vec3, Vec4, Quat, Mat4};

/// WASM uses the scalar Mat2 until a v128 Mat2 is implemented.
#[cfg(all(any(target_arch = "wasm32", target_arch = "wasm64"), target_feature = "simd128"))]
pub use mat2::Mat2;

// ── Portable SIMD (coresimd) ──────────────────────────────────────────────────

#[cfg(feature = "coresimd")]
pub(crate) mod coresimd;

#[cfg(all(
    feature = "coresimd",
    not(any(
        target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64",
        all(any(target_arch = "wasm32", target_arch = "wasm64"), target_feature = "simd128"),
    )),
))]
pub use coresimd::{Vec3, Vec4, Quat, Mat4};

#[cfg(all(
    feature = "coresimd",
    not(any(
        target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64",
        all(any(target_arch = "wasm32", target_arch = "wasm64"), target_feature = "simd128"),
    )),
))]
pub use mat2::Mat2;

// ── Scalar fallback ───────────────────────────────────────────────────────────

#[cfg(not(any(
    target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64",
    all(any(target_arch = "wasm32", target_arch = "wasm64"), target_feature = "simd128"),
    feature = "coresimd",
)))]
pub use scalar::{Vec3, Vec4, Quat, Mat4};

#[cfg(not(any(
    target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64",
    all(any(target_arch = "wasm32", target_arch = "wasm64"), target_feature = "simd128"),
    feature = "coresimd",
)))]
pub use mat2::Mat2;
