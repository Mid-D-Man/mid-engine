// crates/mid-math/src/f32/mod.rs  — updated for Build 27: new platform stubs
pub(crate) mod math;

mod vec2;
pub mod mat2;
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

// AVX + FMA fast path — Mat4::mul only (~4.0 ns vs 7.0 ns SSE2).
// Gate: avx+fma present AND avx512f absent (avx512 supersedes when implemented).
// TODO: add `not(target_feature = "avx512f")` once avx512/mat4.rs has Mul<Mat4>.
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    target_feature = "avx",
    target_feature = "fma",
))]
pub(crate) mod avx;

// AVX-512 fast paths — STUB (no Mul<Mat4> yet, no conflict with avx/).
// Compiled when avx512f is present (CI ubuntu-latest has it).
// When avx512/mat4.rs implements Mul<Mat4>:
//   1. Gate avx/mat4.rs Mul<Mat4> with: not(target_feature = "avx512f")
//   2. Add "x86-64-v4" to bench-vs-all workflow target_cpu choices.
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    target_feature = "avx512f",
))]
pub(crate) mod avx512;

// ── aarch64 ──────────────────────────────────────────────────────────────────

#[cfg(target_arch = "aarch64")]
pub(crate) mod neon;

#[cfg(target_arch = "aarch64")]
pub use neon::{Vec3, Vec4, Quat, Mat4};

/// aarch64 uses the scalar Mat2 until a NEON Mat2 is implemented.
#[cfg(target_arch = "aarch64")]
pub use mat2::Mat2;

// SVE / SVE2 fast paths — STUB (nightly Rust only as of 2026-06).
// This cfg never fires on stable Rust. Wire-up is safe.
// When Rust SVE stabilizes + GitHub has SVE runners:
//   Extend to vec3/vec4/quat/mat4 submodules parallel to neon/.
#[cfg(all(target_arch = "aarch64", target_feature = "sve"))]
pub(crate) mod sve;

// SME (Scalable Matrix Extension) — STUB (no Rust support as of 2026-06).
// Requires: Rust SME intrinsics + macOS Sequoia/Linux 6.1 ZA context save.
// Hardware: Apple M4, Cortex-X4. This cfg never fires on any current toolchain.
#[cfg(all(target_arch = "aarch64", target_feature = "sme"))]
pub(crate) mod sme;

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
