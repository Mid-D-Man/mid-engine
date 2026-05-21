// crates/mid-math/src/f32/coresimd/mod.rs
//! Rust portable SIMD backend — `core::simd` / `std::simd`.
//!
//! Uses `f32x4` (= `Simd<f32, 4>`) as the storage type for Vec3, Vec4, Quat.
//! A single implementation covers ANY target: x86, ARM, RISC-V, LoongArch,
//! WASM, future ISAs. The compiler lowers to the best available instructions.
//!
//! Enabled via `--features coresimd`. Requires `#![feature(portable_simd)]`
//! on nightly Rust; once portable_simd stabilizes the feature gate becomes
//! a no-op.
//!
//! ## Why not replace SSE2/NEON/WASM?
//!
//! Platform-specific backends give exact control over instruction selection
//! and are already verified against benchmarks. `coresimd` fills the gap for
//! platforms those don't cover — RISC-V with V extension, LoongArch LASX, etc.
//!
//! ## Key portable SIMD ops used
//!
//! | Operation            | Portable SIMD             | vs SSE2 equivalent    |
//! |----------------------|---------------------------|-----------------------|
//! | Lane broadcast       | `Simd::splat(v)`          | `_mm_set1_ps`         |
//! | Swizzle / shuffle    | `simd_swizzle!(v, [...])`  | `_mm_shuffle_ps`      |
//! | Comparison → mask    | `.simd_gt(other)`          | `_mm_cmpgt_ps`        |
//! | Select with mask     | `mask.select(a, b)`        | `_mm_and_ps` + OR     |
//! | Bitwise as f32       | `f32x4::from_bits(u32x4)` | `_mm_and_ps`          |
//! | Horizontal reduce    | `.reduce_sum()`            | shuffle + add chain   |
//! | Floor / ceil / round | `StdFloat::floor()`        | `_mm_floor_ps` (SSE4) |
//! | Sqrt                 | `StdFloat::sqrt()`         | `_mm_sqrt_ps`         |
//!
//! ## Cross-compilation
//!
//! Build for RISC-V (requires nightly + cross):
//!   cargo +nightly build --target riscv64gc-unknown-linux-gnu --features coresimd

#[cfg(feature = "coresimd")]
pub mod vec3;
#[cfg(feature = "coresimd")]
pub mod vec4;
#[cfg(feature = "coresimd")]
pub mod quat;
#[cfg(feature = "coresimd")]
pub mod mat4;

#[cfg(feature = "coresimd")]
pub use vec3::Vec3;
#[cfg(feature = "coresimd")]
pub use vec4::Vec4;
#[cfg(feature = "coresimd")]
pub use quat::Quat;
#[cfg(feature = "coresimd")]
pub use mat4::Mat4;

// ── Shared portable-SIMD helpers ──────────────────────────────────────────────
// These are pub(crate) so vec3/vec4/quat/mat4 can import without repeating.

#[cfg(feature = "coresimd")]
use core::simd::prelude::*;
#[cfg(feature = "coresimd")]
use core::simd::{cmp::SimdPartialOrd, num::SimdFloat};
#[cfg(feature = "coresimd")]
use std::simd::StdFloat;

/// 3-lane dot product. Zeroes lane 3 before reduce_sum to avoid padding noise.
///
/// `simd_swizzle!(mul, zero, [0,1,2,4])`:
///   indices 0-3 select from `mul`, indices 4-7 from `zero`.
///   Result = [mul[0], mul[1], mul[2], 0.0] → reduce_sum = x+y+z.
#[cfg(feature = "coresimd")]
#[inline(always)]
pub(crate) fn dot3(a: f32x4, b: f32x4) -> f32 {
    let mul  = a * b;
    let zero = f32x4::splat(0.0);
    simd_swizzle!(mul, zero, [0, 1, 2, 4]).reduce_sum()
}

/// Broadcast dot3 result to all 4 lanes.
#[cfg(feature = "coresimd")]
#[inline(always)]
pub(crate) fn dot3_into_f32x4(a: f32x4, b: f32x4) -> f32x4 {
    f32x4::splat(dot3(a, b))
}

/// 4-lane dot product.
#[cfg(feature = "coresimd")]
#[inline(always)]
pub(crate) fn dot4(a: f32x4, b: f32x4) -> f32 {
    (a * b).reduce_sum()
}

/// Broadcast dot4 result to all 4 lanes.
#[cfg(feature = "coresimd")]
#[inline(always)]
pub(crate) fn dot4_into_f32x4(a: f32x4, b: f32x4) -> f32x4 {
    f32x4::splat(dot4(a, b))
}

/// Bitwise AND on f32x4 via u32x4 reinterpret — used for normalize guard mask.
#[cfg(feature = "coresimd")]
#[inline(always)]
pub(crate) fn f32x4_bitand(a: f32x4, b: f32x4) -> f32x4 {
    f32x4::from_bits(a.to_bits() & b.to_bits())
}

/// Bitwise OR on f32x4 via u32x4 reinterpret.
#[cfg(feature = "coresimd")]
#[inline(always)]
pub(crate) fn f32x4_bitor(a: f32x4, b: f32x4) -> f32x4 {
    f32x4::from_bits(a.to_bits() | b.to_bits())
}

/// Bitwise XOR on f32x4 via u32x4 reinterpret — used for neg / conjugate.
#[cfg(feature = "coresimd")]
#[inline(always)]
pub(crate) fn f32x4_bitxor(a: f32x4, b: f32x4) -> f32x4 {
    f32x4::from_bits(a.to_bits() ^ b.to_bits())
}
