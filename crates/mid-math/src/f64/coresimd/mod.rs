// crates/mid-math/src/f64/coresimd/mod.rs
//! Rust portable SIMD backend for f64 — `core::simd` / `std::simd`.
//!
//! Mirrors f32/coresimd/mod.rs exactly, at f64 width. Uses `f64x4`
//! (= `Simd<f64, 4>`) as the storage type for DVec3 -- the same padded-to-4
//! trick f32's coresimd Vec3 uses with f32x4.
//!
//! ## Why DVec3 is the interesting case here, more than it was for f32
//!
//! f32 Vec3 already gets real SIMD on every hardware-specific backend (SSE2
//! alone fits 3+1 f32 lanes in one __m128) -- coresimd's f32 Vec3 exists for
//! portability to unlisted architectures (RISC-V, LoongArch), not because
//! hardware-specific f32 Vec3 was missing anything.
//!
//! f64 DVec3 is different: it's scalar EVERYWHERE right now (see
//! f64/mod.rs's "always-scalar" list), and not by oversight -- a padded
//! 3+1 f64 vector needs 256 bits, so SSE2's 128-bit __m128d literally
//! cannot hold it the way __m128 holds f32 Vec3. AVX (256-bit __m256d)
//! could, but nobody's built that hardware-specific backend yet, and it'd
//! only help x86 anyway. `f64x4` sidesteps the whole problem: it doesn't
//! commit to a specific register width at the Rust source level, so the
//! compiler can lower it to one 256-bit AVX op, two 128-bit SSE2 ops, NEON
//! pairs, or a portable fallback, whatever the target actually has -- DVec3
//! gets a real chance at vectorization on hardware that literally cannot
//! give it one any other way in this codebase.
//!
//! Enabled via `--features coresimd`. Same pub(crate) visibility and
//! zero-CI-coverage status as f32's version -- see the note in this crate's
//! session history about that; wasn't invented for this file, inherited
//! from the pattern it mirrors. Added a smoke test in tests/f64_tests.rs
//! for at least this one type since nothing else in coresimd had any
//! coverage at all before now.
//!
//! Scope note: this only covers DVec3 for now, matching the highest-value/
//! most-novel case first rather than porting all four of f32 coresimd's
//! types (vec3/vec4/quat/mat4) uncompiled in one pass. DVec4 and DQuat
//! already have real hardware-specific backends (sse2/neon/wasm) elsewhere
//! in the crate, so they don't have the same "structurally can't get SIMD
//! any other way" story DVec3 does -- lower priority, add if/when wanted.

#[cfg(feature = "coresimd")]
pub mod dvec3;

#[cfg(feature = "coresimd")]
pub use dvec3::DVec3;

// ── Shared portable-SIMD helpers ──────────────────────────────────────────────

#[cfg(feature = "coresimd")]
use core::simd::prelude::*;
#[cfg(feature = "coresimd")]
use core::simd::{cmp::SimdPartialOrd, num::SimdFloat};
#[cfg(feature = "coresimd")]
use std::simd::StdFloat;

/// 3-lane dot product. Zeroes lane 3 before reduce_sum to avoid padding noise.
#[cfg(feature = "coresimd")]
#[inline(always)]
pub(crate) fn dot3(a: f64x4, b: f64x4) -> f64 {
    let mul  = a * b;
    let zero = f64x4::splat(0.0);
    simd_swizzle!(mul, zero, [0, 1, 2, 4]).reduce_sum()
}

/// Broadcast dot3 result to all 4 lanes.
#[cfg(feature = "coresimd")]
#[inline(always)]
pub(crate) fn dot3_into_f64x4(a: f64x4, b: f64x4) -> f64x4 {
    f64x4::splat(dot3(a, b))
}
