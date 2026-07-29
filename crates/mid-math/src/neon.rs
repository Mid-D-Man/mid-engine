// crates/mid-math/src/neon.rs
//! Shared AArch64 NEON helper primitives.
//!
//! The f32 type files (vec3, vec4, quat, mat4) use intrinsics directly — NEON
//! has richer primitives (vaddvq_f32, mandatory FMA, direct vabsq_f32) that
//! need no abstraction shims unlike SSE2.
//!
//! One primitive NEON does *not* give you for free: a fast reciprocal square
//! root. `vsqrtq_f32` + `vdivq_f32` (the naive `normalize` path) are both
//! non-pipelined, double-digit-cycle instructions on most AArch64 cores.
//! `rsqrt_nr_f32` below mirrors `crate::sse2::rsqrt_nr` — NEON's `vrsqrteq_f32`
//! estimate (~8-bit) + one `vrsqrtsq_f32` Newton–Raphson step (~23-bit) is a
//! couple of fast multiply-class instructions instead.
//!
//! This module also holds f64 NEON (float64x2_t) helpers, where a few
//! centralized helpers avoid copy-paste across dvec2/dvec4/dquat:
//!   - Const init helper (float64x2_t cannot be built in const context)
//!   - 4-lane dot product from two (lo, hi) float64x2_t pairs
//!   - 2-lane dot product
//!
//! All functions are `pub(crate) unsafe`.

use core::arch::aarch64::*;

// ── f32 reciprocal square root ────────────────────────────────────────────────

/// Reciprocal square root: `vrsqrteq_f32` (NEON RSQRTE, ~8-bit estimate) +
/// one Newton–Raphson step via `vrsqrtsq_f32` (NEON RSQRTS) → ~23-bit precision.
///
/// Replaces the expensive `vsqrtq_f32` + `vdivq_f32` pair used in `normalize`.
/// On AArch64 the estimate+refine sequence is two pipelined multiply-class
/// instructions vs. two non-pipelined ~10+ cycle ops.
///
/// Formula: `r1 = r0 * vrsqrtsq_f32(v, r0*r0)`, where `vrsqrtsq_f32(v, x)`
/// computes the standard NR correction term `(3 - v*x) / 2`.
///
/// `v` should be a broadcast — typically all 4 lanes hold the same
/// squared-length value (from `vdupq_n_f32(dot)`), so all 4 output lanes
/// receive the same refined reciprocal sqrt.
#[inline(always)]
pub(crate) unsafe fn rsqrt_nr_f32(v: float32x4_t) -> float32x4_t {
    let r0 = vrsqrteq_f32(v);
    vmulq_f32(r0, vrsqrtsq_f32(v, vmulq_f32(r0, r0)))
}

// ── Compile-time constant helper ──────────────────────────────────────────────

/// Build a `float64x2_t` from `[f64; 2]` at compile time via transmute.
///
/// `float64x2_t` has no const constructor. Both types are 16 bytes, align 16
/// on AArch64. Lane 0 = a[0], lane 1 = a[1].
#[inline(always)]
pub(crate) const fn f64x2_from_f64x2(a: [f64; 2]) -> float64x2_t {
    unsafe { core::mem::transmute(a) }
}

// ── Dot products ──────────────────────────────────────────────────────────────

/// 2-lane f64 dot product using `vaddvq_f64` (AArch64 ADDP.2D instruction).
///
/// `vaddvq_f64(vmulq_f64(a, b))` = a[0]*b[0] + a[1]*b[1]
/// No shuffle tricks needed — NEON handles horizontal add natively.
#[inline(always)]
pub(crate) unsafe fn dot2d_neon(a: float64x2_t, b: float64x2_t) -> f64 {
    vaddvq_f64(vmulq_f64(a, b))
}

/// 4-lane f64 dot product from two `(lo, hi)` float64x2_t pairs.
///
/// Computes `lo_a·lo_b + hi_a·hi_b` using two `vaddvq_f64` calls.
/// This is optimal on AArch64 — no shuffle needed, both partial sums
/// are scalar adds to get the final result.
#[inline(always)]
pub(crate) unsafe fn dot4d_neon(
    lo_a: float64x2_t, hi_a: float64x2_t,
    lo_b: float64x2_t, hi_b: float64x2_t,
) -> f64 {
    vaddvq_f64(vmulq_f64(lo_a, lo_b)) + vaddvq_f64(vmulq_f64(hi_a, hi_b))
                               }
