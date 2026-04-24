// crates/mid-math/src/sse2.rs
//! Shared SSE2 helper primitives.
//!
//! Used by Vec3, Vec4, Quat and Mat4 on x86 / x86_64.
//! All functions are `pub(crate) unsafe` — callers guarantee the target.
//!
//! Source reference: adapted from glam/src/sse2.rs (MIT/Apache-2.0).

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

// ── Compile-time constant helper ──────────────────────────────────────────────

/// Build a `__m128` from a `[f32; 4]` constant at compile time.
///
/// `_mm_set_ps` is not `const`, so this transmute is needed for
/// compile-time sign masks, identity constants etc.
///
/// # Safety
/// `[f32; 4]` and `__m128` have identical size and alignment on all
/// x86 targets. Every bit pattern of f32 is valid.
#[inline(always)]
pub(crate) const fn m128_from_f32x4(a: [f32; 4]) -> __m128 {
    unsafe { core::mem::transmute(a) }
}

// ── Dot products ──────────────────────────────────────────────────────────────

/// 3-lane dot product. Result lands in lane 0; lanes 1-3 are unspecified.
///
/// Equivalent to `a.x*b.x + a.y*b.y + a.z*b.z` but avoids scalar extraction.
#[inline(always)]
pub(crate) unsafe fn dot3_in_x(lhs: __m128, rhs: __m128) -> __m128 {
    let mul = _mm_mul_ps(lhs, rhs);
    // Shuffle lane 1 (y) into lane 0: IMM bits[1:0]=01 → lane0 = src[1]
    let y = _mm_shuffle_ps::<0b00_00_00_01>(mul, mul);
    // Shuffle lane 2 (z) into lane 0: IMM bits[1:0]=10 → lane0 = src[2]
    let z = _mm_shuffle_ps::<0b00_00_00_10>(mul, mul);
    let xy = _mm_add_ps(mul, y);
    _mm_add_ps(xy, z)
}

/// 4-lane dot product. Result lands in lane 0; lanes 1-3 are unspecified.
#[inline(always)]
pub(crate) unsafe fn dot4_in_x(lhs: __m128, rhs: __m128) -> __m128 {
    let mul = _mm_mul_ps(lhs, rhs);
    // [z, w, x, x] via IMM=0b00_00_11_10: lane0=src[2], lane1=src[3]
    let zw_in_xy = _mm_shuffle_ps::<0b00_00_11_10>(mul, mul);
    // lane0=x+z, lane1=y+w
    let xz_yw = _mm_add_ps(mul, zw_in_xy);
    // move lane1 (y+w) into lane0: IMM bits[1:0]=01 → lane0=src[1]
    let yw_in_0 = _mm_shuffle_ps::<0b00_00_00_01>(xz_yw, xz_yw);
    // lane0 = (x+z) + (y+w) = dot4
    _mm_add_ps(xz_yw, yw_in_0)
}

/// Broadcast dot3 result to all 4 lanes.
#[inline(always)]
pub(crate) unsafe fn dot3_into_m128(lhs: __m128, rhs: __m128) -> __m128 {
    let dot = dot3_in_x(lhs, rhs);
    _mm_shuffle_ps::<0b00_00_00_00>(dot, dot)
}

/// Broadcast dot4 result to all 4 lanes.
#[inline(always)]
pub(crate) unsafe fn dot4_into_m128(lhs: __m128, rhs: __m128) -> __m128 {
    let dot = dot4_in_x(lhs, rhs);
    _mm_shuffle_ps::<0b00_00_00_00>(dot, dot)
}

/// Scalar f32 dot3.
#[inline(always)]
pub(crate) unsafe fn dot3(lhs: __m128, rhs: __m128) -> f32 {
    _mm_cvtss_f32(dot3_in_x(lhs, rhs))
}

/// Scalar f32 dot4.
#[inline(always)]
pub(crate) unsafe fn dot4(lhs: __m128, rhs: __m128) -> f32 {
    _mm_cvtss_f32(dot4_in_x(lhs, rhs))
}

// ── Absolute value ────────────────────────────────────────────────────────────

/// Component-wise absolute value. Clears sign bit via ANDNOT with -0.0.
#[inline(always)]
pub(crate) unsafe fn m128_abs(v: __m128) -> __m128 {
    _mm_andnot_ps(_mm_set1_ps(-0.0), v)
}

// ── Rounding (SSE2 — no SSE4.1 assumed) ──────────────────────────────────────
//
// SSE2 only provides truncation-toward-zero via _mm_cvttps_epi32.
// floor / ceil / round are emulated. SSE4.1 adds _mm_floor_ps etc., but
// we target SSE2 baseline (every x86_64 CPU since 2003).

/// Per-lane floor (round toward negative infinity).
#[inline(always)]
pub(crate) unsafe fn m128_floor(v: __m128) -> __m128 {
    let i  = _mm_cvttps_epi32(v);        // truncate toward zero
    let fi = _mm_cvtepi32_ps(i);          // back to float
    // if fi > v the truncation overshot (negative non-integer) → subtract 1
    let mask = _mm_cmpgt_ps(fi, v);
    let one  = _mm_set1_ps(1.0);
    _mm_sub_ps(fi, _mm_and_ps(mask, one))
}

/// Per-lane ceil (round toward positive infinity).
#[inline(always)]
pub(crate) unsafe fn m128_ceil(v: __m128) -> __m128 {
    let i  = _mm_cvttps_epi32(v);
    let fi = _mm_cvtepi32_ps(i);
    // if fi < v the truncation undershot (positive non-integer) → add 1
    let mask = _mm_cmplt_ps(fi, v);
    let one  = _mm_set1_ps(1.0);
    _mm_add_ps(fi, _mm_and_ps(mask, one))
}

/// Per-lane truncation toward zero.
#[inline(always)]
pub(crate) unsafe fn m128_trunc(v: __m128) -> __m128 {
    _mm_cvtepi32_ps(_mm_cvttps_epi32(v))
}

/// Per-lane round-to-nearest (half-away-from-zero).
#[inline(always)]
pub(crate) unsafe fn m128_round(v: __m128) -> __m128 {
    // Copy sign bit of v onto 0.5, add, then truncate.
    let sign_mask = _mm_set1_ps(-0.0);
    let sign_bit  = _mm_and_ps(v, sign_mask);
    let half      = _mm_or_ps(sign_bit, _mm_set1_ps(0.5));
    m128_trunc(_mm_add_ps(v, half))
}

// ── Scalar sin (lane-by-lane bridge for slerp / euler) ───────────────────────
//
// There is no SSE2 sin instruction. Until we add a polynomial approximation,
// we extract scalars, call libm, and repack. This is only called from
// Quat::slerp and euler conversions — not in the hot Vec3 path.

/// Apply `f32::sin` to each lane independently.
#[inline(always)]
pub(crate) unsafe fn m128_sin(v: __m128) -> __m128 {
    let x = _mm_cvtss_f32(v);
    let y = _mm_cvtss_f32(_mm_shuffle_ps::<0b01_01_01_01>(v, v));
    let z = _mm_cvtss_f32(_mm_shuffle_ps::<0b10_10_10_10>(v, v));
    let w = _mm_cvtss_f32(_mm_shuffle_ps::<0b11_11_11_11>(v, v));
    // _mm_set_ps(lane3, lane2, lane1, lane0)
    _mm_set_ps(w.sin(), z.sin(), y.sin(), x.sin())
            }
