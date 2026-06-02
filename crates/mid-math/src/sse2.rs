// crates/mid-math/src/sse2.rs
//! Shared SSE2 helper primitives — f32 and f64.
//!
//! Used by Vec3, Vec4, Quat, Mat4 (f32) and DVec2, DVec4, DQuat (f64)
//! on x86 / x86_64. All functions are `pub(crate) unsafe`.

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

// ═══════════════════════════════════════════════════════════════════════════════
// ── F32 (__m128) helpers ─────────────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════════

/// Build a `__m128` from a `[f32; 4]` constant at compile time.
#[inline(always)]
pub(crate) const fn m128_from_f32x4(a: [f32; 4]) -> __m128 {
    unsafe { core::mem::transmute(a) }
}

/// 3-lane dot product. Result lands in lane 0; lanes 1-3 are unspecified.
#[inline(always)]
pub(crate) unsafe fn dot3_in_x(lhs: __m128, rhs: __m128) -> __m128 {
    let mul = _mm_mul_ps(lhs, rhs);
    let y   = _mm_shuffle_ps::<0b00_00_00_01>(mul, mul);
    let z   = _mm_shuffle_ps::<0b00_00_00_10>(mul, mul);
    let xy  = _mm_add_ps(mul, y);
    _mm_add_ps(xy, z)
}

/// 4-lane dot product. Result lands in lane 0; lanes 1-3 are unspecified.
#[inline(always)]
pub(crate) unsafe fn dot4_in_x(lhs: __m128, rhs: __m128) -> __m128 {
    let mul      = _mm_mul_ps(lhs, rhs);
    let zw_in_xy = _mm_shuffle_ps::<0b00_00_11_10>(mul, mul);
    let xz_yw    = _mm_add_ps(mul, zw_in_xy);
    let yw_in_0  = _mm_shuffle_ps::<0b00_00_00_01>(xz_yw, xz_yw);
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

/// Component-wise absolute value for f32x4. Clears sign bit via ANDNOT.
#[inline(always)]
pub(crate) unsafe fn m128_abs(v: __m128) -> __m128 {
    _mm_andnot_ps(_mm_set1_ps(-0.0), v)
}

/// Per-lane floor (SSE2, no SSE4.1 assumed).
#[inline(always)]
pub(crate) unsafe fn m128_floor(v: __m128) -> __m128 {
    let i    = _mm_cvttps_epi32(v);
    let fi   = _mm_cvtepi32_ps(i);
    let mask = _mm_cmpgt_ps(fi, v);
    let one  = _mm_set1_ps(1.0);
    _mm_sub_ps(fi, _mm_and_ps(mask, one))
}

/// Per-lane ceil (SSE2).
#[inline(always)]
pub(crate) unsafe fn m128_ceil(v: __m128) -> __m128 {
    let i    = _mm_cvttps_epi32(v);
    let fi   = _mm_cvtepi32_ps(i);
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
    let sign_mask = _mm_set1_ps(-0.0);
    let sign_bit  = _mm_and_ps(v, sign_mask);
    let half      = _mm_or_ps(sign_bit, _mm_set1_ps(0.5));
    m128_trunc(_mm_add_ps(v, half))
}

/// Apply `f32::sin` to each lane independently (scalar fallback for slerp/euler).
#[inline(always)]
pub(crate) unsafe fn m128_sin(v: __m128) -> __m128 {
    let x = _mm_cvtss_f32(v);
    let y = _mm_cvtss_f32(_mm_shuffle_ps::<0b01_01_01_01>(v, v));
    let z = _mm_cvtss_f32(_mm_shuffle_ps::<0b10_10_10_10>(v, v));
    let w = _mm_cvtss_f32(_mm_shuffle_ps::<0b11_11_11_11>(v, v));
    _mm_set_ps(w.sin(), z.sin(), y.sin(), x.sin())
}

/// Reciprocal square root: `_mm_rsqrt_ps` (14-bit) + one Newton–Raphson step (~23-bit).
///
/// Replaces the expensive `sqrt` + `div` pair in `normalize`.  On modern x86,
/// `rsqrt` is 1–3 cycles; `sqrt`+`div` is ~20–30 cycles combined.
///
/// Formula:  r₁ = r₀ · (1.5 − 0.5 · v · r₀²)
///
/// `v` must be a broadcast — all 4 lanes holding the same squared-length value.
/// All 4 output lanes receive the same refined reciprocal sqrt.
#[inline(always)]
pub(crate) unsafe fn rsqrt_nr(v: __m128) -> __m128 {
    let r    = _mm_rsqrt_ps(v);
    let half = _mm_set1_ps(0.5_f32);
    let c    = _mm_set1_ps(1.5_f32);
    // r₁ = r₀ · (1.5 − 0.5 · v · r₀²)
    let rr   = _mm_mul_ps(r, r);
    let nr   = _mm_sub_ps(c, _mm_mul_ps(half, _mm_mul_ps(v, rr)));
    _mm_mul_ps(r, nr)
}

// ═══════════════════════════════════════════════════════════════════════════════
// ── F64 (__m128d) helpers ────────────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════════

/// Build a `__m128d` from `[f64; 2]` at compile time via transmute.
///
/// Layout: lane 0 (lower 64 bits) = a[0], lane 1 (upper 64 bits) = a[1].
/// Safe: `__m128d` and `[f64; 2]` have identical size and 16-byte alignment.
#[inline(always)]
pub(crate) const fn m128d_from_f64x2(a: [f64; 2]) -> __m128d {
    unsafe { core::mem::transmute(a) }
}

/// 2-lane f64 dot product. Result in lane 0; lane 1 is unspecified.
///
/// Algorithm (SSE2, no HADD):
///   mul  = [x*rx, y*ry]
///   shuf = [y*ry, x*rx]  (swap via _mm_shuffle_pd imm=0b01)
///   sum  = [x*rx + y*ry, ...]
///
/// `_mm_shuffle_pd::<0b01>(a, b)`: result[0] = a[1], result[1] = b[0].
#[inline(always)]
pub(crate) unsafe fn dot2d_in_x(lhs: __m128d, rhs: __m128d) -> __m128d {
    let mul  = _mm_mul_pd(lhs, rhs);
    let shuf = _mm_shuffle_pd::<0b01>(mul, mul); // swap: [y*ry, x*rx]
    _mm_add_pd(mul, shuf)                         // [x*rx+y*ry, y*ry+x*rx]
}

/// Scalar f64 dot2 — extracts lane 0.
#[inline(always)]
pub(crate) unsafe fn dot2d(lhs: __m128d, rhs: __m128d) -> f64 {
    _mm_cvtsd_f64(dot2d_in_x(lhs, rhs))
}

/// Broadcast dot2 result to both lanes.
#[inline(always)]
pub(crate) unsafe fn dot2d_into_m128d(lhs: __m128d, rhs: __m128d) -> __m128d {
    // dot2d_in_x gives [dot, _]; _mm_shuffle_pd::<0b00>(d, d) → [d[0], d[0]]
    let d = dot2d_in_x(lhs, rhs);
    _mm_shuffle_pd::<0b00>(d, d)
}

/// 4-lane f64 dot product from two pairs of `__m128d` (lo=[x,y], hi=[z,w]).
/// Returns scalar result.
///
/// Algorithm:
///   lo_mul = [x*rx, y*ry];  lo_sum = x*rx + y*ry  (lane 0)
///   hi_mul = [z*rz, w*rw];  hi_sum = z*rz + w*rw  (lane 0)
///   total  = lo_sum + hi_sum
#[inline(always)]
pub(crate) unsafe fn dot4d(lo_a: __m128d, hi_a: __m128d,
                            lo_b: __m128d, hi_b: __m128d) -> f64 {
    let lo_sum = dot2d_in_x(lo_a, lo_b); // [x*rx+y*ry, _]
    let hi_sum = dot2d_in_x(hi_a, hi_b); // [z*rz+w*rw, _]
    _mm_cvtsd_f64(_mm_add_pd(lo_sum, hi_sum))
}

/// Broadcast dot4d result to a `__m128d` (both lanes = dot).
#[inline(always)]
pub(crate) unsafe fn dot4d_into_m128d(lo_a: __m128d, hi_a: __m128d,
                                       lo_b: __m128d, hi_b: __m128d) -> __m128d {
    let d = dot4d(lo_a, hi_a, lo_b, hi_b);
    _mm_set1_pd(d)
}

/// Absolute value for packed doubles — clears sign bit via ANDNOT.
///
/// `_mm_andnot_pd(a, b)` = `~a & b`. Clearing the sign bit: `~(-0.0) & v`.
#[inline(always)]
pub(crate) unsafe fn m128d_abs(v: __m128d) -> __m128d {
    _mm_andnot_pd(_mm_set1_pd(-0.0), v)
}
