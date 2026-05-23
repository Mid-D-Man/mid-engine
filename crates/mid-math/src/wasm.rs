// crates/mid-math/src/wasm.rs
//! Shared WASM SIMD helper primitives (mirrors sse2.rs for wasm32/wasm64 + simd128).
//!
//! All helpers are `pub(crate) unsafe` — the target_feature gate is enforced at
//! the module level in lib.rs / f32/mod.rs.

#[cfg(target_arch = "wasm32")]
use core::arch::wasm32::*;
#[cfg(target_arch = "wasm64")]
use core::arch::wasm64::*;

// ═══════════════════════════════════════════════════════════════════════════════
// ── F32 helpers ───────────────────────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════════

/// Build a `v128` from `[f32; 4]` at compile time via transmute.
///
/// Required because `f32x4(...)` is not usable in fully const contexts.
/// Layout: lane 0 = a[0], lane 1 = a[1], lane 2 = a[2], lane 3 = a[3].
#[inline(always)]
pub(crate) const fn v128_from_f32x4(a: [f32; 4]) -> v128 {
    unsafe { core::mem::transmute(a) }
}

/// 3-lane dot product.  Result lands in lane 0; lanes 1-3 are unspecified.
///
/// Horizontal add pattern:
///   mul   = [ax·bx, ay·by, az·bz, 0]
///   y     = [ay·by, ay·by, ay·by, ay·by]   (splat lane 1)
///   z     = [az·bz, az·bz, az·bz, az·bz]   (splat lane 2)
///   lane0 = ax·bx + ay·by + az·bz
#[inline(always)]
pub(crate) unsafe fn dot3_in_x(a: v128, b: v128) -> v128 {
    let mul = f32x4_mul(a, b);
    let y   = i32x4_shuffle::<1, 1, 1, 1>(mul, mul);
    let z   = i32x4_shuffle::<2, 2, 2, 2>(mul, mul);
    f32x4_add(f32x4_add(mul, y), z)
}

/// Scalar f32 dot3.
#[inline(always)]
pub(crate) unsafe fn dot3(a: v128, b: v128) -> f32 {
    f32x4_extract_lane::<0>(dot3_in_x(a, b))
}

/// Broadcast dot3 result to all 4 lanes.
#[inline(always)]
pub(crate) unsafe fn dot3_into_v128(a: v128, b: v128) -> v128 {
    let d = dot3_in_x(a, b);
    i32x4_shuffle::<0, 0, 0, 0>(d, d)
}

/// 4-lane dot product.  Result lands in lane 0; lanes 1-3 are unspecified.
///
/// [x+z, y+w, ...] then add shifted [y+w, ...] into lane 0.
#[inline(always)]
pub(crate) unsafe fn dot4_in_x(a: v128, b: v128) -> v128 {
    let mul  = f32x4_mul(a, b);
    // [z, w, z, w] from second half of mul
    let zw   = i32x4_shuffle::<2, 3, 6, 7>(mul, mul);
    // [x+z, y+w, z+z, w+w]
    let xyzw = f32x4_add(mul, zw);
    // [y+w, y+w, ...]
    let yw   = i32x4_shuffle::<1, 1, 5, 5>(xyzw, xyzw);
    // lane 0 = (x+z) + (y+w) = sum of all 4
    f32x4_add(xyzw, yw)
}

/// Scalar f32 dot4.
#[inline(always)]
pub(crate) unsafe fn dot4(a: v128, b: v128) -> f32 {
    f32x4_extract_lane::<0>(dot4_in_x(a, b))
}

/// Broadcast dot4 result to all 4 lanes.
#[inline(always)]
pub(crate) unsafe fn dot4_into_v128(a: v128, b: v128) -> v128 {
    let d = dot4_in_x(a, b);
    i32x4_shuffle::<0, 0, 0, 0>(d, d)
}

// ═══════════════════════════════════════════════════════════════════════════════
// ── F64 (v128 as f64x2) helpers ──────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════════
//
// WASM f64x2: lane 0 = bytes 0-7 (x), lane 1 = bytes 8-15 (y).
// Same memory layout as [f64; 2] and SSE2 __m128d — Deref to XY<f64> is safe.
//
// Key differences from SSE2:
//   f64x2_abs, f64x2_neg   — direct instructions (no sign-mask tricks)
//   v128_andnot(a, b) = a & ~b  ← WASM (note: SSE2 _mm_andnot_pd is ~a & b)
//   No horizontal-add instruction; use lane-swap + add pattern.
//   No native FMA in SIMD128 baseline (relaxed-simd adds it).

/// Build a `v128` from `[f64; 2]` at compile time via transmute.
///
/// Both [f64; 2] and v128 are 16 bytes on WASM. Lane 0 = a[0], Lane 1 = a[1].
#[inline(always)]
pub(crate) const fn v128_from_f64x2(a: [f64; 2]) -> v128 {
    unsafe { core::mem::transmute(a) }
}

/// 2-lane f64 dot. Result broadcast to both lanes of the returned v128.
///
/// Algorithm — swap f64 lanes (each f64 = 2 i32, so swap pairs of i32):
///   mul     = [x·rx, y·ry]
///   swapped = i32x4_shuffle::<2,3,0,1>(mul, mul) → [y·ry, x·rx]
///   sum     = f64x2_add(mul, swapped) → [x·rx+y·ry, y·ry+x·rx]
///
/// Both output lanes equal the dot product — no extra broadcast needed.
#[inline(always)]
pub(crate) unsafe fn dot2d_in_x(a: v128, b: v128) -> v128 {
    let mul     = f64x2_mul(a, b);
    // i32x4_shuffle swapping i32 pairs [0,1] ↔ [2,3] = swapping f64 lanes.
    let swapped = i32x4_shuffle::<2, 3, 0, 1>(mul, mul);
    f64x2_add(mul, swapped)
}

/// Scalar f64 dot2 — extracts lane 0.
#[inline(always)]
pub(crate) unsafe fn dot2d(a: v128, b: v128) -> f64 {
    f64x2_extract_lane::<0>(dot2d_in_x(a, b))
}

/// Broadcast dot2 result to both f64 lanes.
///
/// `dot2d_in_x` already produces [dot, dot] so this is a no-op wrapper
/// kept for API symmetry with the SSE2 / NEON helpers.
#[inline(always)]
pub(crate) unsafe fn dot2d_into_v128(a: v128, b: v128) -> v128 {
    dot2d_in_x(a, b) // both lanes already equal the dot value
}

/// 4-lane f64 dot from two (lo, hi) v128 pairs.
///
/// lo = [x, y], hi = [z, w].
/// Returns scalar: x·rx + y·ry + z·rz + w·rw.
#[inline(always)]
pub(crate) unsafe fn dot4d(
    lo_a: v128, hi_a: v128,
    lo_b: v128, hi_b: v128,
) -> f64 {
    dot2d(lo_a, lo_b) + dot2d(hi_a, hi_b)
}

/// Broadcast dot4d result to both lanes of a v128.
#[inline(always)]
pub(crate) unsafe fn dot4d_into_v128(
    lo_a: v128, hi_a: v128,
    lo_b: v128, hi_b: v128,
) -> v128 {
    f64x2_splat(dot4d(lo_a, hi_a, lo_b, hi_b))
        }
