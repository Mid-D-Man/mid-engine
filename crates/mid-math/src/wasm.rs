// crates/mid-math/src/wasm.rs
//! Shared WASM SIMD helper primitives (mirrors sse2.rs for wasm32/wasm64 + simd128).
//!
//! All helpers are `pub(crate) unsafe` — the target_feature gate is enforced at
//! the module level in lib.rs / f32/mod.rs.

#[cfg(target_arch = "wasm32")]
use core::arch::wasm32::*;
#[cfg(target_arch = "wasm64")]
use core::arch::wasm64::*;

// ── Compile-time constant helper ──────────────────────────────────────────────

/// Build a `v128` from `[f32; 4]` at compile time via transmute.
///
/// Required because `f32x4(...)` is not usable in fully const contexts.
/// Layout: lane 0 = a[0], lane 1 = a[1], lane 2 = a[2], lane 3 = a[3].
#[inline(always)]
pub(crate) const fn v128_from_f32x4(a: [f32; 4]) -> v128 {
    unsafe { core::mem::transmute(a) }
}

// ── Dot products ──────────────────────────────────────────────────────────────

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
