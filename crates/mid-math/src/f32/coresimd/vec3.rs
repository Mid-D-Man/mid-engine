// crates/mid-math/src/f32/coresimd/vec3.rs
//! Vec3 backed by `f32x4` (Rust portable SIMD).
//!
//! Lane layout: [x, y, z, 0] — lane 3 is always 0 (padding).
//! Same API and layout guarantees as SSE2/NEON/WASM Vec3.
//!
//! `simd_swizzle!` replaces `_mm_shuffle_ps` / `vextq_f32` / `i32x4_shuffle`:
//!   simd_swizzle!(v, [2, 0, 1, 1])   →   [v[2], v[0], v[1], v[1]]
//!   simd_swizzle!(a, b, [0, 1, 2, 4]) →  [a[0], a[1], a[2], b[0]]
//!
//! `mask.select(a, b)` replaces the SSE2 ANDNOT+OR mask pattern.

use core::fmt;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use core::simd::prelude::*;
use core::simd::{cmp::SimdPartialEq, cmp::SimdPartialOrd, num::SimdFloat};
use std::simd::StdFloat;

use super::{dot3, dot3_into_f32x4, f32x4_bitand};
use crate::f32::vec2::Vec2;
use crate::impl_vec3_deref;
use crate::EPSILON;

// Vec4 import — needed for extend()
use crate::f32::coresimd::vec4::Vec4;

// ── Type ──────────────────────────────────────────────────────────────────────

/// 3-dimensional vector. 16 bytes, 16-byte aligned. Backed by `f32x4`.
///
/// Lane 3 is always 0 (padding). Arithmetic ops that commute with zero in
/// lane 3 (add, sub, mul, lerp, normalize) preserve it automatically.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct Vec3(pub(crate) f32x4);

// Deref to XYZ<f32>: f32x4 has same layout as [f32; 4], so first 3 floats
// are x, y, z — identical to XYZ<f32> {x@0, y@4, z@8}.
impl_vec3_deref!(Vec3);

// ── Constants ─────────────────────────────────────────────────────────────────

impl Vec3 {
    // f32x4::from_array is const — no union trick needed.
    pub const ZERO:  Self = Self(f32x4::from_array([ 0.0,  0.0,  0.0, 0.0]));
    pub const ONE:   Self = Self(f32x4::from_array([ 1.0,  1.0,  1.0, 0.0]));
    pub const X:     Self = Self(f32x4::from_array([ 1.0,  0.0,  0.0, 0.0]));
    pub const Y:     Self = Self(f32x4::from_array([ 0.0,  1.0,  0.0, 0.0]));
    pub const Z:     Self = Self(f32x4::from_array([ 0.0,  0.0,  1.0, 0.0]));
    pub const NEG_X: Self = Self(f32x4::from_array([-1.0,  0.0,  0.0, 0.0]));
    pub const NEG_Y: Self = Self(f32x4::from_array([ 0.0, -1.0,  0.0, 0.0]));
    pub const NEG_Z: Self = Self(f32x4::from_array([ 0.0,  0.0, -1.0, 0.0]));

    // ── Constructors ─────────────────────────────────────────────────────────

    #[inline(always)]
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self(f32x4::from_array([x, y, z, 0.0]))
    }

    #[inline(always)]
    pub fn splat(v: f32) -> Self {
        // Explicit new() keeps lane 3 = 0 (f32x4::splat would set it to v).
        Self::new(v, v, v)
    }

    #[inline(always)] pub fn from_array(a: [f32; 3]) -> Self { Self::new(a[0], a[1], a[2]) }
    #[inline(always)] pub fn to_array(self) -> [f32; 3] { [self.x, self.y, self.z] }

    /// Extend to Vec4, setting lane 3 = `w`.
    #[inline(always)]
    pub fn extend(self, w: f32) -> Vec4 {
        Vec4::new(self.x, self.y, self.z, w)
    }

    #[inline(always)]
    pub fn truncate(self) -> Vec2 { Vec2::new(self.x, self.y) }

    // ── Arithmetic ────────────────────────────────────────────────────────────

    /// Dot product — zeroes lane 3 before reduce_sum.
    #[inline(always)]
    pub fn dot(self, rhs: Self) -> f32 { dot3(self.0, rhs.0) }

    #[inline]
    pub fn dot_into_vec(self, rhs: Self) -> Self { Self(dot3_into_f32x4(self.0, rhs.0)) }

    /// Cross product via cyclic-permutation swizzle.
    ///
    /// Derivation (same as WASM/SSE2/NEON versions):
    ///   lhszxy = [lz, lx, ly, ly]  swizzle [2,0,1,1]
    ///   rhszxy = [rz, rx, ry, ry]  swizzle [2,0,1,1]
    ///   sub    = lhszxy*rhs - rhszxy*lhs = [lz*rx-rz*lx, lx*ry-rx*ly, ly*rz-ry*lz, _]
    ///   result = swizzle(sub, [2,0,1,1]) = [ly*rz-ry*lz, lz*rx-rz*lx, lx*ry-rx*ly]
    ///          = [result.x, result.y, result.z] ✓
    #[inline(always)]
    pub fn cross(self, rhs: Self) -> Self {
        let lhszxy     = simd_swizzle!(self.0, [2, 0, 1, 1]);
        let rhszxy     = simd_swizzle!(rhs.0,  [2, 0, 1, 1]);
        let lhszxy_rhs = lhszxy * rhs.0;
        let rhszxy_lhs = rhszxy * self.0;
        let sub        = lhszxy_rhs - rhszxy_lhs;
        Self(simd_swizzle!(sub, [2, 0, 1, 1]))
    }

    // ── Length / normalise ────────────────────────────────────────────────────

    #[inline(always)] pub fn length_sq(self) -> f32 { self.dot(self) }

    #[inline]
    pub fn length(self) -> f32 {
        // sqrt of dot3 — StdFloat provides .sqrt() for f32x4
        dot3_into_f32x4(self.0, self.0).sqrt()[0]
    }

    #[inline]
    pub fn length_recip(self) -> f32 {
        let l = self.length();
        if l < EPSILON { 0.0 } else { 1.0 / l }
    }

    /// Normalize. `mask.select(norm, ZERO)` replaces SSE2's ANDNOT+OR pattern.
    #[inline]
    pub fn normalize(self) -> Self {
        let len    = dot3_into_f32x4(self.0, self.0).sqrt();
        let norm   = self.0 / len;
        let mask   = len.simd_gt(f32x4::splat(EPSILON));
        Self(mask.select(norm, f32x4::splat(0.0)))
    }

    #[inline]
    pub fn try_normalize(self) -> Option<Self> {
        let rcp = self.length_recip();
        if rcp.is_finite() && rcp > 0.0 { Some(self * rcp) } else { None }
    }

    #[inline] pub fn normalize_or(self, fb: Self) -> Self { self.try_normalize().unwrap_or(fb) }
    #[inline] pub fn normalize_or_zero(self) -> Self { self.normalize_or(Self::ZERO) }
    #[inline] pub fn is_normalized(self) -> bool { (self.length_sq() - 1.0).abs() <= 2e-4 }

    // ── Interpolation / geometry ──────────────────────────────────────────────

    /// Lerp: `self + (rhs - self) * t`.
    /// No FMA intrinsic needed — compiler fuses on FMA-capable targets automatically.
    #[inline]
    pub fn lerp(self, rhs: Self, t: f32) -> Self {
        let tt = f32x4::splat(t);
        Self(self.0 + (rhs.0 - self.0) * tt)
    }

    #[inline] pub fn reflect(self, n: Self) -> Self { self - n * (2.0 * self.dot(n)) }
    #[inline] pub fn distance(self, rhs: Self)    -> f32 { (self - rhs).length() }
    #[inline] pub fn distance_sq(self, rhs: Self) -> f32 { (self - rhs).length_sq() }

    // ── Component-wise ────────────────────────────────────────────────────────

    /// `simd_lt(rhs).select(self, rhs)` — portable pmin (NaN handling may vary by platform).
    #[inline]
    pub fn min(self, rhs: Self) -> Self {
        Self(self.0.simd_lt(rhs.0).select(self.0, rhs.0))
    }

    #[inline]
    pub fn max(self, rhs: Self) -> Self {
        Self(self.0.simd_gt(rhs.0).select(self.0, rhs.0))
    }

    #[inline] pub fn clamp(self, lo: Self, hi: Self) -> Self { self.max(lo).min(hi) }

    /// Abs via `SimdFloat::abs()` — direct instruction on all platforms.
    #[inline] pub fn abs(self) -> Self { Self(self.0.abs()) }

    /// Floor via `StdFloat::floor()` (needs std; scalar fallback otherwise).
    #[inline] pub fn floor(self) -> Self { Self(self.0.floor()) }
    #[inline] pub fn ceil(self)  -> Self { Self(self.0.ceil()) }
    #[inline] pub fn round(self) -> Self { Self(self.0.round()) }

    // ── Predicates ────────────────────────────────────────────────────────────

    #[inline] pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite()
    }
    #[inline] pub fn is_nan(self) -> bool {
        self.x.is_nan() || self.y.is_nan() || self.z.is_nan()
    }
    #[inline] pub fn approx_eq(self, rhs: Self) -> bool {
        (self - rhs).abs().length_sq() < EPSILON * EPSILON
    }
    #[inline]
    pub fn approx_eq_eps(self, rhs: Self, eps: f32) -> bool {
        (self.x - rhs.x).abs() < eps
            && (self.y - rhs.y).abs() < eps
            && (self.z - rhs.z).abs() < eps
    }
}

// ── Operators ─────────────────────────────────────────────────────────────────

impl Add for Vec3 {
    type Output = Self;
    #[inline(always)] fn add(self, r: Self) -> Self { Self(self.0 + r.0) }
}
impl Sub for Vec3 {
    type Output = Self;
    #[inline(always)] fn sub(self, r: Self) -> Self { Self(self.0 - r.0) }
}
impl Mul<f32> for Vec3 {
    type Output = Self;
    #[inline(always)] fn mul(self, s: f32) -> Self { Self(self.0 * f32x4::splat(s)) }
}
impl Mul<Vec3> for f32 {
    type Output = Vec3;
    #[inline(always)] fn mul(self, v: Vec3) -> Vec3 { Vec3(f32x4::splat(self) * v.0) }
}
impl Mul for Vec3 {
    type Output = Self;
    #[inline(always)] fn mul(self, r: Self) -> Self { Self(self.0 * r.0) }
}
impl Div<f32> for Vec3 {
    type Output = Self;
    #[inline(always)] fn div(self, s: f32) -> Self { Self(self.0 / f32x4::splat(s)) }
}
impl Neg for Vec3 {
    type Output = Self;
    #[inline(always)] fn neg(self) -> Self { Self(-self.0) }
}

impl AddAssign for Vec3 { #[inline(always)] fn add_assign(&mut self, r: Self) { self.0 += r.0; } }
impl SubAssign for Vec3 { #[inline(always)] fn sub_assign(&mut self, r: Self) { self.0 -= r.0; } }
impl MulAssign<f32> for Vec3 { #[inline(always)] fn mul_assign(&mut self, s: f32) { self.0 *= f32x4::splat(s); } }
impl DivAssign<f32> for Vec3 { #[inline(always)] fn div_assign(&mut self, s: f32) { self.0 /= f32x4::splat(s); } }

impl PartialEq for Vec3 {
    #[inline]
    fn eq(&self, rhs: &Self) -> bool {
        // Check lanes 0,1,2 only — lane 3 is padding and may differ after ops.
        (self.0.simd_eq(rhs.0).to_bitmask() & 0b0111) == 0b0111
    }
}

impl Default for Vec3 { fn default() -> Self { Self::ZERO } }

impl fmt::Debug for Vec3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Vec3").field(&self.x).field(&self.y).field(&self.z).finish()
    }
}
impl fmt::Display for Vec3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {}, {})", self.x, self.y, self.z)
    }
}

impl From<[f32; 3]> for Vec3 { #[inline] fn from(a: [f32; 3]) -> Self { Self::new(a[0], a[1], a[2]) } }
impl From<Vec3> for [f32; 3] { #[inline] fn from(v: Vec3) -> Self { [v.x, v.y, v.z] } }
impl From<(f32, f32, f32)> for Vec3 { #[inline] fn from(t: (f32, f32, f32)) -> Self { Self::new(t.0, t.1, t.2) } }
impl From<Vec3> for (f32, f32, f32) { #[inline] fn from(v: Vec3) -> Self { (v.x, v.y, v.z) } }
