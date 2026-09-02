// crates/mid-math/src/f32/wasm/vec3.rs
//! Vec3 backed by `v128` on wasm32/wasm64 with simd128 target feature.

#[cfg(target_arch = "wasm32")]
use core::arch::wasm32::*;
#[cfg(target_arch = "wasm64")]
use core::arch::wasm64::*;

use core::fmt;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use crate::f32::wasm::vec4::Vec4;
use crate::f32::vec2::Vec2;
use crate::impl_vec3_deref;
use crate::wasm::{dot3, dot3_in_x, dot3_into_v128};
use crate::EPSILON;

#[repr(C)]
union UnionCast { f: [f32; 4], v: Vec3 }

/// 3-dimensional vector. 16 bytes, 16-byte aligned. Backed by `v128`.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct Vec3(pub(crate) v128);

impl_vec3_deref!(Vec3);

impl Vec3 {
    pub const ZERO:  Self = unsafe { UnionCast { f: [ 0.0,  0.0,  0.0, 0.0] }.v };
    pub const ONE:   Self = unsafe { UnionCast { f: [ 1.0,  1.0,  1.0, 0.0] }.v };
    pub const X:     Self = unsafe { UnionCast { f: [ 1.0,  0.0,  0.0, 0.0] }.v };
    pub const Y:     Self = unsafe { UnionCast { f: [ 0.0,  1.0,  0.0, 0.0] }.v };
    pub const Z:     Self = unsafe { UnionCast { f: [ 0.0,  0.0,  1.0, 0.0] }.v };
    pub const NEG_X: Self = unsafe { UnionCast { f: [-1.0,  0.0,  0.0, 0.0] }.v };
    pub const NEG_Y: Self = unsafe { UnionCast { f: [ 0.0, -1.0,  0.0, 0.0] }.v };
    pub const NEG_Z: Self = unsafe { UnionCast { f: [ 0.0,  0.0, -1.0, 0.0] }.v };
    pub const NAN:   Self = unsafe { UnionCast { f: [f32::NAN, f32::NAN, f32::NAN, f32::NAN] }.v };

    #[inline(always)]
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        unsafe { UnionCast { f: [x, y, z, 0.0] }.v }
    }

    #[inline(always)] pub fn splat(v: f32) -> Self { unsafe { UnionCast { f: [v, v, v, 0.0] }.v } }
    #[inline(always)] pub fn from_array(a: [f32; 3]) -> Self { Self::new(a[0], a[1], a[2]) }
    #[inline(always)] pub fn to_array(self) -> [f32; 3] { [self.x, self.y, self.z] }

    #[inline(always)] pub fn extend(self, w: f32) -> Vec4 { Vec4::new(self.x, self.y, self.z, w) }
    #[inline(always)] pub fn truncate(self) -> Vec2 { Vec2::new(self.x, self.y) }

    #[inline(always)]
    pub fn dot(self, rhs: Self) -> f32 { unsafe { dot3(self.0, rhs.0) } }

    #[inline]
    pub fn dot_into_vec(self, rhs: Self) -> Self { Self(unsafe { dot3_into_v128(self.0, rhs.0) }) }

    #[inline(always)]
    pub fn cross(self, rhs: Self) -> Self {
        let lhszxy     = i32x4_shuffle::<2, 0, 1, 1>(self.0, self.0);
        let rhszxy     = i32x4_shuffle::<2, 0, 1, 1>(rhs.0,  rhs.0);
        let lhszxy_rhs = f32x4_mul(lhszxy, rhs.0);
        let rhszxy_lhs = f32x4_mul(rhszxy, self.0);
        let sub        = f32x4_sub(lhszxy_rhs, rhszxy_lhs);
        Self(i32x4_shuffle::<2, 0, 1, 1>(sub, sub))
    }

    #[inline(always)] pub fn length_sq(self) -> f32 { self.dot(self) }

    #[inline]
    pub fn length(self) -> f32 {
        unsafe { f32x4_extract_lane::<0>(f32x4_sqrt(dot3_in_x(self.0, self.0))) }
    }

    #[inline]
    pub fn length_recip(self) -> f32 {
        let l = self.length();
        if l < EPSILON { 0.0 } else { 1.0 / l }
    }

    #[inline]
    pub fn normalize(self) -> Self {
        unsafe {
            let len_v      = f32x4_sqrt(dot3_into_v128(self.0, self.0));
            let normalized = Self(f32x4_div(self.0, len_v));
            let ok         = f32x4_gt(len_v, f32x4_splat(EPSILON));
            Self(v128_and(normalized.0, ok))
        }
    }

    #[inline]
    pub fn try_normalize(self) -> Option<Self> {
        let rcp = self.length_recip();
        if rcp > 0.0 && rcp.is_finite() { Some(self * rcp) } else { None }
    }

    #[inline] pub fn normalize_or(self, fallback: Self) -> Self { self.try_normalize().unwrap_or(fallback) }
    #[inline] pub fn normalize_or_zero(self) -> Self { self.normalize_or(Self::ZERO) }
    #[inline] pub fn is_normalized(self) -> bool { (self.length_sq() - 1.0).abs() <= 2e-4 }

    #[inline]
    pub fn lerp(self, rhs: Self, t: f32) -> Self {
        let tt   = f32x4_splat(t);
        let diff = f32x4_sub(rhs.0, self.0);
        Self(f32x4_add(self.0, f32x4_mul(diff, tt)))
    }

    #[inline] pub fn reflect(self, n: Self) -> Self { self - n * (2.0 * self.dot(n)) }
    #[inline] pub fn distance(self, rhs: Self)    -> f32 { (self - rhs).length() }
    #[inline] pub fn distance_sq(self, rhs: Self) -> f32 { (self - rhs).length_sq() }

    #[inline] pub fn min(self, rhs: Self) -> Self { Self(f32x4_pmin(self.0, rhs.0)) }
    #[inline] pub fn max(self, rhs: Self) -> Self { Self(f32x4_pmax(self.0, rhs.0)) }
    #[inline] pub fn clamp(self, lo: Self, hi: Self) -> Self { self.max(lo).min(hi) }
    #[inline] pub fn abs(self) -> Self { Self(f32x4_abs(self.0)) }

    #[inline] pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite()
    }
    #[inline] pub fn is_nan(self) -> bool {
        self.x.is_nan() || self.y.is_nan() || self.z.is_nan()
    }
    #[inline] pub fn approx_eq(self, rhs: Self) -> bool {
        (self - rhs).abs().length_sq() < EPSILON * EPSILON
    }
    #[inline] pub fn abs_diff_eq(self, rhs: Self, max_abs_diff: f32) -> bool {
        (self.x - rhs.x).abs() < max_abs_diff
            && (self.y - rhs.y).abs() < max_abs_diff
            && (self.z - rhs.z).abs() < max_abs_diff
    }
    #[inline] pub fn approx_eq_eps(self, rhs: Self, eps: f32) -> bool {
        (self.x - rhs.x).abs() < eps
            && (self.y - rhs.y).abs() < eps
            && (self.z - rhs.z).abs() < eps
    }

    // ── p1: angle between ─────────────────────────────────────────────────────

    #[inline]
    pub fn angle_between(self, rhs: Self) -> f32 {
        let denom = (self.length_sq() * rhs.length_sq()).sqrt();
        if denom < EPSILON { 0.0 } else { (self.dot(rhs) / denom).clamp(-1.0, 1.0).acos() }
    }

    // ── p2: project / reject ──────────────────────────────────────────────────

    #[inline]
    pub fn project_onto(self, rhs: Self) -> Self {
        let d = rhs.length_sq();
        if d < EPSILON { Self::ZERO } else { rhs * (self.dot(rhs) / d) }
    }

    #[inline]
    pub fn reject_from(self, rhs: Self) -> Self { self - self.project_onto(rhs) }

    // ── p7: movement / clamping helpers ──────────────────────────────────────

    #[inline]
    pub fn move_towards(self, target: Self, max_dist: f32) -> Self {
        let d = target - self;
        let len = d.length();
        if len <= max_dist || len < EPSILON { target } else { self + d / len * max_dist }
    }

    #[inline]
    pub fn clamp_length(self, min: f32, max: f32) -> Self {
        let len = self.length();
        if len < EPSILON { return Self::ZERO; }
        let clamped = len.clamp(min, max);
        if (clamped - len).abs() < EPSILON { self } else { self * (clamped / len) }
    }

    #[inline]
    pub fn clamp_length_max(self, max: f32) -> Self {
        let len = self.length();
        if len > max && len > EPSILON { self * (max / len) } else { self }
    }

    #[inline]
    pub fn clamp_length_min(self, min: f32) -> Self {
        let len = self.length();
        if len < min && len > EPSILON { self * (min / len) } else { self }
    }

    #[inline] pub fn midpoint(self, rhs: Self) -> Self { (self + rhs) * 0.5 }

    #[inline]
    pub fn is_parallel(self, rhs: Self) -> bool {
        self.cross(rhs).length_sq() < EPSILON * EPSILON
    }

    #[inline]
    pub fn is_perpendicular(self, rhs: Self) -> bool { self.dot(rhs).abs() < EPSILON }

    // ── p6: spherical coordinates ─────────────────────────────────────────────

    #[inline]
    pub fn to_spherical(self) -> (f32, f32, f32) {
        let r = self.length();
        if r < EPSILON { return (0.0, 0.0, 0.0); }
        let theta = (self.z / r).clamp(-1.0, 1.0).acos();
        let phi   = self.y.atan2(self.x);
        (r, theta, phi)
    }

    #[inline]
    pub fn from_spherical(r: f32, theta: f32, phi: f32) -> Self {
        let sin_theta = theta.sin();
        Self::new(r * sin_theta * phi.cos(), r * sin_theta * phi.sin(), r * theta.cos())
    }
}

impl Add for Vec3 {
    type Output = Self;
    #[inline(always)] fn add(self, r: Self) -> Self { Self(f32x4_add(self.0, r.0)) }
}
impl Sub for Vec3 {
    type Output = Self;
    #[inline(always)] fn sub(self, r: Self) -> Self { Self(f32x4_sub(self.0, r.0)) }
}
impl Mul<f32> for Vec3 {
    type Output = Self;
    #[inline(always)] fn mul(self, s: f32) -> Self { Self(f32x4_mul(self.0, f32x4_splat(s))) }
}
impl Mul<Vec3> for f32 {
    type Output = Vec3;
    #[inline(always)] fn mul(self, v: Vec3) -> Vec3 { Vec3(f32x4_mul(f32x4_splat(self), v.0)) }
}
impl Mul for Vec3 {
    type Output = Self;
    #[inline(always)] fn mul(self, r: Self) -> Self { Self(f32x4_mul(self.0, r.0)) }
}
impl Div<f32> for Vec3 {
    type Output = Self;
    #[inline(always)] fn div(self, s: f32) -> Self { Self(f32x4_div(self.0, f32x4_splat(s))) }
}
impl Neg for Vec3 {
    type Output = Self;
    #[inline(always)] fn neg(self) -> Self { Self(f32x4_neg(self.0)) }
}
impl AddAssign for Vec3 {
    #[inline(always)] fn add_assign(&mut self, r: Self) { self.0 = f32x4_add(self.0, r.0); }
}
impl SubAssign for Vec3 {
    #[inline(always)] fn sub_assign(&mut self, r: Self) { self.0 = f32x4_sub(self.0, r.0); }
}
impl MulAssign<f32> for Vec3 {
    #[inline(always)] fn mul_assign(&mut self, s: f32) { self.0 = f32x4_mul(self.0, f32x4_splat(s)); }
}
impl DivAssign<f32> for Vec3 {
    #[inline(always)] fn div_assign(&mut self, s: f32) { self.0 = f32x4_div(self.0, f32x4_splat(s)); }
}

impl PartialEq for Vec3 {
    #[inline]
    fn eq(&self, rhs: &Self) -> bool {
        (u32x4_bitmask(f32x4_eq(self.0, rhs.0)) & 0b0111) == 0b0111
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

