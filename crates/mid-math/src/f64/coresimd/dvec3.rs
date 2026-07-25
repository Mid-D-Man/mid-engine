// crates/mid-math/src/f64/coresimd/dvec3.rs
//! DVec3 backed by `f64x4` (Rust portable SIMD).
//!
//! Structural translation of f32/coresimd/vec3.rs: same algorithms, same
//! swizzle index patterns (cross product especially -- [2,0,1,1] is
//! width-independent, it's an index pattern not a value, so it carries over
//! unchanged). What differs: DEPSILON instead of EPSILON (1e-12 vs f32's
//! looser tolerance, matching the existing scalar DVec3's own constant, not
//! copied from the f32 side), the is_normalized threshold (2e-10, taken
//! from the existing scalar DVec3 rather than f32 coresimd's 2e-4 -- f64
//! actually earns a tighter tolerance), and extend()/truncate()/as_vec3()
//! return the crate's normal DVec2/DVec4/Vec3 (whichever backend those
//! dispatch to) rather than a coresimd-specific type, so this doesn't pull
//! in a dependency on a DVec4 coresimd module that doesn't exist yet.

use core::fmt;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use core::simd::prelude::*;
use core::simd::{cmp::SimdPartialEq, cmp::SimdPartialOrd, num::SimdFloat};
use std::simd::StdFloat;

use super::{dot3, dot3_into_f64x4};
use crate::impl_dvec3_deref;
use crate::f64::dvec2::DEPSILON;

/// 3-dimensional double-precision vector. 32 bytes, backed by `f64x4`.
///
/// Unlike every other f64 type in this crate (see f64/mod.rs's
/// "always-scalar" list), this one gets a real SIMD shape -- see the note
/// in coresimd/mod.rs for why DVec3 specifically couldn't get that from any
/// hardware-specific backend, and can from this one.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct DVec3(pub(crate) f64x4);

impl_dvec3_deref!(DVec3);

impl DVec3 {
    pub const ZERO:  Self = Self(f64x4::from_array([ 0.0,  0.0,  0.0, 0.0]));
    pub const ONE:   Self = Self(f64x4::from_array([ 1.0,  1.0,  1.0, 0.0]));
    pub const X:     Self = Self(f64x4::from_array([ 1.0,  0.0,  0.0, 0.0]));
    pub const Y:     Self = Self(f64x4::from_array([ 0.0,  1.0,  0.0, 0.0]));
    pub const Z:     Self = Self(f64x4::from_array([ 0.0,  0.0,  1.0, 0.0]));
    pub const NEG_X: Self = Self(f64x4::from_array([-1.0,  0.0,  0.0, 0.0]));
    pub const NEG_Y: Self = Self(f64x4::from_array([ 0.0, -1.0,  0.0, 0.0]));
    pub const NEG_Z: Self = Self(f64x4::from_array([ 0.0,  0.0, -1.0, 0.0]));
    pub const NAN:   Self = Self(f64x4::from_array([f64::NAN, f64::NAN, f64::NAN, f64::NAN]));

    #[inline(always)]
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self(f64x4::from_array([x, y, z, 0.0]))
    }

    #[inline(always)] pub fn splat(v: f64) -> Self { Self::new(v, v, v) }
    #[inline(always)] pub fn from_array(a: [f64; 3]) -> Self { Self::new(a[0], a[1], a[2]) }
    #[inline(always)] pub fn to_array(self) -> [f64; 3] { [self.x, self.y, self.z] }

    #[inline(always)]
    pub fn extend(self, w: f64) -> crate::DVec4 {
        crate::DVec4::new(self.x, self.y, self.z, w)
    }

    #[inline(always)]
    pub fn truncate(self) -> crate::DVec2 {
        crate::DVec2::new(self.x, self.y)
    }

    #[inline(always)]
    pub fn dot(self, rhs: Self) -> f64 { dot3(self.0, rhs.0) }

    #[inline]
    pub fn dot_into_vec(self, rhs: Self) -> Self { Self(dot3_into_f64x4(self.0, rhs.0)) }

    #[inline(always)]
    pub fn cross(self, rhs: Self) -> Self {
        // Same [2,0,1,1] shuffle-multiply-subtract-shuffle pattern as
        // f32/coresimd/vec3.rs's cross() -- an index pattern is width-
        // independent, carries over unchanged from f32x4 to f64x4.
        let lhszxy     = simd_swizzle!(self.0, [2, 0, 1, 1]);
        let rhszxy     = simd_swizzle!(rhs.0,  [2, 0, 1, 1]);
        let lhszxy_rhs = lhszxy * rhs.0;
        let rhszxy_lhs = rhszxy * self.0;
        let sub        = lhszxy_rhs - rhszxy_lhs;
        Self(simd_swizzle!(sub, [2, 0, 1, 1]))
    }

    #[inline(always)] pub fn length_sq(self) -> f64 { self.dot(self) }

    #[inline]
    pub fn length(self) -> f64 {
        dot3_into_f64x4(self.0, self.0).sqrt()[0]
    }

    #[inline]
    pub fn length_recip(self) -> f64 {
        let l = self.length();
        if l < DEPSILON { 0.0 } else { 1.0 / l }
    }

    #[inline]
    pub fn normalize(self) -> Self {
        let len  = dot3_into_f64x4(self.0, self.0).sqrt();
        let norm = self.0 / len;
        let mask = len.simd_gt(f64x4::splat(DEPSILON));
        Self(mask.select(norm, f64x4::splat(0.0)))
    }

    #[inline]
    pub fn try_normalize(self) -> Option<Self> {
        let rcp = self.length_recip();
        if rcp.is_finite() && rcp > 0.0 { Some(self * rcp) } else { None }
    }

    #[inline] pub fn normalize_or(self, fb: Self) -> Self { self.try_normalize().unwrap_or(fb) }
    #[inline] pub fn normalize_or_zero(self) -> Self { self.normalize_or(Self::ZERO) }
    // 2e-10, matching the existing scalar DVec3's own tolerance -- not
    // f32 coresimd's 2e-4, f64 earns tighter here.
    #[inline] pub fn is_normalized(self) -> bool { (self.length_sq() - 1.0).abs() <= 2e-10 }

    #[inline]
    pub fn lerp(self, rhs: Self, t: f64) -> Self {
        let tt = f64x4::splat(t);
        Self(self.0 + (rhs.0 - self.0) * tt)
    }

    #[inline] pub fn reflect(self, n: Self) -> Self { self - n * (2.0 * self.dot(n)) }
    #[inline] pub fn distance(self, rhs: Self)    -> f64 { (self - rhs).length() }
    #[inline] pub fn distance_sq(self, rhs: Self) -> f64 { (self - rhs).length_sq() }

    #[inline] pub fn min(self, rhs: Self) -> Self { Self(self.0.simd_lt(rhs.0).select(self.0, rhs.0)) }
    #[inline] pub fn max(self, rhs: Self) -> Self { Self(self.0.simd_gt(rhs.0).select(self.0, rhs.0)) }
    #[inline] pub fn clamp(self, lo: Self, hi: Self) -> Self { self.max(lo).min(hi) }
    #[inline] pub fn abs(self) -> Self { Self(self.0.abs()) }
    #[inline] pub fn floor(self) -> Self { Self(self.0.floor()) }
    #[inline] pub fn ceil(self)  -> Self { Self(self.0.ceil()) }
    #[inline] pub fn round(self) -> Self { Self(self.0.round()) }

    #[inline] pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite()
    }
    #[inline] pub fn is_nan(self) -> bool {
        self.x.is_nan() || self.y.is_nan() || self.z.is_nan()
    }
    #[inline] pub fn approx_eq(self, rhs: Self) -> bool {
        (self.x - rhs.x).abs() < DEPSILON
            && (self.y - rhs.y).abs() < DEPSILON
            && (self.z - rhs.z).abs() < DEPSILON
    }

    #[inline]
    pub fn angle_between(self, rhs: Self) -> f64 {
        let denom = (self.length_sq() * rhs.length_sq()).sqrt();
        if denom < DEPSILON { 0.0 } else { (self.dot(rhs) / denom).clamp(-1.0, 1.0).acos() }
    }

    #[inline]
    pub fn project_onto(self, rhs: Self) -> Self {
        let d = rhs.length_sq();
        if d < DEPSILON { Self::ZERO } else { rhs * (self.dot(rhs) / d) }
    }

    #[inline]
    pub fn reject_from(self, rhs: Self) -> Self { self - self.project_onto(rhs) }

    #[inline]
    pub fn move_towards(self, target: Self, max_dist: f64) -> Self {
        let d = target - self;
        let len = d.length();
        if len <= max_dist || len < DEPSILON { target } else { self + d / len * max_dist }
    }

    #[inline]
    pub fn clamp_length(self, min: f64, max: f64) -> Self {
        let len = self.length();
        if len < DEPSILON { return Self::ZERO; }
        let clamped = len.clamp(min, max);
        if (clamped - len).abs() < DEPSILON { self } else { self * (clamped / len) }
    }

    #[inline]
    pub fn clamp_length_max(self, max: f64) -> Self {
        let len = self.length();
        if len > max && len > DEPSILON { self * (max / len) } else { self }
    }

    #[inline]
    pub fn clamp_length_min(self, min: f64) -> Self {
        let len = self.length();
        if len < min && len > DEPSILON { self * (min / len) } else { self }
    }

    #[inline] pub fn midpoint(self, rhs: Self) -> Self { (self + rhs) * 0.5 }

    #[inline]
    pub fn is_parallel(self, rhs: Self) -> bool {
        self.cross(rhs).length_sq() < DEPSILON * DEPSILON
    }

    #[inline]
    pub fn is_perpendicular(self, rhs: Self) -> bool { self.dot(rhs).abs() < DEPSILON }

    #[inline]
    pub fn to_spherical(self) -> (f64, f64, f64) {
        let r = self.length();
        if r < DEPSILON { return (0.0, 0.0, 0.0); }
        let theta = (self.z / r).clamp(-1.0, 1.0).acos();
        let phi   = self.y.atan2(self.x);
        (r, theta, phi)
    }

    #[inline]
    pub fn from_spherical(r: f64, theta: f64, phi: f64) -> Self {
        let sin_theta = theta.sin();
        Self::new(r * sin_theta * phi.cos(), r * sin_theta * phi.sin(), r * theta.cos())
    }

    #[inline(always)]
    pub fn as_vec3(self) -> crate::Vec3 {
        crate::Vec3::new(self.x as f32, self.y as f32, self.z as f32)
    }
    #[inline(always)] pub fn as_vec3a(self) -> crate::Vec3 { self.as_vec3() }
}

impl Add for DVec3 {
    type Output = Self;
    #[inline(always)] fn add(self, r: Self) -> Self { Self(self.0 + r.0) }
}
impl Sub for DVec3 {
    type Output = Self;
    #[inline(always)] fn sub(self, r: Self) -> Self { Self(self.0 - r.0) }
}
impl Mul<f64> for DVec3 {
    type Output = Self;
    #[inline(always)] fn mul(self, s: f64) -> Self { Self(self.0 * f64x4::splat(s)) }
}
impl Mul<DVec3> for f64 {
    type Output = DVec3;
    #[inline(always)] fn mul(self, v: DVec3) -> DVec3 { DVec3(f64x4::splat(self) * v.0) }
}
impl Mul for DVec3 {
    type Output = Self;
    #[inline(always)] fn mul(self, r: Self) -> Self { Self(self.0 * r.0) }
}
impl Div<f64> for DVec3 {
    type Output = Self;
    #[inline(always)] fn div(self, s: f64) -> Self { Self(self.0 / f64x4::splat(s)) }
}
impl Div for DVec3 {
    type Output = Self;
    #[inline(always)] fn div(self, r: Self) -> Self { Self(self.0 / r.0) }
}
impl Neg for DVec3 {
    type Output = Self;
    #[inline(always)] fn neg(self) -> Self { Self(-self.0) }
}
impl AddAssign for DVec3 { #[inline(always)] fn add_assign(&mut self, r: Self) { self.0 += r.0; } }
impl SubAssign for DVec3 { #[inline(always)] fn sub_assign(&mut self, r: Self) { self.0 -= r.0; } }
impl MulAssign<f64> for DVec3 { #[inline(always)] fn mul_assign(&mut self, s: f64) { self.0 *= f64x4::splat(s); } }
impl DivAssign<f64> for DVec3 { #[inline(always)] fn div_assign(&mut self, s: f64) { self.0 /= f64x4::splat(s); } }

impl PartialEq for DVec3 {
    #[inline]
    fn eq(&self, rhs: &Self) -> bool {
        (self.0.simd_eq(rhs.0).to_bitmask() & 0b0111) == 0b0111
    }
}

impl Default for DVec3 { fn default() -> Self { Self::ZERO } }

impl fmt::Debug for DVec3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("DVec3").field(&self.x).field(&self.y).field(&self.z).finish()
    }
}
impl fmt::Display for DVec3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {}, {})", self.x, self.y, self.z)
    }
}
impl From<[f64; 3]> for DVec3 { #[inline] fn from(a: [f64; 3]) -> Self { Self::new(a[0], a[1], a[2]) } }
impl From<DVec3> for [f64; 3] { #[inline] fn from(v: DVec3) -> Self { [v.x, v.y, v.z] } }
impl From<(f64, f64, f64)> for DVec3 { #[inline] fn from(t: (f64, f64, f64)) -> Self { Self::new(t.0, t.1, t.2) } }
impl From<DVec3> for (f64, f64, f64) { #[inline] fn from(v: DVec3) -> Self { (v.x, v.y, v.z) } }
