// crates/mid-math/src/f32/coresimd/vec4.rs
//! Vec4 backed by `f32x4` (Rust portable SIMD).

use core::fmt;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use core::simd::prelude::*;
use core::simd::{cmp::SimdPartialEq, cmp::SimdPartialOrd, num::SimdFloat};
use std::simd::StdFloat;

use super::{dot4, dot4_into_f32x4};
use crate::f32::coresimd::vec3::Vec3;
use crate::f32::vec2::Vec2;
use crate::impl_vec4_deref;
use crate::EPSILON;

// ── Type ──────────────────────────────────────────────────────────────────────

/// 4-dimensional vector. 16 bytes, 16-byte aligned. Backed by `f32x4`.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct Vec4(pub(crate) f32x4);

impl_vec4_deref!(Vec4);

// ── Constants ─────────────────────────────────────────────────────────────────

impl Vec4 {
    pub const ZERO: Self = Self(f32x4::from_array([0.0; 4]));
    pub const ONE:  Self = Self(f32x4::from_array([1.0; 4]));
    pub const X:    Self = Self(f32x4::from_array([1.0, 0.0, 0.0, 0.0]));
    pub const Y:    Self = Self(f32x4::from_array([0.0, 1.0, 0.0, 0.0]));
    pub const Z:    Self = Self(f32x4::from_array([0.0, 0.0, 1.0, 0.0]));
    pub const W:    Self = Self(f32x4::from_array([0.0, 0.0, 0.0, 1.0]));

    // ── Constructors ─────────────────────────────────────────────────────────

    #[inline(always)]
    pub fn new(x: f32, y: f32, z: f32, w: f32) -> Self {
        Self(f32x4::from_array([x, y, z, w]))
    }

    #[inline(always)] pub fn splat(v: f32) -> Self { Self(f32x4::splat(v)) }
    #[inline(always)] pub fn from_array(a: [f32; 4]) -> Self { Self(f32x4::from_array(a)) }
    #[inline(always)] pub fn to_array(self) -> [f32; 4] { self.0.to_array() }

    /// Truncate to Vec3: zero lane 3 via swizzle.
    ///
    /// `simd_swizzle!(v, zero, [0,1,2,4])` → [v[0], v[1], v[2], 0.0]
    #[inline(always)]
    pub fn truncate(self) -> Vec3 {
        Vec3(simd_swizzle!(self.0, f32x4::splat(0.0), [0, 1, 2, 4]))
    }

    // ── Arithmetic ────────────────────────────────────────────────────────────

    #[inline(always)] pub fn dot(self, rhs: Self) -> f32 { dot4(self.0, rhs.0) }
    #[inline(always)] pub fn dot_into_vec(self, rhs: Self) -> Self { Self(dot4_into_f32x4(self.0, rhs.0)) }
    #[inline(always)] pub fn length_sq(self) -> f32 { self.dot(self) }

    #[inline]
    pub fn length(self) -> f32 {
        dot4_into_f32x4(self.0, self.0).sqrt()[0]
    }

    #[inline]
    pub fn length_recip(self) -> f32 {
        let l = self.length();
        if l < EPSILON { 0.0 } else { 1.0 / l }
    }

    #[inline]
    pub fn normalize(self) -> Self {
        let len  = dot4_into_f32x4(self.0, self.0).sqrt();
        let norm = self.0 / len;
        let mask = len.simd_gt(f32x4::splat(EPSILON));
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

    // ── Interpolation ─────────────────────────────────────────────────────────

    #[inline]
    pub fn lerp(self, rhs: Self, t: f32) -> Self {
        Self(self.0 + (rhs.0 - self.0) * f32x4::splat(t))
    }

    // ── Component-wise ────────────────────────────────────────────────────────

    #[inline] pub fn min(self, r: Self) -> Self { Self(self.0.simd_lt(r.0).select(self.0, r.0)) }
    #[inline] pub fn max(self, r: Self) -> Self { Self(self.0.simd_gt(r.0).select(self.0, r.0)) }
    #[inline] pub fn clamp(self, lo: Self, hi: Self) -> Self { self.max(lo).min(hi) }
    #[inline] pub fn abs(self) -> Self { Self(self.0.abs()) }
    #[inline] pub fn floor(self) -> Self { Self(self.0.floor()) }
    #[inline] pub fn ceil(self)  -> Self { Self(self.0.ceil()) }
    #[inline] pub fn round(self) -> Self { Self(self.0.round()) }

    // ── Predicates ────────────────────────────────────────────────────────────

    #[inline] pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite()
            && self.z.is_finite() && self.w.is_finite()
    }
    #[inline] pub fn is_nan(self) -> bool {
        self.x.is_nan() || self.y.is_nan() || self.z.is_nan() || self.w.is_nan()
    }
    #[inline]
    pub fn approx_eq(self, rhs: Self) -> bool {
        let diff = (self.0 - rhs.0).abs();
        (diff.simd_lt(f32x4::splat(EPSILON)).to_bitmask() & 0b1111) == 0b1111
    }
}

// ── Operators ─────────────────────────────────────────────────────────────────

impl Add for Vec4 { type Output=Self; #[inline(always)] fn add(self,r:Self)->Self{Self(self.0+r.0)} }
impl Sub for Vec4 { type Output=Self; #[inline(always)] fn sub(self,r:Self)->Self{Self(self.0-r.0)} }
impl Mul for Vec4 { type Output=Self; #[inline(always)] fn mul(self,r:Self)->Self{Self(self.0*r.0)} }
impl Div for Vec4 { type Output=Self; #[inline(always)] fn div(self,r:Self)->Self{Self(self.0/r.0)} }
impl Mul<f32> for Vec4 { type Output=Self; #[inline(always)] fn mul(self,s:f32)->Self{Self(self.0*f32x4::splat(s))} }
impl Mul<Vec4> for f32 { type Output=Vec4; #[inline(always)] fn mul(self,v:Vec4)->Vec4{Vec4(f32x4::splat(self)*v.0)} }
impl Div<f32> for Vec4 { type Output=Self; #[inline(always)] fn div(self,s:f32)->Self{Self(self.0/f32x4::splat(s))} }
impl Neg for Vec4  { type Output=Self; #[inline(always)] fn neg(self)->Self{Self(-self.0)} }

impl AddAssign for Vec4 { #[inline(always)] fn add_assign(&mut self,r:Self){self.0+=r.0;} }
impl SubAssign for Vec4 { #[inline(always)] fn sub_assign(&mut self,r:Self){self.0-=r.0;} }
impl MulAssign<f32> for Vec4 { #[inline(always)] fn mul_assign(&mut self,s:f32){self.0*=f32x4::splat(s);} }
impl DivAssign<f32> for Vec4 { #[inline(always)] fn div_assign(&mut self,s:f32){self.0/=f32x4::splat(s);} }

impl PartialEq for Vec4 {
    #[inline]
    fn eq(&self, rhs: &Self) -> bool {
        (self.0.simd_eq(rhs.0).to_bitmask() & 0b1111) == 0b1111
    }
}

impl Default for Vec4 { fn default() -> Self { Self::ZERO } }

impl fmt::Debug for Vec4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Vec4")
            .field(&self.x).field(&self.y).field(&self.z).field(&self.w).finish()
    }
}
impl fmt::Display for Vec4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {}, {}, {})", self.x, self.y, self.z, self.w)
    }
}

impl From<[f32; 4]> for Vec4 { #[inline] fn from(a: [f32; 4]) -> Self { Self(f32x4::from_array(a)) } }
impl From<Vec4> for [f32; 4] { #[inline] fn from(v: Vec4) -> Self { v.0.to_array() } }
impl From<(f32,f32,f32,f32)> for Vec4 { #[inline] fn from(t:(f32,f32,f32,f32))->Self{Self::new(t.0,t.1,t.2,t.3)} }
impl From<Vec4> for (f32,f32,f32,f32) { #[inline] fn from(v:Vec4)->(f32,f32,f32,f32){(v.x,v.y,v.z,v.w)} }

