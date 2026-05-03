// crates/mid-math/src/f64/dvec4.rs
//! Double-precision 4D vector. 32 bytes, align(32). Always scalar.
//!
//! Used as homogeneous coordinate carrier, quaternion storage base,
//! and general 4-component math at f64 precision.

use core::fmt;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use super::dvec2::DEPSILON;

/// 4D double-precision vector. 32 bytes, align(32). Always scalar.
///
/// **C interop:** use [`CDVec4`][crate::ffi::types::CDVec4] at the FFI boundary.
#[derive(Clone, Copy, PartialEq)]
#[repr(C, align(32))]
pub struct DVec4 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub w: f64,
}

impl DVec4 {
    pub const ZERO: Self = Self { x: 0.0, y: 0.0, z: 0.0, w: 0.0 };
    pub const ONE:  Self = Self { x: 1.0, y: 1.0, z: 1.0, w: 1.0 };
    pub const X:    Self = Self { x: 1.0, y: 0.0, z: 0.0, w: 0.0 };
    pub const Y:    Self = Self { x: 0.0, y: 1.0, z: 0.0, w: 0.0 };
    pub const Z:    Self = Self { x: 0.0, y: 0.0, z: 1.0, w: 0.0 };
    pub const W:    Self = Self { x: 0.0, y: 0.0, z: 0.0, w: 1.0 };

    // ── Constructors ──────────────────────────────────────────────────────────

    #[inline(always)]
    pub const fn new(x: f64, y: f64, z: f64, w: f64) -> Self { Self { x, y, z, w } }

    #[inline(always)]
    pub fn splat(v: f64) -> Self { Self::new(v, v, v, v) }

    #[inline(always)]
    pub fn from_array(a: [f64; 4]) -> Self { Self::new(a[0], a[1], a[2], a[3]) }

    #[inline(always)]
    pub fn to_array(self) -> [f64; 4] { [self.x, self.y, self.z, self.w] }

    /// Truncate to DVec3, discarding w.
    #[inline(always)]
    pub fn truncate(self) -> super::dvec3::DVec3 {
        super::dvec3::DVec3::new(self.x, self.y, self.z)
    }

    // ── Core ops ──────────────────────────────────────────────────────────────

    #[inline(always)]
    pub fn dot(self, rhs: Self) -> f64 {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z + self.w * rhs.w
    }

    #[inline(always)]
    pub fn length_sq(self) -> f64 { self.dot(self) }

    #[inline(always)]
    pub fn length(self) -> f64 { self.length_sq().sqrt() }

    #[inline(always)]
    pub fn length_recip(self) -> f64 {
        let l = self.length();
        if l < DEPSILON { 0.0 } else { 1.0 / l }
    }

    #[inline(always)]
    pub fn normalize(self) -> Self {
        let l = self.length();
        if l < DEPSILON { Self::ZERO } else { self * (1.0 / l) }
    }

    #[inline(always)]
    pub fn try_normalize(self) -> Option<Self> {
        let rcp = self.length_recip();
        if rcp > 0.0 && rcp.is_finite() { Some(self * rcp) } else { None }
    }

    #[inline(always)]
    pub fn normalize_or_zero(self) -> Self {
        self.try_normalize().unwrap_or(Self::ZERO)
    }

    #[inline(always)]
    pub fn is_normalized(self) -> bool {
        (self.length_sq() - 1.0).abs() <= 2e-10
    }

    #[inline(always)]
    pub fn lerp(self, rhs: Self, t: f64) -> Self { self + (rhs - self) * t }

    // ── Component-wise ────────────────────────────────────────────────────────

    #[inline(always)]
    pub fn abs(self) -> Self {
        Self::new(self.x.abs(), self.y.abs(), self.z.abs(), self.w.abs())
    }

    #[inline(always)]
    pub fn min(self, rhs: Self) -> Self {
        Self::new(self.x.min(rhs.x), self.y.min(rhs.y),
                  self.z.min(rhs.z), self.w.min(rhs.w))
    }

    #[inline(always)]
    pub fn max(self, rhs: Self) -> Self {
        Self::new(self.x.max(rhs.x), self.y.max(rhs.y),
                  self.z.max(rhs.z), self.w.max(rhs.w))
    }

    #[inline(always)]
    pub fn clamp(self, lo: Self, hi: Self) -> Self { self.max(lo).min(hi) }

    // ── Predicates ────────────────────────────────────────────────────────────

    #[inline(always)]
    pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite()
            && self.z.is_finite() && self.w.is_finite()
    }

    #[inline(always)]
    pub fn is_nan(self) -> bool {
        self.x.is_nan() || self.y.is_nan() || self.z.is_nan() || self.w.is_nan()
    }

    #[inline(always)]
    pub fn approx_eq(self, rhs: Self) -> bool {
        (self.x - rhs.x).abs() < DEPSILON
            && (self.y - rhs.y).abs() < DEPSILON
            && (self.z - rhs.z).abs() < DEPSILON
            && (self.w - rhs.w).abs() < DEPSILON
    }

    // ── Casting ───────────────────────────────────────────────────────────────

    /// Lossy cast to single-precision `Vec4`.
    #[inline(always)]
    pub fn as_vec4(self) -> crate::Vec4 {
        crate::Vec4::new(self.x as f32, self.y as f32, self.z as f32, self.w as f32)
    }
}

// ── Operators ─────────────────────────────────────────────────────────────────

impl Add  for DVec4 { type Output=Self; #[inline(always)] fn add(self,r:Self)->Self{Self::new(self.x+r.x,self.y+r.y,self.z+r.z,self.w+r.w)} }
impl Sub  for DVec4 { type Output=Self; #[inline(always)] fn sub(self,r:Self)->Self{Self::new(self.x-r.x,self.y-r.y,self.z-r.z,self.w-r.w)} }
impl Neg  for DVec4 { type Output=Self; #[inline(always)] fn neg(self)->Self{Self::new(-self.x,-self.y,-self.z,-self.w)} }
impl Mul<f64> for DVec4 { type Output=Self; #[inline(always)] fn mul(self,s:f64)->Self{Self::new(self.x*s,self.y*s,self.z*s,self.w*s)} }
impl Mul<DVec4> for f64  { type Output=DVec4; #[inline(always)] fn mul(self,v:DVec4)->DVec4{DVec4::new(self*v.x,self*v.y,self*v.z,self*v.w)} }
impl Mul  for DVec4 { type Output=Self; #[inline(always)] fn mul(self,r:Self)->Self{Self::new(self.x*r.x,self.y*r.y,self.z*r.z,self.w*r.w)} }
impl Div<f64> for DVec4 { type Output=Self; #[inline(always)] fn div(self,s:f64)->Self{Self::new(self.x/s,self.y/s,self.z/s,self.w/s)} }

impl AddAssign for DVec4 { #[inline(always)] fn add_assign(&mut self,r:Self){self.x+=r.x;self.y+=r.y;self.z+=r.z;self.w+=r.w;} }
impl SubAssign for DVec4 { #[inline(always)] fn sub_assign(&mut self,r:Self){self.x-=r.x;self.y-=r.y;self.z-=r.z;self.w-=r.w;} }
impl MulAssign<f64> for DVec4 { #[inline(always)] fn mul_assign(&mut self,s:f64){self.x*=s;self.y*=s;self.z*=s;self.w*=s;} }
impl DivAssign<f64> for DVec4 { #[inline(always)] fn div_assign(&mut self,s:f64){self.x/=s;self.y/=s;self.z/=s;self.w/=s;} }

impl Default for DVec4 { fn default() -> Self { Self::ZERO } }

impl fmt::Debug for DVec4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("DVec4")
            .field(&self.x).field(&self.y).field(&self.z).field(&self.w)
            .finish()
    }
}

impl fmt::Display for DVec4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {}, {}, {})", self.x, self.y, self.z, self.w)
    }
}

impl From<[f64; 4]> for DVec4 { fn from(a:[f64;4])->Self{Self::new(a[0],a[1],a[2],a[3])} }
impl From<DVec4> for [f64; 4] { fn from(v:DVec4)->[f64;4]{[v.x,v.y,v.z,v.w]} }
impl From<(f64, f64, f64, f64)> for DVec4 { fn from(t:(f64,f64,f64,f64))->Self{Self::new(t.0,t.1,t.2,t.3)} }
impl From<DVec4> for (f64, f64, f64, f64) { fn from(v:DVec4)->(f64,f64,f64,f64){(v.x,v.y,v.z,v.w)} }
