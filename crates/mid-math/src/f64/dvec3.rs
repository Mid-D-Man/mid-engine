// crates/mid-math/src/f64/dvec3.rs
//! Double-precision 3D vector. 32 bytes, align(32). Always scalar.
//!
//! Layout: x, y, z, _pad — four f64 fields, 32 bytes total.
//! The padding lane reserves space for future AVX2 packed-double ops
//! (four f64 → one ymm register), exactly mirroring what the f32 Vec3
//! does with its 4th SSE2 lane.
//!
//! DEPSILON for f64 = 1e-12 (vs 1e-6 for f32).

use core::fmt;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use super::dvec2::DEPSILON;

/// 3D double-precision vector. 32 bytes, align(32). Always scalar.
///
/// **C interop:** use [`CDVec3`][crate::ffi::types::CDVec3] at the FFI boundary.
#[derive(Clone, Copy)]
#[repr(C, align(32))]
pub struct DVec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub _pad: f64,
}

impl DVec3 {
    pub const ZERO:  Self = Self { x:  0.0, y:  0.0, z:  0.0, _pad: 0.0 };
    pub const ONE:   Self = Self { x:  1.0, y:  1.0, z:  1.0, _pad: 0.0 };
    pub const X:     Self = Self { x:  1.0, y:  0.0, z:  0.0, _pad: 0.0 };
    pub const Y:     Self = Self { x:  0.0, y:  1.0, z:  0.0, _pad: 0.0 };
    pub const Z:     Self = Self { x:  0.0, y:  0.0, z:  1.0, _pad: 0.0 };
    pub const NEG_X: Self = Self { x: -1.0, y:  0.0, z:  0.0, _pad: 0.0 };
    pub const NEG_Y: Self = Self { x:  0.0, y: -1.0, z:  0.0, _pad: 0.0 };
    pub const NEG_Z: Self = Self { x:  0.0, y:  0.0, z: -1.0, _pad: 0.0 };

    // ── Constructors ──────────────────────────────────────────────────────────

    #[inline(always)]
    pub const fn new(x: f64, y: f64, z: f64) -> Self { Self { x, y, z, _pad: 0.0 } }

    #[inline(always)]
    pub fn splat(v: f64) -> Self { Self::new(v, v, v) }

    #[inline(always)]
    pub fn from_array(a: [f64; 3]) -> Self { Self::new(a[0], a[1], a[2]) }

    #[inline(always)]
    pub fn to_array(self) -> [f64; 3] { [self.x, self.y, self.z] }

    /// Extend to DVec4 by appending `w`.
    #[inline(always)]
    pub fn extend(self, w: f64) -> super::dvec4::DVec4 {
        super::dvec4::DVec4::new(self.x, self.y, self.z, w)
    }

    /// Truncate to DVec2 (drop z).
    #[inline(always)]
    pub fn truncate(self) -> super::dvec2::DVec2 {
        super::dvec2::DVec2::new(self.x, self.y)
    }

    // ── Core ops ──────────────────────────────────────────────────────────────

    #[inline(always)]
    pub fn dot(self, rhs: Self) -> f64 {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
    }

    #[inline(always)]
    pub fn cross(self, rhs: Self) -> Self {
        Self::new(
            self.y * rhs.z - self.z * rhs.y,
            self.z * rhs.x - self.x * rhs.z,
            self.x * rhs.y - self.y * rhs.x,
        )
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

    #[inline(always)]
    pub fn reflect(self, n: Self) -> Self { self - n * (2.0 * self.dot(n)) }

    #[inline(always)]
    pub fn distance(self, rhs: Self) -> f64 { (self - rhs).length() }

    #[inline(always)]
    pub fn distance_sq(self, rhs: Self) -> f64 { (self - rhs).length_sq() }

    /// Angle between `self` and `rhs` in radians `[0, π]`.
    ///
    /// Both must be non-zero (need not be normalised).
    #[inline(always)]
    pub fn angle_between(self, rhs: Self) -> f64 {
        let denom = (self.length_sq() * rhs.length_sq()).sqrt();
        if denom < DEPSILON { 0.0 } else { (self.dot(rhs) / denom).clamp(-1.0, 1.0).acos() }
    }

    // ── Component-wise ────────────────────────────────────────────────────────

    #[inline(always)]
    pub fn abs(self) -> Self { Self::new(self.x.abs(), self.y.abs(), self.z.abs()) }

    #[inline(always)]
    pub fn min(self, rhs: Self) -> Self {
        Self::new(self.x.min(rhs.x), self.y.min(rhs.y), self.z.min(rhs.z))
    }

    #[inline(always)]
    pub fn max(self, rhs: Self) -> Self {
        Self::new(self.x.max(rhs.x), self.y.max(rhs.y), self.z.max(rhs.z))
    }

    #[inline(always)]
    pub fn clamp(self, lo: Self, hi: Self) -> Self { self.max(lo).min(hi) }

    #[inline(always)]
    pub fn floor(self) -> Self { Self::new(self.x.floor(), self.y.floor(), self.z.floor()) }

    #[inline(always)]
    pub fn ceil(self) -> Self { Self::new(self.x.ceil(), self.y.ceil(), self.z.ceil()) }

    #[inline(always)]
    pub fn round(self) -> Self { Self::new(self.x.round(), self.y.round(), self.z.round()) }

    // ── Predicates ────────────────────────────────────────────────────────────

    #[inline(always)]
    pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite()
    }

    #[inline(always)]
    pub fn is_nan(self) -> bool {
        self.x.is_nan() || self.y.is_nan() || self.z.is_nan()
    }

    #[inline(always)]
    pub fn approx_eq(self, rhs: Self) -> bool {
        (self.x - rhs.x).abs() < DEPSILON
            && (self.y - rhs.y).abs() < DEPSILON
            && (self.z - rhs.z).abs() < DEPSILON
    }

    // ── Casting ───────────────────────────────────────────────────────────────

    /// Lossy cast to single-precision `Vec3`.
    #[inline(always)]
    pub fn as_vec3(self) -> crate::Vec3 {
        crate::Vec3::new(self.x as f32, self.y as f32, self.z as f32)
    }

    /// Lossy cast to single-precision `Vec3` (alias for clarity).
    #[inline(always)]
    pub fn as_vec3a(self) -> crate::Vec3 { self.as_vec3() }
}

// ── Operators ─────────────────────────────────────────────────────────────────

impl Add  for DVec3 { type Output=Self; #[inline(always)] fn add(self,r:Self)->Self{Self::new(self.x+r.x,self.y+r.y,self.z+r.z)} }
impl Sub  for DVec3 { type Output=Self; #[inline(always)] fn sub(self,r:Self)->Self{Self::new(self.x-r.x,self.y-r.y,self.z-r.z)} }
impl Neg  for DVec3 { type Output=Self; #[inline(always)] fn neg(self)->Self{Self::new(-self.x,-self.y,-self.z)} }
impl Mul<f64> for DVec3 { type Output=Self; #[inline(always)] fn mul(self,s:f64)->Self{Self::new(self.x*s,self.y*s,self.z*s)} }
impl Mul<DVec3> for f64  { type Output=DVec3; #[inline(always)] fn mul(self,v:DVec3)->DVec3{DVec3::new(self*v.x,self*v.y,self*v.z)} }
impl Mul  for DVec3 { type Output=Self; #[inline(always)] fn mul(self,r:Self)->Self{Self::new(self.x*r.x,self.y*r.y,self.z*r.z)} }
impl Div<f64> for DVec3 { type Output=Self; #[inline(always)] fn div(self,s:f64)->Self{Self::new(self.x/s,self.y/s,self.z/s)} }
impl Div  for DVec3 { type Output=Self; #[inline(always)] fn div(self,r:Self)->Self{Self::new(self.x/r.x,self.y/r.y,self.z/r.z)} }

impl AddAssign for DVec3 { #[inline(always)] fn add_assign(&mut self,r:Self){self.x+=r.x;self.y+=r.y;self.z+=r.z;} }
impl SubAssign for DVec3 { #[inline(always)] fn sub_assign(&mut self,r:Self){self.x-=r.x;self.y-=r.y;self.z-=r.z;} }
impl MulAssign<f64> for DVec3 { #[inline(always)] fn mul_assign(&mut self,s:f64){self.x*=s;self.y*=s;self.z*=s;} }
impl DivAssign<f64> for DVec3 { #[inline(always)] fn div_assign(&mut self,s:f64){self.x/=s;self.y/=s;self.z/=s;} }

impl PartialEq for DVec3 {
    fn eq(&self, rhs: &Self) -> bool {
        self.x == rhs.x && self.y == rhs.y && self.z == rhs.z
        // _pad intentionally ignored
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

impl From<[f64; 3]> for DVec3 { fn from(a:[f64;3])->Self{Self::new(a[0],a[1],a[2])} }
impl From<DVec3> for [f64; 3] { fn from(v:DVec3)->[f64;3]{[v.x,v.y,v.z]} }
impl From<(f64, f64, f64)> for DVec3 { fn from(t:(f64,f64,f64))->Self{Self::new(t.0,t.1,t.2)} }
impl From<DVec3> for (f64, f64, f64) { fn from(v:DVec3)->(f64,f64,f64){(v.x,v.y,v.z)} }
