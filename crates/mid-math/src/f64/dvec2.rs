// crates/mid-math/src/f64/dvec2.rs
//! Double-precision 2D vector. 16 bytes, align(16). Always scalar.

use core::fmt;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

pub(crate) const DEPSILON: f64 = 1e-12;

/// 2D double-precision vector. 16 bytes, align(16). Always scalar.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C, align(16))]
pub struct DVec2 {
    pub x: f64,
    pub y: f64,
}

impl DVec2 {
    pub const ZERO:  Self = Self { x:  0.0, y:  0.0 };
    pub const ONE:   Self = Self { x:  1.0, y:  1.0 };
    pub const X:     Self = Self { x:  1.0, y:  0.0 };
    pub const Y:     Self = Self { x:  0.0, y:  1.0 };
    pub const NEG_X: Self = Self { x: -1.0, y:  0.0 };
    pub const NEG_Y: Self = Self { x:  0.0, y: -1.0 };

    #[inline(always)] pub const fn new(x: f64, y: f64) -> Self { Self { x, y } }
    #[inline(always)] pub const fn splat(v: f64) -> Self { Self { x: v, y: v } }
    #[inline(always)] pub fn from_array(a: [f64; 2]) -> Self { Self::new(a[0], a[1]) }
    #[inline(always)] pub fn to_array(self) -> [f64; 2] { [self.x, self.y] }

    #[inline(always)]
    pub fn extend(self, z: f64) -> crate::DVec3 { crate::DVec3::new(self.x, self.y, z) }

    #[inline(always)] pub fn dot(self, rhs: Self) -> f64 { self.x*rhs.x + self.y*rhs.y }
    #[inline(always)] pub fn length_sq(self) -> f64 { self.dot(self) }
    #[inline(always)] pub fn length(self) -> f64 { self.length_sq().sqrt() }

    #[inline(always)]
    pub fn length_recip(self) -> f64 {
        let l = self.length();
        if l < DEPSILON { 0.0 } else { 1.0 / l }
    }

    #[inline(always)]
    pub fn normalize(self) -> Self {
        let l = self.length();
        if l < DEPSILON { Self::ZERO } else { self / l }
    }

    #[inline(always)]
    pub fn try_normalize(self) -> Option<Self> {
        let rcp = self.length_recip();
        if rcp > 0.0 && rcp.is_finite() { Some(self * rcp) } else { None }
    }

    #[inline(always)] pub fn normalize_or_zero(self) -> Self { self.try_normalize().unwrap_or(Self::ZERO) }
    #[inline(always)] pub fn is_normalized(self) -> bool { (self.length_sq() - 1.0).abs() <= 2e-10 }
    #[inline(always)] pub fn lerp(self, rhs: Self, t: f64) -> Self { self + (rhs - self) * t }
    #[inline(always)] pub fn distance(self, rhs: Self) -> f64 { (self - rhs).length() }
    #[inline(always)] pub fn distance_sq(self, rhs: Self) -> f64 { (self - rhs).length_sq() }

    #[inline(always)] pub fn perp(self) -> Self { Self::new(-self.y, self.x) }
    #[inline(always)] pub fn perp_dot(self, rhs: Self) -> f64 { self.x*rhs.y - self.y*rhs.x }

    #[inline(always)]
    pub fn angle_to(self, rhs: Self) -> f64 { self.perp_dot(rhs).atan2(self.dot(rhs)) }

    #[inline(always)] pub fn to_angle(self) -> f64 { self.y.atan2(self.x) }

    #[inline(always)]
    pub fn from_angle(angle: f64) -> Self {
        let (s, c) = angle.sin_cos();
        Self::new(c, s)
    }

    // ── p6: polar coordinates ─────────────────────────────────────────────────

    /// Convert to polar `(radius, angle)`. `angle` ∈ `[-π, π]`.
    #[inline(always)]
    pub fn to_polar(self) -> (f64, f64) { (self.length(), self.y.atan2(self.x)) }

    /// Build from polar `(radius, angle)`.
    #[inline(always)]
    pub fn from_polar(radius: f64, angle: f64) -> Self {
        let (s, c) = angle.sin_cos();
        Self::new(radius * c, radius * s)
    }

    // ── p2: project / reject ──────────────────────────────────────────────────

    #[inline(always)]
    pub fn project_onto(self, rhs: Self) -> Self {
        let d = rhs.length_sq();
        if d < DEPSILON { Self::ZERO } else { rhs * (self.dot(rhs) / d) }
    }

    #[inline(always)]
    pub fn reject_from(self, rhs: Self) -> Self { self - self.project_onto(rhs) }

    // ── p7: movement helpers ──────────────────────────────────────────────────

    #[inline(always)]
    pub fn move_towards(self, target: Self, max_dist: f64) -> Self {
        let d = target - self;
        let len = d.length();
        if len <= max_dist || len < DEPSILON { target } else { self + d / len * max_dist }
    }

    #[inline(always)]
    pub fn clamp_length(self, min: f64, max: f64) -> Self {
        let len = self.length();
        if len < DEPSILON { return Self::ZERO; }
        let clamped = len.clamp(min, max);
        if (clamped - len).abs() < DEPSILON { self } else { self * (clamped / len) }
    }

    #[inline(always)]
    pub fn clamp_length_max(self, max: f64) -> Self {
        let len = self.length();
        if len > max && len > DEPSILON { self * (max / len) } else { self }
    }

    #[inline(always)]
    pub fn clamp_length_min(self, min: f64) -> Self {
        let len = self.length();
        if len < min && len > DEPSILON { self * (min / len) } else { self }
    }

    #[inline(always)] pub fn midpoint(self, rhs: Self) -> Self { (self + rhs) * 0.5 }

    // ── component-wise ────────────────────────────────────────────────────────

    #[inline(always)] pub fn abs(self) -> Self { Self::new(self.x.abs(), self.y.abs()) }
    #[inline(always)] pub fn min(self, rhs: Self) -> Self { Self::new(self.x.min(rhs.x), self.y.min(rhs.y)) }
    #[inline(always)] pub fn max(self, rhs: Self) -> Self { Self::new(self.x.max(rhs.x), self.y.max(rhs.y)) }
    #[inline(always)] pub fn clamp(self, lo: Self, hi: Self) -> Self { self.max(lo).min(hi) }
    #[inline(always)] pub fn floor(self) -> Self { Self::new(self.x.floor(), self.y.floor()) }
    #[inline(always)] pub fn ceil(self)  -> Self { Self::new(self.x.ceil(),  self.y.ceil())  }
    #[inline(always)] pub fn round(self) -> Self { Self::new(self.x.round(), self.y.round()) }

    #[inline(always)] pub fn is_finite(self) -> bool { self.x.is_finite() && self.y.is_finite() }
    #[inline(always)] pub fn is_nan(self)    -> bool { self.x.is_nan()    || self.y.is_nan()    }

    #[inline(always)]
    pub fn approx_eq(self, rhs: Self) -> bool {
        (self.x-rhs.x).abs() < DEPSILON && (self.y-rhs.y).abs() < DEPSILON
    }

    #[inline(always)]
    pub fn as_vec2(self) -> crate::Vec2 { crate::Vec2::new(self.x as f32, self.y as f32) }
}

impl Add  for DVec2 { type Output=Self; #[inline(always)] fn add(self,r:Self)->Self{Self::new(self.x+r.x,self.y+r.y)} }
impl Sub  for DVec2 { type Output=Self; #[inline(always)] fn sub(self,r:Self)->Self{Self::new(self.x-r.x,self.y-r.y)} }
impl Neg  for DVec2 { type Output=Self; #[inline(always)] fn neg(self)->Self{Self::new(-self.x,-self.y)} }
impl Mul<f64> for DVec2 { type Output=Self; #[inline(always)] fn mul(self,s:f64)->Self{Self::new(self.x*s,self.y*s)} }
impl Mul<DVec2> for f64  { type Output=DVec2; #[inline(always)] fn mul(self,v:DVec2)->DVec2{DVec2::new(self*v.x,self*v.y)} }
impl Mul for DVec2 { type Output=Self; #[inline(always)] fn mul(self,r:Self)->Self{Self::new(self.x*r.x,self.y*r.y)} }
impl Div<f64> for DVec2 { type Output=Self; #[inline(always)] fn div(self,s:f64)->Self{Self::new(self.x/s,self.y/s)} }
impl Div for DVec2 { type Output=Self; #[inline(always)] fn div(self,r:Self)->Self{Self::new(self.x/r.x,self.y/r.y)} }

impl AddAssign for DVec2 { #[inline(always)] fn add_assign(&mut self,r:Self){self.x+=r.x;self.y+=r.y;} }
impl SubAssign for DVec2 { #[inline(always)] fn sub_assign(&mut self,r:Self){self.x-=r.x;self.y-=r.y;} }
impl MulAssign<f64> for DVec2 { #[inline(always)] fn mul_assign(&mut self,s:f64){self.x*=s;self.y*=s;} }
impl DivAssign<f64> for DVec2 { #[inline(always)] fn div_assign(&mut self,s:f64){self.x/=s;self.y/=s;} }

impl Default for DVec2 { fn default() -> Self { Self::ZERO } }

impl fmt::Display for DVec2 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "({}, {})", self.x, self.y) }
}
impl From<[f64; 2]> for DVec2 { fn from(a:[f64;2])->Self{Self::new(a[0],a[1])} }
impl From<DVec2> for [f64; 2] { fn from(v:DVec2)->[f64;2]{[v.x,v.y]} }
impl From<(f64, f64)> for DVec2 { fn from(t:(f64,f64))->Self{Self::new(t.0,t.1)} }
impl From<DVec2> for (f64, f64) { fn from(v:DVec2)->(f64,f64){(v.x,v.y)} }
