// crates/mid-math/src/f32/scalar/vec3.rs
//! Scalar Vec3 — fallback for non-SIMD targets and correctness reference.

use core::fmt;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use crate::EPSILON;

/// 3D vector. 16 bytes (12 data + 4 pad), align(16). Scalar storage.
#[derive(Debug, Clone, Copy)]
#[repr(C, align(16))]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub _pad: f32,
}

impl Vec3 {
    pub const ZERO:  Self = Self { x:0.0, y:0.0, z:0.0, _pad:0.0 };
    pub const ONE:   Self = Self { x:1.0, y:1.0, z:1.0, _pad:0.0 };
    pub const X:     Self = Self { x:1.0, y:0.0, z:0.0, _pad:0.0 };
    pub const Y:     Self = Self { x:0.0, y:1.0, z:0.0, _pad:0.0 };
    pub const Z:     Self = Self { x:0.0, y:0.0, z:1.0, _pad:0.0 };
    pub const NEG_X: Self = Self { x:-1.0, y:0.0,  z:0.0, _pad:0.0 };
    pub const NEG_Y: Self = Self { x:0.0,  y:-1.0, z:0.0, _pad:0.0 };
    pub const NEG_Z: Self = Self { x:0.0,  y:0.0, z:-1.0, _pad:0.0 };

    #[inline(always)] pub fn new(x:f32, y:f32, z:f32) -> Self { Self{x,y,z,_pad:0.0} }
    #[inline(always)] pub fn splat(v:f32) -> Self { Self::new(v,v,v) }
    #[inline(always)] pub fn from_array(a:[f32;3]) -> Self { Self::new(a[0],a[1],a[2]) }
    #[inline(always)] pub fn to_array(self) -> [f32;3] { [self.x,self.y,self.z] }

    #[inline(always)]
    pub fn extend(self, w: f32) -> super::vec4::Vec4 {
        super::vec4::Vec4::new(self.x, self.y, self.z, w)
    }

    #[inline(always)]
    pub fn truncate(self) -> crate::f32::vec2::Vec2 {
        crate::f32::vec2::Vec2::new(self.x, self.y)
    }

    #[inline(always)] pub fn dot(self, r:Self) -> f32 { self.x*r.x+self.y*r.y+self.z*r.z }
    #[inline(always)] pub fn cross(self, r:Self) -> Self {
        Self::new(self.y*r.z-self.z*r.y, self.z*r.x-self.x*r.z, self.x*r.y-self.y*r.x)
    }
    #[inline(always)] pub fn length_sq(self) -> f32 { self.dot(self) }
    #[inline(always)] pub fn length(self) -> f32 { self.length_sq().sqrt() }
    #[inline(always)] pub fn normalize(self) -> Self {
        let l = self.length();
        if l < EPSILON { Self::ZERO } else { self / l }
    }
    #[inline(always)] pub fn lerp(self, r:Self, t:f32) -> Self { self + (r-self)*t }
    #[inline(always)] pub fn reflect(self, n:Self) -> Self { self - n*(2.0*self.dot(n)) }
    #[inline(always)] pub fn distance(self, r:Self) -> f32 { (self-r).length() }
    #[inline(always)] pub fn distance_sq(self, r:Self) -> f32 { (self-r).length_sq() }
    #[inline(always)] pub fn abs(self) -> Self { Self::new(self.x.abs(),self.y.abs(),self.z.abs()) }
    #[inline(always)] pub fn min(self, r:Self) -> Self { Self::new(self.x.min(r.x),self.y.min(r.y),self.z.min(r.z)) }
    #[inline(always)] pub fn max(self, r:Self) -> Self { Self::new(self.x.max(r.x),self.y.max(r.y),self.z.max(r.z)) }
    #[inline(always)] pub fn clamp(self, lo:Self, hi:Self) -> Self { self.max(lo).min(hi) }
    #[inline(always)] pub fn is_finite(self) -> bool { self.x.is_finite()&&self.y.is_finite()&&self.z.is_finite() }
    #[inline(always)] pub fn approx_eq(self, r:Self) -> bool {
        (self.x-r.x).abs()<EPSILON&&(self.y-r.y).abs()<EPSILON&&(self.z-r.z).abs()<EPSILON
    }

    // ── p1: angle between ─────────────────────────────────────────────────────

    /// Angle in radians between `self` and `rhs`. Returns `0.0` for zero-length inputs.
    #[inline(always)]
    pub fn angle_between(self, rhs: Self) -> f32 {
        let denom = (self.length_sq() * rhs.length_sq()).sqrt();
        if denom < EPSILON { 0.0 } else { (self.dot(rhs) / denom).clamp(-1.0, 1.0).acos() }
    }

    // ── p2: project / reject ──────────────────────────────────────────────────

    /// Project `self` onto `rhs` (component of `self` along `rhs`).
    /// Returns `ZERO` if `rhs` is zero-length.
    #[inline(always)]
    pub fn project_onto(self, rhs: Self) -> Self {
        let d = rhs.length_sq();
        if d < EPSILON { Self::ZERO } else { rhs * (self.dot(rhs) / d) }
    }

    /// Component of `self` perpendicular to `rhs`.
    #[inline(always)]
    pub fn reject_from(self, rhs: Self) -> Self { self - self.project_onto(rhs) }

    // ── p7: movement / clamping helpers ──────────────────────────────────────

    /// Move `self` toward `target` by at most `max_dist`. Never overshoots.
    #[inline(always)]
    pub fn move_towards(self, target: Self, max_dist: f32) -> Self {
        let d = target - self;
        let len = d.length();
        if len <= max_dist || len < EPSILON { target } else { self + d / len * max_dist }
    }

    /// Clamp the length to `[min, max]`. Zero vector is returned unchanged.
    #[inline(always)]
    pub fn clamp_length(self, min: f32, max: f32) -> Self {
        let len = self.length();
        if len < EPSILON { return Self::ZERO; }
        let clamped = len.clamp(min, max);
        if (clamped - len).abs() < EPSILON { self } else { self * (clamped / len) }
    }

    /// Clamp the length to at most `max`. Zero vector is returned unchanged.
    #[inline(always)]
    pub fn clamp_length_max(self, max: f32) -> Self {
        let len = self.length();
        if len > max && len > EPSILON { self * (max / len) } else { self }
    }

    /// Clamp the length to at least `min`. Zero vector is returned unchanged.
    #[inline(always)]
    pub fn clamp_length_min(self, min: f32) -> Self {
        let len = self.length();
        if len < min && len > EPSILON { self * (min / len) } else { self }
    }

    /// Midpoint between `self` and `rhs`.
    #[inline(always)]
    pub fn midpoint(self, rhs: Self) -> Self { (self + rhs) * 0.5 }

    /// True when `self` and `rhs` point in the same or opposite direction (cross product ≈ 0).
    #[inline(always)]
    pub fn is_parallel(self, rhs: Self) -> bool {
        self.cross(rhs).length_sq() < EPSILON * EPSILON
    }

    /// True when `self` and `rhs` are perpendicular (dot product ≈ 0).
    #[inline(always)]
    pub fn is_perpendicular(self, rhs: Self) -> bool { self.dot(rhs).abs() < EPSILON }

    // ── p6: spherical coordinates ─────────────────────────────────────────────

    /// Convert to spherical coordinates `(radius, theta, phi)`.
    ///
    /// `theta` = polar angle from +Z axis, range `[0, π]`.
    /// `phi`   = azimuthal angle from +X axis, range `[-π, π]`.
    #[inline]
    pub fn to_spherical(self) -> (f32, f32, f32) {
        let r = self.length();
        if r < EPSILON { return (0.0, 0.0, 0.0); }
        let theta = (self.z / r).clamp(-1.0, 1.0).acos();
        let phi   = self.y.atan2(self.x);
        (r, theta, phi)
    }

    /// Build from spherical coordinates `(r, theta, phi)`.
    ///
    /// `theta` = polar angle from +Z, `phi` = azimuthal from +X.
    #[inline]
    pub fn from_spherical(r: f32, theta: f32, phi: f32) -> Self {
        let sin_theta = theta.sin();
        Self::new(
            r * sin_theta * phi.cos(),
            r * sin_theta * phi.sin(),
            r * theta.cos(),
        )
    }
}

impl PartialEq for Vec3 {
    fn eq(&self, r:&Self) -> bool { self.x==r.x && self.y==r.y && self.z==r.z }
}
impl Default for Vec3 { fn default() -> Self { Self::ZERO } }
impl fmt::Display for Vec3 {
    fn fmt(&self, f:&mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f,"({}, {}, {})",self.x,self.y,self.z)
    }
}

impl Add  for Vec3 { type Output=Self; #[inline(always)] fn add(self,r:Self)->Self{Self::new(self.x+r.x,self.y+r.y,self.z+r.z)} }
impl Sub  for Vec3 { type Output=Self; #[inline(always)] fn sub(self,r:Self)->Self{Self::new(self.x-r.x,self.y-r.y,self.z-r.z)} }
impl Neg  for Vec3 { type Output=Self; #[inline(always)] fn neg(self)->Self{Self::new(-self.x,-self.y,-self.z)} }
impl Mul<f32> for Vec3 { type Output=Self; #[inline(always)] fn mul(self,s:f32)->Self{Self::new(self.x*s,self.y*s,self.z*s)} }
impl Mul<Vec3> for f32 { type Output=Vec3; #[inline(always)] fn mul(self,v:Vec3)->Vec3{Vec3::new(self*v.x,self*v.y,self*v.z)} }
impl Div<f32> for Vec3 { type Output=Self; #[inline(always)] fn div(self,s:f32)->Self{Self::new(self.x/s,self.y/s,self.z/s)} }
impl AddAssign for Vec3 { #[inline(always)] fn add_assign(&mut self,r:Self){self.x+=r.x;self.y+=r.y;self.z+=r.z;} }
impl SubAssign for Vec3 { #[inline(always)] fn sub_assign(&mut self,r:Self){self.x-=r.x;self.y-=r.y;self.z-=r.z;} }
impl MulAssign<f32> for Vec3 { #[inline(always)] fn mul_assign(&mut self,s:f32){self.x*=s;self.y*=s;self.z*=s;} }
impl DivAssign<f32> for Vec3 { #[inline(always)] fn div_assign(&mut self,s:f32){self.x/=s;self.y/=s;self.z/=s;} }

impl From<[f32;3]> for Vec3 { fn from(a:[f32;3])->Self{Self::new(a[0],a[1],a[2])} }
impl From<Vec3> for [f32;3] { fn from(v:Vec3)->[f32;3]{[v.x,v.y,v.z]} }
