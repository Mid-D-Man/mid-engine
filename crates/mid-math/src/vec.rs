// crates/mid-math/src/vec.rs

//! Vec2 · Vec3 · Vec4
//!
//! All types are `#[repr(C)]` and safe across the FFI boundary.
//! Vec3 carries an explicit `_pad` field so its on-wire size is always
//! 16 bytes (matching the 16-byte alignment needed for SSE2).

use std::fmt;
use std::ops::{Add, Sub, Mul, Div, Neg, AddAssign, SubAssign, MulAssign, DivAssign};
use crate::EPSILON;

// ─────────────────────────────────────────────────────────────────────────────
// Vec2
// ─────────────────────────────────────────────────────────────────────────────

/// 2D vector. 8 bytes, no alignment padding.
/// Used for UV coordinates, 2D physics, screen-space positions.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct Vec2 {
    pub x: f32,
    pub y: f32,
}

impl Vec2 {
    pub const ZERO: Self = Self { x: 0.0, y: 0.0 };
    pub const ONE:  Self = Self { x: 1.0, y: 1.0 };
    pub const X:    Self = Self { x: 1.0, y: 0.0 };
    pub const Y:    Self = Self { x: 0.0, y: 1.0 };

    #[inline] pub fn new(x: f32, y: f32) -> Self { Self { x, y } }

    /// Dot product.
    #[inline] pub fn dot(self, rhs: Self) -> f32 { self.x*rhs.x + self.y*rhs.y }

    /// Squared magnitude (avoids sqrt).
    #[inline] pub fn length_sq(self) -> f32 { self.dot(self) }

    /// Magnitude.
    #[inline] pub fn length(self) -> f32 { self.length_sq().sqrt() }

    /// Unit vector. Returns ZERO if length < EPSILON.
    #[inline] pub fn normalize(self) -> Self {
        let l = self.length();
        if l < EPSILON { Self::ZERO } else { self / l }
    }

    /// Distance to `rhs`.
    #[inline] pub fn distance(self, rhs: Self) -> f32 { (self - rhs).length() }

    /// Squared distance (avoids sqrt).
    #[inline] pub fn distance_sq(self, rhs: Self) -> f32 { (self - rhs).length_sq() }

    /// Linear interpolation.
    #[inline] pub fn lerp(self, rhs: Self, t: f32) -> Self { self + (rhs - self) * t }

    /// Rotate 90° counter-clockwise (perpendicular).
    #[inline] pub fn perpendicular(self) -> Self { Self { x: -self.y, y: self.x } }

    /// Component-wise min.
    #[inline] pub fn min(self, rhs: Self) -> Self { Self::new(self.x.min(rhs.x), self.y.min(rhs.y)) }

    /// Component-wise max.
    #[inline] pub fn max(self, rhs: Self) -> Self { Self::new(self.x.max(rhs.x), self.y.max(rhs.y)) }

    /// Component-wise clamp.
    #[inline] pub fn clamp(self, lo: Self, hi: Self) -> Self { self.max(lo).min(hi) }

    /// Component-wise absolute value.
    #[inline] pub fn abs(self) -> Self { Self::new(self.x.abs(), self.y.abs()) }

    /// Approximate equality (per-component epsilon).
    #[inline] pub fn approx_eq(self, rhs: Self) -> bool {
        (self.x - rhs.x).abs() < EPSILON && (self.y - rhs.y).abs() < EPSILON
    }

    /// Extend to Vec3 by appending `z`.
    #[inline] pub fn extend(self, z: f32) -> Vec3 { Vec3::new(self.x, self.y, z) }
}

impl Default  for Vec2 { fn default() -> Self { Self::ZERO } }
impl PartialEq for Vec2 {
    fn eq(&self, r: &Self) -> bool { self.x == r.x && self.y == r.y }
}

impl fmt::Display for Vec2 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {})", self.x, self.y)
    }
}

impl Add    for Vec2 { type Output=Self; #[inline] fn add(self,r:Self)->Self { Self::new(self.x+r.x,self.y+r.y) } }
impl Sub    for Vec2 { type Output=Self; #[inline] fn sub(self,r:Self)->Self { Self::new(self.x-r.x,self.y-r.y) } }
impl Neg    for Vec2 { type Output=Self; #[inline] fn neg(self)->Self { Self::new(-self.x,-self.y) } }
impl Mul<f32> for Vec2 { type Output=Self; #[inline] fn mul(self,s:f32)->Self { Self::new(self.x*s,self.y*s) } }
impl Mul<Vec2> for f32 { type Output=Vec2; #[inline] fn mul(self,v:Vec2)->Vec2 { Vec2::new(self*v.x,self*v.y) } }
impl Div<f32> for Vec2 { type Output=Self; #[inline] fn div(self,s:f32)->Self { Self::new(self.x/s,self.y/s) } }
impl AddAssign for Vec2 { #[inline] fn add_assign(&mut self,r:Self) { self.x+=r.x; self.y+=r.y; } }
impl SubAssign for Vec2 { #[inline] fn sub_assign(&mut self,r:Self) { self.x-=r.x; self.y-=r.y; } }
impl MulAssign<f32> for Vec2 { #[inline] fn mul_assign(&mut self,s:f32) { self.x*=s; self.y*=s; } }
impl DivAssign<f32> for Vec2 { #[inline] fn div_assign(&mut self,s:f32) { self.x/=s; self.y/=s; } }

// ─────────────────────────────────────────────────────────────────────────────
// Vec3
// ─────────────────────────────────────────────────────────────────────────────

/// 3D vector. 16 bytes (12 data + 4 explicit pad for 16-byte SSE2 alignment).
///
/// The `_pad` field must always be 0.0. `Vec3::new()` guarantees this.
/// Do not read or write `_pad` directly.
///
/// **C layout:** `struct MidVec3 { float x, y, z, _pad; }` — 16 bytes.
#[derive(Debug, Clone, Copy)]
#[repr(C, align(16))]
pub struct Vec3 {
    pub x:    f32,
    pub y:    f32,
    pub z:    f32,
    /// Alignment padding. Always 0. Do not use directly.
    pub _pad: f32,
}

impl Vec3 {
    pub const ZERO:    Self = Self { x:0.0,y:0.0,z:0.0,_pad:0.0 };
    pub const ONE:     Self = Self { x:1.0,y:1.0,z:1.0,_pad:0.0 };
    pub const X:       Self = Self { x:1.0,y:0.0,z:0.0,_pad:0.0 };
    pub const Y:       Self = Self { x:0.0,y:1.0,z:0.0,_pad:0.0 };
    pub const Z:       Self = Self { x:0.0,y:0.0,z:1.0,_pad:0.0 };
    pub const NEG_X:   Self = Self { x:-1.0,y:0.0,z:0.0,_pad:0.0 };
    pub const NEG_Y:   Self = Self { x:0.0,y:-1.0,z:0.0,_pad:0.0 };
    pub const NEG_Z:   Self = Self { x:0.0,y:0.0,z:-1.0,_pad:0.0 };

    #[inline] pub fn new(x: f32, y: f32, z: f32) -> Self { Self { x, y, z, _pad: 0.0 } }

    /// Dot product.
    #[inline] pub fn dot(self, rhs: Self) -> f32 { self.x*rhs.x + self.y*rhs.y + self.z*rhs.z }

    /// Cross product (`self × rhs`).
    #[inline] pub fn cross(self, rhs: Self) -> Self {
        Self::new(
            self.y*rhs.z - self.z*rhs.y,
            self.z*rhs.x - self.x*rhs.z,
            self.x*rhs.y - self.y*rhs.x,
        )
    }

    /// Squared magnitude (avoids sqrt).
    #[inline] pub fn length_sq(self) -> f32 { self.dot(self) }

    /// Magnitude.
    #[inline] pub fn length(self) -> f32 { self.length_sq().sqrt() }

    /// Unit vector. Returns ZERO if length < EPSILON.
    #[inline] pub fn normalize(self) -> Self {
        let l = self.length();
        if l < EPSILON { Self::ZERO } else { self / l }
    }

    /// Distance to `rhs`.
    #[inline] pub fn distance(self, rhs: Self) -> f32 { (self - rhs).length() }

    /// Squared distance.
    #[inline] pub fn distance_sq(self, rhs: Self) -> f32 { (self - rhs).length_sq() }

    /// Linear interpolation.
    #[inline] pub fn lerp(self, rhs: Self, t: f32) -> Self { self + (rhs - self) * t }

    /// Reflect around `normal` (normal must be unit length).
    #[inline] pub fn reflect(self, normal: Self) -> Self { self - normal * (2.0 * self.dot(normal)) }

    /// Component-wise min.
    #[inline] pub fn min(self, rhs: Self) -> Self { Self::new(self.x.min(rhs.x),self.y.min(rhs.y),self.z.min(rhs.z)) }

    /// Component-wise max.
    #[inline] pub fn max(self, rhs: Self) -> Self { Self::new(self.x.max(rhs.x),self.y.max(rhs.y),self.z.max(rhs.z)) }

    /// Component-wise clamp.
    #[inline] pub fn clamp(self, lo: Self, hi: Self) -> Self { self.max(lo).min(hi) }

    /// Component-wise absolute value.
    #[inline] pub fn abs(self) -> Self { Self::new(self.x.abs(),self.y.abs(),self.z.abs()) }

    /// Approximate equality (per-component epsilon).
    #[inline] pub fn approx_eq(self, rhs: Self) -> bool {
        (self.x-rhs.x).abs() < EPSILON &&
        (self.y-rhs.y).abs() < EPSILON &&
        (self.z-rhs.z).abs() < EPSILON
    }

    /// Drop z — returns the XY components as Vec2.
    #[inline] pub fn truncate(self) -> Vec2 { Vec2::new(self.x, self.y) }

    /// Promote to Vec4 by appending `w`.
    #[inline] pub fn extend(self, w: f32) -> Vec4 { Vec4::new(self.x, self.y, self.z, w) }
}

impl Default  for Vec3 { fn default() -> Self { Self::ZERO } }
impl PartialEq for Vec3 {
    /// Ignores `_pad`.
    fn eq(&self, r: &Self) -> bool { self.x==r.x && self.y==r.y && self.z==r.z }
}

impl fmt::Display for Vec3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {}, {})", self.x, self.y, self.z)
    }
}

impl Add    for Vec3 { type Output=Self; #[inline] fn add(self,r:Self)->Self { Self::new(self.x+r.x,self.y+r.y,self.z+r.z) } }
impl Sub    for Vec3 { type Output=Self; #[inline] fn sub(self,r:Self)->Self { Self::new(self.x-r.x,self.y-r.y,self.z-r.z) } }
impl Neg    for Vec3 { type Output=Self; #[inline] fn neg(self)->Self { Self::new(-self.x,-self.y,-self.z) } }
impl Mul<f32> for Vec3 { type Output=Self; #[inline] fn mul(self,s:f32)->Self { Self::new(self.x*s,self.y*s,self.z*s) } }
impl Mul<Vec3> for f32 { type Output=Vec3; #[inline] fn mul(self,v:Vec3)->Vec3 { Vec3::new(self*v.x,self*v.y,self*v.z) } }
impl Div<f32> for Vec3 { type Output=Self; #[inline] fn div(self,s:f32)->Self { Self::new(self.x/s,self.y/s,self.z/s) } }
impl AddAssign for Vec3 { #[inline] fn add_assign(&mut self,r:Self) { self.x+=r.x; self.y+=r.y; self.z+=r.z; } }
impl SubAssign for Vec3 { #[inline] fn sub_assign(&mut self,r:Self) { self.x-=r.x; self.y-=r.y; self.z-=r.z; } }
impl MulAssign<f32> for Vec3 { #[inline] fn mul_assign(&mut self,s:f32) { self.x*=s; self.y*=s; self.z*=s; } }
impl DivAssign<f32> for Vec3 { #[inline] fn div_assign(&mut self,s:f32) { self.x/=s; self.y/=s; self.z/=s; } }

// ─────────────────────────────────────────────────────────────────────────────
// Vec4
// ─────────────────────────────────────────────────────────────────────────────

/// 4D vector. 16 bytes, 16-byte aligned.
/// Used for homogeneous coordinates, RGBA color, and quaternion storage.
#[derive(Debug, Clone, Copy)]
#[repr(C, align(16))]
pub struct Vec4 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

impl Vec4 {
    pub const ZERO: Self = Self { x:0.0,y:0.0,z:0.0,w:0.0 };
    pub const ONE:  Self = Self { x:1.0,y:1.0,z:1.0,w:1.0 };
    pub const X:    Self = Self { x:1.0,y:0.0,z:0.0,w:0.0 };
    pub const Y:    Self = Self { x:0.0,y:1.0,z:0.0,w:0.0 };
    pub const Z:    Self = Self { x:0.0,y:0.0,z:1.0,w:0.0 };
    pub const W:    Self = Self { x:0.0,y:0.0,z:0.0,w:1.0 };

    #[inline] pub fn new(x: f32, y: f32, z: f32, w: f32) -> Self { Self { x, y, z, w } }

    /// Dot product (all four components).
    #[inline] pub fn dot(self, rhs: Self) -> f32 { self.x*rhs.x+self.y*rhs.y+self.z*rhs.z+self.w*rhs.w }

    /// Squared magnitude.
    #[inline] pub fn length_sq(self) -> f32 { self.dot(self) }

    /// Magnitude.
    #[inline] pub fn length(self) -> f32 { self.length_sq().sqrt() }

    /// Unit vector. Returns ZERO if length < EPSILON.
    #[inline] pub fn normalize(self) -> Self {
        let l = self.length();
        if l < EPSILON { Self::ZERO } else { self / l }
    }

    /// Linear interpolation.
    #[inline] pub fn lerp(self, rhs: Self, t: f32) -> Self { self + (rhs - self) * t }

    /// Component-wise min.
    #[inline] pub fn min(self, rhs: Self) -> Self { Self::new(self.x.min(rhs.x),self.y.min(rhs.y),self.z.min(rhs.z),self.w.min(rhs.w)) }

    /// Component-wise max.
    #[inline] pub fn max(self, rhs: Self) -> Self { Self::new(self.x.max(rhs.x),self.y.max(rhs.y),self.z.max(rhs.z),self.w.max(rhs.w)) }

    /// Approximate equality.
    #[inline] pub fn approx_eq(self, rhs: Self) -> bool {
        (self.x-rhs.x).abs() < EPSILON && (self.y-rhs.y).abs() < EPSILON &&
        (self.z-rhs.z).abs() < EPSILON && (self.w-rhs.w).abs() < EPSILON
    }

    /// Drop w — returns XYZ as Vec3.
    #[inline] pub fn truncate(self) -> Vec3 { Vec3::new(self.x, self.y, self.z) }
}

impl Default  for Vec4 { fn default() -> Self { Self::ZERO } }
impl PartialEq for Vec4 {
    fn eq(&self, r: &Self) -> bool { self.x==r.x && self.y==r.y && self.z==r.z && self.w==r.w }
}

impl fmt::Display for Vec4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {}, {}, {})", self.x, self.y, self.z, self.w)
    }
}

impl Add    for Vec4 { type Output=Self; #[inline] fn add(self,r:Self)->Self { Self::new(self.x+r.x,self.y+r.y,self.z+r.z,self.w+r.w) } }
impl Sub    for Vec4 { type Output=Self; #[inline] fn sub(self,r:Self)->Self { Self::new(self.x-r.x,self.y-r.y,self.z-r.z,self.w-r.w) } }
impl Neg    for Vec4 { type Output=Self; #[inline] fn neg(self)->Self { Self::new(-self.x,-self.y,-self.z,-self.w) } }
impl Mul<f32> for Vec4 { type Output=Self; #[inline] fn mul(self,s:f32)->Self { Self::new(self.x*s,self.y*s,self.z*s,self.w*s) } }
impl Mul<Vec4> for f32 { type Output=Vec4; #[inline] fn mul(self,v:Vec4)->Vec4 { Vec4::new(self*v.x,self*v.y,self*v.z,self*v.w) } }
impl Div<f32> for Vec4 { type Output=Self; #[inline] fn div(self,s:f32)->Self { Self::new(self.x/s,self.y/s,self.z/s,self.w/s) } }
impl AddAssign for Vec4 { #[inline] fn add_assign(&mut self,r:Self) { self.x+=r.x; self.y+=r.y; self.z+=r.z; self.w+=r.w; } }
impl SubAssign for Vec4 { #[inline] fn sub_assign(&mut self,r:Self) { self.x-=r.x; self.y-=r.y; self.z-=r.z; self.w-=r.w; } }
impl MulAssign<f32> for Vec4 { #[inline] fn mul_assign(&mut self,s:f32) { self.x*=s; self.y*=s; self.z*=s; self.w*=s; } }
impl DivAssign<f32> for Vec4 { #[inline] fn div_assign(&mut self,s:f32) { self.x/=s; self.y/=s; self.z/=s; self.w/=s; } }
