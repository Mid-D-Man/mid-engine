// crates/mid-math/src/vec.rs

//! Vec2 · Vec3 · Vec4
//!
//! All types are `#[repr(C)]` and safe across the FFI boundary.
//! Vec3 carries an explicit `_pad` field so its on-wire size is 16 bytes,
//! matching the 16-byte alignment needed for SSE2.
//!
//! Every method that is called in a hot loop carries `#[inline(always)]`.
//! This removes the function-call overhead that separates a "good" math
//! library from a "great" one — it is the primary reason glam is fast.

use std::fmt;
use std::ops::{Add, Sub, Mul, Div, Neg, AddAssign, SubAssign, MulAssign, DivAssign};
use crate::EPSILON;

// ─────────────────────────────────────────────────────────────────────────────
// Vec2
// ─────────────────────────────────────────────────────────────────────────────

/// 2D vector. 8 bytes, no padding.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct Vec2 { pub x: f32, pub y: f32 }

impl Vec2 {
    pub const ZERO: Self = Self { x: 0.0, y: 0.0 };
    pub const ONE:  Self = Self { x: 1.0, y: 1.0 };
    pub const X:    Self = Self { x: 1.0, y: 0.0 };
    pub const Y:    Self = Self { x: 0.0, y: 1.0 };

    #[inline(always)] pub fn new(x: f32, y: f32) -> Self { Self { x, y } }
    #[inline(always)] pub fn dot(self, r: Self) -> f32 { self.x*r.x + self.y*r.y }
    #[inline(always)] pub fn length_sq(self) -> f32 { self.dot(self) }
    #[inline(always)] pub fn length(self) -> f32 { self.length_sq().sqrt() }
    #[inline(always)] pub fn normalize(self) -> Self {
        let l = self.length();
        if l < EPSILON { Self::ZERO } else { self / l }
    }
    #[inline(always)] pub fn distance(self, r: Self) -> f32 { (self - r).length() }
    #[inline(always)] pub fn distance_sq(self, r: Self) -> f32 { (self - r).length_sq() }
    #[inline(always)] pub fn lerp(self, r: Self, t: f32) -> Self { self + (r - self) * t }
    #[inline(always)] pub fn perpendicular(self) -> Self { Self { x: -self.y, y: self.x } }
    #[inline(always)] pub fn min(self, r: Self) -> Self { Self::new(self.x.min(r.x), self.y.min(r.y)) }
    #[inline(always)] pub fn max(self, r: Self) -> Self { Self::new(self.x.max(r.x), self.y.max(r.y)) }
    #[inline(always)] pub fn clamp(self, lo: Self, hi: Self) -> Self { self.max(lo).min(hi) }
    #[inline(always)] pub fn abs(self) -> Self { Self::new(self.x.abs(), self.y.abs()) }
    #[inline(always)] pub fn approx_eq(self, r: Self) -> bool {
        (self.x-r.x).abs() < EPSILON && (self.y-r.y).abs() < EPSILON
    }
    #[inline(always)] pub fn extend(self, z: f32) -> Vec3 { Vec3::new(self.x, self.y, z) }
}

impl Default   for Vec2 { fn default() -> Self { Self::ZERO } }
impl PartialEq for Vec2 { fn eq(&self, r: &Self) -> bool { self.x==r.x && self.y==r.y } }
impl fmt::Display for Vec2 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "({}, {})", self.x, self.y) }
}

impl Add    for Vec2 { type Output=Self; #[inline(always)] fn add(self,r:Self)->Self { Self::new(self.x+r.x,self.y+r.y) } }
impl Sub    for Vec2 { type Output=Self; #[inline(always)] fn sub(self,r:Self)->Self { Self::new(self.x-r.x,self.y-r.y) } }
impl Neg    for Vec2 { type Output=Self; #[inline(always)] fn neg(self)->Self { Self::new(-self.x,-self.y) } }
impl Mul<f32> for Vec2 { type Output=Self; #[inline(always)] fn mul(self,s:f32)->Self { Self::new(self.x*s,self.y*s) } }
impl Mul<Vec2> for f32 { type Output=Vec2; #[inline(always)] fn mul(self,v:Vec2)->Vec2 { Vec2::new(self*v.x,self*v.y) } }
impl Div<f32> for Vec2 { type Output=Self; #[inline(always)] fn div(self,s:f32)->Self { Self::new(self.x/s,self.y/s) } }
impl AddAssign for Vec2 { #[inline(always)] fn add_assign(&mut self,r:Self){self.x+=r.x;self.y+=r.y;} }
impl SubAssign for Vec2 { #[inline(always)] fn sub_assign(&mut self,r:Self){self.x-=r.x;self.y-=r.y;} }
impl MulAssign<f32> for Vec2 { #[inline(always)] fn mul_assign(&mut self,s:f32){self.x*=s;self.y*=s;} }
impl DivAssign<f32> for Vec2 { #[inline(always)] fn div_assign(&mut self,s:f32){self.x/=s;self.y/=s;} }

// ─────────────────────────────────────────────────────────────────────────────
// Vec3
// ─────────────────────────────────────────────────────────────────────────────

/// 3D vector. 16 bytes (12 data + 4 explicit pad for SSE2 alignment).
///
/// `_pad` is always 0.0. `Vec3::new()` guarantees this.
/// **C layout:** `struct MidVec3 { float x, y, z, _pad; }` — 16 bytes.
#[derive(Debug, Clone, Copy)]
#[repr(C, align(16))]
pub struct Vec3 { pub x: f32, pub y: f32, pub z: f32, pub _pad: f32 }

impl Vec3 {
    pub const ZERO:  Self = Self { x:0.0, y:0.0, z:0.0, _pad:0.0 };
    pub const ONE:   Self = Self { x:1.0, y:1.0, z:1.0, _pad:0.0 };
    pub const X:     Self = Self { x:1.0, y:0.0, z:0.0, _pad:0.0 };
    pub const Y:     Self = Self { x:0.0, y:1.0, z:0.0, _pad:0.0 };
    pub const Z:     Self = Self { x:0.0, y:0.0, z:1.0, _pad:0.0 };
    pub const NEG_X: Self = Self { x:-1.0,y:0.0, z:0.0, _pad:0.0 };
    pub const NEG_Y: Self = Self { x:0.0, y:-1.0,z:0.0, _pad:0.0 };
    pub const NEG_Z: Self = Self { x:0.0, y:0.0, z:-1.0,_pad:0.0 };

    #[inline(always)] pub fn new(x: f32, y: f32, z: f32) -> Self { Self { x, y, z, _pad: 0.0 } }
    #[inline(always)] pub fn dot(self, r: Self) -> f32 { self.x*r.x + self.y*r.y + self.z*r.z }
    #[inline(always)] pub fn cross(self, r: Self) -> Self {
        Self::new(self.y*r.z-self.z*r.y, self.z*r.x-self.x*r.z, self.x*r.y-self.y*r.x)
    }
    #[inline(always)] pub fn length_sq(self) -> f32 { self.dot(self) }
    #[inline(always)] pub fn length(self) -> f32 { self.length_sq().sqrt() }
    #[inline(always)] pub fn normalize(self) -> Self {
        let l = self.length();
        if l < EPSILON { Self::ZERO } else { self / l }
    }
    #[inline(always)] pub fn distance(self, r: Self) -> f32 { (self - r).length() }
    #[inline(always)] pub fn distance_sq(self, r: Self) -> f32 { (self - r).length_sq() }
    #[inline(always)] pub fn lerp(self, r: Self, t: f32) -> Self { self + (r - self) * t }
    #[inline(always)] pub fn reflect(self, n: Self) -> Self { self - n * (2.0 * self.dot(n)) }
    #[inline(always)] pub fn min(self, r: Self) -> Self { Self::new(self.x.min(r.x),self.y.min(r.y),self.z.min(r.z)) }
    #[inline(always)] pub fn max(self, r: Self) -> Self { Self::new(self.x.max(r.x),self.y.max(r.y),self.z.max(r.z)) }
    #[inline(always)] pub fn clamp(self, lo: Self, hi: Self) -> Self { self.max(lo).min(hi) }
    #[inline(always)] pub fn abs(self) -> Self { Self::new(self.x.abs(),self.y.abs(),self.z.abs()) }
    #[inline(always)] pub fn approx_eq(self, r: Self) -> bool {
        (self.x-r.x).abs()<EPSILON && (self.y-r.y).abs()<EPSILON && (self.z-r.z).abs()<EPSILON
    }
    #[inline(always)] pub fn truncate(self) -> Vec2 { Vec2::new(self.x, self.y) }
    #[inline(always)] pub fn extend(self, w: f32) -> Vec4 { Vec4::new(self.x, self.y, self.z, w) }
}

impl Default   for Vec3 { fn default() -> Self { Self::ZERO } }
impl PartialEq for Vec3 {
    /// Ignores `_pad`.
    fn eq(&self, r: &Self) -> bool { self.x==r.x && self.y==r.y && self.z==r.z }
}
impl fmt::Display for Vec3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f,"({}, {}, {})",self.x,self.y,self.z) }
}

impl Add    for Vec3 { type Output=Self; #[inline(always)] fn add(self,r:Self)->Self{Self::new(self.x+r.x,self.y+r.y,self.z+r.z)} }
impl Sub    for Vec3 { type Output=Self; #[inline(always)] fn sub(self,r:Self)->Self{Self::new(self.x-r.x,self.y-r.y,self.z-r.z)} }
impl Neg    for Vec3 { type Output=Self; #[inline(always)] fn neg(self)->Self{Self::new(-self.x,-self.y,-self.z)} }
impl Mul<f32> for Vec3 { type Output=Self; #[inline(always)] fn mul(self,s:f32)->Self{Self::new(self.x*s,self.y*s,self.z*s)} }
impl Mul<Vec3> for f32 { type Output=Vec3; #[inline(always)] fn mul(self,v:Vec3)->Vec3{Vec3::new(self*v.x,self*v.y,self*v.z)} }
impl Div<f32> for Vec3 { type Output=Self; #[inline(always)] fn div(self,s:f32)->Self{Self::new(self.x/s,self.y/s,self.z/s)} }
impl AddAssign for Vec3 { #[inline(always)] fn add_assign(&mut self,r:Self){self.x+=r.x;self.y+=r.y;self.z+=r.z;} }
impl SubAssign for Vec3 { #[inline(always)] fn sub_assign(&mut self,r:Self){self.x-=r.x;self.y-=r.y;self.z-=r.z;} }
impl MulAssign<f32> for Vec3 { #[inline(always)] fn mul_assign(&mut self,s:f32){self.x*=s;self.y*=s;self.z*=s;} }
impl DivAssign<f32> for Vec3 { #[inline(always)] fn div_assign(&mut self,s:f32){self.x/=s;self.y/=s;self.z/=s;} }

// ─────────────────────────────────────────────────────────────────────────────
// Vec4
// ─────────────────────────────────────────────────────────────────────────────

/// 4D vector. 16 bytes, 16-byte aligned.
#[derive(Debug, Clone, Copy)]
#[repr(C, align(16))]
pub struct Vec4 { pub x: f32, pub y: f32, pub z: f32, pub w: f32 }

impl Vec4 {
    pub const ZERO: Self = Self { x:0.0,y:0.0,z:0.0,w:0.0 };
    pub const ONE:  Self = Self { x:1.0,y:1.0,z:1.0,w:1.0 };
    pub const X:    Self = Self { x:1.0,y:0.0,z:0.0,w:0.0 };
    pub const Y:    Self = Self { x:0.0,y:1.0,z:0.0,w:0.0 };
    pub const Z:    Self = Self { x:0.0,y:0.0,z:1.0,w:0.0 };
    pub const W:    Self = Self { x:0.0,y:0.0,z:0.0,w:1.0 };

    #[inline(always)] pub fn new(x:f32,y:f32,z:f32,w:f32)->Self{Self{x,y,z,w}}
    #[inline(always)] pub fn dot(self,r:Self)->f32{self.x*r.x+self.y*r.y+self.z*r.z+self.w*r.w}
    #[inline(always)] pub fn length_sq(self)->f32{self.dot(self)}
    #[inline(always)] pub fn length(self)->f32{self.length_sq().sqrt()}
    #[inline(always)] pub fn normalize(self)->Self{
        let l=self.length(); if l<EPSILON{Self::ZERO}else{self/l}
    }
    #[inline(always)] pub fn lerp(self,r:Self,t:f32)->Self{self+(r-self)*t}
    #[inline(always)] pub fn min(self,r:Self)->Self{Self::new(self.x.min(r.x),self.y.min(r.y),self.z.min(r.z),self.w.min(r.w))}
    #[inline(always)] pub fn max(self,r:Self)->Self{Self::new(self.x.max(r.x),self.y.max(r.y),self.z.max(r.z),self.w.max(r.w))}
    #[inline(always)] pub fn approx_eq(self,r:Self)->bool{
        (self.x-r.x).abs()<EPSILON&&(self.y-r.y).abs()<EPSILON&&(self.z-r.z).abs()<EPSILON&&(self.w-r.w).abs()<EPSILON
    }
    #[inline(always)] pub fn truncate(self)->Vec3{Vec3::new(self.x,self.y,self.z)}
}

impl Default   for Vec4 { fn default()->Self{Self::ZERO} }
impl PartialEq for Vec4 { fn eq(&self,r:&Self)->bool{self.x==r.x&&self.y==r.y&&self.z==r.z&&self.w==r.w} }
impl fmt::Display for Vec4 {
    fn fmt(&self,f:&mut fmt::Formatter<'_>)->fmt::Result{write!(f,"({}, {}, {}, {})",self.x,self.y,self.z,self.w)}
}

impl Add    for Vec4 { type Output=Self; #[inline(always)] fn add(self,r:Self)->Self{Self::new(self.x+r.x,self.y+r.y,self.z+r.z,self.w+r.w)} }
impl Sub    for Vec4 { type Output=Self; #[inline(always)] fn sub(self,r:Self)->Self{Self::new(self.x-r.x,self.y-r.y,self.z-r.z,self.w-r.w)} }
impl Neg    for Vec4 { type Output=Self; #[inline(always)] fn neg(self)->Self{Self::new(-self.x,-self.y,-self.z,-self.w)} }
impl Mul<f32> for Vec4 { type Output=Self; #[inline(always)] fn mul(self,s:f32)->Self{Self::new(self.x*s,self.y*s,self.z*s,self.w*s)} }
impl Mul<Vec4> for f32 { type Output=Vec4; #[inline(always)] fn mul(self,v:Vec4)->Vec4{Vec4::new(self*v.x,self*v.y,self*v.z,self*v.w)} }
impl Div<f32> for Vec4 { type Output=Self; #[inline(always)] fn div(self,s:f32)->Self{Self::new(self.x/s,self.y/s,self.z/s,self.w/s)} }
impl AddAssign for Vec4 { #[inline(always)] fn add_assign(&mut self,r:Self){self.x+=r.x;self.y+=r.y;self.z+=r.z;self.w+=r.w;} }
impl SubAssign for Vec4 { #[inline(always)] fn sub_assign(&mut self,r:Self){self.x-=r.x;self.y-=r.y;self.z-=r.z;self.w-=r.w;} }
impl MulAssign<f32> for Vec4 { #[inline(always)] fn mul_assign(&mut self,s:f32){self.x*=s;self.y*=s;self.z*=s;self.w*=s;} }
impl DivAssign<f32> for Vec4 { #[inline(always)] fn div_assign(&mut self,s:f32){self.x/=s;self.y/=s;self.z/=s;self.w/=s;} }
