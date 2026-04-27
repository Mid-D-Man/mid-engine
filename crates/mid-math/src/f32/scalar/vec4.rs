// crates/mid-math/src/f32/scalar/vec4.rs
//! Scalar Vec4 — fallback and reference.

use core::fmt;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use crate::EPSILON;

#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C, align(16))]
pub struct Vec4 { pub x:f32, pub y:f32, pub z:f32, pub w:f32 }

impl Vec4 {
    pub const ZERO: Self = Self{x:0.0,y:0.0,z:0.0,w:0.0};
    pub const ONE:  Self = Self{x:1.0,y:1.0,z:1.0,w:1.0};
    pub const X:    Self = Self{x:1.0,y:0.0,z:0.0,w:0.0};
    pub const Y:    Self = Self{x:0.0,y:1.0,z:0.0,w:0.0};
    pub const Z:    Self = Self{x:0.0,y:0.0,z:1.0,w:0.0};
    pub const W:    Self = Self{x:0.0,y:0.0,z:0.0,w:1.0};

    #[inline(always)] pub fn new(x:f32,y:f32,z:f32,w:f32)->Self{Self{x,y,z,w}}
    #[inline(always)] pub fn splat(v:f32)->Self{Self::new(v,v,v,v)}
    #[inline(always)] pub fn from_array(a:[f32;4])->Self{Self::new(a[0],a[1],a[2],a[3])}
    #[inline(always)] pub fn to_array(self)->[f32;4]{[self.x,self.y,self.z,self.w]}
    #[inline(always)] pub fn truncate(self)->super::vec3::Vec3{super::vec3::Vec3::new(self.x,self.y,self.z)}

    #[inline(always)] pub fn dot(self,r:Self)->f32{self.x*r.x+self.y*r.y+self.z*r.z+self.w*r.w}
    #[inline(always)] pub fn length_sq(self)->f32{self.dot(self)}
    #[inline(always)] pub fn length(self)->f32{self.length_sq().sqrt()}
    #[inline(always)] pub fn normalize(self)->Self{
        let l=self.length(); if l<EPSILON{Self::ZERO}else{self/l}
    }
    #[inline(always)] pub fn lerp(self,r:Self,t:f32)->Self{self+(r-self)*t}
    #[inline(always)] pub fn abs(self)->Self{Self::new(self.x.abs(),self.y.abs(),self.z.abs(),self.w.abs())}
    #[inline(always)] pub fn min(self,r:Self)->Self{Self::new(self.x.min(r.x),self.y.min(r.y),self.z.min(r.z),self.w.min(r.w))}
    #[inline(always)] pub fn max(self,r:Self)->Self{Self::new(self.x.max(r.x),self.y.max(r.y),self.z.max(r.z),self.w.max(r.w))}
    #[inline(always)] pub fn approx_eq(self,r:Self)->bool{
        (self.x-r.x).abs()<EPSILON&&(self.y-r.y).abs()<EPSILON&&
        (self.z-r.z).abs()<EPSILON&&(self.w-r.w).abs()<EPSILON
    }
}

impl Default for Vec4 { fn default()->Self{Self::ZERO} }
impl fmt::Display for Vec4 {
    fn fmt(&self,f:&mut fmt::Formatter<'_>)->fmt::Result{
        write!(f,"({}, {}, {}, {})",self.x,self.y,self.z,self.w)
    }
}

impl Add  for Vec4{type Output=Self;#[inline(always)]fn add(self,r:Self)->Self{Self::new(self.x+r.x,self.y+r.y,self.z+r.z,self.w+r.w)}}
impl Sub  for Vec4{type Output=Self;#[inline(always)]fn sub(self,r:Self)->Self{Self::new(self.x-r.x,self.y-r.y,self.z-r.z,self.w-r.w)}}
impl Neg  for Vec4{type Output=Self;#[inline(always)]fn neg(self)->Self{Self::new(-self.x,-self.y,-self.z,-self.w)}}
impl Mul<f32> for Vec4{type Output=Self;#[inline(always)]fn mul(self,s:f32)->Self{Self::new(self.x*s,self.y*s,self.z*s,self.w*s)}}
impl Mul<Vec4> for f32{type Output=Vec4;#[inline(always)]fn mul(self,v:Vec4)->Vec4{Vec4::new(self*v.x,self*v.y,self*v.z,self*v.w)}}
impl Div<f32> for Vec4{type Output=Self;#[inline(always)]fn div(self,s:f32)->Self{Self::new(self.x/s,self.y/s,self.z/s,self.w/s)}}
impl AddAssign for Vec4{#[inline(always)]fn add_assign(&mut self,r:Self){self.x+=r.x;self.y+=r.y;self.z+=r.z;self.w+=r.w;}}
impl SubAssign for Vec4{#[inline(always)]fn sub_assign(&mut self,r:Self){self.x-=r.x;self.y-=r.y;self.z-=r.z;self.w-=r.w;}}
impl MulAssign<f32> for Vec4{#[inline(always)]fn mul_assign(&mut self,s:f32){self.x*=s;self.y*=s;self.z*=s;self.w*=s;}}
impl DivAssign<f32> for Vec4{#[inline(always)]fn div_assign(&mut self,s:f32){self.x/=s;self.y/=s;self.z/=s;self.w/=s;}}

impl From<[f32;4]> for Vec4{fn from(a:[f32;4])->Self{Self::new(a[0],a[1],a[2],a[3])}}
impl From<Vec4> for [f32;4]{fn from(v:Vec4)->[f32;4]{[v.x,v.y,v.z,v.w]}}
