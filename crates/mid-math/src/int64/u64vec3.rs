// crates/mid-math/src/int64/u64vec3.rs
//! 3D unsigned 64-bit integer vector. 24 bytes, align 8. No padding. Always scalar.

use core::fmt;
use core::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign,
    Div, DivAssign, Index, IndexMut, Mul, MulAssign, Not,
    Rem, RemAssign, Shl, ShlAssign, Shr, ShrAssign, Sub, SubAssign,
};
use crate::{BVec3, U64Vec2, U64Vec4, UVec3, IVec3, DVec3, Vec3};

/// 3D unsigned 64-bit integer vector. 24 bytes, align 8. No padding.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(C)]
pub struct U64Vec3 {
    pub x: u64,
    pub y: u64,
    pub z: u64,
}

impl U64Vec3 {
    pub const ZERO: Self = Self::splat(0);
    pub const ONE:  Self = Self::splat(1);
    pub const MIN:  Self = Self::splat(u64::MIN);
    pub const MAX:  Self = Self::splat(u64::MAX);
    pub const X:    Self = Self::new(1, 0, 0);
    pub const Y:    Self = Self::new(0, 1, 0);
    pub const Z:    Self = Self::new(0, 0, 1);

    #[inline(always)] pub const fn new(x: u64, y: u64, z: u64) -> Self { Self { x, y, z } }
    #[inline(always)] pub const fn splat(v: u64) -> Self { Self { x: v, y: v, z: v } }
    #[inline(always)] pub const fn from_array(a: [u64; 3]) -> Self { Self::new(a[0], a[1], a[2]) }
    #[inline(always)] pub const fn to_array(self) -> [u64; 3] { [self.x, self.y, self.z] }
    #[inline(always)] pub fn extend(self, w: u64) -> U64Vec4 { U64Vec4::new(self.x, self.y, self.z, w) }
    #[inline(always)] pub const fn truncate(self) -> U64Vec2 { U64Vec2::new(self.x, self.y) }

    #[inline]
    pub fn select(mask: BVec3, if_true: Self, if_false: Self) -> Self {
        Self::new(
            if mask.x { if_true.x } else { if_false.x },
            if mask.y { if_true.y } else { if_false.y },
            if mask.z { if_true.z } else { if_false.z },
        )
    }

    #[inline] pub fn dot(self, r: Self) -> u64 { self.x*r.x + self.y*r.y + self.z*r.z }
    #[inline] pub fn cross(self, r: Self) -> Self {
        Self::new(
            self.y.wrapping_mul(r.z).wrapping_sub(self.z.wrapping_mul(r.y)),
            self.z.wrapping_mul(r.x).wrapping_sub(self.x.wrapping_mul(r.z)),
            self.x.wrapping_mul(r.y).wrapping_sub(self.y.wrapping_mul(r.x)),
        )
    }
    #[inline] pub fn length_sq(self) -> u64 { self.dot(self) }
    #[inline] pub fn min(self, r: Self) -> Self { Self::new(self.x.min(r.x), self.y.min(r.y), self.z.min(r.z)) }
    #[inline] pub fn max(self, r: Self) -> Self { Self::new(self.x.max(r.x), self.y.max(r.y), self.z.max(r.z)) }
    #[inline] pub fn clamp(self, lo: Self, hi: Self) -> Self { self.max(lo).min(hi) }
    #[inline] pub fn min_element(self) -> u64 { self.x.min(self.y).min(self.z) }
    #[inline] pub fn max_element(self) -> u64 { self.x.max(self.y).max(self.z) }
    #[inline] pub fn element_sum(self) -> u64 { self.x + self.y + self.z }
    #[inline] pub fn element_product(self) -> u64 { self.x * self.y * self.z }

    #[inline] pub fn cmpeq(self, r: Self) -> BVec3 { BVec3::new(self.x==r.x, self.y==r.y, self.z==r.z) }
    #[inline] pub fn cmpne(self, r: Self) -> BVec3 { BVec3::new(self.x!=r.x, self.y!=r.y, self.z!=r.z) }
    #[inline] pub fn cmpge(self, r: Self) -> BVec3 { BVec3::new(self.x>=r.x, self.y>=r.y, self.z>=r.z) }
    #[inline] pub fn cmpgt(self, r: Self) -> BVec3 { BVec3::new(self.x>r.x,  self.y>r.y,  self.z>r.z)  }
    #[inline] pub fn cmple(self, r: Self) -> BVec3 { BVec3::new(self.x<=r.x, self.y<=r.y, self.z<=r.z) }
    #[inline] pub fn cmplt(self, r: Self) -> BVec3 { BVec3::new(self.x<r.x,  self.y<r.y,  self.z<r.z)  }

    #[inline] pub fn wrapping_add(self, r: Self) -> Self { Self::new(self.x.wrapping_add(r.x), self.y.wrapping_add(r.y), self.z.wrapping_add(r.z)) }
    #[inline] pub fn wrapping_sub(self, r: Self) -> Self { Self::new(self.x.wrapping_sub(r.x), self.y.wrapping_sub(r.y), self.z.wrapping_sub(r.z)) }
    #[inline] pub fn saturating_add(self, r: Self) -> Self { Self::new(self.x.saturating_add(r.x), self.y.saturating_add(r.y), self.z.saturating_add(r.z)) }
    #[inline] pub fn saturating_sub(self, r: Self) -> Self { Self::new(self.x.saturating_sub(r.x), self.y.saturating_sub(r.y), self.z.saturating_sub(r.z)) }
    #[inline] pub fn checked_add(self, r: Self) -> Option<Self> { Some(Self::new(self.x.checked_add(r.x)?, self.y.checked_add(r.y)?, self.z.checked_add(r.z)?)) }
    #[inline] pub fn checked_sub(self, r: Self) -> Option<Self> { Some(Self::new(self.x.checked_sub(r.x)?, self.y.checked_sub(r.y)?, self.z.checked_sub(r.z)?)) }
    #[inline] pub fn checked_mul(self, r: Self) -> Option<Self> { Some(Self::new(self.x.checked_mul(r.x)?, self.y.checked_mul(r.y)?, self.z.checked_mul(r.z)?)) }

    #[inline] pub fn as_vec3(self)    -> Vec3  { Vec3::new(self.x as f32, self.y as f32, self.z as f32) }
    #[inline] pub fn as_dvec3(self)   -> DVec3 { DVec3::new(self.x as f64, self.y as f64, self.z as f64) }
    #[inline] pub fn as_ivec3(self)   -> IVec3 { IVec3::new(self.x as i32, self.y as i32, self.z as i32) }
    #[inline] pub fn as_uvec3(self)   -> UVec3 { UVec3::new(self.x as u32, self.y as u32, self.z as u32) }
    #[inline] pub fn as_i64vec3(self) -> crate::I64Vec3 { crate::I64Vec3::new(self.x as i64, self.y as i64, self.z as i64) }
}

impl Add  for U64Vec3 { type Output=Self; #[inline] fn add(self,r:Self)->Self { Self::new(self.x+r.x, self.y+r.y, self.z+r.z) } }
impl Sub  for U64Vec3 { type Output=Self; #[inline] fn sub(self,r:Self)->Self { Self::new(self.x-r.x, self.y-r.y, self.z-r.z) } }
impl Mul  for U64Vec3 { type Output=Self; #[inline] fn mul(self,r:Self)->Self { Self::new(self.x*r.x, self.y*r.y, self.z*r.z) } }
impl Div  for U64Vec3 { type Output=Self; #[inline] fn div(self,r:Self)->Self { Self::new(self.x/r.x, self.y/r.y, self.z/r.z) } }
impl Rem  for U64Vec3 { type Output=Self; #[inline] fn rem(self,r:Self)->Self { Self::new(self.x%r.x, self.y%r.y, self.z%r.z) } }
impl Not  for U64Vec3 { type Output=Self; #[inline] fn not(self)->Self { Self::new(!self.x, !self.y, !self.z) } }

impl Mul<u64> for U64Vec3 { type Output=Self; #[inline] fn mul(self,s:u64)->Self { Self::new(self.x*s, self.y*s, self.z*s) } }
impl Div<u64> for U64Vec3 { type Output=Self; #[inline] fn div(self,s:u64)->Self { Self::new(self.x/s, self.y/s, self.z/s) } }
impl Mul<U64Vec3> for u64 { type Output=U64Vec3; #[inline] fn mul(self,v:U64Vec3)->U64Vec3 { U64Vec3::new(self*v.x, self*v.y, self*v.z) } }

impl AddAssign      for U64Vec3 { #[inline] fn add_assign(&mut self,r:Self) { self.x+=r.x; self.y+=r.y; self.z+=r.z; } }
impl SubAssign      for U64Vec3 { #[inline] fn sub_assign(&mut self,r:Self) { self.x-=r.x; self.y-=r.y; self.z-=r.z; } }
impl MulAssign      for U64Vec3 { #[inline] fn mul_assign(&mut self,r:Self) { self.x*=r.x; self.y*=r.y; self.z*=r.z; } }
impl DivAssign      for U64Vec3 { #[inline] fn div_assign(&mut self,r:Self) { self.x/=r.x; self.y/=r.y; self.z/=r.z; } }
impl RemAssign      for U64Vec3 { #[inline] fn rem_assign(&mut self,r:Self) { self.x%=r.x; self.y%=r.y; self.z%=r.z; } }
impl MulAssign<u64> for U64Vec3 { #[inline] fn mul_assign(&mut self,s:u64) { self.x*=s; self.y*=s; self.z*=s; } }
impl DivAssign<u64> for U64Vec3 { #[inline] fn div_assign(&mut self,s:u64) { self.x/=s; self.y/=s; self.z/=s; } }

impl BitAnd for U64Vec3 { type Output=Self; #[inline] fn bitand(self,r:Self)->Self { Self::new(self.x&r.x, self.y&r.y, self.z&r.z) } }
impl BitOr  for U64Vec3 { type Output=Self; #[inline] fn bitor (self,r:Self)->Self { Self::new(self.x|r.x, self.y|r.y, self.z|r.z) } }
impl BitXor for U64Vec3 { type Output=Self; #[inline] fn bitxor(self,r:Self)->Self { Self::new(self.x^r.x, self.y^r.y, self.z^r.z) } }
impl BitAndAssign for U64Vec3 { #[inline] fn bitand_assign(&mut self,r:Self) { *self = *self & r; } }
impl BitOrAssign  for U64Vec3 { #[inline] fn bitor_assign (&mut self,r:Self) { *self = *self | r; } }
impl BitXorAssign for U64Vec3 { #[inline] fn bitxor_assign(&mut self,r:Self) { *self = *self ^ r; } }

impl Shl<u32> for U64Vec3 { type Output=Self; #[inline] fn shl(self,s:u32)->Self { Self::new(self.x<<s, self.y<<s, self.z<<s) } }
impl Shr<u32> for U64Vec3 { type Output=Self; #[inline] fn shr(self,s:u32)->Self { Self::new(self.x>>s, self.y>>s, self.z>>s) } }
impl ShlAssign<u32> for U64Vec3 { #[inline] fn shl_assign(&mut self,s:u32) { self.x<<=s; self.y<<=s; self.z<<=s; } }
impl ShrAssign<u32> for U64Vec3 { #[inline] fn shr_assign(&mut self,s:u32) { self.x>>=s; self.y>>=s; self.z>>=s; } }

impl Index<usize> for U64Vec3 {
    type Output = u64;
    #[inline] fn index(&self, i: usize) -> &u64 {
        match i { 0=>&self.x, 1=>&self.y, 2=>&self.z, _=>panic!("U64Vec3 index {i} out of bounds") }
    }
}
impl IndexMut<usize> for U64Vec3 {
    #[inline] fn index_mut(&mut self, i: usize) -> &mut u64 {
        match i { 0=>&mut self.x, 1=>&mut self.y, 2=>&mut self.z, _=>panic!("U64Vec3 index {i} out of bounds") }
    }
}

impl From<[u64;3]> for U64Vec3 { #[inline] fn from(a:[u64;3])->Self { Self::from_array(a) } }
impl From<U64Vec3> for [u64;3] { #[inline] fn from(v:U64Vec3)->[u64;3] { v.to_array() } }
impl From<(u64,u64,u64)> for U64Vec3 { #[inline] fn from(t:(u64,u64,u64))->Self { Self::new(t.0,t.1,t.2) } }
impl From<(U64Vec2,u64)> for U64Vec3 { #[inline] fn from((v,z):(U64Vec2,u64))->Self { Self::new(v.x,v.y,z) } }

impl fmt::Debug for U64Vec3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "U64Vec3({}, {}, {})", self.x, self.y, self.z) }
}
impl fmt::Display for U64Vec3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "[{}, {}, {}]", self.x, self.y, self.z) }
  }
