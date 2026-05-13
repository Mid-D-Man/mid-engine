// crates/mid-math/src/int32/uvec4.rs
//! 4D unsigned-integer vector. 16 bytes, align 4. Always scalar.

use core::fmt;
use core::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign,
    Div, DivAssign, Index, IndexMut, Mul, MulAssign, Not,
    Rem, Shl, ShlAssign, Shr, ShrAssign, Sub, SubAssign,
};
use crate::{BVec4, IVec4, UVec2, UVec3, Vec4};

/// 4D unsigned-integer vector. 16 bytes, align 4.
///
/// Used for packed entity/component IDs, RGBA8 color (packed), and
/// any 4-component unsigned integer parameter.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(C)]
pub struct UVec4 {
    pub x: u32,
    pub y: u32,
    pub z: u32,
    pub w: u32,
}

impl UVec4 {
    pub const ZERO: Self = Self::splat(0);
    pub const ONE:  Self = Self::splat(1);
    pub const MIN:  Self = Self::splat(u32::MIN);
    pub const MAX:  Self = Self::splat(u32::MAX);
    pub const X:    Self = Self::new(1, 0, 0, 0);
    pub const Y:    Self = Self::new(0, 1, 0, 0);
    pub const Z:    Self = Self::new(0, 0, 1, 0);
    pub const W:    Self = Self::new(0, 0, 0, 1);

    #[inline(always)] pub const fn new(x: u32, y: u32, z: u32, w: u32) -> Self { Self { x, y, z, w } }
    #[inline(always)] pub const fn splat(v: u32) -> Self { Self { x: v, y: v, z: v, w: v } }
    #[inline(always)] pub const fn from_array(a: [u32; 4]) -> Self { Self::new(a[0], a[1], a[2], a[3]) }
    #[inline(always)] pub const fn to_array(self) -> [u32; 4] { [self.x, self.y, self.z, self.w] }
    #[inline(always)] pub const fn truncate(self) -> UVec3 { UVec3::new(self.x, self.y, self.z) }

    #[inline]
    pub fn select(mask: BVec4, if_true: Self, if_false: Self) -> Self {
        Self::new(
            if mask.x { if_true.x } else { if_false.x },
            if mask.y { if_true.y } else { if_false.y },
            if mask.z { if_true.z } else { if_false.z },
            if mask.w { if_true.w } else { if_false.w },
        )
    }

    #[inline] pub fn dot(self, r: Self) -> u32 { self.x*r.x + self.y*r.y + self.z*r.z + self.w*r.w }
    #[inline] pub fn length_sq(self) -> u32 { self.dot(self) }
    #[inline] pub fn min(self, r: Self) -> Self { Self::new(self.x.min(r.x), self.y.min(r.y), self.z.min(r.z), self.w.min(r.w)) }
    #[inline] pub fn max(self, r: Self) -> Self { Self::new(self.x.max(r.x), self.y.max(r.y), self.z.max(r.z), self.w.max(r.w)) }
    #[inline] pub fn clamp(self, lo: Self, hi: Self) -> Self { self.max(lo).min(hi) }
    #[inline] pub fn min_element(self) -> u32 { self.x.min(self.y).min(self.z).min(self.w) }
    #[inline] pub fn max_element(self) -> u32 { self.x.max(self.y).max(self.z).max(self.w) }
    #[inline] pub fn element_sum(self) -> u32 { self.x + self.y + self.z + self.w }
    #[inline] pub fn element_product(self) -> u32 { self.x * self.y * self.z * self.w }

    #[inline] pub fn cmpeq(self, r: Self) -> BVec4 { BVec4::new(self.x==r.x, self.y==r.y, self.z==r.z, self.w==r.w) }
    #[inline] pub fn cmpne(self, r: Self) -> BVec4 { BVec4::new(self.x!=r.x, self.y!=r.y, self.z!=r.z, self.w!=r.w) }
    #[inline] pub fn cmpge(self, r: Self) -> BVec4 { BVec4::new(self.x>=r.x, self.y>=r.y, self.z>=r.z, self.w>=r.w) }
    #[inline] pub fn cmpgt(self, r: Self) -> BVec4 { BVec4::new(self.x>r.x,  self.y>r.y,  self.z>r.z,  self.w>r.w)  }
    #[inline] pub fn cmple(self, r: Self) -> BVec4 { BVec4::new(self.x<=r.x, self.y<=r.y, self.z<=r.z, self.w<=r.w) }
    #[inline] pub fn cmplt(self, r: Self) -> BVec4 { BVec4::new(self.x<r.x,  self.y<r.y,  self.z<r.z,  self.w<r.w)  }

    #[inline] pub fn wrapping_add(self, r: Self) -> Self {
        Self::new(self.x.wrapping_add(r.x), self.y.wrapping_add(r.y), self.z.wrapping_add(r.z), self.w.wrapping_add(r.w))
    }
    #[inline] pub fn wrapping_sub(self, r: Self) -> Self {
        Self::new(self.x.wrapping_sub(r.x), self.y.wrapping_sub(r.y), self.z.wrapping_sub(r.z), self.w.wrapping_sub(r.w))
    }
    #[inline] pub fn saturating_add(self, r: Self) -> Self {
        Self::new(self.x.saturating_add(r.x), self.y.saturating_add(r.y), self.z.saturating_add(r.z), self.w.saturating_add(r.w))
    }
    #[inline] pub fn saturating_sub(self, r: Self) -> Self {
        Self::new(self.x.saturating_sub(r.x), self.y.saturating_sub(r.y), self.z.saturating_sub(r.z), self.w.saturating_sub(r.w))
    }

    #[inline] pub fn as_vec4(self) -> Vec4 { Vec4::new(self.x as f32, self.y as f32, self.z as f32, self.w as f32) }
    #[inline] pub fn as_dvec4(self) -> crate::DVec4 { crate::DVec4::new(self.x as f64, self.y as f64, self.z as f64, self.w as f64) }
    #[inline] pub fn as_ivec4(self) -> IVec4 { IVec4::new(self.x as i32, self.y as i32, self.z as i32, self.w as i32) }
}

impl Add  for UVec4 { type Output=Self; #[inline] fn add(self,r:Self)->Self { Self::new(self.x+r.x, self.y+r.y, self.z+r.z, self.w+r.w) } }
impl Sub  for UVec4 { type Output=Self; #[inline] fn sub(self,r:Self)->Self { Self::new(self.x-r.x, self.y-r.y, self.z-r.z, self.w-r.w) } }
impl Mul  for UVec4 { type Output=Self; #[inline] fn mul(self,r:Self)->Self { Self::new(self.x*r.x, self.y*r.y, self.z*r.z, self.w*r.w) } }
impl Div  for UVec4 { type Output=Self; #[inline] fn div(self,r:Self)->Self { Self::new(self.x/r.x, self.y/r.y, self.z/r.z, self.w/r.w) } }
impl Rem  for UVec4 { type Output=Self; #[inline] fn rem(self,r:Self)->Self { Self::new(self.x%r.x, self.y%r.y, self.z%r.z, self.w%r.w) } }
impl Not  for UVec4 { type Output=Self; #[inline] fn not(self)->Self { Self::new(!self.x, !self.y, !self.z, !self.w) } }

impl Mul<u32> for UVec4 { type Output=Self; #[inline] fn mul(self,s:u32)->Self { Self::new(self.x*s, self.y*s, self.z*s, self.w*s) } }
impl Div<u32> for UVec4 { type Output=Self; #[inline] fn div(self,s:u32)->Self { Self::new(self.x/s, self.y/s, self.z/s, self.w/s) } }
impl Mul<UVec4> for u32 { type Output=UVec4; #[inline] fn mul(self,v:UVec4)->UVec4 { UVec4::new(self*v.x, self*v.y, self*v.z, self*v.w) } }

impl AddAssign     for UVec4 { #[inline] fn add_assign(&mut self,r:Self) { self.x+=r.x; self.y+=r.y; self.z+=r.z; self.w+=r.w; } }
impl SubAssign     for UVec4 { #[inline] fn sub_assign(&mut self,r:Self) { self.x-=r.x; self.y-=r.y; self.z-=r.z; self.w-=r.w; } }
impl MulAssign     for UVec4 { #[inline] fn mul_assign(&mut self,r:Self) { self.x*=r.x; self.y*=r.y; self.z*=r.z; self.w*=r.w; } }
impl DivAssign     for UVec4 { #[inline] fn div_assign(&mut self,r:Self) { self.x/=r.x; self.y/=r.y; self.z/=r.z; self.w/=r.w; } }
impl MulAssign<u32> for UVec4 { #[inline] fn mul_assign(&mut self,s:u32) { self.x*=s; self.y*=s; self.z*=s; self.w*=s; } }
impl DivAssign<u32> for UVec4 { #[inline] fn div_assign(&mut self,s:u32) { self.x/=s; self.y/=s; self.z/=s; self.w/=s; } }

impl BitAnd for UVec4 { type Output=Self; #[inline] fn bitand(self,r:Self)->Self { Self::new(self.x&r.x, self.y&r.y, self.z&r.z, self.w&r.w) } }
impl BitOr  for UVec4 { type Output=Self; #[inline] fn bitor (self,r:Self)->Self { Self::new(self.x|r.x, self.y|r.y, self.z|r.z, self.w|r.w) } }
impl BitXor for UVec4 { type Output=Self; #[inline] fn bitxor(self,r:Self)->Self { Self::new(self.x^r.x, self.y^r.y, self.z^r.z, self.w^r.w) } }
impl BitAndAssign for UVec4 { #[inline] fn bitand_assign(&mut self,r:Self) { *self = *self & r; } }
impl BitOrAssign  for UVec4 { #[inline] fn bitor_assign (&mut self,r:Self) { *self = *self | r; } }
impl BitXorAssign for UVec4 { #[inline] fn bitxor_assign(&mut self,r:Self) { *self = *self ^ r; } }
impl Shl<u32> for UVec4 { type Output=Self; #[inline] fn shl(self,s:u32)->Self { Self::new(self.x<<s, self.y<<s, self.z<<s, self.w<<s) } }
impl Shr<u32> for UVec4 { type Output=Self; #[inline] fn shr(self,s:u32)->Self { Self::new(self.x>>s, self.y>>s, self.z>>s, self.w>>s) } }
impl ShlAssign<u32> for UVec4 { #[inline] fn shl_assign(&mut self,s:u32) { self.x<<=s; self.y<<=s; self.z<<=s; self.w<<=s; } }
impl ShrAssign<u32> for UVec4 { #[inline] fn shr_assign(&mut self,s:u32) { self.x>>=s; self.y>>=s; self.z>>=s; self.w>>=s; } }

impl Index<usize> for UVec4 {
    type Output = u32;
    #[inline] fn index(&self, i: usize) -> &u32 {
        match i { 0=>&self.x, 1=>&self.y, 2=>&self.z, 3=>&self.w, _=>panic!("UVec4 index {i} out of bounds") }
    }
}
impl IndexMut<usize> for UVec4 {
    #[inline] fn index_mut(&mut self, i: usize) -> &mut u32 {
        match i { 0=>&mut self.x, 1=>&mut self.y, 2=>&mut self.z, 3=>&mut self.w, _=>panic!("UVec4 index {i} out of bounds") }
    }
}

impl From<[u32;4]> for UVec4 { #[inline] fn from(a:[u32;4])->Self { Self::from_array(a) } }
impl From<UVec4> for [u32;4] { #[inline] fn from(v:UVec4)->[u32;4] { v.to_array() } }
impl From<(u32,u32,u32,u32)> for UVec4 { #[inline] fn from(t:(u32,u32,u32,u32))->Self { Self::new(t.0,t.1,t.2,t.3) } }
impl From<(UVec3, u32)> for UVec4 { #[inline] fn from((v,w):(UVec3,u32))->Self { Self::new(v.x,v.y,v.z,w) } }
impl From<(UVec2, UVec2)> for UVec4 { #[inline] fn from((a,b):(UVec2,UVec2))->Self { Self::new(a.x,a.y,b.x,b.y) } }

impl fmt::Debug for UVec4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "UVec4({}, {}, {}, {})", self.x, self.y, self.z, self.w) }
}
impl fmt::Display for UVec4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "[{}, {}, {}, {}]", self.x, self.y, self.z, self.w) }
}
