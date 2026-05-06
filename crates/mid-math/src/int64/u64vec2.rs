// crates/mid-math/src/int64/u64vec2.rs
//! 2D unsigned 64-bit integer vector. 16 bytes, align 8. Always scalar.

use core::fmt;
use core::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign,
    Div, DivAssign, Index, IndexMut, Mul, MulAssign, Not,
    Rem, RemAssign, Shl, ShlAssign, Shr, ShrAssign, Sub, SubAssign,
};
use crate::{BVec2, U64Vec3, UVec2, IVec2, DVec2, Vec2};

/// 2D unsigned 64-bit integer vector. 16 bytes, align 8.
///
/// Used for texture atlas coordinates at extreme resolution, file offsets,
/// and any 2D domain where u32 overflows.
///
/// **C interop:** use [`CU64Vec2`][crate::ffi::types::CU64Vec2] at the FFI boundary.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(C)]
pub struct U64Vec2 {
    pub x: u64,
    pub y: u64,
}

impl U64Vec2 {
    pub const ZERO: Self = Self::splat(0);
    pub const ONE:  Self = Self::splat(1);
    pub const MIN:  Self = Self::splat(u64::MIN);
    pub const MAX:  Self = Self::splat(u64::MAX);
    pub const X:    Self = Self::new(1, 0);
    pub const Y:    Self = Self::new(0, 1);

    #[inline(always)] pub const fn new(x: u64, y: u64) -> Self { Self { x, y } }
    #[inline(always)] pub const fn splat(v: u64) -> Self { Self { x: v, y: v } }
    #[inline(always)] pub const fn from_array(a: [u64; 2]) -> Self { Self::new(a[0], a[1]) }
    #[inline(always)] pub const fn to_array(self) -> [u64; 2] { [self.x, self.y] }
    #[inline(always)] pub const fn extend(self, z: u64) -> U64Vec3 { U64Vec3::new(self.x, self.y, z) }

    #[inline]
    pub fn select(mask: BVec2, if_true: Self, if_false: Self) -> Self {
        Self::new(
            if mask.x { if_true.x } else { if_false.x },
            if mask.y { if_true.y } else { if_false.y },
        )
    }

    #[inline] pub fn dot(self, rhs: Self) -> u64 { self.x * rhs.x + self.y * rhs.y }
    #[inline] pub fn length_sq(self) -> u64 { self.dot(self) }
    #[inline] pub fn min(self, r: Self) -> Self { Self::new(self.x.min(r.x), self.y.min(r.y)) }
    #[inline] pub fn max(self, r: Self) -> Self { Self::new(self.x.max(r.x), self.y.max(r.y)) }
    #[inline] pub fn clamp(self, lo: Self, hi: Self) -> Self { self.max(lo).min(hi) }
    #[inline] pub fn min_element(self) -> u64 { self.x.min(self.y) }
    #[inline] pub fn max_element(self) -> u64 { self.x.max(self.y) }
    #[inline] pub fn element_sum(self) -> u64 { self.x + self.y }
    #[inline] pub fn element_product(self) -> u64 { self.x * self.y }

    #[inline] pub fn cmpeq(self, r: Self) -> BVec2 { BVec2::new(self.x==r.x, self.y==r.y) }
    #[inline] pub fn cmpne(self, r: Self) -> BVec2 { BVec2::new(self.x!=r.x, self.y!=r.y) }
    #[inline] pub fn cmpge(self, r: Self) -> BVec2 { BVec2::new(self.x>=r.x, self.y>=r.y) }
    #[inline] pub fn cmpgt(self, r: Self) -> BVec2 { BVec2::new(self.x>r.x,  self.y>r.y)  }
    #[inline] pub fn cmple(self, r: Self) -> BVec2 { BVec2::new(self.x<=r.x, self.y<=r.y) }
    #[inline] pub fn cmplt(self, r: Self) -> BVec2 { BVec2::new(self.x<r.x,  self.y<r.y)  }

    #[inline] pub fn wrapping_add(self, r: Self) -> Self { Self::new(self.x.wrapping_add(r.x), self.y.wrapping_add(r.y)) }
    #[inline] pub fn wrapping_sub(self, r: Self) -> Self { Self::new(self.x.wrapping_sub(r.x), self.y.wrapping_sub(r.y)) }
    #[inline] pub fn wrapping_mul(self, r: Self) -> Self { Self::new(self.x.wrapping_mul(r.x), self.y.wrapping_mul(r.y)) }
    #[inline] pub fn saturating_add(self, r: Self) -> Self { Self::new(self.x.saturating_add(r.x), self.y.saturating_add(r.y)) }
    #[inline] pub fn saturating_sub(self, r: Self) -> Self { Self::new(self.x.saturating_sub(r.x), self.y.saturating_sub(r.y)) }
    #[inline] pub fn checked_add(self, r: Self) -> Option<Self> { Some(Self::new(self.x.checked_add(r.x)?, self.y.checked_add(r.y)?)) }
    #[inline] pub fn checked_sub(self, r: Self) -> Option<Self> { Some(Self::new(self.x.checked_sub(r.x)?, self.y.checked_sub(r.y)?)) }
    #[inline] pub fn checked_mul(self, r: Self) -> Option<Self> { Some(Self::new(self.x.checked_mul(r.x)?, self.y.checked_mul(r.y)?)) }

    #[inline] pub fn as_vec2(self)   -> Vec2  { Vec2::new(self.x as f32, self.y as f32) }
    #[inline] pub fn as_dvec2(self)  -> DVec2 { DVec2::new(self.x as f64, self.y as f64) }
    #[inline] pub fn as_ivec2(self)  -> IVec2 { IVec2::new(self.x as i32, self.y as i32) }
    #[inline] pub fn as_uvec2(self)  -> UVec2 { UVec2::new(self.x as u32, self.y as u32) }
    #[inline] pub fn as_i64vec2(self) -> crate::I64Vec2 { crate::I64Vec2::new(self.x as i64, self.y as i64) }
}

impl Add  for U64Vec2 { type Output=Self; #[inline] fn add(self,r:Self)->Self { Self::new(self.x+r.x, self.y+r.y) } }
impl Sub  for U64Vec2 { type Output=Self; #[inline] fn sub(self,r:Self)->Self { Self::new(self.x-r.x, self.y-r.y) } }
impl Mul  for U64Vec2 { type Output=Self; #[inline] fn mul(self,r:Self)->Self { Self::new(self.x*r.x, self.y*r.y) } }
impl Div  for U64Vec2 { type Output=Self; #[inline] fn div(self,r:Self)->Self { Self::new(self.x/r.x, self.y/r.y) } }
impl Rem  for U64Vec2 { type Output=Self; #[inline] fn rem(self,r:Self)->Self { Self::new(self.x%r.x, self.y%r.y) } }
impl Not  for U64Vec2 { type Output=Self; #[inline] fn not(self)->Self { Self::new(!self.x, !self.y) } }

impl Mul<u64> for U64Vec2 { type Output=Self; #[inline] fn mul(self,s:u64)->Self { Self::new(self.x*s, self.y*s) } }
impl Div<u64> for U64Vec2 { type Output=Self; #[inline] fn div(self,s:u64)->Self { Self::new(self.x/s, self.y/s) } }
impl Mul<U64Vec2> for u64  { type Output=U64Vec2; #[inline] fn mul(self,v:U64Vec2)->U64Vec2 { U64Vec2::new(self*v.x, self*v.y) } }

impl AddAssign      for U64Vec2 { #[inline] fn add_assign(&mut self,r:Self) { self.x+=r.x; self.y+=r.y; } }
impl SubAssign      for U64Vec2 { #[inline] fn sub_assign(&mut self,r:Self) { self.x-=r.x; self.y-=r.y; } }
impl MulAssign      for U64Vec2 { #[inline] fn mul_assign(&mut self,r:Self) { self.x*=r.x; self.y*=r.y; } }
impl DivAssign      for U64Vec2 { #[inline] fn div_assign(&mut self,r:Self) { self.x/=r.x; self.y/=r.y; } }
impl RemAssign      for U64Vec2 { #[inline] fn rem_assign(&mut self,r:Self) { self.x%=r.x; self.y%=r.y; } }
impl MulAssign<u64> for U64Vec2 { #[inline] fn mul_assign(&mut self,s:u64) { self.x*=s; self.y*=s; } }
impl DivAssign<u64> for U64Vec2 { #[inline] fn div_assign(&mut self,s:u64) { self.x/=s; self.y/=s; } }

impl BitAnd for U64Vec2 { type Output=Self; #[inline] fn bitand(self,r:Self)->Self { Self::new(self.x&r.x, self.y&r.y) } }
impl BitOr  for U64Vec2 { type Output=Self; #[inline] fn bitor (self,r:Self)->Self { Self::new(self.x|r.x, self.y|r.y) } }
impl BitXor for U64Vec2 { type Output=Self; #[inline] fn bitxor(self,r:Self)->Self { Self::new(self.x^r.x, self.y^r.y) } }
impl BitAndAssign for U64Vec2 { #[inline] fn bitand_assign(&mut self,r:Self) { *self = *self & r; } }
impl BitOrAssign  for U64Vec2 { #[inline] fn bitor_assign (&mut self,r:Self) { *self = *self | r; } }
impl BitXorAssign for U64Vec2 { #[inline] fn bitxor_assign(&mut self,r:Self) { *self = *self ^ r; } }

impl Shl<u32> for U64Vec2 { type Output=Self; #[inline] fn shl(self,s:u32)->Self { Self::new(self.x<<s, self.y<<s) } }
impl Shr<u32> for U64Vec2 { type Output=Self; #[inline] fn shr(self,s:u32)->Self { Self::new(self.x>>s, self.y>>s) } }
impl ShlAssign<u32> for U64Vec2 { #[inline] fn shl_assign(&mut self,s:u32) { self.x<<=s; self.y<<=s; } }
impl ShrAssign<u32> for U64Vec2 { #[inline] fn shr_assign(&mut self,s:u32) { self.x>>=s; self.y>>=s; } }

impl Index<usize> for U64Vec2 {
    type Output = u64;
    #[inline] fn index(&self, i: usize) -> &u64 {
        match i { 0=>&self.x, 1=>&self.y, _=>panic!("U64Vec2 index {i} out of bounds") }
    }
}
impl IndexMut<usize> for U64Vec2 {
    #[inline] fn index_mut(&mut self, i: usize) -> &mut u64 {
        match i { 0=>&mut self.x, 1=>&mut self.y, _=>panic!("U64Vec2 index {i} out of bounds") }
    }
}

impl From<[u64;2]> for U64Vec2 { #[inline] fn from(a:[u64;2])->Self { Self::from_array(a) } }
impl From<U64Vec2> for [u64;2] { #[inline] fn from(v:U64Vec2)->[u64;2] { v.to_array() } }
impl From<(u64,u64)> for U64Vec2 { #[inline] fn from(t:(u64,u64))->Self { Self::new(t.0,t.1) } }

impl fmt::Debug for U64Vec2 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "U64Vec2({}, {})", self.x, self.y) }
}
impl fmt::Display for U64Vec2 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "[{}, {}]", self.x, self.y) }
            }
