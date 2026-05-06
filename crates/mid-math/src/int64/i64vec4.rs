// crates/mid-math/src/int64/i64vec4.rs
//! 4D signed 64-bit integer vector. 32 bytes, align 8. Always scalar.

use core::fmt;
use core::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign,
    Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Not,
    Rem, RemAssign, Shl, ShlAssign, Shr, ShrAssign, Sub, SubAssign,
};
use crate::{BVec4, I64Vec2, I64Vec3, IVec4, UVec4, DVec4, Vec4};

/// 4D signed 64-bit integer vector. 32 bytes, align 8.
///
/// Used for packed large-coordinate data, bone indices with high entity counts,
/// and 4-component integer parameters requiring i64 range.
///
/// **C interop:** use [`CI64Vec4`][crate::ffi::types::CI64Vec4] at the FFI boundary.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(C)]
pub struct I64Vec4 {
    pub x: i64,
    pub y: i64,
    pub z: i64,
    pub w: i64,
}

impl I64Vec4 {
    pub const ZERO:    Self = Self::splat(0);
    pub const ONE:     Self = Self::splat(1);
    pub const NEG_ONE: Self = Self::splat(-1);
    pub const MIN:     Self = Self::splat(i64::MIN);
    pub const MAX:     Self = Self::splat(i64::MAX);
    pub const X:       Self = Self::new(1, 0, 0, 0);
    pub const Y:       Self = Self::new(0, 1, 0, 0);
    pub const Z:       Self = Self::new(0, 0, 1, 0);
    pub const W:       Self = Self::new(0, 0, 0, 1);
    pub const NEG_X:   Self = Self::new(-1,  0,  0,  0);
    pub const NEG_Y:   Self = Self::new( 0, -1,  0,  0);
    pub const NEG_Z:   Self = Self::new( 0,  0, -1,  0);
    pub const NEG_W:   Self = Self::new( 0,  0,  0, -1);

    #[inline(always)] pub const fn new(x: i64, y: i64, z: i64, w: i64) -> Self { Self { x, y, z, w } }
    #[inline(always)] pub const fn splat(v: i64) -> Self { Self { x: v, y: v, z: v, w: v } }
    #[inline(always)] pub const fn from_array(a: [i64; 4]) -> Self { Self::new(a[0], a[1], a[2], a[3]) }
    #[inline(always)] pub const fn to_array(self) -> [i64; 4] { [self.x, self.y, self.z, self.w] }
    #[inline(always)] pub const fn truncate(self) -> I64Vec3 { I64Vec3::new(self.x, self.y, self.z) }
    #[inline(always)] pub const fn xy(self) -> I64Vec2 { I64Vec2::new(self.x, self.y) }

    #[inline]
    pub fn select(mask: BVec4, if_true: Self, if_false: Self) -> Self {
        Self::new(
            if mask.x { if_true.x } else { if_false.x },
            if mask.y { if_true.y } else { if_false.y },
            if mask.z { if_true.z } else { if_false.z },
            if mask.w { if_true.w } else { if_false.w },
        )
    }

    #[inline] pub fn dot(self, rhs: Self) -> i64 {
        self.x*rhs.x + self.y*rhs.y + self.z*rhs.z + self.w*rhs.w
    }
    #[inline] pub fn length_sq(self) -> i64 { self.dot(self) }
    #[inline] pub fn distance_sq(self, rhs: Self) -> i64 { (self - rhs).length_sq() }
    #[inline] pub fn abs(self) -> Self { Self::new(self.x.abs(), self.y.abs(), self.z.abs(), self.w.abs()) }
    #[inline] pub fn signum(self) -> Self { Self::new(self.x.signum(), self.y.signum(), self.z.signum(), self.w.signum()) }

    #[inline] pub fn min(self, r: Self) -> Self { Self::new(self.x.min(r.x), self.y.min(r.y), self.z.min(r.z), self.w.min(r.w)) }
    #[inline] pub fn max(self, r: Self) -> Self { Self::new(self.x.max(r.x), self.y.max(r.y), self.z.max(r.z), self.w.max(r.w)) }
    #[inline] pub fn clamp(self, lo: Self, hi: Self) -> Self { self.max(lo).min(hi) }
    #[inline] pub fn min_element(self) -> i64 { self.x.min(self.y).min(self.z).min(self.w) }
    #[inline] pub fn max_element(self) -> i64 { self.x.max(self.y).max(self.z).max(self.w) }
    #[inline] pub fn element_sum(self) -> i64 { self.x + self.y + self.z + self.w }
    #[inline] pub fn element_product(self) -> i64 { self.x * self.y * self.z * self.w }

    #[inline] pub fn cmpeq(self, r: Self) -> BVec4 { BVec4::new(self.x==r.x, self.y==r.y, self.z==r.z, self.w==r.w) }
    #[inline] pub fn cmpne(self, r: Self) -> BVec4 { BVec4::new(self.x!=r.x, self.y!=r.y, self.z!=r.z, self.w!=r.w) }
    #[inline] pub fn cmpge(self, r: Self) -> BVec4 { BVec4::new(self.x>=r.x, self.y>=r.y, self.z>=r.z, self.w>=r.w) }
    #[inline] pub fn cmpgt(self, r: Self) -> BVec4 { BVec4::new(self.x>r.x,  self.y>r.y,  self.z>r.z,  self.w>r.w)  }
    #[inline] pub fn cmple(self, r: Self) -> BVec4 { BVec4::new(self.x<=r.x, self.y<=r.y, self.z<=r.z, self.w<=r.w) }
    #[inline] pub fn cmplt(self, r: Self) -> BVec4 { BVec4::new(self.x<r.x,  self.y<r.y,  self.z<r.z,  self.w<r.w)  }

    #[inline] pub fn manhattan_distance(self, rhs: Self) -> u64 {
        self.x.abs_diff(rhs.x) + self.y.abs_diff(rhs.y) + self.z.abs_diff(rhs.z) + self.w.abs_diff(rhs.w)
    }
    #[inline] pub fn checked_manhattan_distance(self, rhs: Self) -> Option<u64> {
        self.x.abs_diff(rhs.x)
            .checked_add(self.y.abs_diff(rhs.y))?
            .checked_add(self.z.abs_diff(rhs.z))?
            .checked_add(self.w.abs_diff(rhs.w))
    }

    #[inline] pub fn wrapping_add(self, r: Self) -> Self { Self::new(self.x.wrapping_add(r.x), self.y.wrapping_add(r.y), self.z.wrapping_add(r.z), self.w.wrapping_add(r.w)) }
    #[inline] pub fn wrapping_sub(self, r: Self) -> Self { Self::new(self.x.wrapping_sub(r.x), self.y.wrapping_sub(r.y), self.z.wrapping_sub(r.z), self.w.wrapping_sub(r.w)) }
    #[inline] pub fn wrapping_mul(self, r: Self) -> Self { Self::new(self.x.wrapping_mul(r.x), self.y.wrapping_mul(r.y), self.z.wrapping_mul(r.z), self.w.wrapping_mul(r.w)) }
    #[inline] pub fn saturating_add(self, r: Self) -> Self { Self::new(self.x.saturating_add(r.x), self.y.saturating_add(r.y), self.z.saturating_add(r.z), self.w.saturating_add(r.w)) }
    #[inline] pub fn saturating_sub(self, r: Self) -> Self { Self::new(self.x.saturating_sub(r.x), self.y.saturating_sub(r.y), self.z.saturating_sub(r.z), self.w.saturating_sub(r.w)) }
    #[inline] pub fn saturating_mul(self, r: Self) -> Self { Self::new(self.x.saturating_mul(r.x), self.y.saturating_mul(r.y), self.z.saturating_mul(r.z), self.w.saturating_mul(r.w)) }
    #[inline] pub fn checked_add(self, r: Self) -> Option<Self> { Some(Self::new(self.x.checked_add(r.x)?, self.y.checked_add(r.y)?, self.z.checked_add(r.z)?, self.w.checked_add(r.w)?)) }
    #[inline] pub fn checked_sub(self, r: Self) -> Option<Self> { Some(Self::new(self.x.checked_sub(r.x)?, self.y.checked_sub(r.y)?, self.z.checked_sub(r.z)?, self.w.checked_sub(r.w)?)) }
    #[inline] pub fn checked_mul(self, r: Self) -> Option<Self> { Some(Self::new(self.x.checked_mul(r.x)?, self.y.checked_mul(r.y)?, self.z.checked_mul(r.z)?, self.w.checked_mul(r.w)?)) }

    #[inline] pub fn as_vec4(self)  -> Vec4  { Vec4::new(self.x as f32, self.y as f32, self.z as f32, self.w as f32) }
    #[inline] pub fn as_dvec4(self) -> DVec4 { DVec4::new(self.x as f64, self.y as f64, self.z as f64, self.w as f64) }
    #[inline] pub fn as_ivec4(self) -> IVec4 { IVec4::new(self.x as i32, self.y as i32, self.z as i32, self.w as i32) }
    #[inline] pub fn as_uvec4(self) -> UVec4 { UVec4::new(self.x as u32, self.y as u32, self.z as u32, self.w as u32) }
    #[inline] pub fn as_u64vec4(self) -> crate::U64Vec4 { crate::U64Vec4::new(self.x as u64, self.y as u64, self.z as u64, self.w as u64) }
}

impl Add  for I64Vec4 { type Output=Self; #[inline] fn add(self,r:Self)->Self { Self::new(self.x+r.x, self.y+r.y, self.z+r.z, self.w+r.w) } }
impl Sub  for I64Vec4 { type Output=Self; #[inline] fn sub(self,r:Self)->Self { Self::new(self.x-r.x, self.y-r.y, self.z-r.z, self.w-r.w) } }
impl Mul  for I64Vec4 { type Output=Self; #[inline] fn mul(self,r:Self)->Self { Self::new(self.x*r.x, self.y*r.y, self.z*r.z, self.w*r.w) } }
impl Div  for I64Vec4 { type Output=Self; #[inline] fn div(self,r:Self)->Self { Self::new(self.x/r.x, self.y/r.y, self.z/r.z, self.w/r.w) } }
impl Rem  for I64Vec4 { type Output=Self; #[inline] fn rem(self,r:Self)->Self { Self::new(self.x%r.x, self.y%r.y, self.z%r.z, self.w%r.w) } }
impl Neg  for I64Vec4 { type Output=Self; #[inline] fn neg(self)->Self { Self::new(-self.x, -self.y, -self.z, -self.w) } }
impl Not  for I64Vec4 { type Output=Self; #[inline] fn not(self)->Self { Self::new(!self.x, !self.y, !self.z, !self.w) } }

impl Mul<i64> for I64Vec4 { type Output=Self; #[inline] fn mul(self,s:i64)->Self { Self::new(self.x*s, self.y*s, self.z*s, self.w*s) } }
impl Div<i64> for I64Vec4 { type Output=Self; #[inline] fn div(self,s:i64)->Self { Self::new(self.x/s, self.y/s, self.z/s, self.w/s) } }
impl Add<i64> for I64Vec4 { type Output=Self; #[inline] fn add(self,s:i64)->Self { Self::new(self.x+s, self.y+s, self.z+s, self.w+s) } }
impl Sub<i64> for I64Vec4 { type Output=Self; #[inline] fn sub(self,s:i64)->Self { Self::new(self.x-s, self.y-s, self.z-s, self.w-s) } }
impl Mul<I64Vec4> for i64 { type Output=I64Vec4; #[inline] fn mul(self,v:I64Vec4)->I64Vec4 { I64Vec4::new(self*v.x, self*v.y, self*v.z, self*v.w) } }

impl AddAssign      for I64Vec4 { #[inline] fn add_assign(&mut self,r:Self) { self.x+=r.x; self.y+=r.y; self.z+=r.z; self.w+=r.w; } }
impl SubAssign      for I64Vec4 { #[inline] fn sub_assign(&mut self,r:Self) { self.x-=r.x; self.y-=r.y; self.z-=r.z; self.w-=r.w; } }
impl MulAssign      for I64Vec4 { #[inline] fn mul_assign(&mut self,r:Self) { self.x*=r.x; self.y*=r.y; self.z*=r.z; self.w*=r.w; } }
impl DivAssign      for I64Vec4 { #[inline] fn div_assign(&mut self,r:Self) { self.x/=r.x; self.y/=r.y; self.z/=r.z; self.w/=r.w; } }
impl RemAssign      for I64Vec4 { #[inline] fn rem_assign(&mut self,r:Self) { self.x%=r.x; self.y%=r.y; self.z%=r.z; self.w%=r.w; } }
impl MulAssign<i64> for I64Vec4 { #[inline] fn mul_assign(&mut self,s:i64) { self.x*=s; self.y*=s; self.z*=s; self.w*=s; } }
impl DivAssign<i64> for I64Vec4 { #[inline] fn div_assign(&mut self,s:i64) { self.x/=s; self.y/=s; self.z/=s; self.w/=s; } }
impl AddAssign<i64> for I64Vec4 { #[inline] fn add_assign(&mut self,s:i64) { self.x+=s; self.y+=s; self.z+=s; self.w+=s; } }
impl SubAssign<i64> for I64Vec4 { #[inline] fn sub_assign(&mut self,s:i64) { self.x-=s; self.y-=s; self.z-=s; self.w-=s; } }
impl RemAssign<i64> for I64Vec4 { #[inline] fn rem_assign(&mut self,s:i64) { self.x%=s; self.y%=s; self.z%=s; self.w%=s; } }

impl BitAnd for I64Vec4 { type Output=Self; #[inline] fn bitand(self,r:Self)->Self { Self::new(self.x&r.x, self.y&r.y, self.z&r.z, self.w&r.w) } }
impl BitOr  for I64Vec4 { type Output=Self; #[inline] fn bitor (self,r:Self)->Self { Self::new(self.x|r.x, self.y|r.y, self.z|r.z, self.w|r.w) } }
impl BitXor for I64Vec4 { type Output=Self; #[inline] fn bitxor(self,r:Self)->Self { Self::new(self.x^r.x, self.y^r.y, self.z^r.z, self.w^r.w) } }
impl BitAndAssign for I64Vec4 { #[inline] fn bitand_assign(&mut self,r:Self) { *self = *self & r; } }
impl BitOrAssign  for I64Vec4 { #[inline] fn bitor_assign (&mut self,r:Self) { *self = *self | r; } }
impl BitXorAssign for I64Vec4 { #[inline] fn bitxor_assign(&mut self,r:Self) { *self = *self ^ r; } }

impl Shl<u32> for I64Vec4 { type Output=Self; #[inline] fn shl(self,s:u32)->Self { Self::new(self.x<<s, self.y<<s, self.z<<s, self.w<<s) } }
impl Shr<u32> for I64Vec4 { type Output=Self; #[inline] fn shr(self,s:u32)->Self { Self::new(self.x>>s, self.y>>s, self.z>>s, self.w>>s) } }
impl Shl<i32> for I64Vec4 { type Output=Self; #[inline] fn shl(self,s:i32)->Self { Self::new(self.x<<s, self.y<<s, self.z<<s, self.w<<s) } }
impl Shr<i32> for I64Vec4 { type Output=Self; #[inline] fn shr(self,s:i32)->Self { Self::new(self.x>>s, self.y>>s, self.z>>s, self.w>>s) } }
impl ShlAssign<u32> for I64Vec4 { #[inline] fn shl_assign(&mut self,s:u32) { self.x<<=s; self.y<<=s; self.z<<=s; self.w<<=s; } }
impl ShrAssign<u32> for I64Vec4 { #[inline] fn shr_assign(&mut self,s:u32) { self.x>>=s; self.y>>=s; self.z>>=s; self.w>>=s; } }
impl ShlAssign<i32> for I64Vec4 { #[inline] fn shl_assign(&mut self,s:i32) { self.x<<=s; self.y<<=s; self.z<<=s; self.w<<=s; } }
impl ShrAssign<i32> for I64Vec4 { #[inline] fn shr_assign(&mut self,s:i32) { self.x>>=s; self.y>>=s; self.z>>=s; self.w>>=s; } }

impl Index<usize> for I64Vec4 {
    type Output = i64;
    #[inline] fn index(&self, i: usize) -> &i64 {
        match i { 0=>&self.x, 1=>&self.y, 2=>&self.z, 3=>&self.w, _=>panic!("I64Vec4 index {i} out of bounds") }
    }
}
impl IndexMut<usize> for I64Vec4 {
    #[inline] fn index_mut(&mut self, i: usize) -> &mut i64 {
        match i { 0=>&mut self.x, 1=>&mut self.y, 2=>&mut self.z, 3=>&mut self.w, _=>panic!("I64Vec4 index {i} out of bounds") }
    }
}

impl From<[i64;4]> for I64Vec4 { #[inline] fn from(a:[i64;4])->Self { Self::from_array(a) } }
impl From<I64Vec4> for [i64;4] { #[inline] fn from(v:I64Vec4)->[i64;4] { v.to_array() } }
impl From<(i64,i64,i64,i64)> for I64Vec4 { #[inline] fn from(t:(i64,i64,i64,i64))->Self { Self::new(t.0,t.1,t.2,t.3) } }
impl From<(I64Vec3,i64)> for I64Vec4 { #[inline] fn from((v,w):(I64Vec3,i64))->Self { Self::new(v.x,v.y,v.z,w) } }
impl From<(I64Vec2,i64,i64)> for I64Vec4 { #[inline] fn from((v,z,w):(I64Vec2,i64,i64))->Self { Self::new(v.x,v.y,z,w) } }
impl From<(I64Vec2,I64Vec2)> for I64Vec4 { #[inline] fn from((a,b):(I64Vec2,I64Vec2))->Self { Self::new(a.x,a.y,b.x,b.y) } }

impl fmt::Debug for I64Vec4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "I64Vec4({}, {}, {}, {})", self.x, self.y, self.z, self.w) }
}
impl fmt::Display for I64Vec4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "[{}, {}, {}, {}]", self.x, self.y, self.z, self.w) }
  }
