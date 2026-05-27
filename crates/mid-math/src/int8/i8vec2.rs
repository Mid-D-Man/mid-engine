// crates/mid-math/src/int8/i8vec2.rs
//! 2D signed 8-bit integer vector. 2 bytes, align 1. Always scalar.

use core::fmt;
use core::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign,
    Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Not,
    Rem, RemAssign, Shl, ShlAssign, Shr, ShrAssign, Sub, SubAssign,
};
use crate::{BVec2, I8Vec3, U8Vec2};

/// 2D signed 8-bit integer vector. 2 bytes, align 1.
///
/// Used for packed vertex normals, bone weights, small offsets,
/// and any 2D domain where the range [-128, 127] is sufficient.
///
/// **Dot product returns i16** to avoid silent overflow (127²+127²=32258).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(C)]
pub struct I8Vec2 {
    pub x: i8,
    pub y: i8,
}

impl I8Vec2 {
    pub const ZERO:    Self = Self::splat(0);
    pub const ONE:     Self = Self::splat(1);
    pub const NEG_ONE: Self = Self::splat(-1);
    pub const MIN:     Self = Self::splat(i8::MIN);
    pub const MAX:     Self = Self::splat(i8::MAX);
    pub const X:       Self = Self::new(1, 0);
    pub const Y:       Self = Self::new(0, 1);
    pub const NEG_X:   Self = Self::new(-1, 0);
    pub const NEG_Y:   Self = Self::new(0, -1);

    #[inline(always)] pub const fn new(x: i8, y: i8) -> Self { Self { x, y } }
    #[inline(always)] pub const fn splat(v: i8) -> Self { Self { x: v, y: v } }
    #[inline(always)] pub const fn from_array(a: [i8; 2]) -> Self { Self::new(a[0], a[1]) }
    #[inline(always)] pub const fn to_array(self) -> [i8; 2] { [self.x, self.y] }

    #[inline(always)] pub const fn extend(self, z: i8) -> I8Vec3 { I8Vec3::new(self.x, self.y, z) }

    #[inline]
    pub fn select(mask: BVec2, if_true: Self, if_false: Self) -> Self {
        Self::new(
            if mask.x { if_true.x } else { if_false.x },
            if mask.y { if_true.y } else { if_false.y },
        )
    }

    /// Dot product, widened to i16 to prevent overflow.
    #[inline] pub fn dot(self, rhs: Self) -> i16 {
        (self.x as i16) * (rhs.x as i16) + (self.y as i16) * (rhs.y as i16)
    }

    /// Squared length, widened to i16.
    #[inline] pub fn length_sq(self) -> i16 { self.dot(self) }

    /// Squared distance, widened to i16.
    #[inline] pub fn distance_sq(self, rhs: Self) -> i16 { (self - rhs).length_sq() }

    #[inline] pub fn abs(self) -> Self { Self::new(self.x.abs(), self.y.abs()) }
    #[inline] pub fn signum(self) -> Self { Self::new(self.x.signum(), self.y.signum()) }
    #[inline] pub fn perp(self) -> Self { Self::new(-self.y, self.x) }

    /// 2D cross / perp-dot, widened to i16.
    #[inline] pub fn perp_dot(self, rhs: Self) -> i16 {
        (self.x as i16) * (rhs.y as i16) - (self.y as i16) * (rhs.x as i16)
    }

    #[inline] pub fn min(self, rhs: Self) -> Self { Self::new(self.x.min(rhs.x), self.y.min(rhs.y)) }
    #[inline] pub fn max(self, rhs: Self) -> Self { Self::new(self.x.max(rhs.x), self.y.max(rhs.y)) }
    #[inline] pub fn clamp(self, lo: Self, hi: Self) -> Self { self.max(lo).min(hi) }
    #[inline] pub fn min_element(self) -> i8 { self.x.min(self.y) }
    #[inline] pub fn max_element(self) -> i8 { self.x.max(self.y) }
    #[inline] pub fn element_sum(self) -> i8 { self.x.wrapping_add(self.y) }

    #[inline] pub fn cmpeq(self, r: Self) -> BVec2 { BVec2::new(self.x == r.x, self.y == r.y) }
    #[inline] pub fn cmpne(self, r: Self) -> BVec2 { BVec2::new(self.x != r.x, self.y != r.y) }
    #[inline] pub fn cmpge(self, r: Self) -> BVec2 { BVec2::new(self.x >= r.x, self.y >= r.y) }
    #[inline] pub fn cmpgt(self, r: Self) -> BVec2 { BVec2::new(self.x >  r.x, self.y >  r.y) }
    #[inline] pub fn cmple(self, r: Self) -> BVec2 { BVec2::new(self.x <= r.x, self.y <= r.y) }
    #[inline] pub fn cmplt(self, r: Self) -> BVec2 { BVec2::new(self.x <  r.x, self.y <  r.y) }

    #[inline] pub fn wrapping_add(self, r: Self) -> Self { Self::new(self.x.wrapping_add(r.x), self.y.wrapping_add(r.y)) }
    #[inline] pub fn wrapping_sub(self, r: Self) -> Self { Self::new(self.x.wrapping_sub(r.x), self.y.wrapping_sub(r.y)) }
    #[inline] pub fn wrapping_mul(self, r: Self) -> Self { Self::new(self.x.wrapping_mul(r.x), self.y.wrapping_mul(r.y)) }
    #[inline] pub fn wrapping_neg(self) -> Self { Self::new(self.x.wrapping_neg(), self.y.wrapping_neg()) }
    #[inline] pub fn wrapping_abs(self) -> Self { Self::new(self.x.wrapping_abs(), self.y.wrapping_abs()) }
    #[inline] pub fn saturating_add(self, r: Self) -> Self { Self::new(self.x.saturating_add(r.x), self.y.saturating_add(r.y)) }
    #[inline] pub fn saturating_sub(self, r: Self) -> Self { Self::new(self.x.saturating_sub(r.x), self.y.saturating_sub(r.y)) }
    #[inline] pub fn checked_add(self, r: Self) -> Option<Self> { Some(Self::new(self.x.checked_add(r.x)?, self.y.checked_add(r.y)?)) }
    #[inline] pub fn checked_sub(self, r: Self) -> Option<Self> { Some(Self::new(self.x.checked_sub(r.x)?, self.y.checked_sub(r.y)?)) }
    #[inline] pub fn checked_mul(self, r: Self) -> Option<Self> { Some(Self::new(self.x.checked_mul(r.x)?, self.y.checked_mul(r.y)?)) }

    // ── Casts ─────────────────────────────────────────────────────────────────
    #[inline] pub fn as_u8vec2(self)   -> U8Vec2          { U8Vec2::new(self.x as u8, self.y as u8) }
    #[inline] pub fn as_i16vec2(self)  -> crate::I16Vec2  { crate::I16Vec2::new(self.x as i16, self.y as i16) }
    #[inline] pub fn as_u16vec2(self)  -> crate::U16Vec2  { crate::U16Vec2::new(self.x as u16, self.y as u16) }
    #[inline] pub fn as_ivec2(self)    -> crate::IVec2    { crate::IVec2::new(self.x as i32, self.y as i32) }
    #[inline] pub fn as_uvec2(self)    -> crate::UVec2    { crate::UVec2::new(self.x as u32, self.y as u32) }
    #[inline] pub fn as_vec2(self)     -> crate::Vec2     { crate::Vec2::new(self.x as f32, self.y as f32) }
    #[inline] pub fn as_dvec2(self)    -> crate::DVec2    { crate::DVec2::new(self.x as f64, self.y as f64) }
}

impl Add  for I8Vec2 { type Output=Self; #[inline] fn add(self,r:Self)->Self { Self::new(self.x.wrapping_add(r.x), self.y.wrapping_add(r.y)) } }
impl Sub  for I8Vec2 { type Output=Self; #[inline] fn sub(self,r:Self)->Self { Self::new(self.x.wrapping_sub(r.x), self.y.wrapping_sub(r.y)) } }
impl Mul  for I8Vec2 { type Output=Self; #[inline] fn mul(self,r:Self)->Self { Self::new(self.x.wrapping_mul(r.x), self.y.wrapping_mul(r.y)) } }
impl Div  for I8Vec2 { type Output=Self; #[inline] fn div(self,r:Self)->Self { Self::new(self.x/r.x, self.y/r.y) } }
impl Rem  for I8Vec2 { type Output=Self; #[inline] fn rem(self,r:Self)->Self { Self::new(self.x%r.x, self.y%r.y) } }
impl Neg  for I8Vec2 { type Output=Self; #[inline] fn neg(self)->Self { Self::new(self.x.wrapping_neg(), self.y.wrapping_neg()) } }
impl Not  for I8Vec2 { type Output=Self; #[inline] fn not(self)->Self { Self::new(!self.x, !self.y) } }

impl Mul<i8> for I8Vec2 { type Output=Self; #[inline] fn mul(self,s:i8)->Self { Self::new(self.x.wrapping_mul(s), self.y.wrapping_mul(s)) } }
impl Mul<I8Vec2> for i8 { type Output=I8Vec2; #[inline] fn mul(self,v:I8Vec2)->I8Vec2 { I8Vec2::new(self.wrapping_mul(v.x), self.wrapping_mul(v.y)) } }

impl AddAssign for I8Vec2 { #[inline] fn add_assign(&mut self,r:Self) { *self = *self + r; } }
impl SubAssign for I8Vec2 { #[inline] fn sub_assign(&mut self,r:Self) { *self = *self - r; } }
impl MulAssign for I8Vec2 { #[inline] fn mul_assign(&mut self,r:Self) { *self = *self * r; } }
impl DivAssign for I8Vec2 { #[inline] fn div_assign(&mut self,r:Self) { self.x/=r.x; self.y/=r.y; } }
impl RemAssign for I8Vec2 { #[inline] fn rem_assign(&mut self,r:Self) { self.x%=r.x; self.y%=r.y; } }
impl MulAssign<i8> for I8Vec2 { #[inline] fn mul_assign(&mut self,s:i8) { *self = *self * s; } }

impl BitAnd for I8Vec2 { type Output=Self; #[inline] fn bitand(self,r:Self)->Self { Self::new(self.x&r.x, self.y&r.y) } }
impl BitOr  for I8Vec2 { type Output=Self; #[inline] fn bitor (self,r:Self)->Self { Self::new(self.x|r.x, self.y|r.y) } }
impl BitXor for I8Vec2 { type Output=Self; #[inline] fn bitxor(self,r:Self)->Self { Self::new(self.x^r.x, self.y^r.y) } }
impl BitAndAssign for I8Vec2 { #[inline] fn bitand_assign(&mut self,r:Self) { *self = *self & r; } }
impl BitOrAssign  for I8Vec2 { #[inline] fn bitor_assign (&mut self,r:Self) { *self = *self | r; } }
impl BitXorAssign for I8Vec2 { #[inline] fn bitxor_assign(&mut self,r:Self) { *self = *self ^ r; } }
impl Shl<u32> for I8Vec2 { type Output=Self; #[inline] fn shl(self,s:u32)->Self { Self::new(self.x<<s, self.y<<s) } }
impl Shr<u32> for I8Vec2 { type Output=Self; #[inline] fn shr(self,s:u32)->Self { Self::new(self.x>>s, self.y>>s) } }
impl ShlAssign<u32> for I8Vec2 { #[inline] fn shl_assign(&mut self,s:u32) { self.x<<=s; self.y<<=s; } }
impl ShrAssign<u32> for I8Vec2 { #[inline] fn shr_assign(&mut self,s:u32) { self.x>>=s; self.y>>=s; } }

impl Index<usize> for I8Vec2 {
    type Output = i8;
    #[inline] fn index(&self, i: usize) -> &i8 { match i { 0=>&self.x, 1=>&self.y, _=>panic!("I8Vec2 index {i} out of bounds") } }
}
impl IndexMut<usize> for I8Vec2 {
    #[inline] fn index_mut(&mut self, i: usize) -> &mut i8 { match i { 0=>&mut self.x, 1=>&mut self.y, _=>panic!("I8Vec2 index {i} out of bounds") } }
}

impl From<[i8;2]> for I8Vec2 { #[inline] fn from(a:[i8;2])->Self { Self::from_array(a) } }
impl From<I8Vec2> for [i8;2] { #[inline] fn from(v:I8Vec2)->[i8;2] { v.to_array() } }
impl From<(i8,i8)> for I8Vec2 { #[inline] fn from(t:(i8,i8))->Self { Self::new(t.0,t.1) } }
impl From<I8Vec2> for (i8,i8) { #[inline] fn from(v:I8Vec2)->(i8,i8) { (v.x,v.y) } }

impl fmt::Debug for I8Vec2 { fn fmt(&self,f:&mut fmt::Formatter<'_>)->fmt::Result { write!(f,"I8Vec2({}, {})",self.x,self.y) } }
impl fmt::Display for I8Vec2 { fn fmt(&self,f:&mut fmt::Formatter<'_>)->fmt::Result { write!(f,"[{}, {}]",self.x,self.y) } }
