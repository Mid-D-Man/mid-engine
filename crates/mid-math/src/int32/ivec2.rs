// crates/mid-math/src/int/ivec2.rs
//! 2D signed-integer vector. 8 bytes, align 4. Always scalar.

use core::fmt;
use core::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign,
    Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Not,
    Rem, RemAssign, Shl, ShlAssign, Shr, ShrAssign, Sub, SubAssign,
};
use crate::{BVec2, IVec3, UVec2, Vec2};

/// 2D signed-integer vector. 8 bytes, align 4.
///
/// Used for screen-space pixel coordinates, tile/grid indices, and any
/// 2D integer domain. Signed allows off-screen / negative coordinates.
///
/// **C interop:** use [`CIVec2`][crate::ffi::types::CIVec2] at the FFI boundary.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(C)]
pub struct IVec2 {
    pub x: i32,
    pub y: i32,
}

impl IVec2 {
    // ── Constants ─────────────────────────────────────────────────────────────

    pub const ZERO:  Self = Self::splat(0);
    pub const ONE:   Self = Self::splat(1);
    pub const NEG_ONE: Self = Self::splat(-1);
    pub const MIN:   Self = Self::splat(i32::MIN);
    pub const MAX:   Self = Self::splat(i32::MAX);
    pub const X:     Self = Self::new(1, 0);
    pub const Y:     Self = Self::new(0, 1);
    pub const NEG_X: Self = Self::new(-1, 0);
    pub const NEG_Y: Self = Self::new(0, -1);

    // ── Constructors ──────────────────────────────────────────────────────────

    #[inline(always)] pub const fn new(x: i32, y: i32) -> Self { Self { x, y } }
    #[inline(always)] pub const fn splat(v: i32) -> Self { Self { x: v, y: v } }
    #[inline(always)] pub const fn from_array(a: [i32; 2]) -> Self { Self::new(a[0], a[1]) }
    #[inline(always)] pub const fn to_array(self) -> [i32; 2] { [self.x, self.y] }

    /// Extend to IVec3 by appending `z`.
    #[inline(always)]
    pub const fn extend(self, z: i32) -> IVec3 { IVec3::new(self.x, self.y, z) }

    /// Element-wise select: choose `if_true` where mask is `true`, else `if_false`.
    #[inline]
    pub fn select(mask: BVec2, if_true: Self, if_false: Self) -> Self {
        Self::new(
            if mask.x { if_true.x } else { if_false.x },
            if mask.y { if_true.y } else { if_false.y },
        )
    }

    // ── Arithmetic ────────────────────────────────────────────────────────────

    #[inline] pub fn dot(self, rhs: Self) -> i32 { self.x * rhs.x + self.y * rhs.y }
    #[inline] pub fn length_sq(self) -> i32 { self.dot(self) }
    #[inline] pub fn distance_sq(self, rhs: Self) -> i32 { (self - rhs).length_sq() }
    #[inline] pub fn abs(self) -> Self { Self::new(self.x.abs(), self.y.abs()) }
    #[inline] pub fn signum(self) -> Self { Self::new(self.x.signum(), self.y.signum()) }
    #[inline] pub fn perp(self) -> Self { Self::new(-self.y, self.x) }
    #[inline] pub fn perp_dot(self, rhs: Self) -> i32 { self.x * rhs.y - self.y * rhs.x }

    // ── Component-wise ────────────────────────────────────────────────────────

    #[inline] pub fn min(self, rhs: Self) -> Self {
        Self::new(self.x.min(rhs.x), self.y.min(rhs.y))
    }
    #[inline] pub fn max(self, rhs: Self) -> Self {
        Self::new(self.x.max(rhs.x), self.y.max(rhs.y))
    }
    #[inline] pub fn clamp(self, lo: Self, hi: Self) -> Self { self.max(lo).min(hi) }
    #[inline] pub fn min_element(self) -> i32 { self.x.min(self.y) }
    #[inline] pub fn max_element(self) -> i32 { self.x.max(self.y) }
    #[inline] pub fn element_sum(self) -> i32 { self.x + self.y }
    #[inline] pub fn element_product(self) -> i32 { self.x * self.y }

    // ── Comparisons → BVec2 ───────────────────────────────────────────────────

    #[inline] pub fn cmpeq(self, rhs: Self) -> BVec2 { BVec2::new(self.x == rhs.x, self.y == rhs.y) }
    #[inline] pub fn cmpne(self, rhs: Self) -> BVec2 { BVec2::new(self.x != rhs.x, self.y != rhs.y) }
    #[inline] pub fn cmpge(self, rhs: Self) -> BVec2 { BVec2::new(self.x >= rhs.x, self.y >= rhs.y) }
    #[inline] pub fn cmpgt(self, rhs: Self) -> BVec2 { BVec2::new(self.x > rhs.x,  self.y > rhs.y)  }
    #[inline] pub fn cmple(self, rhs: Self) -> BVec2 { BVec2::new(self.x <= rhs.x, self.y <= rhs.y) }
    #[inline] pub fn cmplt(self, rhs: Self) -> BVec2 { BVec2::new(self.x < rhs.x,  self.y < rhs.y)  }

    // ── Wrapping / Saturating ─────────────────────────────────────────────────

    #[inline] pub fn wrapping_add(self, rhs: Self) -> Self {
        Self::new(self.x.wrapping_add(rhs.x), self.y.wrapping_add(rhs.y))
    }
    #[inline] pub fn wrapping_sub(self, rhs: Self) -> Self {
        Self::new(self.x.wrapping_sub(rhs.x), self.y.wrapping_sub(rhs.y))
    }
    #[inline] pub fn wrapping_mul(self, rhs: Self) -> Self {
        Self::new(self.x.wrapping_mul(rhs.x), self.y.wrapping_mul(rhs.y))
    }
    #[inline] pub fn saturating_add(self, rhs: Self) -> Self {
        Self::new(self.x.saturating_add(rhs.x), self.y.saturating_add(rhs.y))
    }
    #[inline] pub fn saturating_sub(self, rhs: Self) -> Self {
        Self::new(self.x.saturating_sub(rhs.x), self.y.saturating_sub(rhs.y))
    }

    // ── Casts ─────────────────────────────────────────────────────────────────

    #[inline] pub fn as_vec2(self) -> Vec2 { Vec2::new(self.x as f32, self.y as f32) }
    #[inline] pub fn as_dvec2(self) -> crate::DVec2 { crate::DVec2::new(self.x as f64, self.y as f64) }
    #[inline] pub fn as_uvec2(self) -> UVec2 { UVec2::new(self.x as u32, self.y as u32) }
}

// ── Operators ─────────────────────────────────────────────────────────────────

impl Add  for IVec2 { type Output=Self; #[inline] fn add(self,r:Self)->Self { Self::new(self.x+r.x, self.y+r.y) } }
impl Sub  for IVec2 { type Output=Self; #[inline] fn sub(self,r:Self)->Self { Self::new(self.x-r.x, self.y-r.y) } }
impl Mul  for IVec2 { type Output=Self; #[inline] fn mul(self,r:Self)->Self { Self::new(self.x*r.x, self.y*r.y) } }
impl Div  for IVec2 { type Output=Self; #[inline] fn div(self,r:Self)->Self { Self::new(self.x/r.x, self.y/r.y) } }
impl Rem  for IVec2 { type Output=Self; #[inline] fn rem(self,r:Self)->Self { Self::new(self.x%r.x, self.y%r.y) } }
impl Neg  for IVec2 { type Output=Self; #[inline] fn neg(self)->Self { Self::new(-self.x, -self.y) } }
impl Not  for IVec2 { type Output=Self; #[inline] fn not(self)->Self { Self::new(!self.x, !self.y) } }

impl Add<i32> for IVec2 { type Output=Self; #[inline] fn add(self,s:i32)->Self { Self::new(self.x+s, self.y+s) } }
impl Sub<i32> for IVec2 { type Output=Self; #[inline] fn sub(self,s:i32)->Self { Self::new(self.x-s, self.y-s) } }
impl Mul<i32> for IVec2 { type Output=Self; #[inline] fn mul(self,s:i32)->Self { Self::new(self.x*s, self.y*s) } }
impl Div<i32> for IVec2 { type Output=Self; #[inline] fn div(self,s:i32)->Self { Self::new(self.x/s, self.y/s) } }
impl Rem<i32> for IVec2 { type Output=Self; #[inline] fn rem(self,s:i32)->Self { Self::new(self.x%s, self.y%s) } }
impl Mul<IVec2> for i32 { type Output=IVec2; #[inline] fn mul(self,v:IVec2)->IVec2 { IVec2::new(self*v.x, self*v.y) } }

impl AddAssign     for IVec2 { #[inline] fn add_assign(&mut self,r:Self) { self.x+=r.x; self.y+=r.y; } }
impl SubAssign     for IVec2 { #[inline] fn sub_assign(&mut self,r:Self) { self.x-=r.x; self.y-=r.y; } }
impl MulAssign     for IVec2 { #[inline] fn mul_assign(&mut self,r:Self) { self.x*=r.x; self.y*=r.y; } }
impl DivAssign     for IVec2 { #[inline] fn div_assign(&mut self,r:Self) { self.x/=r.x; self.y/=r.y; } }
impl RemAssign     for IVec2 { #[inline] fn rem_assign(&mut self,r:Self) { self.x%=r.x; self.y%=r.y; } }
impl AddAssign<i32> for IVec2 { #[inline] fn add_assign(&mut self,s:i32) { self.x+=s; self.y+=s; } }
impl SubAssign<i32> for IVec2 { #[inline] fn sub_assign(&mut self,s:i32) { self.x-=s; self.y-=s; } }
impl MulAssign<i32> for IVec2 { #[inline] fn mul_assign(&mut self,s:i32) { self.x*=s; self.y*=s; } }
impl DivAssign<i32> for IVec2 { #[inline] fn div_assign(&mut self,s:i32) { self.x/=s; self.y/=s; } }
impl RemAssign<i32> for IVec2 { #[inline] fn rem_assign(&mut self,s:i32) { self.x%=s; self.y%=s; } }

impl BitAnd for IVec2 { type Output=Self; #[inline] fn bitand(self,r:Self)->Self { Self::new(self.x&r.x, self.y&r.y) } }
impl BitOr  for IVec2 { type Output=Self; #[inline] fn bitor (self,r:Self)->Self { Self::new(self.x|r.x, self.y|r.y) } }
impl BitXor for IVec2 { type Output=Self; #[inline] fn bitxor(self,r:Self)->Self { Self::new(self.x^r.x, self.y^r.y) } }
impl BitAndAssign for IVec2 { #[inline] fn bitand_assign(&mut self,r:Self) { *self = *self & r; } }
impl BitOrAssign  for IVec2 { #[inline] fn bitor_assign (&mut self,r:Self) { *self = *self | r; } }
impl BitXorAssign for IVec2 { #[inline] fn bitxor_assign(&mut self,r:Self) { *self = *self ^ r; } }
impl Shl<i32> for IVec2 { type Output=Self; #[inline] fn shl(self,s:i32)->Self { Self::new(self.x<<s, self.y<<s) } }
impl Shr<i32> for IVec2 { type Output=Self; #[inline] fn shr(self,s:i32)->Self { Self::new(self.x>>s, self.y>>s) } }
impl Shl<u32> for IVec2 { type Output=Self; #[inline] fn shl(self,s:u32)->Self { Self::new(self.x<<s, self.y<<s) } }
impl Shr<u32> for IVec2 { type Output=Self; #[inline] fn shr(self,s:u32)->Self { Self::new(self.x>>s, self.y>>s) } }
impl ShlAssign<i32> for IVec2 { #[inline] fn shl_assign(&mut self,s:i32) { self.x<<=s; self.y<<=s; } }
impl ShrAssign<i32> for IVec2 { #[inline] fn shr_assign(&mut self,s:i32) { self.x>>=s; self.y>>=s; } }
impl ShlAssign<u32> for IVec2 { #[inline] fn shl_assign(&mut self,s:u32) { self.x<<=s; self.y<<=s; } }
impl ShrAssign<u32> for IVec2 { #[inline] fn shr_assign(&mut self,s:u32) { self.x>>=s; self.y>>=s; } }

impl Index<usize> for IVec2 {
    type Output = i32;
    #[inline] fn index(&self, i: usize) -> &i32 {
        match i { 0=>&self.x, 1=>&self.y, _=>panic!("IVec2 index {i} out of bounds") }
    }
}
impl IndexMut<usize> for IVec2 {
    #[inline] fn index_mut(&mut self, i: usize) -> &mut i32 {
        match i { 0=>&mut self.x, 1=>&mut self.y, _=>panic!("IVec2 index {i} out of bounds") }
    }
}

impl From<[i32;2]> for IVec2 { #[inline] fn from(a:[i32;2])->Self { Self::from_array(a) } }
impl From<IVec2> for [i32;2] { #[inline] fn from(v:IVec2)->[i32;2] { v.to_array() } }
impl From<(i32,i32)> for IVec2 { #[inline] fn from(t:(i32,i32))->Self { Self::new(t.0,t.1) } }
impl From<IVec2> for (i32,i32) { #[inline] fn from(v:IVec2)->(i32,i32) { (v.x,v.y) } }

impl fmt::Debug for IVec2 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "IVec2({}, {})", self.x, self.y)
    }
}
impl fmt::Display for IVec2 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}, {}]", self.x, self.y)
    }
      }
