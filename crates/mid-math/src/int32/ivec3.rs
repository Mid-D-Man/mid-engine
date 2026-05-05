// crates/mid-math/src/int/ivec3.rs
//! 3D signed-integer vector. 12 bytes, align 4. No padding. Always scalar.

use core::fmt;
use core::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign,
    Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Not,
    Rem, RemAssign, Shl, ShlAssign, Shr, ShrAssign, Sub, SubAssign,
};
use crate::{BVec3, IVec2, IVec4, UVec3, Vec3};

/// 3D signed-integer vector. 12 bytes, align 4. No padding.
///
/// Used for voxel/chunk coordinates, 3D grid indices, and integer-domain
/// spatial queries. 12 bytes — no padding unlike f32 Vec3 which pads to 16.
///
/// **C interop:** use [`CIVec3`][crate::ffi::types::CIVec3] at the FFI boundary.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(C)]
pub struct IVec3 {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

impl IVec3 {
    // ── Constants ─────────────────────────────────────────────────────────────

    pub const ZERO:    Self = Self::splat(0);
    pub const ONE:     Self = Self::splat(1);
    pub const NEG_ONE: Self = Self::splat(-1);
    pub const MIN:     Self = Self::splat(i32::MIN);
    pub const MAX:     Self = Self::splat(i32::MAX);
    pub const X:       Self = Self::new(1, 0, 0);
    pub const Y:       Self = Self::new(0, 1, 0);
    pub const Z:       Self = Self::new(0, 0, 1);
    pub const NEG_X:   Self = Self::new(-1,  0,  0);
    pub const NEG_Y:   Self = Self::new( 0, -1,  0);
    pub const NEG_Z:   Self = Self::new( 0,  0, -1);

    // ── Constructors ──────────────────────────────────────────────────────────

    #[inline(always)] pub const fn new(x: i32, y: i32, z: i32) -> Self { Self { x, y, z } }
    #[inline(always)] pub const fn splat(v: i32) -> Self { Self { x: v, y: v, z: v } }
    #[inline(always)] pub const fn from_array(a: [i32; 3]) -> Self { Self::new(a[0], a[1], a[2]) }
    #[inline(always)] pub const fn to_array(self) -> [i32; 3] { [self.x, self.y, self.z] }

    /// Extend to IVec4 by appending `w`.
    #[inline(always)]
    pub const fn extend(self, w: i32) -> IVec4 { IVec4::new(self.x, self.y, self.z, w) }

    /// Truncate to IVec2, discarding z.
    #[inline(always)]
    pub const fn truncate(self) -> IVec2 { IVec2::new(self.x, self.y) }

    /// Element-wise select.
    #[inline]
    pub fn select(mask: BVec3, if_true: Self, if_false: Self) -> Self {
        Self::new(
            if mask.x { if_true.x } else { if_false.x },
            if mask.y { if_true.y } else { if_false.y },
            if mask.z { if_true.z } else { if_false.z },
        )
    }

    // ── Arithmetic ────────────────────────────────────────────────────────────

    #[inline]
    pub fn dot(self, rhs: Self) -> i32 {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
    }

    #[inline]
    pub fn cross(self, rhs: Self) -> Self {
        Self::new(
            self.y * rhs.z - self.z * rhs.y,
            self.z * rhs.x - self.x * rhs.z,
            self.x * rhs.y - self.y * rhs.x,
        )
    }

    #[inline] pub fn length_sq(self) -> i32 { self.dot(self) }
    #[inline] pub fn distance_sq(self, rhs: Self) -> i32 { (self - rhs).length_sq() }
    #[inline] pub fn abs(self) -> Self { Self::new(self.x.abs(), self.y.abs(), self.z.abs()) }
    #[inline] pub fn signum(self) -> Self { Self::new(self.x.signum(), self.y.signum(), self.z.signum()) }

    // ── Component-wise ────────────────────────────────────────────────────────

    #[inline] pub fn min(self, rhs: Self) -> Self {
        Self::new(self.x.min(rhs.x), self.y.min(rhs.y), self.z.min(rhs.z))
    }
    #[inline] pub fn max(self, rhs: Self) -> Self {
        Self::new(self.x.max(rhs.x), self.y.max(rhs.y), self.z.max(rhs.z))
    }
    #[inline] pub fn clamp(self, lo: Self, hi: Self) -> Self { self.max(lo).min(hi) }
    #[inline] pub fn min_element(self) -> i32 { self.x.min(self.y).min(self.z) }
    #[inline] pub fn max_element(self) -> i32 { self.x.max(self.y).max(self.z) }
    #[inline] pub fn element_sum(self) -> i32 { self.x + self.y + self.z }
    #[inline] pub fn element_product(self) -> i32 { self.x * self.y * self.z }

    // ── Comparisons → BVec3 ───────────────────────────────────────────────────

    #[inline] pub fn cmpeq(self, r: Self) -> BVec3 { BVec3::new(self.x==r.x, self.y==r.y, self.z==r.z) }
    #[inline] pub fn cmpne(self, r: Self) -> BVec3 { BVec3::new(self.x!=r.x, self.y!=r.y, self.z!=r.z) }
    #[inline] pub fn cmpge(self, r: Self) -> BVec3 { BVec3::new(self.x>=r.x, self.y>=r.y, self.z>=r.z) }
    #[inline] pub fn cmpgt(self, r: Self) -> BVec3 { BVec3::new(self.x>r.x,  self.y>r.y,  self.z>r.z)  }
    #[inline] pub fn cmple(self, r: Self) -> BVec3 { BVec3::new(self.x<=r.x, self.y<=r.y, self.z<=r.z) }
    #[inline] pub fn cmplt(self, r: Self) -> BVec3 { BVec3::new(self.x<r.x,  self.y<r.y,  self.z<r.z)  }

    // ── Wrapping / Saturating ─────────────────────────────────────────────────

    #[inline] pub fn wrapping_add(self, r: Self) -> Self {
        Self::new(self.x.wrapping_add(r.x), self.y.wrapping_add(r.y), self.z.wrapping_add(r.z))
    }
    #[inline] pub fn wrapping_sub(self, r: Self) -> Self {
        Self::new(self.x.wrapping_sub(r.x), self.y.wrapping_sub(r.y), self.z.wrapping_sub(r.z))
    }
    #[inline] pub fn wrapping_mul(self, r: Self) -> Self {
        Self::new(self.x.wrapping_mul(r.x), self.y.wrapping_mul(r.y), self.z.wrapping_mul(r.z))
    }
    #[inline] pub fn saturating_add(self, r: Self) -> Self {
        Self::new(self.x.saturating_add(r.x), self.y.saturating_add(r.y), self.z.saturating_add(r.z))
    }
    #[inline] pub fn saturating_sub(self, r: Self) -> Self {
        Self::new(self.x.saturating_sub(r.x), self.y.saturating_sub(r.y), self.z.saturating_sub(r.z))
    }

    // ── Casts ─────────────────────────────────────────────────────────────────

    #[inline] pub fn as_vec3(self) -> Vec3 { Vec3::new(self.x as f32, self.y as f32, self.z as f32) }
    #[inline] pub fn as_dvec3(self) -> crate::DVec3 {
        crate::DVec3::new(self.x as f64, self.y as f64, self.z as f64)
    }
    #[inline] pub fn as_uvec3(self) -> UVec3 { UVec3::new(self.x as u32, self.y as u32, self.z as u32) }
}

// ── Operators ─────────────────────────────────────────────────────────────────

impl Add  for IVec3 { type Output=Self; #[inline] fn add(self,r:Self)->Self { Self::new(self.x+r.x, self.y+r.y, self.z+r.z) } }
impl Sub  for IVec3 { type Output=Self; #[inline] fn sub(self,r:Self)->Self { Self::new(self.x-r.x, self.y-r.y, self.z-r.z) } }
impl Mul  for IVec3 { type Output=Self; #[inline] fn mul(self,r:Self)->Self { Self::new(self.x*r.x, self.y*r.y, self.z*r.z) } }
impl Div  for IVec3 { type Output=Self; #[inline] fn div(self,r:Self)->Self { Self::new(self.x/r.x, self.y/r.y, self.z/r.z) } }
impl Rem  for IVec3 { type Output=Self; #[inline] fn rem(self,r:Self)->Self { Self::new(self.x%r.x, self.y%r.y, self.z%r.z) } }
impl Neg  for IVec3 { type Output=Self; #[inline] fn neg(self)->Self { Self::new(-self.x, -self.y, -self.z) } }
impl Not  for IVec3 { type Output=Self; #[inline] fn not(self)->Self { Self::new(!self.x, !self.y, !self.z) } }

impl Add<i32> for IVec3 { type Output=Self; #[inline] fn add(self,s:i32)->Self { Self::new(self.x+s, self.y+s, self.z+s) } }
impl Sub<i32> for IVec3 { type Output=Self; #[inline] fn sub(self,s:i32)->Self { Self::new(self.x-s, self.y-s, self.z-s) } }
impl Mul<i32> for IVec3 { type Output=Self; #[inline] fn mul(self,s:i32)->Self { Self::new(self.x*s, self.y*s, self.z*s) } }
impl Div<i32> for IVec3 { type Output=Self; #[inline] fn div(self,s:i32)->Self { Self::new(self.x/s, self.y/s, self.z/s) } }
impl Rem<i32> for IVec3 { type Output=Self; #[inline] fn rem(self,s:i32)->Self { Self::new(self.x%s, self.y%s, self.z%s) } }
impl Mul<IVec3> for i32 { type Output=IVec3; #[inline] fn mul(self,v:IVec3)->IVec3 { IVec3::new(self*v.x, self*v.y, self*v.z) } }

impl AddAssign      for IVec3 { #[inline] fn add_assign(&mut self,r:Self) { self.x+=r.x; self.y+=r.y; self.z+=r.z; } }
impl SubAssign      for IVec3 { #[inline] fn sub_assign(&mut self,r:Self) { self.x-=r.x; self.y-=r.y; self.z-=r.z; } }
impl MulAssign      for IVec3 { #[inline] fn mul_assign(&mut self,r:Self) { self.x*=r.x; self.y*=r.y; self.z*=r.z; } }
impl DivAssign      for IVec3 { #[inline] fn div_assign(&mut self,r:Self) { self.x/=r.x; self.y/=r.y; self.z/=r.z; } }
impl RemAssign      for IVec3 { #[inline] fn rem_assign(&mut self,r:Self) { self.x%=r.x; self.y%=r.y; self.z%=r.z; } }
impl AddAssign<i32> for IVec3 { #[inline] fn add_assign(&mut self,s:i32) { self.x+=s; self.y+=s; self.z+=s; } }
impl SubAssign<i32> for IVec3 { #[inline] fn sub_assign(&mut self,s:i32) { self.x-=s; self.y-=s; self.z-=s; } }
impl MulAssign<i32> for IVec3 { #[inline] fn mul_assign(&mut self,s:i32) { self.x*=s; self.y*=s; self.z*=s; } }
impl DivAssign<i32> for IVec3 { #[inline] fn div_assign(&mut self,s:i32) { self.x/=s; self.y/=s; self.z/=s; } }
impl RemAssign<i32> for IVec3 { #[inline] fn rem_assign(&mut self,s:i32) { self.x%=s; self.y%=s; self.z%=s; } }

impl BitAnd for IVec3 { type Output=Self; #[inline] fn bitand(self,r:Self)->Self { Self::new(self.x&r.x, self.y&r.y, self.z&r.z) } }
impl BitOr  for IVec3 { type Output=Self; #[inline] fn bitor (self,r:Self)->Self { Self::new(self.x|r.x, self.y|r.y, self.z|r.z) } }
impl BitXor for IVec3 { type Output=Self; #[inline] fn bitxor(self,r:Self)->Self { Self::new(self.x^r.x, self.y^r.y, self.z^r.z) } }
impl BitAndAssign for IVec3 { #[inline] fn bitand_assign(&mut self,r:Self) { *self = *self & r; } }
impl BitOrAssign  for IVec3 { #[inline] fn bitor_assign (&mut self,r:Self) { *self = *self | r; } }
impl BitXorAssign for IVec3 { #[inline] fn bitxor_assign(&mut self,r:Self) { *self = *self ^ r; } }
impl Shl<i32> for IVec3 { type Output=Self; #[inline] fn shl(self,s:i32)->Self { Self::new(self.x<<s, self.y<<s, self.z<<s) } }
impl Shr<i32> for IVec3 { type Output=Self; #[inline] fn shr(self,s:i32)->Self { Self::new(self.x>>s, self.y>>s, self.z>>s) } }
impl Shl<u32> for IVec3 { type Output=Self; #[inline] fn shl(self,s:u32)->Self { Self::new(self.x<<s, self.y<<s, self.z<<s) } }
impl Shr<u32> for IVec3 { type Output=Self; #[inline] fn shr(self,s:u32)->Self { Self::new(self.x>>s, self.y>>s, self.z>>s) } }
impl ShlAssign<i32> for IVec3 { #[inline] fn shl_assign(&mut self,s:i32) { self.x<<=s; self.y<<=s; self.z<<=s; } }
impl ShrAssign<i32> for IVec3 { #[inline] fn shr_assign(&mut self,s:i32) { self.x>>=s; self.y>>=s; self.z>>=s; } }
impl ShlAssign<u32> for IVec3 { #[inline] fn shl_assign(&mut self,s:u32) { self.x<<=s; self.y<<=s; self.z<<=s; } }
impl ShrAssign<u32> for IVec3 { #[inline] fn shr_assign(&mut self,s:u32) { self.x>>=s; self.y>>=s; self.z>>=s; } }

impl Index<usize> for IVec3 {
    type Output = i32;
    #[inline] fn index(&self, i: usize) -> &i32 {
        match i { 0=>&self.x, 1=>&self.y, 2=>&self.z, _=>panic!("IVec3 index {i} out of bounds") }
    }
}
impl IndexMut<usize> for IVec3 {
    #[inline] fn index_mut(&mut self, i: usize) -> &mut i32 {
        match i { 0=>&mut self.x, 1=>&mut self.y, 2=>&mut self.z, _=>panic!("IVec3 index {i} out of bounds") }
    }
}

impl From<[i32;3]> for IVec3 { #[inline] fn from(a:[i32;3])->Self { Self::from_array(a) } }
impl From<IVec3> for [i32;3] { #[inline] fn from(v:IVec3)->[i32;3] { v.to_array() } }
impl From<(i32,i32,i32)> for IVec3 { #[inline] fn from(t:(i32,i32,i32))->Self { Self::new(t.0,t.1,t.2) } }
impl From<IVec3> for (i32,i32,i32) { #[inline] fn from(v:IVec3)->(i32,i32,i32) { (v.x,v.y,v.z) } }
impl From<(IVec2, i32)> for IVec3 { #[inline] fn from((v,z):(IVec2,i32))->Self { Self::new(v.x,v.y,z) } }

impl fmt::Debug for IVec3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "IVec3({}, {}, {})", self.x, self.y, self.z)
    }
}
impl fmt::Display for IVec3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}, {}, {}]", self.x, self.y, self.z)
    }
  }
