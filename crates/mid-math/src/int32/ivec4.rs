// crates/mid-math/src/int32/ivec4.rs
//! 4D signed-integer vector. 16 bytes, align 4. Always scalar.

use core::fmt;
use core::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign,
    Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Not,
    Rem, Shl, ShlAssign, Shr, ShrAssign, Sub, SubAssign,
};
use crate::{BVec4, IVec2, IVec3, UVec4, Vec4};

/// 4D signed-integer vector. 16 bytes, align 4.
///
/// Used for packed integer data (e.g. RGBA8 unpacked), bone indices,
/// and 4-component integer parameters.
///
/// **C interop:** use [`CIVec4`][crate::ffi::types::CIVec4] at the FFI boundary.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(C)]
pub struct IVec4 {
    pub x: i32,
    pub y: i32,
    pub z: i32,
    pub w: i32,
}

impl IVec4 {
    // ── Constants ─────────────────────────────────────────────────────────────

    pub const ZERO:    Self = Self::splat(0);
    pub const ONE:     Self = Self::splat(1);
    pub const NEG_ONE: Self = Self::splat(-1);
    pub const MIN:     Self = Self::splat(i32::MIN);
    pub const MAX:     Self = Self::splat(i32::MAX);
    pub const X:       Self = Self::new(1, 0, 0, 0);
    pub const Y:       Self = Self::new(0, 1, 0, 0);
    pub const Z:       Self = Self::new(0, 0, 1, 0);
    pub const W:       Self = Self::new(0, 0, 0, 1);
    pub const NEG_X:   Self = Self::new(-1,  0,  0,  0);
    pub const NEG_Y:   Self = Self::new( 0, -1,  0,  0);
    pub const NEG_Z:   Self = Self::new( 0,  0, -1,  0);
    pub const NEG_W:   Self = Self::new( 0,  0,  0, -1);

    // ── Constructors ──────────────────────────────────────────────────────────

    #[inline(always)] pub const fn new(x: i32, y: i32, z: i32, w: i32) -> Self { Self { x, y, z, w } }
    #[inline(always)] pub const fn splat(v: i32) -> Self { Self { x: v, y: v, z: v, w: v } }
    #[inline(always)] pub const fn from_array(a: [i32; 4]) -> Self { Self::new(a[0], a[1], a[2], a[3]) }
    #[inline(always)] pub const fn to_array(self) -> [i32; 4] { [self.x, self.y, self.z, self.w] }

    /// Truncate to IVec3, discarding w.
    #[inline(always)] pub const fn truncate(self) -> IVec3 { IVec3::new(self.x, self.y, self.z) }

    /// Element-wise select.
    #[inline]
    pub fn select(mask: BVec4, if_true: Self, if_false: Self) -> Self {
        Self::new(
            if mask.x { if_true.x } else { if_false.x },
            if mask.y { if_true.y } else { if_false.y },
            if mask.z { if_true.z } else { if_false.z },
            if mask.w { if_true.w } else { if_false.w },
        )
    }

    // ── Arithmetic ────────────────────────────────────────────────────────────

    #[inline] pub fn dot(self, rhs: Self) -> i32 {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z + self.w * rhs.w
    }
    #[inline] pub fn length_sq(self) -> i32 { self.dot(self) }
    #[inline] pub fn distance_sq(self, rhs: Self) -> i32 { (self - rhs).length_sq() }
    #[inline] pub fn abs(self) -> Self { Self::new(self.x.abs(), self.y.abs(), self.z.abs(), self.w.abs()) }
    #[inline] pub fn signum(self) -> Self { Self::new(self.x.signum(), self.y.signum(), self.z.signum(), self.w.signum()) }

    // ── Component-wise ────────────────────────────────────────────────────────

    #[inline] pub fn min(self, r: Self) -> Self {
        Self::new(self.x.min(r.x), self.y.min(r.y), self.z.min(r.z), self.w.min(r.w))
    }
    #[inline] pub fn max(self, r: Self) -> Self {
        Self::new(self.x.max(r.x), self.y.max(r.y), self.z.max(r.z), self.w.max(r.w))
    }
    #[inline] pub fn clamp(self, lo: Self, hi: Self) -> Self { self.max(lo).min(hi) }
    #[inline] pub fn min_element(self) -> i32 { self.x.min(self.y).min(self.z).min(self.w) }
    #[inline] pub fn max_element(self) -> i32 { self.x.max(self.y).max(self.z).max(self.w) }
    #[inline] pub fn element_sum(self) -> i32 { self.x + self.y + self.z + self.w }
    #[inline] pub fn element_product(self) -> i32 { self.x * self.y * self.z * self.w }

    // ── Comparisons → BVec4 ───────────────────────────────────────────────────

    #[inline] pub fn cmpeq(self, r: Self) -> BVec4 { BVec4::new(self.x==r.x, self.y==r.y, self.z==r.z, self.w==r.w) }
    #[inline] pub fn cmpne(self, r: Self) -> BVec4 { BVec4::new(self.x!=r.x, self.y!=r.y, self.z!=r.z, self.w!=r.w) }
    #[inline] pub fn cmpge(self, r: Self) -> BVec4 { BVec4::new(self.x>=r.x, self.y>=r.y, self.z>=r.z, self.w>=r.w) }
    #[inline] pub fn cmpgt(self, r: Self) -> BVec4 { BVec4::new(self.x>r.x,  self.y>r.y,  self.z>r.z,  self.w>r.w)  }
    #[inline] pub fn cmple(self, r: Self) -> BVec4 { BVec4::new(self.x<=r.x, self.y<=r.y, self.z<=r.z, self.w<=r.w) }
    #[inline] pub fn cmplt(self, r: Self) -> BVec4 { BVec4::new(self.x<r.x,  self.y<r.y,  self.z<r.z,  self.w<r.w)  }

    // ── Wrapping / Saturating ─────────────────────────────────────────────────

    #[inline] pub fn wrapping_add(self, r: Self) -> Self {
        Self::new(self.x.wrapping_add(r.x), self.y.wrapping_add(r.y), self.z.wrapping_add(r.z), self.w.wrapping_add(r.w))
    }
    #[inline] pub fn wrapping_sub(self, r: Self) -> Self {
        Self::new(self.x.wrapping_sub(r.x), self.y.wrapping_sub(r.y), self.z.wrapping_sub(r.z), self.w.wrapping_sub(r.w))
    }
    #[inline] pub fn wrapping_mul(self, r: Self) -> Self {
        Self::new(self.x.wrapping_mul(r.x), self.y.wrapping_mul(r.y), self.z.wrapping_mul(r.z), self.w.wrapping_mul(r.w))
    }
    #[inline] pub fn saturating_add(self, r: Self) -> Self {
        Self::new(self.x.saturating_add(r.x), self.y.saturating_add(r.y), self.z.saturating_add(r.z), self.w.saturating_add(r.w))
    }
    #[inline] pub fn saturating_sub(self, r: Self) -> Self {
        Self::new(self.x.saturating_sub(r.x), self.y.saturating_sub(r.y), self.z.saturating_sub(r.z), self.w.saturating_sub(r.w))
    }

    // ── Casts ─────────────────────────────────────────────────────────────────

    #[inline] pub fn as_vec4(self) -> Vec4 {
        Vec4::new(self.x as f32, self.y as f32, self.z as f32, self.w as f32)
    }
    #[inline] pub fn as_dvec4(self) -> crate::DVec4 {
        crate::DVec4::new(self.x as f64, self.y as f64, self.z as f64, self.w as f64)
    }
    #[inline] pub fn as_uvec4(self) -> UVec4 {
        UVec4::new(self.x as u32, self.y as u32, self.z as u32, self.w as u32)
    }
}

// ── Operators ─────────────────────────────────────────────────────────────────

impl Add  for IVec4 { type Output=Self; #[inline] fn add(self,r:Self)->Self { Self::new(self.x+r.x, self.y+r.y, self.z+r.z, self.w+r.w) } }
impl Sub  for IVec4 { type Output=Self; #[inline] fn sub(self,r:Self)->Self { Self::new(self.x-r.x, self.y-r.y, self.z-r.z, self.w-r.w) } }
impl Mul  for IVec4 { type Output=Self; #[inline] fn mul(self,r:Self)->Self { Self::new(self.x*r.x, self.y*r.y, self.z*r.z, self.w*r.w) } }
impl Div  for IVec4 { type Output=Self; #[inline] fn div(self,r:Self)->Self { Self::new(self.x/r.x, self.y/r.y, self.z/r.z, self.w/r.w) } }
impl Rem  for IVec4 { type Output=Self; #[inline] fn rem(self,r:Self)->Self { Self::new(self.x%r.x, self.y%r.y, self.z%r.z, self.w%r.w) } }
impl Neg  for IVec4 { type Output=Self; #[inline] fn neg(self)->Self { Self::new(-self.x, -self.y, -self.z, -self.w) } }
impl Not  for IVec4 { type Output=Self; #[inline] fn not(self)->Self { Self::new(!self.x, !self.y, !self.z, !self.w) } }

impl Add<i32> for IVec4 { type Output=Self; #[inline] fn add(self,s:i32)->Self { Self::new(self.x+s, self.y+s, self.z+s, self.w+s) } }
impl Sub<i32> for IVec4 { type Output=Self; #[inline] fn sub(self,s:i32)->Self { Self::new(self.x-s, self.y-s, self.z-s, self.w-s) } }
impl Mul<i32> for IVec4 { type Output=Self; #[inline] fn mul(self,s:i32)->Self { Self::new(self.x*s, self.y*s, self.z*s, self.w*s) } }
impl Div<i32> for IVec4 { type Output=Self; #[inline] fn div(self,s:i32)->Self { Self::new(self.x/s, self.y/s, self.z/s, self.w/s) } }
impl Mul<IVec4> for i32 { type Output=IVec4; #[inline] fn mul(self,v:IVec4)->IVec4 { IVec4::new(self*v.x, self*v.y, self*v.z, self*v.w) } }

impl AddAssign      for IVec4 { #[inline] fn add_assign(&mut self,r:Self) { self.x+=r.x; self.y+=r.y; self.z+=r.z; self.w+=r.w; } }
impl SubAssign      for IVec4 { #[inline] fn sub_assign(&mut self,r:Self) { self.x-=r.x; self.y-=r.y; self.z-=r.z; self.w-=r.w; } }
impl MulAssign      for IVec4 { #[inline] fn mul_assign(&mut self,r:Self) { self.x*=r.x; self.y*=r.y; self.z*=r.z; self.w*=r.w; } }
impl DivAssign      for IVec4 { #[inline] fn div_assign(&mut self,r:Self) { self.x/=r.x; self.y/=r.y; self.z/=r.z; self.w/=r.w; } }
impl AddAssign<i32> for IVec4 { #[inline] fn add_assign(&mut self,s:i32) { self.x+=s; self.y+=s; self.z+=s; self.w+=s; } }
impl SubAssign<i32> for IVec4 { #[inline] fn sub_assign(&mut self,s:i32) { self.x-=s; self.y-=s; self.z-=s; self.w-=s; } }
impl MulAssign<i32> for IVec4 { #[inline] fn mul_assign(&mut self,s:i32) { self.x*=s; self.y*=s; self.z*=s; self.w*=s; } }
impl DivAssign<i32> for IVec4 { #[inline] fn div_assign(&mut self,s:i32) { self.x/=s; self.y/=s; self.z/=s; self.w/=s; } }

impl BitAnd for IVec4 { type Output=Self; #[inline] fn bitand(self,r:Self)->Self { Self::new(self.x&r.x, self.y&r.y, self.z&r.z, self.w&r.w) } }
impl BitOr  for IVec4 { type Output=Self; #[inline] fn bitor (self,r:Self)->Self { Self::new(self.x|r.x, self.y|r.y, self.z|r.z, self.w|r.w) } }
impl BitXor for IVec4 { type Output=Self; #[inline] fn bitxor(self,r:Self)->Self { Self::new(self.x^r.x, self.y^r.y, self.z^r.z, self.w^r.w) } }
impl BitAndAssign for IVec4 { #[inline] fn bitand_assign(&mut self,r:Self) { *self = *self & r; } }
impl BitOrAssign  for IVec4 { #[inline] fn bitor_assign (&mut self,r:Self) { *self = *self | r; } }
impl BitXorAssign for IVec4 { #[inline] fn bitxor_assign(&mut self,r:Self) { *self = *self ^ r; } }
impl Shl<i32> for IVec4 { type Output=Self; #[inline] fn shl(self,s:i32)->Self { Self::new(self.x<<s, self.y<<s, self.z<<s, self.w<<s) } }
impl Shr<i32> for IVec4 { type Output=Self; #[inline] fn shr(self,s:i32)->Self { Self::new(self.x>>s, self.y>>s, self.z>>s, self.w>>s) } }
impl Shl<u32> for IVec4 { type Output=Self; #[inline] fn shl(self,s:u32)->Self { Self::new(self.x<<s, self.y<<s, self.z<<s, self.w<<s) } }
impl Shr<u32> for IVec4 { type Output=Self; #[inline] fn shr(self,s:u32)->Self { Self::new(self.x>>s, self.y>>s, self.z>>s, self.w>>s) } }
impl ShlAssign<u32> for IVec4 { #[inline] fn shl_assign(&mut self,s:u32) { self.x<<=s; self.y<<=s; self.z<<=s; self.w<<=s; } }
impl ShrAssign<u32> for IVec4 { #[inline] fn shr_assign(&mut self,s:u32) { self.x>>=s; self.y>>=s; self.z>>=s; self.w>>=s; } }

impl Index<usize> for IVec4 {
    type Output = i32;
    #[inline] fn index(&self, i: usize) -> &i32 {
        match i { 0=>&self.x, 1=>&self.y, 2=>&self.z, 3=>&self.w, _=>panic!("IVec4 index {i} out of bounds") }
    }
}
impl IndexMut<usize> for IVec4 {
    #[inline] fn index_mut(&mut self, i: usize) -> &mut i32 {
        match i { 0=>&mut self.x, 1=>&mut self.y, 2=>&mut self.z, 3=>&mut self.w, _=>panic!("IVec4 index {i} out of bounds") }
    }
}

impl From<[i32;4]> for IVec4 { #[inline] fn from(a:[i32;4])->Self { Self::from_array(a) } }
impl From<IVec4> for [i32;4] { #[inline] fn from(v:IVec4)->[i32;4] { v.to_array() } }
impl From<(i32,i32,i32,i32)> for IVec4 { #[inline] fn from(t:(i32,i32,i32,i32))->Self { Self::new(t.0,t.1,t.2,t.3) } }
impl From<(IVec3, i32)> for IVec4 { #[inline] fn from((v,w):(IVec3,i32))->Self { Self::new(v.x,v.y,v.z,w) } }
impl From<(IVec2, i32, i32)> for IVec4 { #[inline] fn from((v,z,w):(IVec2,i32,i32))->Self { Self::new(v.x,v.y,z,w) } }
impl From<(IVec2, IVec2)> for IVec4 { #[inline] fn from((a,b):(IVec2,IVec2))->Self { Self::new(a.x,a.y,b.x,b.y) } }

impl fmt::Debug for IVec4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "IVec4({}, {}, {}, {})", self.x, self.y, self.z, self.w)
    }
}
impl fmt::Display for IVec4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}, {}, {}, {}]", self.x, self.y, self.z, self.w)
    }
}
