// crates/mid-math/src/int8/u8vec2.rs
//! 2D unsigned 8-bit integer vector. 2 bytes, align 1. Always scalar.

use core::fmt;
use core::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign,
    Div, DivAssign, Index, IndexMut, Mul, MulAssign, Not,
    Rem, RemAssign, Shl, ShlAssign, Shr, ShrAssign, Sub, SubAssign,
};
use crate::{BVec2, I8Vec2, U8Vec3};

/// 2D unsigned 8-bit integer vector. 2 bytes, align 1.
///
/// Used for packed RGBA8 color channels, UV coordinates in 0-255 range,
/// and any 2D unsigned byte domain.
/// Dot product returns u16 to prevent overflow.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(C)]
pub struct U8Vec2 {
    pub x: u8,
    pub y: u8,
}

impl U8Vec2 {
    pub const ZERO: Self = Self::splat(0);
    pub const ONE:  Self = Self::splat(1);
    pub const MIN:  Self = Self::splat(u8::MIN);
    pub const MAX:  Self = Self::splat(u8::MAX);
    pub const X:    Self = Self::new(1, 0);
    pub const Y:    Self = Self::new(0, 1);

    #[inline(always)] pub const fn new(x: u8, y: u8) -> Self { Self { x, y } }
    #[inline(always)] pub const fn splat(v: u8) -> Self { Self { x: v, y: v } }
    #[inline(always)] pub const fn from_array(a: [u8; 2]) -> Self { Self::new(a[0], a[1]) }
    #[inline(always)] pub const fn to_array(self) -> [u8; 2] { [self.x, self.y] }
    #[inline(always)] pub const fn extend(self, z: u8) -> U8Vec3 { U8Vec3::new(self.x, self.y, z) }

    #[inline]
    pub fn select(mask: BVec2, if_true: Self, if_false: Self) -> Self {
        Self::new(
            if mask.x { if_true.x } else { if_false.x },
            if mask.y { if_true.y } else { if_false.y },
        )
    }

    /// Dot product, widened to u16.
    #[inline] pub fn dot(self, rhs: Self) -> u16 {
        (self.x as u16)*(rhs.x as u16) + (self.y as u16)*(rhs.y as u16)
    }

    #[inline] pub fn length_sq(self) -> u16 { self.dot(self) }
    #[inline] pub fn distance_sq(self, rhs: Self) -> u16 {
        let dx = (self.x as i16) - (rhs.x as i16);
        let dy = (self.y as i16) - (rhs.y as i16);
        (dx*dx + dy*dy) as u16
    }

    #[inline] pub fn min(self,r:Self)->Self{Self::new(self.x.min(r.x),self.y.min(r.y))}
    #[inline] pub fn max(self,r:Self)->Self{Self::new(self.x.max(r.x),self.y.max(r.y))}
    #[inline] pub fn clamp(self,lo:Self,hi:Self)->Self{self.max(lo).min(hi)}
    #[inline] pub fn min_element(self)->u8{self.x.min(self.y)}
    #[inline] pub fn max_element(self)->u8{self.x.max(self.y)}
    #[inline] pub fn element_sum(self)->u8{self.x.wrapping_add(self.y)}

    #[inline] pub fn cmpeq(self,r:Self)->BVec2{BVec2::new(self.x==r.x,self.y==r.y)}
    #[inline] pub fn cmpne(self,r:Self)->BVec2{BVec2::new(self.x!=r.x,self.y!=r.y)}
    #[inline] pub fn cmpge(self,r:Self)->BVec2{BVec2::new(self.x>=r.x,self.y>=r.y)}
    #[inline] pub fn cmpgt(self,r:Self)->BVec2{BVec2::new(self.x>r.x, self.y>r.y)}
    #[inline] pub fn cmple(self,r:Self)->BVec2{BVec2::new(self.x<=r.x,self.y<=r.y)}
    #[inline] pub fn cmplt(self,r:Self)->BVec2{BVec2::new(self.x<r.x, self.y<r.y)}

    #[inline] pub fn wrapping_add(self,r:Self)->Self{Self::new(self.x.wrapping_add(r.x),self.y.wrapping_add(r.y))}
    #[inline] pub fn wrapping_sub(self,r:Self)->Self{Self::new(self.x.wrapping_sub(r.x),self.y.wrapping_sub(r.y))}
    #[inline] pub fn wrapping_mul(self,r:Self)->Self{Self::new(self.x.wrapping_mul(r.x),self.y.wrapping_mul(r.y))}
    #[inline] pub fn saturating_add(self,r:Self)->Self{Self::new(self.x.saturating_add(r.x),self.y.saturating_add(r.y))}
    #[inline] pub fn saturating_sub(self,r:Self)->Self{Self::new(self.x.saturating_sub(r.x),self.y.saturating_sub(r.y))}
    #[inline] pub fn checked_add(self,r:Self)->Option<Self>{Some(Self::new(self.x.checked_add(r.x)?,self.y.checked_add(r.y)?))}
    #[inline] pub fn checked_sub(self,r:Self)->Option<Self>{Some(Self::new(self.x.checked_sub(r.x)?,self.y.checked_sub(r.y)?))}
    #[inline] pub fn checked_mul(self,r:Self)->Option<Self>{Some(Self::new(self.x.checked_mul(r.x)?,self.y.checked_mul(r.y)?))}

    #[inline] pub fn as_i8vec2(self)  -> I8Vec2          { I8Vec2::new(self.x as i8, self.y as i8) }
    #[inline] pub fn as_i16vec2(self) -> crate::I16Vec2  { crate::I16Vec2::new(self.x as i16, self.y as i16) }
    #[inline] pub fn as_u16vec2(self) -> crate::U16Vec2  { crate::U16Vec2::new(self.x as u16, self.y as u16) }
    #[inline] pub fn as_ivec2(self)   -> crate::IVec2    { crate::IVec2::new(self.x as i32, self.y as i32) }
    #[inline] pub fn as_uvec2(self)   -> crate::UVec2    { crate::UVec2::new(self.x as u32, self.y as u32) }
    #[inline] pub fn as_vec2(self)    -> crate::Vec2     { crate::Vec2::new(self.x as f32, self.y as f32) }
    #[inline] pub fn as_dvec2(self)   -> crate::DVec2    { crate::DVec2::new(self.x as f64, self.y as f64) }
}

impl Add  for U8Vec2{type Output=Self;#[inline]fn add(self,r:Self)->Self{Self::new(self.x.wrapping_add(r.x),self.y.wrapping_add(r.y))}}
impl Sub  for U8Vec2{type Output=Self;#[inline]fn sub(self,r:Self)->Self{Self::new(self.x.wrapping_sub(r.x),self.y.wrapping_sub(r.y))}}
impl Mul  for U8Vec2{type Output=Self;#[inline]fn mul(self,r:Self)->Self{Self::new(self.x.wrapping_mul(r.x),self.y.wrapping_mul(r.y))}}
impl Div  for U8Vec2{type Output=Self;#[inline]fn div(self,r:Self)->Self{Self::new(self.x/r.x,self.y/r.y)}}
impl Rem  for U8Vec2{type Output=Self;#[inline]fn rem(self,r:Self)->Self{Self::new(self.x%r.x,self.y%r.y)}}
impl Not  for U8Vec2{type Output=Self;#[inline]fn not(self)->Self{Self::new(!self.x,!self.y)}}

impl Mul<u8> for U8Vec2{type Output=Self;#[inline]fn mul(self,s:u8)->Self{Self::new(self.x.wrapping_mul(s),self.y.wrapping_mul(s))}}
impl Mul<U8Vec2> for u8{type Output=U8Vec2;#[inline]fn mul(self,v:U8Vec2)->U8Vec2{U8Vec2::new(self.wrapping_mul(v.x),self.wrapping_mul(v.y))}}

impl AddAssign for U8Vec2{#[inline]fn add_assign(&mut self,r:Self){*self=*self+r;}}
impl SubAssign for U8Vec2{#[inline]fn sub_assign(&mut self,r:Self){*self=*self-r;}}
impl MulAssign for U8Vec2{#[inline]fn mul_assign(&mut self,r:Self){*self=*self*r;}}
impl DivAssign for U8Vec2{#[inline]fn div_assign(&mut self,r:Self){self.x/=r.x;self.y/=r.y;}}
impl RemAssign for U8Vec2{#[inline]fn rem_assign(&mut self,r:Self){self.x%=r.x;self.y%=r.y;}}
impl MulAssign<u8> for U8Vec2{#[inline]fn mul_assign(&mut self,s:u8){*self=*self*s;}}

impl BitAnd for U8Vec2{type Output=Self;#[inline]fn bitand(self,r:Self)->Self{Self::new(self.x&r.x,self.y&r.y)}}
impl BitOr  for U8Vec2{type Output=Self;#[inline]fn bitor (self,r:Self)->Self{Self::new(self.x|r.x,self.y|r.y)}}
impl BitXor for U8Vec2{type Output=Self;#[inline]fn bitxor(self,r:Self)->Self{Self::new(self.x^r.x,self.y^r.y)}}
impl BitAndAssign for U8Vec2{#[inline]fn bitand_assign(&mut self,r:Self){*self=*self&r;}}
impl BitOrAssign  for U8Vec2{#[inline]fn bitor_assign (&mut self,r:Self){*self=*self|r;}}
impl BitXorAssign for U8Vec2{#[inline]fn bitxor_assign(&mut self,r:Self){*self=*self^r;}}
impl Shl<u32> for U8Vec2{type Output=Self;#[inline]fn shl(self,s:u32)->Self{Self::new(self.x<<s,self.y<<s)}}
impl Shr<u32> for U8Vec2{type Output=Self;#[inline]fn shr(self,s:u32)->Self{Self::new(self.x>>s,self.y>>s)}}
impl ShlAssign<u32> for U8Vec2{#[inline]fn shl_assign(&mut self,s:u32){self.x<<=s;self.y<<=s;}}
impl ShrAssign<u32> for U8Vec2{#[inline]fn shr_assign(&mut self,s:u32){self.x>>=s;self.y>>=s;}}

impl Index<usize> for U8Vec2 {
    type Output=u8;
    #[inline] fn index(&self,i:usize)->&u8{match i{0=>&self.x,1=>&self.y,_=>panic!("U8Vec2 index {i} out of bounds")}}
}
impl IndexMut<usize> for U8Vec2 {
    #[inline] fn index_mut(&mut self,i:usize)->&mut u8{match i{0=>&mut self.x,1=>&mut self.y,_=>panic!("U8Vec2 index {i} out of bounds")}}
}

impl From<[u8;2]> for U8Vec2{#[inline]fn from(a:[u8;2])->Self{Self::from_array(a)}}
impl From<U8Vec2> for [u8;2]{#[inline]fn from(v:U8Vec2)->[u8;2]{v.to_array()}}
impl From<(u8,u8)> for U8Vec2{#[inline]fn from(t:(u8,u8))->Self{Self::new(t.0,t.1)}}

impl fmt::Debug for U8Vec2{fn fmt(&self,f:&mut fmt::Formatter<'_>)->fmt::Result{write!(f,"U8Vec2({}, {})",self.x,self.y)}}
impl fmt::Display for U8Vec2{fn fmt(&self,f:&mut fmt::Formatter<'_>)->fmt::Result{write!(f,"[{}, {}]",self.x,self.y)}}
