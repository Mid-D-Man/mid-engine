// crates/mid-math/src/int16/u16vec2.rs
//! 2D unsigned 16-bit integer vector. 4 bytes, align 2. Always scalar.

use core::fmt;
use core::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign,
    Div, DivAssign, Index, IndexMut, Mul, MulAssign, Not,
    Rem, RemAssign, Shl, ShlAssign, Shr, ShrAssign, Sub, SubAssign,
};
use crate::{BVec2, I16Vec2, U16Vec3};

/// 2D unsigned 16-bit integer vector. 4 bytes, align 2.
///
/// Used for texture dimensions, 16-bit UV coordinates, audio sample indices.
/// Dot product returns u32 to prevent overflow.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(C)]
pub struct U16Vec2 {
    pub x: u16,
    pub y: u16,
}

impl U16Vec2 {
    pub const ZERO: Self = Self::splat(0);
    pub const ONE:  Self = Self::splat(1);
    pub const MIN:  Self = Self::splat(u16::MIN);
    pub const MAX:  Self = Self::splat(u16::MAX);
    pub const X:    Self = Self::new(1, 0);
    pub const Y:    Self = Self::new(0, 1);

    #[inline(always)] pub const fn new(x: u16, y: u16) -> Self { Self { x, y } }
    #[inline(always)] pub const fn splat(v: u16) -> Self { Self { x: v, y: v } }
    #[inline(always)] pub const fn from_array(a: [u16; 2]) -> Self { Self::new(a[0], a[1]) }
    #[inline(always)] pub const fn to_array(self) -> [u16; 2] { [self.x, self.y] }
    #[inline(always)] pub const fn extend(self, z: u16) -> U16Vec3 { U16Vec3::new(self.x, self.y, z) }

    #[inline]
    pub fn select(mask: BVec2, if_true: Self, if_false: Self) -> Self {
        Self::new(
            if mask.x { if_true.x } else { if_false.x },
            if mask.y { if_true.y } else { if_false.y },
        )
    }

    /// Dot product, widened to u32.
    #[inline] pub fn dot(self, rhs: Self) -> u32 {
        (self.x as u32)*(rhs.x as u32) + (self.y as u32)*(rhs.y as u32)
    }

    #[inline] pub fn length_sq(self) -> u32 { self.dot(self) }
    #[inline] pub fn distance_sq(self, rhs: Self) -> u32 {
        let dx = (self.x as i32) - (rhs.x as i32);
        let dy = (self.y as i32) - (rhs.y as i32);
        (dx*dx + dy*dy) as u32
    }

    #[inline] pub fn min(self,r:Self)->Self{Self::new(self.x.min(r.x),self.y.min(r.y))}
    #[inline] pub fn max(self,r:Self)->Self{Self::new(self.x.max(r.x),self.y.max(r.y))}
    #[inline] pub fn clamp(self,lo:Self,hi:Self)->Self{self.max(lo).min(hi)}
    #[inline] pub fn min_element(self)->u16{self.x.min(self.y)}
    #[inline] pub fn max_element(self)->u16{self.x.max(self.y)}
    #[inline] pub fn element_sum(self)->u16{self.x.wrapping_add(self.y)}

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

    #[inline] pub fn as_i16vec2(self) -> I16Vec2         { I16Vec2::new(self.x as i16, self.y as i16) }
    #[inline] pub fn as_i8vec2(self)  -> crate::I8Vec2   { crate::I8Vec2::new(self.x as i8, self.y as i8) }
    #[inline] pub fn as_u8vec2(self)  -> crate::U8Vec2   { crate::U8Vec2::new(self.x as u8, self.y as u8) }
    #[inline] pub fn as_ivec2(self)   -> crate::IVec2    { crate::IVec2::new(self.x as i32, self.y as i32) }
    #[inline] pub fn as_uvec2(self)   -> crate::UVec2    { crate::UVec2::new(self.x as u32, self.y as u32) }
    #[inline] pub fn as_u64vec2(self) -> crate::U64Vec2  { crate::U64Vec2::new(self.x as u64, self.y as u64) }
    #[inline] pub fn as_vec2(self)    -> crate::Vec2     { crate::Vec2::new(self.x as f32, self.y as f32) }
    #[inline] pub fn as_dvec2(self)   -> crate::DVec2    { crate::DVec2::new(self.x as f64, self.y as f64) }
}

impl Add  for U16Vec2{type Output=Self;#[inline]fn add(self,r:Self)->Self{Self::new(self.x.wrapping_add(r.x),self.y.wrapping_add(r.y))}}
impl Sub  for U16Vec2{type Output=Self;#[inline]fn sub(self,r:Self)->Self{Self::new(self.x.wrapping_sub(r.x),self.y.wrapping_sub(r.y))}}
impl Mul  for U16Vec2{type Output=Self;#[inline]fn mul(self,r:Self)->Self{Self::new(self.x.wrapping_mul(r.x),self.y.wrapping_mul(r.y))}}
impl Div  for U16Vec2{type Output=Self;#[inline]fn div(self,r:Self)->Self{Self::new(self.x/r.x,self.y/r.y)}}
impl Rem  for U16Vec2{type Output=Self;#[inline]fn rem(self,r:Self)->Self{Self::new(self.x%r.x,self.y%r.y)}}
impl Not  for U16Vec2{type Output=Self;#[inline]fn not(self)->Self{Self::new(!self.x,!self.y)}}

impl Mul<u16> for U16Vec2{type Output=Self;#[inline]fn mul(self,s:u16)->Self{Self::new(self.x.wrapping_mul(s),self.y.wrapping_mul(s))}}
impl Mul<U16Vec2> for u16{type Output=U16Vec2;#[inline]fn mul(self,v:U16Vec2)->U16Vec2{U16Vec2::new(self.wrapping_mul(v.x),self.wrapping_mul(v.y))}}

impl AddAssign for U16Vec2{#[inline]fn add_assign(&mut self,r:Self){*self=*self+r;}}
impl SubAssign for U16Vec2{#[inline]fn sub_assign(&mut self,r:Self){*self=*self-r;}}
impl MulAssign for U16Vec2{#[inline]fn mul_assign(&mut self,r:Self){*self=*self*r;}}
impl DivAssign for U16Vec2{#[inline]fn div_assign(&mut self,r:Self){self.x/=r.x;self.y/=r.y;}}
impl RemAssign for U16Vec2{#[inline]fn rem_assign(&mut self,r:Self){self.x%=r.x;self.y%=r.y;}}
impl MulAssign<u16> for U16Vec2{#[inline]fn mul_assign(&mut self,s:u16){*self=*self*s;}}

impl BitAnd for U16Vec2{type Output=Self;#[inline]fn bitand(self,r:Self)->Self{Self::new(self.x&r.x,self.y&r.y)}}
impl BitOr  for U16Vec2{type Output=Self;#[inline]fn bitor (self,r:Self)->Self{Self::new(self.x|r.x,self.y|r.y)}}
impl BitXor for U16Vec2{type Output=Self;#[inline]fn bitxor(self,r:Self)->Self{Self::new(self.x^r.x,self.y^r.y)}}
impl BitAndAssign for U16Vec2{#[inline]fn bitand_assign(&mut self,r:Self){*self=*self&r;}}
impl BitOrAssign  for U16Vec2{#[inline]fn bitor_assign (&mut self,r:Self){*self=*self|r;}}
impl BitXorAssign for U16Vec2{#[inline]fn bitxor_assign(&mut self,r:Self){*self=*self^r;}}
impl Shl<u32> for U16Vec2{type Output=Self;#[inline]fn shl(self,s:u32)->Self{Self::new(self.x<<s,self.y<<s)}}
impl Shr<u32> for U16Vec2{type Output=Self;#[inline]fn shr(self,s:u32)->Self{Self::new(self.x>>s,self.y>>s)}}
impl ShlAssign<u32> for U16Vec2{#[inline]fn shl_assign(&mut self,s:u32){self.x<<=s;self.y<<=s;}}
impl ShrAssign<u32> for U16Vec2{#[inline]fn shr_assign(&mut self,s:u32){self.x>>=s;self.y>>=s;}}

impl Index<usize> for U16Vec2{type Output=u16;#[inline]fn index(&self,i:usize)->&u16{match i{0=>&self.x,1=>&self.y,_=>panic!("U16Vec2 index {i} out of bounds")}}}
impl IndexMut<usize> for U16Vec2{#[inline]fn index_mut(&mut self,i:usize)->&mut u16{match i{0=>&mut self.x,1=>&mut self.y,_=>panic!("U16Vec2 index {i} out of bounds")}}}

impl From<[u16;2]> for U16Vec2{#[inline]fn from(a:[u16;2])->Self{Self::from_array(a)}}
impl From<U16Vec2> for [u16;2]{#[inline]fn from(v:U16Vec2)->[u16;2]{v.to_array()}}
impl From<(u16,u16)> for U16Vec2{#[inline]fn from(t:(u16,u16))->Self{Self::new(t.0,t.1)}}

impl fmt::Debug for U16Vec2{fn fmt(&self,f:&mut fmt::Formatter<'_>)->fmt::Result{write!(f,"U16Vec2({}, {})",self.x,self.y)}}
impl fmt::Display for U16Vec2{fn fmt(&self,f:&mut fmt::Formatter<'_>)->fmt::Result{write!(f,"[{}, {}]",self.x,self.y)}}
