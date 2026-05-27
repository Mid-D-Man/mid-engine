// crates/mid-math/src/int8/i8vec4.rs
//! 4D signed 8-bit integer vector. 4 bytes, align 1. Always scalar.

use core::fmt;
use core::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign,
    Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Not,
    Rem, RemAssign, Shl, ShlAssign, Shr, ShrAssign, Sub, SubAssign,
};
use crate::{BVec4, I8Vec2, I8Vec3, U8Vec4};

/// 4D signed 8-bit integer vector. 4 bytes, align 1.
///
/// Commonly used for packed RGBA8 signed normals, bone weights, etc.
/// Dot product returns i16 (widened) to prevent overflow.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(C)]
pub struct I8Vec4 {
    pub x: i8,
    pub y: i8,
    pub z: i8,
    pub w: i8,
}

impl I8Vec4 {
    pub const ZERO:    Self = Self::splat(0);
    pub const ONE:     Self = Self::splat(1);
    pub const NEG_ONE: Self = Self::splat(-1);
    pub const MIN:     Self = Self::splat(i8::MIN);
    pub const MAX:     Self = Self::splat(i8::MAX);
    pub const X:       Self = Self::new(1, 0, 0, 0);
    pub const Y:       Self = Self::new(0, 1, 0, 0);
    pub const Z:       Self = Self::new(0, 0, 1, 0);
    pub const W:       Self = Self::new(0, 0, 0, 1);

    #[inline(always)] pub const fn new(x: i8, y: i8, z: i8, w: i8) -> Self { Self { x, y, z, w } }
    #[inline(always)] pub const fn splat(v: i8) -> Self { Self { x: v, y: v, z: v, w: v } }
    #[inline(always)] pub const fn from_array(a: [i8; 4]) -> Self { Self::new(a[0], a[1], a[2], a[3]) }
    #[inline(always)] pub const fn to_array(self) -> [i8; 4] { [self.x, self.y, self.z, self.w] }
    #[inline(always)] pub const fn truncate(self) -> I8Vec3 { I8Vec3::new(self.x, self.y, self.z) }
    #[inline(always)] pub const fn xy(self) -> I8Vec2 { I8Vec2::new(self.x, self.y) }

    #[inline]
    pub fn select(mask: BVec4, if_true: Self, if_false: Self) -> Self {
        Self::new(
            if mask.x { if_true.x } else { if_false.x },
            if mask.y { if_true.y } else { if_false.y },
            if mask.z { if_true.z } else { if_false.z },
            if mask.w { if_true.w } else { if_false.w },
        )
    }

    /// Dot product, widened to i16.
    #[inline] pub fn dot(self, rhs: Self) -> i16 {
        (self.x as i16)*(rhs.x as i16) + (self.y as i16)*(rhs.y as i16)
        + (self.z as i16)*(rhs.z as i16) + (self.w as i16)*(rhs.w as i16)
    }

    #[inline] pub fn length_sq(self) -> i16 { self.dot(self) }
    #[inline] pub fn distance_sq(self, rhs: Self) -> i16 { (self - rhs).length_sq() }
    #[inline] pub fn abs(self) -> Self { Self::new(self.x.abs(), self.y.abs(), self.z.abs(), self.w.abs()) }
    #[inline] pub fn signum(self) -> Self { Self::new(self.x.signum(), self.y.signum(), self.z.signum(), self.w.signum()) }
    #[inline] pub fn wrapping_abs(self) -> Self { Self::new(self.x.wrapping_abs(), self.y.wrapping_abs(), self.z.wrapping_abs(), self.w.wrapping_abs()) }

    #[inline] pub fn min(self,r:Self)->Self{Self::new(self.x.min(r.x),self.y.min(r.y),self.z.min(r.z),self.w.min(r.w))}
    #[inline] pub fn max(self,r:Self)->Self{Self::new(self.x.max(r.x),self.y.max(r.y),self.z.max(r.z),self.w.max(r.w))}
    #[inline] pub fn clamp(self,lo:Self,hi:Self)->Self{self.max(lo).min(hi)}
    #[inline] pub fn min_element(self)->i8{self.x.min(self.y).min(self.z).min(self.w)}
    #[inline] pub fn max_element(self)->i8{self.x.max(self.y).max(self.z).max(self.w)}
    #[inline] pub fn element_sum(self)->i8{self.x.wrapping_add(self.y).wrapping_add(self.z).wrapping_add(self.w)}

    #[inline] pub fn cmpeq(self,r:Self)->BVec4{BVec4::new(self.x==r.x,self.y==r.y,self.z==r.z,self.w==r.w)}
    #[inline] pub fn cmpne(self,r:Self)->BVec4{BVec4::new(self.x!=r.x,self.y!=r.y,self.z!=r.z,self.w!=r.w)}
    #[inline] pub fn cmpge(self,r:Self)->BVec4{BVec4::new(self.x>=r.x,self.y>=r.y,self.z>=r.z,self.w>=r.w)}
    #[inline] pub fn cmpgt(self,r:Self)->BVec4{BVec4::new(self.x>r.x, self.y>r.y, self.z>r.z, self.w>r.w)}
    #[inline] pub fn cmple(self,r:Self)->BVec4{BVec4::new(self.x<=r.x,self.y<=r.y,self.z<=r.z,self.w<=r.w)}
    #[inline] pub fn cmplt(self,r:Self)->BVec4{BVec4::new(self.x<r.x, self.y<r.y, self.z<r.z, self.w<r.w)}

    #[inline] pub fn wrapping_add(self,r:Self)->Self{Self::new(self.x.wrapping_add(r.x),self.y.wrapping_add(r.y),self.z.wrapping_add(r.z),self.w.wrapping_add(r.w))}
    #[inline] pub fn wrapping_sub(self,r:Self)->Self{Self::new(self.x.wrapping_sub(r.x),self.y.wrapping_sub(r.y),self.z.wrapping_sub(r.z),self.w.wrapping_sub(r.w))}
    #[inline] pub fn wrapping_mul(self,r:Self)->Self{Self::new(self.x.wrapping_mul(r.x),self.y.wrapping_mul(r.y),self.z.wrapping_mul(r.z),self.w.wrapping_mul(r.w))}
    #[inline] pub fn wrapping_neg(self)->Self{Self::new(self.x.wrapping_neg(),self.y.wrapping_neg(),self.z.wrapping_neg(),self.w.wrapping_neg())}
    #[inline] pub fn saturating_add(self,r:Self)->Self{Self::new(self.x.saturating_add(r.x),self.y.saturating_add(r.y),self.z.saturating_add(r.z),self.w.saturating_add(r.w))}
    #[inline] pub fn saturating_sub(self,r:Self)->Self{Self::new(self.x.saturating_sub(r.x),self.y.saturating_sub(r.y),self.z.saturating_sub(r.z),self.w.saturating_sub(r.w))}
    #[inline] pub fn checked_add(self,r:Self)->Option<Self>{Some(Self::new(self.x.checked_add(r.x)?,self.y.checked_add(r.y)?,self.z.checked_add(r.z)?,self.w.checked_add(r.w)?))}
    #[inline] pub fn checked_sub(self,r:Self)->Option<Self>{Some(Self::new(self.x.checked_sub(r.x)?,self.y.checked_sub(r.y)?,self.z.checked_sub(r.z)?,self.w.checked_sub(r.w)?))}
    #[inline] pub fn checked_mul(self,r:Self)->Option<Self>{Some(Self::new(self.x.checked_mul(r.x)?,self.y.checked_mul(r.y)?,self.z.checked_mul(r.z)?,self.w.checked_mul(r.w)?))}

    #[inline] pub fn as_u8vec4(self)  -> U8Vec4          { U8Vec4::new(self.x as u8, self.y as u8, self.z as u8, self.w as u8) }
    #[inline] pub fn as_i16vec4(self) -> crate::I16Vec4  { crate::I16Vec4::new(self.x as i16, self.y as i16, self.z as i16, self.w as i16) }
    #[inline] pub fn as_u16vec4(self) -> crate::U16Vec4  { crate::U16Vec4::new(self.x as u16, self.y as u16, self.z as u16, self.w as u16) }
    #[inline] pub fn as_ivec4(self)   -> crate::IVec4    { crate::IVec4::new(self.x as i32, self.y as i32, self.z as i32, self.w as i32) }
    #[inline] pub fn as_uvec4(self)   -> crate::UVec4    { crate::UVec4::new(self.x as u32, self.y as u32, self.z as u32, self.w as u32) }
    #[inline] pub fn as_vec4(self)    -> crate::Vec4     { crate::Vec4::new(self.x as f32, self.y as f32, self.z as f32, self.w as f32) }
    #[inline] pub fn as_dvec4(self)   -> crate::DVec4    { crate::DVec4::new(self.x as f64, self.y as f64, self.z as f64, self.w as f64) }
}

impl Add  for I8Vec4{type Output=Self;#[inline]fn add(self,r:Self)->Self{Self::new(self.x.wrapping_add(r.x),self.y.wrapping_add(r.y),self.z.wrapping_add(r.z),self.w.wrapping_add(r.w))}}
impl Sub  for I8Vec4{type Output=Self;#[inline]fn sub(self,r:Self)->Self{Self::new(self.x.wrapping_sub(r.x),self.y.wrapping_sub(r.y),self.z.wrapping_sub(r.z),self.w.wrapping_sub(r.w))}}
impl Mul  for I8Vec4{type Output=Self;#[inline]fn mul(self,r:Self)->Self{Self::new(self.x.wrapping_mul(r.x),self.y.wrapping_mul(r.y),self.z.wrapping_mul(r.z),self.w.wrapping_mul(r.w))}}
impl Div  for I8Vec4{type Output=Self;#[inline]fn div(self,r:Self)->Self{Self::new(self.x/r.x,self.y/r.y,self.z/r.z,self.w/r.w)}}
impl Rem  for I8Vec4{type Output=Self;#[inline]fn rem(self,r:Self)->Self{Self::new(self.x%r.x,self.y%r.y,self.z%r.z,self.w%r.w)}}
impl Neg  for I8Vec4{type Output=Self;#[inline]fn neg(self)->Self{Self::new(self.x.wrapping_neg(),self.y.wrapping_neg(),self.z.wrapping_neg(),self.w.wrapping_neg())}}
impl Not  for I8Vec4{type Output=Self;#[inline]fn not(self)->Self{Self::new(!self.x,!self.y,!self.z,!self.w)}}

impl Mul<i8> for I8Vec4{type Output=Self;#[inline]fn mul(self,s:i8)->Self{Self::new(self.x.wrapping_mul(s),self.y.wrapping_mul(s),self.z.wrapping_mul(s),self.w.wrapping_mul(s))}}
impl Mul<I8Vec4> for i8{type Output=I8Vec4;#[inline]fn mul(self,v:I8Vec4)->I8Vec4{I8Vec4::new(self.wrapping_mul(v.x),self.wrapping_mul(v.y),self.wrapping_mul(v.z),self.wrapping_mul(v.w))}}

impl AddAssign for I8Vec4{#[inline]fn add_assign(&mut self,r:Self){*self=*self+r;}}
impl SubAssign for I8Vec4{#[inline]fn sub_assign(&mut self,r:Self){*self=*self-r;}}
impl MulAssign for I8Vec4{#[inline]fn mul_assign(&mut self,r:Self){*self=*self*r;}}
impl DivAssign for I8Vec4{#[inline]fn div_assign(&mut self,r:Self){self.x/=r.x;self.y/=r.y;self.z/=r.z;self.w/=r.w;}}
impl RemAssign for I8Vec4{#[inline]fn rem_assign(&mut self,r:Self){self.x%=r.x;self.y%=r.y;self.z%=r.z;self.w%=r.w;}}
impl MulAssign<i8> for I8Vec4{#[inline]fn mul_assign(&mut self,s:i8){*self=*self*s;}}

impl BitAnd for I8Vec4{type Output=Self;#[inline]fn bitand(self,r:Self)->Self{Self::new(self.x&r.x,self.y&r.y,self.z&r.z,self.w&r.w)}}
impl BitOr  for I8Vec4{type Output=Self;#[inline]fn bitor (self,r:Self)->Self{Self::new(self.x|r.x,self.y|r.y,self.z|r.z,self.w|r.w)}}
impl BitXor for I8Vec4{type Output=Self;#[inline]fn bitxor(self,r:Self)->Self{Self::new(self.x^r.x,self.y^r.y,self.z^r.z,self.w^r.w)}}
impl BitAndAssign for I8Vec4{#[inline]fn bitand_assign(&mut self,r:Self){*self=*self&r;}}
impl BitOrAssign  for I8Vec4{#[inline]fn bitor_assign (&mut self,r:Self){*self=*self|r;}}
impl BitXorAssign for I8Vec4{#[inline]fn bitxor_assign(&mut self,r:Self){*self=*self^r;}}
impl Shl<u32> for I8Vec4{type Output=Self;#[inline]fn shl(self,s:u32)->Self{Self::new(self.x<<s,self.y<<s,self.z<<s,self.w<<s)}}
impl Shr<u32> for I8Vec4{type Output=Self;#[inline]fn shr(self,s:u32)->Self{Self::new(self.x>>s,self.y>>s,self.z>>s,self.w>>s)}}
impl ShlAssign<u32> for I8Vec4{#[inline]fn shl_assign(&mut self,s:u32){self.x<<=s;self.y<<=s;self.z<<=s;self.w<<=s;}}
impl ShrAssign<u32> for I8Vec4{#[inline]fn shr_assign(&mut self,s:u32){self.x>>=s;self.y>>=s;self.z>>=s;self.w>>=s;}}

impl Index<usize> for I8Vec4 {
    type Output=i8;
    #[inline] fn index(&self,i:usize)->&i8{match i{0=>&self.x,1=>&self.y,2=>&self.z,3=>&self.w,_=>panic!("I8Vec4 index {i} out of bounds")}}
}
impl IndexMut<usize> for I8Vec4 {
    #[inline] fn index_mut(&mut self,i:usize)->&mut i8{match i{0=>&mut self.x,1=>&mut self.y,2=>&mut self.z,3=>&mut self.w,_=>panic!("I8Vec4 index {i} out of bounds")}}
}

impl From<[i8;4]> for I8Vec4{#[inline]fn from(a:[i8;4])->Self{Self::from_array(a)}}
impl From<I8Vec4> for [i8;4]{#[inline]fn from(v:I8Vec4)->[i8;4]{v.to_array()}}
impl From<(i8,i8,i8,i8)> for I8Vec4{#[inline]fn from(t:(i8,i8,i8,i8))->Self{Self::new(t.0,t.1,t.2,t.3)}}
impl From<(I8Vec3,i8)> for I8Vec4{#[inline]fn from((v,w):(I8Vec3,i8))->Self{Self::new(v.x,v.y,v.z,w)}}
impl From<(I8Vec2,i8,i8)> for I8Vec4{#[inline]fn from((v,z,w):(I8Vec2,i8,i8))->Self{Self::new(v.x,v.y,z,w)}}

impl fmt::Debug for I8Vec4{fn fmt(&self,f:&mut fmt::Formatter<'_>)->fmt::Result{write!(f,"I8Vec4({}, {}, {}, {})",self.x,self.y,self.z,self.w)}}
impl fmt::Display for I8Vec4{fn fmt(&self,f:&mut fmt::Formatter<'_>)->fmt::Result{write!(f,"[{}, {}, {}, {}]",self.x,self.y,self.z,self.w)}}
