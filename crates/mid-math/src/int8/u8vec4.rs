// crates/mid-math/src/int8/u8vec4.rs
//! 4D unsigned 8-bit integer vector. 4 bytes, align 1. Always scalar.

use core::fmt;
use core::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign,
    Div, DivAssign, Index, IndexMut, Mul, MulAssign, Not,
    Rem, RemAssign, Shl, ShlAssign, Shr, ShrAssign, Sub, SubAssign,
};
use crate::{BVec4, I8Vec4, U8Vec2, U8Vec3};

/// 4D unsigned 8-bit integer vector. 4 bytes, align 1.
///
/// Primary use: RGBA8 color. Also used for 4-component bone weights
/// in [0,255] range before normalisation to f32.
/// Dot product returns u16 to prevent overflow.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(C)]
pub struct U8Vec4 {
    pub x: u8,
    pub y: u8,
    pub z: u8,
    pub w: u8,
}

impl U8Vec4 {
    pub const ZERO: Self = Self::splat(0);
    pub const ONE:  Self = Self::splat(1);
    pub const MIN:  Self = Self::splat(u8::MIN);
    pub const MAX:  Self = Self::splat(u8::MAX);
    pub const X:    Self = Self::new(1, 0, 0, 0);
    pub const Y:    Self = Self::new(0, 1, 0, 0);
    pub const Z:    Self = Self::new(0, 0, 1, 0);
    pub const W:    Self = Self::new(0, 0, 0, 1);

    /// RGBA convenience constant.
    pub const BLACK_OPAQUE: Self = Self::new(0, 0, 0, 255);
    pub const WHITE_OPAQUE: Self = Self::new(255, 255, 255, 255);

    #[inline(always)] pub const fn new(x: u8, y: u8, z: u8, w: u8) -> Self { Self { x, y, z, w } }
    #[inline(always)] pub const fn splat(v: u8) -> Self { Self { x: v, y: v, z: v, w: v } }
    #[inline(always)] pub const fn from_array(a: [u8; 4]) -> Self { Self::new(a[0], a[1], a[2], a[3]) }
    #[inline(always)] pub const fn to_array(self) -> [u8; 4] { [self.x, self.y, self.z, self.w] }
    #[inline(always)] pub const fn truncate(self) -> U8Vec3 { U8Vec3::new(self.x, self.y, self.z) }
    #[inline(always)] pub const fn xy(self) -> U8Vec2 { U8Vec2::new(self.x, self.y) }

    /// Build from a packed u32 in RGBA byte order (x=R, y=G, z=B, w=A).
    #[inline(always)] pub const fn from_rgba_u32(v: u32) -> Self {
        Self::new((v >> 24) as u8, (v >> 16) as u8, (v >> 8) as u8, v as u8)
    }

    /// Pack to u32 in RGBA byte order.
    #[inline(always)] pub const fn to_rgba_u32(self) -> u32 {
        ((self.x as u32) << 24) | ((self.y as u32) << 16) | ((self.z as u32) << 8) | (self.w as u32)
    }

    #[inline]
    pub fn select(mask: BVec4, if_true: Self, if_false: Self) -> Self {
        Self::new(
            if mask.x { if_true.x } else { if_false.x },
            if mask.y { if_true.y } else { if_false.y },
            if mask.z { if_true.z } else { if_false.z },
            if mask.w { if_true.w } else { if_false.w },
        )
    }

    /// Dot product, widened to u16.
    #[inline] pub fn dot(self, rhs: Self) -> u16 {
        (self.x as u16)*(rhs.x as u16) + (self.y as u16)*(rhs.y as u16)
        + (self.z as u16)*(rhs.z as u16) + (self.w as u16)*(rhs.w as u16)
    }

    #[inline] pub fn length_sq(self) -> u16 { self.dot(self) }

    #[inline] pub fn min(self,r:Self)->Self{Self::new(self.x.min(r.x),self.y.min(r.y),self.z.min(r.z),self.w.min(r.w))}
    #[inline] pub fn max(self,r:Self)->Self{Self::new(self.x.max(r.x),self.y.max(r.y),self.z.max(r.z),self.w.max(r.w))}
    #[inline] pub fn clamp(self,lo:Self,hi:Self)->Self{self.max(lo).min(hi)}
    #[inline] pub fn min_element(self)->u8{self.x.min(self.y).min(self.z).min(self.w)}
    #[inline] pub fn max_element(self)->u8{self.x.max(self.y).max(self.z).max(self.w)}
    #[inline] pub fn element_sum(self)->u8{self.x.wrapping_add(self.y).wrapping_add(self.z).wrapping_add(self.w)}

    #[inline] pub fn cmpeq(self,r:Self)->BVec4{BVec4::new(self.x==r.x,self.y==r.y,self.z==r.z,self.w==r.w)}
    #[inline] pub fn cmpne(self,r:Self)->BVec4{BVec4::new(self.x!=r.x,self.y!=r.y,self.z!=r.z,self.w!=r.w)}
    #[inline] pub fn cmpge(self,r:Self)->BVec4{BVec4::new(self.x>=r.x,self.y>=r.y,self.z>=r.z,self.w>=r.w)}
    #[inline] pub fn cmpgt(self,r:Self)->BVec4{BVec4::new(self.x>r.x, self.y>r.y, self.z>r.z, self.w>r.w)}
    #[inline] pub fn cmple(self,r:Self)->BVec4{BVec4::new(self.x<=r.x,self.y<=r.y,self.z<=r.z,self.w<=r.w)}
    #[inline] pub fn cmplt(self,r:Self)->BVec4{BVec4::new(self.x<r.x, self.y<r.y, self.z<r.z, self.w<r.w)}

    #[inline] pub fn wrapping_add(self,r:Self)->Self{Self::new(self.x.wrapping_add(r.x),self.y.wrapping_add(r.y),self.z.wrapping_add(r.z),self.w.wrapping_add(r.w))}
    #[inline] pub fn wrapping_sub(self,r:Self)->Self{Self::new(self.x.wrapping_sub(r.x),self.y.wrapping_sub(r.y),self.z.wrapping_sub(r.z),self.w.wrapping_sub(r.w))}
    #[inline] pub fn wrapping_mul(self,r:Self)->Self{Self::new(self.x.wrapping_mul(r.x),self.y.wrapping_mul(r.y),self.z.wrapping_mul(r.z),self.w.wrapping_mul(r.w))}
    #[inline] pub fn saturating_add(self,r:Self)->Self{Self::new(self.x.saturating_add(r.x),self.y.saturating_add(r.y),self.z.saturating_add(r.z),self.w.saturating_add(r.w))}
    #[inline] pub fn saturating_sub(self,r:Self)->Self{Self::new(self.x.saturating_sub(r.x),self.y.saturating_sub(r.y),self.z.saturating_sub(r.z),self.w.saturating_sub(r.w))}
    #[inline] pub fn checked_add(self,r:Self)->Option<Self>{Some(Self::new(self.x.checked_add(r.x)?,self.y.checked_add(r.y)?,self.z.checked_add(r.z)?,self.w.checked_add(r.w)?))}
    #[inline] pub fn checked_sub(self,r:Self)->Option<Self>{Some(Self::new(self.x.checked_sub(r.x)?,self.y.checked_sub(r.y)?,self.z.checked_sub(r.z)?,self.w.checked_sub(r.w)?))}

    #[inline] pub fn as_i8vec4(self)  -> I8Vec4          { I8Vec4::new(self.x as i8, self.y as i8, self.z as i8, self.w as i8) }
    #[inline] pub fn as_i16vec4(self) -> crate::I16Vec4  { crate::I16Vec4::new(self.x as i16, self.y as i16, self.z as i16, self.w as i16) }
    #[inline] pub fn as_u16vec4(self) -> crate::U16Vec4  { crate::U16Vec4::new(self.x as u16, self.y as u16, self.z as u16, self.w as u16) }
    #[inline] pub fn as_ivec4(self)   -> crate::IVec4    { crate::IVec4::new(self.x as i32, self.y as i32, self.z as i32, self.w as i32) }
    #[inline] pub fn as_uvec4(self)   -> crate::UVec4    { crate::UVec4::new(self.x as u32, self.y as u32, self.z as u32, self.w as u32) }
    #[inline] pub fn as_vec4(self)    -> crate::Vec4     { crate::Vec4::new(self.x as f32, self.y as f32, self.z as f32, self.w as f32) }
    #[inline] pub fn as_dvec4(self)   -> crate::DVec4    { crate::DVec4::new(self.x as f64, self.y as f64, self.z as f64, self.w as f64) }

    /// Normalise bone weights so they sum to 255 (lossy integer normalisation).
    /// Useful for u8 skinning weights before upload to GPU.
    #[inline]
    pub fn normalize_weights(self) -> Self {
        let sum = self.x as u16 + self.y as u16 + self.z as u16 + self.w as u16;
        if sum == 0 { return Self::new(255, 0, 0, 0); }
        let s = 255.0 / sum as f32;
        Self::new(
            (self.x as f32 * s).round() as u8,
            (self.y as f32 * s).round() as u8,
            (self.z as f32 * s).round() as u8,
            (self.w as f32 * s).round() as u8,
        )
    }
}

impl Add  for U8Vec4{type Output=Self;#[inline]fn add(self,r:Self)->Self{Self::new(self.x.wrapping_add(r.x),self.y.wrapping_add(r.y),self.z.wrapping_add(r.z),self.w.wrapping_add(r.w))}}
impl Sub  for U8Vec4{type Output=Self;#[inline]fn sub(self,r:Self)->Self{Self::new(self.x.wrapping_sub(r.x),self.y.wrapping_sub(r.y),self.z.wrapping_sub(r.z),self.w.wrapping_sub(r.w))}}
impl Mul  for U8Vec4{type Output=Self;#[inline]fn mul(self,r:Self)->Self{Self::new(self.x.wrapping_mul(r.x),self.y.wrapping_mul(r.y),self.z.wrapping_mul(r.z),self.w.wrapping_mul(r.w))}}
impl Div  for U8Vec4{type Output=Self;#[inline]fn div(self,r:Self)->Self{Self::new(self.x/r.x,self.y/r.y,self.z/r.z,self.w/r.w)}}
impl Rem  for U8Vec4{type Output=Self;#[inline]fn rem(self,r:Self)->Self{Self::new(self.x%r.x,self.y%r.y,self.z%r.z,self.w%r.w)}}
impl Not  for U8Vec4{type Output=Self;#[inline]fn not(self)->Self{Self::new(!self.x,!self.y,!self.z,!self.w)}}

impl Mul<u8> for U8Vec4{type Output=Self;#[inline]fn mul(self,s:u8)->Self{Self::new(self.x.wrapping_mul(s),self.y.wrapping_mul(s),self.z.wrapping_mul(s),self.w.wrapping_mul(s))}}
impl Mul<U8Vec4> for u8{type Output=U8Vec4;#[inline]fn mul(self,v:U8Vec4)->U8Vec4{U8Vec4::new(self.wrapping_mul(v.x),self.wrapping_mul(v.y),self.wrapping_mul(v.z),self.wrapping_mul(v.w))}}

impl AddAssign for U8Vec4{#[inline]fn add_assign(&mut self,r:Self){*self=*self+r;}}
impl SubAssign for U8Vec4{#[inline]fn sub_assign(&mut self,r:Self){*self=*self-r;}}
impl MulAssign for U8Vec4{#[inline]fn mul_assign(&mut self,r:Self){*self=*self*r;}}
impl DivAssign for U8Vec4{#[inline]fn div_assign(&mut self,r:Self){self.x/=r.x;self.y/=r.y;self.z/=r.z;self.w/=r.w;}}
impl RemAssign for U8Vec4{#[inline]fn rem_assign(&mut self,r:Self){self.x%=r.x;self.y%=r.y;self.z%=r.z;self.w%=r.w;}}
impl MulAssign<u8> for U8Vec4{#[inline]fn mul_assign(&mut self,s:u8){*self=*self*s;}}

impl BitAnd for U8Vec4{type Output=Self;#[inline]fn bitand(self,r:Self)->Self{Self::new(self.x&r.x,self.y&r.y,self.z&r.z,self.w&r.w)}}
impl BitOr  for U8Vec4{type Output=Self;#[inline]fn bitor (self,r:Self)->Self{Self::new(self.x|r.x,self.y|r.y,self.z|r.z,self.w|r.w)}}
impl BitXor for U8Vec4{type Output=Self;#[inline]fn bitxor(self,r:Self)->Self{Self::new(self.x^r.x,self.y^r.y,self.z^r.z,self.w^r.w)}}
impl BitAndAssign for U8Vec4{#[inline]fn bitand_assign(&mut self,r:Self){*self=*self&r;}}
impl BitOrAssign  for U8Vec4{#[inline]fn bitor_assign (&mut self,r:Self){*self=*self|r;}}
impl BitXorAssign for U8Vec4{#[inline]fn bitxor_assign(&mut self,r:Self){*self=*self^r;}}
impl Shl<u32> for U8Vec4{type Output=Self;#[inline]fn shl(self,s:u32)->Self{Self::new(self.x<<s,self.y<<s,self.z<<s,self.w<<s)}}
impl Shr<u32> for U8Vec4{type Output=Self;#[inline]fn shr(self,s:u32)->Self{Self::new(self.x>>s,self.y>>s,self.z>>s,self.w>>s)}}
impl ShlAssign<u32> for U8Vec4{#[inline]fn shl_assign(&mut self,s:u32){self.x<<=s;self.y<<=s;self.z<<=s;self.w<<=s;}}
impl ShrAssign<u32> for U8Vec4{#[inline]fn shr_assign(&mut self,s:u32){self.x>>=s;self.y>>=s;self.z>>=s;self.w>>=s;}}

impl Index<usize> for U8Vec4 {
    type Output=u8;
    #[inline] fn index(&self,i:usize)->&u8{match i{0=>&self.x,1=>&self.y,2=>&self.z,3=>&self.w,_=>panic!("U8Vec4 index {i} out of bounds")}}
}
impl IndexMut<usize> for U8Vec4 {
    #[inline] fn index_mut(&mut self,i:usize)->&mut u8{match i{0=>&mut self.x,1=>&mut self.y,2=>&mut self.z,3=>&mut self.w,_=>panic!("U8Vec4 index {i} out of bounds")}}
}

impl From<[u8;4]> for U8Vec4{#[inline]fn from(a:[u8;4])->Self{Self::from_array(a)}}
impl From<U8Vec4> for [u8;4]{#[inline]fn from(v:U8Vec4)->[u8;4]{v.to_array()}}
impl From<(u8,u8,u8,u8)> for U8Vec4{#[inline]fn from(t:(u8,u8,u8,u8))->Self{Self::new(t.0,t.1,t.2,t.3)}}
impl From<(U8Vec3,u8)> for U8Vec4{#[inline]fn from((v,w):(U8Vec3,u8))->Self{Self::new(v.x,v.y,v.z,w)}}
impl From<(U8Vec2,U8Vec2)> for U8Vec4{#[inline]fn from((a,b):(U8Vec2,U8Vec2))->Self{Self::new(a.x,a.y,b.x,b.y)}}

impl fmt::Debug for U8Vec4{fn fmt(&self,f:&mut fmt::Formatter<'_>)->fmt::Result{write!(f,"U8Vec4({}, {}, {}, {})",self.x,self.y,self.z,self.w)}}
impl fmt::Display for U8Vec4{fn fmt(&self,f:&mut fmt::Formatter<'_>)->fmt::Result{write!(f,"[{}, {}, {}, {}]",self.x,self.y,self.z,self.w)}}
