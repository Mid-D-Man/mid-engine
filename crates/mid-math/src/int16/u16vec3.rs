// crates/mid-math/src/int16/u16vec3.rs
//! 3D unsigned 16-bit integer vector. 6 bytes, align 2. No padding. Always scalar.

use core::fmt;
use core::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign,
    Div, DivAssign, Index, IndexMut, Mul, MulAssign, Not,
    Rem, RemAssign, Shl, ShlAssign, Shr, ShrAssign, Sub, SubAssign,
};
use crate::{BVec3, I16Vec3, U16Vec2, U16Vec4};

/// 3D unsigned 16-bit integer vector. 6 bytes, align 2. No padding.
///
/// Dot product returns u32 to prevent overflow.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(C)]
pub struct U16Vec3 {
    pub x: u16,
    pub y: u16,
    pub z: u16,
}

impl U16Vec3 {
    pub const ZERO: Self = Self::splat(0);
    pub const ONE:  Self = Self::splat(1);
    pub const MIN:  Self = Self::splat(u16::MIN);
    pub const MAX:  Self = Self::splat(u16::MAX);
    pub const X:    Self = Self::new(1, 0, 0);
    pub const Y:    Self = Self::new(0, 1, 0);
    pub const Z:    Self = Self::new(0, 0, 1);

    #[inline(always)] pub const fn new(x: u16, y: u16, z: u16) -> Self { Self { x, y, z } }
    #[inline(always)] pub const fn splat(v: u16) -> Self { Self { x: v, y: v, z: v } }
    #[inline(always)] pub const fn from_array(a: [u16; 3]) -> Self { Self::new(a[0], a[1], a[2]) }
    #[inline(always)] pub const fn to_array(self) -> [u16; 3] { [self.x, self.y, self.z] }
    #[inline(always)] pub const fn extend(self, w: u16) -> U16Vec4 { U16Vec4::new(self.x, self.y, self.z, w) }
    #[inline(always)] pub const fn truncate(self) -> U16Vec2 { U16Vec2::new(self.x, self.y) }

    #[inline]
    pub fn select(mask: BVec3, if_true: Self, if_false: Self) -> Self {
        Self::new(
            if mask.x { if_true.x } else { if_false.x },
            if mask.y { if_true.y } else { if_false.y },
            if mask.z { if_true.z } else { if_false.z },
        )
    }

    /// Dot product, widened to u32.
    #[inline] pub fn dot(self, rhs: Self) -> u32 {
        (self.x as u32)*(rhs.x as u32) + (self.y as u32)*(rhs.y as u32) + (self.z as u32)*(rhs.z as u32)
    }

    #[inline] pub fn length_sq(self) -> u32 { self.dot(self) }
    #[inline] pub fn distance_sq(self, rhs: Self) -> u32 {
        let dx = (self.x as i32) - (rhs.x as i32);
        let dy = (self.y as i32) - (rhs.y as i32);
        let dz = (self.z as i32) - (rhs.z as i32);
        (dx*dx + dy*dy + dz*dz) as u32
    }

    #[inline] pub fn min(self,r:Self)->Self{Self::new(self.x.min(r.x),self.y.min(r.y),self.z.min(r.z))}
    #[inline] pub fn max(self,r:Self)->Self{Self::new(self.x.max(r.x),self.y.max(r.y),self.z.max(r.z))}
    #[inline] pub fn clamp(self,lo:Self,hi:Self)->Self{self.max(lo).min(hi)}
    #[inline] pub fn min_element(self)->u16{self.x.min(self.y).min(self.z)}
    #[inline] pub fn max_element(self)->u16{self.x.max(self.y).max(self.z)}
    #[inline] pub fn element_sum(self)->u16{self.x.wrapping_add(self.y).wrapping_add(self.z)}

    #[inline] pub fn cmpeq(self,r:Self)->BVec3{BVec3::new(self.x==r.x,self.y==r.y,self.z==r.z)}
    #[inline] pub fn cmpne(self,r:Self)->BVec3{BVec3::new(self.x!=r.x,self.y!=r.y,self.z!=r.z)}
    #[inline] pub fn cmpge(self,r:Self)->BVec3{BVec3::new(self.x>=r.x,self.y>=r.y,self.z>=r.z)}
    #[inline] pub fn cmpgt(self,r:Self)->BVec3{BVec3::new(self.x>r.x, self.y>r.y, self.z>r.z)}
    #[inline] pub fn cmple(self,r:Self)->BVec3{BVec3::new(self.x<=r.x,self.y<=r.y,self.z<=r.z)}
    #[inline] pub fn cmplt(self,r:Self)->BVec3{BVec3::new(self.x<r.x, self.y<r.y, self.z<r.z)}

    #[inline] pub fn wrapping_add(self,r:Self)->Self{Self::new(self.x.wrapping_add(r.x),self.y.wrapping_add(r.y),self.z.wrapping_add(r.z))}
    #[inline] pub fn wrapping_sub(self,r:Self)->Self{Self::new(self.x.wrapping_sub(r.x),self.y.wrapping_sub(r.y),self.z.wrapping_sub(r.z))}
    #[inline] pub fn wrapping_mul(self,r:Self)->Self{Self::new(self.x.wrapping_mul(r.x),self.y.wrapping_mul(r.y),self.z.wrapping_mul(r.z))}
    #[inline] pub fn saturating_add(self,r:Self)->Self{Self::new(self.x.saturating_add(r.x),self.y.saturating_add(r.y),self.z.saturating_add(r.z))}
    #[inline] pub fn saturating_sub(self,r:Self)->Self{Self::new(self.x.saturating_sub(r.x),self.y.saturating_sub(r.y),self.z.saturating_sub(r.z))}
    #[inline] pub fn checked_add(self,r:Self)->Option<Self>{Some(Self::new(self.x.checked_add(r.x)?,self.y.checked_add(r.y)?,self.z.checked_add(r.z)?))}
    #[inline] pub fn checked_sub(self,r:Self)->Option<Self>{Some(Self::new(self.x.checked_sub(r.x)?,self.y.checked_sub(r.y)?,self.z.checked_sub(r.z)?))}

    #[inline] pub fn as_i16vec3(self) -> I16Vec3         { I16Vec3::new(self.x as i16, self.y as i16, self.z as i16) }
    #[inline] pub fn as_i8vec3(self)  -> crate::I8Vec3   { crate::I8Vec3::new(self.x as i8, self.y as i8, self.z as i8) }
    #[inline] pub fn as_u8vec3(self)  -> crate::U8Vec3   { crate::U8Vec3::new(self.x as u8, self.y as u8, self.z as u8) }
    #[inline] pub fn as_ivec3(self)   -> crate::IVec3    { crate::IVec3::new(self.x as i32, self.y as i32, self.z as i32) }
    #[inline] pub fn as_uvec3(self)   -> crate::UVec3    { crate::UVec3::new(self.x as u32, self.y as u32, self.z as u32) }
    #[inline] pub fn as_u64vec3(self) -> crate::U64Vec3  { crate::U64Vec3::new(self.x as u64, self.y as u64, self.z as u64) }
    #[inline] pub fn as_vec3(self)    -> crate::Vec3     { crate::Vec3::new(self.x as f32, self.y as f32, self.z as f32) }
    #[inline] pub fn as_dvec3(self)   -> crate::DVec3    { crate::DVec3::new(self.x as f64, self.y as f64, self.z as f64) }
}

impl Add  for U16Vec3{type Output=Self;#[inline]fn add(self,r:Self)->Self{Self::new(self.x.wrapping_add(r.x),self.y.wrapping_add(r.y),self.z.wrapping_add(r.z))}}
impl Sub  for U16Vec3{type Output=Self;#[inline]fn sub(self,r:Self)->Self{Self::new(self.x.wrapping_sub(r.x),self.y.wrapping_sub(r.y),self.z.wrapping_sub(r.z))}}
impl Mul  for U16Vec3{type Output=Self;#[inline]fn mul(self,r:Self)->Self{Self::new(self.x.wrapping_mul(r.x),self.y.wrapping_mul(r.y),self.z.wrapping_mul(r.z))}}
impl Div  for U16Vec3{type Output=Self;#[inline]fn div(self,r:Self)->Self{Self::new(self.x/r.x,self.y/r.y,self.z/r.z)}}
impl Rem  for U16Vec3{type Output=Self;#[inline]fn rem(self,r:Self)->Self{Self::new(self.x%r.x,self.y%r.y,self.z%r.z)}}
impl Not  for U16Vec3{type Output=Self;#[inline]fn not(self)->Self{Self::new(!self.x,!self.y,!self.z)}}

impl Mul<u16> for U16Vec3{type Output=Self;#[inline]fn mul(self,s:u16)->Self{Self::new(self.x.wrapping_mul(s),self.y.wrapping_mul(s),self.z.wrapping_mul(s))}}
impl Mul<U16Vec3> for u16{type Output=U16Vec3;#[inline]fn mul(self,v:U16Vec3)->U16Vec3{U16Vec3::new(self.wrapping_mul(v.x),self.wrapping_mul(v.y),self.wrapping_mul(v.z))}}

impl AddAssign for U16Vec3{#[inline]fn add_assign(&mut self,r:Self){*self=*self+r;}}
impl SubAssign for U16Vec3{#[inline]fn sub_assign(&mut self,r:Self){*self=*self-r;}}
impl MulAssign for U16Vec3{#[inline]fn mul_assign(&mut self,r:Self){*self=*self*r;}}
impl DivAssign for U16Vec3{#[inline]fn div_assign(&mut self,r:Self){self.x/=r.x;self.y/=r.y;self.z/=r.z;}}
impl RemAssign for U16Vec3{#[inline]fn rem_assign(&mut self,r:Self){self.x%=r.x;self.y%=r.y;self.z%=r.z;}}
impl MulAssign<u16> for U16Vec3{#[inline]fn mul_assign(&mut self,s:u16){*self=*self*s;}}

impl BitAnd for U16Vec3{type Output=Self;#[inline]fn bitand(self,r:Self)->Self{Self::new(self.x&r.x,self.y&r.y,self.z&r.z)}}
impl BitOr  for U16Vec3{type Output=Self;#[inline]fn bitor (self,r:Self)->Self{Self::new(self.x|r.x,self.y|r.y,self.z|r.z)}}
impl BitXor for U16Vec3{type Output=Self;#[inline]fn bitxor(self,r:Self)->Self{Self::new(self.x^r.x,self.y^r.y,self.z^r.z)}}
impl BitAndAssign for U16Vec3{#[inline]fn bitand_assign(&mut self,r:Self){*self=*self&r;}}
impl BitOrAssign  for U16Vec3{#[inline]fn bitor_assign (&mut self,r:Self){*self=*self|r;}}
impl BitXorAssign for U16Vec3{#[inline]fn bitxor_assign(&mut self,r:Self){*self=*self^r;}}
impl Shl<u32> for U16Vec3{type Output=Self;#[inline]fn shl(self,s:u32)->Self{Self::new(self.x<<s,self.y<<s,self.z<<s)}}
impl Shr<u32> for U16Vec3{type Output=Self;#[inline]fn shr(self,s:u32)->Self{Self::new(self.x>>s,self.y>>s,self.z>>s)}}
impl ShlAssign<u32> for U16Vec3{#[inline]fn shl_assign(&mut self,s:u32){self.x<<=s;self.y<<=s;self.z<<=s;}}
impl ShrAssign<u32> for U16Vec3{#[inline]fn shr_assign(&mut self,s:u32){self.x>>=s;self.y>>=s;self.z>>=s;}}

impl Index<usize> for U16Vec3{type Output=u16;#[inline]fn index(&self,i:usize)->&u16{match i{0=>&self.x,1=>&self.y,2=>&self.z,_=>panic!("U16Vec3 index {i} out of bounds")}}}
impl IndexMut<usize> for U16Vec3{#[inline]fn index_mut(&mut self,i:usize)->&mut u16{match i{0=>&mut self.x,1=>&mut self.y,2=>&mut self.z,_=>panic!("U16Vec3 index {i} out of bounds")}}}

impl From<[u16;3]> for U16Vec3{#[inline]fn from(a:[u16;3])->Self{Self::from_array(a)}}
impl From<U16Vec3> for [u16;3]{#[inline]fn from(v:U16Vec3)->[u16;3]{v.to_array()}}
impl From<(u16,u16,u16)> for U16Vec3{#[inline]fn from(t:(u16,u16,u16))->Self{Self::new(t.0,t.1,t.2)}}
impl From<(U16Vec2,u16)> for U16Vec3{#[inline]fn from((v,z):(U16Vec2,u16))->Self{Self::new(v.x,v.y,z)}}

impl fmt::Debug for U16Vec3{fn fmt(&self,f:&mut fmt::Formatter<'_>)->fmt::Result{write!(f,"U16Vec3({}, {}, {})",self.x,self.y,self.z)}}
impl fmt::Display for U16Vec3{fn fmt(&self,f:&mut fmt::Formatter<'_>)->fmt::Result{write!(f,"[{}, {}, {}]",self.x,self.y,self.z)}}
