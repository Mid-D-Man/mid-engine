// crates/mid-math/src/wide/int/sse2/u32x4.rs
//! 4-lane unsigned 32-bit integer vector — SSE2, x86 / x86_64.

#![allow(non_camel_case_types)]

use core::fmt;
use core::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign,
    BitXor, BitXorAssign, Mul, MulAssign, Not, Sub, SubAssign,
};

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use super::imask4::IMask4;

#[repr(C)]
union UnionCast { u: [u32; 4], v: u32x4 }

#[inline(always)]
unsafe fn ucmpgt(a: __m128i, b: __m128i) -> __m128i {
    let sign = _mm_set1_epi32(i32::MIN);
    _mm_cmpgt_epi32(_mm_xor_si128(a, sign), _mm_xor_si128(b, sign))
}

/// 4-lane unsigned 32-bit integer vector. 16 bytes, 16-byte aligned. Backed by `__m128i`.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct u32x4(pub(crate) __m128i);

impl u32x4 {
    pub const ZERO: Self = unsafe { UnionCast { u: [0; 4] }.v };
    pub const ONE:  Self = unsafe { UnionCast { u: [1; 4] }.v };
    pub const MIN:  Self = unsafe { UnionCast { u: [u32::MIN; 4] }.v };
    pub const MAX:  Self = unsafe { UnionCast { u: [u32::MAX; 4] }.v };

    #[inline(always)]
    pub fn splat(v: u32) -> Self { Self(unsafe { _mm_set1_epi32(v as i32) }) }

    #[inline(always)]
    pub fn new(a: u32, b: u32, c: u32, d: u32) -> Self {
        Self(unsafe { _mm_set_epi32(d as i32, c as i32, b as i32, a as i32) })
    }

    #[inline(always)]
    pub fn from_array(a: [u32; 4]) -> Self {
        Self(unsafe { _mm_loadu_si128(a.as_ptr() as *const __m128i) })
    }

    #[inline(always)]
    pub fn to_array(self) -> [u32; 4] {
        unsafe { let mut a=[0u32;4]; _mm_storeu_si128(a.as_mut_ptr() as *mut __m128i, self.0); a }
    }

    #[inline]
    pub fn get(self, i: usize) -> u32 {
        assert!(i < 4, "u32x4::get — lane {i} out of bounds (max 3)");
        unsafe { UnionCast { v: self }.u[i] }
    }

    #[inline] pub fn min(self, rhs: Self) -> Self { Self::blend(self.cmplt(rhs), self, rhs) }
    #[inline] pub fn max(self, rhs: Self) -> Self { Self::blend(self.cmpgt(rhs), self, rhs) }
    #[inline] pub fn clamp(self, lo: Self, hi: Self) -> Self { self.max(lo).min(hi) }

    #[inline]
    pub fn min_element(self) -> u32 {
        let a = self.to_array(); a[0].min(a[1]).min(a[2]).min(a[3])
    }
    #[inline]
    pub fn max_element(self) -> u32 {
        let a = self.to_array(); a[0].max(a[1]).max(a[2]).max(a[3])
    }
    #[inline]
    pub fn element_sum(self) -> u32 {
        let a = self.to_array();
        a[0].wrapping_add(a[1]).wrapping_add(a[2]).wrapping_add(a[3])
    }

    /// Logical left shift — uses variable-count `_mm_sll_epi32`.
    #[inline(always)]
    pub fn shl(self, count: u32) -> Self {
        unsafe {
            let cnt = _mm_cvtsi32_si128(count as i32);
            Self(_mm_sll_epi32(self.0, cnt))
        }
    }

    /// Logical right shift (zero-fill) — uses variable-count `_mm_srl_epi32`.
    #[inline(always)]
    pub fn shr(self, count: u32) -> Self {
        unsafe {
            let cnt = _mm_cvtsi32_si128(count as i32);
            Self(_mm_srl_epi32(self.0, cnt))
        }
    }

    #[inline(always)] pub fn cmpeq(self, rhs: Self) -> IMask4 { IMask4(unsafe{_mm_cmpeq_epi32(self.0,rhs.0)}) }
    #[inline(always)] pub fn cmpne(self, rhs: Self) -> IMask4 { !self.cmpeq(rhs) }
    #[inline(always)] pub fn cmpgt(self, rhs: Self) -> IMask4 { IMask4(unsafe{ucmpgt(self.0,rhs.0)}) }
    #[inline(always)] pub fn cmplt(self, rhs: Self) -> IMask4 { rhs.cmpgt(self) }
    #[inline(always)] pub fn cmpge(self, rhs: Self) -> IMask4 { !self.cmplt(rhs) }
    #[inline(always)] pub fn cmple(self, rhs: Self) -> IMask4 { !self.cmpgt(rhs) }

    #[inline(always)]
    pub fn blend(mask: IMask4, if_true: Self, if_false: Self) -> Self {
        unsafe {
            Self(_mm_or_si128(_mm_and_si128(mask.0,if_true.0),_mm_andnot_si128(mask.0,if_false.0)))
        }
    }

    #[inline(always)] pub fn wrapping_add(self, r: Self) -> Self { self + r }
    #[inline(always)] pub fn wrapping_sub(self, r: Self) -> Self { self - r }
    #[inline(always)] pub fn wrapping_mul(self, r: Self) -> Self { self * r }

    #[inline]
    pub fn saturating_add(self, rhs: Self) -> Self {
        unsafe {
            let sum = _mm_add_epi32(self.0, rhs.0);
            let overflowed = ucmpgt(self.0, sum);
            let all_ones = _mm_cmpeq_epi32(sum, sum);
            Self(_mm_or_si128(_mm_and_si128(overflowed, all_ones), _mm_andnot_si128(overflowed, sum)))
        }
    }

    #[inline]
    pub fn saturating_sub(self, rhs: Self) -> Self {
        unsafe {
            let underflowed = ucmpgt(rhs.0, self.0);
            let diff = _mm_sub_epi32(self.0, rhs.0);
            Self(_mm_andnot_si128(underflowed, diff))
        }
    }
}

impl Add  for u32x4 { type Output=Self; #[inline(always)] fn add(self,r:Self)->Self{Self(unsafe{_mm_add_epi32(self.0,r.0)})} }
impl AddAssign for u32x4 { #[inline(always)] fn add_assign(&mut self,r:Self){*self=*self+r;} }
impl Sub  for u32x4 { type Output=Self; #[inline(always)] fn sub(self,r:Self)->Self{Self(unsafe{_mm_sub_epi32(self.0,r.0)})} }
impl SubAssign for u32x4 { #[inline(always)] fn sub_assign(&mut self,r:Self){*self=*self-r;} }

impl Mul for u32x4 {
    type Output=Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        unsafe {
            let a13=_mm_shuffle_epi32(self.0,0xF5); let b13=_mm_shuffle_epi32(rhs.0,0xF5);
            let p02=_mm_mul_epu32(self.0,rhs.0); let p13=_mm_mul_epu32(a13,b13);
            let lo02=_mm_shuffle_epi32(p02,0x08); let lo13=_mm_shuffle_epi32(p13,0x08);
            Self(_mm_unpacklo_epi32(lo02,lo13))
        }
    }
}
impl MulAssign for u32x4 { #[inline(always)] fn mul_assign(&mut self,r:Self){*self=*self*r;} }

impl BitAnd for u32x4 { type Output=Self; #[inline(always)] fn bitand(self,r:Self)->Self{Self(unsafe{_mm_and_si128(self.0,r.0)})} }
impl BitAndAssign for u32x4 { #[inline(always)] fn bitand_assign(&mut self,r:Self){*self=*self&r;} }
impl BitOr  for u32x4 { type Output=Self; #[inline(always)] fn bitor (self,r:Self)->Self{Self(unsafe{_mm_or_si128(self.0,r.0)})} }
impl BitOrAssign  for u32x4 { #[inline(always)] fn bitor_assign (&mut self,r:Self){*self=*self|r;} }
impl BitXor for u32x4 { type Output=Self; #[inline(always)] fn bitxor(self,r:Self)->Self{Self(unsafe{_mm_xor_si128(self.0,r.0)})} }
impl BitXorAssign for u32x4 { #[inline(always)] fn bitxor_assign(&mut self,r:Self){*self=*self^r;} }
impl Not for u32x4 { type Output=Self; #[inline(always)] fn not(self)->Self{unsafe{let o=_mm_cmpeq_epi32(self.0,self.0);Self(_mm_xor_si128(self.0,o))}} }

impl PartialEq for u32x4 {
    fn eq(&self,r:&Self)->bool{unsafe{_mm_movemask_ps(_mm_castsi128_ps(_mm_cmpeq_epi32(self.0,r.0)))==0b1111}}
}
impl Eq for u32x4 {}
impl fmt::Debug for u32x4 {
    fn fmt(&self,f:&mut fmt::Formatter<'_>)->fmt::Result{let a=self.to_array();write!(f,"u32x4({},{},{},{})",a[0],a[1],a[2],a[3])}
}
impl fmt::Display for u32x4 {
    fn fmt(&self,f:&mut fmt::Formatter<'_>)->fmt::Result{let a=self.to_array();write!(f,"[{},{},{},{}]",a[0],a[1],a[2],a[3])}
}
impl From<[u32;4]> for u32x4 { fn from(a:[u32;4])->Self{Self::from_array(a)} }
impl From<u32x4> for [u32;4] { fn from(v:u32x4)->[u32;4]{v.to_array()} }
