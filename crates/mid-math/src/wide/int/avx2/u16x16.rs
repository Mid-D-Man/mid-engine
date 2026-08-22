// crates/mid-math/src/wide/int/avx2/u16x16.rs
//! 16-lane unsigned 16-bit integer vector — AVX2, x86 / x86_64.
//!
//! Widens sse2/u16x8.rs to `__m256i`. AVX2 adds native
//! `_mm256_min_epu16`/`_mm256_max_epu16` — SSE2 had to emulate both via
//! the XOR-sign-flip cmpgt + blend trick (see sse2/u16x8.rs's
//! `ucmpgt_u16`). That trick is still needed here for `cmpgt` itself
//! (no native unsigned compare in AVX2, same gap as SSE2).
//!
//! `as_u32x8_lo`/`as_u32x8_hi` zero-extend via the dedicated
//! `_mm256_cvtepu16_epi32` widen instruction — not a shuffle, so no
//! per-128-bit-lane cross-lane hazard (see i16x16.rs's module docs for
//! the general explanation of why that hazard doesn't apply here).

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

use super::imask16x16::IMask16x16;

#[repr(C)]
union UnionCast { u: [u16; 16], v: u16x16 }

#[inline(always)]
unsafe fn ucmpgt_u16(a: __m256i, b: __m256i) -> __m256i {
    let sign = _mm256_set1_epi16(i16::MIN);
    _mm256_cmpgt_epi16(_mm256_xor_si256(a, sign), _mm256_xor_si256(b, sign))
}

/// 16-lane unsigned 16-bit integer vector. 32 bytes, 32-byte aligned. Backed by `__m256i`.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct u16x16(pub(crate) __m256i);

impl u16x16 {
    pub const ZERO: Self = unsafe { UnionCast { u: [0; 16] }.v };
    pub const ONE:  Self = unsafe { UnionCast { u: [1; 16] }.v };
    pub const MIN:  Self = unsafe { UnionCast { u: [u16::MIN; 16] }.v };
    pub const MAX:  Self = unsafe { UnionCast { u: [u16::MAX; 16] }.v };

    #[inline(always)]
    pub fn splat(v: u16) -> Self { Self(unsafe { _mm256_set1_epi16(v as i16) }) }

    #[inline(always)]
    pub fn new(a:u16,b:u16,c:u16,d:u16,e:u16,f:u16,g:u16,h:u16,i_:u16,j:u16,k:u16,l:u16,m:u16,n:u16,o:u16,p:u16) -> Self {
        Self(unsafe { _mm256_set_epi16(
            p as i16,o as i16,n as i16,m as i16,l as i16,k as i16,j as i16,i_ as i16,
            h as i16,g as i16,f as i16,e as i16,d as i16,c as i16,b as i16,a as i16,
        ) })
    }

    #[inline(always)]
    pub fn from_array(a: [u16; 16]) -> Self {
        Self(unsafe { _mm256_loadu_si256(a.as_ptr() as *const __m256i) })
    }

    #[inline(always)]
    pub fn to_array(self) -> [u16; 16] {
        unsafe { let mut a=[0u16;16]; _mm256_storeu_si256(a.as_mut_ptr() as *mut __m256i,self.0); a }
    }

    #[inline]
    pub fn get(self, i: usize) -> u16 {
        assert!(i < 16, "u16x16::get — lane {i} out of bounds (max 15)");
        unsafe { UnionCast { v: self }.u[i] }
    }

    /// Zero-extend the low 8 lanes (indices 0-7) to `u32x8`.
    #[inline(always)]
    pub fn as_u32x8_lo(self) -> super::u32x8::u32x8 {
        unsafe {
            let lo_128 = _mm256_castsi256_si128(self.0);
            super::u32x8::u32x8(_mm256_cvtepu16_epi32(lo_128))
        }
    }
    /// Zero-extend the high 8 lanes (indices 8-15) to `u32x8`.
    #[inline(always)]
    pub fn as_u32x8_hi(self) -> super::u32x8::u32x8 {
        unsafe {
            let hi_128 = _mm256_extracti128_si256::<1>(self.0);
            super::u32x8::u32x8(_mm256_cvtepu16_epi32(hi_128))
        }
    }

    /// Per-lane minimum. AVX2 native `_mm256_min_epu16`.
    #[inline(always)] pub fn min(self, rhs: Self) -> Self { Self(unsafe{_mm256_min_epu16(self.0,rhs.0)}) }
    /// Per-lane maximum. AVX2 native `_mm256_max_epu16`.
    #[inline(always)] pub fn max(self, rhs: Self) -> Self { Self(unsafe{_mm256_max_epu16(self.0,rhs.0)}) }
    #[inline(always)] pub fn clamp(self, lo: Self, hi: Self) -> Self { self.max(lo).min(hi) }

    #[inline] pub fn min_element(self) -> u16 { self.to_array().iter().copied().reduce(u16::min).unwrap() }
    #[inline] pub fn max_element(self) -> u16 { self.to_array().iter().copied().reduce(u16::max).unwrap() }
    #[inline] pub fn element_sum(self) -> u32 { self.to_array().iter().map(|&x| x as u32).sum() }

    #[inline(always)] pub fn mul_lo(self, rhs: Self) -> Self { Self(unsafe{_mm256_mullo_epi16(self.0,rhs.0)}) }
    #[inline(always)] pub fn mul_high_u(self, rhs: Self) -> Self { Self(unsafe{_mm256_mulhi_epu16(self.0,rhs.0)}) }
    #[inline(always)] pub fn saturating_add(self, rhs: Self) -> Self { Self(unsafe{_mm256_adds_epu16(self.0,rhs.0)}) }
    #[inline(always)] pub fn saturating_sub(self, rhs: Self) -> Self { Self(unsafe{_mm256_subs_epu16(self.0,rhs.0)}) }

    #[inline(always)]
    pub fn shl(self, count: u32) -> Self {
        unsafe { let cnt = _mm_cvtsi32_si128(count as i32); Self(_mm256_sll_epi16(self.0, cnt)) }
    }
    #[inline(always)]
    pub fn shr(self, count: u32) -> Self {
        unsafe { let cnt = _mm_cvtsi32_si128(count as i32); Self(_mm256_srl_epi16(self.0, cnt)) }
    }

    #[inline(always)] pub fn cmpeq(self, rhs: Self) -> IMask16x16 { IMask16x16(unsafe{_mm256_cmpeq_epi16(self.0,rhs.0)}) }
    #[inline(always)] pub fn cmpne(self, rhs: Self) -> IMask16x16 { !self.cmpeq(rhs) }
    #[inline(always)] pub fn cmpgt(self, rhs: Self) -> IMask16x16 { IMask16x16(unsafe{ucmpgt_u16(self.0,rhs.0)}) }
    #[inline(always)] pub fn cmplt(self, rhs: Self) -> IMask16x16 { rhs.cmpgt(self) }
    #[inline(always)] pub fn cmpge(self, rhs: Self) -> IMask16x16 { !self.cmplt(rhs) }
    #[inline(always)] pub fn cmple(self, rhs: Self) -> IMask16x16 { !self.cmpgt(rhs) }

    #[inline(always)]
    pub fn blend(mask: IMask16x16, if_true: Self, if_false: Self) -> Self {
        unsafe {
            Self(_mm256_or_si256(_mm256_and_si256(mask.0,if_true.0),_mm256_andnot_si256(mask.0,if_false.0)))
        }
    }

    #[inline(always)] pub fn wrapping_add(self, r: Self) -> Self { self + r }
    #[inline(always)] pub fn wrapping_sub(self, r: Self) -> Self { self - r }
    #[inline(always)] pub fn wrapping_mul(self, r: Self) -> Self { self.mul_lo(r) }
}

impl Add  for u16x16{type Output=Self;#[inline(always)]fn add(self,r:Self)->Self{Self(unsafe{_mm256_add_epi16(self.0,r.0)})}}
impl AddAssign for u16x16{#[inline(always)]fn add_assign(&mut self,r:Self){*self=*self+r;}}
impl Sub  for u16x16{type Output=Self;#[inline(always)]fn sub(self,r:Self)->Self{Self(unsafe{_mm256_sub_epi16(self.0,r.0)})}}
impl SubAssign for u16x16{#[inline(always)]fn sub_assign(&mut self,r:Self){*self=*self-r;}}
impl Mul  for u16x16{type Output=Self;#[inline(always)]fn mul(self,r:Self)->Self{self.mul_lo(r)}}
impl MulAssign for u16x16{#[inline(always)]fn mul_assign(&mut self,r:Self){*self=*self*r;}}
impl BitAnd for u16x16{type Output=Self;#[inline(always)]fn bitand(self,r:Self)->Self{Self(unsafe{_mm256_and_si256(self.0,r.0)})}}
impl BitAndAssign for u16x16{#[inline(always)]fn bitand_assign(&mut self,r:Self){*self=*self&r;}}
impl BitOr  for u16x16{type Output=Self;#[inline(always)]fn bitor (self,r:Self)->Self{Self(unsafe{_mm256_or_si256(self.0,r.0)})}}
impl BitOrAssign  for u16x16{#[inline(always)]fn bitor_assign (&mut self,r:Self){*self=*self|r;}}
impl BitXor for u16x16{type Output=Self;#[inline(always)]fn bitxor(self,r:Self)->Self{Self(unsafe{_mm256_xor_si256(self.0,r.0)})}}
impl BitXorAssign for u16x16{#[inline(always)]fn bitxor_assign(&mut self,r:Self){*self=*self^r;}}
impl Not for u16x16{type Output=Self;#[inline(always)]fn not(self)->Self{unsafe{let o=_mm256_cmpeq_epi16(self.0,self.0);Self(_mm256_xor_si256(self.0,o))}}}

impl PartialEq for u16x16{fn eq(&self,r:&Self)->bool{unsafe{_mm256_movemask_epi8(_mm256_cmpeq_epi16(self.0,r.0))==-1}}}
impl Eq for u16x16{}
impl fmt::Debug for u16x16{
    fn fmt(&self,f:&mut fmt::Formatter<'_>)->fmt::Result{
        let a=self.to_array();
        write!(f,"u16x16({},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{})",
            a[0],a[1],a[2],a[3],a[4],a[5],a[6],a[7],a[8],a[9],a[10],a[11],a[12],a[13],a[14],a[15])
    }
}
impl fmt::Display for u16x16{
    fn fmt(&self,f:&mut fmt::Formatter<'_>)->fmt::Result{
        let a=self.to_array();
        write!(f,"[{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}]",
            a[0],a[1],a[2],a[3],a[4],a[5],a[6],a[7],a[8],a[9],a[10],a[11],a[12],a[13],a[14],a[15])
    }
}
impl From<[u16;16]> for u16x16{fn from(a:[u16;16])->Self{Self::from_array(a)}}
impl From<u16x16> for [u16;16]{fn from(v:u16x16)->[u16;16]{v.to_array()}}
