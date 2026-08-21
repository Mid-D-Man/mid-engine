// crates/mid-math/src/wide/int/avx2/u32x8.rs
//! 8-lane unsigned 32-bit integer vector — AVX2, x86 / x86_64.
//!
//! Widens sse2/u32x4.rs to `__m256i`. AVX2 adds native
//! `_mm256_min_epu32`/`_mm256_max_epu32`/`_mm256_mullo_epi32` — SSE2 had
//! to emulate min/max via the same XOR-sign-flip cmpgt trick still needed
//! here for `cmpgt` itself (AVX2 has no native unsigned compare, same gap
//! as SSE2).

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

use super::imask32x8::IMask32x8;

#[repr(C)]
union UnionCast { u: [u32; 8], v: u32x8 }

#[inline(always)]
unsafe fn ucmpgt(a: __m256i, b: __m256i) -> __m256i {
    let sign = _mm256_set1_epi32(i32::MIN);
    _mm256_cmpgt_epi32(_mm256_xor_si256(a, sign), _mm256_xor_si256(b, sign))
}

/// 8-lane unsigned 32-bit integer vector. 32 bytes, 32-byte aligned. Backed by `__m256i`.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct u32x8(pub(crate) __m256i);

impl u32x8 {
    pub const ZERO: Self = unsafe { UnionCast { u: [0; 8] }.v };
    pub const ONE:  Self = unsafe { UnionCast { u: [1; 8] }.v };
    pub const MIN:  Self = unsafe { UnionCast { u: [u32::MIN; 8] }.v };
    pub const MAX:  Self = unsafe { UnionCast { u: [u32::MAX; 8] }.v };

    #[inline(always)]
    pub fn splat(v: u32) -> Self { Self(unsafe { _mm256_set1_epi32(v as i32) }) }

    #[inline(always)]
    pub fn new(a: u32, b: u32, c: u32, d: u32, e: u32, f: u32, g: u32, h: u32) -> Self {
        Self(unsafe { _mm256_set_epi32(h as i32, g as i32, f as i32, e as i32, d as i32, c as i32, b as i32, a as i32) })
    }

    #[inline(always)]
    pub fn from_array(a: [u32; 8]) -> Self {
        Self(unsafe { _mm256_loadu_si256(a.as_ptr() as *const __m256i) })
    }

    #[inline(always)]
    pub fn to_array(self) -> [u32; 8] {
        unsafe { let mut a=[0u32;8]; _mm256_storeu_si256(a.as_mut_ptr() as *mut __m256i, self.0); a }
    }

    #[inline]
    pub fn get(self, i: usize) -> u32 {
        assert!(i < 8, "u32x8::get — lane {i} out of bounds (max 7)");
        unsafe { UnionCast { v: self }.u[i] }
    }

    /// Per-lane minimum. AVX2 native `_mm256_min_epu32`.
    #[inline(always)] pub fn min(self, rhs: Self) -> Self { Self(unsafe { _mm256_min_epu32(self.0, rhs.0) }) }
    /// Per-lane maximum. AVX2 native `_mm256_max_epu32`.
    #[inline(always)] pub fn max(self, rhs: Self) -> Self { Self(unsafe { _mm256_max_epu32(self.0, rhs.0) }) }
    #[inline(always)] pub fn clamp(self, lo: Self, hi: Self) -> Self { self.max(lo).min(hi) }

    #[inline]
    pub fn min_element(self) -> u32 { self.to_array().iter().copied().reduce(u32::min).unwrap() }
    #[inline]
    pub fn max_element(self) -> u32 { self.to_array().iter().copied().reduce(u32::max).unwrap() }
    #[inline]
    pub fn element_sum(self) -> u32 {
        self.to_array().iter().fold(0u32, |acc, &x| acc.wrapping_add(x))
    }

    /// Logical left shift — uses variable-count `_mm256_sll_epi32`.
    #[inline(always)]
    pub fn shl(self, count: u32) -> Self {
        unsafe { let cnt = _mm_cvtsi32_si128(count as i32); Self(_mm256_sll_epi32(self.0, cnt)) }
    }
    /// Logical right shift (zero-fill) — uses variable-count `_mm256_srl_epi32`.
    #[inline(always)]
    pub fn shr(self, count: u32) -> Self {
        unsafe { let cnt = _mm_cvtsi32_si128(count as i32); Self(_mm256_srl_epi32(self.0, cnt)) }
    }

    #[inline(always)] pub fn cmpeq(self, rhs: Self) -> IMask32x8 { IMask32x8(unsafe { _mm256_cmpeq_epi32(self.0, rhs.0) }) }
    #[inline(always)] pub fn cmpne(self, rhs: Self) -> IMask32x8 { !self.cmpeq(rhs) }
    #[inline(always)] pub fn cmpgt(self, rhs: Self) -> IMask32x8 { IMask32x8(unsafe { ucmpgt(self.0, rhs.0) }) }
    #[inline(always)] pub fn cmplt(self, rhs: Self) -> IMask32x8 { rhs.cmpgt(self) }
    #[inline(always)] pub fn cmpge(self, rhs: Self) -> IMask32x8 { !self.cmplt(rhs) }
    #[inline(always)] pub fn cmple(self, rhs: Self) -> IMask32x8 { !self.cmpgt(rhs) }

    #[inline(always)]
    pub fn blend(mask: IMask32x8, if_true: Self, if_false: Self) -> Self {
        unsafe {
            Self(_mm256_or_si256(_mm256_and_si256(mask.0,if_true.0),_mm256_andnot_si256(mask.0,if_false.0)))
        }
    }

    #[inline(always)] pub fn wrapping_add(self, r: Self) -> Self { self + r }
    #[inline(always)] pub fn wrapping_sub(self, r: Self) -> Self { self - r }
    #[inline(always)] pub fn wrapping_mul(self, r: Self) -> Self { self * r }

    /// No native 32-bit saturating add in AVX2 — manual overflow detection
    /// via the same unsigned-compare trick as sse2/u32x4.
    #[inline]
    pub fn saturating_add(self, rhs: Self) -> Self {
        unsafe {
            let sum = _mm256_add_epi32(self.0, rhs.0);
            let overflowed = ucmpgt(self.0, sum);
            let all_ones = _mm256_cmpeq_epi32(sum, sum);
            Self(_mm256_or_si256(_mm256_and_si256(overflowed, all_ones), _mm256_andnot_si256(overflowed, sum)))
        }
    }
    #[inline]
    pub fn saturating_sub(self, rhs: Self) -> Self {
        unsafe {
            let underflowed = ucmpgt(rhs.0, self.0);
            let diff = _mm256_sub_epi32(self.0, rhs.0);
            Self(_mm256_andnot_si256(underflowed, diff))
        }
    }
}

impl Add  for u32x8 { type Output=Self; #[inline(always)] fn add(self,r:Self)->Self{Self(unsafe{_mm256_add_epi32(self.0,r.0)})} }
impl AddAssign for u32x8 { #[inline(always)] fn add_assign(&mut self,r:Self){*self=*self+r;} }
impl Sub  for u32x8 { type Output=Self; #[inline(always)] fn sub(self,r:Self)->Self{Self(unsafe{_mm256_sub_epi32(self.0,r.0)})} }
impl SubAssign for u32x8 { #[inline(always)] fn sub_assign(&mut self,r:Self){*self=*self-r;} }

/// Native `_mm256_mullo_epi32` — no shuffle/unpack emulation needed.
impl Mul for u32x8 { type Output=Self; #[inline(always)] fn mul(self,r:Self)->Self{Self(unsafe{_mm256_mullo_epi32(self.0,r.0)})} }
impl MulAssign for u32x8 { #[inline(always)] fn mul_assign(&mut self,r:Self){*self=*self*r;} }

impl BitAnd for u32x8 { type Output=Self; #[inline(always)] fn bitand(self,r:Self)->Self{Self(unsafe{_mm256_and_si256(self.0,r.0)})} }
impl BitAndAssign for u32x8 { #[inline(always)] fn bitand_assign(&mut self,r:Self){*self=*self&r;} }
impl BitOr  for u32x8 { type Output=Self; #[inline(always)] fn bitor (self,r:Self)->Self{Self(unsafe{_mm256_or_si256(self.0,r.0)})} }
impl BitOrAssign  for u32x8 { #[inline(always)] fn bitor_assign (&mut self,r:Self){*self=*self|r;} }
impl BitXor for u32x8 { type Output=Self; #[inline(always)] fn bitxor(self,r:Self)->Self{Self(unsafe{_mm256_xor_si256(self.0,r.0)})} }
impl BitXorAssign for u32x8 { #[inline(always)] fn bitxor_assign(&mut self,r:Self){*self=*self^r;} }
impl Not for u32x8 { type Output=Self; #[inline(always)] fn not(self)->Self{unsafe{let o=_mm256_cmpeq_epi32(self.0,self.0);Self(_mm256_xor_si256(self.0,o))}} }

impl PartialEq for u32x8 {
    fn eq(&self,r:&Self)->bool{unsafe{_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(self.0,r.0)))==0xFF}}
}
impl Eq for u32x8 {}
impl fmt::Debug for u32x8 {
    fn fmt(&self,f:&mut fmt::Formatter<'_>)->fmt::Result{let a=self.to_array();write!(f,"u32x8({},{},{},{},{},{},{},{})",a[0],a[1],a[2],a[3],a[4],a[5],a[6],a[7])}
}
impl fmt::Display for u32x8 {
    fn fmt(&self,f:&mut fmt::Formatter<'_>)->fmt::Result{let a=self.to_array();write!(f,"[{},{},{},{},{},{},{},{}]",a[0],a[1],a[2],a[3],a[4],a[5],a[6],a[7])}
}
impl From<[u32;8]> for u32x8 { fn from(a:[u32;8])->Self{Self::from_array(a)} }
impl From<u32x8> for [u32;8] { fn from(v:u32x8)->[u32;8]{v.to_array()} }
