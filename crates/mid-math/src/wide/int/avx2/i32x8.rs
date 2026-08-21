// crates/mid-math/src/wide/int/avx2/i32x8.rs
//! 8-lane signed 32-bit integer vector — AVX2, x86 / x86_64.
//!
//! Widens sse2/i32x4.rs to `__m256i`. AVX2 adds native `_mm256_min_epi32`
//! / `_mm256_max_epi32` / `_mm256_mullo_epi32` / `_mm256_abs_epi32` —
//! none of these exist in plain SSE2, which had to emulate min/max via
//! compare+blend and multiply via a shuffle/unpack chain (see
//! sse2/i32x4.rs). No shuffle emulation needed here.

#![allow(non_camel_case_types)]

use core::fmt;
use core::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign,
    BitXor, BitXorAssign, Mul, MulAssign, Neg, Not, Sub, SubAssign,
};

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use super::imask32x8::IMask32x8;

#[repr(C)]
union UnionCast { i: [i32; 8], v: i32x8 }

/// 8-lane signed 32-bit integer vector. 32 bytes, 32-byte aligned. Backed by `__m256i`.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct i32x8(pub(crate) __m256i);

impl i32x8 {
    pub const ZERO: Self = unsafe { UnionCast { i: [0; 8] }.v };
    pub const ONE:  Self = unsafe { UnionCast { i: [1; 8] }.v };
    pub const MIN:  Self = unsafe { UnionCast { i: [i32::MIN; 8] }.v };
    pub const MAX:  Self = unsafe { UnionCast { i: [i32::MAX; 8] }.v };

    #[inline(always)]
    pub fn splat(v: i32) -> Self { Self(unsafe { _mm256_set1_epi32(v) }) }

    #[inline(always)]
    pub fn new(a: i32, b: i32, c: i32, d: i32, e: i32, f: i32, g: i32, h: i32) -> Self {
        Self(unsafe { _mm256_set_epi32(h, g, f, e, d, c, b, a) })
    }

    #[inline(always)]
    pub fn from_array(a: [i32; 8]) -> Self {
        Self(unsafe { _mm256_loadu_si256(a.as_ptr() as *const __m256i) })
    }

    #[inline(always)]
    pub fn to_array(self) -> [i32; 8] {
        unsafe {
            let mut a = [0i32; 8];
            _mm256_storeu_si256(a.as_mut_ptr() as *mut __m256i, self.0);
            a
        }
    }

    #[inline]
    pub fn get(self, i: usize) -> i32 {
        assert!(i < 8, "i32x8::get — lane {i} out of bounds (max 7)");
        unsafe { UnionCast { v: self }.i[i] }
    }

    /// Absolute value per lane. AVX2 native `_mm256_abs_epi32` —
    /// SSE2 has no equivalent (needs cmplt+sub+blend, see i32x4::abs).
    #[inline(always)]
    pub fn abs(self) -> Self { Self(unsafe { _mm256_abs_epi32(self.0) }) }

    /// Per-lane minimum. AVX2 native — SSE2 must emulate via cmplt+blend.
    #[inline(always)]
    pub fn min(self, rhs: Self) -> Self { Self(unsafe { _mm256_min_epi32(self.0, rhs.0) }) }
    /// Per-lane maximum. AVX2 native — SSE2 must emulate via cmpgt+blend.
    #[inline(always)]
    pub fn max(self, rhs: Self) -> Self { Self(unsafe { _mm256_max_epi32(self.0, rhs.0) }) }
    #[inline(always)]
    pub fn clamp(self, lo: Self, hi: Self) -> Self { self.max(lo).min(hi) }

    #[inline]
    pub fn min_element(self) -> i32 { self.to_array().iter().copied().reduce(i32::min).unwrap() }
    #[inline]
    pub fn max_element(self) -> i32 { self.to_array().iter().copied().reduce(i32::max).unwrap() }
    #[inline]
    pub fn element_sum(self) -> i32 {
        self.to_array().iter().fold(0i32, |acc, &x| acc.wrapping_add(x))
    }

    // ── Shifts — variable-count via _mm256_sll/_mm256_sra/_mm256_srl, count
    // lives in the low 64 bits of a __m128i (same calling convention as the
    // SSE2 128-bit shift intrinsics — count is NOT per-lane, it's uniform).

    #[inline(always)]
    pub fn shl(self, count: u32) -> Self {
        unsafe {
            let cnt = _mm_cvtsi32_si128(count as i32);
            Self(_mm256_sll_epi32(self.0, cnt))
        }
    }
    #[inline(always)]
    pub fn shr_arithmetic(self, count: u32) -> Self {
        unsafe {
            let cnt = _mm_cvtsi32_si128(count as i32);
            Self(_mm256_sra_epi32(self.0, cnt))
        }
    }
    #[inline(always)]
    pub fn shr_logical(self, count: u32) -> Self {
        unsafe {
            let cnt = _mm_cvtsi32_si128(count as i32);
            Self(_mm256_srl_epi32(self.0, cnt))
        }
    }

    #[inline(always)] pub fn cmpeq(self, rhs: Self) -> IMask32x8 { IMask32x8(unsafe { _mm256_cmpeq_epi32(self.0, rhs.0) }) }
    #[inline(always)] pub fn cmpne(self, rhs: Self) -> IMask32x8 { !self.cmpeq(rhs) }
    #[inline(always)] pub fn cmpgt(self, rhs: Self) -> IMask32x8 { IMask32x8(unsafe { _mm256_cmpgt_epi32(self.0, rhs.0) }) }
    /// No native `_mm256_cmplt_epi32` (unlike the 128-bit convenience
    /// wrapper SSE2 gets) — implemented as swapped-operand cmpgt.
    #[inline(always)] pub fn cmplt(self, rhs: Self) -> IMask32x8 { rhs.cmpgt(self) }
    #[inline(always)] pub fn cmpge(self, rhs: Self) -> IMask32x8 { !self.cmplt(rhs) }
    #[inline(always)] pub fn cmple(self, rhs: Self) -> IMask32x8 { !self.cmpgt(rhs) }

    #[inline(always)]
    pub fn blend(mask: IMask32x8, if_true: Self, if_false: Self) -> Self {
        unsafe {
            Self(_mm256_or_si256(
                _mm256_and_si256(mask.0, if_true.0),
                _mm256_andnot_si256(mask.0, if_false.0),
            ))
        }
    }

    #[inline(always)] pub fn wrapping_add(self, r: Self) -> Self { self + r }
    #[inline(always)] pub fn wrapping_sub(self, r: Self) -> Self { self - r }
    #[inline(always)] pub fn wrapping_mul(self, r: Self) -> Self { self * r }

    /// No native 32-bit saturating add in AVX2 (same gap as SSE2) — scalar loop.
    #[inline]
    pub fn saturating_add(self, rhs: Self) -> Self {
        let a = self.to_array();
        let b = rhs.to_array();
        Self::from_array(core::array::from_fn(|i| a[i].saturating_add(b[i])))
    }
    #[inline]
    pub fn saturating_sub(self, rhs: Self) -> Self {
        let a = self.to_array();
        let b = rhs.to_array();
        Self::from_array(core::array::from_fn(|i| a[i].saturating_sub(b[i])))
    }
}

impl Add for i32x8 { type Output=Self; #[inline(always)] fn add(self,r:Self)->Self{Self(unsafe{_mm256_add_epi32(self.0,r.0)})} }
impl AddAssign for i32x8 { #[inline(always)] fn add_assign(&mut self,r:Self){*self=*self+r;} }
impl Sub for i32x8 { type Output=Self; #[inline(always)] fn sub(self,r:Self)->Self{Self(unsafe{_mm256_sub_epi32(self.0,r.0)})} }
impl SubAssign for i32x8 { #[inline(always)] fn sub_assign(&mut self,r:Self){*self=*self-r;} }
impl Neg for i32x8 { type Output=Self; #[inline(always)] fn neg(self)->Self{Self(unsafe{_mm256_sub_epi32(_mm256_setzero_si256(),self.0)})} }

/// Native `_mm256_mullo_epi32` (SSE4.1/AVX2) — no shuffle/unpack emulation
/// needed, unlike SSE2's i32x4::mul.
impl Mul for i32x8 { type Output=Self; #[inline(always)] fn mul(self,r:Self)->Self{Self(unsafe{_mm256_mullo_epi32(self.0,r.0)})} }
impl MulAssign for i32x8 { #[inline(always)] fn mul_assign(&mut self,r:Self){*self=*self*r;} }

impl BitAnd for i32x8 { type Output=Self; #[inline(always)] fn bitand(self,r:Self)->Self{Self(unsafe{_mm256_and_si256(self.0,r.0)})} }
impl BitAndAssign for i32x8 { #[inline(always)] fn bitand_assign(&mut self,r:Self){*self=*self&r;} }
impl BitOr  for i32x8 { type Output=Self; #[inline(always)] fn bitor (self,r:Self)->Self{Self(unsafe{_mm256_or_si256(self.0,r.0)})} }
impl BitOrAssign  for i32x8 { #[inline(always)] fn bitor_assign (&mut self,r:Self){*self=*self|r;} }
impl BitXor for i32x8 { type Output=Self; #[inline(always)] fn bitxor(self,r:Self)->Self{Self(unsafe{_mm256_xor_si256(self.0,r.0)})} }
impl BitXorAssign for i32x8 { #[inline(always)] fn bitxor_assign(&mut self,r:Self){*self=*self^r;} }
impl Not for i32x8 {
    type Output=Self;
    #[inline(always)]
    fn not(self)->Self{unsafe{let ones=_mm256_cmpeq_epi32(self.0,self.0);Self(_mm256_xor_si256(self.0,ones))}}
}

impl PartialEq for i32x8 {
    #[inline]
    fn eq(&self,r:&Self)->bool{unsafe{_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(self.0,r.0)))==0xFF}}
}
impl Eq for i32x8 {}

impl fmt::Debug for i32x8 {
    fn fmt(&self,f:&mut fmt::Formatter<'_>)->fmt::Result{
        let a=self.to_array();write!(f,"i32x8({},{},{},{},{},{},{},{})",a[0],a[1],a[2],a[3],a[4],a[5],a[6],a[7])
    }
}
impl fmt::Display for i32x8 {
    fn fmt(&self,f:&mut fmt::Formatter<'_>)->fmt::Result{
        let a=self.to_array();write!(f,"[{},{},{},{},{},{},{},{}]",a[0],a[1],a[2],a[3],a[4],a[5],a[6],a[7])
    }
}
impl From<[i32;8]> for i32x8 { #[inline] fn from(a:[i32;8])->Self{Self::from_array(a)} }
impl From<i32x8> for [i32;8] { #[inline] fn from(v:i32x8)->[i32;8]{v.to_array()} }
