// crates/mid-math/src/wide/int/sse2/i32x4.rs
//! 4-lane signed 32-bit integer vector — SSE2, x86 / x86_64.

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

use super::imask4::IMask4;

#[repr(C)]
union UnionCast { i: [i32; 4], v: i32x4 }

/// 4-lane signed 32-bit integer vector. 16 bytes, 16-byte aligned. Backed by `__m128i`.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct i32x4(pub(crate) __m128i);

impl i32x4 {
    pub const ZERO: Self = unsafe { UnionCast { i: [0; 4] }.v };
    pub const ONE:  Self = unsafe { UnionCast { i: [1; 4] }.v };
    pub const MIN:  Self = unsafe { UnionCast { i: [i32::MIN; 4] }.v };
    pub const MAX:  Self = unsafe { UnionCast { i: [i32::MAX; 4] }.v };

    #[inline(always)]
    pub fn splat(v: i32) -> Self { Self(unsafe { _mm_set1_epi32(v) }) }

    #[inline(always)]
    pub fn new(a: i32, b: i32, c: i32, d: i32) -> Self {
        Self(unsafe { _mm_set_epi32(d, c, b, a) })
    }

    #[inline(always)]
    pub fn from_array(a: [i32; 4]) -> Self {
        Self(unsafe { _mm_loadu_si128(a.as_ptr() as *const __m128i) })
    }

    #[inline(always)]
    pub fn to_array(self) -> [i32; 4] {
        unsafe {
            let mut a = [0i32; 4];
            _mm_storeu_si128(a.as_mut_ptr() as *mut __m128i, self.0);
            a
        }
    }

    #[inline]
    pub fn get(self, i: usize) -> i32 {
        assert!(i < 4, "i32x4::get — lane {i} out of bounds (max 3)");
        unsafe { UnionCast { v: self }.i[i] }
    }

    #[inline]
    pub fn abs(self) -> Self {
        unsafe {
            let zero = _mm_setzero_si128();
            let is_neg = _mm_cmplt_epi32(self.0, zero);
            let negated = _mm_sub_epi32(zero, self.0);
            Self(_mm_or_si128(
                _mm_and_si128(is_neg, negated),
                _mm_andnot_si128(is_neg, self.0),
            ))
        }
    }

    #[inline]
    pub fn min(self, rhs: Self) -> Self { Self::blend(self.cmplt(rhs), self, rhs) }

    #[inline]
    pub fn max(self, rhs: Self) -> Self { Self::blend(self.cmpgt(rhs), self, rhs) }

    #[inline]
    pub fn clamp(self, lo: Self, hi: Self) -> Self { self.max(lo).min(hi) }

    #[inline]
    pub fn min_element(self) -> i32 {
        let a = self.to_array();
        a[0].min(a[1]).min(a[2]).min(a[3])
    }

    #[inline]
    pub fn max_element(self) -> i32 {
        let a = self.to_array();
        a[0].max(a[1]).max(a[2]).max(a[3])
    }

    #[inline]
    pub fn element_sum(self) -> i32 {
        let a = self.to_array();
        a[0].wrapping_add(a[1]).wrapping_add(a[2]).wrapping_add(a[3])
    }

    // ── Shifts — use variable-count intrinsics (_mm_sll/_mm_sra/_mm_srl)
    // NOT the immediate variants (_mm_slli/_mm_srai/_mm_srli) which require
    // a compile-time constant. Load count into lowest lane of __m128i via
    // _mm_cvtsi32_si128 and pass to the variable-count shift.

    /// Logical left shift all lanes by `count` bits.
    #[inline(always)]
    pub fn shl(self, count: u32) -> Self {
        unsafe {
            let cnt = _mm_cvtsi32_si128(count as i32);
            Self(_mm_sll_epi32(self.0, cnt))
        }
    }

    /// Arithmetic right shift (sign-extend) all lanes by `count` bits.
    #[inline(always)]
    pub fn shr_arithmetic(self, count: u32) -> Self {
        unsafe {
            let cnt = _mm_cvtsi32_si128(count as i32);
            Self(_mm_sra_epi32(self.0, cnt))
        }
    }

    /// Logical right shift (zero-fill) all lanes by `count` bits.
    #[inline(always)]
    pub fn shr_logical(self, count: u32) -> Self {
        unsafe {
            let cnt = _mm_cvtsi32_si128(count as i32);
            Self(_mm_srl_epi32(self.0, cnt))
        }
    }

    #[inline(always)]
    pub fn cmpeq(self, rhs: Self) -> IMask4 { IMask4(unsafe { _mm_cmpeq_epi32(self.0, rhs.0) }) }
    #[inline(always)]
    pub fn cmpne(self, rhs: Self) -> IMask4 { !self.cmpeq(rhs) }
    #[inline(always)]
    pub fn cmpgt(self, rhs: Self) -> IMask4 { IMask4(unsafe { _mm_cmpgt_epi32(self.0, rhs.0) }) }
    #[inline(always)]
    pub fn cmplt(self, rhs: Self) -> IMask4 { IMask4(unsafe { _mm_cmplt_epi32(self.0, rhs.0) }) }
    #[inline(always)]
    pub fn cmpge(self, rhs: Self) -> IMask4 { !self.cmplt(rhs) }
    #[inline(always)]
    pub fn cmple(self, rhs: Self) -> IMask4 { !self.cmpgt(rhs) }

    #[inline(always)]
    pub fn blend(mask: IMask4, if_true: Self, if_false: Self) -> Self {
        unsafe {
            Self(_mm_or_si128(
                _mm_and_si128(mask.0, if_true.0),
                _mm_andnot_si128(mask.0, if_false.0),
            ))
        }
    }

    #[inline(always)] pub fn wrapping_add(self, r: Self) -> Self { self + r }
    #[inline(always)] pub fn wrapping_sub(self, r: Self) -> Self { self - r }
    #[inline(always)] pub fn wrapping_mul(self, r: Self) -> Self { self * r }

    #[inline]
    pub fn saturating_add(self, rhs: Self) -> Self {
        let a = self.to_array();
        let b = rhs.to_array();
        Self::from_array([
            a[0].saturating_add(b[0]), a[1].saturating_add(b[1]),
            a[2].saturating_add(b[2]), a[3].saturating_add(b[3]),
        ])
    }

    #[inline]
    pub fn saturating_sub(self, rhs: Self) -> Self {
        let a = self.to_array();
        let b = rhs.to_array();
        Self::from_array([
            a[0].saturating_sub(b[0]), a[1].saturating_sub(b[1]),
            a[2].saturating_sub(b[2]), a[3].saturating_sub(b[3]),
        ])
    }
}

impl Add for i32x4 { type Output=Self; #[inline(always)] fn add(self,r:Self)->Self{Self(unsafe{_mm_add_epi32(self.0,r.0)})} }
impl AddAssign for i32x4 { #[inline(always)] fn add_assign(&mut self,r:Self){*self=*self+r;} }
impl Sub for i32x4 { type Output=Self; #[inline(always)] fn sub(self,r:Self)->Self{Self(unsafe{_mm_sub_epi32(self.0,r.0)})} }
impl SubAssign for i32x4 { #[inline(always)] fn sub_assign(&mut self,r:Self){*self=*self-r;} }
impl Neg for i32x4 { type Output=Self; #[inline(always)] fn neg(self)->Self{Self(unsafe{_mm_sub_epi32(_mm_setzero_si128(),self.0)})} }

impl Mul for i32x4 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        unsafe {
            let a13 = _mm_shuffle_epi32(self.0, 0xF5);
            let b13 = _mm_shuffle_epi32(rhs.0,  0xF5);
            let prod02 = _mm_mul_epu32(self.0, rhs.0);
            let prod13 = _mm_mul_epu32(a13, b13);
            let lo02 = _mm_shuffle_epi32(prod02, 0x08);
            let lo13 = _mm_shuffle_epi32(prod13, 0x08);
            Self(_mm_unpacklo_epi32(lo02, lo13))
        }
    }
}
impl MulAssign for i32x4 { #[inline(always)] fn mul_assign(&mut self,r:Self){*self=*self*r;} }

impl BitAnd for i32x4 { type Output=Self; #[inline(always)] fn bitand(self,r:Self)->Self{Self(unsafe{_mm_and_si128(self.0,r.0)})} }
impl BitAndAssign for i32x4 { #[inline(always)] fn bitand_assign(&mut self,r:Self){*self=*self&r;} }
impl BitOr  for i32x4 { type Output=Self; #[inline(always)] fn bitor (self,r:Self)->Self{Self(unsafe{_mm_or_si128(self.0,r.0)})} }
impl BitOrAssign  for i32x4 { #[inline(always)] fn bitor_assign (&mut self,r:Self){*self=*self|r;} }
impl BitXor for i32x4 { type Output=Self; #[inline(always)] fn bitxor(self,r:Self)->Self{Self(unsafe{_mm_xor_si128(self.0,r.0)})} }
impl BitXorAssign for i32x4 { #[inline(always)] fn bitxor_assign(&mut self,r:Self){*self=*self^r;} }
impl Not for i32x4 {
    type Output=Self;
    #[inline(always)]
    fn not(self)->Self{unsafe{let ones=_mm_cmpeq_epi32(self.0,self.0);Self(_mm_xor_si128(self.0,ones))}}
}

impl PartialEq for i32x4 {
    #[inline]
    fn eq(&self,r:&Self)->bool{unsafe{_mm_movemask_ps(_mm_castsi128_ps(_mm_cmpeq_epi32(self.0,r.0)))==0b1111}}
}
impl Eq for i32x4 {}

impl fmt::Debug for i32x4 {
    fn fmt(&self,f:&mut fmt::Formatter<'_>)->fmt::Result{
        let a=self.to_array();write!(f,"i32x4({},{},{},{})",a[0],a[1],a[2],a[3])
    }
}
impl fmt::Display for i32x4 {
    fn fmt(&self,f:&mut fmt::Formatter<'_>)->fmt::Result{
        let a=self.to_array();write!(f,"[{},{},{},{}]",a[0],a[1],a[2],a[3])
    }
}
impl From<[i32;4]> for i32x4 { #[inline] fn from(a:[i32;4])->Self{Self::from_array(a)} }
impl From<i32x4> for [i32;4] { #[inline] fn from(v:i32x4)->[i32;4]{v.to_array()} }
