// crates/mid-math/src/wide/int/avx2/i16x16.rs
//! 16-lane signed 16-bit integer vector — AVX2, x86 / x86_64.
//!
//! Widens sse2/i16x8.rs to `__m256i`. All ops here are per-lane
//! (add/sub/cmp/min/max/mul/shift/saturating) so widening from the SSE2
//! 128-bit forms is direct — no cross-lane shuffle involved.
//!
//! Deliberately omits `as_i32x8_lo`/`as_i32x8_hi`/`pack_i32x8` widen/pack
//! bridges that sse2/i16x8.rs has (`as_i32x4_lo`/`as_i32x4_hi`/
//! `pack_i32x4`): AVX2's `_mm256_unpacklo_epi16`/`_mm256_unpackhi_epi16`
//! operate PER 128-BIT LANE, not across the full 256 bits — a naive port
//! of the SSE2 sign-extend-via-unpack trick would silently interleave
//! lanes [0..4) with [8..12) instead of [0..8) with [8..16), which is a
//! correctness bug, not a style choice. Doing this right needs an extra
//! `_mm256_permute4x64_epi64` lane-fixup that I didn't want to ship
//! without being able to compile-check it here — flagging as a follow-up
//! rather than guessing.

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

use super::imask16x16::IMask16x16;

#[repr(C)]
union UnionCast { i: [i16; 16], v: i16x16 }

/// 16-lane signed 16-bit integer vector. 32 bytes, 32-byte aligned. Backed by `__m256i`.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct i16x16(pub(crate) __m256i);

impl i16x16 {
    pub const ZERO: Self = unsafe { UnionCast { i: [0; 16] }.v };
    pub const ONE:  Self = unsafe { UnionCast { i: [1; 16] }.v };
    pub const MIN:  Self = unsafe { UnionCast { i: [i16::MIN; 16] }.v };
    pub const MAX:  Self = unsafe { UnionCast { i: [i16::MAX; 16] }.v };

    #[inline(always)]
    pub fn splat(v: i16) -> Self { Self(unsafe { _mm256_set1_epi16(v) }) }

    #[inline(always)]
    pub fn new(a:i16,b:i16,c:i16,d:i16,e:i16,f:i16,g:i16,h:i16,i_:i16,j:i16,k:i16,l:i16,m:i16,n:i16,o:i16,p:i16) -> Self {
        Self(unsafe { _mm256_set_epi16(p,o,n,m,l,k,j,i_,h,g,f,e,d,c,b,a) })
    }

    #[inline(always)]
    pub fn from_array(a: [i16; 16]) -> Self {
        Self(unsafe { _mm256_loadu_si256(a.as_ptr() as *const __m256i) })
    }

    #[inline(always)]
    pub fn to_array(self) -> [i16; 16] {
        unsafe { let mut a=[0i16;16]; _mm256_storeu_si256(a.as_mut_ptr() as *mut __m256i,self.0); a }
    }

    #[inline]
    pub fn get(self, i: usize) -> i16 {
        assert!(i < 16, "i16x16::get — lane {i} out of bounds (max 15)");
        unsafe { UnionCast { v: self }.i[i] }
    }

    /// Absolute value per lane. AVX2 native `_mm256_abs_epi16`.
    #[inline(always)]
    pub fn abs(self) -> Self { Self(unsafe { _mm256_abs_epi16(self.0) }) }

    #[inline(always)] pub fn min(self, rhs: Self) -> Self { Self(unsafe{_mm256_min_epi16(self.0,rhs.0)}) }
    #[inline(always)] pub fn max(self, rhs: Self) -> Self { Self(unsafe{_mm256_max_epi16(self.0,rhs.0)}) }
    #[inline(always)] pub fn clamp(self, lo: Self, hi: Self) -> Self { self.max(lo).min(hi) }

    #[inline] pub fn min_element(self) -> i16 { self.to_array().iter().copied().reduce(i16::min).unwrap() }
    #[inline] pub fn max_element(self) -> i16 { self.to_array().iter().copied().reduce(i16::max).unwrap() }
    #[inline] pub fn element_sum(self) -> i32 { self.to_array().iter().map(|&x| x as i32).sum() }

    #[inline(always)] pub fn mul_lo(self, rhs: Self) -> Self { Self(unsafe{_mm256_mullo_epi16(self.0,rhs.0)}) }
    #[inline(always)] pub fn mul_high(self, rhs: Self) -> Self { Self(unsafe{_mm256_mulhi_epi16(self.0,rhs.0)}) }

    #[inline(always)] pub fn saturating_add(self, rhs: Self) -> Self { Self(unsafe{_mm256_adds_epi16(self.0,rhs.0)}) }
    #[inline(always)] pub fn saturating_sub(self, rhs: Self) -> Self { Self(unsafe{_mm256_subs_epi16(self.0,rhs.0)}) }

    #[inline(always)]
    pub fn shl(self, count: u32) -> Self {
        unsafe { let cnt = _mm_cvtsi32_si128(count as i32); Self(_mm256_sll_epi16(self.0, cnt)) }
    }
    #[inline(always)]
    pub fn shr_arithmetic(self, count: u32) -> Self {
        unsafe { let cnt = _mm_cvtsi32_si128(count as i32); Self(_mm256_sra_epi16(self.0, cnt)) }
    }
    #[inline(always)]
    pub fn shr_logical(self, count: u32) -> Self {
        unsafe { let cnt = _mm_cvtsi32_si128(count as i32); Self(_mm256_srl_epi16(self.0, cnt)) }
    }

    #[inline(always)] pub fn cmpeq(self, rhs: Self) -> IMask16x16 { IMask16x16(unsafe{_mm256_cmpeq_epi16(self.0,rhs.0)}) }
    #[inline(always)] pub fn cmpne(self, rhs: Self) -> IMask16x16 { !self.cmpeq(rhs) }
    #[inline(always)] pub fn cmpgt(self, rhs: Self) -> IMask16x16 { IMask16x16(unsafe{_mm256_cmpgt_epi16(self.0,rhs.0)}) }
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

impl Add  for i16x16{type Output=Self;#[inline(always)]fn add(self,r:Self)->Self{Self(unsafe{_mm256_add_epi16(self.0,r.0)})}}
impl AddAssign for i16x16{#[inline(always)]fn add_assign(&mut self,r:Self){*self=*self+r;}}
impl Sub  for i16x16{type Output=Self;#[inline(always)]fn sub(self,r:Self)->Self{Self(unsafe{_mm256_sub_epi16(self.0,r.0)})}}
impl SubAssign for i16x16{#[inline(always)]fn sub_assign(&mut self,r:Self){*self=*self-r;}}
impl Neg  for i16x16{type Output=Self;#[inline(always)]fn neg(self)->Self{Self(unsafe{_mm256_sub_epi16(_mm256_setzero_si256(),self.0)})}}
impl Mul  for i16x16{type Output=Self;#[inline(always)]fn mul(self,r:Self)->Self{self.mul_lo(r)}}
impl MulAssign for i16x16{#[inline(always)]fn mul_assign(&mut self,r:Self){*self=*self*r;}}
impl BitAnd for i16x16{type Output=Self;#[inline(always)]fn bitand(self,r:Self)->Self{Self(unsafe{_mm256_and_si256(self.0,r.0)})}}
impl BitAndAssign for i16x16{#[inline(always)]fn bitand_assign(&mut self,r:Self){*self=*self&r;}}
impl BitOr  for i16x16{type Output=Self;#[inline(always)]fn bitor (self,r:Self)->Self{Self(unsafe{_mm256_or_si256(self.0,r.0)})}}
impl BitOrAssign  for i16x16{#[inline(always)]fn bitor_assign (&mut self,r:Self){*self=*self|r;}}
impl BitXor for i16x16{type Output=Self;#[inline(always)]fn bitxor(self,r:Self)->Self{Self(unsafe{_mm256_xor_si256(self.0,r.0)})}}
impl BitXorAssign for i16x16{#[inline(always)]fn bitxor_assign(&mut self,r:Self){*self=*self^r;}}
impl Not for i16x16{type Output=Self;#[inline(always)]fn not(self)->Self{unsafe{let o=_mm256_cmpeq_epi16(self.0,self.0);Self(_mm256_xor_si256(self.0,o))}}}

impl PartialEq for i16x16{fn eq(&self,r:&Self)->bool{unsafe{_mm256_movemask_epi8(_mm256_cmpeq_epi16(self.0,r.0))==-1}}}
impl Eq for i16x16{}
impl fmt::Debug for i16x16{
    fn fmt(&self,f:&mut fmt::Formatter<'_>)->fmt::Result{
        let a=self.to_array();
        write!(f,"i16x16({},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{})",
            a[0],a[1],a[2],a[3],a[4],a[5],a[6],a[7],a[8],a[9],a[10],a[11],a[12],a[13],a[14],a[15])
    }
}
impl fmt::Display for i16x16{
    fn fmt(&self,f:&mut fmt::Formatter<'_>)->fmt::Result{
        let a=self.to_array();
        write!(f,"[{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}]",
            a[0],a[1],a[2],a[3],a[4],a[5],a[6],a[7],a[8],a[9],a[10],a[11],a[12],a[13],a[14],a[15])
    }
}
impl From<[i16;16]> for i16x16{fn from(a:[i16;16])->Self{Self::from_array(a)}}
impl From<i16x16> for [i16;16]{fn from(v:i16x16)->[i16;16]{v.to_array()}}
