// crates/mid-math/src/wide/int/avx2/i8x32.rs
//! 32-lane signed 8-bit integer vector — AVX2, x86 / x86_64.
//!
//! Widens sse2/i8x16.rs to `__m256i`. AVX2 adds native
//! `_mm256_min_epi8`/`_mm256_max_epi8`/`_mm256_abs_epi8` (SSE2 had none
//! of these — SSE2's i8x16 needed cmplt+blend for both abs and min/max).
//!
//! Deliberately omits sse2/i8x16.rs's `shuffle_bytes` (SSSE3
//! `_mm_shuffle_epi8`) and `as_i16x8_lo`/`as_i16x8_hi` widening: AVX2's
//! `_mm256_shuffle_epi8`/`_mm256_unpacklo_epi8`/`_mm256_unpackhi_epi8`
//! all operate PER 128-BIT LANE, not across the full 32 bytes — a direct
//! port would silently restrict shuffle indices to their own 16-byte
//! half and scramble the widen split (lanes [0..8)+[16..24) instead of
//! [0..16)), which is a correctness bug, not a style choice. Doing this
//! right needs an explicit lane-fixup (`_mm256_permute4x64_epi64` or
//! `_mm256_permute2x128_si256`) that I didn't want to ship without being
//! able to compile-check it here — flagging as a follow-up.
//!
//! No multiply, same as sse2/i8x16.rs — no native 8-bit SIMD multiply on
//! x86 (SSE2 or AVX2). No shift, same as sse2/i8x16.rs — no byte-granularity
//! shift instruction exists on x86 at all.

#![allow(non_camel_case_types)]

use core::fmt;
use core::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign,
    BitXor, BitXorAssign, Neg, Not, Sub, SubAssign,
};

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use super::imask8x32::IMask8x32;

#[repr(C)]
union UnionCast { i: [i8; 32], v: i8x32 }

/// 32-lane signed 8-bit integer vector. 32 bytes, 32-byte aligned. Backed by `__m256i`.
///
/// Note: no multiply — i8 mul would require widening to i16 first.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct i8x32(pub(crate) __m256i);

impl i8x32 {
    pub const ZERO: Self = unsafe { UnionCast { i: [0; 32] }.v };
    pub const ONE:  Self = unsafe { UnionCast { i: [1; 32] }.v };
    pub const MIN:  Self = unsafe { UnionCast { i: [i8::MIN; 32] }.v };
    pub const MAX:  Self = unsafe { UnionCast { i: [i8::MAX; 32] }.v };

    #[inline(always)]
    pub fn splat(v: i8) -> Self { Self(unsafe { _mm256_set1_epi8(v) }) }

    #[inline(always)]
    pub fn from_array(a: [i8; 32]) -> Self {
        Self(unsafe { _mm256_loadu_si256(a.as_ptr() as *const __m256i) })
    }

    /// Load from a `[u8; 32]` — common when processing raw byte streams.
    #[inline(always)]
    pub fn from_bytes(b: [u8; 32]) -> Self {
        // Safety: [u8;32] and [i8;32] have identical layout.
        Self(unsafe { _mm256_loadu_si256(b.as_ptr() as *const __m256i) })
    }

    #[inline(always)]
    pub fn to_array(self) -> [i8; 32] {
        unsafe { let mut a=[0i8;32]; _mm256_storeu_si256(a.as_mut_ptr() as *mut __m256i, self.0); a }
    }

    #[inline(always)]
    pub fn to_bytes(self) -> [u8; 32] {
        unsafe { let mut a=[0u8;32]; _mm256_storeu_si256(a.as_mut_ptr() as *mut __m256i, self.0); a }
    }

    #[inline]
    pub fn get(self, i: usize) -> i8 {
        assert!(i < 32, "i8x32::get — lane {i} out of bounds (max 31)");
        unsafe { UnionCast { v: self }.i[i] }
    }

    /// Saturating add — clamps to `[i8::MIN, i8::MAX]`. AVX2 native.
    #[inline(always)] pub fn saturating_add(self, rhs: Self) -> Self { Self(unsafe { _mm256_adds_epi8(self.0, rhs.0) }) }
    /// Saturating sub — clamps to `[i8::MIN, i8::MAX]`. AVX2 native.
    #[inline(always)] pub fn saturating_sub(self, rhs: Self) -> Self { Self(unsafe { _mm256_subs_epi8(self.0, rhs.0) }) }

    /// Absolute value per lane (wrapping — `abs(i8::MIN) == i8::MIN`).
    /// AVX2 native `_mm256_abs_epi8` — SSE2 needs cmplt+sub+blend.
    #[inline(always)]
    pub fn abs(self) -> Self { Self(unsafe { _mm256_abs_epi8(self.0) }) }

    /// Per-lane minimum (signed). AVX2 native `_mm256_min_epi8`.
    #[inline(always)] pub fn min(self, rhs: Self) -> Self { Self(unsafe { _mm256_min_epi8(self.0, rhs.0) }) }
    /// Per-lane maximum (signed). AVX2 native `_mm256_max_epi8`.
    #[inline(always)] pub fn max(self, rhs: Self) -> Self { Self(unsafe { _mm256_max_epi8(self.0, rhs.0) }) }
    #[inline(always)] pub fn clamp(self, lo: Self, hi: Self) -> Self { self.max(lo).min(hi) }

    #[inline] pub fn min_element(self) -> i8 { self.to_array().iter().copied().reduce(i8::min).unwrap() }
    #[inline] pub fn max_element(self) -> i8 { self.to_array().iter().copied().reduce(i8::max).unwrap() }
    /// Horizontal sum. Result is i32 to avoid multi-level overflow.
    #[inline] pub fn element_sum(self) -> i32 { self.to_array().iter().map(|&x| x as i32).sum() }

    #[inline(always)] pub fn cmpeq(self, rhs: Self) -> IMask8x32 { IMask8x32(unsafe { _mm256_cmpeq_epi8(self.0, rhs.0) }) }
    #[inline(always)] pub fn cmpne(self, rhs: Self) -> IMask8x32 { !self.cmpeq(rhs) }
    #[inline(always)] pub fn cmpgt(self, rhs: Self) -> IMask8x32 { IMask8x32(unsafe { _mm256_cmpgt_epi8(self.0, rhs.0) }) }
    #[inline(always)] pub fn cmplt(self, rhs: Self) -> IMask8x32 { rhs.cmpgt(self) }
    #[inline(always)] pub fn cmpge(self, rhs: Self) -> IMask8x32 { !self.cmplt(rhs) }
    #[inline(always)] pub fn cmple(self, rhs: Self) -> IMask8x32 { !self.cmpgt(rhs) }

    #[inline(always)]
    pub fn blend(mask: IMask8x32, if_true: Self, if_false: Self) -> Self {
        unsafe {
            Self(_mm256_or_si256(
                _mm256_and_si256(mask.0, if_true.0),
                _mm256_andnot_si256(mask.0, if_false.0),
            ))
        }
    }

    /// Number of lanes that compare equal to `needle`.
    #[inline]
    pub fn count_eq(self, needle: Self) -> u32 { self.cmpeq(needle).count_true() }
    /// True if any lane equals `needle`.
    #[inline]
    pub fn contains(self, needle: i8) -> bool { self.count_eq(Self::splat(needle)) > 0 }

    #[inline(always)] pub fn wrapping_add(self, r: Self) -> Self { self + r }
    #[inline(always)] pub fn wrapping_sub(self, r: Self) -> Self { self - r }
}

impl Add for i8x32 { type Output=Self; #[inline(always)] fn add(self,r:Self)->Self{Self(unsafe{_mm256_add_epi8(self.0,r.0)})} }
impl AddAssign for i8x32 { #[inline(always)] fn add_assign(&mut self,r:Self){*self=*self+r;} }
impl Sub for i8x32 { type Output=Self; #[inline(always)] fn sub(self,r:Self)->Self{Self(unsafe{_mm256_sub_epi8(self.0,r.0)})} }
impl SubAssign for i8x32 { #[inline(always)] fn sub_assign(&mut self,r:Self){*self=*self-r;} }
impl Neg for i8x32 { type Output=Self; #[inline(always)] fn neg(self)->Self{Self(unsafe{_mm256_sub_epi8(_mm256_setzero_si256(),self.0)})} }

impl BitAnd for i8x32 { type Output=Self; #[inline(always)] fn bitand(self,r:Self)->Self{Self(unsafe{_mm256_and_si256(self.0,r.0)})} }
impl BitAndAssign for i8x32 { #[inline(always)] fn bitand_assign(&mut self,r:Self){*self=*self&r;} }
impl BitOr  for i8x32 { type Output=Self; #[inline(always)] fn bitor (self,r:Self)->Self{Self(unsafe{_mm256_or_si256(self.0,r.0)})} }
impl BitOrAssign  for i8x32 { #[inline(always)] fn bitor_assign (&mut self,r:Self){*self=*self|r;} }
impl BitXor for i8x32 { type Output=Self; #[inline(always)] fn bitxor(self,r:Self)->Self{Self(unsafe{_mm256_xor_si256(self.0,r.0)})} }
impl BitXorAssign for i8x32 { #[inline(always)] fn bitxor_assign(&mut self,r:Self){*self=*self^r;} }
impl Not for i8x32 {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self { unsafe { let ones = _mm256_cmpeq_epi8(self.0, self.0); Self(_mm256_xor_si256(self.0, ones)) } }
}

impl PartialEq for i8x32 {
    #[inline]
    fn eq(&self, r: &Self) -> bool { unsafe { _mm256_movemask_epi8(_mm256_cmpeq_epi8(self.0, r.0)) == -1 } }
}
impl Eq for i8x32 {}

impl fmt::Debug for i8x32 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "i8x32({:?})", self.to_array()) }
}
impl fmt::Display for i8x32 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "{:?}", self.to_array()) }
}
impl From<[i8; 32]> for i8x32 { #[inline] fn from(a: [i8;32]) -> Self { Self::from_array(a) } }
impl From<i8x32> for [i8; 32] { #[inline] fn from(v: i8x32) -> Self { v.to_array() } }
impl From<[u8; 32]> for i8x32 { #[inline] fn from(b: [u8;32]) -> Self { Self::from_bytes(b) } }
impl From<i8x32> for [u8; 32] { #[inline] fn from(v: i8x32) -> Self { v.to_bytes() } }
