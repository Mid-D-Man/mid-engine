// crates/mid-math/src/wide/int/avx2/i8x32.rs
//! 32-lane signed 8-bit integer vector — AVX2, x86 / x86_64.
//!
//! Widens sse2/i8x16.rs to `__m256i`. AVX2 adds native
//! `_mm256_min_epi8`/`_mm256_max_epi8`/`_mm256_abs_epi8` (SSE2 had none
//! of these — SSE2's i8x16 needed cmplt+blend for both abs and min/max).
//!
//! `as_i16x16_lo`/`as_i16x16_hi` sign-extend via the dedicated
//! `_mm256_cvtepi8_epi16` widen instruction — not a shuffle, so no
//! per-128-bit-lane cross-lane hazard (see i16x16.rs's module docs).
//!
//! `shuffle_bytes` uses `_mm256_shuffle_epi8` directly. That genuinely
//! IS per-128-bit-lane (it's a real shuffle, unlike the cvt widen
//! instructions above) — documented explicitly on the method itself
//! rather than silently ported as if it behaved like a flat 32-byte
//! shuffle. See `to_i8x16_pair`/`from_i8x16_pair` below for splitting
//! into two independently-shuffleable `sse2::i8x16` halves if a true
//! cross-half shuffle is ever needed.
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

    /// Sign-extend the low 16 lanes (indices 0-15) to `i16x16`.
    /// AVX2 native `_mm256_cvtepi8_epi16` — a dedicated widen
    /// instruction, not a shuffle, so no cross-lane hazard.
    #[inline(always)]
    pub fn as_i16x16_lo(self) -> super::i16x16::i16x16 {
        unsafe {
            let lo_128 = _mm256_castsi256_si128(self.0);
            super::i16x16::i16x16(_mm256_cvtepi8_epi16(lo_128))
        }
    }
    /// Sign-extend the high 16 lanes (indices 16-31) to `i16x16`.
    #[inline(always)]
    pub fn as_i16x16_hi(self) -> super::i16x16::i16x16 {
        unsafe {
            let hi_128 = _mm256_extracti128_si256::<1>(self.0);
            super::i16x16::i16x16(_mm256_cvtepi8_epi16(hi_128))
        }
    }

    /// Split into two `sse2::i8x16` halves (low 16 bytes, high 16 bytes).
    /// Use this to run `sse2::i8x16::shuffle_bytes` independently on each
    /// half if you need indices that logically reach across the 32-byte
    /// boundary — a single `_mm256_shuffle_epi8` call can't do that (see
    /// this file's `shuffle_bytes` and the module docs).
    #[inline(always)]
    pub fn to_i8x16_pair(self) -> (super::super::sse2::i8x16::i8x16, super::super::sse2::i8x16::i8x16) {
        unsafe {
            let lo = _mm256_castsi256_si128(self.0);
            let hi = _mm256_extracti128_si256::<1>(self.0);
            (
                super::super::sse2::i8x16::i8x16(lo),
                super::super::sse2::i8x16::i8x16(hi),
            )
        }
    }
    /// Combine two `sse2::i8x16` halves back into one `i8x32` (inverse of `to_i8x16_pair`).
    #[inline(always)]
    pub fn from_i8x16_pair(lo: super::super::sse2::i8x16::i8x16, hi: super::super::sse2::i8x16::i8x16) -> Self {
        unsafe {
            let joined = _mm256_set_m128i(hi.0, lo.0);
            Self(joined)
        }
    }

    /// Byte shuffle **within each 16-byte half independently** —
    /// `_mm256_shuffle_epi8` is a per-128-bit-lane instruction, NOT a
    /// full 32-byte cross-lane shuffle. An index byte `i` in `[0,15]`
    /// selects byte `i` of the LOW half for output bytes 0-15, and
    /// selects byte `i` of the HIGH half for output bytes 16-31 (i.e.
    /// each half only ever draws from its own 16 source bytes — an
    /// index of, say, 20 in the low half's output position does NOT
    /// reach into the high half). Index bytes with the high bit set
    /// zero that output byte, same as SSE2's `_mm_shuffle_epi8`. If you
    /// need a true cross-half shuffle, split via `to_i8x16_pair` first.
    #[inline(always)]
    pub fn shuffle_bytes(self, indices: Self) -> Self {
        Self(unsafe { _mm256_shuffle_epi8(self.0, indices.0) })
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
