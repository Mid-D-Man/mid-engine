// crates/mid-math/src/wide/int/avx2/u8x32.rs
//! 32-lane unsigned 8-bit integer vector — AVX2, x86 / x86_64.
//!
//! Widens sse2/u8x16.rs to `__m256i`. `_mm256_min_epu8`/`_mm256_max_epu8`/
//! `_mm256_adds_epu8`/`_mm256_subs_epu8` are all AVX2 native (mirrors
//! SSE2, which already had these natively for unsigned 8-bit — the one
//! width where SSE2 itself needed no emulation). `cmpgt` still needs the
//! XOR-sign-flip trick (no native unsigned compare on x86 at any width).
//!
//! `element_sum` uses `_mm256_sad_epu8` against zero, same as
//! sse2/u8x16.rs — NOT a cross-lane hazard like unpack/shuffle: SAD runs
//! independently per 8-byte group and simply produces four 64-bit
//! partial sums instead of SSE2's two, extracted via `_mm256_extract_epi64`.
//!
//! Deliberately omits `shuffle_bytes` (SSSE3 `_mm_shuffle_epi8`) and
//! `as_u16x8_lo`/`as_u16x8_hi` widening — same per-128-bit-lane
//! correctness hazard as i8x32.rs, see that file's header for detail.
//!
//! No multiply, same as sse2/u8x16.rs — no native 8-bit SIMD multiply.

#![allow(non_camel_case_types)]

use core::fmt;
use core::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign,
    BitXor, BitXorAssign, Not, Sub, SubAssign,
};

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use super::imask8x32::IMask8x32;

#[repr(C)]
union UnionCast { u: [u8; 32], v: u8x32 }

#[inline(always)]
unsafe fn ucmpgt_u8(a: __m256i, b: __m256i) -> __m256i {
    let sign = _mm256_set1_epi8(-128i8); // 0x80 per byte
    _mm256_cmpgt_epi8(_mm256_xor_si256(a, sign), _mm256_xor_si256(b, sign))
}

/// 32-lane unsigned 8-bit integer vector. 32 bytes, 32-byte aligned. Backed by `__m256i`.
///
/// Note: no Mul — unsigned byte multiply requires widening (no cross-lane
/// widen provided here, see module docs).
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct u8x32(pub(crate) __m256i);

impl u8x32 {
    pub const ZERO: Self = unsafe { UnionCast { u: [0; 32] }.v };
    pub const ONE:  Self = unsafe { UnionCast { u: [1; 32] }.v };
    pub const MIN:  Self = unsafe { UnionCast { u: [u8::MIN; 32] }.v };
    pub const MAX:  Self = unsafe { UnionCast { u: [u8::MAX; 32] }.v };

    #[inline(always)]
    pub fn splat(v: u8) -> Self { Self(unsafe { _mm256_set1_epi8(v as i8) }) }

    #[inline(always)]
    pub fn from_array(a: [u8; 32]) -> Self {
        Self(unsafe { _mm256_loadu_si256(a.as_ptr() as *const __m256i) })
    }

    /// Convenience alias — identical to `from_array`.
    #[inline(always)]
    pub fn from_bytes(b: [u8; 32]) -> Self { Self::from_array(b) }

    #[inline(always)]
    pub fn to_array(self) -> [u8; 32] {
        unsafe { let mut a=[0u8;32]; _mm256_storeu_si256(a.as_mut_ptr() as *mut __m256i, self.0); a }
    }

    #[inline]
    pub fn get(self, i: usize) -> u8 {
        assert!(i < 32, "u8x32::get — lane {i} out of bounds (max 31)");
        unsafe { UnionCast { v: self }.u[i] }
    }

    /// Per-lane minimum — AVX2 native `_mm256_min_epu8`.
    #[inline(always)] pub fn min(self, rhs: Self) -> Self { Self(unsafe { _mm256_min_epu8(self.0, rhs.0) }) }
    /// Per-lane maximum — AVX2 native `_mm256_max_epu8`.
    #[inline(always)] pub fn max(self, rhs: Self) -> Self { Self(unsafe { _mm256_max_epu8(self.0, rhs.0) }) }
    #[inline(always)] pub fn clamp(self, lo: Self, hi: Self) -> Self { self.max(lo).min(hi) }

    #[inline] pub fn min_element(self) -> u8 { self.to_array().iter().copied().reduce(u8::min).unwrap() }
    #[inline] pub fn max_element(self) -> u8 { self.to_array().iter().copied().reduce(u8::max).unwrap() }

    /// Horizontal sum via `_mm256_sad_epu8` against zero — four independent
    /// 8-byte-group partial sums (one per `_mm256_extract_epi64` lane),
    /// added together. See module docs re: why this isn't a cross-lane hazard.
    #[inline]
    pub fn element_sum(self) -> u32 {
        unsafe {
            let zero = _mm256_setzero_si256();
            let sad  = _mm256_sad_epu8(self.0, zero);
            let s0 = _mm256_extract_epi64::<0>(sad) as u32;
            let s1 = _mm256_extract_epi64::<1>(sad) as u32;
            let s2 = _mm256_extract_epi64::<2>(sad) as u32;
            let s3 = _mm256_extract_epi64::<3>(sad) as u32;
            s0 + s1 + s2 + s3
        }
    }

    /// Saturating unsigned add — clamps to `u8::MAX`. AVX2 native.
    #[inline(always)] pub fn saturating_add(self, rhs: Self) -> Self { Self(unsafe { _mm256_adds_epu8(self.0, rhs.0) }) }
    /// Saturating unsigned sub — clamps to `0`. AVX2 native.
    #[inline(always)] pub fn saturating_sub(self, rhs: Self) -> Self { Self(unsafe { _mm256_subs_epu8(self.0, rhs.0) }) }

    /// Equality — AVX2 native (same bit pattern for signed/unsigned).
    #[inline(always)] pub fn cmpeq(self, rhs: Self) -> IMask8x32 { IMask8x32(unsafe { _mm256_cmpeq_epi8(self.0, rhs.0) }) }
    #[inline(always)] pub fn cmpne(self, rhs: Self) -> IMask8x32 { !self.cmpeq(rhs) }
    /// Unsigned greater-than. Uses sign-bit XOR trick.
    #[inline(always)] pub fn cmpgt(self, rhs: Self) -> IMask8x32 { IMask8x32(unsafe { ucmpgt_u8(self.0, rhs.0) }) }
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

    /// Number of lanes equal to `needle`.
    #[inline]
    pub fn count_eq(self, needle: Self) -> u32 { self.cmpeq(needle).count_true() }
    /// True if any lane equals `needle`.
    #[inline]
    pub fn contains(self, needle: u8) -> bool { self.count_eq(Self::splat(needle)) > 0 }

    #[inline(always)] pub fn wrapping_add(self, r: Self) -> Self { self + r }
    #[inline(always)] pub fn wrapping_sub(self, r: Self) -> Self { self - r }
}

impl Add for u8x32 { type Output=Self; #[inline(always)] fn add(self,r:Self)->Self{Self(unsafe{_mm256_add_epi8(self.0,r.0)})} }
impl AddAssign for u8x32 { #[inline(always)] fn add_assign(&mut self,r:Self){*self=*self+r;} }
impl Sub for u8x32 { type Output=Self; #[inline(always)] fn sub(self,r:Self)->Self{Self(unsafe{_mm256_sub_epi8(self.0,r.0)})} }
impl SubAssign for u8x32 { #[inline(always)] fn sub_assign(&mut self,r:Self){*self=*self-r;} }

impl BitAnd for u8x32 { type Output=Self; #[inline(always)] fn bitand(self,r:Self)->Self{Self(unsafe{_mm256_and_si256(self.0,r.0)})} }
impl BitAndAssign for u8x32 { #[inline(always)] fn bitand_assign(&mut self,r:Self){*self=*self&r;} }
impl BitOr  for u8x32 { type Output=Self; #[inline(always)] fn bitor (self,r:Self)->Self{Self(unsafe{_mm256_or_si256(self.0,r.0)})} }
impl BitOrAssign  for u8x32 { #[inline(always)] fn bitor_assign (&mut self,r:Self){*self=*self|r;} }
impl BitXor for u8x32 { type Output=Self; #[inline(always)] fn bitxor(self,r:Self)->Self{Self(unsafe{_mm256_xor_si256(self.0,r.0)})} }
impl BitXorAssign for u8x32 { #[inline(always)] fn bitxor_assign(&mut self,r:Self){*self=*self^r;} }
impl Not for u8x32 {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self { unsafe { let ones = _mm256_cmpeq_epi8(self.0, self.0); Self(_mm256_xor_si256(self.0, ones)) } }
}

impl PartialEq for u8x32 {
    #[inline]
    fn eq(&self, r: &Self) -> bool { unsafe { _mm256_movemask_epi8(_mm256_cmpeq_epi8(self.0, r.0)) == -1 } }
}
impl Eq for u8x32 {}

impl fmt::Debug for u8x32 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "u8x32({:?})", self.to_array()) }
}
impl fmt::Display for u8x32 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "{:?}", self.to_array()) }
}
impl From<[u8; 32]> for u8x32 { #[inline] fn from(a: [u8;32]) -> Self { Self::from_array(a) } }
impl From<u8x32> for [u8; 32] { #[inline] fn from(v: u8x32) -> Self { v.to_array() } }
