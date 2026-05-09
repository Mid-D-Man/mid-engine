// crates/mid-math/src/wide/int/sse2/u8x16.rs
//! 16-lane unsigned 8-bit integer vector — SSE2, x86 / x86_64.
//!
//! Engine uses: RGBA pixel batch processing (4 pixels × RGBA per register),
//! raw byte stream parsing, UTF-8 byte validation, texture data staging,
//! DixScript packet payload inspection.
//!
//! Primary type for image/pixel operations and any byte-stream work.
//!
//! Unsigned vs i8x16 differences:
//!   - _mm_min_epu8 / _mm_max_epu8 are SSE2 native for unsigned 8-bit (unlike 16/32 bit)
//!   - _mm_adds_epu8 / _mm_subs_epu8 for unsigned saturating arithmetic
//!   - No signed cmpgt — unsigned comparison via XOR sign-bit flip
//!   - Zero-extension for widening (vs sign-extension)

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

use super::imask16::IMask16;

#[repr(C)]
union UnionCast { u: [u8; 16], v: u8x16 }

// ── Unsigned 8-bit comparison helper ─────────────────────────────────────────
//
// SSE2 has no unsigned 8-bit cmpgt. XOR flip:
//   unsigned a > b  ≡  signed (a ^ 0x80) > signed (b ^ 0x80)

#[inline(always)]
unsafe fn ucmpgt_u8(a: __m128i, b: __m128i) -> __m128i {
    let sign = _mm_set1_epi8(-128i8); // 0x80 per byte
    _mm_cmpgt_epi8(_mm_xor_si128(a, sign), _mm_xor_si128(b, sign))
}

/// 16-lane unsigned 8-bit integer vector. 16 bytes, 16-byte aligned. Backed by `__m128i`.
///
/// Note: no Mul/Neg — unsigned byte multiply requires widening. Use `as_u16x8_lo/hi`
/// then [`u16x8::mul_lo`] or [`u16x8::mul_high_u`].
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct u8x16(pub(crate) __m128i);

impl u8x16 {
    pub const ZERO: Self = unsafe { UnionCast { u: [0; 16] }.v };
    pub const ONE:  Self = unsafe { UnionCast { u: [1; 16] }.v };
    pub const MIN:  Self = unsafe { UnionCast { u: [u8::MIN; 16] }.v };
    pub const MAX:  Self = unsafe { UnionCast { u: [u8::MAX; 16] }.v };

    // ── Constructors ──────────────────────────────────────────────────────────

    #[inline(always)]
    pub fn splat(v: u8) -> Self {
        Self(unsafe { _mm_set1_epi8(v as i8) })
    }

    #[inline(always)]
    pub fn from_array(a: [u8; 16]) -> Self {
        Self(unsafe { _mm_loadu_si128(a.as_ptr() as *const __m128i) })
    }

    /// Convenience alias — identical to `from_array`.
    #[inline(always)]
    pub fn from_bytes(b: [u8; 16]) -> Self { Self::from_array(b) }

    #[inline(always)]
    pub fn to_array(self) -> [u8; 16] {
        unsafe {
            let mut a = [0u8; 16];
            _mm_storeu_si128(a.as_mut_ptr() as *mut __m128i, self.0);
            a
        }
    }

    #[inline]
    pub fn get(self, i: usize) -> u8 {
        assert!(i < 16, "u8x16::get — lane {i} out of bounds (max 15)");
        unsafe { UnionCast { v: self }.u[i] }
    }

    // ── Arithmetic ────────────────────────────────────────────────────────────

    /// Per-lane minimum — `_mm_min_epu8` is SSE2 native for unsigned 8-bit.
    #[inline(always)]
    pub fn min(self, rhs: Self) -> Self {
        Self(unsafe { _mm_min_epu8(self.0, rhs.0) })
    }

    /// Per-lane maximum — `_mm_max_epu8` is SSE2 native for unsigned 8-bit.
    #[inline(always)]
    pub fn max(self, rhs: Self) -> Self {
        Self(unsafe { _mm_max_epu8(self.0, rhs.0) })
    }

    #[inline(always)]
    pub fn clamp(self, lo: Self, hi: Self) -> Self { self.max(lo).min(hi) }

    #[inline]
    pub fn min_element(self) -> u8 {
        self.to_array().iter().copied().reduce(u8::min).unwrap()
    }

    #[inline]
    pub fn max_element(self) -> u8 {
        self.to_array().iter().copied().reduce(u8::max).unwrap()
    }

    #[inline]
    pub fn element_sum(self) -> u32 {
        // _mm_sad_epu8 against zero gives horizontal sum in two 64-bit halves.
        // Extract both halves and add — much faster than scalar loop.
        unsafe {
            let zero = _mm_setzero_si128();
            let sad  = _mm_sad_epu8(self.0, zero);
            // Lane 0 of sad = sum of bytes 0-7, lane 4 = sum of bytes 8-15 (as u16 each)
            let lo = _mm_cvtsi128_si32(sad) as u32;
            let hi = _mm_cvtsi128_si32(_mm_srli_si128(sad, 8)) as u32;
            lo + hi
        }
    }

    // ── Saturating arithmetic ─────────────────────────────────────────────────

    /// Saturating unsigned add — clamps to `u8::MAX`. SSE2 native.
    #[inline(always)]
    pub fn saturating_add(self, rhs: Self) -> Self {
        Self(unsafe { _mm_adds_epu8(self.0, rhs.0) })
    }

    /// Saturating unsigned sub — clamps to `0`. SSE2 native.
    #[inline(always)]
    pub fn saturating_sub(self, rhs: Self) -> Self {
        Self(unsafe { _mm_subs_epu8(self.0, rhs.0) })
    }

    // ── Comparisons → IMask16 ─────────────────────────────────────────────────

    /// Equality — SSE2 native (same bit pattern for signed/unsigned).
    #[inline(always)]
    pub fn cmpeq(self, rhs: Self) -> IMask16 {
        IMask16(unsafe { _mm_cmpeq_epi8(self.0, rhs.0) })
    }

    #[inline(always)]
    pub fn cmpne(self, rhs: Self) -> IMask16 { !self.cmpeq(rhs) }

    /// Unsigned greater-than. Uses sign-bit XOR trick.
    #[inline(always)]
    pub fn cmpgt(self, rhs: Self) -> IMask16 {
        IMask16(unsafe { ucmpgt_u8(self.0, rhs.0) })
    }

    #[inline(always)]
    pub fn cmplt(self, rhs: Self) -> IMask16 { rhs.cmpgt(self) }

    #[inline(always)]
    pub fn cmpge(self, rhs: Self) -> IMask16 { !self.cmplt(rhs) }

    #[inline(always)]
    pub fn cmple(self, rhs: Self) -> IMask16 { !self.cmpgt(rhs) }

    // ── Branchless select ─────────────────────────────────────────────────────

    #[inline(always)]
    pub fn blend(mask: IMask16, if_true: Self, if_false: Self) -> Self {
        unsafe {
            Self(_mm_or_si128(
                _mm_and_si128(mask.0, if_true.0),
                _mm_andnot_si128(mask.0, if_false.0),
            ))
        }
    }

    // ── Byte shuffle (SSSE3) ─────────────────────────────────────────────────

    /// Byte shuffle using `indices` (SSSE3 `PSHUFB`).
    ///
    /// For each lane `i`:
    /// - If `indices[i] & 0x80 != 0`: output = **0**.
    /// - Else: output = `self[indices[i] & 0x0F]`.
    ///
    /// Use for RGBA channel reordering, broadcast patterns, hash permutations.
    #[cfg(target_feature = "ssse3")]
    #[inline(always)]
    pub fn shuffle_bytes(self, indices: Self) -> Self {
        unsafe {
            #[cfg(target_arch = "x86")]
            use core::arch::x86::_mm_shuffle_epi8;
            #[cfg(target_arch = "x86_64")]
            use core::arch::x86_64::_mm_shuffle_epi8;
            // Reinterpret u8 indices as i8 for the intrinsic (sign bit = zero control)
            Self(_mm_shuffle_epi8(self.0, indices.0))
        }
    }

    #[cfg(not(target_feature = "ssse3"))]
    #[inline]
    pub fn shuffle_bytes(self, indices: Self) -> Self {
        let src = self.to_array();
        let idx = indices.to_array();
        let mut out = [0u8; 16];
        for i in 0..16 {
            out[i] = if idx[i] & 0x80 != 0 { 0 } else { src[(idx[i] & 0x0F) as usize] };
        }
        Self::from_array(out)
    }

    // ── Population / search ───────────────────────────────────────────────────

    /// Number of lanes equal to `needle`.
    #[inline]
    pub fn count_eq(self, needle: Self) -> u32 {
        self.cmpeq(needle).count_true()
    }

    /// True if any lane equals `needle`.
    #[inline]
    pub fn contains(self, needle: u8) -> bool {
        self.count_eq(Self::splat(needle)) > 0
    }

    // ── Widening (zero-extend 8-bit lanes to 16-bit) ──────────────────────────

    /// Zero-extend lower 8 lanes (0–7) to `u16x8`.
    #[inline(always)]
    pub fn as_u16x8_lo(self) -> super::u16x8::u16x8 {
        unsafe {
            let zero = _mm_setzero_si128();
            super::u16x8::u16x8(_mm_unpacklo_epi8(self.0, zero))
        }
    }

    /// Zero-extend upper 8 lanes (8–15) to `u16x8`.
    #[inline(always)]
    pub fn as_u16x8_hi(self) -> super::u16x8::u16x8 {
        unsafe {
            let zero = _mm_setzero_si128();
            super::u16x8::u16x8(_mm_unpackhi_epi8(self.0, zero))
        }
    }

    // ── Narrowing ────────────────────────────────────────────────────────────

    /// Pack two `u16x8` to `u8x16` with unsigned saturation `[0, 255]`.
    /// SSE2 native: `_mm_packus_epi16`.
    #[inline(always)]
    pub fn pack_u16x8(lo: super::u16x8::u16x8, hi: super::u16x8::u16x8) -> Self {
        Self(unsafe { _mm_packus_epi16(lo.0, hi.0) })
    }

    // ── Wrapping aliases ─────────────────────────────────────────────────────

    #[inline(always)] pub fn wrapping_add(self, r: Self) -> Self { self + r }
    #[inline(always)] pub fn wrapping_sub(self, r: Self) -> Self { self - r }
}

// ── Operators ─────────────────────────────────────────────────────────────────

impl Add for u8x16 {
    type Output = Self;
    #[inline(always)]
    fn add(self, r: Self) -> Self { Self(unsafe { _mm_add_epi8(self.0, r.0) }) }
}
impl AddAssign for u8x16 { #[inline(always)] fn add_assign(&mut self, r: Self) { *self = *self + r; } }

impl Sub for u8x16 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, r: Self) -> Self { Self(unsafe { _mm_sub_epi8(self.0, r.0) }) }
}
impl SubAssign for u8x16 { #[inline(always)] fn sub_assign(&mut self, r: Self) { *self = *self - r; } }

impl BitAnd for u8x16 {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, r: Self) -> Self { Self(unsafe { _mm_and_si128(self.0, r.0) }) }
}
impl BitAndAssign for u8x16 { #[inline(always)] fn bitand_assign(&mut self, r: Self) { *self = *self & r; } }

impl BitOr for u8x16 {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, r: Self) -> Self { Self(unsafe { _mm_or_si128(self.0, r.0) }) }
}
impl BitOrAssign for u8x16 { #[inline(always)] fn bitor_assign(&mut self, r: Self) { *self = *self | r; } }

impl BitXor for u8x16 {
    type Output = Self;
    #[inline(always)]
    fn bitxor(self, r: Self) -> Self { Self(unsafe { _mm_xor_si128(self.0, r.0) }) }
}
impl BitXorAssign for u8x16 { #[inline(always)] fn bitxor_assign(&mut self, r: Self) { *self = *self ^ r; } }

impl Not for u8x16 {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        unsafe {
            let ones = _mm_cmpeq_epi8(self.0, self.0);
            Self(_mm_xor_si128(self.0, ones))
        }
    }
}

impl PartialEq for u8x16 {
    #[inline]
    fn eq(&self, r: &Self) -> bool {
        unsafe { _mm_movemask_epi8(_mm_cmpeq_epi8(self.0, r.0)) == 0xFFFF }
    }
}
impl Eq for u8x16 {}

impl fmt::Debug for u8x16 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "u8x16({:?})", self.to_array())
    }
}
impl fmt::Display for u8x16 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.to_array())
    }
}
impl From<[u8; 16]> for u8x16 { #[inline] fn from(a: [u8;16]) -> Self { Self::from_array(a) } }
impl From<u8x16> for [u8; 16] { #[inline] fn from(v: u8x16) -> Self { v.to_array() } }
