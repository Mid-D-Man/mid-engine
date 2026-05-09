// crates/mid-math/src/wide/int/sse2/u16x8.rs
//! 8-lane unsigned 16-bit integer vector — SSE2, x86 / x86_64.
//!
//! Engine uses: texture UV coordinates quantized to [0, 65535], quantized normals,
//! 16-bit depth buffer values, texture dimensions, unsigned grid indices.
//!
//! Unsigned vs i16x8 differences:
//!   - No native SSE2 unsigned 16-bit min/max → sign-bit XOR trick (_mm_cmpgt_epi16)
//!   - _mm_adds_epu16 / _mm_subs_epu16 for unsigned saturating arithmetic
//!   - _mm_mulhi_epu16 for unsigned upper-half multiply
//!   - Zero-extension for widening (vs sign-extension for i16x8)
//!   - pack_u32x4 uses SSE4.1 _mm_packus_epi32 with scalar fallback

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

use super::imask8::IMask8;
use super::u32x4::u32x4;

#[repr(C)]
union UnionCast { u: [u16; 8], v: u16x8 }

// ── Unsigned 16-bit comparison helper ────────────────────────────────────────
//
// SSE2 has no unsigned 16-bit comparison.
// unsigned a > b  ≡  signed (a ^ 0x8000) > signed (b ^ 0x8000)
// Same XOR trick as u32x4, but 0x8000 per 16-bit lane.

#[inline(always)]
unsafe fn ucmpgt_u16(a: __m128i, b: __m128i) -> __m128i {
    let sign = _mm_set1_epi16(i16::MIN); // 0x8000 per lane
    _mm_cmpgt_epi16(_mm_xor_si128(a, sign), _mm_xor_si128(b, sign))
}

/// 8-lane unsigned 16-bit integer vector. 16 bytes, 16-byte aligned. Backed by `__m128i`.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct u16x8(pub(crate) __m128i);

impl u16x8 {
    pub const ZERO: Self = unsafe { UnionCast { u: [0; 8] }.v };
    pub const ONE:  Self = unsafe { UnionCast { u: [1; 8] }.v };
    pub const MIN:  Self = unsafe { UnionCast { u: [u16::MIN; 8] }.v };
    pub const MAX:  Self = unsafe { UnionCast { u: [u16::MAX; 8] }.v };

    // ── Constructors ──────────────────────────────────────────────────────────

    #[inline(always)]
    pub fn splat(v: u16) -> Self {
        Self(unsafe { _mm_set1_epi16(v as i16) })
    }

    /// Create from 8 values. `a` = lane 0, `h` = lane 7.
    #[inline(always)]
    pub fn new(a: u16, b: u16, c: u16, d: u16, e: u16, f: u16, g: u16, h: u16) -> Self {
        Self(unsafe {
            _mm_set_epi16(h as i16, g as i16, f as i16, e as i16,
                          d as i16, c as i16, b as i16, a as i16)
        })
    }

    #[inline(always)]
    pub fn from_array(a: [u16; 8]) -> Self {
        Self(unsafe { _mm_loadu_si128(a.as_ptr() as *const __m128i) })
    }

    #[inline(always)]
    pub fn to_array(self) -> [u16; 8] {
        unsafe {
            let mut a = [0u16; 8];
            _mm_storeu_si128(a.as_mut_ptr() as *mut __m128i, self.0);
            a
        }
    }

    #[inline]
    pub fn get(self, i: usize) -> u16 {
        assert!(i < 8, "u16x8::get — lane {i} out of bounds (max 7)");
        unsafe { UnionCast { v: self }.u[i] }
    }

    // ── Arithmetic ────────────────────────────────────────────────────────────

    /// Per-lane minimum (unsigned). No native SSE2 — uses sign-bit XOR comparison + blend.
    #[inline]
    pub fn min(self, rhs: Self) -> Self {
        unsafe {
            let gt = ucmpgt_u16(self.0, rhs.0); // 0xFFFF where self > rhs
            // where self > rhs: take rhs (the smaller)
            Self(_mm_or_si128(
                _mm_andnot_si128(gt, self.0),
                _mm_and_si128(gt, rhs.0),
            ))
        }
    }

    /// Per-lane maximum (unsigned). No native SSE2 — uses sign-bit XOR comparison + blend.
    #[inline]
    pub fn max(self, rhs: Self) -> Self {
        unsafe {
            let gt = ucmpgt_u16(self.0, rhs.0); // 0xFFFF where self > rhs
            // where self > rhs: take self (the larger)
            Self(_mm_or_si128(
                _mm_and_si128(gt, self.0),
                _mm_andnot_si128(gt, rhs.0),
            ))
        }
    }

    #[inline(always)]
    pub fn clamp(self, lo: Self, hi: Self) -> Self { self.max(lo).min(hi) }

    #[inline]
    pub fn min_element(self) -> u16 {
        self.to_array().iter().copied().reduce(u16::min).unwrap()
    }

    #[inline]
    pub fn max_element(self) -> u16 {
        self.to_array().iter().copied().reduce(u16::max).unwrap()
    }

    #[inline]
    pub fn element_sum(self) -> u32 {
        self.to_array().iter().map(|&x| x as u32).sum()
    }

    /// Lower 16 bits of each unsigned 16×16 lane product (wrapping).
    /// Same `_mm_mullo_epi16` as i16x8 — bit pattern is identical; interpretation differs.
    #[inline(always)]
    pub fn mul_lo(self, rhs: Self) -> Self {
        Self(unsafe { _mm_mullo_epi16(self.0, rhs.0) })
    }

    /// Upper 16 bits of each unsigned 16×16→32 lane product.
    #[inline(always)]
    pub fn mul_high_u(self, rhs: Self) -> Self {
        Self(unsafe { _mm_mulhi_epu16(self.0, rhs.0) })
    }

    // ── Saturating arithmetic (unsigned) ─────────────────────────────────────

    /// Saturating unsigned add — clamps to `u16::MAX`. SSE2 native.
    #[inline(always)]
    pub fn saturating_add(self, rhs: Self) -> Self {
        Self(unsafe { _mm_adds_epu16(self.0, rhs.0) })
    }

    /// Saturating unsigned sub — clamps to `0`. SSE2 native.
    #[inline(always)]
    pub fn saturating_sub(self, rhs: Self) -> Self {
        Self(unsafe { _mm_subs_epu16(self.0, rhs.0) })
    }

    // ── Shifts ────────────────────────────────────────────────────────────────

    #[inline(always)]
    pub fn shl(self, count: u32) -> Self {
        Self(unsafe { _mm_slli_epi16(self.0, count as i32) })
    }

    /// Logical right shift (zero-fill) — always unsigned.
    #[inline(always)]
    pub fn shr(self, count: u32) -> Self {
        Self(unsafe { _mm_srli_epi16(self.0, count as i32) })
    }

    // ── Comparisons → IMask8 ─────────────────────────────────────────────────

    #[inline(always)]
    pub fn cmpeq(self, rhs: Self) -> IMask8 {
        IMask8(unsafe { _mm_cmpeq_epi16(self.0, rhs.0) })
    }

    #[inline(always)]
    pub fn cmpne(self, rhs: Self) -> IMask8 { !self.cmpeq(rhs) }

    /// Unsigned greater-than.
    #[inline(always)]
    pub fn cmpgt(self, rhs: Self) -> IMask8 {
        IMask8(unsafe { ucmpgt_u16(self.0, rhs.0) })
    }

    /// Unsigned less-than.
    #[inline(always)]
    pub fn cmplt(self, rhs: Self) -> IMask8 { rhs.cmpgt(self) }

    #[inline(always)]
    pub fn cmpge(self, rhs: Self) -> IMask8 { !self.cmplt(rhs) }

    #[inline(always)]
    pub fn cmple(self, rhs: Self) -> IMask8 { !self.cmpgt(rhs) }

    // ── Branchless select ─────────────────────────────────────────────────────

    #[inline(always)]
    pub fn blend(mask: IMask8, if_true: Self, if_false: Self) -> Self {
        unsafe {
            Self(_mm_or_si128(
                _mm_and_si128(mask.0, if_true.0),
                _mm_andnot_si128(mask.0, if_false.0),
            ))
        }
    }

    // ── Widening (zero-extend 16-bit lanes to 32-bit) ─────────────────────────

    /// Zero-extend lower 4 lanes (0–3) to `u32x4`.
    #[inline(always)]
    pub fn as_u32x4_lo(self) -> u32x4 {
        unsafe {
            let zero = _mm_setzero_si128();
            u32x4(_mm_unpacklo_epi16(self.0, zero))
        }
    }

    /// Zero-extend upper 4 lanes (4–7) to `u32x4`.
    #[inline(always)]
    pub fn as_u32x4_hi(self) -> u32x4 {
        unsafe {
            let zero = _mm_setzero_si128();
            u32x4(_mm_unpackhi_epi16(self.0, zero))
        }
    }

    // ── Narrowing ────────────────────────────────────────────────────────────

    /// Pack two `u32x4` to `u16x8` with unsigned saturation `[0, 65535]`.
    /// Uses `_mm_packus_epi32` (SSE4.1) when available; scalar fallback otherwise.
    #[inline]
    pub fn pack_u32x4(lo: u32x4, hi: u32x4) -> Self {
        #[cfg(target_feature = "sse4.1")]
        unsafe {
            Self(_mm_packus_epi32(lo.0, hi.0))
        }
        #[cfg(not(target_feature = "sse4.1"))]
        {
            let a = lo.to_array();
            let b = hi.to_array();
            let sat = |x: u32| x.min(65535) as u16;
            Self::from_array([
                sat(a[0]), sat(a[1]), sat(a[2]), sat(a[3]),
                sat(b[0]), sat(b[1]), sat(b[2]), sat(b[3]),
            ])
        }
    }

    // ── Wrapping aliases ─────────────────────────────────────────────────────

    #[inline(always)] pub fn wrapping_add(self, r: Self) -> Self { self + r }
    #[inline(always)] pub fn wrapping_sub(self, r: Self) -> Self { self - r }
    #[inline(always)] pub fn wrapping_mul(self, r: Self) -> Self { self.mul_lo(r) }
}

// ── Operators ─────────────────────────────────────────────────────────────────

impl Add for u16x8 {
    type Output = Self;
    #[inline(always)]
    fn add(self, r: Self) -> Self { Self(unsafe { _mm_add_epi16(self.0, r.0) }) }
}
impl AddAssign for u16x8 { #[inline(always)] fn add_assign(&mut self, r: Self) { *self = *self + r; } }

impl Sub for u16x8 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, r: Self) -> Self { Self(unsafe { _mm_sub_epi16(self.0, r.0) }) }
}
impl SubAssign for u16x8 { #[inline(always)] fn sub_assign(&mut self, r: Self) { *self = *self - r; } }

impl Mul for u16x8 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, r: Self) -> Self { self.mul_lo(r) }
}
impl MulAssign for u16x8 { #[inline(always)] fn mul_assign(&mut self, r: Self) { *self = *self * r; } }

impl BitAnd for u16x8 {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, r: Self) -> Self { Self(unsafe { _mm_and_si128(self.0, r.0) }) }
}
impl BitAndAssign for u16x8 { #[inline(always)] fn bitand_assign(&mut self, r: Self) { *self = *self & r; } }

impl BitOr for u16x8 {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, r: Self) -> Self { Self(unsafe { _mm_or_si128(self.0, r.0) }) }
}
impl BitOrAssign for u16x8 { #[inline(always)] fn bitor_assign(&mut self, r: Self) { *self = *self | r; } }

impl BitXor for u16x8 {
    type Output = Self;
    #[inline(always)]
    fn bitxor(self, r: Self) -> Self { Self(unsafe { _mm_xor_si128(self.0, r.0) }) }
}
impl BitXorAssign for u16x8 { #[inline(always)] fn bitxor_assign(&mut self, r: Self) { *self = *self ^ r; } }

impl Not for u16x8 {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        unsafe {
            let ones = _mm_cmpeq_epi16(self.0, self.0);
            Self(_mm_xor_si128(self.0, ones))
        }
    }
}

impl PartialEq for u16x8 {
    #[inline]
    fn eq(&self, r: &Self) -> bool {
        unsafe { _mm_movemask_epi8(_mm_cmpeq_epi16(self.0, r.0)) == 0xFFFF }
    }
}
impl Eq for u16x8 {}

impl fmt::Debug for u16x8 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let a = self.to_array();
        write!(f, "u16x8({},{},{},{},{},{},{},{})",
            a[0],a[1],a[2],a[3],a[4],a[5],a[6],a[7])
    }
}
impl fmt::Display for u16x8 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let a = self.to_array();
        write!(f, "[{},{},{},{},{},{},{},{}]",
            a[0],a[1],a[2],a[3],a[4],a[5],a[6],a[7])
    }
}
impl From<[u16; 8]> for u16x8 { #[inline] fn from(a: [u16;8]) -> Self { Self::from_array(a) } }
impl From<u16x8> for [u16; 8] { #[inline] fn from(v: u16x8) -> Self { v.to_array() } }
