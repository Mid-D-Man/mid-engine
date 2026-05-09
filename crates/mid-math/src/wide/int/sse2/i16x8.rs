// crates/mid-math/src/wide/int/sse2/i16x8.rs
//! 8-lane signed 16-bit integer vector — SSE2, x86 / x86_64.
//!
//! Engine uses: quantized animation quaternion components (8 values per register),
//! 16-bit audio processing, packed 2D grid coordinate pairs.
//!
//! SSE2 native advantages over i32x4:
//!   _mm_min_epi16 / _mm_max_epi16  — signed 16-bit min/max is SSE2 (32-bit needs SSE4.1)
//!   _mm_mullo_epi16 / _mm_mulhi_epi16 — both 16-bit mul variants are SSE2
//!   _mm_adds_epi16 / _mm_subs_epi16   — saturating arithmetic is SSE2
//!   Double lanes vs i32x4 = 2× throughput for same register width

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

use super::imask8::IMask8;
use super::i32x4::i32x4;

#[repr(C)]
union UnionCast { i: [i16; 8], v: i16x8 }

/// 8-lane signed 16-bit integer vector. 16 bytes, 16-byte aligned. Backed by `__m128i`.
///
/// All operations are branchless. Comparison ops return [`IMask8`] for use with [`i16x8::blend`].
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct i16x8(pub(crate) __m128i);

impl i16x8 {
    pub const ZERO: Self = unsafe { UnionCast { i: [0; 8] }.v };
    pub const ONE:  Self = unsafe { UnionCast { i: [1; 8] }.v };
    pub const MIN:  Self = unsafe { UnionCast { i: [i16::MIN; 8] }.v };
    pub const MAX:  Self = unsafe { UnionCast { i: [i16::MAX; 8] }.v };

    // ── Constructors ──────────────────────────────────────────────────────────

    /// Broadcast `v` to all 8 lanes.
    #[inline(always)]
    pub fn splat(v: i16) -> Self {
        Self(unsafe { _mm_set1_epi16(v) })
    }

    /// Create from 8 values. `a` = lane 0, `h` = lane 7.
    /// Note: `_mm_set_epi16` takes args highest-lane first, so args are reversed internally.
    #[inline(always)]
    pub fn new(a: i16, b: i16, c: i16, d: i16, e: i16, f: i16, g: i16, h: i16) -> Self {
        Self(unsafe { _mm_set_epi16(h, g, f, e, d, c, b, a) })
    }

    #[inline(always)]
    pub fn from_array(a: [i16; 8]) -> Self {
        Self(unsafe { _mm_loadu_si128(a.as_ptr() as *const __m128i) })
    }

    #[inline(always)]
    pub fn to_array(self) -> [i16; 8] {
        unsafe {
            let mut a = [0i16; 8];
            _mm_storeu_si128(a.as_mut_ptr() as *mut __m128i, self.0);
            a
        }
    }

    /// Extract a single lane. Panics if `i >= 8`.
    #[inline]
    pub fn get(self, i: usize) -> i16 {
        assert!(i < 8, "i16x8::get — lane {i} out of bounds (max 7)");
        unsafe { UnionCast { v: self }.i[i] }
    }

    // ── Arithmetic ────────────────────────────────────────────────────────────

    /// Wrapping absolute value per lane. `abs(i16::MIN) == i16::MIN`.
    #[inline]
    pub fn abs(self) -> Self {
        unsafe {
            let zero = _mm_setzero_si128();
            let neg  = _mm_sub_epi16(zero, self.0);        // 0 - self = -self (wrapping)
            let mask = _mm_cmplt_epi16(self.0, zero);      // 0xFFFF where self < 0
            Self(_mm_or_si128(
                _mm_and_si128(mask, neg),
                _mm_andnot_si128(mask, self.0),
            ))
        }
    }

    /// Per-lane minimum — `_mm_min_epi16` is SSE2 native for signed 16-bit.
    #[inline(always)]
    pub fn min(self, rhs: Self) -> Self {
        Self(unsafe { _mm_min_epi16(self.0, rhs.0) })
    }

    /// Per-lane maximum — `_mm_max_epi16` is SSE2 native for signed 16-bit.
    #[inline(always)]
    pub fn max(self, rhs: Self) -> Self {
        Self(unsafe { _mm_max_epi16(self.0, rhs.0) })
    }

    #[inline(always)]
    pub fn clamp(self, lo: Self, hi: Self) -> Self { self.max(lo).min(hi) }

    #[inline]
    pub fn min_element(self) -> i16 {
        self.to_array().iter().copied().reduce(i16::min).unwrap()
    }

    #[inline]
    pub fn max_element(self) -> i16 {
        self.to_array().iter().copied().reduce(i16::max).unwrap()
    }

    /// Horizontal sum. Result is i32 to avoid overflow on large inputs.
    #[inline]
    pub fn element_sum(self) -> i32 {
        self.to_array().iter().map(|&x| x as i32).sum()
    }

    /// Lower 16 bits of each signed 16×16 lane product (wrapping).
    #[inline(always)]
    pub fn mul_lo(self, rhs: Self) -> Self {
        Self(unsafe { _mm_mullo_epi16(self.0, rhs.0) })
    }

    /// Upper 16 bits of each signed 16×16→32 lane product.
    ///
    /// Useful for fixed-point arithmetic: `(a * b) >> 16` gives the high half.
    #[inline(always)]
    pub fn mul_high(self, rhs: Self) -> Self {
        Self(unsafe { _mm_mulhi_epi16(self.0, rhs.0) })
    }

    // ── Saturating arithmetic ─────────────────────────────────────────────────

    /// Saturating add — clamps to `[i16::MIN, i16::MAX]`. SSE2 native.
    #[inline(always)]
    pub fn saturating_add(self, rhs: Self) -> Self {
        Self(unsafe { _mm_adds_epi16(self.0, rhs.0) })
    }

    /// Saturating sub — clamps to `[i16::MIN, i16::MAX]`. SSE2 native.
    #[inline(always)]
    pub fn saturating_sub(self, rhs: Self) -> Self {
        Self(unsafe { _mm_subs_epi16(self.0, rhs.0) })
    }

    // ── Shifts ────────────────────────────────────────────────────────────────

    /// Logical left shift all lanes by `count` bits.
    #[inline(always)]
    pub fn shl(self, count: u32) -> Self {
        Self(unsafe { _mm_slli_epi16(self.0, count as i32) })
    }

    /// Arithmetic right shift (sign-fill) all lanes by `count` bits.
    #[inline(always)]
    pub fn shr_arithmetic(self, count: u32) -> Self {
        Self(unsafe { _mm_srai_epi16(self.0, count as i32) })
    }

    /// Logical right shift (zero-fill) all lanes by `count` bits.
    #[inline(always)]
    pub fn shr_logical(self, count: u32) -> Self {
        Self(unsafe { _mm_srli_epi16(self.0, count as i32) })
    }

    // ── Comparisons → IMask8 ─────────────────────────────────────────────────

    #[inline(always)]
    pub fn cmpeq(self, rhs: Self) -> IMask8 {
        IMask8(unsafe { _mm_cmpeq_epi16(self.0, rhs.0) })
    }

    #[inline(always)]
    pub fn cmpne(self, rhs: Self) -> IMask8 { !self.cmpeq(rhs) }

    #[inline(always)]
    pub fn cmpgt(self, rhs: Self) -> IMask8 {
        IMask8(unsafe { _mm_cmpgt_epi16(self.0, rhs.0) })
    }

    #[inline(always)]
    pub fn cmplt(self, rhs: Self) -> IMask8 {
        IMask8(unsafe { _mm_cmplt_epi16(self.0, rhs.0) })
    }

    #[inline(always)]
    pub fn cmpge(self, rhs: Self) -> IMask8 { !self.cmplt(rhs) }

    #[inline(always)]
    pub fn cmple(self, rhs: Self) -> IMask8 { !self.cmpgt(rhs) }

    // ── Branchless select ─────────────────────────────────────────────────────

    /// Select per lane: `if_true[i]` where `mask[i]` is true, else `if_false[i]`.
    #[inline(always)]
    pub fn blend(mask: IMask8, if_true: Self, if_false: Self) -> Self {
        unsafe {
            Self(_mm_or_si128(
                _mm_and_si128(mask.0, if_true.0),
                _mm_andnot_si128(mask.0, if_false.0),
            ))
        }
    }

    // ── Widening (sign-extend 16-bit lanes to 32-bit) ─────────────────────────
    //
    // SSE2 sign-extension technique:
    //   1. srai_epi16 by 15 fills each lane with its sign bit (0x0000 or 0xFFFF).
    //   2. unpacklo/hi_epi16 interleaves data lanes with their sign lanes.
    //   3. The resulting __m128i contains 4 proper sign-extended i32 values.

    /// Sign-extend lower 4 lanes (0–3) to `i32x4`.
    #[inline(always)]
    pub fn as_i32x4_lo(self) -> i32x4 {
        unsafe {
            let sign = _mm_srai_epi16(self.0, 15); // 0x0000 or 0xFFFF per lane
            i32x4(_mm_unpacklo_epi16(self.0, sign))
        }
    }

    /// Sign-extend upper 4 lanes (4–7) to `i32x4`.
    #[inline(always)]
    pub fn as_i32x4_hi(self) -> i32x4 {
        unsafe {
            let sign = _mm_srai_epi16(self.0, 15);
            i32x4(_mm_unpackhi_epi16(self.0, sign))
        }
    }

    // ── Narrowing ────────────────────────────────────────────────────────────

    /// Pack two `i32x4` to `i16x8` with signed saturation `[-32768, 32767]`.
    /// `lo` fills lanes 0–3, `hi` fills lanes 4–7.
    #[inline(always)]
    pub fn pack_i32x4(lo: i32x4, hi: i32x4) -> Self {
        Self(unsafe { _mm_packs_epi32(lo.0, hi.0) })
    }

    // ── Wrapping aliases ─────────────────────────────────────────────────────

    #[inline(always)] pub fn wrapping_add(self, r: Self) -> Self { self + r }
    #[inline(always)] pub fn wrapping_sub(self, r: Self) -> Self { self - r }
    #[inline(always)] pub fn wrapping_mul(self, r: Self) -> Self { self.mul_lo(r) }
}

// ── Operators ─────────────────────────────────────────────────────────────────

impl Add for i16x8 {
    type Output = Self;
    #[inline(always)]
    fn add(self, r: Self) -> Self { Self(unsafe { _mm_add_epi16(self.0, r.0) }) }
}
impl AddAssign for i16x8 { #[inline(always)] fn add_assign(&mut self, r: Self) { *self = *self + r; } }

impl Sub for i16x8 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, r: Self) -> Self { Self(unsafe { _mm_sub_epi16(self.0, r.0) }) }
}
impl SubAssign for i16x8 { #[inline(always)] fn sub_assign(&mut self, r: Self) { *self = *self - r; } }

impl Neg for i16x8 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self { Self(unsafe { _mm_sub_epi16(_mm_setzero_si128(), self.0) }) }
}

impl Mul for i16x8 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, r: Self) -> Self { self.mul_lo(r) }
}
impl MulAssign for i16x8 { #[inline(always)] fn mul_assign(&mut self, r: Self) { *self = *self * r; } }

impl BitAnd for i16x8 {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, r: Self) -> Self { Self(unsafe { _mm_and_si128(self.0, r.0) }) }
}
impl BitAndAssign for i16x8 { #[inline(always)] fn bitand_assign(&mut self, r: Self) { *self = *self & r; } }

impl BitOr for i16x8 {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, r: Self) -> Self { Self(unsafe { _mm_or_si128(self.0, r.0) }) }
}
impl BitOrAssign for i16x8 { #[inline(always)] fn bitor_assign(&mut self, r: Self) { *self = *self | r; } }

impl BitXor for i16x8 {
    type Output = Self;
    #[inline(always)]
    fn bitxor(self, r: Self) -> Self { Self(unsafe { _mm_xor_si128(self.0, r.0) }) }
}
impl BitXorAssign for i16x8 { #[inline(always)] fn bitxor_assign(&mut self, r: Self) { *self = *self ^ r; } }

impl Not for i16x8 {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        unsafe {
            let ones = _mm_cmpeq_epi16(self.0, self.0);
            Self(_mm_xor_si128(self.0, ones))
        }
    }
}

impl PartialEq for i16x8 {
    #[inline]
    fn eq(&self, r: &Self) -> bool {
        unsafe { _mm_movemask_epi8(_mm_cmpeq_epi16(self.0, r.0)) == 0xFFFF }
    }
}
impl Eq for i16x8 {}

impl fmt::Debug for i16x8 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let a = self.to_array();
        write!(f, "i16x8({},{},{},{},{},{},{},{})",
            a[0],a[1],a[2],a[3],a[4],a[5],a[6],a[7])
    }
}
impl fmt::Display for i16x8 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let a = self.to_array();
        write!(f, "[{},{},{},{},{},{},{},{}]",
            a[0],a[1],a[2],a[3],a[4],a[5],a[6],a[7])
    }
}
impl From<[i16; 8]> for i16x8 { #[inline] fn from(a: [i16;8]) -> Self { Self::from_array(a) } }
impl From<i16x8> for [i16; 8] { #[inline] fn from(v: i16x8) -> Self { v.to_array() } }
