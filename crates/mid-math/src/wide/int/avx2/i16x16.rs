// crates/mid-math/src/wide/int/avx2/i16x16.rs
//! 16-lane signed 16-bit integer vector.
//!
//! Same storage/dispatch design as `i32x8.rs` — see that file's doc
//! comment for the full reasoning. Portable `{lo, hi}: i16x8` storage.
//!
//! `as_i32x8_lo`/`as_i32x8_hi` (sign-extend 8 of the 16 lanes to `i32x8`)
//! compose from `i16x8::as_i32x4_lo`/`as_i32x4_hi`, which already exist —
//! `self.lo` (an `i16x8` holding lanes 0-7) widens to a full `i32x8` by
//! widening both of *its own* halves; same for `self.hi` (lanes 8-15).

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

use crate::wide::int::sse2::i16x8::i16x8;
use super::i32x8::i32x8;
use super::imask16x16::IMask16x16;

/// 16-lane signed 16-bit integer vector. Two `i16x8` halves.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct i16x16 {
    lo: i16x8,
    hi: i16x8,
}

impl i16x16 {
    pub const ZERO: Self = Self { lo: i16x8::ZERO, hi: i16x8::ZERO };
    pub const ONE:  Self = Self { lo: i16x8::ONE,  hi: i16x8::ONE };
    pub const MIN:  Self = Self { lo: i16x8::MIN,  hi: i16x8::MIN };
    pub const MAX:  Self = Self { lo: i16x8::MAX,  hi: i16x8::MAX };

    #[inline(always)]
    pub(crate) fn from_halves(lo: i16x8, hi: i16x8) -> Self { Self { lo, hi } }

    #[inline(always)]
    pub fn splat(v: i16) -> Self { Self::from_halves(i16x8::splat(v), i16x8::splat(v)) }

    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    pub fn new(a: i16, b: i16, c: i16, d: i16, e: i16, f: i16, g: i16, h: i16,
               i_: i16, j: i16, k: i16, l: i16, m: i16, n: i16, o: i16, p: i16) -> Self {
        Self::from_halves(i16x8::new(a, b, c, d, e, f, g, h), i16x8::new(i_, j, k, l, m, n, o, p))
    }

    #[inline(always)]
    pub fn from_array(a: [i16; 16]) -> Self {
        let lo: [i16; 8] = [a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]];
        let hi: [i16; 8] = [a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15]];
        Self::from_halves(i16x8::from_array(lo), i16x8::from_array(hi))
    }

    #[inline(always)]
    pub fn to_array(self) -> [i16; 16] {
        let lo = self.lo.to_array();
        let hi = self.hi.to_array();
        [lo[0], lo[1], lo[2], lo[3], lo[4], lo[5], lo[6], lo[7],
         hi[0], hi[1], hi[2], hi[3], hi[4], hi[5], hi[6], hi[7]]
    }

    #[inline]
    pub fn get(self, i: usize) -> i16 {
        assert!(i < 16, "i16x16::get — lane {i} out of bounds (max 15)");
        if i < 8 { self.lo.get(i) } else { self.hi.get(i - 8) }
    }

    // ── AVX2 pack/unpack helpers — only place a __m256i exists ──

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx2")]
    unsafe fn to_m256i(self) -> __m256i {
        unsafe { _mm256_set_m128i(self.hi.0, self.lo.0) }
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx2")]
    unsafe fn from_m256i(v: __m256i) -> Self {
        unsafe {
            Self::from_halves(
                i16x8(_mm256_castsi256_si128(v)),
                i16x8(_mm256_extracti128_si256::<1>(v)),
            )
        }
    }

    #[inline]
    pub fn abs(self) -> Self {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if crate::wide::avx2_available() {
            return unsafe { self.abs_avx2() };
        }
        Self::from_halves(self.lo.abs(), self.hi.abs())
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx2")]
    unsafe fn abs_avx2(self) -> Self {
        unsafe { Self::from_m256i(_mm256_abs_epi16(self.to_m256i())) }
    }

    /// Sign-extend lanes 0-7 to `i32x8`. Portable — composes from
    /// `i16x8::as_i32x4_lo`/`as_i32x4_hi`, which already exist and are
    /// themselves already dispatch-free (SSE2 is always available).
    #[inline(always)]
    pub fn as_i32x8_lo(self) -> i32x8 {
        i32x8::from_halves(self.lo.as_i32x4_lo(), self.lo.as_i32x4_hi())
    }
    /// Sign-extend lanes 8-15 to `i32x8`.
    #[inline(always)]
    pub fn as_i32x8_hi(self) -> i32x8 {
        i32x8::from_halves(self.hi.as_i32x4_lo(), self.hi.as_i32x4_hi())
    }

    #[inline]
    pub fn min(self, rhs: Self) -> Self {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if crate::wide::avx2_available() {
            return unsafe { self.min_avx2(rhs) };
        }
        Self::from_halves(self.lo.min(rhs.lo), self.hi.min(rhs.hi))
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx2")]
    unsafe fn min_avx2(self, rhs: Self) -> Self {
        unsafe { Self::from_m256i(_mm256_min_epi16(self.to_m256i(), rhs.to_m256i())) }
    }

    #[inline]
    pub fn max(self, rhs: Self) -> Self {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if crate::wide::avx2_available() {
            return unsafe { self.max_avx2(rhs) };
        }
        Self::from_halves(self.lo.max(rhs.lo), self.hi.max(rhs.hi))
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx2")]
    unsafe fn max_avx2(self, rhs: Self) -> Self {
        unsafe { Self::from_m256i(_mm256_max_epi16(self.to_m256i(), rhs.to_m256i())) }
    }

    #[inline(always)]
    pub fn clamp(self, lo: Self, hi: Self) -> Self { self.max(lo).min(hi) }

    #[inline] pub fn min_element(self) -> i16 { self.lo.min_element().min(self.hi.min_element()) }
    #[inline] pub fn max_element(self) -> i16 { self.lo.max_element().max(self.hi.max_element()) }
    #[inline] pub fn element_sum(self) -> i32 { self.lo.element_sum() + self.hi.element_sum() }

    #[inline]
    pub fn mul_lo(self, rhs: Self) -> Self {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if crate::wide::avx2_available() {
            return unsafe { self.mul_lo_avx2(rhs) };
        }
        Self::from_halves(self.lo.mul_lo(rhs.lo), self.hi.mul_lo(rhs.hi))
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx2")]
    unsafe fn mul_lo_avx2(self, rhs: Self) -> Self {
        unsafe { Self::from_m256i(_mm256_mullo_epi16(self.to_m256i(), rhs.to_m256i())) }
    }

    #[inline]
    pub fn mul_high(self, rhs: Self) -> Self {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if crate::wide::avx2_available() {
            return unsafe { self.mul_high_avx2(rhs) };
        }
        Self::from_halves(self.lo.mul_high(rhs.lo), self.hi.mul_high(rhs.hi))
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx2")]
    unsafe fn mul_high_avx2(self, rhs: Self) -> Self {
        unsafe { Self::from_m256i(_mm256_mulhi_epi16(self.to_m256i(), rhs.to_m256i())) }
    }

    #[inline]
    pub fn saturating_add(self, rhs: Self) -> Self {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if crate::wide::avx2_available() {
            return unsafe { self.saturating_add_avx2(rhs) };
        }
        Self::from_halves(self.lo.saturating_add(rhs.lo), self.hi.saturating_add(rhs.hi))
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx2")]
    unsafe fn saturating_add_avx2(self, rhs: Self) -> Self {
        unsafe { Self::from_m256i(_mm256_adds_epi16(self.to_m256i(), rhs.to_m256i())) }
    }

    #[inline]
    pub fn saturating_sub(self, rhs: Self) -> Self {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if crate::wide::avx2_available() {
            return unsafe { self.saturating_sub_avx2(rhs) };
        }
        Self::from_halves(self.lo.saturating_sub(rhs.lo), self.hi.saturating_sub(rhs.hi))
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx2")]
    unsafe fn saturating_sub_avx2(self, rhs: Self) -> Self {
        unsafe { Self::from_m256i(_mm256_subs_epi16(self.to_m256i(), rhs.to_m256i())) }
    }

    #[inline(always)]
    pub fn shl(self, count: u32) -> Self {
        Self::from_halves(self.lo.shl(count), self.hi.shl(count))
    }
    #[inline(always)]
    pub fn shr_arithmetic(self, count: u32) -> Self {
        Self::from_halves(self.lo.shr_arithmetic(count), self.hi.shr_arithmetic(count))
    }
    #[inline(always)]
    pub fn shr_logical(self, count: u32) -> Self {
        Self::from_halves(self.lo.shr_logical(count), self.hi.shr_logical(count))
    }

    #[inline(always)] pub fn cmpeq(self, rhs: Self) -> IMask16x16 { IMask16x16::from_halves(self.lo.cmpeq(rhs.lo), self.hi.cmpeq(rhs.hi)) }
    #[inline(always)] pub fn cmpne(self, rhs: Self) -> IMask16x16 { !self.cmpeq(rhs) }
    #[inline(always)] pub fn cmpgt(self, rhs: Self) -> IMask16x16 { IMask16x16::from_halves(self.lo.cmpgt(rhs.lo), self.hi.cmpgt(rhs.hi)) }
    #[inline(always)] pub fn cmplt(self, rhs: Self) -> IMask16x16 { IMask16x16::from_halves(self.lo.cmplt(rhs.lo), self.hi.cmplt(rhs.hi)) }
    #[inline(always)] pub fn cmpge(self, rhs: Self) -> IMask16x16 { !self.cmplt(rhs) }
    #[inline(always)] pub fn cmple(self, rhs: Self) -> IMask16x16 { !self.cmpgt(rhs) }

    #[inline(always)]
    pub fn blend(mask: IMask16x16, if_true: Self, if_false: Self) -> Self {
        Self::from_halves(
            i16x8::blend(mask.lo, if_true.lo, if_false.lo),
            i16x8::blend(mask.hi, if_true.hi, if_false.hi),
        )
    }

    #[inline(always)] pub fn wrapping_add(self, r: Self) -> Self { self + r }
    #[inline(always)] pub fn wrapping_sub(self, r: Self) -> Self { self - r }
    #[inline(always)] pub fn wrapping_mul(self, r: Self) -> Self { self.mul_lo(r) }
}

impl Add for i16x16 {
    type Output = Self;
    #[inline]
    fn add(self, r: Self) -> Self {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if crate::wide::avx2_available() {
            return unsafe { self.add_avx2(r) };
        }
        Self::from_halves(self.lo + r.lo, self.hi + r.hi)
    }
}
impl i16x16 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx2")]
    unsafe fn add_avx2(self, r: Self) -> Self {
        unsafe { Self::from_m256i(_mm256_add_epi16(self.to_m256i(), r.to_m256i())) }
    }
}
impl AddAssign for i16x16 { #[inline(always)] fn add_assign(&mut self, r: Self) { *self = *self + r; } }

impl Sub for i16x16 {
    type Output = Self;
    #[inline]
    fn sub(self, r: Self) -> Self {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if crate::wide::avx2_available() {
            return unsafe { self.sub_avx2(r) };
        }
        Self::from_halves(self.lo - r.lo, self.hi - r.hi)
    }
}
impl i16x16 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx2")]
    unsafe fn sub_avx2(self, r: Self) -> Self {
        unsafe { Self::from_m256i(_mm256_sub_epi16(self.to_m256i(), r.to_m256i())) }
    }
}
impl SubAssign for i16x16 { #[inline(always)] fn sub_assign(&mut self, r: Self) { *self = *self - r; } }

impl Neg for i16x16 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self { Self::ZERO - self }
}

impl Mul for i16x16 { type Output = Self; #[inline(always)] fn mul(self, r: Self) -> Self { self.mul_lo(r) } }
impl MulAssign for i16x16 { #[inline(always)] fn mul_assign(&mut self, r: Self) { *self = *self * r; } }

impl BitAnd for i16x16 { type Output = Self; #[inline(always)] fn bitand(self, r: Self) -> Self { Self::from_halves(self.lo & r.lo, self.hi & r.hi) } }
impl BitAndAssign for i16x16 { #[inline(always)] fn bitand_assign(&mut self, r: Self) { *self = *self & r; } }
impl BitOr for i16x16 { type Output = Self; #[inline(always)] fn bitor(self, r: Self) -> Self { Self::from_halves(self.lo | r.lo, self.hi | r.hi) } }
impl BitOrAssign for i16x16 { #[inline(always)] fn bitor_assign(&mut self, r: Self) { *self = *self | r; } }
impl BitXor for i16x16 { type Output = Self; #[inline(always)] fn bitxor(self, r: Self) -> Self { Self::from_halves(self.lo ^ r.lo, self.hi ^ r.hi) } }
impl BitXorAssign for i16x16 { #[inline(always)] fn bitxor_assign(&mut self, r: Self) { *self = *self ^ r; } }
impl Not for i16x16 { type Output = Self; #[inline(always)] fn not(self) -> Self { Self::from_halves(!self.lo, !self.hi) } }

impl fmt::Debug for i16x16 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let a = self.to_array();
        write!(f, "i16x16({},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{})",
            a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7],
            a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15])
    }
}
impl fmt::Display for i16x16 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let a = self.to_array();
        write!(f, "[{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}]",
            a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7],
            a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15])
    }
}
impl From<[i16; 16]> for i16x16 { #[inline(always)] fn from(a: [i16; 16]) -> Self { Self::from_array(a) } }
impl From<i16x16> for [i16; 16] { #[inline(always)] fn from(v: i16x16) -> [i16; 16] { v.to_array() } }
