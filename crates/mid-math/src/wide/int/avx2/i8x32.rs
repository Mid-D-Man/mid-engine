// crates/mid-math/src/wide/int/avx2/i8x32.rs
//! 32-lane signed 8-bit integer vector.
//!
//! Same storage/dispatch design as `i32x8.rs` — see that file's doc
//! comment for the full reasoning. Portable `{lo, hi}: i8x16` storage —
//! which makes `to_i8x16_pair`/`from_i8x16_pair` trivial field access now
//! (they used to be the AVX2-specific split/join primitives; the type
//! itself already stores its data that way).
//!
//! `shuffle_bytes` genuinely is per-128-bit-lane on real AVX2 hardware
//! (`_mm256_shuffle_epi8`), so the fallback (`i8x16::shuffle_bytes` on
//! each half independently) has identical semantics to the real
//! instruction — not just an approximation, the actual documented
//! behavior of `_mm256_shuffle_epi8` already stays within each half.
//!
//! No multiply/shift, same as `sse2/i8x16.rs` — no native 8-bit SIMD
//! multiply or byte-granularity shift exists on x86 at all, AVX2 included.

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

use crate::wide::int::sse2::i8x16::i8x16;
use super::i16x16::i16x16;
use super::imask8x32::IMask8x32;

/// 32-lane signed 8-bit integer vector. Two `i8x16` halves.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct i8x32 {
    lo: i8x16,
    hi: i8x16,
}

impl i8x32 {
    pub const ZERO: Self = Self { lo: i8x16::ZERO, hi: i8x16::ZERO };
    pub const ONE:  Self = Self { lo: i8x16::ONE,  hi: i8x16::ONE };
    pub const MIN:  Self = Self { lo: i8x16::MIN,  hi: i8x16::MIN };
    pub const MAX:  Self = Self { lo: i8x16::MAX,  hi: i8x16::MAX };

    #[inline(always)]
    pub(crate) fn from_halves(lo: i8x16, hi: i8x16) -> Self { Self { lo, hi } }

    #[inline(always)]
    pub fn splat(v: i8) -> Self { Self::from_halves(i8x16::splat(v), i8x16::splat(v)) }

    #[inline(always)]
    pub fn from_array(a: [i8; 32]) -> Self {
        let mut lo = [0i8; 16];
        let mut hi = [0i8; 16];
        lo.copy_from_slice(&a[0..16]);
        hi.copy_from_slice(&a[16..32]);
        Self::from_halves(i8x16::from_array(lo), i8x16::from_array(hi))
    }
    #[inline(always)]
    pub fn from_bytes(b: [u8; 32]) -> Self {
        let mut lo = [0u8; 16];
        let mut hi = [0u8; 16];
        lo.copy_from_slice(&b[0..16]);
        hi.copy_from_slice(&b[16..32]);
        Self::from_halves(i8x16::from_bytes(lo), i8x16::from_bytes(hi))
    }

    #[inline(always)]
    pub fn to_array(self) -> [i8; 32] {
        let lo = self.lo.to_array();
        let hi = self.hi.to_array();
        let mut out = [0i8; 32];
        out[0..16].copy_from_slice(&lo);
        out[16..32].copy_from_slice(&hi);
        out
    }
    #[inline(always)]
    pub fn to_bytes(self) -> [u8; 32] {
        let lo = self.lo.to_bytes();
        let hi = self.hi.to_bytes();
        let mut out = [0u8; 32];
        out[0..16].copy_from_slice(&lo);
        out[16..32].copy_from_slice(&hi);
        out
    }

    /// Sign-extend lanes 0-15 to `i16x16`. Portable — composes from
    /// `i8x16::as_i16x8_lo`/`as_i16x8_hi`, same reasoning as
    /// `i16x16::as_i32x8_lo`/`hi`.
    #[inline(always)]
    pub fn as_i16x16_lo(self) -> i16x16 {
        i16x16::from_halves(self.lo.as_i16x8_lo(), self.lo.as_i16x8_hi())
    }
    /// Sign-extend lanes 16-31 to `i16x16`.
    #[inline(always)]
    pub fn as_i16x16_hi(self) -> i16x16 {
        i16x16::from_halves(self.hi.as_i16x8_lo(), self.hi.as_i16x8_hi())
    }

    /// Split into two `sse2::i8x16` halves. Trivial now — this type
    /// already stores its data this way.
    #[inline(always)]
    pub fn to_i8x16_pair(self) -> (i8x16, i8x16) { (self.lo, self.hi) }
    /// Combine two `sse2::i8x16` halves into one `i8x32`.
    #[inline(always)]
    pub fn from_i8x16_pair(lo: i8x16, hi: i8x16) -> Self { Self::from_halves(lo, hi) }

    /// Byte shuffle within each 16-byte half independently — matches
    /// `_mm256_shuffle_epi8`'s actual documented per-128-bit-lane
    /// behavior exactly (see this file's own doc comment), so the
    /// fallback (`i8x16::shuffle_bytes` on each half) isn't an
    /// approximation of the real instruction, it's the same semantics.
    #[inline]
    pub fn shuffle_bytes(self, indices: Self) -> Self {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if crate::wide::avx2_available() {
            return unsafe { self.shuffle_bytes_avx2(indices) };
        }
        Self::from_halves(self.lo.shuffle_bytes(indices.lo), self.hi.shuffle_bytes(indices.hi))
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx2")]
    unsafe fn shuffle_bytes_avx2(self, indices: Self) -> Self {
        unsafe { Self::from_m256i(_mm256_shuffle_epi8(self.to_m256i(), indices.to_m256i())) }
    }

    #[inline]
    pub fn get(self, i: usize) -> i8 {
        assert!(i < 32, "i8x32::get — lane {i} out of bounds (max 31)");
        if i < 16 { self.lo.get(i) } else { self.hi.get(i - 16) }
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
                i8x16(_mm256_castsi256_si128(v)),
                i8x16(_mm256_extracti128_si256::<1>(v)),
            )
        }
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
        unsafe { Self::from_m256i(_mm256_adds_epi8(self.to_m256i(), rhs.to_m256i())) }
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
        unsafe { Self::from_m256i(_mm256_subs_epi8(self.to_m256i(), rhs.to_m256i())) }
    }

    /// Absolute value. AVX2 native `_mm256_abs_epi8` fast path — SSE2's
    /// `i8x16::abs` needs cmplt+sub+blend, so unlike i16/i32, this one
    /// genuinely differs between the fast path and the fallback (both
    /// produce the same result, the AVX2 path just has a real dedicated
    /// instruction where SSE2 has none).
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
        unsafe { Self::from_m256i(_mm256_abs_epi8(self.to_m256i())) }
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
        unsafe { Self::from_m256i(_mm256_min_epi8(self.to_m256i(), rhs.to_m256i())) }
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
        unsafe { Self::from_m256i(_mm256_max_epi8(self.to_m256i(), rhs.to_m256i())) }
    }

    #[inline(always)]
    pub fn clamp(self, lo: Self, hi: Self) -> Self { self.max(lo).min(hi) }

    #[inline] pub fn min_element(self) -> i8 { self.lo.min_element().min(self.hi.min_element()) }
    #[inline] pub fn max_element(self) -> i8 { self.lo.max_element().max(self.hi.max_element()) }
    #[inline] pub fn element_sum(self) -> i32 { self.lo.element_sum() + self.hi.element_sum() }

    #[inline(always)] pub fn cmpeq(self, rhs: Self) -> IMask8x32 { IMask8x32::from_halves(self.lo.cmpeq(rhs.lo), self.hi.cmpeq(rhs.hi)) }
    #[inline(always)] pub fn cmpne(self, rhs: Self) -> IMask8x32 { !self.cmpeq(rhs) }
    #[inline(always)] pub fn cmpgt(self, rhs: Self) -> IMask8x32 { IMask8x32::from_halves(self.lo.cmpgt(rhs.lo), self.hi.cmpgt(rhs.hi)) }
    #[inline(always)] pub fn cmplt(self, rhs: Self) -> IMask8x32 { IMask8x32::from_halves(self.lo.cmplt(rhs.lo), self.hi.cmplt(rhs.hi)) }
    #[inline(always)] pub fn cmpge(self, rhs: Self) -> IMask8x32 { !self.cmplt(rhs) }
    #[inline(always)] pub fn cmple(self, rhs: Self) -> IMask8x32 { !self.cmpgt(rhs) }

    #[inline(always)]
    pub fn blend(mask: IMask8x32, if_true: Self, if_false: Self) -> Self {
        Self::from_halves(
            i8x16::blend(mask.lo, if_true.lo, if_false.lo),
            i8x16::blend(mask.hi, if_true.hi, if_false.hi),
        )
    }

    #[inline]
    pub fn count_eq(self, needle: Self) -> u32 { self.cmpeq(needle).count_true() }
    #[inline]
    pub fn contains(self, needle: i8) -> bool { self.count_eq(Self::splat(needle)) > 0 }

    #[inline(always)] pub fn wrapping_add(self, r: Self) -> Self { self + r }
    #[inline(always)] pub fn wrapping_sub(self, r: Self) -> Self { self - r }
}

impl Add for i8x32 {
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
impl i8x32 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx2")]
    unsafe fn add_avx2(self, r: Self) -> Self {
        unsafe { Self::from_m256i(_mm256_add_epi8(self.to_m256i(), r.to_m256i())) }
    }
}
impl AddAssign for i8x32 { #[inline(always)] fn add_assign(&mut self, r: Self) { *self = *self + r; } }

impl Sub for i8x32 {
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
impl i8x32 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx2")]
    unsafe fn sub_avx2(self, r: Self) -> Self {
        unsafe { Self::from_m256i(_mm256_sub_epi8(self.to_m256i(), r.to_m256i())) }
    }
}
impl SubAssign for i8x32 { #[inline(always)] fn sub_assign(&mut self, r: Self) { *self = *self - r; } }

impl Neg for i8x32 { type Output = Self; #[inline(always)] fn neg(self) -> Self { Self::ZERO - self } }

impl BitAnd for i8x32 { type Output = Self; #[inline(always)] fn bitand(self, r: Self) -> Self { Self::from_halves(self.lo & r.lo, self.hi & r.hi) } }
impl BitAndAssign for i8x32 { #[inline(always)] fn bitand_assign(&mut self, r: Self) { *self = *self & r; } }
impl BitOr for i8x32 { type Output = Self; #[inline(always)] fn bitor(self, r: Self) -> Self { Self::from_halves(self.lo | r.lo, self.hi | r.hi) } }
impl BitOrAssign for i8x32 { #[inline(always)] fn bitor_assign(&mut self, r: Self) { *self = *self | r; } }
impl BitXor for i8x32 { type Output = Self; #[inline(always)] fn bitxor(self, r: Self) -> Self { Self::from_halves(self.lo ^ r.lo, self.hi ^ r.hi) } }
impl BitXorAssign for i8x32 { #[inline(always)] fn bitxor_assign(&mut self, r: Self) { *self = *self ^ r; } }
impl Not for i8x32 { type Output = Self; #[inline(always)] fn not(self) -> Self { Self::from_halves(!self.lo, !self.hi) } }

impl fmt::Debug for i8x32 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "i8x32({:?})", self.to_array()) }
}
impl fmt::Display for i8x32 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "{:?}", self.to_array()) }
}
impl From<[i8; 32]> for i8x32 { #[inline(always)] fn from(a: [i8; 32]) -> Self { Self::from_array(a) } }
impl From<i8x32> for [i8; 32] { #[inline(always)] fn from(v: i8x32) -> Self { v.to_array() } }
impl From<[u8; 32]> for i8x32 { #[inline(always)] fn from(b: [u8; 32]) -> Self { Self::from_bytes(b) } }
impl From<i8x32> for [u8; 32] { #[inline(always)] fn from(v: i8x32) -> Self { v.to_bytes() } }
