// crates/mid-math/src/wide/int/avx2/u8x32.rs
//! 32-lane unsigned 8-bit integer vector.
//!
//! Same storage/dispatch design as `i32x8.rs` — see that file's doc
//! comment for the full reasoning. Portable `{lo, hi}: u8x16` storage.
//!
//! `shuffle_bytes` now takes `indices: Self` (matches `u8x16::shuffle_bytes`'s
//! own convention) rather than the original's `indices: i8x32` — a small,
//! deliberate signature change, not preserved from the original, since
//! nothing in this crate's own benches calls `u8x32::shuffle_bytes`
//! (confirmed before making the change) and matching `u8x16`'s own
//! convention is more internally consistent than the original's mixed
//! signed/unsigned parameter.
//!
//! `element_sum` no longer uses `_mm256_sad_epu8` on the fast path composed
//! from two `u8x16::element_sum` calls — each half's own `element_sum`
//! already does the equivalent SAD-based reduction at width 16 (see
//! `sse2/u8x16.rs`), so this stays portable-only like the other
//! reduction methods, not dispatch-wrapped.

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

use crate::wide::int::sse2::u8x16::u8x16;
use super::u16x16::u16x16;
use super::imask8x32::IMask8x32;

/// 32-lane unsigned 8-bit integer vector. Two `u8x16` halves.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct u8x32 {
    lo: u8x16,
    hi: u8x16,
}

impl u8x32 {
    pub const ZERO: Self = Self { lo: u8x16::ZERO, hi: u8x16::ZERO };
    pub const ONE:  Self = Self { lo: u8x16::ONE,  hi: u8x16::ONE };
    pub const MIN:  Self = Self { lo: u8x16::MIN,  hi: u8x16::MIN };
    pub const MAX:  Self = Self { lo: u8x16::MAX,  hi: u8x16::MAX };

    #[inline(always)]
    pub(crate) fn from_halves(lo: u8x16, hi: u8x16) -> Self { Self { lo, hi } }

    #[inline(always)]
    pub fn splat(v: u8) -> Self { Self::from_halves(u8x16::splat(v), u8x16::splat(v)) }

    #[inline(always)]
    pub fn from_array(a: [u8; 32]) -> Self {
        let mut lo = [0u8; 16];
        let mut hi = [0u8; 16];
        lo.copy_from_slice(&a[0..16]);
        hi.copy_from_slice(&a[16..32]);
        Self::from_halves(u8x16::from_array(lo), u8x16::from_array(hi))
    }
    #[inline(always)]
    pub fn from_bytes(b: [u8; 32]) -> Self { Self::from_array(b) }

    #[inline(always)]
    pub fn to_array(self) -> [u8; 32] {
        let lo = self.lo.to_array();
        let hi = self.hi.to_array();
        let mut out = [0u8; 32];
        out[0..16].copy_from_slice(&lo);
        out[16..32].copy_from_slice(&hi);
        out
    }

    /// Zero-extend lanes 0-15 to `u16x16`. Portable — composes from
    /// `u8x16::as_u16x8_lo`/`as_u16x8_hi`.
    #[inline(always)]
    pub fn as_u16x16_lo(self) -> u16x16 {
        u16x16::from_halves(self.lo.as_u16x8_lo(), self.lo.as_u16x8_hi())
    }
    /// Zero-extend lanes 16-31 to `u16x16`.
    #[inline(always)]
    pub fn as_u16x16_hi(self) -> u16x16 {
        u16x16::from_halves(self.hi.as_u16x8_lo(), self.hi.as_u16x8_hi())
    }

    /// Split into two `sse2::u8x16` halves. Trivial now — this type
    /// already stores its data this way.
    #[inline(always)]
    pub fn to_u8x16_pair(self) -> (u8x16, u8x16) { (self.lo, self.hi) }
    /// Combine two `sse2::u8x16` halves into one `u8x32`.
    #[inline(always)]
    pub fn from_u8x16_pair(lo: u8x16, hi: u8x16) -> Self { Self::from_halves(lo, hi) }

    /// Byte shuffle within each 16-byte half independently — see
    /// `i8x32::shuffle_bytes`'s doc comment for why the fallback has
    /// identical semantics to the real `_mm256_shuffle_epi8` instruction.
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
    pub fn get(self, i: usize) -> u8 {
        assert!(i < 32, "u8x32::get — lane {i} out of bounds (max 31)");
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
                u8x16(_mm256_castsi256_si128(v)),
                u8x16(_mm256_extracti128_si256::<1>(v)),
            )
        }
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
        unsafe { Self::from_m256i(_mm256_min_epu8(self.to_m256i(), rhs.to_m256i())) }
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
        unsafe { Self::from_m256i(_mm256_max_epu8(self.to_m256i(), rhs.to_m256i())) }
    }

    #[inline(always)]
    pub fn clamp(self, lo: Self, hi: Self) -> Self { self.max(lo).min(hi) }

    #[inline] pub fn min_element(self) -> u8 { self.lo.min_element().min(self.hi.min_element()) }
    #[inline] pub fn max_element(self) -> u8 { self.lo.max_element().max(self.hi.max_element()) }
    #[inline] pub fn element_sum(self) -> u32 { self.lo.element_sum() + self.hi.element_sum() }

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
        unsafe { Self::from_m256i(_mm256_adds_epu8(self.to_m256i(), rhs.to_m256i())) }
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
        unsafe { Self::from_m256i(_mm256_subs_epu8(self.to_m256i(), rhs.to_m256i())) }
    }

    #[inline(always)] pub fn cmpeq(self, rhs: Self) -> IMask8x32 { IMask8x32::from_halves(self.lo.cmpeq(rhs.lo), self.hi.cmpeq(rhs.hi)) }
    #[inline(always)] pub fn cmpne(self, rhs: Self) -> IMask8x32 { !self.cmpeq(rhs) }
    #[inline(always)] pub fn cmpgt(self, rhs: Self) -> IMask8x32 { IMask8x32::from_halves(self.lo.cmpgt(rhs.lo), self.hi.cmpgt(rhs.hi)) }
    #[inline(always)] pub fn cmplt(self, rhs: Self) -> IMask8x32 { IMask8x32::from_halves(self.lo.cmplt(rhs.lo), self.hi.cmplt(rhs.hi)) }
    #[inline(always)] pub fn cmpge(self, rhs: Self) -> IMask8x32 { !self.cmplt(rhs) }
    #[inline(always)] pub fn cmple(self, rhs: Self) -> IMask8x32 { !self.cmpgt(rhs) }

    #[inline(always)]
    pub fn blend(mask: IMask8x32, if_true: Self, if_false: Self) -> Self {
        Self::from_halves(
            u8x16::blend(mask.lo, if_true.lo, if_false.lo),
            u8x16::blend(mask.hi, if_true.hi, if_false.hi),
        )
    }

    #[inline]
    pub fn count_eq(self, needle: Self) -> u32 { self.cmpeq(needle).count_true() }
    #[inline]
    pub fn contains(self, needle: u8) -> bool { self.count_eq(Self::splat(needle)) > 0 }

    #[inline(always)] pub fn wrapping_add(self, r: Self) -> Self { self + r }
    #[inline(always)] pub fn wrapping_sub(self, r: Self) -> Self { self - r }
}

impl Add for u8x32 {
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
impl u8x32 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx2")]
    unsafe fn add_avx2(self, r: Self) -> Self {
        unsafe { Self::from_m256i(_mm256_add_epi8(self.to_m256i(), r.to_m256i())) }
    }
}
impl AddAssign for u8x32 { #[inline(always)] fn add_assign(&mut self, r: Self) { *self = *self + r; } }

impl Sub for u8x32 {
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
impl u8x32 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx2")]
    unsafe fn sub_avx2(self, r: Self) -> Self {
        unsafe { Self::from_m256i(_mm256_sub_epi8(self.to_m256i(), r.to_m256i())) }
    }
}
impl SubAssign for u8x32 { #[inline(always)] fn sub_assign(&mut self, r: Self) { *self = *self - r; } }

impl BitAnd for u8x32 { type Output = Self; #[inline(always)] fn bitand(self, r: Self) -> Self { Self::from_halves(self.lo & r.lo, self.hi & r.hi) } }
impl BitAndAssign for u8x32 { #[inline(always)] fn bitand_assign(&mut self, r: Self) { *self = *self & r; } }
impl BitOr for u8x32 { type Output = Self; #[inline(always)] fn bitor(self, r: Self) -> Self { Self::from_halves(self.lo | r.lo, self.hi | r.hi) } }
impl BitOrAssign for u8x32 { #[inline(always)] fn bitor_assign(&mut self, r: Self) { *self = *self | r; } }
impl BitXor for u8x32 { type Output = Self; #[inline(always)] fn bitxor(self, r: Self) -> Self { Self::from_halves(self.lo ^ r.lo, self.hi ^ r.hi) } }
impl BitXorAssign for u8x32 { #[inline(always)] fn bitxor_assign(&mut self, r: Self) { *self = *self ^ r; } }
impl Not for u8x32 { type Output = Self; #[inline(always)] fn not(self) -> Self { Self::from_halves(!self.lo, !self.hi) } }

impl fmt::Debug for u8x32 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "u8x32({:?})", self.to_array()) }
}
impl fmt::Display for u8x32 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "{:?}", self.to_array()) }
}
impl From<[u8; 32]> for u8x32 { #[inline(always)] fn from(a: [u8; 32]) -> Self { Self::from_array(a) } }
impl From<u8x32> for [u8; 32] { #[inline(always)] fn from(v: u8x32) -> Self { v.to_array() } }
