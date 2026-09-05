// crates/mid-math/src/wide/int/avx2/u32x8.rs
//! 8-lane unsigned 32-bit integer vector.
//!
//! Same storage/dispatch design as `i32x8.rs` — see that file's doc
//! comment for the full reasoning (portable `{lo, hi}: u32x4` storage,
//! never a raw `__m256i` outside a `#[target_feature(enable = "avx2")]`
//! scope; real AVX2 fast path + portable fallback per method).

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

use crate::wide::int::sse2::u32x4::u32x4;
use super::imask32x8::IMask32x8;

/// 8-lane unsigned 32-bit integer vector. Two `u32x4` halves.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct u32x8 {
    lo: u32x4,
    hi: u32x4,
}

impl u32x8 {
    pub const ZERO: Self = Self { lo: u32x4::ZERO, hi: u32x4::ZERO };
    pub const ONE:  Self = Self { lo: u32x4::ONE,  hi: u32x4::ONE };
    pub const MIN:  Self = Self { lo: u32x4::MIN,  hi: u32x4::MIN };
    pub const MAX:  Self = Self { lo: u32x4::MAX,  hi: u32x4::MAX };

    #[inline(always)]
    pub(crate) fn from_halves(lo: u32x4, hi: u32x4) -> Self { Self { lo, hi } }

    #[inline(always)]
    pub fn splat(v: u32) -> Self { Self::from_halves(u32x4::splat(v), u32x4::splat(v)) }

    #[inline(always)]
    pub fn new(a: u32, b: u32, c: u32, d: u32, e: u32, f: u32, g: u32, h: u32) -> Self {
        Self::from_halves(u32x4::new(a, b, c, d), u32x4::new(e, f, g, h))
    }

    #[inline(always)]
    pub fn from_array(a: [u32; 8]) -> Self {
        let lo: [u32; 4] = [a[0], a[1], a[2], a[3]];
        let hi: [u32; 4] = [a[4], a[5], a[6], a[7]];
        Self::from_halves(u32x4::from_array(lo), u32x4::from_array(hi))
    }

    #[inline(always)]
    pub fn to_array(self) -> [u32; 8] {
        let lo = self.lo.to_array();
        let hi = self.hi.to_array();
        [lo[0], lo[1], lo[2], lo[3], hi[0], hi[1], hi[2], hi[3]]
    }

    #[inline]
    pub fn get(self, i: usize) -> u32 {
        assert!(i < 8, "u32x8::get — lane {i} out of bounds (max 7)");
        if i < 4 { self.lo.get(i) } else { self.hi.get(i - 4) }
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
                u32x4(_mm256_castsi256_si128(v)),
                u32x4(_mm256_extracti128_si256::<1>(v)),
            )
        }
    }

    /// Per-lane minimum. AVX2 native `_mm256_min_epu32` fast path.
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
        unsafe { Self::from_m256i(_mm256_min_epu32(self.to_m256i(), rhs.to_m256i())) }
    }

    /// Per-lane maximum. AVX2 native `_mm256_max_epu32` fast path.
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
        unsafe { Self::from_m256i(_mm256_max_epu32(self.to_m256i(), rhs.to_m256i())) }
    }

    #[inline(always)]
    pub fn clamp(self, lo: Self, hi: Self) -> Self { self.max(lo).min(hi) }

    #[inline]
    pub fn min_element(self) -> u32 { self.lo.min_element().min(self.hi.min_element()) }
    #[inline]
    pub fn max_element(self) -> u32 { self.lo.max_element().max(self.hi.max_element()) }
    #[inline]
    pub fn element_sum(self) -> u32 { self.lo.element_sum().wrapping_add(self.hi.element_sum()) }

    #[inline(always)]
    pub fn shl(self, count: u32) -> Self {
        Self::from_halves(self.lo.shl(count), self.hi.shl(count))
    }
    #[inline(always)]
    pub fn shr(self, count: u32) -> Self {
        Self::from_halves(self.lo.shr(count), self.hi.shr(count))
    }

    #[inline(always)] pub fn cmpeq(self, rhs: Self) -> IMask32x8 { IMask32x8::from_halves(self.lo.cmpeq(rhs.lo), self.hi.cmpeq(rhs.hi)) }
    #[inline(always)] pub fn cmpne(self, rhs: Self) -> IMask32x8 { !self.cmpeq(rhs) }
    #[inline(always)] pub fn cmpgt(self, rhs: Self) -> IMask32x8 { IMask32x8::from_halves(self.lo.cmpgt(rhs.lo), self.hi.cmpgt(rhs.hi)) }
    #[inline(always)] pub fn cmplt(self, rhs: Self) -> IMask32x8 { IMask32x8::from_halves(self.lo.cmplt(rhs.lo), self.hi.cmplt(rhs.hi)) }
    #[inline(always)] pub fn cmpge(self, rhs: Self) -> IMask32x8 { !self.cmplt(rhs) }
    #[inline(always)] pub fn cmple(self, rhs: Self) -> IMask32x8 { !self.cmpgt(rhs) }

    #[inline(always)]
    pub fn blend(mask: IMask32x8, if_true: Self, if_false: Self) -> Self {
        Self::from_halves(
            u32x4::blend(mask.lo, if_true.lo, if_false.lo),
            u32x4::blend(mask.hi, if_true.hi, if_false.hi),
        )
    }

    #[inline(always)] pub fn wrapping_add(self, r: Self) -> Self { self + r }
    #[inline(always)] pub fn wrapping_sub(self, r: Self) -> Self { self - r }
    #[inline(always)] pub fn wrapping_mul(self, r: Self) -> Self { self * r }

    #[inline]
    pub fn saturating_add(self, rhs: Self) -> Self {
        Self::from_halves(self.lo.saturating_add(rhs.lo), self.hi.saturating_add(rhs.hi))
    }
    #[inline]
    pub fn saturating_sub(self, rhs: Self) -> Self {
        Self::from_halves(self.lo.saturating_sub(rhs.lo), self.hi.saturating_sub(rhs.hi))
    }
}

impl Add for u32x8 {
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
impl u32x8 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx2")]
    unsafe fn add_avx2(self, r: Self) -> Self {
        unsafe { Self::from_m256i(_mm256_add_epi32(self.to_m256i(), r.to_m256i())) }
    }
}
impl AddAssign for u32x8 { #[inline(always)] fn add_assign(&mut self, r: Self) { *self = *self + r; } }

impl Sub for u32x8 {
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
impl u32x8 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx2")]
    unsafe fn sub_avx2(self, r: Self) -> Self {
        unsafe { Self::from_m256i(_mm256_sub_epi32(self.to_m256i(), r.to_m256i())) }
    }
}
impl SubAssign for u32x8 { #[inline(always)] fn sub_assign(&mut self, r: Self) { *self = *self - r; } }

/// AVX2 has native `_mm256_mullo_epi32` (same instruction as the signed
/// version — 32-bit multiply low bits are identical for both signedness).
impl Mul for u32x8 {
    type Output = Self;
    #[inline]
    fn mul(self, r: Self) -> Self {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if crate::wide::avx2_available() {
            return unsafe { self.mul_avx2(r) };
        }
        Self::from_halves(self.lo * r.lo, self.hi * r.hi)
    }
}
impl u32x8 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx2")]
    unsafe fn mul_avx2(self, r: Self) -> Self {
        unsafe { Self::from_m256i(_mm256_mullo_epi32(self.to_m256i(), r.to_m256i())) }
    }
}
impl MulAssign for u32x8 { #[inline(always)] fn mul_assign(&mut self, r: Self) { *self = *self * r; } }

impl BitAnd for u32x8 { type Output = Self; #[inline(always)] fn bitand(self, r: Self) -> Self { Self::from_halves(self.lo & r.lo, self.hi & r.hi) } }
impl BitAndAssign for u32x8 { #[inline(always)] fn bitand_assign(&mut self, r: Self) { *self = *self & r; } }
impl BitOr for u32x8 { type Output = Self; #[inline(always)] fn bitor(self, r: Self) -> Self { Self::from_halves(self.lo | r.lo, self.hi | r.hi) } }
impl BitOrAssign for u32x8 { #[inline(always)] fn bitor_assign(&mut self, r: Self) { *self = *self | r; } }
impl BitXor for u32x8 { type Output = Self; #[inline(always)] fn bitxor(self, r: Self) -> Self { Self::from_halves(self.lo ^ r.lo, self.hi ^ r.hi) } }
impl BitXorAssign for u32x8 { #[inline(always)] fn bitxor_assign(&mut self, r: Self) { *self = *self ^ r; } }
impl Not for u32x8 { type Output = Self; #[inline(always)] fn not(self) -> Self { Self::from_halves(!self.lo, !self.hi) } }

impl fmt::Debug for u32x8 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let a = self.to_array();
        write!(f, "u32x8({},{},{},{},{},{},{},{})", a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7])
    }
}
impl fmt::Display for u32x8 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let a = self.to_array();
        write!(f, "[{},{},{},{},{},{},{},{}]", a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7])
    }
}
impl From<[u32; 8]> for u32x8 { #[inline(always)] fn from(a: [u32; 8]) -> Self { Self::from_array(a) } }
impl From<u32x8> for [u32; 8] { #[inline(always)] fn from(v: u32x8) -> [u32; 8] { v.to_array() } }
