// crates/mid-math/src/wide/int/avx2/i32x8.rs
//! 8-lane signed 32-bit integer vector.
//!
//! Storage is two portable `i32x4` halves (`lo`, `hi`), never a raw
//! `__m256i` outside a `#[target_feature(enable = "avx2")]`-gated function.
//! This is not a style preference -- Rust's own documented multiversioning
//! pattern (https://doc.rust-lang.org/std/arch/index.html) never lets a raw
//! vector register type exist outside such a function, and confirmed
//! separately: even just moving/storing a 256-bit register value needs at
//! least the `avx` target feature, so a type that might run on a non-AVX2
//! CPU cannot hold one as an ordinary field.
//!
//! Every arithmetic method is a safe function that checks
//! `crate::wide::avx2_available()` (cached after the first call) and either
//! calls a `#[target_feature(enable = "avx2")]`-gated inner function (packs
//! `lo`/`hi` into a real `__m256i`, runs the actual AVX2 intrinsic, unpacks
//! back to two halves) or falls straight through to calling the matching
//! `i32x4` method on each half and recombining -- reusing that
//! already-correct code rather than writing a second implementation by
//! hand. The AVX2 path costs a pack/unpack per call that a single call
//! site does not recoup; it pays off when many operations run inside one
//! `#[target_feature(enable = "avx2")]` scope without unpacking in between
//! (a batch/FFI function processing a whole slice, for instance), not when
//! called one at a time the way this file's own methods do.

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

use crate::wide::int::sse2::i32x4::i32x4;
use super::imask32x8::IMask32x8;

/// 8-lane signed 32-bit integer vector. Two `i32x4` halves — see this
/// file's own doc comment for why that is the storage, not a raw `__m256i`.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct i32x8 {
    lo: i32x4,
    hi: i32x4,
}

impl i32x8 {
    pub const ZERO: Self = Self { lo: i32x4::ZERO, hi: i32x4::ZERO };
    pub const ONE:  Self = Self { lo: i32x4::ONE,  hi: i32x4::ONE };
    pub const MIN:  Self = Self { lo: i32x4::MIN,  hi: i32x4::MIN };
    pub const MAX:  Self = Self { lo: i32x4::MAX,  hi: i32x4::MAX };

    #[inline(always)]
    pub(crate) fn from_halves(lo: i32x4, hi: i32x4) -> Self { Self { lo, hi } }

    #[inline(always)]
    pub fn splat(v: i32) -> Self { Self::from_halves(i32x4::splat(v), i32x4::splat(v)) }

    #[inline(always)]
    pub fn new(a: i32, b: i32, c: i32, d: i32, e: i32, f: i32, g: i32, h: i32) -> Self {
        Self::from_halves(i32x4::new(a, b, c, d), i32x4::new(e, f, g, h))
    }

    #[inline(always)]
    pub fn from_array(a: [i32; 8]) -> Self {
        let lo: [i32; 4] = [a[0], a[1], a[2], a[3]];
        let hi: [i32; 4] = [a[4], a[5], a[6], a[7]];
        Self::from_halves(i32x4::from_array(lo), i32x4::from_array(hi))
    }

    #[inline(always)]
    pub fn to_array(self) -> [i32; 8] {
        let lo = self.lo.to_array();
        let hi = self.hi.to_array();
        [lo[0], lo[1], lo[2], lo[3], hi[0], hi[1], hi[2], hi[3]]
    }

    #[inline]
    pub fn get(self, i: usize) -> i32 {
        assert!(i < 8, "i32x8::get — lane {i} out of bounds (max 7)");
        if i < 4 { self.lo.get(i) } else { self.hi.get(i - 4) }
    }

    // ── AVX2 pack/unpack helpers — the ONLY place a __m256i exists, ever,
    // and only inside functions confirmed to run on AVX2-capable hardware ──

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
                i32x4(_mm256_castsi256_si128(v)),
                i32x4(_mm256_extracti128_si256::<1>(v)),
            )
        }
    }

    // ── Elementwise arithmetic — real AVX2 fast path + portable fallback ──

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
        unsafe { Self::from_m256i(_mm256_abs_epi32(self.to_m256i())) }
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
        unsafe { Self::from_m256i(_mm256_min_epi32(self.to_m256i(), rhs.to_m256i())) }
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
        unsafe { Self::from_m256i(_mm256_max_epi32(self.to_m256i(), rhs.to_m256i())) }
    }

    #[inline(always)]
    pub fn clamp(self, lo: Self, hi: Self) -> Self { self.max(lo).min(hi) }

    #[inline]
    pub fn min_element(self) -> i32 { self.lo.min_element().min(self.hi.min_element()) }
    #[inline]
    pub fn max_element(self) -> i32 { self.lo.max_element().max(self.hi.max_element()) }
    #[inline]
    pub fn element_sum(self) -> i32 { self.lo.element_sum().wrapping_add(self.hi.element_sum()) }

    // ── Shifts (uniform count, not per-lane) ──

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

    // ── Comparisons — cheap enough (movemask, not arithmetic throughput)
    // that they go straight to the portable halves, same reasoning as
    // IMask32x8 itself; see that file's doc comment ──

    #[inline(always)] pub fn cmpeq(self, rhs: Self) -> IMask32x8 { IMask32x8::from_halves(self.lo.cmpeq(rhs.lo), self.hi.cmpeq(rhs.hi)) }
    #[inline(always)] pub fn cmpne(self, rhs: Self) -> IMask32x8 { !self.cmpeq(rhs) }
    #[inline(always)] pub fn cmpgt(self, rhs: Self) -> IMask32x8 { IMask32x8::from_halves(self.lo.cmpgt(rhs.lo), self.hi.cmpgt(rhs.hi)) }
    #[inline(always)] pub fn cmplt(self, rhs: Self) -> IMask32x8 { IMask32x8::from_halves(self.lo.cmplt(rhs.lo), self.hi.cmplt(rhs.hi)) }
    #[inline(always)] pub fn cmpge(self, rhs: Self) -> IMask32x8 { !self.cmplt(rhs) }
    #[inline(always)] pub fn cmple(self, rhs: Self) -> IMask32x8 { !self.cmpgt(rhs) }

    #[inline(always)]
    pub fn blend(mask: IMask32x8, if_true: Self, if_false: Self) -> Self {
        Self::from_halves(
            i32x4::blend(mask.lo, if_true.lo, if_false.lo),
            i32x4::blend(mask.hi, if_true.hi, if_false.hi),
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

impl Add for i32x8 {
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
impl i32x8 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx2")]
    unsafe fn add_avx2(self, r: Self) -> Self {
        unsafe { Self::from_m256i(_mm256_add_epi32(self.to_m256i(), r.to_m256i())) }
    }
}
impl AddAssign for i32x8 { #[inline(always)] fn add_assign(&mut self, r: Self) { *self = *self + r; } }

impl Sub for i32x8 {
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
impl i32x8 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx2")]
    unsafe fn sub_avx2(self, r: Self) -> Self {
        unsafe { Self::from_m256i(_mm256_sub_epi32(self.to_m256i(), r.to_m256i())) }
    }
}
impl SubAssign for i32x8 { #[inline(always)] fn sub_assign(&mut self, r: Self) { *self = *self - r; } }

impl Neg for i32x8 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self { Self::ZERO - self }
}

/// AVX2 has native `_mm256_mullo_epi32` — no shuffle/unpack emulation
/// needed on the fast path, unlike SSE2's `i32x4::mul`.
impl Mul for i32x8 {
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
impl i32x8 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx2")]
    unsafe fn mul_avx2(self, r: Self) -> Self {
        unsafe { Self::from_m256i(_mm256_mullo_epi32(self.to_m256i(), r.to_m256i())) }
    }
}
impl MulAssign for i32x8 { #[inline(always)] fn mul_assign(&mut self, r: Self) { *self = *self * r; } }

impl BitAnd for i32x8 { type Output = Self; #[inline(always)] fn bitand(self, r: Self) -> Self { Self::from_halves(self.lo & r.lo, self.hi & r.hi) } }
impl BitAndAssign for i32x8 { #[inline(always)] fn bitand_assign(&mut self, r: Self) { *self = *self & r; } }
impl BitOr for i32x8 { type Output = Self; #[inline(always)] fn bitor(self, r: Self) -> Self { Self::from_halves(self.lo | r.lo, self.hi | r.hi) } }
impl BitOrAssign for i32x8 { #[inline(always)] fn bitor_assign(&mut self, r: Self) { *self = *self | r; } }
impl BitXor for i32x8 { type Output = Self; #[inline(always)] fn bitxor(self, r: Self) -> Self { Self::from_halves(self.lo ^ r.lo, self.hi ^ r.hi) } }
impl BitXorAssign for i32x8 { #[inline(always)] fn bitxor_assign(&mut self, r: Self) { *self = *self ^ r; } }
impl Not for i32x8 { type Output = Self; #[inline(always)] fn not(self) -> Self { Self::from_halves(!self.lo, !self.hi) } }

impl fmt::Debug for i32x8 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let a = self.to_array();
        write!(f, "i32x8({},{},{},{},{},{},{},{})", a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7])
    }
}
impl fmt::Display for i32x8 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let a = self.to_array();
        write!(f, "[{},{},{},{},{},{},{},{}]", a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7])
    }
}
impl From<[i32; 8]> for i32x8 { #[inline(always)] fn from(a: [i32; 8]) -> Self { Self::from_array(a) } }
impl From<i32x8> for [i32; 8] { #[inline(always)] fn from(v: i32x8) -> [i32; 8] { v.to_array() } }
