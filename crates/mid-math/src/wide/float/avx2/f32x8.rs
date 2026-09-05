// crates/mid-math/src/wide/float/avx2/f32x8.rs
//! 8-lane independent f32 scalar.
//!
//! NOT a vector — a bag of 8 independent scalars, same role `f32x4` plays
//! for `Vec3x4` (per-lane t-values, dot-product results, scalar
//! multipliers). Portable `{lo, hi}: f32x4` storage, never a raw `__m256`
//! outside a `#[target_feature(enable = "avx2")]` scope — same reasoning
//! as `wide/int/avx2/i32x8.rs`'s doc comment (never let a raw vector
//! register type exist outside such a function; even just moving/storing
//! one needs at least the `avx` target feature).
//!
//! This type exists specifically so `Vec3x8::dot`/`length_sq`/`length`/
//! `lerp` can return/take a portable type instead of a raw `__m256`,
//! matching `Vec3x4`'s own convention (`f32x4`, not `__m128`) exactly.

#![allow(non_camel_case_types)]

use core::fmt;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use crate::wide::float::sse2::f32x4::f32x4;
use super::mask8::Mask8;

/// 8-lane independent f32 scalar. Two `f32x4` halves.
#[derive(Clone, Copy)]
pub struct f32x8 {
    lo: f32x4,
    hi: f32x4,
}

impl f32x8 {
    pub const ZERO: Self = Self { lo: f32x4::ZERO, hi: f32x4::ZERO };
    pub const ONE:  Self = Self { lo: f32x4::ONE,  hi: f32x4::ONE };
    pub const NEG_ONE: Self = Self { lo: f32x4::NEG_ONE, hi: f32x4::NEG_ONE };
    pub const INFINITY: Self = Self { lo: f32x4::INFINITY, hi: f32x4::INFINITY };
    pub const NEG_INFINITY: Self = Self { lo: f32x4::NEG_INFINITY, hi: f32x4::NEG_INFINITY };

    #[inline(always)]
    pub(crate) fn from_halves(lo: f32x4, hi: f32x4) -> Self { Self { lo, hi } }
    #[inline(always)]
    pub(crate) fn halves(self) -> (f32x4, f32x4) { (self.lo, self.hi) }

    #[inline(always)]
    pub fn splat(v: f32) -> Self { Self::from_halves(f32x4::splat(v), f32x4::splat(v)) }

    #[inline(always)]
    pub fn new(a: f32, b: f32, c: f32, d: f32, e: f32, f: f32, g: f32, h: f32) -> Self {
        Self::from_halves(f32x4::new(a, b, c, d), f32x4::new(e, f, g, h))
    }

    #[inline(always)]
    pub fn from_array(a: [f32; 8]) -> Self {
        let lo: [f32; 4] = [a[0], a[1], a[2], a[3]];
        let hi: [f32; 4] = [a[4], a[5], a[6], a[7]];
        Self::from_halves(f32x4::from_array(lo), f32x4::from_array(hi))
    }

    #[inline(always)]
    pub fn to_array(self) -> [f32; 8] {
        let lo = self.lo.to_array();
        let hi = self.hi.to_array();
        [lo[0], lo[1], lo[2], lo[3], hi[0], hi[1], hi[2], hi[3]]
    }

    #[inline]
    pub fn get(self, i: usize) -> f32 {
        assert!(i < 8, "f32x8::get — lane {i} out of bounds (max 7)");
        if i < 4 { self.lo.get(i) } else { self.hi.get(i - 4) }
    }

    // ── AVX2 pack/unpack helpers — only place a __m256 exists ──

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx2")]
    unsafe fn to_m256(self) -> __m256 {
        unsafe { _mm256_set_m128(self.hi.0, self.lo.0) }
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx2")]
    unsafe fn from_m256(v: __m256) -> Self {
        unsafe {
            Self::from_halves(
                f32x4(_mm256_castps256_ps128(v)),
                f32x4(_mm256_extractf128_ps::<1>(v)),
            )
        }
    }

    #[inline]
    pub fn sqrt(self) -> Self {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if crate::wide::avx2_available() {
            return unsafe { self.sqrt_avx2() };
        }
        Self::from_halves(self.lo.sqrt(), self.hi.sqrt())
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx2")]
    unsafe fn sqrt_avx2(self) -> Self {
        unsafe { Self::from_m256(_mm256_sqrt_ps(self.to_m256())) }
    }

    /// Fast reciprocal sqrt (rsqrt + one Newton-Raphson step, ~23-bit
    /// accuracy) — portable-only, composes from `f32x4::recip_sqrt`
    /// (which already does the same NR refinement at width 4) rather
    /// than re-deriving the AVX2 rsqrtps+NR sequence a second time.
    #[inline(always)]
    pub fn recip_sqrt(self) -> Self {
        Self::from_halves(self.lo.recip_sqrt(), self.hi.recip_sqrt())
    }
    /// Fast reciprocal (rcp + one Newton-Raphson step) — portable-only,
    /// same reasoning as `recip_sqrt`.
    #[inline(always)]
    pub fn recip(self) -> Self {
        Self::from_halves(self.lo.recip(), self.hi.recip())
    }

    #[inline(always)]
    pub fn abs(self) -> Self { Self::from_halves(self.lo.abs(), self.hi.abs()) }

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
        unsafe { Self::from_m256(_mm256_min_ps(self.to_m256(), rhs.to_m256())) }
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
        unsafe { Self::from_m256(_mm256_max_ps(self.to_m256(), rhs.to_m256())) }
    }

    #[inline(always)]
    pub fn clamp(self, lo: Self, hi: Self) -> Self { self.max(lo).min(hi) }

    #[inline] pub fn min_element(self) -> f32 { self.lo.min_element().min(self.hi.min_element()) }
    #[inline] pub fn max_element(self) -> f32 { self.lo.max_element().max(self.hi.max_element()) }

    #[inline(always)]
    pub fn mul_add(self, b: Self, c: Self) -> Self {
        Self::from_halves(self.lo.mul_add(b.lo, c.lo), self.hi.mul_add(b.hi, c.hi))
    }

    #[inline(always)]
    pub fn blend(mask: Mask8, if_true: Self, if_false: Self) -> Self {
        Self::from_halves(
            f32x4::blend(mask.lo, if_true.lo, if_false.lo),
            f32x4::blend(mask.hi, if_true.hi, if_false.hi),
        )
    }

    #[inline(always)] pub fn cmpeq(self, rhs: Self) -> Mask8 { Mask8::from_halves(self.lo.cmpeq(rhs.lo), self.hi.cmpeq(rhs.hi)) }
    #[inline(always)] pub fn cmpne(self, rhs: Self) -> Mask8 { Mask8::from_halves(self.lo.cmpne(rhs.lo), self.hi.cmpne(rhs.hi)) }
    #[inline(always)] pub fn cmplt(self, rhs: Self) -> Mask8 { Mask8::from_halves(self.lo.cmplt(rhs.lo), self.hi.cmplt(rhs.hi)) }
    #[inline(always)] pub fn cmple(self, rhs: Self) -> Mask8 { Mask8::from_halves(self.lo.cmple(rhs.lo), self.hi.cmple(rhs.hi)) }
    #[inline(always)] pub fn cmpgt(self, rhs: Self) -> Mask8 { Mask8::from_halves(self.lo.cmpgt(rhs.lo), self.hi.cmpgt(rhs.hi)) }
    #[inline(always)] pub fn cmpge(self, rhs: Self) -> Mask8 { Mask8::from_halves(self.lo.cmpge(rhs.lo), self.hi.cmpge(rhs.hi)) }

    #[inline] pub fn is_finite(self) -> bool { self.lo.is_finite() && self.hi.is_finite() }
    #[inline] pub fn is_nan(self) -> bool { self.lo.is_nan() || self.hi.is_nan() }
}

impl Add for f32x8 {
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
impl f32x8 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx2")]
    unsafe fn add_avx2(self, r: Self) -> Self {
        unsafe { Self::from_m256(_mm256_add_ps(self.to_m256(), r.to_m256())) }
    }
}
impl AddAssign for f32x8 { #[inline(always)] fn add_assign(&mut self, r: Self) { *self = *self + r; } }

impl Sub for f32x8 {
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
impl f32x8 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx2")]
    unsafe fn sub_avx2(self, r: Self) -> Self {
        unsafe { Self::from_m256(_mm256_sub_ps(self.to_m256(), r.to_m256())) }
    }
}
impl SubAssign for f32x8 { #[inline(always)] fn sub_assign(&mut self, r: Self) { *self = *self - r; } }

impl Mul for f32x8 {
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
impl f32x8 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx2")]
    unsafe fn mul_avx2(self, r: Self) -> Self {
        unsafe { Self::from_m256(_mm256_mul_ps(self.to_m256(), r.to_m256())) }
    }
}
impl MulAssign for f32x8 { #[inline(always)] fn mul_assign(&mut self, r: Self) { *self = *self * r; } }

impl Div for f32x8 {
    type Output = Self;
    #[inline]
    fn div(self, r: Self) -> Self {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if crate::wide::avx2_available() {
            return unsafe { self.div_avx2(r) };
        }
        Self::from_halves(self.lo / r.lo, self.hi / r.hi)
    }
}
impl f32x8 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx2")]
    unsafe fn div_avx2(self, r: Self) -> Self {
        unsafe { Self::from_m256(_mm256_div_ps(self.to_m256(), r.to_m256())) }
    }
}
impl DivAssign for f32x8 { #[inline(always)] fn div_assign(&mut self, r: Self) { *self = *self / r; } }

impl Neg for f32x8 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self { Self::from_halves(-self.lo, -self.hi) }
}

impl fmt::Debug for f32x8 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let a = self.to_array();
        write!(f, "f32x8({},{},{},{},{},{},{},{})", a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7])
    }
}
impl fmt::Display for f32x8 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let a = self.to_array();
        write!(f, "[{},{},{},{},{},{},{},{}]", a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7])
    }
}
