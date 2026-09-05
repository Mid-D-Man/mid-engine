// crates/mid-math/src/wide/float/avx2/vec3x8.rs
//! 8 x Vec3 packed in SoA layout.
//!
//! Same storage/dispatch design as `wide/int/avx2/i32x8.rs` — see that
//! file's doc comment for the full reasoning. Portable `{lo, hi}: Vec3x4`
//! storage, never a raw `__m256` field outside a
//! `#[target_feature(enable = "avx2")]` scope.
//!
//! `dot`/`length_sq`/`length` return `f32x8` and `lerp` takes `t: f32x8`,
//! `select` takes `mask: Mask8` — matching `Vec3x4`'s own convention
//! (`f32x4`/`Mask4`, never a raw `__m128`/`__m128`-as-mask) exactly. The
//! original AVX2-only version of this file returned/took raw `__m256`
//! directly in these four methods, which cannot be preserved: a signature
//! exposing the raw register type cannot be called safely from ordinary
//! code either, for the same reason the register cannot live in a
//! struct field outside a guarded scope. One real call site in this
//! crate's own benches (`vs_wide_float.rs`'s `lerp/vec3x8` case) does
//! pass a raw `__m256` today and needs updating to build `f32x8` instead
//! — flagged, not fixed as part of this file, since the bench file's own
//! AVX2 dispatch wrappers need the same "always compiled, runtime
//! checked" update this file just got, which is its own separate pass.
//!
//! `normalize`/`normalize_precise`'s fast-rsqrt math is unchanged in
//! substance — still rsqrt + one Newton-Raphson step — just expressed via
//! `f32x8`'s own portable methods instead of raw intrinsics recomputed
//! here a second time.

#![allow(non_camel_case_types)]

use core::fmt;
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use crate::f32::sse2::vec3::Vec3;
use crate::wide::float::sse2::vec3x4::Vec3x4;
use super::f32x8::f32x8;
use super::mask8::Mask8;

/// 8 x Vec3 in SoA layout. Two `Vec3x4` halves.
#[derive(Clone, Copy)]
pub struct Vec3x8 {
    lo: Vec3x4,
    hi: Vec3x4,
}

impl Vec3x8 {
    pub const ZERO: Self = Self { lo: Vec3x4::ZERO, hi: Vec3x4::ZERO };

    #[inline(always)]
    pub(crate) fn from_halves(lo: Vec3x4, hi: Vec3x4) -> Self { Self { lo, hi } }

    /// Build from 8 individual Vec3s.
    #[inline]
    pub fn from_vec3s(a: Vec3, b: Vec3, c: Vec3, d: Vec3, e: Vec3, f: Vec3, g: Vec3, h: Vec3) -> Self {
        Self::from_halves(Vec3x4::from_vec3s(a, b, c, d), Vec3x4::from_vec3s(e, f, g, h))
    }

    #[inline(always)]
    pub fn from_slice(s: &[Vec3; 8]) -> Self {
        Self::from_vec3s(s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7])
    }

    #[inline(always)]
    pub fn splat(v: Vec3) -> Self { Self::from_halves(Vec3x4::splat(v), Vec3x4::splat(v)) }

    #[inline]
    pub fn to_array(self) -> [Vec3; 8] {
        let lo = self.lo.to_array();
        let hi = self.hi.to_array();
        [lo[0], lo[1], lo[2], lo[3], hi[0], hi[1], hi[2], hi[3]]
    }

    #[inline(always)]
    pub fn write_to_slice(self, s: &mut [Vec3; 8]) {
        let a = self.to_array();
        s.copy_from_slice(&a);
    }

    #[inline]
    pub fn get(self, lane: usize) -> Vec3 {
        assert!(lane < 8, "Vec3x8::get — lane {lane} out of bounds (max 7)");
        if lane < 4 { self.lo.get(lane) } else { self.hi.get(lane - 4) }
    }

    // ── AVX2 pack/unpack helpers — only place a __m256 exists ──
    // (Vec3x8 has 3 __m256-shaped fields conceptually — x/y/z — so this
    // packs/unpacks each of Vec3x4's own x/y/z __m128 fields separately.)

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx2")]
    unsafe fn to_m256_xyz(self) -> (__m256, __m256, __m256) {
        unsafe {
            (
                _mm256_set_m128(self.hi.x, self.lo.x),
                _mm256_set_m128(self.hi.y, self.lo.y),
                _mm256_set_m128(self.hi.z, self.lo.z),
            )
        }
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx2")]
    unsafe fn from_m256_xyz(x: __m256, y: __m256, z: __m256) -> Self {
        unsafe {
            Self::from_halves(
                Vec3x4 { x: _mm256_castps256_ps128(x), y: _mm256_castps256_ps128(y), z: _mm256_castps256_ps128(z) },
                Vec3x4 { x: _mm256_extractf128_ps::<1>(x), y: _mm256_extractf128_ps::<1>(y), z: _mm256_extractf128_ps::<1>(z) },
            )
        }
    }

    #[inline(always)]
    pub fn mul_elem(self, rhs: Self) -> Self {
        Self::from_halves(self.lo.mul_elem(rhs.lo), self.hi.mul_elem(rhs.hi))
    }

    #[inline(always)]
    pub fn scale(self, s: f32x8) -> Self {
        let (slo, shi) = s.halves();
        Self::from_halves(self.lo.scale(slo), self.hi.scale(shi))
    }

    #[inline(always)]
    pub fn scale_uniform(self, s: f32) -> Self {
        Self::from_halves(self.lo.scale_uniform(s), self.hi.scale_uniform(s))
    }

    #[inline(always)]
    pub fn madd(self, b: Self, c: Self) -> Self {
        Self::from_halves(self.lo.madd(b.lo, c.lo), self.hi.madd(b.hi, c.hi))
    }

    /// 8 independent dot products. AVX2 native fast path (3 muls + 2 adds
    /// on real 256-bit registers instead of two width-4 passes).
    #[inline]
    pub fn dot(self, rhs: Self) -> f32x8 {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if crate::wide::avx2_available() {
            return unsafe { self.dot_avx2(rhs) };
        }
        f32x8::from_halves(self.lo.dot(rhs.lo), self.hi.dot(rhs.hi))
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx2")]
    unsafe fn dot_avx2(self, rhs: Self) -> f32x8 {
        unsafe {
            let (ax, ay, az) = self.to_m256_xyz();
            let (bx, by, bz) = rhs.to_m256_xyz();
            let xx = _mm256_mul_ps(ax, bx);
            let yy = _mm256_mul_ps(ay, by);
            let zz = _mm256_mul_ps(az, bz);
            let sum = _mm256_add_ps(_mm256_add_ps(xx, yy), zz);
            f32x8::from_halves(
                crate::wide::float::sse2::f32x4::f32x4(_mm256_castps256_ps128(sum)),
                crate::wide::float::sse2::f32x4::f32x4(_mm256_extractf128_ps::<1>(sum)),
            )
        }
    }

    #[inline]
    pub fn cross(self, rhs: Self) -> Self {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if crate::wide::avx2_available() {
            return unsafe { self.cross_avx2(rhs) };
        }
        Self::from_halves(self.lo.cross(rhs.lo), self.hi.cross(rhs.hi))
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx2")]
    unsafe fn cross_avx2(self, rhs: Self) -> Self {
        unsafe {
            let (ax, ay, az) = self.to_m256_xyz();
            let (bx, by, bz) = rhs.to_m256_xyz();
            let x = _mm256_sub_ps(_mm256_mul_ps(ay, bz), _mm256_mul_ps(az, by));
            let y = _mm256_sub_ps(_mm256_mul_ps(az, bx), _mm256_mul_ps(ax, bz));
            let z = _mm256_sub_ps(_mm256_mul_ps(ax, by), _mm256_mul_ps(ay, bx));
            Self::from_m256_xyz(x, y, z)
        }
    }

    #[inline(always)]
    pub fn length_sq(self) -> f32x8 { self.dot(self) }
    #[inline(always)]
    pub fn length(self) -> f32x8 { self.length_sq().sqrt() }

    #[inline(always)]
    pub fn normalize(self) -> Self {
        Self::from_halves(self.lo.normalize(), self.hi.normalize())
    }
    #[inline(always)]
    pub fn normalize_precise(self) -> Self {
        Self::from_halves(self.lo.normalize_precise(), self.hi.normalize_precise())
    }

    #[inline(always)]
    pub fn lerp(self, rhs: Self, t: f32x8) -> Self {
        let (tlo, thi) = t.halves();
        Self::from_halves(self.lo.lerp(rhs.lo, tlo), self.hi.lerp(rhs.hi, thi))
    }
    #[inline(always)]
    pub fn lerp_uniform(self, rhs: Self, t: f32) -> Self { self.lerp(rhs, f32x8::splat(t)) }

    #[inline(always)]
    pub fn min(self, rhs: Self) -> Self { Self::from_halves(self.lo.min(rhs.lo), self.hi.min(rhs.hi)) }
    #[inline(always)]
    pub fn max(self, rhs: Self) -> Self { Self::from_halves(self.lo.max(rhs.lo), self.hi.max(rhs.hi)) }

    #[inline(always)]
    pub fn select(mask: Mask8, if_true: Self, if_false: Self) -> Self {
        Self::from_halves(
            Vec3x4::select(mask.lo, if_true.lo, if_false.lo),
            Vec3x4::select(mask.hi, if_true.hi, if_false.hi),
        )
    }

    #[inline(always)]
    pub fn length_lt(self, rhs: Self) -> Mask8 {
        Mask8::from_halves(self.lo.length_lt(rhs.lo), self.hi.length_lt(rhs.hi))
    }

    #[inline]
    pub fn is_finite(self) -> bool { self.lo.is_finite() && self.hi.is_finite() }
}

impl Add for Vec3x8 {
    type Output = Self;
    #[inline(always)]
    fn add(self, r: Self) -> Self { Self::from_halves(self.lo + r.lo, self.hi + r.hi) }
}
impl AddAssign for Vec3x8 { #[inline(always)] fn add_assign(&mut self, r: Self) { *self = *self + r; } }

impl Sub for Vec3x8 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, r: Self) -> Self { Self::from_halves(self.lo - r.lo, self.hi - r.hi) }
}
impl SubAssign for Vec3x8 { #[inline(always)] fn sub_assign(&mut self, r: Self) { *self = *self - r; } }

impl Neg for Vec3x8 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self { Self::from_halves(-self.lo, -self.hi) }
}

impl Mul for Vec3x8 { type Output = Self; #[inline(always)] fn mul(self, r: Self) -> Self { self.mul_elem(r) } }
impl MulAssign for Vec3x8 { #[inline(always)] fn mul_assign(&mut self, r: Self) { *self = *self * r; } }
impl Mul<f32> for Vec3x8 { type Output = Self; #[inline(always)] fn mul(self, s: f32) -> Self { self.scale_uniform(s) } }

impl PartialEq for Vec3x8 {
    #[inline]
    fn eq(&self, r: &Self) -> bool { self.lo == r.lo && self.hi == r.hi }
}

impl Default for Vec3x8 { #[inline(always)] fn default() -> Self { Self::ZERO } }

impl fmt::Debug for Vec3x8 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Vec3x8({:?})", self.to_array())
    }
}
impl fmt::Display for Vec3x8 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.to_array())
    }
}
impl From<[Vec3; 8]> for Vec3x8 { #[inline(always)] fn from(a: [Vec3; 8]) -> Self { Self::from_slice(&a) } }
impl From<Vec3x8> for [Vec3; 8] { #[inline(always)] fn from(v: Vec3x8) -> Self { v.to_array() } }
