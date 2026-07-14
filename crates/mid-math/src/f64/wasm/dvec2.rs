// crates/mid-math/src/f64/wasm/dvec2.rs
//! DVec2 backed by `v128` (f64x2) on wasm32/wasm64 with simd128.
//!
//! Lane layout: lane 0 = x (bytes 0-7), lane 1 = y (bytes 8-15).
//! Memory is byte-identical to XY<f64> {x, y} — impl_dvec2_deref! is zero-cost.
//!
//! All arithmetic uses f64x2_* intrinsics. Dot uses the lane-swap horizontal
//! add pattern (see wasm.rs:dot2d_in_x). No FMA in SIMD128 baseline.

#[cfg(target_arch = "wasm32")]
use core::arch::wasm32::*;
#[cfg(target_arch = "wasm64")]
use core::arch::wasm64::*;

use core::fmt;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use crate::wasm::{dot2d, dot2d_in_x, dot2d_into_v128, v128_from_f64x2};
use crate::impl_dvec2_deref;
use crate::DEPSILON;

// ── Union for compile-time constant initialisation ────────────────────────────
// v128 has no const constructor. Both [f64;2] and v128 are 16 bytes on WASM.
#[repr(C)]
union UnionCast { f: [f64; 2], v: DVec2 }

// ── Type ──────────────────────────────────────────────────────────────────────

/// 2D double-precision vector. 16 bytes, 16-byte aligned. Backed by `v128` (f64x2).
///
/// **C interop:** use `CDVec2` at the FFI boundary.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct DVec2(pub(crate) v128);

// Deref to XY<f64>: v128 f64x2 layout [lane0=x@0, lane1=y@8] ≡ XY<f64>{x@0,y@8}.
impl_dvec2_deref!(DVec2);

// ── Constants ─────────────────────────────────────────────────────────────────

impl DVec2 {
    pub const ZERO:  Self = unsafe { UnionCast { f: [ 0.0,  0.0] }.v };
    pub const ONE:   Self = unsafe { UnionCast { f: [ 1.0,  1.0] }.v };
    pub const X:     Self = unsafe { UnionCast { f: [ 1.0,  0.0] }.v };
    pub const Y:     Self = unsafe { UnionCast { f: [ 0.0,  1.0] }.v };
    pub const NEG_X: Self = unsafe { UnionCast { f: [-1.0,  0.0] }.v };
    pub const NEG_Y: Self = unsafe { UnionCast { f: [ 0.0, -1.0] }.v };

    // ── Constructors ─────────────────────────────────────────────────────────

    #[inline(always)]
    pub fn new(x: f64, y: f64) -> Self {
        unsafe { UnionCast { f: [x, y] }.v }
    }

    #[inline(always)]
    pub fn splat(v: f64) -> Self { Self(f64x2_splat(v)) }

    #[inline(always)] pub fn from_array(a: [f64; 2]) -> Self { Self::new(a[0], a[1]) }
    #[inline(always)] pub fn to_array(self) -> [f64; 2] { [self.x, self.y] }

    /// Extend to DVec3 by appending `z` (scalar).
    #[inline(always)]
    pub fn extend(self, z: f64) -> crate::DVec3 {
        crate::DVec3::new(self.x, self.y, z)
    }

    // ── Arithmetic ────────────────────────────────────────────────────────────

    /// 2-lane dot product using lane-swap horizontal add.
    #[inline(always)]
    pub fn dot(self, rhs: Self) -> f64 {
        unsafe { dot2d(self.0, rhs.0) }
    }

    #[inline(always)] pub fn length_sq(self) -> f64 { self.dot(self) }

    /// Euclidean length via f64x2_sqrt on the dot result.
    #[inline]
    pub fn length(self) -> f64 {
        unsafe {
            let d = dot2d_into_v128(self.0, self.0); // [dot, dot]
            f64x2_extract_lane::<0>(f64x2_sqrt(d))
        }
    }

    #[inline]
    pub fn length_recip(self) -> f64 {
        let l = self.length();
        if l < DEPSILON { 0.0 } else { 1.0 / l }
    }

    /// Normalize. Returns ZERO for near-zero-length vectors.
    ///
    /// Guard: `f64x2_gt(len, eps)` → v128 mask; `v128_and(norm, mask)` zeroes
    /// degenerate lanes without branching.
    #[inline]
    pub fn normalize(self) -> Self {
        unsafe {
            let len_v = f64x2_sqrt(dot2d_into_v128(self.0, self.0));
            let norm  = Self(f64x2_div(self.0, len_v));
            let ok    = f64x2_gt(len_v, f64x2_splat(DEPSILON));
            Self(v128_and(norm.0, ok))
        }
    }

    #[inline]
    pub fn try_normalize(self) -> Option<Self> {
        let rcp = self.length_recip();
        if rcp > 0.0 && rcp.is_finite() { Some(self * rcp) } else { None }
    }

    #[inline] pub fn normalize_or(self, fb: Self) -> Self { self.try_normalize().unwrap_or(fb) }
    #[inline] pub fn normalize_or_zero(self) -> Self { self.normalize_or(Self::ZERO) }
    #[inline] pub fn is_normalized(self) -> bool { (self.length_sq() - 1.0).abs() <= 2e-10 }

    // ── Interpolation / geometry ──────────────────────────────────────────────

    /// Linear interpolation: `self + (rhs - self) * t`.
    /// No FMA in WASM SIMD128 baseline — two instructions per component.
    #[inline]
    pub fn lerp(self, rhs: Self, t: f64) -> Self {
        let tt   = f64x2_splat(t);
        let diff = f64x2_sub(rhs.0, self.0);
        Self(f64x2_add(self.0, f64x2_mul(diff, tt)))
    }

    #[inline] pub fn distance(self, rhs: Self)    -> f64 { (self - rhs).length() }
    #[inline] pub fn distance_sq(self, rhs: Self) -> f64 { (self - rhs).length_sq() }

    /// Perpendicular vector (90° counter-clockwise): `(-y, x)`.
    #[inline] pub fn perp(self) -> Self { Self::new(-self.y, self.x) }

    /// 2D cross / perp-dot: `x·rhs.y - y·rhs.x`.
    #[inline]
    pub fn perp_dot(self, rhs: Self) -> f64 { self.x * rhs.y - self.y * rhs.x }

    /// Signed angle from `self` to `rhs` in `[-π, +π]`.
    #[inline]
    pub fn angle_to(self, rhs: Self) -> f64 {
        self.perp_dot(rhs).atan2(self.dot(rhs))
    }

    #[inline] pub fn to_angle(self) -> f64 { self.y.atan2(self.x) }

    #[inline]
    pub fn from_angle(angle: f64) -> Self {
        let (s, c) = angle.sin_cos();
        Self::new(c, s)
    }

    // ── Component-wise ────────────────────────────────────────────────────────

    /// f64x2_min — propagates NaN from either operand.
    #[inline] pub fn min(self, rhs: Self) -> Self { Self(f64x2_min(self.0, rhs.0)) }
    /// f64x2_max — propagates NaN from either operand.
    #[inline] pub fn max(self, rhs: Self) -> Self { Self(f64x2_max(self.0, rhs.0)) }
    #[inline] pub fn clamp(self, lo: Self, hi: Self) -> Self { self.max(lo).min(hi) }
    /// Direct f64x2_abs — no sign-mask trick needed unlike SSE2.
    #[inline] pub fn abs(self) -> Self { Self(f64x2_abs(self.0)) }

    // floor/ceil/round: no f64x2 rounding in WASM SIMD128 baseline; scalar fallback.
    #[inline] pub fn floor(self) -> Self { Self::new(self.x.floor(), self.y.floor()) }
    #[inline] pub fn ceil(self)  -> Self { Self::new(self.x.ceil(),  self.y.ceil())  }
    #[inline] pub fn round(self) -> Self { Self::new(self.x.round(), self.y.round()) }

    // ── Predicates ────────────────────────────────────────────────────────────

    #[inline] pub fn is_finite(self) -> bool { self.x.is_finite() && self.y.is_finite() }
    #[inline] pub fn is_nan(self)    -> bool { self.x.is_nan()    || self.y.is_nan() }

    #[inline]
    pub fn approx_eq(self, rhs: Self) -> bool {
        (self.x - rhs.x).abs() < DEPSILON && (self.y - rhs.y).abs() < DEPSILON
    }

    /// Lossy cast to single-precision `Vec2`.
    #[inline]
    pub fn as_vec2(self) -> crate::Vec2 {
        crate::Vec2::new(self.x as f32, self.y as f32)
    }
}

// ── Operators ─────────────────────────────────────────────────────────────────

impl Add for DVec2 {
    type Output = Self;
    #[inline(always)] fn add(self, r: Self) -> Self { Self(f64x2_add(self.0, r.0)) }
}
impl Sub for DVec2 {
    type Output = Self;
    #[inline(always)] fn sub(self, r: Self) -> Self { Self(f64x2_sub(self.0, r.0)) }
}
impl Mul for DVec2 {
    type Output = Self;
    #[inline(always)] fn mul(self, r: Self) -> Self { Self(f64x2_mul(self.0, r.0)) }
}
impl Div for DVec2 {
    type Output = Self;
    #[inline(always)] fn div(self, r: Self) -> Self { Self(f64x2_div(self.0, r.0)) }
}
impl Mul<f64> for DVec2 {
    type Output = Self;
    #[inline(always)] fn mul(self, s: f64) -> Self { Self(f64x2_mul(self.0, f64x2_splat(s))) }
}
impl Mul<DVec2> for f64 {
    type Output = DVec2;
    #[inline(always)] fn mul(self, v: DVec2) -> DVec2 { DVec2(f64x2_mul(f64x2_splat(self), v.0)) }
}
impl Div<f64> for DVec2 {
    type Output = Self;
    #[inline(always)] fn div(self, s: f64) -> Self { Self(f64x2_div(self.0, f64x2_splat(s))) }
}
/// Direct f64x2_neg — no XOR trick unlike SSE2.
impl Neg for DVec2 {
    type Output = Self;
    #[inline(always)] fn neg(self) -> Self { Self(f64x2_neg(self.0)) }
}

impl AddAssign for DVec2 {
    #[inline(always)] fn add_assign(&mut self, r: Self) { self.0 = f64x2_add(self.0, r.0); }
}
impl SubAssign for DVec2 {
    #[inline(always)] fn sub_assign(&mut self, r: Self) { self.0 = f64x2_sub(self.0, r.0); }
}
impl MulAssign<f64> for DVec2 {
    #[inline(always)] fn mul_assign(&mut self, s: f64) { self.0 = f64x2_mul(self.0, f64x2_splat(s)); }
}
impl DivAssign<f64> for DVec2 {
    #[inline(always)] fn div_assign(&mut self, s: f64) { self.0 = f64x2_div(self.0, f64x2_splat(s)); }
}

// ── PartialEq ─────────────────────────────────────────────────────────────────

impl PartialEq for DVec2 {
    #[inline]
    fn eq(&self, rhs: &Self) -> bool {
        // f64x2_eq → all-1s per lane if equal.
        // u64x2_bitmask extracts MSB of each 64-bit lane → u16 with 2 bits.
        (u64x2_bitmask(f64x2_eq(self.0, rhs.0)) & 0b11) == 0b11
    }
}

impl Default for DVec2 { fn default() -> Self { Self::ZERO } }

// ── Display / Debug ───────────────────────────────────────────────────────────

impl fmt::Debug for DVec2 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("DVec2").field(&self.x).field(&self.y).finish()
    }
}
impl fmt::Display for DVec2 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {})", self.x, self.y)
    }
}

// ── Conversions ───────────────────────────────────────────────────────────────

impl From<[f64; 2]> for DVec2 { #[inline] fn from(a: [f64; 2]) -> Self { Self::new(a[0], a[1]) } }
impl From<DVec2> for [f64; 2] { #[inline] fn from(v: DVec2) -> Self { [v.x, v.y] } }
impl From<(f64, f64)> for DVec2 { #[inline] fn from(t: (f64, f64)) -> Self { Self::new(t.0, t.1) } }
impl From<DVec2> for (f64, f64) { #[inline] fn from(v: DVec2) -> Self { (v.x, v.y) } }
