// crates/mid-math/src/f64/sse2/dvec2.rs
//! DVec2 backed by `__m128d` on x86 / x86_64.
//!
//! `__m128d` holds 2× f64: lane 0 = x (lower 64 bits), lane 1 = y (upper 64 bits).
//! The memory layout is identical to `{x: f64, y: f64}` — the Deref to XY<f64>
//! is a zero-cost bitcast.
//!
//! Key SSE2 double-precision ops used:
//!   _mm_add_pd, _mm_sub_pd, _mm_mul_pd, _mm_div_pd  — basic arithmetic
//!   _mm_sqrt_pd                                       — square root
//!   _mm_min_pd, _mm_max_pd                           — component min/max
//!   _mm_cmpgt_pd, _mm_and_pd                         — masked normalize guard
//!   _mm_shuffle_pd                                    — lane swap for dot product
//!   _mm_andnot_pd                                     — abs (sign bit clear)
//!
//! No SSE4.1 required (floor/ceil/round are scalar fallbacks; the scalar
//! cost is negligible since floor/ceil are rare hot-path operations).

use core::fmt;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use crate::sse2::{dot2d, dot2d_in_x, dot2d_into_m128d, m128d_abs};
use crate::impl_dvec2_deref;
use crate::DEPSILON;

// ── Union for compile-time constant initialisation ────────────────────────────
// `_mm_set_pd` is not const. We transmute [f64; 2] ↔ __m128d. Safe because
// both types are 16-byte aligned and 16 bytes in size on all x86 targets.
#[repr(C)]
union UnionCast {
    f: [f64; 2],
    v: DVec2,
}

// ── Type ──────────────────────────────────────────────────────────────────────

/// 2D double-precision vector. 16 bytes, 16-byte aligned.
/// Backed by `__m128d` on x86 / x86_64.
///
/// **C interop:** use `CDVec2` at the FFI boundary.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct DVec2(pub(crate) __m128d);

// Provides `.x` and `.y` via Deref to XY<f64>. The memory layout of
// __m128d [lane0=x, lane1=y] is byte-identical to XY<f64> {x, y}.
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

    /// Create from components.
    ///
    /// `_mm_set_pd(e1, e0)`: lane 0 = e0 = x, lane 1 = e1 = y.
    #[inline(always)]
    pub fn new(x: f64, y: f64) -> Self {
        Self(unsafe { _mm_set_pd(y, x) })
    }

    #[inline(always)]
    pub fn splat(v: f64) -> Self { Self(unsafe { _mm_set1_pd(v) }) }

    #[inline(always)] pub fn from_array(a: [f64; 2]) -> Self { Self::new(a[0], a[1]) }
    #[inline(always)] pub fn to_array(self) -> [f64; 2] { [self.x, self.y] }

    /// Extend to DVec3 by appending `z` (always scalar).
    #[inline(always)]
    pub fn extend(self, z: f64) -> crate::DVec3 {
        crate::DVec3::new(self.x, self.y, z)
    }

    // ── Arithmetic ────────────────────────────────────────────────────────────

    /// 2-lane dot product. Both components multiplied and summed.
    #[inline(always)]
    pub fn dot(self, rhs: Self) -> f64 {
        unsafe { dot2d(self.0, rhs.0) }
    }

    #[inline(always)] pub fn length_sq(self) -> f64 { self.dot(self) }

    /// Euclidean length. Uses SSE2 `_mm_sqrt_pd` on the dot result.
    #[inline]
    pub fn length(self) -> f64 {
        unsafe { _mm_cvtsd_f64(_mm_sqrt_pd(dot2d_in_x(self.0, self.0))) }
    }

    #[inline]
    pub fn length_recip(self) -> f64 {
        let l = self.length();
        if l < DEPSILON { 0.0 } else { 1.0 / l }
    }

    /// Normalize to unit length. Returns ZERO for near-zero-length vectors.
    ///
    /// Guard: `_mm_cmpgt_pd(len, DEPSILON)` → mask zeroes the result when
    /// the length is too small, avoiding division by zero without branching.
    #[inline]
    pub fn normalize(self) -> Self {
        unsafe {
            let len  = _mm_sqrt_pd(dot2d_into_m128d(self.0, self.0));
            let norm = Self(_mm_div_pd(self.0, len));
            let ok   = _mm_cmpgt_pd(len, _mm_set1_pd(DEPSILON));
            Self(_mm_and_pd(norm.0, ok))
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
    #[inline]
    pub fn lerp(self, rhs: Self, t: f64) -> Self {
        unsafe {
            let tt   = _mm_set1_pd(t);
            let diff = _mm_sub_pd(rhs.0, self.0);
            Self(_mm_add_pd(self.0, _mm_mul_pd(diff, tt)))
        }
    }

    #[inline] pub fn distance(self, rhs: Self)    -> f64 { (self - rhs).length() }
    #[inline] pub fn distance_sq(self, rhs: Self) -> f64 { (self - rhs).length_sq() }

    /// Perpendicular vector (90° counter-clockwise): `(-y, x)`.
    #[inline] pub fn perp(self) -> Self { Self::new(-self.y, self.x) }

    /// 2D cross product (perp-dot / wedge): `x*rhs.y - y*rhs.x`.
    #[inline]
    pub fn perp_dot(self, rhs: Self) -> f64 {
        self.x * rhs.y - self.y * rhs.x
    }

    /// Signed angle from `self` to `rhs` in radians, range `[-π, +π]`.
    #[inline]
    pub fn angle_to(self, rhs: Self) -> f64 {
        self.perp_dot(rhs).atan2(self.dot(rhs))
    }

    /// Angle of this vector: `y.atan2(x)`.
    #[inline] pub fn to_angle(self) -> f64 { self.y.atan2(self.x) }

    /// Unit vector from angle: `(cos(angle), sin(angle))`.
    #[inline]
    pub fn from_angle(angle: f64) -> Self {
        let (s, c) = angle.sin_cos();
        Self::new(c, s)
    }

    // ── Component-wise ────────────────────────────────────────────────────────

    #[inline] pub fn min(self, rhs: Self) -> Self { Self(unsafe { _mm_min_pd(self.0, rhs.0) }) }
    #[inline] pub fn max(self, rhs: Self) -> Self { Self(unsafe { _mm_max_pd(self.0, rhs.0) }) }
    #[inline] pub fn clamp(self, lo: Self, hi: Self) -> Self { self.max(lo).min(hi) }
    #[inline] pub fn abs(self) -> Self { Self(unsafe { m128d_abs(self.0) }) }

    // SSE4.1 _mm_floor_pd/_mm_ceil_pd not assumed — scalar fallback is fine
    // since floor/ceil are rarely in hot paths.
    #[inline] pub fn floor(self) -> Self { Self::new(self.x.floor(), self.y.floor()) }
    #[inline] pub fn ceil(self)  -> Self { Self::new(self.x.ceil(),  self.y.ceil()) }
    #[inline] pub fn round(self) -> Self { Self::new(self.x.round(), self.y.round()) }

    // ── Predicates ────────────────────────────────────────────────────────────

    #[inline] pub fn is_finite(self) -> bool { self.x.is_finite() && self.y.is_finite() }
    #[inline] pub fn is_nan(self)    -> bool { self.x.is_nan()    || self.y.is_nan() }

    #[inline]
    pub fn approx_eq(self, rhs: Self) -> bool {
        (self.x - rhs.x).abs() < DEPSILON && (self.y - rhs.y).abs() < DEPSILON
    }

    // ── Cast ─────────────────────────────────────────────────────────────────

    /// Lossy cast to single-precision `Vec2`.
    #[inline]
    pub fn as_vec2(self) -> crate::Vec2 {
        crate::Vec2::new(self.x as f32, self.y as f32)
    }
}

// ── Operators ─────────────────────────────────────────────────────────────────

impl Add for DVec2 {
    type Output = Self;
    #[inline(always)] fn add(self, r: Self) -> Self { Self(unsafe { _mm_add_pd(self.0, r.0) }) }
}
impl Sub for DVec2 {
    type Output = Self;
    #[inline(always)] fn sub(self, r: Self) -> Self { Self(unsafe { _mm_sub_pd(self.0, r.0) }) }
}
impl Mul<f64> for DVec2 {
    type Output = Self;
    #[inline(always)] fn mul(self, s: f64) -> Self { Self(unsafe { _mm_mul_pd(self.0, _mm_set1_pd(s)) }) }
}
impl Mul<DVec2> for f64 {
    type Output = DVec2;
    #[inline(always)] fn mul(self, v: DVec2) -> DVec2 { DVec2(unsafe { _mm_mul_pd(_mm_set1_pd(self), v.0) }) }
}
impl Mul for DVec2 {
    type Output = Self;
    #[inline(always)] fn mul(self, r: Self) -> Self { Self(unsafe { _mm_mul_pd(self.0, r.0) }) }
}
impl Div<f64> for DVec2 {
    type Output = Self;
    #[inline(always)] fn div(self, s: f64) -> Self { Self(unsafe { _mm_div_pd(self.0, _mm_set1_pd(s)) }) }
}
impl Div for DVec2 {
    type Output = Self;
    #[inline(always)] fn div(self, r: Self) -> Self { Self(unsafe { _mm_div_pd(self.0, r.0) }) }
}
impl Neg for DVec2 {
    type Output = Self;
    #[inline(always)] fn neg(self) -> Self { Self(unsafe { _mm_xor_pd(self.0, _mm_set1_pd(-0.0)) }) }
}

impl AddAssign for DVec2 {
    #[inline(always)] fn add_assign(&mut self, r: Self) { self.0 = unsafe { _mm_add_pd(self.0, r.0) }; }
}
impl SubAssign for DVec2 {
    #[inline(always)] fn sub_assign(&mut self, r: Self) { self.0 = unsafe { _mm_sub_pd(self.0, r.0) }; }
}
impl MulAssign<f64> for DVec2 {
    #[inline(always)] fn mul_assign(&mut self, s: f64) { self.0 = unsafe { _mm_mul_pd(self.0, _mm_set1_pd(s)) }; }
}
impl DivAssign<f64> for DVec2 {
    #[inline(always)] fn div_assign(&mut self, s: f64) { self.0 = unsafe { _mm_div_pd(self.0, _mm_set1_pd(s)) }; }
}

// ── PartialEq ─────────────────────────────────────────────────────────────────

impl PartialEq for DVec2 {
    #[inline]
    fn eq(&self, rhs: &Self) -> bool {
        unsafe {
            // _mm_cmpeq_pd → all-1s per lane if equal. movmskpd extracts sign bits.
            (_mm_movemask_pd(_mm_cmpeq_pd(self.0, rhs.0)) & 0b11) == 0b11
        }
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
