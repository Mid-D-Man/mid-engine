// crates/mid-math/src/f64/neon/dvec2.rs
//! DVec2 backed by `float64x2_t` on aarch64.
//!
//! Lane layout: lane 0 = x (bytes 0-7), lane 1 = y (bytes 8-15).
//! Memory is byte-identical to XY<f64> {x, y} — Deref is a zero-cost bitcast.
//!
//! Key NEON f64 advantages over SSE2:
//!   vaddvq_f64  — single-instruction horizontal add (no shuffle trick needed)
//!   vfmaq_f64   — mandatory FMA on AArch64 (no target_feature gate required)
//!   vabsq_f64   — direct abs, no sign-mask ANDNOT needed
//!   vnegq_f64   — direct neg, no XOR -0.0 needed
//!   vcgtq_f64   — comparison to uint64x2_t mask for normalize guard

use core::fmt;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use core::arch::aarch64::*;

use crate::neon::{dot2d_neon, dot2d_neon_into_f64x2};
use crate::impl_dvec2_deref;
use crate::DEPSILON;

// ── Union for compile-time constant initialisation ────────────────────────────

#[repr(C)]
union UnionCast { f: [f64; 2], v: DVec2 }

// ── Type ──────────────────────────────────────────────────────────────────────

/// 2D double-precision vector. 16 bytes, 16-byte aligned.
/// Backed by `float64x2_t` on aarch64.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct DVec2(pub(crate) float64x2_t);

// Deref to XY<f64>: float64x2_t [lane0=x@0, lane1=y@8] ≡ XY<f64> {x@0, y@8}.
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
    pub fn splat(v: f64) -> Self { Self(unsafe { vdupq_n_f64(v) }) }

    #[inline(always)] pub fn from_array(a: [f64; 2]) -> Self { Self::new(a[0], a[1]) }
    #[inline(always)] pub fn to_array(self) -> [f64; 2] { [self.x, self.y] }

    #[inline(always)]
    pub fn extend(self, z: f64) -> crate::DVec3 {
        crate::DVec3::new(self.x, self.y, z)
    }

    // ── Arithmetic ────────────────────────────────────────────────────────────

    /// Dot product via `vaddvq_f64(vmulq_f64(...))` — single ADDP instruction.
    #[inline(always)]
    pub fn dot(self, rhs: Self) -> f64 { unsafe { dot2d_neon(self.0, rhs.0) } }

    #[inline(always)] pub fn length_sq(self) -> f64 { self.dot(self) }

    #[inline]
    pub fn length(self) -> f64 {
        // vsqrtq_f64 then vgetq_lane_f64 is common on AArch64.
        unsafe {
            let d = vdupq_n_f64(dot2d_neon(self.0, self.0));
            vgetq_lane_f64::<0>(vsqrtq_f64(d))
        }
    }

    #[inline]
    pub fn length_recip(self) -> f64 {
        let l = self.length();
        if l < DEPSILON { 0.0 } else { 1.0 / l }
    }

    /// Normalize. Guard: zero lanes where length ≤ DEPSILON via uint64 AND mask.
    #[inline]
    pub fn normalize(self) -> Self {
        unsafe {
            let len_v  = dot2d_neon_into_f64x2(self.0, self.0); // [dot, dot]
            let sqrt_v = vsqrtq_f64(len_v);
            let norm   = Self(vdivq_f64(self.0, sqrt_v));
            let ok     = vcgtq_f64(sqrt_v, vdupq_n_f64(DEPSILON)); // uint64x2_t
            // AND float bits with mask: zeroes lanes where len <= DEPSILON
            Self(vreinterpretq_f64_u64(vandq_u64(
                vreinterpretq_u64_f64(norm.0),
                ok,
            )))
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

    /// Linear interpolation using mandatory AArch64 FMA: `self + (rhs-self)*t`.
    #[inline]
    pub fn lerp(self, rhs: Self, t: f64) -> Self {
        unsafe {
            let tt   = vdupq_n_f64(t);
            let diff = vsubq_f64(rhs.0, self.0);
            // vfmaq_f64(a, b, c) = a + b*c
            Self(vfmaq_f64(self.0, diff, tt))
        }
    }

    #[inline] pub fn distance(self, rhs: Self)    -> f64 { (self - rhs).length() }
    #[inline] pub fn distance_sq(self, rhs: Self) -> f64 { (self - rhs).length_sq() }

    #[inline] pub fn perp(self) -> Self { Self::new(-self.y, self.x) }

    #[inline]
    pub fn perp_dot(self, rhs: Self) -> f64 { self.x * rhs.y - self.y * rhs.x }

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

    #[inline] pub fn min(self, r: Self) -> Self { Self(unsafe { vminq_f64(self.0, r.0) }) }
    #[inline] pub fn max(self, r: Self) -> Self { Self(unsafe { vmaxq_f64(self.0, r.0) }) }
    #[inline] pub fn clamp(self, lo: Self, hi: Self) -> Self { self.max(lo).min(hi) }
    /// Direct `vabsq_f64` — no sign-mask trick needed unlike SSE2.
    #[inline] pub fn abs(self) -> Self { Self(unsafe { vabsq_f64(self.0) }) }

    // floor/ceil/round: no native NEON f64 rounding before ARMv8.1 (frint*),
    // scalar fallback is safe and negligible cost for these rare ops.
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

    #[inline]
    pub fn as_vec2(self) -> crate::Vec2 {
        crate::Vec2::new(self.x as f32, self.y as f32)
    }
}

// ── Operators ─────────────────────────────────────────────────────────────────

impl Add for DVec2 {
    type Output = Self;
    #[inline(always)] fn add(self, r: Self) -> Self { Self(unsafe { vaddq_f64(self.0, r.0) }) }
}
impl Sub for DVec2 {
    type Output = Self;
    #[inline(always)] fn sub(self, r: Self) -> Self { Self(unsafe { vsubq_f64(self.0, r.0) }) }
}
impl Mul<f64> for DVec2 {
    type Output = Self;
    #[inline(always)] fn mul(self, s: f64) -> Self { Self(unsafe { vmulq_n_f64(self.0, s) }) }
}
impl Mul<DVec2> for f64 {
    type Output = DVec2;
    #[inline(always)] fn mul(self, v: DVec2) -> DVec2 { DVec2(unsafe { vmulq_n_f64(v.0, self) }) }
}
impl Mul for DVec2 {
    type Output = Self;
    #[inline(always)] fn mul(self, r: Self) -> Self { Self(unsafe { vmulq_f64(self.0, r.0) }) }
}
impl Div<f64> for DVec2 {
    type Output = Self;
    #[inline(always)] fn div(self, s: f64) -> Self { Self(unsafe { vdivq_f64(self.0, vdupq_n_f64(s)) }) }
}
impl Div for DVec2 {
    type Output = Self;
    #[inline(always)] fn div(self, r: Self) -> Self { Self(unsafe { vdivq_f64(self.0, r.0) }) }
}
/// Direct `vnegq_f64` — no XOR trick unlike SSE2.
impl Neg for DVec2 {
    type Output = Self;
    #[inline(always)] fn neg(self) -> Self { Self(unsafe { vnegq_f64(self.0) }) }
}

impl AddAssign for DVec2 {
    #[inline(always)] fn add_assign(&mut self, r: Self) { self.0 = unsafe { vaddq_f64(self.0, r.0) }; }
}
impl SubAssign for DVec2 {
    #[inline(always)] fn sub_assign(&mut self, r: Self) { self.0 = unsafe { vsubq_f64(self.0, r.0) }; }
}
impl MulAssign<f64> for DVec2 {
    #[inline(always)] fn mul_assign(&mut self, s: f64) { self.0 = unsafe { vmulq_n_f64(self.0, s) }; }
}
impl DivAssign<f64> for DVec2 {
    #[inline(always)] fn div_assign(&mut self, s: f64) { self.0 = unsafe { vdivq_f64(self.0, vdupq_n_f64(s)) }; }
}

impl PartialEq for DVec2 {
    #[inline]
    fn eq(&self, rhs: &Self) -> bool {
        unsafe {
            let cmp = vceqq_f64(self.0, rhs.0);
            vgetq_lane_u64::<0>(cmp) != 0 && vgetq_lane_u64::<1>(cmp) != 0
        }
    }
}

impl Default for DVec2 { fn default() -> Self { Self::ZERO } }

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

impl From<[f64; 2]> for DVec2 { #[inline] fn from(a: [f64; 2]) -> Self { Self::new(a[0], a[1]) } }
impl From<DVec2> for [f64; 2] { #[inline] fn from(v: DVec2) -> Self { [v.x, v.y] } }
impl From<(f64, f64)> for DVec2 { #[inline] fn from(t: (f64, f64)) -> Self { Self::new(t.0, t.1) } }
impl From<DVec2> for (f64, f64) { #[inline] fn from(v: DVec2) -> Self { (v.x, v.y) } }
