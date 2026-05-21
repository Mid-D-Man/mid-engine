// crates/mid-math/src/f64/neon/dvec4.rs
//! DVec4 backed by 2× `float64x2_t` on aarch64.
//!
//! Storage layout (32 bytes, 32-byte aligned):
//!   lo: float64x2_t → [x (bytes 0-7),  y (bytes 8-15)]
//!   hi: float64x2_t → [z (bytes 16-23), w (bytes 24-31)]
//!
//! Byte-identical to XYZW<f64> {x@0, y@8, z@16, w@24}. Deref is a zero-cost cast.
//!
//! AArch64 advantages for DVec4 vs SSE2:
//!   vaddvq_f64  — one-instruction horizontal add per pair (SSE2 needs shuffles)
//!   vfmaq_f64   — mandatory FMA for lerp (SSE2 needs explicit target_feature)
//!   vnegq_f64   — direct negate (SSE2 needs XOR -0.0)
//!   vabsq_f64   — direct abs (SSE2 needs ANDNOT)

use core::fmt;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use core::arch::aarch64::*;

use crate::neon::{dot4d_neon, dot4d_neon_into_f64x2};
use crate::impl_dvec4_deref;
use crate::DEPSILON;

// ── Union for compile-time constant initialisation ────────────────────────────
// DVec4 {lo=[x,y], hi=[z,w]} has same byte layout as [f64; 4] = [x, y, z, w].
#[repr(C)]
union UnionCast { f: [f64; 4], v: DVec4 }

// ── Type ──────────────────────────────────────────────────────────────────────

/// 4D double-precision vector. 32 bytes, 32-byte aligned.
/// Backed by 2× `float64x2_t` on aarch64: `lo=[x,y]`, `hi=[z,w]`.
#[derive(Clone, Copy)]
#[repr(C, align(32))]
pub struct DVec4 {
    pub(crate) lo: float64x2_t, // [x, y]
    pub(crate) hi: float64x2_t, // [z, w]
}

// Layout {lo@0: x@0,y@8; hi@16: z@16,w@24} ≡ XYZW<f64> {x@0,y@8,z@16,w@24}.
impl_dvec4_deref!(DVec4);

// ── Constants ─────────────────────────────────────────────────────────────────

impl DVec4 {
    pub const ZERO: Self = unsafe { UnionCast { f: [0.0; 4] }.v };
    pub const ONE:  Self = unsafe { UnionCast { f: [1.0; 4] }.v };
    pub const X:    Self = unsafe { UnionCast { f: [1.0, 0.0, 0.0, 0.0] }.v };
    pub const Y:    Self = unsafe { UnionCast { f: [0.0, 1.0, 0.0, 0.0] }.v };
    pub const Z:    Self = unsafe { UnionCast { f: [0.0, 0.0, 1.0, 0.0] }.v };
    pub const W:    Self = unsafe { UnionCast { f: [0.0, 0.0, 0.0, 1.0] }.v };

    // ── Constructors ─────────────────────────────────────────────────────────

    #[inline(always)]
    pub fn new(x: f64, y: f64, z: f64, w: f64) -> Self {
        unsafe { UnionCast { f: [x, y, z, w] }.v }
    }

    #[inline(always)]
    pub fn splat(v: f64) -> Self {
        let r = unsafe { vdupq_n_f64(v) };
        Self { lo: r, hi: r }
    }

    #[inline(always)] pub fn from_array(a: [f64; 4]) -> Self { Self::new(a[0], a[1], a[2], a[3]) }
    #[inline(always)] pub fn to_array(self) -> [f64; 4] { [self.x, self.y, self.z, self.w] }

    #[inline(always)]
    pub fn truncate(self) -> crate::DVec3 {
        crate::DVec3::new(self.x, self.y, self.z)
    }

    // ── Arithmetic ────────────────────────────────────────────────────────────

    /// 4-lane dot: two `vaddvq_f64` calls (one per pair), then scalar add.
    #[inline(always)]
    pub fn dot(self, rhs: Self) -> f64 {
        unsafe { dot4d_neon(self.lo, self.hi, rhs.lo, rhs.hi) }
    }

    #[inline(always)] pub fn length_sq(self) -> f64 { self.dot(self) }

    #[inline]
    pub fn length(self) -> f64 { self.dot(self).sqrt() }

    #[inline]
    pub fn length_recip(self) -> f64 {
        let l = self.length();
        if l < DEPSILON { 0.0 } else { 1.0 / l }
    }

    /// Normalize. Guard mask applied to both lo and hi via uint64 AND.
    #[inline]
    pub fn normalize(self) -> Self {
        unsafe {
            let len_v  = dot4d_neon_into_f64x2(self.lo, self.hi, self.lo, self.hi);
            let sqrt_v = vsqrtq_f64(len_v);
            let lo_n   = vdivq_f64(self.lo, sqrt_v);
            let hi_n   = vdivq_f64(self.hi, sqrt_v);
            let ok     = vcgtq_f64(sqrt_v, vdupq_n_f64(DEPSILON));
            Self {
                lo: vreinterpretq_f64_u64(vandq_u64(vreinterpretq_u64_f64(lo_n), ok)),
                hi: vreinterpretq_f64_u64(vandq_u64(vreinterpretq_u64_f64(hi_n), ok)),
            }
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

    // ── Interpolation ─────────────────────────────────────────────────────────

    /// Lerp using mandatory AArch64 FMA: `vfmaq_f64(base, diff, t)`.
    #[inline]
    pub fn lerp(self, rhs: Self, t: f64) -> Self {
        unsafe {
            let tt      = vdupq_n_f64(t);
            let lo_diff = vsubq_f64(rhs.lo, self.lo);
            let hi_diff = vsubq_f64(rhs.hi, self.hi);
            Self {
                lo: vfmaq_f64(self.lo, lo_diff, tt),
                hi: vfmaq_f64(self.hi, hi_diff, tt),
            }
        }
    }

    #[inline] pub fn distance(self, rhs: Self)    -> f64 { (self - rhs).length() }
    #[inline] pub fn distance_sq(self, rhs: Self) -> f64 { (self - rhs).length_sq() }

    // ── Component-wise ────────────────────────────────────────────────────────

    #[inline]
    pub fn min(self, r: Self) -> Self {
        Self { lo: unsafe { vminq_f64(self.lo, r.lo) }, hi: unsafe { vminq_f64(self.hi, r.hi) } }
    }
    #[inline]
    pub fn max(self, r: Self) -> Self {
        Self { lo: unsafe { vmaxq_f64(self.lo, r.lo) }, hi: unsafe { vmaxq_f64(self.hi, r.hi) } }
    }
    #[inline] pub fn clamp(self, lo: Self, hi: Self) -> Self { self.max(lo).min(hi) }
    #[inline]
    pub fn abs(self) -> Self {
        Self { lo: unsafe { vabsq_f64(self.lo) }, hi: unsafe { vabsq_f64(self.hi) } }
    }

    // ── Predicates ────────────────────────────────────────────────────────────

    #[inline]
    pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite()
            && self.z.is_finite() && self.w.is_finite()
    }
    #[inline]
    pub fn is_nan(self) -> bool {
        self.x.is_nan() || self.y.is_nan() || self.z.is_nan() || self.w.is_nan()
    }

    #[inline]
    pub fn approx_eq(self, rhs: Self) -> bool {
        unsafe {
            let lo_diff = vabsq_f64(vsubq_f64(self.lo, rhs.lo));
            let hi_diff = vabsq_f64(vsubq_f64(self.hi, rhs.hi));
            let eps     = vdupq_n_f64(DEPSILON);
            let lo_ok   = vcltq_f64(lo_diff, eps); // uint64x2_t all-1s where ok
            let hi_ok   = vcltq_f64(hi_diff, eps);
            vgetq_lane_u64::<0>(lo_ok) != 0 && vgetq_lane_u64::<1>(lo_ok) != 0
                && vgetq_lane_u64::<0>(hi_ok) != 0 && vgetq_lane_u64::<1>(hi_ok) != 0
        }
    }

    #[inline]
    pub fn as_vec4(self) -> crate::Vec4 {
        crate::Vec4::new(self.x as f32, self.y as f32, self.z as f32, self.w as f32)
    }
}

// ── Operators ─────────────────────────────────────────────────────────────────

impl Add for DVec4 {
    type Output = Self;
    #[inline(always)]
    fn add(self, r: Self) -> Self {
        Self { lo: unsafe { vaddq_f64(self.lo, r.lo) }, hi: unsafe { vaddq_f64(self.hi, r.hi) } }
    }
}
impl Sub for DVec4 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, r: Self) -> Self {
        Self { lo: unsafe { vsubq_f64(self.lo, r.lo) }, hi: unsafe { vsubq_f64(self.hi, r.hi) } }
    }
}
impl Mul for DVec4 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, r: Self) -> Self {
        Self { lo: unsafe { vmulq_f64(self.lo, r.lo) }, hi: unsafe { vmulq_f64(self.hi, r.hi) } }
    }
}
impl Div for DVec4 {
    type Output = Self;
    #[inline(always)]
    fn div(self, r: Self) -> Self {
        Self { lo: unsafe { vdivq_f64(self.lo, r.lo) }, hi: unsafe { vdivq_f64(self.hi, r.hi) } }
    }
}
impl Mul<f64> for DVec4 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, s: f64) -> Self {
        Self { lo: unsafe { vmulq_n_f64(self.lo, s) }, hi: unsafe { vmulq_n_f64(self.hi, s) } }
    }
}
impl Mul<DVec4> for f64 {
    type Output = DVec4;
    #[inline(always)] fn mul(self, v: DVec4) -> DVec4 { v * self }
}
impl Div<f64> for DVec4 {
    type Output = Self;
    #[inline(always)]
    fn div(self, s: f64) -> Self {
        unsafe {
            let sv = vdupq_n_f64(s);
            Self { lo: vdivq_f64(self.lo, sv), hi: vdivq_f64(self.hi, sv) }
        }
    }
}
impl Neg for DVec4 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self { lo: unsafe { vnegq_f64(self.lo) }, hi: unsafe { vnegq_f64(self.hi) } }
    }
}

impl AddAssign for DVec4 {
    #[inline(always)]
    fn add_assign(&mut self, r: Self) {
        self.lo = unsafe { vaddq_f64(self.lo, r.lo) };
        self.hi = unsafe { vaddq_f64(self.hi, r.hi) };
    }
}
impl SubAssign for DVec4 {
    #[inline(always)]
    fn sub_assign(&mut self, r: Self) {
        self.lo = unsafe { vsubq_f64(self.lo, r.lo) };
        self.hi = unsafe { vsubq_f64(self.hi, r.hi) };
    }
}
impl MulAssign<f64> for DVec4 {
    #[inline(always)]
    fn mul_assign(&mut self, s: f64) {
        self.lo = unsafe { vmulq_n_f64(self.lo, s) };
        self.hi = unsafe { vmulq_n_f64(self.hi, s) };
    }
}
impl DivAssign<f64> for DVec4 {
    #[inline(always)]
    fn div_assign(&mut self, s: f64) {
        unsafe {
            let sv = vdupq_n_f64(s);
            self.lo = vdivq_f64(self.lo, sv);
            self.hi = vdivq_f64(self.hi, sv);
        }
    }
}

impl PartialEq for DVec4 {
    #[inline]
    fn eq(&self, rhs: &Self) -> bool {
        unsafe {
            let lo_cmp = vceqq_f64(self.lo, rhs.lo);
            let hi_cmp = vceqq_f64(self.hi, rhs.hi);
            vgetq_lane_u64::<0>(lo_cmp) != 0 && vgetq_lane_u64::<1>(lo_cmp) != 0
                && vgetq_lane_u64::<0>(hi_cmp) != 0 && vgetq_lane_u64::<1>(hi_cmp) != 0
        }
    }
}

impl Default for DVec4 { fn default() -> Self { Self::ZERO } }

impl fmt::Debug for DVec4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("DVec4")
            .field(&self.x).field(&self.y).field(&self.z).field(&self.w).finish()
    }
}
impl fmt::Display for DVec4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {}, {}, {})", self.x, self.y, self.z, self.w)
    }
}

impl From<[f64; 4]> for DVec4 { #[inline] fn from(a: [f64; 4]) -> Self { Self::new(a[0],a[1],a[2],a[3]) } }
impl From<DVec4> for [f64; 4] { #[inline] fn from(v: DVec4) -> Self { [v.x,v.y,v.z,v.w] } }
impl From<(f64,f64,f64,f64)> for DVec4 { #[inline] fn from(t: (f64,f64,f64,f64)) -> Self { Self::new(t.0,t.1,t.2,t.3) } }
impl From<DVec4> for (f64,f64,f64,f64) { #[inline] fn from(v: DVec4) -> Self { (v.x,v.y,v.z,v.w) } }
