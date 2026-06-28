// crates/mid-math/src/f32/neon/vec3.rs
//! Vec3 backed by `float32x4_t` on aarch64.

use core::arch::aarch64::*;
use core::fmt;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use crate::f32::neon::vec4::Vec4;
use crate::f32::vec2::Vec2;
use crate::impl_vec3_deref;
use crate::EPSILON;

#[repr(C)]
union UnionCast { f: [f32; 4], v: Vec3 }

/// 3-dimensional vector. 16 bytes, 16-byte aligned. Backed by `float32x4_t`.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct Vec3(pub(crate) float32x4_t);

impl_vec3_deref!(Vec3);

impl Vec3 {
    pub const ZERO:  Self = unsafe { UnionCast { f: [ 0.0,  0.0,  0.0, 0.0] }.v };
    pub const ONE:   Self = unsafe { UnionCast { f: [ 1.0,  1.0,  1.0, 0.0] }.v };
    pub const X:     Self = unsafe { UnionCast { f: [ 1.0,  0.0,  0.0, 0.0] }.v };
    pub const Y:     Self = unsafe { UnionCast { f: [ 0.0,  1.0,  0.0, 0.0] }.v };
    pub const Z:     Self = unsafe { UnionCast { f: [ 0.0,  0.0,  1.0, 0.0] }.v };
    pub const NEG_X: Self = unsafe { UnionCast { f: [-1.0,  0.0,  0.0, 0.0] }.v };
    pub const NEG_Y: Self = unsafe { UnionCast { f: [ 0.0, -1.0,  0.0, 0.0] }.v };
    pub const NEG_Z: Self = unsafe { UnionCast { f: [ 0.0,  0.0, -1.0, 0.0] }.v };
    pub const NAN:   Self = unsafe { UnionCast { f: [f32::NAN, f32::NAN, f32::NAN, f32::NAN] }.v };

    #[inline(always)]
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        unsafe { UnionCast { f: [x, y, z, 0.0] }.v }
    }

    #[inline(always)] pub fn splat(v: f32) -> Self { Self::new(v, v, v) }
    #[inline(always)] pub fn from_array(a: [f32; 3]) -> Self { Self::new(a[0], a[1], a[2]) }
    #[inline(always)] pub fn to_array(self) -> [f32; 3] { [self.x, self.y, self.z] }

    #[inline(always)]
    pub fn extend(self, w: f32) -> Vec4 { Vec4::new(self.x, self.y, self.z, w) }

    #[inline(always)]
    pub fn truncate(self) -> Vec2 { Vec2::new(self.x, self.y) }

    /// NEON vectorized 3D dot product.
    ///
    /// `vmulq_f32` multiplies all 4 lanes in parallel; `vaddvq_f32` (NEON ADDV)
    /// horizontally sums them. Vec3's w lane is always 0, so the sum equals
    /// x·rx + y·ry + z·rz + 0 — correct without any masking.
    ///
    /// Replaces the previous scalar path (6 lane-extract `vgetq_lane_f32` +
    /// 3 fmul + 2 fadd), which stalled the pipeline crossing the SIMD/GP boundary.
    #[inline(always)]
    pub fn dot(self, rhs: Self) -> f32 {
        unsafe { vaddvq_f32(vmulq_f32(self.0, rhs.0)) }
    }

    #[inline]
    pub fn dot_into_vec(self, rhs: Self) -> Self { Self::splat(self.dot(rhs)) }

    /// Vectorized cross product (no scalar lane extraction).
    ///
    /// `cross(a,b) = (a.yzx * b.zxy) - (a.zxy * b.yzx)`, built via `vextq_f32`
    /// rotates + single-lane `vsetq_lane_f32` swaps, same algorithm as glam's
    /// NEON `Vec3A::cross`. Previous version read `.x/.y/.z` (6 lane extracts
    /// + 3 scalar muls + 3 scalar subs) — this is 2 vector muls + 1 vmlsq_f32
    /// fused multiply-subtract, no scalar detour.
    ///
    /// Lane 3 of the result is explicitly zeroed to preserve the "w is always
    /// 0" invariant every other Vec3 ctor/op maintains (the shuffle leaves
    /// it as don't-care garbage otherwise).
    #[inline(always)]
    pub fn cross(self, rhs: Self) -> Self {
        unsafe {
            let lhs = self.0;
            let rhs = rhs.0;

            let lhs_yzwx = vextq_f32::<1>(lhs, lhs);
            let rhs_wxyz = vextq_f32::<3>(rhs, rhs);
            let lhs_yzx  = vsetq_lane_f32::<2>(vgetq_lane_f32::<0>(lhs), lhs_yzwx);
            let rhs_zxy  = vsetq_lane_f32::<0>(vgetq_lane_f32::<2>(rhs), rhs_wxyz);
            let part_a   = vmulq_f32(lhs_yzx, rhs_zxy);

            let lhs_wxyz = vextq_f32::<3>(lhs, lhs);
            let rhs_yzwx = vextq_f32::<1>(rhs, rhs);
            let lhs_zxy  = vsetq_lane_f32::<0>(vgetq_lane_f32::<2>(lhs), lhs_wxyz);
            let rhs_yzx  = vsetq_lane_f32::<2>(vgetq_lane_f32::<0>(rhs), rhs_yzwx);

            let result = vmlsq_f32(part_a, lhs_zxy, rhs_yzx); // part_a - lhs_zxy*rhs_yzx
            Self(vsetq_lane_f32::<3>(0.0, result))
        }
    }

    #[inline(always)] pub fn length_sq(self) -> f32 { self.dot(self) }

    #[inline]
    pub fn length(self) -> f32 { self.length_sq().sqrt() }

    #[inline]
    pub fn length_recip(self) -> f32 {
        let l = self.length();
        if l < EPSILON { 0.0 } else { 1.0 / l }
    }

    /// Normalize to unit length.
    ///
    /// **Undefined (NaN in practice) for zero-length input.**
    /// Use [`Self::normalize_or_zero()`] for safe behaviour.
    ///
    /// NEON-OPT (Build 22): two fixes vs the previous version:
    ///
    /// 1. Squared-length now uses `vmulq_f32` + `vaddvq_f32` (NEON ADDV.4S)
    ///    to stay fully in SIMD registers. The old path called `self.dot(self)`
    ///    which was scalar (6 lane-extract + 5 scalar ops + `vdupq_n_f32`),
    ///    paying a SIMD→GP→SIMD round-trip penalty every call.
    ///
    /// 2. The `vcgtq_f32` + `vandq_u32` zero-guard is removed, matching
    ///    glam's `normalize()` contract (equivalent of SSE2 OPT-3 in Build 7).
    ///    Safe callers use `normalize_or_zero()`.
    #[inline]
    pub fn normalize(self) -> Self {
        unsafe {
            // w lane is 0 → vaddvq gives x²+y²+z²+0 = correct ‖self‖²
            let dot = vdupq_n_f32(vaddvq_f32(vmulq_f32(self.0, self.0)));
            let inv = crate::neon::rsqrt_nr_f32(dot);
            Self(vmulq_f32(self.0, inv))
        }
    }

    #[inline]
    pub fn try_normalize(self) -> Option<Self> {
        let rcp = self.length_recip();
        if rcp > 0.0 && rcp.is_finite() { Some(self * rcp) } else { None }
    }

    #[inline] pub fn normalize_or(self, fallback: Self) -> Self { self.try_normalize().unwrap_or(fallback) }
    #[inline] pub fn normalize_or_zero(self) -> Self { self.normalize_or(Self::ZERO) }
    #[inline] pub fn is_normalized(self) -> bool { (self.length_sq() - 1.0).abs() <= 2e-4 }

    #[inline]
    pub fn lerp(self, rhs: Self, t: f32) -> Self {
        unsafe {
            let diff = vsubq_f32(rhs.0, self.0);
            let t_v  = vdupq_n_f32(t);
            Self(vfmaq_f32(self.0, diff, t_v))
        }
    }

    #[inline] pub fn reflect(self, n: Self) -> Self { self - n * (2.0 * self.dot(n)) }
    #[inline] pub fn distance(self, rhs: Self)    -> f32 { (self - rhs).length() }
    #[inline] pub fn distance_sq(self, rhs: Self) -> f32 { (self - rhs).length_sq() }

    #[inline] pub fn min(self, rhs: Self) -> Self { Self(unsafe { vminq_f32(self.0, rhs.0) }) }
    #[inline] pub fn max(self, rhs: Self) -> Self { Self(unsafe { vmaxq_f32(self.0, rhs.0) }) }
    #[inline] pub fn clamp(self, lo: Self, hi: Self) -> Self { self.max(lo).min(hi) }
    #[inline] pub fn abs(self) -> Self { Self(unsafe { vabsq_f32(self.0) }) }

    #[inline] pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite()
    }
    #[inline] pub fn is_nan(self) -> bool {
        self.x.is_nan() || self.y.is_nan() || self.z.is_nan()
    }
    #[inline] pub fn approx_eq(self, rhs: Self) -> bool {
        (self - rhs).abs().length_sq() < EPSILON * EPSILON
    }
    #[inline] pub fn abs_diff_eq(self, rhs: Self, max_abs_diff: f32) -> bool {
        (self.x - rhs.x).abs() < max_abs_diff
            && (self.y - rhs.y).abs() < max_abs_diff
            && (self.z - rhs.z).abs() < max_abs_diff
    }
    #[inline] pub fn approx_eq_eps(self, rhs: Self, eps: f32) -> bool {
        (self.x - rhs.x).abs() < eps
            && (self.y - rhs.y).abs() < eps
            && (self.z - rhs.z).abs() < eps
    }

    // ── p1: angle between ─────────────────────────────────────────────────────

    #[inline]
    pub fn angle_between(self, rhs: Self) -> f32 {
        let denom = (self.length_sq() * rhs.length_sq()).sqrt();
        if denom < EPSILON { 0.0 } else { (self.dot(rhs) / denom).clamp(-1.0, 1.0).acos() }
    }

    // ── p2: project / reject ──────────────────────────────────────────────────

    #[inline]
    pub fn project_onto(self, rhs: Self) -> Self {
        let d = rhs.length_sq();
        if d < EPSILON { Self::ZERO } else { rhs * (self.dot(rhs) / d) }
    }

    #[inline]
    pub fn reject_from(self, rhs: Self) -> Self { self - self.project_onto(rhs) }

    // ── p7: movement / clamping helpers ──────────────────────────────────────

    #[inline]
    pub fn move_towards(self, target: Self, max_dist: f32) -> Self {
        let d = target - self;
        let len = d.length();
        if len <= max_dist || len < EPSILON { target } else { self + d / len * max_dist }
    }

    #[inline]
    pub fn clamp_length(self, min: f32, max: f32) -> Self {
        let len = self.length();
        if len < EPSILON { return Self::ZERO; }
        let clamped = len.clamp(min, max);
        if (clamped - len).abs() < EPSILON { self } else { self * (clamped / len) }
    }

    #[inline]
    pub fn clamp_length_max(self, max: f32) -> Self {
        let len = self.length();
        if len > max && len > EPSILON { self * (max / len) } else { self }
    }

    #[inline]
    pub fn clamp_length_min(self, min: f32) -> Self {
        let len = self.length();
        if len < min && len > EPSILON { self * (min / len) } else { self }
    }

    #[inline] pub fn midpoint(self, rhs: Self) -> Self { (self + rhs) * 0.5 }

    #[inline]
    pub fn is_parallel(self, rhs: Self) -> bool {
        self.cross(rhs).length_sq() < EPSILON * EPSILON
    }

    #[inline]
    pub fn is_perpendicular(self, rhs: Self) -> bool { self.dot(rhs).abs() < EPSILON }

    // ── p6: spherical coordinates ─────────────────────────────────────────────

    #[inline]
    pub fn to_spherical(self) -> (f32, f32, f32) {
        let r = self.length();
        if r < EPSILON { return (0.0, 0.0, 0.0); }
        let theta = (self.z / r).clamp(-1.0, 1.0).acos();
        let phi   = self.y.atan2(self.x);
        (r, theta, phi)
    }

    #[inline]
    pub fn from_spherical(r: f32, theta: f32, phi: f32) -> Self {
        let sin_theta = theta.sin();
        Self::new(r * sin_theta * phi.cos(), r * sin_theta * phi.sin(), r * theta.cos())
    }
}

impl Add for Vec3 {
    type Output = Self;
    #[inline(always)] fn add(self, r: Self) -> Self { Self(unsafe { vaddq_f32(self.0, r.0) }) }
}
impl Sub for Vec3 {
    type Output = Self;
    #[inline(always)] fn sub(self, r: Self) -> Self { Self(unsafe { vsubq_f32(self.0, r.0) }) }
}
impl Mul<f32> for Vec3 {
    type Output = Self;
    #[inline(always)] fn mul(self, s: f32) -> Self { Self(unsafe { vmulq_n_f32(self.0, s) }) }
}
impl Mul<Vec3> for f32 {
    type Output = Vec3;
    #[inline(always)] fn mul(self, v: Vec3) -> Vec3 { Vec3(unsafe { vmulq_n_f32(v.0, self) }) }
}
impl Mul for Vec3 {
    type Output = Self;
    #[inline(always)] fn mul(self, r: Self) -> Self { Self(unsafe { vmulq_f32(self.0, r.0) }) }
}
impl Div<f32> for Vec3 {
    type Output = Self;
    #[inline(always)] fn div(self, s: f32) -> Self { Self(unsafe { vdivq_f32(self.0, vdupq_n_f32(s)) }) }
}
impl Neg for Vec3 {
    type Output = Self;
    #[inline(always)] fn neg(self) -> Self { Self(unsafe { vnegq_f32(self.0) }) }
}
impl AddAssign for Vec3 {
    #[inline(always)] fn add_assign(&mut self, r: Self) { self.0 = unsafe { vaddq_f32(self.0, r.0) }; }
}
impl SubAssign for Vec3 {
    #[inline(always)] fn sub_assign(&mut self, r: Self) { self.0 = unsafe { vsubq_f32(self.0, r.0) }; }
}
impl MulAssign<f32> for Vec3 {
    #[inline(always)] fn mul_assign(&mut self, s: f32) { self.0 = unsafe { vmulq_n_f32(self.0, s) }; }
}
impl DivAssign<f32> for Vec3 {
    #[inline(always)] fn div_assign(&mut self, s: f32) { self.0 = unsafe { vdivq_f32(self.0, vdupq_n_f32(s)) }; }
}

impl PartialEq for Vec3 {
    #[inline]
    fn eq(&self, rhs: &Self) -> bool {
        unsafe {
            let cmp = vceqq_f32(self.0, rhs.0);
            vgetq_lane_u32::<0>(cmp) != 0
                && vgetq_lane_u32::<1>(cmp) != 0
                && vgetq_lane_u32::<2>(cmp) != 0
        }
    }
}

impl Default for Vec3 { fn default() -> Self { Self::ZERO } }

impl fmt::Debug for Vec3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Vec3").field(&self.x).field(&self.y).field(&self.z).finish()
    }
}
impl fmt::Display for Vec3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {}, {})", self.x, self.y, self.z)
    }
}
impl From<[f32; 3]> for Vec3 { #[inline] fn from(a: [f32; 3]) -> Self { Self::new(a[0], a[1], a[2]) } }
impl From<Vec3> for [f32; 3] { #[inline] fn from(v: Vec3) -> Self { [v.x, v.y, v.z] } }
impl From<(f32, f32, f32)> for Vec3 { #[inline] fn from(t: (f32, f32, f32)) -> Self { Self::new(t.0, t.1, t.2) } }
impl From<Vec3> for (f32, f32, f32) { #[inline] fn from(v: Vec3) -> Self { (v.x, v.y, v.z) } }
