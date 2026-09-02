// crates/mid-math/src/f32/neon/vec4.rs
//! Vec4 backed by `float32x4_t` on aarch64.
//!
//! Key advantage over SSE2: `vaddvq_f32` is a single AArch64 ADDV.4S
//! instruction for 4-lane horizontal add — perfect for dot product.
//! FMA (`vfmaq_f32`) is mandatory on AArch64 (unlike optional FMA3 on x86).

use core::arch::aarch64::*;
use core::fmt;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use crate::f32::neon::vec3::Vec3;
use crate::f32::vec2::Vec2;
use crate::impl_vec4_deref;
use crate::EPSILON;

#[repr(C)]
union UnionCast { f: [f32; 4], v: Vec4 }

/// 4-dimensional vector. 16 bytes, 16-byte aligned. Backed by `float32x4_t`.
///
/// **C interop:** use [`CVec4`][crate::ffi::types::CVec4] at the FFI boundary.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct Vec4(pub(crate) float32x4_t);

impl_vec4_deref!(Vec4);

impl Vec4 {
    pub const ZERO: Self = unsafe { UnionCast { f: [0.0; 4] }.v };
    pub const ONE:  Self = unsafe { UnionCast { f: [1.0; 4] }.v };
    pub const X:    Self = unsafe { UnionCast { f: [1.0, 0.0, 0.0, 0.0] }.v };
    pub const Y:    Self = unsafe { UnionCast { f: [0.0, 1.0, 0.0, 0.0] }.v };
    pub const Z:    Self = unsafe { UnionCast { f: [0.0, 0.0, 1.0, 0.0] }.v };
    pub const W:    Self = unsafe { UnionCast { f: [0.0, 0.0, 0.0, 1.0] }.v };

    // ── Constructors ─────────────────────────────────────────────────────────

    #[inline(always)]
    pub fn new(x: f32, y: f32, z: f32, w: f32) -> Self {
        unsafe { UnionCast { f: [x, y, z, w] }.v }
    }

    #[inline(always)]
    pub fn splat(v: f32) -> Self { Self(unsafe { vdupq_n_f32(v) }) }

    #[inline(always)] pub fn from_array(a: [f32; 4]) -> Self { Self::new(a[0], a[1], a[2], a[3]) }
    #[inline(always)] pub fn to_array(self)  -> [f32; 4] { [self.x, self.y, self.z, self.w] }

    /// Drop w, return Vec3 (zeros lane 3).
    #[inline(always)]
    pub fn truncate(self) -> Vec3 {
        unsafe { Vec3(vsetq_lane_f32::<3>(0.0, self.0)) }
    }

    // ── Arithmetic ────────────────────────────────────────────────────────────

    /// 4-lane dot product via single AArch64 ADDV.4S instruction.
    #[inline(always)]
    pub fn dot(self, rhs: Self) -> f32 {
        unsafe { vaddvq_f32(vmulq_f32(self.0, rhs.0)) }
    }

    /// Broadcast dot to all lanes.
    #[inline]
    pub fn dot_into_vec(self, rhs: Self) -> Self {
        Self::splat(self.dot(rhs))
    }

    // ── Length / normalise ────────────────────────────────────────────────────

    #[inline(always)] pub fn length_sq(self) -> f32 { self.dot(self) }

    #[inline]
    pub fn length(self) -> f32 {
        unsafe { vaddvq_f32(vmulq_f32(self.0, self.0)).sqrt() }
    }

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
    /// NEON-OPT (Build 22): `vcgtq_f32` + `vandq_u32` zero-guard removed,
    /// matching glam's `normalize()` contract. Saves 3 NEON ops on every
    /// non-zero input. Safe path: `normalize_or_zero()`.
    #[inline]
    pub fn normalize(self) -> Self {
        unsafe {
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

    #[inline] pub fn normalize_or(self, fb: Self) -> Self { self.try_normalize().unwrap_or(fb) }
    #[inline] pub fn normalize_or_zero(self) -> Self { self.normalize_or(Self::ZERO) }
    #[inline] pub fn is_normalized(self) -> bool { (self.length_sq() - 1.0).abs() <= 2e-4 }

    // ── Interpolation ─────────────────────────────────────────────────────────

    /// Linear interpolation using FMA (mandatory on AArch64).
    #[inline]
    pub fn lerp(self, rhs: Self, t: f32) -> Self {
        unsafe {
            let t_v  = vdupq_n_f32(t);
            let diff = vsubq_f32(rhs.0, self.0);
            Self(vfmaq_f32(self.0, diff, t_v))
        }
    }

    // ── Component-wise ────────────────────────────────────────────────────────

    #[inline] pub fn min(self, r: Self) -> Self { Self(unsafe { vminq_f32(self.0, r.0) }) }
    #[inline] pub fn max(self, r: Self) -> Self { Self(unsafe { vmaxq_f32(self.0, r.0) }) }
    #[inline] pub fn clamp(self, lo: Self, hi: Self) -> Self { self.max(lo).min(hi) }
    #[inline] pub fn abs(self) -> Self { Self(unsafe { vabsq_f32(self.0) }) }

    // ── Predicates ────────────────────────────────────────────────────────────

    #[inline] pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite() && self.w.is_finite()
    }
    #[inline] pub fn is_nan(self) -> bool {
        self.x.is_nan() || self.y.is_nan() || self.z.is_nan() || self.w.is_nan()
    }
    #[inline]
    pub fn approx_eq(self, rhs: Self) -> bool {
        unsafe {
            let diff = vabsq_f32(vsubq_f32(self.0, rhs.0));
            let eps  = vdupq_n_f32(EPSILON);
            let ok   = vcltq_f32(diff, eps);
            vgetq_lane_u32::<0>(ok) != 0
                && vgetq_lane_u32::<1>(ok) != 0
                && vgetq_lane_u32::<2>(ok) != 0
                && vgetq_lane_u32::<3>(ok) != 0
        }
    }
}

// ── Operators ─────────────────────────────────────────────────────────────────

impl Add for Vec4 {
    type Output = Self;
    #[inline(always)] fn add(self, r: Self) -> Self { Self(unsafe { vaddq_f32(self.0, r.0) }) }
}
impl Sub for Vec4 {
    type Output = Self;
    #[inline(always)] fn sub(self, r: Self) -> Self { Self(unsafe { vsubq_f32(self.0, r.0) }) }
}
impl Mul<f32> for Vec4 {
    type Output = Self;
    #[inline(always)] fn mul(self, s: f32) -> Self { Self(unsafe { vmulq_n_f32(self.0, s) }) }
}
impl Mul<Vec4> for f32 {
    type Output = Vec4;
    #[inline(always)] fn mul(self, v: Vec4) -> Vec4 { Vec4(unsafe { vmulq_n_f32(v.0, self) }) }
}
impl Mul for Vec4 {
    type Output = Self;
    #[inline(always)] fn mul(self, r: Self) -> Self { Self(unsafe { vmulq_f32(self.0, r.0) }) }
}
impl Div<f32> for Vec4 {
    type Output = Self;
    #[inline(always)] fn div(self, s: f32) -> Self {
        Self(unsafe { vdivq_f32(self.0, vdupq_n_f32(s)) })
    }
}
impl Neg for Vec4 {
    type Output = Self;
    #[inline(always)] fn neg(self) -> Self { Self(unsafe { vnegq_f32(self.0) }) }
}

impl AddAssign for Vec4 {
    #[inline(always)] fn add_assign(&mut self, r: Self) { self.0 = unsafe { vaddq_f32(self.0, r.0) }; }
}
impl SubAssign for Vec4 {
    #[inline(always)] fn sub_assign(&mut self, r: Self) { self.0 = unsafe { vsubq_f32(self.0, r.0) }; }
}
impl MulAssign<f32> for Vec4 {
    #[inline(always)] fn mul_assign(&mut self, s: f32) { self.0 = unsafe { vmulq_n_f32(self.0, s) }; }
}
impl DivAssign<f32> for Vec4 {
    #[inline(always)] fn div_assign(&mut self, s: f32) {
        self.0 = unsafe { vdivq_f32(self.0, vdupq_n_f32(s)) };
    }
}

impl PartialEq for Vec4 {
    #[inline]
    fn eq(&self, rhs: &Self) -> bool {
        unsafe {
            let cmp = vceqq_f32(self.0, rhs.0);
            vgetq_lane_u32::<0>(cmp) != 0
                && vgetq_lane_u32::<1>(cmp) != 0
                && vgetq_lane_u32::<2>(cmp) != 0
                && vgetq_lane_u32::<3>(cmp) != 0
        }
    }
}

impl Default for Vec4 { fn default() -> Self { Self::ZERO } }

impl fmt::Debug for Vec4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Vec4")
            .field(&self.x).field(&self.y).field(&self.z).field(&self.w)
            .finish()
    }
}
impl fmt::Display for Vec4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {}, {}, {})", self.x, self.y, self.z, self.w)
    }
}

impl From<[f32; 4]> for Vec4 {
    #[inline] fn from(a: [f32; 4]) -> Self { Self::new(a[0], a[1], a[2], a[3]) }
}
impl From<Vec4> for [f32; 4] {
    #[inline] fn from(v: Vec4) -> Self { [v.x, v.y, v.z, v.w] }
}
impl From<(f32, f32, f32, f32)> for Vec4 {
    #[inline] fn from(t: (f32, f32, f32, f32)) -> Self { Self::new(t.0, t.1, t.2, t.3) }
}
impl From<Vec4> for (f32, f32, f32, f32) {
    #[inline] fn from(v: Vec4) -> Self { (v.x, v.y, v.z, v.w) }
    }

