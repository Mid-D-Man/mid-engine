// crates/mid-math/src/f32/wasm/vec4.rs
//! Vec4 backed by `v128` on wasm32/wasm64 with simd128 target feature.
//!
//! Key WASM advantages:
//!   f32x4_abs / f32x4_neg — direct instructions, no bit-mask tricks needed
//!   v128_andnot(a, b)     — a & ~b  (note: opposite argument order to SSE2 _mm_andnot_ps)
//!
//! Dot product uses the crate::wasm::dot4 horizontal-add helper.

#[cfg(target_arch = "wasm32")]
use core::arch::wasm32::*;
#[cfg(target_arch = "wasm64")]
use core::arch::wasm64::*;

use core::fmt;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use crate::f32::wasm::vec3::Vec3;
use crate::impl_vec4_deref;
use crate::wasm::{dot4, dot4_in_x, dot4_into_v128};
use crate::EPSILON;

#[repr(C)]
union UnionCast { f: [f32; 4], v: Vec4 }

/// 4-dimensional vector. 16 bytes, 16-byte aligned. Backed by `v128`.
///
/// **C interop:** use [`CVec4`][crate::ffi::types::CVec4] at the FFI boundary.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct Vec4(pub(crate) v128);

impl_vec4_deref!(Vec4);

impl Vec4 {
    pub const ZERO: Self = unsafe { UnionCast { f: [0.0; 4] }.v };
    pub const ONE:  Self = unsafe { UnionCast { f: [1.0; 4] }.v };
    pub const X:    Self = unsafe { UnionCast { f: [1.0, 0.0, 0.0, 0.0] }.v };
    pub const Y:    Self = unsafe { UnionCast { f: [0.0, 1.0, 0.0, 0.0] }.v };
    pub const Z:    Self = unsafe { UnionCast { f: [0.0, 0.0, 1.0, 0.0] }.v };
    pub const W:    Self = unsafe { UnionCast { f: [0.0, 0.0, 0.0, 1.0] }.v };

    // ── Constructors ──────────────────────────────────────────────────────────

    #[inline(always)]
    pub fn new(x: f32, y: f32, z: f32, w: f32) -> Self {
        unsafe { UnionCast { f: [x, y, z, w] }.v }
    }

    #[inline(always)]
    pub fn splat(v: f32) -> Self { Self(unsafe { f32x4_splat(v) }) }

    #[inline(always)] pub fn from_array(a: [f32; 4]) -> Self { Self::new(a[0], a[1], a[2], a[3]) }
    #[inline(always)] pub fn to_array(self) -> [f32; 4] { [self.x, self.y, self.z, self.w] }

    /// Drop w, zero lane 3, return Vec3.
    ///
    /// AND with [0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0x00000000].
    #[inline(always)]
    pub fn truncate(self) -> Vec3 {
        use crate::wasm::v128_from_f32x4;
        const MASK: v128 = v128_from_f32x4([
            f32::from_bits(0xFFFF_FFFF),
            f32::from_bits(0xFFFF_FFFF),
            f32::from_bits(0xFFFF_FFFF),
            0.0,
        ]);
        Vec3(unsafe { v128_and(self.0, MASK) })
    }

    // ── Arithmetic ────────────────────────────────────────────────────────────

    #[inline(always)]
    pub fn dot(self, rhs: Self) -> f32 { unsafe { dot4(self.0, rhs.0) } }

    #[inline]
    pub fn dot_into_vec(self, rhs: Self) -> Self {
        Self(unsafe { dot4_into_v128(self.0, rhs.0) })
    }

    // ── Length / normalise ────────────────────────────────────────────────────

    #[inline(always)] pub fn length_sq(self) -> f32 { self.dot(self) }

    #[inline]
    pub fn length(self) -> f32 {
        unsafe {
            let dot = dot4_in_x(self.0, self.0);
            f32x4_extract_lane::<0>(f32x4_sqrt(dot))
        }
    }

    #[inline]
    pub fn length_recip(self) -> f32 {
        let l = self.length();
        if l < EPSILON { 0.0 } else { 1.0 / l }
    }

    /// Normalise to unit length.  Returns `ZERO` for near-zero-length vectors.
    #[inline]
    pub fn normalize(self) -> Self {
        unsafe {
            let len  = f32x4_sqrt(dot4_into_v128(self.0, self.0));
            let n    = Self(f32x4_div(self.0, len));
            let ok   = f32x4_gt(len, f32x4_splat(EPSILON));
            Self(v128_and(n.0, ok))
        }
    }

    #[inline]
    pub fn try_normalize(self) -> Option<Self> {
        let rcp = self.length_recip();
        if rcp.is_finite() && rcp > 0.0 { Some(self * rcp) } else { None }
    }

    #[inline] pub fn normalize_or(self, fb: Self) -> Self { self.try_normalize().unwrap_or(fb) }
    #[inline] pub fn normalize_or_zero(self) -> Self { self.normalize_or(Self::ZERO) }
    #[inline] pub fn is_normalized(self) -> bool { (self.length_sq() - 1.0).abs() <= 2e-4 }

    // ── Interpolation ─────────────────────────────────────────────────────────

    /// Linear interpolation: `self + (rhs - self) * t`.
    #[inline]
    pub fn lerp(self, rhs: Self, t: f32) -> Self {
        unsafe {
            let tt   = f32x4_splat(t);
            let diff = f32x4_sub(rhs.0, self.0);
            Self(f32x4_add(self.0, f32x4_mul(diff, tt)))
        }
    }

    // ── Component-wise ────────────────────────────────────────────────────────

    #[inline] pub fn min(self, r: Self) -> Self { Self(unsafe { f32x4_pmin(self.0, r.0) }) }
    #[inline] pub fn max(self, r: Self) -> Self { Self(unsafe { f32x4_pmax(self.0, r.0) }) }
    #[inline] pub fn clamp(self, lo: Self, hi: Self) -> Self { self.max(lo).min(hi) }
    #[inline] pub fn abs(self) -> Self { Self(unsafe { f32x4_abs(self.0) }) }

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
            // f32x4_abs gives the absolute difference, compare < eps each lane
            let diff = f32x4_abs(f32x4_sub(self.0, rhs.0));
            let eps  = f32x4_splat(EPSILON);
            let lt   = f32x4_lt(diff, eps);
            // All 4 lanes must pass — bitmask gives 1 bit per lane
            (u32x4_bitmask(lt) & 0b1111) == 0b1111
        }
    }
}

// ── Operators ─────────────────────────────────────────────────────────────────

impl Add for Vec4 {
    type Output = Self;
    #[inline(always)] fn add(self, r: Self) -> Self { Self(unsafe { f32x4_add(self.0, r.0) }) }
}
impl Sub for Vec4 {
    type Output = Self;
    #[inline(always)] fn sub(self, r: Self) -> Self { Self(unsafe { f32x4_sub(self.0, r.0) }) }
}
impl Mul<f32> for Vec4 {
    type Output = Self;
    #[inline(always)] fn mul(self, s: f32) -> Self { Self(unsafe { f32x4_mul(self.0, f32x4_splat(s)) }) }
}
impl Mul<Vec4> for f32 {
    type Output = Vec4;
    #[inline(always)] fn mul(self, v: Vec4) -> Vec4 { Vec4(unsafe { f32x4_mul(f32x4_splat(self), v.0) }) }
}
impl Mul for Vec4 {
    type Output = Self;
    #[inline(always)] fn mul(self, r: Self) -> Self { Self(unsafe { f32x4_mul(self.0, r.0) }) }
}
impl Div<f32> for Vec4 {
    type Output = Self;
    #[inline(always)] fn div(self, s: f32) -> Self { Self(unsafe { f32x4_div(self.0, f32x4_splat(s)) }) }
}
impl Neg for Vec4 {
    type Output = Self;
    #[inline(always)] fn neg(self) -> Self { Self(unsafe { f32x4_neg(self.0) }) }
}
impl AddAssign for Vec4 {
    #[inline(always)] fn add_assign(&mut self, r: Self) { self.0 = unsafe { f32x4_add(self.0, r.0) }; }
}
impl SubAssign for Vec4 {
    #[inline(always)] fn sub_assign(&mut self, r: Self) { self.0 = unsafe { f32x4_sub(self.0, r.0) }; }
}
impl MulAssign<f32> for Vec4 {
    #[inline(always)] fn mul_assign(&mut self, s: f32) { self.0 = unsafe { f32x4_mul(self.0, f32x4_splat(s)) }; }
}
impl DivAssign<f32> for Vec4 {
    #[inline(always)] fn div_assign(&mut self, s: f32) { self.0 = unsafe { f32x4_div(self.0, f32x4_splat(s)) }; }
}

impl PartialEq for Vec4 {
    #[inline]
    fn eq(&self, rhs: &Self) -> bool {
        unsafe { (u32x4_bitmask(f32x4_eq(self.0, rhs.0)) & 0b1111) == 0b1111 }
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
