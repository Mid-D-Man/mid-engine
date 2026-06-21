// crates/mid-math/src/f32/sse2/mat2.rs
//! Mat2 backed by a single `__m128` on x86 / x86_64.
//!
//! Build 21: added `inverse_unchecked()` — same SSE2 cofactor as
//! `inverse_or_zero` but without the singular-guard mask. Returns Self
//! directly (no Option), matching glam's API contract. Use when the caller
//! guarantees the matrix is invertible.

use core::fmt;
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use crate::sse2::{m128_abs, m128_from_f32x4};
use crate::f32::vec2::Vec2;
use crate::EPSILON;

#[repr(C)]
union UnionCast { f: [f32; 4], v: Mat2 }

/// Sign pattern for 2×2 adjugate: [+d, -c, -b, +a] / det.
const SIGN: __m128 = m128_from_f32x4([1.0_f32, -1.0, -1.0, 1.0]);

/// 2×2 column-major matrix. 16 bytes, 16-byte aligned.
/// Both columns packed into a single `__m128`.
/// Layout (low→high lanes): [x_axis.x, x_axis.y, y_axis.x, y_axis.y].
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct Mat2(pub(crate) __m128);

impl core::ops::Deref for Mat2 {
    type Target = crate::deref::Cols2<Vec2>;
    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        unsafe { &*(self as *const Self).cast() }
    }
}

impl core::ops::DerefMut for Mat2 {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { &mut *(self as *mut Self).cast() }
    }
}

impl Mat2 {
    pub const ZERO:     Self = unsafe { UnionCast { f: [0.0; 4] }.v };
    pub const IDENTITY: Self = unsafe { UnionCast { f: [1.0, 0.0, 0.0, 1.0] }.v };

    #[inline(always)]
    const fn new(m00: f32, m01: f32, m10: f32, m11: f32) -> Self {
        unsafe { UnionCast { f: [m00, m01, m10, m11] }.v }
    }

    #[inline(always)]
    pub fn from_cols(x_axis: Vec2, y_axis: Vec2) -> Self {
        Self::new(x_axis.x, x_axis.y, y_axis.x, y_axis.y)
    }

    #[inline]
    pub fn from_cols_array(m: &[f32; 4]) -> Self { Self::new(m[0], m[1], m[2], m[3]) }

    #[inline]
    pub fn to_cols_array(self) -> [f32; 4] {
        [self.x_axis.x, self.x_axis.y, self.y_axis.x, self.y_axis.y]
    }

    #[inline]
    pub fn from_cols_array_2d(m: &[[f32; 2]; 2]) -> Self {
        Self::from_cols(Vec2::from(m[0]), Vec2::from(m[1]))
    }

    #[inline]
    pub fn from_diagonal(d: Vec2) -> Self { Self::new(d.x, 0.0, 0.0, d.y) }

    #[inline]
    pub fn from_scale_angle(scale: Vec2, angle: f32) -> Self {
        let (s, c) = angle.sin_cos();
        Self::new(c * scale.x, s * scale.x, -s * scale.y, c * scale.y)
    }

    #[inline]
    pub fn from_angle(angle: f32) -> Self {
        let (s, c) = angle.sin_cos();
        Self::new(c, s, -s, c)
    }

    #[inline]
    pub fn from_scale(scale: Vec2) -> Self { Self::new(scale.x, 0.0, 0.0, scale.y) }

    // ── Core ops ──────────────────────────────────────────────────────────────

    /// Transpose — one `_mm_shuffle_ps`.
    #[inline]
    pub fn transpose(self) -> Self {
        Self(unsafe { _mm_shuffle_ps::<0b11_01_10_00>(self.0, self.0) })
    }

    #[inline]
    pub fn diagonal(self) -> Vec2 { Vec2::new(self.x_axis.x, self.y_axis.y) }

    /// Signed determinant.
    #[inline]
    pub fn determinant(self) -> f32 {
        unsafe {
            let abcd = self.0;
            let dcba = _mm_shuffle_ps::<0b00_01_10_11>(abcd, abcd);
            let prod = _mm_mul_ps(abcd, dcba);
            let sub  = _mm_sub_ps(prod, _mm_shuffle_ps::<0b01_01_01_01>(prod, prod));
            _mm_cvtss_f32(sub)
        }
    }

    // ── Inverse inner ──────────────────────────────────────────────────────────

    #[inline(always)]
    unsafe fn inverse_raw(self) -> (__m128, f32) {
        let abcd    = self.0;
        let dcba    = _mm_shuffle_ps::<0b00_01_10_11>(abcd, abcd);
        let prod    = _mm_mul_ps(abcd, dcba);
        let sub     = _mm_sub_ps(prod, _mm_shuffle_ps::<0b01_01_01_01>(prod, prod));
        let det_f32 = _mm_cvtss_f32(sub);
        let det     = _mm_shuffle_ps::<0b00_00_00_00>(sub, sub);
        let tmp     = _mm_div_ps(SIGN, det);
        let reorder = _mm_shuffle_ps::<0b00_10_01_11>(abcd, abcd);
        (_mm_mul_ps(reorder, tmp), det_f32)
    }

    /// Checked inverse — `None` if |det| < EPSILON.
    #[inline]
    pub fn inverse(self) -> Option<Self> {
        let (m, det) = unsafe { self.inverse_raw() };
        if det.abs() < EPSILON { None } else { Some(Self(m)) }
    }

    /// Branchless inverse — returns `ZERO` if singular.
    ///
    /// Uses SSE2 compare-and-mask, no conditional jump. Preferred over
    /// `inverse()` in throughput-critical paths when a zero fallback is acceptable.
    #[inline]
    pub fn inverse_or_zero(self) -> Self {
        unsafe {
            let abcd    = self.0;
            let dcba    = _mm_shuffle_ps::<0b00_01_10_11>(abcd, abcd);
            let prod    = _mm_mul_ps(abcd, dcba);
            let sub     = _mm_sub_ps(prod, _mm_shuffle_ps::<0b01_01_01_01>(prod, prod));
            let det     = _mm_shuffle_ps::<0b00_00_00_00>(sub, sub);
            let mask    = _mm_cmpge_ps(m128_abs(det), _mm_set1_ps(EPSILON));
            let tmp     = _mm_div_ps(SIGN, det);
            let reorder = _mm_shuffle_ps::<0b00_10_01_11>(abcd, abcd);
            Self(_mm_and_ps(_mm_mul_ps(reorder, tmp), mask))
        }
    }

    /// Unchecked inverse — returns Self directly, no Option wrapper.
    ///
    /// **Undefined result (NaN/inf) if the matrix is singular.**
    ///
    /// ~30% faster than `inverse()` — no Option tag, no branch.
    /// Matches glam's `Mat2::inverse()` API contract (glam panics in debug,
    /// returns invalid in release). Use when input is known invertible.
    ///
    /// Build 21: added to close mat2/inverse bench gap (1.30 vs 0.985 ns).
    #[inline(always)]
    pub fn inverse_unchecked(self) -> Self {
        Self(unsafe { self.inverse_raw().0 })
    }

    // ── Transform helpers ─────────────────────────────────────────────────────

    #[inline]
    pub fn mul_vec2(self, v: Vec2) -> Vec2 {
        unsafe {
            let abcd = self.0;
            let xxyy = _mm_set_ps(v.y, v.y, v.x, v.x);
            let axbx = _mm_mul_ps(abcd, xxyy);
            let cydy = _mm_shuffle_ps::<0b01_00_11_10>(axbx, axbx);
            let res  = _mm_add_ps(axbx, cydy);
            Vec2::new(
                _mm_cvtss_f32(res),
                _mm_cvtss_f32(_mm_shuffle_ps::<0b01_01_01_01>(res, res)),
            )
        }
    }

    #[inline]
    pub fn mul_transpose_vec2(self, v: Vec2) -> Vec2 {
        Vec2::new(self.x_axis.dot(v), self.y_axis.dot(v))
    }

    #[inline]
    pub fn mul_mat2(self, rhs: Self) -> Self { self.mul(rhs) }

    #[inline]
    pub fn mul_scalar(self, s: f32) -> Self {
        Self(unsafe { _mm_mul_ps(self.0, _mm_set1_ps(s)) })
    }

    #[inline]
    pub fn div_scalar(self, s: f32) -> Self {
        Self(unsafe { _mm_div_ps(self.0, _mm_set1_ps(s)) })
    }

    // ── Predicates ────────────────────────────────────────────────────────────

    #[inline]
    pub fn is_finite(self) -> bool {
        self.x_axis.is_finite() && self.y_axis.is_finite()
    }

    #[inline]
    pub fn is_nan(self) -> bool {
        self.x_axis.is_nan() || self.y_axis.is_nan()
    }

    #[inline]
    pub fn abs_diff_eq(self, rhs: Self, max_abs_diff: f32) -> bool {
        unsafe {
            let diff  = _mm_sub_ps(self.0, rhs.0);
            let adiff = m128_abs(diff);
            let eps   = _mm_set1_ps(max_abs_diff);
            (_mm_movemask_ps(_mm_cmplt_ps(adiff, eps)) & 0b1111) == 0b1111
        }
    }
}

// ── Operators ─────────────────────────────────────────────────────────────────

impl Mul for Mat2 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        unsafe {
            let abcd  = self.0;
            let rhs   = rhs.0;
            let xxyy0 = _mm_shuffle_ps::<0b01_01_00_00>(rhs, rhs);
            let xxyy1 = _mm_shuffle_ps::<0b11_11_10_10>(rhs, rhs);
            let t0    = _mm_mul_ps(abcd, xxyy0);
            let t1    = _mm_mul_ps(abcd, xxyy1);
            let s0    = _mm_shuffle_ps::<0b01_00_11_10>(t0, t0);
            let s1    = _mm_shuffle_ps::<0b01_00_11_10>(t1, t1);
            let r0    = _mm_add_ps(t0, s0);
            let r1    = _mm_add_ps(t1, s1);
            Self(_mm_shuffle_ps::<0b01_00_01_00>(r0, r1))
        }
    }
}

impl MulAssign for Mat2 {
    #[inline(always)] fn mul_assign(&mut self, rhs: Self) { *self = self.mul(rhs); }
}
impl Mul<Vec2> for Mat2 {
    type Output = Vec2;
    #[inline(always)] fn mul(self, rhs: Vec2) -> Vec2 { self.mul_vec2(rhs) }
}
impl Mul<f32> for Mat2 {
    type Output = Self;
    #[inline(always)] fn mul(self, s: f32) -> Self { self.mul_scalar(s) }
}
impl Mul<Mat2> for f32 {
    type Output = Mat2;
    #[inline(always)] fn mul(self, m: Mat2) -> Mat2 { m.mul_scalar(self) }
}
impl Add for Mat2 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self { Self(unsafe { _mm_add_ps(self.0, rhs.0) }) }
}
impl AddAssign for Mat2 {
    #[inline(always)] fn add_assign(&mut self, rhs: Self) { *self = *self + rhs; }
}
impl Sub for Mat2 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self { Self(unsafe { _mm_sub_ps(self.0, rhs.0) }) }
}
impl SubAssign for Mat2 {
    #[inline(always)] fn sub_assign(&mut self, rhs: Self) { *self = *self - rhs; }
}
impl Neg for Mat2 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self { Self(unsafe { _mm_xor_ps(self.0, _mm_set1_ps(-0.0)) }) }
}
impl PartialEq for Mat2 {
    #[inline]
    fn eq(&self, rhs: &Self) -> bool {
        unsafe { (_mm_movemask_ps(_mm_cmpeq_ps(self.0, rhs.0)) & 0b1111) == 0b1111 }
    }
}
impl Default for Mat2 { #[inline] fn default() -> Self { Self::IDENTITY } }

impl fmt::Debug for Mat2 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Mat2")
            .field("x_axis", &self.x_axis)
            .field("y_axis", &self.y_axis)
            .finish()
    }
}
impl fmt::Display for Mat2 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}, {}]", self.x_axis, self.y_axis)
    }
}
impl From<[[f32; 2]; 2]> for Mat2 {
    #[inline]
    fn from(m: [[f32; 2]; 2]) -> Self {
        Self::from_cols(Vec2::from(m[0]), Vec2::from(m[1]))
    }
}
impl From<Mat2> for [[f32; 2]; 2] {
    #[inline]
    fn from(m: Mat2) -> Self { [m.x_axis.to_array(), m.y_axis.to_array()] }
             }
