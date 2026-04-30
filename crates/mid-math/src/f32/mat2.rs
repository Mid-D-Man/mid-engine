// crates/mid-math/src/f32/mat2.rs
//! 2×2 column-major matrix — always scalar, 16-byte aligned.
//!
//! Used for 2D rotation, scale, shear and as the inner matrix
//! of any future Affine2 type. Fits entirely in one __m128 lane
//! but the benefit of SIMD on 2 floats is negligible — LLVM
//! auto-vectorises the hot paths (lerp, mul) anyway.

use core::fmt;
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use crate::{Vec2, EPSILON};

/// 2×2 column-major matrix. 16 bytes, 16-byte aligned.
///
/// Layout: `cols[0]` = x_axis (first column), `cols[1]` = y_axis (second column).
/// Element (row r, col c) is `cols[c][r]`.
///
/// **C interop:** `CMat2` (in `crate::ffi::types`) is the FFI-safe equivalent.
#[derive(Clone, Copy, PartialEq)]
#[repr(C, align(16))]
pub struct Mat2 {
    pub x_axis: Vec2,
    pub y_axis: Vec2,
}

impl Mat2 {
    /// All zeros — not invertible.
    pub const ZERO: Self = Self { x_axis: Vec2::ZERO, y_axis: Vec2::ZERO };

    /// Identity — no transform.
    pub const IDENTITY: Self = Self { x_axis: Vec2::X, y_axis: Vec2::Y };

    // ── Constructors ──────────────────────────────────────────────────────────

    #[inline(always)]
    pub fn from_cols(x_axis: Vec2, y_axis: Vec2) -> Self {
        Self { x_axis, y_axis }
    }

    /// Column-major flat array: `[x0, x1, y0, y1]`.
    #[inline]
    pub fn from_cols_array(m: &[f32; 4]) -> Self {
        Self::from_cols(Vec2::new(m[0], m[1]), Vec2::new(m[2], m[3]))
    }

    /// Column-major flat array: `[x0, x1, y0, y1]`.
    #[inline]
    pub fn to_cols_array(self) -> [f32; 4] {
        [self.x_axis.x, self.x_axis.y, self.y_axis.x, self.y_axis.y]
    }

    /// Column-major 2D array.
    #[inline]
    pub fn from_cols_array_2d(m: &[[f32; 2]; 2]) -> Self {
        Self::from_cols(Vec2::from(m[0]), Vec2::from(m[1]))
    }

    /// Diagonal scale matrix — off-diagonals are zero.
    #[inline]
    pub fn from_diagonal(d: Vec2) -> Self {
        Self::from_cols(Vec2::new(d.x, 0.0), Vec2::new(0.0, d.y))
    }

    /// Counter-clockwise rotation by `angle` radians.
    #[inline]
    pub fn from_angle(angle: f32) -> Self {
        let (s, c) = (angle.sin(), angle.cos());
        // column-major: col0 = (cos, sin), col1 = (-sin, cos)
        Self::from_cols(Vec2::new(c, s), Vec2::new(-s, c))
    }

    /// Uniform scale.
    #[inline]
    pub fn from_scale(scale: Vec2) -> Self {
        Self::from_cols(Vec2::new(scale.x, 0.0), Vec2::new(0.0, scale.y))
    }

    /// Non-uniform scale combined with a counter-clockwise rotation.
    ///
    /// Equivalent to `Mat2::from_angle(angle) * Mat2::from_scale(scale)`.
    #[inline]
    pub fn from_scale_angle(scale: Vec2, angle: f32) -> Self {
        let (s, c) = (angle.sin(), angle.cos());
        Self::from_cols(
            Vec2::new(c * scale.x,  s * scale.x),
            Vec2::new(-s * scale.y, c * scale.y),
        )
    }

    // ── Core ops ──────────────────────────────────────────────────────────────

    /// Transpose — swap rows and columns.
    #[inline]
    pub fn transpose(self) -> Self {
        Self::from_cols(
            Vec2::new(self.x_axis.x, self.y_axis.x),
            Vec2::new(self.x_axis.y, self.y_axis.y),
        )
    }

    /// Signed determinant: `x.x * y.y - x.y * y.x`.
    #[inline]
    pub fn determinant(self) -> f32 {
        self.x_axis.x * self.y_axis.y - self.x_axis.y * self.y_axis.x
    }

    /// Inverse via the 2×2 adjugate formula. Returns `None` if singular.
    #[inline]
    pub fn inverse(self) -> Option<Self> {
        let det = self.determinant();
        if det.abs() < EPSILON {
            return None;
        }
        let inv = 1.0 / det;
        Some(Self::from_cols(
            Vec2::new( self.y_axis.y * inv, -self.x_axis.y * inv),
            Vec2::new(-self.y_axis.x * inv,  self.x_axis.x * inv),
        ))
    }

    /// Inverse — returns `Mat2::ZERO` when singular instead of `None`.
    #[inline]
    pub fn inverse_or_zero(self) -> Self {
        self.inverse().unwrap_or(Self::ZERO)
    }

    // ── Transform helpers ─────────────────────────────────────────────────────

    /// Transform a 2D vector: `M * v`.
    #[inline]
    pub fn mul_vec2(self, v: Vec2) -> Vec2 {
        Vec2::new(
            self.x_axis.x * v.x + self.y_axis.x * v.y,
            self.x_axis.y * v.x + self.y_axis.y * v.y,
        )
    }

    /// Matrix multiply: `self * rhs`.
    #[inline]
    pub fn mul_mat2(self, rhs: Self) -> Self {
        Self::from_cols(
            self.mul_vec2(rhs.x_axis),
            self.mul_vec2(rhs.y_axis),
        )
    }

    /// Element-wise scalar multiply.
    #[inline]
    pub fn mul_scalar(self, s: f32) -> Self {
        Self::from_cols(self.x_axis * s, self.y_axis * s)
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

    /// Approximate element-wise equality within `max_abs_diff`.
    #[inline]
    pub fn abs_diff_eq(self, rhs: Self, max_abs_diff: f32) -> bool {
        self.x_axis.approx_eq_eps(rhs.x_axis, max_abs_diff)
            && self.y_axis.approx_eq_eps(rhs.y_axis, max_abs_diff)
    }
}

// ── Operators ─────────────────────────────────────────────────────────────────

impl Mul for Mat2 {
    type Output = Self;
    #[inline(always)] fn mul(self, rhs: Self) -> Self { self.mul_mat2(rhs) }
}
impl MulAssign for Mat2 {
    #[inline(always)] fn mul_assign(&mut self, rhs: Self) { *self = self.mul_mat2(rhs); }
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
    fn add(self, rhs: Self) -> Self {
        Self::from_cols(self.x_axis + rhs.x_axis, self.y_axis + rhs.y_axis)
    }
}
impl AddAssign for Mat2 {
    #[inline(always)] fn add_assign(&mut self, rhs: Self) { *self = *self + rhs; }
}
impl Sub for Mat2 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self::from_cols(self.x_axis - rhs.x_axis, self.y_axis - rhs.y_axis)
    }
}
impl SubAssign for Mat2 {
    #[inline(always)] fn sub_assign(&mut self, rhs: Self) { *self = *self - rhs; }
}
impl Neg for Mat2 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self { Self::from_cols(-self.x_axis, -self.y_axis) }
}

impl Default for Mat2 { fn default() -> Self { Self::IDENTITY } }

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
    #[inline] fn from(m: [[f32; 2]; 2]) -> Self {
        Self::from_cols(Vec2::from(m[0]), Vec2::from(m[1]))
    }
}
impl From<Mat2> for [[f32; 2]; 2] {
    #[inline] fn from(m: Mat2) -> Self {
        [m.x_axis.to_array(), m.y_axis.to_array()]
    }
         }
