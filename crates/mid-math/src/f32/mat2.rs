// crates/mid-math/src/f32/mat2.rs
//! Mat2 — always scalar on all platforms. 16 bytes, column-major.

use core::fmt;
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use crate::{Vec2, EPSILON};

/// 2×2 column-major matrix.
///
/// Layout: `[x_axis | y_axis]` in memory (column 0 then column 1).
/// Always scalar — too small for SIMD to offer benefit.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct Mat2 {
    pub x_axis: Vec2,
    pub y_axis: Vec2,
}

impl Mat2 {
    pub const ZERO:     Self = Self { x_axis: Vec2::ZERO, y_axis: Vec2::ZERO };
    pub const IDENTITY: Self = Self { x_axis: Vec2::X,    y_axis: Vec2::Y    };

    // ── Constructors ──────────────────────────────────────────────────────────

    /// Build from two column vectors.
    #[inline(always)]
    pub fn from_cols(x_axis: Vec2, y_axis: Vec2) -> Self { Self { x_axis, y_axis } }

    /// Build from a flat array in column-major order.
    #[inline]
    pub fn from_cols_array(m: &[f32; 4]) -> Self {
        Self::from_cols(Vec2::new(m[0], m[1]), Vec2::new(m[2], m[3]))
    }

    /// Flatten to column-major array.
    #[inline]
    pub fn to_cols_array(self) -> [f32; 4] {
        [self.x_axis.x, self.x_axis.y, self.y_axis.x, self.y_axis.y]
    }

    /// Build from a 2×2 array `[[col0_x, col0_y], [col1_x, col1_y]]`.
    #[inline]
    pub fn from_cols_array_2d(m: &[[f32; 2]; 2]) -> Self {
        Self::from_cols(Vec2::from_array(m[0]), Vec2::from_array(m[1]))
    }

    /// Expand to 2×2 nested column array.
    #[inline]
    pub fn to_cols_array_2d(self) -> [[f32; 2]; 2] {
        [self.x_axis.to_array(), self.y_axis.to_array()]
    }

    /// Write columns to a flat slice in column-major order. Panics if `slice.len() < 4`.
    #[inline]
    pub fn write_cols_to_slice(self, slice: &mut [f32]) {
        slice[0] = self.x_axis.x; slice[1] = self.x_axis.y;
        slice[2] = self.y_axis.x; slice[3] = self.y_axis.y;
    }

    /// Build from a diagonal vector — off-diagonal elements are zero.
    #[inline]
    pub fn from_diagonal(diagonal: Vec2) -> Self {
        Self::from_cols(Vec2::new(diagonal.x, 0.0), Vec2::new(0.0, diagonal.y))
    }

    /// 2D rotation matrix — counter-clockwise by `angle` radians.
    #[inline]
    pub fn from_angle(angle: f32) -> Self {
        let (s, c) = angle.sin_cos();
        Self::from_cols(Vec2::new(c, s), Vec2::new(-s, c))
    }

    /// Uniform scale matrix.
    #[inline]
    pub fn from_scale(scale: Vec2) -> Self {
        Self::from_cols(Vec2::new(scale.x, 0.0), Vec2::new(0.0, scale.y))
    }

    /// Combined scale + rotation (scale applied before rotation).
    #[inline]
    pub fn from_scale_angle(scale: Vec2, angle: f32) -> Self {
        let (s, c) = angle.sin_cos();
        Self::from_cols(Vec2::new(c * scale.x, s * scale.x), Vec2::new(-s * scale.y, c * scale.y))
    }

    /// Extract the upper-left 2×2 of a Mat3.
    #[inline]
    pub fn from_mat3(m: crate::Mat3) -> Self {
        Self::from_cols(
            Vec2::new(m.cols[0][0], m.cols[0][1]),
            Vec2::new(m.cols[1][0], m.cols[1][1]),
        )
    }

    // ── Column / row accessors ────────────────────────────────────────────────

    /// Column vector at `index` (0 = x_axis, 1 = y_axis). Panics if index ≥ 2.
    #[inline]
    pub fn col(&self, index: usize) -> Vec2 {
        match index {
            0 => self.x_axis,
            1 => self.y_axis,
            _ => panic!("Mat2::col index {index} out of bounds"),
        }
    }

    /// Mutable column reference. Panics if index ≥ 2.
    #[inline]
    pub fn col_mut(&mut self, index: usize) -> &mut Vec2 {
        match index {
            0 => &mut self.x_axis,
            1 => &mut self.y_axis,
            _ => panic!("Mat2::col_mut index {index} out of bounds"),
        }
    }

    /// Row vector at `index` (0 = top, 1 = bottom). Panics if index ≥ 2.
    /// Rows are not contiguous in memory — this builds a new Vec2.
    #[inline]
    pub fn row(&self, index: usize) -> Vec2 {
        match index {
            0 => Vec2::new(self.x_axis.x, self.y_axis.x),
            1 => Vec2::new(self.x_axis.y, self.y_axis.y),
            _ => panic!("Mat2::row index {index} out of bounds"),
        }
    }

    /// Main diagonal as a Vec2: `(m[0][0], m[1][1])`.
    #[inline]
    pub fn diagonal(&self) -> Vec2 { Vec2::new(self.x_axis.x, self.y_axis.y) }

    // ── Core operations ───────────────────────────────────────────────────────

    /// Transpose: swap rows and columns.
    #[inline]
    pub fn transpose(self) -> Self {
        Self::from_cols(
            Vec2::new(self.x_axis.x, self.y_axis.x),
            Vec2::new(self.x_axis.y, self.y_axis.y),
        )
    }

    /// Determinant: `ad - bc`.
    #[inline]
    pub fn determinant(self) -> f32 {
        self.x_axis.x * self.y_axis.y - self.x_axis.y * self.y_axis.x
    }

    /// Try to invert. Returns `None` if the matrix is singular (|det| < ε).
    #[inline]
    pub fn try_inverse(self) -> Option<Self> {
        let det = self.determinant();
        if det.abs() < EPSILON {
            None
        } else {
            let inv_det = 1.0 / det;
            Some(Self::from_cols(
                Vec2::new( self.y_axis.y * inv_det, -self.x_axis.y * inv_det),
                Vec2::new(-self.y_axis.x * inv_det,  self.x_axis.x * inv_det),
            ))
        }
    }

    /// Alias for `try_inverse` (original mid-math name kept).
    #[inline]
    pub fn inverse(self) -> Option<Self> { self.try_inverse() }

    /// Invert, returning `ZERO` if singular.
    #[inline]
    pub fn inverse_or_zero(self) -> Self { self.try_inverse().unwrap_or(Self::ZERO) }

    // ── Transform ─────────────────────────────────────────────────────────────

    /// Multiply matrix by a column vector: `M * v`.
    #[inline]
    pub fn mul_vec2(self, rhs: Vec2) -> Vec2 {
        Vec2::new(
            self.x_axis.x * rhs.x + self.y_axis.x * rhs.y,
            self.x_axis.y * rhs.x + self.y_axis.y * rhs.y,
        )
    }

    /// Multiply transpose by a column vector: `M^T * v` (= row-vector `v * M`).
    /// Cheaper than `self.transpose().mul_vec2(rhs)` — no extra allocation.
    #[inline]
    pub fn mul_transpose_vec2(self, rhs: Vec2) -> Vec2 {
        Vec2::new(self.x_axis.dot(rhs), self.y_axis.dot(rhs))
    }

    /// Matrix × matrix.
    #[inline]
    pub fn mul_mat2(self, rhs: Self) -> Self {
        Self::from_cols(self.mul_vec2(rhs.x_axis), self.mul_vec2(rhs.y_axis))
    }

    /// Multiply each column by the matching component of `scale` (column scale).
    /// Equivalent to `self * Mat2::from_diagonal(scale)`.
    #[inline]
    pub fn mul_diagonal_scale(self, scale: Vec2) -> Self {
        Self::from_cols(self.x_axis * scale.x, self.y_axis * scale.y)
    }

    /// Scale all elements by `rhs`.
    #[inline]
    pub fn mul_scalar(self, rhs: f32) -> Self {
        Self::from_cols(self.x_axis * rhs, self.y_axis * rhs)
    }

    /// Divide all elements by `rhs`.
    #[inline]
    pub fn div_scalar(self, rhs: f32) -> Self { self.mul_scalar(1.0 / rhs) }

    /// Element-wise reciprocal.
    #[inline]
    pub fn recip(self) -> Self {
        Self::from_cols(self.x_axis.recip(), self.y_axis.recip())
    }

    /// Element-wise absolute value.
    #[inline]
    pub fn abs(self) -> Self {
        Self::from_cols(self.x_axis.abs(), self.y_axis.abs())
    }

    // ── Named add/sub aliases ─────────────────────────────────────────────────

    /// Element-wise addition (named method, mirrors operator).
    #[inline]
    pub fn add_mat2(self, rhs: Self) -> Self {
        Self::from_cols(self.x_axis + rhs.x_axis, self.y_axis + rhs.y_axis)
    }

    /// Element-wise subtraction (named method, mirrors operator).
    #[inline]
    pub fn sub_mat2(self, rhs: Self) -> Self {
        Self::from_cols(self.x_axis - rhs.x_axis, self.y_axis - rhs.y_axis)
    }

    // ── Predicates ────────────────────────────────────────────────────────────

    #[inline]
    pub fn is_finite(self) -> bool { self.x_axis.is_finite() && self.y_axis.is_finite() }
    #[inline]
    pub fn is_nan(self) -> bool { self.x_axis.is_nan() || self.y_axis.is_nan() }

    // ── Approx equality ───────────────────────────────────────────────────────

    #[inline]
    pub fn abs_diff_eq(self, rhs: Self, max_abs_diff: f32) -> bool {
        self.x_axis.abs_diff_eq(rhs.x_axis, max_abs_diff)
            && self.y_axis.abs_diff_eq(rhs.y_axis, max_abs_diff)
    }

    #[inline]
    pub fn approx_eq(self, rhs: Self) -> bool { self.abs_diff_eq(rhs, EPSILON) }
}

// ── Default / Display ─────────────────────────────────────────────────────────

impl Default for Mat2 {
    fn default() -> Self { Self::IDENTITY }
}

impl fmt::Display for Mat2 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}, {}]", self.x_axis, self.y_axis)
    }
}

// ── Operators ─────────────────────────────────────────────────────────────────

impl Mul for Mat2 {
    type Output = Self;
    #[inline] fn mul(self, rhs: Self) -> Self { self.mul_mat2(rhs) }
}
impl MulAssign for Mat2 {
    #[inline] fn mul_assign(&mut self, rhs: Self) { *self = self.mul_mat2(rhs); }
}
impl Mul<Vec2> for Mat2 {
    type Output = Vec2;
    #[inline] fn mul(self, rhs: Vec2) -> Vec2 { self.mul_vec2(rhs) }
}
impl Mul<f32> for Mat2 {
    type Output = Self;
    #[inline] fn mul(self, rhs: f32) -> Self { self.mul_scalar(rhs) }
}
impl Mul<Mat2> for f32 {
    type Output = Mat2;
    #[inline] fn mul(self, rhs: Mat2) -> Mat2 { rhs.mul_scalar(self) }
}
impl MulAssign<f32> for Mat2 {
    #[inline] fn mul_assign(&mut self, rhs: f32) { *self = self.mul_scalar(rhs); }
}
impl Add for Mat2 {
    type Output = Self;
    #[inline] fn add(self, rhs: Self) -> Self { self.add_mat2(rhs) }
}
impl AddAssign for Mat2 {
    #[inline] fn add_assign(&mut self, rhs: Self) { *self = self.add_mat2(rhs); }
}
impl Sub for Mat2 {
    type Output = Self;
    #[inline] fn sub(self, rhs: Self) -> Self { self.sub_mat2(rhs) }
}
impl SubAssign for Mat2 {
    #[inline] fn sub_assign(&mut self, rhs: Self) { *self = self.sub_mat2(rhs); }
}
impl Neg for Mat2 {
    type Output = Self;
    #[inline] fn neg(self) -> Self { Self::from_cols(-self.x_axis, -self.y_axis) }
}

// ── Conversions ───────────────────────────────────────────────────────────────

impl From<[[f32; 2]; 2]> for Mat2 {
    fn from(m: [[f32; 2]; 2]) -> Self { Self::from_cols_array_2d(&m) }
}
impl From<Mat2> for [[f32; 2]; 2] {
    fn from(m: Mat2) -> Self { m.to_cols_array_2d() }
}
impl From<[f32; 4]> for Mat2 {
    fn from(m: [f32; 4]) -> Self { Self::from_cols_array(&m) }
}
impl From<Mat2> for [f32; 4] {
    fn from(m: Mat2) -> Self { m.to_cols_array() }
}
