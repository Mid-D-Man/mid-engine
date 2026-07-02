// crates/mid-math/src/f64/dmat2.rs
//! Double-precision 2×2 column-major matrix. 32 bytes, align(16).
//!
//! Used for 2D rotation/scale, and as the inner matrix of DAffine2
//! (which we'll add in Phase 3A). Scalar only — SIMD on 2 f64s gives
//! essentially no gain.

use core::fmt;
use core::ops::{Add, Mul, Neg, Sub};

use crate::DVec2;
use super::dvec2::DEPSILON;

/// 2×2 column-major double-precision matrix. 32 bytes, align(16).
///
/// `x_axis` = first column, `y_axis` = second column.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C, align(16))]
pub struct DMat2 {
    pub x_axis: DVec2,
    pub y_axis: DVec2,
}

impl DMat2 {
    pub const ZERO:     Self = Self { x_axis: DVec2::ZERO, y_axis: DVec2::ZERO };
    pub const IDENTITY: Self = Self { x_axis: DVec2::X,    y_axis: DVec2::Y    };

    // ── Constructors ──────────────────────────────────────────────────────────

    #[inline(always)]
    pub fn from_cols(x_axis: DVec2, y_axis: DVec2) -> Self { Self { x_axis, y_axis } }

    /// Column-major flat array: `[x0, x1, y0, y1]`.
    #[inline]
    pub fn from_cols_array(m: &[f64; 4]) -> Self {
        Self::from_cols(DVec2::new(m[0], m[1]), DVec2::new(m[2], m[3]))
    }

    #[inline]
    pub fn to_cols_array(self) -> [f64; 4] {
        [self.x_axis.x, self.x_axis.y, self.y_axis.x, self.y_axis.y]
    }

    /// Diagonal scale matrix.
    #[inline]
    pub fn from_diagonal(d: DVec2) -> Self {
        Self::from_cols(DVec2::new(d.x, 0.0), DVec2::new(0.0, d.y))
    }

    /// Counter-clockwise rotation by `angle` radians.
    #[inline]
    pub fn from_angle(angle: f64) -> Self {
        let (s, c) = angle.sin_cos();
        Self::from_cols(DVec2::new(c, s), DVec2::new(-s, c))
    }

    // ── Core ops ──────────────────────────────────────────────────────────────

    #[inline]
    pub fn transpose(self) -> Self {
        Self::from_cols(
            DVec2::new(self.x_axis.x, self.y_axis.x),
            DVec2::new(self.x_axis.y, self.y_axis.y),
        )
    }

    #[inline]
    pub fn determinant(self) -> f64 {
        self.x_axis.x * self.y_axis.y - self.x_axis.y * self.y_axis.x
    }

    pub fn inverse(self) -> Option<Self> {
        let det = self.determinant();
        if det.abs() < DEPSILON { return None; }
        let inv = 1.0 / det;
        Some(Self::from_cols(
            DVec2::new( self.y_axis.y * inv, -self.x_axis.y * inv),
            DVec2::new(-self.y_axis.x * inv,  self.x_axis.x * inv),
        ))
    }

    #[inline]
    pub fn inverse_or_zero(self) -> Self {
        self.inverse().unwrap_or(Self::ZERO)
    }

    #[inline]
    pub fn mul_vec2(self, v: DVec2) -> DVec2 {
        DVec2::new(
            self.x_axis.x * v.x + self.y_axis.x * v.y,
            self.x_axis.y * v.x + self.y_axis.y * v.y,
        )
    }

    #[inline]
    pub fn mul_scalar(self, s: f64) -> Self {
        Self::from_cols(self.x_axis * s, self.y_axis * s)
    }

    pub fn is_finite(self) -> bool {
        self.x_axis.is_finite() && self.y_axis.is_finite()
    }

    pub fn abs_diff_eq(self, rhs: Self, eps: f64) -> bool {
        self.x_axis.approx_eq(rhs.x_axis) && {
            // use explicit eps rather than DEPSILON
            (self.x_axis.x - rhs.x_axis.x).abs() < eps
                && (self.x_axis.y - rhs.x_axis.y).abs() < eps
                && (self.y_axis.x - rhs.y_axis.x).abs() < eps
                && (self.y_axis.y - rhs.y_axis.y).abs() < eps
        }
    }

    /// Lossy cast to single-precision `Mat2`.
    pub fn as_mat2(self) -> crate::Mat2 {
        crate::Mat2::from_cols(
            crate::Vec2::new(self.x_axis.x as f32, self.x_axis.y as f32),
            crate::Vec2::new(self.y_axis.x as f32, self.y_axis.y as f32),
        )
    }
}

impl Default for DMat2 { fn default() -> Self { Self::IDENTITY } }

impl Mul for DMat2 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self::from_cols(self.mul_vec2(rhs.x_axis), self.mul_vec2(rhs.y_axis))
    }
}
impl Mul<DVec2> for DMat2 {
    type Output = DVec2;
    #[inline(always)] fn mul(self, v: DVec2) -> DVec2 { self.mul_vec2(v) }
}
impl Mul<f64> for DMat2 {
    type Output = Self;
    #[inline(always)] fn mul(self, s: f64) -> Self { self.mul_scalar(s) }
}
impl Add for DMat2 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self::from_cols(self.x_axis + rhs.x_axis, self.y_axis + rhs.y_axis)
    }
}
impl Sub for DMat2 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self::from_cols(self.x_axis - rhs.x_axis, self.y_axis - rhs.y_axis)
    }
}
impl Neg for DMat2 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self { Self::from_cols(-self.x_axis, -self.y_axis) }
}

impl fmt::Display for DMat2 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}, {}]", self.x_axis, self.y_axis)
    }
}

impl From<[[f64; 2]; 2]> for DMat2 {
    fn from(m: [[f64; 2]; 2]) -> Self {
        Self::from_cols(DVec2::from(m[0]), DVec2::from(m[1]))
    }
}
impl From<DMat2> for [[f64; 2]; 2] {
    fn from(m: DMat2) -> Self {
        [m.x_axis.to_array(), m.y_axis.to_array()]
    }
    }
