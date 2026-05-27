// crates/mid-math/src/f64/daffine2.rs
//! Double-precision 2D affine transform — rotation · scale · translation.
//!
//! Direct f64 port of f32/affine2.rs.
//! All field types are DVec2; DEPSILON replaces EPSILON.
//!
//! # Use cases
//!   - High-precision 2D physics simulation
//!   - Large-world 2D coordinate systems (sub-millimetre accuracy)
//!   - Authoring tools requiring f64 UI transforms
//!   - Any 2D domain where f32 rounding is detectable

use core::fmt;
use core::ops::Mul;
use super::dvec2::{DVec2, DEPSILON};
use super::dmat3::DMat3;

/// Double-precision 2D affine transform.
///
/// 48 bytes, 8-byte aligned. Always scalar.
///
/// **C interop:** use `CDAffine2` at the FFI boundary.
#[derive(Clone, Copy, PartialEq)]
#[repr(C, align(8))]
pub struct DAffine2 {
    /// First column of the 2×2 matrix.
    pub x_axis: DVec2,
    /// Second column of the 2×2 matrix.
    pub y_axis: DVec2,
    /// Translation component.
    pub translation: DVec2,
}

impl DAffine2 {
    /// Identity — no rotation, no scale, no translation.
    pub const IDENTITY: Self = Self {
        x_axis:      DVec2::X,
        y_axis:      DVec2::Y,
        translation: DVec2::ZERO,
    };

    /// All zeros — not a valid transform.
    pub const ZERO: Self = Self {
        x_axis:      DVec2::ZERO,
        y_axis:      DVec2::ZERO,
        translation: DVec2::ZERO,
    };

    // ── Constructors ──────────────────────────────────────────────────────────

    #[inline(always)]
    pub fn from_cols(x_axis: DVec2, y_axis: DVec2, translation: DVec2) -> Self {
        Self { x_axis, y_axis, translation }
    }

    #[inline]
    pub fn from_translation(t: DVec2) -> Self {
        Self { x_axis: DVec2::X, y_axis: DVec2::Y, translation: t }
    }

    /// Counter-clockwise rotation by `angle` radians (f64 precision).
    #[inline]
    pub fn from_angle(angle: f64) -> Self {
        let (s, c) = angle.sin_cos();
        Self {
            x_axis:      DVec2::new(c, s),
            y_axis:      DVec2::new(-s, c),
            translation: DVec2::ZERO,
        }
    }

    #[inline]
    pub fn from_scale(s: DVec2) -> Self {
        Self {
            x_axis:      DVec2::new(s.x, 0.0),
            y_axis:      DVec2::new(0.0, s.y),
            translation: DVec2::ZERO,
        }
    }

    #[inline]
    pub fn from_scale_uniform(s: f64) -> Self {
        Self::from_scale(DVec2::splat(s))
    }

    #[inline]
    pub fn from_scale_angle(scale: DVec2, angle: f64) -> Self {
        let (s, c) = angle.sin_cos();
        Self {
            x_axis:      DVec2::new(c * scale.x,  s * scale.x),
            y_axis:      DVec2::new(-s * scale.y, c * scale.y),
            translation: DVec2::ZERO,
        }
    }

    /// Full TRS — scale, then rotate, then translate.
    #[inline]
    pub fn from_trs(t: DVec2, angle: f64, scale: DVec2) -> Self {
        let (s, c) = angle.sin_cos();
        Self {
            x_axis:      DVec2::new(c * scale.x,  s * scale.x),
            y_axis:      DVec2::new(-s * scale.y, c * scale.y),
            translation: t,
        }
    }

    /// Extract from the upper-left 2×2 + translation column of a DMat3.
    #[inline]
    pub fn from_mat3(m: DMat3) -> Self {
        Self {
            x_axis:      DVec2::new(m.cols[0][0], m.cols[0][1]),
            y_axis:      DVec2::new(m.cols[1][0], m.cols[1][1]),
            translation: DVec2::new(m.cols[2][0], m.cols[2][1]),
        }
    }

    // ── Decomposition ─────────────────────────────────────────────────────────

    /// Extract the rotation angle in radians.
    #[inline]
    pub fn rotation_angle(self) -> f64 {
        self.x_axis.y.atan2(self.x_axis.x)
    }

    /// Extract the scale vector.
    #[inline]
    pub fn scale(self) -> DVec2 {
        DVec2::new(self.x_axis.length(), self.y_axis.length())
    }

    /// Decompose into `(translation, angle_radians, scale)`.
    #[inline]
    pub fn decompose(self) -> (DVec2, f64, DVec2) {
        let sx = self.x_axis.length();
        let sy = self.y_axis.length();
        let angle = self.x_axis.y.atan2(self.x_axis.x);
        (self.translation, angle, DVec2::new(sx, sy))
    }

    // ── Conversion ────────────────────────────────────────────────────────────

    /// Expand to a 3×3 homogeneous matrix.
    #[inline]
    pub fn to_mat3(self) -> DMat3 {
        DMat3::from_cols(
            [self.x_axis.x, self.x_axis.y, 0.0],
            [self.y_axis.x, self.y_axis.y, 0.0],
            [self.translation.x, self.translation.y, 1.0],
        )
    }

    /// Lossy downcast to single-precision `Affine2`.
    #[inline]
    pub fn as_affine2(self) -> crate::Affine2 {
        crate::Affine2::from_cols(
            crate::Vec2::new(self.x_axis.x as f32, self.x_axis.y as f32),
            crate::Vec2::new(self.y_axis.x as f32, self.y_axis.y as f32),
            crate::Vec2::new(self.translation.x as f32, self.translation.y as f32),
        )
    }

    // ── Transform helpers ─────────────────────────────────────────────────────

    #[inline(always)]
    pub fn transform_point(self, p: DVec2) -> DVec2 {
        self.x_axis * p.x + self.y_axis * p.y + self.translation
    }

    #[inline(always)]
    pub fn transform_vector(self, v: DVec2) -> DVec2 {
        self.x_axis * v.x + self.y_axis * v.y
    }

    // ── Inverse ───────────────────────────────────────────────────────────────

    /// Inverse of a TRS affine transform (fast path — no shear).
    #[inline]
    pub fn inverse(self) -> Self {
        let sx2 = self.x_axis.length_sq();
        let sy2 = self.y_axis.length_sq();

        let isx = if sx2 < DEPSILON { 0.0 } else { 1.0 / sx2 };
        let isy = if sy2 < DEPSILON { 0.0 } else { 1.0 / sy2 };

        let inv_x = DVec2::new(self.x_axis.x * isx, self.y_axis.x * isy);
        let inv_y = DVec2::new(self.x_axis.y * isx, self.y_axis.y * isy);

        let t = self.translation;
        let inv_t = -(inv_x * t.x + inv_y * t.y);

        Self { x_axis: inv_x, y_axis: inv_y, translation: inv_t }
    }

    /// General 2D affine inverse (handles shear). Returns `None` if singular.
    #[inline]
    pub fn inverse_general(self) -> Option<Self> {
        let det = self.x_axis.x * self.y_axis.y - self.x_axis.y * self.y_axis.x;
        if det.abs() < DEPSILON { return None; }
        let inv = 1.0 / det;

        let inv_x = DVec2::new( self.y_axis.y * inv, -self.x_axis.y * inv);
        let inv_y = DVec2::new(-self.y_axis.x * inv,  self.x_axis.x * inv);

        let t = self.translation;
        let inv_t = DVec2::new(
            -(inv_x.x * t.x + inv_y.x * t.y),
            -(inv_x.y * t.x + inv_y.y * t.y),
        );

        Some(Self { x_axis: inv_x, y_axis: inv_y, translation: inv_t })
    }

    // ── Predicates ────────────────────────────────────────────────────────────

    #[inline]
    pub fn is_finite(self) -> bool {
        self.x_axis.is_finite()
            && self.y_axis.is_finite()
            && self.translation.is_finite()
    }

    #[inline]
    pub fn is_axis_aligned(self) -> bool {
        self.x_axis.y.abs() < DEPSILON && self.y_axis.x.abs() < DEPSILON
    }
}

// ── Operators ─────────────────────────────────────────────────────────────────

impl Mul for DAffine2 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self {
            x_axis:      self.transform_vector(rhs.x_axis),
            y_axis:      self.transform_vector(rhs.y_axis),
            translation: self.transform_point(rhs.translation),
        }
    }
}

impl Mul<DVec2> for DAffine2 {
    type Output = DVec2;
    #[inline(always)]
    fn mul(self, rhs: DVec2) -> DVec2 {
        self.transform_point(rhs)
    }
}

impl Default for DAffine2 {
    #[inline] fn default() -> Self { Self::IDENTITY }
}

impl fmt::Debug for DAffine2 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DAffine2")
            .field("x_axis",      &self.x_axis)
            .field("y_axis",      &self.y_axis)
            .field("translation", &self.translation)
            .finish()
    }
}

impl fmt::Display for DAffine2 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let p = f.precision().unwrap_or(6);
        write!(
            f,
            "DAffine2 {{ x:{:.*?} y:{:.*?} t:{:.*?} }}",
            p, self.x_axis, p, self.y_axis, p, self.translation
        )
    }
}

impl From<DMat3> for DAffine2 {
    #[inline] fn from(m: DMat3) -> Self { Self::from_mat3(m) }
}

impl From<DAffine2> for DMat3 {
    #[inline] fn from(a: DAffine2) -> Self { a.to_mat3() }
         }
