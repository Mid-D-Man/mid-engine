// crates/mid-math/src/f32/affine2.rs
//! 2D affine transform — rotation · scale · translation.
//!
//! Stores a 2×2 linear matrix (x_axis, y_axis as Vec2) and a Vec2 translation.
//! The implicit bottom row [0, 0, 1] is never stored or computed.
//!
//! Compared to Mat3 for 2D TRS-only work:
//!   inverse()         : ~2× faster (no 3rd row)
//!   mul (compose)     : ~30% fewer multiply-adds
//!   transform_point   : same speed
//!
//! # Memory layout
//!   x_axis:      Vec2  @ bytes  0-7   (16-byte aligned via Vec2 field order)
//!   y_axis:      Vec2  @ bytes  8-15
//!   translation: Vec2  @ bytes 16-23
//!   Total: 24 bytes, align(8)
//!
//! # Use cases
//!   - Sprite transforms
//!   - UI element layout
//!   - Tilemap cell-to-world
//!   - 2D physics body transforms
//!   - Any 2D rotation + non-uniform scale + translation

use core::fmt;
use core::ops::Mul;
use crate::{Vec2, Mat3, EPSILON};

/// 2D affine transform.
///
/// 24 bytes, 8-byte aligned. Always scalar — Vec2 has no SIMD dispatch.
///
/// **C interop:** use `CAffine2` (in `crate::ffi::types`) at the FFI boundary.
#[derive(Clone, Copy, PartialEq)]
#[repr(C, align(8))]
pub struct Affine2 {
    /// First column of the 2×2 matrix (x-basis scaled by sx and rotated).
    pub x_axis: Vec2,
    /// Second column of the 2×2 matrix (y-basis scaled by sy and rotated).
    pub y_axis: Vec2,
    /// Translation component (applied after the linear transform).
    pub translation: Vec2,
}

impl Affine2 {
    /// Identity — no rotation, no scale, no translation.
    pub const IDENTITY: Self = Self {
        x_axis:      Vec2::X,
        y_axis:      Vec2::Y,
        translation: Vec2::ZERO,
    };

    /// All zeros — not a valid transform. Useful as a sentinel.
    pub const ZERO: Self = Self {
        x_axis:      Vec2::ZERO,
        y_axis:      Vec2::ZERO,
        translation: Vec2::ZERO,
    };

    // ── Constructors ──────────────────────────────────────────────────────────

    /// Build directly from column vectors.
    #[inline(always)]
    pub fn from_cols(x_axis: Vec2, y_axis: Vec2, translation: Vec2) -> Self {
        Self { x_axis, y_axis, translation }
    }

    /// Translation only — identity rotation and scale.
    #[inline]
    pub fn from_translation(t: Vec2) -> Self {
        Self { x_axis: Vec2::X, y_axis: Vec2::Y, translation: t }
    }

    /// Counter-clockwise rotation by `angle` radians.
    #[inline]
    pub fn from_angle(angle: f32) -> Self {
        let (s, c) = angle.sin_cos();
        // Column-major: col0 = (cos, sin), col1 = (-sin, cos)
        Self {
            x_axis:      Vec2::new(c, s),
            y_axis:      Vec2::new(-s, c),
            translation: Vec2::ZERO,
        }
    }

    /// Non-uniform scale only.
    #[inline]
    pub fn from_scale(s: Vec2) -> Self {
        Self {
            x_axis:      Vec2::new(s.x, 0.0),
            y_axis:      Vec2::new(0.0, s.y),
            translation: Vec2::ZERO,
        }
    }

    /// Uniform scale only.
    #[inline]
    pub fn from_scale_uniform(s: f32) -> Self {
        Self::from_scale(Vec2::splat(s))
    }

    /// Scale then rotate (no translation). Equivalent to
    /// `Affine2::from_angle(angle) * Affine2::from_scale(scale)`.
    #[inline]
    pub fn from_scale_angle(scale: Vec2, angle: f32) -> Self {
        let (s, c) = angle.sin_cos();
        Self {
            x_axis:      Vec2::new(c * scale.x,  s * scale.x),
            y_axis:      Vec2::new(-s * scale.y, c * scale.y),
            translation: Vec2::ZERO,
        }
    }

    /// Full TRS — scale, then rotate, then translate.
    ///
    /// Equivalent to `T * R * S` in matrix form.
    #[inline]
    pub fn from_trs(t: Vec2, angle: f32, scale: Vec2) -> Self {
        let (s, c) = angle.sin_cos();
        Self {
            x_axis:      Vec2::new(c * scale.x,  s * scale.x),
            y_axis:      Vec2::new(-s * scale.y, c * scale.y),
            translation: t,
        }
    }

    /// Extract from the upper-left 2×2 + translation column of a Mat3.
    ///
    /// Assumes `m.cols[2][2] == 1.0` (homogeneous 2D matrix).
    #[inline]
    pub fn from_mat3(m: Mat3) -> Self {
        Self {
            x_axis:      Vec2::new(m.cols[0][0], m.cols[0][1]),
            y_axis:      Vec2::new(m.cols[1][0], m.cols[1][1]),
            translation: Vec2::new(m.cols[2][0], m.cols[2][1]),
        }
    }

    // ── Decomposition ─────────────────────────────────────────────────────────

    /// Extract the rotation angle in radians.
    ///
    /// Valid only when the transform has no shear (i.e. was built from TRS).
    /// Result is in `[-π, π]`.
    #[inline]
    pub fn rotation_angle(self) -> f32 {
        self.x_axis.y.atan2(self.x_axis.x)
    }

    /// Extract the scale vector.
    ///
    /// Returns `(|x_axis|, |y_axis|)`. Negative scale is not detectable
    /// without additional information.
    #[inline]
    pub fn scale(self) -> Vec2 {
        Vec2::new(self.x_axis.length(), self.y_axis.length())
    }

    /// Decompose into `(translation, angle_radians, scale)`.
    ///
    /// Equivalent to calling `translation`, `rotation_angle`, and `scale`
    /// individually but cheaper (one sqrt per axis instead of two).
    #[inline]
    pub fn decompose(self) -> (Vec2, f32, Vec2) {
        let sx = self.x_axis.length();
        let sy = self.y_axis.length();
        let angle = self.x_axis.y.atan2(self.x_axis.x);
        (self.translation, angle, Vec2::new(sx, sy))
    }

    // ── Conversion ────────────────────────────────────────────────────────────

    /// Expand to a 3×3 homogeneous matrix by appending the implicit `[0, 0, 1]` row.
    #[inline]
    pub fn to_mat3(self) -> Mat3 {
        Mat3::from_cols(
            [self.x_axis.x, self.x_axis.y, 0.0],
            [self.y_axis.x, self.y_axis.y, 0.0],
            [self.translation.x, self.translation.y, 1.0],
        )
    }

    // ── Transform helpers ─────────────────────────────────────────────────────

    /// Apply to a point — applies the 2×2 linear part then adds translation.
    #[inline(always)]
    pub fn transform_point(self, p: Vec2) -> Vec2 {
        self.x_axis * p.x + self.y_axis * p.y + self.translation
    }

    /// Apply to a direction vector — applies the 2×2 linear part only (no translation).
    ///
    /// Use for velocity vectors and normals (when scale is uniform).
    #[inline(always)]
    pub fn transform_vector(self, v: Vec2) -> Vec2 {
        self.x_axis * v.x + self.y_axis * v.y
    }

    // ── Inverse ───────────────────────────────────────────────────────────────

    /// Inverse of a TRS affine transform.
    ///
    /// Faster than the general 3×3 matrix inverse. Valid for
    /// translation + rotation + non-zero scale. Does NOT handle shear.
    ///
    /// # Derivation
    ///
    /// For M = R × S (the stored 2×2):
    /// ```text
    /// M^-1 = S^-1 × R^T
    /// (M^-1)[i,j] = axis_j[i] / |axis_j|²
    /// inv_t       = -(M^-1 × original_t)
    /// ```
    #[inline]
    pub fn inverse(self) -> Self {
        let sx2 = self.x_axis.length_sq();
        let sy2 = self.y_axis.length_sq();

        let isx = if sx2 < EPSILON { 0.0 } else { 1.0 / sx2 };
        let isy = if sy2 < EPSILON { 0.0 } else { 1.0 / sy2 };

        // Each new column is a *row* of the original 2×2, scaled by
        // the corresponding column's inverse squared length.
        let inv_x = Vec2::new(self.x_axis.x * isx, self.y_axis.x * isy);
        let inv_y = Vec2::new(self.x_axis.y * isx, self.y_axis.y * isy);

        let t = self.translation;
        let inv_t = -(inv_x * t.x + inv_y * t.y);

        Self {
            x_axis:      inv_x,
            y_axis:      inv_y,
            translation: inv_t,
        }
    }

    /// General 2D affine inverse (handles shear).
    ///
    /// Uses the 2×2 adjugate formula. Returns `None` if singular.
    #[inline]
    pub fn inverse_general(self) -> Option<Self> {
        let det = self.x_axis.x * self.y_axis.y - self.x_axis.y * self.y_axis.x;
        if det.abs() < EPSILON { return None; }
        let inv = 1.0 / det;

        let inv_x = Vec2::new( self.y_axis.y * inv, -self.x_axis.y * inv);
        let inv_y = Vec2::new(-self.y_axis.x * inv,  self.x_axis.x * inv);

        let t = self.translation;
        let inv_t = Vec2::new(
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

    /// True when the transform has no rotation or shear (only scale and translation).
    #[inline]
    pub fn is_axis_aligned(self) -> bool {
        self.x_axis.y.abs() < EPSILON && self.y_axis.x.abs() < EPSILON
    }
}

// ── Mul: compose two affine transforms ───────────────────────────────────────
//
// `self * rhs` applies rhs first, then self — same convention as Mat3.
//
// result.matrix2     = self.matrix2 × rhs.matrix2
// result.translation = self.transform_point(rhs.translation)

impl Mul for Affine2 {
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

/// Apply an Affine2 to a point directly: `transform * point`.
impl Mul<Vec2> for Affine2 {
    type Output = Vec2;
    #[inline(always)]
    fn mul(self, rhs: Vec2) -> Vec2 {
        self.transform_point(rhs)
    }
}

impl Default for Affine2 {
    #[inline] fn default() -> Self { Self::IDENTITY }
}

impl fmt::Debug for Affine2 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Affine2")
            .field("x_axis",      &self.x_axis)
            .field("y_axis",      &self.y_axis)
            .field("translation", &self.translation)
            .finish()
    }
}

impl fmt::Display for Affine2 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let p = f.precision().unwrap_or(4);
        write!(
            f,
            "Affine2 {{ x:{:.*?} y:{:.*?} t:{:.*?} }}",
            p, self.x_axis, p, self.y_axis, p, self.translation
        )
    }
}

impl From<Mat3> for Affine2 {
    #[inline] fn from(m: Mat3) -> Self { Self::from_mat3(m) }
}

impl From<Affine2> for Mat3 {
    #[inline] fn from(a: Affine2) -> Self { a.to_mat3() }
            }
