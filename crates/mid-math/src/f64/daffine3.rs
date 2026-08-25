// crates/mid-math/src/f64/daffine3.rs
//! Double-precision 3D affine transform — rotation · scale · translation.
//!
//! Stores a 3×3 linear matrix (x_axis, y_axis, z_axis) and a DVec3 translation.
//! The implicit bottom row [0, 0, 0, 1] is never stored or computed.
//!
//! Direct port of src/f32/affine3.rs with f32 → f64 and Vec3/Quat/Mat4
//! replaced by DVec3/DQuat/DMat4.
//!
//! Compared to DMat4 for TRS-only work:
//!   inverse()       : ~2× faster (no 4th row)
//!   mul (compose)   : ~40% fewer multiply-adds
//!   transform_point : same speed

use core::fmt;
use core::ops::Mul;

use super::dvec3::DVec3;
use crate::DQuat;
use super::dmat4::DMat4;
use super::dvec2::DEPSILON;

/// 3D double-precision affine transform.
///
/// 96 bytes, align(8). Four DVec3 fields (each now 24 bytes).
/// Faster basic ops than the 32-byte padded version at the cost of
/// AVX2 readiness — wide transforms use Vec3x4/Vec3x8 SoA types instead.
#[derive(Clone, Copy, PartialEq)]
#[repr(C, align(8))]
pub struct DAffine3 {
    pub x_axis:      DVec3,
    pub y_axis:      DVec3,
    pub z_axis:      DVec3,
    pub translation: DVec3,
}

impl DAffine3 {
    /// Identity — no rotation, no scale, no translation.
    pub const IDENTITY: Self = Self {
        x_axis:      DVec3::X,
        y_axis:      DVec3::Y,
        z_axis:      DVec3::Z,
        translation: DVec3::ZERO,
    };

    // ── Constructors ──────────────────────────────────────────────────────────

    /// Translation only.
    #[inline]
    pub fn from_translation(t: DVec3) -> Self {
        Self { x_axis: DVec3::X, y_axis: DVec3::Y, z_axis: DVec3::Z, translation: t }
    }

    /// Non-uniform scale only.
    #[inline]
    pub fn from_scale(s: DVec3) -> Self {
        Self {
            x_axis:      DVec3::new(s.x, 0.0, 0.0),
            y_axis:      DVec3::new(0.0, s.y, 0.0),
            z_axis:      DVec3::new(0.0, 0.0, s.z),
            translation: DVec3::ZERO,
        }
    }

    /// Rotation only. `q` is normalised internally.
    #[inline]
    pub fn from_rotation(q: DQuat) -> Self {
        let q = q.normalize();
        let (x, y, z, w) = (q.x, q.y, q.z, q.w);
        let (x2, y2, z2) = (x+x, y+y, z+z);
        let (xx, yy, zz) = (x*x2, y*y2, z*z2);
        let (xy, xz, yz) = (x*y2, x*z2, y*z2);
        let (wx, wy, wz) = (w*x2, w*y2, w*z2);
        Self {
            x_axis:      DVec3::new(1.0-yy-zz, xy+wz,       xz-wy),
            y_axis:      DVec3::new(xy-wz,      1.0-xx-zz,   yz+wx),
            z_axis:      DVec3::new(xz+wy,      yz-wx,       1.0-xx-yy),
            translation: DVec3::ZERO,
        }
    }

    /// Full TRS — scale, then rotate, then translate. `r` is normalised internally.
    #[inline]
    pub fn from_trs(t: DVec3, r: DQuat, s: DVec3) -> Self {
        let q = r.normalize();
        let (x, y, z, w) = (q.x, q.y, q.z, q.w);
        let (x2, y2, z2) = (x+x, y+y, z+z);
        let (xx, yy, zz) = (x*x2, y*y2, z*z2);
        let (xy, xz, yz) = (x*y2, x*z2, y*z2);
        let (wx, wy, wz) = (w*x2, w*y2, w*z2);
        Self {
            x_axis:      DVec3::new((1.0-yy-zz)*s.x, (xy+wz)*s.x,       (xz-wy)*s.x),
            y_axis:      DVec3::new((xy-wz)*s.y,      (1.0-xx-zz)*s.y,   (yz+wx)*s.y),
            z_axis:      DVec3::new((xz+wy)*s.z,      (yz-wx)*s.z,       (1.0-xx-yy)*s.z),
            translation: t,
        }
    }

    /// Extract from a DMat4 (assumes bottom row is `[0,0,0,1]`).
    #[inline]
    pub fn from_mat4(m: DMat4) -> Self {
        Self {
            x_axis:      DVec3::new(m.cols[0][0], m.cols[0][1], m.cols[0][2]),
            y_axis:      DVec3::new(m.cols[1][0], m.cols[1][1], m.cols[1][2]),
            z_axis:      DVec3::new(m.cols[2][0], m.cols[2][1], m.cols[2][2]),
            translation: DVec3::new(m.cols[3][0], m.cols[3][1], m.cols[3][2]),
        }
    }

    // ── Conversion ────────────────────────────────────────────────────────────

    /// Expand to DMat4 by appending the implicit `[0,0,0,1]` row.
    #[inline]
    pub fn to_mat4(self) -> DMat4 {
        DMat4::from_cols(
            [self.x_axis.x, self.x_axis.y, self.x_axis.z, 0.0],
            [self.y_axis.x, self.y_axis.y, self.y_axis.z, 0.0],
            [self.z_axis.x, self.z_axis.y, self.z_axis.z, 0.0],
            [self.translation.x, self.translation.y, self.translation.z, 1.0],
        )
    }

    // ── Transform helpers ─────────────────────────────────────────────────────

    /// Apply to a point — scale, rotation, and translation.
    #[inline(always)]
    pub fn transform_point(self, p: DVec3) -> DVec3 {
        self.x_axis * p.x + self.y_axis * p.y + self.z_axis * p.z + self.translation
    }

    /// Apply to a direction vector — scale and rotation only, NO translation.
    #[inline(always)]
    pub fn transform_vector(self, v: DVec3) -> DVec3 {
        self.x_axis * v.x + self.y_axis * v.y + self.z_axis * v.z
    }

    // ── Inverse ───────────────────────────────────────────────────────────────

    /// Inverse of a TRS affine transform.
    ///
    /// ~2× faster than `DMat4::inverse` because the implicit `[0,0,0,1]` row
    /// is never computed. Valid for translation + rotation + non-zero scale.
    /// Does NOT handle shear.
    ///
    /// # Derivation
    ///
    /// For M = R × S (the stored 3×3 where axis_j = R[:,j] × sj):
    /// ```text
    /// M^-1 = S^-1 × R^T
    /// (M^-1)[i,j] = axis_j[i] / |axis_j|²
    /// inv_t = -(inv_matrix3 × original_t)
    /// ```
    #[inline]
    pub fn inverse(self) -> Self {
        let sx2 = self.x_axis.length_sq();
        let sy2 = self.y_axis.length_sq();
        let sz2 = self.z_axis.length_sq();

        let isx = if sx2 < DEPSILON { 0.0 } else { 1.0 / sx2 };
        let isy = if sy2 < DEPSILON { 0.0 } else { 1.0 / sy2 };
        let isz = if sz2 < DEPSILON { 0.0 } else { 1.0 / sz2 };

        // Each new column is a *row* of the original 3×3, scaled by
        // the corresponding column's inverse squared length.
        let inv_x = DVec3::new(
            self.x_axis.x * isx, self.y_axis.x * isy, self.z_axis.x * isz,
        );
        let inv_y = DVec3::new(
            self.x_axis.y * isx, self.y_axis.y * isy, self.z_axis.y * isz,
        );
        let inv_z = DVec3::new(
            self.x_axis.z * isx, self.y_axis.z * isy, self.z_axis.z * isz,
        );

        let t = self.translation;
        let inv_t = -(inv_x * t.x + inv_y * t.y + inv_z * t.z);

        Self {
            x_axis:      inv_x,
            y_axis:      inv_y,
            z_axis:      inv_z,
            translation: inv_t,
        }
    }

    // ── Predicates ────────────────────────────────────────────────────────────

    #[inline]
    pub fn is_finite(self) -> bool {
        self.x_axis.is_finite()
            && self.y_axis.is_finite()
            && self.z_axis.is_finite()
            && self.translation.is_finite()
    }

    // ── Cast ─────────────────────────────────────────────────────────────────

    /// Lossy cast to single-precision `Affine3`.
    ///
    /// Axes are copied directly — scale is already baked into axis lengths,
    /// matching the f32 Affine3 internal layout exactly.
    #[inline]
    pub fn as_affine3(self) -> crate::Affine3 {
        crate::Affine3 {
            x_axis:      self.x_axis.as_vec3(),
            y_axis:      self.y_axis.as_vec3(),
            z_axis:      self.z_axis.as_vec3(),
            translation: self.translation.as_vec3(),
        }
    }

    /// Large World Coordinates: the real "View Space Shift" step —
    /// shifts `self` so `origin` (typically the camera's own current
    /// `DAffine3`/`DVec3` position) becomes the new coordinate zero,
    /// composed in f64, *then* truncates to f32 via [`Self::as_affine3`].
    /// Unlike calling `as_affine3` directly, this is safe regardless of
    /// how far `self` is from the *world* origin: rotation and scale
    /// (`matrix3`) are unaffected by the shift (they're not
    /// position-magnitude-dependent, so they never needed f64 in the
    /// first place) — only `translation` is shifted, from a
    /// world-magnitude value down to a small, camera-relative one,
    /// which is what makes truncating it to f32 safe. Precision is
    /// highest exactly where `origin` is, i.e. right where the camera
    /// is looking, which is the whole point of calling this once per
    /// frame with the camera's own current transform before sending
    /// per-vertex or per-instance data to the GPU.
    #[inline]
    pub fn to_view_relative(self, origin: DVec3) -> crate::Affine3 {
        (Self::from_translation(-origin) * self).as_affine3()
    }
}

// ── Mul: compose two affine transforms ───────────────────────────────────────
//
// `self * rhs` applies rhs first, then self.
//
// result.matrix3     = self.matrix3 × rhs.matrix3
// result.translation = self.transform_point(rhs.translation)

impl Mul for DAffine3 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self {
            x_axis:      self.transform_vector(rhs.x_axis),
            y_axis:      self.transform_vector(rhs.y_axis),
            z_axis:      self.transform_vector(rhs.z_axis),
            translation: self.transform_point(rhs.translation),
        }
    }
}

impl Default for DAffine3 {
    #[inline] fn default() -> Self { Self::IDENTITY }
}

impl fmt::Debug for DAffine3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DAffine3")
            .field("x_axis",      &self.x_axis)
            .field("y_axis",      &self.y_axis)
            .field("z_axis",      &self.z_axis)
            .field("translation", &self.translation)
            .finish()
    }
}

impl fmt::Display for DAffine3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let p = f.precision().unwrap_or(6);
        write!(
            f,
            "DAffine3 {{ x:{:.*?} y:{:.*?} z:{:.*?} t:{:.*?} }}",
            p, self.x_axis, p, self.y_axis, p, self.z_axis, p, self.translation
        )
    }
}

impl From<DMat4> for DAffine3 {
    #[inline] fn from(m: DMat4) -> Self { Self::from_mat4(m) }
}

impl From<DAffine3> for DMat4 {
    #[inline] fn from(a: DAffine3) -> Self { a.to_mat4() }
       }
