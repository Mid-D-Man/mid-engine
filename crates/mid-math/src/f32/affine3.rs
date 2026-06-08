// crates/mid-math/src/f32/affine3.rs
//! 3D affine transform — rotation · scale · translation.
//!
//! Stores a 3×3 linear matrix (x_axis, y_axis, z_axis) and a Vec3 translation.
//! The implicit bottom row [0, 0, 0, 1] is never stored or computed.
//!
//! Updated for Build 8: from_mat4 and to_mat4 use Vec3::truncate / Vec3::extend
//! instead of element-by-element access, matching the new Mat4 Vec4-field layout.

use core::fmt;
use core::ops::Mul;

use crate::{Mat4, Quat, Vec3};
use crate::EPSILON;

/// 3D affine transform.
///
/// 64 bytes, 16-byte aligned. On x86_64 the Vec3 fields are `__m128`-backed,
/// so all operations use SSE2 without any extra loads.
///
/// **C interop:** use [`CAffine3`][crate::ffi::types::CAffine3] at the FFI boundary.
#[derive(Clone, Copy, PartialEq)]
#[repr(C, align(16))]
pub struct Affine3 {
    /// First column of the 3×3 matrix (x-basis scaled by sx and rotated).
    pub x_axis: Vec3,
    /// Second column of the 3×3 matrix (y-basis scaled by sy and rotated).
    pub y_axis: Vec3,
    /// Third column of the 3×3 matrix (z-basis scaled by sz and rotated).
    pub z_axis: Vec3,
    /// Translation component (applied after the linear transform).
    pub translation: Vec3,
}

impl Affine3 {
    /// Identity — no rotation, no scale, no translation.
    pub const IDENTITY: Self = Self {
        x_axis:      Vec3::X,
        y_axis:      Vec3::Y,
        z_axis:      Vec3::Z,
        translation: Vec3::ZERO,
    };

    // ── Constructors ──────────────────────────────────────────────────────────

    /// Translation only.
    #[inline]
    pub fn from_translation(t: Vec3) -> Self {
        Self { x_axis: Vec3::X, y_axis: Vec3::Y, z_axis: Vec3::Z, translation: t }
    }

    /// Non-uniform scale only.
    #[inline]
    pub fn from_scale(s: Vec3) -> Self {
        Self {
            x_axis:      Vec3::new(s.x, 0.0, 0.0),
            y_axis:      Vec3::new(0.0, s.y, 0.0),
            z_axis:      Vec3::new(0.0, 0.0, s.z),
            translation: Vec3::ZERO,
        }
    }

    /// Rotation only. `q` is normalised internally.
    #[inline]
    pub fn from_rotation(q: Quat) -> Self {
        let q = q.normalize();
        let (x, y, z, w) = (q.x, q.y, q.z, q.w);
        let (x2, y2, z2) = (x + x, y + y, z + z);
        let (xx, yy, zz) = (x * x2, y * y2, z * z2);
        let (xy, xz, yz) = (x * y2, x * z2, y * z2);
        let (wx, wy, wz) = (w * x2, w * y2, w * z2);
        Self {
            x_axis:      Vec3::new(1.0 - yy - zz, xy + wz,       xz - wy),
            y_axis:      Vec3::new(xy - wz,        1.0 - xx - zz, yz + wx),
            z_axis:      Vec3::new(xz + wy,        yz - wx,       1.0 - xx - yy),
            translation: Vec3::ZERO,
        }
    }

    /// Full TRS — scale, then rotate, then translate. `r` is normalised internally.
    ///
    /// Equivalent to `Mat4::from_trs` but stores the result as Affine3.
    #[inline]
    pub fn from_trs(t: Vec3, r: Quat, s: Vec3) -> Self {
        let q = r.normalize();
        let (x, y, z, w) = (q.x, q.y, q.z, q.w);
        let (x2, y2, z2) = (x + x, y + y, z + z);
        let (xx, yy, zz) = (x * x2, y * y2, z * z2);
        let (xy, xz, yz) = (x * y2, x * z2, y * z2);
        let (wx, wy, wz) = (w * x2, w * y2, w * z2);
        Self {
            x_axis:      Vec3::new((1.0 - yy - zz) * s.x, (xy + wz) * s.x, (xz - wy) * s.x),
            y_axis:      Vec3::new((xy - wz) * s.y, (1.0 - xx - zz) * s.y, (yz + wx) * s.y),
            z_axis:      Vec3::new((xz + wy) * s.z, (yz - wx) * s.z, (1.0 - xx - yy) * s.z),
            translation: t,
        }
    }

    /// Extract from a Mat4.
    ///
    /// Assumes the bottom row of `m` is `[0, 0, 0, 1]`.
    ///
    /// Uses `Vec4::truncate()` (zero lane 3 via SSE2 AND-mask) rather than
    /// element-by-element scalar extraction — aligns with the Build-8 Vec4-field layout.
    #[inline]
    pub fn from_mat4(m: Mat4) -> Self {
        Self {
            x_axis:      m.x_axis.truncate(),
            y_axis:      m.y_axis.truncate(),
            z_axis:      m.z_axis.truncate(),
            translation: m.w_axis.truncate(),
        }
    }

    // ── Conversion ────────────────────────────────────────────────────────────

    /// Expand to Mat4 by appending the implicit `[0, 0, 0, 1]` row.
    ///
    /// Uses `Vec3::extend(w)` to build each Vec4 column — zero-cost SSE2 blend.
    #[inline]
    pub fn to_mat4(self) -> Mat4 {
        Mat4 {
            x_axis: self.x_axis.extend(0.0),
            y_axis: self.y_axis.extend(0.0),
            z_axis: self.z_axis.extend(0.0),
            w_axis: self.translation.extend(1.0),
        }
    }

    // ── Transform helpers ─────────────────────────────────────────────────────

    /// Apply to a point — applies scale, rotation, and translation.
    ///
    /// Equivalent to `to_mat4().transform_point(p)` but without constructing Mat4.
    #[inline(always)]
    pub fn transform_point(self, p: Vec3) -> Vec3 {
        self.x_axis * p.x + self.y_axis * p.y + self.z_axis * p.z + self.translation
    }

    /// Apply to a direction vector — applies scale and rotation only, NO translation.
    ///
    /// Use this for velocity vectors and normals (when scale is uniform).
    #[inline(always)]
    pub fn transform_vector(self, v: Vec3) -> Vec3 {
        self.x_axis * v.x + self.y_axis * v.y + self.z_axis * v.z
    }

    // ── Inverse ───────────────────────────────────────────────────────────────

    /// Inverse of a TRS affine transform.
    ///
    /// ~2× faster than `Mat4::inverse_general` because the implicit bottom row
    /// [0,0,0,1] is never computed. Valid for translation + rotation + non-zero
    /// scale. Does NOT handle shear.
    #[inline]
    pub fn inverse(self) -> Self {
        let sx2 = self.x_axis.length_sq();
        let sy2 = self.y_axis.length_sq();
        let sz2 = self.z_axis.length_sq();

        let isx = if sx2 < EPSILON { 0.0 } else { 1.0 / sx2 };
        let isy = if sy2 < EPSILON { 0.0 } else { 1.0 / sy2 };
        let isz = if sz2 < EPSILON { 0.0 } else { 1.0 / sz2 };

        let inv_x = Vec3::new(
            self.x_axis.x * isx, self.y_axis.x * isy, self.z_axis.x * isz,
        );
        let inv_y = Vec3::new(
            self.x_axis.y * isx, self.y_axis.y * isy, self.z_axis.y * isz,
        );
        let inv_z = Vec3::new(
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
}

// ── Mul: compose two affine transforms ───────────────────────────────────────

impl Mul for Affine3 {
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

impl Default for Affine3 {
    #[inline]
    fn default() -> Self { Self::IDENTITY }
}

impl fmt::Debug for Affine3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Affine3")
            .field("x_axis",      &self.x_axis)
            .field("y_axis",      &self.y_axis)
            .field("z_axis",      &self.z_axis)
            .field("translation", &self.translation)
            .finish()
    }
}

impl fmt::Display for Affine3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let p = f.precision().unwrap_or(4);
        write!(
            f,
            "Affine3 {{ x:{:.*?} y:{:.*?} z:{:.*?} t:{:.*?} }}",
            p, self.x_axis, p, self.y_axis, p, self.z_axis, p, self.translation
        )
    }
}

impl From<Mat4> for Affine3 {
    #[inline]
    fn from(m: Mat4) -> Self { Self::from_mat4(m) }
}

impl From<Affine3> for Mat4 {
    #[inline]
    fn from(a: Affine3) -> Self { a.to_mat4() }
    }
