// crates/mid-math/src/f32/affine3.rs
//! 3D affine transform — rotation · scale · translation.
//!
//! Stores a 3×3 linear matrix (x_axis, y_axis, z_axis) and a Vec3 translation.
//! The implicit bottom row [0, 0, 0, 1] is never stored or computed.
//!
//! Updated for Build 8: from_mat4 and to_mat4 use Vec3::truncate / Vec3::extend
//! instead of element-by-element access, matching the new Mat4 Vec4-field layout.
//!
//! `inverse()` gained an SSE2 fast path on x86/x86_64, ported from
//! `Mat4::inverse_trs`'s column-transpose + masked-reciprocal-scale algorithm
//! (see the Inverse section below). `inverse_scalar()` retains the original
//! portable implementation as a fallback and correctness cross-check reference.

use core::fmt;
use core::ops::Mul;

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

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

    /// Inverse of a TRS affine transform — SSE2 fast path.
    ///
    /// ~2× faster than `Mat4::inverse_general` because the implicit bottom row
    /// [0,0,0,1] is never computed. Valid for translation + rotation + non-zero
    /// scale. Does NOT handle shear.
    ///
    /// # Algorithm
    ///
    /// Ported verbatim from `Mat4::inverse_trs`'s 3-column transpose
    /// (`unpacklo`/`unpackhi`/`movelh`/`movehl`) + masked-reciprocal scale +
    /// dot-product translation negate. The Mat4 version finishes by forcing
    /// the translation column's lane 3 to `1.0` (homogeneous row contract).
    /// Affine3 has no implicit row, so that fixup is dropped entirely:
    ///
    /// - `x_axis`/`y_axis`/`z_axis`/`translation` are `Vec3` ⇒ lane 3 = 0 going in.
    /// - `sums[3]` (sum of squared lane-3 components) is therefore `0`.
    /// - `0 < EPSILON` ⇒ the reciprocal mask is false for lane 3 ⇒ `inv_scales[3] = 0`.
    /// - Every output column's lane 3 = `(transposed row lane 3) * inv_scales[3]
    ///   = 0 * 0 = 0`, and the translation's lane 3 = `0 - dot_col[3] = -0.0`.
    ///
    /// Both `0` and `-0.0` are correct/ignored padding for `Vec3` (its `PartialEq`,
    /// `Debug`, `Display`, and arithmetic all operate on lanes 0-2 only), so lane 3
    /// stays naturally well-formed with zero extra masking.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[inline]
    pub fn inverse(self) -> Self {
        unsafe {
            let c0 = self.x_axis.0;
            let c1 = self.y_axis.0;
            let c2 = self.z_axis.0;
            let c3 = self.translation.0;

            // Squared lengths of the three rotation columns.
            let sq0  = _mm_mul_ps(c0, c0);
            let sq1  = _mm_mul_ps(c1, c1);
            let sq2  = _mm_mul_ps(c2, c2);
            let zero = _mm_setzero_ps();

            // Horizontal sum: sums[i] = sq_i.x + sq_i.y + sq_i.z
            // Using 3-way transpose + column-wise add.
            let lo01 = _mm_unpacklo_ps(sq0, sq1);
            let lo2z = _mm_unpacklo_ps(sq2, zero);
            let hi01 = _mm_unpackhi_ps(sq0, sq1);
            let hi2z = _mm_unpackhi_ps(sq2, zero);
            let row0 = _mm_movelh_ps(lo01, lo2z); // [sq0.x, sq1.x, sq2.x, 0]
            let row1 = _mm_movehl_ps(lo2z, lo01); // [sq0.y, sq1.y, sq2.y, 0]
            let row2 = _mm_movelh_ps(hi01, hi2z); // [sq0.z, sq1.z, sq2.z, 0]
            let sums = _mm_add_ps(_mm_add_ps(row0, row1), row2);
            // sums = [sx², sy², sz², 0]

            // Safe reciprocals: guard against near-zero scale.
            let eps  = _mm_set1_ps(EPSILON);
            let mask = _mm_cmpge_ps(sums, eps);
            let safe = _mm_or_ps(
                _mm_and_ps(mask, sums),
                _mm_andnot_ps(mask, _mm_set1_ps(1.0)),
            );
            let inv_scales = _mm_and_ps(mask, _mm_div_ps(_mm_set1_ps(1.0), safe));
            // inv_scales = [1/sx², 1/sy², 1/sz², 0]

            // Transpose the 3×3 of the rotation columns.
            let lo01_r = _mm_unpacklo_ps(c0, c1);
            let lo2z_r = _mm_unpacklo_ps(c2, zero);
            let hi01_r = _mm_unpackhi_ps(c0, c1);
            let hi2z_r = _mm_unpackhi_ps(c2, zero);
            let trow0 = _mm_movelh_ps(lo01_r, lo2z_r); // [c0.x, c1.x, c2.x, 0]
            let trow1 = _mm_movehl_ps(lo2z_r, lo01_r); // [c0.y, c1.y, c2.y, 0]
            let trow2 = _mm_movelh_ps(hi01_r, hi2z_r); // [c0.z, c1.z, c2.z, 0]

            // Scale each transposed row by the corresponding inverse squared scale.
            let ic0 = _mm_mul_ps(trow0, inv_scales);
            let ic1 = _mm_mul_ps(trow1, inv_scales);
            let ic2 = _mm_mul_ps(trow2, inv_scales);

            // Inverse translation: -(inv_rot × original_t)
            let tx = _mm_shuffle_ps::<0b00_00_00_00>(c3, c3);
            let ty = _mm_shuffle_ps::<0b01_01_01_01>(c3, c3);
            let tz = _mm_shuffle_ps::<0b10_10_10_10>(c3, c3);
            let dot_col = _mm_add_ps(
                _mm_add_ps(_mm_mul_ps(ic0, tx), _mm_mul_ps(ic1, ty)),
                _mm_mul_ps(ic2, tz),
            );
            let inv_t = _mm_sub_ps(zero, dot_col);

            Self {
                x_axis:      Vec3(ic0),
                y_axis:      Vec3(ic1),
                z_axis:      Vec3(ic2),
                translation: Vec3(inv_t),
            }
        }
    }

    /// Inverse of a TRS affine transform — portable fallback for non-x86 targets.
    ///
    /// Identical contract to the SSE2 `inverse()`: valid for translation +
    /// rotation + non-zero scale, does NOT handle shear.
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    #[inline]
    pub fn inverse(self) -> Self {
        self.inverse_scalar()
    }

    /// Scalar fallback inverse — exact same algorithm as the SSE2 path, no SIMD.
    ///
    /// Kept as a portable reference implementation for non-x86 targets and as
    /// a correctness cross-check against the SSE2 `inverse()` (both produce
    /// bit-equivalent results).
    ///
    /// # Derivation
    ///
    /// For M = R × S (the stored 3×3):
    /// ```text
    /// M^-1 = S^-1 × R^T
    /// (M^-1)[i,j] = axis_j[i] / |axis_j|²
    /// inv_t       = -(M^-1 × original_t)
    /// ```
    #[inline]
    pub fn inverse_scalar(self) -> Self {
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
