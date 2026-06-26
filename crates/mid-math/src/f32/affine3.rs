// crates/mid-math/src/f32/affine3.rs
//! 3D affine transform — rotation · scale · translation, no shear.
//!
//! Stores a 3×3 linear matrix (x_axis, y_axis, z_axis) and a Vec3 translation.
//! The implicit bottom row [0, 0, 0, 1] is never stored or computed.

use core::fmt;
use core::ops::{Mul, MulAssign};

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use crate::{Mat4, Quat, Vec3};
use crate::EPSILON;

/// 3D affine transform. 64 bytes, 16-byte aligned.
///
/// On x86/x86_64 the Vec3 fields are `__m128`-backed, so all operations
/// naturally hit SSE2 without extra loads.
///
/// **C interop:** use `CAffine3` at the FFI boundary.
#[derive(Clone, Copy, PartialEq)]
#[repr(C, align(16))]
pub struct Affine3 {
    pub x_axis:      Vec3,
    pub y_axis:      Vec3,
    pub z_axis:      Vec3,
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

    /// Fully degenerate zero transform (not useful for transforms, provided for completeness).
    pub const ZERO: Self = Self {
        x_axis:      Vec3::ZERO,
        y_axis:      Vec3::ZERO,
        z_axis:      Vec3::ZERO,
        translation: Vec3::ZERO,
    };

    pub const NAN: Self = Self {
        x_axis:      Vec3::NAN,
        y_axis:      Vec3::NAN,
        z_axis:      Vec3::NAN,
        translation: Vec3::NAN,
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

    /// Rotation only from quaternion. `q` is normalised internally.
    #[inline]
    pub fn from_rotation(q: Quat) -> Self {
        let q = q.normalize();
        let (x,y,z,w) = (q.x,q.y,q.z,q.w);
        let (x2,y2,z2) = (x+x,y+y,z+z);
        let (xx,yy,zz) = (x*x2,y*y2,z*z2);
        let (xy,xz,yz) = (x*y2,x*z2,y*z2);
        let (wx,wy,wz) = (w*x2,w*y2,w*z2);
        Self {
            x_axis:      Vec3::new(1.0-yy-zz, xy+wz,     xz-wy),
            y_axis:      Vec3::new(xy-wz,      1.0-xx-zz, yz+wx),
            z_axis:      Vec3::new(xz+wy,      yz-wx,     1.0-xx-yy),
            translation: Vec3::ZERO,
        }
    }

    /// Alias for `from_rotation` (glam-compat name).
    #[inline] pub fn from_quat(q: Quat) -> Self { Self::from_rotation(q) }

    /// Rotation-only affine from axis + angle (axis need not be unit).
    #[inline]
    pub fn from_axis_angle(axis: Vec3, angle: f32) -> Self {
        let m = crate::Mat3::from_axis_angle(axis, angle);
        Self::from_mat3(m)
    }

    #[inline]
    pub fn from_rotation_x(angle: f32) -> Self {
        Self::from_mat3(crate::Mat3::from_rotation_x(angle))
    }
    #[inline]
    pub fn from_rotation_y(angle: f32) -> Self {
        Self::from_mat3(crate::Mat3::from_rotation_y(angle))
    }
    #[inline]
    pub fn from_rotation_z(angle: f32) -> Self {
        Self::from_mat3(crate::Mat3::from_rotation_z(angle))
    }

    /// Full TRS — scale, then rotate, then translate. `r` is normalised internally.
    /// Same semantics as `from_scale_rotation_translation`.
    #[inline]
    pub fn from_trs(t: Vec3, r: Quat, s: Vec3) -> Self {
        let q = r.normalize();
        let (x,y,z,w) = (q.x,q.y,q.z,q.w);
        let (x2,y2,z2) = (x+x,y+y,z+z);
        let (xx,yy,zz) = (x*x2,y*y2,z*z2);
        let (xy,xz,yz) = (x*y2,x*z2,y*z2);
        let (wx,wy,wz) = (w*x2,w*y2,w*z2);
        Self {
            x_axis:      Vec3::new((1.0-yy-zz)*s.x, (xy+wz)*s.x, (xz-wy)*s.x),
            y_axis:      Vec3::new((xy-wz)*s.y, (1.0-xx-zz)*s.y, (yz+wx)*s.y),
            z_axis:      Vec3::new((xz+wy)*s.z, (yz-wx)*s.z, (1.0-xx-yy)*s.z),
            translation: t,
        }
    }

    /// glam-compatible argument order: `(scale, rotation, translation)`.
    #[inline]
    pub fn from_scale_rotation_translation(s: Vec3, r: Quat, t: Vec3) -> Self {
        Self::from_trs(t, r, s)
    }

    /// Rotation + translation only (scale = 1).
    #[inline]
    pub fn from_rotation_translation(r: Quat, t: Vec3) -> Self {
        Self::from_trs(t, r, Vec3::ONE)
    }

    /// Build from a `Mat3` (3×3 rotation/scale), translation = ZERO.
    #[inline]
    pub fn from_mat3(m: crate::Mat3) -> Self {
        Self {
            x_axis:      Vec3::new(m.cols[0][0], m.cols[0][1], m.cols[0][2]),
            y_axis:      Vec3::new(m.cols[1][0], m.cols[1][1], m.cols[1][2]),
            z_axis:      Vec3::new(m.cols[2][0], m.cols[2][1], m.cols[2][2]),
            translation: Vec3::ZERO,
        }
    }

    /// Build from a `Mat3` plus a translation.
    #[inline]
    pub fn from_mat3_translation(m: crate::Mat3, t: Vec3) -> Self {
        Self {
            x_axis:      Vec3::new(m.cols[0][0], m.cols[0][1], m.cols[0][2]),
            y_axis:      Vec3::new(m.cols[1][0], m.cols[1][1], m.cols[1][2]),
            z_axis:      Vec3::new(m.cols[2][0], m.cols[2][1], m.cols[2][2]),
            translation: t,
        }
    }

    /// Extract from a Mat4. Assumes bottom row is `[0, 0, 0, 1]`.
    #[inline]
    pub fn from_mat4(m: Mat4) -> Self {
        Self {
            x_axis:      m.x_axis.truncate(),
            y_axis:      m.y_axis.truncate(),
            z_axis:      m.z_axis.truncate(),
            translation: m.w_axis.truncate(),
        }
    }

    // ── Camera / view constructors ────────────────────────────────────────────

    /// Right-handed look-to view transform (full affine including eye translation).
    #[inline]
    pub fn look_to_rh(eye: Vec3, dir: Vec3, up: Vec3) -> Self {
        let f = dir.normalize();
        let r = f.cross(up).normalize();
        let u = r.cross(f);
        Self {
            x_axis:      Vec3::new(r.x, u.x, -f.x),
            y_axis:      Vec3::new(r.y, u.y, -f.y),
            z_axis:      Vec3::new(r.z, u.z, -f.z),
            translation: Vec3::new(-r.dot(eye), -u.dot(eye), f.dot(eye)),
        }
    }

    #[inline] pub fn look_to_lh(eye: Vec3, dir: Vec3, up: Vec3) -> Self { Self::look_to_rh(eye, -dir, up) }
    #[inline] pub fn look_at_rh(eye: Vec3, center: Vec3, up: Vec3) -> Self { Self::look_to_rh(eye, center-eye, up) }
    #[inline] pub fn look_at_lh(eye: Vec3, center: Vec3, up: Vec3) -> Self { Self::look_to_lh(eye, center-eye, up) }

    // ── Conversion ────────────────────────────────────────────────────────────

    /// Expand to Mat4 by appending the implicit `[0, 0, 0, 1]` row.
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
    #[inline(always)]
    pub fn transform_point(self, p: Vec3) -> Vec3 {
        self.x_axis * p.x + self.y_axis * p.y + self.z_axis * p.z + self.translation
    }

    /// Alias for `transform_point` (glam-compat name).
    #[inline(always)] pub fn transform_point3(self, p: Vec3) -> Vec3 { self.transform_point(p) }

    /// Apply to a direction — scale and rotation only, NO translation.
    #[inline(always)]
    pub fn transform_vector(self, v: Vec3) -> Vec3 {
        self.x_axis * v.x + self.y_axis * v.y + self.z_axis * v.z
    }

    /// Alias for `transform_vector` (glam-compat name).
    #[inline(always)] pub fn transform_vector3(self, v: Vec3) -> Vec3 { self.transform_vector(v) }

    // ── Decompose ─────────────────────────────────────────────────────────────

    /// Decompose into `(scale, rotation, translation)`.
    ///
    /// Scale is the length of each axis. Rotation is extracted via Shepperd's method.
    /// Does not handle shear correctly — axes must be orthogonal for rotation to be valid.
    #[inline]
    pub fn decompose(self) -> (Vec3, Quat, Vec3) {
        let t  = self.translation;
        let sx = self.x_axis.length();
        let sy = self.y_axis.length();
        let sz = self.z_axis.length();
        let inv_sx = if sx < EPSILON { 0.0 } else { 1.0 / sx };
        let inv_sy = if sy < EPSILON { 0.0 } else { 1.0 / sy };
        let inv_sz = if sz < EPSILON { 0.0 } else { 1.0 / sz };
        // Build normalised rotation as a Mat3 then convert to quat.
        let rot = crate::Mat3 { cols: [
            [self.x_axis.x*inv_sx, self.x_axis.y*inv_sx, self.x_axis.z*inv_sx],
            [self.y_axis.x*inv_sy, self.y_axis.y*inv_sy, self.y_axis.z*inv_sy],
            [self.z_axis.x*inv_sz, self.z_axis.y*inv_sz, self.z_axis.z*inv_sz],
        ]};
        (Vec3::new(sx, sy, sz), rot.to_quat(), t)
    }

    // ── Inverse ───────────────────────────────────────────────────────────────

    /// Inverse of a TRS affine transform — SSE2 fast path on x86/x86_64.
    /// Valid for translation + rotation + non-zero scale. Does NOT handle shear.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[inline]
    pub fn inverse(self) -> Self {
        unsafe {
            let c0 = self.x_axis.0;
            let c1 = self.y_axis.0;
            let c2 = self.z_axis.0;
            let c3 = self.translation.0;

            let sq0  = _mm_mul_ps(c0, c0);
            let sq1  = _mm_mul_ps(c1, c1);
            let sq2  = _mm_mul_ps(c2, c2);
            let zero = _mm_setzero_ps();

            let lo01 = _mm_unpacklo_ps(sq0, sq1);
            let lo2z = _mm_unpacklo_ps(sq2, zero);
            let hi01 = _mm_unpackhi_ps(sq0, sq1);
            let hi2z = _mm_unpackhi_ps(sq2, zero);
            let row0 = _mm_movelh_ps(lo01, lo2z);
            let row1 = _mm_movehl_ps(lo2z, lo01);
            let row2 = _mm_movelh_ps(hi01, hi2z);
            let sums = _mm_add_ps(_mm_add_ps(row0, row1), row2);

            let eps  = _mm_set1_ps(EPSILON);
            let mask = _mm_cmpge_ps(sums, eps);
            let safe = _mm_or_ps(
                _mm_and_ps(mask, sums),
                _mm_andnot_ps(mask, _mm_set1_ps(1.0)),
            );
            let inv_scales = _mm_and_ps(mask, _mm_div_ps(_mm_set1_ps(1.0), safe));

            let lo01_r = _mm_unpacklo_ps(c0, c1);
            let lo2z_r = _mm_unpacklo_ps(c2, zero);
            let hi01_r = _mm_unpackhi_ps(c0, c1);
            let hi2z_r = _mm_unpackhi_ps(c2, zero);
            let trow0 = _mm_movelh_ps(lo01_r, lo2z_r);
            let trow1 = _mm_movehl_ps(lo2z_r, lo01_r);
            let trow2 = _mm_movelh_ps(hi01_r, hi2z_r);

            let ic0 = _mm_mul_ps(trow0, inv_scales);
            let ic1 = _mm_mul_ps(trow1, inv_scales);
            let ic2 = _mm_mul_ps(trow2, inv_scales);

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

    /// Inverse — portable scalar fallback (non-x86 targets).
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    #[inline]
    pub fn inverse(self) -> Self { self.inverse_scalar() }

    /// Invert or return `IDENTITY` if a scale axis is degenerate.
    #[inline]
    pub fn inverse_or_identity(self) -> Self {
        if self.x_axis.length_sq() < EPSILON
            || self.y_axis.length_sq() < EPSILON
            || self.z_axis.length_sq() < EPSILON
        {
            Self::IDENTITY
        } else {
            self.inverse()
        }
    }

    /// Scalar inverse — portable reference. Valid for TRS, no shear.
    #[inline]
    pub fn inverse_scalar(self) -> Self {
        let sx2 = self.x_axis.length_sq();
        let sy2 = self.y_axis.length_sq();
        let sz2 = self.z_axis.length_sq();
        let isx = if sx2 < EPSILON { 0.0 } else { 1.0 / sx2 };
        let isy = if sy2 < EPSILON { 0.0 } else { 1.0 / sy2 };
        let isz = if sz2 < EPSILON { 0.0 } else { 1.0 / sz2 };
        let inv_x = Vec3::new(self.x_axis.x*isx, self.y_axis.x*isy, self.z_axis.x*isz);
        let inv_y = Vec3::new(self.x_axis.y*isx, self.y_axis.y*isy, self.z_axis.y*isz);
        let inv_z = Vec3::new(self.x_axis.z*isx, self.y_axis.z*isy, self.z_axis.z*isz);
        let t     = self.translation;
        let inv_t = -(inv_x*t.x + inv_y*t.y + inv_z*t.z);
        Self { x_axis: inv_x, y_axis: inv_y, z_axis: inv_z, translation: inv_t }
    }

    // ── Predicates ────────────────────────────────────────────────────────────

    #[inline]
    pub fn is_finite(self) -> bool {
        self.x_axis.is_finite() && self.y_axis.is_finite()
            && self.z_axis.is_finite() && self.translation.is_finite()
    }

    #[inline]
    pub fn is_nan(self) -> bool {
        self.x_axis.is_nan() || self.y_axis.is_nan()
            || self.z_axis.is_nan() || self.translation.is_nan()
    }

    #[inline]
    pub fn abs_diff_eq(self, rhs: Self, max_abs_diff: f32) -> bool {
        self.x_axis.abs_diff_eq(rhs.x_axis, max_abs_diff)
            && self.y_axis.abs_diff_eq(rhs.y_axis, max_abs_diff)
            && self.z_axis.abs_diff_eq(rhs.z_axis, max_abs_diff)
            && self.translation.abs_diff_eq(rhs.translation, max_abs_diff)
    }

    #[inline]
    pub fn approx_eq(self, rhs: Self) -> bool { self.abs_diff_eq(rhs, EPSILON) }
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

impl MulAssign for Affine3 {
    #[inline] fn mul_assign(&mut self, rhs: Self) { *self = *self * rhs; }
}

/// Transform a point (applies scale, rotation, and translation).
impl Mul<Vec3> for Affine3 {
    type Output = Vec3;
    #[inline(always)] fn mul(self, rhs: Vec3) -> Vec3 { self.transform_point(rhs) }
}

// ── Default / Debug / Display / From ─────────────────────────────────────────

impl Default for Affine3 { fn default() -> Self { Self::IDENTITY } }

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
        write!(f, "Affine3 {{ x:{:.*?} y:{:.*?} z:{:.*?} t:{:.*?} }}",
            p, self.x_axis, p, self.y_axis, p, self.z_axis, p, self.translation)
    }
}

impl From<Mat4>    for Affine3 { fn from(m: Mat4)    -> Self { Self::from_mat4(m)     } }
impl From<Affine3> for Mat4    { fn from(a: Affine3) -> Self { a.to_mat4()            } }
