// crates/mid-math/src/f32/dual_quat.rs
//! Dual quaternions for f32 — rigid body skinning with dual-linear blending (DLB).
//!
//! Moved from helpers/dual_quat.rs — lives in f32/ because it depends on
//! the platform-dispatched f32 Quat and Vec3 types.
//!
//! A dual quaternion Q = q_real + ε·q_dual encodes:
//!   q_real = rotation quaternion
//!   q_dual = 0.5 * t * q_real  (translation encoded in dual part)
//!
//! # Skinning workflow
//!
//!   1. Build per-bone `DualQuat::from_rotation_translation(rot, pos)`.
//!   2. Blend across influences with `blend2` or `blend4`.
//!   3. Call `transform_point(vertex_pos)` on the blended result.
//!
//! # Reference
//!   Kavan et al. "Skinning with Dual Quaternions" (2007) — the standard
//!   algorithm used in most game engines that avoid SQT blending artifacts.

use core::fmt;
use core::ops::Mul;
use crate::{Quat, Vec3, EPSILON};

/// Single-precision dual quaternion. 32 bytes, 16-byte aligned.
///
/// `real` encodes rotation. `dual` encodes translation as `0.5 * t * real`.
#[derive(Clone, Copy)]
#[repr(C, align(16))]
pub struct DualQuat {
    pub real: Quat,
    pub dual: Quat,
}

impl DualQuat {
    /// Identity — no rotation, no translation.
    pub const IDENTITY: Self = Self {
        real: Quat::IDENTITY,
        dual: Quat::ZERO,
    };

    // ── Constructors ──────────────────────────────────────────────────────────

    /// Build from rotation + translation.
    ///
    /// `rotation` is normalised internally.
    #[inline]
    pub fn from_rotation_translation(rotation: Quat, translation: Vec3) -> Self {
        let r = rotation.normalize();
        // dual = 0.5 * pure_quat(translation) * r
        // pure_quat(t) = Quat(t.x, t.y, t.z, 0)
        let t = Quat::new(translation.x, translation.y, translation.z, 0.0);
        let dual = Quat::new(
            0.5 * ( t.w * r.x + t.x * r.w + t.y * r.z - t.z * r.y),
            0.5 * ( t.w * r.y - t.x * r.z + t.y * r.w + t.z * r.x),
            0.5 * ( t.w * r.z + t.x * r.y - t.y * r.x + t.z * r.w),
            0.5 * (-t.x * r.x - t.y * r.y - t.z * r.z),
        );
        Self { real: r, dual }
    }

    /// Rotation only — no translation.
    #[inline]
    pub fn from_rotation(rotation: Quat) -> Self {
        Self { real: rotation.normalize(), dual: Quat::ZERO }
    }

    /// Translation only — no rotation.
    #[inline]
    pub fn from_translation(translation: Vec3) -> Self {
        Self::from_rotation_translation(Quat::IDENTITY, translation)
    }

    // ── Decomposition ─────────────────────────────────────────────────────────

    /// Extract the rotation quaternion (normalised).
    #[inline]
    pub fn rotation(self) -> Quat { self.real.normalize() }

    /// Extract the translation vector.
    #[inline]
    pub fn translation(self) -> Vec3 {
        let r = self.real;
        let d = self.dual;
        // t = 2 * d * r*  (r* = conjugate of r)
        Vec3::new(
            2.0 * (-d.w * r.x + d.x * r.w - d.y * r.z + d.z * r.y),
            2.0 * (-d.w * r.y + d.x * r.z + d.y * r.w - d.z * r.x),
            2.0 * (-d.w * r.z - d.x * r.y + d.y * r.x + d.z * r.w),
        )
    }

    // ── Transform ─────────────────────────────────────────────────────────────

    /// Transform a point (applies rotation then translation).
    #[inline]
    pub fn transform_point(self, p: Vec3) -> Vec3 {
        self.real.rotate(p) + self.translation()
    }

    /// Transform a direction vector (rotation only, no translation).
    #[inline]
    pub fn transform_vector(self, v: Vec3) -> Vec3 {
        self.real.rotate(v)
    }

    // ── Normalisation ──────────────────────────────────────────────────────────

    /// Normalise so that `|real| = 1`.
    ///
    /// Required before use if `blend2`/`blend4` was called, because the
    /// interpolated sum is not unit by construction.
    #[inline]
    pub fn normalize(self) -> Self {
        let mag = self.real.length();
        if mag < EPSILON { return Self::IDENTITY; }
        let inv = 1.0 / mag;
        Self {
            real: Quat::new(self.real.x*inv, self.real.y*inv, self.real.z*inv, self.real.w*inv),
            dual: Quat::new(self.dual.x*inv, self.dual.y*inv, self.dual.z*inv, self.dual.w*inv),
        }
    }

    // ── Dual-linear blending (DLB) ────────────────────────────────────────────

    /// Scale all components by a scalar weight.
    #[inline]
    pub fn scale(self, w: f32) -> Self {
        Self {
            real: Quat::new(self.real.x*w, self.real.y*w, self.real.z*w, self.real.w*w),
            dual: Quat::new(self.dual.x*w, self.dual.y*w, self.dual.z*w, self.dual.w*w),
        }
    }

    /// Component-wise add (used during blending accumulation).
    #[inline]
    pub fn add_dq(self, rhs: Self) -> Self {
        Self {
            real: Quat::new(self.real.x+rhs.real.x, self.real.y+rhs.real.y,
                            self.real.z+rhs.real.z, self.real.w+rhs.real.w),
            dual: Quat::new(self.dual.x+rhs.dual.x, self.dual.y+rhs.dual.y,
                            self.dual.z+rhs.dual.z, self.dual.w+rhs.dual.w),
        }
    }

    /// Dual-linear blend of two dual quaternions with weights.
    ///
    /// Ensures shortest-path interpolation via antipodal flip on the real part.
    /// Result is normalised.
    #[inline]
    pub fn blend2(dq0: Self, w0: f32, dq1: Self, w1: f32) -> Self {
        let dot = dq0.real.x*dq1.real.x + dq0.real.y*dq1.real.y
                + dq0.real.z*dq1.real.z + dq0.real.w*dq1.real.w;
        let dq1 = if dot < 0.0 { dq1.scale(-1.0) } else { dq1 };
        dq0.scale(w0).add_dq(dq1.scale(w1)).normalize()
    }

    /// Dual-linear blend of up to four weighted dual quaternions.
    ///
    /// All influences are flipped to the same hemisphere as `influences[0]`
    /// before blending. Result is normalised.
    pub fn blend4(influences: [(Self, f32); 4]) -> Self {
        let (ref_dq, _) = influences[0];
        let pivot_dot = |a: &Self, b: &Self| -> f32 {
            a.real.x*b.real.x + a.real.y*b.real.y
          + a.real.z*b.real.z + a.real.w*b.real.w
        };
        let mut acc = Self::IDENTITY.scale(0.0);
        for (dq, w) in &influences {
            let sign = if pivot_dot(&ref_dq, dq) < 0.0 { -1.0 } else { 1.0 };
            acc = acc.add_dq(dq.scale(w * sign));
        }
        acc.normalize()
    }

    // ── Composition ───────────────────────────────────────────────────────────

    /// Compose two dual quaternions: `self * rhs` applies rhs first, then self.
    pub fn mul_dual_quat(self, rhs: Self) -> Self {
        let (a, b, c, d) = (self.real.w, self.real.x, self.real.y, self.real.z);
        let (e, f, g, h) = (rhs.real.w, rhs.real.x, rhs.real.y, rhs.real.z);
        let (da, db, dc, dd) = (self.dual.w, self.dual.x, self.dual.y, self.dual.z);
        let (de, df, dg, dh) = (rhs.dual.w, rhs.dual.x, rhs.dual.y, rhs.dual.z);
        Self {
            real: Quat::new(
                a*f + b*e + c*h - d*g,
                a*g - b*h + c*e + d*f,
                a*h + b*g - c*f + d*e,
                a*e - b*f - c*g - d*h,
            ),
            dual: Quat::new(
                a*df + b*de + c*dh - d*dg + da*f + db*e + dc*h - dd*g,
                a*dg - b*dh + c*de + d*df + da*g - db*h + dc*e + dd*f,
                a*dh + b*dg - c*df + d*de + da*h + db*g - dc*f + dd*e,
                a*de - b*df - c*dg - d*dh + da*e - db*f - dc*g - dd*h,
            ),
        }
    }

    /// Conjugate — negates the bivector parts (used to invert unit DualQuat).
    #[inline]
    pub fn conjugate(self) -> Self {
        Self {
            real: self.real.conjugate(),
            dual: Quat::new(-self.dual.x, -self.dual.y, -self.dual.z, self.dual.w),
        }
    }

/// Convert to f64 DDualQuat (lossless upcast).
#[inline]
pub fn as_ddual_quat(self) -> crate::DDualQuat {
    use crate::f64::dquat::DQuat as ScalarDQuat;
    use crate::f64::dvec3::DVec3 as ScalarDVec3;
    let t = self.translation();
    crate::DDualQuat::from_rotation_translation(
        ScalarDQuat::new(
            self.real.x as f64,
            self.real.y as f64,
            self.real.z as f64,
            self.real.w as f64,
        ),
        ScalarDVec3::new(t.x as f64, t.y as f64, t.z as f64),
    )
}

    #[inline] pub fn is_finite(self) -> bool { self.real.is_finite() && self.dual.is_finite() }
}

impl Mul for DualQuat {
    type Output = Self;
    #[inline] fn mul(self, rhs: Self) -> Self { self.mul_dual_quat(rhs) }
}

impl Default for DualQuat { fn default() -> Self { Self::IDENTITY } }

impl PartialEq for DualQuat {
    fn eq(&self, r: &Self) -> bool { self.real == r.real && self.dual == r.dual }
}

impl fmt::Debug for DualQuat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DualQuat(real={:?}, dual={:?})", self.real, self.dual)
    }
}

impl fmt::Display for DualQuat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let t = self.translation();
        write!(f, "DualQuat(rot={:?}, t=({:.4},{:.4},{:.4}))",
               self.real, t.x, t.y, t.z)
    }
      }
