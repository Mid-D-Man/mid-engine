// crates/mid-math/src/f64/ddual_quat.rs
//! Double-precision dual quaternions — rigid body skinning at f64 precision.
//!
//! Direct f64 port of f32/dual_quat.rs. Use when sub-millimetre accuracy is
//! required: physics ground truth, authoring tools, large-world coordinates.
//!
//! For real-time skinning of meshes use the f32 `DualQuat` — GPU shaders
//! operate on f32 and the precision difference is imperceptible at typical
//! distances. DDualQuat is for the solver/authoring layer.
//!
//! See f32/dual_quat.rs for algorithm documentation.

use core::fmt;
use core::ops::Mul;
use crate::DQuat;
use super::dvec3::DVec3;
use super::dvec2::DEPSILON;

/// Double-precision dual quaternion. 64 bytes, 32-byte aligned.
///
/// `real` encodes rotation. `dual` encodes translation as `0.5 * t * real`.
#[derive(Clone, Copy)]
#[repr(C, align(32))]
pub struct DDualQuat {
    pub real: DQuat,
    pub dual: DQuat,
}

impl DDualQuat {
    /// Identity — no rotation, no translation.
    pub const IDENTITY: Self = Self {
        real: DQuat::IDENTITY,
        dual: DQuat::ZERO,
    };

    // ── Constructors ──────────────────────────────────────────────────────────

    /// Build from rotation + translation.
    ///
    /// `rotation` is normalised internally.
    #[inline]
    pub fn from_rotation_translation(rotation: DQuat, translation: DVec3) -> Self {
        let r = rotation.normalize();
        let t = DQuat::new(translation.x, translation.y, translation.z, 0.0);
        let dual = DQuat::new(
            0.5 * ( t.w * r.x + t.x * r.w + t.y * r.z - t.z * r.y),
            0.5 * ( t.w * r.y - t.x * r.z + t.y * r.w + t.z * r.x),
            0.5 * ( t.w * r.z + t.x * r.y - t.y * r.x + t.z * r.w),
            0.5 * (-t.x * r.x - t.y * r.y - t.z * r.z),
        );
        Self { real: r, dual }
    }

    /// Rotation only — no translation.
    #[inline]
    pub fn from_rotation(rotation: DQuat) -> Self {
        Self { real: rotation.normalize(), dual: DQuat::ZERO }
    }

    /// Translation only — no rotation.
    #[inline]
    pub fn from_translation(translation: DVec3) -> Self {
        Self::from_rotation_translation(DQuat::IDENTITY, translation)
    }

    // ── Decomposition ─────────────────────────────────────────────────────────

    /// Extract the rotation quaternion (normalised).
    #[inline]
    pub fn rotation(self) -> DQuat { self.real.normalize() }

    /// Extract the translation vector.
    #[inline]
    pub fn translation(self) -> DVec3 {
        let r = self.real;
        let d = self.dual;
        DVec3::new(
            2.0 * (-d.w * r.x + d.x * r.w - d.y * r.z + d.z * r.y),
            2.0 * (-d.w * r.y + d.x * r.z + d.y * r.w - d.z * r.x),
            2.0 * (-d.w * r.z - d.x * r.y + d.y * r.x + d.z * r.w),
        )
    }

    // ── Transform ─────────────────────────────────────────────────────────────

    /// Transform a point (rotation then translation).
    #[inline]
    pub fn transform_point(self, p: DVec3) -> DVec3 {
        self.real.rotate(p) + self.translation()
    }

    /// Transform a direction vector (rotation only).
    #[inline]
    pub fn transform_vector(self, v: DVec3) -> DVec3 {
        self.real.rotate(v)
    }

    // ── Normalisation ──────────────────────────────────────────────────────────

    /// Normalise so that `|real| = 1`.
    #[inline]
    pub fn normalize(self) -> Self {
        let mag = self.real.length();
        if mag < DEPSILON { return Self::IDENTITY; }
        let inv = 1.0 / mag;
        Self {
            real: DQuat::new(self.real.x*inv, self.real.y*inv, self.real.z*inv, self.real.w*inv),
            dual: DQuat::new(self.dual.x*inv, self.dual.y*inv, self.dual.z*inv, self.dual.w*inv),
        }
    }

    // ── Dual-linear blending (DLB) ────────────────────────────────────────────

    /// Scale all components by a scalar weight.
    #[inline]
    pub fn scale(self, w: f64) -> Self {
        Self {
            real: DQuat::new(self.real.x*w, self.real.y*w, self.real.z*w, self.real.w*w),
            dual: DQuat::new(self.dual.x*w, self.dual.y*w, self.dual.z*w, self.dual.w*w),
        }
    }

    /// Component-wise add (used during blending accumulation).
    #[inline]
    pub fn add_dq(self, rhs: Self) -> Self {
        Self {
            real: DQuat::new(self.real.x+rhs.real.x, self.real.y+rhs.real.y,
                             self.real.z+rhs.real.z, self.real.w+rhs.real.w),
            dual: DQuat::new(self.dual.x+rhs.dual.x, self.dual.y+rhs.dual.y,
                             self.dual.z+rhs.dual.z, self.dual.w+rhs.dual.w),
        }
    }

    /// Dual-linear blend of two dual quaternions with weights.
    ///
    /// Ensures shortest-path interpolation. Result is normalised.
    #[inline]
    pub fn blend2(dq0: Self, w0: f64, dq1: Self, w1: f64) -> Self {
        let dot = dq0.real.x*dq1.real.x + dq0.real.y*dq1.real.y
                + dq0.real.z*dq1.real.z + dq0.real.w*dq1.real.w;
        let dq1 = if dot < 0.0 { dq1.scale(-1.0) } else { dq1 };
        dq0.scale(w0).add_dq(dq1.scale(w1)).normalize()
    }

    /// Dual-linear blend of up to four weighted dual quaternions.
    ///
    /// All influences are flipped to the same hemisphere as `influences[0]`.
    /// Result is normalised.
    pub fn blend4(influences: [(Self, f64); 4]) -> Self {
        let (ref_dq, _) = influences[0];
        let pivot_dot = |a: &Self, b: &Self| -> f64 {
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
            real: DQuat::new(
                a*f + b*e + c*h - d*g,
                a*g - b*h + c*e + d*f,
                a*h + b*g - c*f + d*e,
                a*e - b*f - c*g - d*h,
            ),
            dual: DQuat::new(
                a*df + b*de + c*dh - d*dg + da*f + db*e + dc*h - dd*g,
                a*dg - b*dh + c*de + d*df + da*g - db*h + dc*e + dd*f,
                a*dh + b*dg - c*df + d*de + da*h + db*g - dc*f + dd*e,
                a*de - b*df - c*dg - d*dh + da*e - db*f - dc*g - dd*h,
            ),
        }
    }

    /// Conjugate (negates bivector parts).
    #[inline]
    pub fn conjugate(self) -> Self {
        Self {
            real: self.real.conjugate(),
            dual: DQuat::new(-self.dual.x, -self.dual.y, -self.dual.z, self.dual.w),
        }
    }

    /// Lossy downcast to f32 `DualQuat`.
    #[inline]
    pub fn as_dual_quat(self) -> crate::DualQuat {
        crate::DualQuat::from_rotation_translation(
            crate::Quat::new(self.real.x as f32, self.real.y as f32,
                             self.real.z as f32, self.real.w as f32),
            crate::Vec3::new(self.translation().x as f32,
                             self.translation().y as f32,
                             self.translation().z as f32),
        )
    }

    #[inline]
    pub fn is_finite(self) -> bool { self.real.is_finite() && self.dual.is_finite() }
}

impl Mul for DDualQuat {
    type Output = Self;
    #[inline] fn mul(self, rhs: Self) -> Self { self.mul_dual_quat(rhs) }
}

impl Default for DDualQuat { fn default() -> Self { Self::IDENTITY } }

impl PartialEq for DDualQuat {
    fn eq(&self, r: &Self) -> bool { self.real == r.real && self.dual == r.dual }
}

impl fmt::Debug for DDualQuat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DDualQuat(real={:?}, dual={:?})", self.real, self.dual)
    }
}

impl fmt::Display for DDualQuat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let t = self.translation();
        write!(f, "DDualQuat(rot={:?}, t=({:.6},{:.6},{:.6}))",
               self.real, t.x, t.y, t.z)
    }
                             }
