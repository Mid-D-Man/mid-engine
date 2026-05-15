// crates/mid-math/src/dual_quat.rs
//! Dual quaternions — the industry standard for skeletal skinning.
//!
//! A dual quaternion DQ = q_r + ε·q_d (where ε² = 0) encodes rigid-body
//! transformation (rotation + translation) in a single structure.
//!
//! Compared to Mat4 for skinning:
//!   - No candy-wrapper artifact under large rotations (key advantage over LBS)
//!   - Volume-preserving under blending
//!   - ~30% fewer operations than Mat4 for transform_point
//!
//! Algorithm: Dual Linear Blending (DLB) — blend dual quats linearly,
//! normalize, then transform. Correct for small influence counts (≤ 4 bones).
//!
//! Reference: Kavan et al. 2007 "Skinning with Dual Quaternions"

use core::fmt;
use core::ops::Mul;
use crate::{Quat, Vec3, EPSILON};

/// Dual quaternion. 32 bytes, 16-byte aligned.
///
/// `real` encodes rotation. `dual` encodes translation as `0.5 * t * real`.
/// For a valid rigid transform: `real` must be unit, `real.dot(dual) ≈ 0`.
#[derive(Clone, Copy)]
#[repr(C, align(16))]
pub struct DualQuat {
    /// Rotation quaternion (unit).
    pub real: Quat,
    /// Translation-encoding quaternion. dual = 0.5 * pure_t * real.
    pub dual: Quat,
}

impl DualQuat {
    /// Identity: no rotation, no translation.
    pub const IDENTITY: Self = Self {
        real: Quat::IDENTITY,
        dual: Quat { x: 0.0, y: 0.0, z: 0.0, w: 0.0 },
    };

    // ── Constructors ──────────────────────────────────────────────────────────

    /// Build from rotation + translation.
    ///
    /// `rotation` is normalised internally.
    #[inline]
    pub fn from_rotation_translation(rotation: Quat, translation: Vec3) -> Self {
        let r = rotation.normalize();
        // dual = 0.5 * t_pure * r  where t_pure = Quat(tx,ty,tz,0)
        let t = Quat::new(translation.x, translation.y, translation.z, 0.0);
        let dual = Quat::new(
            0.5 * ( t.w * r.x + t.x * r.w + t.y * r.z - t.z * r.y),
            0.5 * ( t.w * r.y - t.x * r.z + t.y * r.w + t.z * r.x),
            0.5 * ( t.w * r.z + t.x * r.y - t.y * r.x + t.z * r.w),
            0.5 * (-t.x * r.x - t.y * r.y - t.z * r.z),
        );
        Self { real: r, dual }
    }

    /// Build from rotation only (no translation).
    #[inline]
    pub fn from_rotation(rotation: Quat) -> Self {
        Self { real: rotation.normalize(), dual: Quat::new(0.0, 0.0, 0.0, 0.0) }
    }

    /// Build from translation only (no rotation).
    #[inline]
    pub fn from_translation(translation: Vec3) -> Self {
        Self::from_rotation_translation(Quat::IDENTITY, translation)
    }

    // ── Decomposition ─────────────────────────────────────────────────────────

    /// Extract the rotation component.
    #[inline] pub fn rotation(self) -> Quat { self.real.normalize() }

    /// Extract the translation component.
    #[inline]
    pub fn translation(self) -> Vec3 {
        // t = 2 * dual * conjugate(real)
        let r = self.real;
        let d = self.dual;
        // (a + bi + cj + dk) * (w - xi - yj - zk) for quat conjugate
        let tx = 2.0 * (-d.w * r.x + d.x * r.w - d.y * r.z + d.z * r.y);
        let ty = 2.0 * (-d.w * r.y + d.x * r.z + d.y * r.w - d.z * r.x);
        let tz = 2.0 * (-d.w * r.z - d.x * r.y + d.y * r.x + d.z * r.w);
        Vec3::new(tx, ty, tz)
    }

    // ── Transform ────────────────────────────────────────────────────────────

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

    // ── Normalisation ─────────────────────────────────────────────────────────

    /// Normalise to maintain unit dual quaternion constraint.
    ///
    /// Required after linear blending: `dq = Σ wᵢ·dqᵢ` followed by `normalize()`.
    #[inline]
    pub fn normalize(self) -> Self {
        let mag = (self.real.x * self.real.x
                 + self.real.y * self.real.y
                 + self.real.z * self.real.z
                 + self.real.w * self.real.w).sqrt();
        if mag < EPSILON { return Self::IDENTITY; }
        let inv = 1.0 / mag;
        Self {
            real: Quat::new(self.real.x*inv, self.real.y*inv, self.real.z*inv, self.real.w*inv),
            dual: Quat::new(self.dual.x*inv, self.dual.y*inv, self.dual.z*inv, self.dual.w*inv),
        }
    }

    // ── Blending (DLB) ────────────────────────────────────────────────────────

    /// Scale all components by `w` — used as building block for weighted blend.
    #[inline]
    pub fn scale(self, w: f32) -> Self {
        Self {
            real: Quat::new(self.real.x*w, self.real.y*w, self.real.z*w, self.real.w*w),
            dual: Quat::new(self.dual.x*w, self.dual.y*w, self.dual.z*w, self.dual.w*w),
        }
    }

    /// Add another dual quaternion component-wise. Used in weighted blending.
    #[inline]
    pub fn add(self, rhs: Self) -> Self {
        Self {
            real: Quat::new(self.real.x+rhs.real.x, self.real.y+rhs.real.y,
                            self.real.z+rhs.real.z, self.real.w+rhs.real.w),
            dual: Quat::new(self.dual.x+rhs.dual.x, self.dual.y+rhs.dual.y,
                            self.dual.z+rhs.dual.z, self.dual.w+rhs.dual.w),
        }
    }

    /// Dual Linear Blend of two dual quaternions with weights `w0 + w1 = 1`.
    ///
    /// Ensures shortest-path interpolation by negating `dq1` if needed.
    pub fn blend2(dq0: Self, w0: f32, dq1: Self, w1: f32) -> Self {
        // Shortest path: negate dq1 if dot(real0, real1) < 0
        let dot = dq0.real.x * dq1.real.x + dq0.real.y * dq1.real.y
                + dq0.real.z * dq1.real.z + dq0.real.w * dq1.real.w;
        let dq1 = if dot < 0.0 { dq1.scale(-1.0) } else { dq1 };
        dq0.scale(w0).add(dq1.scale(w1)).normalize()
    }

    /// Blend 4 dual quaternions with given weights. Normalises the result.
    ///
    /// Weights should sum to 1.0 but do not need to — normalise handles it.
    pub fn blend4(influences: [(Self, f32); 4]) -> Self {
        let pivot_dot = |a: &Self, b: &Self| -> f32 {
            a.real.x * b.real.x + a.real.y * b.real.y
          + a.real.z * b.real.z + a.real.w * b.real.w
        };
        let (ref_dq, _) = influences[0];
        let mut acc = Self::IDENTITY.scale(0.0); // zero dual quat
        for (dq, w) in &influences {
            let sign = if pivot_dot(&ref_dq, dq) < 0.0 { -1.0 } else { 1.0 };
            acc = acc.add(dq.scale(w * sign));
        }
        acc.normalize()
    }

    // ── Composition ───────────────────────────────────────────────────────────

    /// Multiply (compose): `self * rhs` applies rhs first, then self.
    pub fn mul_dual_quat(self, rhs: Self) -> Self {
        Self {
            real: Quat::new(
                self.real.w*rhs.real.x + self.real.x*rhs.real.w + self.real.y*rhs.real.z - self.real.z*rhs.real.y,
                self.real.w*rhs.real.y - self.real.x*rhs.real.z + self.real.y*rhs.real.w + self.real.z*rhs.real.x,
                self.real.w*rhs.real.z + self.real.x*rhs.real.y - self.real.y*rhs.real.x + self.real.z*rhs.real.w,
                self.real.w*rhs.real.w - self.real.x*rhs.real.x - self.real.y*rhs.real.y - self.real.z*rhs.real.z,
            ),
            dual: Quat::new(
                self.real.w*rhs.dual.x + self.real.x*rhs.dual.w + self.real.y*rhs.dual.z - self.real.z*rhs.dual.y
              + self.dual.w*rhs.real.x + self.dual.x*rhs.real.w + self.dual.y*rhs.real.z - self.dual.z*rhs.real.y,
                self.real.w*rhs.dual.y - self.real.x*rhs.dual.z + self.real.y*rhs.dual.w + self.real.z*rhs.dual.x
              + self.dual.w*rhs.real.y - self.dual.x*rhs.real.z + self.dual.y*rhs.real.w + self.dual.z*rhs.real.x,
                self.real.w*rhs.dual.z + self.real.x*rhs.dual.y - self.real.y*rhs.dual.x + self.real.z*rhs.dual.w
              + self.dual.w*rhs.real.z + self.dual.x*rhs.real.y - self.dual.y*rhs.real.x + self.dual.z*rhs.real.w,
                self.real.w*rhs.dual.w - self.real.x*rhs.dual.x - self.real.y*rhs.dual.y - self.real.z*rhs.dual.z
              + self.dual.w*rhs.real.w - self.dual.x*rhs.real.x - self.dual.y*rhs.real.y - self.dual.z*rhs.real.z,
            ),
        }
    }

    #[inline] pub fn conjugate(self) -> Self {
        Self {
            real: self.real.conjugate(),
            dual: Quat::new(-self.dual.x, -self.dual.y, -self.dual.z, self.dual.w),
        }
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
