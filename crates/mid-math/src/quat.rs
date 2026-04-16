// crates/mid-math/src/quat.rs

//! Quaternion — rotation without Gimbal Lock.
//!
//! Convention: (x, y, z, w) where w is the scalar part.
//! Euler convention: ZYX — yaw applied first, then pitch, then roll.
//! This matches `to_euler` extraction and aerospace/game standard.

use std::fmt;
use std::ops::{Mul, Neg};
use crate::vec::Vec3;
use crate::mat::Mat4;
use crate::EPSILON;

/// Quaternion. 16 bytes, 16-byte aligned. `#[repr(C)]`.
///
/// **C layout:** `struct MidQuat { float x, y, z, w; }` — 16 bytes.
#[derive(Debug, Clone, Copy)]
#[repr(C, align(16))]
pub struct Quat {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

impl Quat {
    /// No rotation.
    pub const IDENTITY: Self = Self { x: 0.0, y: 0.0, z: 0.0, w: 1.0 };

    #[inline] pub fn new(x: f32, y: f32, z: f32, w: f32) -> Self { Self { x, y, z, w } }

    // ── Constructors ───────────────────────────────────────────────────────

    /// Build from a unit axis and an angle in radians.
    /// `axis` need not be pre-normalized — normalized internally.
    #[inline]
    pub fn from_axis_angle(axis: Vec3, angle_rad: f32) -> Self {
        let (sin, cos) = (angle_rad * 0.5).sin_cos();
        let n = axis.normalize();
        Self::new(n.x * sin, n.y * sin, n.z * sin, cos)
    }

    /// Build from Euler angles in radians, **ZYX convention** (aerospace / game standard).
    ///
    /// Applied as: first rotate around **Z** (yaw), then **Y** (pitch), then **X** (roll).
    /// Equivalent to `q = Rz * Ry * Rx`.
    ///
    /// `roll`  = rotation around X axis  
    /// `pitch` = rotation around Y axis  
    /// `yaw`   = rotation around Z axis
    ///
    /// This matches `to_euler` exactly — round-tripping is exact up to floating-point epsilon.
    pub fn from_euler(roll: f32, pitch: f32, yaw: f32) -> Self {
        let (sx, cx) = (roll  * 0.5).sin_cos();
        let (sy, cy) = (pitch * 0.5).sin_cos();
        let (sz, cz) = (yaw   * 0.5).sin_cos();
        // q = Rz * Ry * Rx
        Self::new(
            cz * cy * sx - sz * sy * cx,  // x
            cz * sy * cx + sz * cy * sx,  // y
            sz * cy * cx - cz * sy * sx,  // z
            cz * cy * cx + sz * sy * sx,  // w
        ).normalize()
    }

    // ── Decomposition ──────────────────────────────────────────────────────

    /// Decompose to Euler angles, **ZYX convention** (matches `from_euler`).
    /// Returns `(roll, pitch, yaw)` in radians.
    ///
    /// Handles the gimbal-lock singularity at `|pitch| ≈ 90°`.
    pub fn to_euler(self) -> (f32, f32, f32) {
        // ZYX extraction — matches Rz*Ry*Rx construction in from_euler.
        let sinp = 2.0 * (self.w * self.y - self.z * self.x);
        let pitch = if sinp.abs() >= 1.0 {
            sinp.signum() * std::f32::consts::FRAC_PI_2
        } else {
            sinp.asin()
        };
        let roll = (2.0 * (self.w * self.x + self.y * self.z))
            .atan2(1.0 - 2.0 * (self.x * self.x + self.y * self.y));
        let yaw  = (2.0 * (self.w * self.z + self.x * self.y))
            .atan2(1.0 - 2.0 * (self.y * self.y + self.z * self.z));
        (roll, pitch, yaw)
    }

    // ── Core ops ───────────────────────────────────────────────────────────

    #[inline] pub fn dot(self, rhs: Self) -> f32 {
        self.x*rhs.x + self.y*rhs.y + self.z*rhs.z + self.w*rhs.w
    }

    #[inline] pub fn length_sq(self) -> f32 { self.dot(self) }
    #[inline] pub fn length(self)    -> f32 { self.length_sq().sqrt() }

    /// Returns a unit quaternion. Falls back to IDENTITY if near-zero.
    #[inline] pub fn normalize(self) -> Self {
        let l = self.length();
        if l < EPSILON { Self::IDENTITY } else {
            Self::new(self.x/l, self.y/l, self.z/l, self.w/l)
        }
    }

    /// Conjugate — equivalent to the inverse for unit quaternions.
    #[inline] pub fn conjugate(self) -> Self { Self::new(-self.x, -self.y, -self.z, self.w) }

    /// Inverse. For unit quaternions, prefer `conjugate()` — it's cheaper.
    #[inline] pub fn inverse(self) -> Self {
        let sq = self.length_sq();
        if sq < EPSILON { Self::IDENTITY } else {
            Self::new(-self.x/sq, -self.y/sq, -self.z/sq, self.w/sq)
        }
    }

    /// Rotate a Vec3. `self` must be normalized.
    ///
    /// Uses the sandwich product `q v q*` via two cross products —
    /// faster than converting to matrix for a single vector.
    #[inline]
    pub fn rotate(self, v: Vec3) -> Vec3 {
        // t = 2 * cross(q.xyz, v);  result = v + w*t + cross(q.xyz, t)
        let qv = Vec3::new(self.x, self.y, self.z);
        let t  = 2.0 * qv.cross(v);
        v + self.w * t + qv.cross(t)
    }

    // ── Interpolation ──────────────────────────────────────────────────────

    /// Spherical linear interpolation — constant angular velocity.
    /// Both quaternions should be normalized. `t` ∈ [0, 1].
    pub fn slerp(self, mut rhs: Self, t: f32) -> Self {
        let mut cos_theta = self.dot(rhs);
        if cos_theta < 0.0 { rhs = -rhs; cos_theta = -cos_theta; }

        if cos_theta > 1.0 - EPSILON {
            return Self::new(
                self.x + (rhs.x - self.x)*t,
                self.y + (rhs.y - self.y)*t,
                self.z + (rhs.z - self.z)*t,
                self.w + (rhs.w - self.w)*t,
            ).normalize();
        }

        let angle = cos_theta.acos();
        let sin_a = angle.sin();
        let s0    = ((1.0 - t) * angle).sin() / sin_a;
        let s1    = (t          * angle).sin() / sin_a;

        Self::new(
            self.x*s0 + rhs.x*s1,
            self.y*s0 + rhs.y*s1,
            self.z*s0 + rhs.z*s1,
            self.w*s0 + rhs.w*s1,
        )
    }

    /// Normalised linear interpolation — cheaper than slerp, slightly
    /// non-constant angular velocity. Fine for small angles.
    #[inline]
    pub fn nlerp(self, rhs: Self, t: f32) -> Self {
        let dot = self.dot(rhs);
        let rhs = if dot < 0.0 { -rhs } else { rhs };
        Self::new(
            self.x + (rhs.x - self.x)*t,
            self.y + (rhs.y - self.y)*t,
            self.z + (rhs.z - self.z)*t,
            self.w + (rhs.w - self.w)*t,
        ).normalize()
    }

    // ── Conversion ─────────────────────────────────────────────────────────

    /// Convert to a column-major rotation Mat4. `self` must be normalized.
    pub fn to_mat4(self) -> Mat4 {
        let q = self.normalize();
        let (x, y, z, w) = (q.x, q.y, q.z, q.w);
        let (x2, y2, z2) = (x+x, y+y, z+z);
        let (xx, yy, zz) = (x*x2, y*y2, z*z2);
        let (xy, xz, yz) = (x*y2, x*z2, y*z2);
        let (wx, wy, wz) = (w*x2, w*y2, w*z2);
        Mat4::from_cols(
            [1.0-yy-zz, xy+wz,     xz-wy,     0.0],
            [xy-wz,     1.0-xx-zz, yz+wx,     0.0],
            [xz+wy,     yz-wx,     1.0-xx-yy, 0.0],
            [0.0,       0.0,       0.0,       1.0],
        )
    }
}

// ── Operators ─────────────────────────────────────────────────────────────────

impl Mul for Quat {
    type Output = Self;
    #[inline]
    fn mul(self, r: Self) -> Self {
        Self::new(
            self.w*r.x + self.x*r.w + self.y*r.z - self.z*r.y,
            self.w*r.y - self.x*r.z + self.y*r.w + self.z*r.x,
            self.w*r.z + self.x*r.y - self.y*r.x + self.z*r.w,
            self.w*r.w - self.x*r.x - self.y*r.y - self.z*r.z,
        )
    }
}

impl Neg for Quat {
    type Output = Self;
    #[inline] fn neg(self) -> Self { Self::new(-self.x, -self.y, -self.z, -self.w) }
}

impl PartialEq for Quat {
    fn eq(&self, r: &Self) -> bool {
        self.x==r.x && self.y==r.y && self.z==r.z && self.w==r.w
    }
}

impl Default for Quat { fn default() -> Self { Self::IDENTITY } }

impl fmt::Display for Quat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Quat({:.4}, {:.4}, {:.4}, {:.4})", self.x, self.y, self.z, self.w)
    }
}
