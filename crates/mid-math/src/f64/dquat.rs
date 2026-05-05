// crates/mid-math/src/f64/dquat.rs
//! Double-precision quaternion. 32 bytes, align(32). Always scalar.
//!
//! Convention: (x, y, z, w) where w is the scalar part.
//! Euler convention: ZYX — same as the f32 Quat.
//!
//! Uses acos_approx from f32::math ported to f64 — polynomial coefficients
//! are more accurate at f64 precision anyway. For f64 quaternions the
//! extra precision of acos matters most in slerp near-parallel branches.
//!
//! DEPSILON = 1e-12 for normalization checks.
//! Near-identity threshold for slerp fallback: cos_theta > 1.0 - 1e-10.

use core::fmt;
use core::ops::{Add, Mul, MulAssign, Neg, Sub};

use super::dvec3::DVec3;
use super::dmat4::DMat4;
use super::dvec2::DEPSILON;

/// Double-precision quaternion. 32 bytes, align(32).
/// Convention: (x, y, z, w). Lane layout: [x, y, z, w].
///
/// **C interop:** use [`CDQuat`][crate::ffi::types::CDQuat] at the FFI boundary.
#[derive(Clone, Copy)]
#[repr(C, align(32))]
pub struct DQuat {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub w: f64,
}

impl DQuat {
    /// Identity — represents no rotation.
    pub const IDENTITY: Self = Self { x: 0.0, y: 0.0, z: 0.0, w: 1.0 };
    const ZERO: Self         = Self { x: 0.0, y: 0.0, z: 0.0, w: 0.0 };

    // ── Constructors ──────────────────────────────────────────────────────────

    #[inline(always)]
    pub const fn new(x: f64, y: f64, z: f64, w: f64) -> Self { Self { x, y, z, w } }

    #[inline(always)]
    pub const fn from_xyzw(x: f64, y: f64, z: f64, w: f64) -> Self { Self::new(x, y, z, w) }

    /// Build from a unit axis and an angle in radians.
    /// `axis` need not be pre-normalised — normalised internally.
    #[inline]
    pub fn from_axis_angle(axis: DVec3, angle_rad: f64) -> Self {
        let (s, c) = (angle_rad * 0.5).sin_cos();
        let n = axis.normalize();
        Self::new(n.x * s, n.y * s, n.z * s, c)
    }

    /// Build from Euler angles (radians), ZYX convention.
    /// Applied as: Rz * Ry * Rx.
    pub fn from_euler(roll: f64, pitch: f64, yaw: f64) -> Self {
        let (sx, cx) = (roll  * 0.5).sin_cos();
        let (sy, cy) = (pitch * 0.5).sin_cos();
        let (sz, cz) = (yaw   * 0.5).sin_cos();
        Self::new(
            cz * cy * sx - sz * sy * cx,
            cz * sy * cx + sz * cy * sx,
            sz * cy * cx - cz * sy * sx,
            cz * cy * cx + sz * sy * sx,
        ).normalize()
    }

    // ── Decomposition ─────────────────────────────────────────────────────────

    /// Extract Euler angles (ZYX). Returns `(roll, pitch, yaw)`.
    pub fn to_euler(self) -> (f64, f64, f64) {
        let sinp  = 2.0 * (self.w * self.y - self.z * self.x);
        let pitch = if sinp.abs() >= 1.0 {
            sinp.signum() * core::f64::consts::FRAC_PI_2
        } else {
            sinp.asin()
        };
        let roll = (2.0 * (self.w * self.x + self.y * self.z))
            .atan2(1.0 - 2.0 * (self.x * self.x + self.y * self.y));
        let yaw  = (2.0 * (self.w * self.z + self.x * self.y))
            .atan2(1.0 - 2.0 * (self.y * self.y + self.z * self.z));
        (roll, pitch, yaw)
    }

    // ── Core ops ──────────────────────────────────────────────────────────────

    #[inline]
    pub fn dot(self, rhs: Self) -> f64 {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z + self.w * rhs.w
    }

    #[inline] pub fn length_sq(self) -> f64 { self.dot(self) }
    #[inline] pub fn length(self)    -> f64 { self.length_sq().sqrt() }

    /// Normalize. Returns IDENTITY if near-zero length.
    #[inline]
    pub fn normalize(self) -> Self {
        let l = self.length();
        if l < DEPSILON { Self::IDENTITY }
        else { Self::new(self.x / l, self.y / l, self.z / l, self.w / l) }
    }

    /// Conjugate — inverse for unit quaternions.
    #[inline]
    pub fn conjugate(self) -> Self {
        Self::new(-self.x, -self.y, -self.z, self.w)
    }

    /// Full inverse (safe for non-unit quaternions).
    #[inline]
    pub fn inverse(self) -> Self {
        let sq = self.length_sq();
        if sq < DEPSILON { return Self::IDENTITY; }
        let r = 1.0 / sq;
        Self::new(-self.x * r, -self.y * r, -self.z * r, self.w * r)
    }

    /// Rotate a DVec3 via sandwich product q v q*.
    ///
    /// `self` must be normalised.
    ///
    /// Uses the two-cross-product formula:
    /// ```text
    /// t = 2 * cross(q.xyz, v)
    /// result = v + w*t + cross(q.xyz, t)
    /// ```
    #[inline]
    pub fn rotate(self, v: DVec3) -> DVec3 {
        let qv = DVec3::new(self.x, self.y, self.z);
        let t  = qv.cross(v) * 2.0;
        v + t * self.w + qv.cross(t)
    }

    /// Hamilton product of two quaternions.
    ///
    /// Composes rotations: `(self * rhs)` applies `rhs` first, then `self`.
    #[inline]
    pub fn mul_quat(self, rhs: Self) -> Self {
        Self::new(
            self.w * rhs.x + self.x * rhs.w + self.y * rhs.z - self.z * rhs.y,
            self.w * rhs.y - self.x * rhs.z + self.y * rhs.w + self.z * rhs.x,
            self.w * rhs.z + self.x * rhs.y - self.y * rhs.x + self.z * rhs.w,
            self.w * rhs.w - self.x * rhs.x - self.y * rhs.y - self.z * rhs.z,
        )
    }

    // ── Interpolation ──────────────────────────────────────────────────────────

    /// Normalised linear interpolation — fast, slightly non-constant velocity.
    ///
    /// Shortest-path via dot-sign flip. Inlined normalize avoids the
    /// method-call overhead and lets the compiler see the full dependency.
    #[inline]
    pub fn nlerp(self, rhs: Self, t: f64) -> Self {
        let dot  = self.dot(rhs);
        let sign = if dot < 0.0 { -1.0f64 } else { 1.0f64 };

        // Lerp toward the correct hemisphere.
        let lx = self.x + (rhs.x * sign - self.x) * t;
        let ly = self.y + (rhs.y * sign - self.y) * t;
        let lz = self.z + (rhs.z * sign - self.z) * t;
        let lw = self.w + (rhs.w * sign - self.w) * t;

        // Inline normalize — avoid the function call overhead.
        // length_sq is fused here so compiler sees all four components.
        let len_sq = lx*lx + ly*ly + lz*lz + lw*lw;
        if len_sq < DEPSILON {
            return Self::IDENTITY;
        }
        let inv_len = 1.0 / len_sq.sqrt();
        Self::new(lx * inv_len, ly * inv_len, lz * inv_len, lw * inv_len)
    }

    /// Spherical linear interpolation — constant angular velocity.
///
/// 2-transcendental path: acos + sin_cos.
///
/// Previous atan2 approach was 3 transcendentals on glibc (71-79 ns).
/// acos (~15 ns) + sin_cos (~15 ns) + sqrt (~5 ns) ≈ 35 ns.
///
/// Algebraic derivation eliminates the 3rd transcendental:
///   sin((1-t)·θ) / sin(θ)
///   = (sin(θ)·cos(t·θ) − cos(θ)·sin(t·θ)) / sin(θ)
///   = cos(t·θ) − cos(θ)·sin(t·θ)/sin(θ)
///   = cos_t − cos_theta·s1
///
/// No normalize() — slerp of two unit quats is unit by construction
/// when sin_theta is non-zero.
pub fn slerp(self, mut rhs: Self, t: f64) -> Self {
    let mut cos_theta = self.dot(rhs);

    // Shortest-path flip.
    if cos_theta < 0.0 {
        rhs = -rhs;
        cos_theta = -cos_theta;
    }

    // Near-identical: fall back to nlerp to avoid division by near-zero sin.
    if cos_theta > 1.0 - 1e-6 {
        return self.nlerp(rhs, t);
    }

    // 1st transcendental: acos.
    // Safe because cos_theta ∈ [0, 1 − 1e-6] after the shortest-path flip.
    let angle = cos_theta.acos();

    // sin(angle) via sqrt — avoids a separate sin() call (~5 ns vs ~15 ns).
    // Equivalent to sin(acos(cos_theta)) = sqrt(1 − cos²θ).
    let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();

    // 2nd transcendental: sin_cos counts as one call on most platforms.
    let (sin_t, cos_t) = (t * angle).sin_cos();

    let inv_sin = 1.0 / sin_theta;
    let s1 = sin_t * inv_sin;
    let s0 = cos_t - cos_theta * s1; // algebraic — no 3rd transcendental

    Self::new(
        self.x * s0 + rhs.x * s1,
        self.y * s0 + rhs.y * s1,
        self.z * s0 + rhs.z * s1,
        self.w * s0 + rhs.w * s1,
    )
}
    // ── Conversion ─────────────────────────────────────────────────────────────

    /// Convert to rotation DMat4. `self` must be normalised.
    pub fn to_mat4(self) -> DMat4 {
        let q = self.normalize();
        let (x, y, z, w) = (q.x, q.y, q.z, q.w);
        let (x2, y2, z2) = (x + x, y + y, z + z);
        let (xx, yy, zz) = (x * x2, y * y2, z * z2);
        let (xy, xz, yz) = (x * y2, x * z2, y * z2);
        let (wx, wy, wz) = (w * x2, w * y2, w * z2);
        DMat4::from_cols(
            [1.0-yy-zz, xy+wz,     xz-wy,     0.0],
            [xy-wz,     1.0-xx-zz, yz+wx,     0.0],
            [xz+wy,     yz-wx,     1.0-xx-yy, 0.0],
            [0.0,       0.0,       0.0,       1.0],
        )
    }

    /// Lossy cast to single-precision `Quat`.
    #[inline]
    pub fn as_quat(self) -> crate::Quat {
        crate::Quat::new(self.x as f32, self.y as f32, self.z as f32, self.w as f32)
    }

    #[inline]
    pub fn is_normalized(self) -> bool { (self.length_sq() - 1.0).abs() <= 4e-10 }

    #[inline]
    pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite()
            && self.z.is_finite() && self.w.is_finite()
    }
}

// ── Operators ─────────────────────────────────────────────────────────────────

impl Mul for DQuat {
    type Output = Self;
    #[inline] fn mul(self, rhs: Self) -> Self { self.mul_quat(rhs) }
}
impl MulAssign for DQuat {
    #[inline] fn mul_assign(&mut self, rhs: Self) { *self = self.mul_quat(rhs); }
}
impl Neg for DQuat {
    type Output = Self;
    #[inline] fn neg(self) -> Self { Self::new(-self.x, -self.y, -self.z, -self.w) }
}
impl Add for DQuat {
    type Output = Self;
    #[inline] fn add(self, r: Self) -> Self {
        Self::new(self.x+r.x, self.y+r.y, self.z+r.z, self.w+r.w)
    }
}
impl Sub for DQuat {
    type Output = Self;
    #[inline] fn sub(self, r: Self) -> Self {
        Self::new(self.x-r.x, self.y-r.y, self.z-r.z, self.w-r.w)
    }
}
impl Mul<f64> for DQuat {
    type Output = Self;
    #[inline] fn mul(self, s: f64) -> Self {
        Self::new(self.x*s, self.y*s, self.z*s, self.w*s)
    }
}

impl PartialEq for DQuat {
    fn eq(&self, rhs: &Self) -> bool {
        self.x == rhs.x && self.y == rhs.y && self.z == rhs.z && self.w == rhs.w
    }
}
impl Default for DQuat { fn default() -> Self { Self::IDENTITY } }

impl fmt::Debug for DQuat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("DQuat")
            .field(&self.x).field(&self.y).field(&self.z).field(&self.w)
            .finish()
    }
}
impl fmt::Display for DQuat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DQuat({:.6}, {:.6}, {:.6}, {:.6})", self.x, self.y, self.z, self.w)
    }
}
