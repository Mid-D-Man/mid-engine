// crates/mid-math/src/wide/float/scalar/quatx4.rs
//! Scalar fallback QuatX4 — non-x86 platforms.
//!
//! SoA layout using [f32; 4] arrays. Matches the SSE2 interface exactly.
//! NEON and WASM SIMD ports of QuatX4 delegate here until Phase 5.

use core::fmt;
use core::ops::Mul;

use crate::Quat;
use crate::EPSILON;
use super::f32x4::f32x4;
use super::vec3x4::Vec3x4;

/// 4 quaternions in SoA layout — scalar fallback. 64 bytes, 16-byte aligned.
#[derive(Clone, Copy, PartialEq)]
#[repr(C, align(16))]
pub struct QuatX4 {
    pub x: [f32; 4],
    pub y: [f32; 4],
    pub z: [f32; 4],
    pub w: [f32; 4],
}

impl QuatX4 {
    pub const IDENTITY: Self = Self {
        x: [0.0; 4],
        y: [0.0; 4],
        z: [0.0; 4],
        w: [1.0; 4],
    };

    // ── Constructors ──────────────────────────────────────────────────────────

    #[inline]
    pub fn from_quats(a: Quat, b: Quat, c: Quat, d: Quat) -> Self {
        Self {
            x: [a.x, b.x, c.x, d.x],
            y: [a.y, b.y, c.y, d.y],
            z: [a.z, b.z, c.z, d.z],
            w: [a.w, b.w, c.w, d.w],
        }
    }

    #[inline(always)]
    pub fn splat(q: Quat) -> Self {
        Self { x: [q.x; 4], y: [q.y; 4], z: [q.z; 4], w: [q.w; 4] }
    }

    #[inline(always)]
    pub fn from_slice(s: &[Quat; 4]) -> Self {
        Self::from_quats(s[0], s[1], s[2], s[3])
    }

    #[inline]
    pub fn to_array(self) -> [Quat; 4] {
        core::array::from_fn(|i| Quat::new(self.x[i], self.y[i], self.z[i], self.w[i]))
    }

    #[inline]
    pub fn get(self, lane: usize) -> Quat {
        assert!(lane < 4, "QuatX4::get — lane {lane} out of bounds (max 3)");
        self.to_array()[lane]
    }

    // ── Core ops ──────────────────────────────────────────────────────────────

    #[inline]
    pub fn dot(self, rhs: Self) -> f32x4 {
        f32x4(core::array::from_fn(|i| {
            self.x[i]*rhs.x[i] + self.y[i]*rhs.y[i]
                + self.z[i]*rhs.z[i] + self.w[i]*rhs.w[i]
        }))
    }

    #[inline(always)] pub fn length_sq(self) -> f32x4 { self.dot(self) }

    #[inline]
    pub fn normalize(self) -> Self {
        let lsq = self.length_sq().0;
        let mut out = Self::IDENTITY;
        for i in 0..4 {
            if lsq[i] > EPSILON * EPSILON {
                let inv = 1.0 / lsq[i].sqrt();
                out.x[i] = self.x[i] * inv;
                out.y[i] = self.y[i] * inv;
                out.z[i] = self.z[i] * inv;
                out.w[i] = self.w[i] * inv;
            }
            // degenerate lane stays as identity (0, 0, 0, 1)
        }
        out
    }

    #[inline]
    pub fn conjugate(self) -> Self {
        Self {
            x: self.x.map(|v| -v),
            y: self.y.map(|v| -v),
            z: self.z.map(|v| -v),
            w: self.w,
        }
    }

    // ── Hamilton product ───────────────────────────────────────────────────────

    #[inline]
    pub fn mul_quatx4(self, rhs: Self) -> Self {
        let mut out = Self::IDENTITY;
        for i in 0..4 {
            let (lx, ly, lz, lw) = (self.x[i], self.y[i], self.z[i], self.w[i]);
            let (rx, ry, rz, rw) = (rhs.x[i],  rhs.y[i],  rhs.z[i],  rhs.w[i]);
            out.x[i] = lw*rx + lx*rw + ly*rz - lz*ry;
            out.y[i] = lw*ry - lx*rz + ly*rw + lz*rx;
            out.z[i] = lw*rz + lx*ry - ly*rx + lz*rw;
            out.w[i] = lw*rw - lx*rx - ly*ry - lz*rz;
        }
        out
    }

    // ── Interpolation ──────────────────────────────────────────────────────────

    #[inline]
    pub fn nlerp(self, rhs: Self, t: f32x4) -> Self {
        // Shortest-path: flip rhs where dot < 0
        let mut rx = rhs.x;
        let mut ry = rhs.y;
        let mut rz = rhs.z;
        let mut rw = rhs.w;
        for i in 0..4 {
            let d = self.x[i]*rhs.x[i] + self.y[i]*rhs.y[i]
                  + self.z[i]*rhs.z[i] + self.w[i]*rhs.w[i];
            if d < 0.0 {
                rx[i] = -rhs.x[i]; ry[i] = -rhs.y[i];
                rz[i] = -rhs.z[i]; rw[i] = -rhs.w[i];
            }
        }
        let lerp = |a: [f32; 4], b: [f32; 4]| -> [f32; 4] {
            core::array::from_fn(|i| a[i] + (b[i] - a[i]) * t.0[i])
        };
        Self {
            x: lerp(self.x, rx),
            y: lerp(self.y, ry),
            z: lerp(self.z, rz),
            w: lerp(self.w, rw),
        }.normalize()
    }

    // ── Rotation ──────────────────────────────────────────────────────────────

    /// Rotate 4 vectors by 4 quaternions simultaneously (scalar fallback).
    ///
    /// t = 2 * cross(q.xyz, v);  result = v + w*t + cross(q.xyz, t)
    #[inline]
    pub fn rotate(self, v: Vec3x4) -> Vec3x4 {
        let qxyz = Vec3x4 { x: self.x, y: self.y, z: self.z };
        let cross1 = qxyz.cross(v);
        let t = Vec3x4 {
            x: cross1.x.map(|v| v * 2.0),
            y: cross1.y.map(|v| v * 2.0),
            z: cross1.z.map(|v| v * 2.0),
        };
        let wt = Vec3x4 {
            x: core::array::from_fn(|i| self.w[i] * t.x[i]),
            y: core::array::from_fn(|i| self.w[i] * t.y[i]),
            z: core::array::from_fn(|i| self.w[i] * t.z[i]),
        };
        let cross2 = qxyz.cross(t);
        Vec3x4 {
            x: core::array::from_fn(|i| v.x[i] + wt.x[i] + cross2.x[i]),
            y: core::array::from_fn(|i| v.y[i] + wt.y[i] + cross2.y[i]),
            z: core::array::from_fn(|i| v.z[i] + wt.z[i] + cross2.z[i]),
        }
    }

    // ── Predicates ────────────────────────────────────────────────────────────

    #[inline]
    pub fn is_finite(self) -> bool {
        self.to_array().iter().all(|q| q.is_finite())
    }
}

impl Mul for QuatX4 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self { self.mul_quatx4(rhs) }
}

impl Default for QuatX4 { fn default() -> Self { Self::IDENTITY } }

impl fmt::Debug for QuatX4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let a = self.to_array();
        write!(f, "QuatX4([{:?}, {:?}, {:?}, {:?}])", a[0], a[1], a[2], a[3])
    }
}
impl fmt::Display for QuatX4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let a = self.to_array();
        write!(f, "[{}, {}, {}, {}]", a[0], a[1], a[2], a[3])
    }
}
impl From<[Quat; 4]> for QuatX4 {
    fn from(a: [Quat; 4]) -> Self { Self::from_slice(&a) }
}
impl From<QuatX4> for [Quat; 4] {
    fn from(v: QuatX4) -> Self { v.to_array() }
      }
