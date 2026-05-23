// crates/mid-math/src/wide/float/neon/quatx4.rs
//! 4 quaternions packed SoA — NEON, aarch64.
//!
//! ## vld4q_f32 for quaternion construction
//!
//! Quat is `#[repr(transparent)] float32x4_t` with lane layout `[x, y, z, w]`.
//! Four Quats in memory: `[x0,y0,z0,w0, x1,y1,z1,w1, ...]` — 16 floats.
//!
//! `vld4q_f32` deinterleaves ALL FOUR components simultaneously in ONE instruction:
//!   val[0]=[x0..x3], val[1]=[y0..y3], val[2]=[z0..z3], val[3]=[w0..w3]
//!
//! vs SSE2 which needs 4 separate `_mm_set_ps` for the same construction.
//! `vst4q_f32` reverses this in one instruction for `to_array`.
//!
//! ## Hamilton product with FMA
//!
//! `vfmaq_f32(acc, b, c)` = `acc + b*c` (mandatory FMA on AArch64).
//! `vfmsq_f32(acc, b, c)` = `acc - b*c` (FMA-subtract, also mandatory).
//! The 4-component Hamilton product uses 12 FMA/FMS operations across all 4 lanes.

use core::arch::aarch64::*;
use core::fmt;
use core::ops::Mul;

use crate::Quat;
use crate::EPSILON;
use super::f32x4::{f32x4, rsqrt_nr};
use super::vec3x4::Vec3x4;

/// 4 quaternions in SoA layout. 64 bytes, 16-byte aligned. Backed by 4 × `float32x4_t`.
#[derive(Clone, Copy)]
#[repr(C, align(16))]
pub struct QuatX4 {
    pub x: float32x4_t,
    pub y: float32x4_t,
    pub z: float32x4_t,
    pub w: float32x4_t,
}

impl QuatX4 {
    pub const IDENTITY: Self = Self {
        x: unsafe { core::mem::transmute([0.0f32; 4]) },
        y: unsafe { core::mem::transmute([0.0f32; 4]) },
        z: unsafe { core::mem::transmute([0.0f32; 4]) },
        w: unsafe { core::mem::transmute([1.0f32, 1.0, 1.0, 1.0]) },
    };

    // ── Constructors ──────────────────────────────────────────────────────────

    /// Build from a slice of 4 Quats using `vld4q_f32` — ONE instruction for all 4 components.
    ///
    /// Quat is `#[repr(transparent)] float32x4_t` with `[x, y, z, w]` layout.
    /// `vld4q_f32` loads `[x0,y0,z0,w0, x1,y1,z1,w1, ...]` and deinterleaves into
    /// `val[0]=[x0..x3]`, `val[1]=[y0..y3]`, `val[2]=[z0..z3]`, `val[3]=[w0..w3]`.
    #[inline]
    pub fn from_slice(s: &[Quat; 4]) -> Self {
        unsafe {
            let loaded = vld4q_f32(s.as_ptr() as *const f32);
            Self { x: loaded.0, y: loaded.1, z: loaded.2, w: loaded.3 }
        }
    }

    #[inline(always)]
    pub fn from_quats(a: Quat, b: Quat, c: Quat, d: Quat) -> Self {
        Self::from_slice(&[a, b, c, d])
    }

    #[inline(always)]
    pub fn splat(q: Quat) -> Self {
        unsafe {
            Self {
                x: vdupq_n_f32(q.x),
                y: vdupq_n_f32(q.y),
                z: vdupq_n_f32(q.z),
                w: vdupq_n_f32(q.w),
            }
        }
    }

    /// Extract all 4 quaternions using `vst4q_f32` — ONE instruction SoA→AoS.
    #[inline]
    pub fn to_array(self) -> [Quat; 4] {
        unsafe {
            let packed = float32x4x4_t(self.x, self.y, self.z, self.w);
            let mut out = [Quat::IDENTITY; 4];
            // vst4q_f32 interleaves back to [x0,y0,z0,w0, x1,y1,z1,w1, ...] = 4 Quats ✓
            vst4q_f32(out.as_mut_ptr() as *mut f32, packed);
            out
        }
    }

    #[inline]
    pub fn get(self, lane: usize) -> Quat {
        assert!(lane < 4, "QuatX4::get — lane {lane} out of bounds (max 3)");
        self.to_array()[lane]
    }

    // ── Core ops ──────────────────────────────────────────────────────────────

    /// 4 independent dot products. Returns `f32x4[i] = dot(self[i], rhs[i])`.
    #[inline(always)]
    pub fn dot(self, rhs: Self) -> f32x4 {
        unsafe {
            let xx = vmulq_f32(self.x, rhs.x);
            let yy = vmulq_f32(self.y, rhs.y);
            let zz = vmulq_f32(self.z, rhs.z);
            let ww = vmulq_f32(self.w, rhs.w);
            f32x4(vaddq_f32(vaddq_f32(xx, yy), vaddq_f32(zz, ww)))
        }
    }

    #[inline(always)] pub fn length_sq(self) -> f32x4 { self.dot(self) }

    /// Normalize all 4 quaternions. Degenerate lanes → identity (0,0,0,1).
    #[inline]
    pub fn normalize(self) -> Self {
        unsafe {
            let lsq     = self.length_sq().0;
            let inv_len = rsqrt_nr(lsq);
            let eps     = vdupq_n_f32(EPSILON);
            let len     = vsqrtq_f32(lsq);
            let ok      = vcgtq_f32(len, eps);
            // Where ok: normalized value. Where not: identity component.
            let id_x = vdupq_n_f32(0.0);
            let id_w = vdupq_n_f32(1.0);
            Self {
                x: vbslq_f32(ok, vmulq_f32(self.x, inv_len), id_x),
                y: vbslq_f32(ok, vmulq_f32(self.y, inv_len), id_x),
                z: vbslq_f32(ok, vmulq_f32(self.z, inv_len), id_x),
                w: vbslq_f32(ok, vmulq_f32(self.w, inv_len), id_w),
            }
        }
    }

    /// Conjugate: negate xyz, keep w. Direct `vnegq_f32` — no XOR trick needed.
    #[inline(always)]
    pub fn conjugate(self) -> Self {
        unsafe {
            Self {
                x: vnegq_f32(self.x),
                y: vnegq_f32(self.y),
                z: vnegq_f32(self.z),
                w: self.w,
            }
        }
    }

    // ── Hamilton product ───────────────────────────────────────────────────────
    //
    // For each of 4 lane-pairs simultaneously using FMA/FMS:
    //   result.x = lw*rx + lx*rw + ly*rz - lz*ry
    //   result.y = lw*ry - lx*rz + ly*rw + lz*rx
    //   result.z = lw*rz + lx*ry - ly*rx + lz*rw
    //   result.w = lw*rw - lx*rx - ly*ry - lz*rz
    //
    // FMA reduces the accumulation ops vs SSE2 (mandatory on AArch64).
    // vfmaq_f32(a, b, c) = a + b*c
    // vfmsq_f32(a, b, c) = a - b*c

    #[inline(always)]
    pub fn mul_quatx4(self, rhs: Self) -> Self {
        unsafe {
            let (lx, ly, lz, lw) = (self.x, self.y, self.z, self.w);
            let (rx, ry, rz, rw) = (rhs.x,  rhs.y,  rhs.z,  rhs.w);

            // result.x = lw*rx + lx*rw + ly*rz - lz*ry
            let x = vfmsq_f32(
                vfmaq_f32(vfmaq_f32(vmulq_f32(lw, rx), lx, rw), ly, rz),
                lz, ry,
            );
            // result.y = lw*ry - lx*rz + ly*rw + lz*rx
            let y = vfmaq_f32(
                vfmaq_f32(vfmsq_f32(vmulq_f32(lw, ry), lx, rz), ly, rw),
                lz, rx,
            );
            // result.z = lw*rz + lx*ry - ly*rx + lz*rw
            let z = vfmaq_f32(
                vfmsq_f32(vfmaq_f32(vmulq_f32(lw, rz), lx, ry), ly, rx),
                lz, rw,
            );
            // result.w = lw*rw - lx*rx - ly*ry - lz*rz
            let w = vfmsq_f32(
                vfmsq_f32(vfmsq_f32(vmulq_f32(lw, rw), lx, rx), ly, ry),
                lz, rz,
            );

            Self { x, y, z, w }
        }
    }

    // ── Interpolation ──────────────────────────────────────────────────────────

    /// Normalised linear interpolation — 4 pairs simultaneously.
    ///
    /// Branchless shortest-path: XOR sign bit into rhs where dot < 0.
    /// FMA lerp: one instruction per component per lane.
    #[inline]
    pub fn nlerp(self, rhs: Self, t: f32x4) -> Self {
        unsafe {
            let dot = self.dot(rhs).0;
            // Extract sign bit of dot: 0x8000_0000 per lane where dot < 0.
            let sign_bit = vandq_u32(
                vreinterpretq_u32_f32(dot),
                vreinterpretq_u32_f32(vdupq_n_f32(-0.0)),
            );
            // XOR sign bit into all rhs components — flips sign where dot was negative.
            let flip = |v: float32x4_t| -> float32x4_t {
                vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(v), sign_bit))
            };
            let rx = flip(rhs.x); let ry = flip(rhs.y);
            let rz = flip(rhs.z); let rw = flip(rhs.w);
            // FMA lerp: self + (rhs_adj - self) * t
            let lerped = Self {
                x: vfmaq_f32(self.x, vsubq_f32(rx, self.x), t.0),
                y: vfmaq_f32(self.y, vsubq_f32(ry, self.y), t.0),
                z: vfmaq_f32(self.z, vsubq_f32(rz, self.z), t.0),
                w: vfmaq_f32(self.w, vsubq_f32(rw, self.w), t.0),
            };
            lerped.normalize()
        }
    }

    // ── Rotation ──────────────────────────────────────────────────────────────

    /// Rotate 4 vectors by 4 quaternions simultaneously using FMA.
    ///
    /// `t = 2 * cross(q.xyz, v);  result = v + w*t + cross(q.xyz, t)`
    #[inline]
    pub fn rotate(self, v: Vec3x4) -> Vec3x4 {
        unsafe {
            let qxyz = Vec3x4 { x: self.x, y: self.y, z: self.z };
            let cross1 = qxyz.cross(v);
            let two = vdupq_n_f32(2.0);
            let t = Vec3x4 {
                x: vmulq_f32(two, cross1.x),
                y: vmulq_f32(two, cross1.y),
                z: vmulq_f32(two, cross1.z),
            };
            // result = v + w*t + cross(qxyz, t)  — use FMA for w*t accumulation
            let cross2 = qxyz.cross(t);
            Vec3x4 {
                x: vfmaq_f32(vaddq_f32(v.x, cross2.x), self.w, t.x),
                y: vfmaq_f32(vaddq_f32(v.y, cross2.y), self.w, t.y),
                z: vfmaq_f32(vaddq_f32(v.z, cross2.z), self.w, t.z),
            }
        }
    }

    #[inline]
    pub fn is_finite(self) -> bool { self.to_array().iter().all(|q| q.is_finite()) }
}

impl Mul for QuatX4 {
    type Output = Self;
    #[inline(always)] fn mul(self, r: Self) -> Self { self.mul_quatx4(r) }
}

impl PartialEq for QuatX4 {
    fn eq(&self, r: &Self) -> bool {
        unsafe {
            vminvq_u32(vceqq_f32(self.x, r.x)) == u32::MAX
                && vminvq_u32(vceqq_f32(self.y, r.y)) == u32::MAX
                && vminvq_u32(vceqq_f32(self.z, r.z)) == u32::MAX
                && vminvq_u32(vceqq_f32(self.w, r.w)) == u32::MAX
        }
    }
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
impl From<[Quat; 4]> for QuatX4 { #[inline] fn from(a: [Quat; 4]) -> Self { Self::from_slice(&a) } }
impl From<QuatX4> for [Quat; 4] { #[inline] fn from(v: QuatX4) -> Self { v.to_array() } }
