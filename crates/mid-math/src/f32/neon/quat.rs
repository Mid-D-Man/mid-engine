// crates/mid-math/src/f32/neon/quat.rs  ← CLEAN FINAL VERSION
//! Quaternion backed by `float32x4_t` on aarch64.

use core::arch::aarch64::*;
use core::fmt;
use core::ops::{Add, Mul, MulAssign, Neg, Sub};

use crate::f32::neon::vec3::Vec3;
use crate::f32::neon::mat4::Mat4;
use crate::f32::math;
use crate::impl_vec4_deref;
use crate::EPSILON;

#[repr(C)] union UnionCast { f: [f32; 4], v: Quat }
#[repr(C)] union SignCast  { f: [f32; 4], v: float32x4_t }

// Sign-control vectors for mul_quat (placed in .rodata, single LDR on AArch64).
const QMUL_WZYX: float32x4_t = unsafe { SignCast { f: [ 1.0, -1.0,  1.0, -1.0] }.v };
const QMUL_ZWXY: float32x4_t = unsafe { SignCast { f: [ 1.0,  1.0, -1.0, -1.0] }.v };
const QMUL_YXWZ: float32x4_t = unsafe { SignCast { f: [-1.0,  1.0,  1.0, -1.0] }.v };
// Mask for conjugate: negate xyz, keep w (sign-bit XOR).
const CONJ_SIGN: float32x4_t = unsafe { SignCast { f: [-0.0, -0.0, -0.0, 0.0] }.v };

/// Quaternion. 16 bytes, 16-byte aligned. Lane layout: [x, y, z, w].
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct Quat(pub(crate) float32x4_t);

impl_vec4_deref!(Quat);

impl Quat {
    pub const IDENTITY: Self = unsafe { UnionCast { f: [0.0, 0.0, 0.0, 1.0] }.v };
    pub const ZERO:     Self = unsafe { UnionCast { f: [0.0; 4] }.v };

    #[inline(always)]
    pub fn new(x: f32, y: f32, z: f32, w: f32) -> Self {
        unsafe { UnionCast { f: [x, y, z, w] }.v }
    }

    #[inline(always)]
    pub fn from_xyzw(x: f32, y: f32, z: f32, w: f32) -> Self { Self::new(x, y, z, w) }

    pub fn from_axis_angle(axis: Vec3, angle_rad: f32) -> Self {
        let (s, c) = math::sin_cos(angle_rad * 0.5);
        let n = axis.normalize();
        Self::new(n.x * s, n.y * s, n.z * s, c)
    }

    pub fn from_euler(roll: f32, pitch: f32, yaw: f32) -> Self {
        let (sx, cx) = math::sin_cos(roll  * 0.5);
        let (sy, cy) = math::sin_cos(pitch * 0.5);
        let (sz, cz) = math::sin_cos(yaw   * 0.5);
        Self::new(
            cz*cy*sx - sz*sy*cx,
            cz*sy*cx + sz*cy*sx,
            sz*cy*cx - cz*sy*sx,
            cz*cy*cx + sz*sy*sx,
        ).normalize()
    }

    pub fn to_euler(self) -> (f32, f32, f32) {
        let sinp  = 2.0 * (self.w * self.y - self.z * self.x);
        let pitch = if sinp.abs() >= 1.0 {
            sinp.signum() * core::f32::consts::FRAC_PI_2
        } else { sinp.asin() };
        let roll = (2.0*(self.w*self.x + self.y*self.z))
            .atan2(1.0 - 2.0*(self.x*self.x + self.y*self.y));
        let yaw  = (2.0*(self.w*self.z + self.x*self.y))
            .atan2(1.0 - 2.0*(self.y*self.y + self.z*self.z));
        (roll, pitch, yaw)
    }

    #[inline(always)]
    pub fn dot(self, rhs: Self) -> f32 {
        unsafe { vaddvq_f32(vmulq_f32(self.0, rhs.0)) }
    }

    #[inline] pub fn length_sq(self) -> f32 { self.dot(self) }
    #[inline] pub fn length(self)    -> f32 { self.length_sq().sqrt() }

    #[inline]
    pub fn normalize(self) -> Self {
        unsafe {
            let dot = vaddvq_f32(vmulq_f32(self.0, self.0));
            if dot < EPSILON { return Self::IDENTITY; }
            Self(vdivq_f32(self.0, vsqrtq_f32(vdupq_n_f32(dot))))
        }
    }

    /// Conjugate: negate xyz, keep w. Single VEOR on AArch64.
    #[inline]
    pub fn conjugate(self) -> Self {
        unsafe {
            Self(vreinterpretq_f32_u32(veorq_u32(
                vreinterpretq_u32_f32(self.0),
                vreinterpretq_u32_f32(CONJ_SIGN),
            )))
        }
    }

    #[inline]
    pub fn inverse(self) -> Self {
        let sq = self.length_sq();
        if sq < EPSILON { return Self::IDENTITY; }
        Self(unsafe { vmulq_n_f32(self.conjugate().0, 1.0 / sq) })
    }

    #[inline]
    pub fn rotate(self, v: Vec3) -> Vec3 {
        let qv = Vec3::new(self.x, self.y, self.z);
        let t  = 2.0 * qv.cross(v);
        v + self.w * t + qv.cross(t)
    }

    /// Quaternion product. NEON permutation derivation (verified lane-by-lane):
    ///   rhs   = [rx, ry, rz, rw]
    ///   wzyx  = [rw, rz, ry, rx]  via vrev64 + vextq::<2>
    ///   zwxy  = [rz, rw, rx, ry]  via vrev64(wzyx)
    ///   yxwz  = [ry, rx, rw, rz]  via vrev64(zwxy) + vextq::<2>
    pub fn mul_quat(self, rhs: Self) -> Self {
        unsafe {
            let lhs = self.0;
            let r   = rhs.0;

            let lw = vdupq_laneq_f32::<3>(lhs);
            let lx = vdupq_laneq_f32::<0>(lhs);
            let ly = vdupq_laneq_f32::<1>(lhs);
            let lz = vdupq_laneq_f32::<2>(lhs);

            // Build permutations of rhs.
            let rev1   = vrev64q_f32(r);
            let l_wzyx = vextq_f32::<2>(rev1, rev1);
            let l_zwxy = vrev64q_f32(l_wzyx);
            let rev2   = vrev64q_f32(l_zwxy);
            let l_yxwz = vextq_f32::<2>(rev2, rev2);

            // Four multiply-then-sign-flip terms.
            let t1 = vmulq_f32(lw, r);
            let t2 = vmulq_f32(vmulq_f32(lx, l_wzyx), QMUL_WZYX);
            let t3 = vmulq_f32(vmulq_f32(ly, l_zwxy), QMUL_ZWXY);
            let t4 = vmulq_f32(vmulq_f32(lz, l_yxwz), QMUL_YXWZ);

            Self(vaddq_f32(vaddq_f32(t1, t2), vaddq_f32(t3, t4)))
        }
    }

    #[inline]
    pub fn nlerp(self, rhs: Self, t: f32) -> Self {
        let rhs_adj = if self.dot(rhs) < 0.0 { -rhs } else { rhs };
        unsafe {
            let t_v  = vdupq_n_f32(t);
            let diff = vsubq_f32(rhs_adj.0, self.0);
            Self(vfmaq_f32(self.0, diff, t_v)).normalize()
        }
    }

    pub fn slerp(self, mut rhs: Self, t: f32) -> Self {
        let mut dot = self.dot(rhs);
        if dot < 0.0 { rhs = -rhs; dot = -dot; }
        if dot > 1.0 - EPSILON { return self.nlerp(rhs, t); }
        let angle     = math::acos_approx(dot);
        let sin_theta = (1.0 - dot * dot).sqrt();
        let s0 = ((1.0 - t) * angle).sin();
        let s1 = (t * angle).sin();
        unsafe {
            let blended = vaddq_f32(
                vmulq_n_f32(self.0, s0),
                vmulq_n_f32(rhs.0,  s1),
            );
            Self(vdivq_f32(blended, vdupq_n_f32(sin_theta))).normalize()
        }
    }

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

    #[inline] pub fn is_normalized(self) -> bool { (self.length_sq() - 1.0).abs() <= 2e-4 }
    #[inline] pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite() && self.w.is_finite()
    }
}

impl Mul     for Quat { type Output = Self; #[inline] fn mul(self, r: Self) -> Self { self.mul_quat(r) } }
impl MulAssign for Quat { #[inline] fn mul_assign(&mut self, r: Self) { *self = self.mul_quat(r); } }
impl Neg for Quat {
    type Output = Self;
    #[inline] fn neg(self) -> Self { Self(unsafe { vnegq_f32(self.0) }) }
}
impl Add for Quat {
    type Output = Self;
    #[inline] fn add(self, r: Self) -> Self { Self(unsafe { vaddq_f32(self.0, r.0) }) }
}
impl Sub for Quat {
    type Output = Self;
    #[inline] fn sub(self, r: Self) -> Self { Self(unsafe { vsubq_f32(self.0, r.0) }) }
}
impl Mul<f32> for Quat {
    type Output = Self;
    #[inline] fn mul(self, s: f32) -> Self { Self(unsafe { vmulq_n_f32(self.0, s) }) }
}
impl PartialEq for Quat {
    #[inline]
    fn eq(&self, rhs: &Self) -> bool {
        unsafe {
            let c = vceqq_f32(self.0, rhs.0);
            vgetq_lane_u32::<0>(c) != 0 && vgetq_lane_u32::<1>(c) != 0
                && vgetq_lane_u32::<2>(c) != 0 && vgetq_lane_u32::<3>(c) != 0
        }
    }
}
impl Default for Quat { fn default() -> Self { Self::IDENTITY } }
impl fmt::Debug for Quat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Quat")
            .field(&self.x).field(&self.y).field(&self.z).field(&self.w).finish()
    }
}
impl fmt::Display for Quat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Quat({:.4}, {:.4}, {:.4}, {:.4})", self.x, self.y, self.z, self.w)
    }
}
