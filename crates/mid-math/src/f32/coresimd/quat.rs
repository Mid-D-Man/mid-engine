// crates/mid-math/src/f32/coresimd/quat.rs
//! Quat backed by `f32x4` (Rust portable SIMD).
//!
//! The `mul_quat` uses the same sign-control vector algorithm as SSE2/NEON/WASM.

use core::fmt;
use core::ops::{Add, Mul, MulAssign, Neg, Sub};
use core::simd::prelude::*;
use core::simd::{cmp::SimdPartialEq, cmp::SimdPartialOrd, num::SimdFloat};
use std::simd::StdFloat;

use super::{dot4, dot4_into_f32x4, f32x4_bitand, f32x4_bitxor};
use crate::f32::coresimd::vec3::Vec3;
use crate::f32::coresimd::mat4::Mat4;
use crate::f32::math;
use crate::impl_vec4_deref;
use crate::EPSILON;

// ── Sign-control constants ────────────────────────────────────────────────────

const CONTROL_WZYX: f32x4 = f32x4::from_array([ 1.0, -1.0,  1.0, -1.0]);
const CONTROL_ZWXY: f32x4 = f32x4::from_array([ 1.0,  1.0, -1.0, -1.0]);
const CONTROL_YXWZ: f32x4 = f32x4::from_array([-1.0,  1.0,  1.0, -1.0]);
const CONJ_SIGN: f32x4 = f32x4::from_array([-0.0, -0.0, -0.0, 0.0]);

// ── Type ──────────────────────────────────────────────────────────────────────

/// Quaternion. 16 bytes, 16-byte aligned. Backed by `f32x4`.
/// Convention: (x, y, z, w). Layout: lane0=x, lane1=y, lane2=z, lane3=w.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct Quat(pub(crate) f32x4);

impl_vec4_deref!(Quat);

impl Quat {
    pub const IDENTITY: Self = Self(f32x4::from_array([0.0, 0.0, 0.0, 1.0]));
    pub const ZERO:     Self = Self(f32x4::from_array([0.0; 4]));

    // ── Constructors ─────────────────────────────────────────────────────────

    #[inline(always)]
    pub fn new(x: f32, y: f32, z: f32, w: f32) -> Self {
        Self(f32x4::from_array([x, y, z, w]))
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

    // ── Core ops ──────────────────────────────────────────────────────────────

    #[inline(always)]
    pub fn dot(self, rhs: Self) -> f32 { dot4(self.0, rhs.0) }

    #[inline(always)] pub fn length_sq(self) -> f32 { self.dot(self) }
    #[inline] pub fn length(self) -> f32 { self.length_sq().sqrt() }

    /// Normalize. Falls back to IDENTITY where length ≤ EPSILON.
    #[inline]
    pub fn normalize(self) -> Self {
        let len  = dot4_into_f32x4(self.0, self.0).sqrt();
        let n    = Self(self.0 / len);
        let ok   = len.simd_gt(f32x4::splat(EPSILON));
        Self(ok.select(n.0, Self::IDENTITY.0))
    }

    /// Fast normalize — **no** IDENTITY fallback guard.
    ///
    /// Precondition: `self` must not be near-zero length. Always satisfied
    /// when input is a lerped blend of two unit quaternions (nlerp/slerp).
    #[inline(always)]
    pub(crate) fn normalize_fast(self) -> Self {
        let len = dot4_into_f32x4(self.0, self.0).sqrt();
        Self(self.0 / len)
    }

    /// Conjugate: negate xyz, keep w. Single XOR via sign-bit mask.
    #[inline]
    pub fn conjugate(self) -> Self {
        Self(f32x4_bitxor(self.0, CONJ_SIGN))
    }

    #[inline]
    pub fn inverse(self) -> Self {
        let sq = self.length_sq();
        if sq < EPSILON { return Self::IDENTITY; }
        Self(self.conjugate().0 * f32x4::splat(1.0 / sq))
    }

    #[inline]
    pub fn rotate(self, v: Vec3) -> Vec3 {
        let qv = Vec3::new(self.x, self.y, self.z);
        let t  = 2.0 * qv.cross(v);
        v + self.w * t + qv.cross(t)
    }

    pub fn mul_quat(self, rhs: Self) -> Self {
        let lhs = self.0;
        let r   = rhs.0;

        let r_xxxx = simd_swizzle!(lhs, [0, 0, 0, 0]);
        let r_yyyy = simd_swizzle!(lhs, [1, 1, 1, 1]);
        let r_zzzz = simd_swizzle!(lhs, [2, 2, 2, 2]);
        let r_wwww = simd_swizzle!(lhs, [3, 3, 3, 3]);

        let lxrw_lyrw_lzrw_lwrw = r_wwww * r;
        let l_wzyx                = simd_swizzle!(r, [3, 2, 1, 0]);
        let lwrx_lzrx_lyrx_lxrx  = r_xxxx * l_wzyx;
        let l_zwxy                = simd_swizzle!(l_wzyx, [1, 0, 3, 2]);
        let lwrx_nlzrx_lyrx_nlxrx = lwrx_lzrx_lyrx_lxrx * CONTROL_WZYX;
        let lzry_lwry_lxry_lyry   = r_yyyy * l_zwxy;
        let l_yxwz                = simd_swizzle!(l_zwxy, [3, 2, 1, 0]);
        let lzry_lwry_nlxry_nlyry = lzry_lwry_lxry_lyry * CONTROL_ZWXY;
        let lyrz_lxrz_lwrz_lzrz  = r_zzzz * l_yxwz;
        let result0               = lxrw_lyrw_lzrw_lwrw + lwrx_nlzrx_lyrx_nlxrx;
        let nlyrz_lxrz_lwrz_nlzrz = lyrz_lxrz_lwrz_lzrz * CONTROL_YXWZ;
        let result1               = lzry_lwry_nlxry_nlyry + nlyrz_lxrz_lwrz_nlzrz;

        Self(result0 + result1)
    }

    // ── Interpolation ──────────────────────────────────────────────────────────

    /// Normalised linear interpolation.
    ///
    /// Uses `normalize_fast()` — blend of two unit quats is always non-zero.
    #[inline]
    pub fn nlerp(self, rhs: Self, t: f32) -> Self {
        let dot_val  = self.dot(rhs);
        let sign_bit = f32x4_bitand(f32x4::splat(dot_val), f32x4::splat(-0.0));
        let rhs_adj  = Self(f32x4_bitxor(rhs.0, sign_bit));
        let lerped   = Self(self.0 + (rhs_adj.0 - self.0) * f32x4::splat(t));
        lerped.normalize_fast()
    }

    pub fn slerp(self, mut rhs: Self, t: f32) -> Self {
        let mut cos_theta = self.dot(rhs);
        if cos_theta < 0.0 { rhs = -rhs; cos_theta = -cos_theta; }
        if cos_theta > 1.0 - EPSILON { return self.nlerp(rhs, t); }
        let angle     = math::acos_approx(cos_theta);
        let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();
        let scale1    = math::sin(angle * (1.0 - t));
        let scale2    = math::sin(angle * t);
        let blended   = self.0 * f32x4::splat(scale1) + rhs.0 * f32x4::splat(scale2);
        // blended / sin_theta is ≈unit — normalize_fast() corrects FP drift.
        Self(blended / f32x4::splat(sin_theta)).normalize_fast()
    }

    // ── Conversion ─────────────────────────────────────────────────────────────

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
        self.x.is_finite() && self.y.is_finite()
            && self.z.is_finite() && self.w.is_finite()
    }
}

// ── Operators ─────────────────────────────────────────────────────────────────

impl Mul for Quat {
    type Output = Self;
    #[inline] fn mul(self, r: Self) -> Self { self.mul_quat(r) }
}
impl MulAssign for Quat {
    #[inline] fn mul_assign(&mut self, r: Self) { *self = self.mul_quat(r); }
}
impl Neg for Quat {
    type Output = Self;
    #[inline] fn neg(self) -> Self { Self(-self.0) }
}
impl Add for Quat {
    type Output = Self;
    #[inline] fn add(self, r: Self) -> Self { Self(self.0 + r.0) }
}
impl Sub for Quat {
    type Output = Self;
    #[inline] fn sub(self, r: Self) -> Self { Self(self.0 - r.0) }
}
impl Mul<f32> for Quat {
    type Output = Self;
    #[inline] fn mul(self, s: f32) -> Self { Self(self.0 * f32x4::splat(s)) }
}

impl PartialEq for Quat {
    #[inline]
    fn eq(&self, rhs: &Self) -> bool {
        (self.0.simd_eq(rhs.0).to_bitmask() & 0b1111) == 0b1111
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

/// Build a quaternion from three orthonormal rotation-matrix axis columns.
/// See sse2::quat::quat_from_rotation_axes for why this is a plain function
/// rather than a QuatExt impl (avoids conflicting with the generic
/// `impl QuatExt for crate::Quat` whenever they'd resolve to the same type).
pub(crate) fn quat_from_rotation_axes(x_axis: Vec3, y_axis: Vec3, z_axis: Vec3) -> Quat {
    let (m00, m10, m20) = (x_axis.x, x_axis.y, x_axis.z);
    let (m01, m11, m21) = (y_axis.x, y_axis.y, y_axis.z);
    let (m02, m12, m22) = (z_axis.x, z_axis.y, z_axis.z);

    if m22 <= 0.0 {
        let dif10 = m11 - m00;
        let omm22 = 1.0 - m22;
        if dif10 <= 0.0 {
            let four_xsq = omm22 - dif10;
            let inv4x = 0.5 / four_xsq.sqrt();
            Quat::new(four_xsq * inv4x, (m10 + m01) * inv4x, (m20 + m02) * inv4x, (m21 - m12) * inv4x)
        } else {
            let four_ysq = omm22 + dif10;
            let inv4y = 0.5 / four_ysq.sqrt();
            Quat::new((m10 + m01) * inv4y, four_ysq * inv4y, (m21 + m12) * inv4y, (m02 - m20) * inv4y)
        }
    } else {
        let sum10 = m11 + m00;
        let opm22 = 1.0 + m22;
        if sum10 <= 0.0 {
            let four_zsq = opm22 - sum10;
            let inv4z = 0.5 / four_zsq.sqrt();
            Quat::new((m20 + m02) * inv4z, (m21 + m12) * inv4z, four_zsq * inv4z, (m10 - m01) * inv4z)
        } else {
            let four_wsq = opm22 + sum10;
            let inv4w = 0.5 / four_wsq.sqrt();
            Quat::new((m21 - m12) * inv4w, (m02 - m20) * inv4w, (m10 - m01) * inv4w, four_wsq * inv4w)
        }
    }
            }
