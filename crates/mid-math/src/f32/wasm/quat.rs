// crates/mid-math/src/f32/wasm/quat.rs
//! Quaternion backed by `v128` on wasm32/wasm64 with simd128.
//!
//! Convention: (x, y, z, w).  Lane layout: 0=x, 1=y, 2=z, 3=w.
//!
//! WASM notes vs SSE2:
//!   v128_andnot(a,b) = a & ~b  ← note reversed from SSE2 _mm_andnot_ps(a,b) = ~a & b
//!   f32x4_neg — direct, no XOR -0.0 needed
//!   No WASM equivalent of _mm_movemask_ps; use u32x4_bitmask instead

#[cfg(target_arch = "wasm32")]
use core::arch::wasm32::*;
#[cfg(target_arch = "wasm64")]
use core::arch::wasm64::*;

use core::fmt;
use core::ops::{Add, Mul, MulAssign, Neg, Sub};

use crate::f32::wasm::vec3::Vec3;
use crate::f32::wasm::mat4::Mat4;
use crate::f32::math;
use crate::impl_vec4_deref;
use crate::wasm::{dot4_into_v128, v128_from_f32x4};
use crate::EPSILON;

#[repr(C)]
union UnionCast { f: [f32; 4], v: Quat }

// Sign-control vectors for mul_quat — placed in .rodata, single load on WASM.
const CONTROL_WZYX: v128 = v128_from_f32x4([ 1.0, -1.0,  1.0, -1.0]);
const CONTROL_ZWXY: v128 = v128_from_f32x4([ 1.0,  1.0, -1.0, -1.0]);
const CONTROL_YXWZ: v128 = v128_from_f32x4([-1.0,  1.0,  1.0, -1.0]);

// Conjugate sign mask: negate xyz, keep w.
const CONJ_SIGN: v128 = v128_from_f32x4([-0.0, -0.0, -0.0, 0.0]);

/// Quaternion. 16 bytes, 16-byte aligned.  Lane layout: [x, y, z, w].
///
/// Backed by `v128` on wasm32/wasm64 with simd128.
///
/// **C interop:** use [`CQuat`][crate::ffi::types::CQuat] at the FFI boundary.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct Quat(pub(crate) v128);

impl_vec4_deref!(Quat);

impl Quat {
    pub const IDENTITY: Self = unsafe { UnionCast { f: [0.0, 0.0, 0.0, 1.0] }.v };
    pub const ZERO:     Self = unsafe { UnionCast { f: [0.0; 4] }.v };

    // ── Constructors ──────────────────────────────────────────────────────────

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

    // ── Core ops ──────────────────────────────────────────────────────────────

    #[inline]
    pub fn dot(self, rhs: Self) -> f32 {
        unsafe {
            let mul = f32x4_mul(self.0, rhs.0);
            // horizontal sum of all 4 lanes
            let lo = i32x4_shuffle::<0, 1, 4, 5>(mul, mul);
            let hi = i32x4_shuffle::<2, 3, 6, 7>(mul, mul);
            let s  = f32x4_add(lo, hi);
            let s2 = i32x4_shuffle::<1, 0, 5, 4>(s, s);
            f32x4_extract_lane::<0>(f32x4_add(s, s2))
        }
    }

    #[inline] pub fn length_sq(self) -> f32 { self.dot(self) }
    #[inline] pub fn length(self)    -> f32 { self.length_sq().sqrt() }

    /// Normalise.  Returns `IDENTITY` for near-zero-length quaternions.
    ///
    /// Uses `v128_andnot(a, b) = a & ~b` (WASM) vs SSE2's `_mm_andnot_ps(a,b) = ~a & b`.
    /// Argument order is swapped to achieve the same "IDENTITY where len<=eps, n where len>eps".
    #[inline]
    pub fn normalize(self) -> Self {
        unsafe {
            let len  = f32x4_sqrt(dot4_into_v128(self.0, self.0));
            let n    = Self(f32x4_div(self.0, len));
            let ok   = f32x4_gt(len, f32x4_splat(EPSILON));
            // keep n where len > EPSILON
            let keep = v128_and(n.0, ok);
            // IDENTITY where len <= EPSILON: v128_andnot(IDENTITY, ok) = IDENTITY & ~ok
            let alt  = v128_andnot(Self::IDENTITY.0, ok);
            Self(v128_or(keep, alt))
        }
    }

    /// Conjugate: negate xyz, keep w.  Single XOR on WASM.
    #[inline]
    pub fn conjugate(self) -> Self {
        Self(unsafe { v128_xor(self.0, CONJ_SIGN) })
    }

    #[inline]
    pub fn inverse(self) -> Self {
        let sq = self.length_sq();
        if sq < EPSILON { return Self::IDENTITY; }
        Self(unsafe { f32x4_mul(self.conjugate().0, f32x4_splat(1.0 / sq)) })
    }

    #[inline]
    pub fn rotate(self, v: Vec3) -> Vec3 {
        let qv = Vec3::new(self.x, self.y, self.z);
        let t  = 2.0 * qv.cross(v);
        v + self.w * t + qv.cross(t)
    }

    /// Quaternion product using WZYX / ZWXY / YXWZ sign-control pattern.
    ///
    /// Identical algorithm to SSE2 and NEON mul_quat; only intrinsic names differ.
    pub fn mul_quat(self, rhs: Self) -> Self {
        unsafe {
            let lhs = self.0;
            let rhs = rhs.0;

            let r_xxxx = i32x4_shuffle::<0, 0, 4, 4>(lhs, lhs);
            let r_yyyy = i32x4_shuffle::<1, 1, 5, 5>(lhs, lhs);
            let r_zzzz = i32x4_shuffle::<2, 2, 6, 6>(lhs, lhs);
            let r_wwww = i32x4_shuffle::<3, 3, 7, 7>(lhs, lhs);

            let lxrw_etc = f32x4_mul(r_wwww, rhs);

            // WZYX permutation of rhs: [w,z,y,x] → shuffle(3,2,1,0) from same register
            let l_wzyx = i32x4_shuffle::<3, 2, 5, 4>(rhs, rhs);
            let lwrx_etc = f32x4_mul(r_xxxx, l_wzyx);
            // ZWXY: reverse l_wzyx
            let l_zwxy = i32x4_shuffle::<1, 0, 7, 6>(l_wzyx, l_wzyx);
            let lwrx_signed = f32x4_mul(lwrx_etc, CONTROL_WZYX);

            let lzry_etc = f32x4_mul(r_yyyy, l_zwxy);
            // YXWZ: reverse l_zwxy
            let l_yxwz = i32x4_shuffle::<3, 2, 5, 4>(l_zwxy, l_zwxy);
            let lzry_signed = f32x4_mul(lzry_etc, CONTROL_ZWXY);

            let lyrz_etc = f32x4_mul(r_zzzz, l_yxwz);
            let result0 = f32x4_add(lxrw_etc, lwrx_signed);
            let lyrz_signed = f32x4_mul(lyrz_etc, CONTROL_YXWZ);
            let result1 = f32x4_add(lzry_signed, lyrz_signed);

            Self(f32x4_add(result0, result1))
        }
    }

    // ── Interpolation ──────────────────────────────────────────────────────────

    #[inline]
    pub fn nlerp(self, rhs: Self, t: f32) -> Self {
        unsafe {
            let dot = self.dot(rhs);
            // Flip rhs if dot < 0
            let sign_mask = f32x4_splat(-0.0);
            let dot_v     = f32x4_splat(dot);
            let sign_bit  = v128_and(dot_v, sign_mask);
            let rhs_adj   = v128_xor(rhs.0, sign_bit);
            let tt        = f32x4_splat(t);
            let diff      = f32x4_sub(rhs_adj, self.0);
            Self(f32x4_add(self.0, f32x4_mul(diff, tt))).normalize()
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
            let blended = f32x4_add(
                f32x4_mul(self.0, f32x4_splat(s0)),
                f32x4_mul(rhs.0,  f32x4_splat(s1)),
            );
            Self(f32x4_div(blended, f32x4_splat(sin_theta))).normalize()
        }
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
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite() && self.w.is_finite()
    }
}

impl Mul for Quat {
    type Output = Self;
    #[inline] fn mul(self, r: Self) -> Self { self.mul_quat(r) }
}
impl MulAssign for Quat {
    #[inline] fn mul_assign(&mut self, r: Self) { *self = self.mul_quat(r); }
}
impl Neg for Quat {
    type Output = Self;
    #[inline] fn neg(self) -> Self { Self(unsafe { f32x4_neg(self.0) }) }
}
impl Add for Quat {
    type Output = Self;
    #[inline] fn add(self, r: Self) -> Self { Self(unsafe { f32x4_add(self.0, r.0) }) }
}
impl Sub for Quat {
    type Output = Self;
    #[inline] fn sub(self, r: Self) -> Self { Self(unsafe { f32x4_sub(self.0, r.0) }) }
}
impl Mul<f32> for Quat {
    type Output = Self;
    #[inline] fn mul(self, s: f32) -> Self { Self(unsafe { f32x4_mul(self.0, f32x4_splat(s)) }) }
}

impl PartialEq for Quat {
    #[inline]
    fn eq(&self, rhs: &Self) -> bool {
        unsafe { (u32x4_bitmask(f32x4_eq(self.0, rhs.0)) & 0b1111) == 0b1111 }
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
