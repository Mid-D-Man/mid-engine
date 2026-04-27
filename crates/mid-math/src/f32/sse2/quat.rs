// crates/mid-math/src/f32/sse2/quat.rs
//! Quaternion backed by `__m128` on x86 / x86_64.
//!
//! Convention: (x, y, z, w) where w is the scalar part.
//! Storage:    lane0=x, lane1=y, lane2=z, lane3=w.
//! Euler convention: ZYX — yaw first, then pitch, then roll.
//!
//! The SIMD multiply uses sign-constant masking (ported from glam's
//! rtm-inspired implementation) — avoids branches and scalar extraction.

use core::fmt;
use core::ops::{Mul, MulAssign, Neg, Add, Sub};

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use crate::sse2::{dot4_into_m128, m128_from_f32x4, m128_sin};
use crate::f32::sse2::vec3::Vec3;
use crate::f32::sse2::mat4::Mat4;
use crate::f32::math;
use crate::EPSILON;
use crate::impl_vec4_deref;

#[repr(C)]
union UnionCast {
    f: [f32; 4],
    v: Quat,
}

/// Quaternion. 16 bytes, 16-byte aligned. Lane layout: [x, y, z, w].
///
/// Backed by `__m128` on x86 / x86_64.
///
/// **C interop:** use [`CQuat`][crate::ffi::types::CQuat] at the FFI boundary.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct Quat(pub(crate) __m128);

// Deref gives .x .y .z .w access on the __m128 storage.
impl_vec4_deref!(Quat);

// ── Sign masks used in SIMD multiply ─────────────────────────────────────────
// These are compile-time constants built from the union trick.
// Each flips the sign of specific lanes to implement the Hamilton product
// without branches.
//
// control_wzyx: [ 1, -1,  1, -1]  (applied to the lhs × wzyx(rhs) term)
// control_zwxy: [ 1,  1, -1, -1]  (applied to the lhs × zwxy(rhs) term)
// control_yxwz: [-1,  1,  1, -1]  (applied to the lhs × yxwz(rhs) term)

const CONTROL_WZYX: __m128 = m128_from_f32x4([ 1.0, -1.0,  1.0, -1.0]);
const CONTROL_ZWXY: __m128 = m128_from_f32x4([ 1.0,  1.0, -1.0, -1.0]);
const CONTROL_YXWZ: __m128 = m128_from_f32x4([-1.0,  1.0,  1.0, -1.0]);

impl Quat {
    // ── Constants ────────────────────────────────────────────────────────────

    /// Identity quaternion — represents no rotation.
    pub const IDENTITY: Self = unsafe { UnionCast { f: [0.0, 0.0, 0.0, 1.0] }.v };
    const ZERO: Self         = unsafe { UnionCast { f: [0.0; 4] }.v };

    // ── Constructors ─────────────────────────────────────────────────────────

    #[inline(always)]
    pub fn new(x: f32, y: f32, z: f32, w: f32) -> Self {
        Self(unsafe { _mm_set_ps(w, z, y, x) })
    }

    #[inline(always)]
    pub fn from_xyzw(x: f32, y: f32, z: f32, w: f32) -> Self {
        Self::new(x, y, z, w)
    }

    /// Build from a unit axis and an angle in radians.
    /// `axis` need not be pre-normalised — normalised internally.
    #[inline]
    pub fn from_axis_angle(axis: Vec3, angle_rad: f32) -> Self {
        let (s, c) = math::sin_cos(angle_rad * 0.5);
        let n = axis.normalize();
        Self::new(n.x * s, n.y * s, n.z * s, c)
    }

    /// Build from Euler angles (radians), ZYX convention.
    /// Applied as: Rz * Ry * Rx.
    pub fn from_euler(roll: f32, pitch: f32, yaw: f32) -> Self {
        let (sx, cx) = math::sin_cos(roll  * 0.5);
        let (sy, cy) = math::sin_cos(pitch * 0.5);
        let (sz, cz) = math::sin_cos(yaw   * 0.5);
        Self::new(
            cz * cy * sx - sz * sy * cx,
            cz * sy * cx + sz * cy * sx,
            sz * cy * cx - cz * sy * sx,
            cz * cy * cx + sz * sy * sx,
        ).normalize()
    }

    // ── Decomposition ─────────────────────────────────────────────────────────

    /// Extract Euler angles (ZYX convention). Returns (roll, pitch, yaw).
    pub fn to_euler(self) -> (f32, f32, f32) {
        let sinp  = 2.0 * (self.w * self.y - self.z * self.x);
        let pitch = if sinp.abs() >= 1.0 {
            sinp.signum() * core::f32::consts::FRAC_PI_2
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
    pub fn dot(self, rhs: Self) -> f32 {
        unsafe { crate::sse2::dot4(self.0, rhs.0) }
    }

    #[inline] pub fn length_sq(self) -> f32 { self.dot(self) }
    #[inline] pub fn length(self)    -> f32 { self.length_sq().sqrt() }

    /// Normalize. Returns IDENTITY if near-zero.
    #[inline]
    pub fn normalize(self) -> Self {
        unsafe {
            let len = _mm_sqrt_ps(dot4_into_m128(self.0, self.0));
            let n   = Self(_mm_div_ps(self.0, len));
            let ok  = _mm_cmpgt_ps(len, _mm_set1_ps(EPSILON));
            // Blend: if ok keep n else use IDENTITY
            let keep = _mm_and_ps(n.0, ok);
            let alt  = _mm_andnot_ps(ok, Self::IDENTITY.0);
            Self(_mm_or_ps(keep, alt))
        }
    }

    /// Conjugate — equivalent to inverse for unit quaternions.
    #[inline]
    pub fn conjugate(self) -> Self {
        // Flip sign of xyz lanes (lanes 0-2), keep w lane (lane 3).
        // sign mask: [-0.0, -0.0, -0.0, 0.0]
        const SIGN: __m128 = m128_from_f32x4([-0.0, -0.0, -0.0, 0.0]);
        Self(unsafe { _mm_xor_ps(self.0, SIGN) })
    }

    /// Inverse. For unit quaternions prefer `conjugate()` — it's faster.
    #[inline]
    pub fn inverse(self) -> Self {
        let sq = self.length_sq();
        if sq < EPSILON { return Self::IDENTITY; }
        let rcp = 1.0 / sq;
        // conjugate / length_sq
        let conj = self.conjugate();
        Self(unsafe { _mm_mul_ps(conj.0, _mm_set1_ps(rcp)) })
    }

    /// Rotate a Vec3. `self` must be normalized.
    ///
    /// Uses the sandwich product via two cross-products (faster than
    /// converting to matrix for a single vector):
    ///   t = 2 * cross(q.xyz, v)
    ///   result = v + w*t + cross(q.xyz, t)
    #[inline]
    pub fn rotate(self, v: Vec3) -> Vec3 {
        let qv = Vec3::new(self.x, self.y, self.z);
        let t  = 2.0 * qv.cross(v);
        v + self.w * t + qv.cross(t)
    }

    // ── Mul (Hamilton product) ─────────────────────────────────────────────────
    //
    // Based on rtm / glam's SIMD quat multiply.
    // Decomposes as four shuffle+multiply+add sequences using sign constants,
    // avoiding any scalar extraction.
    //
    // Hamilton product:
    //   result.x = lw*rx + lx*rw + ly*rz - lz*ry
    //   result.y = lw*ry - lx*rz + ly*rw + lz*rx
    //   result.z = lw*rz + lx*ry - ly*rx + lz*rw
    //   result.w = lw*rw - lx*rx - ly*ry - lz*rz

    #[inline]
    pub fn mul_quat(self, rhs: Self) -> Self {
        unsafe {
            let lhs = self.0;
            let rhs = rhs.0;

            // Broadcast each lane of lhs
            let r_xxxx = _mm_shuffle_ps::<0b00_00_00_00>(lhs, lhs);
            let r_yyyy = _mm_shuffle_ps::<0b01_01_01_01>(lhs, lhs);
            let r_zzzz = _mm_shuffle_ps::<0b10_10_10_10>(lhs, lhs);
            let r_wwww = _mm_shuffle_ps::<0b11_11_11_11>(lhs, lhs);

            // Term 1: lw * [rx, ry, rz, rw]
            let lxrw_lyrw_lzrw_lwrw = _mm_mul_ps(r_wwww, rhs);

            // Shuffle rhs into [rw, rz, ry, rx]
            let l_wzyx = _mm_shuffle_ps::<0b00_01_10_11>(rhs, rhs);

            // Term 2: lx * [rw, rz, ry, rx]
            let lwrx_lzrx_lyrx_lxrx = _mm_mul_ps(r_xxxx, l_wzyx);

            // Shuffle l_wzyx into [rz, rw, rx, ry]
            let l_zwxy = _mm_shuffle_ps::<0b10_11_00_01>(l_wzyx, l_wzyx);

            // Apply sign mask to term 2
            let lwrx_nlzrx_lyrx_nlxrx = _mm_mul_ps(lwrx_lzrx_lyrx_lxrx, CONTROL_WZYX);

            // Term 3: ly * [rz, rw, rx, ry]
            let lzry_lwry_lxry_lyry = _mm_mul_ps(r_yyyy, l_zwxy);

            // Shuffle l_zwxy into [ry, rx, rw, rz]
            let l_yxwz = _mm_shuffle_ps::<0b00_01_10_11>(l_zwxy, l_zwxy);

            // Apply sign mask to term 3
            let lzry_lwry_nlxry_nlyry = _mm_mul_ps(lzry_lwry_lxry_lyry, CONTROL_ZWXY);

            // Term 4: lz * [ry, rx, rw, rz]
            let lyrz_lxrz_lwrz_lzrz = _mm_mul_ps(r_zzzz, l_yxwz);

            // Accumulate terms 1 + 2
            let result0 = _mm_add_ps(lxrw_lyrw_lzrw_lwrw, lwrx_nlzrx_lyrx_nlxrx);

            // Apply sign mask to term 4
            let nlyrz_lxrz_lwrz_nlzrz = _mm_mul_ps(lyrz_lxrz_lwrz_lzrz, CONTROL_YXWZ);

            // Accumulate terms 3 + 4
            let result1 = _mm_add_ps(lzry_lwry_nlxry_nlyry, nlyrz_lxrz_lwrz_nlzrz);

            // Final sum
            Self(_mm_add_ps(result0, result1))
        }
    }

    // ── Interpolation ──────────────────────────────────────────────────────────

    /// Normalised linear interpolation — fast, slightly non-constant velocity.
    #[inline]
    pub fn nlerp(self, mut rhs: Self, t: f32) -> Self {
        // Ensure shortest path
        if self.dot(rhs) < 0.0 { rhs = -rhs; }
        unsafe {
            let tt = _mm_set1_ps(t);
            let lerped = _mm_add_ps(self.0, _mm_mul_ps(_mm_sub_ps(rhs.0, self.0), tt));
            Self(lerped).normalize()
        }
    }

    /// Spherical linear interpolation — constant angular velocity.
    pub fn slerp(self, mut rhs: Self, t: f32) -> Self {
        let mut cos_theta = self.dot(rhs);
        if cos_theta < 0.0 {
            rhs = -rhs;
            cos_theta = -cos_theta;
        }

        // When quaternions are nearly identical, fall back to nlerp to avoid
        // division by near-zero sin(angle).
        if cos_theta > 1.0 - EPSILON {
            return self.nlerp(rhs, t);
        }

        let angle  = math::acos_approx(cos_theta);
        let sin_a  = math::sqrt(1.0 - cos_theta * cos_theta);

        // Use SIMD to compute sin(x*angle), sin(y*angle), sin(z*angle)
        // simultaneously where x=(1-t), y=t, z=1 (for the divisor).
        unsafe {
            let angles = _mm_mul_ps(
                _mm_set1_ps(angle),
                _mm_set_ps(0.0, 1.0, t, 1.0 - t),
            );
            let sins   = m128_sin(angles);
            // lane0 = sin((1-t)*angle), lane1 = sin(t*angle), lane2 = sin(angle)
            let s0     = _mm_shuffle_ps::<0b00_00_00_00>(sins, sins);
            let s1     = _mm_shuffle_ps::<0b01_01_01_01>(sins, sins);
            let theta_sin = _mm_shuffle_ps::<0b10_10_10_10>(sins, sins);
            let _ = sin_a; // already captured via acos_approx path above

            let blended = _mm_add_ps(
                _mm_mul_ps(self.0, s0),
                _mm_mul_ps(rhs.0,  s1),
            );
            Self(_mm_div_ps(blended, theta_sin)).normalize()
        }
    }

    // ── Conversion ─────────────────────────────────────────────────────────────

    /// Convert to rotation Mat4. `self` must be normalised.
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

    /// Returns true if length ≈ 1.0.
    #[inline]
    pub fn is_normalized(self) -> bool { (self.length_sq() - 1.0).abs() <= 2e-4 }

    #[inline]
    pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite() &&
        self.z.is_finite() && self.w.is_finite()
    }
}

// ── Operators ─────────────────────────────────────────────────────────────────

impl Mul for Quat {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self { self.mul_quat(rhs) }
}
impl MulAssign for Quat {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) { *self = self.mul_quat(rhs); }
}
impl Neg for Quat {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self(unsafe { _mm_xor_ps(self.0, _mm_set1_ps(-0.0)) })
    }
}
impl Add for Quat {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self { Self(unsafe { _mm_add_ps(self.0, rhs.0) }) }
}
impl Sub for Quat {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self { Self(unsafe { _mm_sub_ps(self.0, rhs.0) }) }
}
impl Mul<f32> for Quat {
    type Output = Self;
    #[inline]
    fn mul(self, s: f32) -> Self { Self(unsafe { _mm_mul_ps(self.0, _mm_set1_ps(s)) }) }
}

impl PartialEq for Quat {
    #[inline]
    fn eq(&self, rhs: &Self) -> bool {
        unsafe {
            (_mm_movemask_ps(_mm_cmpeq_ps(self.0, rhs.0)) & 0b1111) == 0b1111
        }
    }
}

impl Default for Quat { fn default() -> Self { Self::IDENTITY } }

impl fmt::Debug for Quat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Quat")
            .field(&self.x).field(&self.y).field(&self.z).field(&self.w)
            .finish()
    }
}
impl fmt::Display for Quat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Quat({:.4}, {:.4}, {:.4}, {:.4})", self.x, self.y, self.z, self.w)
    }
  }
