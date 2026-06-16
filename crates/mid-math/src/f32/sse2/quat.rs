// crates/mid-math/src/f32/sse2/quat.rs
//! Quaternion backed by `__m128` on x86 / x86_64.
//!
//! Convention: (x, y, z, w) where w is the scalar part.
//! Storage:    lane0=x, lane1=y, lane2=z, lane3=w.
//! Euler convention: ZYX — yaw first, then pitch, then roll.

use core::fmt;
use core::ops::{Mul, MulAssign, Neg, Add, Sub};

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use crate::sse2::{dot4_into_m128, m128_from_f32x4, m128_sin, rsqrt_nr};
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

impl_vec4_deref!(Quat);

const CONTROL_WZYX: __m128 = m128_from_f32x4([ 1.0, -1.0,  1.0, -1.0]);
const CONTROL_ZWXY: __m128 = m128_from_f32x4([ 1.0,  1.0, -1.0, -1.0]);
const CONTROL_YXWZ: __m128 = m128_from_f32x4([-1.0,  1.0,  1.0, -1.0]);

impl Quat {
    // ── Constants ────────────────────────────────────────────────────────────

    pub const IDENTITY: Self = unsafe { UnionCast { f: [0.0, 0.0, 0.0, 1.0] }.v };
    pub const ZERO: Self     = unsafe { UnionCast { f: [0.0; 4] }.v };

    // ── Constructors ─────────────────────────────────────────────────────────

    #[inline(always)]
    pub fn new(x: f32, y: f32, z: f32, w: f32) -> Self {
        Self(unsafe { _mm_set_ps(w, z, y, x) })
    }

    #[inline(always)]
    pub fn from_xyzw(x: f32, y: f32, z: f32, w: f32) -> Self {
        Self::new(x, y, z, w)
    }

    #[inline]
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
            cz * cy * sx - sz * sy * cx,
            cz * sy * cx + sz * cy * sx,
            sz * cy * cx - cz * sy * sx,
            cz * cy * cx + sz * sy * sx,
        ).normalize()
    }

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

    /// Normalize to unit length. Returns `Quat::IDENTITY` for near-zero input.
    ///
    /// The IDENTITY fallback guard costs 4 SSE ops (cmpgt, and, andnot, or).
    /// For internal hot-paths where the input is known non-zero, use
    /// `normalize_fast()`.
    #[inline]
    pub fn normalize(self) -> Self {
        unsafe {
            let dot     = dot4_into_m128(self.0, self.0);
            let ok      = _mm_cmpgt_ps(dot, _mm_set1_ps(1e-12_f32));
            let inv_len = rsqrt_nr(dot);
            let n       = _mm_mul_ps(self.0, inv_len);
            let keep = _mm_and_ps(n, ok);
            let alt  = _mm_andnot_ps(ok, Self::IDENTITY.0);
            Self(_mm_or_ps(keep, alt))
        }
    }

    /// Fast normalize — **no** IDENTITY fallback guard.
    ///
    /// Precondition: `self` must not be near-zero length. Always satisfied
    /// after lerping two unit quaternions (nlerp/slerp inputs) and after
    /// the slerp division-by-sin step.
    ///
    /// Saves 4 SSE ops vs `normalize()`. Keep `pub(crate)` — external callers
    /// must use the safe `normalize()`.
    #[inline(always)]
    pub(crate) fn normalize_fast(self) -> Self {
        unsafe {
            let dot     = dot4_into_m128(self.0, self.0);
            let inv_len = rsqrt_nr(dot);
            Self(_mm_mul_ps(self.0, inv_len))
        }
    }

    #[inline]
    pub fn conjugate(self) -> Self {
        const SIGN: __m128 = m128_from_f32x4([-0.0, -0.0, -0.0, 0.0]);
        Self(unsafe { _mm_xor_ps(self.0, SIGN) })
    }

    #[inline]
    pub fn inverse(self) -> Self {
        let sq = self.length_sq();
        if sq < EPSILON { return Self::IDENTITY; }
        let rcp = 1.0 / sq;
        let conj = self.conjugate();
        Self(unsafe { _mm_mul_ps(conj.0, _mm_set1_ps(rcp)) })
    }

    #[inline]
    pub fn rotate(self, v: Vec3) -> Vec3 {
        let qv = Vec3::new(self.x, self.y, self.z);
        let t  = 2.0 * qv.cross(v);
        v + self.w * t + qv.cross(t)
    }

    #[inline]
    pub fn mul_quat(self, rhs: Self) -> Self {
        unsafe {
            let lhs = self.0;
            let rhs = rhs.0;

            let r_xxxx = _mm_shuffle_ps::<0b00_00_00_00>(lhs, lhs);
            let r_yyyy = _mm_shuffle_ps::<0b01_01_01_01>(lhs, lhs);
            let r_zzzz = _mm_shuffle_ps::<0b10_10_10_10>(lhs, lhs);
            let r_wwww = _mm_shuffle_ps::<0b11_11_11_11>(lhs, lhs);

            let lxrw_lyrw_lzrw_lwrw = _mm_mul_ps(r_wwww, rhs);
            let l_wzyx = _mm_shuffle_ps::<0b00_01_10_11>(rhs, rhs);
            let lwrx_lzrx_lyrx_lxrx = _mm_mul_ps(r_xxxx, l_wzyx);
            let l_zwxy = _mm_shuffle_ps::<0b10_11_00_01>(l_wzyx, l_wzyx);
            let lwrx_nlzrx_lyrx_nlxrx = _mm_mul_ps(lwrx_lzrx_lyrx_lxrx, CONTROL_WZYX);
            let lzry_lwry_lxry_lyry = _mm_mul_ps(r_yyyy, l_zwxy);
            let l_yxwz = _mm_shuffle_ps::<0b00_01_10_11>(l_zwxy, l_zwxy);
            let lzry_lwry_nlxry_nlyry = _mm_mul_ps(lzry_lwry_lxry_lyry, CONTROL_ZWXY);
            let lyrz_lxrz_lwrz_lzrz = _mm_mul_ps(r_zzzz, l_yxwz);
            let result0 = _mm_add_ps(lxrw_lyrw_lzrw_lwrw, lwrx_nlzrx_lyrx_nlxrx);
            let nlyrz_lxrz_lwrz_nlzrz = _mm_mul_ps(lyrz_lxrz_lwrz_lzrz, CONTROL_YXWZ);
            let result1 = _mm_add_ps(lzry_lwry_nlxry_nlyry, nlyrz_lxrz_lwrz_nlzrz);
            Self(_mm_add_ps(result0, result1))
        }
    }

    // ── Interpolation ──────────────────────────────────────────────────────────

    /// Normalised linear interpolation.
    ///
    /// `dot4_into_m128` (now 5 ops with pairwise-add broadcast) replaces the old
    /// 6-op path to avoid a scalar→SIMD round-trip that cost ~4–6 cycles.
    ///
    /// Uses `normalize_fast()` — blend of two unit quats is always non-zero.
    #[inline]
    pub fn nlerp(self, rhs: Self, t: f32) -> Self {
        unsafe {
            // dot4_into_m128: all 4 lanes = dot(self, rhs). No scalar extraction.
            let dot_v    = dot4_into_m128(self.0, rhs.0);
            // Extract sign bit of dot into every lane, XOR into rhs to flip if dot < 0.
            let sign_bit = _mm_and_ps(dot_v, _mm_set1_ps(-0.0f32));
            let rhs_adj  = _mm_xor_ps(rhs.0, sign_bit);
            let tt       = _mm_set1_ps(t);
            let lerped   = _mm_add_ps(self.0, _mm_mul_ps(_mm_sub_ps(rhs_adj, self.0), tt));
            // lerped is a linear blend of two unit quats — always non-zero.
            Self(lerped).normalize_fast()
        }
    }

    /// Spherical linear interpolation.
    ///
    /// Uses the vectorized sin-table approach: compute `sin(angle * [(1-t), t, 1, 0])`
    /// in one `m128_sin` call, then blend `self * sin(angle*(1-t)) + rhs * sin(angle*t)`
    /// and divide by `sin(angle)`.
    ///
    /// The division already places the result on the unit sphere geometrically —
    /// no `normalize_fast()` needed. Removing it saves: 1 `dot4_into_m128` +
    /// 1 `rsqrt_nr` (mul+mul+sub+mul) + 1 mul = ~8 instructions.
    ///
    /// Equivalent output quality to glam, which also skips the post-slerp normalize.
    pub fn slerp(self, mut rhs: Self, t: f32) -> Self {
        let mut cos_theta = self.dot(rhs);
        if cos_theta < 0.0 {
            rhs = -rhs;
            cos_theta = -cos_theta;
        }
        if cos_theta > 1.0 - EPSILON {
            return self.nlerp(rhs, t);
        }
        let angle = math::acos_approx(cos_theta);
        unsafe {
            // Compute sin(angle * (1-t)), sin(angle * t), sin(angle) in one shot.
            // _mm_set_ps(e3, e2, e1, e0): lane0=(1-t), lane1=t, lane2=1.0, lane3=0.0
            let angles = _mm_mul_ps(
                _mm_set1_ps(angle),
                _mm_set_ps(0.0, 1.0, t, 1.0 - t),
            );
            let sins      = m128_sin(angles);
            let s0        = _mm_shuffle_ps::<0b00_00_00_00>(sins, sins); // sin(angle*(1-t))
            let s1        = _mm_shuffle_ps::<0b01_01_01_01>(sins, sins); // sin(angle*t)
            let theta_sin = _mm_shuffle_ps::<0b10_10_10_10>(sins, sins); // sin(angle)

            let blended = _mm_add_ps(
                _mm_mul_ps(self.0, s0),
                _mm_mul_ps(rhs.0,  s1),
            );

            // Division by sin(angle) places the result on the unit sphere.
            // No normalize_fast() needed — matches glam's contract.
            Self(_mm_div_ps(blended, theta_sin))
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
