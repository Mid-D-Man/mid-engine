// crates/mid-math/src/f32/sse2/quat.rs
//! Quaternion backed by `__m128` on x86 / x86_64.
//!
//! Build 20: from_axis_angle rewritten to stay in XMM registers.
//!           to_mat4 delegates to quat_to_axes_sse2 — no Deref spill.

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
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct Quat(pub(crate) __m128);

impl_vec4_deref!(Quat);

const CONTROL_WZYX: __m128 = m128_from_f32x4([ 1.0, -1.0,  1.0, -1.0]);
const CONTROL_ZWXY: __m128 = m128_from_f32x4([ 1.0,  1.0, -1.0, -1.0]);
const CONTROL_YXWZ: __m128 = m128_from_f32x4([-1.0,  1.0,  1.0, -1.0]);

/// All-ones lanes 0-2, lane 3 = 0.  Used to zero the w/padding lane.
const XYZ_MASK: __m128 = m128_from_f32x4([
    f32::from_bits(0xFFFF_FFFF),
    f32::from_bits(0xFFFF_FFFF),
    f32::from_bits(0xFFFF_FFFF),
    0.0_f32,
]);

impl Quat {
    pub const IDENTITY: Self = unsafe { UnionCast { f: [0.0, 0.0, 0.0, 1.0] }.v };
    pub const ZERO: Self     = unsafe { UnionCast { f: [0.0; 4] }.v };

    // ── Constructors ──────────────────────────────────────────────────────────

    #[inline(always)]
    pub fn new(x: f32, y: f32, z: f32, w: f32) -> Self {
        Self(unsafe { _mm_set_ps(w, z, y, x) })
    }

    #[inline(always)]
    pub fn from_xyzw(x: f32, y: f32, z: f32, w: f32) -> Self { Self::new(x, y, z, w) }

    /// Create from axis-angle, keeping all values in XMM registers.
    ///
    /// Previous path: normalize() → Deref reads n.x/n.y/n.z → stack spill →
    ///   _mm_set_ps(c,z,y,x) → 4 scalar moves ≈ 6.25 ns.
    ///
    /// New path: normalize() → __m128 in register → _mm_mul_ps(n, splat(sin))
    ///   → AND/OR to insert cos into lane 3 ≈ 4.8 ns target.
    ///
    /// Lane-3 insert (SSE2, no blendps needed):
    ///   AND xyz_s with XYZ_MASK → zeroes lane 3
    ///   ANDNOT XYZ_MASK with splat(c) → [0,0,0,c]
    ///   OR → [ax*s, ay*s, az*s, c]
    #[inline]
    pub fn from_axis_angle(axis: Vec3, angle_rad: f32) -> Self {
        let (s, c) = math::sin_cos(angle_rad * 0.5);
        unsafe {
            let n        = axis.normalize().0;
            let sv       = _mm_set1_ps(s);
            let xyz_s    = _mm_mul_ps(n, sv);
            let xyz_only = _mm_and_ps(xyz_s, XYZ_MASK);
            let w_only   = _mm_andnot_ps(XYZ_MASK, _mm_set1_ps(c)); // [0,0,0,c]
            Self(_mm_or_ps(xyz_only, w_only))
        }
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

    #[inline]
    pub fn normalize(self) -> Self {
        unsafe {
            let dot     = dot4_into_m128(self.0, self.0);
            let ok      = _mm_cmpgt_ps(dot, _mm_set1_ps(1e-12_f32));
            let inv_len = rsqrt_nr(dot);
            let n       = _mm_mul_ps(self.0, inv_len);
            let keep    = _mm_and_ps(n, ok);
            let alt     = _mm_andnot_ps(ok, Self::IDENTITY.0);
            Self(_mm_or_ps(keep, alt))
        }
    }

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
        let rcp  = 1.0 / sq;
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

            let lxrw_lyrw_lzrw_lwrw     = _mm_mul_ps(r_wwww, rhs);
            let l_wzyx                   = _mm_shuffle_ps::<0b00_01_10_11>(rhs, rhs);
            let lwrx_lzrx_lyrx_lxrx     = _mm_mul_ps(r_xxxx, l_wzyx);
            let l_zwxy                   = _mm_shuffle_ps::<0b10_11_00_01>(l_wzyx, l_wzyx);
            let lwrx_nlzrx_lyrx_nlxrx   = _mm_mul_ps(lwrx_lzrx_lyrx_lxrx, CONTROL_WZYX);
            let lzry_lwry_lxry_lyry      = _mm_mul_ps(r_yyyy, l_zwxy);
            let l_yxwz                   = _mm_shuffle_ps::<0b00_01_10_11>(l_zwxy, l_zwxy);
            let lzry_lwry_nlxry_nlyry    = _mm_mul_ps(lzry_lwry_lxry_lyry, CONTROL_ZWXY);
            let lyrz_lxrz_lwrz_lzrz     = _mm_mul_ps(r_zzzz, l_yxwz);
            let result0                  = _mm_add_ps(lxrw_lyrw_lzrw_lwrw, lwrx_nlzrx_lyrx_nlxrx);
            let nlyrz_lxrz_lwrz_nlzrz   = _mm_mul_ps(lyrz_lxrz_lwrz_lzrz, CONTROL_YXWZ);
            let result1                  = _mm_add_ps(lzry_lwry_nlxry_nlyry, nlyrz_lxrz_lwrz_nlzrz);
            Self(_mm_add_ps(result0, result1))
        }
    }

    // ── Interpolation ─────────────────────────────────────────────────────────

    #[inline]
    pub fn nlerp(self, rhs: Self, t: f32) -> Self {
        unsafe {
            let dot_v    = dot4_into_m128(self.0, rhs.0);
            let sign_bit = _mm_and_ps(dot_v, _mm_set1_ps(-0.0f32));
            let rhs_adj  = _mm_xor_ps(rhs.0, sign_bit);
            let tt       = _mm_set1_ps(t);
            let lerped   = _mm_add_ps(self.0, _mm_mul_ps(_mm_sub_ps(rhs_adj, self.0), tt));
            Self(lerped).normalize_fast()
        }
    }

    pub fn slerp(self, mut rhs: Self, t: f32) -> Self {
        let mut cos_theta = self.dot(rhs);
        if cos_theta < 0.0 { rhs = -rhs; cos_theta = -cos_theta; }
        if cos_theta > 1.0 - EPSILON { return self.nlerp(rhs, t); }
        let angle = math::acos_approx(cos_theta);
        unsafe {
            let angles    = _mm_mul_ps(_mm_set1_ps(angle), _mm_set_ps(0.0, 1.0, t, 1.0 - t));
            let sins      = m128_sin(angles);
            let s0        = _mm_shuffle_ps::<0b00_00_00_00>(sins, sins);
            let s1        = _mm_shuffle_ps::<0b01_01_01_01>(sins, sins);
            let theta_sin = _mm_shuffle_ps::<0b10_10_10_10>(sins, sins);
            let blended   = _mm_add_ps(_mm_mul_ps(self.0, s0), _mm_mul_ps(rhs.0, s1));
            Self(_mm_div_ps(blended, theta_sin))
        }
    }

    // ── Conversion ────────────────────────────────────────────────────────────

    /// Convert to Mat4 via `quat_to_axes_sse2` — zero Deref reads, zero spills.
    pub fn to_mat4(self) -> Mat4 {
        use crate::f32::sse2::mat4::quat_to_axes_sse2;
        use crate::f32::sse2::vec4::Vec4;
        unsafe {
            let q = self.normalize().0;
            let (x_axis, y_axis, z_axis) = quat_to_axes_sse2(q);
            Mat4 {
                x_axis: Vec4(x_axis),
                y_axis: Vec4(y_axis),
                z_axis: Vec4(z_axis),
                w_axis: Vec4::W,
            }
        }
    }

    #[inline] pub fn is_normalized(self) -> bool { (self.length_sq() - 1.0).abs() <= 2e-4 }

    #[inline]
    pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite() &&
        self.z.is_finite() && self.w.is_finite()
    }
}

// ── Operators ─────────────────────────────────────────────────────────────────

impl Mul for Quat {
    type Output = Self;
    #[inline] fn mul(self, rhs: Self) -> Self { self.mul_quat(rhs) }
}
impl MulAssign for Quat {
    #[inline] fn mul_assign(&mut self, rhs: Self) { *self = self.mul_quat(rhs); }
}
impl Neg for Quat {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self { Self(unsafe { _mm_xor_ps(self.0, _mm_set1_ps(-0.0)) }) }
}
impl Add for Quat {
    type Output = Self;
    #[inline] fn add(self, rhs: Self) -> Self { Self(unsafe { _mm_add_ps(self.0, rhs.0) }) }
}
impl Sub for Quat {
    type Output = Self;
    #[inline] fn sub(self, rhs: Self) -> Self { Self(unsafe { _mm_sub_ps(self.0, rhs.0) }) }
}
impl Mul<f32> for Quat {
    type Output = Self;
    #[inline] fn mul(self, s: f32) -> Self { Self(unsafe { _mm_mul_ps(self.0, _mm_set1_ps(s)) }) }
}

impl PartialEq for Quat {
    #[inline]
    fn eq(&self, rhs: &Self) -> bool {
        unsafe { (_mm_movemask_ps(_mm_cmpeq_ps(self.0, rhs.0)) & 0b1111) == 0b1111 }
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
