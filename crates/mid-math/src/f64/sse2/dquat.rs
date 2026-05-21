// crates/mid-math/src/f64/sse2/dquat.rs
//! DQuat backed by 2× `__m128d` on x86 / x86_64.
//!
//! Storage: `lo=[x,y]`, `hi=[z,w]` — same as DVec4. 32 bytes, align(32).
//! Deref to XYZW<f64> provides .x .y .z .w access.
//!
//! SIMD is used for: add/sub/mul-scalar, dot, length, normalize, nlerp,
//! conjugate, and basic arithmetic operators.
//!
//! The Hamilton product (`mul_quat`) stays scalar: it has 28 operations with
//! mixed signs that don't map cleanly to 2-lane SSE2 without significant
//! overhead. The scalar version is ~5 ns and dominates slerp anyway.

use core::fmt;
use core::ops::{Add, Mul, MulAssign, Neg, Sub};

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use crate::sse2::{dot4d, dot4d_into_m128d, m128d_abs, m128d_from_f64x2};
use crate::impl_dvec4_deref;
use crate::f64::dvec3::DVec3;
use crate::f64::dmat4::DMat4;
use crate::f64::dvec2::DEPSILON;

// ── Union ─────────────────────────────────────────────────────────────────────

#[repr(C)]
union UnionCast { f: [f64; 4], v: DQuat }

// Sign mask for conjugate: negate x,y,z; keep w.
// lo=[x,y]: both negative → [-0.0, -0.0]
// hi=[z,w]: negate z only → [-0.0,  0.0]
// Using m128d_from_f64x2: [a[0]=lane0, a[1]=lane1]
const CONJ_SIGN_LO: __m128d = m128d_from_f64x2([-0.0, -0.0]); // negate x,y
const CONJ_SIGN_HI: __m128d = m128d_from_f64x2([-0.0,  0.0]); // negate z, keep w

// ── Type ──────────────────────────────────────────────────────────────────────

/// Double-precision quaternion. 32 bytes, 32-byte aligned.
/// Convention: (x, y, z, w). Backed by 2× `__m128d`: `lo=[x,y]`, `hi=[z,w]`.
#[derive(Clone, Copy)]
#[repr(C, align(32))]
pub struct DQuat {
    pub(crate) lo: __m128d, // [x, y]
    pub(crate) hi: __m128d, // [z, w]
}

impl_dvec4_deref!(DQuat);

// ── Constants ─────────────────────────────────────────────────────────────────

impl DQuat {
    pub const IDENTITY: Self = unsafe { UnionCast { f: [0.0, 0.0, 0.0, 1.0] }.v };
    pub const ZERO:     Self = unsafe { UnionCast { f: [0.0; 4] }.v };

    // ── Constructors ─────────────────────────────────────────────────────────

    #[inline(always)]
    pub fn new(x: f64, y: f64, z: f64, w: f64) -> Self {
        unsafe { UnionCast { f: [x, y, z, w] }.v }
    }

    #[inline(always)]
    pub fn from_xyzw(x: f64, y: f64, z: f64, w: f64) -> Self { Self::new(x, y, z, w) }

    pub fn from_axis_angle(axis: DVec3, angle_rad: f64) -> Self {
        let (s, c) = (angle_rad * 0.5).sin_cos();
        let n = axis.normalize();
        Self::new(n.x * s, n.y * s, n.z * s, c)
    }

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

    pub fn to_euler(self) -> (f64, f64, f64) {
        let sinp  = 2.0 * (self.w * self.y - self.z * self.x);
        let pitch = if sinp.abs() >= 1.0 {
            sinp.signum() * core::f64::consts::FRAC_PI_2
        } else { sinp.asin() };
        let roll = (2.0*(self.w*self.x + self.y*self.z))
            .atan2(1.0 - 2.0*(self.x*self.x + self.y*self.y));
        let yaw  = (2.0*(self.w*self.z + self.x*self.y))
            .atan2(1.0 - 2.0*(self.y*self.y + self.z*self.z));
        (roll, pitch, yaw)
    }

    // ── Core ops ──────────────────────────────────────────────────────────────

    #[inline(always)]
    pub fn dot(self, rhs: Self) -> f64 {
        unsafe { dot4d(self.lo, self.hi, rhs.lo, rhs.hi) }
    }

    #[inline(always)] pub fn length_sq(self) -> f64 { self.dot(self) }
    #[inline] pub fn length(self)    -> f64 { self.length_sq().sqrt() }

    #[inline]
    pub fn normalize(self) -> Self {
        unsafe {
            let len_v = dot4d_into_m128d(self.lo, self.hi, self.lo, self.hi);
            let sqrt  = _mm_sqrt_pd(len_v);
            let lo_n  = _mm_div_pd(self.lo, sqrt);
            let hi_n  = _mm_div_pd(self.hi, sqrt);
            let ok    = _mm_cmpgt_pd(sqrt, _mm_set1_pd(DEPSILON));
            // Keep normalized value where len > DEPSILON, else fall back to IDENTITY
            let id_lo = Self::IDENTITY.lo;
            let id_hi = Self::IDENTITY.hi;
            Self {
                lo: _mm_or_pd(_mm_and_pd(ok, lo_n), _mm_andnot_pd(ok, id_lo)),
                hi: _mm_or_pd(_mm_and_pd(ok, hi_n), _mm_andnot_pd(ok, id_hi)),
            }
        }
    }

    /// Conjugate: negate xyz, keep w. Two XOR ops (one per __m128d).
    #[inline]
    pub fn conjugate(self) -> Self {
        Self {
            lo: unsafe { _mm_xor_pd(self.lo, CONJ_SIGN_LO) }, // negate x,y
            hi: unsafe { _mm_xor_pd(self.hi, CONJ_SIGN_HI) }, // negate z, keep w
        }
    }

    #[inline]
    pub fn inverse(self) -> Self {
        let sq = self.length_sq();
        if sq < DEPSILON { return Self::IDENTITY; }
        let rcp = 1.0 / sq;
        let c = self.conjugate();
        unsafe {
            let r = _mm_set1_pd(rcp);
            Self { lo: _mm_mul_pd(c.lo, r), hi: _mm_mul_pd(c.hi, r) }
        }
    }

    #[inline]
    pub fn rotate(self, v: DVec3) -> DVec3 {
        let qv = DVec3::new(self.x, self.y, self.z);
        let t  = qv.cross(v) * 2.0;
        v + t * self.w + qv.cross(t)
    }

    /// Hamilton product — scalar (28 ops with mixed signs; SIMD benefit minimal).
    #[inline]
    pub fn mul_quat(self, rhs: Self) -> Self {
        let (lx, ly, lz, lw) = (self.x, self.y, self.z, self.w);
        let (rx, ry, rz, rw) = (rhs.x,  rhs.y,  rhs.z,  rhs.w);
        Self::new(
            lw*rx + lx*rw + ly*rz - lz*ry,
            lw*ry - lx*rz + ly*rw + lz*rx,
            lw*rz + lx*ry - ly*rx + lz*rw,
            lw*rw - lx*rx - ly*ry - lz*rz,
        )
    }

    // ── Interpolation ──────────────────────────────────────────────────────────

    #[inline]
    pub fn nlerp(self, rhs: Self, t: f64) -> Self {
        let dot = self.dot(rhs);
        let sign_bit = dot.to_bits() & 0x8000_0000_0000_0000u64;
        let flip = |x: f64| f64::from_bits(x.to_bits() ^ sign_bit);
        unsafe {
            let rhs_adj = UnionCast { f: [flip(rhs.x), flip(rhs.y), flip(rhs.z), flip(rhs.w)] }.v;
            let tt = _mm_set1_pd(t);
            let lo_d = _mm_sub_pd(rhs_adj.lo, self.lo);
            let hi_d = _mm_sub_pd(rhs_adj.hi, self.hi);
            let lerped = Self {
                lo: _mm_add_pd(self.lo, _mm_mul_pd(lo_d, tt)),
                hi: _mm_add_pd(self.hi, _mm_mul_pd(hi_d, tt)),
            };
            lerped.normalize()
        }
    }

    pub fn slerp(self, mut rhs: Self, t: f64) -> Self {
        let mut cos_theta = self.dot(rhs);
        if cos_theta < 0.0 { rhs = -rhs; cos_theta = -cos_theta; }
        if cos_theta > 1.0 - 1e-6 { return self.nlerp(rhs, t); }
        let angle     = cos_theta.acos();
        let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();
        let (sin_t, cos_t) = (t * angle).sin_cos();
        let s1 = sin_t / sin_theta;
        let s0 = cos_t - cos_theta * s1;
        unsafe {
            let v0 = _mm_set1_pd(s0);
            let v1 = _mm_set1_pd(s1);
            Self {
                lo: _mm_add_pd(_mm_mul_pd(self.lo, v0), _mm_mul_pd(rhs.lo, v1)),
                hi: _mm_add_pd(_mm_mul_pd(self.hi, v0), _mm_mul_pd(rhs.hi, v1)),
            }
        }
    }

    // ── Conversion ─────────────────────────────────────────────────────────────

    pub fn to_mat4(self) -> DMat4 {
        let q = self.normalize();
        let (x, y, z, w) = (q.x, q.y, q.z, q.w);
        let (x2, y2, z2) = (x+x, y+y, z+z);
        let (xx, yy, zz) = (x*x2, y*y2, z*z2);
        let (xy, xz, yz) = (x*y2, x*z2, y*z2);
        let (wx, wy, wz) = (w*x2, w*y2, w*z2);
        DMat4::from_cols(
            [1.0-yy-zz, xy+wz,     xz-wy,     0.0],
            [xy-wz,     1.0-xx-zz, yz+wx,     0.0],
            [xz+wy,     yz-wx,     1.0-xx-yy, 0.0],
            [0.0,       0.0,       0.0,       1.0],
        )
    }

    #[inline]
    pub fn as_quat(self) -> crate::Quat {
        crate::Quat::new(self.x as f32, self.y as f32, self.z as f32, self.w as f32)
    }

    #[inline] pub fn is_normalized(self) -> bool { (self.length_sq() - 1.0).abs() <= 4e-10 }
    #[inline] pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite()
            && self.z.is_finite() && self.w.is_finite()
    }
}

// ── Operators ─────────────────────────────────────────────────────────────────

impl Mul for DQuat {
    type Output = Self;
    #[inline] fn mul(self, r: Self) -> Self { self.mul_quat(r) }
}
impl MulAssign for DQuat {
    #[inline] fn mul_assign(&mut self, r: Self) { *self = self.mul_quat(r); }
}
impl Neg for DQuat {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        unsafe {
            let s = _mm_set1_pd(-0.0);
            Self { lo: _mm_xor_pd(self.lo, s), hi: _mm_xor_pd(self.hi, s) }
        }
    }
}
impl Add for DQuat {
    type Output = Self;
    #[inline]
    fn add(self, r: Self) -> Self {
        Self { lo: unsafe { _mm_add_pd(self.lo, r.lo) }, hi: unsafe { _mm_add_pd(self.hi, r.hi) } }
    }
}
impl Sub for DQuat {
    type Output = Self;
    #[inline]
    fn sub(self, r: Self) -> Self {
        Self { lo: unsafe { _mm_sub_pd(self.lo, r.lo) }, hi: unsafe { _mm_sub_pd(self.hi, r.hi) } }
    }
}
impl Mul<f64> for DQuat {
    type Output = Self;
    #[inline]
    fn mul(self, s: f64) -> Self {
        unsafe {
            let sv = _mm_set1_pd(s);
            Self { lo: _mm_mul_pd(self.lo, sv), hi: _mm_mul_pd(self.hi, sv) }
        }
    }
}

impl PartialEq for DQuat {
    #[inline]
    fn eq(&self, rhs: &Self) -> bool {
        unsafe {
            let lo_ok = (_mm_movemask_pd(_mm_cmpeq_pd(self.lo, rhs.lo)) & 0b11) == 0b11;
            let hi_ok = (_mm_movemask_pd(_mm_cmpeq_pd(self.hi, rhs.hi)) & 0b11) == 0b11;
            lo_ok && hi_ok
        }
    }
}
impl Default for DQuat { fn default() -> Self { Self::IDENTITY } }

impl fmt::Debug for DQuat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("DQuat")
            .field(&self.x).field(&self.y).field(&self.z).field(&self.w).finish()
    }
}
impl fmt::Display for DQuat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DQuat({:.6}, {:.6}, {:.6}, {:.6})", self.x, self.y, self.z, self.w)
    }
}
