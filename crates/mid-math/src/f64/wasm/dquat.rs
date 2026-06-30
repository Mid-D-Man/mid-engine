// crates/mid-math/src/f64/wasm/dquat.rs
//! DQuat backed by 2× `v128` (f64x2) on wasm32/wasm64 with simd128.
//!
//! Storage: `lo=[x,y]`, `hi=[z,w]` — identical layout to SSE2/NEON DQuat.
//!
//! SIMD used for: arithmetic operators, dot, length, normalize, conjugate,
//!                nlerp (blend), slerp (scalar blend with SIMD lerp).
//!
//! Scalar for: Hamilton product — 28 mixed-sign ops on 2-lane f64x2 gives
//!             no win. The scalar path is ~5 ns and isn't a hot loop.
//!
//! WASM vs SSE2 notes:
//!   f64x2_neg — direct instruction; no XOR with -0.0 sign-bit trick needed
//!   f64x2_abs — direct; no ANDNOT mask needed
//!   v128_andnot(a, b) = a & ~b  ← WASM (reversed from SSE2 _mm_andnot_pd)
//!   u64x2_bitmask — extracts MSB of each 64-bit lane (2-bit result)
//!   No FMA in SIMD128 baseline — lerp is 2 muls + 1 add per component
//!
//! ## A note on `unsafe`
//! `core::arch::wasm32`/`wasm64` SIMD intrinsics (`f64x2_*`, `u64x2_*`, `v128_*`)
//! are SAFE functions — unlike x86/ARM intrinsics, WASM SIMD instructions can't
//! cause memory unsafety; a module either validates with simd128 support at
//! load time or fails to load at all. Only two things in this file genuinely
//! need `unsafe`: reading a `union` field (always unsafe in Rust, regardless
//! of payload), and calls to `dot4d`/`dot4d_into_v128` from `crate::wasm`,
//! which ARE declared as `unsafe fn`.

#[cfg(target_arch = "wasm32")]
use core::arch::wasm32::*;
#[cfg(target_arch = "wasm64")]
use core::arch::wasm64::*;

use core::fmt;
use core::ops::{Add, Mul, MulAssign, Neg, Sub};

use crate::f64::dvec3::DVec3;
use crate::f64::dmat4::DMat4;
use crate::impl_dvec4_deref;
use crate::wasm::{dot4d, dot4d_into_v128, v128_from_f64x2};
use crate::DEPSILON;

// ── Conjugate sign masks ──────────────────────────────────────────────────────
//
// lo = [x, y]: negate both → [-0.0, -0.0]
// hi = [z, w]: negate z only, keep w → [-0.0, 0.0]
//
// XOR with sign-bit pattern clears/sets the IEEE754 sign bit:
//   f64 -0.0 = 0x8000_0000_0000_0000  (sign bit set, all else zero)
//   XOR with -0.0 = toggle sign bit = negate

const CONJ_SIGN_LO: v128 = v128_from_f64x2([-0.0, -0.0]); // negate x, y
const CONJ_SIGN_HI: v128 = v128_from_f64x2([-0.0,  0.0]); // negate z, keep w

// ── Union for compile-time constant initialisation ────────────────────────────

#[repr(C)]
union UnionCast { f: [f64; 4], v: DQuat }

// ── Type ──────────────────────────────────────────────────────────────────────

/// Double-precision quaternion. 32 bytes, 32-byte aligned.
/// Convention: (x, y, z, w). Backed by 2× `v128` (f64x2): `lo=[x,y]`, `hi=[z,w]`.
///
/// **C interop:** use `CDQuat` at the FFI boundary.
#[derive(Clone, Copy)]
#[repr(C, align(32))]
pub struct DQuat {
    pub(crate) lo: v128, // [x, y]
    pub(crate) hi: v128, // [z, w]
}

impl_dvec4_deref!(DQuat);

// ── Constants ─────────────────────────────────────────────────────────────────
// Union field read — genuinely unsafe, kept.

impl DQuat {
    pub const IDENTITY: Self = unsafe { UnionCast { f: [0.0, 0.0, 0.0, 1.0] }.v };
    pub const ZERO:     Self = unsafe { UnionCast { f: [0.0; 4] }.v };

    // ── Constructors ─────────────────────────────────────────────────────────

    #[inline(always)]
    pub fn new(x: f64, y: f64, z: f64, w: f64) -> Self {
        // Union field read — genuinely unsafe, kept.
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
        let roll = (2.0 * (self.w * self.x + self.y * self.z))
            .atan2(1.0 - 2.0 * (self.x * self.x + self.y * self.y));
        let yaw  = (2.0 * (self.w * self.z + self.x * self.y))
            .atan2(1.0 - 2.0 * (self.y * self.y + self.z * self.z));
        (roll, pitch, yaw)
    }

    // ── Core ops ──────────────────────────────────────────────────────────────

    #[inline(always)]
    pub fn dot(self, rhs: Self) -> f64 {
        // dot4d is an unsafe fn (crate::wasm) — kept.
        unsafe { dot4d(self.lo, self.hi, rhs.lo, rhs.hi) }
    }

    #[inline(always)] pub fn length_sq(self) -> f64 { self.dot(self) }
    #[inline]         pub fn length(self)    -> f64 { self.length_sq().sqrt() }

    /// Normalize. Falls back to IDENTITY for near-zero-length quaternions.
    ///
    /// `dot4d_into_v128` broadcasts the dot to both f64 lanes, giving us the
    /// same divisor for lo and hi in one sqrt. Guard mask zeroes degenerate
    /// lanes and restores IDENTITY w=1 via:
    ///   keep  = v128_and(norm,     ok)   — normalized lanes where len > eps
    ///   ident = v128_andnot(IDENTITY, ok) — identity in degenerate lanes
    ///
    /// Note: v128_andnot(a, b) = a & ~b  (WASM convention — reversed from SSE2).
    #[inline]
    pub fn normalize(self) -> Self {
        // dot4d_into_v128 is an unsafe fn (crate::wasm) — kept; the rest of
        // this block is safe wasm32 intrinsics riding along inside it.
        unsafe {
            let len_v = f64x2_sqrt(dot4d_into_v128(self.lo, self.hi, self.lo, self.hi));
            let lo_n  = f64x2_div(self.lo, len_v);
            let hi_n  = f64x2_div(self.hi, len_v);
            let ok    = f64x2_gt(len_v, f64x2_splat(DEPSILON));
            // ok lanes: keep normalized; degenerate lanes: use identity (0,0,0,1)
            let id_lo = Self::IDENTITY.lo;
            let id_hi = Self::IDENTITY.hi;
            Self {
                lo: v128_or(v128_and(lo_n, ok), v128_andnot(id_lo, ok)),
                hi: v128_or(v128_and(hi_n, ok), v128_andnot(id_hi, ok)),
            }
        }
    }

    /// Conjugate: negate xyz, keep w. Two v128_xor ops.
    ///
    /// lo=[x,y] XOR [-0.0, -0.0] → [-x, -y]
    /// hi=[z,w] XOR [-0.0,  0.0] → [-z,  w]
    #[inline]
    pub fn conjugate(self) -> Self {
        Self {
            lo: v128_xor(self.lo, CONJ_SIGN_LO),
            hi: v128_xor(self.hi, CONJ_SIGN_HI),
        }
    }

    #[inline]
    pub fn inverse(self) -> Self {
        let sq = self.length_sq();
        if sq < DEPSILON { return Self::IDENTITY; }
        let rcp = 1.0 / sq;
        let c = self.conjugate();
        let r = f64x2_splat(rcp);
        Self { lo: f64x2_mul(c.lo, r), hi: f64x2_mul(c.hi, r) }
    }

    #[inline]
    pub fn rotate(self, v: DVec3) -> DVec3 {
        let qv = DVec3::new(self.x, self.y, self.z);
        let t  = qv.cross(v) * 2.0;
        v + t * self.w + qv.cross(t)
    }

    /// Hamilton product — scalar path.
    ///
    /// 28 mixed-sign multiply-accumulators map poorly to 2-lane f64x2 SIMD:
    /// the result components mix both lanes in non-trivial ways, requiring
    /// shuffles that eat back the SIMD benefit. Scalar ~5 ns is acceptable
    /// since mul_quat is not a per-entity hot path (slerp runs it ≤1×/frame/bone).
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

    /// Normalised linear interpolation — branchless shortest-path flip.
    ///
    /// Extracts sign bit of dot via integer XOR:
    ///   0x8000_0000_0000_0000 if dot < 0 (negative), else 0.
    /// XOR this bit into every rhs component = branchless negate.
    #[inline]
    pub fn nlerp(self, rhs: Self, t: f64) -> Self {
        let dot = self.dot(rhs);
        let sign_bit = dot.to_bits() & 0x8000_0000_0000_0000u64;
        let flip = |x: f64| f64::from_bits(x.to_bits() ^ sign_bit);

        // Only the union field read genuinely needs `unsafe` — everything
        // after it is safe wasm32 intrinsics / a safe `.normalize()` call.
        let rhs_adj = unsafe {
            UnionCast { f: [flip(rhs.x), flip(rhs.y), flip(rhs.z), flip(rhs.w)] }.v
        };

        let tt    = f64x2_splat(t);
        let lo_d  = f64x2_sub(rhs_adj.lo, self.lo);
        let hi_d  = f64x2_sub(rhs_adj.hi, self.hi);
        let lerped = Self {
            lo: f64x2_add(self.lo, f64x2_mul(lo_d, tt)),
            hi: f64x2_add(self.hi, f64x2_mul(hi_d, tt)),
        };
        lerped.normalize()
    }

    pub fn slerp(self, mut rhs: Self, t: f64) -> Self {
        let mut cos_theta = self.dot(rhs);
        if cos_theta < 0.0 { rhs = -rhs; cos_theta = -cos_theta; }
        if cos_theta > 1.0 - 1e-6 { return self.nlerp(rhs, t); }
        let angle     = cos_theta.acos();
        let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();
        let (sin_t, cos_t) = (t * angle).sin_cos();
        let s1 = sin_t / sin_theta;
        let s0 = cos_t - cos_theta * s1; // algebraic — avoids 3rd transcendental
        let v0 = f64x2_splat(s0);
        let v1 = f64x2_splat(s1);
        Self {
            lo: f64x2_add(f64x2_mul(self.lo, v0), f64x2_mul(rhs.lo, v1)),
            hi: f64x2_add(f64x2_mul(self.hi, v0), f64x2_mul(rhs.hi, v1)),
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

/// Direct f64x2_neg on both registers — no XOR trick unlike SSE2.
impl Neg for DQuat {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self {
            lo: f64x2_neg(self.lo),
            hi: f64x2_neg(self.hi),
        }
    }
}

impl Add for DQuat {
    type Output = Self;
    #[inline]
    fn add(self, r: Self) -> Self {
        Self {
            lo: f64x2_add(self.lo, r.lo),
            hi: f64x2_add(self.hi, r.hi),
        }
    }
}

impl Sub for DQuat {
    type Output = Self;
    #[inline]
    fn sub(self, r: Self) -> Self {
        Self {
            lo: f64x2_sub(self.lo, r.lo),
            hi: f64x2_sub(self.hi, r.hi),
        }
    }
}

impl Mul<f64> for DQuat {
    type Output = Self;
    #[inline]
    fn mul(self, s: f64) -> Self {
        let sv = f64x2_splat(s);
        Self { lo: f64x2_mul(self.lo, sv), hi: f64x2_mul(self.hi, sv) }
    }
}

impl PartialEq for DQuat {
    #[inline]
    fn eq(&self, rhs: &Self) -> bool {
        let lo_ok = (u64x2_bitmask(f64x2_eq(self.lo, rhs.lo)) & 0b11) == 0b11;
        let hi_ok = (u64x2_bitmask(f64x2_eq(self.hi, rhs.hi)) & 0b11) == 0b11;
        lo_ok && hi_ok
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
