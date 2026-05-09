// crates/mid-math/src/wide/float/sse2/quatx4.rs
//! 4 quaternions packed SoA — SSE2, x86 / x86_64.
//!
//! Layout: x,y,z,w each a __m128 holding that component across all 4 quats.
//! Hamilton product: 4 independent multiplications in the same instruction
//! count as 1 scalar multiplication — straight SIMD parallelism.
//!
//! Engine uses: animation bone blending (4 joints/cycle), rigid body
//! orientation integration, network quaternion interpolation.

use core::fmt;
use core::ops::Mul;

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use crate::f32::sse2::quat::Quat;
use crate::sse2::m128_from_f32x4;
use crate::EPSILON;
use super::f32x4::f32x4;
use super::vec3x4::Vec3x4;

/// 4 quaternions in SoA layout. 64 bytes, 16-byte aligned. Backed by 4 × __m128.
///
/// Each field holds one component for all 4 quaternions:
///   `x: __m128 = [x₀, x₁, x₂, x₃]`
///   `y: __m128 = [y₀, y₁, y₂, y₃]`
///   `z: __m128 = [z₀, z₁, z₂, z₃]`
///   `w: __m128 = [w₀, w₁, w₂, w₃]`
///
/// Never mix with scalar code inside tight loops — stay in wide registers.
#[derive(Clone, Copy)]
#[repr(C, align(16))]
pub struct QuatX4 {
    pub x: __m128,
    pub y: __m128,
    pub z: __m128,
    pub w: __m128,
}

impl QuatX4 {
    /// All 4 lanes = identity quaternion (0, 0, 0, 1).
    pub const IDENTITY: Self = Self {
        x: m128_from_f32x4([0.0; 4]),
        y: m128_from_f32x4([0.0; 4]),
        z: m128_from_f32x4([0.0; 4]),
        w: m128_from_f32x4([1.0; 4]),
    };

    // ── Constructors ──────────────────────────────────────────────────────────

    /// Build from 4 individual quaternions.
    ///
    /// Cost: 4 × `_mm_set_ps` (no shuffles needed — component scatter is free here).
    #[inline]
    pub fn from_quats(a: Quat, b: Quat, c: Quat, d: Quat) -> Self {
        // _mm_set_ps(lane3, lane2, lane1, lane0)
        unsafe {
            Self {
                x: _mm_set_ps(d.x, c.x, b.x, a.x),
                y: _mm_set_ps(d.y, c.y, b.y, a.y),
                z: _mm_set_ps(d.z, c.z, b.z, a.z),
                w: _mm_set_ps(d.w, c.w, b.w, a.w),
            }
        }
    }

    /// Broadcast one quaternion to all 4 lanes.
    #[inline(always)]
    pub fn splat(q: Quat) -> Self {
        unsafe {
            Self {
                x: _mm_set1_ps(q.x),
                y: _mm_set1_ps(q.y),
                z: _mm_set1_ps(q.z),
                w: _mm_set1_ps(q.w),
            }
        }
    }

    /// Build from a slice of 4 quaternions.
    #[inline(always)]
    pub fn from_slice(s: &[Quat; 4]) -> Self {
        Self::from_quats(s[0], s[1], s[2], s[3])
    }

    /// Extract all 4 quaternions as an array (SoA → AoS).
    #[inline]
    pub fn to_array(self) -> [Quat; 4] {
        unsafe {
            let mut xs = [0.0f32; 4];
            let mut ys = [0.0f32; 4];
            let mut zs = [0.0f32; 4];
            let mut ws = [0.0f32; 4];
            _mm_storeu_ps(xs.as_mut_ptr(), self.x);
            _mm_storeu_ps(ys.as_mut_ptr(), self.y);
            _mm_storeu_ps(zs.as_mut_ptr(), self.z);
            _mm_storeu_ps(ws.as_mut_ptr(), self.w);
            core::array::from_fn(|i| Quat::new(xs[i], ys[i], zs[i], ws[i]))
        }
    }

    /// Extract a single quaternion by lane index. Panics if `lane >= 4`.
    #[inline]
    pub fn get(self, lane: usize) -> Quat {
        assert!(lane < 4, "QuatX4::get — lane {lane} out of bounds (max 3)");
        self.to_array()[lane]
    }

    // ── Core ops ──────────────────────────────────────────────────────────────

    /// 4 independent dot products simultaneously.
    ///
    /// Returns `f32x4` where lane i = dot(self[i], rhs[i]).
    #[inline(always)]
    pub fn dot(self, rhs: Self) -> f32x4 {
        unsafe {
            let xx = _mm_mul_ps(self.x, rhs.x);
            let yy = _mm_mul_ps(self.y, rhs.y);
            let zz = _mm_mul_ps(self.z, rhs.z);
            let ww = _mm_mul_ps(self.w, rhs.w);
            f32x4(_mm_add_ps(_mm_add_ps(xx, yy), _mm_add_ps(zz, ww)))
        }
    }

    /// Squared length of each quaternion.
    #[inline(always)]
    pub fn length_sq(self) -> f32x4 { self.dot(self) }

    /// Normalize all 4 quaternions.
    ///
    /// Degenerate (near-zero length) lanes produce identity (0,0,0,1).
    #[inline]
    pub fn normalize(self) -> Self {
        unsafe {
            let lsq     = self.length_sq().0;
            let len      = _mm_sqrt_ps(lsq);
            let eps      = _mm_set1_ps(EPSILON);
            let ok       = _mm_cmpgt_ps(len, eps); // 0xFFFFFFFF where len > eps

            // Avoid div-by-zero: use 1.0 in degenerate lanes
            let safe_len = _mm_or_ps(
                _mm_and_ps(ok, len),
                _mm_andnot_ps(ok, _mm_set1_ps(1.0)),
            );
            let inv = _mm_div_ps(_mm_set1_ps(1.0), safe_len);

            // xyz: normalized where ok, else 0
            let nx = _mm_and_ps(ok, _mm_mul_ps(self.x, inv));
            let ny = _mm_and_ps(ok, _mm_mul_ps(self.y, inv));
            let nz = _mm_and_ps(ok, _mm_mul_ps(self.z, inv));
            // w: normalized where ok, else 1.0 (identity w)
            let nw = _mm_or_ps(
                _mm_and_ps(ok, _mm_mul_ps(self.w, inv)),
                _mm_andnot_ps(ok, _mm_set1_ps(1.0)),
            );

            Self { x: nx, y: ny, z: nz, w: nw }
        }
    }

    /// Conjugate of all 4 quaternions — negate xyz, keep w.
    ///
    /// Equivalent to inverse for unit quaternions. XOR sign bit = 1 instruction.
    #[inline(always)]
    pub fn conjugate(self) -> Self {
        unsafe {
            let sign = _mm_set1_ps(-0.0); // 0x80000000 per lane
            Self {
                x: _mm_xor_ps(self.x, sign),
                y: _mm_xor_ps(self.y, sign),
                z: _mm_xor_ps(self.z, sign),
                w: self.w,
            }
        }
    }

    // ── Hamilton product ───────────────────────────────────────────────────────
    //
    // For each of 4 independent lane pairs simultaneously:
    //   result.x = lw*rx + lx*rw + ly*rz - lz*ry
    //   result.y = lw*ry - lx*rz + ly*rw + lz*rx
    //   result.z = lw*rz + lx*ry - ly*rx + lz*rw
    //   result.w = lw*rw - lx*rx - ly*ry - lz*rz
    //
    // 16 muls + 12 add/subs → same as 1 scalar Hamilton product but 4× throughput.
    // LLVM fuses mul+add to vfmadd231ps on FMA3 CPUs (CI Xeon/EPYC are FMA3).

    /// 4 independent Hamilton products simultaneously.
    #[inline(always)]
    pub fn mul_quatx4(self, rhs: Self) -> Self {
        unsafe {
            let (lx, ly, lz, lw) = (self.x, self.y, self.z, self.w);
            let (rx, ry, rz, rw) = (rhs.x,  rhs.y,  rhs.z,  rhs.w);

            // result.x = lw*rx + lx*rw + ly*rz - lz*ry
            let x = _mm_add_ps(
                _mm_add_ps(_mm_mul_ps(lw, rx), _mm_mul_ps(lx, rw)),
                _mm_sub_ps(_mm_mul_ps(ly, rz), _mm_mul_ps(lz, ry)),
            );
            // result.y = lw*ry - lx*rz + ly*rw + lz*rx
            let y = _mm_add_ps(
                _mm_sub_ps(_mm_mul_ps(lw, ry), _mm_mul_ps(lx, rz)),
                _mm_add_ps(_mm_mul_ps(ly, rw), _mm_mul_ps(lz, rx)),
            );
            // result.z = lw*rz + lx*ry - ly*rx + lz*rw
            let z = _mm_add_ps(
                _mm_add_ps(_mm_mul_ps(lw, rz), _mm_mul_ps(lx, ry)),
                _mm_sub_ps(_mm_mul_ps(lz, rw), _mm_mul_ps(ly, rx)),
            );
            // result.w = lw*rw - lx*rx - ly*ry - lz*rz
            let w = _mm_sub_ps(
                _mm_mul_ps(lw, rw),
                _mm_add_ps(
                    _mm_add_ps(_mm_mul_ps(lx, rx), _mm_mul_ps(ly, ry)),
                    _mm_mul_ps(lz, rz),
                ),
            );

            Self { x, y, z, w }
        }
    }

    // ── Interpolation ──────────────────────────────────────────────────────────

    /// Normalised linear interpolation — 4 pairs simultaneously.
    ///
    /// `t[i]` blends self[i] toward rhs[i].
    /// Shortest-path flip via sign-bit XOR — branchless.
    #[inline]
    pub fn nlerp(self, rhs: Self, t: f32x4) -> Self {
        unsafe {
            let dot = self.dot(rhs).0;

            // sign_mask: 0x80000000 per lane where dot < 0, else 0
            let sign_mask = _mm_and_ps(dot, _mm_set1_ps(-0.0));

            // Flip all rhs components in lanes where dot was negative
            let rx = _mm_xor_ps(rhs.x, sign_mask);
            let ry = _mm_xor_ps(rhs.y, sign_mask);
            let rz = _mm_xor_ps(rhs.z, sign_mask);
            let rw = _mm_xor_ps(rhs.w, sign_mask);

            // SIMD lerp: a + (b - a) * t
            let tt = t.0;
            let lerped = Self {
                x: _mm_add_ps(self.x, _mm_mul_ps(_mm_sub_ps(rx, self.x), tt)),
                y: _mm_add_ps(self.y, _mm_mul_ps(_mm_sub_ps(ry, self.y), tt)),
                z: _mm_add_ps(self.z, _mm_mul_ps(_mm_sub_ps(rz, self.z), tt)),
                w: _mm_add_ps(self.w, _mm_mul_ps(_mm_sub_ps(rw, self.w), tt)),
            };
            lerped.normalize()
        }
    }

    // ── Rotation ──────────────────────────────────────────────────────────────

    /// Rotate 4 vectors by 4 quaternions simultaneously.
    ///
    /// Each self[i] rotates v[i] via the sandwich product.
    ///
    /// Two-cross-product formula — avoids mat3 construction:
    /// ```text
    /// t      = 2 * cross(q.xyz, v)
    /// result = v + w*t + cross(q.xyz, t)
    /// ```
    #[inline]
    pub fn rotate(self, v: Vec3x4) -> Vec3x4 {
        unsafe {
            let qxyz = Vec3x4 { x: self.x, y: self.y, z: self.z };

            // t = 2 * cross(qxyz, v)
            let cross1 = qxyz.cross(v);
            let two = _mm_set1_ps(2.0);
            let t = Vec3x4 {
                x: _mm_mul_ps(two, cross1.x),
                y: _mm_mul_ps(two, cross1.y),
                z: _mm_mul_ps(two, cross1.z),
            };

            // wt = w * t (broadcast w across xyz of t)
            let wt = Vec3x4 {
                x: _mm_mul_ps(self.w, t.x),
                y: _mm_mul_ps(self.w, t.y),
                z: _mm_mul_ps(self.w, t.z),
            };

            // result = v + wt + cross(qxyz, t)
            let cross2 = qxyz.cross(t);
            Vec3x4 {
                x: _mm_add_ps(v.x, _mm_add_ps(wt.x, cross2.x)),
                y: _mm_add_ps(v.y, _mm_add_ps(wt.y, cross2.y)),
                z: _mm_add_ps(v.z, _mm_add_ps(wt.z, cross2.z)),
            }
        }
    }

    // ── Predicates ────────────────────────────────────────────────────────────

    /// True if all components of all 4 quaternions are finite.
    #[inline]
    pub fn is_finite(self) -> bool {
        self.to_array().iter().all(|q| q.is_finite())
    }
}

// ── Operators ─────────────────────────────────────────────────────────────────

impl Mul for QuatX4 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self { self.mul_quatx4(rhs) }
}

impl PartialEq for QuatX4 {
    #[inline]
    fn eq(&self, rhs: &Self) -> bool {
        unsafe {
            let mx = _mm_movemask_ps(_mm_cmpeq_ps(self.x, rhs.x));
            let my = _mm_movemask_ps(_mm_cmpeq_ps(self.y, rhs.y));
            let mz = _mm_movemask_ps(_mm_cmpeq_ps(self.z, rhs.z));
            let mw = _mm_movemask_ps(_mm_cmpeq_ps(self.w, rhs.w));
            mx == 0b1111 && my == 0b1111 && mz == 0b1111 && mw == 0b1111
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

impl From<[Quat; 4]> for QuatX4 {
    #[inline] fn from(a: [Quat; 4]) -> Self { Self::from_slice(&a) }
}
impl From<QuatX4> for [Quat; 4] {
    #[inline] fn from(v: QuatX4) -> Self { v.to_array() }
  }
