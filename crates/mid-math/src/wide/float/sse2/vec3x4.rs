// crates/mid-math/src/wide/float/sse2/vec3x4.rs
//! 4 × Vec3 packed in SoA layout — SSE2, x86 / x86_64.
//!
//! SoA layout: x = [x0,x1,x2,x3], y = [y0,y1,y2,y3], z = [z0,z1,z2,z3].
//! Each field is a __m128 holding one component for all 4 vectors.
//!
//! All 4 vectors are operated on simultaneously. dot() returns f32x4 with
//! 4 independent dot products. normalize() normalizes all 4 in one call.
//!
//! AoS→SoA transpose (from_vec3s) uses 7 shuffle instructions:
//!   unpacklo×2, unpackhi×2, movelh×2, movehl×1.
//!
//! See sse2.rs for the rsqrt_nr helper used in normalize().

use core::fmt;
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use super::mask4::Mask4;
use super::f32x4::{f32x4, rsqrt_nr};
use crate::f32::sse2::vec3::Vec3;
use crate::sse2::m128_from_f32x4;
use crate::EPSILON;

/// 4 × Vec3 in SoA layout. 48 bytes, 16-byte aligned.
///
/// Fields are public for advanced intrinsic use.
/// Prefer the provided methods for correctness.
#[derive(Clone, Copy)]
#[repr(C, align(16))]
pub struct Vec3x4 {
    /// x-components of all 4 vectors: [x0, x1, x2, x3]
    pub x: __m128,
    /// y-components of all 4 vectors: [y0, y1, y2, y3]
    pub y: __m128,
    /// z-components of all 4 vectors: [z0, z1, z2, z3]
    pub z: __m128,
}

impl Vec3x4 {
    // ── Constants ─────────────────────────────────────────────────────────────

    pub const ZERO: Self = Self {
        x: m128_from_f32x4([0.0; 4]),
        y: m128_from_f32x4([0.0; 4]),
        z: m128_from_f32x4([0.0; 4]),
    };

    pub const X: Self = Self {
        x: m128_from_f32x4([1.0; 4]),
        y: m128_from_f32x4([0.0; 4]),
        z: m128_from_f32x4([0.0; 4]),
    };

    pub const Y: Self = Self {
        x: m128_from_f32x4([0.0; 4]),
        y: m128_from_f32x4([1.0; 4]),
        z: m128_from_f32x4([0.0; 4]),
    };

    pub const Z: Self = Self {
        x: m128_from_f32x4([0.0; 4]),
        y: m128_from_f32x4([0.0; 4]),
        z: m128_from_f32x4([1.0; 4]),
    };

    // ── Constructors ──────────────────────────────────────────────────────────

    /// Build from 4 individual Vec3s.
    ///
    /// Performs a 7-shuffle AoS→SoA transpose:
    /// - unpacklo×2: interleave x/y pairs from adjacent vectors
    /// - unpackhi×2: extract z+pad pairs
    /// - movelh×2 + movehl×1: assemble the 3 SoA lanes
    ///
    /// Cost: 7 SSE2 shuffle instructions. Called once at bulk setup;
    /// all subsequent ops on Vec3x4 are free of per-element overhead.
    #[inline]
    pub fn from_vec3s(a: Vec3, b: Vec3, c: Vec3, d: Vec3) -> Self {
        // Each Vec3 is [x, y, z, 0.0] in __m128.
        unsafe {
            // Step 1: interleave low halves (x and y components) of adjacent pairs
            // unpacklo_ps(p, q) = [p[0], q[0], p[1], q[1]]
            let lo01 = _mm_unpacklo_ps(a.0, b.0); // [x0, x1, y0, y1]
            let lo23 = _mm_unpacklo_ps(c.0, d.0); // [x2, x3, y2, y3]

            // Step 2: interleave high halves (z and pad=0.0) of adjacent pairs
            // unpackhi_ps(p, q) = [p[2], q[2], p[3], q[3]]
            let hi01 = _mm_unpackhi_ps(a.0, b.0); // [z0, z1, 0,  0 ]
            let hi23 = _mm_unpackhi_ps(c.0, d.0); // [z2, z3, 0,  0 ]

            // Step 3: assemble SoA lanes
            // movelh_ps(p, q) = [p[0], p[1], q[0], q[1]]
            let x = _mm_movelh_ps(lo01, lo23); // [x0, x1, x2, x3] ✓

            // movehl_ps(a, b) = [b[2], b[3], a[2], a[3]]
            // movehl_ps(lo23, lo01) = [lo01[2], lo01[3], lo23[2], lo23[3]]
            //                       = [y0,      y1,      y2,      y3     ] ✓
            let y = _mm_movehl_ps(lo23, lo01); // [y0, y1, y2, y3] ✓

            let z = _mm_movelh_ps(hi01, hi23); // [z0, z1, z2, z3] ✓

            Self { x, y, z }
        }
    }

    /// Build from a slice of 4 Vec3s.
    #[inline(always)]
    pub fn from_slice(s: &[Vec3; 4]) -> Self {
        Self::from_vec3s(s[0], s[1], s[2], s[3])
    }

    /// Broadcast a single Vec3 to all 4 lanes.
    #[inline(always)]
    pub fn splat(v: Vec3) -> Self {
        unsafe {
            Self {
                x: _mm_set1_ps(v.x),
                y: _mm_set1_ps(v.y),
                z: _mm_set1_ps(v.z),
            }
        }
    }

    /// Extract all 4 vectors as an array (SoA→AoS transpose).
    ///
    /// Inverse of from_vec3s. Uses the same 7-shuffle pattern in reverse.
    #[inline]
    pub fn to_array(self) -> [Vec3; 4] {
        unsafe {
            // Interleave x and y into pairs per vector
            // unpacklo_ps(x, y) = [x0, y0, x1, y1]
            let xy_lo = _mm_unpacklo_ps(self.x, self.y); // [x0, y0, x1, y1]
            let xy_hi = _mm_unpackhi_ps(self.x, self.y); // [x2, y2, x3, y3]

            // Interleave z and 0 for padding
            let zero    = _mm_setzero_ps();
            let z_lo    = _mm_unpacklo_ps(self.z, zero); // [z0, 0, z1, 0]
            let z_hi    = _mm_unpackhi_ps(self.z, zero); // [z2, 0, z3, 0]

            // Assemble each Vec3 from xy pair + z
            // movelh_ps(p, q) = [p[0], p[1], q[0], q[1]]
            let v0 = Vec3(_mm_movelh_ps(xy_lo, z_lo)); // [x0, y0, z0, 0] ✓
            // movehl_ps(a, b) = [b[2], b[3], a[2], a[3]]
            let v1 = Vec3(_mm_movehl_ps(z_lo, xy_lo));  // [x1, y1, z1, 0] ✓
            let v2 = Vec3(_mm_movelh_ps(xy_hi, z_hi));  // [x2, y2, z2, 0] ✓
            let v3 = Vec3(_mm_movehl_ps(z_hi, xy_hi));  // [x3, y3, z3, 0] ✓

            [v0, v1, v2, v3]
        }
    }

    /// Write to a mutable slice of 4 Vec3s.
    #[inline(always)]
    pub fn write_to_slice(self, s: &mut [Vec3; 4]) {
        let a = self.to_array();
        s.copy_from_slice(&a);
    }

    /// Extract a single lane as a Vec3. Panics if `lane >= 4`.
    #[inline]
    pub fn get(self, lane: usize) -> Vec3 {
        assert!(lane < 4, "Vec3x4::get — lane {lane} out of bounds (max 3)");
        self.to_array()[lane]
    }

    // ── Arithmetic ────────────────────────────────────────────────────────────

    /// Element-wise multiply (NOT dot product — use `.dot()` for that).
    #[inline(always)]
    pub fn mul_elem(self, rhs: Self) -> Self {
        unsafe {
            Self {
                x: _mm_mul_ps(self.x, rhs.x),
                y: _mm_mul_ps(self.y, rhs.y),
                z: _mm_mul_ps(self.z, rhs.z),
            }
        }
    }

    /// Scale each of the 4 vectors by a per-lane scalar.
    ///
    /// `s[i]` multiplies all 3 components of vector i.
    #[inline(always)]
    pub fn scale(self, s: f32x4) -> Self {
        unsafe {
            Self {
                x: _mm_mul_ps(self.x, s.0),
                y: _mm_mul_ps(self.y, s.0),
                z: _mm_mul_ps(self.z, s.0),
            }
        }
    }

    /// Scale all 4 vectors by the same scalar.
    #[inline(always)]
    pub fn scale_uniform(self, s: f32) -> Self {
        self.scale(f32x4::splat(s))
    }

    /// Fused multiply-add: `self * b + c` per component, per lane.
    ///
    /// On FMA3 CPUs LLVM auto-contracts to `vfmadd231ps`.
    #[inline(always)]
    pub fn madd(self, b: Self, c: Self) -> Self {
        unsafe {
            Self {
                x: _mm_add_ps(_mm_mul_ps(self.x, b.x), c.x),
                y: _mm_add_ps(_mm_mul_ps(self.y, b.y), c.y),
                z: _mm_add_ps(_mm_mul_ps(self.z, b.z), c.z),
            }
        }
    }

    // ── Geometric ops ─────────────────────────────────────────────────────────

    /// 4 independent dot products simultaneously.
    ///
    /// Returns `f32x4` where lane i = dot(self[i], rhs[i]).
    /// Cost: 3 muls + 2 adds = same as a single scalar dot product
    /// but processes 4 pairs in parallel.
    #[inline(always)]
    pub fn dot(self, rhs: Self) -> f32x4 {
        unsafe {
            let xx = _mm_mul_ps(self.x, rhs.x);
            let yy = _mm_mul_ps(self.y, rhs.y);
            let zz = _mm_mul_ps(self.z, rhs.z);
            f32x4(_mm_add_ps(_mm_add_ps(xx, yy), zz))
        }
    }

    /// 4 independent cross products simultaneously.
    ///
    /// result[i] = cross(self[i], rhs[i]).
    /// No shuffles needed — SoA layout already has components separated.
    #[inline(always)]
    pub fn cross(self, rhs: Self) -> Self {
        unsafe {
            Self {
                // x = self.y * rhs.z - self.z * rhs.y
                x: _mm_sub_ps(
                    _mm_mul_ps(self.y, rhs.z),
                    _mm_mul_ps(self.z, rhs.y),
                ),
                // y = self.z * rhs.x - self.x * rhs.z
                y: _mm_sub_ps(
                    _mm_mul_ps(self.z, rhs.x),
                    _mm_mul_ps(self.x, rhs.z),
                ),
                // z = self.x * rhs.y - self.y * rhs.x
                z: _mm_sub_ps(
                    _mm_mul_ps(self.x, rhs.y),
                    _mm_mul_ps(self.y, rhs.x),
                ),
            }
        }
    }

    /// Squared length of each vector.
    #[inline(always)]
    pub fn length_sq(self) -> f32x4 { self.dot(self) }

    /// Length of each vector (full sqrt, IEEE754).
    #[inline(always)]
    pub fn length(self) -> f32x4 {
        f32x4(unsafe { _mm_sqrt_ps(self.length_sq().0) })
    }

    /// Normalize all 4 vectors — fast path using rsqrtps + Newton-Raphson.
    ///
    /// Accuracy: ~23-bit mantissa (same as IEEE754 f32 precision after NR).
    /// Zero-length vectors produce zero output (not NaN) — guarded by EPSILON mask.
    ///
    /// ~4× faster than normalize_precise for bulk normalization.
    #[inline]
    pub fn normalize(self) -> Self {
        unsafe {
            let len_sq  = self.length_sq().0;
            let inv_len = rsqrt_nr(len_sq);
            // Zero out lanes where length is below EPSILON (degenerate vectors)
            let ok = _mm_cmpgt_ps(len_sq, _mm_set1_ps(EPSILON * EPSILON));
            let inv = _mm_and_ps(inv_len, ok);
            Self {
                x: _mm_mul_ps(self.x, inv),
                y: _mm_mul_ps(self.y, inv),
                z: _mm_mul_ps(self.z, inv),
            }
        }
    }

    /// Normalize all 4 vectors — precise path using full sqrt + divide.
    ///
    /// IEEE754 accurate. ~4× slower than `normalize()`.
    /// Use when accuracy matters more than throughput.
    #[inline]
    pub fn normalize_precise(self) -> Self {
        unsafe {
            let len_sq = self.length_sq().0;
            let len    = _mm_sqrt_ps(len_sq);
            let ok     = _mm_cmpgt_ps(len, _mm_set1_ps(EPSILON));
            // Safe reciprocal: replace near-zero with 1.0 before dividing
            let safe = _mm_or_ps(
                _mm_and_ps(ok, len),
                _mm_andnot_ps(ok, _mm_set1_ps(1.0)),
            );
            let inv = _mm_div_ps(_mm_set1_ps(1.0), safe);
            let inv = _mm_and_ps(inv, ok); // zero out degenerate lanes
            Self {
                x: _mm_mul_ps(self.x, inv),
                y: _mm_mul_ps(self.y, inv),
                z: _mm_mul_ps(self.z, inv),
            }
        }
    }

    // ── Interpolation / component-wise ────────────────────────────────────────

    /// Per-lane linear interpolation with a per-lane t value.
    ///
    /// `t[i]` is the blend factor for vector i. t=0 → self, t=1 → rhs.
    #[inline(always)]
    pub fn lerp(self, rhs: Self, t: f32x4) -> Self {
        // self + (rhs - self) * t
        unsafe {
            let diff_x = _mm_sub_ps(rhs.x, self.x);
            let diff_y = _mm_sub_ps(rhs.y, self.y);
            let diff_z = _mm_sub_ps(rhs.z, self.z);
            Self {
                x: _mm_add_ps(self.x, _mm_mul_ps(diff_x, t.0)),
                y: _mm_add_ps(self.y, _mm_mul_ps(diff_y, t.0)),
                z: _mm_add_ps(self.z, _mm_mul_ps(diff_z, t.0)),
            }
        }
    }

    /// Component-wise minimum across all 4 vectors.
    #[inline(always)]
    pub fn min(self, rhs: Self) -> Self {
        unsafe {
            Self {
                x: _mm_min_ps(self.x, rhs.x),
                y: _mm_min_ps(self.y, rhs.y),
                z: _mm_min_ps(self.z, rhs.z),
            }
        }
    }

    /// Component-wise maximum across all 4 vectors.
    #[inline(always)]
    pub fn max(self, rhs: Self) -> Self {
        unsafe {
            Self {
                x: _mm_max_ps(self.x, rhs.x),
                y: _mm_max_ps(self.y, rhs.y),
                z: _mm_max_ps(self.z, rhs.z),
            }
        }
    }

    // ── Branchless select ─────────────────────────────────────────────────────

    /// Per-lane branchless select.
    ///
    /// `mask[i]` true → `if_true[i]`, false → `if_false[i]`.
    /// No branch, no misprediction penalty.
    #[inline(always)]
    pub fn select(mask: Mask4, if_true: Self, if_false: Self) -> Self {
        unsafe {
            Self {
                x: _mm_or_ps(
                    _mm_and_ps(mask.0, if_true.x),
                    _mm_andnot_ps(mask.0, if_false.x),
                ),
                y: _mm_or_ps(
                    _mm_and_ps(mask.0, if_true.y),
                    _mm_andnot_ps(mask.0, if_false.y),
                ),
                z: _mm_or_ps(
                    _mm_and_ps(mask.0, if_true.z),
                    _mm_andnot_ps(mask.0, if_false.z),
                ),
            }
        }
    }

    // ── Comparisons → Mask4 ───────────────────────────────────────────────────
    // Comparisons are per-component across all 4 lanes simultaneously.
    // The mask is true in lane i if ALL 3 components of that vector satisfy the condition.

    /// Length-squared comparison: true in lane i if length_sq(self[i]) < length_sq(rhs[i]).
    #[inline(always)]
    pub fn length_lt(self, rhs: Self) -> Mask4 {
        self.length_sq().cmplt(rhs.length_sq())
    }

    // ── Predicates ────────────────────────────────────────────────────────────

    /// True if all components of all 4 vectors are finite.
    #[inline]
    pub fn is_finite(self) -> bool {
        unsafe {
            let inf = _mm_set1_ps(f32::INFINITY);
            let ax  = _mm_andnot_ps(_mm_set1_ps(-0.0), self.x);
            let ay  = _mm_andnot_ps(_mm_set1_ps(-0.0), self.y);
            let az  = _mm_andnot_ps(_mm_set1_ps(-0.0), self.z);
            // All lanes must be strictly less than infinity
            let mx = _mm_movemask_ps(_mm_cmplt_ps(ax, inf));
            let my = _mm_movemask_ps(_mm_cmplt_ps(ay, inf));
            let mz = _mm_movemask_ps(_mm_cmplt_ps(az, inf));
            mx == 0b1111 && my == 0b1111 && mz == 0b1111
        }
    }
}

// ── Operators ─────────────────────────────────────────────────────────────────

impl Add for Vec3x4 {
    type Output = Self;
    #[inline(always)]
    fn add(self, r: Self) -> Self {
        unsafe {
            Self {
                x: _mm_add_ps(self.x, r.x),
                y: _mm_add_ps(self.y, r.y),
                z: _mm_add_ps(self.z, r.z),
            }
        }
    }
}
impl AddAssign for Vec3x4 { #[inline(always)] fn add_assign(&mut self, r: Self) { *self = *self + r; } }

impl Sub for Vec3x4 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, r: Self) -> Self {
        unsafe {
            Self {
                x: _mm_sub_ps(self.x, r.x),
                y: _mm_sub_ps(self.y, r.y),
                z: _mm_sub_ps(self.z, r.z),
            }
        }
    }
}
impl SubAssign for Vec3x4 { #[inline(always)] fn sub_assign(&mut self, r: Self) { *self = *self - r; } }

impl Neg for Vec3x4 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        unsafe {
            let sign = _mm_set1_ps(-0.0);
            Self {
                x: _mm_xor_ps(self.x, sign),
                y: _mm_xor_ps(self.y, sign),
                z: _mm_xor_ps(self.z, sign),
            }
        }
    }
}

/// Element-wise multiply — use `.dot()` for dot products.
impl Mul for Vec3x4 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, r: Self) -> Self { self.mul_elem(r) }
}
impl MulAssign for Vec3x4 { #[inline(always)] fn mul_assign(&mut self, r: Self) { *self = *self * r; } }

/// Scale all 4 vectors by a per-lane f32x4.
impl Mul<f32x4> for Vec3x4 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, s: f32x4) -> Self { self.scale(s) }
}

/// Scale all 4 vectors by a uniform f32.
impl Mul<f32> for Vec3x4 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, s: f32) -> Self { self.scale_uniform(s) }
}

impl PartialEq for Vec3x4 {
    fn eq(&self, r: &Self) -> bool {
        unsafe {
            let mx = _mm_movemask_ps(_mm_cmpeq_ps(self.x, r.x));
            let my = _mm_movemask_ps(_mm_cmpeq_ps(self.y, r.y));
            let mz = _mm_movemask_ps(_mm_cmpeq_ps(self.z, r.z));
            mx == 0b1111 && my == 0b1111 && mz == 0b1111
        }
    }
}

impl Default for Vec3x4 { fn default() -> Self { Self::ZERO } }

impl fmt::Debug for Vec3x4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let a = self.to_array();
        write!(f, "Vec3x4([{:?}, {:?}, {:?}, {:?}])", a[0], a[1], a[2], a[3])
    }
}
impl fmt::Display for Vec3x4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let a = self.to_array();
        write!(f, "[{}, {}, {}, {}]", a[0], a[1], a[2], a[3])
    }
}

impl From<[Vec3; 4]> for Vec3x4 {
    #[inline] fn from(a: [Vec3; 4]) -> Self { Self::from_slice(&a) }
}
impl From<Vec3x4> for [Vec3; 4] {
    #[inline] fn from(v: Vec3x4) -> Self { v.to_array() }
  }
