// crates/mid-math/src/wide/float/avx2/vec3x8.rs
//! 8 × Vec3 packed in SoA layout — AVX2, x86 / x86_64.
//!
//! Requires: RUSTFLAGS="-C target-feature=+avx2"
//! 2010 MacBook Pro (Sandy Bridge) does NOT have AVX2. This path only compiles
//! and runs on Haswell (2013) or later, or on GitHub CI (Xeon/EPYC).
//!
//! Layout: x,y,z each a __m256 holding one component for all 8 vectors.
//! Same SoA philosophy as Vec3x4 — doubles the throughput on AVX2 hardware.

use core::fmt;
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use crate::f32::sse2::vec3::Vec3;
use crate::EPSILON;
use super::super::sse2::f32x4::f32x4 as f32x4_sse;

// Predicate constant for _mm256_cmp_ps: greater-than, ordered, quiet
const CMP_GT_OQ: i32 = 30;

/// Fast reciprocal sqrt per lane via rsqrt + one Newton-Raphson step.
/// Accuracy: ~23-bit mantissa (same as f32 precision after NR).
#[inline(always)]
unsafe fn rsqrt_nr_256(x: __m256) -> __m256 {
    let r     = _mm256_rsqrt_ps(x);
    let half  = _mm256_set1_ps(0.5);
    let three = _mm256_set1_ps(3.0);
    let xrr   = _mm256_mul_ps(x, _mm256_mul_ps(r, r));
    _mm256_mul_ps(_mm256_mul_ps(half, r), _mm256_sub_ps(three, xrr))
}

/// 8 × Vec3 in SoA layout. 96 bytes, 32-byte aligned.
///
/// Fields are public for advanced intrinsic use.
/// All operations process 8 vectors simultaneously — double throughput vs Vec3x4
/// on AVX2-capable hardware.
#[derive(Clone, Copy)]
#[repr(C, align(32))]
pub struct Vec3x8 {
    /// x-components of all 8 vectors: [x₀..x₇]
    pub x: __m256,
    /// y-components of all 8 vectors: [y₀..y₇]
    pub y: __m256,
    /// z-components of all 8 vectors: [z₀..z₇]
    pub z: __m256,
}

impl Vec3x8 {
    // ── Constants ─────────────────────────────────────────────────────────────

    pub const ZERO: Self = unsafe {
        // transmute from [f32; 8] — __m256 and [f32;8] are both 32 bytes.
        // SAFETY: all-zeros is valid for __m256.
        core::mem::transmute::<[f32; 24], Self>([0.0f32; 24])
    };

    // ── Constructors ──────────────────────────────────────────────────────────

    /// Build from 8 individual Vec3s.
    ///
    /// `_mm256_set_ps` takes args highest-lane first, so ordering is reversed.
    #[inline]
    pub fn from_vec3s(
        a: Vec3, b: Vec3, c: Vec3, d: Vec3,
        e: Vec3, f: Vec3, g: Vec3, h: Vec3,
    ) -> Self {
        unsafe {
            Self {
                x: _mm256_set_ps(h.x, g.x, f.x, e.x, d.x, c.x, b.x, a.x),
                y: _mm256_set_ps(h.y, g.y, f.y, e.y, d.y, c.y, b.y, a.y),
                z: _mm256_set_ps(h.z, g.z, f.z, e.z, d.z, c.z, b.z, a.z),
            }
        }
    }

    /// Build from a slice of 8 Vec3s.
    #[inline(always)]
    pub fn from_slice(s: &[Vec3; 8]) -> Self {
        Self::from_vec3s(s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7])
    }

    /// Broadcast a single Vec3 to all 8 lanes.
    #[inline(always)]
    pub fn splat(v: Vec3) -> Self {
        unsafe {
            Self {
                x: _mm256_set1_ps(v.x),
                y: _mm256_set1_ps(v.y),
                z: _mm256_set1_ps(v.z),
            }
        }
    }

    /// Extract all 8 vectors as an array (SoA → AoS).
    #[inline]
    pub fn to_array(self) -> [Vec3; 8] {
        unsafe {
            let mut xs = [0.0f32; 8];
            let mut ys = [0.0f32; 8];
            let mut zs = [0.0f32; 8];
            _mm256_storeu_ps(xs.as_mut_ptr(), self.x);
            _mm256_storeu_ps(ys.as_mut_ptr(), self.y);
            _mm256_storeu_ps(zs.as_mut_ptr(), self.z);
            core::array::from_fn(|i| Vec3::new(xs[i], ys[i], zs[i]))
        }
    }

    /// Write to a mutable slice of 8 Vec3s.
    #[inline(always)]
    pub fn write_to_slice(self, s: &mut [Vec3; 8]) {
        let a = self.to_array();
        s.copy_from_slice(&a);
    }

    /// Extract one lane. Panics if `lane >= 8`.
    #[inline]
    pub fn get(self, lane: usize) -> Vec3 {
        assert!(lane < 8, "Vec3x8::get — lane {lane} out of bounds (max 7)");
        self.to_array()[lane]
    }

    // ── Arithmetic ────────────────────────────────────────────────────────────

    /// Element-wise multiply — NOT dot product.
    #[inline(always)]
    pub fn mul_elem(self, rhs: Self) -> Self {
        unsafe {
            Self {
                x: _mm256_mul_ps(self.x, rhs.x),
                y: _mm256_mul_ps(self.y, rhs.y),
                z: _mm256_mul_ps(self.z, rhs.z),
            }
        }
    }

    /// Scale each of the 8 vectors by a uniform scalar.
    #[inline(always)]
    pub fn scale_uniform(self, s: f32) -> Self {
        unsafe {
            let ss = _mm256_set1_ps(s);
            Self {
                x: _mm256_mul_ps(self.x, ss),
                y: _mm256_mul_ps(self.y, ss),
                z: _mm256_mul_ps(self.z, ss),
            }
        }
    }

    /// Fused multiply-add: `self * b + c` per component, per lane.
    /// LLVM auto-contracts to `vfmadd231ps` on FMA3 CPUs.
    #[inline(always)]
    pub fn madd(self, b: Self, c: Self) -> Self {
        unsafe {
            Self {
                x: _mm256_add_ps(_mm256_mul_ps(self.x, b.x), c.x),
                y: _mm256_add_ps(_mm256_mul_ps(self.y, b.y), c.y),
                z: _mm256_add_ps(_mm256_mul_ps(self.z, b.z), c.z),
            }
        }
    }

    // ── Geometric ops ─────────────────────────────────────────────────────────

    /// 8 independent dot products simultaneously.
    ///
    /// Returns a `__m256` where lane i = dot(self[i], rhs[i]).
    /// 3 muls + 2 adds = same as 1 scalar dot but 8× throughput.
    #[inline(always)]
    pub fn dot(self, rhs: Self) -> __m256 {
        unsafe {
            let xx = _mm256_mul_ps(self.x, rhs.x);
            let yy = _mm256_mul_ps(self.y, rhs.y);
            let zz = _mm256_mul_ps(self.z, rhs.z);
            _mm256_add_ps(_mm256_add_ps(xx, yy), zz)
        }
    }

    /// 8 independent cross products simultaneously.
    #[inline(always)]
    pub fn cross(self, rhs: Self) -> Self {
        unsafe {
            Self {
                x: _mm256_sub_ps(
                    _mm256_mul_ps(self.y, rhs.z),
                    _mm256_mul_ps(self.z, rhs.y),
                ),
                y: _mm256_sub_ps(
                    _mm256_mul_ps(self.z, rhs.x),
                    _mm256_mul_ps(self.x, rhs.z),
                ),
                z: _mm256_sub_ps(
                    _mm256_mul_ps(self.x, rhs.y),
                    _mm256_mul_ps(self.y, rhs.x),
                ),
            }
        }
    }

    #[inline(always)]
    pub fn length_sq(self) -> __m256 { self.dot(self) }

    /// Length of each vector (full sqrt).
    #[inline(always)]
    pub fn length(self) -> __m256 {
        unsafe { _mm256_sqrt_ps(self.length_sq()) }
    }

    /// Normalize all 8 vectors — fast path via rsqrtps + Newton-Raphson.
    ///
    /// Degenerate lanes (length ≈ 0) produce zero — guarded by EPSILON.
    #[inline]
    pub fn normalize(self) -> Self {
        unsafe {
            let len_sq  = self.length_sq();
            let eps2    = _mm256_set1_ps(EPSILON * EPSILON);
            let inv     = rsqrt_nr_256(len_sq);
            let ok      = _mm256_cmp_ps(len_sq, eps2, CMP_GT_OQ);
            let masked  = _mm256_and_ps(ok, inv);
            Self {
                x: _mm256_mul_ps(self.x, masked),
                y: _mm256_mul_ps(self.y, masked),
                z: _mm256_mul_ps(self.z, masked),
            }
        }
    }

    /// Normalize — full sqrt + divide for IEEE754 accuracy.
    #[inline]
    pub fn normalize_precise(self) -> Self {
        unsafe {
            let len_sq = self.length_sq();
            let len    = _mm256_sqrt_ps(len_sq);
            let eps    = _mm256_set1_ps(EPSILON);
            let ok     = _mm256_cmp_ps(len, eps, CMP_GT_OQ);
            let safe   = _mm256_blendv_ps(_mm256_set1_ps(1.0), len, ok);
            let inv    = _mm256_div_ps(_mm256_set1_ps(1.0), safe);
            let masked = _mm256_and_ps(ok, inv);
            Self {
                x: _mm256_mul_ps(self.x, masked),
                y: _mm256_mul_ps(self.y, masked),
                z: _mm256_mul_ps(self.z, masked),
            }
        }
    }

    // ── Interpolation ─────────────────────────────────────────────────────────

    /// Per-lane lerp. `t` is a `__m256` of blend factors.
    #[inline(always)]
    pub fn lerp(self, rhs: Self, t: __m256) -> Self {
        unsafe {
            Self {
                x: _mm256_add_ps(self.x, _mm256_mul_ps(_mm256_sub_ps(rhs.x, self.x), t)),
                y: _mm256_add_ps(self.y, _mm256_mul_ps(_mm256_sub_ps(rhs.y, self.y), t)),
                z: _mm256_add_ps(self.z, _mm256_mul_ps(_mm256_sub_ps(rhs.z, self.z), t)),
            }
        }
    }

    /// Per-lane lerp with uniform t.
    #[inline(always)]
    pub fn lerp_uniform(self, rhs: Self, t: f32) -> Self {
        unsafe { self.lerp(rhs, _mm256_set1_ps(t)) }
    }

    /// Component-wise minimum.
    #[inline(always)]
    pub fn min(self, rhs: Self) -> Self {
        unsafe {
            Self {
                x: _mm256_min_ps(self.x, rhs.x),
                y: _mm256_min_ps(self.y, rhs.y),
                z: _mm256_min_ps(self.z, rhs.z),
            }
        }
    }

    /// Component-wise maximum.
    #[inline(always)]
    pub fn max(self, rhs: Self) -> Self {
        unsafe {
            Self {
                x: _mm256_max_ps(self.x, rhs.x),
                y: _mm256_max_ps(self.y, rhs.y),
                z: _mm256_max_ps(self.z, rhs.z),
            }
        }
    }

    /// Branchless per-lane select. `mask` MSB set = choose `if_true`.
    #[inline(always)]
    pub fn select(mask: __m256, if_true: Self, if_false: Self) -> Self {
        unsafe {
            Self {
                x: _mm256_blendv_ps(if_false.x, if_true.x, mask),
                y: _mm256_blendv_ps(if_false.y, if_true.y, mask),
                z: _mm256_blendv_ps(if_false.z, if_true.z, mask),
            }
        }
    }

    // ── Predicates ────────────────────────────────────────────────────────────

    /// True if all components of all 8 vectors are finite.
    #[inline]
    pub fn is_finite(self) -> bool {
        self.to_array().iter().all(|v| v.is_finite())
    }
}

// ── Operators ─────────────────────────────────────────────────────────────────

impl Add for Vec3x8 {
    type Output = Self;
    #[inline(always)]
    fn add(self, r: Self) -> Self {
        unsafe { Self {
            x: _mm256_add_ps(self.x, r.x),
            y: _mm256_add_ps(self.y, r.y),
            z: _mm256_add_ps(self.z, r.z),
        }}
    }
}
impl AddAssign for Vec3x8 { fn add_assign(&mut self, r: Self) { *self = *self + r; } }

impl Sub for Vec3x8 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, r: Self) -> Self {
        unsafe { Self {
            x: _mm256_sub_ps(self.x, r.x),
            y: _mm256_sub_ps(self.y, r.y),
            z: _mm256_sub_ps(self.z, r.z),
        }}
    }
}
impl SubAssign for Vec3x8 { fn sub_assign(&mut self, r: Self) { *self = *self - r; } }

impl Neg for Vec3x8 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        unsafe {
            let sign = _mm256_set1_ps(-0.0);
            Self {
                x: _mm256_xor_ps(self.x, sign),
                y: _mm256_xor_ps(self.y, sign),
                z: _mm256_xor_ps(self.z, sign),
            }
        }
    }
}

impl Mul for Vec3x8 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, r: Self) -> Self { self.mul_elem(r) }
}
impl MulAssign for Vec3x8 { fn mul_assign(&mut self, r: Self) { *self = *self * r; } }

impl Mul<f32> for Vec3x8 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, s: f32) -> Self { self.scale_uniform(s) }
}

impl PartialEq for Vec3x8 {
    fn eq(&self, r: &Self) -> bool {
        unsafe {
            let mx = _mm256_movemask_ps(_mm256_cmp_ps(self.x, r.x, 0)); // _CMP_EQ_OQ = 0
            let my = _mm256_movemask_ps(_mm256_cmp_ps(self.y, r.y, 0));
            let mz = _mm256_movemask_ps(_mm256_cmp_ps(self.z, r.z, 0));
            mx == 0xFF && my == 0xFF && mz == 0xFF
        }
    }
}

impl Default for Vec3x8 { fn default() -> Self { Self::ZERO } }

impl fmt::Debug for Vec3x8 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let a = self.to_array();
        write!(f, "Vec3x8({:?})", a)
    }
}
impl fmt::Display for Vec3x8 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let a = self.to_array();
        write!(f, "{:?}", a)
    }
}
impl From<[Vec3; 8]> for Vec3x8 {
    fn from(a: [Vec3; 8]) -> Self { Self::from_slice(&a) }
}
impl From<Vec3x8> for [Vec3; 8] {
    fn from(v: Vec3x8) -> Self { v.to_array() }
      }
