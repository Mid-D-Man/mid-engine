// crates/mid-math/src/f32/sse2/vec3.rs
//! Vec3 backed by `__m128` on x86 / x86_64.
//!
//! Layout: [x, y, z, 0.0] — lane 3 is always kept as 0.0 (padding).
//! Size: 16 bytes, align: 16 bytes.
//!
//! C interop: use CVec3 from crate::ffi::types at the boundary.
//! Internal Rust code uses this type directly.

use core::fmt;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use crate::sse2::{dot3, dot3_in_x, dot3_into_m128, m128_abs};
use crate::f32::scalar::vec4::Vec4;
use crate::f32::vec2::Vec2;
use crate::EPSILON;
use crate::{impl_vec3_deref};

// ── Union for const construction ──────────────────────────────────────────────
// `_mm_set_ps` is not const, so we use a union to create compile-time
// constants (ZERO, ONE, X, Y, Z etc.).

#[repr(C)]
union UnionCast {
    f: [f32; 4],
    v: Vec3,
}

/// 3-dimensional vector. 16 bytes, 16-byte aligned.
///
/// Backed by `__m128` on x86 / x86_64 — the value IS the SIMD register.
///
/// **C interop:** use [`CVec3`][crate::ffi::types::CVec3] at the FFI boundary.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct Vec3(pub(crate) __m128);

// Deref gives .x / .y / .z access on the __m128 storage.
impl_vec3_deref!(Vec3);

impl Vec3 {
    // ── Constants ────────────────────────────────────────────────────────────

    pub const ZERO:  Self = unsafe { UnionCast { f: [0.0, 0.0, 0.0, 0.0] }.v };
    pub const ONE:   Self = unsafe { UnionCast { f: [1.0, 1.0, 1.0, 0.0] }.v };
    pub const X:     Self = unsafe { UnionCast { f: [1.0, 0.0, 0.0, 0.0] }.v };
    pub const Y:     Self = unsafe { UnionCast { f: [0.0, 1.0, 0.0, 0.0] }.v };
    pub const Z:     Self = unsafe { UnionCast { f: [0.0, 0.0, 1.0, 0.0] }.v };
    pub const NEG_X: Self = unsafe { UnionCast { f: [-1.0,  0.0,  0.0, 0.0] }.v };
    pub const NEG_Y: Self = unsafe { UnionCast { f: [ 0.0, -1.0,  0.0, 0.0] }.v };
    pub const NEG_Z: Self = unsafe { UnionCast { f: [ 0.0,  0.0, -1.0, 0.0] }.v };

    // ── Constructors ─────────────────────────────────────────────────────────

    /// Create from components. Lane 3 is always set to 0.0.
    #[inline(always)]
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        // _mm_set_ps(lane3, lane2, lane1, lane0) — high to low
        Self(unsafe { _mm_set_ps(0.0, z, y, x) })
    }

    #[inline(always)]
    pub fn splat(v: f32) -> Self {
        Self(unsafe { _mm_set_ps(0.0, v, v, v) })
    }

    #[inline(always)]
    pub fn from_array(a: [f32; 3]) -> Self { Self::new(a[0], a[1], a[2]) }

    #[inline(always)]
    pub fn to_array(self) -> [f32; 3] { [self.x, self.y, self.z] }

    // ── Extend / truncate ─────────────────────────────────────────────────────

    /// Extend to Vec4 by appending `w`.
    #[inline(always)]
    pub fn extend(self, w: f32) -> Vec4 {
        // Insert w into lane 3 of the existing register.
        // _mm_insert_ps needs SSE4.1 — use set instead, it's the same cost.
        Vec4(unsafe { _mm_set_ps(w, self.z, self.y, self.x) })
    }

    /// Truncate to Vec2 (drops z).
    #[inline(always)]
    pub fn truncate(self) -> Vec2 { Vec2::new(self.x, self.y) }

    // ── Arithmetic ────────────────────────────────────────────────────────────

    /// Dot product.
    #[inline]
    pub fn dot(self, rhs: Self) -> f32 {
        unsafe { dot3(self.0, rhs.0) }
    }

    /// Returns a Vec3 where every component holds the dot product.
    /// Useful for normalise / length chains without extracting to scalar.
    #[inline]
    pub fn dot_into_vec(self, rhs: Self) -> Self {
        Self(unsafe { dot3_into_m128(self.0, rhs.0) })
    }

    /// Cross product. Result lane 3 = 0.0.
    ///
    /// Uses the shuffle pattern from glam's Vec3A implementation:
    ///   (self.zxy() * rhs - self * rhs.zxy()).zxy()
    /// which requires only two shuffles and two multiply-subtract ops.
    #[inline]
    pub fn cross(self, rhs: Self) -> Self {
        unsafe {
            // shuffle self into [z, x, y, _] (zxy permutation)
            let lhs_zxy = _mm_shuffle_ps::<0b00_00_10_01>(self.0, self.0);
            // shuffle rhs into [z, x, y, _]
            let rhs_zxy = _mm_shuffle_ps::<0b00_00_10_01>(rhs.0, rhs.0);
            // lhs_zxy * rhs  − lhs * rhs_zxy
            let a = _mm_sub_ps(
                _mm_mul_ps(lhs_zxy, rhs.0),
                _mm_mul_ps(self.0, rhs_zxy),
            );
            // shuffle result back: [z,x,y,_] → [x,y,z,_]
            Self(_mm_shuffle_ps::<0b00_00_10_01>(a, a))
        }
    }

    /// Squared length.
    #[inline]
    pub fn length_sq(self) -> f32 { self.dot(self) }

    /// Euclidean length.
    #[inline]
    pub fn length(self) -> f32 {
        unsafe {
            let dot = dot3_in_x(self.0, self.0);
            _mm_cvtss_f32(_mm_sqrt_ps(dot))
        }
    }

    /// 1.0 / length. Valid only when length > 0.
    #[inline]
    pub fn length_recip(self) -> f32 {
        unsafe {
            let dot = dot3_in_x(self.0, self.0);
            _mm_cvtss_f32(_mm_div_ps(Self::ONE.0, _mm_sqrt_ps(dot)))
        }
    }

    /// Normalize to unit length. Returns ZERO if length < EPSILON.
    #[inline]
    pub fn normalize(self) -> Self {
        unsafe {
            let len = _mm_sqrt_ps(dot3_into_m128(self.0, self.0));
            let normalized = Self(_mm_div_ps(self.0, len));
            // If any lane of len was < EPSILON the result may be ±inf or NaN.
            // Mask those out to ZERO.
            let is_finite = _mm_cmpgt_ps(len, _mm_set1_ps(EPSILON));
            Self(_mm_and_ps(normalized.0, is_finite))
        }
    }

    /// Returns Some(normalized) or None if near-zero.
    #[inline]
    pub fn try_normalize(self) -> Option<Self> {
        let rcp = self.length_recip();
        if rcp.is_finite() && rcp > 0.0 {
            Some(self * rcp)
        } else {
            None
        }
    }

    /// Returns normalized or `fallback` if near-zero.
    #[inline]
    pub fn normalize_or(self, fallback: Self) -> Self {
        self.try_normalize().unwrap_or(fallback)
    }

    /// Returns normalized or ZERO if near-zero.
    #[inline]
    pub fn normalize_or_zero(self) -> Self {
        self.normalize_or(Self::ZERO)
    }

    /// Returns true if length ≈ 1.0 (within 2e-4).
    #[inline]
    pub fn is_normalized(self) -> bool {
        (self.length_sq() - 1.0).abs() <= 2e-4
    }

    /// Linear interpolation. `t = 0` → self, `t = 1` → rhs.
    #[inline]
    pub fn lerp(self, rhs: Self, t: f32) -> Self {
        // self + (rhs - self) * t  — one FMA pattern
        unsafe {
            let tt = _mm_set1_ps(t);
            Self(_mm_add_ps(self.0, _mm_mul_ps(_mm_sub_ps(rhs.0, self.0), tt)))
        }
    }

    /// Reflect incident vector over normal `n` (n must be normalized).
    #[inline]
    pub fn reflect(self, n: Self) -> Self {
        self - n * (2.0 * self.dot(n))
    }

    /// Euclidean distance to `rhs`.
    #[inline]
    pub fn distance(self, rhs: Self) -> f32 { (self - rhs).length() }

    /// Squared Euclidean distance to `rhs`.
    #[inline]
    pub fn distance_sq(self, rhs: Self) -> f32 { (self - rhs).length_sq() }

    // ── Component-wise min / max / clamp / abs ────────────────────────────────

    #[inline]
    pub fn min(self, rhs: Self) -> Self {
        Self(unsafe { _mm_min_ps(self.0, rhs.0) })
    }

    #[inline]
    pub fn max(self, rhs: Self) -> Self {
        Self(unsafe { _mm_max_ps(self.0, rhs.0) })
    }

    #[inline]
    pub fn clamp(self, lo: Self, hi: Self) -> Self {
        self.max(lo).min(hi)
    }

    #[inline]
    pub fn abs(self) -> Self {
        Self(unsafe { m128_abs(self.0) })
    }

    /// Returns true if all components are finite.
    #[inline]
    pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite()
    }

    /// Returns true if any component is NaN.
    #[inline]
    pub fn is_nan(self) -> bool {
        self.x.is_nan() || self.y.is_nan() || self.z.is_nan()
    }

    /// Approximate per-element equality using EPSILON.
    #[inline]
    pub fn approx_eq(self, rhs: Self) -> bool {
        (self - rhs).abs().length_sq() < EPSILON * EPSILON
    }
}

// ── Operator impls ────────────────────────────────────────────────────────────

impl Add for Vec3 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self(unsafe { _mm_add_ps(self.0, rhs.0) })
    }
}

impl Sub for Vec3 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self(unsafe { _mm_sub_ps(self.0, rhs.0) })
    }
}

impl Mul<f32> for Vec3 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, s: f32) -> Self {
        Self(unsafe { _mm_mul_ps(self.0, _mm_set1_ps(s)) })
    }
}

impl Mul<Vec3> for f32 {
    type Output = Vec3;
    #[inline(always)]
    fn mul(self, v: Vec3) -> Vec3 {
        Vec3(unsafe { _mm_mul_ps(_mm_set1_ps(self), v.0) })
    }
}

impl Mul for Vec3 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self(unsafe { _mm_mul_ps(self.0, rhs.0) })
    }
}

impl Div<f32> for Vec3 {
    type Output = Self;
    #[inline(always)]
    fn div(self, s: f32) -> Self {
        Self(unsafe { _mm_div_ps(self.0, _mm_set1_ps(s)) })
    }
}

impl Neg for Vec3 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self(unsafe { _mm_xor_ps(self.0, _mm_set1_ps(-0.0)) })
    }
}

impl AddAssign for Vec3 {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) { self.0 = unsafe { _mm_add_ps(self.0, rhs.0) }; }
}
impl SubAssign for Vec3 {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) { self.0 = unsafe { _mm_sub_ps(self.0, rhs.0) }; }
}
impl MulAssign<f32> for Vec3 {
    #[inline(always)]
    fn mul_assign(&mut self, s: f32) { self.0 = unsafe { _mm_mul_ps(self.0, _mm_set1_ps(s)) }; }
}
impl DivAssign<f32> for Vec3 {
    #[inline(always)]
    fn div_assign(&mut self, s: f32) { self.0 = unsafe { _mm_div_ps(self.0, _mm_set1_ps(s)) }; }
}

// ── Equality / Display ────────────────────────────────────────────────────────

impl PartialEq for Vec3 {
    #[inline]
    fn eq(&self, rhs: &Self) -> bool {
        // Compare only lanes 0-2 (x, y, z). Lane 3 is padding and must be ignored.
        unsafe {
            // _mm_cmpeq_ps returns all-1 bits per lane where equal
            let cmp = _mm_cmpeq_ps(self.0, rhs.0);
            // movemask: bit k = sign bit of lane k
            // We only care about bits 0-2
            (_mm_movemask_ps(cmp) & 0b0111) == 0b0111
        }
    }
}

impl Default for Vec3 { fn default() -> Self { Self::ZERO } }

impl fmt::Debug for Vec3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Vec3")
            .field(&self.x).field(&self.y).field(&self.z)
            .finish()
    }
}

impl fmt::Display for Vec3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {}, {})", self.x, self.y, self.z)
    }
}

// ── Conversions ───────────────────────────────────────────────────────────────

impl From<[f32; 3]> for Vec3 {
    #[inline] fn from(a: [f32; 3]) -> Self { Self::new(a[0], a[1], a[2]) }
}
impl From<Vec3> for [f32; 3] {
    #[inline] fn from(v: Vec3) -> Self { [v.x, v.y, v.z] }
}
impl From<(f32, f32, f32)> for Vec3 {
    #[inline] fn from(t: (f32, f32, f32)) -> Self { Self::new(t.0, t.1, t.2) }
}
impl From<Vec3> for (f32, f32, f32) {
    #[inline] fn from(v: Vec3) -> Self { (v.x, v.y, v.z) }
}
