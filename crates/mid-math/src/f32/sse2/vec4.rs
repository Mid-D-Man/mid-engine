// crates/mid-math/src/f32/sse2/vec4.rs
//! Vec4 backed by `__m128` on x86 / x86_64.
//!
//! Layout: [x, y, z, w] — all four lanes are meaningful.
//! Size: 16 bytes, align: 16 bytes.

use core::fmt;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use crate::sse2::{dot4, dot4_in_x, m128_abs};
use crate::f32::sse2::vec3::Vec3;
use crate::EPSILON;
use crate::impl_vec4_deref;

#[repr(C)]
union UnionCast {
    f: [f32; 4],
    v: Vec4,
}

/// 4-dimensional vector. 16 bytes, 16-byte aligned.
///
/// Backed by `__m128` on x86 / x86_64.
///
/// **C interop:** use [`CVec4`][crate::ffi::types::CVec4] at the FFI boundary.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct Vec4(pub(crate) __m128);

impl_vec4_deref!(Vec4);

impl Vec4 {
    pub const ZERO: Self = unsafe { UnionCast { f: [0.0; 4] }.v };
    pub const ONE:  Self = unsafe { UnionCast { f: [1.0; 4] }.v };
    pub const X:    Self = unsafe { UnionCast { f: [1.0, 0.0, 0.0, 0.0] }.v };
    pub const Y:    Self = unsafe { UnionCast { f: [0.0, 1.0, 0.0, 0.0] }.v };
    pub const Z:    Self = unsafe { UnionCast { f: [0.0, 0.0, 1.0, 0.0] }.v };
    pub const W:    Self = unsafe { UnionCast { f: [0.0, 0.0, 0.0, 1.0] }.v };

    #[inline(always)]
    pub fn new(x: f32, y: f32, z: f32, w: f32) -> Self {
        Self(unsafe { _mm_set_ps(w, z, y, x) })
    }

    #[inline(always)]
    pub fn splat(v: f32) -> Self {
        Self(unsafe { _mm_set1_ps(v) })
    }

    #[inline(always)]
    pub fn from_array(a: [f32; 4]) -> Self { Self::new(a[0], a[1], a[2], a[3]) }

    #[inline(always)]
    pub fn to_array(self) -> [f32; 4] { [self.x, self.y, self.z, self.w] }

    /// Truncate to Vec3 (drops w).
    #[inline(always)]
    pub fn truncate(self) -> Vec3 {
        Vec3(unsafe { _mm_and_ps(self.0, crate::sse2::m128_from_f32x4(
            [f32::from_bits(0xFFFF_FFFF), f32::from_bits(0xFFFF_FFFF),
             f32::from_bits(0xFFFF_FFFF), 0.0]
        ))})
    }

    // ── Arithmetic ─────────────────────────────────────────────────────────

    #[inline]
    pub fn dot(self, rhs: Self) -> f32 { unsafe { dot4(self.0, rhs.0) } }

    #[inline]
    pub fn length_sq(self) -> f32 { self.dot(self) }

    #[inline]
    pub fn length(self) -> f32 {
        unsafe {
            let dot = dot4_in_x(self.0, self.0);
            _mm_cvtss_f32(_mm_sqrt_ps(dot))
        }
    }

    #[inline]
    pub fn length_recip(self) -> f32 {
        unsafe {
            let dot = dot4_in_x(self.0, self.0);
            _mm_cvtss_f32(_mm_div_ps(Self::ONE.0, _mm_sqrt_ps(dot)))
        }
    }

    #[inline]
    pub fn normalize(self) -> Self {
        unsafe {
            let len = _mm_sqrt_ps(crate::sse2::dot4_into_m128(self.0, self.0));
            let n   = Self(_mm_div_ps(self.0, len));
            let ok  = _mm_cmpgt_ps(len, _mm_set1_ps(EPSILON));
            Self(_mm_and_ps(n.0, ok))
        }
    }

    #[inline]
    pub fn try_normalize(self) -> Option<Self> {
        let rcp = self.length_recip();
        if rcp.is_finite() && rcp > 0.0 { Some(self * rcp) } else { None }
    }

    #[inline]
    pub fn normalize_or_zero(self) -> Self {
        self.try_normalize().unwrap_or(Self::ZERO)
    }

    #[inline]
    pub fn lerp(self, rhs: Self, t: f32) -> Self {
        unsafe {
            let tt = _mm_set1_ps(t);
            Self(_mm_add_ps(self.0, _mm_mul_ps(_mm_sub_ps(rhs.0, self.0), tt)))
        }
    }

    #[inline]
    pub fn abs(self) -> Self { Self(unsafe { m128_abs(self.0) }) }

    #[inline]
    pub fn min(self, rhs: Self) -> Self { Self(unsafe { _mm_min_ps(self.0, rhs.0) }) }

    #[inline]
    pub fn max(self, rhs: Self) -> Self { Self(unsafe { _mm_max_ps(self.0, rhs.0) }) }

    #[inline]
    pub fn clamp(self, lo: Self, hi: Self) -> Self { self.max(lo).min(hi) }

    #[inline]
    pub fn is_normalized(self) -> bool { (self.length_sq() - 1.0).abs() <= 2e-4 }

    #[inline]
    pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite() &&
        self.z.is_finite() && self.w.is_finite()
    }

    #[inline]
    pub fn approx_eq(self, rhs: Self) -> bool {
        unsafe {
            let diff = m128_abs(_mm_sub_ps(self.0, rhs.0));
            let eps  = _mm_set1_ps(EPSILON);
            (_mm_movemask_ps(_mm_cmplt_ps(diff, eps)) & 0b1111) == 0b1111
        }
    }
}

// ── Operators ─────────────────────────────────────────────────────────────────

impl Add for Vec4 {
    type Output = Self;
    #[inline(always)]
    fn add(self, r: Self) -> Self { Self(unsafe { _mm_add_ps(self.0, r.0) }) }
}
impl Sub for Vec4 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, r: Self) -> Self { Self(unsafe { _mm_sub_ps(self.0, r.0) }) }
}
impl Mul<f32> for Vec4 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, s: f32) -> Self { Self(unsafe { _mm_mul_ps(self.0, _mm_set1_ps(s)) }) }
}
impl Mul<Vec4> for f32 {
    type Output = Vec4;
    #[inline(always)]
    fn mul(self, v: Vec4) -> Vec4 { Vec4(unsafe { _mm_mul_ps(_mm_set1_ps(self), v.0) }) }
}
impl Mul for Vec4 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, r: Self) -> Self { Self(unsafe { _mm_mul_ps(self.0, r.0) }) }
}
impl Div<f32> for Vec4 {
    type Output = Self;
    #[inline(always)]
    fn div(self, s: f32) -> Self { Self(unsafe { _mm_div_ps(self.0, _mm_set1_ps(s)) }) }
}
impl Neg for Vec4 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self { Self(unsafe { _mm_xor_ps(self.0, _mm_set1_ps(-0.0)) }) }
}
impl AddAssign for Vec4 {
    #[inline(always)]
    fn add_assign(&mut self, r: Self) { self.0 = unsafe { _mm_add_ps(self.0, r.0) }; }
}
impl SubAssign for Vec4 {
    #[inline(always)]
    fn sub_assign(&mut self, r: Self) { self.0 = unsafe { _mm_sub_ps(self.0, r.0) }; }
}
impl MulAssign<f32> for Vec4 {
    #[inline(always)]
    fn mul_assign(&mut self, s: f32) { self.0 = unsafe { _mm_mul_ps(self.0, _mm_set1_ps(s)) }; }
}
impl DivAssign<f32> for Vec4 {
    #[inline(always)]
    fn div_assign(&mut self, s: f32) { self.0 = unsafe { _mm_div_ps(self.0, _mm_set1_ps(s)) }; }
}

impl PartialEq for Vec4 {
    #[inline]
    fn eq(&self, rhs: &Self) -> bool {
        unsafe {
            (_mm_movemask_ps(_mm_cmpeq_ps(self.0, rhs.0)) & 0b1111) == 0b1111
        }
    }
}

impl Default for Vec4 { fn default() -> Self { Self::ZERO } }

impl fmt::Debug for Vec4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Vec4")
            .field(&self.x).field(&self.y).field(&self.z).field(&self.w)
            .finish()
    }
}
impl fmt::Display for Vec4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {}, {}, {})", self.x, self.y, self.z, self.w)
    }
}

impl From<[f32; 4]> for Vec4 {
    #[inline] fn from(a: [f32; 4]) -> Self { Self::new(a[0], a[1], a[2], a[3]) }
}
impl From<Vec4> for [f32; 4] {
    #[inline] fn from(v: Vec4) -> Self { [v.x, v.y, v.z, v.w] }
}
impl From<(f32, f32, f32, f32)> for Vec4 {
    #[inline] fn from(t: (f32, f32, f32, f32)) -> Self { Self::new(t.0, t.1, t.2, t.3) }
}
