// crates/mid-math/src/f32/sse2/vec3.rs
// Fix: import sse2::vec4::Vec4, not scalar::vec4::Vec4

use core::fmt;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use crate::sse2::{dot3, dot3_in_x, dot3_into_m128, m128_abs};
// *** FIX: was crate::f32::scalar::vec4::Vec4 — must be the SSE2 type ***
use crate::f32::sse2::vec4::Vec4;
use crate::f32::vec2::Vec2;
use crate::EPSILON;
use crate::impl_vec3_deref;

#[repr(C)]
union UnionCast {
    f: [f32; 4],
    v: Vec3,
}

/// 3-dimensional vector. 16 bytes, 16-byte aligned. Backed by __m128.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct Vec3(pub(crate) __m128);

impl_vec3_deref!(Vec3);

impl Vec3 {
    pub const ZERO:  Self = unsafe { UnionCast { f: [ 0.0,  0.0,  0.0, 0.0] }.v };
    pub const ONE:   Self = unsafe { UnionCast { f: [ 1.0,  1.0,  1.0, 0.0] }.v };
    pub const X:     Self = unsafe { UnionCast { f: [ 1.0,  0.0,  0.0, 0.0] }.v };
    pub const Y:     Self = unsafe { UnionCast { f: [ 0.0,  1.0,  0.0, 0.0] }.v };
    pub const Z:     Self = unsafe { UnionCast { f: [ 0.0,  0.0,  1.0, 0.0] }.v };
    pub const NEG_X: Self = unsafe { UnionCast { f: [-1.0,  0.0,  0.0, 0.0] }.v };
    pub const NEG_Y: Self = unsafe { UnionCast { f: [ 0.0, -1.0,  0.0, 0.0] }.v };
    pub const NEG_Z: Self = unsafe { UnionCast { f: [ 0.0,  0.0, -1.0, 0.0] }.v };

    #[inline(always)]
    pub fn new(x: f32, y: f32, z: f32) -> Self {
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

    /// Extend to Vec4 (SSE2) by setting lane 3 = w.
    #[inline(always)]
    pub fn extend(self, w: f32) -> Vec4 {
        Vec4(unsafe { _mm_set_ps(w, self.z, self.y, self.x) })
    }

    #[inline(always)]
    pub fn truncate(self) -> Vec2 { Vec2::new(self.x, self.y) }

    #[inline]
    pub fn dot(self, rhs: Self) -> f32 {
        unsafe { dot3(self.0, rhs.0) }
    }

    #[inline]
    pub fn dot_into_vec(self, rhs: Self) -> Self {
        Self(unsafe { dot3_into_m128(self.0, rhs.0) })
    }

  #[inline]
pub fn cross(self, rhs: Self) -> Self {
    unsafe {
        // Cross product: result = self × rhs
        //   result.x = a.y·b.z − a.z·b.y
        //   result.y = a.z·b.x − a.x·b.z
        //   result.z = a.x·b.y − a.y·b.x
        //
        // Requires TWO different cyclic permutations — previous code used the
        // SAME shuffle for both operands which computes -(a × b) = b × a.
        //
        // YZX shuffle 0b00_00_10_01: result[i] = src[(01,10,00)] → [y, z, x]
        // ZXY shuffle 0b00_01_00_10: result[i] = src[(10,00,01)] → [z, x, y]
        let a_yzx = _mm_shuffle_ps::<0b00_00_10_01>(self.0, self.0); // [ay, az, ax]
        let b_zxy = _mm_shuffle_ps::<0b00_01_00_10>(rhs.0,  rhs.0); // [bz, bx, by]
        let a_zxy = _mm_shuffle_ps::<0b00_01_00_10>(self.0, self.0); // [az, ax, ay]
        let b_yzx = _mm_shuffle_ps::<0b00_00_10_01>(rhs.0,  rhs.0); // [by, bz, bx]
        // [ay·bz − az·by,  az·bx − ax·bz,  ax·by − ay·bx]
        Self(_mm_sub_ps(
            _mm_mul_ps(a_yzx, b_zxy),
            _mm_mul_ps(a_zxy, b_yzx),
        ))
    }
}
    #[inline] pub fn length_sq(self) -> f32 { self.dot(self) }

    #[inline]
    pub fn length(self) -> f32 {
        unsafe {
            let dot = dot3_in_x(self.0, self.0);
            _mm_cvtss_f32(_mm_sqrt_ps(dot))
        }
    }

    #[inline]
    pub fn length_recip(self) -> f32 {
        unsafe {
            let dot = dot3_in_x(self.0, self.0);
            _mm_cvtss_f32(_mm_div_ps(Self::ONE.0, _mm_sqrt_ps(dot)))
        }
    }

    #[inline]
    pub fn normalize(self) -> Self {
        unsafe {
            let len = _mm_sqrt_ps(dot3_into_m128(self.0, self.0));
            let normalized = Self(_mm_div_ps(self.0, len));
            let is_finite = _mm_cmpgt_ps(len, _mm_set1_ps(EPSILON));
            Self(_mm_and_ps(normalized.0, is_finite))
        }
    }

    #[inline]
    pub fn try_normalize(self) -> Option<Self> {
        let rcp = self.length_recip();
        if rcp.is_finite() && rcp > 0.0 { Some(self * rcp) } else { None }
    }

    #[inline]
    pub fn normalize_or(self, fallback: Self) -> Self {
        self.try_normalize().unwrap_or(fallback)
    }

    #[inline]
    pub fn normalize_or_zero(self) -> Self { self.normalize_or(Self::ZERO) }

    #[inline]
    pub fn is_normalized(self) -> bool { (self.length_sq() - 1.0).abs() <= 2e-4 }

    #[inline]
    pub fn lerp(self, rhs: Self, t: f32) -> Self {
        unsafe {
            let tt = _mm_set1_ps(t);
            Self(_mm_add_ps(self.0, _mm_mul_ps(_mm_sub_ps(rhs.0, self.0), tt)))
        }
    }

    #[inline]
    pub fn reflect(self, n: Self) -> Self { self - n * (2.0 * self.dot(n)) }

    #[inline]
    pub fn distance(self, rhs: Self) -> f32 { (self - rhs).length() }

    #[inline]
    pub fn distance_sq(self, rhs: Self) -> f32 { (self - rhs).length_sq() }

    #[inline]
    pub fn min(self, rhs: Self) -> Self { Self(unsafe { _mm_min_ps(self.0, rhs.0) }) }

    #[inline]
    pub fn max(self, rhs: Self) -> Self { Self(unsafe { _mm_max_ps(self.0, rhs.0) }) }

    #[inline]
    pub fn clamp(self, lo: Self, hi: Self) -> Self { self.max(lo).min(hi) }

    #[inline]
    pub fn abs(self) -> Self { Self(unsafe { m128_abs(self.0) }) }

    #[inline]
    pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite()
    }

    #[inline]
    pub fn is_nan(self) -> bool {
        self.x.is_nan() || self.y.is_nan() || self.z.is_nan()
    }

    #[inline]
    pub fn approx_eq(self, rhs: Self) -> bool {
        (self - rhs).abs().length_sq() < EPSILON * EPSILON
    }
}

impl Add for Vec3 {
    type Output = Self;
    #[inline(always)]
    fn add(self, r: Self) -> Self { Self(unsafe { _mm_add_ps(self.0, r.0) }) }
}
impl Sub for Vec3 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, r: Self) -> Self { Self(unsafe { _mm_sub_ps(self.0, r.0) }) }
}
impl Mul<f32> for Vec3 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, s: f32) -> Self { Self(unsafe { _mm_mul_ps(self.0, _mm_set1_ps(s)) }) }
}
impl Mul<Vec3> for f32 {
    type Output = Vec3;
    #[inline(always)]
    fn mul(self, v: Vec3) -> Vec3 { Vec3(unsafe { _mm_mul_ps(_mm_set1_ps(self), v.0) }) }
}
impl Mul for Vec3 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, r: Self) -> Self { Self(unsafe { _mm_mul_ps(self.0, r.0) }) }
}
impl Div<f32> for Vec3 {
    type Output = Self;
    #[inline(always)]
    fn div(self, s: f32) -> Self { Self(unsafe { _mm_div_ps(self.0, _mm_set1_ps(s)) }) }
}
impl Neg for Vec3 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self { Self(unsafe { _mm_xor_ps(self.0, _mm_set1_ps(-0.0)) }) }
}
impl AddAssign for Vec3 {
    #[inline(always)]
    fn add_assign(&mut self, r: Self) { self.0 = unsafe { _mm_add_ps(self.0, r.0) }; }
}
impl SubAssign for Vec3 {
    #[inline(always)]
    fn sub_assign(&mut self, r: Self) { self.0 = unsafe { _mm_sub_ps(self.0, r.0) }; }
}
impl MulAssign<f32> for Vec3 {
    #[inline(always)]
    fn mul_assign(&mut self, s: f32) { self.0 = unsafe { _mm_mul_ps(self.0, _mm_set1_ps(s)) }; }
}
impl DivAssign<f32> for Vec3 {
    #[inline(always)]
    fn div_assign(&mut self, s: f32) { self.0 = unsafe { _mm_div_ps(self.0, _mm_set1_ps(s)) }; }
}

impl PartialEq for Vec3 {
    #[inline]
    fn eq(&self, rhs: &Self) -> bool {
        unsafe {
            (_mm_movemask_ps(_mm_cmpeq_ps(self.0, rhs.0)) & 0b0111) == 0b0111
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
