// crates/mid-math/src/f64/sse2/dvec4.rs
//! DVec4 backed by 2× `__m128d` on x86 / x86_64.
//!
//! Storage layout (32 bytes, 32-byte aligned):
//!   lo: __m128d → [x (bytes 0-7), y (bytes 8-15)]
//!   hi: __m128d → [z (bytes 16-23), w (bytes 24-31)]
//!
//! This is byte-identical to `XYZW<f64> {x, y, z, w}` (each 8 bytes, no gaps),
//! enabling the zero-cost `impl_dvec4_deref!` cast.
//!
//! Key ops benefiting from SSE2 f64:
//!   add/sub/mul/div     — 2 ops instead of 4 (parallel lo+hi)
//!   dot product         — parallel mul + sequential hadd
//!   normalize           — sqrt + div across both registers
//!   lerp                — FMA-style (no FMA3 required)
//!
//! Operations that stay scalar:
//!   floor/ceil/round    — no SSE2 _mm_floor_pd (needs SSE4.1)
//!   is_finite/is_nan    — scalar predicate check
//!   extend/truncate     — simple lane manipulation

use core::fmt;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use crate::sse2::{dot4d, dot4d_into_m128d, m128d_abs};
use crate::impl_dvec4_deref;
use crate::DEPSILON;

// ── Union for compile-time constant initialisation ────────────────────────────
// Memory layout: [f64; 4] and DVec4 are byte-identical at their first 32 bytes.
// DVec4.lo @0 = [f[0], f[1]], DVec4.hi @16 = [f[2], f[3]].
#[repr(C)]
union UnionCast {
    f: [f64; 4],
    v: DVec4,
}

// ── Type ──────────────────────────────────────────────────────────────────────

/// 4D double-precision vector. 32 bytes, 32-byte aligned.
/// Backed by 2× `__m128d` on x86 / x86_64: `lo=[x,y]`, `hi=[z,w]`.
///
/// **C interop:** use `CDVec4` at the FFI boundary.
#[derive(Clone, Copy)]
#[repr(C, align(32))]
pub struct DVec4 {
    pub(crate) lo: __m128d, // [x, y]
    pub(crate) hi: __m128d, // [z, w]
}

// Deref to XYZW<f64>: layout is {x@0, y@8, z@16, w@24} — matches DVec4.
impl_dvec4_deref!(DVec4);

// ── Constants ─────────────────────────────────────────────────────────────────

impl DVec4 {
    pub const ZERO: Self = unsafe { UnionCast { f: [0.0; 4] }.v };
    pub const ONE:  Self = unsafe { UnionCast { f: [1.0; 4] }.v };
    pub const X:    Self = unsafe { UnionCast { f: [1.0, 0.0, 0.0, 0.0] }.v };
    pub const Y:    Self = unsafe { UnionCast { f: [0.0, 1.0, 0.0, 0.0] }.v };
    pub const Z:    Self = unsafe { UnionCast { f: [0.0, 0.0, 1.0, 0.0] }.v };
    pub const W:    Self = unsafe { UnionCast { f: [0.0, 0.0, 0.0, 1.0] }.v };

    // ── Constructors ─────────────────────────────────────────────────────────

    /// Create from four components.
    #[inline(always)]
    pub fn new(x: f64, y: f64, z: f64, w: f64) -> Self {
        Self {
            lo: unsafe { _mm_set_pd(y, x) }, // lane0=x, lane1=y
            hi: unsafe { _mm_set_pd(w, z) }, // lane0=z, lane1=w
        }
    }

    #[inline(always)]
    pub fn splat(v: f64) -> Self {
        let r = unsafe { _mm_set1_pd(v) };
        Self { lo: r, hi: r }
    }

    #[inline(always)] pub fn from_array(a: [f64; 4]) -> Self { Self::new(a[0], a[1], a[2], a[3]) }
    #[inline(always)] pub fn to_array(self) -> [f64; 4] { [self.x, self.y, self.z, self.w] }

    /// Drop w, return DVec3.
    #[inline(always)]
    pub fn truncate(self) -> crate::DVec3 {
        crate::DVec3::new(self.x, self.y, self.z)
    }

    // ── Arithmetic ────────────────────────────────────────────────────────────

    /// 4-lane dot product using parallel lo+hi multiplication.
    #[inline(always)]
    pub fn dot(self, rhs: Self) -> f64 {
        unsafe { dot4d(self.lo, self.hi, rhs.lo, rhs.hi) }
    }

    #[inline(always)] pub fn length_sq(self) -> f64 { self.dot(self) }

    #[inline]
    pub fn length(self) -> f64 {
        unsafe { _mm_cvtsd_f64(_mm_sqrt_pd(_mm_set1_pd(self.dot(self)))) }
    }

    #[inline]
    pub fn length_recip(self) -> f64 {
        let l = self.length();
        if l < DEPSILON { 0.0 } else { 1.0 / l }
    }

    /// Normalize to unit length. Returns ZERO for near-zero-length vectors.
    ///
    /// Applies the same guard mask to both lo and hi registers simultaneously.
    #[inline]
    pub fn normalize(self) -> Self {
        unsafe {
            let len_v = dot4d_into_m128d(self.lo, self.hi, self.lo, self.hi);
            let sqrt  = _mm_sqrt_pd(len_v); // [sqrt(dot), sqrt(dot)]
            let lo_n  = _mm_div_pd(self.lo, sqrt);
            let hi_n  = _mm_div_pd(self.hi, sqrt);
            let ok    = _mm_cmpgt_pd(sqrt, _mm_set1_pd(DEPSILON));
            Self {
                lo: _mm_and_pd(lo_n, ok),
                hi: _mm_and_pd(hi_n, ok),
            }
        }
    }

    #[inline]
    pub fn try_normalize(self) -> Option<Self> {
        let rcp = self.length_recip();
        if rcp > 0.0 && rcp.is_finite() { Some(self * rcp) } else { None }
    }

    #[inline] pub fn normalize_or(self, fb: Self) -> Self { self.try_normalize().unwrap_or(fb) }
    #[inline] pub fn normalize_or_zero(self) -> Self { self.normalize_or(Self::ZERO) }
    #[inline] pub fn is_normalized(self) -> bool { (self.length_sq() - 1.0).abs() <= 2e-10 }

    // ── Interpolation ─────────────────────────────────────────────────────────

    /// Linear interpolation: `self + (rhs - self) * t`.
    ///
    /// Applied to both lo and hi registers independently — two parallel
    /// SSE2 subtract + multiply + add chains.
    #[inline]
    pub fn lerp(self, rhs: Self, t: f64) -> Self {
        unsafe {
            let tt      = _mm_set1_pd(t);
            let lo_diff = _mm_sub_pd(rhs.lo, self.lo);
            let hi_diff = _mm_sub_pd(rhs.hi, self.hi);
            Self {
                lo: _mm_add_pd(self.lo, _mm_mul_pd(lo_diff, tt)),
                hi: _mm_add_pd(self.hi, _mm_mul_pd(hi_diff, tt)),
            }
        }
    }

    #[inline] pub fn distance(self, rhs: Self)    -> f64 { (self - rhs).length() }
    #[inline] pub fn distance_sq(self, rhs: Self) -> f64 { (self - rhs).length_sq() }

    // ── Component-wise ────────────────────────────────────────────────────────

    #[inline]
    pub fn min(self, rhs: Self) -> Self {
        Self { lo: unsafe { _mm_min_pd(self.lo, rhs.lo) },
               hi: unsafe { _mm_min_pd(self.hi, rhs.hi) } }
    }
    #[inline]
    pub fn max(self, rhs: Self) -> Self {
        Self { lo: unsafe { _mm_max_pd(self.lo, rhs.lo) },
               hi: unsafe { _mm_max_pd(self.hi, rhs.hi) } }
    }
    #[inline]
    pub fn clamp(self, lo: Self, hi: Self) -> Self { self.max(lo).min(hi) }
    #[inline]
    pub fn abs(self) -> Self {
        Self { lo: unsafe { m128d_abs(self.lo) }, hi: unsafe { m128d_abs(self.hi) } }
    }

    // ── Predicates ────────────────────────────────────────────────────────────

    #[inline]
    pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite()
            && self.z.is_finite() && self.w.is_finite()
    }
    #[inline]
    pub fn is_nan(self) -> bool {
        self.x.is_nan() || self.y.is_nan() || self.z.is_nan() || self.w.is_nan()
    }

    #[inline]
    pub fn approx_eq(self, rhs: Self) -> bool {
        unsafe {
            let lo_diff = m128d_abs(_mm_sub_pd(self.lo, rhs.lo));
            let hi_diff = m128d_abs(_mm_sub_pd(self.hi, rhs.hi));
            let eps     = _mm_set1_pd(DEPSILON);
            let lo_ok   = _mm_movemask_pd(_mm_cmplt_pd(lo_diff, eps));
            let hi_ok   = _mm_movemask_pd(_mm_cmplt_pd(hi_diff, eps));
            (lo_ok & 0b11) == 0b11 && (hi_ok & 0b11) == 0b11
        }
    }

    // ── Cast ─────────────────────────────────────────────────────────────────

    /// Lossy cast to single-precision `Vec4`.
    #[inline]
    pub fn as_vec4(self) -> crate::Vec4 {
        crate::Vec4::new(self.x as f32, self.y as f32, self.z as f32, self.w as f32)
    }
}

// ── Operators ─────────────────────────────────────────────────────────────────

macro_rules! impl_binop {
    ($trait:ident, $method:ident, $op:ident) => {
        impl $trait for DVec4 {
            type Output = Self;
            #[inline(always)]
            fn $method(self, r: Self) -> Self {
                Self { lo: unsafe { $op(self.lo, r.lo) },
                       hi: unsafe { $op(self.hi, r.hi) } }
            }
        }
    };
}

impl_binop!(Add, add, _mm_add_pd);
impl_binop!(Sub, sub, _mm_sub_pd);
impl_binop!(Mul, mul, _mm_mul_pd);
impl_binop!(Div, div, _mm_div_pd);

impl Mul<f64> for DVec4 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, s: f64) -> Self {
        unsafe {
            let sv = _mm_set1_pd(s);
            Self { lo: _mm_mul_pd(self.lo, sv), hi: _mm_mul_pd(self.hi, sv) }
        }
    }
}
impl Mul<DVec4> for f64 {
    type Output = DVec4;
    #[inline(always)]
    fn mul(self, v: DVec4) -> DVec4 { v * self }
}
impl Div<f64> for DVec4 {
    type Output = Self;
    #[inline(always)]
    fn div(self, s: f64) -> Self {
        unsafe {
            let sv = _mm_set1_pd(s);
            Self { lo: _mm_div_pd(self.lo, sv), hi: _mm_div_pd(self.hi, sv) }
        }
    }
}
impl Neg for DVec4 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        unsafe {
            let sign = _mm_set1_pd(-0.0);
            Self { lo: _mm_xor_pd(self.lo, sign), hi: _mm_xor_pd(self.hi, sign) }
        }
    }
}

impl AddAssign for DVec4 {
    #[inline(always)]
    fn add_assign(&mut self, r: Self) {
        self.lo = unsafe { _mm_add_pd(self.lo, r.lo) };
        self.hi = unsafe { _mm_add_pd(self.hi, r.hi) };
    }
}
impl SubAssign for DVec4 {
    #[inline(always)]
    fn sub_assign(&mut self, r: Self) {
        self.lo = unsafe { _mm_sub_pd(self.lo, r.lo) };
        self.hi = unsafe { _mm_sub_pd(self.hi, r.hi) };
    }
}
impl MulAssign<f64> for DVec4 {
    #[inline(always)]
    fn mul_assign(&mut self, s: f64) {
        unsafe {
            let sv = _mm_set1_pd(s);
            self.lo = _mm_mul_pd(self.lo, sv);
            self.hi = _mm_mul_pd(self.hi, sv);
        }
    }
}
impl DivAssign<f64> for DVec4 {
    #[inline(always)]
    fn div_assign(&mut self, s: f64) {
        unsafe {
            let sv = _mm_set1_pd(s);
            self.lo = _mm_div_pd(self.lo, sv);
            self.hi = _mm_div_pd(self.hi, sv);
        }
    }
}

// ── PartialEq ─────────────────────────────────────────────────────────────────

impl PartialEq for DVec4 {
    #[inline]
    fn eq(&self, rhs: &Self) -> bool {
        unsafe {
            let lo_eq = _mm_movemask_pd(_mm_cmpeq_pd(self.lo, rhs.lo));
            let hi_eq = _mm_movemask_pd(_mm_cmpeq_pd(self.hi, rhs.hi));
            (lo_eq & 0b11) == 0b11 && (hi_eq & 0b11) == 0b11
        }
    }
}

impl Default for DVec4 { fn default() -> Self { Self::ZERO } }

// ── Display / Debug ───────────────────────────────────────────────────────────

impl fmt::Debug for DVec4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("DVec4")
            .field(&self.x).field(&self.y).field(&self.z).field(&self.w).finish()
    }
}
impl fmt::Display for DVec4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {}, {}, {})", self.x, self.y, self.z, self.w)
    }
}

// ── Conversions ───────────────────────────────────────────────────────────────

impl From<[f64; 4]> for DVec4 { #[inline] fn from(a: [f64; 4]) -> Self { Self::new(a[0], a[1], a[2], a[3]) } }
impl From<DVec4> for [f64; 4] { #[inline] fn from(v: DVec4) -> Self { [v.x, v.y, v.z, v.w] } }
impl From<(f64, f64, f64, f64)> for DVec4 { #[inline] fn from(t: (f64,f64,f64,f64)) -> Self { Self::new(t.0,t.1,t.2,t.3) } }
impl From<DVec4> for (f64, f64, f64, f64) { #[inline] fn from(v: DVec4) -> Self { (v.x, v.y, v.z, v.w) } }
