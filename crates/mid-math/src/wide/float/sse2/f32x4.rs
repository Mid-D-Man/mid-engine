// crates/mid-math/src/wide/float/sse2/f32x4.rs
//! 4-lane f32 scalar — SSE2, x86 / x86_64.
//!
//! NOT a vector — a bag of 4 independent scalars. Used for:
//!   - Per-lane t-values in Vec3x4::lerp
//!   - 4 dot products returned from Vec3x4::dot
//!   - Scalar multipliers for Vec3x4::scale
//!   - Any per-lane f32 work that pairs with Vec3x4
//!
//! Key methods:
//!   recip_sqrt: rsqrtps + Newton-Raphson → ~23-bit accuracy
//!   recip:      rcpps   + Newton-Raphson → ~23-bit accuracy
//!   sqrt:       _mm_sqrt_ps              → full IEEE754
//!   blend:      branchless select via Mask4

#![allow(non_camel_case_types)]

use core::fmt;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use super::mask4::Mask4;

#[repr(C)]
union UCast { f: [f32; 4], v: f32x4 }

/// 4-lane independent f32 scalar. 16 bytes, 16-byte aligned. Backed by `__m128`.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct f32x4(pub(crate) __m128);

// ── Internal fast math helpers ────────────────────────────────────────────────

/// rsqrtps + one Newton-Raphson step → ~23-bit mantissa accuracy.
///
/// Formula: r_new = 0.5 * r * (3.0 - x * r * r)
/// Starting from rsqrtps (~12-bit), one NR step reaches ~23-bit (IEEE754 f32).
#[inline(always)]
pub(crate) unsafe fn rsqrt_nr(x: __m128) -> __m128 {
    let r     = _mm_rsqrt_ps(x);
    let half  = _mm_set1_ps(0.5);
    let three = _mm_set1_ps(3.0);
    let xrr   = _mm_mul_ps(x, _mm_mul_ps(r, r));      // x * r²
    let nr    = _mm_sub_ps(three, xrr);                // 3 - x*r²
    _mm_mul_ps(_mm_mul_ps(half, r), nr)                // 0.5 * r * (3 - x*r²)
}

/// rcpps + one Newton-Raphson step → ~23-bit mantissa accuracy.
///
/// Formula: r_new = r * (2.0 - x * r)
#[inline(always)]
pub(crate) unsafe fn rcp_nr(x: __m128) -> __m128 {
    let r  = _mm_rcp_ps(x);
    let xr = _mm_mul_ps(x, r);                         // x * r
    let nr = _mm_sub_ps(_mm_set1_ps(2.0), xr);         // 2 - x*r
    _mm_mul_ps(r, nr)                                   // r * (2 - x*r)
}

impl f32x4 {
    // ── Constants ─────────────────────────────────────────────────────────────

    pub const ZERO: Self = unsafe { UCast { f: [0.0; 4] }.v };
    pub const ONE:  Self = unsafe { UCast { f: [1.0; 4] }.v };
    pub const NEG_ONE: Self = unsafe { UCast { f: [-1.0; 4] }.v };

    pub const INFINITY:     Self = unsafe { UCast { f: [f32::INFINITY; 4] }.v };
    pub const NEG_INFINITY: Self = unsafe { UCast { f: [f32::NEG_INFINITY; 4] }.v };

    // ── Constructors ──────────────────────────────────────────────────────────

    /// Broadcast `v` to all 4 lanes.
    #[inline(always)]
    pub fn splat(v: f32) -> Self {
        Self(unsafe { _mm_set1_ps(v) })
    }

    /// Create from 4 values. `a` = lane 0, `d` = lane 3.
    #[inline(always)]
    pub fn new(a: f32, b: f32, c: f32, d: f32) -> Self {
        // _mm_set_ps takes args highest-lane first.
        Self(unsafe { _mm_set_ps(d, c, b, a) })
    }

    #[inline(always)]
    pub fn from_array(a: [f32; 4]) -> Self {
        Self(unsafe { _mm_loadu_ps(a.as_ptr()) })
    }

    #[inline(always)]
    pub fn to_array(self) -> [f32; 4] {
        unsafe {
            let mut a = [0.0f32; 4];
            _mm_storeu_ps(a.as_mut_ptr(), self.0);
            a
        }
    }

    /// Extract a single lane. Panics if `i >= 4`.
    #[inline]
    pub fn get(self, i: usize) -> f32 {
        assert!(i < 4, "f32x4::get — lane {i} out of bounds (max 3)");
        unsafe { UCast { v: self }.f[i] }
    }

    // ── Precise math ──────────────────────────────────────────────────────────

    /// Full-precision square root per lane (IEEE754, `_mm_sqrt_ps`).
    #[inline(always)]
    pub fn sqrt(self) -> Self {
        Self(unsafe { _mm_sqrt_ps(self.0) })
    }

    // ── Fast approximate math ─────────────────────────────────────────────────

    /// Fast reciprocal square root: `1.0 / sqrt(x)` per lane.
    ///
    /// Uses `rsqrtps` + one Newton-Raphson step → ~23-bit mantissa.
    /// ~4× faster than `sqrt` + `div` for bulk normalize operations.
    /// Not IEEE754 rounded — use `sqrt().recip()` for exact results.
    #[inline(always)]
    pub fn recip_sqrt(self) -> Self {
        Self(unsafe { rsqrt_nr(self.0) })
    }

    /// Fast reciprocal: `1.0 / x` per lane.
    ///
    /// Uses `rcpps` + one Newton-Raphson step → ~23-bit mantissa.
    /// Not IEEE754 rounded.
    #[inline(always)]
    pub fn recip(self) -> Self {
        Self(unsafe { rcp_nr(self.0) })
    }

    // ── Component-wise arithmetic ─────────────────────────────────────────────

    #[inline(always)]
    pub fn abs(self) -> Self {
        unsafe {
            // Clear sign bit via ANDNOT with -0.0
            Self(_mm_andnot_ps(_mm_set1_ps(-0.0), self.0))
        }
    }

    #[inline(always)]
    pub fn min(self, rhs: Self) -> Self {
        Self(unsafe { _mm_min_ps(self.0, rhs.0) })
    }

    #[inline(always)]
    pub fn max(self, rhs: Self) -> Self {
        Self(unsafe { _mm_max_ps(self.0, rhs.0) })
    }

    #[inline(always)]
    pub fn clamp(self, lo: Self, hi: Self) -> Self {
        self.max(lo).min(hi)
    }

    /// Horizontal min — extracts all 4 lanes, returns smallest.
    #[inline]
    pub fn min_element(self) -> f32 {
        let a = self.to_array();
        a[0].min(a[1]).min(a[2]).min(a[3])
    }

    /// Horizontal max — extracts all 4 lanes, returns largest.
    #[inline]
    pub fn max_element(self) -> f32 {
        let a = self.to_array();
        a[0].max(a[1]).max(a[2]).max(a[3])
    }

    // ── Fused multiply-add ────────────────────────────────────────────────────

    /// `self * b + c` — fused on FMA3 CPUs (LLVM auto-contracts).
    /// Falls back to two instructions on SSE2-only.
    #[inline(always)]
    pub fn mul_add(self, b: Self, c: Self) -> Self {
        // LLVM will emit vfmadd231ps on FMA3 targets automatically.
        // We write it as two ops; the compiler fuses if target_feature includes fma.
        Self(unsafe { _mm_add_ps(_mm_mul_ps(self.0, b.0), c.0) })
    }

    // ── Branchless select ─────────────────────────────────────────────────────

    /// Per-lane branchless select using [`Mask4`].
    ///
    /// Uses float bitwise ops — no domain crossing vs `_mm_and_si128`.
    #[inline(always)]
    pub fn blend(mask: Mask4, if_true: Self, if_false: Self) -> Self {
        unsafe {
            Self(_mm_or_ps(
                _mm_and_ps(mask.0, if_true.0),
                _mm_andnot_ps(mask.0, if_false.0),
            ))
        }
    }

    // ── Comparisons → Mask4 ───────────────────────────────────────────────────

    #[inline(always)]
    pub fn cmpeq(self, rhs: Self) -> Mask4 {
        Mask4(unsafe { _mm_cmpeq_ps(self.0, rhs.0) })
    }

    #[inline(always)]
    pub fn cmpne(self, rhs: Self) -> Mask4 {
        Mask4(unsafe { _mm_cmpneq_ps(self.0, rhs.0) })
    }

    #[inline(always)]
    pub fn cmplt(self, rhs: Self) -> Mask4 {
        Mask4(unsafe { _mm_cmplt_ps(self.0, rhs.0) })
    }

    #[inline(always)]
    pub fn cmple(self, rhs: Self) -> Mask4 {
        Mask4(unsafe { _mm_cmple_ps(self.0, rhs.0) })
    }

    #[inline(always)]
    pub fn cmpgt(self, rhs: Self) -> Mask4 {
        Mask4(unsafe { _mm_cmpgt_ps(self.0, rhs.0) })
    }

    #[inline(always)]
    pub fn cmpge(self, rhs: Self) -> Mask4 {
        Mask4(unsafe { _mm_cmpge_ps(self.0, rhs.0) })
    }

    // ── Predicates ────────────────────────────────────────────────────────────

    /// True if all 4 lanes are finite (not NaN or infinity).
    #[inline]
    pub fn is_finite(self) -> bool {
        let a = self.to_array();
        a.iter().all(|x| x.is_finite())
    }

    /// True if any lane is NaN.
    #[inline]
    pub fn is_nan(self) -> bool {
        // NaN != NaN: cmpeq returns 0 for NaN lanes
        unsafe { _mm_movemask_ps(_mm_cmpeq_ps(self.0, self.0)) != 0b1111 }
    }
}

// ── Operators ─────────────────────────────────────────────────────────────────

impl Add for f32x4 {
    type Output = Self;
    #[inline(always)]
    fn add(self, r: Self) -> Self { Self(unsafe { _mm_add_ps(self.0, r.0) }) }
}
impl AddAssign for f32x4 { #[inline(always)] fn add_assign(&mut self, r: Self) { *self = *self + r; } }

impl Sub for f32x4 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, r: Self) -> Self { Self(unsafe { _mm_sub_ps(self.0, r.0) }) }
}
impl SubAssign for f32x4 { #[inline(always)] fn sub_assign(&mut self, r: Self) { *self = *self - r; } }

impl Mul for f32x4 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, r: Self) -> Self { Self(unsafe { _mm_mul_ps(self.0, r.0) }) }
}
impl MulAssign for f32x4 { #[inline(always)] fn mul_assign(&mut self, r: Self) { *self = *self * r; } }

impl Div for f32x4 {
    type Output = Self;
    #[inline(always)]
    fn div(self, r: Self) -> Self { Self(unsafe { _mm_div_ps(self.0, r.0) }) }
}
impl DivAssign for f32x4 { #[inline(always)] fn div_assign(&mut self, r: Self) { *self = *self / r; } }

impl Neg for f32x4 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self(unsafe { _mm_xor_ps(self.0, _mm_set1_ps(-0.0)) })
    }
}

impl PartialEq for f32x4 {
    #[inline]
    fn eq(&self, r: &Self) -> bool {
        unsafe { _mm_movemask_ps(_mm_cmpeq_ps(self.0, r.0)) == 0b1111 }
    }
}

impl fmt::Debug for f32x4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let a = self.to_array();
        write!(f, "f32x4({}, {}, {}, {})", a[0], a[1], a[2], a[3])
    }
}
impl fmt::Display for f32x4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let a = self.to_array();
        write!(f, "[{}, {}, {}, {}]", a[0], a[1], a[2], a[3])
    }
}

impl From<[f32; 4]> for f32x4 { #[inline] fn from(a: [f32;4]) -> Self { Self::from_array(a) } }
impl From<f32x4> for [f32; 4] { #[inline] fn from(v: f32x4) -> Self { v.to_array() } }
impl From<f32> for f32x4 { #[inline] fn from(v: f32) -> Self { Self::splat(v) } }
