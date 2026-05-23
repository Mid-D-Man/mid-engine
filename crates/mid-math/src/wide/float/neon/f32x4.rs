// crates/mid-math/src/wide/float/neon/f32x4.rs
//! 4-lane independent f32 scalar — NEON, aarch64.
//!
//! Backed by `float32x4_t`. Mirrors the SSE2 f32x4 API exactly.
//!
//! NEON advantages over SSE2:
//!   - `vrsqrtsq_f32(x, r²)` = `(3-x·r²)/2` in ONE instruction (NR step)
//!   - `vrecpsq_f32(x, r)`   = `(2-x·r)` in ONE instruction (NR step)
//!   - `vabsq_f32`           — direct abs, no ANDNOT sign-mask trick
//!   - `vbslq_f32`           — bitselect in ONE instruction vs AND+ANDNOT+OR
//!   - Comparisons return `uint32x4_t` (Mask4's native type) directly

#![allow(non_camel_case_types)]

use core::arch::aarch64::*;
use core::fmt;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use super::mask4::Mask4;

#[repr(C)]
union UCast { f: [f32; 4], v: f32x4 }

/// 4-lane independent f32 scalar. 16 bytes, 16-byte aligned. Backed by `float32x4_t`.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct f32x4(pub(crate) float32x4_t);

// ── NEON fast-math helpers ────────────────────────────────────────────────────

/// Reciprocal square root: `1/sqrt(x)` per lane via `vrsqrteq_f32` + one NR step.
///
/// `vrsqrtsq_f32(x, r²)` = `(3 - x·r²) / 2` — dedicated NR-step instruction.
/// One NR step from the ~8-bit estimate reaches ~23-bit mantissa accuracy.
#[inline(always)]
pub(crate) unsafe fn rsqrt_nr(x: float32x4_t) -> float32x4_t {
    let r0   = vrsqrteq_f32(x);                   // ~8-bit estimate
    let r0sq = vmulq_f32(r0, r0);                  // r0²
    let step = vrsqrtsq_f32(x, r0sq);              // (3 - x·r0²) / 2  ← one instruction!
    vmulq_f32(r0, step)                             // r0 · (3-x·r0²)/2 = NR result
}

/// Reciprocal: `1/x` per lane via `vrecpeq_f32` + one NR step.
///
/// `vrecpsq_f32(x, r)` = `(2 - x·r)` — dedicated NR-step instruction.
#[inline(always)]
pub(crate) unsafe fn rcp_nr(x: float32x4_t) -> float32x4_t {
    let r0   = vrecpeq_f32(x);                     // ~8-bit estimate
    let step = vrecpsq_f32(x, r0);                 // (2 - x·r0)  ← one instruction!
    vmulq_f32(r0, step)                             // r0 · (2-x·r0) = NR result
}

impl f32x4 {
    // ── Constants ─────────────────────────────────────────────────────────────

    pub const ZERO:         Self = unsafe { UCast { f: [0.0; 4] }.v };
    pub const ONE:          Self = unsafe { UCast { f: [1.0; 4] }.v };
    pub const NEG_ONE:      Self = unsafe { UCast { f: [-1.0; 4] }.v };
    pub const INFINITY:     Self = unsafe { UCast { f: [f32::INFINITY; 4] }.v };
    pub const NEG_INFINITY: Self = unsafe { UCast { f: [f32::NEG_INFINITY; 4] }.v };

    // ── Constructors ──────────────────────────────────────────────────────────

    #[inline(always)] pub fn splat(v: f32) -> Self { Self(unsafe { vdupq_n_f32(v) }) }
    #[inline(always)] pub fn new(a: f32, b: f32, c: f32, d: f32) -> Self {
        unsafe { UCast { f: [a, b, c, d] }.v }
    }
    #[inline(always)]
    pub fn from_array(a: [f32; 4]) -> Self {
        Self(unsafe { vld1q_f32(a.as_ptr()) })
    }
    #[inline(always)]
    pub fn to_array(self) -> [f32; 4] {
        let mut a = [0.0f32; 4];
        unsafe { vst1q_f32(a.as_mut_ptr(), self.0) };
        a
    }
    #[inline]
    pub fn get(self, i: usize) -> f32 {
        assert!(i < 4);
        unsafe { UCast { v: self }.f[i] }
    }

    // ── Precise math ──────────────────────────────────────────────────────────

    /// Full-precision sqrt per lane. Single `FSQRT.4S` on AArch64.
    #[inline(always)]
    pub fn sqrt(self) -> Self { Self(unsafe { vsqrtq_f32(self.0) }) }

    // ── Fast approximate math ─────────────────────────────────────────────────

    /// Fast `1/sqrt(x)` via `vrsqrteq_f32` + Newton-Raphson. ~23-bit accuracy.
    #[inline(always)]
    pub fn recip_sqrt(self) -> Self { Self(unsafe { rsqrt_nr(self.0) }) }

    /// Fast `1/x` via `vrecpeq_f32` + Newton-Raphson. ~23-bit accuracy.
    #[inline(always)]
    pub fn recip(self) -> Self { Self(unsafe { rcp_nr(self.0) }) }

    // ── Component-wise arithmetic ─────────────────────────────────────────────

    /// Direct `vabsq_f32` — no sign-mask ANDNOT trick needed unlike SSE2.
    #[inline(always)]
    pub fn abs(self) -> Self { Self(unsafe { vabsq_f32(self.0) }) }

    #[inline(always)] pub fn min(self, r: Self) -> Self { Self(unsafe { vminq_f32(self.0, r.0) }) }
    #[inline(always)] pub fn max(self, r: Self) -> Self { Self(unsafe { vmaxq_f32(self.0, r.0) }) }
    #[inline(always)] pub fn clamp(self, lo: Self, hi: Self) -> Self { self.max(lo).min(hi) }

    #[inline]
    pub fn min_element(self) -> f32 { unsafe { vminvq_f32(self.0) } }
    #[inline]
    pub fn max_element(self) -> f32 { unsafe { vmaxvq_f32(self.0) } }

    /// Fused multiply-add: `self * b + c`. Mandatory FMA on AArch64.
    ///
    /// `vfmaq_f32(c, self, b)` = `c + self*b` in ONE instruction.
    #[inline(always)]
    pub fn mul_add(self, b: Self, c: Self) -> Self {
        Self(unsafe { vfmaq_f32(c.0, self.0, b.0) })
    }

    // ── Branchless select ─────────────────────────────────────────────────────

    /// Per-lane bitselect using `vbslq_f32` — ONE instruction vs SSE2's AND+ANDNOT+OR.
    ///
    /// `vbslq_f32(mask, if_true, if_false)`:
    ///   result[i] = if_true[i] where mask[i] = 0xFFFF_FFFF,
    ///               if_false[i] where mask[i] = 0x0000_0000.
    #[inline(always)]
    pub fn blend(mask: Mask4, if_true: Self, if_false: Self) -> Self {
        Self(unsafe { vbslq_f32(mask.0, if_true.0, if_false.0) })
    }

    // ── Comparisons → Mask4 ───────────────────────────────────────────────────
    // NEON comparison intrinsics return uint32x4_t directly — Mask4's native type.
    // No float-as-int reinterpretation needed (unlike SSE2).

    #[inline(always)] pub fn cmpeq(self, r: Self) -> Mask4 { Mask4(unsafe { vceqq_f32(self.0, r.0) }) }
    #[inline(always)] pub fn cmpne(self, r: Self) -> Mask4 { !self.cmpeq(r) }
    #[inline(always)] pub fn cmplt(self, r: Self) -> Mask4 { Mask4(unsafe { vcltq_f32(self.0, r.0) }) }
    #[inline(always)] pub fn cmple(self, r: Self) -> Mask4 { Mask4(unsafe { vcleq_f32(self.0, r.0) }) }
    #[inline(always)] pub fn cmpgt(self, r: Self) -> Mask4 { Mask4(unsafe { vcgtq_f32(self.0, r.0) }) }
    #[inline(always)] pub fn cmpge(self, r: Self) -> Mask4 { Mask4(unsafe { vcgeq_f32(self.0, r.0) }) }

    // ── Predicates ────────────────────────────────────────────────────────────

    #[inline]
    pub fn is_finite(self) -> bool { self.to_array().iter().all(|x| x.is_finite()) }
    #[inline]
    pub fn is_nan(self) -> bool {
        // NaN != NaN: all-zero mask where NaN
        unsafe { vminvq_u32(vceqq_f32(self.0, self.0)) == 0 }
    }
}

// ── Operators ─────────────────────────────────────────────────────────────────

impl Add for f32x4 {
    type Output = Self;
    #[inline(always)] fn add(self, r: Self) -> Self { Self(unsafe { vaddq_f32(self.0, r.0) }) }
}
impl AddAssign for f32x4 { #[inline(always)] fn add_assign(&mut self, r: Self) { self.0 = unsafe { vaddq_f32(self.0, r.0) }; } }
impl Sub for f32x4 {
    type Output = Self;
    #[inline(always)] fn sub(self, r: Self) -> Self { Self(unsafe { vsubq_f32(self.0, r.0) }) }
}
impl SubAssign for f32x4 { #[inline(always)] fn sub_assign(&mut self, r: Self) { self.0 = unsafe { vsubq_f32(self.0, r.0) }; } }
impl Mul for f32x4 {
    type Output = Self;
    #[inline(always)] fn mul(self, r: Self) -> Self { Self(unsafe { vmulq_f32(self.0, r.0) }) }
}
impl MulAssign for f32x4 { #[inline(always)] fn mul_assign(&mut self, r: Self) { self.0 = unsafe { vmulq_f32(self.0, r.0) }; } }
impl Div for f32x4 {
    type Output = Self;
    #[inline(always)] fn div(self, r: Self) -> Self { Self(unsafe { vdivq_f32(self.0, r.0) }) }
}
impl DivAssign for f32x4 { #[inline(always)] fn div_assign(&mut self, r: Self) { self.0 = unsafe { vdivq_f32(self.0, r.0) }; } }
impl Neg for f32x4 {
    type Output = Self;
    #[inline(always)] fn neg(self) -> Self { Self(unsafe { vnegq_f32(self.0) }) }
}

impl PartialEq for f32x4 {
    #[inline]
    fn eq(&self, r: &Self) -> bool {
        unsafe { vminvq_u32(vceqq_f32(self.0, r.0)) == u32::MAX }
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
