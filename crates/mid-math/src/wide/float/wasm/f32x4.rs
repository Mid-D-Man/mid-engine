// crates/mid-math/src/wide/float/wasm/f32x4.rs
//! 4-lane f32 scalar — WASM SIMD128.
//!
//! No rsqrt/rcp hardware instructions on baseline simd128:
//!   - recip_sqrt: 1/sqrt via f32x4_sqrt + f32x4_div (Newton-Raphson variant)
//!   - recip:      f32x4_div(splat(1), x)
//! Relaxed-simd adds `f32x4_relaxed_madd` but we stay on baseline here.

#![allow(non_camel_case_types)]

use core::fmt;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use core::arch::wasm32::*;

use super::mask4::Mask4;

/// 4-lane independent f32 scalar backed by `v128`.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct f32x4(pub(crate) v128);

// ── Newton-Raphson helpers ────────────────────────────────────────────────────

/// Approximate reciprocal square root: 1/sqrt(x) via sqrt + div + one NR step.
///
/// WASM SIMD128 has no rsqrtps equivalent, so we use:
///   r = 1.0 / sqrt(x)          (full precision, then refine)
/// One NR step: r_new = 0.5 * r * (3 - x * r^2)
/// On WASM this is cheaper than it looks — LLVM maps sqrt to a single
/// `f32x4.sqrt` instruction; the NR step is 4 more.
#[inline(always)]
pub(crate) fn rsqrt_nr(x: v128) -> v128 {
    let sqrt = f32x4_sqrt(x);
    let one  = f32x4_splat(1.0);
    let r    = f32x4_div(one, sqrt);                    // r = 1/sqrt(x)
    // NR: r_new = 0.5 * r * (3 - x * r^2)
    let half  = f32x4_splat(0.5);
    let three = f32x4_splat(3.0);
    let xrr   = f32x4_mul(x, f32x4_mul(r, r));         // x * r²
    let nr    = f32x4_sub(three, xrr);                  // 3 - x*r²
    f32x4_mul(f32x4_mul(half, r), nr)                   // 0.5 * r * (3 - x*r²)
}

impl f32x4 {
    // ── Constants ─────────────────────────────────────────────────────────────

    pub const ZERO:         Self = Self(f32x4_splat_const::<0>());
    pub const ONE:          Self = Self(f32x4_splat_const::<0x3F800000>());
    pub const NEG_ONE:      Self = Self(f32x4_splat_const::<0xBF800000>());
    pub const INFINITY:     Self = Self(f32x4_splat_const::<0x7F800000>());
    pub const NEG_INFINITY: Self = Self(f32x4_splat_const::<0xFF800000>());

    // ── Constructors ──────────────────────────────────────────────────────────

    #[inline(always)]
    pub fn splat(v: f32) -> Self { Self(f32x4_splat(v)) }

    #[inline(always)]
    pub fn new(a: f32, b: f32, c: f32, d: f32) -> Self {
        Self(f32x4(a, b, c, d))
    }

    #[inline(always)]
    pub fn from_array(a: [f32; 4]) -> Self {
        // Safety: [f32;4] is 16 bytes, no alignment requirement for v128_load
        unsafe { Self(v128_load(a.as_ptr() as *const v128)) }
    }

    #[inline(always)]
    pub fn to_array(self) -> [f32; 4] {
        let mut a = [0.0f32; 4];
        unsafe { v128_store(a.as_mut_ptr() as *mut v128, self.0) };
        a
    }

    #[inline]
    pub fn get(self, i: usize) -> f32 {
        assert!(i < 4, "f32x4::get — lane {i} out of bounds (max 3)");
        self.to_array()[i]
    }

    // ── Precise math ──────────────────────────────────────────────────────────

    #[inline(always)]
    pub fn sqrt(self) -> Self { Self(f32x4_sqrt(self.0)) }

    // ── Fast approximate math ─────────────────────────────────────────────────

    /// Fast reciprocal square root. On WASM uses sqrt+div+NR (no rsqrtps).
    #[inline(always)]
    pub fn recip_sqrt(self) -> Self { Self(rsqrt_nr(self.0)) }

    /// Reciprocal: 1.0 / x per lane.
    #[inline(always)]
    pub fn recip(self) -> Self {
        Self(f32x4_div(f32x4_splat(1.0), self.0))
    }

    // ── Component-wise arithmetic ─────────────────────────────────────────────

    #[inline(always)]
    pub fn abs(self) -> Self {
        Self(f32x4_abs(self.0))
    }

    #[inline(always)]
    pub fn min(self, rhs: Self) -> Self { Self(f32x4_min(self.0, rhs.0)) }

    #[inline(always)]
    pub fn max(self, rhs: Self) -> Self { Self(f32x4_max(self.0, rhs.0)) }

    #[inline(always)]
    pub fn clamp(self, lo: Self, hi: Self) -> Self { self.max(lo).min(hi) }

    #[inline]
    pub fn min_element(self) -> f32 {
        let a = self.to_array();
        a[0].min(a[1]).min(a[2]).min(a[3])
    }

    #[inline]
    pub fn max_element(self) -> f32 {
        let a = self.to_array();
        a[0].max(a[1]).max(a[2]).max(a[3])
    }

    /// `self * b + c`. LLVM may fuse to f32x4_relaxed_madd on relaxed-simd targets.
    #[inline(always)]
    pub fn mul_add(self, b: Self, c: Self) -> Self {
        Self(f32x4_add(f32x4_mul(self.0, b.0), c.0))
    }

    // ── Branchless select ─────────────────────────────────────────────────────

    #[inline(always)]
    pub fn blend(mask: Mask4, if_true: Self, if_false: Self) -> Self {
        Self(v128_or(
            v128_and(mask.0, if_true.0),
            v128_andnot(if_false.0, mask.0),
        ))
    }

    // ── Comparisons → Mask4 ───────────────────────────────────────────────────

    #[inline(always)] pub fn cmpeq(self, r: Self) -> Mask4 { Mask4(f32x4_eq(self.0, r.0)) }
    #[inline(always)] pub fn cmpne(self, r: Self) -> Mask4 { Mask4(f32x4_ne(self.0, r.0)) }
    #[inline(always)] pub fn cmplt(self, r: Self) -> Mask4 { Mask4(f32x4_lt(self.0, r.0)) }
    #[inline(always)] pub fn cmple(self, r: Self) -> Mask4 { Mask4(f32x4_le(self.0, r.0)) }
    #[inline(always)] pub fn cmpgt(self, r: Self) -> Mask4 { Mask4(f32x4_gt(self.0, r.0)) }
    #[inline(always)] pub fn cmpge(self, r: Self) -> Mask4 { Mask4(f32x4_ge(self.0, r.0)) }

    // ── Predicates ────────────────────────────────────────────────────────────

    #[inline]
    pub fn is_finite(self) -> bool {
        let a = self.to_array();
        a.iter().all(|x| x.is_finite())
    }

    #[inline]
    pub fn is_nan(self) -> bool {
        // NaN != NaN
        !i32x4_all_true(f32x4_eq(self.0, self.0))
    }
}

// ── Const splat helper — produces a v128 constant from a bit pattern ──────────
// Used for associated ZERO / ONE / etc. without runtime init.

const fn f32x4_splat_const<const BITS: u32>() -> v128 {
    // transmute [u32;4] → v128 at compile time
    // SAFETY: v128 is a 16-byte type; [u32; 4] is 16 bytes with same alignment.
    unsafe { core::mem::transmute([BITS; 4]) }
}

// ── Operators ─────────────────────────────────────────────────────────────────

impl Add for f32x4 {
    type Output = Self;
    #[inline(always)] fn add(self, r: Self) -> Self { Self(f32x4_add(self.0, r.0)) }
}
impl AddAssign for f32x4 { #[inline(always)] fn add_assign(&mut self, r: Self) { *self = *self + r; } }

impl Sub for f32x4 {
    type Output = Self;
    #[inline(always)] fn sub(self, r: Self) -> Self { Self(f32x4_sub(self.0, r.0)) }
}
impl SubAssign for f32x4 { #[inline(always)] fn sub_assign(&mut self, r: Self) { *self = *self - r; } }

impl Mul for f32x4 {
    type Output = Self;
    #[inline(always)] fn mul(self, r: Self) -> Self { Self(f32x4_mul(self.0, r.0)) }
}
impl MulAssign for f32x4 { #[inline(always)] fn mul_assign(&mut self, r: Self) { *self = *self * r; } }

impl Div for f32x4 {
    type Output = Self;
    #[inline(always)] fn div(self, r: Self) -> Self { Self(f32x4_div(self.0, r.0)) }
}
impl DivAssign for f32x4 { #[inline(always)] fn div_assign(&mut self, r: Self) { *self = *self / r; } }

impl Neg for f32x4 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self { Self(f32x4_neg(self.0)) }
}

impl PartialEq for f32x4 {
    #[inline]
    fn eq(&self, r: &Self) -> bool {
        i32x4_all_true(f32x4_eq(self.0, r.0))
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

impl From<[f32; 4]> for f32x4 { #[inline] fn from(a: [f32; 4]) -> Self { Self::from_array(a) } }
impl From<f32x4> for [f32; 4]  { #[inline] fn from(v: f32x4)   -> Self { v.to_array() } }
impl From<f32>   for f32x4     { #[inline] fn from(v: f32)     -> Self { Self::splat(v) } }
