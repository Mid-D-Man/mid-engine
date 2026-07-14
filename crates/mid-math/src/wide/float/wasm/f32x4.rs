// crates/mid-math/src/wide/float/wasm/f32x4.rs
//! 4-lane f32 scalar — WASM SIMD128.
//!
//! No hardware rsqrt/rcp: recip_sqrt uses sqrt+div+NR (~5 instructions).
//! recip uses f32x4_div(splat(1), x).

#![allow(non_camel_case_types)]

use core::fmt;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

// Import specific wasm intrinsics — deliberately omitting the `f32x4` constructor
// function to avoid a name clash with our f32x4 struct.
#[cfg(target_arch = "wasm32")]
use core::arch::wasm32::{
    v128,
    v128_and, v128_or, v128_andnot,
    i32x4_all_true, i32x4_bitmask,
    f32x4_add, f32x4_sub, f32x4_mul, f32x4_div,
    f32x4_neg, f32x4_abs, f32x4_sqrt,
    f32x4_min, f32x4_max,
    f32x4_eq, f32x4_ne, f32x4_lt, f32x4_le, f32x4_gt, f32x4_ge,
    f32x4_splat, f32x4_extract_lane,
};
#[cfg(target_arch = "wasm64")]
use core::arch::wasm64::{
    v128,
    v128_and, v128_or, v128_andnot,
    i32x4_all_true, i32x4_bitmask,
    f32x4_add, f32x4_sub, f32x4_mul, f32x4_div,
    f32x4_neg, f32x4_abs, f32x4_sqrt,
    f32x4_min, f32x4_max,
    f32x4_eq, f32x4_ne, f32x4_lt, f32x4_le, f32x4_gt, f32x4_ge,
    f32x4_splat, f32x4_extract_lane,
};

use crate::wasm::v128_from_f32x4;
use super::mask4::Mask4;

/// 4-lane independent f32 scalar backed by `v128`.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct f32x4(pub(crate) v128);

// ── Newton-Raphson rsqrt ──────────────────────────────────────────────────────
// WASM has no rsqrtps equivalent. We compute 1/sqrt(x) then refine with NR.
// r0 = 1.0 / sqrt(x)  (full precision sqrt + div as starting point)
// r1 = 0.5 * r0 * (3 - x * r0^2)  ← one NR step, ~4 extra instructions
// This matches SSE2 rsqrtps+NR accuracy (~23-bit mantissa).

#[inline(always)]
pub(crate) fn rsqrt_nr(x: v128) -> v128 {
    let r     = f32x4_div(f32x4_splat(1.0), f32x4_sqrt(x));
    let half  = f32x4_splat(0.5);
    let three = f32x4_splat(3.0);
    let xrr   = f32x4_mul(x, f32x4_mul(r, r));
    let nr    = f32x4_sub(three, xrr);
    f32x4_mul(f32x4_mul(half, r), nr)
}

impl f32x4 {
    // ── Constants ─────────────────────────────────────────────────────────────

    pub const ZERO:         Self = Self(v128_from_f32x4([0.0; 4]));
    pub const ONE:          Self = Self(v128_from_f32x4([1.0; 4]));
    pub const NEG_ONE:      Self = Self(v128_from_f32x4([-1.0; 4]));
    pub const INFINITY:     Self = Self(v128_from_f32x4([f32::INFINITY; 4]));
    pub const NEG_INFINITY: Self = Self(v128_from_f32x4([f32::NEG_INFINITY; 4]));

    // ── Constructors ──────────────────────────────────────────────────────────

    #[inline(always)]
    pub fn splat(v: f32) -> Self { Self(f32x4_splat(v)) }

    #[inline(always)]
    pub fn new(a: f32, b: f32, c: f32, d: f32) -> Self {
        Self::from_array([a, b, c, d])
    }

    #[inline(always)]
    pub fn from_array(a: [f32; 4]) -> Self {
        Self(v128_from_f32x4(a))
    }

    #[inline(always)]
    pub fn to_array(self) -> [f32; 4] {
        [
            f32x4_extract_lane::<0>(self.0), f32x4_extract_lane::<1>(self.0),
            f32x4_extract_lane::<2>(self.0), f32x4_extract_lane::<3>(self.0),
        ]
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

    /// Reciprocal square root: 1/sqrt(x). Uses sqrt+div+NR (~23-bit accuracy).
    #[inline(always)]
    pub fn recip_sqrt(self) -> Self { Self(rsqrt_nr(self.0)) }

    /// Reciprocal: 1/x per lane.
    #[inline(always)]
    pub fn recip(self) -> Self {
        Self(f32x4_div(f32x4_splat(1.0), self.0))
    }

    // ── Component-wise arithmetic ─────────────────────────────────────────────

    #[inline(always)] pub fn abs(self) -> Self  { Self(f32x4_abs(self.0)) }
    #[inline(always)] pub fn min(self, r: Self) -> Self { Self(f32x4_min(self.0, r.0)) }
    #[inline(always)] pub fn max(self, r: Self) -> Self { Self(f32x4_max(self.0, r.0)) }
    #[inline(always)] pub fn clamp(self, lo: Self, hi: Self) -> Self { self.max(lo).min(hi) }

    #[inline] pub fn min_element(self) -> f32 {
        let a = self.to_array(); a[0].min(a[1]).min(a[2]).min(a[3])
    }
    #[inline] pub fn max_element(self) -> f32 {
        let a = self.to_array(); a[0].max(a[1]).max(a[2]).max(a[3])
    }

    /// `self * b + c`. LLVM may fuse to f32x4_relaxed_madd on relaxed-simd targets.
    #[inline(always)]
    pub fn mul_add(self, b: Self, c: Self) -> Self {
        Self(f32x4_add(f32x4_mul(self.0, b.0), c.0))
    }

    // ── Branchless select ─────────────────────────────────────────────────────

    /// WASM: v128_andnot(a, b) = a & !b  (opposite arg order from SSE2 _mm_andnot_ps)
    #[inline(always)]
    pub fn blend(mask: Mask4, if_true: Self, if_false: Self) -> Self {
        Self(v128_or(
            v128_and(mask.0, if_true.0),
            v128_andnot(if_false.0, mask.0),   // if_false & !mask
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

    #[inline] pub fn is_finite(self) -> bool {
        self.to_array().iter().all(|x| x.is_finite())
    }
    #[inline] pub fn is_nan(self) -> bool {
        // NaN != NaN
        !i32x4_all_true(f32x4_eq(self.0, self.0))
    }
}

// ── Operators ─────────────────────────────────────────────────────────────────

impl Add  for f32x4 { type Output=Self; #[inline(always)] fn add(self,r:Self)->Self { Self(f32x4_add(self.0,r.0)) } }
impl Sub  for f32x4 { type Output=Self; #[inline(always)] fn sub(self,r:Self)->Self { Self(f32x4_sub(self.0,r.0)) } }
impl Mul  for f32x4 { type Output=Self; #[inline(always)] fn mul(self,r:Self)->Self { Self(f32x4_mul(self.0,r.0)) } }
impl Div  for f32x4 { type Output=Self; #[inline(always)] fn div(self,r:Self)->Self { Self(f32x4_div(self.0,r.0)) } }
impl Neg  for f32x4 { type Output=Self; #[inline(always)] fn neg(self)       ->Self { Self(f32x4_neg(self.0)) } }

impl AddAssign for f32x4 { #[inline(always)] fn add_assign(&mut self,r:Self){*self=*self+r;} }
impl SubAssign for f32x4 { #[inline(always)] fn sub_assign(&mut self,r:Self){*self=*self-r;} }
impl MulAssign for f32x4 { #[inline(always)] fn mul_assign(&mut self,r:Self){*self=*self*r;} }
impl DivAssign for f32x4 { #[inline(always)] fn div_assign(&mut self,r:Self){*self=*self/r;} }

impl PartialEq for f32x4 {
    #[inline] fn eq(&self, r: &Self) -> bool { i32x4_all_true(f32x4_eq(self.0, r.0)) }
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

impl From<[f32; 4]> for f32x4 { #[inline] fn from(a:[f32;4])->Self { Self::from_array(a) } }
impl From<f32x4> for [f32; 4]  { #[inline] fn from(v:f32x4) ->Self { v.to_array() } }
impl From<f32>   for f32x4     { #[inline] fn from(v:f32)   ->Self { Self::splat(v) } }
