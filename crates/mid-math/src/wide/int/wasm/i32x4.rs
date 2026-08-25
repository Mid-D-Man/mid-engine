// crates/mid-math/src/wide/int/wasm/i32x4.rs
//! 4-lane signed 32-bit integer vector — WASM SIMD128.
//!
//! Mirrors sse2/i32x4.rs. WASM has native `i32x4_mul` (SSE2 needs a
//! shuffle/unpack chain — see sse2/i32x4.rs) and native `i32x4_min`/
//! `i32x4_max`/`u32x4_min`/`u32x4_max` (SSE2 needs cmp+blend emulation
//! for signed, and the XOR-flip trick for unsigned). No native 32-bit
//! saturating add/sub in WASM either (same gap as SSE2/AVX2) — scalar loop.
//!
//! All function names verified directly against stdarch's
//! `core_arch/src/wasm32/simd128.rs` source, not recalled from memory.

#![allow(non_camel_case_types)]

use core::fmt;
use core::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign,
    BitXor, BitXorAssign, Mul, MulAssign, Neg, Not, Sub, SubAssign,
};

#[cfg(target_arch = "wasm32")]
use core::arch::wasm32::*;
#[cfg(target_arch = "wasm64")]
use core::arch::wasm64::*;

use super::imask4::IMask4;
use crate::wasm::v128_from_i32x4;

/// 4-lane signed 32-bit integer vector. Backed by `v128`.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct i32x4(pub(crate) v128);

impl i32x4 {
    pub const ZERO: Self = Self(v128_from_i32x4([0; 4]));
    pub const ONE:  Self = Self(v128_from_i32x4([1; 4]));
    pub const MIN:  Self = Self(v128_from_i32x4([i32::MIN; 4]));
    pub const MAX:  Self = Self(v128_from_i32x4([i32::MAX; 4]));

    #[inline(always)] pub fn splat(v: i32) -> Self { Self(i32x4_splat(v)) }
    #[inline(always)] pub fn new(a: i32, b: i32, c: i32, d: i32) -> Self { Self(v128_from_i32x4([a, b, c, d])) }
    #[inline(always)] pub fn from_array(a: [i32; 4]) -> Self { Self(v128_from_i32x4(a)) }

    #[inline(always)]
    pub fn to_array(self) -> [i32; 4] {
        [
            i32x4_extract_lane::<0>(self.0),
            i32x4_extract_lane::<1>(self.0),
            i32x4_extract_lane::<2>(self.0),
            i32x4_extract_lane::<3>(self.0),
        ]
    }

    #[inline]
    pub fn get(self, i: usize) -> i32 {
        match i {
            0 => i32x4_extract_lane::<0>(self.0),
            1 => i32x4_extract_lane::<1>(self.0),
            2 => i32x4_extract_lane::<2>(self.0),
            3 => i32x4_extract_lane::<3>(self.0),
            _ => panic!("i32x4::get — lane {i} out of bounds (max 3)"),
        }
    }

    /// Absolute value per lane. Native `i32x4_abs`.
    #[inline(always)] pub fn abs(self) -> Self { Self(i32x4_abs(self.0)) }

    /// Per-lane minimum. Native `i32x4_min` — SSE2 needs cmp+blend.
    #[inline(always)] pub fn min(self, rhs: Self) -> Self { Self(i32x4_min(self.0, rhs.0)) }
    /// Per-lane maximum. Native `i32x4_max`.
    #[inline(always)] pub fn max(self, rhs: Self) -> Self { Self(i32x4_max(self.0, rhs.0)) }
    #[inline(always)] pub fn clamp(self, lo: Self, hi: Self) -> Self { self.max(lo).min(hi) }

    #[inline] pub fn min_element(self) -> i32 { self.to_array().iter().copied().reduce(i32::min).unwrap() }
    #[inline] pub fn max_element(self) -> i32 { self.to_array().iter().copied().reduce(i32::max).unwrap() }
    #[inline] pub fn element_sum(self) -> i32 { self.to_array().iter().fold(0i32, |a, &x| a.wrapping_add(x)) }

    #[inline(always)] pub fn shl(self, count: u32) -> Self { Self(i32x4_shl(self.0, count)) }
    #[inline(always)] pub fn shr_arithmetic(self, count: u32) -> Self { Self(i32x4_shr(self.0, count)) }
    #[inline(always)] pub fn shr_logical(self, count: u32) -> Self { Self(u32x4_shr(self.0, count)) }

    #[inline(always)] pub fn cmpeq(self, rhs: Self) -> IMask4 { IMask4(i32x4_eq(self.0, rhs.0)) }
    #[inline(always)] pub fn cmpne(self, rhs: Self) -> IMask4 { IMask4(i32x4_ne(self.0, rhs.0)) }
    #[inline(always)] pub fn cmpgt(self, rhs: Self) -> IMask4 { IMask4(i32x4_gt(self.0, rhs.0)) }
    #[inline(always)] pub fn cmplt(self, rhs: Self) -> IMask4 { IMask4(i32x4_lt(self.0, rhs.0)) }
    #[inline(always)] pub fn cmpge(self, rhs: Self) -> IMask4 { IMask4(i32x4_ge(self.0, rhs.0)) }
    #[inline(always)] pub fn cmple(self, rhs: Self) -> IMask4 { IMask4(i32x4_le(self.0, rhs.0)) }

    /// Branchless select. Native `v128_bitselect` — one instruction,
    /// vs SSE2's manual and/andnot/or chain.
    #[inline(always)]
    pub fn blend(mask: IMask4, if_true: Self, if_false: Self) -> Self {
        Self(v128_bitselect(if_true.0, if_false.0, mask.0))
    }

    #[inline(always)] pub fn wrapping_add(self, r: Self) -> Self { self + r }
    #[inline(always)] pub fn wrapping_sub(self, r: Self) -> Self { self - r }
    #[inline(always)] pub fn wrapping_mul(self, r: Self) -> Self { self * r }

    /// No native 32-bit saturating add in WASM SIMD128 — scalar loop
    /// (same gap as SSE2/AVX2).
    #[inline]
    pub fn saturating_add(self, rhs: Self) -> Self {
        let a = self.to_array();
        let b = rhs.to_array();
        Self::from_array(core::array::from_fn(|i| a[i].saturating_add(b[i])))
    }
    #[inline]
    pub fn saturating_sub(self, rhs: Self) -> Self {
        let a = self.to_array();
        let b = rhs.to_array();
        Self::from_array(core::array::from_fn(|i| a[i].saturating_sub(b[i])))
    }
}

impl Add for i32x4 { type Output=Self; #[inline(always)] fn add(self,r:Self)->Self{Self(i32x4_add(self.0,r.0))} }
impl AddAssign for i32x4 { #[inline(always)] fn add_assign(&mut self,r:Self){*self=*self+r;} }
impl Sub for i32x4 { type Output=Self; #[inline(always)] fn sub(self,r:Self)->Self{Self(i32x4_sub(self.0,r.0))} }
impl SubAssign for i32x4 { #[inline(always)] fn sub_assign(&mut self,r:Self){*self=*self-r;} }
impl Neg for i32x4 { type Output=Self; #[inline(always)] fn neg(self)->Self{Self(i32x4_neg(self.0))} }

/// Native `i32x4_mul` — WASM SIMD128 has 32-bit multiply baseline, no
/// shuffle/unpack emulation needed (unlike SSE2).
impl Mul for i32x4 { type Output=Self; #[inline(always)] fn mul(self,r:Self)->Self{Self(i32x4_mul(self.0,r.0))} }
impl MulAssign for i32x4 { #[inline(always)] fn mul_assign(&mut self,r:Self){*self=*self*r;} }

impl BitAnd for i32x4 { type Output=Self; #[inline(always)] fn bitand(self,r:Self)->Self{Self(v128_and(self.0,r.0))} }
impl BitAndAssign for i32x4 { #[inline(always)] fn bitand_assign(&mut self,r:Self){*self=*self&r;} }
impl BitOr  for i32x4 { type Output=Self; #[inline(always)] fn bitor (self,r:Self)->Self{Self(v128_or(self.0,r.0))} }
impl BitOrAssign  for i32x4 { #[inline(always)] fn bitor_assign (&mut self,r:Self){*self=*self|r;} }
impl BitXor for i32x4 { type Output=Self; #[inline(always)] fn bitxor(self,r:Self)->Self{Self(v128_xor(self.0,r.0))} }
impl BitXorAssign for i32x4 { #[inline(always)] fn bitxor_assign(&mut self,r:Self){*self=*self^r;} }
impl Not for i32x4 { type Output=Self; #[inline(always)] fn not(self)->Self{Self(v128_not(self.0))} }

impl PartialEq for i32x4 { #[inline] fn eq(&self,r:&Self)->bool{i32x4_all_true(i32x4_eq(self.0,r.0))} }
impl Eq for i32x4 {}

impl fmt::Debug for i32x4 {
    fn fmt(&self,f:&mut fmt::Formatter<'_>)->fmt::Result{let a=self.to_array();write!(f,"i32x4({},{},{},{})",a[0],a[1],a[2],a[3])}
}
impl fmt::Display for i32x4 {
    fn fmt(&self,f:&mut fmt::Formatter<'_>)->fmt::Result{let a=self.to_array();write!(f,"[{},{},{},{}]",a[0],a[1],a[2],a[3])}
}
impl From<[i32;4]> for i32x4 { #[inline] fn from(a:[i32;4])->Self{Self::from_array(a)} }
impl From<i32x4> for [i32;4] { #[inline] fn from(v:i32x4)->[i32;4]{v.to_array()} }
