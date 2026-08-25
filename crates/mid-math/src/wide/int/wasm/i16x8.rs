// crates/mid-math/src/wide/int/wasm/i16x8.rs
//! 8-lane signed 16-bit integer vector — WASM SIMD128.
//!
//! Mirrors sse2/i16x8.rs. WASM has native `i16x8_mul` (same as SSE2's
//! `mullo` — 16-bit multiply is baseline everywhere) and native
//! `i16x8_add_sat`/`i16x8_sub_sat` (also matches SSE2). `as_i32x4_lo`/
//! `as_i32x4_hi` use the dedicated `i16x8_extend_low_i8x16`... no —
//! `i32x4_extend_low_i16x8`/`i32x4_extend_high_i16x8` widen instructions,
//! which take ONE v128 argument each (the whole register) and return
//! the correct half directly — no separate extract-then-convert step
//! needed, unlike x86's cvtepi16_epi32 (which needs a 128-bit half
//! extracted first on AVX2, or a full-register unpack trick on SSE2).

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

use super::imask8::IMask8;
use crate::wasm::v128_from_i16x8;

/// 8-lane signed 16-bit integer vector. Backed by `v128`.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct i16x8(pub(crate) v128);

impl i16x8 {
    pub const ZERO: Self = Self(v128_from_i16x8([0; 8]));
    pub const ONE:  Self = Self(v128_from_i16x8([1; 8]));
    pub const MIN:  Self = Self(v128_from_i16x8([i16::MIN; 8]));
    pub const MAX:  Self = Self(v128_from_i16x8([i16::MAX; 8]));

    #[inline(always)] pub fn splat(v: i16) -> Self { Self(i16x8_splat(v)) }
    #[inline(always)]
    pub fn new(a:i16,b:i16,c:i16,d:i16,e:i16,f:i16,g:i16,h:i16) -> Self {
        Self(v128_from_i16x8([a,b,c,d,e,f,g,h]))
    }
    #[inline(always)] pub fn from_array(a: [i16; 8]) -> Self { Self(v128_from_i16x8(a)) }

    #[inline(always)]
    pub fn to_array(self) -> [i16; 8] {
        [
            i16x8_extract_lane::<0>(self.0), i16x8_extract_lane::<1>(self.0),
            i16x8_extract_lane::<2>(self.0), i16x8_extract_lane::<3>(self.0),
            i16x8_extract_lane::<4>(self.0), i16x8_extract_lane::<5>(self.0),
            i16x8_extract_lane::<6>(self.0), i16x8_extract_lane::<7>(self.0),
        ]
    }

    #[inline]
    pub fn get(self, i: usize) -> i16 {
        assert!(i < 8, "i16x8::get — lane {i} out of bounds (max 7)");
        self.to_array()[i]
    }

    /// Sign-extend the low 4 lanes to `i32x4`. Native `i32x4_extend_low_i16x8`.
    #[inline(always)]
    pub fn as_i32x4_lo(self) -> super::i32x4::i32x4 {
        super::i32x4::i32x4(i32x4_extend_low_i16x8(self.0))
    }
    /// Sign-extend the high 4 lanes to `i32x4`. Native `i32x4_extend_high_i16x8`.
    #[inline(always)]
    pub fn as_i32x4_hi(self) -> super::i32x4::i32x4 {
        super::i32x4::i32x4(i32x4_extend_high_i16x8(self.0))
    }

    #[inline(always)] pub fn abs(self) -> Self { Self(i16x8_abs(self.0)) }
    #[inline(always)] pub fn min(self, rhs: Self) -> Self { Self(i16x8_min(self.0, rhs.0)) }
    #[inline(always)] pub fn max(self, rhs: Self) -> Self { Self(i16x8_max(self.0, rhs.0)) }
    #[inline(always)] pub fn clamp(self, lo: Self, hi: Self) -> Self { self.max(lo).min(hi) }

    #[inline] pub fn min_element(self) -> i16 { self.to_array().iter().copied().reduce(i16::min).unwrap() }
    #[inline] pub fn max_element(self) -> i16 { self.to_array().iter().copied().reduce(i16::max).unwrap() }
    #[inline] pub fn element_sum(self) -> i32 { self.to_array().iter().map(|&x| x as i32).sum() }

    /// Native `i16x8_mul` — full lane result (16x16→16, wrapping), no
    /// separate "low"/"high" split needed the way SSE2's mullo/mulhi is;
    /// kept as two methods for API parity with sse2/i16x8.rs anyway.
    #[inline(always)] pub fn mul_lo(self, rhs: Self) -> Self { Self(i16x8_mul(self.0, rhs.0)) }
    /// No native widening 16x16→32-high-bits instruction in WASM
    /// SIMD128 (unlike SSE2's `_mm_mulhi_epi16`) — computed via the
    /// widen-to-i32x4 path instead.
    #[inline]
    pub fn mul_high(self, rhs: Self) -> Self {
        let (alo, ahi) = (self.as_i32x4_lo(), self.as_i32x4_hi());
        let (blo, bhi) = (rhs.as_i32x4_lo(), rhs.as_i32x4_hi());
        let lo = (alo * blo).shr_arithmetic(16).to_array();
        let hi = (ahi * bhi).shr_arithmetic(16).to_array();
        Self::from_array([
            lo[0] as i16, lo[1] as i16, lo[2] as i16, lo[3] as i16,
            hi[0] as i16, hi[1] as i16, hi[2] as i16, hi[3] as i16,
        ])
    }

    #[inline(always)] pub fn saturating_add(self, rhs: Self) -> Self { Self(i16x8_add_sat(self.0, rhs.0)) }
    #[inline(always)] pub fn saturating_sub(self, rhs: Self) -> Self { Self(i16x8_sub_sat(self.0, rhs.0)) }

    #[inline(always)] pub fn shl(self, count: u32) -> Self { Self(i16x8_shl(self.0, count)) }
    #[inline(always)] pub fn shr_arithmetic(self, count: u32) -> Self { Self(i16x8_shr(self.0, count)) }
    #[inline(always)] pub fn shr_logical(self, count: u32) -> Self { Self(u16x8_shr(self.0, count)) }

    #[inline(always)] pub fn cmpeq(self, rhs: Self) -> IMask8 { IMask8(i16x8_eq(self.0, rhs.0)) }
    #[inline(always)] pub fn cmpne(self, rhs: Self) -> IMask8 { IMask8(i16x8_ne(self.0, rhs.0)) }
    #[inline(always)] pub fn cmpgt(self, rhs: Self) -> IMask8 { IMask8(i16x8_gt(self.0, rhs.0)) }
    #[inline(always)] pub fn cmplt(self, rhs: Self) -> IMask8 { IMask8(i16x8_lt(self.0, rhs.0)) }
    #[inline(always)] pub fn cmpge(self, rhs: Self) -> IMask8 { IMask8(i16x8_ge(self.0, rhs.0)) }
    #[inline(always)] pub fn cmple(self, rhs: Self) -> IMask8 { IMask8(i16x8_le(self.0, rhs.0)) }

    #[inline(always)]
    pub fn blend(mask: IMask8, if_true: Self, if_false: Self) -> Self {
        Self(v128_bitselect(if_true.0, if_false.0, mask.0))
    }

    #[inline(always)] pub fn wrapping_add(self, r: Self) -> Self { self + r }
    #[inline(always)] pub fn wrapping_sub(self, r: Self) -> Self { self - r }
    #[inline(always)] pub fn wrapping_mul(self, r: Self) -> Self { self.mul_lo(r) }
}

impl Add for i16x8 { type Output=Self; #[inline(always)] fn add(self,r:Self)->Self{Self(i16x8_add(self.0,r.0))} }
impl AddAssign for i16x8 { #[inline(always)] fn add_assign(&mut self,r:Self){*self=*self+r;} }
impl Sub for i16x8 { type Output=Self; #[inline(always)] fn sub(self,r:Self)->Self{Self(i16x8_sub(self.0,r.0))} }
impl SubAssign for i16x8 { #[inline(always)] fn sub_assign(&mut self,r:Self){*self=*self-r;} }
impl Neg for i16x8 { type Output=Self; #[inline(always)] fn neg(self)->Self{Self(i16x8_neg(self.0))} }
impl Mul for i16x8 { type Output=Self; #[inline(always)] fn mul(self,r:Self)->Self{self.mul_lo(r)} }
impl MulAssign for i16x8 { #[inline(always)] fn mul_assign(&mut self,r:Self){*self=*self*r;} }
impl BitAnd for i16x8 { type Output=Self; #[inline(always)] fn bitand(self,r:Self)->Self{Self(v128_and(self.0,r.0))} }
impl BitAndAssign for i16x8 { #[inline(always)] fn bitand_assign(&mut self,r:Self){*self=*self&r;} }
impl BitOr  for i16x8 { type Output=Self; #[inline(always)] fn bitor (self,r:Self)->Self{Self(v128_or(self.0,r.0))} }
impl BitOrAssign  for i16x8 { #[inline(always)] fn bitor_assign (&mut self,r:Self){*self=*self|r;} }
impl BitXor for i16x8 { type Output=Self; #[inline(always)] fn bitxor(self,r:Self)->Self{Self(v128_xor(self.0,r.0))} }
impl BitXorAssign for i16x8 { #[inline(always)] fn bitxor_assign(&mut self,r:Self){*self=*self^r;} }
impl Not for i16x8 { type Output=Self; #[inline(always)] fn not(self)->Self{Self(v128_not(self.0))} }

impl PartialEq for i16x8 { fn eq(&self,r:&Self)->bool{i16x8_all_true(i16x8_eq(self.0,r.0))} }
impl Eq for i16x8 {}
impl fmt::Debug for i16x8 {
    fn fmt(&self,f:&mut fmt::Formatter<'_>)->fmt::Result{
        let a=self.to_array();write!(f,"i16x8({},{},{},{},{},{},{},{})",a[0],a[1],a[2],a[3],a[4],a[5],a[6],a[7])
    }
}
impl fmt::Display for i16x8 {
    fn fmt(&self,f:&mut fmt::Formatter<'_>)->fmt::Result{
        let a=self.to_array();write!(f,"[{},{},{},{},{},{},{},{}]",a[0],a[1],a[2],a[3],a[4],a[5],a[6],a[7])
    }
}
impl From<[i16;8]> for i16x8 { fn from(a:[i16;8])->Self{Self::from_array(a)} }
impl From<i16x8> for [i16;8] { fn from(v:i16x8)->[i16;8]{v.to_array()} }
