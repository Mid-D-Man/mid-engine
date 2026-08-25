// crates/mid-math/src/wide/int/wasm/i8x16.rs
//! 16-lane signed 8-bit integer vector — WASM SIMD128.
//!
//! Mirrors sse2/i8x16.rs. No multiply — same as SSE2, no native 8-bit
//! SIMD multiply on WASM either. `shuffle_bytes` uses `i8x16_swizzle`,
//! which (unlike x86's per-128-bit-lane `_mm256_shuffle_epi8`) IS a
//! true flat 16-byte shuffle — there's only one 128-bit lane on WASM's
//! v128, so the AVX2 cross-lane hazard documented in avx2/i8x32.rs
//! doesn't apply here at all.

#![allow(non_camel_case_types)]

use core::fmt;
use core::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign,
    BitXor, BitXorAssign, Neg, Not, Sub, SubAssign,
};

#[cfg(target_arch = "wasm32")]
use core::arch::wasm32::*;
#[cfg(target_arch = "wasm64")]
use core::arch::wasm64::*;

use super::imask16::IMask16;
use crate::wasm::v128_from_i8x16;

/// 16-lane signed 8-bit integer vector. Backed by `v128`.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct i8x16(pub(crate) v128);

impl i8x16 {
    pub const ZERO: Self = Self(v128_from_i8x16([0; 16]));
    pub const ONE:  Self = Self(v128_from_i8x16([1; 16]));
    pub const MIN:  Self = Self(v128_from_i8x16([i8::MIN; 16]));
    pub const MAX:  Self = Self(v128_from_i8x16([i8::MAX; 16]));

    #[inline(always)] pub fn splat(v: i8) -> Self { Self(i8x16_splat(v)) }
    #[inline(always)] pub fn from_array(a: [i8; 16]) -> Self { Self(v128_from_i8x16(a)) }
    #[inline(always)]
    pub fn from_bytes(b: [u8; 16]) -> Self {
        Self(v128_from_i8x16(core::array::from_fn(|i| b[i] as i8)))
    }

    #[inline(always)]
    pub fn to_array(self) -> [i8; 16] {
        core::array::from_fn(|i| match i {
            0=>i8x16_extract_lane::<0>(self.0), 1=>i8x16_extract_lane::<1>(self.0),
            2=>i8x16_extract_lane::<2>(self.0), 3=>i8x16_extract_lane::<3>(self.0),
            4=>i8x16_extract_lane::<4>(self.0), 5=>i8x16_extract_lane::<5>(self.0),
            6=>i8x16_extract_lane::<6>(self.0), 7=>i8x16_extract_lane::<7>(self.0),
            8=>i8x16_extract_lane::<8>(self.0), 9=>i8x16_extract_lane::<9>(self.0),
            10=>i8x16_extract_lane::<10>(self.0), 11=>i8x16_extract_lane::<11>(self.0),
            12=>i8x16_extract_lane::<12>(self.0), 13=>i8x16_extract_lane::<13>(self.0),
            14=>i8x16_extract_lane::<14>(self.0), _=>i8x16_extract_lane::<15>(self.0),
        })
    }
    #[inline(always)]
    pub fn to_bytes(self) -> [u8; 16] {
        let a = self.to_array();
        core::array::from_fn(|i| a[i] as u8)
    }

    #[inline]
    pub fn get(self, i: usize) -> i8 {
        assert!(i < 16, "i8x16::get — lane {i} out of bounds (max 15)");
        self.to_array()[i]
    }

    #[inline(always)] pub fn saturating_add(self, rhs: Self) -> Self { Self(i8x16_add_sat(self.0, rhs.0)) }
    #[inline(always)] pub fn saturating_sub(self, rhs: Self) -> Self { Self(i8x16_sub_sat(self.0, rhs.0)) }
    #[inline(always)] pub fn abs(self) -> Self { Self(i8x16_abs(self.0)) }
    #[inline(always)] pub fn min(self, rhs: Self) -> Self { Self(i8x16_min(self.0, rhs.0)) }
    #[inline(always)] pub fn max(self, rhs: Self) -> Self { Self(i8x16_max(self.0, rhs.0)) }
    #[inline(always)] pub fn clamp(self, lo: Self, hi: Self) -> Self { self.max(lo).min(hi) }

    #[inline] pub fn min_element(self) -> i8 { self.to_array().iter().copied().reduce(i8::min).unwrap() }
    #[inline] pub fn max_element(self) -> i8 { self.to_array().iter().copied().reduce(i8::max).unwrap() }
    #[inline] pub fn element_sum(self) -> i32 { self.to_array().iter().map(|&x| x as i32).sum() }

    #[inline(always)] pub fn cmpeq(self, rhs: Self) -> IMask16 { IMask16(i8x16_eq(self.0, rhs.0)) }
    #[inline(always)] pub fn cmpne(self, rhs: Self) -> IMask16 { IMask16(i8x16_ne(self.0, rhs.0)) }
    #[inline(always)] pub fn cmpgt(self, rhs: Self) -> IMask16 { IMask16(i8x16_gt(self.0, rhs.0)) }
    #[inline(always)] pub fn cmplt(self, rhs: Self) -> IMask16 { IMask16(i8x16_lt(self.0, rhs.0)) }
    #[inline(always)] pub fn cmpge(self, rhs: Self) -> IMask16 { IMask16(i8x16_ge(self.0, rhs.0)) }
    #[inline(always)] pub fn cmple(self, rhs: Self) -> IMask16 { IMask16(i8x16_le(self.0, rhs.0)) }

    #[inline(always)]
    pub fn blend(mask: IMask16, if_true: Self, if_false: Self) -> Self {
        Self(v128_bitselect(if_true.0, if_false.0, mask.0))
    }

    /// Number of lanes that compare equal to `needle`.
    #[inline] pub fn count_eq(self, needle: Self) -> u32 { self.cmpeq(needle).count_true() }
    /// True if any lane equals `needle`.
    #[inline] pub fn contains(self, needle: i8) -> bool { self.count_eq(Self::splat(needle)) > 0 }

    /// Full 16-byte shuffle. Native `i8x16_swizzle` — a true flat
    /// shuffle across all 16 lanes (see module docs: no per-128-bit-lane
    /// restriction here, unlike the AVX2 32-byte version). Index >= 16
    /// in `indices` zeroes that output byte (swizzle semantics, matches
    /// SSE2's `shuffle_bytes` high-bit-set behavior).
    #[inline(always)]
    pub fn shuffle_bytes(self, indices: Self) -> Self {
        Self(i8x16_swizzle(self.0, indices.0))
    }

    #[inline(always)] pub fn wrapping_add(self, r: Self) -> Self { self + r }
    #[inline(always)] pub fn wrapping_sub(self, r: Self) -> Self { self - r }
}

impl Add for i8x16 { type Output=Self; #[inline(always)] fn add(self,r:Self)->Self{Self(i8x16_add(self.0,r.0))} }
impl AddAssign for i8x16 { #[inline(always)] fn add_assign(&mut self,r:Self){*self=*self+r;} }
impl Sub for i8x16 { type Output=Self; #[inline(always)] fn sub(self,r:Self)->Self{Self(i8x16_sub(self.0,r.0))} }
impl SubAssign for i8x16 { #[inline(always)] fn sub_assign(&mut self,r:Self){*self=*self-r;} }
impl Neg for i8x16 { type Output=Self; #[inline(always)] fn neg(self)->Self{Self(i8x16_neg(self.0))} }
impl BitAnd for i8x16 { type Output=Self; #[inline(always)] fn bitand(self,r:Self)->Self{Self(v128_and(self.0,r.0))} }
impl BitAndAssign for i8x16 { #[inline(always)] fn bitand_assign(&mut self,r:Self){*self=*self&r;} }
impl BitOr  for i8x16 { type Output=Self; #[inline(always)] fn bitor (self,r:Self)->Self{Self(v128_or(self.0,r.0))} }
impl BitOrAssign  for i8x16 { #[inline(always)] fn bitor_assign (&mut self,r:Self){*self=*self|r;} }
impl BitXor for i8x16 { type Output=Self; #[inline(always)] fn bitxor(self,r:Self)->Self{Self(v128_xor(self.0,r.0))} }
impl BitXorAssign for i8x16 { #[inline(always)] fn bitxor_assign(&mut self,r:Self){*self=*self^r;} }
impl Not for i8x16 { type Output=Self; #[inline(always)] fn not(self)->Self{Self(v128_not(self.0))} }

impl PartialEq for i8x16 { fn eq(&self,r:&Self)->bool{i8x16_all_true(i8x16_eq(self.0,r.0))} }
impl Eq for i8x16 {}
impl fmt::Debug for i8x16 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "i8x16({:?})", self.to_array()) }
}
impl fmt::Display for i8x16 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "{:?}", self.to_array()) }
}
impl From<[i8; 16]> for i8x16 { fn from(a: [i8;16]) -> Self { Self::from_array(a) } }
impl From<i8x16> for [i8; 16] { fn from(v: i8x16) -> Self { v.to_array() } }
impl From<[u8; 16]> for i8x16 { fn from(b: [u8;16]) -> Self { Self::from_bytes(b) } }
impl From<i8x16> for [u8; 16] { fn from(v: i8x16) -> Self { v.to_bytes() } }
