// crates/mid-math/src/wide/int/wasm/u8x16.rs
//! 16-lane unsigned 8-bit integer vector — WASM SIMD128.
//!
//! Mirrors sse2/u8x16.rs. Native `u8x16_min`/`u8x16_max` (real unsigned
//! instructions). `element_sum` — WASM has no direct SAD-style
//! horizontal-sum-of-bytes instruction like x86's `psadbw`, so this
//! stays a widen-and-fold loop rather than reaching for a single
//! intrinsic (unlike sse2/u8x16.rs's `_mm_sad_epu8` trick).

#![allow(non_camel_case_types)]

use core::fmt;
use core::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign,
    BitXor, BitXorAssign, Not, Sub, SubAssign,
};

#[cfg(target_arch = "wasm32")]
use core::arch::wasm32::*;
#[cfg(target_arch = "wasm64")]
use core::arch::wasm64::*;

use super::imask16::IMask16;
use crate::wasm::v128_from_i8x16;

/// 16-lane unsigned 8-bit integer vector. Backed by `v128`.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct u8x16(pub(crate) v128);

impl u8x16 {
    pub const ZERO: Self = Self(v128_from_i8x16([0; 16]));
    pub const ONE:  Self = Self(v128_from_i8x16([1; 16]));
    pub const MIN:  Self = Self(v128_from_i8x16([0; 16]));
    pub const MAX:  Self = Self(v128_from_i8x16([u8::MAX as i8; 16]));

    #[inline(always)] pub fn splat(v: u8) -> Self { Self(u8x16_splat(v)) }
    #[inline(always)]
    pub fn from_array(a: [u8; 16]) -> Self {
        Self(v128_from_i8x16(core::array::from_fn(|i| a[i] as i8)))
    }
    #[inline(always)] pub fn from_bytes(b: [u8; 16]) -> Self { Self::from_array(b) }

    #[inline(always)]
    pub fn to_array(self) -> [u8; 16] {
        core::array::from_fn(|i| match i {
            0=>u8x16_extract_lane::<0>(self.0), 1=>u8x16_extract_lane::<1>(self.0),
            2=>u8x16_extract_lane::<2>(self.0), 3=>u8x16_extract_lane::<3>(self.0),
            4=>u8x16_extract_lane::<4>(self.0), 5=>u8x16_extract_lane::<5>(self.0),
            6=>u8x16_extract_lane::<6>(self.0), 7=>u8x16_extract_lane::<7>(self.0),
            8=>u8x16_extract_lane::<8>(self.0), 9=>u8x16_extract_lane::<9>(self.0),
            10=>u8x16_extract_lane::<10>(self.0), 11=>u8x16_extract_lane::<11>(self.0),
            12=>u8x16_extract_lane::<12>(self.0), 13=>u8x16_extract_lane::<13>(self.0),
            14=>u8x16_extract_lane::<14>(self.0), _=>u8x16_extract_lane::<15>(self.0),
        })
    }

    #[inline]
    pub fn get(self, i: usize) -> u8 {
        assert!(i < 16, "u8x16::get — lane {i} out of bounds (max 15)");
        self.to_array()[i]
    }

    #[inline(always)] pub fn min(self, rhs: Self) -> Self { Self(u8x16_min(self.0, rhs.0)) }
    #[inline(always)] pub fn max(self, rhs: Self) -> Self { Self(u8x16_max(self.0, rhs.0)) }
    #[inline(always)] pub fn clamp(self, lo: Self, hi: Self) -> Self { self.max(lo).min(hi) }

    #[inline] pub fn min_element(self) -> u8 { self.to_array().iter().copied().reduce(u8::min).unwrap() }
    #[inline] pub fn max_element(self) -> u8 { self.to_array().iter().copied().reduce(u8::max).unwrap() }
    /// Horizontal sum. No native SAD-style instruction on WASM SIMD128
    /// (unlike x86's `_mm_sad_epu8` — see module docs) — widen-and-fold.
    #[inline] pub fn element_sum(self) -> u32 { self.to_array().iter().map(|&x| x as u32).sum() }

    #[inline(always)] pub fn saturating_add(self, rhs: Self) -> Self { Self(u8x16_add_sat(self.0, rhs.0)) }
    #[inline(always)] pub fn saturating_sub(self, rhs: Self) -> Self { Self(u8x16_sub_sat(self.0, rhs.0)) }

    #[inline(always)] pub fn cmpeq(self, rhs: Self) -> IMask16 { IMask16(u8x16_eq(self.0, rhs.0)) }
    #[inline(always)] pub fn cmpne(self, rhs: Self) -> IMask16 { IMask16(u8x16_ne(self.0, rhs.0)) }
    #[inline(always)] pub fn cmpgt(self, rhs: Self) -> IMask16 { IMask16(u8x16_gt(self.0, rhs.0)) }
    #[inline(always)] pub fn cmplt(self, rhs: Self) -> IMask16 { IMask16(u8x16_lt(self.0, rhs.0)) }
    #[inline(always)] pub fn cmpge(self, rhs: Self) -> IMask16 { IMask16(u8x16_ge(self.0, rhs.0)) }
    #[inline(always)] pub fn cmple(self, rhs: Self) -> IMask16 { IMask16(u8x16_le(self.0, rhs.0)) }

    #[inline(always)]
    pub fn blend(mask: IMask16, if_true: Self, if_false: Self) -> Self {
        Self(v128_bitselect(if_true.0, if_false.0, mask.0))
    }

    #[inline] pub fn count_eq(self, needle: Self) -> u32 { self.cmpeq(needle).count_true() }
    #[inline] pub fn contains(self, needle: u8) -> bool { self.count_eq(Self::splat(needle)) > 0 }

    /// Full 16-byte shuffle. Native `i8x16_swizzle` — flat, no
    /// per-128-bit-lane restriction (see i8x16.rs's module docs).
    #[inline(always)]
    pub fn shuffle_bytes(self, indices: super::i8x16::i8x16) -> Self {
        Self(i8x16_swizzle(self.0, indices.0))
    }

    #[inline(always)] pub fn wrapping_add(self, r: Self) -> Self { self + r }
    #[inline(always)] pub fn wrapping_sub(self, r: Self) -> Self { self - r }
}

impl Add for u8x16 { type Output=Self; #[inline(always)] fn add(self,r:Self)->Self{Self(i8x16_add(self.0,r.0))} }
impl AddAssign for u8x16 { #[inline(always)] fn add_assign(&mut self,r:Self){*self=*self+r;} }
impl Sub for u8x16 { type Output=Self; #[inline(always)] fn sub(self,r:Self)->Self{Self(i8x16_sub(self.0,r.0))} }
impl SubAssign for u8x16 { #[inline(always)] fn sub_assign(&mut self,r:Self){*self=*self-r;} }
impl BitAnd for u8x16 { type Output=Self; #[inline(always)] fn bitand(self,r:Self)->Self{Self(v128_and(self.0,r.0))} }
impl BitAndAssign for u8x16 { #[inline(always)] fn bitand_assign(&mut self,r:Self){*self=*self&r;} }
impl BitOr  for u8x16 { type Output=Self; #[inline(always)] fn bitor (self,r:Self)->Self{Self(v128_or(self.0,r.0))} }
impl BitOrAssign  for u8x16 { #[inline(always)] fn bitor_assign (&mut self,r:Self){*self=*self|r;} }
impl BitXor for u8x16 { type Output=Self; #[inline(always)] fn bitxor(self,r:Self)->Self{Self(v128_xor(self.0,r.0))} }
impl BitXorAssign for u8x16 { #[inline(always)] fn bitxor_assign(&mut self,r:Self){*self=*self^r;} }
impl Not for u8x16 { type Output=Self; #[inline(always)] fn not(self)->Self{Self(v128_not(self.0))} }

impl PartialEq for u8x16 { fn eq(&self,r:&Self)->bool{i8x16_all_true(u8x16_eq(self.0,r.0))} }
impl Eq for u8x16 {}
impl fmt::Debug for u8x16 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "u8x16({:?})", self.to_array()) }
}
impl fmt::Display for u8x16 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "{:?}", self.to_array()) }
}
impl From<[u8; 16]> for u8x16 { fn from(a: [u8;16]) -> Self { Self::from_array(a) } }
impl From<u8x16> for [u8; 16] { fn from(v: u8x16) -> Self { v.to_array() } }
