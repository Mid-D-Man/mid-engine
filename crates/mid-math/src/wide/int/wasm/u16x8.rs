// crates/mid-math/src/wide/int/wasm/u16x8.rs
//! 8-lane unsigned 16-bit integer vector — WASM SIMD128.
//!
//! Mirrors sse2/u16x8.rs. Native `u16x8_min`/`u16x8_max` (real unsigned
//! instructions, no XOR-flip trick SSE2 needs) and native
//! `u16x8_add_sat`/`u16x8_sub_sat`.

#![allow(non_camel_case_types)]

use core::fmt;
use core::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign,
    BitXor, BitXorAssign, Mul, MulAssign, Not, Sub, SubAssign,
};

#[cfg(target_arch = "wasm32")]
use core::arch::wasm32::*;
#[cfg(target_arch = "wasm64")]
use core::arch::wasm64::*;

use super::imask8::IMask8;
use crate::wasm::v128_from_i16x8;

/// 8-lane unsigned 16-bit integer vector. Backed by `v128`.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct u16x8(pub(crate) v128);

impl u16x8 {
    pub const ZERO: Self = Self(v128_from_i16x8([0; 8]));
    pub const ONE:  Self = Self(v128_from_i16x8([1; 8]));
    pub const MIN:  Self = Self(v128_from_i16x8([0; 8]));
    pub const MAX:  Self = Self(v128_from_i16x8([u16::MAX as i16; 8]));

    #[inline(always)] pub fn splat(v: u16) -> Self { Self(u16x8_splat(v)) }
    #[inline(always)]
    pub fn new(a:u16,b:u16,c:u16,d:u16,e:u16,f:u16,g:u16,h:u16) -> Self {
        Self(v128_from_i16x8([a as i16,b as i16,c as i16,d as i16,e as i16,f as i16,g as i16,h as i16]))
    }
    #[inline(always)]
    pub fn from_array(a: [u16; 8]) -> Self {
        Self(v128_from_i16x8(core::array::from_fn(|i| a[i] as i16)))
    }

    #[inline(always)]
    pub fn to_array(self) -> [u16; 8] {
        [
            u16x8_extract_lane::<0>(self.0), u16x8_extract_lane::<1>(self.0),
            u16x8_extract_lane::<2>(self.0), u16x8_extract_lane::<3>(self.0),
            u16x8_extract_lane::<4>(self.0), u16x8_extract_lane::<5>(self.0),
            u16x8_extract_lane::<6>(self.0), u16x8_extract_lane::<7>(self.0),
        ]
    }

    #[inline]
    pub fn get(self, i: usize) -> u16 {
        assert!(i < 8, "u16x8::get — lane {i} out of bounds (max 7)");
        self.to_array()[i]
    }

    /// Zero-extend the low 4 lanes to `u32x4`. Native `i32x4_extend_low_u16x8`
    /// (name says "i32x4" — that's the output register shape; passing a
    /// `u16x8` source zero-extends, matching Rust's `u32x4` semantics).
    #[inline(always)]
    pub fn as_u32x4_lo(self) -> super::u32x4::u32x4 {
        super::u32x4::u32x4(i32x4_extend_low_u16x8(self.0))
    }
    #[inline(always)]
    pub fn as_u32x4_hi(self) -> super::u32x4::u32x4 {
        super::u32x4::u32x4(i32x4_extend_high_u16x8(self.0))
    }

    #[inline(always)] pub fn min(self, rhs: Self) -> Self { Self(u16x8_min(self.0, rhs.0)) }
    #[inline(always)] pub fn max(self, rhs: Self) -> Self { Self(u16x8_max(self.0, rhs.0)) }
    #[inline(always)] pub fn clamp(self, lo: Self, hi: Self) -> Self { self.max(lo).min(hi) }

    #[inline] pub fn min_element(self) -> u16 { self.to_array().iter().copied().reduce(u16::min).unwrap() }
    #[inline] pub fn max_element(self) -> u16 { self.to_array().iter().copied().reduce(u16::max).unwrap() }
    #[inline] pub fn element_sum(self) -> u32 { self.to_array().iter().map(|&x| x as u32).sum() }

    #[inline(always)] pub fn mul_lo(self, rhs: Self) -> Self { Self(i16x8_mul(self.0, rhs.0)) }
    #[inline(always)] pub fn saturating_add(self, rhs: Self) -> Self { Self(u16x8_add_sat(self.0, rhs.0)) }
    #[inline(always)] pub fn saturating_sub(self, rhs: Self) -> Self { Self(u16x8_sub_sat(self.0, rhs.0)) }

    #[inline(always)] pub fn shl(self, count: u32) -> Self { Self(i16x8_shl(self.0, count)) }
    #[inline(always)] pub fn shr(self, count: u32) -> Self { Self(u16x8_shr(self.0, count)) }

    #[inline(always)] pub fn cmpeq(self, rhs: Self) -> IMask8 { IMask8(u16x8_eq(self.0, rhs.0)) }
    #[inline(always)] pub fn cmpne(self, rhs: Self) -> IMask8 { IMask8(u16x8_ne(self.0, rhs.0)) }
    #[inline(always)] pub fn cmpgt(self, rhs: Self) -> IMask8 { IMask8(u16x8_gt(self.0, rhs.0)) }
    #[inline(always)] pub fn cmplt(self, rhs: Self) -> IMask8 { IMask8(u16x8_lt(self.0, rhs.0)) }
    #[inline(always)] pub fn cmpge(self, rhs: Self) -> IMask8 { IMask8(u16x8_ge(self.0, rhs.0)) }
    #[inline(always)] pub fn cmple(self, rhs: Self) -> IMask8 { IMask8(u16x8_le(self.0, rhs.0)) }

    #[inline(always)]
    pub fn blend(mask: IMask8, if_true: Self, if_false: Self) -> Self {
        Self(v128_bitselect(if_true.0, if_false.0, mask.0))
    }

    #[inline(always)] pub fn wrapping_add(self, r: Self) -> Self { self + r }
    #[inline(always)] pub fn wrapping_sub(self, r: Self) -> Self { self - r }
    #[inline(always)] pub fn wrapping_mul(self, r: Self) -> Self { self.mul_lo(r) }
}

impl Add for u16x8 { type Output=Self; #[inline(always)] fn add(self,r:Self)->Self{Self(i16x8_add(self.0,r.0))} }
impl AddAssign for u16x8 { #[inline(always)] fn add_assign(&mut self,r:Self){*self=*self+r;} }
impl Sub for u16x8 { type Output=Self; #[inline(always)] fn sub(self,r:Self)->Self{Self(i16x8_sub(self.0,r.0))} }
impl SubAssign for u16x8 { #[inline(always)] fn sub_assign(&mut self,r:Self){*self=*self-r;} }
impl Mul for u16x8 { type Output=Self; #[inline(always)] fn mul(self,r:Self)->Self{self.mul_lo(r)} }
impl MulAssign for u16x8 { #[inline(always)] fn mul_assign(&mut self,r:Self){*self=*self*r;} }
impl BitAnd for u16x8 { type Output=Self; #[inline(always)] fn bitand(self,r:Self)->Self{Self(v128_and(self.0,r.0))} }
impl BitAndAssign for u16x8 { #[inline(always)] fn bitand_assign(&mut self,r:Self){*self=*self&r;} }
impl BitOr  for u16x8 { type Output=Self; #[inline(always)] fn bitor (self,r:Self)->Self{Self(v128_or(self.0,r.0))} }
impl BitOrAssign  for u16x8 { #[inline(always)] fn bitor_assign (&mut self,r:Self){*self=*self|r;} }
impl BitXor for u16x8 { type Output=Self; #[inline(always)] fn bitxor(self,r:Self)->Self{Self(v128_xor(self.0,r.0))} }
impl BitXorAssign for u16x8 { #[inline(always)] fn bitxor_assign(&mut self,r:Self){*self=*self^r;} }
impl Not for u16x8 { type Output=Self; #[inline(always)] fn not(self)->Self{Self(v128_not(self.0))} }

impl PartialEq for u16x8 { fn eq(&self,r:&Self)->bool{i16x8_all_true(u16x8_eq(self.0,r.0))} }
impl Eq for u16x8 {}
impl fmt::Debug for u16x8 {
    fn fmt(&self,f:&mut fmt::Formatter<'_>)->fmt::Result{
        let a=self.to_array();write!(f,"u16x8({},{},{},{},{},{},{},{})",a[0],a[1],a[2],a[3],a[4],a[5],a[6],a[7])
    }
}
impl fmt::Display for u16x8 {
    fn fmt(&self,f:&mut fmt::Formatter<'_>)->fmt::Result{
        let a=self.to_array();write!(f,"[{},{},{},{},{},{},{},{}]",a[0],a[1],a[2],a[3],a[4],a[5],a[6],a[7])
    }
}
impl From<[u16;8]> for u16x8 { fn from(a:[u16;8])->Self{Self::from_array(a)} }
impl From<u16x8> for [u16;8] { fn from(v:u16x8)->[u16;8]{v.to_array()} }
