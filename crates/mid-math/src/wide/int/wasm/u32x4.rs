// crates/mid-math/src/wide/int/wasm/u32x4.rs
//! 4-lane unsigned 32-bit integer vector — WASM SIMD128.
//!
//! Mirrors sse2/u32x4.rs. WASM has native `u32x4_min`/`u32x4_max`/
//! `u32x4_gt`/`u32x4_lt`/`u32x4_ge`/`u32x4_le` — real unsigned compare
//! instructions, no XOR-sign-flip trick needed at all (SSE2 has no
//! native unsigned compare of any kind, needs the trick for every op
//! here). `add`/`sub`/`shl`/`eq`/`ne` are `pub use i32x4_* as u32x4_*`
//! aliases in stdarch itself (wrapping arithmetic and bit-equality are
//! identical for signed/unsigned) — called directly here.

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

use super::imask4::IMask4;
use crate::wasm::v128_from_i32x4;

/// 4-lane unsigned 32-bit integer vector. Backed by `v128`.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct u32x4(pub(crate) v128);

impl u32x4 {
    pub const ZERO: Self = Self(v128_from_i32x4([0; 4]));
    pub const ONE:  Self = Self(v128_from_i32x4([1; 4]));
    pub const MIN:  Self = Self(v128_from_i32x4([0; 4]));
    pub const MAX:  Self = Self(v128_from_i32x4([u32::MAX as i32; 4]));

    #[inline(always)] pub fn splat(v: u32) -> Self { Self(u32x4_splat(v)) }
    #[inline(always)]
    pub fn new(a: u32, b: u32, c: u32, d: u32) -> Self {
        Self(v128_from_i32x4([a as i32, b as i32, c as i32, d as i32]))
    }
    #[inline(always)]
    pub fn from_array(a: [u32; 4]) -> Self {
        Self(v128_from_i32x4([a[0] as i32, a[1] as i32, a[2] as i32, a[3] as i32]))
    }

    #[inline(always)]
    pub fn to_array(self) -> [u32; 4] {
        [
            u32x4_extract_lane::<0>(self.0),
            u32x4_extract_lane::<1>(self.0),
            u32x4_extract_lane::<2>(self.0),
            u32x4_extract_lane::<3>(self.0),
        ]
    }

    #[inline]
    pub fn get(self, i: usize) -> u32 {
        match i {
            0 => u32x4_extract_lane::<0>(self.0),
            1 => u32x4_extract_lane::<1>(self.0),
            2 => u32x4_extract_lane::<2>(self.0),
            3 => u32x4_extract_lane::<3>(self.0),
            _ => panic!("u32x4::get — lane {i} out of bounds (max 3)"),
        }
    }

    /// Per-lane minimum. Native `u32x4_min` — real unsigned instruction.
    #[inline(always)] pub fn min(self, rhs: Self) -> Self { Self(u32x4_min(self.0, rhs.0)) }
    /// Per-lane maximum. Native `u32x4_max`.
    #[inline(always)] pub fn max(self, rhs: Self) -> Self { Self(u32x4_max(self.0, rhs.0)) }
    #[inline(always)] pub fn clamp(self, lo: Self, hi: Self) -> Self { self.max(lo).min(hi) }

    #[inline] pub fn min_element(self) -> u32 { self.to_array().iter().copied().reduce(u32::min).unwrap() }
    #[inline] pub fn max_element(self) -> u32 { self.to_array().iter().copied().reduce(u32::max).unwrap() }
    #[inline] pub fn element_sum(self) -> u32 { self.to_array().iter().fold(0u32, |a, &x| a.wrapping_add(x)) }

    #[inline(always)] pub fn shl(self, count: u32) -> Self { Self(i32x4_shl(self.0, count)) }
    #[inline(always)] pub fn shr(self, count: u32) -> Self { Self(u32x4_shr(self.0, count)) }

    #[inline(always)] pub fn cmpeq(self, rhs: Self) -> IMask4 { IMask4(u32x4_eq(self.0, rhs.0)) }
    #[inline(always)] pub fn cmpne(self, rhs: Self) -> IMask4 { IMask4(u32x4_ne(self.0, rhs.0)) }
    #[inline(always)] pub fn cmpgt(self, rhs: Self) -> IMask4 { IMask4(u32x4_gt(self.0, rhs.0)) }
    #[inline(always)] pub fn cmplt(self, rhs: Self) -> IMask4 { IMask4(u32x4_lt(self.0, rhs.0)) }
    #[inline(always)] pub fn cmpge(self, rhs: Self) -> IMask4 { IMask4(u32x4_ge(self.0, rhs.0)) }
    #[inline(always)] pub fn cmple(self, rhs: Self) -> IMask4 { IMask4(u32x4_le(self.0, rhs.0)) }

    #[inline(always)]
    pub fn blend(mask: IMask4, if_true: Self, if_false: Self) -> Self {
        Self(v128_bitselect(if_true.0, if_false.0, mask.0))
    }

    #[inline(always)] pub fn wrapping_add(self, r: Self) -> Self { self + r }
    #[inline(always)] pub fn wrapping_sub(self, r: Self) -> Self { self - r }
    #[inline(always)] pub fn wrapping_mul(self, r: Self) -> Self { self * r }

    /// No native 32-bit saturating add in WASM SIMD128 — scalar loop.
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

impl Add for u32x4 { type Output=Self; #[inline(always)] fn add(self,r:Self)->Self{Self(i32x4_add(self.0,r.0))} }
impl AddAssign for u32x4 { #[inline(always)] fn add_assign(&mut self,r:Self){*self=*self+r;} }
impl Sub for u32x4 { type Output=Self; #[inline(always)] fn sub(self,r:Self)->Self{Self(i32x4_sub(self.0,r.0))} }
impl SubAssign for u32x4 { #[inline(always)] fn sub_assign(&mut self,r:Self){*self=*self-r;} }

/// Native `i32x4_mul` — wrapping multiply is bit-identical for
/// signed/unsigned, no separate u32x4_mul exists (or is needed).
impl Mul for u32x4 { type Output=Self; #[inline(always)] fn mul(self,r:Self)->Self{Self(i32x4_mul(self.0,r.0))} }
impl MulAssign for u32x4 { #[inline(always)] fn mul_assign(&mut self,r:Self){*self=*self*r;} }

impl BitAnd for u32x4 { type Output=Self; #[inline(always)] fn bitand(self,r:Self)->Self{Self(v128_and(self.0,r.0))} }
impl BitAndAssign for u32x4 { #[inline(always)] fn bitand_assign(&mut self,r:Self){*self=*self&r;} }
impl BitOr  for u32x4 { type Output=Self; #[inline(always)] fn bitor (self,r:Self)->Self{Self(v128_or(self.0,r.0))} }
impl BitOrAssign  for u32x4 { #[inline(always)] fn bitor_assign (&mut self,r:Self){*self=*self|r;} }
impl BitXor for u32x4 { type Output=Self; #[inline(always)] fn bitxor(self,r:Self)->Self{Self(v128_xor(self.0,r.0))} }
impl BitXorAssign for u32x4 { #[inline(always)] fn bitxor_assign(&mut self,r:Self){*self=*self^r;} }
impl Not for u32x4 { type Output=Self; #[inline(always)] fn not(self)->Self{Self(v128_not(self.0))} }

impl PartialEq for u32x4 { #[inline] fn eq(&self,r:&Self)->bool{i32x4_all_true(u32x4_eq(self.0,r.0))} }
impl Eq for u32x4 {}
impl fmt::Debug for u32x4 {
    fn fmt(&self,f:&mut fmt::Formatter<'_>)->fmt::Result{let a=self.to_array();write!(f,"u32x4({},{},{},{})",a[0],a[1],a[2],a[3])}
}
impl fmt::Display for u32x4 {
    fn fmt(&self,f:&mut fmt::Formatter<'_>)->fmt::Result{let a=self.to_array();write!(f,"[{},{},{},{}]",a[0],a[1],a[2],a[3])}
}
impl From<[u32;4]> for u32x4 { fn from(a:[u32;4])->Self{Self::from_array(a)} }
impl From<u32x4> for [u32;4] { fn from(v:u32x4)->[u32;4]{v.to_array()} }
