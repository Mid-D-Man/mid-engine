// crates/mid-math/src/wide/int/wasm/imask8.rs
//! 8-lane integer comparison mask for i16x8/u16x8 — WASM SIMD128.
//! Mirrors sse2/imask8.rs. `i16x8_bitmask` returns a real `u8` directly.

use core::fmt;
use core::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not};

#[cfg(target_arch = "wasm32")]
use core::arch::wasm32::{
    v128, v128_and, v128_or, v128_xor, v128_not, v128_any_true,
    i16x8_all_true, i16x8_bitmask,
};
#[cfg(target_arch = "wasm64")]
use core::arch::wasm64::{
    v128, v128_and, v128_or, v128_xor, v128_not, v128_any_true,
    i16x8_all_true, i16x8_bitmask,
};

use crate::wasm::v128_from_i16x8;

/// 8-lane integer comparison mask. Backed by `v128`.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct IMask8(pub(crate) v128);

impl IMask8 {
    pub const FALSE: Self = Self(v128_from_i16x8([0; 8]));
    pub const TRUE:  Self = Self(v128_from_i16x8([-1; 8]));

    #[inline(always)] pub fn any(self) -> bool { v128_any_true(self.0) }
    #[inline(always)] pub fn all(self) -> bool { i16x8_all_true(self.0) }
    #[inline(always)] pub fn none(self) -> bool { !self.any() }
    #[inline(always)] pub fn bitmask(self) -> u8 { i16x8_bitmask(self.0) }
    #[inline] pub fn count_true(self) -> u32 { self.bitmask().count_ones() }
}

impl BitAnd for IMask8 { type Output=Self; #[inline(always)] fn bitand(self,r:Self)->Self{IMask8(v128_and(self.0,r.0))} }
impl BitAndAssign for IMask8 { #[inline(always)] fn bitand_assign(&mut self,r:Self){*self=*self&r;} }
impl BitOr for IMask8 { type Output=Self; #[inline(always)] fn bitor(self,r:Self)->Self{IMask8(v128_or(self.0,r.0))} }
impl BitOrAssign for IMask8 { #[inline(always)] fn bitor_assign(&mut self,r:Self){*self=*self|r;} }
impl BitXor for IMask8 { type Output=Self; #[inline(always)] fn bitxor(self,r:Self)->Self{IMask8(v128_xor(self.0,r.0))} }
impl BitXorAssign for IMask8 { #[inline(always)] fn bitxor_assign(&mut self,r:Self){*self=*self^r;} }
impl Not for IMask8 { type Output=Self; #[inline(always)] fn not(self)->Self{IMask8(v128_not(self.0))} }

impl PartialEq for IMask8 { #[inline] fn eq(&self,r:&Self)->bool{self.bitmask()==r.bitmask()} }
impl Eq for IMask8 {}
impl fmt::Debug for IMask8 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "IMask8({:08b})", self.bitmask()) }
}
