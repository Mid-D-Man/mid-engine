// crates/mid-math/src/wide/int/wasm/imask16.rs
//! 16-lane integer comparison mask for i8x16/u8x16 — WASM SIMD128.
//! Mirrors sse2/imask16.rs. `i8x16_bitmask` returns a real `u16` directly
//! (no bit-pairing extraction needed, unlike SSE2's byte-granularity
//! `_mm_movemask_epi8` trick for this same lane count — WASM's bitmask
//! ops are per-lane-width natively, not always byte-granularity).

use core::fmt;
use core::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not};

#[cfg(target_arch = "wasm32")]
use core::arch::wasm32::{
    v128, v128_and, v128_or, v128_xor, v128_not, v128_any_true,
    i8x16_all_true, i8x16_bitmask,
};
#[cfg(target_arch = "wasm64")]
use core::arch::wasm64::{
    v128, v128_and, v128_or, v128_xor, v128_not, v128_any_true,
    i8x16_all_true, i8x16_bitmask,
};

use crate::wasm::v128_from_i8x16;

/// 16-lane integer comparison mask. Backed by `v128`.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct IMask16(pub(crate) v128);

impl IMask16 {
    pub const FALSE: Self = Self(v128_from_i8x16([0; 16]));
    pub const TRUE:  Self = Self(v128_from_i8x16([-1; 16]));

    #[inline(always)] pub fn any(self) -> bool { v128_any_true(self.0) }
    #[inline(always)] pub fn all(self) -> bool { i8x16_all_true(self.0) }
    #[inline(always)] pub fn none(self) -> bool { !self.any() }
    #[inline(always)] pub fn bitmask(self) -> u16 { i8x16_bitmask(self.0) }
    #[inline] pub fn count_true(self) -> u32 { self.bitmask().count_ones() }
}

impl BitAnd for IMask16 { type Output=Self; #[inline(always)] fn bitand(self,r:Self)->Self{IMask16(v128_and(self.0,r.0))} }
impl BitAndAssign for IMask16 { #[inline(always)] fn bitand_assign(&mut self,r:Self){*self=*self&r;} }
impl BitOr for IMask16 { type Output=Self; #[inline(always)] fn bitor(self,r:Self)->Self{IMask16(v128_or(self.0,r.0))} }
impl BitOrAssign for IMask16 { #[inline(always)] fn bitor_assign(&mut self,r:Self){*self=*self|r;} }
impl BitXor for IMask16 { type Output=Self; #[inline(always)] fn bitxor(self,r:Self)->Self{IMask16(v128_xor(self.0,r.0))} }
impl BitXorAssign for IMask16 { #[inline(always)] fn bitxor_assign(&mut self,r:Self){*self=*self^r;} }
impl Not for IMask16 { type Output=Self; #[inline(always)] fn not(self)->Self{IMask16(v128_not(self.0))} }

impl PartialEq for IMask16 { #[inline] fn eq(&self,r:&Self)->bool{self.bitmask()==r.bitmask()} }
impl Eq for IMask16 {}
impl fmt::Debug for IMask16 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "IMask16({:016b})", self.bitmask()) }
}
