// crates/mid-math/src/wide/int/wasm/imask4.rs
//! 4-lane integer comparison mask for i32x4/u32x4 — WASM SIMD128.
//! Mirrors sse2/imask4.rs. `i32x4_bitmask` returns a real `u8` directly
//! (no cast from a wider int like x86's `_mm_movemask_ps` needs), and
//! `i32x4_all_true`/`v128_any_true` return `bool` directly — WASM's API
//! is a fair bit cleaner here than SSE2's.

use core::fmt;
use core::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not};

#[cfg(target_arch = "wasm32")]
use core::arch::wasm32::{
    v128, v128_and, v128_or, v128_xor, v128_not, v128_any_true,
    i32x4_all_true, i32x4_bitmask,
};
#[cfg(target_arch = "wasm64")]
use core::arch::wasm64::{
    v128, v128_and, v128_or, v128_xor, v128_not, v128_any_true,
    i32x4_all_true, i32x4_bitmask,
};

use crate::wasm::v128_from_i32x4;

/// 4-lane integer comparison mask. Backed by `v128`.
/// Lane i: `0xFFFFFFFF` = true, `0x00000000` = false.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct IMask4(pub(crate) v128);

impl IMask4 {
    pub const FALSE: Self = Self(v128_from_i32x4([0; 4]));
    pub const TRUE:  Self = Self(v128_from_i32x4([-1; 4]));

    #[inline(always)] pub fn any(self) -> bool { v128_any_true(self.0) }
    #[inline(always)] pub fn all(self) -> bool { i32x4_all_true(self.0) }
    #[inline(always)] pub fn none(self) -> bool { !self.any() }

    /// Packed 4-bit bitmask — one bit per lane. Real `u8` return, no cast needed.
    #[inline(always)] pub fn bitmask(self) -> u8 { i32x4_bitmask(self.0) }
    #[inline] pub fn count_true(self) -> u32 { self.bitmask().count_ones() }
}

impl BitAnd for IMask4 { type Output=Self; #[inline(always)] fn bitand(self,r:Self)->Self{IMask4(v128_and(self.0,r.0))} }
impl BitAndAssign for IMask4 { #[inline(always)] fn bitand_assign(&mut self,r:Self){*self=*self&r;} }
impl BitOr for IMask4 { type Output=Self; #[inline(always)] fn bitor(self,r:Self)->Self{IMask4(v128_or(self.0,r.0))} }
impl BitOrAssign for IMask4 { #[inline(always)] fn bitor_assign(&mut self,r:Self){*self=*self|r;} }
impl BitXor for IMask4 { type Output=Self; #[inline(always)] fn bitxor(self,r:Self)->Self{IMask4(v128_xor(self.0,r.0))} }
impl BitXorAssign for IMask4 { #[inline(always)] fn bitxor_assign(&mut self,r:Self){*self=*self^r;} }
impl Not for IMask4 { type Output=Self; #[inline(always)] fn not(self)->Self{IMask4(v128_not(self.0))} }

impl PartialEq for IMask4 { #[inline] fn eq(&self,r:&Self)->bool{self.bitmask()==r.bitmask()} }
impl Eq for IMask4 {}
impl fmt::Debug for IMask4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "IMask4({:04b})", self.bitmask()) }
}

// unused import silencer for i32x4_splat brought in for symmetry with sibling files
#[allow(unused_imports)]
use i32x4_splat as _unused_i32x4_splat_reexport_guard;
