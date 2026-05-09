// crates/mid-math/src/wide/int/scalar/i8x16.rs
//! Scalar fallback 16-lane i8 — non-x86 platforms.

#![allow(non_camel_case_types)]

use core::fmt;
use core::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign,
    BitXor, BitXorAssign, Neg, Not, Sub, SubAssign,
};
use super::imask16::IMask16;

#[inline(always)] fn lane8(b: bool) -> u8 { if b { u8::MAX } else { 0 } }

/// 16-lane signed 8-bit integer — scalar fallback.
#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(C, align(16))]
pub struct i8x16(pub(crate) [i8; 16]);

impl i8x16 {
    pub const ZERO: Self = i8x16([0; 16]);
    pub const ONE:  Self = i8x16([1; 16]);
    pub const MIN:  Self = i8x16([i8::MIN; 16]);
    pub const MAX:  Self = i8x16([i8::MAX; 16]);

    #[inline(always)] pub fn splat(v: i8) -> Self { i8x16([v; 16]) }
    #[inline(always)] pub fn from_array(a: [i8; 16]) -> Self { i8x16(a) }
    #[inline(always)] pub fn from_bytes(b: [u8; 16]) -> Self {
        // Safety: same layout
        unsafe { core::mem::transmute(b) }
    }
    #[inline(always)] pub fn to_array(self) -> [i8; 16] { self.0 }
    #[inline(always)] pub fn to_bytes(self) -> [u8; 16] { unsafe { core::mem::transmute(self.0) } }
    #[inline] pub fn get(self, i: usize) -> i8 { assert!(i < 16); self.0[i] }

    #[inline] pub fn abs(self) -> Self { i8x16(self.0.map(|x| x.wrapping_abs())) }
    #[inline] pub fn min(self, r: Self) -> Self { i8x16(core::array::from_fn(|i| self.0[i].min(r.0[i]))) }
    #[inline] pub fn max(self, r: Self) -> Self { i8x16(core::array::from_fn(|i| self.0[i].max(r.0[i]))) }
    #[inline] pub fn clamp(self, lo: Self, hi: Self) -> Self { self.max(lo).min(hi) }
    #[inline] pub fn min_element(self) -> i8 { self.0.iter().copied().reduce(i8::min).unwrap() }
    #[inline] pub fn max_element(self) -> i8 { self.0.iter().copied().reduce(i8::max).unwrap() }
    #[inline] pub fn element_sum(self) -> i32 { self.0.iter().map(|&x| x as i32).sum() }

    #[inline] pub fn saturating_add(self, r: Self) -> Self {
        i8x16(core::array::from_fn(|i| self.0[i].saturating_add(r.0[i])))
    }
    #[inline] pub fn saturating_sub(self, r: Self) -> Self {
        i8x16(core::array::from_fn(|i| self.0[i].saturating_sub(r.0[i])))
    }

    #[inline] pub fn cmpeq(self, r: Self) -> IMask16 { IMask16(core::array::from_fn(|i| lane8(self.0[i] == r.0[i]))) }
    #[inline] pub fn cmpne(self, r: Self) -> IMask16 { !self.cmpeq(r) }
    #[inline] pub fn cmpgt(self, r: Self) -> IMask16 { IMask16(core::array::from_fn(|i| lane8(self.0[i] > r.0[i]))) }
    #[inline] pub fn cmplt(self, r: Self) -> IMask16 { r.cmpgt(self) }
    #[inline] pub fn cmpge(self, r: Self) -> IMask16 { !self.cmplt(r) }
    #[inline] pub fn cmple(self, r: Self) -> IMask16 { !self.cmpgt(r) }

    #[inline] pub fn blend(mask: IMask16, t: Self, f: Self) -> Self {
        i8x16(core::array::from_fn(|i| if mask.0[i] != 0 { t.0[i] } else { f.0[i] }))
    }

    #[inline] pub fn shuffle_bytes(self, indices: Self) -> Self {
        let mut out = [0i8; 16];
        for i in 0..16 {
            let ix = self.0[i];
            out[i] = if ix < 0 { 0 } else { self.0[(ix & 0x0F) as usize] };
        }
        i8x16(out)
    }

    #[inline] pub fn count_eq(self, needle: Self) -> u32 { self.cmpeq(needle).count_true() }
    #[inline] pub fn contains(self, needle: i8) -> bool { self.count_eq(Self::splat(needle)) > 0 }

    #[inline] pub fn as_i16x8_lo(self) -> super::i16x8::i16x8 {
        super::i16x8::i16x8(core::array::from_fn(|i| self.0[i] as i16))
    }
    #[inline] pub fn as_i16x8_hi(self) -> super::i16x8::i16x8 {
        super::i16x8::i16x8(core::array::from_fn(|i| self.0[i + 8] as i16))
    }

    #[inline(always)] pub fn wrapping_add(self, r: Self) -> Self { self + r }
    #[inline(always)] pub fn wrapping_sub(self, r: Self) -> Self { self - r }
}

impl Add for i8x16 { type Output=Self; fn add(self, r: Self) -> Self { i8x16(core::array::from_fn(|i| self.0[i].wrapping_add(r.0[i]))) } }
impl AddAssign for i8x16 { fn add_assign(&mut self, r: Self) { *self = *self + r; } }
impl Sub for i8x16 { type Output=Self; fn sub(self, r: Self) -> Self { i8x16(core::array::from_fn(|i| self.0[i].wrapping_sub(r.0[i]))) } }
impl SubAssign for i8x16 { fn sub_assign(&mut self, r: Self) { *self = *self - r; } }
impl Neg for i8x16 { type Output=Self; fn neg(self) -> Self { i8x16(self.0.map(|x| x.wrapping_neg())) } }
impl BitAnd for i8x16 { type Output=Self; fn bitand(self, r: Self) -> Self { i8x16(core::array::from_fn(|i| self.0[i] & r.0[i])) } }
impl BitAndAssign for i8x16 { fn bitand_assign(&mut self, r: Self) { *self = *self & r; } }
impl BitOr for i8x16 { type Output=Self; fn bitor(self, r: Self) -> Self { i8x16(core::array::from_fn(|i| self.0[i] | r.0[i])) } }
impl BitOrAssign for i8x16 { fn bitor_assign(&mut self, r: Self) { *self = *self | r; } }
impl BitXor for i8x16 { type Output=Self; fn bitxor(self, r: Self) -> Self { i8x16(core::array::from_fn(|i| self.0[i] ^ r.0[i])) } }
impl BitXorAssign for i8x16 { fn bitxor_assign(&mut self, r: Self) { *self = *self ^ r; } }
impl Not for i8x16 { type Output=Self; fn not(self) -> Self { i8x16(self.0.map(|x| !x)) } }

impl fmt::Debug for i8x16 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "i8x16({:?})", self.0) }
}
impl fmt::Display for i8x16 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "{:?}", self.0) }
}
impl From<[i8;16]> for i8x16 { fn from(a: [i8;16]) -> Self { Self::from_array(a) } }
impl From<i8x16> for [i8;16] { fn from(v: i8x16) -> Self { v.to_array() } }
impl From<[u8;16]> for i8x16 { fn from(b: [u8;16]) -> Self { Self::from_bytes(b) } }
impl From<i8x16> for [u8;16] { fn from(v: i8x16) -> Self { v.to_bytes() } }
