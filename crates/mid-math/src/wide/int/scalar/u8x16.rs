// crates/mid-math/src/wide/int/scalar/u8x16.rs
//! Scalar fallback 16-lane u8 — non-x86 platforms.

#![allow(non_camel_case_types)]

use core::fmt;
use core::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign,
    BitXor, BitXorAssign, Not, Sub, SubAssign,
};
use super::imask16::IMask16;

#[inline(always)] fn lane8u(b: bool) -> u8 { if b { u8::MAX } else { 0 } }

/// 16-lane unsigned 8-bit integer — scalar fallback.
#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(C, align(16))]
pub struct u8x16(pub(crate) [u8; 16]);

impl u8x16 {
    pub const ZERO: Self = u8x16([0; 16]);
    pub const ONE:  Self = u8x16([1; 16]);
    pub const MIN:  Self = u8x16([u8::MIN; 16]);
    pub const MAX:  Self = u8x16([u8::MAX; 16]);

    #[inline(always)] pub fn splat(v: u8) -> Self { u8x16([v; 16]) }
    #[inline(always)] pub fn from_array(a: [u8; 16]) -> Self { u8x16(a) }
    #[inline(always)] pub fn from_bytes(b: [u8; 16]) -> Self { u8x16(b) }
    #[inline(always)] pub fn to_array(self) -> [u8; 16] { self.0 }
    #[inline] pub fn get(self, i: usize) -> u8 { assert!(i < 16); self.0[i] }

    #[inline] pub fn min(self, r: Self) -> Self { u8x16(core::array::from_fn(|i| self.0[i].min(r.0[i]))) }
    #[inline] pub fn max(self, r: Self) -> Self { u8x16(core::array::from_fn(|i| self.0[i].max(r.0[i]))) }
    #[inline] pub fn clamp(self, lo: Self, hi: Self) -> Self { self.max(lo).min(hi) }
    #[inline] pub fn min_element(self) -> u8 { self.0.iter().copied().reduce(u8::min).unwrap() }
    #[inline] pub fn max_element(self) -> u8 { self.0.iter().copied().reduce(u8::max).unwrap() }
    #[inline] pub fn element_sum(self) -> u32 { self.0.iter().map(|&x| x as u32).sum() }

    #[inline] pub fn saturating_add(self, r: Self) -> Self {
        u8x16(core::array::from_fn(|i| self.0[i].saturating_add(r.0[i])))
    }
    #[inline] pub fn saturating_sub(self, r: Self) -> Self {
        u8x16(core::array::from_fn(|i| self.0[i].saturating_sub(r.0[i])))
    }

    #[inline] pub fn cmpeq(self, r: Self) -> IMask16 { IMask16(core::array::from_fn(|i| lane8u(self.0[i] == r.0[i]))) }
    #[inline] pub fn cmpne(self, r: Self) -> IMask16 { !self.cmpeq(r) }
    #[inline] pub fn cmpgt(self, r: Self) -> IMask16 { IMask16(core::array::from_fn(|i| lane8u(self.0[i] > r.0[i]))) }
    #[inline] pub fn cmplt(self, r: Self) -> IMask16 { r.cmpgt(self) }
    #[inline] pub fn cmpge(self, r: Self) -> IMask16 { !self.cmplt(r) }
    #[inline] pub fn cmple(self, r: Self) -> IMask16 { !self.cmpgt(r) }

    #[inline] pub fn blend(mask: IMask16, t: Self, f: Self) -> Self {
        u8x16(core::array::from_fn(|i| if mask.0[i] != 0 { t.0[i] } else { f.0[i] }))
    }

    #[inline] pub fn shuffle_bytes(self, indices: Self) -> Self {
        let mut out = [0u8; 16];
        for i in 0..16 {
            out[i] = if indices.0[i] & 0x80 != 0 { 0 } else { self.0[(indices.0[i] & 0x0F) as usize] };
        }
        u8x16(out)
    }

    #[inline] pub fn count_eq(self, needle: Self) -> u32 { self.cmpeq(needle).count_true() }
    #[inline] pub fn contains(self, needle: u8) -> bool { self.count_eq(Self::splat(needle)) > 0 }

    #[inline] pub fn as_u16x8_lo(self) -> super::u16x8::u16x8 {
        super::u16x8::u16x8(core::array::from_fn(|i| self.0[i] as u16))
    }
    #[inline] pub fn as_u16x8_hi(self) -> super::u16x8::u16x8 {
        super::u16x8::u16x8(core::array::from_fn(|i| self.0[i + 8] as u16))
    }
    #[inline] pub fn pack_u16x8(lo: super::u16x8::u16x8, hi: super::u16x8::u16x8) -> Self {
        let sat = |x: u16| x.min(255) as u8;
        u8x16(core::array::from_fn(|i| {
            if i < 8 { sat(lo.0[i]) } else { sat(hi.0[i - 8]) }
        }))
    }

    #[inline(always)] pub fn wrapping_add(self, r: Self) -> Self { self + r }
    #[inline(always)] pub fn wrapping_sub(self, r: Self) -> Self { self - r }
}

impl Add for u8x16 { type Output=Self; fn add(self, r: Self) -> Self { u8x16(core::array::from_fn(|i| self.0[i].wrapping_add(r.0[i]))) } }
impl AddAssign for u8x16 { fn add_assign(&mut self, r: Self) { *self = *self + r; } }
impl Sub for u8x16 { type Output=Self; fn sub(self, r: Self) -> Self { u8x16(core::array::from_fn(|i| self.0[i].wrapping_sub(r.0[i]))) } }
impl SubAssign for u8x16 { fn sub_assign(&mut self, r: Self) { *self = *self - r; } }
impl BitAnd for u8x16 { type Output=Self; fn bitand(self, r: Self) -> Self { u8x16(core::array::from_fn(|i| self.0[i] & r.0[i])) } }
impl BitAndAssign for u8x16 { fn bitand_assign(&mut self, r: Self) { *self = *self & r; } }
impl BitOr for u8x16 { type Output=Self; fn bitor(self, r: Self) -> Self { u8x16(core::array::from_fn(|i| self.0[i] | r.0[i])) } }
impl BitOrAssign for u8x16 { fn bitor_assign(&mut self, r: Self) { *self = *self | r; } }
impl BitXor for u8x16 { type Output=Self; fn bitxor(self, r: Self) -> Self { u8x16(core::array::from_fn(|i| self.0[i] ^ r.0[i])) } }
impl BitXorAssign for u8x16 { fn bitxor_assign(&mut self, r: Self) { *self = *self ^ r; } }
impl Not for u8x16 { type Output=Self; fn not(self) -> Self { u8x16(self.0.map(|x| !x)) } }

impl fmt::Debug for u8x16 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "u8x16({:?})", self.0) }
}
impl fmt::Display for u8x16 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "{:?}", self.0) }
}
impl From<[u8;16]> for u8x16 { fn from(a: [u8;16]) -> Self { Self::from_array(a) } }
impl From<u8x16> for [u8;16] { fn from(v: u8x16) -> Self { v.to_array() } }
