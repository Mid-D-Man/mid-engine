// crates/mid-math/src/wide/int/scalar/u16x8.rs
//! Scalar fallback 8-lane u16 — non-x86 platforms.

#![allow(non_camel_case_types)]

use core::fmt;
use core::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign,
    BitXor, BitXorAssign, Mul, MulAssign, Not, Sub, SubAssign,
};
use super::imask8::IMask8;
use super::u32x4::u32x4;

#[inline(always)]
fn lane16u(b: bool) -> u16 { if b { u16::MAX } else { 0 } }

/// 8-lane unsigned 16-bit integer — scalar fallback.
#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(C, align(16))]
pub struct u16x8(pub(crate) [u16; 8]);

impl u16x8 {
    pub const ZERO: Self = u16x8([0; 8]);
    pub const ONE:  Self = u16x8([1; 8]);
    pub const MIN:  Self = u16x8([u16::MIN; 8]);
    pub const MAX:  Self = u16x8([u16::MAX; 8]);

    #[inline(always)] pub fn splat(v: u16) -> Self { u16x8([v; 8]) }
    #[inline(always)]
    pub fn new(a: u16, b: u16, c: u16, d: u16, e: u16, f: u16, g: u16, h: u16) -> Self {
        u16x8([a, b, c, d, e, f, g, h])
    }
    #[inline(always)] pub fn from_array(a: [u16; 8]) -> Self { u16x8(a) }
    #[inline(always)] pub fn to_array(self) -> [u16; 8] { self.0 }
    #[inline] pub fn get(self, i: usize) -> u16 { assert!(i < 8); self.0[i] }

    #[inline] pub fn min(self, r: Self) -> Self { u16x8(core::array::from_fn(|i| self.0[i].min(r.0[i]))) }
    #[inline] pub fn max(self, r: Self) -> Self { u16x8(core::array::from_fn(|i| self.0[i].max(r.0[i]))) }
    #[inline] pub fn clamp(self, lo: Self, hi: Self) -> Self { self.max(lo).min(hi) }
    #[inline] pub fn min_element(self) -> u16 { self.0.iter().copied().reduce(u16::min).unwrap() }
    #[inline] pub fn max_element(self) -> u16 { self.0.iter().copied().reduce(u16::max).unwrap() }
    #[inline] pub fn element_sum(self) -> u32 { self.0.iter().map(|&x| x as u32).sum() }

    #[inline] pub fn mul_lo(self, r: Self) -> Self {
        u16x8(core::array::from_fn(|i| self.0[i].wrapping_mul(r.0[i])))
    }
    #[inline] pub fn mul_high_u(self, r: Self) -> Self {
        u16x8(core::array::from_fn(|i| {
            ((self.0[i] as u32 * r.0[i] as u32) >> 16) as u16
        }))
    }

    #[inline] pub fn saturating_add(self, r: Self) -> Self {
        u16x8(core::array::from_fn(|i| self.0[i].saturating_add(r.0[i])))
    }
    #[inline] pub fn saturating_sub(self, r: Self) -> Self {
        u16x8(core::array::from_fn(|i| self.0[i].saturating_sub(r.0[i])))
    }

    #[inline] pub fn shl(self, c: u32) -> Self { u16x8(self.0.map(|x| x << c)) }
    #[inline] pub fn shr(self, c: u32) -> Self { u16x8(self.0.map(|x| x >> c)) }

    #[inline] pub fn cmpeq(self, r: Self) -> IMask8 { IMask8(core::array::from_fn(|i| lane16u(self.0[i] == r.0[i]))) }
    #[inline] pub fn cmpne(self, r: Self) -> IMask8 { !self.cmpeq(r) }
    #[inline] pub fn cmpgt(self, r: Self) -> IMask8 { IMask8(core::array::from_fn(|i| lane16u(self.0[i] > r.0[i]))) }
    #[inline] pub fn cmplt(self, r: Self) -> IMask8 { r.cmpgt(self) }
    #[inline] pub fn cmpge(self, r: Self) -> IMask8 { !self.cmplt(r) }
    #[inline] pub fn cmple(self, r: Self) -> IMask8 { !self.cmpgt(r) }

    #[inline] pub fn blend(mask: IMask8, t: Self, f: Self) -> Self {
        u16x8(core::array::from_fn(|i| if mask.0[i] != 0 { t.0[i] } else { f.0[i] }))
    }

    #[inline] pub fn as_u32x4_lo(self) -> u32x4 {
        u32x4([self.0[0] as u32, self.0[1] as u32, self.0[2] as u32, self.0[3] as u32])
    }
    #[inline] pub fn as_u32x4_hi(self) -> u32x4 {
        u32x4([self.0[4] as u32, self.0[5] as u32, self.0[6] as u32, self.0[7] as u32])
    }
    #[inline] pub fn pack_u32x4(lo: u32x4, hi: u32x4) -> Self {
        let sat = |x: u32| x.min(u16::MAX as u32) as u16;
        u16x8([
            sat(lo.0[0]), sat(lo.0[1]), sat(lo.0[2]), sat(lo.0[3]),
            sat(hi.0[0]), sat(hi.0[1]), sat(hi.0[2]), sat(hi.0[3]),
        ])
    }

    #[inline(always)] pub fn wrapping_add(self, r: Self) -> Self { self + r }
    #[inline(always)] pub fn wrapping_sub(self, r: Self) -> Self { self - r }
    #[inline(always)] pub fn wrapping_mul(self, r: Self) -> Self { self.mul_lo(r) }
}

impl Add for u16x8 { type Output=Self; fn add(self, r: Self) -> Self { u16x8(core::array::from_fn(|i| self.0[i].wrapping_add(r.0[i]))) } }
impl AddAssign for u16x8 { fn add_assign(&mut self, r: Self) { *self = *self + r; } }
impl Sub for u16x8 { type Output=Self; fn sub(self, r: Self) -> Self { u16x8(core::array::from_fn(|i| self.0[i].wrapping_sub(r.0[i]))) } }
impl SubAssign for u16x8 { fn sub_assign(&mut self, r: Self) { *self = *self - r; } }
impl Mul for u16x8 { type Output=Self; fn mul(self, r: Self) -> Self { self.mul_lo(r) } }
impl MulAssign for u16x8 { fn mul_assign(&mut self, r: Self) { *self = *self * r; } }
impl BitAnd for u16x8 { type Output=Self; fn bitand(self, r: Self) -> Self { u16x8(core::array::from_fn(|i| self.0[i] & r.0[i])) } }
impl BitAndAssign for u16x8 { fn bitand_assign(&mut self, r: Self) { *self = *self & r; } }
impl BitOr for u16x8 { type Output=Self; fn bitor(self, r: Self) -> Self { u16x8(core::array::from_fn(|i| self.0[i] | r.0[i])) } }
impl BitOrAssign for u16x8 { fn bitor_assign(&mut self, r: Self) { *self = *self | r; } }
impl BitXor for u16x8 { type Output=Self; fn bitxor(self, r: Self) -> Self { u16x8(core::array::from_fn(|i| self.0[i] ^ r.0[i])) } }
impl BitXorAssign for u16x8 { fn bitxor_assign(&mut self, r: Self) { *self = *self ^ r; } }
impl Not for u16x8 { type Output=Self; fn not(self) -> Self { u16x8(self.0.map(|x| !x)) } }

impl fmt::Debug for u16x8 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "u16x8({},{},{},{},{},{},{},{})",
            self.0[0],self.0[1],self.0[2],self.0[3],
            self.0[4],self.0[5],self.0[6],self.0[7])
    }
}
impl fmt::Display for u16x8 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "{:?}", self.0) }
}
impl From<[u16;8]> for u16x8 { fn from(a: [u16;8]) -> Self { Self::from_array(a) } }
impl From<u16x8> for [u16;8] { fn from(v: u16x8) -> Self { v.to_array() } }
