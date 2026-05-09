// crates/mid-math/src/wide/int/scalar/i16x8.rs
//! Scalar fallback 8-lane i16 — non-x86 platforms.

#![allow(non_camel_case_types)]

use core::fmt;
use core::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign,
    BitXor, BitXorAssign, Mul, MulAssign, Neg, Not, Sub, SubAssign,
};
use super::imask8::IMask8;
use super::i32x4::i32x4;

#[inline(always)]
fn lane16(b: bool) -> u16 { if b { u16::MAX } else { 0 } }

/// 8-lane signed 16-bit integer — scalar fallback.
#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(C, align(16))]
pub struct i16x8(pub(crate) [i16; 8]);

impl i16x8 {
    pub const ZERO: Self = i16x8([0; 8]);
    pub const ONE:  Self = i16x8([1; 8]);
    pub const MIN:  Self = i16x8([i16::MIN; 8]);
    pub const MAX:  Self = i16x8([i16::MAX; 8]);

    #[inline(always)] pub fn splat(v: i16) -> Self { i16x8([v; 8]) }
    #[inline(always)]
    pub fn new(a: i16, b: i16, c: i16, d: i16, e: i16, f: i16, g: i16, h: i16) -> Self {
        i16x8([a, b, c, d, e, f, g, h])
    }
    #[inline(always)] pub fn from_array(a: [i16; 8]) -> Self { i16x8(a) }
    #[inline(always)] pub fn to_array(self) -> [i16; 8] { self.0 }
    #[inline] pub fn get(self, i: usize) -> i16 {
        assert!(i < 8); self.0[i]
    }

    #[inline] pub fn abs(self) -> Self {
        i16x8(self.0.map(|x| x.wrapping_abs()))
    }
    #[inline] pub fn min(self, r: Self) -> Self { i16x8(core::array::from_fn(|i| self.0[i].min(r.0[i]))) }
    #[inline] pub fn max(self, r: Self) -> Self { i16x8(core::array::from_fn(|i| self.0[i].max(r.0[i]))) }
    #[inline] pub fn clamp(self, lo: Self, hi: Self) -> Self { self.max(lo).min(hi) }
    #[inline] pub fn min_element(self) -> i16 { self.0.iter().copied().reduce(i16::min).unwrap() }
    #[inline] pub fn max_element(self) -> i16 { self.0.iter().copied().reduce(i16::max).unwrap() }
    #[inline] pub fn element_sum(self) -> i32 { self.0.iter().map(|&x| x as i32).sum() }

    #[inline] pub fn mul_lo(self, r: Self) -> Self {
        i16x8(core::array::from_fn(|i| self.0[i].wrapping_mul(r.0[i])))
    }
    #[inline] pub fn mul_high(self, r: Self) -> Self {
        i16x8(core::array::from_fn(|i| {
            ((self.0[i] as i32 * r.0[i] as i32) >> 16) as i16
        }))
    }

    #[inline] pub fn saturating_add(self, r: Self) -> Self {
        i16x8(core::array::from_fn(|i| self.0[i].saturating_add(r.0[i])))
    }
    #[inline] pub fn saturating_sub(self, r: Self) -> Self {
        i16x8(core::array::from_fn(|i| self.0[i].saturating_sub(r.0[i])))
    }

    #[inline] pub fn shl(self, c: u32) -> Self { i16x8(self.0.map(|x| x << c)) }
    #[inline] pub fn shr_arithmetic(self, c: u32) -> Self { i16x8(self.0.map(|x| x >> c)) }
    #[inline] pub fn shr_logical(self, c: u32) -> Self {
        i16x8(self.0.map(|x| (x as u16 >> c) as i16))
    }

    #[inline] pub fn cmpeq(self, r: Self) -> IMask8 {
        IMask8(core::array::from_fn(|i| lane16(self.0[i] == r.0[i])))
    }
    #[inline] pub fn cmpne(self, r: Self) -> IMask8 { !self.cmpeq(r) }
    #[inline] pub fn cmpgt(self, r: Self) -> IMask8 {
        IMask8(core::array::from_fn(|i| lane16(self.0[i] > r.0[i])))
    }
    #[inline] pub fn cmplt(self, r: Self) -> IMask8 { r.cmpgt(self) }
    #[inline] pub fn cmpge(self, r: Self) -> IMask8 { !self.cmplt(r) }
    #[inline] pub fn cmple(self, r: Self) -> IMask8 { !self.cmpgt(r) }

    #[inline] pub fn blend(mask: IMask8, t: Self, f: Self) -> Self {
        i16x8(core::array::from_fn(|i| if mask.0[i] != 0 { t.0[i] } else { f.0[i] }))
    }

    #[inline] pub fn as_i32x4_lo(self) -> i32x4 {
        i32x4([self.0[0] as i32, self.0[1] as i32, self.0[2] as i32, self.0[3] as i32])
    }
    #[inline] pub fn as_i32x4_hi(self) -> i32x4 {
        i32x4([self.0[4] as i32, self.0[5] as i32, self.0[6] as i32, self.0[7] as i32])
    }
    #[inline] pub fn pack_i32x4(lo: i32x4, hi: i32x4) -> Self {
        let sat = |x: i32| x.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
        i16x8([
            sat(lo.0[0]), sat(lo.0[1]), sat(lo.0[2]), sat(lo.0[3]),
            sat(hi.0[0]), sat(hi.0[1]), sat(hi.0[2]), sat(hi.0[3]),
        ])
    }

    #[inline(always)] pub fn wrapping_add(self, r: Self) -> Self { self + r }
    #[inline(always)] pub fn wrapping_sub(self, r: Self) -> Self { self - r }
    #[inline(always)] pub fn wrapping_mul(self, r: Self) -> Self { self.mul_lo(r) }
}

impl Add for i16x8 { type Output=Self; fn add(self, r: Self) -> Self { i16x8(core::array::from_fn(|i| self.0[i].wrapping_add(r.0[i]))) } }
impl AddAssign for i16x8 { fn add_assign(&mut self, r: Self) { *self = *self + r; } }
impl Sub for i16x8 { type Output=Self; fn sub(self, r: Self) -> Self { i16x8(core::array::from_fn(|i| self.0[i].wrapping_sub(r.0[i]))) } }
impl SubAssign for i16x8 { fn sub_assign(&mut self, r: Self) { *self = *self - r; } }
impl Neg for i16x8 { type Output=Self; fn neg(self) -> Self { i16x8(self.0.map(|x| x.wrapping_neg())) } }
impl Mul for i16x8 { type Output=Self; fn mul(self, r: Self) -> Self { self.mul_lo(r) } }
impl MulAssign for i16x8 { fn mul_assign(&mut self, r: Self) { *self = *self * r; } }
impl BitAnd for i16x8 { type Output=Self; fn bitand(self, r: Self) -> Self { i16x8(core::array::from_fn(|i| self.0[i] & r.0[i])) } }
impl BitAndAssign for i16x8 { fn bitand_assign(&mut self, r: Self) { *self = *self & r; } }
impl BitOr for i16x8 { type Output=Self; fn bitor(self, r: Self) -> Self { i16x8(core::array::from_fn(|i| self.0[i] | r.0[i])) } }
impl BitOrAssign for i16x8 { fn bitor_assign(&mut self, r: Self) { *self = *self | r; } }
impl BitXor for i16x8 { type Output=Self; fn bitxor(self, r: Self) -> Self { i16x8(core::array::from_fn(|i| self.0[i] ^ r.0[i])) } }
impl BitXorAssign for i16x8 { fn bitxor_assign(&mut self, r: Self) { *self = *self ^ r; } }
impl Not for i16x8 { type Output=Self; fn not(self) -> Self { i16x8(self.0.map(|x| !x)) } }

impl fmt::Debug for i16x8 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "i16x8({},{},{},{},{},{},{},{})",
            self.0[0],self.0[1],self.0[2],self.0[3],
            self.0[4],self.0[5],self.0[6],self.0[7])
    }
}
impl fmt::Display for i16x8 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "{:?}", self.0) }
}
impl From<[i16;8]> for i16x8 { fn from(a: [i16;8]) -> Self { Self::from_array(a) } }
impl From<i16x8> for [i16;8] { fn from(v: i16x8) -> Self { v.to_array() } }
