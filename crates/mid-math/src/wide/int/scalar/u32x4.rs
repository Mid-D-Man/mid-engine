// crates/mid-math/src/wide/int/scalar/u32x4.rs
//! Scalar fallback 4-lane u32 — non-x86 platforms.

#![allow(non_camel_case_types)]

use core::fmt;
use core::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign,
    BitXor, BitXorAssign, Mul, MulAssign, Not, Sub, SubAssign,
};
use super::imask4::IMask4;

/// 4-lane unsigned 32-bit integer — scalar fallback.
#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(C, align(16))]
pub struct u32x4(pub(crate) [u32; 4]);

#[inline(always)]
fn lane(b: bool) -> u32 { if b { u32::MAX } else { 0 } }

impl u32x4 {
    pub const ZERO: Self = u32x4([0; 4]);
    pub const ONE:  Self = u32x4([1; 4]);
    pub const MIN:  Self = u32x4([u32::MIN; 4]);
    pub const MAX:  Self = u32x4([u32::MAX; 4]);

    #[inline(always)] pub fn splat(v: u32) -> Self { u32x4([v; 4]) }
    #[inline(always)] pub fn new(a: u32, b: u32, c: u32, d: u32) -> Self { u32x4([a, b, c, d]) }
    #[inline(always)] pub fn from_array(a: [u32; 4]) -> Self { u32x4(a) }
    #[inline(always)] pub fn to_array(self) -> [u32; 4] { self.0 }
    #[inline] pub fn get(self, i: usize) -> u32 {
        assert!(i < 4, "u32x4::get — lane {i} out of bounds");
        self.0[i]
    }

    #[inline] pub fn min(self, r: Self) -> Self {
        u32x4([self.0[0].min(r.0[0]), self.0[1].min(r.0[1]),
               self.0[2].min(r.0[2]), self.0[3].min(r.0[3])])
    }
    #[inline] pub fn max(self, r: Self) -> Self {
        u32x4([self.0[0].max(r.0[0]), self.0[1].max(r.0[1]),
               self.0[2].max(r.0[2]), self.0[3].max(r.0[3])])
    }
    #[inline] pub fn clamp(self, lo: Self, hi: Self) -> Self { self.max(lo).min(hi) }
    #[inline] pub fn min_element(self) -> u32 { self.0.iter().copied().reduce(u32::min).unwrap() }
    #[inline] pub fn max_element(self) -> u32 { self.0.iter().copied().reduce(u32::max).unwrap() }
    #[inline] pub fn element_sum(self) -> u32 {
        self.0[0].wrapping_add(self.0[1]).wrapping_add(self.0[2]).wrapping_add(self.0[3])
    }

    #[inline(always)] pub fn shl(self, c: u32) -> Self {
        u32x4([self.0[0] << c, self.0[1] << c, self.0[2] << c, self.0[3] << c])
    }
    #[inline(always)] pub fn shr(self, c: u32) -> Self {
        u32x4([self.0[0] >> c, self.0[1] >> c, self.0[2] >> c, self.0[3] >> c])
    }

    #[inline] pub fn cmpeq(self, r: Self) -> IMask4 {
        IMask4([lane(self.0[0]==r.0[0]), lane(self.0[1]==r.0[1]),
                lane(self.0[2]==r.0[2]), lane(self.0[3]==r.0[3])])
    }
    #[inline] pub fn cmpne(self, r: Self) -> IMask4 { !self.cmpeq(r) }
    #[inline] pub fn cmpgt(self, r: Self) -> IMask4 {
        IMask4([lane(self.0[0]>r.0[0]), lane(self.0[1]>r.0[1]),
                lane(self.0[2]>r.0[2]), lane(self.0[3]>r.0[3])])
    }
    #[inline] pub fn cmplt(self, r: Self) -> IMask4 { r.cmpgt(self) }
    #[inline] pub fn cmpge(self, r: Self) -> IMask4 { !self.cmplt(r) }
    #[inline] pub fn cmple(self, r: Self) -> IMask4 { !self.cmpgt(r) }

    #[inline] pub fn blend(mask: IMask4, t: Self, f: Self) -> Self {
        u32x4([
            if mask.0[0] != 0 { t.0[0] } else { f.0[0] },
            if mask.0[1] != 0 { t.0[1] } else { f.0[1] },
            if mask.0[2] != 0 { t.0[2] } else { f.0[2] },
            if mask.0[3] != 0 { t.0[3] } else { f.0[3] },
        ])
    }

    #[inline] pub fn wrapping_add(self, r: Self) -> Self { self + r }
    #[inline] pub fn wrapping_sub(self, r: Self) -> Self { self - r }
    #[inline] pub fn wrapping_mul(self, r: Self) -> Self { self * r }

    #[inline] pub fn saturating_add(self, r: Self) -> Self {
        u32x4([self.0[0].saturating_add(r.0[0]), self.0[1].saturating_add(r.0[1]),
               self.0[2].saturating_add(r.0[2]), self.0[3].saturating_add(r.0[3])])
    }
    #[inline] pub fn saturating_sub(self, r: Self) -> Self {
        u32x4([self.0[0].saturating_sub(r.0[0]), self.0[1].saturating_sub(r.0[1]),
               self.0[2].saturating_sub(r.0[2]), self.0[3].saturating_sub(r.0[3])])
    }
}

impl Add for u32x4 {
    type Output = Self;
    fn add(self, r: Self) -> Self {
        u32x4([self.0[0].wrapping_add(r.0[0]), self.0[1].wrapping_add(r.0[1]),
               self.0[2].wrapping_add(r.0[2]), self.0[3].wrapping_add(r.0[3])])
    }
}
impl AddAssign for u32x4 { fn add_assign(&mut self, r: Self) { *self = *self + r; } }
impl Sub for u32x4 {
    type Output = Self;
    fn sub(self, r: Self) -> Self {
        u32x4([self.0[0].wrapping_sub(r.0[0]), self.0[1].wrapping_sub(r.0[1]),
               self.0[2].wrapping_sub(r.0[2]), self.0[3].wrapping_sub(r.0[3])])
    }
}
impl SubAssign for u32x4 { fn sub_assign(&mut self, r: Self) { *self = *self - r; } }
impl Mul for u32x4 {
    type Output = Self;
    fn mul(self, r: Self) -> Self {
        u32x4([self.0[0].wrapping_mul(r.0[0]), self.0[1].wrapping_mul(r.0[1]),
               self.0[2].wrapping_mul(r.0[2]), self.0[3].wrapping_mul(r.0[3])])
    }
}
impl MulAssign for u32x4 { fn mul_assign(&mut self, r: Self) { *self = *self * r; } }
impl BitAnd for u32x4 {
    type Output = Self;
    fn bitand(self, r: Self) -> Self {
        u32x4([self.0[0]&r.0[0], self.0[1]&r.0[1], self.0[2]&r.0[2], self.0[3]&r.0[3]])
    }
}
impl BitAndAssign for u32x4 { fn bitand_assign(&mut self, r: Self) { *self = *self & r; } }
impl BitOr for u32x4 {
    type Output = Self;
    fn bitor(self, r: Self) -> Self {
        u32x4([self.0[0]|r.0[0], self.0[1]|r.0[1], self.0[2]|r.0[2], self.0[3]|r.0[3]])
    }
}
impl BitOrAssign for u32x4 { fn bitor_assign(&mut self, r: Self) { *self = *self | r; } }
impl BitXor for u32x4 {
    type Output = Self;
    fn bitxor(self, r: Self) -> Self {
        u32x4([self.0[0]^r.0[0], self.0[1]^r.0[1], self.0[2]^r.0[2], self.0[3]^r.0[3]])
    }
}
impl BitXorAssign for u32x4 { fn bitxor_assign(&mut self, r: Self) { *self = *self ^ r; } }
impl Not for u32x4 {
    type Output = Self;
    fn not(self) -> Self {
        u32x4([!self.0[0], !self.0[1], !self.0[2], !self.0[3]])
    }
}

impl fmt::Debug for u32x4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "u32x4({}, {}, {}, {})", self.0[0], self.0[1], self.0[2], self.0[3])
    }
}
impl fmt::Display for u32x4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}, {}, {}, {}]", self.0[0], self.0[1], self.0[2], self.0[3])
    }
}
impl From<[u32;4]> for u32x4 { fn from(a: [u32;4]) -> Self { Self::from_array(a) } }
impl From<u32x4> for [u32;4] { fn from(v: u32x4) -> Self { v.to_array() } }
