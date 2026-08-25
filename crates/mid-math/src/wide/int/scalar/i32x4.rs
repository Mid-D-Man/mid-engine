// crates/mid-math/src/wide/int/scalar/i32x4.rs  (clean, no duplicate)
//! Scalar fallback 4-lane i32 — non-x86 platforms.

#![allow(non_camel_case_types)]

use core::fmt;
use core::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign,
    BitXor, BitXorAssign, Mul, MulAssign, Neg, Not, Sub, SubAssign,
};
use super::imask4::IMask4;

/// 4-lane signed 32-bit integer — scalar fallback.
#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(C, align(16))]
pub struct i32x4(pub(crate) [i32; 4]);

/// bool → u32 mask lane: true → 0xFFFFFFFF, false → 0.
#[inline(always)]
fn lane(b: bool) -> u32 { if b { u32::MAX } else { 0 } }

impl i32x4 {
    pub const ZERO: Self = i32x4([0; 4]);
    pub const ONE:  Self = i32x4([1; 4]);
    pub const MIN:  Self = i32x4([i32::MIN; 4]);
    pub const MAX:  Self = i32x4([i32::MAX; 4]);

    #[inline(always)] pub fn splat(v: i32) -> Self { i32x4([v; 4]) }
    #[inline(always)] pub fn new(a: i32, b: i32, c: i32, d: i32) -> Self { i32x4([a, b, c, d]) }
    #[inline(always)] pub fn from_array(a: [i32; 4]) -> Self { i32x4(a) }
    #[inline(always)] pub fn to_array(self) -> [i32; 4] { self.0 }
    #[inline] pub fn get(self, i: usize) -> i32 {
        assert!(i < 4, "i32x4::get — lane {i} out of bounds");
        self.0[i]
    }

    #[inline] pub fn abs(self) -> Self {
        i32x4([self.0[0].wrapping_abs(), self.0[1].wrapping_abs(),
               self.0[2].wrapping_abs(), self.0[3].wrapping_abs()])
    }
    /// Branchless compare-to-mask + bitwise blend, NOT a direct
    /// `i32::min()` per lane — matches the `wide` crate's own scalar
    /// i32x4 fallback (`self.simd_lt(rhs).select(self, rhs)`, checked
    /// directly against its source), which benched ~3x faster than the
    /// straightforward version this replaced. Confirmed the difference
    /// really is this pattern, not something else: `abs` (identical
    /// per-lane `wrapping_abs()` on both sides) ties exactly, and
    /// `wide`'s own u32x4::min — which still uses the direct
    /// `arr[i].min(rhs[i])` approach rather than this trick — ties us
    /// too. Formula is `n ^ ((n^y) & mask)`, mask = all-ones where
    /// `self < r` else zero — equivalent to `mask&y | !mask&n` in one
    /// fewer op, same trick `wide`'s own `generic_bit_blend` uses.
    #[inline]
    pub fn min(self, r: Self) -> Self {
        let m = [
            if self.0[0] < r.0[0] { -1i32 } else { 0 },
            if self.0[1] < r.0[1] { -1i32 } else { 0 },
            if self.0[2] < r.0[2] { -1i32 } else { 0 },
            if self.0[3] < r.0[3] { -1i32 } else { 0 },
        ];
        i32x4([
            r.0[0] ^ ((r.0[0] ^ self.0[0]) & m[0]),
            r.0[1] ^ ((r.0[1] ^ self.0[1]) & m[1]),
            r.0[2] ^ ((r.0[2] ^ self.0[2]) & m[2]),
            r.0[3] ^ ((r.0[3] ^ self.0[3]) & m[3]),
        ])
    }
    /// See `min` — same pattern, blend operands swapped.
    #[inline]
    pub fn max(self, r: Self) -> Self {
        let m = [
            if self.0[0] < r.0[0] { -1i32 } else { 0 },
            if self.0[1] < r.0[1] { -1i32 } else { 0 },
            if self.0[2] < r.0[2] { -1i32 } else { 0 },
            if self.0[3] < r.0[3] { -1i32 } else { 0 },
        ];
        i32x4([
            self.0[0] ^ ((self.0[0] ^ r.0[0]) & m[0]),
            self.0[1] ^ ((self.0[1] ^ r.0[1]) & m[1]),
            self.0[2] ^ ((self.0[2] ^ r.0[2]) & m[2]),
            self.0[3] ^ ((self.0[3] ^ r.0[3]) & m[3]),
        ])
    }
    #[inline] pub fn clamp(self, lo: Self, hi: Self) -> Self { self.max(lo).min(hi) }
    #[inline] pub fn min_element(self) -> i32 { self.0.iter().copied().reduce(i32::min).unwrap() }
    #[inline] pub fn max_element(self) -> i32 { self.0.iter().copied().reduce(i32::max).unwrap() }
    #[inline] pub fn element_sum(self) -> i32 {
        self.0[0].wrapping_add(self.0[1]).wrapping_add(self.0[2]).wrapping_add(self.0[3])
    }

    #[inline(always)] pub fn shl(self, c: u32) -> Self {
        i32x4([self.0[0] << c, self.0[1] << c, self.0[2] << c, self.0[3] << c])
    }
    #[inline(always)] pub fn shr_arithmetic(self, c: u32) -> Self {
        i32x4([self.0[0] >> c, self.0[1] >> c, self.0[2] >> c, self.0[3] >> c])
    }
    #[inline(always)] pub fn shr_logical(self, c: u32) -> Self {
        i32x4([
            (self.0[0] as u32 >> c) as i32, (self.0[1] as u32 >> c) as i32,
            (self.0[2] as u32 >> c) as i32, (self.0[3] as u32 >> c) as i32,
        ])
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
        i32x4([
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
        i32x4([self.0[0].saturating_add(r.0[0]), self.0[1].saturating_add(r.0[1]),
               self.0[2].saturating_add(r.0[2]), self.0[3].saturating_add(r.0[3])])
    }
    #[inline] pub fn saturating_sub(self, r: Self) -> Self {
        i32x4([self.0[0].saturating_sub(r.0[0]), self.0[1].saturating_sub(r.0[1]),
               self.0[2].saturating_sub(r.0[2]), self.0[3].saturating_sub(r.0[3])])
    }
}

impl Add for i32x4 {
    type Output = Self;
    fn add(self, r: Self) -> Self {
        i32x4([self.0[0].wrapping_add(r.0[0]), self.0[1].wrapping_add(r.0[1]),
               self.0[2].wrapping_add(r.0[2]), self.0[3].wrapping_add(r.0[3])])
    }
}
impl AddAssign for i32x4 { fn add_assign(&mut self, r: Self) { *self = *self + r; } }
impl Sub for i32x4 {
    type Output = Self;
    fn sub(self, r: Self) -> Self {
        i32x4([self.0[0].wrapping_sub(r.0[0]), self.0[1].wrapping_sub(r.0[1]),
               self.0[2].wrapping_sub(r.0[2]), self.0[3].wrapping_sub(r.0[3])])
    }
}
impl SubAssign for i32x4 { fn sub_assign(&mut self, r: Self) { *self = *self - r; } }
impl Neg for i32x4 {
    type Output = Self;
    fn neg(self) -> Self {
        i32x4([self.0[0].wrapping_neg(), self.0[1].wrapping_neg(),
               self.0[2].wrapping_neg(), self.0[3].wrapping_neg()])
    }
}
impl Mul for i32x4 {
    type Output = Self;
    fn mul(self, r: Self) -> Self {
        i32x4([self.0[0].wrapping_mul(r.0[0]), self.0[1].wrapping_mul(r.0[1]),
               self.0[2].wrapping_mul(r.0[2]), self.0[3].wrapping_mul(r.0[3])])
    }
}
impl MulAssign for i32x4 { fn mul_assign(&mut self, r: Self) { *self = *self * r; } }
impl BitAnd for i32x4 {
    type Output = Self;
    fn bitand(self, r: Self) -> Self {
        i32x4([self.0[0]&r.0[0], self.0[1]&r.0[1], self.0[2]&r.0[2], self.0[3]&r.0[3]])
    }
}
impl BitAndAssign for i32x4 { fn bitand_assign(&mut self, r: Self) { *self = *self & r; } }
impl BitOr for i32x4 {
    type Output = Self;
    fn bitor(self, r: Self) -> Self {
        i32x4([self.0[0]|r.0[0], self.0[1]|r.0[1], self.0[2]|r.0[2], self.0[3]|r.0[3]])
    }
}
impl BitOrAssign for i32x4 { fn bitor_assign(&mut self, r: Self) { *self = *self | r; } }
impl BitXor for i32x4 {
    type Output = Self;
    fn bitxor(self, r: Self) -> Self {
        i32x4([self.0[0]^r.0[0], self.0[1]^r.0[1], self.0[2]^r.0[2], self.0[3]^r.0[3]])
    }
}
impl BitXorAssign for i32x4 { fn bitxor_assign(&mut self, r: Self) { *self = *self ^ r; } }
impl Not for i32x4 {
    type Output = Self;
    fn not(self) -> Self {
        i32x4([!self.0[0], !self.0[1], !self.0[2], !self.0[3]])
    }
}

impl fmt::Debug for i32x4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "i32x4({}, {}, {}, {})", self.0[0], self.0[1], self.0[2], self.0[3])
    }
}
impl fmt::Display for i32x4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}, {}, {}, {}]", self.0[0], self.0[1], self.0[2], self.0[3])
    }
}
impl From<[i32;4]> for i32x4 { fn from(a: [i32;4]) -> Self { Self::from_array(a) } }
impl From<i32x4> for [i32;4] { fn from(v: i32x4) -> Self { v.to_array() } }
