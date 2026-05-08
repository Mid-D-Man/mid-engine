// crates/mid-math/src/wide/int/scalar/i32x4.rs
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

// Macro to reduce boilerplate for element-wise ops.
macro_rules! ew {
    ($a:expr, $b:expr, $op:tt) => {
        [$a.0[0] $op $b.0[0], $a.0[1] $op $b.0[1],
         $a.0[2] $op $b.0[2], $a.0[3] $op $b.0[3]]
    };
}

impl i32x4 {
    pub const ZERO: Self = i32x4([0i32; 4]);
    pub const ONE:  Self = i32x4([1i32; 4]);
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
        i32x4([
            self.0[0].wrapping_abs(), self.0[1].wrapping_abs(),
            self.0[2].wrapping_abs(), self.0[3].wrapping_abs(),
        ])
    }

    #[inline] pub fn min(self, r: Self) -> Self {
        i32x4([self.0[0].min(r.0[0]), self.0[1].min(r.0[1]),
               self.0[2].min(r.0[2]), self.0[3].min(r.0[3])])
    }
    #[inline] pub fn max(self, r: Self) -> Self {
        i32x4([self.0[0].max(r.0[0]), self.0[1].max(r.0[1]),
               self.0[2].max(r.0[2]), self.0[3].max(r.0[3])])
    }
    #[inline] pub fn clamp(self, lo: Self, hi: Self) -> Self { self.max(lo).min(hi) }
    #[inline] pub fn min_element(self) -> i32 { self.0[0].min(self.0[1]).min(self.0[2]).min(self.0[3]) }
    #[inline] pub fn max_element(self) -> i32 { self.0[0].max(self.0[1]).max(self.0[2]).max(self.0[3]) }
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
        // Logical right shift on i32 — cast to u32, shift, cast back
        i32x4([
            (self.0[0] as u32 >> c) as i32, (self.0[1] as u32 >> c) as i32,
            (self.0[2] as u32 >> c) as i32, (self.0[3] as u32 >> c) as i32,
        ])
    }

    // ── Comparison ────────────────────────────────────────────────────────────

    #[inline] pub fn cmpeq(self, r: Self) -> IMask4 {
        IMask4([mask(self.0[0] == r.0[0]), mask(self.0[1] == r.0[1]),
                mask(self.0[2] == r.0[2]), mask(self.0[3] == r.0[3])])
    }
    #[inline] pub fn cmpne(self, r: Self) -> IMask4 { !self.cmpeq(r) }
    #[inline] pub fn cmpgt(self, r: Self) -> IMask4 {
        IMask4([mask(self.0[0] > r.0[0]), mask(self.0[1] > r.0[1]),
                mask(self.0[2] > r.0[2]), mask(self.0[3] > r.0[3])])
    }
    #[inline] pub fn cmplt(self, r: Self) -> IMask4 { r.cmpgt(self) }
    #[inline] pub fn cmpge(self, r: Self) -> IMask4 { !self.cmplt(r) }
    #[inline] pub fn cmple(self, r: Self) -> IMask4 { !self.cmpgt(r) }

    // ── Select ────────────────────────────────────────────────────────────────

    #[inline] pub fn blend(mask: IMask4, t: Self, f: Self) -> Self {
        i32x4([
            if mask.0[0] != 0 { t.0[0] } else { f.0[0] },
            if mask.0[1] != 0 { t.0[1] } else { f.0[1] },
            if mask.0[2] != 0 { t.0[2] } else { f.0[2] },
            if mask.0[3] != 0 { t.0[3] } else { f.0[3] },
        ])
    }

    #[inline(always)] pub fn wrapping_add(self, r: Self) -> Self {
        i32x4(ew!(self, r, wrapping_add))  // can't use macro with method call...
    }

    #[inline] pub fn wrapping_add(self, r: Self) -> Self {
        i32x4([self.0[0].wrapping_add(r.0[0]), self.0[1].wrapping_add(r.0[1]),
               self.0[2].wrapping_add(r.0[2]), self.0[3].wrapping_add(r.0[3])])
    }
    #[inline] pub fn wrapping_sub(self, r: Self) -> Self {
        i32x4([self.0[0].wrapping_sub(r.0[0]), self.0[1].wrapping_sub(r.0[1]),
               self.0[2].wrapping_sub(r.0[2]), self.0[3].wrapping_sub(r.0[3])])
    }
    #[inline] pub fn wrapping_mul(self, r: Self) -> Self {
        i32x4([self.0[0].wrapping_mul(r.0[0]), self.0[1].wrapping_mul(r.0[1]),
               self.0[2].wrapping_mul(r.0[2]), self.0[3].wrapping_mul(r.0[3])])
    }
    #[inline] pub fn saturating_add(self, r: Self) -> Self {
        i32x4([self.0[0].saturating_add(r.0[0]), self.0[1].saturating_add(r.0[1]),
               self.0[2].saturating_add(r.0[2]), self.0[3].saturating_add(r.0[3])])
    }
    #[inline] pub fn saturating_sub(self, r: Self) -> Self {
        i32x4([self.0[0].saturating_sub(r.0[0]), self.0[1].saturating_sub(r.0[1]),
               self.0[2].saturating_sub(r.0[2]), self.0[3].saturating_sub(r.0[3])])
    }
}

/// Convert bool to u32 mask lane: true → 0xFFFFFFFF, false → 0.
#[inline(always)]
fn mask(b: bool) -> u32 { if b { u32::MAX } else { 0 } }

// ── Operators ─────────────────────────────────────────────────────────────────

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
        i32x4([self.0[0] & r.0[0], self.0[1] & r.0[1], self.0[2] & r.0[2], self.0[3] & r.0[3]])
    }
}
impl BitAndAssign for i32x4 { fn bitand_assign(&mut self, r: Self) { *self = *self & r; } }
impl BitOr for i32x4 {
    type Output = Self;
    fn bitor(self, r: Self) -> Self {
        i32x4([self.0[0] | r.0[0], self.0[1] | r.0[1], self.0[2] | r.0[2], self.0[3] | r.0[3]])
    }
}
impl BitOrAssign for i32x4 { fn bitor_assign(&mut self, r: Self) { *self = *self | r; } }
impl BitXor for i32x4 {
    type Output = Self;
    fn bitxor(self, r: Self) -> Self {
        i32x4([self.0[0] ^ r.0[0], self.0[1] ^ r.0[1], self.0[2] ^ r.0[2], self.0[3] ^ r.0[3]])
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
impl From<[i32; 4]> for i32x4 { fn from(a: [i32; 4]) -> Self { Self::from_array(a) } }
impl From<i32x4> for [i32; 4] { fn from(v: i32x4) -> Self { v.to_array() } }
