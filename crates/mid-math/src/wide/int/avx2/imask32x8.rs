// crates/mid-math/src/wide/int/avx2/imask32x8.rs
//! 8-lane integer comparison mask for i32x8/u32x8.
//!
//! Each 32-bit lane: 0xFFFFFFFF = true, 0x00000000 = false.
//! Never constructed directly -- always produced by i32x8/u32x8 comparisons.
//!
//! Always compiled on x86/x86_64, not gated on the `avx2` target feature.
//! Storage is two portable `IMask4` halves, never a raw `__m256i` -- for a
//! pure bitwise/reduction type like this one, splitting the work across two
//! `IMask4` halves costs nothing bitwise-op-for-bitwise-op relative to a
//! real 256-bit instruction (modern superscalar execution runs two
//! independent 128-bit ops about as fast as one 256-bit op once dispatch
//! overhead is counted), so there is no runtime-detection fast path here at
//! all -- this type is unconditionally portable and needs none of `i32x8`'s
//! dispatch machinery.

use core::fmt;
use core::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not};

use crate::wide::int::sse2::imask4::IMask4;

/// 8-lane integer comparison mask. Lane i: `0xFFFFFFFF` = true, `0x00000000` = false.
/// Use [`i32x8::blend`][super::i32x8::i32x8::blend] /
/// [`u32x8::blend`][super::u32x8::u32x8::blend] for branchless selection.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct IMask32x8 {
    pub(crate) lo: IMask4,
    pub(crate) hi: IMask4,
}

impl IMask32x8 {
    /// All lanes false.
    pub const FALSE: Self = Self { lo: IMask4::FALSE, hi: IMask4::FALSE };
    /// All lanes true.
    pub const TRUE: Self = Self { lo: IMask4::TRUE, hi: IMask4::TRUE };

    #[inline]
    pub(crate) fn from_halves(lo: IMask4, hi: IMask4) -> Self { Self { lo, hi } }

    /// True if any lane is set.
    #[inline]
    pub fn any(self) -> bool { self.lo.any() || self.hi.any() }

    /// True if all lanes are set.
    #[inline]
    pub fn all(self) -> bool { self.lo.all() && self.hi.all() }

    /// True if no lane is set.
    #[inline]
    pub fn none(self) -> bool { self.lo.none() && self.hi.none() }

    /// Packed 8-bit bitmask -- one bit per 32-bit lane, low half in bits 0-3.
    #[inline]
    pub fn bitmask(self) -> u8 {
        (self.lo.bitmask() as u8) | ((self.hi.bitmask() as u8) << 4)
    }

    /// Number of true lanes.
    #[inline]
    pub fn count_true(self) -> u32 { self.bitmask().count_ones() }
}

impl BitAnd for IMask32x8 {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, r: Self) -> Self { Self { lo: self.lo & r.lo, hi: self.hi & r.hi } }
}
impl BitAndAssign for IMask32x8 {
    #[inline(always)]
    fn bitand_assign(&mut self, r: Self) { *self = *self & r; }
}
impl BitOr for IMask32x8 {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, r: Self) -> Self { Self { lo: self.lo | r.lo, hi: self.hi | r.hi } }
}
impl BitOrAssign for IMask32x8 {
    #[inline(always)]
    fn bitor_assign(&mut self, r: Self) { *self = *self | r; }
}
impl BitXor for IMask32x8 {
    type Output = Self;
    #[inline(always)]
    fn bitxor(self, r: Self) -> Self { Self { lo: self.lo ^ r.lo, hi: self.hi ^ r.hi } }
}
impl BitXorAssign for IMask32x8 {
    #[inline(always)]
    fn bitxor_assign(&mut self, r: Self) { *self = *self ^ r; }
}
impl Not for IMask32x8 {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self { Self { lo: !self.lo, hi: !self.hi } }
}

impl fmt::Debug for IMask32x8 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "IMask32x8({:08b})", self.bitmask())
    }
}
