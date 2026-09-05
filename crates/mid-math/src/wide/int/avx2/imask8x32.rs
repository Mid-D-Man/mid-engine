// crates/mid-math/src/wide/int/avx2/imask8x32.rs
//! 32-lane integer comparison mask for i8x32/u8x32.
//!
//! Each 8-bit lane: 0xFF = true, 0x00 = false. Always compiled on
//! x86/x86_64, not gated on the `avx2` target feature. Storage is two
//! portable `IMask16` halves, never a raw `__m256i` -- same reasoning as
//! `IMask32x8` (see that file's doc comment): no dispatch needed for a
//! pure bitwise/reduction mask type.

use core::fmt;
use core::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not};

use crate::wide::int::sse2::imask16::IMask16;

/// 32-lane integer comparison mask. Lane i: `0xFF` = true, `0x00` = false.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct IMask8x32 {
    pub(crate) lo: IMask16,
    pub(crate) hi: IMask16,
}

impl IMask8x32 {
    pub const FALSE: Self = Self { lo: IMask16::FALSE, hi: IMask16::FALSE };
    pub const TRUE: Self = Self { lo: IMask16::TRUE, hi: IMask16::TRUE };

    #[inline]
    pub(crate) fn from_halves(lo: IMask16, hi: IMask16) -> Self { Self { lo, hi } }

    #[inline]
    pub fn any(self) -> bool { self.lo.any() || self.hi.any() }
    #[inline]
    pub fn all(self) -> bool { self.lo.all() && self.hi.all() }
    #[inline]
    pub fn none(self) -> bool { self.lo.none() && self.hi.none() }

    /// 32-bit bitmask -- one bit per 8-bit lane, low half in bits 0-15.
    #[inline]
    pub fn bitmask(self) -> u32 {
        (self.lo.bitmask() as u32) | ((self.hi.bitmask() as u32) << 16)
    }

    #[inline]
    pub fn count_true(self) -> u32 { self.bitmask().count_ones() }
}

impl BitAnd for IMask8x32 { type Output = Self; #[inline(always)] fn bitand(self, r: Self) -> Self { Self { lo: self.lo & r.lo, hi: self.hi & r.hi } } }
impl BitAndAssign for IMask8x32 { #[inline(always)] fn bitand_assign(&mut self, r: Self) { *self = *self & r; } }
impl BitOr for IMask8x32 { type Output = Self; #[inline(always)] fn bitor(self, r: Self) -> Self { Self { lo: self.lo | r.lo, hi: self.hi | r.hi } } }
impl BitOrAssign for IMask8x32 { #[inline(always)] fn bitor_assign(&mut self, r: Self) { *self = *self | r; } }
impl BitXor for IMask8x32 { type Output = Self; #[inline(always)] fn bitxor(self, r: Self) -> Self { Self { lo: self.lo ^ r.lo, hi: self.hi ^ r.hi } } }
impl BitXorAssign for IMask8x32 { #[inline(always)] fn bitxor_assign(&mut self, r: Self) { *self = *self ^ r; } }
impl Not for IMask8x32 { type Output = Self; #[inline(always)] fn not(self) -> Self { Self { lo: !self.lo, hi: !self.hi } } }

impl fmt::Debug for IMask8x32 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "IMask8x32({:032b})", self.bitmask())
    }
}
