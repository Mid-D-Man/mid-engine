// crates/mid-math/src/wide/int/avx2/imask16x16.rs
//! 16-lane integer comparison mask for i16x16/u16x16.
//!
//! Each 16-bit lane: 0xFFFF = true, 0x0000 = false. Always compiled on
//! x86/x86_64, not gated on the `avx2` target feature. Storage is two
//! portable `IMask8` halves, never a raw `__m256i` -- same reasoning as
//! `IMask32x8` (see that file's doc comment): a pure bitwise/reduction
//! mask type gets no real benefit from a genuine 256-bit instruction over
//! two independent 128-bit ones, so there is no dispatch here at all.

use core::fmt;
use core::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not};

use crate::wide::int::sse2::imask8::IMask8;

/// 16-lane integer comparison mask. Lane i: `0xFFFF` = true, `0x0000` = false.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct IMask16x16 {
    pub(crate) lo: IMask8,
    pub(crate) hi: IMask8,
}

impl IMask16x16 {
    pub const FALSE: Self = Self { lo: IMask8::FALSE, hi: IMask8::FALSE };
    pub const TRUE: Self = Self { lo: IMask8::TRUE, hi: IMask8::TRUE };

    #[inline]
    pub(crate) fn from_halves(lo: IMask8, hi: IMask8) -> Self { Self { lo, hi } }

    #[inline]
    pub fn any(self) -> bool { self.lo.any() || self.hi.any() }
    #[inline]
    pub fn all(self) -> bool { self.lo.all() && self.hi.all() }
    #[inline]
    pub fn none(self) -> bool { self.lo.none() && self.hi.none() }

    /// Packed 16-bit bitmask -- one bit per 16-bit lane, low half in bits 0-7.
    #[inline]
    pub fn bitmask(self) -> u16 {
        (self.lo.bitmask() as u16) | ((self.hi.bitmask() as u16) << 8)
    }

    #[inline]
    pub fn count_true(self) -> u32 { self.bitmask().count_ones() }
}

impl BitAnd for IMask16x16 { type Output = Self; #[inline(always)] fn bitand(self, r: Self) -> Self { Self { lo: self.lo & r.lo, hi: self.hi & r.hi } } }
impl BitAndAssign for IMask16x16 { #[inline(always)] fn bitand_assign(&mut self, r: Self) { *self = *self & r; } }
impl BitOr for IMask16x16 { type Output = Self; #[inline(always)] fn bitor(self, r: Self) -> Self { Self { lo: self.lo | r.lo, hi: self.hi | r.hi } } }
impl BitOrAssign for IMask16x16 { #[inline(always)] fn bitor_assign(&mut self, r: Self) { *self = *self | r; } }
impl BitXor for IMask16x16 { type Output = Self; #[inline(always)] fn bitxor(self, r: Self) -> Self { Self { lo: self.lo ^ r.lo, hi: self.hi ^ r.hi } } }
impl BitXorAssign for IMask16x16 { #[inline(always)] fn bitxor_assign(&mut self, r: Self) { *self = *self ^ r; } }
impl Not for IMask16x16 { type Output = Self; #[inline(always)] fn not(self) -> Self { Self { lo: !self.lo, hi: !self.hi } } }

impl fmt::Debug for IMask16x16 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "IMask16x16({:016b})", self.bitmask())
    }
}
