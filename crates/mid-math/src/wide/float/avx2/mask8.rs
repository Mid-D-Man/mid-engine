// crates/mid-math/src/wide/float/avx2/mask8.rs
//! 8-lane float comparison mask for f32x8/Vec3x8.
//!
//! Each lane: all-ones (`f32::from_bits(0xFFFFFFFF)`) = true, all-zeros =
//! false. Never constructed directly — produced by `f32x8`/`Vec3x8`
//! comparisons. Always compiled on x86/x86_64, storage is two portable
//! `Mask4` halves, never a raw `__m256` — same reasoning as `IMask32x8`
//! (see that file's doc comment): no dispatch needed for a pure
//! bitwise/reduction mask type.

use core::fmt;
use core::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not};

use crate::wide::float::sse2::mask4::Mask4;

/// 8-lane float comparison mask.
#[derive(Clone, Copy)]
pub struct Mask8 {
    pub(crate) lo: Mask4,
    pub(crate) hi: Mask4,
}

impl Mask8 {
    pub const FALSE: Self = Self { lo: Mask4::FALSE, hi: Mask4::FALSE };
    pub const TRUE: Self = Self { lo: Mask4::TRUE, hi: Mask4::TRUE };

    #[inline(always)]
    pub(crate) fn from_halves(lo: Mask4, hi: Mask4) -> Self { Self { lo, hi } }

    #[inline]
    pub fn any(self) -> bool { self.lo.any() || self.hi.any() }
    #[inline]
    pub fn all(self) -> bool { self.lo.all() && self.hi.all() }
    #[inline]
    pub fn none(self) -> bool { self.lo.none() && self.hi.none() }

    /// Packed 8-bit bitmask -- one bit per lane, low half in bits 0-3.
    #[inline]
    pub fn bitmask(self) -> u8 {
        (self.lo.bitmask() as u8) | ((self.hi.bitmask() as u8) << 4)
    }
    #[inline]
    pub fn count_set(self) -> u32 { self.bitmask().count_ones() }
}

impl BitAnd for Mask8 { type Output = Self; #[inline(always)] fn bitand(self, r: Self) -> Self { Self { lo: self.lo & r.lo, hi: self.hi & r.hi } } }
impl BitAndAssign for Mask8 { #[inline(always)] fn bitand_assign(&mut self, r: Self) { *self = *self & r; } }
impl BitOr for Mask8 { type Output = Self; #[inline(always)] fn bitor(self, r: Self) -> Self { Self { lo: self.lo | r.lo, hi: self.hi | r.hi } } }
impl BitOrAssign for Mask8 { #[inline(always)] fn bitor_assign(&mut self, r: Self) { *self = *self | r; } }
impl BitXor for Mask8 { type Output = Self; #[inline(always)] fn bitxor(self, r: Self) -> Self { Self { lo: self.lo ^ r.lo, hi: self.hi ^ r.hi } } }
impl BitXorAssign for Mask8 { #[inline(always)] fn bitxor_assign(&mut self, r: Self) { *self = *self ^ r; } }
impl Not for Mask8 { type Output = Self; #[inline(always)] fn not(self) -> Self { Self { lo: !self.lo, hi: !self.hi } } }

impl fmt::Debug for Mask8 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Mask8({:08b})", self.bitmask())
    }
}
