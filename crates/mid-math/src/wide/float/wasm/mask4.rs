// crates/mid-math/src/wide/float/wasm/mask4.rs
//! 4-lane float comparison mask — WASM SIMD128.

use core::fmt;
use core::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not};
use core::arch::wasm32::*;

/// 4-lane float comparison mask. Backed by `v128`.
///
/// Each lane: all-ones (0xFFFFFFFF) = true, all-zeros = false.
/// Produced by `f32x4` or `Vec3x4` comparisons.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct Mask4(pub(crate) v128);

impl Mask4 {
    pub const FALSE: Self = Self(f32x4(0.0, 0.0, 0.0, 0.0));
    pub const TRUE: Self = Self(i32x4(
        u32::MAX as i32,
        u32::MAX as i32,
        u32::MAX as i32,
        u32::MAX as i32,
    ));

    /// True if any lane is set.
    #[inline]
    pub fn any(self) -> bool {
        v128_any_true(self.0)
    }

    /// True if all 4 lanes are set.
    #[inline]
    pub fn all(self) -> bool {
        // i32x4_all_true checks all lanes non-zero
        i32x4_all_true(self.0)
    }

    /// True if no lanes are set.
    #[inline]
    pub fn none(self) -> bool {
        !v128_any_true(self.0)
    }

    /// Extract 4-bit mask (bit i = lane i sign bit).
    #[inline]
    pub fn bitmask(self) -> u32 {
        i32x4_bitmask(self.0) as u32
    }

    #[inline(always)]
    pub(crate) fn from_v128(m: v128) -> Self { Mask4(m) }
}

impl BitAnd for Mask4 {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, r: Self) -> Self { Mask4(v128_and(self.0, r.0)) }
}
impl BitAndAssign for Mask4 { #[inline(always)] fn bitand_assign(&mut self, r: Self) { *self = *self & r; } }

impl BitOr for Mask4 {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, r: Self) -> Self { Mask4(v128_or(self.0, r.0)) }
}
impl BitOrAssign for Mask4 { #[inline(always)] fn bitor_assign(&mut self, r: Self) { *self = *self | r; } }

impl BitXor for Mask4 {
    type Output = Self;
    #[inline(always)]
    fn bitxor(self, r: Self) -> Self { Mask4(v128_xor(self.0, r.0)) }
}
impl BitXorAssign for Mask4 { #[inline(always)] fn bitxor_assign(&mut self, r: Self) { *self = *self ^ r; } }

impl Not for Mask4 {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self { Mask4(v128_not(self.0)) }
}

impl PartialEq for Mask4 {
    fn eq(&self, r: &Self) -> bool { self.bitmask() == r.bitmask() }
}
impl Eq for Mask4 {}

impl fmt::Debug for Mask4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let b = self.bitmask();
        write!(f, "Mask4({}, {}, {}, {})",
            b & 1 != 0, b >> 1 & 1 != 0, b >> 2 & 1 != 0, b >> 3 & 1 != 0)
    }
      }
