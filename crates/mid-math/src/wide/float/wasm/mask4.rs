// crates/mid-math/src/wide/float/wasm/mask4.rs
//! 4-lane float comparison mask — WASM SIMD128.

use core::fmt;
use core::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not};

#[cfg(target_arch = "wasm32")]
use core::arch::wasm32::*;
#[cfg(target_arch = "wasm64")]
use core::arch::wasm64::*;

use crate::wasm::v128_from_f32x4;

/// 4-lane float comparison mask backed by `v128`.
///
/// Each lane: all-ones (0xFFFF_FFFF) = true, all-zeros = false.
/// Produced by `f32x4` or `Vec3x4` comparisons — never construct directly.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct Mask4(pub(crate) v128);

impl Mask4 {
    pub const FALSE: Self = Self(v128_from_f32x4([0.0; 4]));
    pub const TRUE: Self  = Self(v128_from_f32x4([f32::from_bits(0xFFFF_FFFF); 4]));

    #[inline] pub fn any(self)  -> bool { v128_any_true(self.0) }
    #[inline] pub fn all(self)  -> bool { i32x4_all_true(self.0) }
    #[inline] pub fn none(self) -> bool { !v128_any_true(self.0) }
    #[inline] pub fn bitmask(self) -> u32 { i32x4_bitmask(self.0) as u32 }

    /// Index (0-3) of the lowest-numbered true lane, or `None` if none are true.
    #[inline]
    pub fn first_set_lane(self) -> Option<u32> {
        let b = self.bitmask();
        if b == 0 { None } else { Some(b.trailing_zeros()) }
    }

    /// Number of true lanes (0-4).
    #[inline]
    pub fn count_set(self) -> u32 { self.bitmask().count_ones() }

    /// Iterate the indices (0-3) of true lanes, lowest to highest.
    #[inline]
    pub fn iter_set_lanes(self) -> Mask4LaneIter { Mask4LaneIter { bits: self.bitmask() } }

    #[inline(always)]
    pub(crate) fn from_v128(m: v128) -> Self { Mask4(m) }
}

/// Iterator over the true-lane indices of a [`Mask4`]. See [`Mask4::iter_set_lanes`].
pub struct Mask4LaneIter { bits: u32 }

impl Iterator for Mask4LaneIter {
    type Item = u32;
    #[inline]
    fn next(&mut self) -> Option<u32> {
        if self.bits == 0 { return None; }
        let idx = self.bits.trailing_zeros();
        self.bits &= self.bits - 1; // clear lowest set bit
        Some(idx)
    }
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.bits.count_ones() as usize;
        (n, Some(n))
    }
}
impl ExactSizeIterator for Mask4LaneIter {}

impl BitAnd for Mask4 {
    type Output = Self;
    #[inline(always)] fn bitand(self, r: Self) -> Self { Mask4(v128_and(self.0, r.0)) }
}
impl BitAndAssign for Mask4 { #[inline(always)] fn bitand_assign(&mut self, r: Self) { *self = *self & r; } }

impl BitOr for Mask4 {
    type Output = Self;
    #[inline(always)] fn bitor(self, r: Self) -> Self { Mask4(v128_or(self.0, r.0)) }
}
impl BitOrAssign for Mask4 { #[inline(always)] fn bitor_assign(&mut self, r: Self) { *self = *self | r; } }

impl BitXor for Mask4 {
    type Output = Self;
    #[inline(always)] fn bitxor(self, r: Self) -> Self { Mask4(v128_xor(self.0, r.0)) }
}
impl BitXorAssign for Mask4 { #[inline(always)] fn bitxor_assign(&mut self, r: Self) { *self = *self ^ r; } }

impl Not for Mask4 {
    type Output = Self;
    #[inline(always)] fn not(self) -> Self { Mask4(v128_not(self.0)) }
}

impl PartialEq for Mask4 { fn eq(&self, r: &Self) -> bool { self.bitmask() == r.bitmask() } }
impl Eq for Mask4 {}

impl fmt::Debug for Mask4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let b = self.bitmask();
        write!(f, "Mask4({}, {}, {}, {})",
            b & 1 != 0, b >> 1 & 1 != 0, b >> 2 & 1 != 0, b >> 3 & 1 != 0)
    }
    }
