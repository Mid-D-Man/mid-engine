// crates/mid-math/src/bvec/bvec2.rs
//! 2-component boolean mask.

use core::fmt;
use core::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not};

/// 2-component boolean mask. 2 bytes, align 1.
///
/// Used for element-wise select operations in geometry/intersect modules.
///
/// Bit layout of `bitmask()`: bit 0 = x, bit 1 = y.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(C)]
pub struct BVec2 {
    pub x: bool,
    pub y: bool,
}

impl BVec2 {
    /// All-false mask.
    pub const FALSE: Self = Self { x: false, y: false };
    /// All-true mask.
    pub const TRUE:  Self = Self { x: true,  y: true  };

    /// Creates a new mask from two booleans.
    #[inline(always)]
    pub const fn new(x: bool, y: bool) -> Self { Self { x, y } }

    /// Creates a mask with all elements set to `v`.
    #[inline(always)]
    pub const fn splat(v: bool) -> Self { Self { x: v, y: v } }

    /// Returns `true` if any element is `true`.
    #[inline]
    pub fn any(self) -> bool { self.x || self.y }

    /// Returns `true` if all elements are `true`.
    #[inline]
    pub fn all(self) -> bool { self.x && self.y }

    /// Returns a packed bitmask: bit 0 = x, bit 1 = y.
    ///
    /// A `true` element produces a `1` bit, `false` a `0` bit.
    #[inline]
    pub fn bitmask(self) -> u32 {
        (self.x as u32) | ((self.y as u32) << 1)
    }

    /// Returns the value at `index`. Panics if `index >= 2`.
    #[inline]
    pub fn test(self, index: usize) -> bool {
        match index {
            0 => self.x,
            1 => self.y,
            _ => panic!("BVec2::test — index {} out of bounds (max 1)", index),
        }
    }

    /// Creates a mask from an array `[x, y]`.
    #[inline]
    pub const fn from_array(a: [bool; 2]) -> Self { Self::new(a[0], a[1]) }

    /// Converts the mask to `[x, y]`.
    #[inline]
    pub const fn to_array(self) -> [bool; 2] { [self.x, self.y] }
}

// ── Bitwise operators ─────────────────────────────────────────────────────────

impl BitAnd for BVec2 {
    type Output = Self;
    #[inline] fn bitand(self, r: Self) -> Self { Self::new(self.x & r.x, self.y & r.y) }
}
impl BitAndAssign for BVec2 {
    #[inline] fn bitand_assign(&mut self, r: Self) { *self = *self & r; }
}
impl BitOr for BVec2 {
    type Output = Self;
    #[inline] fn bitor(self, r: Self) -> Self { Self::new(self.x | r.x, self.y | r.y) }
}
impl BitOrAssign for BVec2 {
    #[inline] fn bitor_assign(&mut self, r: Self) { *self = *self | r; }
}
impl BitXor for BVec2 {
    type Output = Self;
    #[inline] fn bitxor(self, r: Self) -> Self { Self::new(self.x ^ r.x, self.y ^ r.y) }
}
impl BitXorAssign for BVec2 {
    #[inline] fn bitxor_assign(&mut self, r: Self) { *self = *self ^ r; }
}
impl Not for BVec2 {
    type Output = Self;
    #[inline] fn not(self) -> Self { Self::new(!self.x, !self.y) }
}

// ── Conversions ───────────────────────────────────────────────────────────────

impl From<[bool; 2]> for BVec2 {
    #[inline] fn from(a: [bool; 2]) -> Self { Self::from_array(a) }
}
impl From<BVec2> for [bool; 2] {
    #[inline] fn from(v: BVec2) -> Self { v.to_array() }
}
impl From<(bool, bool)> for BVec2 {
    #[inline] fn from(t: (bool, bool)) -> Self { Self::new(t.0, t.1) }
}
impl From<BVec2> for (bool, bool) {
    #[inline] fn from(v: BVec2) -> Self { (v.x, v.y) }
}

// ── Display ───────────────────────────────────────────────────────────────────

impl fmt::Debug for BVec2 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BVec2({}, {})", self.x, self.y)
    }
}
impl fmt::Display for BVec2 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}, {}]", self.x, self.y)
    }
  }
