// crates/mid-math/src/bvec/bvec3.rs
//! 3-component boolean mask.

use core::fmt;
use core::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not};

/// 3-component boolean mask. 3 bytes, align 1.
///
/// Used for element-wise select on Vec3 results in geometry/intersect modules.
///
/// Bit layout of `bitmask()`: bit 0 = x, bit 1 = y, bit 2 = z.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(C)]
pub struct BVec3 {
    pub x: bool,
    pub y: bool,
    pub z: bool,
}

impl BVec3 {
    /// All-false mask.
    pub const FALSE: Self = Self { x: false, y: false, z: false };
    /// All-true mask.
    pub const TRUE:  Self = Self { x: true,  y: true,  z: true  };

    #[inline(always)]
    pub const fn new(x: bool, y: bool, z: bool) -> Self { Self { x, y, z } }

    #[inline(always)]
    pub const fn splat(v: bool) -> Self { Self { x: v, y: v, z: v } }

    /// Returns `true` if any element is `true`.
    #[inline]
    pub fn any(self) -> bool { self.x || self.y || self.z }

    /// Returns `true` if all elements are `true`.
    #[inline]
    pub fn all(self) -> bool { self.x && self.y && self.z }

    /// Returns a packed bitmask: bit 0 = x, bit 1 = y, bit 2 = z.
    #[inline]
    pub fn bitmask(self) -> u32 {
        (self.x as u32) | ((self.y as u32) << 1) | ((self.z as u32) << 2)
    }

    /// Returns the value at `index`. Panics if `index >= 3`.
    #[inline]
    pub fn test(self, index: usize) -> bool {
        match index {
            0 => self.x,
            1 => self.y,
            2 => self.z,
            _ => panic!("BVec3::test — index {} out of bounds (max 2)", index),
        }
    }

    #[inline]
    pub const fn from_array(a: [bool; 3]) -> Self { Self::new(a[0], a[1], a[2]) }

    #[inline]
    pub const fn to_array(self) -> [bool; 3] { [self.x, self.y, self.z] }
}

// ── Bitwise operators ─────────────────────────────────────────────────────────

impl BitAnd for BVec3 {
    type Output = Self;
    #[inline]
    fn bitand(self, r: Self) -> Self {
        Self::new(self.x & r.x, self.y & r.y, self.z & r.z)
    }
}
impl BitAndAssign for BVec3 {
    #[inline] fn bitand_assign(&mut self, r: Self) { *self = *self & r; }
}
impl BitOr for BVec3 {
    type Output = Self;
    #[inline]
    fn bitor(self, r: Self) -> Self {
        Self::new(self.x | r.x, self.y | r.y, self.z | r.z)
    }
}
impl BitOrAssign for BVec3 {
    #[inline] fn bitor_assign(&mut self, r: Self) { *self = *self | r; }
}
impl BitXor for BVec3 {
    type Output = Self;
    #[inline]
    fn bitxor(self, r: Self) -> Self {
        Self::new(self.x ^ r.x, self.y ^ r.y, self.z ^ r.z)
    }
}
impl BitXorAssign for BVec3 {
    #[inline] fn bitxor_assign(&mut self, r: Self) { *self = *self ^ r; }
}
impl Not for BVec3 {
    type Output = Self;
    #[inline] fn not(self) -> Self { Self::new(!self.x, !self.y, !self.z) }
}

// ── Conversions ───────────────────────────────────────────────────────────────

impl From<[bool; 3]> for BVec3 {
    #[inline] fn from(a: [bool; 3]) -> Self { Self::from_array(a) }
}
impl From<BVec3> for [bool; 3] {
    #[inline] fn from(v: BVec3) -> Self { v.to_array() }
}
impl From<(bool, bool, bool)> for BVec3 {
    #[inline] fn from(t: (bool, bool, bool)) -> Self { Self::new(t.0, t.1, t.2) }
}
impl From<BVec3> for (bool, bool, bool) {
    #[inline] fn from(v: BVec3) -> Self { (v.x, v.y, v.z) }
}

// ── Display ───────────────────────────────────────────────────────────────────

impl fmt::Debug for BVec3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BVec3({}, {}, {})", self.x, self.y, self.z)
    }
}
impl fmt::Display for BVec3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}, {}, {}]", self.x, self.y, self.z)
    }
  }
