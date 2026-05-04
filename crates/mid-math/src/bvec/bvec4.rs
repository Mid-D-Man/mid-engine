// crates/mid-math/src/bvec/bvec4.rs
//! 4-component boolean mask.

use core::fmt;
use core::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not};

/// 4-component boolean mask. 4 bytes, align 1.
///
/// Used for element-wise select on Vec4/Quat results in geometry/intersect modules.
///
/// Bit layout of `bitmask()`: bit 0 = x, bit 1 = y, bit 2 = z, bit 3 = w.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(C)]
pub struct BVec4 {
    pub x: bool,
    pub y: bool,
    pub z: bool,
    pub w: bool,
}

impl BVec4 {
    /// All-false mask.
    pub const FALSE: Self = Self { x: false, y: false, z: false, w: false };
    /// All-true mask.
    pub const TRUE:  Self = Self { x: true,  y: true,  z: true,  w: true  };

    #[inline(always)]
    pub const fn new(x: bool, y: bool, z: bool, w: bool) -> Self { Self { x, y, z, w } }

    #[inline(always)]
    pub const fn splat(v: bool) -> Self { Self { x: v, y: v, z: v, w: v } }

    /// Returns `true` if any element is `true`.
    #[inline]
    pub fn any(self) -> bool { self.x || self.y || self.z || self.w }

    /// Returns `true` if all elements are `true`.
    #[inline]
    pub fn all(self) -> bool { self.x && self.y && self.z && self.w }

    /// Returns a packed bitmask: bit 0 = x, bit 1 = y, bit 2 = z, bit 3 = w.
    #[inline]
    pub fn bitmask(self) -> u32 {
        (self.x as u32)
            | ((self.y as u32) << 1)
            | ((self.z as u32) << 2)
            | ((self.w as u32) << 3)
    }

    /// Returns the value at `index`. Panics if `index >= 4`.
    #[inline]
    pub fn test(self, index: usize) -> bool {
        match index {
            0 => self.x,
            1 => self.y,
            2 => self.z,
            3 => self.w,
            _ => panic!("BVec4::test — index {} out of bounds (max 3)", index),
        }
    }

    #[inline]
    pub const fn from_array(a: [bool; 4]) -> Self { Self::new(a[0], a[1], a[2], a[3]) }

    #[inline]
    pub const fn to_array(self) -> [bool; 4] { [self.x, self.y, self.z, self.w] }
}

// ── Bitwise operators ─────────────────────────────────────────────────────────

impl BitAnd for BVec4 {
    type Output = Self;
    #[inline]
    fn bitand(self, r: Self) -> Self {
        Self::new(self.x & r.x, self.y & r.y, self.z & r.z, self.w & r.w)
    }
}
impl BitAndAssign for BVec4 {
    #[inline] fn bitand_assign(&mut self, r: Self) { *self = *self & r; }
}
impl BitOr for BVec4 {
    type Output = Self;
    #[inline]
    fn bitor(self, r: Self) -> Self {
        Self::new(self.x | r.x, self.y | r.y, self.z | r.z, self.w | r.w)
    }
}
impl BitOrAssign for BVec4 {
    #[inline] fn bitor_assign(&mut self, r: Self) { *self = *self | r; }
}
impl BitXor for BVec4 {
    type Output = Self;
    #[inline]
    fn bitxor(self, r: Self) -> Self {
        Self::new(self.x ^ r.x, self.y ^ r.y, self.z ^ r.z, self.w ^ r.w)
    }
}
impl BitXorAssign for BVec4 {
    #[inline] fn bitxor_assign(&mut self, r: Self) { *self = *self ^ r; }
}
impl Not for BVec4 {
    type Output = Self;
    #[inline] fn not(self) -> Self { Self::new(!self.x, !self.y, !self.z, !self.w) }
}

// ── Conversions ───────────────────────────────────────────────────────────────

impl From<[bool; 4]> for BVec4 {
    #[inline] fn from(a: [bool; 4]) -> Self { Self::from_array(a) }
}
impl From<BVec4> for [bool; 4] {
    #[inline] fn from(v: BVec4) -> Self { v.to_array() }
}
impl From<(bool, bool, bool, bool)> for BVec4 {
    #[inline] fn from(t: (bool, bool, bool, bool)) -> Self { Self::new(t.0, t.1, t.2, t.3) }
}
impl From<BVec4> for (bool, bool, bool, bool) {
    #[inline] fn from(v: BVec4) -> Self { (v.x, v.y, v.z, v.w) }
}

// ── Display ───────────────────────────────────────────────────────────────────

impl fmt::Debug for BVec4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BVec4({}, {}, {}, {})", self.x, self.y, self.z, self.w)
    }
}
impl fmt::Display for BVec4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}, {}, {}, {}]", self.x, self.y, self.z, self.w)
    }
      }
