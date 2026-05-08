// crates/mid-math/src/wide/int/scalar/imask4.rs
//! Scalar fallback 4-lane integer comparison mask.

use core::fmt;
use core::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not};

/// 4-lane integer comparison mask — scalar fallback.
/// Lane `0u32` = false, `u32::MAX` = true.
#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(C, align(16))]
pub struct IMask4(pub(crate) [u32; 4]);

impl IMask4 {
    pub const FALSE: Self = IMask4([0u32; 4]);
    pub const TRUE:  Self = IMask4([u32::MAX; 4]);

    #[inline]
    pub fn any(self)    -> bool { self.0.iter().any(|&x| x != 0) }
    #[inline]
    pub fn all(self)    -> bool { self.0.iter().all(|&x| x != 0) }
    #[inline]
    pub fn none(self)   -> bool { self.0.iter().all(|&x| x == 0) }
    #[inline]
    pub fn bitmask(self) -> u32 {
        (if self.0[0] != 0 { 1 } else { 0 })
        | (if self.0[1] != 0 { 2 } else { 0 })
        | (if self.0[2] != 0 { 4 } else { 0 })
        | (if self.0[3] != 0 { 8 } else { 0 })
    }
}

impl BitAnd for IMask4 {
    type Output = Self;
    fn bitand(self, r: Self) -> Self {
        IMask4([self.0[0] & r.0[0], self.0[1] & r.0[1], self.0[2] & r.0[2], self.0[3] & r.0[3]])
    }
}
impl BitAndAssign for IMask4 { fn bitand_assign(&mut self, r: Self) { *self = *self & r; } }
impl BitOr for IMask4 {
    type Output = Self;
    fn bitor(self, r: Self) -> Self {
        IMask4([self.0[0] | r.0[0], self.0[1] | r.0[1], self.0[2] | r.0[2], self.0[3] | r.0[3]])
    }
}
impl BitOrAssign for IMask4 { fn bitor_assign(&mut self, r: Self) { *self = *self | r; } }
impl BitXor for IMask4 {
    type Output = Self;
    fn bitxor(self, r: Self) -> Self {
        IMask4([self.0[0] ^ r.0[0], self.0[1] ^ r.0[1], self.0[2] ^ r.0[2], self.0[3] ^ r.0[3]])
    }
}
impl BitXorAssign for IMask4 { fn bitxor_assign(&mut self, r: Self) { *self = *self ^ r; } }
impl Not for IMask4 {
    type Output = Self;
    fn not(self) -> Self {
        IMask4([!self.0[0], !self.0[1], !self.0[2], !self.0[3]])
    }
}

impl fmt::Debug for IMask4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "IMask4({}, {}, {}, {})",
            self.0[0] != 0, self.0[1] != 0, self.0[2] != 0, self.0[3] != 0)
    }
      }
