// crates/mid-math/src/wide/int/scalar/imask8.rs
//! Scalar fallback 8-lane integer comparison mask (for i16x8 / u16x8).

use core::fmt;
use core::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not};

/// 8-lane integer comparison mask — scalar fallback.
/// Lane value: `u16::MAX` (0xFFFF) = true, `0` = false.
#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(C, align(16))]
pub struct IMask8(pub(crate) [u16; 8]);

impl IMask8 {
    pub const FALSE: Self = IMask8([0u16; 8]);
    pub const TRUE:  Self = IMask8([u16::MAX; 8]);

    #[inline] pub fn any(self)  -> bool { self.0.iter().any(|&x| x != 0) }
    #[inline] pub fn all(self)  -> bool { self.0.iter().all(|&x| x != 0) }
    #[inline] pub fn none(self) -> bool { self.0.iter().all(|&x| x == 0) }

    #[inline]
    pub fn bitmask(self) -> u8 {
        self.0.iter().enumerate().fold(0u8, |acc, (i, &x)| {
            acc | (if x != 0 { 1u8 } else { 0 }) << i
        })
    }
}

impl BitAnd for IMask8 {
    type Output = Self;
    fn bitand(self, r: Self) -> Self {
        let mut out = [0u16; 8];
        for i in 0..8 { out[i] = self.0[i] & r.0[i]; }
        IMask8(out)
    }
}
impl BitAndAssign for IMask8 { fn bitand_assign(&mut self, r: Self) { *self = *self & r; } }
impl BitOr for IMask8 {
    type Output = Self;
    fn bitor(self, r: Self) -> Self {
        let mut out = [0u16; 8];
        for i in 0..8 { out[i] = self.0[i] | r.0[i]; }
        IMask8(out)
    }
}
impl BitOrAssign for IMask8 { fn bitor_assign(&mut self, r: Self) { *self = *self | r; } }
impl BitXor for IMask8 {
    type Output = Self;
    fn bitxor(self, r: Self) -> Self {
        let mut out = [0u16; 8];
        for i in 0..8 { out[i] = self.0[i] ^ r.0[i]; }
        IMask8(out)
    }
}
impl BitXorAssign for IMask8 { fn bitxor_assign(&mut self, r: Self) { *self = *self ^ r; } }
impl Not for IMask8 {
    type Output = Self;
    fn not(self) -> Self {
        let mut out = [0u16; 8];
        for i in 0..8 { out[i] = !self.0[i]; }
        IMask8(out)
    }
}
impl fmt::Debug for IMask8 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let bits: Vec<bool> = self.0.iter().map(|&x| x != 0).collect();
        write!(f, "IMask8({:?})", bits)
    }
                                                                   }
