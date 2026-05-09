// crates/mid-math/src/wide/int/scalar/imask16.rs
//! Scalar fallback 16-lane integer comparison mask (for i8x16 / u8x16).

use core::fmt;
use core::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not};

/// 16-lane integer comparison mask — scalar fallback.
/// Lane value: `u8::MAX` (0xFF) = true, `0` = false.
#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(C, align(16))]
pub struct IMask16(pub(crate) [u8; 16]);

impl IMask16 {
    pub const FALSE: Self = IMask16([0u8; 16]);
    pub const TRUE:  Self = IMask16([u8::MAX; 16]);

    #[inline] pub fn any(self)  -> bool { self.0.iter().any(|&x| x != 0) }
    #[inline] pub fn all(self)  -> bool { self.0.iter().all(|&x| x != 0) }
    #[inline] pub fn none(self) -> bool { self.0.iter().all(|&x| x == 0) }

    #[inline]
    pub fn bitmask(self) -> u16 {
        self.0.iter().enumerate().fold(0u16, |acc, (i, &x)| {
            acc | (if x != 0 { 1u16 } else { 0 }) << i
        })
    }

    #[inline]
    pub fn count_true(self) -> u32 { self.bitmask().count_ones() }
}

impl BitAnd for IMask16 {
    type Output = Self;
    fn bitand(self, r: Self) -> Self {
        let mut out = [0u8; 16];
        for i in 0..16 { out[i] = self.0[i] & r.0[i]; }
        IMask16(out)
    }
}
impl BitAndAssign for IMask16 { fn bitand_assign(&mut self, r: Self) { *self = *self & r; } }
impl BitOr for IMask16 {
    type Output = Self;
    fn bitor(self, r: Self) -> Self {
        let mut out = [0u8; 16];
        for i in 0..16 { out[i] = self.0[i] | r.0[i]; }
        IMask16(out)
    }
}
impl BitOrAssign for IMask16 { fn bitor_assign(&mut self, r: Self) { *self = *self | r; } }
impl BitXor for IMask16 {
    type Output = Self;
    fn bitxor(self, r: Self) -> Self {
        let mut out = [0u8; 16];
        for i in 0..16 { out[i] = self.0[i] ^ r.0[i]; }
        IMask16(out)
    }
}
impl BitXorAssign for IMask16 { fn bitxor_assign(&mut self, r: Self) { *self = *self ^ r; } }
impl Not for IMask16 {
    type Output = Self;
    fn not(self) -> Self {
        let mut out = [0u8; 16];
        for i in 0..16 { out[i] = !self.0[i]; }
        IMask16(out)
    }
}
impl fmt::Debug for IMask16 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "IMask16({:016b})", self.bitmask())
    }
                                     }
