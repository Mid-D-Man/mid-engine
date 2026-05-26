// crates/mid-math/src/wide/int/neon/imask8.rs
//! 8-lane integer comparison mask — NEON, aarch64.
//!
//! Backed by `uint16x8_t`.
//! Lane value: `0xFFFF` = true, `0x0000` = false.
//!
//! Produced by i16x8 and u16x8 comparison operations.
//!
//! Key ops:
//!   vmaxvq_u16 — single UMAXV for `any()`
//!   vminvq_u16 — single UMINV for `all()`
//!   vmvnq_u16  — direct NOT

use core::arch::aarch64::*;
use core::fmt;
use core::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not};

#[repr(C)]
union UnionCast { u: [u16; 8], v: IMask8 }

/// 8-lane integer comparison mask. 16 bytes, 16-byte aligned.
/// Backed by `uint16x8_t`. Lane: `0xFFFF` = true, `0x0000` = false.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct IMask8(pub(crate) uint16x8_t);

impl IMask8 {
    pub const FALSE: Self = unsafe { UnionCast { u: [0u16; 8] }.v };
    pub const TRUE:  Self = unsafe { UnionCast { u: [u16::MAX; 8] }.v };

    /// True if any lane is set. Single `UMAXV` instruction.
    #[inline]
    pub fn any(self) -> bool {
        unsafe { vmaxvq_u16(self.0) != 0 }
    }

    /// True if all lanes are set. Single `UMINV` instruction.
    #[inline]
    pub fn all(self) -> bool {
        unsafe { vminvq_u16(self.0) != 0 }
    }

    #[inline]
    pub fn none(self) -> bool { !self.any() }

    /// 8-bit bitmask — bit i set if lane i is true.
    ///
    /// Shifts each lane right by 15 to isolate MSB, then extracts.
    #[inline]
    pub fn bitmask(self) -> u8 {
        unsafe {
            let s = vshrq_n_u16(self.0, 15); // each lane: 0 or 1
            (vgetq_lane_u16::<0>(s) as u8)
                | ((vgetq_lane_u16::<1>(s) as u8) << 1)
                | ((vgetq_lane_u16::<2>(s) as u8) << 2)
                | ((vgetq_lane_u16::<3>(s) as u8) << 3)
                | ((vgetq_lane_u16::<4>(s) as u8) << 4)
                | ((vgetq_lane_u16::<5>(s) as u8) << 5)
                | ((vgetq_lane_u16::<6>(s) as u8) << 6)
                | ((vgetq_lane_u16::<7>(s) as u8) << 7)
        }
    }
}

impl BitAnd for IMask8 {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, r: Self) -> Self { IMask8(unsafe { vandq_u16(self.0, r.0) }) }
}
impl BitAndAssign for IMask8 { #[inline(always)] fn bitand_assign(&mut self, r: Self) { *self = *self & r; } }
impl BitOr for IMask8 {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, r: Self) -> Self { IMask8(unsafe { vorrq_u16(self.0, r.0) }) }
}
impl BitOrAssign for IMask8 { #[inline(always)] fn bitor_assign(&mut self, r: Self) { *self = *self | r; } }
impl BitXor for IMask8 {
    type Output = Self;
    #[inline(always)]
    fn bitxor(self, r: Self) -> Self { IMask8(unsafe { veorq_u16(self.0, r.0) }) }
}
impl BitXorAssign for IMask8 { #[inline(always)] fn bitxor_assign(&mut self, r: Self) { *self = *self ^ r; } }
impl Not for IMask8 {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self { IMask8(unsafe { vmvnq_u16(self.0) }) }
}

impl PartialEq for IMask8 {
    #[inline]
    fn eq(&self, r: &Self) -> bool { self.bitmask() == r.bitmask() }
}
impl Eq for IMask8 {}

impl fmt::Debug for IMask8 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let b = self.bitmask();
        let bits: Vec<bool> = (0..8).map(|i| (b >> i) & 1 != 0).collect();
        write!(f, "IMask8({:?})", bits)
    }
      }
