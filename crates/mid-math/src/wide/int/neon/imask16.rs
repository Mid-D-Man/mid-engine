// crates/mid-math/src/wide/int/neon/imask16.rs
//! 16-lane integer comparison mask — NEON, aarch64.
//!
//! Backed by `uint8x16_t`.
//! Lane value: `0xFF` = true, `0x00` = false.
//!
//! Produced by i8x16 and u8x16 comparison operations.
//!
//! Key ops:
//!   vmaxvq_u8  — single UMAXV for `any()`
//!   vminvq_u8  — single UMINV for `all()`
//!   vmvnq_u8   — direct NOT

use core::arch::aarch64::*;
use core::fmt;
use core::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not};

#[repr(C)]
union UnionCast { u: [u8; 16], v: IMask16 }

/// 16-lane integer comparison mask. 16 bytes, 16-byte aligned.
/// Backed by `uint8x16_t`. Lane: `0xFF` = true, `0x00` = false.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct IMask16(pub(crate) uint8x16_t);

impl IMask16 {
    pub const FALSE: Self = unsafe { UnionCast { u: [0u8; 16] }.v };
    pub const TRUE:  Self = unsafe { UnionCast { u: [u8::MAX; 16] }.v };

    /// True if any lane is set. Single `UMAXV` instruction.
    #[inline]
    pub fn any(self) -> bool {
        unsafe { vmaxvq_u8(self.0) != 0 }
    }

    /// True if all lanes are set. Single `UMINV` instruction.
    #[inline]
    pub fn all(self) -> bool {
        unsafe { vminvq_u8(self.0) != 0 }
    }

    #[inline]
    pub fn none(self) -> bool { !self.any() }

    /// 16-bit bitmask — bit i set if lane i is true.
    ///
    /// Shifts each byte right by 7 to isolate MSB, then extracts all 16 lanes.
    #[inline]
    pub fn bitmask(self) -> u16 {
        unsafe {
            let s = vshrq_n_u8(self.0, 7); // each byte: 0 or 1
            (vgetq_lane_u8::<0>(s)  as u16)
                | ((vgetq_lane_u8::<1>(s)  as u16) << 1)
                | ((vgetq_lane_u8::<2>(s)  as u16) << 2)
                | ((vgetq_lane_u8::<3>(s)  as u16) << 3)
                | ((vgetq_lane_u8::<4>(s)  as u16) << 4)
                | ((vgetq_lane_u8::<5>(s)  as u16) << 5)
                | ((vgetq_lane_u8::<6>(s)  as u16) << 6)
                | ((vgetq_lane_u8::<7>(s)  as u16) << 7)
                | ((vgetq_lane_u8::<8>(s)  as u16) << 8)
                | ((vgetq_lane_u8::<9>(s)  as u16) << 9)
                | ((vgetq_lane_u8::<10>(s) as u16) << 10)
                | ((vgetq_lane_u8::<11>(s) as u16) << 11)
                | ((vgetq_lane_u8::<12>(s) as u16) << 12)
                | ((vgetq_lane_u8::<13>(s) as u16) << 13)
                | ((vgetq_lane_u8::<14>(s) as u16) << 14)
                | ((vgetq_lane_u8::<15>(s) as u16) << 15)
        }
    }

    #[inline]
    pub fn count_true(self) -> u32 { self.bitmask().count_ones() }
}

impl BitAnd for IMask16 {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, r: Self) -> Self { IMask16(unsafe { vandq_u8(self.0, r.0) }) }
}
impl BitAndAssign for IMask16 { #[inline(always)] fn bitand_assign(&mut self, r: Self) { *self = *self & r; } }
impl BitOr for IMask16 {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, r: Self) -> Self { IMask16(unsafe { vorrq_u8(self.0, r.0) }) }
}
impl BitOrAssign for IMask16 { #[inline(always)] fn bitor_assign(&mut self, r: Self) { *self = *self | r; } }
impl BitXor for IMask16 {
    type Output = Self;
    #[inline(always)]
    fn bitxor(self, r: Self) -> Self { IMask16(unsafe { veorq_u8(self.0, r.0) }) }
}
impl BitXorAssign for IMask16 { #[inline(always)] fn bitxor_assign(&mut self, r: Self) { *self = *self ^ r; } }
impl Not for IMask16 {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self { IMask16(unsafe { vmvnq_u8(self.0) }) }
}

impl PartialEq for IMask16 {
    #[inline]
    fn eq(&self, r: &Self) -> bool { self.bitmask() == r.bitmask() }
}
impl Eq for IMask16 {}

impl fmt::Debug for IMask16 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "IMask16({:016b})", self.bitmask())
    }
                   }
