// crates/mid-math/src/wide/int/neon/imask4.rs
//! 4-lane integer comparison mask — NEON, aarch64.
//!
//! Backed by `uint32x4_t`.
//! Lane value: `0xFFFF_FFFF` = true, `0x0000_0000` = false.
//!
//! Produced by i32x4 and u32x4 comparison operations.
//! NEON comparison intrinsics return uint32x4_t directly — no reinterpret needed.
//!
//! Key NEON advantages vs SSE2:
//!   vminvq_u32  — single UMINV instruction for `all()` check
//!   vmaxvq_u32  — single UMAXV instruction for `any()` check
//!   vmvnq_u32   — direct bitwise NOT (no XOR-with-ones trick)
//!   vbslq_s32   — single BSL for blend (vs AND+ANDNOT+OR in SSE2)

use core::arch::aarch64::*;
use core::fmt;
use core::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not};

#[repr(C)]
union UnionCast { u: [u32; 4], v: IMask4 }

/// 4-lane integer comparison mask. 16 bytes, 16-byte aligned.
/// Backed by `uint32x4_t`. Lane: `0xFFFF_FFFF` = true, `0` = false.
///
/// Never construct directly — produced by [`i32x4`][super::i32x4::i32x4] or
/// [`u32x4`][super::u32x4::u32x4] comparison methods.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct IMask4(pub(crate) uint32x4_t);

impl IMask4 {
    pub const FALSE: Self = unsafe { UnionCast { u: [0u32; 4] }.v };
    pub const TRUE:  Self = unsafe { UnionCast { u: [u32::MAX; 4] }.v };

    /// True if any lane is set. Uses single `UMAXV` instruction.
    #[inline]
    pub fn any(self) -> bool {
        unsafe { vmaxvq_u32(self.0) != 0 }
    }

    /// True if all lanes are set. Uses single `UMINV` instruction.
    #[inline]
    pub fn all(self) -> bool {
        unsafe { vminvq_u32(self.0) != 0 }
    }

    #[inline]
    pub fn none(self) -> bool { !self.any() }

    /// 4-bit bitmask — bit i set if lane i is true.
    ///
    /// Shifts each lane right by 31 to isolate the MSB (0 or 1),
    /// then assembles into a scalar via lane extraction.
    #[inline]
    pub fn bitmask(self) -> u32 {
        unsafe {
            let s = vshrq_n_u32(self.0, 31);
            vgetq_lane_u32::<0>(s)
                | (vgetq_lane_u32::<1>(s) << 1)
                | (vgetq_lane_u32::<2>(s) << 2)
                | (vgetq_lane_u32::<3>(s) << 3)
        }
    }
}

impl BitAnd for IMask4 {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, r: Self) -> Self { IMask4(unsafe { vandq_u32(self.0, r.0) }) }
}
impl BitAndAssign for IMask4 {
    #[inline(always)] fn bitand_assign(&mut self, r: Self) { *self = *self & r; }
}
impl BitOr for IMask4 {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, r: Self) -> Self { IMask4(unsafe { vorrq_u32(self.0, r.0) }) }
}
impl BitOrAssign for IMask4 {
    #[inline(always)] fn bitor_assign(&mut self, r: Self) { *self = *self | r; }
}
impl BitXor for IMask4 {
    type Output = Self;
    #[inline(always)]
    fn bitxor(self, r: Self) -> Self { IMask4(unsafe { veorq_u32(self.0, r.0) }) }
}
impl BitXorAssign for IMask4 {
    #[inline(always)] fn bitxor_assign(&mut self, r: Self) { *self = *self ^ r; }
}
/// Direct `vmvnq_u32` — no XOR-with-ones trick needed unlike SSE2.
impl Not for IMask4 {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self { IMask4(unsafe { vmvnq_u32(self.0) }) }
}

impl PartialEq for IMask4 {
    #[inline]
    fn eq(&self, r: &Self) -> bool { self.bitmask() == r.bitmask() }
}
impl Eq for IMask4 {}

impl fmt::Debug for IMask4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let b = self.bitmask();
        write!(f, "IMask4({}, {}, {}, {})",
            b & 1 != 0, b >> 1 & 1 != 0, b >> 2 & 1 != 0, b >> 3 & 1 != 0)
    }
  }
