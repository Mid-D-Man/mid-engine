// crates/mid-math/src/wide/float/neon/mask4.rs
//! 4-lane float comparison mask — NEON, aarch64.
//!
//! Backed by `uint32x4_t` — comparison intrinsics produce this directly,
//! no float-as-int reinterpretation needed (unlike SSE2 which returns __m128).
//! Lane value: `0xFFFF_FFFF` = true, `0x0000_0000` = false.

use core::arch::aarch64::*;
use core::fmt;
use core::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not};

#[repr(C)]
union UCast { u: [u32; 4], v: Mask4 }

/// 4-lane float comparison mask. 16 bytes, 16-byte aligned. Backed by `uint32x4_t`.
///
/// Produced by [`f32x4`] comparison methods. Never construct directly.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct Mask4(pub(crate) uint32x4_t);

impl Mask4 {
    pub const FALSE: Self = unsafe { UCast { u: [0u32; 4] }.v };
    pub const TRUE:  Self = unsafe { UCast { u: [u32::MAX; 4] }.v };

    /// True if any lane is set (non-zero).
    ///
    /// `vmaxvq_u32` — single AArch64 UMAXV instruction.
    #[inline]
    pub fn any(self) -> bool { unsafe { vmaxvq_u32(self.0) != 0 } }

    /// True if all lanes are set.
    ///
    /// `vminvq_u32` — single AArch64 UMINV instruction.
    #[inline]
    pub fn all(self) -> bool { unsafe { vminvq_u32(self.0) != 0 } }

    #[inline]
    pub fn none(self) -> bool { !self.any() }

    /// 4-bit mask — bit i set if lane i is true.
    #[inline]
    pub fn bitmask(self) -> u32 {
        unsafe {
            // Shift each lane right by 31 to extract the sign bit (0 or 1).
            let s = vshrq_n_u32(self.0, 31);
            vgetq_lane_u32::<0>(s)
                | (vgetq_lane_u32::<1>(s) << 1)
                | (vgetq_lane_u32::<2>(s) << 2)
                | (vgetq_lane_u32::<3>(s) << 3)
        }
    }
}

impl BitAnd for Mask4 {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, r: Self) -> Self { Mask4(unsafe { vandq_u32(self.0, r.0) }) }
}
impl BitAndAssign for Mask4 { #[inline(always)] fn bitand_assign(&mut self, r: Self) { *self = *self & r; } }

impl BitOr for Mask4 {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, r: Self) -> Self { Mask4(unsafe { vorrq_u32(self.0, r.0) }) }
}
impl BitOrAssign for Mask4 { #[inline(always)] fn bitor_assign(&mut self, r: Self) { *self = *self | r; } }

impl BitXor for Mask4 {
    type Output = Self;
    #[inline(always)]
    fn bitxor(self, r: Self) -> Self { Mask4(unsafe { veorq_u32(self.0, r.0) }) }
}
impl BitXorAssign for Mask4 { #[inline(always)] fn bitxor_assign(&mut self, r: Self) { *self = *self ^ r; } }

impl Not for Mask4 {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self { Mask4(unsafe { vmvnq_u32(self.0) }) }
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
