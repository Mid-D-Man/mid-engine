// crates/mid-math/src/wide/int/sse2/imask16.rs
//! 16-lane integer comparison mask for i8x16/u8x16 — SSE2, x86 / x86_64.
//! Each 8-bit lane: 0xFF = true, 0x00 = false.
//! bitmask() maps directly to _mm_movemask_epi8 — one bit per byte lane.

use core::fmt;
use core::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not};

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

#[repr(C)]
union UnionCast { i: [i8; 16], v: IMask16 }

/// 16-lane integer comparison mask. 16 bytes, 16-byte aligned.
/// Backed by `__m128i`. Lane i: `0xFF` = true, `0x00` = false.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct IMask16(pub(crate) __m128i);

impl IMask16 {
    /// All lanes false.
    pub const FALSE: Self = unsafe { UnionCast { i: [0; 16] }.v };
    /// All lanes true.
    pub const TRUE: Self  = unsafe { UnionCast { i: [-1; 16] }.v };

    /// True if any lane is set.
    #[inline]
    pub fn any(self) -> bool {
        unsafe { _mm_movemask_epi8(self.0) != 0 }
    }

    /// True if all lanes are set.
    #[inline]
    pub fn all(self) -> bool {
        unsafe { _mm_movemask_epi8(self.0) == 0xFFFF }
    }

    /// True if no lane is set.
    #[inline]
    pub fn none(self) -> bool {
        unsafe { _mm_movemask_epi8(self.0) == 0 }
    }

    /// 16-bit bitmask — one bit per 8-bit lane. Directly from `_mm_movemask_epi8`.
    #[inline]
    pub fn bitmask(self) -> u16 {
        unsafe { _mm_movemask_epi8(self.0) as u16 }
    }

    /// Number of true lanes.
    #[inline]
    pub fn count_true(self) -> u32 {
        self.bitmask().count_ones()
    }
}

impl BitAnd for IMask16 {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, r: Self) -> Self { IMask16(unsafe { _mm_and_si128(self.0, r.0) }) }
}
impl BitAndAssign for IMask16 {
    #[inline(always)] fn bitand_assign(&mut self, r: Self) { *self = *self & r; }
}
impl BitOr for IMask16 {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, r: Self) -> Self { IMask16(unsafe { _mm_or_si128(self.0, r.0) }) }
}
impl BitOrAssign for IMask16 {
    #[inline(always)] fn bitor_assign(&mut self, r: Self) { *self = *self | r; }
}
impl BitXor for IMask16 {
    type Output = Self;
    #[inline(always)]
    fn bitxor(self, r: Self) -> Self { IMask16(unsafe { _mm_xor_si128(self.0, r.0) }) }
}
impl BitXorAssign for IMask16 {
    #[inline(always)] fn bitxor_assign(&mut self, r: Self) { *self = *self ^ r; }
}
impl Not for IMask16 {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        unsafe {
            let ones = _mm_cmpeq_epi8(self.0, self.0);
            IMask16(_mm_xor_si128(self.0, ones))
        }
    }
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
