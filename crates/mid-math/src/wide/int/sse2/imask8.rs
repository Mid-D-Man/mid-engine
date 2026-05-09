// crates/mid-math/src/wide/int/sse2/imask8.rs
//! 8-lane integer comparison mask for i16x8/u16x8 — SSE2, x86 / x86_64.
//! Each 16-bit lane: 0xFFFF = true, 0x0000 = false.
//! Never constructed directly — always produced by i16x8/u16x8 comparisons.

use core::fmt;
use core::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not};

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

#[repr(C)]
union UnionCast { i: [i16; 8], v: IMask8 }

/// 8-lane integer comparison mask. 16 bytes, 16-byte aligned.
/// Backed by `__m128i`. Lane i: `0xFFFF` = true, `0x0000` = false.
/// Use [`i16x8::blend`] / [`u16x8::blend`] for branchless selection.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct IMask8(pub(crate) __m128i);

impl IMask8 {
    /// All lanes false.
    pub const FALSE: Self = unsafe { UnionCast { i: [0; 8] }.v };
    /// All lanes true.
    pub const TRUE: Self  = unsafe { UnionCast { i: [-1; 8] }.v };

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

    /// Packed 8-bit bitmask — one bit per 16-bit lane.
    ///
    /// A true 16-bit lane produces two consecutive set bits in `_mm_movemask_epi8`.
    /// We extract one representative bit per lane (bit `2*i` for lane `i`).
    #[inline]
    pub fn bitmask(self) -> u8 {
        unsafe {
            let m = _mm_movemask_epi8(self.0) as u32;
            // Each true lane contributes 2 identical bits. Extract bit 0 of each pair.
            (0..8u32).fold(0u8, |acc, i| acc | (((m >> (2 * i)) & 1) as u8) << i)
        }
    }
}

impl BitAnd for IMask8 {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, r: Self) -> Self { IMask8(unsafe { _mm_and_si128(self.0, r.0) }) }
}
impl BitAndAssign for IMask8 {
    #[inline(always)] fn bitand_assign(&mut self, r: Self) { *self = *self & r; }
}
impl BitOr for IMask8 {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, r: Self) -> Self { IMask8(unsafe { _mm_or_si128(self.0, r.0) }) }
}
impl BitOrAssign for IMask8 {
    #[inline(always)] fn bitor_assign(&mut self, r: Self) { *self = *self | r; }
}
impl BitXor for IMask8 {
    type Output = Self;
    #[inline(always)]
    fn bitxor(self, r: Self) -> Self { IMask8(unsafe { _mm_xor_si128(self.0, r.0) }) }
}
impl BitXorAssign for IMask8 {
    #[inline(always)] fn bitxor_assign(&mut self, r: Self) { *self = *self ^ r; }
}
impl Not for IMask8 {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        unsafe {
            let ones = _mm_cmpeq_epi16(self.0, self.0); // all-ones trick
            IMask8(_mm_xor_si128(self.0, ones))
        }
    }
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
