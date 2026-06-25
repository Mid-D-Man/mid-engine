// crates/mid-math/src/wide/float/sse2/mask4.rs
// 4-lane float comparison mask — SSE2, x86 / x86_64.

use core::fmt;
use core::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not};

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

#[repr(C)]
union UCast { f: [f32; 4], v: Mask4 }

/// 4-lane float comparison mask. 16 bytes, 16-byte aligned. Backed by `__m128`.
///
/// Each lane: all-ones (`f32::from_bits(0xFFFFFFFF)`) = true, all-zeros = false.
/// Never construct directly — produced by [`f32x4`] or [`Vec3x4`] comparisons.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct Mask4(pub(crate) __m128);

impl Mask4 {
    pub const FALSE: Self = unsafe { UCast { f: [0.0; 4] }.v };
    pub const TRUE: Self  = unsafe { UCast { f: [
        f32::from_bits(0xFFFF_FFFF),
        f32::from_bits(0xFFFF_FFFF),
        f32::from_bits(0xFFFF_FFFF),
        f32::from_bits(0xFFFF_FFFF),
    ] }.v };

    #[inline]
    pub fn any(self) -> bool { unsafe { _mm_movemask_ps(self.0) != 0 } }
    #[inline]
    pub fn all(self) -> bool { unsafe { _mm_movemask_ps(self.0) == 0b1111 } }
    #[inline]
    pub fn none(self) -> bool { unsafe { _mm_movemask_ps(self.0) == 0 } }
    #[inline]
    pub fn bitmask(self) -> u32 { unsafe { _mm_movemask_ps(self.0) as u32 } }

    #[allow(dead_code)]
    #[inline(always)]
    pub(crate) fn from_m128(m: __m128) -> Self { Mask4(m) }
}

impl BitAnd for Mask4 {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, r: Self) -> Self { Mask4(unsafe { _mm_and_ps(self.0, r.0) }) }
}
impl BitAndAssign for Mask4 { #[inline(always)] fn bitand_assign(&mut self, r: Self) { *self = *self & r; } }
impl BitOr for Mask4 {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, r: Self) -> Self { Mask4(unsafe { _mm_or_ps(self.0, r.0) }) }
}
impl BitOrAssign for Mask4 { #[inline(always)] fn bitor_assign(&mut self, r: Self) { *self = *self | r; } }
impl BitXor for Mask4 {
    type Output = Self;
    #[inline(always)]
    fn bitxor(self, r: Self) -> Self { Mask4(unsafe { _mm_xor_ps(self.0, r.0) }) }
}
impl BitXorAssign for Mask4 { #[inline(always)] fn bitxor_assign(&mut self, r: Self) { *self = *self ^ r; } }
impl Not for Mask4 {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        unsafe {
            let ones = _mm_cmpeq_ps(self.0, self.0);
            Mask4(_mm_xor_ps(self.0, ones))
        }
    }
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
