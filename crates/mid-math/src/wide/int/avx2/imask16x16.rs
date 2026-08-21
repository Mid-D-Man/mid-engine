// crates/mid-math/src/wide/int/avx2/imask16x16.rs
//! 16-lane integer comparison mask for i16x16/u16x16 — AVX2, x86 / x86_64.
//! Each 16-bit lane: 0xFFFF = true, 0x0000 = false.
//! Mirrors sse2/imask8.rs, widened to __m256i. `_mm256_movemask_epi8`
//! gives a 32-bit byte-granularity result; each true 16-bit lane
//! contributes 2 identical bits, so we extract bit `2*i` per lane `i`
//! (same trick sse2/imask8.rs uses at half the width).

use core::fmt;
use core::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not};

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

#[repr(C)]
union UnionCast { i: [i16; 16], v: IMask16x16 }

/// 16-lane integer comparison mask. 32 bytes, 32-byte aligned.
/// Backed by `__m256i`. Lane i: `0xFFFF` = true, `0x0000` = false.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct IMask16x16(pub(crate) __m256i);

impl IMask16x16 {
    /// All lanes false.
    pub const FALSE: Self = unsafe { UnionCast { i: [0; 16] }.v };
    /// All lanes true.
    pub const TRUE: Self  = unsafe { UnionCast { i: [-1; 16] }.v };

    /// True if any lane is set.
    #[inline]
    pub fn any(self) -> bool { unsafe { _mm256_movemask_epi8(self.0) != 0 } }

    /// True if all lanes are set.
    #[inline]
    pub fn all(self) -> bool { unsafe { _mm256_movemask_epi8(self.0) == -1 } }

    /// True if no lane is set.
    #[inline]
    pub fn none(self) -> bool { unsafe { _mm256_movemask_epi8(self.0) == 0 } }

    /// Packed 16-bit bitmask — one bit per 16-bit lane.
    ///
    /// A true 16-bit lane produces two consecutive set bits in
    /// `_mm256_movemask_epi8`. We extract one representative bit per
    /// lane (bit `2*i` for lane `i`), same as sse2/imask8.rs.
    #[inline]
    pub fn bitmask(self) -> u16 {
        unsafe {
            let m = _mm256_movemask_epi8(self.0) as u32;
            (0..16u32).fold(0u16, |acc, i| acc | (((m >> (2 * i)) & 1) as u16) << i)
        }
    }

    /// Number of true lanes.
    #[inline]
    pub fn count_true(self) -> u32 { self.bitmask().count_ones() }
}

impl BitAnd for IMask16x16 { type Output=Self; #[inline(always)] fn bitand(self,r:Self)->Self{IMask16x16(unsafe{_mm256_and_si256(self.0,r.0)})} }
impl BitAndAssign for IMask16x16 { #[inline(always)] fn bitand_assign(&mut self,r:Self){*self=*self&r;} }
impl BitOr for IMask16x16 { type Output=Self; #[inline(always)] fn bitor(self,r:Self)->Self{IMask16x16(unsafe{_mm256_or_si256(self.0,r.0)})} }
impl BitOrAssign for IMask16x16 { #[inline(always)] fn bitor_assign(&mut self,r:Self){*self=*self|r;} }
impl BitXor for IMask16x16 { type Output=Self; #[inline(always)] fn bitxor(self,r:Self)->Self{IMask16x16(unsafe{_mm256_xor_si256(self.0,r.0)})} }
impl BitXorAssign for IMask16x16 { #[inline(always)] fn bitxor_assign(&mut self,r:Self){*self=*self^r;} }
impl Not for IMask16x16 {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        unsafe {
            let ones = _mm256_cmpeq_epi16(self.0, self.0);
            IMask16x16(_mm256_xor_si256(self.0, ones))
        }
    }
}
impl PartialEq for IMask16x16 { #[inline] fn eq(&self,r:&Self)->bool{self.bitmask()==r.bitmask()} }
impl Eq for IMask16x16 {}
impl fmt::Debug for IMask16x16 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "IMask16x16({:016b})", self.bitmask())
    }
}
