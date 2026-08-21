// crates/mid-math/src/wide/int/avx2/imask8x32.rs
//! 32-lane integer comparison mask for i8x32/u8x32 — AVX2, x86 / x86_64.
//! Each 8-bit lane: 0xFF = true, 0x00 = false.
//! bitmask() maps directly to `_mm256_movemask_epi8` — one bit per byte
//! lane, no extraction needed (mirrors sse2/imask16.rs exactly, doubled).

use core::fmt;
use core::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not};

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

#[repr(C)]
union UnionCast { i: [i8; 32], v: IMask8x32 }

/// 32-lane integer comparison mask. 32 bytes, 32-byte aligned.
/// Backed by `__m256i`. Lane i: `0xFF` = true, `0x00` = false.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct IMask8x32(pub(crate) __m256i);

impl IMask8x32 {
    /// All lanes false.
    pub const FALSE: Self = unsafe { UnionCast { i: [0; 32] }.v };
    /// All lanes true.
    pub const TRUE: Self  = unsafe { UnionCast { i: [-1; 32] }.v };

    /// True if any lane is set.
    #[inline]
    pub fn any(self) -> bool { unsafe { _mm256_movemask_epi8(self.0) != 0 } }

    /// True if all lanes are set.
    #[inline]
    pub fn all(self) -> bool { unsafe { _mm256_movemask_epi8(self.0) == -1 } }

    /// True if no lane is set.
    #[inline]
    pub fn none(self) -> bool { unsafe { _mm256_movemask_epi8(self.0) == 0 } }

    /// 32-bit bitmask — one bit per 8-bit lane. Directly from `_mm256_movemask_epi8`.
    #[inline]
    pub fn bitmask(self) -> u32 {
        unsafe { _mm256_movemask_epi8(self.0) as u32 }
    }

    /// Number of true lanes.
    #[inline]
    pub fn count_true(self) -> u32 { self.bitmask().count_ones() }
}

impl BitAnd for IMask8x32 { type Output=Self; #[inline(always)] fn bitand(self,r:Self)->Self{IMask8x32(unsafe{_mm256_and_si256(self.0,r.0)})} }
impl BitAndAssign for IMask8x32 { #[inline(always)] fn bitand_assign(&mut self,r:Self){*self=*self&r;} }
impl BitOr for IMask8x32 { type Output=Self; #[inline(always)] fn bitor(self,r:Self)->Self{IMask8x32(unsafe{_mm256_or_si256(self.0,r.0)})} }
impl BitOrAssign for IMask8x32 { #[inline(always)] fn bitor_assign(&mut self,r:Self){*self=*self|r;} }
impl BitXor for IMask8x32 { type Output=Self; #[inline(always)] fn bitxor(self,r:Self)->Self{IMask8x32(unsafe{_mm256_xor_si256(self.0,r.0)})} }
impl BitXorAssign for IMask8x32 { #[inline(always)] fn bitxor_assign(&mut self,r:Self){*self=*self^r;} }
impl Not for IMask8x32 {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        unsafe {
            let ones = _mm256_cmpeq_epi8(self.0, self.0);
            IMask8x32(_mm256_xor_si256(self.0, ones))
        }
    }
}
impl PartialEq for IMask8x32 { #[inline] fn eq(&self,r:&Self)->bool{self.bitmask()==r.bitmask()} }
impl Eq for IMask8x32 {}
impl fmt::Debug for IMask8x32 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "IMask8x32({:032b})", self.bitmask())
    }
}
