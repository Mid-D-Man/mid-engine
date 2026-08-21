// crates/mid-math/src/wide/int/avx2/imask32x8.rs
//! 8-lane integer comparison mask for i32x8/u32x8 — AVX2, x86 / x86_64.
//! Each 32-bit lane: 0xFFFFFFFF = true, 0x00000000 = false.
//! Never constructed directly — always produced by i32x8/u32x8 comparisons.
//! Mirrors sse2/imask4.rs, widened to __m256i (movemask_ps gives an 8-bit
//! result directly for 8× 32-bit lanes — no bit-pairing extraction needed,
//! same reasoning as the SSE2 IMask4 case).

use core::fmt;
use core::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not};

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

#[repr(C)]
union UnionCast { i: [i32; 8], v: IMask32x8 }

/// 8-lane integer comparison mask. 32 bytes, 32-byte aligned.
/// Backed by `__m256i`. Lane i: `0xFFFFFFFF` = true, `0x00000000` = false.
/// Use [`i32x8::blend`][super::i32x8::i32x8::blend] /
/// [`u32x8::blend`][super::u32x8::u32x8::blend] for branchless selection.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct IMask32x8(pub(crate) __m256i);

impl IMask32x8 {
    /// All lanes false.
    pub const FALSE: Self = unsafe { UnionCast { i: [0; 8] }.v };
    /// All lanes true.
    pub const TRUE: Self  = unsafe { UnionCast { i: [-1; 8] }.v };

    /// True if any lane is set.
    #[inline]
    pub fn any(self) -> bool {
        unsafe { _mm256_movemask_ps(_mm256_castsi256_ps(self.0)) != 0 }
    }

    /// True if all lanes are set.
    #[inline]
    pub fn all(self) -> bool {
        unsafe { _mm256_movemask_ps(_mm256_castsi256_ps(self.0)) == 0xFF }
    }

    /// True if no lane is set.
    #[inline]
    pub fn none(self) -> bool {
        unsafe { _mm256_movemask_ps(_mm256_castsi256_ps(self.0)) == 0 }
    }

    /// Packed 8-bit bitmask — one bit per 32-bit lane.
    #[inline]
    pub fn bitmask(self) -> u8 {
        unsafe { _mm256_movemask_ps(_mm256_castsi256_ps(self.0)) as u8 }
    }

    /// Number of true lanes.
    #[inline]
    pub fn count_true(self) -> u32 { self.bitmask().count_ones() }
}

impl BitAnd for IMask32x8 { type Output=Self; #[inline(always)] fn bitand(self,r:Self)->Self{IMask32x8(unsafe{_mm256_and_si256(self.0,r.0)})} }
impl BitAndAssign for IMask32x8 { #[inline(always)] fn bitand_assign(&mut self,r:Self){*self=*self&r;} }
impl BitOr for IMask32x8 { type Output=Self; #[inline(always)] fn bitor(self,r:Self)->Self{IMask32x8(unsafe{_mm256_or_si256(self.0,r.0)})} }
impl BitOrAssign for IMask32x8 { #[inline(always)] fn bitor_assign(&mut self,r:Self){*self=*self|r;} }
impl BitXor for IMask32x8 { type Output=Self; #[inline(always)] fn bitxor(self,r:Self)->Self{IMask32x8(unsafe{_mm256_xor_si256(self.0,r.0)})} }
impl BitXorAssign for IMask32x8 { #[inline(always)] fn bitxor_assign(&mut self,r:Self){*self=*self^r;} }
impl Not for IMask32x8 {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        unsafe {
            let ones = _mm256_cmpeq_epi32(self.0, self.0); // all-ones trick
            IMask32x8(_mm256_xor_si256(self.0, ones))
        }
    }
}
impl PartialEq for IMask32x8 { #[inline] fn eq(&self,r:&Self)->bool{self.bitmask()==r.bitmask()} }
impl Eq for IMask32x8 {}
impl fmt::Debug for IMask32x8 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let b = self.bitmask();
        write!(f, "IMask32x8({:08b})", b)
    }
}
