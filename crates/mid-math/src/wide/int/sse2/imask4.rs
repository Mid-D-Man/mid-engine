// crates/mid-math/src/wide/int/sse2/imask4.rs
//! 4-lane integer comparison mask — SSE2, x86 / x86_64.
//!
//! Each lane: 0xFFFFFFFF = true, 0x00000000 = false.
//! Produced by i32x4 / u32x4 comparison ops.
//! Used for branchless blend (select) on integer lanes.
//!
//! NOT constructed directly — always from a comparison result.

use core::fmt;
use core::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not};

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

// ── Const initialisation helper ───────────────────────────────────────────────
// Mirrors the UnionCast pattern in f32/sse2/quat.rs.
// [i32; 4] and __m128i are both 16 bytes; both are Copy.

#[repr(C)]
union UnionCast {
    i: [i32; 4],
    v: IMask4,
}

/// 4-lane integer comparison mask. 16 bytes, 16-byte aligned.
///
/// Backed by `__m128i`. Each lane is either `0xFFFFFFFF` (true) or
/// `0x00000000` (false). Never construct directly — use comparison
/// operations on [`i32x4`][super::i32x4::i32x4] or
/// [`u32x4`][super::u32x4::u32x4].
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct IMask4(pub(crate) __m128i);

impl IMask4 {
    /// All lanes false.
    pub const FALSE: Self = unsafe { UnionCast { i: [0; 4] }.v };
    /// All lanes true.
    pub const TRUE: Self  = unsafe { UnionCast { i: [-1; 4] }.v }; // -1i32 = 0xFFFFFFFF

    // ── Horizontal predicates ─────────────────────────────────────────────────

    /// Returns `true` if any lane is true.
    #[inline]
    pub fn any(self) -> bool {
        unsafe { _mm_movemask_ps(_mm_castsi128_ps(self.0)) != 0 }
    }

    /// Returns `true` if all lanes are true.
    #[inline]
    pub fn all(self) -> bool {
        unsafe { _mm_movemask_ps(_mm_castsi128_ps(self.0)) == 0b1111 }
    }

    /// Returns `true` if no lane is true.
    #[inline]
    pub fn none(self) -> bool {
        unsafe { _mm_movemask_ps(_mm_castsi128_ps(self.0)) == 0 }
    }

    /// Returns a packed 4-bit bitmask: bit 0 = lane 0, bit 1 = lane 1, etc.
    ///
    /// Uses the high bit of each 32-bit lane (0 or 1 since masks are 0/0xFFFFFFFF).
    #[inline]
    pub fn bitmask(self) -> u32 {
        unsafe { _mm_movemask_ps(_mm_castsi128_ps(self.0)) as u32 }
    }
}

// ── Bitwise operators ─────────────────────────────────────────────────────────

impl BitAnd for IMask4 {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, r: Self) -> Self { IMask4(unsafe { _mm_and_si128(self.0, r.0) }) }
}
impl BitAndAssign for IMask4 {
    #[inline(always)]
    fn bitand_assign(&mut self, r: Self) { *self = *self & r; }
}
impl BitOr for IMask4 {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, r: Self) -> Self { IMask4(unsafe { _mm_or_si128(self.0, r.0) }) }
}
impl BitOrAssign for IMask4 {
    #[inline(always)]
    fn bitor_assign(&mut self, r: Self) { *self = *self | r; }
}
impl BitXor for IMask4 {
    type Output = Self;
    #[inline(always)]
    fn bitxor(self, r: Self) -> Self { IMask4(unsafe { _mm_xor_si128(self.0, r.0) }) }
}
impl BitXorAssign for IMask4 {
    #[inline(always)]
    fn bitxor_assign(&mut self, r: Self) { *self = *self ^ r; }
}
impl Not for IMask4 {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        unsafe {
            // cmpeq(x, x) = all-ones regardless of x — standard SSE2 all-ones trick
            let ones = _mm_cmpeq_epi32(self.0, self.0);
            IMask4(_mm_xor_si128(self.0, ones))
        }
    }
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
