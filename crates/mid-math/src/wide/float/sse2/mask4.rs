// crates/mid-math/src/wide/float/sse2/mask4.rs
//! 4-lane float comparison mask — SSE2, x86 / x86_64.
//!
//! Backed by `__m128` (NOT `__m128i` like IMask4).
//! Float comparisons (_mm_cmplt_ps etc.) produce __m128 results natively —
//! no cast needed. Blend uses _mm_and_ps / _mm_andnot_ps.
//!
//! Distinct from IMask4 which is integer (__m128i). This is the mask
//! returned by f32x4 and Vec3x4 comparisons — do not mix them.

use core::fmt;
use core::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not};

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use crate::sse2::m128_from_f32x4;

#[repr(C)]
union UCast { f: [f32; 4], v: Mask4 }

/// 4-lane float comparison mask. 16 bytes, 16-byte aligned. Backed by `__m128`.
///
/// Each lane: all-ones (`f32::from_bits(0xFFFFFFFF)`) = true, all-zeros = false.
/// Never construct directly — produced by [`f32x4`] or [`Vec3x4`] comparisons.
/// Use [`f32x4::blend`] or [`Vec3x4::select`] for branchless selection.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct Mask4(pub(crate) __m128);

impl Mask4 {
    /// All lanes false.
    pub const FALSE: Self = unsafe { UCast { f: [0.0; 4] }.v };
    /// All lanes true (all bits set per lane — not NaN-safe, only use as mask).
    pub const TRUE: Self  = unsafe { UCast { f: [
        f32::from_bits(0xFFFF_FFFF),
        f32::from_bits(0xFFFF_FFFF),
        f32::from_bits(0xFFFF_FFFF),
        f32::from_bits(0xFFFF_FFFF),
    ] }.v };

    // ── Horizontal predicates ─────────────────────────────────────────────────

    /// Returns `true` if any lane is set.
    #[inline]
    pub fn any(self) -> bool {
        unsafe { _mm_movemask_ps(self.0) != 0 }
    }

    /// Returns `true` if all lanes are set.
    #[inline]
    pub fn all(self) -> bool {
        unsafe { _mm_movemask_ps(self.0) == 0b1111 }
    }

    /// Returns `true` if no lane is set.
    #[inline]
    pub fn none(self) -> bool {
        unsafe { _mm_movemask_ps(self.0) == 0 }
    }

    /// 4-bit packed bitmask: bit 0 = lane 0, bit 3 = lane 3.
    #[inline]
    pub fn bitmask(self) -> u32 {
        unsafe { _mm_movemask_ps(self.0) as u32 }
    }

    // ── Constructors from comparisons ─────────────────────────────────────────
    // These mirror the f32x4 comparison API but are callable directly
    // for constructing masks from raw __m128 comparison results.

    /// Wrap a raw `__m128` comparison result (for internal use by f32x4/Vec3x4).
    #[inline(always)]
    pub(crate) fn from_m128(m: __m128) -> Self { Mask4(m) }
}

// ── Bitwise operators — use float bitwise ops (_mm_and_ps etc.) ───────────────
// Using float variants (not _si128) keeps the mask in the float execution port
// and avoids domain crossing penalties on older CPUs.

impl BitAnd for Mask4 {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, r: Self) -> Self { Mask4(unsafe { _mm_and_ps(self.0, r.0) }) }
}
impl BitAndAssign for Mask4 {
    #[inline(always)] fn bitand_assign(&mut self, r: Self) { *self = *self & r; }
}
impl BitOr for Mask4 {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, r: Self) -> Self { Mask4(unsafe { _mm_or_ps(self.0, r.0) }) }
}
impl BitOrAssign for Mask4 {
    #[inline(always)] fn bitor_assign(&mut self, r: Self) { *self = *self | r; }
}
impl BitXor for Mask4 {
    type Output = Self;
    #[inline(always)]
    fn bitxor(self, r: Self) -> Self { Mask4(unsafe { _mm_xor_ps(self.0, r.0) }) }
}
impl BitXorAssign for Mask4 {
    #[inline(always)] fn bitxor_assign(&mut self, r: Self) { *self = *self ^ r; }
}
impl Not for Mask4 {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        unsafe {
            // XOR with all-ones flips every bit.
            let ones = _mm_cmpeq_ps(self.0, self.0); // always all-ones
            Mask4(_mm_xor_ps(self.0, ones))
        }
    }
}

impl PartialEq for Mask4 {
    #[inline]
    fn eq(&self, r: &Self) -> bool { self.bitmask() == r.bitmask() }
}
impl Eq for Mask4 {}

impl fmt::Debug for Mask4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let b = self.bitmask();
        write!(f, "Mask4({}, {}, {}, {})",
            b & 1 != 0, b >> 1 & 1 != 0, b >> 2 & 1 != 0, b >> 3 & 1 != 0)
    }
}
