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

    // ── Lane iteration ──────────────────────────────────────────────────────
    //
    // All three operate purely on `bitmask()`'s 4-bit value, so `trailing_zeros`/
    // `count_ones` do the work: LLVM lowers those to TZCNT/POPCNT automatically
    // when the target has bmi1/popcnt (x86-64-v2 and up), and falls back to a
    // portable software sequence otherwise. No raw intrinsics needed to pick up
    // that win where it exists, and it stays correct everywhere else too.

    /// Index (0-3) of the lowest-numbered true lane, or `None` if none are true.
    #[inline]
    pub fn first_set_lane(self) -> Option<u32> {
        let b = self.bitmask();
        if b == 0 { None } else { Some(b.trailing_zeros()) }
    }

    /// Number of true lanes (0-4).
    #[inline]
    pub fn count_set(self) -> u32 { self.bitmask().count_ones() }

    /// Iterate the indices (0-3) of true lanes, lowest to highest.
    #[inline]
    pub fn iter_set_lanes(self) -> Mask4LaneIter { Mask4LaneIter { bits: self.bitmask() } }

    #[allow(dead_code)]
    #[inline(always)]
    pub(crate) fn from_m128(m: __m128) -> Self { Mask4(m) }
}

/// Iterator over the true-lane indices of a [`Mask4`]. See [`Mask4::iter_set_lanes`].
pub struct Mask4LaneIter { bits: u32 }

impl Iterator for Mask4LaneIter {
    type Item = u32;
    #[inline]
    fn next(&mut self) -> Option<u32> {
        if self.bits == 0 { return None; }
        let idx = self.bits.trailing_zeros();
        self.bits &= self.bits - 1; // clear lowest set bit (BLSR under bmi1)
        Some(idx)
    }
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.bits.count_ones() as usize;
        (n, Some(n))
    }
}
impl ExactSizeIterator for Mask4LaneIter {}

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
        // Do NOT derive "all-ones" via _mm_cmpeq_ps(self.0, self.0): a TRUE
        // lane's bit pattern (0xFFFFFFFF) reinterpreted as f32 is NaN, and
        // NaN != NaN even against itself. That silently flips the "ones"
        // helper on exactly the lanes that are true, cancelling out the XOR
        // and leaving every lane true regardless of input. XOR against the
        // TRUE constant instead — plain bitwise, no float comparison.
        unsafe { Mask4(_mm_xor_ps(self.0, Mask4::TRUE.0)) }
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
