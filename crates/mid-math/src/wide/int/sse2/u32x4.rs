// crates/mid-math/src/wide/int/sse2/u32x4.rs
//! 4-lane unsigned 32-bit integer vector — SSE2, x86 / x86_64.
//!
//! Engine uses: component type hashes, generation counters, DixScript string
//! hashes, bitfield flags, 8 hashes processed simultaneously (AVX2).
//!
//! Key SSE2 differences from i32x4:
//!   - No Neg trait (unsigned — negation undefined)
//!   - No abs (always non-negative)
//!   - Unsigned comparison: XOR sign bit before signed _mm_cmpgt_epi32
//!   - saturating_add: check carry via (sum < a) unsigned comparison

#![allow(non_camel_case_types)]

use core::fmt;
use core::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign,
    BitXor, BitXorAssign, Mul, MulAssign, Not, Sub, SubAssign,
};

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use super::imask4::IMask4;

#[repr(C)]
union UnionCast {
    u: [u32; 4],
    v: u32x4,
}

/// 4-lane unsigned 32-bit integer vector. 16 bytes, 16-byte aligned.
///
/// Backed by `__m128i`. All comparisons are unsigned.
/// Comparison operations return [`IMask4`].
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct u32x4(pub(crate) __m128i);

// ── Unsigned comparison helper ─────────────────────────────────────────────────
//
// SSE2 has no unsigned 32-bit comparison instruction.
// unsigned a > b  ≡  signed (a ^ 0x80000000) > signed (b ^ 0x80000000)
//
// Proof: the XOR flips the sign bit. After XOR, the ordering of the
// resulting signed values matches the original unsigned ordering exactly.

#[inline(always)]
unsafe fn ucmpgt(a: __m128i, b: __m128i) -> __m128i {
    let sign = _mm_set1_epi32(i32::MIN); // 0x80000000 broadcast
    _mm_cmpgt_epi32(
        _mm_xor_si128(a, sign),
        _mm_xor_si128(b, sign),
    )
}

impl u32x4 {
    // ── Constants ─────────────────────────────────────────────────────────────

    pub const ZERO: Self = unsafe { UnionCast { u: [0; 4] }.v };
    pub const ONE:  Self = unsafe { UnionCast { u: [1; 4] }.v };
    pub const MIN:  Self = unsafe { UnionCast { u: [u32::MIN; 4] }.v };
    pub const MAX:  Self = unsafe { UnionCast { u: [u32::MAX; 4] }.v };

    // ── Constructors ──────────────────────────────────────────────────────────

    #[inline(always)]
    pub fn splat(v: u32) -> Self {
        Self(unsafe { _mm_set1_epi32(v as i32) })
    }

    /// `a` = lane 0, `d` = lane 3.
    #[inline(always)]
    pub fn new(a: u32, b: u32, c: u32, d: u32) -> Self {
        Self(unsafe { _mm_set_epi32(d as i32, c as i32, b as i32, a as i32) })
    }

    #[inline(always)]
    pub fn from_array(a: [u32; 4]) -> Self {
        Self(unsafe { _mm_loadu_si128(a.as_ptr() as *const __m128i) })
    }

    #[inline(always)]
    pub fn to_array(self) -> [u32; 4] {
        unsafe {
            let mut a = [0u32; 4];
            _mm_storeu_si128(a.as_mut_ptr() as *mut __m128i, self.0);
            a
        }
    }

    #[inline]
    pub fn get(self, i: usize) -> u32 {
        assert!(i < 4, "u32x4::get — lane {i} out of bounds (max 3)");
        unsafe { UnionCast { v: self }.u[i] }
    }

    // ── Arithmetic ────────────────────────────────────────────────────────────

    /// Per-lane minimum (unsigned).
    #[inline]
    pub fn min(self, rhs: Self) -> Self {
        let lt = self.cmplt(rhs); // ones where self < rhs (unsigned)
        Self::blend(lt, self, rhs)
    }

    /// Per-lane maximum (unsigned).
    #[inline]
    pub fn max(self, rhs: Self) -> Self {
        let gt = self.cmpgt(rhs); // ones where self > rhs (unsigned)
        Self::blend(gt, self, rhs)
    }

    #[inline]
    pub fn clamp(self, lo: Self, hi: Self) -> Self { self.max(lo).min(hi) }

    #[inline]
    pub fn min_element(self) -> u32 {
        let a = self.to_array();
        a[0].min(a[1]).min(a[2]).min(a[3])
    }

    #[inline]
    pub fn max_element(self) -> u32 {
        let a = self.to_array();
        a[0].max(a[1]).max(a[2]).max(a[3])
    }

    #[inline]
    pub fn element_sum(self) -> u32 {
        let a = self.to_array();
        a[0].wrapping_add(a[1]).wrapping_add(a[2]).wrapping_add(a[3])
    }

    // ── Shift operations ──────────────────────────────────────────────────────

    #[inline(always)] pub fn shl(self, count: u32) -> Self {
        Self(unsafe { _mm_slli_epi32(self.0, count as i32) })
    }
    /// Logical right shift (zero-fill from left).
    #[inline(always)] pub fn shr(self, count: u32) -> Self {
        Self(unsafe { _mm_srli_epi32(self.0, count as i32) })
    }

    // ── Comparison → IMask4 ───────────────────────────────────────────────────

    #[inline(always)]
    pub fn cmpeq(self, rhs: Self) -> IMask4 {
        // Signed equality works correctly for unsigned values.
        IMask4(unsafe { _mm_cmpeq_epi32(self.0, rhs.0) })
    }

    #[inline(always)]
    pub fn cmpne(self, rhs: Self) -> IMask4 { !self.cmpeq(rhs) }

    /// Unsigned greater-than.
    #[inline(always)]
    pub fn cmpgt(self, rhs: Self) -> IMask4 {
        IMask4(unsafe { ucmpgt(self.0, rhs.0) })
    }

    /// Unsigned less-than.
    #[inline(always)]
    pub fn cmplt(self, rhs: Self) -> IMask4 { rhs.cmpgt(self) }

    /// Unsigned greater-than-or-equal.
    #[inline(always)]
    pub fn cmpge(self, rhs: Self) -> IMask4 { !self.cmplt(rhs) }

    /// Unsigned less-than-or-equal.
    #[inline(always)]
    pub fn cmple(self, rhs: Self) -> IMask4 { !self.cmpgt(rhs) }

    // ── Branchless select ─────────────────────────────────────────────────────

    /// Per-lane branchless select.
    #[inline(always)]
    pub fn blend(mask: IMask4, if_true: Self, if_false: Self) -> Self {
        unsafe {
            Self(_mm_or_si128(
                _mm_and_si128(mask.0, if_true.0),
                _mm_andnot_si128(mask.0, if_false.0),
            ))
        }
    }

    // ── Wrapping / saturating ─────────────────────────────────────────────────

    #[inline(always)] pub fn wrapping_add(self, r: Self) -> Self { self + r }
    #[inline(always)] pub fn wrapping_sub(self, r: Self) -> Self { self - r }
    #[inline(always)] pub fn wrapping_mul(self, r: Self) -> Self { self * r }

    /// Saturating unsigned add: clamps to `u32::MAX` on overflow.
    ///
    /// Overflow detected by: `(a + b) < a` unsigned.
    /// No SSE2 instruction for 32-bit unsigned saturation — uses comparison + blend.
    #[inline]
    pub fn saturating_add(self, rhs: Self) -> Self {
        unsafe {
            let sum = _mm_add_epi32(self.0, rhs.0);
            // Overflow if sum < self (unsigned): carry happened
            let overflowed = ucmpgt(self.0, sum); // ones where self > sum (i.e. sum wrapped)
            // Where overflow: u32::MAX (all-ones), else: sum
            let all_ones = _mm_cmpeq_epi32(sum, sum);
            Self(_mm_or_si128(
                _mm_and_si128(overflowed, all_ones),
                _mm_andnot_si128(overflowed, sum),
            ))
        }
    }

    /// Saturating unsigned sub: clamps to `0` on underflow.
    ///
    /// Underflow detected by: `rhs > self` unsigned.
    #[inline]
    pub fn saturating_sub(self, rhs: Self) -> Self {
        unsafe {
            let underflowed = ucmpgt(rhs.0, self.0); // ones where rhs > self
            let diff = _mm_sub_epi32(self.0, rhs.0);
            // Where underflow: 0, else: diff
            Self(_mm_andnot_si128(underflowed, diff))
        }
    }
}

// ── Operators ─────────────────────────────────────────────────────────────────

impl Add for u32x4 {
    type Output = Self;
    #[inline(always)]
    fn add(self, r: Self) -> Self { Self(unsafe { _mm_add_epi32(self.0, r.0) }) }
}
impl AddAssign for u32x4 {
    #[inline(always)] fn add_assign(&mut self, r: Self) { *self = *self + r; }
}
impl Sub for u32x4 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, r: Self) -> Self { Self(unsafe { _mm_sub_epi32(self.0, r.0) }) }
}
impl SubAssign for u32x4 {
    #[inline(always)] fn sub_assign(&mut self, r: Self) { *self = *self - r; }
}
impl Mul for u32x4 {
    type Output = Self;
    /// Low 32 bits of each lane product — same SSE2 trick as i32x4.
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        unsafe {
            let a13 = _mm_shuffle_epi32(self.0, 0xF5);
            let b13 = _mm_shuffle_epi32(rhs.0,  0xF5);
            let prod02 = _mm_mul_epu32(self.0, rhs.0);
            let prod13 = _mm_mul_epu32(a13, b13);
            let lo02 = _mm_shuffle_epi32(prod02, 0x08);
            let lo13 = _mm_shuffle_epi32(prod13, 0x08);
            Self(_mm_unpacklo_epi32(lo02, lo13))
        }
    }
}
impl MulAssign for u32x4 {
    #[inline(always)] fn mul_assign(&mut self, r: Self) { *self = *self * r; }
}
impl BitAnd for u32x4 {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, r: Self) -> Self { Self(unsafe { _mm_and_si128(self.0, r.0) }) }
}
impl BitAndAssign for u32x4 {
    #[inline(always)] fn bitand_assign(&mut self, r: Self) { *self = *self & r; }
}
impl BitOr for u32x4 {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, r: Self) -> Self { Self(unsafe { _mm_or_si128(self.0, r.0) }) }
}
impl BitOrAssign for u32x4 {
    #[inline(always)] fn bitor_assign(&mut self, r: Self) { *self = *self | r; }
}
impl BitXor for u32x4 {
    type Output = Self;
    #[inline(always)]
    fn bitxor(self, r: Self) -> Self { Self(unsafe { _mm_xor_si128(self.0, r.0) }) }
}
impl BitXorAssign for u32x4 {
    #[inline(always)] fn bitxor_assign(&mut self, r: Self) { *self = *self ^ r; }
}
impl Not for u32x4 {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        unsafe {
            let ones = _mm_cmpeq_epi32(self.0, self.0);
            Self(_mm_xor_si128(self.0, ones))
        }
    }
}

impl PartialEq for u32x4 {
    #[inline]
    fn eq(&self, r: &Self) -> bool {
        unsafe { _mm_movemask_ps(_mm_castsi128_ps(_mm_cmpeq_epi32(self.0, r.0))) == 0b1111 }
    }
}
impl Eq for u32x4 {}

impl fmt::Debug for u32x4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let a = self.to_array();
        write!(f, "u32x4({}, {}, {}, {})", a[0], a[1], a[2], a[3])
    }
}
impl fmt::Display for u32x4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let a = self.to_array();
        write!(f, "[{}, {}, {}, {}]", a[0], a[1], a[2], a[3])
    }
}

impl From<[u32; 4]> for u32x4 {
    #[inline] fn from(a: [u32; 4]) -> Self { Self::from_array(a) }
}
impl From<u32x4> for [u32; 4] {
    #[inline] fn from(v: u32x4) -> Self { v.to_array() }
  }
