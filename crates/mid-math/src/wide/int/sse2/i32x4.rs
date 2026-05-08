// crates/mid-math/src/wide/int/sse2/i32x4.rs
//! 4-lane signed 32-bit integer vector — SSE2, x86 / x86_64.
//!
//! Engine uses: entity IDs, voxel chunk coordinates, ECS archetype indices,
//! grid lookups, 4 comparisons in ~3 instructions instead of 4 sequential.
//!
//! Key SSE2 implementation notes:
//!   - mul:   no _mm_mullo_epi32 in SSE2 → 6-instruction trick via _mm_mul_epu32
//!   - min:   no _mm_min_epi32 in SSE2   → cmplt + blend (2 instructions)
//!   - max:   no _mm_max_epi32 in SSE2   → cmpgt + blend (2 instructions)
//!   - abs:   no _mm_abs_epi32 in SSE2   → cmplt + neg + blend
//!   - sat32: no saturating i32 add/sub  → scalar extract (rare hot path)

#![allow(non_camel_case_types)]

use core::fmt;
use core::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign,
    BitXor, BitXorAssign, Mul, MulAssign, Neg, Not, Sub, SubAssign,
};

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use super::imask4::IMask4;

// ── Const helper ──────────────────────────────────────────────────────────────

#[repr(C)]
union UnionCast {
    i: [i32; 4],
    v: i32x4,
}

/// 4-lane signed 32-bit integer vector. 16 bytes, 16-byte aligned.
///
/// Backed by `__m128i`. All operations are branchless.
/// Comparison operations return [`IMask4`] for use with [`i32x4::blend`].
///
/// **C interop:** extract with `to_array()` and pass as `[i32; 4]`.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct i32x4(pub(crate) __m128i);

impl i32x4 {
    // ── Constants ─────────────────────────────────────────────────────────────

    pub const ZERO: Self = unsafe { UnionCast { i: [0; 4] }.v };
    pub const ONE:  Self = unsafe { UnionCast { i: [1; 4] }.v };
    pub const MIN:  Self = unsafe { UnionCast { i: [i32::MIN; 4] }.v };
    pub const MAX:  Self = unsafe { UnionCast { i: [i32::MAX; 4] }.v };

    // ── Constructors ──────────────────────────────────────────────────────────

    /// Broadcast `v` to all 4 lanes.
    #[inline(always)]
    pub fn splat(v: i32) -> Self {
        Self(unsafe { _mm_set1_epi32(v) })
    }

    /// Create from 4 values. `a` = lane 0, `d` = lane 3.
    ///
    /// `_mm_set_epi32` takes args highest-lane first, so args are reversed.
    #[inline(always)]
    pub fn new(a: i32, b: i32, c: i32, d: i32) -> Self {
        Self(unsafe { _mm_set_epi32(d, c, b, a) })
    }

    #[inline(always)]
    pub fn from_array(a: [i32; 4]) -> Self {
        Self(unsafe { _mm_loadu_si128(a.as_ptr() as *const __m128i) })
    }

    #[inline(always)]
    pub fn to_array(self) -> [i32; 4] {
        unsafe {
            let mut a = [0i32; 4];
            _mm_storeu_si128(a.as_mut_ptr() as *mut __m128i, self.0);
            a
        }
    }

    /// Extract one lane. Panics if `i >= 4`.
    #[inline]
    pub fn get(self, i: usize) -> i32 {
        assert!(i < 4, "i32x4::get — lane {i} out of bounds (max 3)");
        unsafe { UnionCast { v: self }.i[i] }
    }

    // ── Arithmetic ────────────────────────────────────────────────────────────

    /// Absolute value of each lane (wrapping — `abs(i32::MIN) = i32::MIN`).
    ///
    /// SSE2: compare < 0, negate negatives via sub(0, x), blend.
    #[inline]
    pub fn abs(self) -> Self {
        unsafe {
            let zero = _mm_setzero_si128();
            let is_neg = _mm_cmplt_epi32(self.0, zero); // 0xFFFFFFFF where self < 0
            let negated = _mm_sub_epi32(zero, self.0);   // 0 - self = -self (wrapping)
            // where is_neg: negated (i.e. |self|), else self
            Self(_mm_or_si128(
                _mm_and_si128(is_neg, negated),
                _mm_andnot_si128(is_neg, self.0),
            ))
        }
    }

    /// Per-lane minimum.
    ///
    /// SSE2: cmplt + blend (SSE4.1 would use _mm_min_epi32 directly).
    #[inline]
    pub fn min(self, rhs: Self) -> Self {
        let lt = self.cmplt(rhs); // ones where self < rhs
        Self::blend(lt, self, rhs) // where self < rhs: self, else: rhs
    }

    /// Per-lane maximum.
    #[inline]
    pub fn max(self, rhs: Self) -> Self {
        let gt = self.cmpgt(rhs); // ones where self > rhs
        Self::blend(gt, self, rhs) // where self > rhs: self, else: rhs
    }

    #[inline]
    pub fn clamp(self, lo: Self, hi: Self) -> Self { self.max(lo).min(hi) }

    /// Horizontal minimum (scalar — extracts all 4 lanes).
    #[inline]
    pub fn min_element(self) -> i32 {
        let a = self.to_array();
        a[0].min(a[1]).min(a[2]).min(a[3])
    }

    /// Horizontal maximum.
    #[inline]
    pub fn max_element(self) -> i32 {
        let a = self.to_array();
        a[0].max(a[1]).max(a[2]).max(a[3])
    }

    /// Horizontal sum (wrapping).
    #[inline]
    pub fn element_sum(self) -> i32 {
        let a = self.to_array();
        a[0].wrapping_add(a[1]).wrapping_add(a[2]).wrapping_add(a[3])
    }

    // ── Shift operations ──────────────────────────────────────────────────────

    /// Logical left shift all lanes by `count` bits (zero-fill from right).
    #[inline(always)]
    pub fn shl(self, count: u32) -> Self {
        Self(unsafe { _mm_slli_epi32(self.0, count as i32) })
    }

    /// Arithmetic right shift (sign-extend) all lanes by `count` bits.
    #[inline(always)]
    pub fn shr_arithmetic(self, count: u32) -> Self {
        Self(unsafe { _mm_srai_epi32(self.0, count as i32) })
    }

    /// Logical right shift (zero-fill from left) all lanes by `count` bits.
    #[inline(always)]
    pub fn shr_logical(self, count: u32) -> Self {
        Self(unsafe { _mm_srli_epi32(self.0, count as i32) })
    }

    // ── Comparison → IMask4 ───────────────────────────────────────────────────

    #[inline(always)]
    pub fn cmpeq(self, rhs: Self) -> IMask4 {
        IMask4(unsafe { _mm_cmpeq_epi32(self.0, rhs.0) })
    }

    #[inline(always)]
    pub fn cmpne(self, rhs: Self) -> IMask4 { !self.cmpeq(rhs) }

    #[inline(always)]
    pub fn cmpgt(self, rhs: Self) -> IMask4 {
        IMask4(unsafe { _mm_cmpgt_epi32(self.0, rhs.0) })
    }

    #[inline(always)]
    pub fn cmplt(self, rhs: Self) -> IMask4 {
        IMask4(unsafe { _mm_cmplt_epi32(self.0, rhs.0) })
    }

    #[inline(always)]
    pub fn cmpge(self, rhs: Self) -> IMask4 { !self.cmplt(rhs) }

    #[inline(always)]
    pub fn cmple(self, rhs: Self) -> IMask4 { !self.cmpgt(rhs) }

    // ── Branchless select ─────────────────────────────────────────────────────

    /// Select per lane: returns `if_true[i]` where `mask[i]` is true, else `if_false[i]`.
    ///
    /// SSE2: `(mask & if_true) | (~mask & if_false)`.
    /// No branch, no misprediction penalty.
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

    /// Wrapping add — identical to `+` operator (SSE2 add already wraps).
    #[inline(always)] pub fn wrapping_add(self, r: Self) -> Self { self + r }
    /// Wrapping sub.
    #[inline(always)] pub fn wrapping_sub(self, r: Self) -> Self { self - r }
    /// Wrapping mul.
    #[inline(always)] pub fn wrapping_mul(self, r: Self) -> Self { self * r }

    /// Saturating add. Clamps to `i32::MIN / i32::MAX` on overflow.
    ///
    /// Note: no SSE2 instruction for 32-bit signed saturation — scalar extract.
    /// Use sparingly in hot loops.
    #[inline]
    pub fn saturating_add(self, rhs: Self) -> Self {
        let a = self.to_array();
        let b = rhs.to_array();
        Self::from_array([
            a[0].saturating_add(b[0]), a[1].saturating_add(b[1]),
            a[2].saturating_add(b[2]), a[3].saturating_add(b[3]),
        ])
    }

    /// Saturating sub.
    #[inline]
    pub fn saturating_sub(self, rhs: Self) -> Self {
        let a = self.to_array();
        let b = rhs.to_array();
        Self::from_array([
            a[0].saturating_sub(b[0]), a[1].saturating_sub(b[1]),
            a[2].saturating_sub(b[2]), a[3].saturating_sub(b[3]),
        ])
    }
}

// ── Operators ─────────────────────────────────────────────────────────────────

impl Add for i32x4 {
    type Output = Self;
    #[inline(always)]
    fn add(self, r: Self) -> Self { Self(unsafe { _mm_add_epi32(self.0, r.0) }) }
}
impl AddAssign for i32x4 {
    #[inline(always)] fn add_assign(&mut self, r: Self) { *self = *self + r; }
}

impl Sub for i32x4 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, r: Self) -> Self { Self(unsafe { _mm_sub_epi32(self.0, r.0) }) }
}
impl SubAssign for i32x4 {
    #[inline(always)] fn sub_assign(&mut self, r: Self) { *self = *self - r; }
}

impl Neg for i32x4 {
    type Output = Self;
    /// Wrapping negate (`i32::MIN` stays `i32::MIN`).
    #[inline(always)]
    fn neg(self) -> Self {
        Self(unsafe { _mm_sub_epi32(_mm_setzero_si128(), self.0) })
    }
}

impl Mul for i32x4 {
    type Output = Self;
    /// Low 32 bits of each lane product (wrapping).
    ///
    /// SSE2 uses `_mm_mul_epu32` on two 64-bit splits then interleaves
    /// the low halves. SSE4.1 would use `_mm_mullo_epi32` directly.
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        unsafe {
            // Shuffle 0xF5 = _MM_SHUFFLE(3,3,1,1): copy lanes 1,1,3,3
            // → extracts odd lanes so _mm_mul_epu32 sees the right pairs.
            let a13 = _mm_shuffle_epi32(self.0, 0xF5); // [a1,a1,a3,a3]
            let b13 = _mm_shuffle_epi32(rhs.0,  0xF5); // [b1,b1,b3,b3]
            // 64-bit products: [a0*b0, a2*b2] and [a1*b1, a3*b3]
            let prod02 = _mm_mul_epu32(self.0, rhs.0);
            let prod13 = _mm_mul_epu32(a13,    b13);
            // Shuffle 0x08 = _MM_SHUFFLE(0,0,2,0): extract low 32 bits of each 64-bit result
            let lo02 = _mm_shuffle_epi32(prod02, 0x08); // [low(a0b0), low(a2b2), ?, ?]
            let lo13 = _mm_shuffle_epi32(prod13, 0x08); // [low(a1b1), low(a3b3), ?, ?]
            // Interleave: [low(a0b0), low(a1b1), low(a2b2), low(a3b3)]
            Self(_mm_unpacklo_epi32(lo02, lo13))
        }
    }
}
impl MulAssign for i32x4 {
    #[inline(always)] fn mul_assign(&mut self, r: Self) { *self = *self * r; }
}

impl BitAnd for i32x4 {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, r: Self) -> Self { Self(unsafe { _mm_and_si128(self.0, r.0) }) }
}
impl BitAndAssign for i32x4 {
    #[inline(always)] fn bitand_assign(&mut self, r: Self) { *self = *self & r; }
}
impl BitOr for i32x4 {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, r: Self) -> Self { Self(unsafe { _mm_or_si128(self.0, r.0) }) }
}
impl BitOrAssign for i32x4 {
    #[inline(always)] fn bitor_assign(&mut self, r: Self) { *self = *self | r; }
}
impl BitXor for i32x4 {
    type Output = Self;
    #[inline(always)]
    fn bitxor(self, r: Self) -> Self { Self(unsafe { _mm_xor_si128(self.0, r.0) }) }
}
impl BitXorAssign for i32x4 {
    #[inline(always)] fn bitxor_assign(&mut self, r: Self) { *self = *self ^ r; }
}
impl Not for i32x4 {
    type Output = Self;
    /// Bitwise NOT of all lanes (`~x`).
    #[inline(always)]
    fn not(self) -> Self {
        unsafe {
            let ones = _mm_cmpeq_epi32(self.0, self.0); // all 0xFFFFFFFF
            Self(_mm_xor_si128(self.0, ones))
        }
    }
}

impl PartialEq for i32x4 {
    #[inline]
    fn eq(&self, r: &Self) -> bool {
        unsafe { _mm_movemask_ps(_mm_castsi128_ps(_mm_cmpeq_epi32(self.0, r.0))) == 0b1111 }
    }
}
impl Eq for i32x4 {}

impl fmt::Debug for i32x4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let a = self.to_array();
        write!(f, "i32x4({}, {}, {}, {})", a[0], a[1], a[2], a[3])
    }
}
impl fmt::Display for i32x4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let a = self.to_array();
        write!(f, "[{}, {}, {}, {}]", a[0], a[1], a[2], a[3])
    }
}

impl From<[i32; 4]> for i32x4 {
    #[inline] fn from(a: [i32; 4]) -> Self { Self::from_array(a) }
}
impl From<i32x4> for [i32; 4] {
    #[inline] fn from(v: i32x4) -> Self { v.to_array() }
}
