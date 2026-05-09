// crates/mid-math/src/wide/int/sse2/i8x16.rs
//! 16-lane signed 8-bit integer vector — SSE2, x86 / x86_64.
//!
//! Engine uses: DixScript string hashing (16 bytes/cycle via FNV-1a),
//! state machine flag arrays, MBFA compression byte processing,
//! UTF-8 byte classification.
//!
//! SSE2 native: add/sub/saturating/cmpeq/cmpgt — all 8-bit.
//! SSSE3 gate: shuffle_bytes (_mm_shuffle_epi8) — available since Penryn 2007.
//!             Detected at compile time via target_feature = "ssse3".

#![allow(non_camel_case_types)]

use core::fmt;
use core::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign,
    BitXor, BitXorAssign, Neg, Not, Sub, SubAssign,
};

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use super::imask16::IMask16;

#[repr(C)]
union UnionCast { i: [i8; 16], v: i8x16 }

/// 16-lane signed 8-bit integer vector. 16 bytes, 16-byte aligned. Backed by `__m128i`.
///
/// All operations are branchless. Comparison ops return [`IMask16`].
///
/// Note: no multiply — i8 mul would require widening to i16 first.
/// Use `as_i16x8_lo` / `as_i16x8_hi` then [`i16x8`] multiply.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct i8x16(pub(crate) __m128i);

impl i8x16 {
    pub const ZERO: Self = unsafe { UnionCast { i: [0; 16] }.v };
    pub const ONE:  Self = unsafe { UnionCast { i: [1; 16] }.v };
    pub const MIN:  Self = unsafe { UnionCast { i: [i8::MIN; 16] }.v };
    pub const MAX:  Self = unsafe { UnionCast { i: [i8::MAX; 16] }.v };

    // ── Constructors ──────────────────────────────────────────────────────────

    /// Broadcast `v` to all 16 lanes.
    #[inline(always)]
    pub fn splat(v: i8) -> Self {
        Self(unsafe { _mm_set1_epi8(v) })
    }

    /// Load from a byte slice reference — idiomatic for string/stream processing.
    #[inline(always)]
    pub fn from_array(a: [i8; 16]) -> Self {
        Self(unsafe { _mm_loadu_si128(a.as_ptr() as *const __m128i) })
    }

    /// Load from a `[u8; 16]` — common when processing raw byte streams.
    #[inline(always)]
    pub fn from_bytes(b: [u8; 16]) -> Self {
        // Safety: [u8;16] and [i8;16] have identical layout.
        Self(unsafe { _mm_loadu_si128(b.as_ptr() as *const __m128i) })
    }

    #[inline(always)]
    pub fn to_array(self) -> [i8; 16] {
        unsafe {
            let mut a = [0i8; 16];
            _mm_storeu_si128(a.as_mut_ptr() as *mut __m128i, self.0);
            a
        }
    }

    #[inline(always)]
    pub fn to_bytes(self) -> [u8; 16] {
        unsafe {
            let mut a = [0u8; 16];
            _mm_storeu_si128(a.as_mut_ptr() as *mut __m128i, self.0);
            a
        }
    }

    /// Extract a single lane. Panics if `i >= 16`.
    #[inline]
    pub fn get(self, i: usize) -> i8 {
        assert!(i < 16, "i8x16::get — lane {i} out of bounds (max 15)");
        unsafe { UnionCast { v: self }.i[i] }
    }

    // ── Arithmetic ────────────────────────────────────────────────────────────

    /// Saturating add — clamps to `[i8::MIN, i8::MAX]`. SSE2 native.
    #[inline(always)]
    pub fn saturating_add(self, rhs: Self) -> Self {
        Self(unsafe { _mm_adds_epi8(self.0, rhs.0) })
    }

    /// Saturating sub — clamps to `[i8::MIN, i8::MAX]`. SSE2 native.
    #[inline(always)]
    pub fn saturating_sub(self, rhs: Self) -> Self {
        Self(unsafe { _mm_subs_epi8(self.0, rhs.0) })
    }

    /// Absolute value per lane (wrapping — `abs(i8::MIN) == i8::MIN`).
    #[inline]
    pub fn abs(self) -> Self {
        unsafe {
            let zero = _mm_setzero_si128();
            let neg  = _mm_sub_epi8(zero, self.0);
            let mask = _mm_cmplt_epi8(self.0, zero); // 0xFF where self < 0
            Self(_mm_or_si128(
                _mm_and_si128(mask, neg),
                _mm_andnot_si128(mask, self.0),
            ))
        }
    }

    /// Per-lane minimum (signed).
    #[inline]
    pub fn min(self, rhs: Self) -> Self {
        let lt = self.cmplt(rhs);
        Self::blend(lt, self, rhs)
    }

    /// Per-lane maximum (signed).
    #[inline]
    pub fn max(self, rhs: Self) -> Self {
        let gt = self.cmpgt(rhs);
        Self::blend(gt, self, rhs)
    }

    #[inline(always)]
    pub fn clamp(self, lo: Self, hi: Self) -> Self { self.max(lo).min(hi) }

    #[inline]
    pub fn min_element(self) -> i8 {
        self.to_array().iter().copied().reduce(i8::min).unwrap()
    }

    #[inline]
    pub fn max_element(self) -> i8 {
        self.to_array().iter().copied().reduce(i8::max).unwrap()
    }

    /// Horizontal sum. Result is i32 to avoid multi-level overflow.
    #[inline]
    pub fn element_sum(self) -> i32 {
        self.to_array().iter().map(|&x| x as i32).sum()
    }

    // ── Comparisons → IMask16 ─────────────────────────────────────────────────

    #[inline(always)]
    pub fn cmpeq(self, rhs: Self) -> IMask16 {
        IMask16(unsafe { _mm_cmpeq_epi8(self.0, rhs.0) })
    }

    #[inline(always)]
    pub fn cmpne(self, rhs: Self) -> IMask16 { !self.cmpeq(rhs) }

    #[inline(always)]
    pub fn cmpgt(self, rhs: Self) -> IMask16 {
        IMask16(unsafe { _mm_cmpgt_epi8(self.0, rhs.0) })
    }

    #[inline(always)]
    pub fn cmplt(self, rhs: Self) -> IMask16 {
        IMask16(unsafe { _mm_cmplt_epi8(self.0, rhs.0) })
    }

    #[inline(always)]
    pub fn cmpge(self, rhs: Self) -> IMask16 { !self.cmplt(rhs) }

    #[inline(always)]
    pub fn cmple(self, rhs: Self) -> IMask16 { !self.cmpgt(rhs) }

    // ── Branchless select ─────────────────────────────────────────────────────

    #[inline(always)]
    pub fn blend(mask: IMask16, if_true: Self, if_false: Self) -> Self {
        unsafe {
            Self(_mm_or_si128(
                _mm_and_si128(mask.0, if_true.0),
                _mm_andnot_si128(mask.0, if_false.0),
            ))
        }
    }

    // ── Byte shuffle (SSSE3) ─────────────────────────────────────────────────
    //
    // PSHUFB: result[i] = if indices[i] & 0x80 != 0 { 0 } else { self[indices[i] & 0x0F] }
    // This zeroing behaviour is intentional and useful for filtering/masking.
    // Available on all CPUs since Penryn (2007) including the 2010 MacBook Pro.

    /// Byte shuffle using `indices`. Requires SSSE3.
    ///
    /// For each output lane `i`:
    /// - If `indices[i] & 0x80 != 0`: output lane is **zero**.
    /// - Else: output lane = `self[indices[i] & 0x0F]`.
    ///
    /// Primary use: reordering bytes for hashing, channel swizzling,
    /// broadcast patterns, and selective zeroing.
    #[cfg(target_feature = "ssse3")]
    #[inline(always)]
    pub fn shuffle_bytes(self, indices: Self) -> Self {
        unsafe {
            extern "C" {
                // LLVM will substitute the correct intrinsic.
            }
            // _mm_shuffle_epi8 is in the ssse3 module.
            // We must import it separately when the feature is active.
            #[cfg(target_arch = "x86")]
            use core::arch::x86::_mm_shuffle_epi8;
            #[cfg(target_arch = "x86_64")]
            use core::arch::x86_64::_mm_shuffle_epi8;
            Self(_mm_shuffle_epi8(self.0, indices.0))
        }
    }

    /// Scalar fallback for `shuffle_bytes` when SSSE3 is unavailable.
    #[cfg(not(target_feature = "ssse3"))]
    #[inline]
    pub fn shuffle_bytes(self, indices: Self) -> Self {
        let src = self.to_array();
        let idx = indices.to_array();
        let mut out = [0i8; 16];
        for i in 0..16 {
            let ix = idx[i];
            out[i] = if ix < 0 { 0 } else { src[(ix & 0x0F) as usize] };
        }
        Self::from_array(out)
    }

    // ── Count / population ────────────────────────────────────────────────────

    /// Number of lanes that compare equal to `needle`.
    #[inline]
    pub fn count_eq(self, needle: Self) -> u32 {
        self.cmpeq(needle).count_true()
    }

    /// True if any lane equals `needle`.
    #[inline]
    pub fn contains(self, needle: i8) -> bool {
        self.count_eq(Self::splat(needle)) > 0
    }

    // ── Widening ─────────────────────────────────────────────────────────────

    /// Sign-extend lower 8 lanes (0–7) to `i16` pairs via unpack with sign fill.
    #[inline(always)]
    pub fn as_i16x8_lo(self) -> super::i16x8::i16x8 {
        unsafe {
            let sign = _mm_cmplt_epi8(self.0, _mm_setzero_si128()); // 0xFF where negative
            super::i16x8::i16x8(_mm_unpacklo_epi8(self.0, sign))
        }
    }

    /// Sign-extend upper 8 lanes (8–15) to `i16`.
    #[inline(always)]
    pub fn as_i16x8_hi(self) -> super::i16x8::i16x8 {
        unsafe {
            let sign = _mm_cmplt_epi8(self.0, _mm_setzero_si128());
            super::i16x8::i16x8(_mm_unpackhi_epi8(self.0, sign))
        }
    }

    // ── Wrapping aliases ─────────────────────────────────────────────────────

    #[inline(always)] pub fn wrapping_add(self, r: Self) -> Self { self + r }
    #[inline(always)] pub fn wrapping_sub(self, r: Self) -> Self { self - r }
}

// ── Operators ─────────────────────────────────────────────────────────────────

impl Add for i8x16 {
    type Output = Self;
    #[inline(always)]
    fn add(self, r: Self) -> Self { Self(unsafe { _mm_add_epi8(self.0, r.0) }) }
}
impl AddAssign for i8x16 { #[inline(always)] fn add_assign(&mut self, r: Self) { *self = *self + r; } }

impl Sub for i8x16 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, r: Self) -> Self { Self(unsafe { _mm_sub_epi8(self.0, r.0) }) }
}
impl SubAssign for i8x16 { #[inline(always)] fn sub_assign(&mut self, r: Self) { *self = *self - r; } }

impl Neg for i8x16 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self { Self(unsafe { _mm_sub_epi8(_mm_setzero_si128(), self.0) }) }
}

impl BitAnd for i8x16 {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, r: Self) -> Self { Self(unsafe { _mm_and_si128(self.0, r.0) }) }
}
impl BitAndAssign for i8x16 { #[inline(always)] fn bitand_assign(&mut self, r: Self) { *self = *self & r; } }

impl BitOr for i8x16 {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, r: Self) -> Self { Self(unsafe { _mm_or_si128(self.0, r.0) }) }
}
impl BitOrAssign for i8x16 { #[inline(always)] fn bitor_assign(&mut self, r: Self) { *self = *self | r; } }

impl BitXor for i8x16 {
    type Output = Self;
    #[inline(always)]
    fn bitxor(self, r: Self) -> Self { Self(unsafe { _mm_xor_si128(self.0, r.0) }) }
}
impl BitXorAssign for i8x16 { #[inline(always)] fn bitxor_assign(&mut self, r: Self) { *self = *self ^ r; } }

impl Not for i8x16 {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        unsafe {
            let ones = _mm_cmpeq_epi8(self.0, self.0);
            Self(_mm_xor_si128(self.0, ones))
        }
    }
}

impl PartialEq for i8x16 {
    #[inline]
    fn eq(&self, r: &Self) -> bool {
        unsafe { _mm_movemask_epi8(_mm_cmpeq_epi8(self.0, r.0)) == 0xFFFF }
    }
}
impl Eq for i8x16 {}

impl fmt::Debug for i8x16 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let a = self.to_array();
        write!(f, "i8x16({:?})", a)
    }
}
impl fmt::Display for i8x16 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let a = self.to_array();
        write!(f, "{:?}", a)
    }
}
impl From<[i8; 16]> for i8x16 { #[inline] fn from(a: [i8;16]) -> Self { Self::from_array(a) } }
impl From<i8x16> for [i8; 16] { #[inline] fn from(v: i8x16) -> Self { v.to_array() } }
impl From<[u8; 16]> for i8x16 { #[inline] fn from(b: [u8;16]) -> Self { Self::from_bytes(b) } }
impl From<i8x16> for [u8; 16] { #[inline] fn from(v: i8x16) -> Self { v.to_bytes() } }
