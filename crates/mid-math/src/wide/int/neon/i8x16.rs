// crates/mid-math/src/wide/int/neon/i8x16.rs
//! 16-lane signed 8-bit integer vector — NEON, aarch64.
//!
//! Backed by `int8x16_t`. Mask: `IMask16` (uint8x16_t).
//!
//! NEON advantages over SSE2 for i8:
//!   vabsq_s8         — direct abs (SSE2: cmplt + blend)
//!   vminq/vmaxq_s8   — native (SSE2: no native i8 min/max before SSE4.1)
//!   vqaddq_s8        — saturating add native
//!   vqsubq_s8        — saturating sub native
//!   vmull_s8         — widening multiply → int16x8_t
//!   vmovl_s8         — sign-extend 8 lanes → int16x8_t in 1 instr
//!   vqtbl1q_u8       — ARMv8 table lookup = pshufb equivalent (shuffle_bytes)
//!   vnegq_s8         — direct negate
//!   vmvnq_s8         — direct NOT
//!   vaddlvq_s8       — widened horizontal sum → i32 (avoids overflow)
//!   vminvq/vmaxvq    — horizontal min/max in 1 instruction

#![allow(non_camel_case_types)]

use core::arch::aarch64::*;
use core::fmt;
use core::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign,
    BitXor, BitXorAssign, Mul, MulAssign, Neg, Not, Sub, SubAssign,
};
use super::imask16::IMask16;
use super::i16x8::i16x8;

#[repr(C)]
union UnionCast { i: [i8; 16], v: i8x16 }

/// 16-lane signed 8-bit integer vector. 16 bytes, 16-byte aligned.
/// Backed by `int8x16_t` on aarch64.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct i8x16(pub(crate) int8x16_t);

impl i8x16 {
    pub const ZERO: Self = unsafe { UnionCast { i: [0;       16] }.v };
    pub const ONE:  Self = unsafe { UnionCast { i: [1;       16] }.v };
    pub const MIN:  Self = unsafe { UnionCast { i: [i8::MIN; 16] }.v };
    pub const MAX:  Self = unsafe { UnionCast { i: [i8::MAX; 16] }.v };

    // ── Constructors ──────────────────────────────────────────────────────────

    #[inline(always)]
    pub fn splat(v: i8) -> Self { Self(unsafe { vdupq_n_s8(v) }) }

    #[inline(always)]
    pub fn new(
        a:i8,b:i8,c:i8,d:i8,e:i8,f:i8,g:i8,h:i8,
        i:i8,j:i8,k:i8,l:i8,m:i8,n:i8,o:i8,p:i8,
    ) -> Self {
        unsafe { UnionCast { i: [a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p] }.v }
    }

    #[inline(always)]
    pub fn from_array(a: [i8; 16]) -> Self {
        Self(unsafe { vld1q_s8(a.as_ptr()) })
    }

    /// Load from a `[u8; 16]` — common when processing raw byte streams.
    /// Added to match sse2/scalar (was missing — surfaced by a
    /// backend-agnostic bench calling it, which compiled fine against
    /// sse2/scalar but not neon).
    #[inline(always)]
    pub fn from_bytes(b: [u8; 16]) -> Self {
        Self(unsafe { vld1q_s8(b.as_ptr() as *const i8) })
    }

    #[inline(always)]
    pub fn to_array(self) -> [i8; 16] {
        let mut a = [0i8; 16];
        unsafe { vst1q_s8(a.as_mut_ptr(), self.0) };
        a
    }

    /// Matches sse2/scalar — see `from_bytes`.
    #[inline(always)]
    pub fn to_bytes(self) -> [u8; 16] {
        let mut a = [0u8; 16];
        unsafe { vst1q_s8(a.as_mut_ptr() as *mut i8, self.0) };
        a
    }

    #[inline]
    pub fn get(self, idx: usize) -> i8 {
        assert!(idx < 16, "i8x16::get — lane {idx} out of bounds");
        unsafe { UnionCast { v: self }.i[idx] }
    }

    // ── Count / population ───────────────────────────────────────────────────
    // Matches sse2/scalar — see `from_bytes`'s note.

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

    // ── Arithmetic ────────────────────────────────────────────────────────────

    /// Direct `vabsq_s8`. SSE2 had no native i8 abs (needed SSE3 `pabsb`).
    #[inline(always)] pub fn abs(self) -> Self { Self(unsafe { vabsq_s8(self.0) }) }
    /// `vminq_s8` — native signed i8 min. SSE2 had NO native i8 min.
    #[inline(always)] pub fn min(self, r: Self) -> Self { Self(unsafe { vminq_s8(self.0, r.0) }) }
    /// `vmaxq_s8` — native signed i8 max. SSE2 had NO native i8 max.
    #[inline(always)] pub fn max(self, r: Self) -> Self { Self(unsafe { vmaxq_s8(self.0, r.0) }) }
    #[inline(always)] pub fn clamp(self, lo: Self, hi: Self) -> Self { self.max(lo).min(hi) }
    /// `vminvq_s8` — single SMINV instruction.
    #[inline] pub fn min_element(self) -> i8 { unsafe { vminvq_s8(self.0) } }
    /// `vmaxvq_s8` — single SMAXV instruction.
    #[inline] pub fn max_element(self) -> i8 { unsafe { vmaxvq_s8(self.0) } }

    /// `vaddlvq_s8` — sign-widened sum → i32, avoids i8 overflow entirely.
    #[inline] pub fn element_sum(self) -> i32 { unsafe { vaddlvq_s8(self.0).into() } }

    // ── Saturating ────────────────────────────────────────────────────────────

    #[inline(always)] pub fn saturating_add(self, r: Self) -> Self { Self(unsafe { vqaddq_s8(self.0, r.0) }) }
    #[inline(always)] pub fn saturating_sub(self, r: Self) -> Self { Self(unsafe { vqsubq_s8(self.0, r.0) }) }

    // ── Shifts ────────────────────────────────────────────────────────────────

    #[inline(always)]
    pub fn shl(self, count: u32) -> Self {
        unsafe { Self(vshlq_s8(self.0, vdupq_n_s8(count as i8))) }
    }

    /// Arithmetic right shift (sign-extend).
    #[inline(always)]
    pub fn shr_arithmetic(self, count: u32) -> Self {
        unsafe { Self(vshlq_s8(self.0, vdupq_n_s8(-(count as i8)))) }
    }

    /// Logical right shift (zero-fill). Reinterpret as u8 for unsigned shift.
    #[inline(always)]
    pub fn shr_logical(self, count: u32) -> Self {
        unsafe {
            let u = vreinterpretq_u8_s8(self.0);
            Self(vreinterpretq_s8_u8(vshlq_u8(u, vdupq_n_s8(-(count as i8)))))
        }
    }

    // ── Shuffle / permute ─────────────────────────────────────────────────────

    /// Byte shuffle via `vqtbl1q_u8` — ARMv8 equivalent of SSSE3 `pshufb`.
    /// Each byte of `indices` selects a source lane; index ≥ 16 → zero output.
    /// Reinterprets self as u8 for the table lookup.
    #[inline(always)]
    pub fn shuffle_bytes(self, indices: Self) -> Self {
        unsafe {
            let tbl = vreinterpretq_u8_s8(self.0);
            let idx = vreinterpretq_u8_s8(indices.0);
            Self(vreinterpretq_s8_u8(vqtbl1q_u8(tbl, idx)))
        }
    }

    // ── Comparisons ───────────────────────────────────────────────────────────

    #[inline(always)] pub fn cmpeq(self, r: Self) -> IMask16 { IMask16(unsafe { vceqq_s8(self.0, r.0) }) }
    #[inline(always)] pub fn cmpne(self, r: Self) -> IMask16 { !self.cmpeq(r) }
    #[inline(always)] pub fn cmpgt(self, r: Self) -> IMask16 { IMask16(unsafe { vcgtq_s8(self.0, r.0) }) }
    #[inline(always)] pub fn cmplt(self, r: Self) -> IMask16 { IMask16(unsafe { vcltq_s8(self.0, r.0) }) }
    #[inline(always)] pub fn cmpge(self, r: Self) -> IMask16 { IMask16(unsafe { vcgeq_s8(self.0, r.0) }) }
    #[inline(always)] pub fn cmple(self, r: Self) -> IMask16 { IMask16(unsafe { vcleq_s8(self.0, r.0) }) }

    // ── Branchless select ─────────────────────────────────────────────────────

    /// `vbslq_u8` reinterpreted — ONE instruction.
    #[inline(always)]
    pub fn blend(mask: IMask16, t: Self, f: Self) -> Self {
        unsafe {
            Self(vreinterpretq_s8_u8(vbslq_u8(
                mask.0,
                vreinterpretq_u8_s8(t.0),
                vreinterpretq_u8_s8(f.0),
            )))
        }
    }

    // ── Widening ──────────────────────────────────────────────────────────────

    /// Sign-extend low 8 lanes → int16x8_t. Single `SXTL` instruction.
    #[inline(always)]
    pub fn as_i16x8_lo(self) -> i16x8 {
        unsafe { i16x8(vmovl_s8(vget_low_s8(self.0))) }
    }

    /// Sign-extend high 8 lanes → int16x8_t. Single `SXTL2` instruction.
    #[inline(always)]
    pub fn as_i16x8_hi(self) -> i16x8 {
        unsafe { i16x8(vmovl_high_s8(self.0)) }
    }

    /// Widening multiply: low 8 lanes × r low 8 lanes → int16x8_t.
    #[inline(always)]
    pub fn mul_widen_lo(self, r: Self) -> i16x8 {
        unsafe { i16x8(vmull_s8(vget_low_s8(self.0), vget_low_s8(r.0))) }
    }

    /// Widening multiply: high 8 lanes × r high 8 lanes → int16x8_t.
    #[inline(always)]
    pub fn mul_widen_hi(self, r: Self) -> i16x8 {
        unsafe { i16x8(vmull_high_s8(self.0, r.0)) }
    }

    /// Narrow two i16x8 to i8x16 with signed saturation.
    #[inline(always)]
    pub fn pack_i16x8(lo: i16x8, hi: i16x8) -> Self {
        unsafe { Self(vcombine_s8(vqmovn_s16(lo.0), vqmovn_s16(hi.0))) }
    }

    #[inline(always)] pub fn wrapping_add(self, r: Self) -> Self { self + r }
    #[inline(always)] pub fn wrapping_sub(self, r: Self) -> Self { self - r }
}

// ── Operators ─────────────────────────────────────────────────────────────────

impl Add for i8x16 {
    type Output = Self;
    #[inline(always)] fn add(self, r: Self) -> Self { Self(unsafe { vaddq_s8(self.0, r.0) }) }
}
impl AddAssign for i8x16 { #[inline(always)] fn add_assign(&mut self, r: Self) { *self = *self + r; } }
impl Sub for i8x16 {
    type Output = Self;
    #[inline(always)] fn sub(self, r: Self) -> Self { Self(unsafe { vsubq_s8(self.0, r.0) }) }
}
impl SubAssign for i8x16 { #[inline(always)] fn sub_assign(&mut self, r: Self) { *self = *self - r; } }
impl Neg for i8x16 {
    type Output = Self;
    #[inline(always)] fn neg(self) -> Self { Self(unsafe { vnegq_s8(self.0) }) }
}
/// Note: i8 has no native `vmulq_s8`. Must widen, multiply, narrow.
/// We expose this honestly — `*` is NOT available on i8x16.
/// Use `mul_widen_lo` / `mul_widen_hi` + `pack_i16x8` instead.
impl Mul for i8x16 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, r: Self) -> Self {
        // Widen both halves, multiply, saturating-narrow back.
        let lo = self.mul_widen_lo(r);
        let hi = self.mul_widen_hi(r);
        Self::pack_i16x8(lo, hi)
    }
}
impl MulAssign for i8x16 { #[inline(always)] fn mul_assign(&mut self, r: Self) { *self = *self * r; } }
impl BitAnd for i8x16 {
    type Output = Self;
    #[inline(always)] fn bitand(self, r: Self) -> Self { Self(unsafe { vandq_s8(self.0, r.0) }) }
}
impl BitAndAssign for i8x16 { #[inline(always)] fn bitand_assign(&mut self, r: Self) { *self = *self & r; } }
impl BitOr for i8x16 {
    type Output = Self;
    #[inline(always)] fn bitor(self, r: Self) -> Self { Self(unsafe { vorrq_s8(self.0, r.0) }) }
}
impl BitOrAssign for i8x16 { #[inline(always)] fn bitor_assign(&mut self, r: Self) { *self = *self | r; } }
impl BitXor for i8x16 {
    type Output = Self;
    #[inline(always)] fn bitxor(self, r: Self) -> Self { Self(unsafe { veorq_s8(self.0, r.0) }) }
}
impl BitXorAssign for i8x16 { #[inline(always)] fn bitxor_assign(&mut self, r: Self) { *self = *self ^ r; } }
impl Not for i8x16 {
    type Output = Self;
    #[inline(always)] fn not(self) -> Self { Self(unsafe { vmvnq_s8(self.0) }) }
}

impl PartialEq for i8x16 {
    #[inline]
    fn eq(&self, r: &Self) -> bool {
        unsafe { vminvq_u8(vceqq_s8(self.0, r.0)) == u8::MAX }
    }
}
impl Eq for i8x16 {}

impl fmt::Debug for i8x16 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let a = self.to_array();
        write!(f, "i8x16({},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{})",
            a[0],a[1],a[2],a[3],a[4],a[5],a[6],a[7],
            a[8],a[9],a[10],a[11],a[12],a[13],a[14],a[15])
    }
}
impl fmt::Display for i8x16 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let a = self.to_array();
        write!(f, "[{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}]",
            a[0],a[1],a[2],a[3],a[4],a[5],a[6],a[7],
            a[8],a[9],a[10],a[11],a[12],a[13],a[14],a[15])
    }
}
impl From<[i8; 16]> for i8x16 { #[inline] fn from(a: [i8;16]) -> Self { Self::from_array(a) } }
impl From<i8x16> for [i8; 16] { #[inline] fn from(v: i8x16) -> Self { v.to_array() } }
