// crates/mid-math/src/wide/int/neon/u8x16.rs
//! 16-lane unsigned 8-bit integer vector — NEON, aarch64.
//!
//! Backed by `uint8x16_t`. Mask: `IMask16` (uint8x16_t).
//!
//! NEON advantages over SSE2 for u8:
//!   vminq/vmaxq_u8   — native (SSE2 has _epi8 unsigned min/max, rare)
//!   vcgtq/vcltq_u8   — native unsigned compare (SSE2: XOR-bias)
//!   vqaddq_u8        — saturating add native
//!   vqsubq_u8        — saturating sub native
//!   vqtbl1q_u8       — ARMv8 table lookup (shuffle_bytes), index≥16 → 0
//!   vmull_u8         — widening multiply → uint16x8_t
//!   vmovl_u8         — zero-extend 8 lanes → uint16x8_t in 1 instr
//!   vaddlvq_u8       — widened horizontal sum → u32

#![allow(non_camel_case_types)]

use core::arch::aarch64::*;
use core::fmt;
use core::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign,
    BitXor, BitXorAssign, Mul, MulAssign, Not, Sub, SubAssign,
};
use super::imask16::IMask16;
use super::u16x8::u16x8;

#[repr(C)]
union UnionCast { u: [u8; 16], v: u8x16 }

/// 16-lane unsigned 8-bit integer vector. 16 bytes, 16-byte aligned.
/// Backed by `uint8x16_t` on aarch64.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct u8x16(pub(crate) uint8x16_t);

impl u8x16 {
    pub const ZERO: Self = unsafe { UnionCast { u: [0;       16] }.v };
    pub const ONE:  Self = unsafe { UnionCast { u: [1;       16] }.v };
    pub const MIN:  Self = unsafe { UnionCast { u: [u8::MIN; 16] }.v };
    pub const MAX:  Self = unsafe { UnionCast { u: [u8::MAX; 16] }.v };

    // ── Constructors ──────────────────────────────────────────────────────────

    #[inline(always)]
    pub fn splat(v: u8) -> Self { Self(unsafe { vdupq_n_u8(v) }) }

    #[inline(always)]
    pub fn new(
        a:u8,b:u8,c:u8,d:u8,e:u8,f:u8,g:u8,h:u8,
        i:u8,j:u8,k:u8,l:u8,m:u8,n:u8,o:u8,p:u8,
    ) -> Self {
        unsafe { UnionCast { u: [a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p] }.v }
    }

    #[inline(always)]
    pub fn from_array(a: [u8; 16]) -> Self {
        Self(unsafe { vld1q_u8(a.as_ptr()) })
    }

    #[inline(always)]
    pub fn to_array(self) -> [u8; 16] {
        let mut a = [0u8; 16];
        unsafe { vst1q_u8(a.as_mut_ptr(), self.0) };
        a
    }

    #[inline]
    pub fn get(self, idx: usize) -> u8 {
        assert!(idx < 16, "u8x16::get — lane {idx} out of bounds");
        unsafe { UnionCast { v: self }.u[idx] }
    }

    // ── Arithmetic ────────────────────────────────────────────────────────────

    /// `vminq_u8` — native unsigned min.
    #[inline(always)] pub fn min(self, r: Self) -> Self { Self(unsafe { vminq_u8(self.0, r.0) }) }
    /// `vmaxq_u8` — native unsigned max.
    #[inline(always)] pub fn max(self, r: Self) -> Self { Self(unsafe { vmaxq_u8(self.0, r.0) }) }
    #[inline(always)] pub fn clamp(self, lo: Self, hi: Self) -> Self { self.max(lo).min(hi) }
    /// `vminvq_u8` — single UMINV instruction.
    #[inline] pub fn min_element(self) -> u8 { unsafe { vminvq_u8(self.0) } }
    /// `vmaxvq_u8` — single UMAXV instruction.
    #[inline] pub fn max_element(self) -> u8 { unsafe { vmaxvq_u8(self.0) } }
    /// `vaddlvq_u8` — zero-widened sum → u32, avoids u8 overflow.
    #[inline] pub fn element_sum(self) -> u32 { unsafe { vaddlvq_u8(self.0).into() } }

    // ── Saturating ────────────────────────────────────────────────────────────

    #[inline(always)] pub fn saturating_add(self, r: Self) -> Self { Self(unsafe { vqaddq_u8(self.0, r.0) }) }
    #[inline(always)] pub fn saturating_sub(self, r: Self) -> Self { Self(unsafe { vqsubq_u8(self.0, r.0) }) }

    // ── Shifts ────────────────────────────────────────────────────────────────

    #[inline(always)]
    pub fn shl(self, count: u32) -> Self {
        unsafe { Self(vshlq_u8(self.0, vdupq_n_s8(count as i8))) }
    }

    /// Logical right shift (zero-fill).
    #[inline(always)]
    pub fn shr(self, count: u32) -> Self {
        unsafe { Self(vshlq_u8(self.0, vdupq_n_s8(-(count as i8)))) }
    }

    // ── Shuffle / permute ─────────────────────────────────────────────────────

    /// `vqtbl1q_u8` — ARMv8 table lookup. Index ≥ 16 produces 0.
    /// Direct equivalent of SSSE3 `pshufb` for unsigned bytes.
    #[inline(always)]
    pub fn shuffle_bytes(self, indices: Self) -> Self {
        unsafe { Self(vqtbl1q_u8(self.0, indices.0)) }
    }

    // ── Comparisons ───────────────────────────────────────────────────────────
    // All native unsigned — no XOR-bias needed unlike SSE2.

    #[inline(always)] pub fn cmpeq(self, r: Self) -> IMask16 { IMask16(unsafe { vceqq_u8(self.0, r.0) }) }
    #[inline(always)] pub fn cmpne(self, r: Self) -> IMask16 { !self.cmpeq(r) }
    /// Native unsigned gt. SSE2 had no `pcmpgtb` unsigned.
    #[inline(always)] pub fn cmpgt(self, r: Self) -> IMask16 { IMask16(unsafe { vcgtq_u8(self.0, r.0) }) }
    #[inline(always)] pub fn cmplt(self, r: Self) -> IMask16 { IMask16(unsafe { vcltq_u8(self.0, r.0) }) }
    #[inline(always)] pub fn cmpge(self, r: Self) -> IMask16 { IMask16(unsafe { vcgeq_u8(self.0, r.0) }) }
    #[inline(always)] pub fn cmple(self, r: Self) -> IMask16 { IMask16(unsafe { vcleq_u8(self.0, r.0) }) }

    // ── Branchless select ─────────────────────────────────────────────────────

    /// `vbslq_u8` — ONE instruction, no reinterpret needed.
    #[inline(always)]
    pub fn blend(mask: IMask16, t: Self, f: Self) -> Self {
        unsafe { Self(vbslq_u8(mask.0, t.0, f.0)) }
    }

    // ── Widening ─────────────────────────────────────────────────────────────

    /// Zero-extend low 8 lanes → uint16x8_t. Single `UXTL` instruction.
    #[inline(always)]
    pub fn as_u16x8_lo(self) -> u16x8 {
        unsafe { u16x8(vmovl_u8(vget_low_u8(self.0))) }
    }

    /// Zero-extend high 8 lanes → uint16x8_t. Single `UXTL2` instruction.
    #[inline(always)]
    pub fn as_u16x8_hi(self) -> u16x8 {
        unsafe { u16x8(vmovl_high_u8(self.0)) }
    }

    /// Widening multiply: low 8 lanes × r low 8 lanes → uint16x8_t.
    #[inline(always)]
    pub fn mul_widen_lo(self, r: Self) -> u16x8 {
        unsafe { u16x8(vmull_u8(vget_low_u8(self.0), vget_low_u8(r.0))) }
    }

    /// Widening multiply: high 8 lanes × r high 8 lanes → uint16x8_t.
    #[inline(always)]
    pub fn mul_widen_hi(self, r: Self) -> u16x8 {
        unsafe { u16x8(vmull_high_u8(self.0, r.0)) }
    }

    /// Narrow two u16x8 to u8x16 with unsigned saturation.
    #[inline(always)]
    pub fn pack_u16x8(lo: u16x8, hi: u16x8) -> Self {
        unsafe { Self(vcombine_u8(vqmovn_u16(lo.0), vqmovn_u16(hi.0))) }
    }

    #[inline(always)] pub fn wrapping_add(self, r: Self) -> Self { self + r }
    #[inline(always)] pub fn wrapping_sub(self, r: Self) -> Self { self - r }
}

// ── Operators ─────────────────────────────────────────────────────────────────

impl Add for u8x16 {
    type Output = Self;
    #[inline(always)] fn add(self, r: Self) -> Self { Self(unsafe { vaddq_u8(self.0, r.0) }) }
}
impl AddAssign for u8x16 { #[inline(always)] fn add_assign(&mut self, r: Self) { *self = *self + r; } }
impl Sub for u8x16 {
    type Output = Self;
    #[inline(always)] fn sub(self, r: Self) -> Self { Self(unsafe { vsubq_u8(self.0, r.0) }) }
}
impl SubAssign for u8x16 { #[inline(always)] fn sub_assign(&mut self, r: Self) { *self = *self - r; } }
/// No native `vmulq_u8` — widen, multiply, saturating-narrow back.
impl Mul for u8x16 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, r: Self) -> Self {
        let lo = self.mul_widen_lo(r);
        let hi = self.mul_widen_hi(r);
        Self::pack_u16x8(lo, hi)
    }
}
impl MulAssign for u8x16 { #[inline(always)] fn mul_assign(&mut self, r: Self) { *self = *self * r; } }
impl BitAnd for u8x16 {
    type Output = Self;
    #[inline(always)] fn bitand(self, r: Self) -> Self { Self(unsafe { vandq_u8(self.0, r.0) }) }
}
impl BitAndAssign for u8x16 { #[inline(always)] fn bitand_assign(&mut self, r: Self) { *self = *self & r; } }
impl BitOr for u8x16 {
    type Output = Self;
    #[inline(always)] fn bitor(self, r: Self) -> Self { Self(unsafe { vorrq_u8(self.0, r.0) }) }
}
impl BitOrAssign for u8x16 { #[inline(always)] fn bitor_assign(&mut self, r: Self) { *self = *self | r; } }
impl BitXor for u8x16 {
    type Output = Self;
    #[inline(always)] fn bitxor(self, r: Self) -> Self { Self(unsafe { veorq_u8(self.0, r.0) }) }
}
impl BitXorAssign for u8x16 { #[inline(always)] fn bitxor_assign(&mut self, r: Self) { *self = *self ^ r; } }
impl Not for u8x16 {
    type Output = Self;
    #[inline(always)] fn not(self) -> Self { Self(unsafe { vmvnq_u8(self.0) }) }
}

impl PartialEq for u8x16 {
    #[inline]
    fn eq(&self, r: &Self) -> bool {
        unsafe { vminvq_u8(vceqq_u8(self.0, r.0)) == u8::MAX }
    }
}
impl Eq for u8x16 {}

impl fmt::Debug for u8x16 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let a = self.to_array();
        write!(f, "u8x16({},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{})",
            a[0],a[1],a[2],a[3],a[4],a[5],a[6],a[7],
            a[8],a[9],a[10],a[11],a[12],a[13],a[14],a[15])
    }
}
impl fmt::Display for u8x16 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let a = self.to_array();
        write!(f, "[{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}]",
            a[0],a[1],a[2],a[3],a[4],a[5],a[6],a[7],
            a[8],a[9],a[10],a[11],a[12],a[13],a[14],a[15])
    }
}
impl From<[u8; 16]> for u8x16 { #[inline] fn from(a: [u8;16]) -> Self { Self::from_array(a) } }
impl From<u8x16> for [u8; 16] { #[inline] fn from(v: u8x16) -> Self { v.to_array() } }
