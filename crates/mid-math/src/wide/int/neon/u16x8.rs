// crates/mid-math/src/wide/int/neon/u16x8.rs
//! 8-lane unsigned 16-bit integer vector — NEON, aarch64.
//!
//! Backed by `uint16x8_t`. Mask: `IMask8` (uint16x8_t).
//!
//! NEON advantages over SSE2 for u16:
//!   vminq/vmaxq_u16  — native unsigned min/max (SSE2 has these)
//!   vcgtq/vcltq_u16  — native unsigned compare (SSE2 needs XOR-bias)
//!   vqaddq_u16       — saturating add native
//!   vqsubq_u16       — saturating sub native
//!   vaddvq_u16       — horizontal sum in 1 ADDV
//!   vmull_u16        — widening multiply to uint32x4_t
//!   vmovl_u16        — zero-extend 4 lanes to u32x4 in 1 instruction

#![allow(non_camel_case_types)]

use core::arch::aarch64::*;
use core::fmt;
use core::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign,
    BitXor, BitXorAssign, Mul, MulAssign, Not, Sub, SubAssign,
};
use super::imask8::IMask8;
use super::u32x4::u32x4;

#[repr(C)]
union UnionCast { u: [u16; 8], v: u16x8 }

/// 8-lane unsigned 16-bit integer vector. 16 bytes, 16-byte aligned.
/// Backed by `uint16x8_t` on aarch64.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct u16x8(pub(crate) uint16x8_t);

impl u16x8 {
    pub const ZERO: Self = unsafe { UnionCast { u: [0;        8] }.v };
    pub const ONE:  Self = unsafe { UnionCast { u: [1;        8] }.v };
    pub const MIN:  Self = unsafe { UnionCast { u: [u16::MIN; 8] }.v };
    pub const MAX:  Self = unsafe { UnionCast { u: [u16::MAX; 8] }.v };

    // ── Constructors ──────────────────────────────────────────────────────────

    #[inline(always)]
    pub fn splat(v: u16) -> Self { Self(unsafe { vdupq_n_u16(v) }) }

    #[inline(always)]
    pub fn new(a:u16,b:u16,c:u16,d:u16,e:u16,f:u16,g:u16,h:u16) -> Self {
        unsafe { UnionCast { u: [a,b,c,d,e,f,g,h] }.v }
    }

    #[inline(always)]
    pub fn from_array(a: [u16; 8]) -> Self {
        Self(unsafe { vld1q_u16(a.as_ptr()) })
    }

    #[inline(always)]
    pub fn to_array(self) -> [u16; 8] {
        let mut a = [0u16; 8];
        unsafe { vst1q_u16(a.as_mut_ptr(), self.0) };
        a
    }

    #[inline]
    pub fn get(self, i: usize) -> u16 {
        assert!(i < 8, "u16x8::get — lane {i} out of bounds");
        unsafe { UnionCast { v: self }.u[i] }
    }

    // ── Arithmetic ────────────────────────────────────────────────────────────

    /// `vminq_u16` — native unsigned min.
    #[inline(always)] pub fn min(self, r: Self) -> Self { Self(unsafe { vminq_u16(self.0, r.0) }) }
    /// `vmaxq_u16` — native unsigned max.
    #[inline(always)] pub fn max(self, r: Self) -> Self { Self(unsafe { vmaxq_u16(self.0, r.0) }) }
    #[inline(always)] pub fn clamp(self, lo: Self, hi: Self) -> Self { self.max(lo).min(hi) }
    /// `vminvq_u16` — single UMINV instruction.
    #[inline] pub fn min_element(self) -> u16 { unsafe { vminvq_u16(self.0) } }
    /// `vmaxvq_u16` — single UMAXV instruction.
    #[inline] pub fn max_element(self) -> u16 { unsafe { vmaxvq_u16(self.0) } }
    /// `vaddvq_u16` — single ADDV instruction.
    #[inline] pub fn element_sum(self) -> u32 {
        unsafe { vaddvq_u16(self.0) as u32 }
    }

    // ── Multiply variants ─────────────────────────────────────────────────────

    /// Low 16 bits of 16×16 multiply. `vmulq_u16`.
    #[inline(always)] pub fn mul_lo(self, r: Self) -> Self { Self(unsafe { vmulq_u16(self.0, r.0) }) }

    /// High 16 bits of unsigned 16×16 multiply via widening then shift.
    /// No single `vmulhq_u16` on NEON — use widening multiply + shift.
    #[inline(always)]
    pub fn mul_high_u(self, r: Self) -> Self {
        unsafe {
            // Widen low halves, multiply, shift right 16, narrow back
            let lo = vshrn_n_u32::<16>(vmull_u16(vget_low_u16(self.0), vget_low_u16(r.0)));
            let hi = vshrn_n_u32::<16>(vmull_high_u16(self.0, r.0));
            Self(vcombine_u16(lo, hi))
        }
    }

    /// Widening multiply: low 4 lanes → uint32x4_t.
    #[inline(always)]
    pub fn mul_widen_lo(self, r: Self) -> u32x4 {
        unsafe { u32x4(vmull_u16(vget_low_u16(self.0), vget_low_u16(r.0))) }
    }

    /// Widening multiply: high 4 lanes → uint32x4_t.
    #[inline(always)]
    pub fn mul_widen_hi(self, r: Self) -> u32x4 {
        unsafe { u32x4(vmull_high_u16(self.0, r.0)) }
    }

    // ── Saturating ────────────────────────────────────────────────────────────

    #[inline(always)] pub fn saturating_add(self, r: Self) -> Self { Self(unsafe { vqaddq_u16(self.0, r.0) }) }
    #[inline(always)] pub fn saturating_sub(self, r: Self) -> Self { Self(unsafe { vqsubq_u16(self.0, r.0) }) }

    // ── Shifts ────────────────────────────────────────────────────────────────

    /// Logical left shift.
    #[inline(always)]
    pub fn shl(self, count: u32) -> Self {
        unsafe { Self(vshlq_u16(self.0, vdupq_n_s16(count as i16))) }
    }

    /// Logical right shift (zero-fill).
    #[inline(always)]
    pub fn shr(self, count: u32) -> Self {
        unsafe { Self(vshlq_u16(self.0, vdupq_n_s16(-(count as i16)))) }
    }

    // ── Comparisons ───────────────────────────────────────────────────────────
    // All unsigned — no XOR-bias trick needed unlike SSE2.

    #[inline(always)] pub fn cmpeq(self, r: Self) -> IMask8 { IMask8(unsafe { vceqq_u16(self.0, r.0) }) }
    #[inline(always)] pub fn cmpne(self, r: Self) -> IMask8 { !self.cmpeq(r) }
    #[inline(always)] pub fn cmpgt(self, r: Self) -> IMask8 { IMask8(unsafe { vcgtq_u16(self.0, r.0) }) }
    #[inline(always)] pub fn cmplt(self, r: Self) -> IMask8 { IMask8(unsafe { vcltq_u16(self.0, r.0) }) }
    #[inline(always)] pub fn cmpge(self, r: Self) -> IMask8 { IMask8(unsafe { vcgeq_u16(self.0, r.0) }) }
    #[inline(always)] pub fn cmple(self, r: Self) -> IMask8 { IMask8(unsafe { vcleq_u16(self.0, r.0) }) }

    // ── Branchless select ─────────────────────────────────────────────────────

    /// `vbslq_u16` — ONE instruction, no reinterpret needed.
    #[inline(always)]
    pub fn blend(mask: IMask8, t: Self, f: Self) -> Self {
        unsafe { Self(vbslq_u16(mask.0, t.0, f.0)) }
    }

    // ── Widening ─────────────────────────────────────────────────────────────

    /// Zero-extend low 4 lanes to u32x4. Single `UXTL` instruction.
    #[inline(always)]
    pub fn as_u32x4_lo(self) -> u32x4 {
        unsafe { u32x4(vmovl_u16(vget_low_u16(self.0))) }
    }

    /// Zero-extend high 4 lanes to u32x4. Single `UXTL2` instruction.
    #[inline(always)]
    pub fn as_u32x4_hi(self) -> u32x4 {
        unsafe { u32x4(vmovl_high_u16(self.0)) }
    }

    /// Narrow two u32x4 to u16x8 with unsigned saturation.
    /// `vqmovn_u32` + `vcombine_u16`.
    #[inline(always)]
    pub fn pack_u32x4(lo: u32x4, hi: u32x4) -> Self {
        unsafe { Self(vcombine_u16(vqmovn_u32(lo.0), vqmovn_u32(hi.0))) }
    }

    #[inline(always)] pub fn wrapping_add(self, r: Self) -> Self { self + r }
    #[inline(always)] pub fn wrapping_sub(self, r: Self) -> Self { self - r }
    #[inline(always)] pub fn wrapping_mul(self, r: Self) -> Self { self.mul_lo(r) }
}

// ── Operators ─────────────────────────────────────────────────────────────────

impl Add for u16x8 {
    type Output = Self;
    #[inline(always)] fn add(self, r: Self) -> Self { Self(unsafe { vaddq_u16(self.0, r.0) }) }
}
impl AddAssign for u16x8 { #[inline(always)] fn add_assign(&mut self, r: Self) { *self = *self + r; } }
impl Sub for u16x8 {
    type Output = Self;
    #[inline(always)] fn sub(self, r: Self) -> Self { Self(unsafe { vsubq_u16(self.0, r.0) }) }
}
impl SubAssign for u16x8 { #[inline(always)] fn sub_assign(&mut self, r: Self) { *self = *self - r; } }
impl Mul for u16x8 {
    type Output = Self;
    #[inline(always)] fn mul(self, r: Self) -> Self { self.mul_lo(r) }
}
impl MulAssign for u16x8 { #[inline(always)] fn mul_assign(&mut self, r: Self) { *self = *self * r; } }
impl BitAnd for u16x8 {
    type Output = Self;
    #[inline(always)] fn bitand(self, r: Self) -> Self { Self(unsafe { vandq_u16(self.0, r.0) }) }
}
impl BitAndAssign for u16x8 { #[inline(always)] fn bitand_assign(&mut self, r: Self) { *self = *self & r; } }
impl BitOr for u16x8 {
    type Output = Self;
    #[inline(always)] fn bitor(self, r: Self) -> Self { Self(unsafe { vorrq_u16(self.0, r.0) }) }
}
impl BitOrAssign for u16x8 { #[inline(always)] fn bitor_assign(&mut self, r: Self) { *self = *self | r; } }
impl BitXor for u16x8 {
    type Output = Self;
    #[inline(always)] fn bitxor(self, r: Self) -> Self { Self(unsafe { veorq_u16(self.0, r.0) }) }
}
impl BitXorAssign for u16x8 { #[inline(always)] fn bitxor_assign(&mut self, r: Self) { *self = *self ^ r; } }
impl Not for u16x8 {
    type Output = Self;
    #[inline(always)] fn not(self) -> Self { Self(unsafe { vmvnq_u16(self.0) }) }
}

impl PartialEq for u16x8 {
    #[inline]
    fn eq(&self, r: &Self) -> bool {
        unsafe { vminvq_u16(vceqq_u16(self.0, r.0)) == u16::MAX }
    }
}
impl Eq for u16x8 {}

impl fmt::Debug for u16x8 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let a = self.to_array();
        write!(f, "u16x8({},{},{},{},{},{},{},{})",
            a[0],a[1],a[2],a[3],a[4],a[5],a[6],a[7])
    }
}
impl fmt::Display for u16x8 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let a = self.to_array();
        write!(f, "[{},{},{},{},{},{},{},{}]",
            a[0],a[1],a[2],a[3],a[4],a[5],a[6],a[7])
    }
}
impl From<[u16; 8]> for u16x8 { #[inline] fn from(a: [u16;8]) -> Self { Self::from_array(a) } }
impl From<u16x8> for [u16; 8] { #[inline] fn from(v: u16x8) -> Self { v.to_array() } }
