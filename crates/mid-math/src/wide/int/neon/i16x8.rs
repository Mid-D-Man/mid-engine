// crates/mid-math/src/wide/int/neon/i16x8.rs
//! 8-lane signed 16-bit integer vector — NEON, aarch64.
//!
//! Backed by `int16x8_t`. Mask: `IMask8` (uint16x8_t).
//!
//! NEON advantages over SSE2 for i16:
//!   vabsq_s16        — direct abs (SSE2: cmplt + blend)
//!   vminq/vmaxq_s16  — native (SSE2 has these but not for all widths)
//!   vaddvq_s16       — horizontal sum in 1 ADDV instruction
//!   vminvq/vmaxvq    — horizontal min/max in 1 instruction
//!   vqaddq_s16       — saturating add native
//!   vqsubq_s16       — saturating sub native
//!   vqdmulhq_s16     — saturating doubling multiply high (audio DSP)
//!   vnegq_s16        — direct negate
//!   vmvnq_s16        — direct NOT
//!   vshlq_s16        — unified shift (pos=left, neg=arith-right)
//!   vmull_s16        — widening multiply to int32x4_t (lower 4 lanes)

#![allow(non_camel_case_types)]

use core::arch::aarch64::*;
use core::fmt;
use core::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign,
    BitXor, BitXorAssign, Mul, MulAssign, Neg, Not, Sub, SubAssign,
};
use super::imask8::IMask8;
use super::i32x4::i32x4;

#[repr(C)]
union UnionCast { i: [i16; 8], v: i16x8 }

/// 8-lane signed 16-bit integer vector. 16 bytes, 16-byte aligned.
/// Backed by `int16x8_t` on aarch64.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct i16x8(pub(crate) int16x8_t);

impl i16x8 {
    pub const ZERO: Self = unsafe { UnionCast { i: [0;        8] }.v };
    pub const ONE:  Self = unsafe { UnionCast { i: [1;        8] }.v };
    pub const MIN:  Self = unsafe { UnionCast { i: [i16::MIN; 8] }.v };
    pub const MAX:  Self = unsafe { UnionCast { i: [i16::MAX; 8] }.v };

    // ── Constructors ──────────────────────────────────────────────────────────

    #[inline(always)]
    pub fn splat(v: i16) -> Self { Self(unsafe { vdupq_n_s16(v) }) }

    #[inline(always)]
    pub fn new(a:i16,b:i16,c:i16,d:i16,e:i16,f:i16,g:i16,h:i16) -> Self {
        unsafe { UnionCast { i: [a,b,c,d,e,f,g,h] }.v }
    }

    #[inline(always)]
    pub fn from_array(a: [i16; 8]) -> Self {
        Self(unsafe { vld1q_s16(a.as_ptr()) })
    }

    #[inline(always)]
    pub fn to_array(self) -> [i16; 8] {
        let mut a = [0i16; 8];
        unsafe { vst1q_s16(a.as_mut_ptr(), self.0) };
        a
    }

    #[inline]
    pub fn get(self, i: usize) -> i16 {
        assert!(i < 8, "i16x8::get — lane {i} out of bounds");
        unsafe { UnionCast { v: self }.i[i] }
    }

    // ── Arithmetic ────────────────────────────────────────────────────────────

    /// Direct `vabsq_s16` — single instruction.
    #[inline(always)] pub fn abs(self) -> Self { Self(unsafe { vabsq_s16(self.0) }) }
    #[inline(always)] pub fn min(self, r: Self) -> Self { Self(unsafe { vminq_s16(self.0, r.0) }) }
    #[inline(always)] pub fn max(self, r: Self) -> Self { Self(unsafe { vmaxq_s16(self.0, r.0) }) }
    #[inline(always)] pub fn clamp(self, lo: Self, hi: Self) -> Self { self.max(lo).min(hi) }
    /// `vminvq_s16` — single SMINV instruction.
    #[inline] pub fn min_element(self) -> i16 { unsafe { vminvq_s16(self.0) } }
    /// `vmaxvq_s16` — single SMAXV instruction.
    #[inline] pub fn max_element(self) -> i16 { unsafe { vmaxvq_s16(self.0) } }
    /// `vaddvq_s16` — single ADDV instruction.
    #[inline] pub fn element_sum(self) -> i32 {
        // widen to i32 to avoid overflow on full sum
        unsafe { vaddvq_s16(self.0) as i32 }
    }

    // ── Multiply variants ─────────────────────────────────────────────────────

    /// Low 16 bits of 16×16 multiply. `vmulq_s16`.
    #[inline(always)] pub fn mul_lo(self, r: Self) -> Self { Self(unsafe { vmulq_s16(self.0, r.0) }) }

    /// High 16 bits of signed 16×16 multiply. `vqdmulhq_s16` gives
    /// `(2*a*b) >> 16` with saturation — standard DSP multiply-high.
    #[inline(always)] pub fn mul_high(self, r: Self) -> Self { Self(unsafe { vqdmulhq_s16(self.0, r.0) }) }

    /// Widening multiply: low 4 lanes → int32x4_t.
    /// `vmull_s16` takes the low half (int16x4_t) of each operand.
    #[inline(always)]
    pub fn mul_widen_lo(self, r: Self) -> i32x4 {
        unsafe { i32x4(vmull_s16(vget_low_s16(self.0), vget_low_s16(r.0))) }
    }

    /// Widening multiply: high 4 lanes → int32x4_t.
    #[inline(always)]
    pub fn mul_widen_hi(self, r: Self) -> i32x4 {
        unsafe { i32x4(vmull_high_s16(self.0, r.0)) }
    }

    // ── Saturating ────────────────────────────────────────────────────────────

    #[inline(always)] pub fn saturating_add(self, r: Self) -> Self { Self(unsafe { vqaddq_s16(self.0, r.0) }) }
    #[inline(always)] pub fn saturating_sub(self, r: Self) -> Self { Self(unsafe { vqsubq_s16(self.0, r.0) }) }

    // ── Shifts ────────────────────────────────────────────────────────────────

    /// Logical left shift. `vshlq_s16` with positive count.
    #[inline(always)]
    pub fn shl(self, count: u32) -> Self {
        unsafe { Self(vshlq_s16(self.0, vdupq_n_s16(count as i16))) }
    }

    /// Arithmetic right shift (sign-extend). `vshlq_s16` with negative count.
    #[inline(always)]
    pub fn shr_arithmetic(self, count: u32) -> Self {
        unsafe { Self(vshlq_s16(self.0, vdupq_n_s16(-(count as i16)))) }
    }

    /// Logical right shift (zero-fill). Reinterpret as u16, shift, reinterpret back.
    #[inline(always)]
    pub fn shr_logical(self, count: u32) -> Self {
        unsafe {
            let u = vreinterpretq_u16_s16(self.0);
            Self(vreinterpretq_s16_u16(vshlq_u16(u, vdupq_n_s16(-(count as i16)))))
        }
    }

    // ── Comparisons ───────────────────────────────────────────────────────────

    #[inline(always)] pub fn cmpeq(self, r: Self) -> IMask8 { IMask8(unsafe { vceqq_s16(self.0, r.0) }) }
    #[inline(always)] pub fn cmpne(self, r: Self) -> IMask8 { !self.cmpeq(r) }
    #[inline(always)] pub fn cmpgt(self, r: Self) -> IMask8 { IMask8(unsafe { vcgtq_s16(self.0, r.0) }) }
    #[inline(always)] pub fn cmplt(self, r: Self) -> IMask8 { IMask8(unsafe { vcltq_s16(self.0, r.0) }) }
    #[inline(always)] pub fn cmpge(self, r: Self) -> IMask8 { IMask8(unsafe { vcgeq_s16(self.0, r.0) }) }
    #[inline(always)] pub fn cmple(self, r: Self) -> IMask8 { IMask8(unsafe { vcleq_s16(self.0, r.0) }) }

    // ── Branchless select ─────────────────────────────────────────────────────

    /// `vbslq_u16` reinterpreted — ONE instruction.
    #[inline(always)]
    pub fn blend(mask: IMask8, t: Self, f: Self) -> Self {
        unsafe {
            Self(vreinterpretq_s16_u16(vbslq_u16(
                mask.0,
                vreinterpretq_u16_s16(t.0),
                vreinterpretq_u16_s16(f.0),
            )))
        }
    }

    // ── Widening ─────────────────────────────────────────────────────────────

    /// Sign-extend low 4 lanes to i32x4.
    #[inline(always)]
    pub fn as_i32x4_lo(self) -> i32x4 {
        unsafe { i32x4(vmovl_s16(vget_low_s16(self.0))) }
    }

    /// Sign-extend high 4 lanes to i32x4.
    #[inline(always)]
    pub fn as_i32x4_hi(self) -> i32x4 {
        unsafe { i32x4(vmovl_high_s16(self.0)) }
    }

    /// Narrow two i32x4 to i16x8 with signed saturation.
    #[inline(always)]
    pub fn pack_i32x4(lo: i32x4, hi: i32x4) -> Self {
        unsafe { Self(vcombine_s16(vqmovn_s32(lo.0), vqmovn_s32(hi.0))) }
    }

    #[inline(always)] pub fn wrapping_add(self, r: Self) -> Self { self + r }
    #[inline(always)] pub fn wrapping_sub(self, r: Self) -> Self { self - r }
    #[inline(always)] pub fn wrapping_mul(self, r: Self) -> Self { self.mul_lo(r) }
}

// ── Operators ─────────────────────────────────────────────────────────────────

impl Add for i16x8 {
    type Output = Self;
    #[inline(always)] fn add(self, r: Self) -> Self { Self(unsafe { vaddq_s16(self.0, r.0) }) }
}
impl AddAssign for i16x8 { #[inline(always)] fn add_assign(&mut self, r: Self) { *self = *self + r; } }
impl Sub for i16x8 {
    type Output = Self;
    #[inline(always)] fn sub(self, r: Self) -> Self { Self(unsafe { vsubq_s16(self.0, r.0) }) }
}
impl SubAssign for i16x8 { #[inline(always)] fn sub_assign(&mut self, r: Self) { *self = *self - r; } }
impl Neg for i16x8 {
    type Output = Self;
    #[inline(always)] fn neg(self) -> Self { Self(unsafe { vnegq_s16(self.0) }) }
}
impl Mul for i16x8 {
    type Output = Self;
    #[inline(always)] fn mul(self, r: Self) -> Self { self.mul_lo(r) }
}
impl MulAssign for i16x8 { #[inline(always)] fn mul_assign(&mut self, r: Self) { *self = *self * r; } }
impl BitAnd for i16x8 {
    type Output = Self;
    #[inline(always)] fn bitand(self, r: Self) -> Self { Self(unsafe { vandq_s16(self.0, r.0) }) }
}
impl BitAndAssign for i16x8 { #[inline(always)] fn bitand_assign(&mut self, r: Self) { *self = *self & r; } }
impl BitOr for i16x8 {
    type Output = Self;
    #[inline(always)] fn bitor(self, r: Self) -> Self { Self(unsafe { vorrq_s16(self.0, r.0) }) }
}
impl BitOrAssign for i16x8 { #[inline(always)] fn bitor_assign(&mut self, r: Self) { *self = *self | r; } }
impl BitXor for i16x8 {
    type Output = Self;
    #[inline(always)] fn bitxor(self, r: Self) -> Self { Self(unsafe { veorq_s16(self.0, r.0) }) }
}
impl BitXorAssign for i16x8 { #[inline(always)] fn bitxor_assign(&mut self, r: Self) { *self = *self ^ r; } }
impl Not for i16x8 {
    type Output = Self;
    #[inline(always)] fn not(self) -> Self { Self(unsafe { vmvnq_s16(self.0) }) }
}

impl PartialEq for i16x8 {
    #[inline]
    fn eq(&self, r: &Self) -> bool {
        unsafe { vminvq_u16(vceqq_s16(self.0, r.0)) == u16::MAX }
    }
}
impl Eq for i16x8 {}

impl fmt::Debug for i16x8 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let a = self.to_array();
        write!(f, "i16x8({},{},{},{},{},{},{},{})",
            a[0],a[1],a[2],a[3],a[4],a[5],a[6],a[7])
    }
}
impl fmt::Display for i16x8 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let a = self.to_array();
        write!(f, "[{},{},{},{},{},{},{},{}]",
            a[0],a[1],a[2],a[3],a[4],a[5],a[6],a[7])
    }
}
impl From<[i16; 8]> for i16x8 { #[inline] fn from(a: [i16;8]) -> Self { Self::from_array(a) } }
impl From<i16x8> for [i16; 8] { #[inline] fn from(v: i16x8) -> Self { v.to_array() } }
