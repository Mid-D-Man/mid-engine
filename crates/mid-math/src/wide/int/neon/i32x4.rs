// crates/mid-math/src/wide/int/neon/i32x4.rs
//! 4-lane signed 32-bit integer vector — NEON, aarch64.
//!
//! Backed by `int32x4_t`. Mask: `IMask4` (uint32x4_t).
//!
//! NEON advantages over SSE2 for i32:
//!   vabsq_s32       — direct abs (SSE2: cmplt + blend, 3 instr)
//!   vminq/vmaxq_s32 — native signed min/max (SSE2 needs blend workaround)
//!   vaddvq_s32      — horizontal sum in 1 ADDV instruction
//!   vminvq/vmaxvq   — horizontal min/max in 1 SMINV/SMAXV instruction
//!   vqaddq_s32      — saturating add (NO SSE2 equivalent for i32)
//!   vqsubq_s32      — saturating sub (NO SSE2 equivalent for i32)
//!   vnegq_s32       — direct negate (SSE2: sub from zero)
//!   vmvnq_s32       — direct bitwise NOT (SSE2: XOR-with-ones trick)
//!   vshlq_s32       — unified shift: positive=left, negative=arith-right

#![allow(non_camel_case_types)]

use core::arch::aarch64::*;
use core::fmt;
use core::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign,
    BitXor, BitXorAssign, Mul, MulAssign, Neg, Not, Sub, SubAssign,
};
use super::imask4::IMask4;

#[repr(C)]
union UnionCast { i: [i32; 4], v: i32x4 }

/// 4-lane signed 32-bit integer vector. 16 bytes, 16-byte aligned.
/// Backed by `int32x4_t` on aarch64.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct i32x4(pub(crate) int32x4_t);

impl i32x4 {
    pub const ZERO: Self = unsafe { UnionCast { i: [0;       4] }.v };
    pub const ONE:  Self = unsafe { UnionCast { i: [1;       4] }.v };
    pub const MIN:  Self = unsafe { UnionCast { i: [i32::MIN; 4] }.v };
    pub const MAX:  Self = unsafe { UnionCast { i: [i32::MAX; 4] }.v };

    // ── Constructors ──────────────────────────────────────────────────────────

    #[inline(always)] pub fn splat(v: i32) -> Self { Self(unsafe { vdupq_n_s32(v) }) }
    #[inline(always)] pub fn new(a: i32, b: i32, c: i32, d: i32) -> Self {
        unsafe { UnionCast { i: [a, b, c, d] }.v }
    }
    #[inline(always)] pub fn from_array(a: [i32; 4]) -> Self {
        Self(unsafe { vld1q_s32(a.as_ptr()) })
    }
    #[inline(always)] pub fn to_array(self) -> [i32; 4] {
        let mut a = [0i32; 4];
        unsafe { vst1q_s32(a.as_mut_ptr(), self.0) };
        a
    }
    #[inline] pub fn get(self, i: usize) -> i32 {
        assert!(i < 4, "i32x4::get — lane {i} out of bounds");
        unsafe { UnionCast { v: self }.i[i] }
    }

    // ── Arithmetic ────────────────────────────────────────────────────────────

    /// Direct `vabsq_s32` — no compare+blend trick needed unlike SSE2.
    #[inline(always)] pub fn abs(self) -> Self { Self(unsafe { vabsq_s32(self.0) }) }
    /// `vminq_s32` — native signed min, no workaround like SSE2.
    #[inline(always)] pub fn min(self, r: Self) -> Self { Self(unsafe { vminq_s32(self.0, r.0) }) }
    /// `vmaxq_s32` — native signed max.
    #[inline(always)] pub fn max(self, r: Self) -> Self { Self(unsafe { vmaxq_s32(self.0, r.0) }) }
    #[inline(always)] pub fn clamp(self, lo: Self, hi: Self) -> Self { self.max(lo).min(hi) }
    /// `vminvq_s32` — single SMINV instruction.
    #[inline] pub fn min_element(self) -> i32 { unsafe { vminvq_s32(self.0) } }
    /// `vmaxvq_s32` — single SMAXV instruction.
    #[inline] pub fn max_element(self) -> i32 { unsafe { vmaxvq_s32(self.0) } }
    /// `vaddvq_s32` — single ADDV instruction.
    #[inline] pub fn element_sum(self) -> i32 { unsafe { vaddvq_s32(self.0) } }

    // ── Shifts ────────────────────────────────────────────────────────────────
    //
    // vshlq_s32(a, b): positive b = left shift, negative b = arithmetic right shift.
    // One instruction family covers both directions.

    /// Logical left shift all lanes by `count` bits.
    #[inline(always)]
    pub fn shl(self, count: u32) -> Self {
        unsafe { Self(vshlq_s32(self.0, vdupq_n_s32(count as i32))) }
    }

    /// Arithmetic right shift (sign-extend) all lanes by `count` bits.
    #[inline(always)]
    pub fn shr_arithmetic(self, count: u32) -> Self {
        unsafe { Self(vshlq_s32(self.0, vdupq_n_s32(-(count as i32)))) }
    }

    /// Logical right shift (zero-fill) all lanes by `count` bits.
    /// Reinterprets as u32 for unsigned shift, then back.
    #[inline(always)]
    pub fn shr_logical(self, count: u32) -> Self {
        unsafe {
            let u = vreinterpretq_u32_s32(self.0);
            Self(vreinterpretq_s32_u32(vshlq_u32(u, vdupq_n_s32(-(count as i32)))))
        }
    }

    // ── Comparisons ───────────────────────────────────────────────────────────
    // NEON comparisons return uint32x4_t — IMask4's native backing type.
    // No float-as-int reinterpretation needed unlike SSE2.

    #[inline(always)] pub fn cmpeq(self, r: Self) -> IMask4 { IMask4(unsafe { vceqq_s32(self.0, r.0) }) }
    #[inline(always)] pub fn cmpne(self, r: Self) -> IMask4 { !self.cmpeq(r) }
    #[inline(always)] pub fn cmpgt(self, r: Self) -> IMask4 { IMask4(unsafe { vcgtq_s32(self.0, r.0) }) }
    #[inline(always)] pub fn cmplt(self, r: Self) -> IMask4 { IMask4(unsafe { vcltq_s32(self.0, r.0) }) }
    #[inline(always)] pub fn cmpge(self, r: Self) -> IMask4 { IMask4(unsafe { vcgeq_s32(self.0, r.0) }) }
    #[inline(always)] pub fn cmple(self, r: Self) -> IMask4 { IMask4(unsafe { vcleq_s32(self.0, r.0) }) }

    // ── Branchless select ─────────────────────────────────────────────────────

    /// Per-lane blend via `vbslq_u32` — ONE instruction vs SSE2's AND+ANDNOT+OR.
    /// Reinterprets signed to unsigned for the BSL instruction.
    #[inline(always)]
    pub fn blend(mask: IMask4, t: Self, f: Self) -> Self {
        unsafe {
            Self(vreinterpretq_s32_u32(vbslq_u32(
                mask.0,
                vreinterpretq_u32_s32(t.0),
                vreinterpretq_u32_s32(f.0),
            )))
        }
    }

    // ── Saturating ────────────────────────────────────────────────────────────
    // SSE2 has NO native i32 saturating add/sub — required scalar fallback there.
    // NEON provides these natively via vqaddq/vqsubq.

    /// Saturating signed add — clamps to `[i32::MIN, i32::MAX]`. Native NEON.
    #[inline(always)] pub fn saturating_add(self, r: Self) -> Self { Self(unsafe { vqaddq_s32(self.0, r.0) }) }
    /// Saturating signed sub — clamps to `[i32::MIN, i32::MAX]`. Native NEON.
    #[inline(always)] pub fn saturating_sub(self, r: Self) -> Self { Self(unsafe { vqsubq_s32(self.0, r.0) }) }

    #[inline(always)] pub fn wrapping_add(self, r: Self) -> Self { self + r }
    #[inline(always)] pub fn wrapping_sub(self, r: Self) -> Self { self - r }
    #[inline(always)] pub fn wrapping_mul(self, r: Self) -> Self { self * r }
}

// ── Operators ─────────────────────────────────────────────────────────────────

impl Add for i32x4 {
    type Output = Self;
    #[inline(always)] fn add(self, r: Self) -> Self { Self(unsafe { vaddq_s32(self.0, r.0) }) }
}
impl AddAssign for i32x4 { #[inline(always)] fn add_assign(&mut self, r: Self) { *self = *self + r; } }
impl Sub for i32x4 {
    type Output = Self;
    #[inline(always)] fn sub(self, r: Self) -> Self { Self(unsafe { vsubq_s32(self.0, r.0) }) }
}
impl SubAssign for i32x4 { #[inline(always)] fn sub_assign(&mut self, r: Self) { *self = *self - r; } }
/// Direct `vnegq_s32` — no sub-from-zero trick needed unlike SSE2.
impl Neg for i32x4 {
    type Output = Self;
    #[inline(always)] fn neg(self) -> Self { Self(unsafe { vnegq_s32(self.0) }) }
}
/// `vmulq_s32` — low 32 bits of 32×32 multiply. Available on all AArch64.
impl Mul for i32x4 {
    type Output = Self;
    #[inline(always)] fn mul(self, r: Self) -> Self { Self(unsafe { vmulq_s32(self.0, r.0) }) }
}
impl MulAssign for i32x4 { #[inline(always)] fn mul_assign(&mut self, r: Self) { *self = *self * r; } }
impl BitAnd for i32x4 {
    type Output = Self;
    #[inline(always)] fn bitand(self, r: Self) -> Self { Self(unsafe { vandq_s32(self.0, r.0) }) }
}
impl BitAndAssign for i32x4 { #[inline(always)] fn bitand_assign(&mut self, r: Self) { *self = *self & r; } }
impl BitOr for i32x4 {
    type Output = Self;
    #[inline(always)] fn bitor(self, r: Self) -> Self { Self(unsafe { vorrq_s32(self.0, r.0) }) }
}
impl BitOrAssign for i32x4 { #[inline(always)] fn bitor_assign(&mut self, r: Self) { *self = *self | r; } }
impl BitXor for i32x4 {
    type Output = Self;
    #[inline(always)] fn bitxor(self, r: Self) -> Self { Self(unsafe { veorq_s32(self.0, r.0) }) }
}
impl BitXorAssign for i32x4 { #[inline(always)] fn bitxor_assign(&mut self, r: Self) { *self = *self ^ r; } }
/// `vmvnq_s32` — direct bitwise NOT. No XOR-with-ones trick needed.
impl Not for i32x4 {
    type Output = Self;
    #[inline(always)] fn not(self) -> Self { Self(unsafe { vmvnq_s32(self.0) }) }
}

impl PartialEq for i32x4 {
    #[inline]
    fn eq(&self, r: &Self) -> bool {
        unsafe { vminvq_u32(vceqq_s32(self.0, r.0)) == u32::MAX }
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
impl From<[i32; 4]> for i32x4 { #[inline] fn from(a: [i32;4]) -> Self { Self::from_array(a) } }
impl From<i32x4> for [i32; 4] { #[inline] fn from(v: i32x4) -> Self { v.to_array() } }
