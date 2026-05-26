// crates/mid-math/src/wide/int/neon/u32x4.rs
//! 4-lane unsigned 32-bit integer vector — NEON, aarch64.
//!
//! Backed by `uint32x4_t`. Mask: `IMask4` (uint32x4_t).
//!
//! NEON advantages over SSE2 for u32:
//!   vminq/vmaxq_u32  — native unsigned min/max (SSE2 needs XOR-flip trick)
//!   vcgtq/vcltq_u32  — native unsigned comparisons (SSE2: XOR bias)
//!   vqaddq_u32       — saturating add (SSE2: overflow detection dance)
//!   vqsubq_u32       — saturating sub (SSE2: underflow detection dance)
//!   vaddvq_u32       — horizontal sum in 1 ADDV instruction
//!   vminvq/vmaxvq    — horizontal min/max in 1 UMINV/UMAXV instruction

#![allow(non_camel_case_types)]

use core::arch::aarch64::*;
use core::fmt;
use core::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign,
    BitXor, BitXorAssign, Mul, MulAssign, Not, Sub, SubAssign,
};
use super::imask4::IMask4;

#[repr(C)]
union UnionCast { u: [u32; 4], v: u32x4 }

/// 4-lane unsigned 32-bit integer vector. 16 bytes, 16-byte aligned.
/// Backed by `uint32x4_t` on aarch64.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct u32x4(pub(crate) uint32x4_t);

impl u32x4 {
    pub const ZERO: Self = unsafe { UnionCast { u: [0;       4] }.v };
    pub const ONE:  Self = unsafe { UnionCast { u: [1;       4] }.v };
    pub const MIN:  Self = unsafe { UnionCast { u: [u32::MIN; 4] }.v };
    pub const MAX:  Self = unsafe { UnionCast { u: [u32::MAX; 4] }.v };

    // ── Constructors ──────────────────────────────────────────────────────────

    #[inline(always)] pub fn splat(v: u32) -> Self { Self(unsafe { vdupq_n_u32(v) }) }
    #[inline(always)] pub fn new(a: u32, b: u32, c: u32, d: u32) -> Self {
        unsafe { UnionCast { u: [a, b, c, d] }.v }
    }
    #[inline(always)] pub fn from_array(a: [u32; 4]) -> Self {
        Self(unsafe { vld1q_u32(a.as_ptr()) })
    }
    #[inline(always)] pub fn to_array(self) -> [u32; 4] {
        let mut a = [0u32; 4];
        unsafe { vst1q_u32(a.as_mut_ptr(), self.0) };
        a
    }
    #[inline] pub fn get(self, i: usize) -> u32 {
        assert!(i < 4, "u32x4::get — lane {i} out of bounds");
        unsafe { UnionCast { v: self }.u[i] }
    }

    // ── Arithmetic ────────────────────────────────────────────────────────────

    /// `vminq_u32` — native unsigned min. SSE2 required XOR-bias workaround.
    #[inline(always)] pub fn min(self, r: Self) -> Self { Self(unsafe { vminq_u32(self.0, r.0) }) }
    /// `vmaxq_u32` — native unsigned max.
    #[inline(always)] pub fn max(self, r: Self) -> Self { Self(unsafe { vmaxq_u32(self.0, r.0) }) }
    #[inline(always)] pub fn clamp(self, lo: Self, hi: Self) -> Self { self.max(lo).min(hi) }
    /// `vminvq_u32` — single UMINV instruction.
    #[inline] pub fn min_element(self) -> u32 { unsafe { vminvq_u32(self.0) } }
    /// `vmaxvq_u32` — single UMAXV instruction.
    #[inline] pub fn max_element(self) -> u32 { unsafe { vmaxvq_u32(self.0) } }
    /// `vaddvq_u32` — single ADDV instruction.
    #[inline] pub fn element_sum(self) -> u32 { unsafe { vaddvq_u32(self.0) } }

    // ── Shifts ────────────────────────────────────────────────────────────────
    //
    // vshlq_u32(a, b): positive b = left shift, negative b = logical right shift.

    /// Logical left shift all lanes by `count` bits.
    #[inline(always)]
    pub fn shl(self, count: u32) -> Self {
        unsafe { Self(vshlq_u32(self.0, vdupq_n_s32(count as i32))) }
    }

    /// Logical right shift (zero-fill) all lanes by `count` bits.
    #[inline(always)]
    pub fn shr(self, count: u32) -> Self {
        unsafe { Self(vshlq_u32(self.0, vdupq_n_s32(-(count as i32)))) }
    }

    // ── Comparisons ───────────────────────────────────────────────────────────
    // NEON unsigned comparisons are native — no XOR sign-bit flip like SSE2.

    #[inline(always)] pub fn cmpeq(self, r: Self) -> IMask4 { IMask4(unsafe { vceqq_u32(self.0, r.0) }) }
    #[inline(always)] pub fn cmpne(self, r: Self) -> IMask4 { !self.cmpeq(r) }
    /// Native unsigned greater-than. SSE2 needed XOR-bias sign-flip trick.
    #[inline(always)] pub fn cmpgt(self, r: Self) -> IMask4 { IMask4(unsafe { vcgtq_u32(self.0, r.0) }) }
    #[inline(always)] pub fn cmplt(self, r: Self) -> IMask4 { IMask4(unsafe { vcltq_u32(self.0, r.0) }) }
    #[inline(always)] pub fn cmpge(self, r: Self) -> IMask4 { IMask4(unsafe { vcgeq_u32(self.0, r.0) }) }
    #[inline(always)] pub fn cmple(self, r: Self) -> IMask4 { IMask4(unsafe { vcleq_u32(self.0, r.0) }) }

    // ── Branchless select ─────────────────────────────────────────────────────

    /// `vbslq_u32` — ONE instruction. No reinterpret needed (already u32).
    #[inline(always)]
    pub fn blend(mask: IMask4, t: Self, f: Self) -> Self {
        unsafe { Self(vbslq_u32(mask.0, t.0, f.0)) }
    }

    // ── Saturating ────────────────────────────────────────────────────────────
    // SSE2 saturating u32 required complex overflow detection. NEON is native.

    /// Saturating unsigned add — clamps to `u32::MAX`. Native `vqaddq_u32`.
    #[inline(always)] pub fn saturating_add(self, r: Self) -> Self { Self(unsafe { vqaddq_u32(self.0, r.0) }) }
    /// Saturating unsigned sub — clamps to `0`. Native `vqsubq_u32`.
    #[inline(always)] pub fn saturating_sub(self, r: Self) -> Self { Self(unsafe { vqsubq_u32(self.0, r.0) }) }

    #[inline(always)] pub fn wrapping_add(self, r: Self) -> Self { self + r }
    #[inline(always)] pub fn wrapping_sub(self, r: Self) -> Self { self - r }
    #[inline(always)] pub fn wrapping_mul(self, r: Self) -> Self { self * r }
}

// ── Operators ─────────────────────────────────────────────────────────────────

impl Add for u32x4 {
    type Output = Self;
    #[inline(always)] fn add(self, r: Self) -> Self { Self(unsafe { vaddq_u32(self.0, r.0) }) }
}
impl AddAssign for u32x4 { #[inline(always)] fn add_assign(&mut self, r: Self) { *self = *self + r; } }
impl Sub for u32x4 {
    type Output = Self;
    #[inline(always)] fn sub(self, r: Self) -> Self { Self(unsafe { vsubq_u32(self.0, r.0) }) }
}
impl SubAssign for u32x4 { #[inline(always)] fn sub_assign(&mut self, r: Self) { *self = *self - r; } }
/// `vmulq_u32` — low 32 bits of 32×32 multiply.
impl Mul for u32x4 {
    type Output = Self;
    #[inline(always)] fn mul(self, r: Self) -> Self { Self(unsafe { vmulq_u32(self.0, r.0) }) }
}
impl MulAssign for u32x4 { #[inline(always)] fn mul_assign(&mut self, r: Self) { *self = *self * r; } }
impl BitAnd for u32x4 {
    type Output = Self;
    #[inline(always)] fn bitand(self, r: Self) -> Self { Self(unsafe { vandq_u32(self.0, r.0) }) }
}
impl BitAndAssign for u32x4 { #[inline(always)] fn bitand_assign(&mut self, r: Self) { *self = *self & r; } }
impl BitOr for u32x4 {
    type Output = Self;
    #[inline(always)] fn bitor(self, r: Self) -> Self { Self(unsafe { vorrq_u32(self.0, r.0) }) }
}
impl BitOrAssign for u32x4 { #[inline(always)] fn bitor_assign(&mut self, r: Self) { *self = *self | r; } }
impl BitXor for u32x4 {
    type Output = Self;
    #[inline(always)] fn bitxor(self, r: Self) -> Self { Self(unsafe { veorq_u32(self.0, r.0) }) }
}
impl BitXorAssign for u32x4 { #[inline(always)] fn bitxor_assign(&mut self, r: Self) { *self = *self ^ r; } }
impl Not for u32x4 {
    type Output = Self;
    #[inline(always)] fn not(self) -> Self { Self(unsafe { vmvnq_u32(self.0) }) }
}

impl PartialEq for u32x4 {
    #[inline]
    fn eq(&self, r: &Self) -> bool {
        unsafe { vminvq_u32(vceqq_u32(self.0, r.0)) == u32::MAX }
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
impl From<[u32; 4]> for u32x4 { #[inline] fn from(a: [u32;4]) -> Self { Self::from_array(a) } }
impl From<u32x4> for [u32; 4] { #[inline] fn from(v: u32x4) -> Self { v.to_array() } }
