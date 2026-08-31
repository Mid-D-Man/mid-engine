// crates/mid-math/src/ffi/wide_int.rs
//! C-ABI types and #[no_mangle] exports for the always-available
//! (SSE2/NEON/scalar/wasm-tier) wide/int SIMD-lane types.
//!
//! Deliberately excludes the AVX2-only additive types (i32x8/u32x8/
//! i16x16/u16x16/i8x32/u8x32) -- exposing a conditionally-compiled type
//! over a stable C ABI needs its own design pass, not decided here.
//!
//! C representation: a single array field per type (v: [T; N]), not
//! named x/y/z/w fields -- these are N packed scalar lanes, not a
//! math vector (confirmed directly against source: no dot/length_sq/
//! distance_sq exist on any of these types, unlike IVec2/3/4).
//!
//! cmpXX/blend intentionally not exported here -- folded into the
//! separate bvec+comparisons FFI work instead.

use crate::{i32x4, u32x4, i16x8, u16x8, i8x16, u8x16};

// =============================================================================
//  C types
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)] #[repr(C)]
pub struct CI32x4 { pub v: [i32; 4] }
impl From<i32x4>  for CI32x4 { #[inline(always)] fn from(w: i32x4)  -> Self { Self { v: w.to_array() } } }
impl From<CI32x4> for i32x4  { #[inline(always)] fn from(c: CI32x4) -> Self { i32x4::from_array(c.v) } }

#[derive(Debug, Clone, Copy, PartialEq, Eq)] #[repr(C)]
pub struct CU32x4 { pub v: [u32; 4] }
impl From<u32x4>  for CU32x4 { #[inline(always)] fn from(w: u32x4)  -> Self { Self { v: w.to_array() } } }
impl From<CU32x4> for u32x4  { #[inline(always)] fn from(c: CU32x4) -> Self { u32x4::from_array(c.v) } }

#[derive(Debug, Clone, Copy, PartialEq, Eq)] #[repr(C)]
pub struct CI16x8 { pub v: [i16; 8] }
impl From<i16x8>  for CI16x8 { #[inline(always)] fn from(w: i16x8)  -> Self { Self { v: w.to_array() } } }
impl From<CI16x8> for i16x8  { #[inline(always)] fn from(c: CI16x8) -> Self { i16x8::from_array(c.v) } }

#[derive(Debug, Clone, Copy, PartialEq, Eq)] #[repr(C)]
pub struct CU16x8 { pub v: [u16; 8] }
impl From<u16x8>  for CU16x8 { #[inline(always)] fn from(w: u16x8)  -> Self { Self { v: w.to_array() } } }
impl From<CU16x8> for u16x8  { #[inline(always)] fn from(c: CU16x8) -> Self { u16x8::from_array(c.v) } }

#[derive(Debug, Clone, Copy, PartialEq, Eq)] #[repr(C)]
pub struct CI8x16 { pub v: [i8; 16] }
impl From<i8x16>  for CI8x16 { #[inline(always)] fn from(w: i8x16)  -> Self { Self { v: w.to_array() } } }
impl From<CI8x16> for i8x16  { #[inline(always)] fn from(c: CI8x16) -> Self { i8x16::from_array(c.v) } }

#[derive(Debug, Clone, Copy, PartialEq, Eq)] #[repr(C)]
pub struct CU8x16 { pub v: [u8; 16] }
impl From<u8x16>  for CU8x16 { #[inline(always)] fn from(w: u8x16)  -> Self { Self { v: w.to_array() } } }
impl From<CU8x16> for u8x16  { #[inline(always)] fn from(c: CU8x16) -> Self { u8x16::from_array(c.v) } }

// =============================================================================
//  Exports
// =============================================================================

// ── Exports — i32x4 ───────────────────────────────────────────────────────────────
#[no_mangle] pub extern "C" fn mid_i32x4_new(v0:i32, v1:i32, v2:i32, v3:i32)->CI32x4{i32x4::from_array([v0, v1, v2, v3]).into()}
#[no_mangle] pub extern "C" fn mid_i32x4_splat(v:i32)->CI32x4{i32x4::splat(v).into()}
#[no_mangle] pub extern "C" fn mid_i32x4_add(a:CI32x4,b:CI32x4)->CI32x4{(i32x4::from(a)+i32x4::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_i32x4_sub(a:CI32x4,b:CI32x4)->CI32x4{(i32x4::from(a)-i32x4::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_i32x4_mul(a:CI32x4,b:CI32x4)->CI32x4{(i32x4::from(a)*i32x4::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_i32x4_min(a:CI32x4,b:CI32x4)->CI32x4{i32x4::from(a).min(i32x4::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_i32x4_max(a:CI32x4,b:CI32x4)->CI32x4{i32x4::from(a).max(i32x4::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_i32x4_clamp(v:CI32x4,lo:CI32x4,hi:CI32x4)->CI32x4{i32x4::from(v).clamp(i32x4::from(lo),i32x4::from(hi)).into()}
#[no_mangle] pub extern "C" fn mid_i32x4_abs(v:CI32x4)->CI32x4{i32x4::from(v).abs().into()}
#[no_mangle] pub extern "C" fn mid_i32x4_min_element(v:CI32x4)->i32{i32x4::from(v).min_element()}
#[no_mangle] pub extern "C" fn mid_i32x4_max_element(v:CI32x4)->i32{i32x4::from(v).max_element()}
#[no_mangle] pub extern "C" fn mid_i32x4_element_sum(v:CI32x4)->i32{i32x4::from(v).element_sum()}
#[no_mangle] pub extern "C" fn mid_i32x4_wrapping_add(a:CI32x4,b:CI32x4)->CI32x4{i32x4::from(a).wrapping_add(i32x4::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_i32x4_wrapping_sub(a:CI32x4,b:CI32x4)->CI32x4{i32x4::from(a).wrapping_sub(i32x4::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_i32x4_saturating_add(a:CI32x4,b:CI32x4)->CI32x4{i32x4::from(a).saturating_add(i32x4::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_i32x4_saturating_sub(a:CI32x4,b:CI32x4)->CI32x4{i32x4::from(a).saturating_sub(i32x4::from(b)).into()}

// ── Exports — u32x4 ───────────────────────────────────────────────────────────────
#[no_mangle] pub extern "C" fn mid_u32x4_new(v0:u32, v1:u32, v2:u32, v3:u32)->CU32x4{u32x4::from_array([v0, v1, v2, v3]).into()}
#[no_mangle] pub extern "C" fn mid_u32x4_splat(v:u32)->CU32x4{u32x4::splat(v).into()}
#[no_mangle] pub extern "C" fn mid_u32x4_add(a:CU32x4,b:CU32x4)->CU32x4{(u32x4::from(a)+u32x4::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_u32x4_sub(a:CU32x4,b:CU32x4)->CU32x4{(u32x4::from(a)-u32x4::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_u32x4_mul(a:CU32x4,b:CU32x4)->CU32x4{(u32x4::from(a)*u32x4::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_u32x4_min(a:CU32x4,b:CU32x4)->CU32x4{u32x4::from(a).min(u32x4::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_u32x4_max(a:CU32x4,b:CU32x4)->CU32x4{u32x4::from(a).max(u32x4::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_u32x4_clamp(v:CU32x4,lo:CU32x4,hi:CU32x4)->CU32x4{u32x4::from(v).clamp(u32x4::from(lo),u32x4::from(hi)).into()}
#[no_mangle] pub extern "C" fn mid_u32x4_min_element(v:CU32x4)->u32{u32x4::from(v).min_element()}
#[no_mangle] pub extern "C" fn mid_u32x4_max_element(v:CU32x4)->u32{u32x4::from(v).max_element()}
#[no_mangle] pub extern "C" fn mid_u32x4_element_sum(v:CU32x4)->u32{u32x4::from(v).element_sum()}
#[no_mangle] pub extern "C" fn mid_u32x4_wrapping_add(a:CU32x4,b:CU32x4)->CU32x4{u32x4::from(a).wrapping_add(u32x4::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_u32x4_wrapping_sub(a:CU32x4,b:CU32x4)->CU32x4{u32x4::from(a).wrapping_sub(u32x4::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_u32x4_saturating_add(a:CU32x4,b:CU32x4)->CU32x4{u32x4::from(a).saturating_add(u32x4::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_u32x4_saturating_sub(a:CU32x4,b:CU32x4)->CU32x4{u32x4::from(a).saturating_sub(u32x4::from(b)).into()}

// ── Exports — i16x8 ───────────────────────────────────────────────────────────────
#[no_mangle] pub extern "C" fn mid_i16x8_new(v0:i16, v1:i16, v2:i16, v3:i16, v4:i16, v5:i16, v6:i16, v7:i16)->CI16x8{i16x8::from_array([v0, v1, v2, v3, v4, v5, v6, v7]).into()}
#[no_mangle] pub extern "C" fn mid_i16x8_splat(v:i16)->CI16x8{i16x8::splat(v).into()}
#[no_mangle] pub extern "C" fn mid_i16x8_add(a:CI16x8,b:CI16x8)->CI16x8{(i16x8::from(a)+i16x8::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_i16x8_sub(a:CI16x8,b:CI16x8)->CI16x8{(i16x8::from(a)-i16x8::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_i16x8_mul(a:CI16x8,b:CI16x8)->CI16x8{(i16x8::from(a)*i16x8::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_i16x8_min(a:CI16x8,b:CI16x8)->CI16x8{i16x8::from(a).min(i16x8::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_i16x8_max(a:CI16x8,b:CI16x8)->CI16x8{i16x8::from(a).max(i16x8::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_i16x8_clamp(v:CI16x8,lo:CI16x8,hi:CI16x8)->CI16x8{i16x8::from(v).clamp(i16x8::from(lo),i16x8::from(hi)).into()}
#[no_mangle] pub extern "C" fn mid_i16x8_abs(v:CI16x8)->CI16x8{i16x8::from(v).abs().into()}
#[no_mangle] pub extern "C" fn mid_i16x8_min_element(v:CI16x8)->i16{i16x8::from(v).min_element()}
#[no_mangle] pub extern "C" fn mid_i16x8_max_element(v:CI16x8)->i16{i16x8::from(v).max_element()}
// element_sum() returns i32, not i16 — a deliberate widened accumulator
// (crates/mid-math/src/wide/int/sse2/i16x8.rs), since summing 8 i16
// lanes can overflow i16's own range. Matching that here, not
// truncating with .try_into().unwrap() (rustc's own suggested fix for
// the mismatch this used to be) — that would reintroduce exactly the
// overflow-panic risk the widened accumulator exists to avoid.
#[no_mangle] pub extern "C" fn mid_i16x8_element_sum(v:CI16x8)->i32{i16x8::from(v).element_sum()}
#[no_mangle] pub extern "C" fn mid_i16x8_wrapping_add(a:CI16x8,b:CI16x8)->CI16x8{i16x8::from(a).wrapping_add(i16x8::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_i16x8_wrapping_sub(a:CI16x8,b:CI16x8)->CI16x8{i16x8::from(a).wrapping_sub(i16x8::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_i16x8_saturating_add(a:CI16x8,b:CI16x8)->CI16x8{i16x8::from(a).saturating_add(i16x8::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_i16x8_saturating_sub(a:CI16x8,b:CI16x8)->CI16x8{i16x8::from(a).saturating_sub(i16x8::from(b)).into()}

// ── Exports — u16x8 ───────────────────────────────────────────────────────────────
#[no_mangle] pub extern "C" fn mid_u16x8_new(v0:u16, v1:u16, v2:u16, v3:u16, v4:u16, v5:u16, v6:u16, v7:u16)->CU16x8{u16x8::from_array([v0, v1, v2, v3, v4, v5, v6, v7]).into()}
#[no_mangle] pub extern "C" fn mid_u16x8_splat(v:u16)->CU16x8{u16x8::splat(v).into()}
#[no_mangle] pub extern "C" fn mid_u16x8_add(a:CU16x8,b:CU16x8)->CU16x8{(u16x8::from(a)+u16x8::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_u16x8_sub(a:CU16x8,b:CU16x8)->CU16x8{(u16x8::from(a)-u16x8::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_u16x8_mul(a:CU16x8,b:CU16x8)->CU16x8{(u16x8::from(a)*u16x8::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_u16x8_min(a:CU16x8,b:CU16x8)->CU16x8{u16x8::from(a).min(u16x8::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_u16x8_max(a:CU16x8,b:CU16x8)->CU16x8{u16x8::from(a).max(u16x8::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_u16x8_clamp(v:CU16x8,lo:CU16x8,hi:CU16x8)->CU16x8{u16x8::from(v).clamp(u16x8::from(lo),u16x8::from(hi)).into()}
#[no_mangle] pub extern "C" fn mid_u16x8_min_element(v:CU16x8)->u16{u16x8::from(v).min_element()}
#[no_mangle] pub extern "C" fn mid_u16x8_max_element(v:CU16x8)->u16{u16x8::from(v).max_element()}
// See mid_i16x8_element_sum's comment above — same reasoning, u32.
#[no_mangle] pub extern "C" fn mid_u16x8_element_sum(v:CU16x8)->u32{u16x8::from(v).element_sum()}
#[no_mangle] pub extern "C" fn mid_u16x8_wrapping_add(a:CU16x8,b:CU16x8)->CU16x8{u16x8::from(a).wrapping_add(u16x8::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_u16x8_wrapping_sub(a:CU16x8,b:CU16x8)->CU16x8{u16x8::from(a).wrapping_sub(u16x8::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_u16x8_saturating_add(a:CU16x8,b:CU16x8)->CU16x8{u16x8::from(a).saturating_add(u16x8::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_u16x8_saturating_sub(a:CU16x8,b:CU16x8)->CU16x8{u16x8::from(a).saturating_sub(u16x8::from(b)).into()}

// ── Exports — i8x16 ───────────────────────────────────────────────────────────────
#[no_mangle] pub extern "C" fn mid_i8x16_new(v0:i8, v1:i8, v2:i8, v3:i8, v4:i8, v5:i8, v6:i8, v7:i8, v8:i8, v9:i8, v10:i8, v11:i8, v12:i8, v13:i8, v14:i8, v15:i8)->CI8x16{i8x16::from_array([v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15]).into()}
#[no_mangle] pub extern "C" fn mid_i8x16_splat(v:i8)->CI8x16{i8x16::splat(v).into()}
#[no_mangle] pub extern "C" fn mid_i8x16_add(a:CI8x16,b:CI8x16)->CI8x16{(i8x16::from(a)+i8x16::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_i8x16_sub(a:CI8x16,b:CI8x16)->CI8x16{(i8x16::from(a)-i8x16::from(b)).into()}
// mid_i8x16_mul intentionally not exported: SSE2's i8x16 has no `Mul`
// impl (crates/mid-math/src/wide/int/sse2/i8x16.rs) -- unlike NEON's
// i8x16, which gets there by widening to i16 (mul_widen_lo/
// mul_widen_hi), multiplying, and narrowing back (pack_i16x8; see that
// file's own doc comment, though it reads oddly against what the code
// actually does). SSE2's i8x16 doesn't have those widen/pack helpers
// yet, so replicating that isn't a quick fix -- confirmed no other
// caller anywhere in mid-math depends on this operator existing.
// Real follow-up if 8-bit SIMD multiply is actually needed on x86, not
// invented here under time pressure.#[no_mangle] pub extern "C" fn mid_i8x16_min(a:CI8x16,b:CI8x16)->CI8x16{i8x16::from(a).min(i8x16::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_i8x16_max(a:CI8x16,b:CI8x16)->CI8x16{i8x16::from(a).max(i8x16::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_i8x16_clamp(v:CI8x16,lo:CI8x16,hi:CI8x16)->CI8x16{i8x16::from(v).clamp(i8x16::from(lo),i8x16::from(hi)).into()}
#[no_mangle] pub extern "C" fn mid_i8x16_abs(v:CI8x16)->CI8x16{i8x16::from(v).abs().into()}
#[no_mangle] pub extern "C" fn mid_i8x16_min_element(v:CI8x16)->i8{i8x16::from(v).min_element()}
#[no_mangle] pub extern "C" fn mid_i8x16_max_element(v:CI8x16)->i8{i8x16::from(v).max_element()}
// See mid_i16x8_element_sum's comment above — same reasoning, i32.
#[no_mangle] pub extern "C" fn mid_i8x16_element_sum(v:CI8x16)->i32{i8x16::from(v).element_sum()}
#[no_mangle] pub extern "C" fn mid_i8x16_wrapping_add(a:CI8x16,b:CI8x16)->CI8x16{i8x16::from(a).wrapping_add(i8x16::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_i8x16_wrapping_sub(a:CI8x16,b:CI8x16)->CI8x16{i8x16::from(a).wrapping_sub(i8x16::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_i8x16_saturating_add(a:CI8x16,b:CI8x16)->CI8x16{i8x16::from(a).saturating_add(i8x16::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_i8x16_saturating_sub(a:CI8x16,b:CI8x16)->CI8x16{i8x16::from(a).saturating_sub(i8x16::from(b)).into()}

// ── Exports — u8x16 ───────────────────────────────────────────────────────────────
#[no_mangle] pub extern "C" fn mid_u8x16_new(v0:u8, v1:u8, v2:u8, v3:u8, v4:u8, v5:u8, v6:u8, v7:u8, v8:u8, v9:u8, v10:u8, v11:u8, v12:u8, v13:u8, v14:u8, v15:u8)->CU8x16{u8x16::from_array([v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15]).into()}
#[no_mangle] pub extern "C" fn mid_u8x16_splat(v:u8)->CU8x16{u8x16::splat(v).into()}
#[no_mangle] pub extern "C" fn mid_u8x16_add(a:CU8x16,b:CU8x16)->CU8x16{(u8x16::from(a)+u8x16::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_u8x16_sub(a:CU8x16,b:CU8x16)->CU8x16{(u8x16::from(a)-u8x16::from(b)).into()}
// mid_u8x16_mul intentionally not exported — same reasoning as
// mid_i8x16_mul above.
#[no_mangle] pub extern "C" fn mid_u8x16_min(a:CU8x16,b:CU8x16)->CU8x16{u8x16::from(a).min(u8x16::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_u8x16_max(a:CU8x16,b:CU8x16)->CU8x16{u8x16::from(a).max(u8x16::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_u8x16_clamp(v:CU8x16,lo:CU8x16,hi:CU8x16)->CU8x16{u8x16::from(v).clamp(u8x16::from(lo),u8x16::from(hi)).into()}
#[no_mangle] pub extern "C" fn mid_u8x16_min_element(v:CU8x16)->u8{u8x16::from(v).min_element()}
#[no_mangle] pub extern "C" fn mid_u8x16_max_element(v:CU8x16)->u8{u8x16::from(v).max_element()}
// See mid_i16x8_element_sum's comment above — same reasoning, u32.
#[no_mangle] pub extern "C" fn mid_u8x16_element_sum(v:CU8x16)->u32{u8x16::from(v).element_sum()}
#[no_mangle] pub extern "C" fn mid_u8x16_wrapping_add(a:CU8x16,b:CU8x16)->CU8x16{u8x16::from(a).wrapping_add(u8x16::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_u8x16_wrapping_sub(a:CU8x16,b:CU8x16)->CU8x16{u8x16::from(a).wrapping_sub(u8x16::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_u8x16_saturating_add(a:CU8x16,b:CU8x16)->CU8x16{u8x16::from(a).saturating_add(u8x16::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_u8x16_saturating_sub(a:CU8x16,b:CU8x16)->CU8x16{u8x16::from(a).saturating_sub(u8x16::from(b)).into()}
