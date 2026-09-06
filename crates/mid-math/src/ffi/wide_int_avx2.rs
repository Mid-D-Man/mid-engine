// crates/mid-math/src/ffi/wide_int_avx2.rs
//! C-ABI types and #[no_mangle] exports for the AVX2-only additive
//! wide/int SIMD-lane types: i32x8/u32x8/i16x16/u16x16/i8x32/u8x32.
//!
//! This is "FFI option 1" from the AVX2 rework's design pass: direct,
//! dispatched access to these types across the C ABI. It's safe now in
//! a way it wasn't before that rework -- these types are always
//! compiled on x86/x86_64 (not gated on the crate's own `avx2`
//! target-feature baseline), and every arithmetic method checks CPU
//! support at runtime internally, falling back to a portable two-half
//! implementation when AVX2 isn't actually present. A C caller linking
//! against a non-AVX2 build/CPU gets a working (slower) result, not a
//! link error -- see wide/int/avx2/i32x8.rs's doc comment for the full
//! reasoning, and wide_int.rs's own header for why that file excludes
//! these types (this file is the design pass it deferred).
//!
//! Gated `#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]` at
//! the module level (see ffi/mod.rs) -- these types don't exist on
//! other architectures at all, same as the underlying wide types
//! themselves.
//!
//! Scope mirrors wide_int.rs exactly, applied to the wider types:
//! construction, splat, core arithmetic, min/max/clamp, the widened-
//! accumulator element_sum convention, wrapping_*, saturating_*.
//! Deliberately excludes, same as wide_int.rs: cmpXX/blend (folded
//! into the same future bvec+comparisons FFI work wide_int.rs defers
//! to), shl/shr (not exposed for the narrow tier either), and the
//! AVX2-specific extras that have no narrow-tier equivalent to mirror
//! (widen conversions, shuffle_bytes, count_eq/contains, to/from
//! half-pair) -- not decided here, no established convention to match.
//!
//! C representation: a single array field per type (v: [T; N]), same
//! reasoning as wide_int.rs -- N packed scalar lanes, not a math
//! vector.
//!
//! `new` takes N individual scalar args and builds via `from_array`
//! (not a `new()` call on the wide type itself -- i8x32/u8x32 have no
//! such constructor; matches wide_int.rs's own i8x16/u8x16 convention
//! exactly, just wider).

use crate::{i32x8, u32x8, i16x16, u16x16, i8x32, u8x32};

// =============================================================================
//  C types
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)] #[repr(C)]
pub struct CI32x8 { pub v: [i32; 8] }
impl From<i32x8>  for CI32x8 { #[inline(always)] fn from(w: i32x8)  -> Self { Self { v: w.to_array() } } }
impl From<CI32x8> for i32x8  { #[inline(always)] fn from(c: CI32x8) -> Self { i32x8::from_array(c.v) } }

#[derive(Debug, Clone, Copy, PartialEq, Eq)] #[repr(C)]
pub struct CU32x8 { pub v: [u32; 8] }
impl From<u32x8>  for CU32x8 { #[inline(always)] fn from(w: u32x8)  -> Self { Self { v: w.to_array() } } }
impl From<CU32x8> for u32x8  { #[inline(always)] fn from(c: CU32x8) -> Self { u32x8::from_array(c.v) } }

#[derive(Debug, Clone, Copy, PartialEq, Eq)] #[repr(C)]
pub struct CI16x16 { pub v: [i16; 16] }
impl From<i16x16>  for CI16x16 { #[inline(always)] fn from(w: i16x16)  -> Self { Self { v: w.to_array() } } }
impl From<CI16x16> for i16x16  { #[inline(always)] fn from(c: CI16x16) -> Self { i16x16::from_array(c.v) } }

#[derive(Debug, Clone, Copy, PartialEq, Eq)] #[repr(C)]
pub struct CU16x16 { pub v: [u16; 16] }
impl From<u16x16>  for CU16x16 { #[inline(always)] fn from(w: u16x16)  -> Self { Self { v: w.to_array() } } }
impl From<CU16x16> for u16x16  { #[inline(always)] fn from(c: CU16x16) -> Self { u16x16::from_array(c.v) } }

#[derive(Debug, Clone, Copy, PartialEq, Eq)] #[repr(C)]
pub struct CI8x32 { pub v: [i8; 32] }
impl From<i8x32>  for CI8x32 { #[inline(always)] fn from(w: i8x32)  -> Self { Self { v: w.to_array() } } }
impl From<CI8x32> for i8x32  { #[inline(always)] fn from(c: CI8x32) -> Self { i8x32::from_array(c.v) } }

#[derive(Debug, Clone, Copy, PartialEq, Eq)] #[repr(C)]
pub struct CU8x32 { pub v: [u8; 32] }
impl From<u8x32>  for CU8x32 { #[inline(always)] fn from(w: u8x32)  -> Self { Self { v: w.to_array() } } }
impl From<CU8x32> for u8x32  { #[inline(always)] fn from(c: CU8x32) -> Self { u8x32::from_array(c.v) } }

// =============================================================================
//  Exports
// =============================================================================

// ── Exports — i32x8 ───────────────────────────────────────────────────────────────
#[no_mangle] pub extern "C" fn mid_i32x8_new(v0:i32,v1:i32,v2:i32,v3:i32,v4:i32,v5:i32,v6:i32,v7:i32)->CI32x8{i32x8::from_array([v0,v1,v2,v3,v4,v5,v6,v7]).into()}
#[no_mangle] pub extern "C" fn mid_i32x8_splat(v:i32)->CI32x8{i32x8::splat(v).into()}
#[no_mangle] pub extern "C" fn mid_i32x8_add(a:CI32x8,b:CI32x8)->CI32x8{(i32x8::from(a)+i32x8::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_i32x8_sub(a:CI32x8,b:CI32x8)->CI32x8{(i32x8::from(a)-i32x8::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_i32x8_mul(a:CI32x8,b:CI32x8)->CI32x8{(i32x8::from(a)*i32x8::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_i32x8_min(a:CI32x8,b:CI32x8)->CI32x8{i32x8::from(a).min(i32x8::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_i32x8_max(a:CI32x8,b:CI32x8)->CI32x8{i32x8::from(a).max(i32x8::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_i32x8_clamp(v:CI32x8,lo:CI32x8,hi:CI32x8)->CI32x8{i32x8::from(v).clamp(i32x8::from(lo),i32x8::from(hi)).into()}
#[no_mangle] pub extern "C" fn mid_i32x8_abs(v:CI32x8)->CI32x8{i32x8::from(v).abs().into()}
#[no_mangle] pub extern "C" fn mid_i32x8_min_element(v:CI32x8)->i32{i32x8::from(v).min_element()}
#[no_mangle] pub extern "C" fn mid_i32x8_max_element(v:CI32x8)->i32{i32x8::from(v).max_element()}
#[no_mangle] pub extern "C" fn mid_i32x8_element_sum(v:CI32x8)->i32{i32x8::from(v).element_sum()}
#[no_mangle] pub extern "C" fn mid_i32x8_wrapping_add(a:CI32x8,b:CI32x8)->CI32x8{i32x8::from(a).wrapping_add(i32x8::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_i32x8_wrapping_sub(a:CI32x8,b:CI32x8)->CI32x8{i32x8::from(a).wrapping_sub(i32x8::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_i32x8_saturating_add(a:CI32x8,b:CI32x8)->CI32x8{i32x8::from(a).saturating_add(i32x8::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_i32x8_saturating_sub(a:CI32x8,b:CI32x8)->CI32x8{i32x8::from(a).saturating_sub(i32x8::from(b)).into()}

// ── Exports — u32x8 ───────────────────────────────────────────────────────────────
#[no_mangle] pub extern "C" fn mid_u32x8_new(v0:u32,v1:u32,v2:u32,v3:u32,v4:u32,v5:u32,v6:u32,v7:u32)->CU32x8{u32x8::from_array([v0,v1,v2,v3,v4,v5,v6,v7]).into()}
#[no_mangle] pub extern "C" fn mid_u32x8_splat(v:u32)->CU32x8{u32x8::splat(v).into()}
#[no_mangle] pub extern "C" fn mid_u32x8_add(a:CU32x8,b:CU32x8)->CU32x8{(u32x8::from(a)+u32x8::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_u32x8_sub(a:CU32x8,b:CU32x8)->CU32x8{(u32x8::from(a)-u32x8::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_u32x8_mul(a:CU32x8,b:CU32x8)->CU32x8{(u32x8::from(a)*u32x8::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_u32x8_min(a:CU32x8,b:CU32x8)->CU32x8{u32x8::from(a).min(u32x8::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_u32x8_max(a:CU32x8,b:CU32x8)->CU32x8{u32x8::from(a).max(u32x8::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_u32x8_clamp(v:CU32x8,lo:CU32x8,hi:CU32x8)->CU32x8{u32x8::from(v).clamp(u32x8::from(lo),u32x8::from(hi)).into()}
#[no_mangle] pub extern "C" fn mid_u32x8_min_element(v:CU32x8)->u32{u32x8::from(v).min_element()}
#[no_mangle] pub extern "C" fn mid_u32x8_max_element(v:CU32x8)->u32{u32x8::from(v).max_element()}
#[no_mangle] pub extern "C" fn mid_u32x8_element_sum(v:CU32x8)->u32{u32x8::from(v).element_sum()}
#[no_mangle] pub extern "C" fn mid_u32x8_wrapping_add(a:CU32x8,b:CU32x8)->CU32x8{u32x8::from(a).wrapping_add(u32x8::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_u32x8_wrapping_sub(a:CU32x8,b:CU32x8)->CU32x8{u32x8::from(a).wrapping_sub(u32x8::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_u32x8_saturating_add(a:CU32x8,b:CU32x8)->CU32x8{u32x8::from(a).saturating_add(u32x8::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_u32x8_saturating_sub(a:CU32x8,b:CU32x8)->CU32x8{u32x8::from(a).saturating_sub(u32x8::from(b)).into()}

// ── Exports — i16x16 ──────────────────────────────────────────────────────────────
#[no_mangle] pub extern "C" fn mid_i16x16_new(v0:i16,v1:i16,v2:i16,v3:i16,v4:i16,v5:i16,v6:i16,v7:i16,v8:i16,v9:i16,v10:i16,v11:i16,v12:i16,v13:i16,v14:i16,v15:i16)->CI16x16{i16x16::from_array([v0,v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14,v15]).into()}
#[no_mangle] pub extern "C" fn mid_i16x16_splat(v:i16)->CI16x16{i16x16::splat(v).into()}
#[no_mangle] pub extern "C" fn mid_i16x16_add(a:CI16x16,b:CI16x16)->CI16x16{(i16x16::from(a)+i16x16::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_i16x16_sub(a:CI16x16,b:CI16x16)->CI16x16{(i16x16::from(a)-i16x16::from(b)).into()}
// mul goes through the Mul impl, which itself delegates to mul_lo
// (crates/mid-math/src/wide/int/avx2/i16x16.rs) -- matching narrow-tier
// i16x8's own convention of exposing plain `*` for the low-16-bits result.
#[no_mangle] pub extern "C" fn mid_i16x16_mul(a:CI16x16,b:CI16x16)->CI16x16{(i16x16::from(a)*i16x16::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_i16x16_min(a:CI16x16,b:CI16x16)->CI16x16{i16x16::from(a).min(i16x16::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_i16x16_max(a:CI16x16,b:CI16x16)->CI16x16{i16x16::from(a).max(i16x16::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_i16x16_clamp(v:CI16x16,lo:CI16x16,hi:CI16x16)->CI16x16{i16x16::from(v).clamp(i16x16::from(lo),i16x16::from(hi)).into()}
#[no_mangle] pub extern "C" fn mid_i16x16_abs(v:CI16x16)->CI16x16{i16x16::from(v).abs().into()}
#[no_mangle] pub extern "C" fn mid_i16x16_min_element(v:CI16x16)->i16{i16x16::from(v).min_element()}
#[no_mangle] pub extern "C" fn mid_i16x16_max_element(v:CI16x16)->i16{i16x16::from(v).max_element()}
// Widened accumulator (i32) -- same reasoning as narrow-tier i16x8's
// own mid_i16x8_element_sum: summing 16 i16 lanes can overflow i16's range.
#[no_mangle] pub extern "C" fn mid_i16x16_element_sum(v:CI16x16)->i32{i16x16::from(v).element_sum()}
#[no_mangle] pub extern "C" fn mid_i16x16_wrapping_add(a:CI16x16,b:CI16x16)->CI16x16{i16x16::from(a).wrapping_add(i16x16::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_i16x16_wrapping_sub(a:CI16x16,b:CI16x16)->CI16x16{i16x16::from(a).wrapping_sub(i16x16::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_i16x16_saturating_add(a:CI16x16,b:CI16x16)->CI16x16{i16x16::from(a).saturating_add(i16x16::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_i16x16_saturating_sub(a:CI16x16,b:CI16x16)->CI16x16{i16x16::from(a).saturating_sub(i16x16::from(b)).into()}

// ── Exports — u16x16 ──────────────────────────────────────────────────────────────
#[no_mangle] pub extern "C" fn mid_u16x16_new(v0:u16,v1:u16,v2:u16,v3:u16,v4:u16,v5:u16,v6:u16,v7:u16,v8:u16,v9:u16,v10:u16,v11:u16,v12:u16,v13:u16,v14:u16,v15:u16)->CU16x16{u16x16::from_array([v0,v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14,v15]).into()}
#[no_mangle] pub extern "C" fn mid_u16x16_splat(v:u16)->CU16x16{u16x16::splat(v).into()}
#[no_mangle] pub extern "C" fn mid_u16x16_add(a:CU16x16,b:CU16x16)->CU16x16{(u16x16::from(a)+u16x16::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_u16x16_sub(a:CU16x16,b:CU16x16)->CU16x16{(u16x16::from(a)-u16x16::from(b)).into()}
// See mid_i16x16_mul's comment above -- same reasoning, mul_lo via Mul.
#[no_mangle] pub extern "C" fn mid_u16x16_mul(a:CU16x16,b:CU16x16)->CU16x16{(u16x16::from(a)*u16x16::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_u16x16_min(a:CU16x16,b:CU16x16)->CU16x16{u16x16::from(a).min(u16x16::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_u16x16_max(a:CU16x16,b:CU16x16)->CU16x16{u16x16::from(a).max(u16x16::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_u16x16_clamp(v:CU16x16,lo:CU16x16,hi:CU16x16)->CU16x16{u16x16::from(v).clamp(u16x16::from(lo),u16x16::from(hi)).into()}
#[no_mangle] pub extern "C" fn mid_u16x16_min_element(v:CU16x16)->u16{u16x16::from(v).min_element()}
#[no_mangle] pub extern "C" fn mid_u16x16_max_element(v:CU16x16)->u16{u16x16::from(v).max_element()}
// See mid_i16x16_element_sum's comment above -- same reasoning, u32.
#[no_mangle] pub extern "C" fn mid_u16x16_element_sum(v:CU16x16)->u32{u16x16::from(v).element_sum()}
#[no_mangle] pub extern "C" fn mid_u16x16_wrapping_add(a:CU16x16,b:CU16x16)->CU16x16{u16x16::from(a).wrapping_add(u16x16::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_u16x16_wrapping_sub(a:CU16x16,b:CU16x16)->CU16x16{u16x16::from(a).wrapping_sub(u16x16::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_u16x16_saturating_add(a:CU16x16,b:CU16x16)->CU16x16{u16x16::from(a).saturating_add(u16x16::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_u16x16_saturating_sub(a:CU16x16,b:CU16x16)->CU16x16{u16x16::from(a).saturating_sub(u16x16::from(b)).into()}

// ── Exports — i8x32 ───────────────────────────────────────────────────────────────
#[no_mangle] pub extern "C" fn mid_i8x32_new(v0:i8,v1:i8,v2:i8,v3:i8,v4:i8,v5:i8,v6:i8,v7:i8,v8:i8,v9:i8,v10:i8,v11:i8,v12:i8,v13:i8,v14:i8,v15:i8,v16:i8,v17:i8,v18:i8,v19:i8,v20:i8,v21:i8,v22:i8,v23:i8,v24:i8,v25:i8,v26:i8,v27:i8,v28:i8,v29:i8,v30:i8,v31:i8)->CI8x32{i8x32::from_array([v0,v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14,v15,v16,v17,v18,v19,v20,v21,v22,v23,v24,v25,v26,v27,v28,v29,v30,v31]).into()}
#[no_mangle] pub extern "C" fn mid_i8x32_splat(v:i8)->CI8x32{i8x32::splat(v).into()}
#[no_mangle] pub extern "C" fn mid_i8x32_add(a:CI8x32,b:CI8x32)->CI8x32{(i8x32::from(a)+i8x32::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_i8x32_sub(a:CI8x32,b:CI8x32)->CI8x32{(i8x32::from(a)-i8x32::from(b)).into()}
// mid_i8x32_mul intentionally not exported -- i8x32 has no `Mul` impl,
// same reasoning as narrow-tier i8x16's own omission (see wide_int.rs).
#[no_mangle] pub extern "C" fn mid_i8x32_min(a:CI8x32,b:CI8x32)->CI8x32{i8x32::from(a).min(i8x32::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_i8x32_max(a:CI8x32,b:CI8x32)->CI8x32{i8x32::from(a).max(i8x32::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_i8x32_clamp(v:CI8x32,lo:CI8x32,hi:CI8x32)->CI8x32{i8x32::from(v).clamp(i8x32::from(lo),i8x32::from(hi)).into()}
#[no_mangle] pub extern "C" fn mid_i8x32_abs(v:CI8x32)->CI8x32{i8x32::from(v).abs().into()}
#[no_mangle] pub extern "C" fn mid_i8x32_min_element(v:CI8x32)->i8{i8x32::from(v).min_element()}
#[no_mangle] pub extern "C" fn mid_i8x32_max_element(v:CI8x32)->i8{i8x32::from(v).max_element()}
// See mid_i16x16_element_sum's comment above -- same reasoning, i32.
#[no_mangle] pub extern "C" fn mid_i8x32_element_sum(v:CI8x32)->i32{i8x32::from(v).element_sum()}
#[no_mangle] pub extern "C" fn mid_i8x32_wrapping_add(a:CI8x32,b:CI8x32)->CI8x32{i8x32::from(a).wrapping_add(i8x32::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_i8x32_wrapping_sub(a:CI8x32,b:CI8x32)->CI8x32{i8x32::from(a).wrapping_sub(i8x32::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_i8x32_saturating_add(a:CI8x32,b:CI8x32)->CI8x32{i8x32::from(a).saturating_add(i8x32::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_i8x32_saturating_sub(a:CI8x32,b:CI8x32)->CI8x32{i8x32::from(a).saturating_sub(i8x32::from(b)).into()}

// ── Exports — u8x32 ───────────────────────────────────────────────────────────────
#[no_mangle] pub extern "C" fn mid_u8x32_new(v0:u8,v1:u8,v2:u8,v3:u8,v4:u8,v5:u8,v6:u8,v7:u8,v8:u8,v9:u8,v10:u8,v11:u8,v12:u8,v13:u8,v14:u8,v15:u8,v16:u8,v17:u8,v18:u8,v19:u8,v20:u8,v21:u8,v22:u8,v23:u8,v24:u8,v25:u8,v26:u8,v27:u8,v28:u8,v29:u8,v30:u8,v31:u8)->CU8x32{u8x32::from_array([v0,v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14,v15,v16,v17,v18,v19,v20,v21,v22,v23,v24,v25,v26,v27,v28,v29,v30,v31]).into()}
#[no_mangle] pub extern "C" fn mid_u8x32_splat(v:u8)->CU8x32{u8x32::splat(v).into()}
#[no_mangle] pub extern "C" fn mid_u8x32_add(a:CU8x32,b:CU8x32)->CU8x32{(u8x32::from(a)+u8x32::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_u8x32_sub(a:CU8x32,b:CU8x32)->CU8x32{(u8x32::from(a)-u8x32::from(b)).into()}
// mid_u8x32_mul intentionally not exported -- same reasoning as
// mid_i8x32_mul above.
#[no_mangle] pub extern "C" fn mid_u8x32_min(a:CU8x32,b:CU8x32)->CU8x32{u8x32::from(a).min(u8x32::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_u8x32_max(a:CU8x32,b:CU8x32)->CU8x32{u8x32::from(a).max(u8x32::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_u8x32_clamp(v:CU8x32,lo:CU8x32,hi:CU8x32)->CU8x32{u8x32::from(v).clamp(u8x32::from(lo),u8x32::from(hi)).into()}
#[no_mangle] pub extern "C" fn mid_u8x32_min_element(v:CU8x32)->u8{u8x32::from(v).min_element()}
#[no_mangle] pub extern "C" fn mid_u8x32_max_element(v:CU8x32)->u8{u8x32::from(v).max_element()}
// See mid_i16x16_element_sum's comment above -- same reasoning, u32.
#[no_mangle] pub extern "C" fn mid_u8x32_element_sum(v:CU8x32)->u32{u8x32::from(v).element_sum()}
#[no_mangle] pub extern "C" fn mid_u8x32_wrapping_add(a:CU8x32,b:CU8x32)->CU8x32{u8x32::from(a).wrapping_add(u8x32::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_u8x32_wrapping_sub(a:CU8x32,b:CU8x32)->CU8x32{u8x32::from(a).wrapping_sub(u8x32::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_u8x32_saturating_add(a:CU8x32,b:CU8x32)->CU8x32{u8x32::from(a).saturating_add(u8x32::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_u8x32_saturating_sub(a:CU8x32,b:CU8x32)->CU8x32{u8x32::from(a).saturating_sub(u8x32::from(b)).into()}
