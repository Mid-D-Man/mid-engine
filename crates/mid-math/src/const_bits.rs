// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-math.md, section "const_bits.rs"
// ============================================================================
//! Bit casts usable in `const` contexts on rustc older than 1.83, where
//! `f32::from_bits`, `f32::to_bits` and `f64::to_bits` are not yet `const fn`.
//!
//! Same technique as `sse2::m128_from_f32x4`: `core::mem::transmute` has been
//! `const` since 1.56, and a bit cast between same-size plain-data types is
//! exactly what the std functions do.

// Not every backend (scalar, NEON) has a const site that uses all three.
#![allow(dead_code)]

/// `f32::from_bits`, usable in `const`.
#[inline(always)]
pub(crate) const fn f32_from_bits(bits: u32) -> f32 {
    // SAFETY: `u32` and `f32` have the same size and every bit pattern is a
    // valid `f32`.
    unsafe { core::mem::transmute::<u32, f32>(bits) }
}

/// `f32::to_bits`, usable in `const`.
#[inline(always)]
pub(crate) const fn f32_to_bits(v: f32) -> u32 {
    // SAFETY: `f32` and `u32` have the same size and every bit pattern is a
    // valid `u32`.
    unsafe { core::mem::transmute::<f32, u32>(v) }
}

/// `f64::to_bits`, usable in `const`.
#[inline(always)]
pub(crate) const fn f64_to_bits(v: f64) -> u64 {
    // SAFETY: `f64` and `u64` have the same size and every bit pattern is a
    // valid `u64`.
    unsafe { core::mem::transmute::<f64, u64>(v) }
}
