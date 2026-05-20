// crates/mid-math/src/f32/avx2/mod.rs
//! AVX2 fast-paths for x86 / x86_64 with `target_feature = "avx2"`.
//!
//! AVX2 adds 256-bit integer ops and mandatory FMA (vfmadd*) over SSE2.
//! A 256-bit ymm register holds 8× f32 or 4× f64.
//!
//! # Status
//!
//! | Type    | Op         | Status              | OPT tag |
//! |---------|------------|---------------------|---------|
//! | Mat4    | Mul (f32×8)| Stub — not yet impl | OPT-7   |
//!
//! # Integration plan for OPT-7
//!
//! When OPT-7 lands:
//!   1. Gate the SSE2 `Mul for Mat4` impl with `#[cfg(not(target_feature = "avx2"))]`.
//!   2. Implement the AVX2 `Mul for Mat4` in `avx2/mat4.rs`.
//!   3. The `sse2/mat4.rs` type definition and all other methods remain unchanged.
//!
//! This module is compiled only when `target_feature = "avx2"` is set, but it
//! exports nothing until OPT-7 is complete.
//!
//! # f64 AVX2 (future)
//!
//! `DVec4` (align 32) and `DMat4` (align 32) are pre-aligned for AVX2 ymm registers
//! (4× f64). When f64 SIMD work begins it will live in `crates/mid-math/src/f64/avx2/`
//! following the same pattern as this module.

pub(crate) mod mat4;
