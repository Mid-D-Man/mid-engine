// crates/mid-math/src/wide/int/wasm/mod.rs
//! WASM SIMD128-backed integer wide types — wasm32/wasm64 with
//! `target_feature = "simd128"`.
//!
//! Same widths as sse2/neon (i32x4/u32x4/i16x8/u16x8/i8x16/u8x16 +
//! IMask4/8/16) — no AVX2-style wider addition here, WASM SIMD128 is a
//! single 128-bit v128 register, there's no wider tier to add.
//!
//! Every function name in this backend was checked directly against
//! stdarch's `core_arch/src/wasm32/simd128.rs` source (not recalled
//! from memory) before use.
//!
//! ## WASM vs SSE2 — where WASM is simpler
//!
//! | Operation              | SSE2                              | WASM SIMD128            |
//! |-------------------------|------------------------------------|--------------------------|
//! | i32 multiply            | shuffle/unpack chain (5+ instr)    | `i32x4_mul` (1)          |
//! | i32/u32 min/max          | cmp+blend / XOR-flip+blend         | native, all 4 combos (1) |
//! | u32/u16/i8 unsigned cmp  | XOR-sign-flip trick                | native `gt`/`lt`/`ge`/`le` (1) |
//! | blend/select             | and + andnot + or (3 instr)        | `v128_bitselect` (1)     |
//! | widen i16→i32            | unpacklo/unpackhi + shuffle        | `extend_low`/`extend_high` (1, takes whole register) |
//! | shuffle_bytes (16 lanes) | `_mm_shuffle_epi8`                 | `i8x16_swizzle` — same shape, no cross-lane hazard since there's only one lane |
//!
//! `add`/`sub`/`shl`/`eq`/`ne` are `pub use i*_op as u*_op` aliases in
//! stdarch itself for the signed/unsigned pairs — wrapping arithmetic
//! and bit-equality don't care about signedness. Called directly by
//! name in the u32x4/u16x8/u8x16 files here, same as stdarch intends.
//!
//! ## Known gaps (matches SSE2 for the same underlying hardware reasons)
//!
//! - No i32/u32 saturating add/sub — no such instruction on any of
//!   SSE2/AVX2/WASM SIMD128. Scalar loop, same as sse2/i32x4.rs.
//! - No i8/u8 multiply — no byte-granularity SIMD multiply anywhere.
//! - No 16x16→32-high-bits instruction (`mulhi`) — computed via the
//!   widen-to-i32x4 path in i16x8.rs's `mul_high` instead.
//! - `u8x16::element_sum` — no SAD-style horizontal-sum-of-bytes
//!   instruction on WASM SIMD128 (x86 has `psadbw`); falls back to a
//!   widen-and-fold loop.

pub mod imask4;
pub mod imask8;
pub mod imask16;
pub mod i32x4;
pub mod u32x4;
pub mod i16x8;
pub mod u16x8;
pub mod i8x16;
pub mod u8x16;

pub use imask4::IMask4;
pub use imask8::IMask8;
pub use imask16::IMask16;
