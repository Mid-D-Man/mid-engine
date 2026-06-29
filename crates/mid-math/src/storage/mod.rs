// crates/mid-math/src/storage/mod.rs
//! Low-precision storage types — the boundary layer of Mid-Engine.
//!
//! ## Purpose
//! Store data compressed. Unpack to `f32` only when arithmetic is needed.
//!
//! ## Type map
//! | Type           | Bits | Range            | Primary use                      |
//! |----------------|------|------------------|----------------------------------|
//! | `f16`          |  16  | ±65504           | GPU normals, HDR, bone transforms|
//! | `bf16`         |  16  | ±3.39e38         | ML training (same range as f32)  |
//! | `F8Norm`       |   8  | [0.0, 1.0]       | Colors, alpha, blend weights     |
//! | `F8E4M3`       |   8  | ±448.0           | ML weights / activations (FP8)   |
//! | `F8E5M2`       |   8  | ±57344.0         | ML gradients (FP8)               |
//! | `F4E2M1`       |   4  | ±6.0  (MXFP4)   | Ultra-compressed ML weights      |
//! | `F4E3M0`       |   4  | ±16.0 (powers²) | Approximate ML inference         |
//! | `F4E2M1Pair`   |   8  | 2×F4E2M1         | Packed storage unit for F4E2M1   |
//! | `F4E3M0Pair`   |   8  | 2×F4E3M0         | Packed storage unit for F4E3M0   |
//! | `BitMask*`     | 1/bool | {false, true}  | ECS masks, bone flags, visibility|
//!
//! ## The two-mask rule
//! There are TWO distinct "boolean array" types in this engine:
//!
//! **SIMD computation masks** (`Mask4`, `IMask4`, etc. in `wide/`):
//! * Each boolean = full 32-bit SIMD lane (`0xFFFF_FFFF` = true, `0` = false)
//! * Purpose: branchless blending inside wide-vector math
//! * Never stored to memory — only live in registers during computation
//!
//! **Storage masks** (`BitMask*` below):
//! * 1 bit per boolean, packed into `u8` / `u64` / `[u64; N]`
//! * Purpose: ECS component presence, animation bone flags, entity visibility
//! * Lives in RAM — designed for compact storage and fast bitwise queries
//!
//! A `BitMask64` for 64 booleans = **8 bytes**.
//! A `Mask4` for 4 booleans = **16 bytes** (wrong bit pattern for storage).
//! Never confuse them.
//!
//! ## Storage masks are format-agnostic
//! Whether your array holds `f32`, `f16`, `F8E4M3`, or `F4E2M1` values,
//! the same `BitMask64` marks which slots are active. The mask is just bits.
//!
//! ## Quantization architecture
//! These types are the **primitive layer** of a quantization pipeline:
//! ```text
//! f32  (compute)
//!  │  ◄── unpack / dequant
//!  │
//! f16 / bf16 / F8* / F4*   (storage — this module)
//!  │
//!  │  ── block scale + zero-point  ──►  mid-quant (future crate)
//! ```
//! Block quantization (N values + shared f16 scale, à la ggml Q4_0/Q8_0)
//! belongs in `mid-quant`, which imports these primitives.
//! `mid-math` stays zero-dependency.

pub mod bf16;
pub mod f16;
pub mod f4;
pub mod f8;
pub mod storage_mask;

// ── f16 ───────────────────────────────────────────────────────────────────────
#[allow(non_camel_case_types)]
pub use f16::f16;
pub use f16::{
    f32x4_to_f16x4, f16x4_to_f32x4,
    f32x8_to_f16x8, f16x8_to_f32x8,
    f32_slice_to_f16, f16_slice_to_f32,
};

// ── bf16 ──────────────────────────────────────────────────────────────────────
#[allow(non_camel_case_types)]
pub use bf16::bf16;
pub use bf16::{
    f32x4_to_bf16x4, bf16x4_to_f32x4,
    f32x8_to_bf16x8, bf16x8_to_f32x8,
    f32_slice_to_bf16, bf16_slice_to_f32,
};

// ── f8 ───────────────────────────────────────────────────────────────────────
pub use f8::{F8Norm, F8E4M3, F8E5M2};
pub use f8::{
    f32x4_to_f8e4m3x4, f8e4m3x4_to_f32x4,
    f32x4_to_f8e5m2x4, f8e5m2x4_to_f32x4,
};

// ── f4 ───────────────────────────────────────────────────────────────────────
pub use f4::{
    F4E2M1, F4E3M0,
    F4E2M1Pair, F4E3M0Pair,
    f32x8_to_f4e2m1x4pairs, f4e2m1x4pairs_to_f32x8,
    f32x8_to_f4e3m0x4pairs, f4e3m0x4pairs_to_f32x8,
    f32_slice_to_f4e2m1_packed, f4e2m1_packed_to_f32_slice,
};

// ── storage masks ─────────────────────────────────────────────────────────────
pub use storage_mask::{
    BitMask8, BitMask16, BitMask32, BitMask64,
    BitMask128, BitMask256,
    IterOnes, WideIterOnes,
};
