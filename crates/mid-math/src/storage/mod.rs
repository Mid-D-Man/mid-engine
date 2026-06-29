// crates/mid-math/src/storage/mod.rs
//! Low-precision storage types — the "boundary layer" of Mid-Engine.
//!
//! ## Purpose
//! These types exist purely for MEMORY EFFICIENCY at the edges of the engine:
//! GPU uploads, network packets, animation clip storage, ML inference weights.
//! They are NOT for arithmetic. Convert to `f32` before computing.
//!
//! ## Type map
//! | Type         | Bits | Range               | Game engine use case               |
//! |--------------|------|---------------------|------------------------------------|
//! | `f16`        |  16  | ±65504              | GPU normals, HDR, bone transforms  |
//! | `F8Norm`     |   8  | [0.0, 1.0]          | Colors, blend weights, alpha       |
//! | `F8E4M3`     |   8  | ±448.0              | ML weights / activations (FP8)     |
//! | `F8E5M2`     |   8  | ±57344.0            | ML gradients (FP8)                 |
//! | `BitMask*`   | 1/bool | {false, true}     | ECS component masks, bone flags    |
//!
//! ## Two-mask rule
//! There are TWO conceptually distinct "boolean arrays" in this engine:
//!
//! * **SIMD computation masks** (`Mask4`, `IMask4`, etc. in `wide/`) — each
//!   boolean inflated to a full 32-bit lane (`0xFFFF_FFFF` = true). Used for
//!   branchless blending inside wide-vector math. NOT stored, only computed.
//!
//! * **Storage masks** (`BitMask*` below) — 1 bit per boolean, packed into
//!   `u8`/`u16`/`u32`/`u64`/`[u64;2]`/`[u64;4]`. Used for ECS component
//!   presence, animation bone active flags, per-entity visibility, etc.
//!
//! These serve completely different purposes. Never confuse them.
//! A `BitMask64` for 64 booleans costs 8 bytes.
//! A `Mask4` for 4 booleans costs 16 bytes and has the WRONG bit pattern for storage.
//!
//! ## f4 — PLANNED, not yet implemented
//! 4-bit float formats (`F4E2M1`, `F4E3M0`, signed nibble) are planned for
//! embedded ML inference. When ready:
//!   1. Add `pub mod f4;`
//!   2. Add `pub use f4::{F4E2M1, F4E3M0};`
//!   3. Add batch helpers `f32x8_to_f4e2m1x8` etc.
//! The `BitMask*` family works with f4 arrays unchanged.
//!
//! ## Quantization note
//! These types are the storage layer of a quantization pipeline.
//! The "block quantization" concept (e.g., 32 weights sharing one scale,
//! as used by ggml/llama.cpp) belongs in a separate `mid-quant` crate that
//! imports these primitives. mid-math stays dependency-free.

pub mod f16;
pub mod f8;
pub mod storage_mask;

// ── Re-exports ────────────────────────────────────────────────────────────────

#[allow(non_camel_case_types)]
pub use f16::f16;

pub use f16::{
    f32x4_to_f16x4, f16x4_to_f32x4,
    f32x8_to_f16x8, f16x8_to_f32x8,
    f32_slice_to_f16, f16_slice_to_f32,
};

pub use f8::{
    F8Norm, F8E4M3, F8E5M2,
    f32x4_to_f8e4m3x4, f8e4m3x4_to_f32x4,
    f32x4_to_f8e5m2x4, f8e5m2x4_to_f32x4,
};

pub use storage_mask::{
    BitMask8, BitMask16, BitMask32, BitMask64,
    BitMask128, BitMask256,
    IterOnes, WideIterOnes,
};
