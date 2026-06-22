// crates/mid-math/src/f32/sme/mod.rs
//! SME (Scalable Matrix Extension) fast paths for aarch64.
//!
//! ## Status: STUB — no Rust support as of 2026-06
//!
//! SME is ARM's dedicated matrix acceleration ISA, introduced with
//! the Cortex-X4 / Apple M4.
//!
//! ## Hardware
//! | SoC            | SME version | Context                  |
//! |----------------|-------------|--------------------------|
//! | Apple M4       | SME2        | MacBook Pro / Mac mini   |
//! | Apple M4 Ultra | SME2        | Mac Studio               |
//! | Cortex-X4      | SME         | High-end Android (2024+) |
//!
//! ## Gate
//! #[cfg(all(target_arch = "aarch64", target_feature = "sme"))]
//!
//! ## What SME provides
//! ZA registers: on-chip matrix tile storage (2D accumulator array).
//! Key instructions:
//!   fmopa — outer-product accumulate into ZA tile (A_col × B_row)
//!   fmops — outer-product subtract
//!   ld1w  — tile load from memory
//!   st1w  — tile store
//!   zero  — clear ZA tile
//!
//! For mid-math:
//!   - Mat4 multiply: 4×4 f32 fits in ZA[0] tile.
//!     fmopa(A_col_k, B_row_k) for k=0..3 → complete C matrix in ZA.
//!     Expected: sub-nanosecond for single mat4 mul on M4.
//!   - 8-bone blend skinning: accumulate 8 DualQuat weighted sums in ZA.
//!   - Neural inference for AI characters (very future, mid-net integration).
//!
//! ## SME2 additions (Apple M4)
//! ZT0: 512-bit lookup table register.
//! Multi-vector: operates on 2 or 4 SVE vectors simultaneously.
//! Potential for game math:
//!   - ZT0 polynomial table → faster sin_cos → faster from_axis_angle/slerp
//!   - Multi-vector FMA → 4 independent dot products in one instruction
//!
//! ## Blockers (all must resolve before implementation)
//!   1. Rust SME intrinsics stabilization (no timeline)
//!   2. OS support: macOS Sequoia+ saves/restores ZA context.
//!      Linux aarch64 SME context switching landed in kernel 6.1.
//!   3. GitHub Actions M4 runners (currently M1/M2/M3 only)
//!
//! ## Why the stub exists now
//! Architecture documentation. When blockers clear, this module
//! follows the same pattern as avx/mat4.rs: one impl gated at the
//! correct feature level, everything else falls through to neon/.
