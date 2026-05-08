// crates/mid-math/src/wide/int/mod.rs
//! Integer wide types.
//!
//! Platform dispatch mirrors the existing f32/mod.rs pattern:
//!   x86 / x86_64  → sse2/ implementations
//!   everything else → scalar/ fallback
//!
//! Types exposed (all via IMask4 for comparison results):
//!   IMask4   — 4-lane integer comparison mask
//!   i32x4    — 4-lane signed 32-bit integer
//!   u32x4    — 4-lane unsigned 32-bit integer
//!
//! Coming in Batch 2:
//!   i16x8, u16x8, i8x16, u8x16

pub(crate) mod scalar;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub(crate) mod sse2;

// ── Platform dispatch ─────────────────────────────────────────────────────────

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub use sse2::{IMask4, i32x4, u32x4};

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
pub use scalar::{IMask4, i32x4, u32x4};
