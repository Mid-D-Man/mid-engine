// crates/mid-math/src/wide/mod.rs
//! Wide SIMD types — vertical operations on N values simultaneously.
//!
//! Organised into two separate sub-folders:
//!   wide/float/  — f32x4, Vec3x4, QuatX4, Mask4  (SSE2 baseline, AVX2 gated)
//!   wide/int/    — IMask4, i32x4, u32x4, i16x8, u16x8, i8x16, u8x16
//!
//! Philosophy: every operation is branchless. Branches in wide/ are a bug.
//! Integer ALU and FPU run independently — processing integer and float wide
//! types simultaneously is free throughput on modern CPUs.
//!
//! Phase 3C build order:
//!   Batch 1 (this file): IMask4, i32x4, u32x4
//!   Batch 2: i16x8, u16x8, i8x16, u8x16
//!   Batch 3: float/Mask4, float/f32x4
//!   Batch 4: float/Vec3x4 (most complex — AoS→SoA transpose)
//!   Batch 5: float/QuatX4

pub mod float;
pub mod int;

// ── Integer wide re-exports ───────────────────────────────────────────────────

pub use int::IMask4;
#[allow(non_camel_case_types)]
pub use int::i32x4;
#[allow(non_camel_case_types)]
pub use int::u32x4;

// Float wide re-exports arrive in Batch 3–5.
// pub use float::...;
