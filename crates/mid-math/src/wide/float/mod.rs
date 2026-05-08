// crates/mid-math/src/wide/float/mod.rs
//! Float wide types — Phase 3C Batch 3–5.
//!
//! Planned types (not yet implemented):
//!   Mask4    — 4-lane float comparison mask  (batch 3)
//!   f32x4    — 4-lane f32 scalar             (batch 3)
//!   Vec3x4   — 4 × Vec3 SoA, 3 × __m128      (batch 4)
//!   QuatX4   — 4 × Quat SoA, 4 × __m128      (batch 5)
//!
//! AVX2 variants (feature-gated):
//!   Mask8, f32x8, Vec3x8
//!
//! This module stub exists so `wide/mod.rs` compiles.
//! Populate after integer batch is stable.
