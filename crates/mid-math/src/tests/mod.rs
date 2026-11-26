// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-math.md, section "tests/mod.rs"
// ============================================================================

mod vec_tests;
mod quat_tests;
mod mat_tests;
mod stress_tests;
mod f64_tests;
mod bool_tests;
mod int64_tests;
mod wide_tests;
mod fixed;
mod noise;
mod camera;
mod geom;
mod mid_vec;
#[cfg(feature = "mint")]
mod mint_tests;
mod color32_rounded;
