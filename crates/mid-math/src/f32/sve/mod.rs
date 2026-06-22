// crates/mid-math/src/f32/sve/mod.rs
//! SVE / SVE2 (Scalable Vector Extension) fast paths for aarch64.
//!
//! ## Status: STUB — nightly Rust only as of 2026-06
//!
//! Tracking issue: https://github.com/rust-lang/rust/issues/111167
//!
//! ## Hardware
//! | Core                  | SVE width  | Where                      |
//! |-----------------------|------------|----------------------------|
//! | Apple M4              | 128-bit    | MacBook Pro, Mac mini      |
//! | Apple M4 Ultra        | 128-bit    | Mac Studio (+ SME2)        |
//! | ARM Neoverse N2       | 128-256b   | AWS Graviton3, Azure D-v5  |
//! | ARM Neoverse V2       | 256-bit    | AWS Graviton4              |
//! | Fujitsu A64FX         | 512-bit    | HPC clusters               |
//!
//! ## Gate
//! #[cfg(all(target_arch = "aarch64", target_feature = "sve"))]
//!
//! On stable Rust, `target_feature = "sve"` is never set → this module
//! is never compiled on stable. Safe to leave wired into f32/mod.rs.
//!
//! To test on nightly (requires Graviton3 or Apple Silicon):
//!   cargo +nightly build --target aarch64-unknown-linux-gnu \
//!     -Z unstable-options \
//!     -C target-feature=+sve
//!
//! ## How SVE differs from NEON (float32x4_t)
//! NEON: fixed 128-bit, 4 f32 lanes, standard intrinsics.
//! SVE:  scalable — vl=128..2048 bits, predicate registers (pg), VLA code.
//!
//!   svbool_t pg  = svptrue_b32();  // all-true predicate
//!   svfloat32_t a = svld1_f32(pg, ptr);  // load vl/32 floats
//!   svfloat32_t r = svmla_f32(acc, a, b); // FMA, all active lanes
//!
//! For mid-math:
//!   - Batch normalize/dot over N floats without padding loops
//!   - VLA Vec3 normalize: predicated tail for arbitrary N
//!   - Mat4 mul: svmla_f32 FMA chains, width-agnostic
//!   - Replaces Vec3x4/Vec3x8 with a single VLA Vec3 batch type
//!
//! ## SVE2 additions
//! Complex arithmetic, histogram, bitwise rotation. Minimal benefit
//! for game math beyond SVE.
//!
//! ## GitHub CI
//! No runner has SVE yet (2026-06). Watch for:
//!   - AWS Graviton4 runners (Neoverse V2, 256-bit SVE)
//!   - Apple M4 mac runners (128-bit SVE)
//!
//! ## Future structure (when Rust SVE stabilizes)
//! pub mod vec3;   // svfloat32_t Vec3 with predicated ops
//! pub mod vec4;   // svfloat32_t Vec4
//! pub mod quat;   // svfloat32_t Quat
//! pub mod mat4;   // svmla-based Mat4 multiply
