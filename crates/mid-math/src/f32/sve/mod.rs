// crates/mid-math/src/f32/sve/mod.rs
//! SVE / SVE2 (Scalable Vector Extension) fast paths for aarch64.
//!
//! ## Status: STUB — buildable now, but not worth implementing yet
//!
//! Two things changed since this file was first written:
//!
//! 1. `target_feature = "sve"`/`"sve2"` are **no longer nightly-only**.
//!    Confirmed directly: bench build #65 (rustc 1.97.1, 2026-07-14,
//!    `-C target-cpu=native` on the GitHub-hosted aarch64 runner) shows
//!    both in the `rustc --print cfg` dump. Tracking issue
//!    (rust-lang/rust#111167) was presumably resolved between whenever
//!    this comment was first written and now. The `#[cfg(...)]` gate below
//!    is live and will compile this module in on that runner today.
//!
//! 2. That runner is confirmed **Neoverse N2** (Azure Cobalt 100 — GitHub's
//!    own arm64-runner docs). Neoverse N2's SVE2 implementation is
//!    **128-bit** per Arm's own Technical Reference Manual — identical
//!    width to NEON's fixed 128-bit `float32x4_t`. SVE's entire value
//!    proposition is processing more lanes per instruction; at matched
//!    width there's nothing to gain from hand-writing predicated SVE
//!    intrinsics over the NEON code that already exists. It would be real,
//!    non-trivial work (different intrinsic surface, predicate-register
//!    programming model, less mature Rust support than NEON's) for a
//!    measured zero on the only ARM hardware in this repo's CI.
//!
//! So: don't implement this against current CI hardware. Revisit if/when
//! a wider-SVE runner is available (see table below) — implementing width-
//! agnostic VLA code now, without hardware to confirm a real gain on,
//! repeats the same mistake the AVX-512 module made (see f32/avx512/mod.rs).
//!
//! ## Hardware
//! | Core                  | SVE width  | Where                      |
//! |-----------------------|------------|----------------------------|
//! | Apple M4               | 128-bit    | MacBook Pro, Mac mini      |
//! | Apple M4 Ultra          | 128-bit    | Mac Studio (+ SME2)        |
//! | ARM Neoverse N2         | 128-bit    | AWS Graviton3, **GitHub Actions arm64 runners** |
//! | ARM Neoverse V2         | 256-bit    | AWS Graviton4              |
//! | Fujitsu A64FX           | 512-bit    | HPC clusters               |
//!
//! ## Gate
//! #[cfg(all(target_arch = "aarch64", target_feature = "sve"))]
//!
//! Live as of rustc 1.97.1 on this repo's CI runner (see above) — just not
//! worth turning on given the width situation.
//!
//! ## How SVE differs from NEON (float32x4_t)
//! NEON: fixed 128-bit, 4 f32 lanes, standard intrinsics.
//! SVE:  scalable — vl=128..2048 bits, predicate registers (pg), VLA code.
//!
//!   svbool_t pg  = svptrue_b32();  // all-true predicate
//!   svfloat32_t a = svld1_f32(pg, ptr);  // load vl/32 floats
//!   svfloat32_t r = svmla_f32(acc, a, b); // FMA, all active lanes
//!
//! For mid-math, if/when a wider-than-128-bit runner is available:
//!   - Batch normalize/dot over N floats without padding loops
//!   - VLA Vec3 normalize: predicated tail for arbitrary N
//!   - Mat4 mul: svmla_f32 FMA chains, width-agnostic
//!   - Replaces Vec3x4/Vec3x8 with a single VLA Vec3 batch type
//!
//! ## SVE2 additions
//! Complex arithmetic, histogram, bitwise rotation. Minimal benefit
//! for game math beyond SVE.
//!
//! ## Future structure (when there's hardware worth targeting)
//! pub mod vec3;   // svfloat32_t Vec3 with predicated ops
//! pub mod vec4;   // svfloat32_t Vec4
//! pub mod quat;   // svfloat32_t Quat
//! pub mod mat4;   // svmla-based Mat4 multiply
