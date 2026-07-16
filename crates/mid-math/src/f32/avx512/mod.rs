// crates/mid-math/src/f32/avx512/mod.rs
//! AVX-512 fast paths for x86 / x86_64.
//!
//! Gate: `#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), target_feature = "avx512f"))]`
//!
//! ## Hardware availability
//! GitHub Actions `ubuntu-latest` runners do **not** have AVX-512 — they're
//! AMD EPYC, confirmed by `/proc/cpuinfo` on the `native` bench build (no
//! `avx512*` flags present at all) and by the `x86-64-v4` job in
//! `Abemch-vs-all.yml`, which soft-skips itself via a hardware gate rather
//! than SIGILL. This module has therefore never executed in CI — not for
//! perf, and not for correctness (no tests live under `f32/avx512/`).
//! Treat the "~2.0 ns target" below as a design target, not a measured
//! result, until this runs on real AVX-512 hardware (Ice Lake / Zen4+,
//! self-hosted or a different CI tier). The optimization itself stays —
//! this note is just correcting what used to be claimed here.
//! Activate with: `-C target-cpu=x86-64-v4` or `-C target-cpu=native` (on
//! hardware that actually has it).
//!
//! ## Gate interaction with avx/
//! avx512f implies avx+fma on all existing silicon.
//! f32/mod.rs gates `avx/` with `not(target_feature = "avx512f")` so exactly
//! one Mul<Mat4> impl is compiled per target. MulAssign lives ungated in
//! sse2/mat4.rs and delegates to whichever Mul<Mat4> is active.
//!
//! ## Contents
//! mat4: Mat4::mul — all 4 output columns in one ZMM, ~2.0 ns target.
//!
//! ## Planned additions
//! - f32x16 wide SIMD type (16-wide SoA: 16 normalizes per instruction)
//! - Vec3x16 (extends Vec3x4/Vec3x8 family)
//! - Masked AABB frustum cull (k-register masks, no dummy padding needed)

pub mod mat4;
