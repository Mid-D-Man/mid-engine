// crates/mid-math/src/f32/avx512/mod.rs
//! AVX-512 fast paths for x86 / x86_64.
//!
//! Gate: `#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), target_feature = "avx512f"))]`
//!
//! ## Hardware availability
//! GitHub Actions ubuntu-latest runners (2026) have full AVX-512:
//!   avx512f avx512dq avx512bw avx512vl avx512vnni avx512ifma
//!   avx512vbmi avx512bitalg avx512vbmi2 avx512vpopcntdq
//! Activate with: `-C target-cpu=x86-64-v4` or `-C target-cpu=native`
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
