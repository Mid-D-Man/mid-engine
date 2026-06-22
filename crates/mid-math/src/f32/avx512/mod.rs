// crates/mid-math/src/f32/avx512/mod.rs
//! AVX-512 fast paths for x86 / x86_64.
//!
//! ## Hardware
//! GitHub Actions ubuntu-latest runners (2026) confirm:
//!   avx512f avx512dq avx512bw avx512vl avx512vnni
//!   avx512ifma avx512vbmi avx512bitalg avx512vbmi2 avx512vpopcntdq
//!
//! ## Activate
//!   RUSTFLAGS="-C target-cpu=native"  (CI already has avx512f)
//!   RUSTFLAGS="-C target-feature=+avx512f,+avx512dq,+avx512bw,+avx512vl"
//!
//! Add `x86-64-v4` to the bench-vs-all workflow target_cpu choices once
//! avx512/mat4.rs is implemented (x86-64-v4 = AVX-512F+BW+CD+DQ+VL).
//!
//! ## Gate interaction with avx/
//! avx/mat4.rs Mul<Mat4> guard is currently:
//!   cfg(all(target_feature="avx", target_feature="fma"))
//! avx512 implies avx+fma on Intel, so BOTH modules compile when avx512f
//! is set. This is safe only while avx512/mat4.rs is a stub (no Mul<Mat4>).
//! Once avx512/mat4.rs implements Mul<Mat4>, update avx/mat4.rs to:
//!   cfg(all(avx, fma, not(target_feature="avx512f")))
//!
//! ## Planned ops (priority order)
//!
//! ### 1. Mat4::mul via _mm512_fmadd_ps — target ~2.0-2.5 ns
//! Current AVX2: ~4.0 ns (256-bit, 2 output columns per YMM).
//! AVX-512: 4 output columns per ZMM, two mat4 muls per cycle burst.
//! See mat4.rs for full algorithm sketch.
//!
//! ### 2. Masked AABB frustum cull (avx512f + avx512dq + avx512vl)
//! k-register masks: no dummy padding, 16 AABBs per instruction.
//!
//! ### 3. f32x16 wide SIMD type
//! Extends f32x4 (SSE2) → f32x8 (AVX2) → f32x16 (AVX-512) type family.
//!
//! ### 4. Vec3x16 (SoA, 16 Vec3s per ZMM triple)
//! Extends Vec3x4 (SSE2) and Vec3x8 (AVX2).

pub mod mat4;
