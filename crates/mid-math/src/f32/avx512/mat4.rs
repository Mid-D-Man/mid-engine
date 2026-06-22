// crates/mid-math/src/f32/avx512/mat4.rs
//! AVX-512 Mat4 multiply — STUB (not yet implemented).
//!
//! ## Planned algorithm: 4 output columns per ZMM register
//!
//! For C = A × B (column-major, A has cols a0..a3):
//!
//! AVX+FMA (current avx/mat4.rs): packs 2 LHS cols into one YMM, processes
//! 2 output columns simultaneously → 26 instructions, ~4.0 ns.
//!
//! AVX-512 plan: pack all 4 LHS columns into one ZMM [a0|a1|a2|a3].
//! Process all 4 output columns simultaneously in a single ZMM.
//!
//! Sketch:
//!   // Pack 4 LHS __m128 columns into one ZMM
//!   lhs_zmm = _mm512_insertf32x4(
//!       _mm512_insertf32x4(
//!           _mm512_insertf32x4(
//!               _mm512_castps128_ps512(x_axis.0), y_axis.0, 1),
//!           z_axis.0, 2),
//!       w_axis.0, 3);
//!
//!   // For each pair of RHS columns (j, j+1):
//!   //   broadcast B[j][k] to the appropriate ZMM lane groups
//!   //   fmadd chain: acc += a_k * B_col_j[k]
//!   // _mm512_fmadd_ps throughput: 0.5 CPI (Skylake-X, Ice Lake, Zen4)
//!
//!   // Extract 4 output cols via _mm512_extractf32x4_ps
//!
//! ## Expected instruction count
//! AVX2 (avx/mat4.rs): 26 256-bit instructions
//! AVX-512 target:     ~14-16 512-bit instructions
//! → ~3× throughput gain over SSE2, ~1.6× over AVX2
//!
//! ## Bench target
//! SSE2:        ~7.0 ns
//! AVX2:        ~4.0 ns
//! AVX-512:     ~2.0-2.5 ns

// TODO: implement. No items here yet — avx/mat4.rs handles Mul<Mat4>
// for all avx+fma targets until this file is complete.
//
// Implementation checklist:
//   [ ] use core::arch::x86_64::*;
//   [ ] impl Mul<Mat4> for Mat4 behind #[cfg(target_feature = "avx512f")]
//   [ ] Update avx/mat4.rs Mul<Mat4> to also gate on not(target_feature="avx512f")
//   [ ] Add "x86-64-v4" to bench-vs-all workflow target_cpu choices
//   [ ] Bench: cargo bench --bench vs_all -p mid-math (with native/avx512 flags)
