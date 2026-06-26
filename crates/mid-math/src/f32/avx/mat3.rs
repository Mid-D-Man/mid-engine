// crates/mid-math/src/f32/avx/mat3.rs
//! AVX + FMA accelerated 3×3 matrix multiplication.
//!
//! Uses 256-bit YMM registers to compute **two output columns per pass**,
//! halving the number of SIMD multiply instructions vs the XMM scalar path.
//!
//! ## Performance profile (vs scalar mul_mat3)
//!
//! | Path        | Multiply ops | FMA ops |
//! |-------------|-------------|---------|
//! | Scalar      | 27 mul + 18 add = 45 | 0 |
//! | SSE2 XMM    | 9 _mm_mul_ps + 6 _mm_add_ps = 15 | 0 |
//! | **AVX+FMA** | **6 _mm256_fmadd_ps + 3 _mm_fmadd_ps = 9** | **9** |
//!
//! ## Gate
//!
//! Compiled only when **both** `avx` and `fma` target features are active:
//! ```
//! RUSTFLAGS="-C target-feature=+avx,+fma" cargo build
//! ```
//!
//! In `f32/mat3.rs`, `mul_mat3` dispatches here automatically when this cfg is set:
//! ```rust
//! #[cfg(all(any(target_arch="x86",target_arch="x86_64"), target_feature="avx", target_feature="fma"))]
//! return unsafe { crate::f32::avx::mat3::mat3_mul_avx(self, rhs) };
//! ```
//!
//! ## Safety
//!
//! The function is marked `unsafe` because it uses raw SIMD intrinsics. The
//! temporary 4-float buffers are fully stack-allocated; no pointer arithmetic
//! reaches outside those buffers. The caller guarantees `avx` and `fma` are
//! available (enforced by the `#[target_feature]` attribute + cfg gate).

use crate::f32::mat3::Mat3;

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

/// Multiply `a * b` (column-major) using AVX YMM + FMA.
///
/// Requires: `avx` and `fma` target features.
/// Called automatically from `Mat3::mul_mat3` via the cfg dispatch block.
#[inline]
#[target_feature(enable = "avx,fma")]
pub unsafe fn mat3_mul_avx(a: Mat3, b: Mat3) -> Mat3 {
    // ── Load A columns into XMM (lane 3 = 0.0 for safety) ────────────────────
    //
    // _mm_set_ps(lane3, lane2, lane1, lane0) — note reversed argument order.
    let a_c0 = _mm_set_ps(0.0, a.cols[0][2], a.cols[0][1], a.cols[0][0]);
    let a_c1 = _mm_set_ps(0.0, a.cols[1][2], a.cols[1][1], a.cols[1][0]);
    let a_c2 = _mm_set_ps(0.0, a.cols[2][2], a.cols[2][1], a.cols[2][0]);

    // ── Pass 1: output columns 0 and 1 via YMM ───────────────────────────────
    //
    // Pack each A column into both 128-bit lanes of a YMM register:
    //   Y_ax = [a_cx | a_cx]
    // This lets us compute two output columns simultaneously.
    let y_a0 = _mm256_set_m128(a_c0, a_c0);
    let y_a1 = _mm256_set_m128(a_c1, a_c1);
    let y_a2 = _mm256_set_m128(a_c2, a_c2);

    // Scale factors from B:
    //   Low  128-bit lane → col 0 of output (B.cols[0][*])
    //   High 128-bit lane → col 1 of output (B.cols[1][*])
    let y_s0 = _mm256_set_m128(
        _mm_set1_ps(b.cols[1][0]),   // high lane: B col1 row0
        _mm_set1_ps(b.cols[0][0]),   // low  lane: B col0 row0
    );
    let y_s1 = _mm256_set_m128(
        _mm_set1_ps(b.cols[1][1]),
        _mm_set1_ps(b.cols[0][1]),
    );
    let y_s2 = _mm256_set_m128(
        _mm_set1_ps(b.cols[1][2]),
        _mm_set1_ps(b.cols[0][2]),
    );

    // FMA chain: out_01 = A_c0*B[*][0] + A_c1*B[*][1] + A_c2*B[*][2]
    //   step 1: tmp     = A_c0 * s0
    //   step 2: tmp     = A_c1 * s1 + tmp    (fmadd)
    //   step 3: out_01  = A_c2 * s2 + tmp    (fmadd)
    let y_tmp  = _mm256_mul_ps(y_a0, y_s0);
    let y_tmp  = _mm256_fmadd_ps(y_a1, y_s1, y_tmp);
    let y_out01 = _mm256_fmadd_ps(y_a2, y_s2, y_tmp);

    // Extract col 0 (low 128-bit lane) and col 1 (high 128-bit lane).
    let x_out0 = _mm256_castps256_ps128(y_out01);
    let x_out1 = _mm256_extractf128_ps(y_out01, 1);

    // ── Pass 2: output column 2 via XMM ──────────────────────────────────────

    let x_s0 = _mm_set1_ps(b.cols[2][0]);
    let x_s1 = _mm_set1_ps(b.cols[2][1]);
    let x_s2 = _mm_set1_ps(b.cols[2][2]);

    let x_tmp  = _mm_mul_ps(a_c0, x_s0);
    let x_tmp  = _mm_fmadd_ps(a_c1, x_s1, x_tmp);
    let x_out2 = _mm_fmadd_ps(a_c2, x_s2, x_tmp);

    // ── Store back to [f32; 3] columns ───────────────────────────────────────
    //
    // We can't _mm_storeu_ps directly into [f32;3] (only 12 bytes) without
    // overwriting adjacent memory. Use a 4-float stack buffer, take first 3.
    let mut result = Mat3::ZERO;
    let mut buf = [0.0f32; 4];

    _mm_storeu_ps(buf.as_mut_ptr(), x_out0);
    result.cols[0] = [buf[0], buf[1], buf[2]];

    _mm_storeu_ps(buf.as_mut_ptr(), x_out1);
    result.cols[1] = [buf[0], buf[1], buf[2]];

    _mm_storeu_ps(buf.as_mut_ptr(), x_out2);
    result.cols[2] = [buf[0], buf[1], buf[2]];

    result
  }
