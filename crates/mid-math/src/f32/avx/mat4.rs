// crates/mid-math/src/f32/avx/mat4.rs
//! AVX + FMA  4×4 matrix multiply.
//!
//! ## Algorithm — two output columns per 256-bit register
//!
//! For C = A × B (column-major, A has cols a0..a3):
//!   C_col_j = Σ_k  A_col_k · B_col_j[k]
//!
//! We process two output columns at once by packing them into one YMM:
//!   result_low  = C_col_j
//!   result_high = C_col_{j+1}
//!
//! Steps:
//!  1. `lhs_01 = [a0 | a1]`, `lhs_23 = [a2 | a3]`  (pack LHS column pairs)
//!  2. Hoist: `ak_dup = [a_k | a_k]` for k=0..3   (done ONCE, reused for all rhs pairs)
//!  3. Per output pair  (j, j+1):
//!     a. `rhs_pair = [B_col_j | B_col_{j+1}]`
//!     b. `r_k = _mm256_permute_ps(rhs_pair, k<<6|k<<4|k<<2|k)` →
//!             `[B_col_j[k]×4 | B_col_{j+1}[k]×4]`
//!        (permute_ps works on each 128-bit half independently)
//!     c. acc = a0_dup·r0 + a1_dup·r1 + a2_dup·r2 + a3_dup·r3  (FMA chain)
//!  4. Extract low/high halves → C output columns.
//!
//! ## Instruction count
//! LHS setup (once): 2 set_m128 + 4 permute2f128 = 6
//! Per RHS pair (×2): 1 set_m128 + 4 permute_ps + 1 mul + 3 fmadd = 9  → 18
//! Extract (×4): 2 cast (free) + 2 extractf128 = 2
//! Total: ≈ 26 AVX/FMA instructions (each 256-bit) vs ≈ 32 SSE2 (each 128-bit)
//! → ~1.9× throughput gain; target latency ≈ 3.0–3.5 ns.
//!
//! Source: cglm `include/cglm/simd/avx/mat4.h` → `glm_mat4_mul_avx`

use core::ops::Mul;

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use crate::f32::sse2::mat4::Mat4;
use crate::f32::sse2::vec4::Vec4;

// ── Inner helper ──────────────────────────────────────────────────────────────

/// Compute 2 output columns of `C = A × B` simultaneously.
///
/// # Arguments
/// * `a0..a3` — LHS columns each duplicated to both YMM halves: `[A_col_k | A_col_k]`
/// * `rhs_pair` — two consecutive RHS columns packed: `[B_col_j | B_col_{j+1}]`
///
/// # Returns
/// `[C_col_j | C_col_{j+1}]` packed in one `__m256`.
///
/// `_mm256_permute_ps::<IMM>` applies `IMM` independently to both 128-bit halves.
/// IMM `0b_kk_kk_kk_kk` broadcasts lane `k` within each half:
///   low half  → `[B_col_j[k]     × 4]`
///   high half → `[B_col_{j+1}[k] × 4]`
///
/// FMA chain: `acc = a0*r0 + a1*r1 + a2*r2 + a3*r3`.
/// Both halves accumulate their respective output column in parallel.
#[inline(always)]
unsafe fn col_pair(
    a0: __m256, a1: __m256, a2: __m256, a3: __m256,
    rhs_pair: __m256,
) -> __m256 {
    let r0 = _mm256_permute_ps::<0b00_00_00_00>(rhs_pair); // row 0 of each rhs col
    let r1 = _mm256_permute_ps::<0b01_01_01_01>(rhs_pair); // row 1
    let r2 = _mm256_permute_ps::<0b10_10_10_10>(rhs_pair); // row 2
    let r3 = _mm256_permute_ps::<0b11_11_11_11>(rhs_pair); // row 3

    let acc = _mm256_mul_ps(a0, r0);
    let acc = _mm256_fmadd_ps(a1, r1, acc);
    let acc = _mm256_fmadd_ps(a2, r2, acc);
    _mm256_fmadd_ps(a3, r3, acc)
}

// ── Mul<Mat4> for Mat4 ────────────────────────────────────────────────────────
//
// This impl is compiled ONLY when avx+fma are present.
// `sse2/mat4.rs` gates its Mul<Mat4> with `#[cfg(not(all(target_feature="avx",
// target_feature="fma")))]` so exactly one implementation exists per target.

impl Mul<Mat4> for Mat4 {
    type Output = Mat4;

    /// AVX + FMA 4×4 matrix multiply.
    ///
    /// `_mm256_set_m128(hi, lo)` — lo → low 128 bits, hi → high 128 bits.
    ///
    /// `_mm256_permute2f128_ps::<IMM>(a, a)`:
    ///   `0x00` → both output halves = a low half  → `[col_k   | col_k  ]`
    ///   `0x11` → both output halves = a high half → `[col_k+1 | col_k+1]`
    ///   (IMM encoding: bits[1:0] = source for low out; bits[5:4] = source for high out;
    ///    value 0 = src low half, 1 = src high half)
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        unsafe {
            // Pack LHS column pairs into 256-bit registers.
            //   lhs_01: low=x_axis (col 0), high=y_axis (col 1)
            //   lhs_23: low=z_axis (col 2), high=w_axis (col 3)
            let lhs_01 = _mm256_set_m128(self.y_axis.0, self.x_axis.0);
            let lhs_23 = _mm256_set_m128(self.w_axis.0, self.z_axis.0);

            // Hoist: duplicate each LHS column into BOTH YMM halves.
            // Done once here; reused for both RHS column pairs below.
            let a0 = _mm256_permute2f128_ps::<0x00>(lhs_01, lhs_01); // [x_axis | x_axis]
            let a1 = _mm256_permute2f128_ps::<0x11>(lhs_01, lhs_01); // [y_axis | y_axis]
            let a2 = _mm256_permute2f128_ps::<0x00>(lhs_23, lhs_23); // [z_axis | z_axis]
            let a3 = _mm256_permute2f128_ps::<0x11>(lhs_23, lhs_23); // [w_axis | w_axis]

            // Compute output columns 0+1 simultaneously
            let rhs_01 = _mm256_set_m128(rhs.y_axis.0, rhs.x_axis.0);
            let c01    = col_pair(a0, a1, a2, a3, rhs_01);

            // Compute output columns 2+3 simultaneously
            let rhs_23 = _mm256_set_m128(rhs.w_axis.0, rhs.z_axis.0);
            let c23    = col_pair(a0, a1, a2, a3, rhs_23);

            // Split 256-bit results back into 128-bit output columns.
            // _mm256_castps256_ps128 is a zero-cost reinterpret (no instruction emitted).
            Self {
                x_axis: Vec4(_mm256_castps256_ps128(c01)),
                y_axis: Vec4(_mm256_extractf128_ps::<1>(c01)),
                z_axis: Vec4(_mm256_castps256_ps128(c23)),
                w_axis: Vec4(_mm256_extractf128_ps::<1>(c23)),
            }
        }
    }
}
// Note: MulAssign lives in sse2/mat4.rs (ungated) and delegates to whichever
// Mul<Mat4> is in scope — no second definition needed here.
