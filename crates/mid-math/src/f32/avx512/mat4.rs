// crates/mid-math/src/f32/avx512/mat4.rs
//! AVX-512 Mat4 multiply — all 4 output columns in one ZMM register.
//!
//! ## Algorithm
//!
//! For C = A × B (column-major, A has cols a0..a3):
//!   C_col_j = Σ_k  A_col_k · B_col_j[k]
//!
//! AVX2 (avx/mat4.rs) packs 2 RHS cols into one YMM, processes 2 output
//! columns per pass, 2 passes total — 26 256-bit instructions, ~4.0 ns.
//!
//! AVX-512 packs ALL 4 RHS cols into one ZMM, processes all 4 output
//! columns in a SINGLE pass — ~21 512-bit instructions, target ~2.0 ns.
//!
//! Pass structure:
//!  1. Broadcast each LHS column to all 4 ZMM 128-bit lanes:
//!       a_k_dup = [A_col_k | A_col_k | A_col_k | A_col_k]
//!  2. Pack all 4 RHS columns into one ZMM:
//!       rhs_zmm = [B_col_0 | B_col_1 | B_col_2 | B_col_3]
//!  3. For k=0..3:
//!       r_k = permute(rhs_zmm, broadcast_k)
//!           = [B_col_0[k]×4 | B_col_1[k]×4 | B_col_2[k]×4 | B_col_3[k]×4]
//!       acc = fmadd(a_k_dup, r_k, acc)
//!  4. extract128(acc, j) = C_col_j
//!
//! Instruction count:
//!   LHS broadcast (once): 4 × broadcastf32x4_ps              =  4
//!   RHS pack     (once):  1 castps128_ps512 + 3 insertf32x4  =  4
//!   k-loop       (×4):   1 permute_ps + 1 fmadd (or mul)     =  8
//!   Extract      (×4):   4 × extractf32x4_ps                 =  4
//!   Total:                                                    = 20 AVX-512 instructions
//!
//! All AVX-512 intrinsics used require only `target_feature = "avx512f"`.
//!
//! ## Benchmarks (expected, subject to CI confirmation)
//! SSE2 (x86-64):       ~7.0 ns
//! AVX2+FMA (x86-64-v3): ~4.0 ns
//! AVX-512 (x86-64-v4): ~2.0-2.5 ns  ← this file

use core::ops::Mul;

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use crate::f32::sse2::mat4::Mat4;
use crate::f32::sse2::vec4::Vec4;

impl Mul<Mat4> for Mat4 {
    type Output = Mat4;

    /// AVX-512 4×4 matrix multiply.
    ///
    /// All 4 output columns accumulate in parallel inside a single ZMM (512-bit) register.
    ///
    /// `_mm512_broadcastf32x4_ps(__m128)`:
    ///   Copies a 128-bit lane to all 4 positions → [a | a | a | a].
    ///   LLVM lowers to VBROADCASTF32X4 (from memory) or a register shuffle
    ///   when the __m128 is already in an XMM register (our case — Vec4 fields).
    ///
    /// `_mm512_permute_ps::<IMM>(a)`:
    ///   VPERMILPS ZMM — permutes within each 128-bit lane independently.
    ///   IMM = 0b_kk_kk_kk_kk broadcasts element k to all 4 positions in
    ///   each lane. Identical semantics to `_mm256_permute_ps` in avx/mat4.rs
    ///   but applied to all four 128-bit lanes of the ZMM simultaneously.
    ///
    /// `_mm512_fmadd_ps(a, b, c)`:
    ///   VFMADD213PS ZMM — fused multiply-add on 16 f32 lanes.
    ///   Part of AVX-512F (not the separate FMA extension for XMM/YMM).
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        unsafe {
            // ── Broadcast each LHS column to all 4 ZMM 128-bit lanes ─────────
            //   a0 = [x_axis | x_axis | x_axis | x_axis]
            //   a1 = [y_axis | y_axis | y_axis | y_axis]  etc.
            //
            // These are hoisted — reused for every element of all 4 output cols.
            let a0 = _mm512_broadcastf32x4_ps(self.x_axis.0);
            let a1 = _mm512_broadcastf32x4_ps(self.y_axis.0);
            let a2 = _mm512_broadcastf32x4_ps(self.z_axis.0);
            let a3 = _mm512_broadcastf32x4_ps(self.w_axis.0);

            // ── Pack all 4 RHS columns into one ZMM ──────────────────────────
            //   rhs_zmm = [B_col_0 | B_col_1 | B_col_2 | B_col_3]
            //
            // _mm512_castps128_ps512: zero-cost reinterpret, upper 384 bits undefined.
            // _mm512_insertf32x4::<N>: VINSERTF32X4 zmm — insert __m128 at 128-bit lane N.
            // Requires avx512f (ZMM destination; the YMM destination form needs avx512dq).
            let rhs_zmm = {
                let z = _mm512_castps128_ps512(rhs.x_axis.0);        // lane 0 = B_col_0
                let z = _mm512_insertf32x4::<1>(z, rhs.y_axis.0);    // lane 1 = B_col_1
                let z = _mm512_insertf32x4::<2>(z, rhs.z_axis.0);    // lane 2 = B_col_2
                _mm512_insertf32x4::<3>(z, rhs.w_axis.0)             // lane 3 = B_col_3
            };

            // ── Broadcast element k within each 128-bit lane of rhs_zmm ──────
            //
            // After permute with IMM = 0b_kk_kk_kk_kk:
            //   r_k lane j = [B_col_j[k], B_col_j[k], B_col_j[k], B_col_j[k]]
            //
            // Each 128-bit lane independently broadcasts its k-th element.
            // The 4 lanes remain independent — they hold B_col_{0..3}[k] respectively.
            let r0 = _mm512_permute_ps::<0b00_00_00_00>(rhs_zmm); // element 0 broadcast
            let r1 = _mm512_permute_ps::<0b01_01_01_01>(rhs_zmm); // element 1 broadcast
            let r2 = _mm512_permute_ps::<0b10_10_10_10>(rhs_zmm); // element 2 broadcast
            let r3 = _mm512_permute_ps::<0b11_11_11_11>(rhs_zmm); // element 3 broadcast

            // ── FMA chain — all 4 output columns accumulate in parallel ───────
            //
            // After iteration k:
            //   acc 128-bit lane j = Σ_{i=0..k} A_col_i × B_col_j[i]
            //
            // After k=3:
            //   acc lane j = Σ_{i=0..3} A_col_i × B_col_j[i] = C_col_j  ✓
            //
            // a0 = [A_col_0 | A_col_0 | A_col_0 | A_col_0]
            // r0 lane j = [B_col_j[0] × 4]
            // a0 × r0 lane j = A_col_0 × B_col_j[0]  (contributing row-0 of B to col j)
            let acc = _mm512_mul_ps(a0, r0);           // k=0: no accumulator yet
            let acc = _mm512_fmadd_ps(a1, r1, acc);   // k=1: acc += A_col_1 × B_col_j[1]
            let acc = _mm512_fmadd_ps(a2, r2, acc);   // k=2: acc += A_col_2 × B_col_j[2]
            let acc = _mm512_fmadd_ps(a3, r3, acc);   // k=3: acc += A_col_3 × B_col_j[3]

            // ── Extract 4 output columns from their respective ZMM lanes ─────
            //   VEXTRACTF32X4 xmm, zmm, imm8  (avx512f)
            Self {
                x_axis: Vec4(_mm512_extractf32x4_ps::<0>(acc)), // C_col_0
                y_axis: Vec4(_mm512_extractf32x4_ps::<1>(acc)), // C_col_1
                z_axis: Vec4(_mm512_extractf32x4_ps::<2>(acc)), // C_col_2
                w_axis: Vec4(_mm512_extractf32x4_ps::<3>(acc)), // C_col_3
            }
        }
    }
}
// MulAssign lives in sse2/mat4.rs (ungated) — delegates to whichever Mul<Mat4>
// is active. No second definition needed here.
