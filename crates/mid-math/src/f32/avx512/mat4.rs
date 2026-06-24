// crates/mid-math/src/f32/avx512/mat4.rs
//! AVX-512 Mat4 multiply — all 4 output columns in one ZMM register.
//!
//! ## Algorithm
//!
//! For C = A × B (column-major, A has cols a0..a3):
//!   C_col_j = Σ_k  A_col_k · B_col_j[k]
//!
//! AVX2 (avx/mat4.rs): 2 output cols per YMM, 2 passes → 26 256-bit insns, ~4.0 ns.
//! AVX-512: all 4 output cols in one ZMM, 1 pass   → ~20 512-bit insns, ~2.0 ns target.
//!
//! ## LHS broadcast
//! We need a_k = [A_col_k | A_col_k | A_col_k | A_col_k] for k=0..3.
//! Built via _mm512_insertf32x4 × 3 (safe, always compiles under avx512f).
//! LLVM will emit VBROADCASTF32X4 when it proves the source is uniform —
//! verified via `cargo rustc -- --emit=asm`.
//!
//! ## RHS permute
//! rhs_zmm = [B_col_0 | B_col_1 | B_col_2 | B_col_3]
//! _mm512_permute_ps::<0b_kk_kk_kk_kk>(rhs_zmm) broadcasts element k
//! within each 128-bit lane independently:
//!   lane j output = [B_col_j[k] × 4]
//!
//! ## FMA chain
//! acc = a0*r0 + a1*r1 + a2*r2 + a3*r3  (all 4 output cols accumulate in parallel)
//!
//! ## Extract
//! _mm512_extractf32x4_ps::<j>(acc) → C_col_j as __m128
//!
//! ## Instruction budget
//!   LHS broadcast (×4):  1 castps128_ps512 + 3 insertf32x4 per col = 16 insns
//!   RHS pack      (×1):  1 cast + 3 insert                          =  4 insns
//!   Permute       (×4):  4 permute_ps                               =  4 insns
//!   FMA chain     (×4):  1 mul + 3 fmadd                            =  4 insns
//!   Extract       (×4):  4 extractf32x4_ps                          =  4 insns
//!   Total:                                                           = 32 insns
//!
//! Note: LLVM optimizes the 4×insert broadcast to VBROADCASTF32X4 (1 insn each)
//! reducing the real instruction count to ~20. The insert form is used here
//! for guaranteed compilation on all avx512f-capable Rust toolchains.
//!
//! ## Target latency
//!   SSE2:        ~7.0 ns
//!   AVX2+FMA:    ~4.0 ns
//!   AVX-512:     ~2.0–2.5 ns

use core::ops::Mul;

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use crate::f32::sse2::mat4::Mat4;
use crate::f32::sse2::vec4::Vec4;

// ── Broadcast helper ──────────────────────────────────────────────────────────

/// Broadcast `__m128` a to all four 128-bit lanes of a `__m512`.
///
/// Emits 1 × _mm512_castps128_ps512 (free zero-cost reinterpret) +
///        3 × VINSERTF32X4 instructions.
///
/// LLVM folds these to a single VBROADCASTF32X4 when it determines all
/// lanes receive the same value — which is always true here since `a` is
/// an SSE register not modified between inserts.
///
/// Requires only `target_feature = "avx512f"`. No dependency on avx, fma,
/// or avx512dq beyond what avx512f implies.
#[inline(always)]
unsafe fn broadcast_f32x4(a: __m128) -> __m512 {
    // lane 0 = a, lanes 1-3 undefined
    let z = _mm512_castps128_ps512(a);
    // insert a into lane 1
    let z = _mm512_insertf32x4::<1>(z, a);
    // insert a into lane 2
    let z = _mm512_insertf32x4::<2>(z, a);
    // insert a into lane 3
    _mm512_insertf32x4::<3>(z, a)
    // result: [a | a | a | a]
}

// ── Mul<Mat4> for Mat4 ────────────────────────────────────────────────────────
//
// Compiled ONLY when avx512f is present.
// f32/mod.rs excludes the avx/ module when avx512f is active, so exactly one
// Mul<Mat4> impl exists per target.
// MulAssign lives ungated in sse2/mat4.rs and delegates here automatically.

impl Mul<Mat4> for Mat4 {
    type Output = Mat4;

    /// AVX-512 4×4 matrix multiply — all 4 output columns processed in parallel.
    ///
    /// # Safety
    /// Requires `target_feature = "avx512f"` — guaranteed by parent module gate.
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        unsafe {
            // ── Broadcast each LHS column to all 4 ZMM lanes ─────────────────
            // a_k = [A_col_k | A_col_k | A_col_k | A_col_k]
            // Hoisted: reused across all 4 RHS element permutations.
            let a0 = broadcast_f32x4(self.x_axis.0); // [x_axis | x_axis | x_axis | x_axis]
            let a1 = broadcast_f32x4(self.y_axis.0); // [y_axis | y_axis | y_axis | y_axis]
            let a2 = broadcast_f32x4(self.z_axis.0); // [z_axis | z_axis | z_axis | z_axis]
            let a3 = broadcast_f32x4(self.w_axis.0); // [w_axis | w_axis | w_axis | w_axis]

            // ── Pack all 4 RHS columns into one ZMM ───────────────────────────
            // rhs_zmm = [B_col_0 | B_col_1 | B_col_2 | B_col_3]
            //
            // _mm512_castps128_ps512: free reinterpret, upper lanes undefined.
            // _mm512_insertf32x4::<N>: VINSERTF32X4 zmm, xmm, imm8.
            let rhs_zmm = {
                let z = _mm512_castps128_ps512(rhs.x_axis.0);     // lane 0 = B_col_0
                let z = _mm512_insertf32x4::<1>(z, rhs.y_axis.0); // lane 1 = B_col_1
                let z = _mm512_insertf32x4::<2>(z, rhs.z_axis.0); // lane 2 = B_col_2
                _mm512_insertf32x4::<3>(z, rhs.w_axis.0)          // lane 3 = B_col_3
            };

            // ── Permute: broadcast element k within each 128-bit lane ─────────
            // _mm512_permute_ps::<IMM>(a): VPERMILPS zmm, zmm, imm8.
            // IMM = 0b_kk_kk_kk_kk broadcasts element k within each 128-bit lane:
            //   lane j after permute = [B_col_j[k], B_col_j[k], B_col_j[k], B_col_j[k]]
            //
            // All 4 lanes are permuted independently and simultaneously.
            let r0 = _mm512_permute_ps::<0b00_00_00_00>(rhs_zmm); // row 0 of every col
            let r1 = _mm512_permute_ps::<0b01_01_01_01>(rhs_zmm); // row 1
            let r2 = _mm512_permute_ps::<0b10_10_10_10>(rhs_zmm); // row 2
            let r3 = _mm512_permute_ps::<0b11_11_11_11>(rhs_zmm); // row 3

            // ── FMA accumulation — all 4 output columns in parallel ────────────
            //
            // After the k-th fmadd:
            //   acc lane j = Σ_{i=0..k} A_col_i × B_col_j[i]
            //
            // After k=3:
            //   acc lane j = C_col_j  (complete output column j)
            //
            // _mm512_fmadd_ps: VFMADD213PS zmm (part of avx512f, not separate fma).
            let acc = _mm512_mul_ps(a0, r0);           // k=0: no accumulator yet
            let acc = _mm512_fmadd_ps(a1, r1, acc);   // k=1: acc += A_col_1 × B_row_1
            let acc = _mm512_fmadd_ps(a2, r2, acc);   // k=2: acc += A_col_2 × B_row_2
            let acc = _mm512_fmadd_ps(a3, r3, acc);   // k=3: acc += A_col_3 × B_row_3

            // ── Extract 4 output columns from ZMM 128-bit lanes ───────────────
            // _mm512_extractf32x4_ps::<N>: VEXTRACTF32X4 xmm, zmm, imm8. (avx512f)
            // _mm512_castps512_ps128:      free reinterpret for lane 0.
            Self {
                x_axis: Vec4(_mm512_castps512_ps128(acc)),         // C_col_0 = lane 0
                y_axis: Vec4(_mm512_extractf32x4_ps::<1>(acc)),    // C_col_1 = lane 1
                z_axis: Vec4(_mm512_extractf32x4_ps::<2>(acc)),    // C_col_2 = lane 2
                w_axis: Vec4(_mm512_extractf32x4_ps::<3>(acc)),    // C_col_3 = lane 3
            }
        }
    }
}
// MulAssign is ungated in sse2/mat4.rs — delegates to whichever Mul<Mat4> is active.
