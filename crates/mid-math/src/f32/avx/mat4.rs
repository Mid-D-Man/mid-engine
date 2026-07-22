// crates/mid-math/src/f32/avx/mat4.rs
//! AVX + FMA  4×4 matrix multiply.
//!
//! Active when: avx + fma present AND avx512f absent.
//! When avx512f is available, avx512/mat4.rs takes over (all 4 output cols
//! in one ZMM instead of 2-column pairs per YMM).
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
//! → ~1.9× throughput gain; target latency ≈ 3.5–4.0 ns.
//!
//! Source: cglm `include/cglm/simd/avx/mat4.h` → `glm_mat4_mul_avx`

use core::ops::Mul;

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use crate::f32::sse2::mat4::Mat4;
use crate::f32::sse2::vec4::Vec4;
use crate::f32::sse2::vec3::Vec3;
use crate::wide::float::sse2::vec3x4::Vec3x4;

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
// Compiled ONLY when avx+fma present AND avx512f absent.
// avx512/mat4.rs provides the superior 4-column-per-ZMM implementation when
// avx512f is available.
// sse2/mat4.rs gates its Mul<Mat4> with not(all(avx,fma)) so exactly one
// implementation exists per target at all times.

#[cfg(not(target_feature = "avx512f"))]
// Steps aside when avx512f is present: avx512/mat4.rs supersedes this specific
// impl with an all-4-columns-in-one-ZMM approach (~2.0 ns vs ~4.0 ns here).
// This exclusion used to live on the whole `avx` module in f32/mod.rs, but
// that incorrectly took vec4.rs/quat.rs down with it — they have no
// avx512-specific replacement and should always be active under avx+fma.
#[cfg(not(target_feature = "avx512f"))]
impl Mul<Mat4> for Mat4 {
    type Output = Mat4;

    /// AVX + FMA 4×4 matrix multiply.
    ///
    /// `_mm256_set_m128(hi, lo)` — lo → low 128 bits, hi → high 128 bits.
    ///
    /// `_mm256_permute2f128_ps::<IMM>(a, a)`:
    ///   `0x00` → both output halves = a low half  → `[col_k   | col_k  ]`
    ///   `0x11` → both output halves = a high half → `[col_k+1 | col_k+1]`
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        unsafe {
            // Pack LHS column pairs into 256-bit registers.
            let lhs_01 = _mm256_set_m128(self.y_axis.0, self.x_axis.0);
            let lhs_23 = _mm256_set_m128(self.w_axis.0, self.z_axis.0);

            // Hoist: duplicate each LHS column into BOTH YMM halves.
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

            Self {
                x_axis: Vec4(_mm256_castps256_ps128(c01)),
                y_axis: Vec4(_mm256_extractf128_ps::<1>(c01)),
                z_axis: Vec4(_mm256_castps256_ps128(c23)),
                w_axis: Vec4(_mm256_extractf128_ps::<1>(c23)),
            }
        }
    }
}
// MulAssign lives in sse2/mat4.rs (ungated) — delegates to whichever Mul<Mat4>
// is in scope. No second definition needed here.

// ── Transform helpers, FMA-fused ────────────────────────────────────────────
//
// sse2/mat4.rs's plain versions step aside under this same
// not(avx512f) && avx+fma gate (see the #[cfg] on each method there).
//
// Same 128-bit __m128 shape as the SSE2 version -- this isn't the 256-bit
// two-column trick col_pair() uses above, transform_point only ever has one
// output column. The win is purely instruction count: 3 mul + 3 add (6 ops)
// collapses to 1 mul + 2 fmadd + 1 add (4 ops) via _mm_fmadd_ps, which is
// available on any target with target_feature="fma" regardless of whether
// 256-bit AVX registers are in play elsewhere.
//
// This previously didn't exist at all -- transform_point ran the plain SSE2
// path unconditionally on every x86 tier, including native/x86-64-v3, which
// left FMA on the table for the single most-called op in the entity-transform
// benchmarks. Re-bench after this lands; whether it closes the loop-level gap
// against glam or not, it's a straightforward win on its own terms.
#[cfg(not(target_feature = "avx512f"))]
impl Mat4 {
    #[inline(always)]
    pub fn transform_point(self, p: Vec3) -> Vec3 {
        unsafe {
            let bx = _mm_shuffle_ps::<0b00_00_00_00>(p.0, p.0);
            let by = _mm_shuffle_ps::<0b01_01_01_01>(p.0, p.0);
            let bz = _mm_shuffle_ps::<0b10_10_10_10>(p.0, p.0);
            let res = _mm_mul_ps(self.x_axis.0, bx);
            let res = _mm_fmadd_ps(self.y_axis.0, by, res);
            let res = _mm_fmadd_ps(self.z_axis.0, bz, res);
            Vec3(_mm_add_ps(res, self.w_axis.0))
        }
    }

    #[inline(always)]
    pub fn transform_vector(self, v: Vec3) -> Vec3 {
        unsafe {
            let bx = _mm_shuffle_ps::<0b00_00_00_00>(v.0, v.0);
            let by = _mm_shuffle_ps::<0b01_01_01_01>(v.0, v.0);
            let bz = _mm_shuffle_ps::<0b10_10_10_10>(v.0, v.0);
            let res = _mm_mul_ps(self.x_axis.0, bx);
            let res = _mm_fmadd_ps(self.y_axis.0, by, res);
            Vec3(_mm_fmadd_ps(self.z_axis.0, bz, res))
        }
    }

    // ── Wide SIMD batch transforms, FMA-fused ───────────────────────────────
    //
    // FMA counterpart to sse2/mat4.rs's transform_vec3x4/transform_vec3x4_dir
    // -- those had the exact same missing-FMA gap transform_point had before
    // its own fix (always ran the plain SSE2 path regardless of tier, no
    // #[cfg] to step aside). Same row/broadcast math as the SSE2 version,
    // just fused, matching the exact function names and signatures so
    // existing callers and the existing correctness test
    // (vec3x4_mat4_transform_matches_scalar in tests/wide_tests.rs) don't
    // need to change at all -- this is a pure step-aside-and-replace under
    // the same #[cfg] the SSE2 side now has, not a new API.
    #[inline(always)]
    pub fn transform_vec3x4(
        self,
        v: Vec3x4,
    ) -> Vec3x4 {
        unsafe {
            let c0x = _mm_shuffle_ps::<0b00_00_00_00>(self.x_axis.0, self.x_axis.0);
            let c0y = _mm_shuffle_ps::<0b01_01_01_01>(self.x_axis.0, self.x_axis.0);
            let c0z = _mm_shuffle_ps::<0b10_10_10_10>(self.x_axis.0, self.x_axis.0);
            let c1x = _mm_shuffle_ps::<0b00_00_00_00>(self.y_axis.0, self.y_axis.0);
            let c1y = _mm_shuffle_ps::<0b01_01_01_01>(self.y_axis.0, self.y_axis.0);
            let c1z = _mm_shuffle_ps::<0b10_10_10_10>(self.y_axis.0, self.y_axis.0);
            let c2x = _mm_shuffle_ps::<0b00_00_00_00>(self.z_axis.0, self.z_axis.0);
            let c2y = _mm_shuffle_ps::<0b01_01_01_01>(self.z_axis.0, self.z_axis.0);
            let c2z = _mm_shuffle_ps::<0b10_10_10_10>(self.z_axis.0, self.z_axis.0);
            let c3x = _mm_shuffle_ps::<0b00_00_00_00>(self.w_axis.0, self.w_axis.0);
            let c3y = _mm_shuffle_ps::<0b01_01_01_01>(self.w_axis.0, self.w_axis.0);
            let c3z = _mm_shuffle_ps::<0b10_10_10_10>(self.w_axis.0, self.w_axis.0);

            let rx = _mm_add_ps(_mm_fmadd_ps(c1x, v.y, _mm_mul_ps(c0x, v.x)), _mm_fmadd_ps(c2x, v.z, c3x));
            let ry = _mm_add_ps(_mm_fmadd_ps(c1y, v.y, _mm_mul_ps(c0y, v.x)), _mm_fmadd_ps(c2y, v.z, c3y));
            let rz = _mm_add_ps(_mm_fmadd_ps(c1z, v.y, _mm_mul_ps(c0z, v.x)), _mm_fmadd_ps(c2z, v.z, c3z));

            Vec3x4 { x: rx, y: ry, z: rz }
        }
    }

    #[inline(always)]
    pub fn transform_vec3x4_dir(
        self,
        v: Vec3x4,
    ) -> Vec3x4 {
        unsafe {
            let c0x = _mm_shuffle_ps::<0b00_00_00_00>(self.x_axis.0, self.x_axis.0);
            let c0y = _mm_shuffle_ps::<0b01_01_01_01>(self.x_axis.0, self.x_axis.0);
            let c0z = _mm_shuffle_ps::<0b10_10_10_10>(self.x_axis.0, self.x_axis.0);
            let c1x = _mm_shuffle_ps::<0b00_00_00_00>(self.y_axis.0, self.y_axis.0);
            let c1y = _mm_shuffle_ps::<0b01_01_01_01>(self.y_axis.0, self.y_axis.0);
            let c1z = _mm_shuffle_ps::<0b10_10_10_10>(self.y_axis.0, self.y_axis.0);
            let c2x = _mm_shuffle_ps::<0b00_00_00_00>(self.z_axis.0, self.z_axis.0);
            let c2y = _mm_shuffle_ps::<0b01_01_01_01>(self.z_axis.0, self.z_axis.0);
            let c2z = _mm_shuffle_ps::<0b10_10_10_10>(self.z_axis.0, self.z_axis.0);

            let rx = _mm_fmadd_ps(c2x, v.z, _mm_fmadd_ps(c1x, v.y, _mm_mul_ps(c0x, v.x)));
            let ry = _mm_fmadd_ps(c2y, v.z, _mm_fmadd_ps(c1y, v.y, _mm_mul_ps(c0y, v.x)));
            let rz = _mm_fmadd_ps(c2z, v.z, _mm_fmadd_ps(c1z, v.y, _mm_mul_ps(c0z, v.x)));

            Vec3x4 { x: rx, y: ry, z: rz }
        }
    }
                }
