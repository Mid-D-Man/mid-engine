// crates/mid-math/src/f32/avx2/mat4.rs
//! AVX2 Mat4 operations — OPT-7 placeholder.
//!
//! # Why AVX2 for Mat4 multiply?
//!
//! SSE2 processes one 128-bit xmm register at a time (4× f32).
//! AVX2 processes one 256-bit ymm register at a time (8× f32).
//!
//! Mat4 × Mat4 produces 4 output columns, each a 4× f32 Vec4.
//! With SSE2 we compute one output column per instruction group (4 muls + 3 adds = 7 instr).
//! With AVX2 we compute TWO output columns simultaneously in a single ymm register,
//! halving the instruction count for the full multiply.
//!
//! Reference: cglm `glm_mul_avx` in `include/cglm/simd/avx/affine.h` —
//! that file is in-tree at `include/cglm/simd/avx/` for reference.
//!
//! # Implementation plan (OPT-7)
//!
//! 1. Run `cargo bench --bench vs_all -p mid-math` and record SSE2 baseline.
//! 2. Implement `mul_avx2(self, rhs: Mat4) -> Mat4` using `__m256` intrinsics.
//!    Algorithm (from cglm glm_mul_avx):
//!      a. Load cols 0+1 of m1 as single __m256 (glmm_load256 equivalent).
//!      b. Load cols 2+3 of m1 as single __m256.
//!      c. Load cols 0+1 of m2, broadcast each element to its 8-lane position.
//!      d. FMA accumulate: result = Σ(col_i * broadcast(m2_element_i)).
//!      e. Store result cols 0+1 and 2+3.
//! 3. Add `#[cfg(not(target_feature = "avx2"))]` to the `Mul` impl in `sse2/mat4.rs`.
//! 4. Add `#[cfg(target_feature = "avx2")]` `Mul` impl here.
//! 5. Run bench again. Target: ~3.5 ns (≈ 2× SSE2 throughput of ~7.1 ns).
//! 6. Paste both bench outputs before and after for sign-off.
//!
//! # DO NOT implement until OPT-1 and OPT-2 are complete and benched.
//!
//! OPT-1: SSE2 general inverse  (~117 ns → target ~20 ns)
//! OPT-2: SSE2 TRS inverse      (~ 78 ns → target ~10 ns)
//! OPT-7: AVX2 Mat4 multiply    (~  7 ns → target ~3.5 ns)

// Nothing exported yet. Impl block lands here during OPT-7.
