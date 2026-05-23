// crates/mid-math/src/f64/avx2/dvec4.rs
//! DVec4 using `__m256d` — OPT-F64-1 placeholder.
//!
//! # Why __m256d for DVec4?
//!
//! DVec4 is exactly 4× f64 = 256 bits = one AVX2 `__m256d` register.
//! The current SSE2 implementation uses 2× `__m128d` (lo=[x,y], hi=[z,w]),
//! requiring paired instructions for every operation.
//!
//! With a single `__m256d`:
//!   add / sub / mul / div  : 2 instructions (SSE2) → 1 (AVX2)
//!   abs / neg              : 1 instruction each (direct)
//!   dot product            : `_mm256_dp_pd` or hadd chain in 3 instructions
//!   normalize              : sqrt → div → mask in 3 paired → 3 single
//!   lerp                   : 4 instructions → 2 (with optional FMA256)
//!
//! # Implementation plan (OPT-F64-1)
//!
//! 1. Bench SSE2 DVec4:
//!      RUSTFLAGS="-C target-feature=+avx2" cargo bench --bench vs_all_f64 -p mid-math
//!    Record baseline (SSE2 still active since OPT-F64-1 not landed yet).
//!
//! 2. Implement `pub struct DVec4(__m256d)` here with `#[repr(transparent)]`.
//!    Lane layout: lane0=x, lane1=y, lane2=z, lane3=w (same byte order as
//!    XYZW<f64> — Deref via impl_dvec4_deref! is still valid).
//!
//! 3. Key intrinsics:
//!      _mm256_add_pd / _mm256_sub_pd / _mm256_mul_pd / _mm256_div_pd
//!      _mm256_sqrt_pd      — normalize inner sqrt
//!      _mm256_cmp_pd::<30> — OQ greater-than for normalize guard (30 = _CMP_GT_OQ)
//!      _mm256_and_pd       — mask degenerate lanes
//!      _mm256_set1_pd      — broadcast scalar
//!      _mm256_loadu_pd / _mm256_storeu_pd — load/store [f64; 4]
//!
//! 4. Dot product algorithm (no _mm256_dp_pd equivalent for packed f64):
//!      mul  = _mm256_mul_pd(a, b)              // [x*rx, y*ry, z*rz, w*rw]
//!      hadd = _mm256_hadd_pd(mul, mul)          // [(x*rx+y*ry), ..., (z*rz+w*rw), ...]
//!      // hadd produces [lo+lo, lo+lo, hi+hi, hi+hi] — extract and add pairs
//!      lo   = _mm256_extractf128_pd(hadd, 0)   // [x*rx+y*ry, x*rx+y*ry]
//!      hi   = _mm256_extractf128_pd(hadd, 1)   // [z*rz+w*rw, z*rz+w*rw]
//!      sum  = _mm_add_pd(lo, hi)               // [dot, dot] in __m128d
//!      f64  = _mm_cvtsd_f64(sum)
//!
//! 5. Gate in sse2/mod.rs:
//!      Add `#[cfg(not(target_feature = "avx2"))]` to the SSE2 DVec4 type/impls.
//!      Add `#[cfg(target_feature = "avx2")]` here.
//!
//! 6. Run bench again. Target: ≥1.5× SSE2 on bulk add/normalize.
//!    Paste both [RELEASE] bench outputs as GitHub Step Summary before merging.
//!
//! # DO NOT implement until:
//!
//!   - [RELEASE] f64 SSE2 numbers are recorded in a Step Summary
//!   - f64 NEON numbers are recorded (cross bench to aarch64)
//!   - The f32 AVX2 Mat4 multiply (OPT-7) is completed as a reference
//!     for the __m256-based implementation pattern
//!
//! Nothing exported yet. Implementation lands here during OPT-F64-1.
