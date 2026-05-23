// crates/mid-math/src/f64/avx2/mod.rs
//! AVX2 fast-paths for f64 types — x86 / x86_64 with `target_feature = "avx2"`.
//!
//! `__m256d` holds 4× f64 (256 bits, 32-byte aligned).
//! This is the natural register for DVec4 (4 f64) and DMat4 columns (4 f64 each).
//!
//! # Status
//!
//! | Type   | Op                  | Status              | OPT tag    |
//! |--------|---------------------|---------------------|------------|
//! | DVec4  | all ops (f64×4)     | Stub — not yet impl | OPT-F64-1  |
//! | DMat4  | Mul (column ops)    | Stub — not yet impl | OPT-F64-2  |
//! | DVec3x4d | wide SoA (4×DVec3) | Stub — not yet impl | OPT-F64-3  |
//!
//! # Why AVX2 matters for f64
//!
//! SSE2 holds 2× f64 per `__m128d`. DVec4 currently uses 2× `__m128d` (lo+hi).
//! AVX2 `__m256d` holds 4× f64 in a **single** register:
//!   - DVec4 add/sub/mul: 2 SSE2 instructions → 1 AVX2 instruction (2× throughput)
//!   - DVec4 dot: 2 separate horizontal adds → 1 `_mm256_hadd_pd` chain
//!   - DMat4 × DVec4: 4 columns × 4 f64 = each column fits in one `__m256d`
//!   - DMat4 × DMat4: 4 outer products over `__m256d` columns
//!
//! # Implementation plan
//!
//! ## OPT-F64-1 — DVec4 via `__m256d`
//!
//! 1. Run `cargo bench --bench vs_all_f64 -p mid-math` → record SSE2 baseline.
//! 2. Implement `DVec4` storing a single `__m256d` in `avx2/dvec4.rs`.
//!    Algorithm for add: `_mm256_add_pd(self.0, rhs.0)` — one instruction.
//!    Algorithm for dot: `_mm256_hadd_pd` pair then extract lane 0.
//!    Algorithm for normalize: dot → `_mm256_sqrt_pd` → `_mm256_div_pd` → mask.
//! 3. Gate SSE2 DVec4 with `#[cfg(not(target_feature = "avx2"))]`.
//! 4. Gate AVX2 DVec4 with `#[cfg(target_feature = "avx2")]`.
//! 5. Run bench again. Target: ~1.5× SSE2 throughput on add/mul ops.
//! 6. Paste both bench outputs for sign-off.
//!
//! DO NOT implement OPT-F64-1 until:
//!   - f64 SSE2 DVec4 is fully benched ([RELEASE] numbers recorded in Step Summary)
//!   - f64 NEON DVec4 benched on aarch64 (cross build via `cross`)
//!
//! ## OPT-F64-2 — DMat4 column multiply via `__m256d`
//!
//! DMat4 stores `[[f64; 4]; 4]` (4 columns, each 4 f64).
//! With AVX2: load each column as `__m256d`, broadcast each element of rhs,
//! accumulate 4 FMA256 operations for the full multiply.
//! Expected: ~2× SSE2 throughput for DMat4 × DMat4.
//!
//! Reference: same algorithm as `f32/avx2/mat4.rs` (OPT-7) but operating
//! on `__m256d` (4 f64) instead of `__m256` (8 f32).
//!
//! DO NOT implement OPT-F64-2 until OPT-F64-1 is benched and merged.
//!
//! ## OPT-F64-3 — DVec3x4d: 4× DVec3 in SoA layout
//!
//! Store x,y,z each as `__m256d` holding 4 independent f64 values.
//! Enables 4 DVec3 dot products / cross products simultaneously at f64 precision.
//! Use case: physics solver operating on f64 positions/velocities in batches.
//!
//! Layout: `{ x: __m256d, y: __m256d, z: __m256d }` — 96 bytes, 32-byte aligned.
//! Mirrors the f32 `Vec3x4` design but at double precision.
//!
//! DO NOT implement OPT-F64-3 until OPT-F64-1 and OPT-F64-2 are benched and merged.
//!
//! # Build / test
//!
//! To compile AVX2 paths locally (requires Haswell+ CPU, not 2010 MBP):
//!   RUSTFLAGS="-C target-feature=+avx2" cargo build -p mid-math
//!   RUSTFLAGS="-C target-feature=+avx2" cargo test  -p mid-math --release
//!   RUSTFLAGS="-C target-feature=+avx2" cargo bench --bench vs_all_f64 -p mid-math
//!
//! CI (GitHub Actions runners are Haswell+) compiles and runs AVX2 paths automatically
//! when RUSTFLAGS includes `+avx2` — add this to the release CI job when OPT-F64-1 lands.

pub(crate) mod dvec4;
