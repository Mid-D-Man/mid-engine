// crates/mid-math/src/f64/mod.rs
//! Double-precision (f64) math types.
//!
//! ## Platform dispatch
//!
//! Three types receive platform-specific SIMD implementations:
//!   DVec2 — 2×f64 maps perfectly to one SIMD register on all targets
//!   DVec4 — 4×f64 stored as 2×register (lo=[x,y], hi=[z,w])
//!   DQuat — same 2×register layout as DVec4
//!
//! All other types remain scalar:
//!   DVec3    — 3×f64, no padding; 3-lane f64 SIMD needs 2 registers + masking
//!              for marginal gain. AVX2 __m256d (4 f64) is the justified target
//!              but that lives in f64/avx2/ (see OPT-F64-3).
//!   DMat2/3/4, DAffine3 — scalar until AVX2 column ops are implemented.
//!
//! ## SIMD backends (priority order)
//!
//! | Target              | Backend         | Types              |
//! |---------------------|-----------------|--------------------|
//! | x86 / x86_64        | SSE2 (__m128d)  | DVec2, DVec4, DQuat|
//! | aarch64             | NEON (f64x2)    | DVec2, DVec4, DQuat|
//! | wasm32/64 + simd128 | WASM (f64x2)    | DVec2, DVec4, DQuat|
//! | all others          | scalar          | all types          |
//!
//! ## AVX2 roadmap (f64/avx2/)
//!
//! When OPT-F64-1 lands: DVec4 via __m256d (4 f64 in one ymm register).
//! When OPT-F64-2 lands: DMat4 column multiply via __m256d.
//! When OPT-F64-3 lands: DVec3x4d wide type (4×DVec3 SoA in __m256d).
//! See f64/avx2/mod.rs for full implementation plan and sequencing rules.
//!
//! DEPSILON = 1e-12 (vs 1e-6 for f32 types).

// ── Scalar submodules — always compiled ───────────────────────────────────────
//
// DVec2/DVec4/DQuat scalar versions live here too. On SIMD targets they are
// NOT re-exported as the canonical types, but they remain compiled so that:
//   (a) DEPSILON is always accessible via dvec2::DEPSILON
//   (b) The scalar types are usable as reference implementations in tests
//   (c) Internal users can path-qualify them explicitly if needed

pub mod dvec2;
pub mod dvec3;
pub mod dvec4;
pub mod dquat;
pub mod dmat2;
pub mod dmat3;
pub mod dmat4;
pub mod daffine3;

// ── SSE2 — x86 / x86_64 ──────────────────────────────────────────────────────

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub(crate) mod sse2;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub use sse2::{DVec2, DVec4, DQuat};

// ── AVX2 — x86 / x86_64 with target_feature = "avx2" ────────────────────────
//
// Exports nothing yet. Stubs live in f64/avx2/ with implementation plans.
// Gated here so the module compiles on AVX2 targets and OPT tags are visible.

#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    target_feature = "avx2",
))]
pub(crate) mod avx2;

// ── NEON — aarch64 ────────────────────────────────────────────────────────────
//
// float64x2_t is mandatory on all AArch64 targets. No runtime check needed.
// vaddvq_f64 (single-instruction horizontal add) gives NEON a clear advantage
// over the SSE2 shuffle-based dot product.

#[cfg(target_arch = "aarch64")]
pub(crate) mod neon;

#[cfg(target_arch = "aarch64")]
pub use neon::{DVec2, DVec4, DQuat};

// ── WASM SIMD128 ──────────────────────────────────────────────────────────────
//
// f64x2_* intrinsics available in WASM simd128. Build with:
//   RUSTFLAGS="-C target-feature=+simd128" cargo build --target wasm32-unknown-unknown
//
// f64x2_abs and f64x2_neg are direct instructions — simpler than SSE2's
// sign-mask ANDNOT pattern.

#[cfg(all(
    any(target_arch = "wasm32", target_arch = "wasm64"),
    target_feature = "simd128",
))]
pub(crate) mod wasm;

#[cfg(all(
    any(target_arch = "wasm32", target_arch = "wasm64"),
    target_feature = "simd128",
))]
pub use wasm::{DVec2, DVec4, DQuat};

// ── Scalar fallback ───────────────────────────────────────────────────────────
//
// Active when no SIMD backend applies:
//   - Not x86/x86_64
//   - Not aarch64
//   - Not wasm32/wasm64 + simd128

#[cfg(not(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(
        any(target_arch = "wasm32", target_arch = "wasm64"),
        target_feature = "simd128",
    ),
)))]
pub use self::{dvec2::DVec2, dvec4::DVec4, dquat::DQuat};

// ── Always-scalar re-exports ──────────────────────────────────────────────────

pub use dvec3::DVec3;
pub use dmat2::DMat2;
pub use dmat3::DMat3;
pub use dmat4::DMat4;
pub use daffine3::DAffine3;

/// f64 comparison epsilon. 1e-12 — tighter than f32 EPSILON (1e-6).
///
/// Used internally by all f64 types. Exported so users don't need to
/// hardcode the magic number.
pub const DEPSILON: f64 = dvec2::DEPSILON;
