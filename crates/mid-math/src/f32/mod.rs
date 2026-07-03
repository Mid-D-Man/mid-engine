// crates/mid-math/src/f32/mod.rs
pub(crate) mod math;

mod vec2;
pub mod mat2;
pub mod mat3;
pub mod affine2;
pub mod affine3;
pub mod dual_quat;
pub mod mat4_projection;

pub use vec2::Vec2;
pub use mat3::Mat3;
pub use affine2::Affine2;
pub use affine3::Affine3;
pub use dual_quat::DualQuat;

pub(crate) mod scalar;

// ── force-scalar override ─────────────────────────────────────────────────────
// `--features force-scalar` forces the pure-scalar Vec3/Vec4/Quat/Mat4/Mat2
// path regardless of target_arch. Without this, `scalar::*` was UNREACHABLE
// on any x86/x86_64/aarch64/wasm+simd128 runner — module selection below is
// keyed on target_arch, not target_feature or -C target-cpu, so even
// `-C target-cpu=x86-64` (SSE2 baseline, no AVX) still pulled in the sse2
// module. There was no way to bench true scalar on any GitHub-hosted runner.
//
// Scoped to f32 only for now — f64::mod.rs and wide/*/mod.rs are untouched
// and unaffected either way; they keep their normal arch-specific selection
// regardless of this feature. Nothing here changes their compilation.
//
// mod sse2 / mod neon / mod wasm below stay unconditional (not gated on this
// feature) — avx/mat4.rs and avx512/mat4.rs import `crate::f32::sse2::mat4::Mat4`
// directly by module path, not through the top-level alias, so they keep
// compiling fine even when force-scalar wins the alias below. neon/wasm just
// become unused dead code in that case (no -D warnings on bench builds, so
// this is silent).

// ── x86 / x86_64 ─────────────────────────────────────────────────────────────

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub(crate) mod sse2;

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), not(feature = "force-scalar")))]
pub use sse2::{Vec3, Vec4, Quat, Mat4, Mat2};

// AVX + FMA fast path — Mat4::mul only (~4.0 ns vs 7.0 ns SSE2).
// Excluded when avx512f is present: avx512/mat4.rs supersedes with
// all-4-columns-in-one-ZMM approach (~2.0 ns).
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    target_feature = "avx",
    target_feature = "fma",
    not(target_feature = "avx512f"),
))]
pub(crate) mod avx;

// AVX-512 fast paths — avx512f required.
// Currently provides: Mat4::mul via _mm512_fmadd_ps (~2.0 ns target).
// Activate: RUSTFLAGS="-C target-cpu=x86-64-v4" or "-C target-cpu=native"
// (NOT guaranteed on GitHub-hosted runners — see hwcheck gate in the bench
// workflow; rustc emits avx512f instructions on request regardless of the
// actual host, so this needs a runtime feature-detect, not just the cfg).
//
// Gating chain:
//   avx512f active → avx512/mat4.rs provides Mul<Mat4>
//   avx+fma but no avx512f → avx/mat4.rs provides Mul<Mat4>
//   SSE2 only → sse2/mat4.rs provides Mul<Mat4>
// MulAssign is ungated in sse2/mat4.rs and delegates to whichever is active.
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    target_feature = "avx512f",
))]
pub(crate) mod avx512;

// ── aarch64 ──────────────────────────────────────────────────────────────────

#[cfg(target_arch = "aarch64")]
pub(crate) mod neon;

#[cfg(all(target_arch = "aarch64", not(feature = "force-scalar")))]
pub use neon::{Vec3, Vec4, Quat, Mat4, Mat2};

// SVE / SVE2 — STUB. cfg never fires on stable Rust (nightly-only as of 2026-06).
// Hardware: Apple M4, Neoverse N2 (AWS Graviton3), Snapdragon 8 Gen 3, X Elite.
#[cfg(all(target_arch = "aarch64", target_feature = "sve"))]
pub(crate) mod sve;

// SME (Scalable Matrix Extension) — STUB. No Rust support as of 2026-06.
// Hardware: Apple M4, Cortex-X4.
#[cfg(all(target_arch = "aarch64", target_feature = "sme"))]
pub(crate) mod sme;

// ── WASM SIMD128 ─────────────────────────────────────────────────────────────

#[cfg(all(any(target_arch = "wasm32", target_arch = "wasm64"), target_feature = "simd128"))]
pub(crate) mod wasm;

// Mat2 comes from wasm::mat2 (v128-backed) — was scalar-fallback before,
// see wasm/mat2.rs header for the bench numbers that gave it away.
#[cfg(all(
    any(target_arch = "wasm32", target_arch = "wasm64"), target_feature = "simd128",
    not(feature = "force-scalar"),
))]
pub use wasm::{Vec3, Vec4, Quat, Mat4, Mat2};

// ── Portable SIMD (coresimd) ──────────────────────────────────────────────────

#[cfg(feature = "coresimd")]
pub(crate) mod coresimd;

#[cfg(all(
    feature = "coresimd",
    not(feature = "force-scalar"),
    not(any(
        target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64",
        all(any(target_arch = "wasm32", target_arch = "wasm64"), target_feature = "simd128"),
    )),
))]
pub use coresimd::{Vec3, Vec4, Quat, Mat4};

#[cfg(all(
    feature = "coresimd",
    not(feature = "force-scalar"),
    not(any(
        target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64",
        all(any(target_arch = "wasm32", target_arch = "wasm64"), target_feature = "simd128"),
    )),
))]
pub use mat2::Mat2;

// ── Scalar fallback ───────────────────────────────────────────────────────────
// Fires either naturally (no SIMD backend exists for this target) or when
// force-scalar wins the vote regardless of what's available.

#[cfg(any(
    feature = "force-scalar",
    not(any(
        target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64",
        all(any(target_arch = "wasm32", target_arch = "wasm64"), target_feature = "simd128"),
        feature = "coresimd",
    )),
))]
pub use scalar::{Vec3, Vec4, Quat, Mat4};

#[cfg(any(
    feature = "force-scalar",
    not(any(
        target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64",
        all(any(target_arch = "wasm32", target_arch = "wasm64"), target_feature = "simd128"),
        feature = "coresimd",
    )),
))]
pub use mat2::Mat2;
