// crates/mid-math/src/f64/mod.rs
pub mod dvec2;
pub mod dvec3;
pub mod dvec4;
pub mod dquat;
pub mod dmat2;
pub mod dmat3;
pub mod dmat4;
pub mod daffine2;
pub mod daffine3;
pub mod ddual_quat;

// ── Always-scalar types (no SIMD variant) ─────────────────────────────────────
//
// True for the default build. DVec3 is the one exception when built with
// --features coresimd: crate::f64::coresimd::DVec3 (pub(crate), f64x4-backed)
// exists as an additional, separate implementation -- same non-invasive
// pattern f32/mod.rs already uses for its own coresimd module. Doesn't
// change what plain `DVec3` below resolves to; that's still always this
// scalar one regardless of the coresimd feature.
pub use dvec3::DVec3;
pub use dmat2::DMat2;
pub use dmat3::DMat3;
pub use dmat4::DMat4;
pub use daffine2::DAffine2;
pub use daffine3::DAffine3;
pub use ddual_quat::DDualQuat;
pub use dvec2::DEPSILON;

// ── Portable SIMD (coresimd) ───────────────────────────────────────────────────
//
// See coresimd/mod.rs for why DVec3 is the one worth having here that the
// hardware-specific backends structurally can't provide. pub(crate) same as
// f32's version -- not part of the public API yet, mirrors that module's
// current (also unreviewed-by-a-compiler-in-this-session, also
// zero-CI-coverage-until-now) status.
#[cfg(feature = "coresimd")]
pub(crate) mod coresimd;

// ── SIMD-dispatched: x86 / x86_64 ────────────────────────────────────────────
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub(crate) mod sse2;
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), not(feature = "force-scalar")))]
pub use sse2::{DVec2, DVec4, DQuat};

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), target_feature = "avx2"))]
pub(crate) mod avx2;

// ── SIMD-dispatched: aarch64 ──────────────────────────────────────────────────
#[cfg(target_arch = "aarch64")]
pub(crate) mod neon;
#[cfg(all(target_arch = "aarch64", not(feature = "force-scalar")))]
pub use neon::{DVec2, DVec4, DQuat};

// ── SIMD-dispatched: wasm32/wasm64 with simd128 ───────────────────────────────
#[cfg(all(any(target_arch = "wasm32", target_arch = "wasm64"), target_feature = "simd128"))]
pub(crate) mod wasm;
#[cfg(all(
    any(target_arch = "wasm32", target_arch = "wasm64"), target_feature = "simd128",
    not(feature = "force-scalar"),
))]
pub use wasm::{DVec2, DVec4, DQuat};

// ── Scalar fallback ───────────────────────────────────────────────────────────
#[cfg(any(feature = "force-scalar", not(any(
    target_arch = "x86", target_arch = "x86_64",
    target_arch = "aarch64",
    all(any(target_arch = "wasm32", target_arch = "wasm64"), target_feature = "simd128"),
))))]
pub use dvec2::DVec2;

#[cfg(any(feature = "force-scalar", not(any(
    target_arch = "x86", target_arch = "x86_64",
    target_arch = "aarch64",
    all(any(target_arch = "wasm32", target_arch = "wasm64"), target_feature = "simd128"),
))))]
pub use dvec4::DVec4;

#[cfg(any(feature = "force-scalar", not(any(
    target_arch = "x86", target_arch = "x86_64",
    target_arch = "aarch64",
    all(any(target_arch = "wasm32", target_arch = "wasm64"), target_feature = "simd128"),
))))]
pub use dquat::DQuat;
