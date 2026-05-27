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

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub(crate) mod sse2;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub use sse2::{DVec2, DVec4, DQuat};

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), target_feature = "avx2"))]
pub(crate) mod avx2;

#[cfg(target_arch = "aarch64")]
pub(crate) mod neon;
#[cfg(target_arch = "aarch64")]
pub use neon::{DVec2, DVec4, DQuat};

#[cfg(all(any(target_arch = "wasm32", target_arch = "wasm64"), target_feature = "simd128"))]
pub(crate) mod wasm;
#[cfg(all(any(target_arch = "wasm32", target_arch = "wasm64"), target_feature = "simd128"))]
pub use wasm::{DVec2, DVec4, DQuat};

#[cfg(not(any(
    target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64",
    all(any(target_arch = "wasm32", target_arch = "wasm64"), target_feature = "simd128"),
)))]
// ── f64 types ─────────────────────────────────────────────────────────────────
pub use f64::{
    DVec2, DVec3, DVec4, DQuat,
    DMat2, DMat3, DMat4,
    DAffine2, DAffine3,
    DDualQuat,
    DEPSILON,
};
