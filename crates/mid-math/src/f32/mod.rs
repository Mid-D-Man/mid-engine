// crates/mid-math/src/f32/mod.rs

pub(crate) mod math;

mod vec2;
pub mod mat2;
pub mod mat3;
pub mod affine3;   // ← NEW

pub use vec2::Vec2;
pub use mat2::Mat2;
pub use mat3::Mat3;
pub use affine3::Affine3;  // ← NEW

pub(crate) mod scalar;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub(crate) mod sse2;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub use sse2::{Vec3, Vec4, Quat, Mat4};

#[cfg(target_arch = "aarch64")]
pub(crate) mod neon;

#[cfg(target_arch = "aarch64")]
pub use neon::{Vec3, Vec4, Quat, Mat4};

#[cfg(all(
    any(target_arch = "wasm32", target_arch = "wasm64"),
    target_feature = "simd128",
))]
pub(crate) mod wasm;

#[cfg(all(
    any(target_arch = "wasm32", target_arch = "wasm64"),
    target_feature = "simd128",
))]
pub use wasm::{Vec3, Vec4, Quat, Mat4};

#[cfg(not(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(
        any(target_arch = "wasm32", target_arch = "wasm64"),
        target_feature = "simd128",
    ),
)))]
pub use scalar::{Vec3, Vec4, Quat, Mat4};
