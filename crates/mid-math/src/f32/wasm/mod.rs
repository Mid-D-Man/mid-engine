// crates/mid-math/src/f32/wasm/mod.rs
//! WASM SIMD implementations — wasm32 / wasm64 with simd128.
//!
//! TODO: Replace with v128-backed Vec3/Vec4/Quat and WASM Mat4 ops.
//!       Requires: RUSTFLAGS="-C target-feature=+simd128"
//!       Reference: glam/src/f32/wasm/ for the proven patterns.
//!       Note: WASM scalar is ~5-10x slower than native — this is non-negotiable
//!       for any 3D web context.
//!
//! For now, delegate to scalar so wasm32 targets compile and are correct.

pub use crate::f32::scalar::vec3::Vec3;
pub use crate::f32::scalar::vec4::Vec4;
pub use crate::f32::scalar::quat::Quat;
pub use crate::f32::scalar::mat4::Mat4;
