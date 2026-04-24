// crates/mid-math/src/f32/neon/mod.rs
//! NEON implementations — aarch64 (iOS, Android, Apple Silicon).
//!
//! TODO: Replace with float32x4_t-backed Vec3/Vec4/Quat and NEON Mat4 ops.
//!       Priority after SSE2 path is stable.
//!       Reference: glam/src/f32/neon/ for the proven NEON patterns.
//!
//! For now, delegate to scalar so aarch64 targets compile and are correct.

pub use crate::f32::scalar::vec3::Vec3;
pub use crate::f32::scalar::vec4::Vec4;
pub use crate::f32::scalar::quat::Quat;
pub use crate::f32::scalar::mat4::Mat4;
