// crates/mid-math/src/f32/neon/mod.rs
//! NEON implementations — aarch64 (iOS, Android, Apple Silicon).
//!
//! Status:
//!   Vec3  — NEON float32x4_t 
//!   Vec4  — transitional scalar re-export (next)
//!   Quat  — transitional scalar re-export (next)
//!   Mat4  — transitional scalar re-export (next)
//!
//! Cross-compilation for local testing:
//!   cargo install cross
//!   cross test -p mid-math --target aarch64-unknown-linux-gnu
//!   cross test -p mid-math --target aarch64-unknown-linux-gnu --release

pub mod vec3;
pub mod vec4;

// Quat and Mat4 not yet NEON-ified — keep scalar until next pass.
pub use crate::f32::scalar::quat::Quat;
pub use crate::f32::scalar::mat4::Mat4;

pub use vec3::Vec3;
pub use vec4::Vec4;
