// crates/mid-math/src/f32/neon/mod.rs
//! NEON implementations — aarch64 (iOS, Android, Apple Silicon, Linux ARM).
//!
//! Status:
//!   Vec3    float32x4_t, full NEON
//!   Vec4    float32x4_t, full NEON, vaddvq_f32 dot
//!   Quat    float32x4_t, full NEON mul_quat + FMA slerp
//!   Mat4    NEON Mul<Vec4> + Mul<Mat4> (FMA); scalar inverse (Phase 2)
//!   Mat2    float32x4_t, single-register pack (Build 22 — was scalar fallback)
//!
//! Cross-compilation from x86_64 dev machine:
//!   cargo install cross
//!   cross test  -p mid-math --target aarch64-unknown-linux-gnu
//!   cross bench -p mid-math --target aarch64-unknown-linux-gnu --release

pub mod vec3;
pub mod vec4;
pub mod quat;
pub mod mat4;
pub mod mat2;

pub use vec3::Vec3;
pub use vec4::Vec4;
pub use quat::Quat;
pub use mat4::Mat4;
pub use mat2::Mat2;
