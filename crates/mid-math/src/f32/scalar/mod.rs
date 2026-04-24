// crates/mid-math/src/f32/scalar/mod.rs
//! Scalar (non-SIMD) implementations — always compiled.
//!
//! These serve two purposes:
//!   1. Fallback on non-SIMD targets (MIPS, RISC-V, etc.)
//!   2. Correctness reference and test baseline on SIMD targets
//!      (via `inverse_scalar()`, `inverse_trs_scalar()` methods)

pub mod vec3;
pub mod vec4;
pub mod quat;
pub mod mat4;

pub use vec3::Vec3;
pub use vec4::Vec4;
pub use quat::Quat;
pub use mat4::Mat4;
