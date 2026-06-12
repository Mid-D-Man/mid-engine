// crates/mid-math/src/f32/sse2/mod.rs
//! SSE2-backed implementations for x86 / x86_64.
//!
//! Vec3, Vec4 and Quat store a `__m128` register as their only field
//! (`#[repr(transparent)]`). This means the value literally IS the SIMD
//! register — no load/store needed to perform arithmetic, matching glam's
//! approach and eliminating the scalar-extraction overhead.
//!
//! Mat4 uses four named Vec4 fields (Build 8 storage fix), keeping all four
//! columns live in XMM registers across Mul<Mat4> calls — zero memory traffic
//! for the LHS.
//!
//! Mat2 packs both columns into a single __m128: transpose = 1 shuffle,
//! determinant = ~4 instructions, mul_mat2 = 6 instructions.

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
