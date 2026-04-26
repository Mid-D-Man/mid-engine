// crates/mid-math/src/f32/sse2/mod.rs
//! SSE2-backed implementations for x86 / x86_64.
//!
//! Vec3, Vec4 and Quat store a `__m128` register as their only field
//! (`#[repr(transparent)]`). This means the value literally IS the SIMD
//! register — no load/store needed to perform arithmetic, matching glam's
//! approach and eliminating the scalar-extraction overhead that caused the
//! 4x criterion gap.
//!
//! Mat4 keeps `[[f32;4];4]` column-major storage (align 16) but gains
//! SSE2 fast-paths for multiply, inverse and transform.
//!
//! FFI boundary: C callers never see __m128. They use the #[repr(C)] types
//! in crate::ffi::types (CVec3, CVec4, CQuat, CMat4). Conversion between
//! the internal type and the C type is zero-cost — LLVM sees through it.

pub mod vec3;
pub mod vec4;
pub mod quat;
pub mod mat4;

pub use vec3::Vec3;
pub use vec4::Vec4;
pub use quat::Quat;
pub use mat4::Mat4;
