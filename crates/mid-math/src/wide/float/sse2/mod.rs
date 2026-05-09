// crates/mid-math/src/wide/float/sse2/mod.rs
//! SSE2-backed float wide types — x86 / x86_64 only.

pub mod mask4;
pub mod f32x4;
pub mod vec3x4;
pub mod quatx4;

pub use mask4::Mask4;
#[allow(non_camel_case_types)]
pub use f32x4::f32x4;
pub use vec3x4::Vec3x4;
pub use quatx4::QuatX4;
