// crates/mid-math/src/wide/float/avx2/mod.rs
//! AVX2-associated float wide types — x86 / x86_64, always compiled (see
//! `wide/float/mod.rs`'s doc comment for why this is no longer gated on
//! the `avx2` target feature).

pub mod f32x8;
pub mod mask8;
pub mod vec3x8;

pub use vec3x8::Vec3x8;
pub use mask8::Mask8;
