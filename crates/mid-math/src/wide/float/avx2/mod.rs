// crates/mid-math/src/wide/float/avx2/mod.rs
//! AVX2-backed float wide types — x86 / x86_64 with target_feature = "avx2".

pub mod vec3x8;
pub use vec3x8::Vec3x8;
