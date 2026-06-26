// crates/mid-math/src/f32/avx/mod.rs
//! AVX + FMA fast paths for x86 / x86_64.
//!
//! Compiled only when both `target_feature = "avx"` and `target_feature = "fma"`
//! are present. The parent `f32::mod` enforces this via cfg on the module itself.
//! Only operations that benefit from 256-bit width live here.
pub mod mat4;
pub mod mat3;
