// crates/mid-math/src/wide/int/scalar/mod.rs
//! Scalar fallback integer wide types — non-x86 platforms.
//!
//! Same interface as the SSE2 versions, implemented with arrays.
//! AARCH64 NEON and WASM SIMD128 fast paths arrive in Phase 5.

pub mod imask4;
pub mod i32x4;
pub mod u32x4;

pub use imask4::IMask4;
#[allow(non_camel_case_types)]
pub use i32x4::i32x4;
#[allow(non_camel_case_types)]
pub use u32x4::u32x4;
