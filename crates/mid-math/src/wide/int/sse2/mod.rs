// crates/mid-math/src/wide/int/sse2/mod.rs
//! SSE2-backed integer wide types — x86 / x86_64 only.
//!
//! All types are #[repr(transparent)] over __m128i.
//! Constants use the UnionCast pattern established in f32/sse2/.

pub mod imask4;
pub mod i32x4;
pub mod u32x4;

pub use imask4::IMask4;
#[allow(non_camel_case_types)]
pub use i32x4::i32x4;
#[allow(non_camel_case_types)]
pub use u32x4::u32x4;
