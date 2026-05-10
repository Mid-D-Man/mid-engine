// crates/mid-math/src/wide/int/sse2/mod.rs
// SSE2-backed integer wide types — x86 / x86_64 only.
//
// Same name-clash constraint as scalar: module i32x4 vs type i32x4.
// Parent imports types via sse2::i32x4::i32x4 etc.

pub mod imask4;
pub mod imask8;
pub mod imask16;
pub mod i32x4;
pub mod u32x4;
pub mod i16x8;
pub mod u16x8;
pub mod i8x16;
pub mod u8x16;

// IMask* safe to re-export (different names from modules).
pub use imask4::IMask4;
pub use imask8::IMask8;
pub use imask16::IMask16;
// Lowercase types resolved by parent via full sub-module path.
