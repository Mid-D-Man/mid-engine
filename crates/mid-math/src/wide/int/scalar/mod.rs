// crates/mid-math/src/wide/int/scalar/mod.rs
// Scalar fallback integer wide types — non-x86 platforms.
//
// NOTE: module names (i32x4, u32x4, …) match their contained type names.
// Re-exporting the type here would clash with the module in the same namespace.
// The parent (wide/int/mod.rs) imports types via the full sub-module path,
// e.g. scalar::i32x4::i32x4.

pub mod imask4;
pub mod imask8;
pub mod imask16;
pub mod i32x4;
pub mod u32x4;
pub mod i16x8;
pub mod u16x8;
pub mod i8x16;
pub mod u8x16;

// IMask* names differ from module names — safe to re-export here.
pub use imask4::IMask4;
pub use imask8::IMask8;
pub use imask16::IMask16;
// lowercase types are NOT re-exported here; parent resolves via scalar::i32x4::i32x4 etc.
