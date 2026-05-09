// crates/mid-math/src/wide/int/scalar/mod.rs  (updated)
//! Scalar fallback integer wide types — non-x86 platforms.

pub mod imask4;
pub mod imask8;
pub mod imask16;
pub mod i32x4;
pub mod u32x4;
pub mod i16x8;
pub mod u16x8;
pub mod i8x16;
pub mod u8x16;

pub use imask4::IMask4;
pub use imask8::IMask8;
pub use imask16::IMask16;

#[allow(non_camel_case_types)]
pub use i32x4::i32x4;
#[allow(non_camel_case_types)]
pub use u32x4::u32x4;
#[allow(non_camel_case_types)]
pub use i16x8::i16x8;
#[allow(non_camel_case_types)]
pub use u16x8::u16x8;
#[allow(non_camel_case_types)]
pub use i8x16::i8x16;
#[allow(non_camel_case_types)]
pub use u8x16::u8x16;
