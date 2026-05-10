// crates/mid-math/src/wide/int/mod.rs
// Integer wide types — platform dispatch.

pub(crate) mod scalar;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub(crate) mod sse2;

// ── Platform dispatch ─────────────────────────────────────────────────────────
// IMask types differ from their module names → can be imported with short path.
// Lowercase types (i32x4 etc.) share their name with their module; we must use
// the full sub-module path (e.g. sse2::i32x4::i32x4) to avoid the name clash.

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub use sse2::{IMask4, IMask8, IMask16};

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(non_camel_case_types)]
pub use sse2::{
    i32x4::i32x4, u32x4::u32x4,
    i16x8::i16x8, u16x8::u16x8,
    i8x16::i8x16, u8x16::u8x16,
};

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
pub use scalar::{IMask4, IMask8, IMask16};

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
#[allow(non_camel_case_types)]
pub use scalar::{
    i32x4::i32x4, u32x4::u32x4,
    i16x8::i16x8, u16x8::u16x8,
    i8x16::i8x16, u8x16::u8x16,
};
