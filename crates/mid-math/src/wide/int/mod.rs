// crates/mid-math/src/wide/int/mod.rs  (updated)
//! Integer wide types — platform dispatch.

pub(crate) mod scalar;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub(crate) mod sse2;

// ── Platform dispatch ─────────────────────────────────────────────────────────

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub use sse2::{
    IMask4, IMask8, IMask16,
    i32x4, u32x4,
    i16x8, u16x8,
    i8x16, u8x16,
};

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
pub use scalar::{
    IMask4, IMask8, IMask16,
    i32x4, u32x4,
    i16x8, u16x8,
    i8x16, u8x16,
};
