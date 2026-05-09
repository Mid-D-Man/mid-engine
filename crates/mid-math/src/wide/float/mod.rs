// crates/mid-math/src/wide/float/mod.rs
pub(crate) mod scalar;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub(crate) mod sse2;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub use sse2::{Mask4, f32x4, Vec3x4};

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
pub use scalar::{Mask4, f32x4, Vec3x4};
