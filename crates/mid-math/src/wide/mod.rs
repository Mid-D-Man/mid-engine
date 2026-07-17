// crates/mid-math/src/wide/mod.rs
//! Wide SIMD types — vertical operations on N values simultaneously.

pub mod float;
pub mod int;

// ── Integer wide re-exports ───────────────────────────────────────────────────

pub use int::{IMask4, IMask8, IMask16};

#[allow(non_camel_case_types)]
pub use int::{i32x4, u32x4, i16x8, u16x8, i8x16, u8x16};

// ── Float wide re-exports ─────────────────────────────────────────────────────

pub use float::{Mask4, Mask4LaneIter};

#[allow(non_camel_case_types)]
pub use float::f32x4;

pub use float::Vec3x4;
pub use float::QuatX4;

// Vec3x8 is only available when targeting AVX2
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    target_feature = "avx2",
))]
pub use float::Vec3x8;
