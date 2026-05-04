// crates/mid-math/src/bvec/mod.rs
//! Boolean vector masks — element-wise selection for geometry and intersect modules.
//!
//! Scalar only. SIMD-aligned variants (BVec3A, BVec4A) arrive with Phase 4A
//! wide vector types — they need the __m128 backing that wide types provide.
//!
//! Layout: 1-byte-aligned packed bools, no padding.
//!   BVec2 — 2 bytes, align 1
//!   BVec3 — 3 bytes, align 1
//!   BVec4 — 4 bytes, align 1

mod bvec2;
mod bvec3;
mod bvec4;

pub use bvec2::BVec2;
pub use bvec3::BVec3;
pub use bvec4::BVec4;
