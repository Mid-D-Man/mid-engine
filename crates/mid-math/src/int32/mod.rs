// crates/mid-math/src/int/mod.rs
//! Integer vector types — i32 and u32 only.
//!
//! Used for grid coordinates, voxel indices, screen pixel positions,
//! entity IDs and any integer domain the engine needs.
//!
//! All types are scalar — no SIMD for integers yet.
//! AVX2 packed-int wide types are Phase 4A alongside Vec3x4/Vec3x8.
//!
//! Layout (no padding unlike f32 Vec3):
//!   IVec2 / UVec2 — 8  bytes, align 4
//!   IVec3 / UVec3 — 12 bytes, align 4   ← no padding, intentional
//!   IVec4 / UVec4 — 16 bytes, align 4

mod ivec2;
mod ivec3;
mod ivec4;
mod uvec2;
mod uvec3;
mod uvec4;

pub use ivec2::IVec2;
pub use ivec3::IVec3;
pub use ivec4::IVec4;
pub use uvec2::UVec2;
pub use uvec3::UVec3;
pub use uvec4::UVec4;
