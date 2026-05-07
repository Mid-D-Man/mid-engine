// crates/mid-math/src/ffi/mod.rs
//! C-ABI boundary layer for mid-math.
//!
//! PLANNED REFACTOR (Phase 3C prep): split into domain submodules:
//!   mod float32  → types + exports for f32 types
//!   mod float64  → types + exports for f64 types
//!   mod int32    → types + exports for i32/u32 types
//!   mod int64    → types + exports for i64/u64 types
//!   mod geometry → types + exports for Transform, AABB, etc. (Phase 3C)
//!
//! The refactor requires no ABI changes — purely file reorganisation.
//! Planned before geometry module lands to keep exports.rs manageable.

pub mod types;
pub mod exports;

pub use types::{
    // f32
    CAffine3, CMat3, CMat4, CQuat, CVec2, CVec3, CVec4,
    // f64
    CDAffine3, CDMat2, CDMat3, CDMat4, CDQuat, CDVec2, CDVec3, CDVec4,
    // i32 / u32
    CIVec2, CIVec3, CIVec4,
    CUVec2, CUVec3, CUVec4,
    // i64 / u64
    CI64Vec2, CI64Vec3, CI64Vec4,
    CU64Vec2, CU64Vec3, CU64Vec4,
};
