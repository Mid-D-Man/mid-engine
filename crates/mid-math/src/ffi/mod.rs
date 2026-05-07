// crates/mid-math/src/ffi/mod.rs
//! C-ABI boundary layer for mid-math.
//!
//! Split by numeric domain — no ABI change, same #[no_mangle] symbols,
//! same #[repr(C)] layouts. Each domain file contains both its types
//! and its export functions.
//!
//! Domain files:
//!   float32.rs  — CVec2, CVec3, CVec4, CQuat, CMat3, CMat4, CAffine3
//!   float64.rs  — CDVec2..4, CDQuat, CDMat2..4, CDAffine3
//!   int32.rs    — CIVec2..4, CUVec2..4
//!   int64.rs    — CI64Vec2..4, CU64Vec2..4
//!   geometry.rs — CTransform, CAABB, CSphere (Phase 3C stub)

pub mod float32;
pub mod float64;
pub mod int32;
pub mod int64;
pub mod geometry;

// ── Flat re-exports so existing code using crate::ffi::types::X still works ──
// (callers can use crate::ffi::CVec3 directly after this)

pub use float32::{CAffine3, CMat3, CMat4, CQuat, CVec2, CVec3, CVec4};
pub use float64::{CDAffine3, CDMat2, CDMat3, CDMat4, CDQuat, CDVec2, CDVec3, CDVec4};
pub use int32::{CIVec2, CIVec3, CIVec4, CUVec2, CUVec3, CUVec4};
pub use int64::{CI64Vec2, CI64Vec3, CI64Vec4, CU64Vec2, CU64Vec3, CU64Vec4};

// Legacy path: anything that did `use crate::ffi::types::X` still compiles
// via this re-export module alias.
pub mod types {
    pub use super::float32::{CAffine3, CMat3, CMat4, CQuat, CVec2, CVec3, CVec4};
    pub use super::float64::{CDAffine3, CDMat2, CDMat3, CDMat4, CDQuat, CDVec2, CDVec3, CDVec4};
    pub use super::int32::{CIVec2, CIVec3, CIVec4, CUVec2, CUVec3, CUVec4};
    pub use super::int64::{CI64Vec2, CI64Vec3, CI64Vec4, CU64Vec2, CU64Vec3, CU64Vec4};
}
