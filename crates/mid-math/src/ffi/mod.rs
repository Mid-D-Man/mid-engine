// crates/mid-math/src/ffi/mod.rs
//! C-ABI boundary layer for mid-math.

pub mod float32;
pub mod float64;
pub mod int32;
pub mod int64;
pub mod curves;
pub mod color;
pub mod helpers;
pub mod rng;
pub mod fixed;

// ── Flat re-exports ───────────────────────────────────────────────────────────

pub use float32::{CAffine3, CMat3, CMat4, CQuat, CVec2, CVec3, CVec4};
pub use float64::{CDAffine3, CDMat2, CDMat3, CDMat4, CDQuat, CDVec2, CDVec3, CDVec4};
pub use int32::{CIVec2, CIVec3, CIVec4, CUVec2, CUVec3, CUVec4};
pub use int64::{CI64Vec2, CI64Vec3, CI64Vec4, CU64Vec2, CU64Vec3, CU64Vec4};

pub use color::{CColor32, CRgb, CRgba, CHsv, CHsl, CRgbe, CYCbCr};
pub use helpers::{CDualQuat, CRotor3, CTangentFrame, CPackedTangent,
                  CSpatialVelocity, CSpatialForce, CSpatialInertia};
pub use rng::{CXorshift64State, CPcg32State};
pub use fixed::{CFixed8, CFixed12, CFixed16,
                CFixed8Vec2, CFixed12Vec2, CFixed16Vec2,
                CFixed8Vec3, CFixed12Vec3, CFixed16Vec3};

// Legacy path — anything that did `use crate::ffi::types::X` still compiles.
pub mod types {
    pub use super::float32::{CAffine3, CMat3, CMat4, CQuat, CVec2, CVec3, CVec4};
    pub use super::float64::{CDAffine3, CDMat2, CDMat3, CDMat4, CDQuat, CDVec2, CDVec3, CDVec4};
    pub use super::int32::{CIVec2, CIVec3, CIVec4, CUVec2, CUVec3, CUVec4};
    pub use super::int64::{CI64Vec2, CI64Vec3, CI64Vec4, CU64Vec2, CU64Vec3, CU64Vec4};
}
