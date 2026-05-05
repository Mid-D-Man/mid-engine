// crates/mid-math/src/ffi/mod.rs

pub mod types;
pub mod exports;

pub use types::{
    // f32
    CAffine3, CMat3, CMat4, CQuat, CVec2, CVec3, CVec4,
    // f64
    CDAffine3, CDMat2, CDMat3, CDMat4, CDQuat, CDVec2, CDVec3, CDVec4,
    // i32
    CIVec2, CIVec3, CIVec4,
    // u32
    CUVec2, CUVec3, CUVec4,
};
