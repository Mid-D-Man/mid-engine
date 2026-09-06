// crates/mid-math/src/ffi/mod.rs
//! C-ABI boundary layer for mid-math.

pub mod float32;
pub mod float64;
pub mod int32;
pub mod int64;
pub mod int8;
pub mod int16;
pub mod curves;
pub mod color;
pub mod helpers;
pub mod rng;
pub mod fixed;
pub mod noise;
pub mod camera;
pub mod geom;
pub mod wide_int;
pub mod wide_float;
pub mod storage;
pub mod bvec;

// AVX2-only additive wide types (i32x8 and friends, f32x8, Vec3x8) get
// their own FFI modules, gated to the architectures they exist on at
// all -- the types themselves always compile on x86/x86_64 now and
// dispatch to AVX2 or a portable fallback internally at runtime (see
// each module's own header), but they simply don't exist elsewhere
// (aarch64, wasm32/64), so the module declaration itself is gated the
// same way the underlying types already are.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub mod wide_int_avx2;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub mod wide_float_avx2;

// "FFI option 3" -- width-hiding batch functions. NOT architecture-gated
// like the two modules above: that's the whole point (a C caller gets
// one function name that works everywhere, with the AVX2 fast path
// used opportunistically inside it on x86/x86_64 only). See its own
// header for the full design.
pub mod wide_batch;

// ── Flat re-exports ───────────────────────────────────────────────────────────

pub use float32::{CAffine3, CMat3, CMat4, CQuat, CVec2, CVec3, CVec4};
pub use float64::{CDAffine3, CDMat2, CDMat3, CDMat4, CDQuat, CDVec2, CDVec3, CDVec4};
pub use int32::{CIVec2, CIVec3, CIVec4, CUVec2, CUVec3, CUVec4};
pub use int64::{CI64Vec2, CI64Vec3, CI64Vec4, CU64Vec2, CU64Vec3, CU64Vec4};
pub use int8::{CI8Vec2, CI8Vec3, CI8Vec4, CU8Vec2, CU8Vec3, CU8Vec4};
pub use int16::{CI16Vec2, CI16Vec3, CI16Vec4, CU16Vec2, CU16Vec3, CU16Vec4};
pub use color::{CColor32, CRgb, CRgba, CHsv, CHsl, CRgbe, CYCbCr};
pub use helpers::{CDualQuat, CRotor3, CTangentFrame, CPackedTangent,
                  CSpatialVelocity, CSpatialForce, CSpatialInertia};
pub use rng::{CXorshift64State, CPcg32State};
pub use fixed::{CFixed8, CFixed12, CFixed16,
                CFixed8Vec2, CFixed12Vec2, CFixed16Vec2,
                CFixed8Vec3, CFixed12Vec3, CFixed16Vec3};
pub use camera::{CFrustum, CVisibility, CPerspectiveParams};
pub use geom::{CBarycentricCoords, CTriangle2, CTriangle3, CCircumcircle, CRayHit3};
pub use wide_int::{CI32x4, CU32x4, CI16x8, CU16x8, CI8x16, CU8x16};
pub use wide_float::{Cf32x4, CVec3x4, CQuatX4};
// storage: no new C struct types -- exports use raw u8/u16/u32/u64/[u64;2]/[u64;4]
// directly (see ffi/storage.rs's header for why). bvec: no new C struct types
// either -- BVec2/3/4 are already crate::BVec2/3/4, re-exported from the crate
// root already, not redefined here.

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub use wide_int_avx2::{CI32x8, CU32x8, CI16x16, CU16x16, CI8x32, CU8x32};
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub use wide_float_avx2::{Cf32x8, CVec3x8};
// wide_batch: no new C struct types either -- exports operate on raw
// scalar pointers (i32/u32/i16/u16/i8/u8/f32) directly, same reasoning
// as storage/bvec above.

// Legacy path — anything that did `use crate::ffi::types::X` still compiles.
pub mod types {
    pub use super::float32::{CAffine3, CMat3, CMat4, CQuat, CVec2, CVec3, CVec4};
    pub use super::float64::{CDAffine3, CDMat2, CDMat3, CDMat4, CDQuat, CDVec2, CDVec3, CDVec4};
    pub use super::int32::{CIVec2, CIVec3, CIVec4, CUVec2, CUVec3, CUVec4};
    pub use super::int64::{CI64Vec2, CI64Vec3, CI64Vec4, CU64Vec2, CU64Vec3, CU64Vec4};
}
