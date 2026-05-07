// crates/mid-math/src/ffi/types.rs
//! C-ABI types for all mid-math vector/matrix types.
//!
//! PLANNED REFACTOR (Phase 3C prep): split this file into domain modules:
//!   ffi/float32_types.rs  — CVec2, CVec3, CVec4, CQuat, CMat3, CMat4, CAffine3
//!   ffi/float64_types.rs  — CDVec2..4, CDQuat, CDMat2..4, CDAffine3
//!   ffi/int32_types.rs    — CIVec2..4, CUVec2..4
//!   ffi/int64_types.rs    — CI64Vec2..4, CU64Vec2..4
//!   ffi/geometry_types.rs — CTransform, CAABB, CSphere (Phase 3C)
//!
//! For now everything lives here. The refactor is purely cosmetic — no ABI
//! change, just file organisation. Scheduled for before geometry module lands.

use crate::{
    Affine3, DAffine3, DMat2, DMat3, DMat4, DQuat, DVec2, DVec3, DVec4,
    Mat3, Mat4, Quat, Vec2, Vec3, Vec4,
    IVec2, IVec3, IVec4, UVec2, UVec3, UVec4,
    I64Vec2, I64Vec3, I64Vec4, U64Vec2, U64Vec3, U64Vec4,
};

// ═══════════════════════════════════════════════════════════════════════════
//  f32 C types
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct CVec2 { pub x: f32, pub y: f32 }
impl From<Vec2>  for CVec2 { #[inline(always)] fn from(v: Vec2)  -> Self { Self { x: v.x, y: v.y } } }
impl From<CVec2> for Vec2  { #[inline(always)] fn from(v: CVec2) -> Self { Vec2::new(v.x, v.y) } }

#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct CVec3 { pub x: f32, pub y: f32, pub z: f32, pub _pad: f32 }
impl CVec3 { #[inline(always)] pub fn new(x: f32, y: f32, z: f32) -> Self { Self { x, y, z, _pad: 0.0 } } }
impl From<Vec3>  for CVec3 { #[inline(always)] fn from(v: Vec3)  -> Self { Self::new(v.x, v.y, v.z) } }
impl From<CVec3> for Vec3  { #[inline(always)] fn from(v: CVec3) -> Self { Vec3::new(v.x, v.y, v.z) } }

#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct CVec4 { pub x: f32, pub y: f32, pub z: f32, pub w: f32 }
impl From<Vec4>  for CVec4 { #[inline(always)] fn from(v: Vec4)  -> Self { Self { x: v.x, y: v.y, z: v.z, w: v.w } } }
impl From<CVec4> for Vec4  { #[inline(always)] fn from(v: CVec4) -> Self { Vec4::new(v.x, v.y, v.z, v.w) } }

#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct CQuat { pub x: f32, pub y: f32, pub z: f32, pub w: f32 }
impl From<Quat>  for CQuat { #[inline(always)] fn from(q: Quat)  -> Self { Self { x: q.x, y: q.y, z: q.z, w: q.w } } }
impl From<CQuat> for Quat  { #[inline(always)] fn from(q: CQuat) -> Self { Quat::new(q.x, q.y, q.z, q.w) } }

#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct CMat3 { pub cols: [[f32; 3]; 3] }
impl From<Mat3>  for CMat3 { #[inline(always)] fn from(m: Mat3)  -> Self { Self { cols: m.cols } } }
impl From<CMat3> for Mat3  { #[inline(always)] fn from(m: CMat3) -> Self { Mat3 { cols: m.cols } } }

#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C, align(16))]
pub struct CMat4 { pub cols: [[f32; 4]; 4] }
impl From<Mat4>  for CMat4 { #[inline(always)] fn from(m: Mat4)  -> Self { Self { cols: m.cols } } }
impl From<CMat4> for Mat4  { #[inline(always)] fn from(m: CMat4) -> Self { Mat4 { cols: m.cols } } }

#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C, align(16))]
pub struct CAffine3 {
    pub x_axis: CVec3, pub y_axis: CVec3,
    pub z_axis: CVec3, pub translation: CVec3,
}
impl CAffine3 {
    #[inline(always)]
    pub fn new(x_axis: CVec3, y_axis: CVec3, z_axis: CVec3, translation: CVec3) -> Self {
        Self { x_axis, y_axis, z_axis, translation }
    }
}
impl From<Affine3>  for CAffine3 {
    #[inline(always)] fn from(a: Affine3) -> Self {
        Self { x_axis: a.x_axis.into(), y_axis: a.y_axis.into(),
               z_axis: a.z_axis.into(), translation: a.translation.into() }
    }
}
impl From<CAffine3> for Affine3 {
    #[inline(always)] fn from(a: CAffine3) -> Self {
        Self { x_axis: a.x_axis.into(), y_axis: a.y_axis.into(),
               z_axis: a.z_axis.into(), translation: a.translation.into() }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  f64 C types
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C, align(16))]
pub struct CDVec2 { pub x: f64, pub y: f64 }
impl CDVec2 { #[inline(always)] pub fn new(x: f64, y: f64) -> Self { Self { x, y } } }
impl From<DVec2>  for CDVec2 { #[inline(always)] fn from(v: DVec2)  -> Self { Self::new(v.x, v.y) } }
impl From<CDVec2> for DVec2  { #[inline(always)] fn from(v: CDVec2) -> Self { DVec2::new(v.x, v.y) } }

#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C, align(8))]
pub struct CDVec3 { pub x: f64, pub y: f64, pub z: f64 }
impl CDVec3 { #[inline(always)] pub fn new(x: f64, y: f64, z: f64) -> Self { Self { x, y, z } } }
impl From<DVec3>  for CDVec3 { #[inline(always)] fn from(v: DVec3)  -> Self { Self::new(v.x, v.y, v.z) } }
impl From<CDVec3> for DVec3  { #[inline(always)] fn from(v: CDVec3) -> Self { DVec3::new(v.x, v.y, v.z) } }

#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C, align(32))]
pub struct CDVec4 { pub x: f64, pub y: f64, pub z: f64, pub w: f64 }
impl From<DVec4>  for CDVec4 { #[inline(always)] fn from(v: DVec4)  -> Self { Self { x: v.x, y: v.y, z: v.z, w: v.w } } }
impl From<CDVec4> for DVec4  { #[inline(always)] fn from(v: CDVec4) -> Self { DVec4::new(v.x, v.y, v.z, v.w) } }

#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C, align(32))]
pub struct CDQuat { pub x: f64, pub y: f64, pub z: f64, pub w: f64 }
impl From<DQuat>  for CDQuat { #[inline(always)] fn from(q: DQuat)  -> Self { Self { x: q.x, y: q.y, z: q.z, w: q.w } } }
impl From<CDQuat> for DQuat  { #[inline(always)] fn from(q: CDQuat) -> Self { DQuat::new(q.x, q.y, q.z, q.w) } }

#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C, align(16))]
pub struct CDMat2 { pub x_axis: CDVec2, pub y_axis: CDVec2 }
impl From<DMat2>  for CDMat2 {
    #[inline(always)] fn from(m: DMat2) -> Self { Self { x_axis: m.x_axis.into(), y_axis: m.y_axis.into() } }
}
impl From<CDMat2> for DMat2 {
    #[inline(always)] fn from(m: CDMat2) -> Self { DMat2::from_cols(m.x_axis.into(), m.y_axis.into()) }
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct CDMat3 { pub cols: [[f64; 3]; 3] }
impl From<DMat3>  for CDMat3 { #[inline(always)] fn from(m: DMat3)  -> Self { Self { cols: m.cols } } }
impl From<CDMat3> for DMat3  { #[inline(always)] fn from(m: CDMat3) -> Self { DMat3 { cols: m.cols } } }

#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C, align(32))]
pub struct CDMat4 { pub cols: [[f64; 4]; 4] }
impl From<DMat4>  for CDMat4 { #[inline(always)] fn from(m: DMat4)  -> Self { Self { cols: m.cols } } }
impl From<CDMat4> for DMat4  { #[inline(always)] fn from(m: CDMat4) -> Self { DMat4 { cols: m.cols } } }

#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C, align(8))]
pub struct CDAffine3 {
    pub x_axis: CDVec3, pub y_axis: CDVec3,
    pub z_axis: CDVec3, pub translation: CDVec3,
}
impl CDAffine3 {
    #[inline(always)]
    pub fn new(x: CDVec3, y: CDVec3, z: CDVec3, t: CDVec3) -> Self {
        Self { x_axis: x, y_axis: y, z_axis: z, translation: t }
    }
}
impl From<DAffine3>  for CDAffine3 {
    #[inline(always)] fn from(a: DAffine3) -> Self {
        Self { x_axis: a.x_axis.into(), y_axis: a.y_axis.into(),
               z_axis: a.z_axis.into(), translation: a.translation.into() }
    }
}
impl From<CDAffine3> for DAffine3 {
    #[inline(always)] fn from(a: CDAffine3) -> Self {
        Self { x_axis: a.x_axis.into(), y_axis: a.y_axis.into(),
               z_axis: a.z_axis.into(), translation: a.translation.into() }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  i32 / u32 C types
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Copy, PartialEq, Eq)] #[repr(C)]
pub struct CIVec2 { pub x: i32, pub y: i32 }
impl From<IVec2>  for CIVec2 { #[inline(always)] fn from(v: IVec2)  -> Self { Self { x: v.x, y: v.y } } }
impl From<CIVec2> for IVec2  { #[inline(always)] fn from(v: CIVec2) -> Self { IVec2::new(v.x, v.y) } }

#[derive(Debug, Clone, Copy, PartialEq, Eq)] #[repr(C)]
pub struct CIVec3 { pub x: i32, pub y: i32, pub z: i32 }
impl From<IVec3>  for CIVec3 { #[inline(always)] fn from(v: IVec3)  -> Self { Self { x: v.x, y: v.y, z: v.z } } }
impl From<CIVec3> for IVec3  { #[inline(always)] fn from(v: CIVec3) -> Self { IVec3::new(v.x, v.y, v.z) } }

#[derive(Debug, Clone, Copy, PartialEq, Eq)] #[repr(C)]
pub struct CIVec4 { pub x: i32, pub y: i32, pub z: i32, pub w: i32 }
impl From<IVec4>  for CIVec4 { #[inline(always)] fn from(v: IVec4)  -> Self { Self { x: v.x, y: v.y, z: v.z, w: v.w } } }
impl From<CIVec4> for IVec4  { #[inline(always)] fn from(v: CIVec4) -> Self { IVec4::new(v.x, v.y, v.z, v.w) } }

#[derive(Debug, Clone, Copy, PartialEq, Eq)] #[repr(C)]
pub struct CUVec2 { pub x: u32, pub y: u32 }
impl From<UVec2>  for CUVec2 { #[inline(always)] fn from(v: UVec2)  -> Self { Self { x: v.x, y: v.y } } }
impl From<CUVec2> for UVec2  { #[inline(always)] fn from(v: CUVec2) -> Self { UVec2::new(v.x, v.y) } }

#[derive(Debug, Clone, Copy, PartialEq, Eq)] #[repr(C)]
pub struct CUVec3 { pub x: u32, pub y: u32, pub z: u32 }
impl From<UVec3>  for CUVec3 { #[inline(always)] fn from(v: UVec3)  -> Self { Self { x: v.x, y: v.y, z: v.z } } }
impl From<CUVec3> for UVec3  { #[inline(always)] fn from(v: CUVec3) -> Self { UVec3::new(v.x, v.y, v.z) } }

#[derive(Debug, Clone, Copy, PartialEq, Eq)] #[repr(C)]
pub struct CUVec4 { pub x: u32, pub y: u32, pub z: u32, pub w: u32 }
impl From<UVec4>  for CUVec4 { #[inline(always)] fn from(v: UVec4)  -> Self { Self { x: v.x, y: v.y, z: v.z, w: v.w } } }
impl From<CUVec4> for UVec4  { #[inline(always)] fn from(v: CUVec4) -> Self { UVec4::new(v.x, v.y, v.z, v.w) } }

// ═══════════════════════════════════════════════════════════════════════════
//  i64 / u64 C types
// ═══════════════════════════════════════════════════════════════════════════

/// C-ABI I64Vec2. 16 bytes, align 8.
#[derive(Debug, Clone, Copy, PartialEq, Eq)] #[repr(C, align(8))]
pub struct CI64Vec2 { pub x: i64, pub y: i64 }
impl From<I64Vec2>  for CI64Vec2 { #[inline(always)] fn from(v: I64Vec2)  -> Self { Self { x: v.x, y: v.y } } }
impl From<CI64Vec2> for I64Vec2  { #[inline(always)] fn from(v: CI64Vec2) -> Self { I64Vec2::new(v.x, v.y) } }

/// C-ABI I64Vec3. 24 bytes, align 8. No padding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)] #[repr(C, align(8))]
pub struct CI64Vec3 { pub x: i64, pub y: i64, pub z: i64 }
impl From<I64Vec3>  for CI64Vec3 { #[inline(always)] fn from(v: I64Vec3)  -> Self { Self { x: v.x, y: v.y, z: v.z } } }
impl From<CI64Vec3> for I64Vec3  { #[inline(always)] fn from(v: CI64Vec3) -> Self { I64Vec3::new(v.x, v.y, v.z) } }

/// C-ABI I64Vec4. 32 bytes, align 8.
#[derive(Debug, Clone, Copy, PartialEq, Eq)] #[repr(C, align(8))]
pub struct CI64Vec4 { pub x: i64, pub y: i64, pub z: i64, pub w: i64 }
impl From<I64Vec4>  for CI64Vec4 { #[inline(always)] fn from(v: I64Vec4)  -> Self { Self { x: v.x, y: v.y, z: v.z, w: v.w } } }
impl From<CI64Vec4> for I64Vec4  { #[inline(always)] fn from(v: CI64Vec4) -> Self { I64Vec4::new(v.x, v.y, v.z, v.w) } }

/// C-ABI U64Vec2. 16 bytes, align 8.
#[derive(Debug, Clone, Copy, PartialEq, Eq)] #[repr(C, align(8))]
pub struct CU64Vec2 { pub x: u64, pub y: u64 }
impl From<U64Vec2>  for CU64Vec2 { #[inline(always)] fn from(v: U64Vec2)  -> Self { Self { x: v.x, y: v.y } } }
impl From<CU64Vec2> for U64Vec2  { #[inline(always)] fn from(v: CU64Vec2) -> Self { U64Vec2::new(v.x, v.y) } }

/// C-ABI U64Vec3. 24 bytes, align 8. No padding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)] #[repr(C, align(8))]
pub struct CU64Vec3 { pub x: u64, pub y: u64, pub z: u64 }
impl From<U64Vec3>  for CU64Vec3 { #[inline(always)] fn from(v: U64Vec3)  -> Self { Self { x: v.x, y: v.y, z: v.z } } }
impl From<CU64Vec3> for U64Vec3  { #[inline(always)] fn from(v: CU64Vec3) -> Self { U64Vec3::new(v.x, v.y, v.z) } }

/// C-ABI U64Vec4. 32 bytes, align 8.
#[derive(Debug, Clone, Copy, PartialEq, Eq)] #[repr(C, align(8))]
pub struct CU64Vec4 { pub x: u64, pub y: u64, pub z: u64, pub w: u64 }
impl From<U64Vec4>  for CU64Vec4 { #[inline(always)] fn from(v: U64Vec4)  -> Self { Self { x: v.x, y: v.y, z: v.z, w: v.w } } }
impl From<CU64Vec4> for U64Vec4  { #[inline(always)] fn from(v: CU64Vec4) -> Self { U64Vec4::new(v.x, v.y, v.z, v.w) } }
