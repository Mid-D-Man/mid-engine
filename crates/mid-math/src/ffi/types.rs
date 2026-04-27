// crates/mid-math/src/ffi/types.rs
//! C-compatible types for the FFI boundary.
//!
//! These are the ONLY types that cross the FFI boundary. Internal code
//! uses the SIMD-backed types (Vec3, Vec4, Quat, Mat4) directly.
//!
//! All types are #[repr(C)] — safe to pass by value or pointer from
//! C, C++, C#/Unity P/Invoke, GDScript, or any language with a C FFI.
//!
//! Naming convention: C prefix = "C-ABI type"
//!   CVec2, CVec3, CVec4, CQuat, CMat3, CMat4
//!
//! Conversions between internal SIMD types and C types are zero-cost —
//! LLVM sees through the field copies on every platform we target.

use crate::{Vec2, Vec3, Vec4, Quat, Mat3, Mat4};

// ── CVec2 ─────────────────────────────────────────────────────────────────────

/// C-ABI Vec2. 8 bytes, no padding.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct CVec2 {
    pub x: f32,
    pub y: f32,
}

impl From<Vec2> for CVec2 {
    #[inline(always)]
    fn from(v: Vec2) -> Self { Self { x: v.x, y: v.y } }
}
impl From<CVec2> for Vec2 {
    #[inline(always)]
    fn from(v: CVec2) -> Self { Vec2::new(v.x, v.y) }
}

// ── CVec3 ─────────────────────────────────────────────────────────────────────

/// C-ABI Vec3. 16 bytes (12 data + 4 explicit pad for 16-byte alignment).
///
/// The `_pad` field must be kept as 0.0 by all C callers.
/// C struct: `typedef struct { float x, y, z, _pad; } MidVec3;`
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct CVec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub _pad: f32,
}

impl CVec3 {
    #[inline(always)]
    pub fn new(x: f32, y: f32, z: f32) -> Self { Self { x, y, z, _pad: 0.0 } }
}

impl From<Vec3> for CVec3 {
    #[inline(always)]
    fn from(v: Vec3) -> Self { Self::new(v.x, v.y, v.z) }
}
impl From<CVec3> for Vec3 {
    #[inline(always)]
    fn from(v: CVec3) -> Self { Vec3::new(v.x, v.y, v.z) }
}

// ── CVec4 ─────────────────────────────────────────────────────────────────────

/// C-ABI Vec4. 16 bytes, 16-byte aligned.
/// C struct: `typedef struct { float x, y, z, w; } MidVec4;`
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct CVec4 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

impl From<Vec4> for CVec4 {
    #[inline(always)]
    fn from(v: Vec4) -> Self { Self { x: v.x, y: v.y, z: v.z, w: v.w } }
}
impl From<CVec4> for Vec4 {
    #[inline(always)]
    fn from(v: CVec4) -> Self { Vec4::new(v.x, v.y, v.z, v.w) }
}

// ── CQuat ─────────────────────────────────────────────────────────────────────

/// C-ABI Quaternion. 16 bytes. Layout: (x, y, z, w).
/// C struct: `typedef struct { float x, y, z, w; } MidQuat;`
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct CQuat {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

impl From<Quat> for CQuat {
    #[inline(always)]
    fn from(q: Quat) -> Self { Self { x: q.x, y: q.y, z: q.z, w: q.w } }
}
impl From<CQuat> for Quat {
    #[inline(always)]
    fn from(q: CQuat) -> Self { Quat::new(q.x, q.y, q.z, q.w) }
}

// ── CMat3 ─────────────────────────────────────────────────────────────────────

/// C-ABI Mat3. 36 bytes, column-major.
/// C struct: `typedef struct { float cols[3][3]; } MidMat3;`
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct CMat3 {
    pub cols: [[f32; 3]; 3],
}

impl From<Mat3> for CMat3 {
    #[inline(always)]
    fn from(m: Mat3) -> Self { Self { cols: m.cols } }
}
impl From<CMat3> for Mat3 {
    #[inline(always)]
    fn from(m: CMat3) -> Self { Mat3 { cols: m.cols } }
}

// ── CMat4 ─────────────────────────────────────────────────────────────────────

/// C-ABI Mat4. 64 bytes, 16-byte aligned, column-major.
/// C struct: `typedef struct { float cols[4][4]; } MidMat4;`
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C, align(16))]
pub struct CMat4 {
    pub cols: [[f32; 4]; 4],
}

impl From<Mat4> for CMat4 {
    #[inline(always)]
    fn from(m: Mat4) -> Self { Self { cols: m.cols } }
}
impl From<CMat4> for Mat4 {
    #[inline(always)]
    fn from(m: CMat4) -> Self { Mat4 { cols: m.cols } }
}
