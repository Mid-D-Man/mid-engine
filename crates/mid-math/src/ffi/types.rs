// crates/mid-math/src/ffi/types.rs
// Fix 3: Mat3 now resolves — but CMat3 has no FFI exports yet so
// keep the type but note it's unused until we add mat3 FFI functions.

use crate::{Vec2, Vec3, Vec4, Quat, Mat3, Mat4};

#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct CVec2 { pub x: f32, pub y: f32 }

impl From<Vec2> for CVec2 {
    #[inline(always)] fn from(v: Vec2) -> Self { Self { x: v.x, y: v.y } }
}
impl From<CVec2> for Vec2 {
    #[inline(always)] fn from(v: CVec2) -> Self { Vec2::new(v.x, v.y) }
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct CVec3 { pub x: f32, pub y: f32, pub z: f32, pub _pad: f32 }

impl CVec3 {
    #[inline(always)] pub fn new(x: f32, y: f32, z: f32) -> Self { Self{x,y,z,_pad:0.0} }
}
impl From<Vec3> for CVec3 {
    #[inline(always)] fn from(v: Vec3) -> Self { Self::new(v.x, v.y, v.z) }
}
impl From<CVec3> for Vec3 {
    #[inline(always)] fn from(v: CVec3) -> Self { Vec3::new(v.x, v.y, v.z) }
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct CVec4 { pub x: f32, pub y: f32, pub z: f32, pub w: f32 }

impl From<Vec4> for CVec4 {
    #[inline(always)] fn from(v: Vec4) -> Self { Self{x:v.x,y:v.y,z:v.z,w:v.w} }
}
impl From<CVec4> for Vec4 {
    #[inline(always)] fn from(v: CVec4) -> Self { Vec4::new(v.x,v.y,v.z,v.w) }
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct CQuat { pub x: f32, pub y: f32, pub z: f32, pub w: f32 }

impl From<Quat> for CQuat {
    #[inline(always)] fn from(q: Quat) -> Self { Self{x:q.x,y:q.y,z:q.z,w:q.w} }
}
impl From<CQuat> for Quat {
    #[inline(always)] fn from(q: CQuat) -> Self { Quat::new(q.x,q.y,q.z,q.w) }
}

/// C-ABI Mat3. 36 bytes, column-major.
/// Unused in exports for now — mat3 FFI ops to be added with normal matrix support.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct CMat3 { pub cols: [[f32; 3]; 3] }

impl From<Mat3> for CMat3 {
    #[inline(always)] fn from(m: Mat3) -> Self { Self { cols: m.cols } }
}
impl From<CMat3> for Mat3 {
    #[inline(always)] fn from(m: CMat3) -> Self { Mat3 { cols: m.cols } }
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C, align(16))]
pub struct CMat4 { pub cols: [[f32; 4]; 4] }

impl From<Mat4> for CMat4 {
    #[inline(always)] fn from(m: Mat4) -> Self { Self { cols: m.cols } }
}
impl From<CMat4> for Mat4 {
    #[inline(always)] fn from(m: CMat4) -> Self { Mat4 { cols: m.cols } }
}
