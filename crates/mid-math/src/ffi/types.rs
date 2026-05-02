// crates/mid-math/src/ffi/types.rs

use crate::{Affine3, Mat3, Mat4, Quat, Vec2, Vec3, Vec4};

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

/// C-ABI Affine3. 64 bytes, 16-byte aligned. Four CVec3 (each 16 bytes with _pad).
///
/// Layout mirrors Affine3 exactly — memcpy between them is safe.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C, align(16))]
pub struct CAffine3 {
    pub x_axis:      CVec3,
    pub y_axis:      CVec3,
    pub z_axis:      CVec3,
    pub translation: CVec3,
}

impl CAffine3 {
    #[inline(always)]
    pub fn new(x_axis: CVec3, y_axis: CVec3, z_axis: CVec3, translation: CVec3) -> Self {
        Self { x_axis, y_axis, z_axis, translation }
    }
}

impl From<Affine3> for CAffine3 {
    #[inline(always)]
    fn from(a: Affine3) -> Self {
        Self {
            x_axis:      CVec3::from(a.x_axis),
            y_axis:      CVec3::from(a.y_axis),
            z_axis:      CVec3::from(a.z_axis),
            translation: CVec3::from(a.translation),
        }
    }
}

impl From<CAffine3> for Affine3 {
    #[inline(always)]
    fn from(a: CAffine3) -> Self {
        Self {
            x_axis:      Vec3::from(a.x_axis),
            y_axis:      Vec3::from(a.y_axis),
            z_axis:      Vec3::from(a.z_axis),
            translation: Vec3::from(a.translation),
        }
    }
}
