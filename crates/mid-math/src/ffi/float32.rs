// crates/mid-math/src/ffi/float32.rs
//! C-ABI types and #[no_mangle] exports for f32 math types.
//!
//! CMat4 retains `cols: [[f32;4];4]` — immutable C ABI contract.
//! From impls updated for Build 8 to convert between that array layout
//! and Mat4's new Vec4-field layout.

use crate::{Affine3, Mat3, Mat4, Quat, Vec2, Vec3, Vec4};

// ═══════════════════════════════════════════════════════════════════════════
//  C types
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct CVec2 { pub x: f32, pub y: f32 }
impl From<Vec2>  for CVec2 { #[inline(always)] fn from(v: Vec2)  -> Self { Self { x: v.x, y: v.y } } }
impl From<CVec2> for Vec2  { #[inline(always)] fn from(v: CVec2) -> Self { Vec2::new(v.x, v.y) } }

#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct CVec3 { pub x: f32, pub y: f32, pub z: f32, pub _pad: f32 }
impl CVec3 {
    #[inline(always)] pub fn new(x: f32, y: f32, z: f32) -> Self { Self { x, y, z, _pad: 0.0 } }
}
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

/// CMat4: immutable C ABI type. Always `cols: [[f32;4];4]`.
///
/// Conversions explicitly bridge between this array layout and Mat4's
/// Vec4-field layout (Build 8). LLVM folds to 4 loads + 4 stores.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C, align(16))]
pub struct CMat4 { pub cols: [[f32; 4]; 4] }

impl From<Mat4> for CMat4 {
    #[inline(always)]
    fn from(m: Mat4) -> Self {
        Self {
            cols: [
                m.x_axis.to_array(),
                m.y_axis.to_array(),
                m.z_axis.to_array(),
                m.w_axis.to_array(),
            ],
        }
    }
}

impl From<CMat4> for Mat4 {
    #[inline(always)]
    fn from(m: CMat4) -> Self {
        Mat4::from_cols(m.cols[0], m.cols[1], m.cols[2], m.cols[3])
    }
}

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
            x_axis:      a.x_axis.into(),
            y_axis:      a.y_axis.into(),
            z_axis:      a.z_axis.into(),
            translation: a.translation.into(),
        }
    }
}
impl From<CAffine3> for Affine3 {
    #[inline(always)]
    fn from(a: CAffine3) -> Self {
        Self {
            x_axis:      a.x_axis.into(),
            y_axis:      a.y_axis.into(),
            z_axis:      a.z_axis.into(),
            translation: a.translation.into(),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Exports
// ═══════════════════════════════════════════════════════════════════════════

#[no_mangle] pub extern "C" fn mid_vec2_new(x:f32,y:f32)->CVec2{Vec2::new(x,y).into()}
#[no_mangle] pub extern "C" fn mid_vec2_add(a:CVec2,b:CVec2)->CVec2{(Vec2::from(a)+Vec2::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_vec2_sub(a:CVec2,b:CVec2)->CVec2{(Vec2::from(a)-Vec2::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_vec2_scale(v:CVec2,s:f32)->CVec2{(Vec2::from(v)*s).into()}
#[no_mangle] pub extern "C" fn mid_vec2_dot(a:CVec2,b:CVec2)->f32{Vec2::from(a).dot(Vec2::from(b))}
#[no_mangle] pub extern "C" fn mid_vec2_length(v:CVec2)->f32{Vec2::from(v).length()}
#[no_mangle] pub extern "C" fn mid_vec2_normalize(v:CVec2)->CVec2{Vec2::from(v).normalize().into()}
#[no_mangle] pub extern "C" fn mid_vec2_lerp(a:CVec2,b:CVec2,t:f32)->CVec2{Vec2::from(a).lerp(Vec2::from(b),t).into()}
#[no_mangle] pub extern "C" fn mid_vec2_distance(a:CVec2,b:CVec2)->f32{Vec2::from(a).distance(Vec2::from(b))}

#[no_mangle] pub extern "C" fn mid_vec3_new(x:f32,y:f32,z:f32)->CVec3{Vec3::new(x,y,z).into()}
#[no_mangle] pub extern "C" fn mid_vec3_add(a:CVec3,b:CVec3)->CVec3{(Vec3::from(a)+Vec3::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_vec3_sub(a:CVec3,b:CVec3)->CVec3{(Vec3::from(a)-Vec3::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_vec3_scale(v:CVec3,s:f32)->CVec3{(Vec3::from(v)*s).into()}
#[no_mangle] pub extern "C" fn mid_vec3_dot(a:CVec3,b:CVec3)->f32{Vec3::from(a).dot(Vec3::from(b))}
#[no_mangle] pub extern "C" fn mid_vec3_cross(a:CVec3,b:CVec3)->CVec3{Vec3::from(a).cross(Vec3::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_vec3_length(v:CVec3)->f32{Vec3::from(v).length()}
#[no_mangle] pub extern "C" fn mid_vec3_normalize(v:CVec3)->CVec3{Vec3::from(v).normalize().into()}
#[no_mangle] pub extern "C" fn mid_vec3_lerp(a:CVec3,b:CVec3,t:f32)->CVec3{Vec3::from(a).lerp(Vec3::from(b),t).into()}
#[no_mangle] pub extern "C" fn mid_vec3_distance(a:CVec3,b:CVec3)->f32{Vec3::from(a).distance(Vec3::from(b))}
#[no_mangle] pub extern "C" fn mid_vec3_reflect(v:CVec3,n:CVec3)->CVec3{Vec3::from(v).reflect(Vec3::from(n)).into()}

#[no_mangle] pub extern "C" fn mid_vec4_new(x:f32,y:f32,z:f32,w:f32)->CVec4{Vec4::new(x,y,z,w).into()}
#[no_mangle] pub extern "C" fn mid_vec4_add(a:CVec4,b:CVec4)->CVec4{(Vec4::from(a)+Vec4::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_vec4_dot(a:CVec4,b:CVec4)->f32{Vec4::from(a).dot(Vec4::from(b))}
#[no_mangle] pub extern "C" fn mid_vec4_normalize(v:CVec4)->CVec4{Vec4::from(v).normalize().into()}
#[no_mangle] pub extern "C" fn mid_vec4_lerp(a:CVec4,b:CVec4,t:f32)->CVec4{Vec4::from(a).lerp(Vec4::from(b),t).into()}

#[no_mangle] pub extern "C" fn mid_quat_identity()->CQuat{Quat::IDENTITY.into()}
#[no_mangle] pub extern "C" fn mid_quat_new(x:f32,y:f32,z:f32,w:f32)->CQuat{Quat::new(x,y,z,w).into()}
#[no_mangle] pub extern "C" fn mid_quat_from_axis_angle(axis:CVec3,angle_rad:f32)->CQuat{
    Quat::from_axis_angle(Vec3::from(axis),angle_rad).into()
}
#[no_mangle] pub extern "C" fn mid_quat_from_euler(roll:f32,pitch:f32,yaw:f32)->CQuat{
    Quat::from_euler(roll,pitch,yaw).into()
}
#[no_mangle] pub extern "C" fn mid_quat_mul(a:CQuat,b:CQuat)->CQuat{(Quat::from(a)*Quat::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_quat_normalize(q:CQuat)->CQuat{Quat::from(q).normalize().into()}
#[no_mangle] pub extern "C" fn mid_quat_conjugate(q:CQuat)->CQuat{Quat::from(q).conjugate().into()}
#[no_mangle] pub extern "C" fn mid_quat_rotate(q:CQuat,v:CVec3)->CVec3{Quat::from(q).rotate(Vec3::from(v)).into()}
#[no_mangle] pub extern "C" fn mid_quat_slerp(a:CQuat,b:CQuat,t:f32)->CQuat{Quat::from(a).slerp(Quat::from(b),t).into()}
#[no_mangle] pub extern "C" fn mid_quat_to_mat4(q:CQuat)->CMat4{Quat::from(q).to_mat4().into()}

#[no_mangle] pub extern "C" fn mid_mat4_identity()->CMat4{Mat4::IDENTITY.into()}
#[no_mangle] pub extern "C" fn mid_mat4_from_translation(t:CVec3)->CMat4{Mat4::from_translation(Vec3::from(t)).into()}
#[no_mangle] pub extern "C" fn mid_mat4_from_scale(s:CVec3)->CMat4{Mat4::from_scale(Vec3::from(s)).into()}
#[no_mangle] pub extern "C" fn mid_mat4_from_rotation(q:CQuat)->CMat4{Mat4::from_rotation(Quat::from(q)).into()}
#[no_mangle] pub extern "C" fn mid_mat4_from_trs(t:CVec3,r:CQuat,s:CVec3)->CMat4{
    Mat4::from_trs(Vec3::from(t),Quat::from(r),Vec3::from(s)).into()
}
#[no_mangle] pub extern "C" fn mid_mat4_mul(a:CMat4,b:CMat4)->CMat4{(Mat4::from(a)*Mat4::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_mat4_transpose(m:CMat4)->CMat4{Mat4::from(m).transpose().into()}
#[no_mangle] pub extern "C" fn mid_mat4_transform_point(m:CMat4,p:CVec3)->CVec3{Mat4::from(m).transform_point(Vec3::from(p)).into()}
#[no_mangle] pub extern "C" fn mid_mat4_transform_vector(m:CMat4,v:CVec3)->CVec3{Mat4::from(m).transform_vector(Vec3::from(v)).into()}
#[no_mangle] pub extern "C" fn mid_mat4_look_at_rh(eye:CVec3,center:CVec3,up:CVec3)->CMat4{
    Mat4::look_at_rh(Vec3::from(eye),Vec3::from(center),Vec3::from(up)).into()
}
#[no_mangle] pub extern "C" fn mid_mat4_perspective_rh(fov_y:f32,aspect:f32,near:f32,far:f32)->CMat4{
    Mat4::perspective_rh(fov_y,aspect,near,far).into()
}
#[no_mangle] pub extern "C" fn mid_mat4_ortho_rh(l:f32,r:f32,b:f32,t:f32,n:f32,f:f32)->CMat4{
    Mat4::ortho_rh(l,r,b,t,n,f).into()
}
#[no_mangle] pub extern "C" fn mid_mat4_inverse(m:CMat4)->CMat4{
    Mat4::from(m).inverse().unwrap_or(Mat4::IDENTITY).into()
}

#[no_mangle] pub extern "C" fn mid_affine3_identity()->CAffine3{Affine3::IDENTITY.into()}
#[no_mangle] pub extern "C" fn mid_affine3_from_trs(t:CVec3,r:CQuat,s:CVec3)->CAffine3{
    Affine3::from_trs(Vec3::from(t),Quat::from(r),Vec3::from(s)).into()
}
#[no_mangle] pub extern "C" fn mid_affine3_from_translation(t:CVec3)->CAffine3{Affine3::from_translation(Vec3::from(t)).into()}
#[no_mangle] pub extern "C" fn mid_affine3_from_rotation(q:CQuat)->CAffine3{Affine3::from_rotation(Quat::from(q)).into()}
#[no_mangle] pub extern "C" fn mid_affine3_from_scale(s:CVec3)->CAffine3{Affine3::from_scale(Vec3::from(s)).into()}
#[no_mangle] pub extern "C" fn mid_affine3_from_mat4(m:CMat4)->CAffine3{Affine3::from_mat4(Mat4::from(m)).into()}
#[no_mangle] pub extern "C" fn mid_affine3_to_mat4(a:CAffine3)->CMat4{Affine3::from(a).to_mat4().into()}
#[no_mangle] pub extern "C" fn mid_affine3_mul(a:CAffine3,b:CAffine3)->CAffine3{(Affine3::from(a)*Affine3::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_affine3_inverse(a:CAffine3)->CAffine3{Affine3::from(a).inverse().into()}
#[no_mangle] pub extern "C" fn mid_affine3_transform_point(a:CAffine3,p:CVec3)->CVec3{Affine3::from(a).transform_point(Vec3::from(p)).into()}
#[no_mangle] pub extern "C" fn mid_affine3_transform_vector(a:CAffine3,v:CVec3)->CVec3{Affine3::from(a).transform_vector(Vec3::from(v)).into()}
