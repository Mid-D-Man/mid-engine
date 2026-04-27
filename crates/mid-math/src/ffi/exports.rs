// crates/mid-math/src/ffi/exports.rs
//! #[no_mangle] C export functions.
//!
//! Every function:
//!   1. Takes C types (CVec3 etc.) as parameters
//!   2. Converts to internal SIMD types immediately (zero-cost)
//!   3. Calls the SIMD implementation
//!   4. Converts result back to C type (zero-cost)
//!   5. Returns C type
//!
//! The conversion cost is zero — LLVM folds the field copies into
//! register moves. Verified by the wrapped_glam criterion benchmark.
//!
//! Naming: mid_<type>_<operation>
//! Generate a C header with cbindgen or hand-maintain headers/mid_math.h

use crate::ffi::types::{CVec2, CVec3, CVec4, CQuat, CMat3, CMat4};
use crate::{Vec2, Vec3, Vec4, Quat, Mat3, Mat4};
use crate::constants::*;

// ── Vec2 ──────────────────────────────────────────────────────────────────────

#[no_mangle] pub extern "C" fn mid_vec2_new(x:f32, y:f32) -> CVec2 { Vec2::new(x,y).into() }
#[no_mangle] pub extern "C" fn mid_vec2_add(a:CVec2, b:CVec2) -> CVec2 { (Vec2::from(a)+Vec2::from(b)).into() }
#[no_mangle] pub extern "C" fn mid_vec2_sub(a:CVec2, b:CVec2) -> CVec2 { (Vec2::from(a)-Vec2::from(b)).into() }
#[no_mangle] pub extern "C" fn mid_vec2_scale(v:CVec2, s:f32) -> CVec2 { (Vec2::from(v)*s).into() }
#[no_mangle] pub extern "C" fn mid_vec2_dot(a:CVec2, b:CVec2) -> f32 { Vec2::from(a).dot(Vec2::from(b)) }
#[no_mangle] pub extern "C" fn mid_vec2_length(v:CVec2) -> f32 { Vec2::from(v).length() }
#[no_mangle] pub extern "C" fn mid_vec2_normalize(v:CVec2) -> CVec2 { Vec2::from(v).normalize().into() }
#[no_mangle] pub extern "C" fn mid_vec2_lerp(a:CVec2, b:CVec2, t:f32) -> CVec2 { Vec2::from(a).lerp(Vec2::from(b),t).into() }
#[no_mangle] pub extern "C" fn mid_vec2_distance(a:CVec2, b:CVec2) -> f32 { Vec2::from(a).distance(Vec2::from(b)) }

// ── Vec3 ──────────────────────────────────────────────────────────────────────

#[no_mangle] pub extern "C" fn mid_vec3_new(x:f32, y:f32, z:f32) -> CVec3 { Vec3::new(x,y,z).into() }
#[no_mangle] pub extern "C" fn mid_vec3_add(a:CVec3, b:CVec3) -> CVec3 { (Vec3::from(a)+Vec3::from(b)).into() }
#[no_mangle] pub extern "C" fn mid_vec3_sub(a:CVec3, b:CVec3) -> CVec3 { (Vec3::from(a)-Vec3::from(b)).into() }
#[no_mangle] pub extern "C" fn mid_vec3_scale(v:CVec3, s:f32) -> CVec3 { (Vec3::from(v)*s).into() }
#[no_mangle] pub extern "C" fn mid_vec3_dot(a:CVec3, b:CVec3) -> f32 { Vec3::from(a).dot(Vec3::from(b)) }
#[no_mangle] pub extern "C" fn mid_vec3_cross(a:CVec3, b:CVec3) -> CVec3 { Vec3::from(a).cross(Vec3::from(b)).into() }
#[no_mangle] pub extern "C" fn mid_vec3_length(v:CVec3) -> f32 { Vec3::from(v).length() }
#[no_mangle] pub extern "C" fn mid_vec3_normalize(v:CVec3) -> CVec3 { Vec3::from(v).normalize().into() }
#[no_mangle] pub extern "C" fn mid_vec3_lerp(a:CVec3, b:CVec3, t:f32) -> CVec3 { Vec3::from(a).lerp(Vec3::from(b),t).into() }
#[no_mangle] pub extern "C" fn mid_vec3_distance(a:CVec3, b:CVec3) -> f32 { Vec3::from(a).distance(Vec3::from(b)) }
#[no_mangle] pub extern "C" fn mid_vec3_reflect(v:CVec3, n:CVec3) -> CVec3 { Vec3::from(v).reflect(Vec3::from(n)).into() }

// ── Vec4 ──────────────────────────────────────────────────────────────────────

#[no_mangle] pub extern "C" fn mid_vec4_new(x:f32,y:f32,z:f32,w:f32) -> CVec4 { Vec4::new(x,y,z,w).into() }
#[no_mangle] pub extern "C" fn mid_vec4_add(a:CVec4, b:CVec4) -> CVec4 { (Vec4::from(a)+Vec4::from(b)).into() }
#[no_mangle] pub extern "C" fn mid_vec4_dot(a:CVec4, b:CVec4) -> f32 { Vec4::from(a).dot(Vec4::from(b)) }
#[no_mangle] pub extern "C" fn mid_vec4_normalize(v:CVec4) -> CVec4 { Vec4::from(v).normalize().into() }
#[no_mangle] pub extern "C" fn mid_vec4_lerp(a:CVec4, b:CVec4, t:f32) -> CVec4 { Vec4::from(a).lerp(Vec4::from(b),t).into() }

// ── Quat ──────────────────────────────────────────────────────────────────────

#[no_mangle] pub extern "C" fn mid_quat_identity() -> CQuat { Quat::IDENTITY.into() }
#[no_mangle] pub extern "C" fn mid_quat_new(x:f32,y:f32,z:f32,w:f32) -> CQuat { Quat::new(x,y,z,w).into() }
#[no_mangle] pub extern "C" fn mid_quat_from_axis_angle(axis:CVec3, angle_rad:f32) -> CQuat {
    Quat::from_axis_angle(Vec3::from(axis), angle_rad).into()
}
#[no_mangle] pub extern "C" fn mid_quat_from_euler(roll:f32, pitch:f32, yaw:f32) -> CQuat {
    Quat::from_euler(roll, pitch, yaw).into()
}
#[no_mangle] pub extern "C" fn mid_quat_mul(a:CQuat, b:CQuat) -> CQuat {
    (Quat::from(a)*Quat::from(b)).into()
}
#[no_mangle] pub extern "C" fn mid_quat_normalize(q:CQuat) -> CQuat {
    Quat::from(q).normalize().into()
}
#[no_mangle] pub extern "C" fn mid_quat_conjugate(q:CQuat) -> CQuat {
    Quat::from(q).conjugate().into()
}
#[no_mangle] pub extern "C" fn mid_quat_rotate(q:CQuat, v:CVec3) -> CVec3 {
    Quat::from(q).rotate(Vec3::from(v)).into()
}
#[no_mangle] pub extern "C" fn mid_quat_slerp(a:CQuat, b:CQuat, t:f32) -> CQuat {
    Quat::from(a).slerp(Quat::from(b), t).into()
}
#[no_mangle] pub extern "C" fn mid_quat_to_mat4(q:CQuat) -> CMat4 {
    Quat::from(q).to_mat4().into()
}

// ── Mat4 ──────────────────────────────────────────────────────────────────────

#[no_mangle] pub extern "C" fn mid_mat4_identity() -> CMat4 { Mat4::IDENTITY.into() }
#[no_mangle] pub extern "C" fn mid_mat4_from_translation(t:CVec3) -> CMat4 {
    Mat4::from_translation(Vec3::from(t)).into()
}
#[no_mangle] pub extern "C" fn mid_mat4_from_scale(s:CVec3) -> CMat4 {
    Mat4::from_scale(Vec3::from(s)).into()
}
#[no_mangle] pub extern "C" fn mid_mat4_from_rotation(q:CQuat) -> CMat4 {
    Mat4::from_rotation(Quat::from(q)).into()
}
#[no_mangle] pub extern "C" fn mid_mat4_from_trs(t:CVec3, r:CQuat, s:CVec3) -> CMat4 {
    Mat4::from_trs(Vec3::from(t), Quat::from(r), Vec3::from(s)).into()
}
#[no_mangle] pub extern "C" fn mid_mat4_mul(a:CMat4, b:CMat4) -> CMat4 {
    (Mat4::from(a)*Mat4::from(b)).into()
}
#[no_mangle] pub extern "C" fn mid_mat4_transpose(m:CMat4) -> CMat4 {
    Mat4::from(m).transpose().into()
}
#[no_mangle] pub extern "C" fn mid_mat4_transform_point(m:CMat4, p:CVec3) -> CVec3 {
    Mat4::from(m).transform_point(Vec3::from(p)).into()
}
#[no_mangle] pub extern "C" fn mid_mat4_transform_vector(m:CMat4, v:CVec3) -> CVec3 {
    Mat4::from(m).transform_vector(Vec3::from(v)).into()
}
#[no_mangle] pub extern "C" fn mid_mat4_look_at_rh(eye:CVec3, center:CVec3, up:CVec3) -> CMat4 {
    Mat4::look_at_rh(Vec3::from(eye), Vec3::from(center), Vec3::from(up)).into()
}
#[no_mangle] pub extern "C" fn mid_mat4_perspective_rh(fov_y:f32, aspect:f32, near:f32, far:f32) -> CMat4 {
    Mat4::perspective_rh(fov_y, aspect, near, far).into()
}
#[no_mangle] pub extern "C" fn mid_mat4_ortho_rh(l:f32,r:f32,b:f32,t:f32,n:f32,f:f32) -> CMat4 {
    Mat4::ortho_rh(l, r, b, t, n, f).into()
}
/// Inverse. Returns identity if singular — C callers cannot handle Option<T>.
#[no_mangle] pub extern "C" fn mid_mat4_inverse(m:CMat4) -> CMat4 {
    Mat4::from(m).inverse().unwrap_or(Mat4::IDENTITY).into()
}
