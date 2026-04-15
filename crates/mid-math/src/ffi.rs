// crates/mid-math/src/ffi.rs

//! C-compatible FFI exports for mid-math.
//!
//! All types are `#[repr(C)]` so they can be passed by value or pointer
//! directly from C, C++, or any P/Invoke binding without translation.
//!
//! Naming: `mid_<type>_<operation>`
//!
//! Generate a C header with `cbindgen` or hand-maintain `headers/mid_math.h`.

use crate::vec::{Vec2, Vec3, Vec4};
use crate::quat::Quat;
use crate::mat::Mat4;

// ── Vec2 ──────────────────────────────────────────────────────────────────────

#[no_mangle] pub extern "C" fn mid_vec2_new(x: f32, y: f32) -> Vec2 { Vec2::new(x, y) }
#[no_mangle] pub extern "C" fn mid_vec2_add(a: Vec2, b: Vec2) -> Vec2 { a + b }
#[no_mangle] pub extern "C" fn mid_vec2_sub(a: Vec2, b: Vec2) -> Vec2 { a - b }
#[no_mangle] pub extern "C" fn mid_vec2_scale(v: Vec2, s: f32) -> Vec2 { v * s }
#[no_mangle] pub extern "C" fn mid_vec2_dot(a: Vec2, b: Vec2) -> f32  { a.dot(b) }
#[no_mangle] pub extern "C" fn mid_vec2_length(v: Vec2) -> f32        { v.length() }
#[no_mangle] pub extern "C" fn mid_vec2_normalize(v: Vec2) -> Vec2    { v.normalize() }
#[no_mangle] pub extern "C" fn mid_vec2_lerp(a: Vec2, b: Vec2, t: f32) -> Vec2 { a.lerp(b, t) }
#[no_mangle] pub extern "C" fn mid_vec2_distance(a: Vec2, b: Vec2) -> f32 { a.distance(b) }

// ── Vec3 ──────────────────────────────────────────────────────────────────────

#[no_mangle] pub extern "C" fn mid_vec3_new(x: f32, y: f32, z: f32) -> Vec3 { Vec3::new(x, y, z) }
#[no_mangle] pub extern "C" fn mid_vec3_add(a: Vec3, b: Vec3) -> Vec3 { a + b }
#[no_mangle] pub extern "C" fn mid_vec3_sub(a: Vec3, b: Vec3) -> Vec3 { a - b }
#[no_mangle] pub extern "C" fn mid_vec3_scale(v: Vec3, s: f32) -> Vec3 { v * s }
#[no_mangle] pub extern "C" fn mid_vec3_dot(a: Vec3, b: Vec3) -> f32   { a.dot(b) }
#[no_mangle] pub extern "C" fn mid_vec3_cross(a: Vec3, b: Vec3) -> Vec3 { a.cross(b) }
#[no_mangle] pub extern "C" fn mid_vec3_length(v: Vec3) -> f32          { v.length() }
#[no_mangle] pub extern "C" fn mid_vec3_normalize(v: Vec3) -> Vec3      { v.normalize() }
#[no_mangle] pub extern "C" fn mid_vec3_lerp(a: Vec3, b: Vec3, t: f32) -> Vec3 { a.lerp(b, t) }
#[no_mangle] pub extern "C" fn mid_vec3_distance(a: Vec3, b: Vec3) -> f32 { a.distance(b) }
#[no_mangle] pub extern "C" fn mid_vec3_reflect(v: Vec3, n: Vec3) -> Vec3  { v.reflect(n) }

// ── Vec4 ──────────────────────────────────────────────────────────────────────

#[no_mangle] pub extern "C" fn mid_vec4_new(x: f32, y: f32, z: f32, w: f32) -> Vec4 { Vec4::new(x,y,z,w) }
#[no_mangle] pub extern "C" fn mid_vec4_add(a: Vec4, b: Vec4) -> Vec4 { a + b }
#[no_mangle] pub extern "C" fn mid_vec4_dot(a: Vec4, b: Vec4) -> f32  { a.dot(b) }
#[no_mangle] pub extern "C" fn mid_vec4_normalize(v: Vec4) -> Vec4    { v.normalize() }
#[no_mangle] pub extern "C" fn mid_vec4_lerp(a: Vec4, b: Vec4, t: f32) -> Vec4 { a.lerp(b, t) }

// ── Quat ──────────────────────────────────────────────────────────────────────

#[no_mangle] pub extern "C" fn mid_quat_identity() -> Quat { Quat::IDENTITY }
#[no_mangle] pub extern "C" fn mid_quat_new(x: f32, y: f32, z: f32, w: f32) -> Quat { Quat::new(x,y,z,w) }
#[no_mangle] pub extern "C" fn mid_quat_from_axis_angle(axis: Vec3, angle_rad: f32) -> Quat {
    Quat::from_axis_angle(axis, angle_rad)
}
#[no_mangle] pub extern "C" fn mid_quat_from_euler(roll: f32, pitch: f32, yaw: f32) -> Quat {
    Quat::from_euler(roll, pitch, yaw)
}
#[no_mangle] pub extern "C" fn mid_quat_mul(a: Quat, b: Quat) -> Quat      { a * b }
#[no_mangle] pub extern "C" fn mid_quat_normalize(q: Quat) -> Quat          { q.normalize() }
#[no_mangle] pub extern "C" fn mid_quat_conjugate(q: Quat) -> Quat          { q.conjugate() }
#[no_mangle] pub extern "C" fn mid_quat_rotate(q: Quat, v: Vec3) -> Vec3    { q.rotate(v) }
#[no_mangle] pub extern "C" fn mid_quat_slerp(a: Quat, b: Quat, t: f32) -> Quat { a.slerp(b, t) }
#[no_mangle] pub extern "C" fn mid_quat_to_mat4(q: Quat) -> Mat4            { q.to_mat4() }

// ── Mat4 ──────────────────────────────────────────────────────────────────────

#[no_mangle] pub extern "C" fn mid_mat4_identity() -> Mat4 { Mat4::IDENTITY }
#[no_mangle] pub extern "C" fn mid_mat4_from_translation(t: Vec3) -> Mat4  { Mat4::from_translation(t) }
#[no_mangle] pub extern "C" fn mid_mat4_from_scale(s: Vec3) -> Mat4        { Mat4::from_scale(s) }
#[no_mangle] pub extern "C" fn mid_mat4_from_rotation(q: Quat) -> Mat4     { Mat4::from_rotation(q) }
#[no_mangle] pub extern "C" fn mid_mat4_from_trs(t: Vec3, r: Quat, s: Vec3) -> Mat4 { Mat4::from_trs(t, r, s) }
#[no_mangle] pub extern "C" fn mid_mat4_mul(a: Mat4, b: Mat4) -> Mat4      { a * b }
#[no_mangle] pub extern "C" fn mid_mat4_transpose(m: Mat4) -> Mat4         { m.transpose() }
#[no_mangle] pub extern "C" fn mid_mat4_transform_point(m: Mat4, p: Vec3) -> Vec3  { m.transform_point(p) }
#[no_mangle] pub extern "C" fn mid_mat4_transform_vector(m: Mat4, v: Vec3) -> Vec3 { m.transform_vector(v) }
#[no_mangle] pub extern "C" fn mid_mat4_look_at_rh(eye: Vec3, center: Vec3, up: Vec3) -> Mat4 {
    Mat4::look_at_rh(eye, center, up)
}
#[no_mangle] pub extern "C" fn mid_mat4_perspective_rh(fov_y: f32, aspect: f32, near: f32, far: f32) -> Mat4 {
    Mat4::perspective_rh(fov_y, aspect, near, far)
}
#[no_mangle] pub extern "C" fn mid_mat4_ortho_rh(l: f32, r: f32, b: f32, t: f32, n: f32, f: f32) -> Mat4 {
    Mat4::ortho_rh(l, r, b, t, n, f)
}
/// Inverse. Returns identity if singular — C callers cannot handle Option<T>.
#[no_mangle] pub extern "C" fn mid_mat4_inverse(m: Mat4) -> Mat4 {
    m.inverse().unwrap_or(Mat4::IDENTITY)
}
