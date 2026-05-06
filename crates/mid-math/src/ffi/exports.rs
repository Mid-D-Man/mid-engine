// crates/mid-math/src/ffi/exports.rs

use crate::ffi::types::{
    CAffine3, CMat4, CQuat, CVec2, CVec3, CVec4,
    CDAffine3, CDMat2, CDMat3, CDMat4, CDQuat, CDVec2, CDVec3, CDVec4,
    CIVec2, CIVec3, CIVec4,
    CUVec2, CUVec3, CUVec4,
};
use crate::{Affine3, Mat4, Quat, Vec2, Vec3, Vec4};
use crate::{DAffine3, DMat2, DMat3, DMat4, DQuat, DVec2, DVec3, DVec4};
use crate::{IVec2, IVec3, IVec4, UVec2, UVec3, UVec4};

// ═══════════════════════════════════════════════════════════════════════════
//  f32 exports
// ═══════════════════════════════════════════════════════════════════════════

// ── Vec2 ─────────────────────────────────────────────────────────────────────
#[no_mangle] pub extern "C" fn mid_vec2_new(x:f32,y:f32)->CVec2{Vec2::new(x,y).into()}
#[no_mangle] pub extern "C" fn mid_vec2_add(a:CVec2,b:CVec2)->CVec2{(Vec2::from(a)+Vec2::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_vec2_sub(a:CVec2,b:CVec2)->CVec2{(Vec2::from(a)-Vec2::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_vec2_scale(v:CVec2,s:f32)->CVec2{(Vec2::from(v)*s).into()}
#[no_mangle] pub extern "C" fn mid_vec2_dot(a:CVec2,b:CVec2)->f32{Vec2::from(a).dot(Vec2::from(b))}
#[no_mangle] pub extern "C" fn mid_vec2_length(v:CVec2)->f32{Vec2::from(v).length()}
#[no_mangle] pub extern "C" fn mid_vec2_normalize(v:CVec2)->CVec2{Vec2::from(v).normalize().into()}
#[no_mangle] pub extern "C" fn mid_vec2_lerp(a:CVec2,b:CVec2,t:f32)->CVec2{Vec2::from(a).lerp(Vec2::from(b),t).into()}
#[no_mangle] pub extern "C" fn mid_vec2_distance(a:CVec2,b:CVec2)->f32{Vec2::from(a).distance(Vec2::from(b))}

// ── Vec3 ─────────────────────────────────────────────────────────────────────
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

// ── Vec4 ─────────────────────────────────────────────────────────────────────
#[no_mangle] pub extern "C" fn mid_vec4_new(x:f32,y:f32,z:f32,w:f32)->CVec4{Vec4::new(x,y,z,w).into()}
#[no_mangle] pub extern "C" fn mid_vec4_add(a:CVec4,b:CVec4)->CVec4{(Vec4::from(a)+Vec4::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_vec4_dot(a:CVec4,b:CVec4)->f32{Vec4::from(a).dot(Vec4::from(b))}
#[no_mangle] pub extern "C" fn mid_vec4_normalize(v:CVec4)->CVec4{Vec4::from(v).normalize().into()}
#[no_mangle] pub extern "C" fn mid_vec4_lerp(a:CVec4,b:CVec4,t:f32)->CVec4{Vec4::from(a).lerp(Vec4::from(b),t).into()}

// ── Quat ─────────────────────────────────────────────────────────────────────
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

// ── Mat4 ─────────────────────────────────────────────────────────────────────
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

// ── Affine3 ───────────────────────────────────────────────────────────────────
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

// ═══════════════════════════════════════════════════════════════════════════
//  f64 exports
// ═══════════════════════════════════════════════════════════════════════════

// ── DVec2 ────────────────────────────────────────────────────────────────────
#[no_mangle] pub extern "C" fn mid_dvec2_new(x:f64,y:f64)->CDVec2{DVec2::new(x,y).into()}
#[no_mangle] pub extern "C" fn mid_dvec2_add(a:CDVec2,b:CDVec2)->CDVec2{(DVec2::from(a)+DVec2::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_dvec2_sub(a:CDVec2,b:CDVec2)->CDVec2{(DVec2::from(a)-DVec2::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_dvec2_scale(v:CDVec2,s:f64)->CDVec2{(DVec2::from(v)*s).into()}
#[no_mangle] pub extern "C" fn mid_dvec2_dot(a:CDVec2,b:CDVec2)->f64{DVec2::from(a).dot(DVec2::from(b))}
#[no_mangle] pub extern "C" fn mid_dvec2_length(v:CDVec2)->f64{DVec2::from(v).length()}
#[no_mangle] pub extern "C" fn mid_dvec2_normalize(v:CDVec2)->CDVec2{DVec2::from(v).normalize().into()}
#[no_mangle] pub extern "C" fn mid_dvec2_lerp(a:CDVec2,b:CDVec2,t:f64)->CDVec2{DVec2::from(a).lerp(DVec2::from(b),t).into()}
#[no_mangle] pub extern "C" fn mid_dvec2_distance(a:CDVec2,b:CDVec2)->f64{DVec2::from(a).distance(DVec2::from(b))}
#[no_mangle] pub extern "C" fn mid_dvec2_perp_dot(a:CDVec2,b:CDVec2)->f64{DVec2::from(a).perp_dot(DVec2::from(b))}
#[no_mangle] pub extern "C" fn mid_dvec2_angle_to(a:CDVec2,b:CDVec2)->f64{DVec2::from(a).angle_to(DVec2::from(b))}
#[no_mangle] pub extern "C" fn mid_dvec2_from_angle(angle:f64)->CDVec2{DVec2::from_angle(angle).into()}

// ── DVec3 ────────────────────────────────────────────────────────────────────
#[no_mangle] pub extern "C" fn mid_dvec3_new(x:f64,y:f64,z:f64)->CDVec3{DVec3::new(x,y,z).into()}
#[no_mangle] pub extern "C" fn mid_dvec3_add(a:CDVec3,b:CDVec3)->CDVec3{(DVec3::from(a)+DVec3::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_dvec3_sub(a:CDVec3,b:CDVec3)->CDVec3{(DVec3::from(a)-DVec3::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_dvec3_scale(v:CDVec3,s:f64)->CDVec3{(DVec3::from(v)*s).into()}
#[no_mangle] pub extern "C" fn mid_dvec3_dot(a:CDVec3,b:CDVec3)->f64{DVec3::from(a).dot(DVec3::from(b))}
#[no_mangle] pub extern "C" fn mid_dvec3_cross(a:CDVec3,b:CDVec3)->CDVec3{DVec3::from(a).cross(DVec3::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_dvec3_length(v:CDVec3)->f64{DVec3::from(v).length()}
#[no_mangle] pub extern "C" fn mid_dvec3_normalize(v:CDVec3)->CDVec3{DVec3::from(v).normalize().into()}
#[no_mangle] pub extern "C" fn mid_dvec3_lerp(a:CDVec3,b:CDVec3,t:f64)->CDVec3{DVec3::from(a).lerp(DVec3::from(b),t).into()}
#[no_mangle] pub extern "C" fn mid_dvec3_distance(a:CDVec3,b:CDVec3)->f64{DVec3::from(a).distance(DVec3::from(b))}
#[no_mangle] pub extern "C" fn mid_dvec3_reflect(v:CDVec3,n:CDVec3)->CDVec3{DVec3::from(v).reflect(DVec3::from(n)).into()}
#[no_mangle] pub extern "C" fn mid_dvec3_angle_between(a:CDVec3,b:CDVec3)->f64{DVec3::from(a).angle_between(DVec3::from(b))}

// ── DVec4 ────────────────────────────────────────────────────────────────────
#[no_mangle] pub extern "C" fn mid_dvec4_new(x:f64,y:f64,z:f64,w:f64)->CDVec4{DVec4::new(x,y,z,w).into()}
#[no_mangle] pub extern "C" fn mid_dvec4_add(a:CDVec4,b:CDVec4)->CDVec4{(DVec4::from(a)+DVec4::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_dvec4_sub(a:CDVec4,b:CDVec4)->CDVec4{(DVec4::from(a)-DVec4::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_dvec4_scale(v:CDVec4,s:f64)->CDVec4{(DVec4::from(v)*s).into()}
#[no_mangle] pub extern "C" fn mid_dvec4_dot(a:CDVec4,b:CDVec4)->f64{DVec4::from(a).dot(DVec4::from(b))}
#[no_mangle] pub extern "C" fn mid_dvec4_length(v:CDVec4)->f64{DVec4::from(v).length()}
#[no_mangle] pub extern "C" fn mid_dvec4_normalize(v:CDVec4)->CDVec4{DVec4::from(v).normalize().into()}
#[no_mangle] pub extern "C" fn mid_dvec4_lerp(a:CDVec4,b:CDVec4,t:f64)->CDVec4{DVec4::from(a).lerp(DVec4::from(b),t).into()}

// ── DQuat ────────────────────────────────────────────────────────────────────
#[no_mangle] pub extern "C" fn mid_dquat_identity()->CDQuat{DQuat::IDENTITY.into()}
#[no_mangle] pub extern "C" fn mid_dquat_new(x:f64,y:f64,z:f64,w:f64)->CDQuat{DQuat::new(x,y,z,w).into()}
#[no_mangle] pub extern "C" fn mid_dquat_from_axis_angle(axis:CDVec3,angle_rad:f64)->CDQuat{
    DQuat::from_axis_angle(DVec3::from(axis),angle_rad).into()
}
#[no_mangle] pub extern "C" fn mid_dquat_from_euler(roll:f64,pitch:f64,yaw:f64)->CDQuat{
    DQuat::from_euler(roll,pitch,yaw).into()
}
#[no_mangle] pub extern "C" fn mid_dquat_mul(a:CDQuat,b:CDQuat)->CDQuat{(DQuat::from(a)*DQuat::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_dquat_normalize(q:CDQuat)->CDQuat{DQuat::from(q).normalize().into()}
#[no_mangle] pub extern "C" fn mid_dquat_conjugate(q:CDQuat)->CDQuat{DQuat::from(q).conjugate().into()}
#[no_mangle] pub extern "C" fn mid_dquat_inverse(q:CDQuat)->CDQuat{DQuat::from(q).inverse().into()}
#[no_mangle] pub extern "C" fn mid_dquat_rotate(q:CDQuat,v:CDVec3)->CDVec3{DQuat::from(q).rotate(DVec3::from(v)).into()}
#[no_mangle] pub extern "C" fn mid_dquat_slerp(a:CDQuat,b:CDQuat,t:f64)->CDQuat{DQuat::from(a).slerp(DQuat::from(b),t).into()}
#[no_mangle] pub extern "C" fn mid_dquat_nlerp(a:CDQuat,b:CDQuat,t:f64)->CDQuat{DQuat::from(a).nlerp(DQuat::from(b),t).into()}
#[no_mangle] pub extern "C" fn mid_dquat_to_mat4(q:CDQuat)->CDMat4{DQuat::from(q).to_mat4().into()}

// ── DMat2 ────────────────────────────────────────────────────────────────────
#[no_mangle] pub extern "C" fn mid_dmat2_identity()->CDMat2{DMat2::IDENTITY.into()}
#[no_mangle] pub extern "C" fn mid_dmat2_from_angle(angle:f64)->CDMat2{DMat2::from_angle(angle).into()}
#[no_mangle] pub extern "C" fn mid_dmat2_mul(a:CDMat2,b:CDMat2)->CDMat2{(DMat2::from(a)*DMat2::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_dmat2_transpose(m:CDMat2)->CDMat2{DMat2::from(m).transpose().into()}
#[no_mangle] pub extern "C" fn mid_dmat2_determinant(m:CDMat2)->f64{DMat2::from(m).determinant()}
#[no_mangle] pub extern "C" fn mid_dmat2_inverse(m:CDMat2)->CDMat2{
    DMat2::from(m).inverse().unwrap_or(DMat2::ZERO).into()
}

// ── DMat3 ────────────────────────────────────────────────────────────────────
#[no_mangle] pub extern "C" fn mid_dmat3_identity()->CDMat3{DMat3::IDENTITY.into()}
#[no_mangle] pub extern "C" fn mid_dmat3_mul(a:CDMat3,b:CDMat3)->CDMat3{(DMat3::from(a)*DMat3::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_dmat3_transpose(m:CDMat3)->CDMat3{DMat3::from(m).transpose().into()}
#[no_mangle] pub extern "C" fn mid_dmat3_determinant(m:CDMat3)->f64{DMat3::from(m).determinant()}
#[no_mangle] pub extern "C" fn mid_dmat3_inverse(m:CDMat3)->CDMat3{
    DMat3::from(m).inverse().unwrap_or(DMat3::ZERO).into()
}
#[no_mangle] pub extern "C" fn mid_dmat3_normal_matrix(model:CDMat4)->CDMat3{
    DMat3::normal_matrix(&DMat4::from(model)).unwrap_or(DMat3::IDENTITY).into()
}

// ── DMat4 ────────────────────────────────────────────────────────────────────
#[no_mangle] pub extern "C" fn mid_dmat4_identity()->CDMat4{DMat4::IDENTITY.into()}
#[no_mangle] pub extern "C" fn mid_dmat4_from_translation(t:CDVec3)->CDMat4{DMat4::from_translation(DVec3::from(t)).into()}
#[no_mangle] pub extern "C" fn mid_dmat4_from_scale(s:CDVec3)->CDMat4{DMat4::from_scale(DVec3::from(s)).into()}
#[no_mangle] pub extern "C" fn mid_dmat4_from_rotation(q:CDQuat)->CDMat4{DMat4::from_rotation(DQuat::from(q)).into()}
#[no_mangle] pub extern "C" fn mid_dmat4_from_trs(t:CDVec3,r:CDQuat,s:CDVec3)->CDMat4{
    DMat4::from_trs(DVec3::from(t),DQuat::from(r),DVec3::from(s)).into()
}
#[no_mangle] pub extern "C" fn mid_dmat4_mul(a:CDMat4,b:CDMat4)->CDMat4{(DMat4::from(a)*DMat4::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_dmat4_transpose(m:CDMat4)->CDMat4{DMat4::from(m).transpose().into()}
#[no_mangle] pub extern "C" fn mid_dmat4_transform_point(m:CDMat4,p:CDVec3)->CDVec3{
    DMat4::from(m).transform_point(DVec3::from(p)).into()
}
#[no_mangle] pub extern "C" fn mid_dmat4_transform_vector(m:CDMat4,v:CDVec3)->CDVec3{
    DMat4::from(m).transform_vector(DVec3::from(v)).into()
}
#[no_mangle] pub extern "C" fn mid_dmat4_look_at_rh(eye:CDVec3,center:CDVec3,up:CDVec3)->CDMat4{
    DMat4::look_at_rh(DVec3::from(eye),DVec3::from(center),DVec3::from(up)).into()
}
#[no_mangle] pub extern "C" fn mid_dmat4_perspective_rh(fov_y:f64,aspect:f64,near:f64,far:f64)->CDMat4{
    DMat4::perspective_rh(fov_y,aspect,near,far).into()
}
#[no_mangle] pub extern "C" fn mid_dmat4_ortho_rh(l:f64,r:f64,b:f64,t:f64,n:f64,f:f64)->CDMat4{
    DMat4::ortho_rh(l,r,b,t,n,f).into()
}
#[no_mangle] pub extern "C" fn mid_dmat4_inverse(m:CDMat4)->CDMat4{
    DMat4::from(m).inverse().unwrap_or(DMat4::IDENTITY).into()
}
#[no_mangle] pub extern "C" fn mid_dmat4_inverse_trs(m:CDMat4)->CDMat4{
    DMat4::from(m).inverse_trs().into()
}

// ── DAffine3 ─────────────────────────────────────────────────────────────────
#[no_mangle] pub extern "C" fn mid_daffine3_identity()->CDAffine3{DAffine3::IDENTITY.into()}
#[no_mangle] pub extern "C" fn mid_daffine3_from_trs(t:CDVec3,r:CDQuat,s:CDVec3)->CDAffine3{
    DAffine3::from_trs(DVec3::from(t),DQuat::from(r),DVec3::from(s)).into()
}
#[no_mangle] pub extern "C" fn mid_daffine3_from_translation(t:CDVec3)->CDAffine3{
    DAffine3::from_translation(DVec3::from(t)).into()
}
#[no_mangle] pub extern "C" fn mid_daffine3_from_rotation(q:CDQuat)->CDAffine3{
    DAffine3::from_rotation(DQuat::from(q)).into()
}
#[no_mangle] pub extern "C" fn mid_daffine3_from_scale(s:CDVec3)->CDAffine3{
    DAffine3::from_scale(DVec3::from(s)).into()
}
#[no_mangle] pub extern "C" fn mid_daffine3_from_mat4(m:CDMat4)->CDAffine3{
    DAffine3::from_mat4(DMat4::from(m)).into()
}
#[no_mangle] pub extern "C" fn mid_daffine3_to_mat4(a:CDAffine3)->CDMat4{
    DAffine3::from(a).to_mat4().into()
}
#[no_mangle] pub extern "C" fn mid_daffine3_mul(a:CDAffine3,b:CDAffine3)->CDAffine3{
    (DAffine3::from(a)*DAffine3::from(b)).into()
}
#[no_mangle] pub extern "C" fn mid_daffine3_inverse(a:CDAffine3)->CDAffine3{
    DAffine3::from(a).inverse().into()
}
#[no_mangle] pub extern "C" fn mid_daffine3_transform_point(a:CDAffine3,p:CDVec3)->CDVec3{
    DAffine3::from(a).transform_point(DVec3::from(p)).into()
}
#[no_mangle] pub extern "C" fn mid_daffine3_transform_vector(a:CDAffine3,v:CDVec3)->CDVec3{
    DAffine3::from(a).transform_vector(DVec3::from(v)).into()
}

// ═══════════════════════════════════════════════════════════════════════════
//  i32 integer vector exports
//  Naming: mid_ivec2_*, mid_ivec3_*, mid_ivec4_*
//  All ops pass by value — integer types are trivially copyable.
// ═══════════════════════════════════════════════════════════════════════════

// ── IVec2 ────────────────────────────────────────────────────────────────────
#[no_mangle] pub extern "C" fn mid_ivec2_new(x:i32,y:i32)->CIVec2{IVec2::new(x,y).into()}
#[no_mangle] pub extern "C" fn mid_ivec2_add(a:CIVec2,b:CIVec2)->CIVec2{(IVec2::from(a)+IVec2::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_ivec2_sub(a:CIVec2,b:CIVec2)->CIVec2{(IVec2::from(a)-IVec2::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_ivec2_mul(a:CIVec2,b:CIVec2)->CIVec2{(IVec2::from(a)*IVec2::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_ivec2_scale(v:CIVec2,s:i32)->CIVec2{(IVec2::from(v)*s).into()}
#[no_mangle] pub extern "C" fn mid_ivec2_dot(a:CIVec2,b:CIVec2)->i32{IVec2::from(a).dot(IVec2::from(b))}
#[no_mangle] pub extern "C" fn mid_ivec2_min(a:CIVec2,b:CIVec2)->CIVec2{IVec2::from(a).min(IVec2::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_ivec2_max(a:CIVec2,b:CIVec2)->CIVec2{IVec2::from(a).max(IVec2::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_ivec2_clamp(v:CIVec2,lo:CIVec2,hi:CIVec2)->CIVec2{IVec2::from(v).clamp(IVec2::from(lo),IVec2::from(hi)).into()}
#[no_mangle] pub extern "C" fn mid_ivec2_abs(v:CIVec2)->CIVec2{IVec2::from(v).abs().into()}
#[no_mangle] pub extern "C" fn mid_ivec2_neg(v:CIVec2)->CIVec2{(-IVec2::from(v)).into()}
#[no_mangle] pub extern "C" fn mid_ivec2_length_sq(v:CIVec2)->i32{IVec2::from(v).length_sq()}
#[no_mangle] pub extern "C" fn mid_ivec2_distance_sq(a:CIVec2,b:CIVec2)->i32{IVec2::from(a).distance_sq(IVec2::from(b))}
#[no_mangle] pub extern "C" fn mid_ivec2_min_element(v:CIVec2)->i32{IVec2::from(v).min_element()}
#[no_mangle] pub extern "C" fn mid_ivec2_max_element(v:CIVec2)->i32{IVec2::from(v).max_element()}
#[no_mangle] pub extern "C" fn mid_ivec2_element_sum(v:CIVec2)->i32{IVec2::from(v).element_sum()}
#[no_mangle] pub extern "C" fn mid_ivec2_wrapping_add(a:CIVec2,b:CIVec2)->CIVec2{IVec2::from(a).wrapping_add(IVec2::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_ivec2_wrapping_sub(a:CIVec2,b:CIVec2)->CIVec2{IVec2::from(a).wrapping_sub(IVec2::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_ivec2_saturating_add(a:CIVec2,b:CIVec2)->CIVec2{IVec2::from(a).saturating_add(IVec2::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_ivec2_saturating_sub(a:CIVec2,b:CIVec2)->CIVec2{IVec2::from(a).saturating_sub(IVec2::from(b)).into()}

// ── IVec3 ────────────────────────────────────────────────────────────────────
#[no_mangle] pub extern "C" fn mid_ivec3_new(x:i32,y:i32,z:i32)->CIVec3{IVec3::new(x,y,z).into()}
#[no_mangle] pub extern "C" fn mid_ivec3_add(a:CIVec3,b:CIVec3)->CIVec3{(IVec3::from(a)+IVec3::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_ivec3_sub(a:CIVec3,b:CIVec3)->CIVec3{(IVec3::from(a)-IVec3::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_ivec3_mul(a:CIVec3,b:CIVec3)->CIVec3{(IVec3::from(a)*IVec3::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_ivec3_scale(v:CIVec3,s:i32)->CIVec3{(IVec3::from(v)*s).into()}
#[no_mangle] pub extern "C" fn mid_ivec3_dot(a:CIVec3,b:CIVec3)->i32{IVec3::from(a).dot(IVec3::from(b))}
#[no_mangle] pub extern "C" fn mid_ivec3_cross(a:CIVec3,b:CIVec3)->CIVec3{IVec3::from(a).cross(IVec3::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_ivec3_min(a:CIVec3,b:CIVec3)->CIVec3{IVec3::from(a).min(IVec3::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_ivec3_max(a:CIVec3,b:CIVec3)->CIVec3{IVec3::from(a).max(IVec3::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_ivec3_clamp(v:CIVec3,lo:CIVec3,hi:CIVec3)->CIVec3{IVec3::from(v).clamp(IVec3::from(lo),IVec3::from(hi)).into()}
#[no_mangle] pub extern "C" fn mid_ivec3_abs(v:CIVec3)->CIVec3{IVec3::from(v).abs().into()}
#[no_mangle] pub extern "C" fn mid_ivec3_neg(v:CIVec3)->CIVec3{(-IVec3::from(v)).into()}
#[no_mangle] pub extern "C" fn mid_ivec3_length_sq(v:CIVec3)->i32{IVec3::from(v).length_sq()}
#[no_mangle] pub extern "C" fn mid_ivec3_distance_sq(a:CIVec3,b:CIVec3)->i32{IVec3::from(a).distance_sq(IVec3::from(b))}
#[no_mangle] pub extern "C" fn mid_ivec3_min_element(v:CIVec3)->i32{IVec3::from(v).min_element()}
#[no_mangle] pub extern "C" fn mid_ivec3_max_element(v:CIVec3)->i32{IVec3::from(v).max_element()}
#[no_mangle] pub extern "C" fn mid_ivec3_element_sum(v:CIVec3)->i32{IVec3::from(v).element_sum()}
#[no_mangle] pub extern "C" fn mid_ivec3_wrapping_add(a:CIVec3,b:CIVec3)->CIVec3{IVec3::from(a).wrapping_add(IVec3::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_ivec3_wrapping_sub(a:CIVec3,b:CIVec3)->CIVec3{IVec3::from(a).wrapping_sub(IVec3::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_ivec3_saturating_add(a:CIVec3,b:CIVec3)->CIVec3{IVec3::from(a).saturating_add(IVec3::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_ivec3_saturating_sub(a:CIVec3,b:CIVec3)->CIVec3{IVec3::from(a).saturating_sub(IVec3::from(b)).into()}

// ── IVec4 ────────────────────────────────────────────────────────────────────
#[no_mangle] pub extern "C" fn mid_ivec4_new(x:i32,y:i32,z:i32,w:i32)->CIVec4{IVec4::new(x,y,z,w).into()}
#[no_mangle] pub extern "C" fn mid_ivec4_add(a:CIVec4,b:CIVec4)->CIVec4{(IVec4::from(a)+IVec4::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_ivec4_sub(a:CIVec4,b:CIVec4)->CIVec4{(IVec4::from(a)-IVec4::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_ivec4_mul(a:CIVec4,b:CIVec4)->CIVec4{(IVec4::from(a)*IVec4::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_ivec4_scale(v:CIVec4,s:i32)->CIVec4{(IVec4::from(v)*s).into()}
#[no_mangle] pub extern "C" fn mid_ivec4_dot(a:CIVec4,b:CIVec4)->i32{IVec4::from(a).dot(IVec4::from(b))}
#[no_mangle] pub extern "C" fn mid_ivec4_min(a:CIVec4,b:CIVec4)->CIVec4{IVec4::from(a).min(IVec4::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_ivec4_max(a:CIVec4,b:CIVec4)->CIVec4{IVec4::from(a).max(IVec4::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_ivec4_clamp(v:CIVec4,lo:CIVec4,hi:CIVec4)->CIVec4{IVec4::from(v).clamp(IVec4::from(lo),IVec4::from(hi)).into()}
#[no_mangle] pub extern "C" fn mid_ivec4_abs(v:CIVec4)->CIVec4{IVec4::from(v).abs().into()}
#[no_mangle] pub extern "C" fn mid_ivec4_neg(v:CIVec4)->CIVec4{(-IVec4::from(v)).into()}
#[no_mangle] pub extern "C" fn mid_ivec4_length_sq(v:CIVec4)->i32{IVec4::from(v).length_sq()}
#[no_mangle] pub extern "C" fn mid_ivec4_distance_sq(a:CIVec4,b:CIVec4)->i32{IVec4::from(a).distance_sq(IVec4::from(b))}
#[no_mangle] pub extern "C" fn mid_ivec4_min_element(v:CIVec4)->i32{IVec4::from(v).min_element()}
#[no_mangle] pub extern "C" fn mid_ivec4_max_element(v:CIVec4)->i32{IVec4::from(v).max_element()}
#[no_mangle] pub extern "C" fn mid_ivec4_element_sum(v:CIVec4)->i32{IVec4::from(v).element_sum()}
#[no_mangle] pub extern "C" fn mid_ivec4_wrapping_add(a:CIVec4,b:CIVec4)->CIVec4{IVec4::from(a).wrapping_add(IVec4::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_ivec4_wrapping_sub(a:CIVec4,b:CIVec4)->CIVec4{IVec4::from(a).wrapping_sub(IVec4::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_ivec4_saturating_add(a:CIVec4,b:CIVec4)->CIVec4{IVec4::from(a).saturating_add(IVec4::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_ivec4_saturating_sub(a:CIVec4,b:CIVec4)->CIVec4{IVec4::from(a).saturating_sub(IVec4::from(b)).into()}

// ═══════════════════════════════════════════════════════════════════════════
//  u32 integer vector exports
//  Naming: mid_uvec2_*, mid_uvec3_*, mid_uvec4_*
// ═══════════════════════════════════════════════════════════════════════════

// ── UVec2 ────────────────────────────────────────────────────────────────────
#[no_mangle] pub extern "C" fn mid_uvec2_new(x:u32,y:u32)->CUVec2{UVec2::new(x,y).into()}
#[no_mangle] pub extern "C" fn mid_uvec2_add(a:CUVec2,b:CUVec2)->CUVec2{(UVec2::from(a)+UVec2::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_uvec2_sub(a:CUVec2,b:CUVec2)->CUVec2{(UVec2::from(a)-UVec2::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_uvec2_mul(a:CUVec2,b:CUVec2)->CUVec2{(UVec2::from(a)*UVec2::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_uvec2_scale(v:CUVec2,s:u32)->CUVec2{(UVec2::from(v)*s).into()}
#[no_mangle] pub extern "C" fn mid_uvec2_dot(a:CUVec2,b:CUVec2)->u32{UVec2::from(a).dot(UVec2::from(b))}
#[no_mangle] pub extern "C" fn mid_uvec2_min(a:CUVec2,b:CUVec2)->CUVec2{UVec2::from(a).min(UVec2::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_uvec2_max(a:CUVec2,b:CUVec2)->CUVec2{UVec2::from(a).max(UVec2::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_uvec2_clamp(v:CUVec2,lo:CUVec2,hi:CUVec2)->CUVec2{UVec2::from(v).clamp(UVec2::from(lo),UVec2::from(hi)).into()}
#[no_mangle] pub extern "C" fn mid_uvec2_length_sq(v:CUVec2)->u32{UVec2::from(v).length_sq()}
#[no_mangle] pub extern "C" fn mid_uvec2_min_element(v:CUVec2)->u32{UVec2::from(v).min_element()}
#[no_mangle] pub extern "C" fn mid_uvec2_max_element(v:CUVec2)->u32{UVec2::from(v).max_element()}
#[no_mangle] pub extern "C" fn mid_uvec2_element_sum(v:CUVec2)->u32{UVec2::from(v).element_sum()}
#[no_mangle] pub extern "C" fn mid_uvec2_wrapping_add(a:CUVec2,b:CUVec2)->CUVec2{UVec2::from(a).wrapping_add(UVec2::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_uvec2_wrapping_sub(a:CUVec2,b:CUVec2)->CUVec2{UVec2::from(a).wrapping_sub(UVec2::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_uvec2_saturating_add(a:CUVec2,b:CUVec2)->CUVec2{UVec2::from(a).saturating_add(UVec2::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_uvec2_saturating_sub(a:CUVec2,b:CUVec2)->CUVec2{UVec2::from(a).saturating_sub(UVec2::from(b)).into()}

// ── UVec3 ────────────────────────────────────────────────────────────────────
#[no_mangle] pub extern "C" fn mid_uvec3_new(x:u32,y:u32,z:u32)->CUVec3{UVec3::new(x,y,z).into()}
#[no_mangle] pub extern "C" fn mid_uvec3_add(a:CUVec3,b:CUVec3)->CUVec3{(UVec3::from(a)+UVec3::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_uvec3_sub(a:CUVec3,b:CUVec3)->CUVec3{(UVec3::from(a)-UVec3::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_uvec3_mul(a:CUVec3,b:CUVec3)->CUVec3{(UVec3::from(a)*UVec3::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_uvec3_scale(v:CUVec3,s:u32)->CUVec3{(UVec3::from(v)*s).into()}
#[no_mangle] pub extern "C" fn mid_uvec3_dot(a:CUVec3,b:CUVec3)->u32{UVec3::from(a).dot(UVec3::from(b))}
#[no_mangle] pub extern "C" fn mid_uvec3_cross(a:CUVec3,b:CUVec3)->CUVec3{UVec3::from(a).cross(UVec3::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_uvec3_min(a:CUVec3,b:CUVec3)->CUVec3{UVec3::from(a).min(UVec3::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_uvec3_max(a:CUVec3,b:CUVec3)->CUVec3{UVec3::from(a).max(UVec3::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_uvec3_clamp(v:CUVec3,lo:CUVec3,hi:CUVec3)->CUVec3{UVec3::from(v).clamp(UVec3::from(lo),UVec3::from(hi)).into()}
#[no_mangle] pub extern "C" fn mid_uvec3_length_sq(v:CUVec3)->u32{UVec3::from(v).length_sq()}
#[no_mangle] pub extern "C" fn mid_uvec3_min_element(v:CUVec3)->u32{UVec3::from(v).min_element()}
#[no_mangle] pub extern "C" fn mid_uvec3_max_element(v:CUVec3)->u32{UVec3::from(v).max_element()}
#[no_mangle] pub extern "C" fn mid_uvec3_element_sum(v:CUVec3)->u32{UVec3::from(v).element_sum()}
#[no_mangle] pub extern "C" fn mid_uvec3_wrapping_add(a:CUVec3,b:CUVec3)->CUVec3{UVec3::from(a).wrapping_add(UVec3::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_uvec3_wrapping_sub(a:CUVec3,b:CUVec3)->CUVec3{UVec3::from(a).wrapping_sub(UVec3::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_uvec3_saturating_add(a:CUVec3,b:CUVec3)->CUVec3{UVec3::from(a).saturating_add(UVec3::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_uvec3_saturating_sub(a:CUVec3,b:CUVec3)->CUVec3{UVec3::from(a).saturating_sub(UVec3::from(b)).into()}

// ── UVec4 ────────────────────────────────────────────────────────────────────
#[no_mangle] pub extern "C" fn mid_uvec4_new(x:u32,y:u32,z:u32,w:u32)->CUVec4{UVec4::new(x,y,z,w).into()}
#[no_mangle] pub extern "C" fn mid_uvec4_add(a:CUVec4,b:CUVec4)->CUVec4{(UVec4::from(a)+UVec4::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_uvec4_sub(a:CUVec4,b:CUVec4)->CUVec4{(UVec4::from(a)-UVec4::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_uvec4_mul(a:CUVec4,b:CUVec4)->CUVec4{(UVec4::from(a)*UVec4::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_uvec4_scale(v:CUVec4,s:u32)->CUVec4{(UVec4::from(v)*s).into()}
#[no_mangle] pub extern "C" fn mid_uvec4_dot(a:CUVec4,b:CUVec4)->u32{UVec4::from(a).dot(UVec4::from(b))}
#[no_mangle] pub extern "C" fn mid_uvec4_min(a:CUVec4,b:CUVec4)->CUVec4{UVec4::from(a).min(UVec4::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_uvec4_max(a:CUVec4,b:CUVec4)->CUVec4{UVec4::from(a).max(UVec4::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_uvec4_clamp(v:CUVec4,lo:CUVec4,hi:CUVec4)->CUVec4{UVec4::from(v).clamp(UVec4::from(lo),UVec4::from(hi)).into()}
#[no_mangle] pub extern "C" fn mid_uvec4_length_sq(v:CUVec4)->u32{UVec4::from(v).length_sq()}
#[no_mangle] pub extern "C" fn mid_uvec4_min_element(v:CUVec4)->u32{UVec4::from(v).min_element()}
#[no_mangle] pub extern "C" fn mid_uvec4_max_element(v:CUVec4)->u32{UVec4::from(v).max_element()}
#[no_mangle] pub extern "C" fn mid_uvec4_element_sum(v:CUVec4)->u32{UVec4::from(v).element_sum()}
#[no_mangle] pub extern "C" fn mid_uvec4_wrapping_add(a:CUVec4,b:CUVec4)->CUVec4{UVec4::from(a).wrapping_add(UVec4::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_uvec4_wrapping_sub(a:CUVec4,b:CUVec4)->CUVec4{UVec4::from(a).wrapping_sub(UVec4::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_uvec4_saturating_add(a:CUVec4,b:CUVec4)->CUVec4{UVec4::from(a).saturating_add(UVec4::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_uvec4_saturating_sub(a:CUVec4,b:CUVec4)->CUVec4{UVec4::from(a).saturating_sub(UVec4::from(b)).into()}
