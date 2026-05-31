// crates/mid-math/src/ffi/float64.rs
//! C-ABI types and #[no_mangle] exports for f64 math types.
//!
//! Types:  CDVec2..4, CDQuat, CDMat2..4, CDAffine3
//! Exports: mid_dvec2_*, mid_dvec3_*, mid_dvec4_*, mid_dquat_*,
//!          mid_dmat2_*, mid_dmat3_*, mid_dmat4_*, mid_daffine3_*

use crate::{DAffine3, DMat2, DMat3, DMat4, DQuat, DVec2, DVec3, DVec4};

// Explicit scalar-path aliases needed because on x86_64 `crate::DVec2` and
// `crate::DQuat` resolve to the SSE2 variants, but DMat2/DMat4/DAffine3
// store the scalar types internally (those structs are always scalar).
use crate::f64::dvec2::DVec2 as ScalarDVec2;
use crate::f64::dquat::DQuat as ScalarDQuat;

// ═══════════════════════════════════════════════════════════════════════════
//  C types
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

// DMat2 stores crate::f64::dvec2::DVec2 (always scalar) regardless of platform.
// We must NOT use .into() here because on x86_64 `crate::DVec2` is the SSE2
// variant which is a different type. Use explicit CDVec2::new / ScalarDVec2::new.
impl From<DMat2> for CDMat2 {
    #[inline(always)]
    fn from(m: DMat2) -> Self {
        Self {
            x_axis: CDVec2::new(m.x_axis.x, m.x_axis.y),
            y_axis: CDVec2::new(m.y_axis.x, m.y_axis.y),
        }
    }
}
impl From<CDMat2> for DMat2 {
    #[inline(always)]
    fn from(m: CDMat2) -> Self {
        DMat2::from_cols(
            ScalarDVec2::new(m.x_axis.x, m.x_axis.y),
            ScalarDVec2::new(m.y_axis.x, m.y_axis.y),
        )
    }
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
//  Exports
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
#[no_mangle] pub extern "C" fn mid_dmat4_from_rotation(q:CDQuat)->CDMat4{
    // DMat4::from_rotation expects crate::f64::dquat::DQuat (scalar).
    // We must not pass `DQuat::from(q)` which on x86_64 is the SSE2 variant.
    DMat4::from_rotation(ScalarDQuat::new(q.x, q.y, q.z, q.w)).into()
}
#[no_mangle] pub extern "C" fn mid_dmat4_from_trs(t:CDVec3,r:CDQuat,s:CDVec3)->CDMat4{
    DMat4::from_trs(
        DVec3::from(t),
        ScalarDQuat::new(r.x, r.y, r.z, r.w),
        DVec3::from(s),
    ).into()
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
    // DAffine3::from_trs expects crate::f64::dquat::DQuat (scalar).
    DAffine3::from_trs(
        DVec3::from(t),
        ScalarDQuat::new(r.x, r.y, r.z, r.w),
        DVec3::from(s),
    ).into()
}
#[no_mangle] pub extern "C" fn mid_daffine3_from_translation(t:CDVec3)->CDAffine3{
    DAffine3::from_translation(DVec3::from(t)).into()
}
#[no_mangle] pub extern "C" fn mid_daffine3_from_rotation(q:CDQuat)->CDAffine3{
    // DAffine3::from_rotation expects crate::f64::dquat::DQuat (scalar).
    DAffine3::from_rotation(ScalarDQuat::new(q.x, q.y, q.z, q.w)).into()
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
