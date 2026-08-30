// crates/mid-math/src/ffi/wide_float.rs
//! C-ABI types and #[no_mangle] exports for the always-available
//! (SSE2/NEON/scalar/wasm-tier) wide/float SIMD types: f32x4, Vec3x4, QuatX4.
//!
//! Deliberately excludes the AVX2-only Vec3x8 -- same reasoning as
//! wide_int.rs's exclusion of the AVX2-only wide/int types: exposing a
//! conditionally-compiled type over a stable C ABI needs its own design
//! pass, not decided here. No QuatX8/Rotor3x8-equivalent exists in
//! mid-math yet either way (confirmed against source this session).
//!
//! C representation differs by type, deliberately:
//!   Cf32x4  -- a bare `v: [f32; 4]` array (4 independent packed scalar
//!              lanes, not a vector -- same reasoning as wide_int.rs).
//!   CVec3x4 -- `v: [CVec3; 4]`, an AoS array of the EXISTING CVec3 type
//!              from ffi/float32.rs, not a raw SoA `x/y/z: [f32;4]`
//!              layout. Vec3x4's internal SoA __m128 fields are a
//!              platform-specific implementation detail (raw __m128
//!              is never exposed across this FFI boundary, matching
//!              every other file in this module); reusing CVec3 keeps
//!              the C-side representation consistent with the rest of
//!              the API and makes the per-lane conversion a direct
//!              array map, not a manual transpose.
//!   CQuatX4 -- `v: [CQuat; 4]`, same reasoning.
//!
//! dot() returns Cf32x4 (4 independent per-lane dot products), matching
//! the underlying Vec3x4::dot/QuatX4::dot -> f32x4 signature exactly.
//! lerp/nlerp take a Cf32x4 `t` (per-lane blend factor, confirmed NOT
//! a single scalar broadcast across lanes -- t[i] applies to lane i).

use crate::{f32x4, Vec3x4, QuatX4};
use super::float32::{CVec3, CQuat};

// ═══════════════════════════════════════════════════════════════════════════
//  C types
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Copy, PartialEq)] #[repr(C)]
pub struct Cf32x4 { pub v: [f32; 4] }
impl From<f32x4>  for Cf32x4 { #[inline(always)] fn from(w: f32x4)  -> Self { Self { v: w.to_array() } } }
impl From<Cf32x4> for f32x4  { #[inline(always)] fn from(c: Cf32x4) -> Self { f32x4::from_array(c.v) } }

#[derive(Debug, Clone, Copy, PartialEq)] #[repr(C)]
pub struct CVec3x4 { pub v: [CVec3; 4] }
impl From<Vec3x4> for CVec3x4 {
    #[inline(always)]
    fn from(w: Vec3x4) -> Self { Self { v: w.to_array().map(CVec3::from) } }
}
impl From<CVec3x4> for Vec3x4 {
    #[inline(always)]
    fn from(c: CVec3x4) -> Self {
        let arr: [crate::Vec3; 4] = c.v.map(crate::Vec3::from);
        Vec3x4::from_slice(&arr)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)] #[repr(C)]
pub struct CQuatX4 { pub v: [CQuat; 4] }
impl From<QuatX4> for CQuatX4 {
    #[inline(always)]
    fn from(w: QuatX4) -> Self { Self { v: w.to_array().map(CQuat::from) } }
}
impl From<CQuatX4> for QuatX4 {
    #[inline(always)]
    fn from(c: CQuatX4) -> Self {
        let arr: [crate::Quat; 4] = c.v.map(crate::Quat::from);
        QuatX4::from_slice(&arr)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Exports — f32x4
// ═══════════════════════════════════════════════════════════════════════════

#[no_mangle] pub extern "C" fn mid_f32x4_new(v0:f32,v1:f32,v2:f32,v3:f32)->Cf32x4{f32x4::new(v0,v1,v2,v3).into()}
#[no_mangle] pub extern "C" fn mid_f32x4_splat(v:f32)->Cf32x4{f32x4::splat(v).into()}
#[no_mangle] pub extern "C" fn mid_f32x4_add(a:Cf32x4,b:Cf32x4)->Cf32x4{(f32x4::from(a)+f32x4::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_f32x4_sub(a:Cf32x4,b:Cf32x4)->Cf32x4{(f32x4::from(a)-f32x4::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_f32x4_mul(a:Cf32x4,b:Cf32x4)->Cf32x4{(f32x4::from(a)*f32x4::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_f32x4_min(a:Cf32x4,b:Cf32x4)->Cf32x4{f32x4::from(a).min(f32x4::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_f32x4_max(a:Cf32x4,b:Cf32x4)->Cf32x4{f32x4::from(a).max(f32x4::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_f32x4_clamp(v:Cf32x4,lo:Cf32x4,hi:Cf32x4)->Cf32x4{f32x4::from(v).clamp(f32x4::from(lo),f32x4::from(hi)).into()}
#[no_mangle] pub extern "C" fn mid_f32x4_abs(v:Cf32x4)->Cf32x4{f32x4::from(v).abs().into()}
#[no_mangle] pub extern "C" fn mid_f32x4_sqrt(v:Cf32x4)->Cf32x4{f32x4::from(v).sqrt().into()}
#[no_mangle] pub extern "C" fn mid_f32x4_recip(v:Cf32x4)->Cf32x4{f32x4::from(v).recip().into()}
#[no_mangle] pub extern "C" fn mid_f32x4_recip_sqrt(v:Cf32x4)->Cf32x4{f32x4::from(v).recip_sqrt().into()}
#[no_mangle] pub extern "C" fn mid_f32x4_mul_add(a:Cf32x4,b:Cf32x4,c:Cf32x4)->Cf32x4{f32x4::from(a).mul_add(f32x4::from(b),f32x4::from(c)).into()}

// ═══════════════════════════════════════════════════════════════════════════
//  Exports — Vec3x4
// ═══════════════════════════════════════════════════════════════════════════

#[no_mangle] pub extern "C" fn mid_vec3x4_new(v0:CVec3,v1:CVec3,v2:CVec3,v3:CVec3)->CVec3x4{CVec3x4{v:[v0,v1,v2,v3]}}
#[no_mangle] pub extern "C" fn mid_vec3x4_splat(v:CVec3)->CVec3x4{Vec3x4::splat(crate::Vec3::from(v)).into()}
#[no_mangle] pub extern "C" fn mid_vec3x4_add(a:CVec3x4,b:CVec3x4)->CVec3x4{(Vec3x4::from(a)+Vec3x4::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_vec3x4_sub(a:CVec3x4,b:CVec3x4)->CVec3x4{(Vec3x4::from(a)-Vec3x4::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_vec3x4_dot(a:CVec3x4,b:CVec3x4)->Cf32x4{Vec3x4::from(a).dot(Vec3x4::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_vec3x4_cross(a:CVec3x4,b:CVec3x4)->CVec3x4{Vec3x4::from(a).cross(Vec3x4::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_vec3x4_normalize(v:CVec3x4)->CVec3x4{Vec3x4::from(v).normalize().into()}
#[no_mangle] pub extern "C" fn mid_vec3x4_normalize_precise(v:CVec3x4)->CVec3x4{Vec3x4::from(v).normalize_precise().into()}
#[no_mangle] pub extern "C" fn mid_vec3x4_lerp(a:CVec3x4,b:CVec3x4,t:Cf32x4)->CVec3x4{Vec3x4::from(a).lerp(Vec3x4::from(b),f32x4::from(t)).into()}

// ═══════════════════════════════════════════════════════════════════════════
//  Exports — QuatX4
// ═══════════════════════════════════════════════════════════════════════════

#[no_mangle] pub extern "C" fn mid_quatx4_new(v0:CQuat,v1:CQuat,v2:CQuat,v3:CQuat)->CQuatX4{CQuatX4{v:[v0,v1,v2,v3]}}
#[no_mangle] pub extern "C" fn mid_quatx4_splat(q:CQuat)->CQuatX4{QuatX4::splat(crate::Quat::from(q)).into()}
#[no_mangle] pub extern "C" fn mid_quatx4_mul(a:CQuatX4,b:CQuatX4)->CQuatX4{(QuatX4::from(a)*QuatX4::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_quatx4_dot(a:CQuatX4,b:CQuatX4)->Cf32x4{QuatX4::from(a).dot(QuatX4::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_quatx4_rotate(q:CQuatX4,v:CVec3x4)->CVec3x4{QuatX4::from(q).rotate(Vec3x4::from(v)).into()}
#[no_mangle] pub extern "C" fn mid_quatx4_nlerp(a:CQuatX4,b:CQuatX4,t:Cf32x4)->CQuatX4{QuatX4::from(a).nlerp(QuatX4::from(b),f32x4::from(t)).into()}
#[no_mangle] pub extern "C" fn mid_quatx4_normalize(q:CQuatX4)->CQuatX4{QuatX4::from(q).normalize().into()}
