// crates/mid-math/src/ffi/int32.rs
//! C-ABI types and #[no_mangle] exports for i32/u32 integer vector types.
//!
//! Types:  CIVec2..4, CUVec2..4
//! Exports: mid_ivec2_*, mid_ivec3_*, mid_ivec4_*,
//!          mid_uvec2_*, mid_uvec3_*, mid_uvec4_*

use crate::{IVec2, IVec3, IVec4, UVec2, UVec3, UVec4};

// ═══════════════════════════════════════════════════════════════════════════
//  C types
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
//  Exports — IVec2
// ═══════════════════════════════════════════════════════════════════════════

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

// ── Exports — IVec3 ───────────────────────────────────────────────────────────
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

// ── Exports — IVec4 ───────────────────────────────────────────────────────────
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

// ── Exports — UVec2 ───────────────────────────────────────────────────────────
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

// ── Exports — UVec3 ───────────────────────────────────────────────────────────
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

// ── Exports — UVec4 ───────────────────────────────────────────────────────────
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
