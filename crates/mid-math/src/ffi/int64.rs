// crates/mid-math/src/ffi/int64.rs
//! C-ABI types and #[no_mangle] exports for i64/u64 integer vector types.
//!
//! Types:  CI64Vec2..4, CU64Vec2..4
//! Exports: mid_i64vec2_*, mid_i64vec3_*, mid_i64vec4_*,
//!          mid_u64vec2_*, mid_u64vec3_*, mid_u64vec4_*

use crate::{I64Vec2, I64Vec3, I64Vec4, U64Vec2, U64Vec3, U64Vec4};

// ═══════════════════════════════════════════════════════════════════════════
//  C types
// ═══════════════════════════════════════════════════════════════════════════

/// C-ABI I64Vec2. 16 bytes, align 8.
#[derive(Debug, Clone, Copy, PartialEq, Eq)] #[repr(C, align(8))]
pub struct CI64Vec2 { pub x: i64, pub y: i64 }
impl From<I64Vec2>  for CI64Vec2 { #[inline(always)] fn from(v: I64Vec2)  -> Self { Self { x: v.x, y: v.y } } }
impl From<CI64Vec2> for I64Vec2  { #[inline(always)] fn from(v: CI64Vec2) -> Self { I64Vec2::new(v.x, v.y) } }

/// C-ABI I64Vec3. 24 bytes, align 8. No padding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)] #[repr(C, align(8))]
pub struct CI64Vec3 { pub x: i64, pub y: i64, pub z: i64 }
impl From<I64Vec3>  for CI64Vec3 { #[inline(always)] fn from(v: I64Vec3)  -> Self { Self { x: v.x, y: v.y, z: v.z } } }
impl From<CI64Vec3> for I64Vec3  { #[inline(always)] fn from(v: CI64Vec3) -> Self { I64Vec3::new(v.x, v.y, v.z) } }

/// C-ABI I64Vec4. 32 bytes, align 8.
#[derive(Debug, Clone, Copy, PartialEq, Eq)] #[repr(C, align(8))]
pub struct CI64Vec4 { pub x: i64, pub y: i64, pub z: i64, pub w: i64 }
impl From<I64Vec4>  for CI64Vec4 { #[inline(always)] fn from(v: I64Vec4)  -> Self { Self { x: v.x, y: v.y, z: v.z, w: v.w } } }
impl From<CI64Vec4> for I64Vec4  { #[inline(always)] fn from(v: CI64Vec4) -> Self { I64Vec4::new(v.x, v.y, v.z, v.w) } }

/// C-ABI U64Vec2. 16 bytes, align 8.
#[derive(Debug, Clone, Copy, PartialEq, Eq)] #[repr(C, align(8))]
pub struct CU64Vec2 { pub x: u64, pub y: u64 }
impl From<U64Vec2>  for CU64Vec2 { #[inline(always)] fn from(v: U64Vec2)  -> Self { Self { x: v.x, y: v.y } } }
impl From<CU64Vec2> for U64Vec2  { #[inline(always)] fn from(v: CU64Vec2) -> Self { U64Vec2::new(v.x, v.y) } }

/// C-ABI U64Vec3. 24 bytes, align 8. No padding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)] #[repr(C, align(8))]
pub struct CU64Vec3 { pub x: u64, pub y: u64, pub z: u64 }
impl From<U64Vec3>  for CU64Vec3 { #[inline(always)] fn from(v: U64Vec3)  -> Self { Self { x: v.x, y: v.y, z: v.z } } }
impl From<CU64Vec3> for U64Vec3  { #[inline(always)] fn from(v: CU64Vec3) -> Self { U64Vec3::new(v.x, v.y, v.z) } }

/// C-ABI U64Vec4. 32 bytes, align 8.
#[derive(Debug, Clone, Copy, PartialEq, Eq)] #[repr(C, align(8))]
pub struct CU64Vec4 { pub x: u64, pub y: u64, pub z: u64, pub w: u64 }
impl From<U64Vec4>  for CU64Vec4 { #[inline(always)] fn from(v: U64Vec4)  -> Self { Self { x: v.x, y: v.y, z: v.z, w: v.w } } }
impl From<CU64Vec4> for U64Vec4  { #[inline(always)] fn from(v: CU64Vec4) -> Self { U64Vec4::new(v.x, v.y, v.z, v.w) } }

// ═══════════════════════════════════════════════════════════════════════════
//  Exports — I64Vec2
// ═══════════════════════════════════════════════════════════════════════════

#[no_mangle] pub extern "C" fn mid_i64vec2_new(x:i64,y:i64)->CI64Vec2{I64Vec2::new(x,y).into()}
#[no_mangle] pub extern "C" fn mid_i64vec2_add(a:CI64Vec2,b:CI64Vec2)->CI64Vec2{(I64Vec2::from(a)+I64Vec2::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_i64vec2_sub(a:CI64Vec2,b:CI64Vec2)->CI64Vec2{(I64Vec2::from(a)-I64Vec2::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_i64vec2_mul(a:CI64Vec2,b:CI64Vec2)->CI64Vec2{(I64Vec2::from(a)*I64Vec2::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_i64vec2_scale(v:CI64Vec2,s:i64)->CI64Vec2{(I64Vec2::from(v)*s).into()}
#[no_mangle] pub extern "C" fn mid_i64vec2_dot(a:CI64Vec2,b:CI64Vec2)->i64{I64Vec2::from(a).dot(I64Vec2::from(b))}
#[no_mangle] pub extern "C" fn mid_i64vec2_min(a:CI64Vec2,b:CI64Vec2)->CI64Vec2{I64Vec2::from(a).min(I64Vec2::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_i64vec2_max(a:CI64Vec2,b:CI64Vec2)->CI64Vec2{I64Vec2::from(a).max(I64Vec2::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_i64vec2_clamp(v:CI64Vec2,lo:CI64Vec2,hi:CI64Vec2)->CI64Vec2{I64Vec2::from(v).clamp(I64Vec2::from(lo),I64Vec2::from(hi)).into()}
#[no_mangle] pub extern "C" fn mid_i64vec2_abs(v:CI64Vec2)->CI64Vec2{I64Vec2::from(v).abs().into()}
#[no_mangle] pub extern "C" fn mid_i64vec2_neg(v:CI64Vec2)->CI64Vec2{(-I64Vec2::from(v)).into()}
#[no_mangle] pub extern "C" fn mid_i64vec2_length_sq(v:CI64Vec2)->i64{I64Vec2::from(v).length_sq()}
#[no_mangle] pub extern "C" fn mid_i64vec2_min_element(v:CI64Vec2)->i64{I64Vec2::from(v).min_element()}
#[no_mangle] pub extern "C" fn mid_i64vec2_max_element(v:CI64Vec2)->i64{I64Vec2::from(v).max_element()}
#[no_mangle] pub extern "C" fn mid_i64vec2_element_sum(v:CI64Vec2)->i64{I64Vec2::from(v).element_sum()}
#[no_mangle] pub extern "C" fn mid_i64vec2_wrapping_add(a:CI64Vec2,b:CI64Vec2)->CI64Vec2{I64Vec2::from(a).wrapping_add(I64Vec2::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_i64vec2_wrapping_sub(a:CI64Vec2,b:CI64Vec2)->CI64Vec2{I64Vec2::from(a).wrapping_sub(I64Vec2::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_i64vec2_saturating_add(a:CI64Vec2,b:CI64Vec2)->CI64Vec2{I64Vec2::from(a).saturating_add(I64Vec2::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_i64vec2_saturating_sub(a:CI64Vec2,b:CI64Vec2)->CI64Vec2{I64Vec2::from(a).saturating_sub(I64Vec2::from(b)).into()}

// ── Exports — I64Vec3 ─────────────────────────────────────────────────────────
#[no_mangle] pub extern "C" fn mid_i64vec3_new(x:i64,y:i64,z:i64)->CI64Vec3{I64Vec3::new(x,y,z).into()}
#[no_mangle] pub extern "C" fn mid_i64vec3_add(a:CI64Vec3,b:CI64Vec3)->CI64Vec3{(I64Vec3::from(a)+I64Vec3::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_i64vec3_sub(a:CI64Vec3,b:CI64Vec3)->CI64Vec3{(I64Vec3::from(a)-I64Vec3::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_i64vec3_mul(a:CI64Vec3,b:CI64Vec3)->CI64Vec3{(I64Vec3::from(a)*I64Vec3::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_i64vec3_scale(v:CI64Vec3,s:i64)->CI64Vec3{(I64Vec3::from(v)*s).into()}
#[no_mangle] pub extern "C" fn mid_i64vec3_dot(a:CI64Vec3,b:CI64Vec3)->i64{I64Vec3::from(a).dot(I64Vec3::from(b))}
#[no_mangle] pub extern "C" fn mid_i64vec3_cross(a:CI64Vec3,b:CI64Vec3)->CI64Vec3{I64Vec3::from(a).cross(I64Vec3::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_i64vec3_min(a:CI64Vec3,b:CI64Vec3)->CI64Vec3{I64Vec3::from(a).min(I64Vec3::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_i64vec3_max(a:CI64Vec3,b:CI64Vec3)->CI64Vec3{I64Vec3::from(a).max(I64Vec3::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_i64vec3_clamp(v:CI64Vec3,lo:CI64Vec3,hi:CI64Vec3)->CI64Vec3{I64Vec3::from(v).clamp(I64Vec3::from(lo),I64Vec3::from(hi)).into()}
#[no_mangle] pub extern "C" fn mid_i64vec3_abs(v:CI64Vec3)->CI64Vec3{I64Vec3::from(v).abs().into()}
#[no_mangle] pub extern "C" fn mid_i64vec3_neg(v:CI64Vec3)->CI64Vec3{(-I64Vec3::from(v)).into()}
#[no_mangle] pub extern "C" fn mid_i64vec3_length_sq(v:CI64Vec3)->i64{I64Vec3::from(v).length_sq()}
#[no_mangle] pub extern "C" fn mid_i64vec3_min_element(v:CI64Vec3)->i64{I64Vec3::from(v).min_element()}
#[no_mangle] pub extern "C" fn mid_i64vec3_max_element(v:CI64Vec3)->i64{I64Vec3::from(v).max_element()}
#[no_mangle] pub extern "C" fn mid_i64vec3_element_sum(v:CI64Vec3)->i64{I64Vec3::from(v).element_sum()}
#[no_mangle] pub extern "C" fn mid_i64vec3_wrapping_add(a:CI64Vec3,b:CI64Vec3)->CI64Vec3{I64Vec3::from(a).wrapping_add(I64Vec3::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_i64vec3_wrapping_sub(a:CI64Vec3,b:CI64Vec3)->CI64Vec3{I64Vec3::from(a).wrapping_sub(I64Vec3::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_i64vec3_saturating_add(a:CI64Vec3,b:CI64Vec3)->CI64Vec3{I64Vec3::from(a).saturating_add(I64Vec3::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_i64vec3_saturating_sub(a:CI64Vec3,b:CI64Vec3)->CI64Vec3{I64Vec3::from(a).saturating_sub(I64Vec3::from(b)).into()}

// ── Exports — I64Vec4 ─────────────────────────────────────────────────────────
#[no_mangle] pub extern "C" fn mid_i64vec4_new(x:i64,y:i64,z:i64,w:i64)->CI64Vec4{I64Vec4::new(x,y,z,w).into()}
#[no_mangle] pub extern "C" fn mid_i64vec4_add(a:CI64Vec4,b:CI64Vec4)->CI64Vec4{(I64Vec4::from(a)+I64Vec4::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_i64vec4_sub(a:CI64Vec4,b:CI64Vec4)->CI64Vec4{(I64Vec4::from(a)-I64Vec4::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_i64vec4_mul(a:CI64Vec4,b:CI64Vec4)->CI64Vec4{(I64Vec4::from(a)*I64Vec4::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_i64vec4_scale(v:CI64Vec4,s:i64)->CI64Vec4{(I64Vec4::from(v)*s).into()}
#[no_mangle] pub extern "C" fn mid_i64vec4_dot(a:CI64Vec4,b:CI64Vec4)->i64{I64Vec4::from(a).dot(I64Vec4::from(b))}
#[no_mangle] pub extern "C" fn mid_i64vec4_min(a:CI64Vec4,b:CI64Vec4)->CI64Vec4{I64Vec4::from(a).min(I64Vec4::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_i64vec4_max(a:CI64Vec4,b:CI64Vec4)->CI64Vec4{I64Vec4::from(a).max(I64Vec4::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_i64vec4_clamp(v:CI64Vec4,lo:CI64Vec4,hi:CI64Vec4)->CI64Vec4{I64Vec4::from(v).clamp(I64Vec4::from(lo),I64Vec4::from(hi)).into()}
#[no_mangle] pub extern "C" fn mid_i64vec4_abs(v:CI64Vec4)->CI64Vec4{I64Vec4::from(v).abs().into()}
#[no_mangle] pub extern "C" fn mid_i64vec4_neg(v:CI64Vec4)->CI64Vec4{(-I64Vec4::from(v)).into()}
#[no_mangle] pub extern "C" fn mid_i64vec4_length_sq(v:CI64Vec4)->i64{I64Vec4::from(v).length_sq()}
#[no_mangle] pub extern "C" fn mid_i64vec4_min_element(v:CI64Vec4)->i64{I64Vec4::from(v).min_element()}
#[no_mangle] pub extern "C" fn mid_i64vec4_max_element(v:CI64Vec4)->i64{I64Vec4::from(v).max_element()}
#[no_mangle] pub extern "C" fn mid_i64vec4_element_sum(v:CI64Vec4)->i64{I64Vec4::from(v).element_sum()}
#[no_mangle] pub extern "C" fn mid_i64vec4_wrapping_add(a:CI64Vec4,b:CI64Vec4)->CI64Vec4{I64Vec4::from(a).wrapping_add(I64Vec4::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_i64vec4_wrapping_sub(a:CI64Vec4,b:CI64Vec4)->CI64Vec4{I64Vec4::from(a).wrapping_sub(I64Vec4::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_i64vec4_saturating_add(a:CI64Vec4,b:CI64Vec4)->CI64Vec4{I64Vec4::from(a).saturating_add(I64Vec4::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_i64vec4_saturating_sub(a:CI64Vec4,b:CI64Vec4)->CI64Vec4{I64Vec4::from(a).saturating_sub(I64Vec4::from(b)).into()}

// ── Exports — U64Vec2 ─────────────────────────────────────────────────────────
#[no_mangle] pub extern "C" fn mid_u64vec2_new(x:u64,y:u64)->CU64Vec2{U64Vec2::new(x,y).into()}
#[no_mangle] pub extern "C" fn mid_u64vec2_add(a:CU64Vec2,b:CU64Vec2)->CU64Vec2{(U64Vec2::from(a)+U64Vec2::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_u64vec2_sub(a:CU64Vec2,b:CU64Vec2)->CU64Vec2{(U64Vec2::from(a)-U64Vec2::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_u64vec2_mul(a:CU64Vec2,b:CU64Vec2)->CU64Vec2{(U64Vec2::from(a)*U64Vec2::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_u64vec2_scale(v:CU64Vec2,s:u64)->CU64Vec2{(U64Vec2::from(v)*s).into()}
#[no_mangle] pub extern "C" fn mid_u64vec2_dot(a:CU64Vec2,b:CU64Vec2)->u64{U64Vec2::from(a).dot(U64Vec2::from(b))}
#[no_mangle] pub extern "C" fn mid_u64vec2_min(a:CU64Vec2,b:CU64Vec2)->CU64Vec2{U64Vec2::from(a).min(U64Vec2::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_u64vec2_max(a:CU64Vec2,b:CU64Vec2)->CU64Vec2{U64Vec2::from(a).max(U64Vec2::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_u64vec2_clamp(v:CU64Vec2,lo:CU64Vec2,hi:CU64Vec2)->CU64Vec2{U64Vec2::from(v).clamp(U64Vec2::from(lo),U64Vec2::from(hi)).into()}
#[no_mangle] pub extern "C" fn mid_u64vec2_length_sq(v:CU64Vec2)->u64{U64Vec2::from(v).length_sq()}
#[no_mangle] pub extern "C" fn mid_u64vec2_min_element(v:CU64Vec2)->u64{U64Vec2::from(v).min_element()}
#[no_mangle] pub extern "C" fn mid_u64vec2_max_element(v:CU64Vec2)->u64{U64Vec2::from(v).max_element()}
#[no_mangle] pub extern "C" fn mid_u64vec2_element_sum(v:CU64Vec2)->u64{U64Vec2::from(v).element_sum()}
#[no_mangle] pub extern "C" fn mid_u64vec2_wrapping_add(a:CU64Vec2,b:CU64Vec2)->CU64Vec2{U64Vec2::from(a).wrapping_add(U64Vec2::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_u64vec2_wrapping_sub(a:CU64Vec2,b:CU64Vec2)->CU64Vec2{U64Vec2::from(a).wrapping_sub(U64Vec2::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_u64vec2_saturating_add(a:CU64Vec2,b:CU64Vec2)->CU64Vec2{U64Vec2::from(a).saturating_add(U64Vec2::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_u64vec2_saturating_sub(a:CU64Vec2,b:CU64Vec2)->CU64Vec2{U64Vec2::from(a).saturating_sub(U64Vec2::from(b)).into()}

// ── Exports — U64Vec3 ─────────────────────────────────────────────────────────
#[no_mangle] pub extern "C" fn mid_u64vec3_new(x:u64,y:u64,z:u64)->CU64Vec3{U64Vec3::new(x,y,z).into()}
#[no_mangle] pub extern "C" fn mid_u64vec3_add(a:CU64Vec3,b:CU64Vec3)->CU64Vec3{(U64Vec3::from(a)+U64Vec3::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_u64vec3_sub(a:CU64Vec3,b:CU64Vec3)->CU64Vec3{(U64Vec3::from(a)-U64Vec3::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_u64vec3_mul(a:CU64Vec3,b:CU64Vec3)->CU64Vec3{(U64Vec3::from(a)*U64Vec3::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_u64vec3_scale(v:CU64Vec3,s:u64)->CU64Vec3{(U64Vec3::from(v)*s).into()}
#[no_mangle] pub extern "C" fn mid_u64vec3_dot(a:CU64Vec3,b:CU64Vec3)->u64{U64Vec3::from(a).dot(U64Vec3::from(b))}
#[no_mangle] pub extern "C" fn mid_u64vec3_cross(a:CU64Vec3,b:CU64Vec3)->CU64Vec3{U64Vec3::from(a).cross(U64Vec3::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_u64vec3_min(a:CU64Vec3,b:CU64Vec3)->CU64Vec3{U64Vec3::from(a).min(U64Vec3::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_u64vec3_max(a:CU64Vec3,b:CU64Vec3)->CU64Vec3{U64Vec3::from(a).max(U64Vec3::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_u64vec3_clamp(v:CU64Vec3,lo:CU64Vec3,hi:CU64Vec3)->CU64Vec3{U64Vec3::from(v).clamp(U64Vec3::from(lo),U64Vec3::from(hi)).into()}
#[no_mangle] pub extern "C" fn mid_u64vec3_length_sq(v:CU64Vec3)->u64{U64Vec3::from(v).length_sq()}
#[no_mangle] pub extern "C" fn mid_u64vec3_min_element(v:CU64Vec3)->u64{U64Vec3::from(v).min_element()}
#[no_mangle] pub extern "C" fn mid_u64vec3_max_element(v:CU64Vec3)->u64{U64Vec3::from(v).max_element()}
#[no_mangle] pub extern "C" fn mid_u64vec3_element_sum(v:CU64Vec3)->u64{U64Vec3::from(v).element_sum()}
#[no_mangle] pub extern "C" fn mid_u64vec3_wrapping_add(a:CU64Vec3,b:CU64Vec3)->CU64Vec3{U64Vec3::from(a).wrapping_add(U64Vec3::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_u64vec3_wrapping_sub(a:CU64Vec3,b:CU64Vec3)->CU64Vec3{U64Vec3::from(a).wrapping_sub(U64Vec3::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_u64vec3_saturating_add(a:CU64Vec3,b:CU64Vec3)->CU64Vec3{U64Vec3::from(a).saturating_add(U64Vec3::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_u64vec3_saturating_sub(a:CU64Vec3,b:CU64Vec3)->CU64Vec3{U64Vec3::from(a).saturating_sub(U64Vec3::from(b)).into()}

// ── Exports — U64Vec4 ─────────────────────────────────────────────────────────
#[no_mangle] pub extern "C" fn mid_u64vec4_new(x:u64,y:u64,z:u64,w:u64)->CU64Vec4{U64Vec4::new(x,y,z,w).into()}
#[no_mangle] pub extern "C" fn mid_u64vec4_add(a:CU64Vec4,b:CU64Vec4)->CU64Vec4{(U64Vec4::from(a)+U64Vec4::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_u64vec4_sub(a:CU64Vec4,b:CU64Vec4)->CU64Vec4{(U64Vec4::from(a)-U64Vec4::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_u64vec4_mul(a:CU64Vec4,b:CU64Vec4)->CU64Vec4{(U64Vec4::from(a)*U64Vec4::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_u64vec4_scale(v:CU64Vec4,s:u64)->CU64Vec4{(U64Vec4::from(v)*s).into()}
#[no_mangle] pub extern "C" fn mid_u64vec4_dot(a:CU64Vec4,b:CU64Vec4)->u64{U64Vec4::from(a).dot(U64Vec4::from(b))}
#[no_mangle] pub extern "C" fn mid_u64vec4_min(a:CU64Vec4,b:CU64Vec4)->CU64Vec4{U64Vec4::from(a).min(U64Vec4::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_u64vec4_max(a:CU64Vec4,b:CU64Vec4)->CU64Vec4{U64Vec4::from(a).max(U64Vec4::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_u64vec4_clamp(v:CU64Vec4,lo:CU64Vec4,hi:CU64Vec4)->CU64Vec4{U64Vec4::from(v).clamp(U64Vec4::from(lo),U64Vec4::from(hi)).into()}
#[no_mangle] pub extern "C" fn mid_u64vec4_length_sq(v:CU64Vec4)->u64{U64Vec4::from(v).length_sq()}
#[no_mangle] pub extern "C" fn mid_u64vec4_min_element(v:CU64Vec4)->u64{U64Vec4::from(v).min_element()}
#[no_mangle] pub extern "C" fn mid_u64vec4_max_element(v:CU64Vec4)->u64{U64Vec4::from(v).max_element()}
#[no_mangle] pub extern "C" fn mid_u64vec4_element_sum(v:CU64Vec4)->u64{U64Vec4::from(v).element_sum()}
#[no_mangle] pub extern "C" fn mid_u64vec4_wrapping_add(a:CU64Vec4,b:CU64Vec4)->CU64Vec4{U64Vec4::from(a).wrapping_add(U64Vec4::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_u64vec4_wrapping_sub(a:CU64Vec4,b:CU64Vec4)->CU64Vec4{U64Vec4::from(a).wrapping_sub(U64Vec4::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_u64vec4_saturating_add(a:CU64Vec4,b:CU64Vec4)->CU64Vec4{U64Vec4::from(a).saturating_add(U64Vec4::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_u64vec4_saturating_sub(a:CU64Vec4,b:CU64Vec4)->CU64Vec4{U64Vec4::from(a).saturating_sub(U64Vec4::from(b)).into()}
