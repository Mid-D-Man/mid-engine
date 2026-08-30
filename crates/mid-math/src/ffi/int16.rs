// crates/mid-math/src/ffi/int16.rs
//! C-ABI types and #[no_mangle] exports for i16/u16 integer vector types.
//!
//! Mirrors ffi/int32.rs's curated subset (not the full Rust API — no
//! checked_*/wrapping_mul/signum/select/splat/as_XXX conversions), plus
//! the 6 cmpXX comparisons (-> BVec2/3/4, matching what int32.rs's own
//! FFI does NOT yet expose — see ffi/bvec.rs for the BVec2/3/4 type itself).
//! dot/length_sq/distance_sq widen one step (i16->i32, u16->u32) to
//! match the underlying Rust methods' own overflow-safety widening —
//! verified directly against i16vec*.rs/u16vec*.rs source, not assumed
//! from the i32/u32 FFI pattern (which does NOT widen, since i32 dot doesn't
//! need to). Two real asymmetries also verified against source and encoded
//! here, not assumed: u16vec3 has no cross() (unlike UVec3/u32, which
//! does); u16vec4 has no distance_sq() (unlike every other type here).

use crate::{I16Vec2, I16Vec3, I16Vec4, U16Vec2, U16Vec3, U16Vec4, BVec2, BVec3, BVec4};

// =============================================================================
//  C types
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)] #[repr(C)]
pub struct CI16Vec2 { pub x: i16, pub y: i16 }
impl From<I16Vec2>  for CI16Vec2 { #[inline(always)] fn from(v: I16Vec2)  -> Self { Self { x: v.x, y: v.y } } }
impl From<CI16Vec2> for I16Vec2  { #[inline(always)] fn from(v: CI16Vec2) -> Self { I16Vec2::new(v.x, v.y) } }

#[derive(Debug, Clone, Copy, PartialEq, Eq)] #[repr(C)]
pub struct CI16Vec3 { pub x: i16, pub y: i16, pub z: i16 }
impl From<I16Vec3>  for CI16Vec3 { #[inline(always)] fn from(v: I16Vec3)  -> Self { Self { x: v.x, y: v.y, z: v.z } } }
impl From<CI16Vec3> for I16Vec3  { #[inline(always)] fn from(v: CI16Vec3) -> Self { I16Vec3::new(v.x, v.y, v.z) } }

#[derive(Debug, Clone, Copy, PartialEq, Eq)] #[repr(C)]
pub struct CI16Vec4 { pub x: i16, pub y: i16, pub z: i16, pub w: i16 }
impl From<I16Vec4>  for CI16Vec4 { #[inline(always)] fn from(v: I16Vec4)  -> Self { Self { x: v.x, y: v.y, z: v.z, w: v.w } } }
impl From<CI16Vec4> for I16Vec4  { #[inline(always)] fn from(v: CI16Vec4) -> Self { I16Vec4::new(v.x, v.y, v.z, v.w) } }

#[derive(Debug, Clone, Copy, PartialEq, Eq)] #[repr(C)]
pub struct CU16Vec2 { pub x: u16, pub y: u16 }
impl From<U16Vec2>  for CU16Vec2 { #[inline(always)] fn from(v: U16Vec2)  -> Self { Self { x: v.x, y: v.y } } }
impl From<CU16Vec2> for U16Vec2  { #[inline(always)] fn from(v: CU16Vec2) -> Self { U16Vec2::new(v.x, v.y) } }

#[derive(Debug, Clone, Copy, PartialEq, Eq)] #[repr(C)]
pub struct CU16Vec3 { pub x: u16, pub y: u16, pub z: u16 }
impl From<U16Vec3>  for CU16Vec3 { #[inline(always)] fn from(v: U16Vec3)  -> Self { Self { x: v.x, y: v.y, z: v.z } } }
impl From<CU16Vec3> for U16Vec3  { #[inline(always)] fn from(v: CU16Vec3) -> Self { U16Vec3::new(v.x, v.y, v.z) } }

#[derive(Debug, Clone, Copy, PartialEq, Eq)] #[repr(C)]
pub struct CU16Vec4 { pub x: u16, pub y: u16, pub z: u16, pub w: u16 }
impl From<U16Vec4>  for CU16Vec4 { #[inline(always)] fn from(v: U16Vec4)  -> Self { Self { x: v.x, y: v.y, z: v.z, w: v.w } } }
impl From<CU16Vec4> for U16Vec4  { #[inline(always)] fn from(v: CU16Vec4) -> Self { U16Vec4::new(v.x, v.y, v.z, v.w) } }

// =============================================================================
//  Exports
// =============================================================================

// ── Exports — I16Vec2 ───────────────────────────────────────────────────────────────
#[no_mangle] pub extern "C" fn mid_i16vec2_new(x:i16, y:i16)->CI16Vec2{I16Vec2::new(x, y).into()}
#[no_mangle] pub extern "C" fn mid_i16vec2_add(a:CI16Vec2,b:CI16Vec2)->CI16Vec2{(I16Vec2::from(a)+I16Vec2::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_i16vec2_sub(a:CI16Vec2,b:CI16Vec2)->CI16Vec2{(I16Vec2::from(a)-I16Vec2::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_i16vec2_mul(a:CI16Vec2,b:CI16Vec2)->CI16Vec2{(I16Vec2::from(a)*I16Vec2::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_i16vec2_scale(v:CI16Vec2,s:i16)->CI16Vec2{(I16Vec2::from(v)*s).into()}
#[no_mangle] pub extern "C" fn mid_i16vec2_dot(a:CI16Vec2,b:CI16Vec2)->i32{I16Vec2::from(a).dot(I16Vec2::from(b))}
#[no_mangle] pub extern "C" fn mid_i16vec2_min(a:CI16Vec2,b:CI16Vec2)->CI16Vec2{I16Vec2::from(a).min(I16Vec2::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_i16vec2_max(a:CI16Vec2,b:CI16Vec2)->CI16Vec2{I16Vec2::from(a).max(I16Vec2::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_i16vec2_clamp(v:CI16Vec2,lo:CI16Vec2,hi:CI16Vec2)->CI16Vec2{I16Vec2::from(v).clamp(I16Vec2::from(lo),I16Vec2::from(hi)).into()}
#[no_mangle] pub extern "C" fn mid_i16vec2_abs(v:CI16Vec2)->CI16Vec2{I16Vec2::from(v).abs().into()}
#[no_mangle] pub extern "C" fn mid_i16vec2_neg(v:CI16Vec2)->CI16Vec2{(-I16Vec2::from(v)).into()}
#[no_mangle] pub extern "C" fn mid_i16vec2_length_sq(v:CI16Vec2)->i32{I16Vec2::from(v).length_sq()}
#[no_mangle] pub extern "C" fn mid_i16vec2_distance_sq(a:CI16Vec2,b:CI16Vec2)->i32{I16Vec2::from(a).distance_sq(I16Vec2::from(b))}
#[no_mangle] pub extern "C" fn mid_i16vec2_min_element(v:CI16Vec2)->i16{I16Vec2::from(v).min_element()}
#[no_mangle] pub extern "C" fn mid_i16vec2_max_element(v:CI16Vec2)->i16{I16Vec2::from(v).max_element()}
#[no_mangle] pub extern "C" fn mid_i16vec2_element_sum(v:CI16Vec2)->i16{I16Vec2::from(v).element_sum()}
#[no_mangle] pub extern "C" fn mid_i16vec2_wrapping_add(a:CI16Vec2,b:CI16Vec2)->CI16Vec2{I16Vec2::from(a).wrapping_add(I16Vec2::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_i16vec2_wrapping_sub(a:CI16Vec2,b:CI16Vec2)->CI16Vec2{I16Vec2::from(a).wrapping_sub(I16Vec2::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_i16vec2_saturating_add(a:CI16Vec2,b:CI16Vec2)->CI16Vec2{I16Vec2::from(a).saturating_add(I16Vec2::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_i16vec2_saturating_sub(a:CI16Vec2,b:CI16Vec2)->CI16Vec2{I16Vec2::from(a).saturating_sub(I16Vec2::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_i16vec2_cmpeq(a:CI16Vec2,b:CI16Vec2)->BVec2{I16Vec2::from(a).cmpeq(I16Vec2::from(b))}
#[no_mangle] pub extern "C" fn mid_i16vec2_cmpne(a:CI16Vec2,b:CI16Vec2)->BVec2{I16Vec2::from(a).cmpne(I16Vec2::from(b))}
#[no_mangle] pub extern "C" fn mid_i16vec2_cmpge(a:CI16Vec2,b:CI16Vec2)->BVec2{I16Vec2::from(a).cmpge(I16Vec2::from(b))}
#[no_mangle] pub extern "C" fn mid_i16vec2_cmpgt(a:CI16Vec2,b:CI16Vec2)->BVec2{I16Vec2::from(a).cmpgt(I16Vec2::from(b))}
#[no_mangle] pub extern "C" fn mid_i16vec2_cmple(a:CI16Vec2,b:CI16Vec2)->BVec2{I16Vec2::from(a).cmple(I16Vec2::from(b))}
#[no_mangle] pub extern "C" fn mid_i16vec2_cmplt(a:CI16Vec2,b:CI16Vec2)->BVec2{I16Vec2::from(a).cmplt(I16Vec2::from(b))}

// ── Exports — I16Vec3 ───────────────────────────────────────────────────────────────
#[no_mangle] pub extern "C" fn mid_i16vec3_new(x:i16, y:i16, z:i16)->CI16Vec3{I16Vec3::new(x, y, z).into()}
#[no_mangle] pub extern "C" fn mid_i16vec3_add(a:CI16Vec3,b:CI16Vec3)->CI16Vec3{(I16Vec3::from(a)+I16Vec3::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_i16vec3_sub(a:CI16Vec3,b:CI16Vec3)->CI16Vec3{(I16Vec3::from(a)-I16Vec3::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_i16vec3_mul(a:CI16Vec3,b:CI16Vec3)->CI16Vec3{(I16Vec3::from(a)*I16Vec3::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_i16vec3_scale(v:CI16Vec3,s:i16)->CI16Vec3{(I16Vec3::from(v)*s).into()}
#[no_mangle] pub extern "C" fn mid_i16vec3_dot(a:CI16Vec3,b:CI16Vec3)->i32{I16Vec3::from(a).dot(I16Vec3::from(b))}
#[no_mangle] pub extern "C" fn mid_i16vec3_cross(a:CI16Vec3,b:CI16Vec3)->CI16Vec3{I16Vec3::from(a).cross(I16Vec3::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_i16vec3_min(a:CI16Vec3,b:CI16Vec3)->CI16Vec3{I16Vec3::from(a).min(I16Vec3::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_i16vec3_max(a:CI16Vec3,b:CI16Vec3)->CI16Vec3{I16Vec3::from(a).max(I16Vec3::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_i16vec3_clamp(v:CI16Vec3,lo:CI16Vec3,hi:CI16Vec3)->CI16Vec3{I16Vec3::from(v).clamp(I16Vec3::from(lo),I16Vec3::from(hi)).into()}
#[no_mangle] pub extern "C" fn mid_i16vec3_abs(v:CI16Vec3)->CI16Vec3{I16Vec3::from(v).abs().into()}
#[no_mangle] pub extern "C" fn mid_i16vec3_neg(v:CI16Vec3)->CI16Vec3{(-I16Vec3::from(v)).into()}
#[no_mangle] pub extern "C" fn mid_i16vec3_length_sq(v:CI16Vec3)->i32{I16Vec3::from(v).length_sq()}
#[no_mangle] pub extern "C" fn mid_i16vec3_distance_sq(a:CI16Vec3,b:CI16Vec3)->i32{I16Vec3::from(a).distance_sq(I16Vec3::from(b))}
#[no_mangle] pub extern "C" fn mid_i16vec3_min_element(v:CI16Vec3)->i16{I16Vec3::from(v).min_element()}
#[no_mangle] pub extern "C" fn mid_i16vec3_max_element(v:CI16Vec3)->i16{I16Vec3::from(v).max_element()}
#[no_mangle] pub extern "C" fn mid_i16vec3_element_sum(v:CI16Vec3)->i16{I16Vec3::from(v).element_sum()}
#[no_mangle] pub extern "C" fn mid_i16vec3_wrapping_add(a:CI16Vec3,b:CI16Vec3)->CI16Vec3{I16Vec3::from(a).wrapping_add(I16Vec3::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_i16vec3_wrapping_sub(a:CI16Vec3,b:CI16Vec3)->CI16Vec3{I16Vec3::from(a).wrapping_sub(I16Vec3::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_i16vec3_saturating_add(a:CI16Vec3,b:CI16Vec3)->CI16Vec3{I16Vec3::from(a).saturating_add(I16Vec3::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_i16vec3_saturating_sub(a:CI16Vec3,b:CI16Vec3)->CI16Vec3{I16Vec3::from(a).saturating_sub(I16Vec3::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_i16vec3_cmpeq(a:CI16Vec3,b:CI16Vec3)->BVec3{I16Vec3::from(a).cmpeq(I16Vec3::from(b))}
#[no_mangle] pub extern "C" fn mid_i16vec3_cmpne(a:CI16Vec3,b:CI16Vec3)->BVec3{I16Vec3::from(a).cmpne(I16Vec3::from(b))}
#[no_mangle] pub extern "C" fn mid_i16vec3_cmpge(a:CI16Vec3,b:CI16Vec3)->BVec3{I16Vec3::from(a).cmpge(I16Vec3::from(b))}
#[no_mangle] pub extern "C" fn mid_i16vec3_cmpgt(a:CI16Vec3,b:CI16Vec3)->BVec3{I16Vec3::from(a).cmpgt(I16Vec3::from(b))}
#[no_mangle] pub extern "C" fn mid_i16vec3_cmple(a:CI16Vec3,b:CI16Vec3)->BVec3{I16Vec3::from(a).cmple(I16Vec3::from(b))}
#[no_mangle] pub extern "C" fn mid_i16vec3_cmplt(a:CI16Vec3,b:CI16Vec3)->BVec3{I16Vec3::from(a).cmplt(I16Vec3::from(b))}

// ── Exports — I16Vec4 ───────────────────────────────────────────────────────────────
#[no_mangle] pub extern "C" fn mid_i16vec4_new(x:i16, y:i16, z:i16, w:i16)->CI16Vec4{I16Vec4::new(x, y, z, w).into()}
#[no_mangle] pub extern "C" fn mid_i16vec4_add(a:CI16Vec4,b:CI16Vec4)->CI16Vec4{(I16Vec4::from(a)+I16Vec4::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_i16vec4_sub(a:CI16Vec4,b:CI16Vec4)->CI16Vec4{(I16Vec4::from(a)-I16Vec4::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_i16vec4_mul(a:CI16Vec4,b:CI16Vec4)->CI16Vec4{(I16Vec4::from(a)*I16Vec4::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_i16vec4_scale(v:CI16Vec4,s:i16)->CI16Vec4{(I16Vec4::from(v)*s).into()}
#[no_mangle] pub extern "C" fn mid_i16vec4_dot(a:CI16Vec4,b:CI16Vec4)->i32{I16Vec4::from(a).dot(I16Vec4::from(b))}
#[no_mangle] pub extern "C" fn mid_i16vec4_min(a:CI16Vec4,b:CI16Vec4)->CI16Vec4{I16Vec4::from(a).min(I16Vec4::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_i16vec4_max(a:CI16Vec4,b:CI16Vec4)->CI16Vec4{I16Vec4::from(a).max(I16Vec4::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_i16vec4_clamp(v:CI16Vec4,lo:CI16Vec4,hi:CI16Vec4)->CI16Vec4{I16Vec4::from(v).clamp(I16Vec4::from(lo),I16Vec4::from(hi)).into()}
#[no_mangle] pub extern "C" fn mid_i16vec4_abs(v:CI16Vec4)->CI16Vec4{I16Vec4::from(v).abs().into()}
#[no_mangle] pub extern "C" fn mid_i16vec4_neg(v:CI16Vec4)->CI16Vec4{(-I16Vec4::from(v)).into()}
#[no_mangle] pub extern "C" fn mid_i16vec4_length_sq(v:CI16Vec4)->i32{I16Vec4::from(v).length_sq()}
#[no_mangle] pub extern "C" fn mid_i16vec4_distance_sq(a:CI16Vec4,b:CI16Vec4)->i32{I16Vec4::from(a).distance_sq(I16Vec4::from(b))}
#[no_mangle] pub extern "C" fn mid_i16vec4_min_element(v:CI16Vec4)->i16{I16Vec4::from(v).min_element()}
#[no_mangle] pub extern "C" fn mid_i16vec4_max_element(v:CI16Vec4)->i16{I16Vec4::from(v).max_element()}
#[no_mangle] pub extern "C" fn mid_i16vec4_element_sum(v:CI16Vec4)->i16{I16Vec4::from(v).element_sum()}
#[no_mangle] pub extern "C" fn mid_i16vec4_wrapping_add(a:CI16Vec4,b:CI16Vec4)->CI16Vec4{I16Vec4::from(a).wrapping_add(I16Vec4::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_i16vec4_wrapping_sub(a:CI16Vec4,b:CI16Vec4)->CI16Vec4{I16Vec4::from(a).wrapping_sub(I16Vec4::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_i16vec4_saturating_add(a:CI16Vec4,b:CI16Vec4)->CI16Vec4{I16Vec4::from(a).saturating_add(I16Vec4::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_i16vec4_saturating_sub(a:CI16Vec4,b:CI16Vec4)->CI16Vec4{I16Vec4::from(a).saturating_sub(I16Vec4::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_i16vec4_cmpeq(a:CI16Vec4,b:CI16Vec4)->BVec4{I16Vec4::from(a).cmpeq(I16Vec4::from(b))}
#[no_mangle] pub extern "C" fn mid_i16vec4_cmpne(a:CI16Vec4,b:CI16Vec4)->BVec4{I16Vec4::from(a).cmpne(I16Vec4::from(b))}
#[no_mangle] pub extern "C" fn mid_i16vec4_cmpge(a:CI16Vec4,b:CI16Vec4)->BVec4{I16Vec4::from(a).cmpge(I16Vec4::from(b))}
#[no_mangle] pub extern "C" fn mid_i16vec4_cmpgt(a:CI16Vec4,b:CI16Vec4)->BVec4{I16Vec4::from(a).cmpgt(I16Vec4::from(b))}
#[no_mangle] pub extern "C" fn mid_i16vec4_cmple(a:CI16Vec4,b:CI16Vec4)->BVec4{I16Vec4::from(a).cmple(I16Vec4::from(b))}
#[no_mangle] pub extern "C" fn mid_i16vec4_cmplt(a:CI16Vec4,b:CI16Vec4)->BVec4{I16Vec4::from(a).cmplt(I16Vec4::from(b))}

// ── Exports — U16Vec2 ───────────────────────────────────────────────────────────────
#[no_mangle] pub extern "C" fn mid_u16vec2_new(x:u16, y:u16)->CU16Vec2{U16Vec2::new(x, y).into()}
#[no_mangle] pub extern "C" fn mid_u16vec2_add(a:CU16Vec2,b:CU16Vec2)->CU16Vec2{(U16Vec2::from(a)+U16Vec2::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_u16vec2_sub(a:CU16Vec2,b:CU16Vec2)->CU16Vec2{(U16Vec2::from(a)-U16Vec2::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_u16vec2_mul(a:CU16Vec2,b:CU16Vec2)->CU16Vec2{(U16Vec2::from(a)*U16Vec2::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_u16vec2_scale(v:CU16Vec2,s:u16)->CU16Vec2{(U16Vec2::from(v)*s).into()}
#[no_mangle] pub extern "C" fn mid_u16vec2_dot(a:CU16Vec2,b:CU16Vec2)->u32{U16Vec2::from(a).dot(U16Vec2::from(b))}
#[no_mangle] pub extern "C" fn mid_u16vec2_min(a:CU16Vec2,b:CU16Vec2)->CU16Vec2{U16Vec2::from(a).min(U16Vec2::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_u16vec2_max(a:CU16Vec2,b:CU16Vec2)->CU16Vec2{U16Vec2::from(a).max(U16Vec2::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_u16vec2_clamp(v:CU16Vec2,lo:CU16Vec2,hi:CU16Vec2)->CU16Vec2{U16Vec2::from(v).clamp(U16Vec2::from(lo),U16Vec2::from(hi)).into()}
#[no_mangle] pub extern "C" fn mid_u16vec2_length_sq(v:CU16Vec2)->u32{U16Vec2::from(v).length_sq()}
#[no_mangle] pub extern "C" fn mid_u16vec2_distance_sq(a:CU16Vec2,b:CU16Vec2)->u32{U16Vec2::from(a).distance_sq(U16Vec2::from(b))}
#[no_mangle] pub extern "C" fn mid_u16vec2_min_element(v:CU16Vec2)->u16{U16Vec2::from(v).min_element()}
#[no_mangle] pub extern "C" fn mid_u16vec2_max_element(v:CU16Vec2)->u16{U16Vec2::from(v).max_element()}
#[no_mangle] pub extern "C" fn mid_u16vec2_element_sum(v:CU16Vec2)->u16{U16Vec2::from(v).element_sum()}
#[no_mangle] pub extern "C" fn mid_u16vec2_wrapping_add(a:CU16Vec2,b:CU16Vec2)->CU16Vec2{U16Vec2::from(a).wrapping_add(U16Vec2::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_u16vec2_wrapping_sub(a:CU16Vec2,b:CU16Vec2)->CU16Vec2{U16Vec2::from(a).wrapping_sub(U16Vec2::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_u16vec2_saturating_add(a:CU16Vec2,b:CU16Vec2)->CU16Vec2{U16Vec2::from(a).saturating_add(U16Vec2::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_u16vec2_saturating_sub(a:CU16Vec2,b:CU16Vec2)->CU16Vec2{U16Vec2::from(a).saturating_sub(U16Vec2::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_u16vec2_cmpeq(a:CU16Vec2,b:CU16Vec2)->BVec2{U16Vec2::from(a).cmpeq(U16Vec2::from(b))}
#[no_mangle] pub extern "C" fn mid_u16vec2_cmpne(a:CU16Vec2,b:CU16Vec2)->BVec2{U16Vec2::from(a).cmpne(U16Vec2::from(b))}
#[no_mangle] pub extern "C" fn mid_u16vec2_cmpge(a:CU16Vec2,b:CU16Vec2)->BVec2{U16Vec2::from(a).cmpge(U16Vec2::from(b))}
#[no_mangle] pub extern "C" fn mid_u16vec2_cmpgt(a:CU16Vec2,b:CU16Vec2)->BVec2{U16Vec2::from(a).cmpgt(U16Vec2::from(b))}
#[no_mangle] pub extern "C" fn mid_u16vec2_cmple(a:CU16Vec2,b:CU16Vec2)->BVec2{U16Vec2::from(a).cmple(U16Vec2::from(b))}
#[no_mangle] pub extern "C" fn mid_u16vec2_cmplt(a:CU16Vec2,b:CU16Vec2)->BVec2{U16Vec2::from(a).cmplt(U16Vec2::from(b))}

// ── Exports — U16Vec3 ───────────────────────────────────────────────────────────────
#[no_mangle] pub extern "C" fn mid_u16vec3_new(x:u16, y:u16, z:u16)->CU16Vec3{U16Vec3::new(x, y, z).into()}
#[no_mangle] pub extern "C" fn mid_u16vec3_add(a:CU16Vec3,b:CU16Vec3)->CU16Vec3{(U16Vec3::from(a)+U16Vec3::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_u16vec3_sub(a:CU16Vec3,b:CU16Vec3)->CU16Vec3{(U16Vec3::from(a)-U16Vec3::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_u16vec3_mul(a:CU16Vec3,b:CU16Vec3)->CU16Vec3{(U16Vec3::from(a)*U16Vec3::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_u16vec3_scale(v:CU16Vec3,s:u16)->CU16Vec3{(U16Vec3::from(v)*s).into()}
#[no_mangle] pub extern "C" fn mid_u16vec3_dot(a:CU16Vec3,b:CU16Vec3)->u32{U16Vec3::from(a).dot(U16Vec3::from(b))}
#[no_mangle] pub extern "C" fn mid_u16vec3_min(a:CU16Vec3,b:CU16Vec3)->CU16Vec3{U16Vec3::from(a).min(U16Vec3::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_u16vec3_max(a:CU16Vec3,b:CU16Vec3)->CU16Vec3{U16Vec3::from(a).max(U16Vec3::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_u16vec3_clamp(v:CU16Vec3,lo:CU16Vec3,hi:CU16Vec3)->CU16Vec3{U16Vec3::from(v).clamp(U16Vec3::from(lo),U16Vec3::from(hi)).into()}
#[no_mangle] pub extern "C" fn mid_u16vec3_length_sq(v:CU16Vec3)->u32{U16Vec3::from(v).length_sq()}
#[no_mangle] pub extern "C" fn mid_u16vec3_distance_sq(a:CU16Vec3,b:CU16Vec3)->u32{U16Vec3::from(a).distance_sq(U16Vec3::from(b))}
#[no_mangle] pub extern "C" fn mid_u16vec3_min_element(v:CU16Vec3)->u16{U16Vec3::from(v).min_element()}
#[no_mangle] pub extern "C" fn mid_u16vec3_max_element(v:CU16Vec3)->u16{U16Vec3::from(v).max_element()}
#[no_mangle] pub extern "C" fn mid_u16vec3_element_sum(v:CU16Vec3)->u16{U16Vec3::from(v).element_sum()}
#[no_mangle] pub extern "C" fn mid_u16vec3_wrapping_add(a:CU16Vec3,b:CU16Vec3)->CU16Vec3{U16Vec3::from(a).wrapping_add(U16Vec3::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_u16vec3_wrapping_sub(a:CU16Vec3,b:CU16Vec3)->CU16Vec3{U16Vec3::from(a).wrapping_sub(U16Vec3::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_u16vec3_saturating_add(a:CU16Vec3,b:CU16Vec3)->CU16Vec3{U16Vec3::from(a).saturating_add(U16Vec3::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_u16vec3_saturating_sub(a:CU16Vec3,b:CU16Vec3)->CU16Vec3{U16Vec3::from(a).saturating_sub(U16Vec3::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_u16vec3_cmpeq(a:CU16Vec3,b:CU16Vec3)->BVec3{U16Vec3::from(a).cmpeq(U16Vec3::from(b))}
#[no_mangle] pub extern "C" fn mid_u16vec3_cmpne(a:CU16Vec3,b:CU16Vec3)->BVec3{U16Vec3::from(a).cmpne(U16Vec3::from(b))}
#[no_mangle] pub extern "C" fn mid_u16vec3_cmpge(a:CU16Vec3,b:CU16Vec3)->BVec3{U16Vec3::from(a).cmpge(U16Vec3::from(b))}
#[no_mangle] pub extern "C" fn mid_u16vec3_cmpgt(a:CU16Vec3,b:CU16Vec3)->BVec3{U16Vec3::from(a).cmpgt(U16Vec3::from(b))}
#[no_mangle] pub extern "C" fn mid_u16vec3_cmple(a:CU16Vec3,b:CU16Vec3)->BVec3{U16Vec3::from(a).cmple(U16Vec3::from(b))}
#[no_mangle] pub extern "C" fn mid_u16vec3_cmplt(a:CU16Vec3,b:CU16Vec3)->BVec3{U16Vec3::from(a).cmplt(U16Vec3::from(b))}

// ── Exports — U16Vec4 ───────────────────────────────────────────────────────────────
#[no_mangle] pub extern "C" fn mid_u16vec4_new(x:u16, y:u16, z:u16, w:u16)->CU16Vec4{U16Vec4::new(x, y, z, w).into()}
#[no_mangle] pub extern "C" fn mid_u16vec4_add(a:CU16Vec4,b:CU16Vec4)->CU16Vec4{(U16Vec4::from(a)+U16Vec4::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_u16vec4_sub(a:CU16Vec4,b:CU16Vec4)->CU16Vec4{(U16Vec4::from(a)-U16Vec4::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_u16vec4_mul(a:CU16Vec4,b:CU16Vec4)->CU16Vec4{(U16Vec4::from(a)*U16Vec4::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_u16vec4_scale(v:CU16Vec4,s:u16)->CU16Vec4{(U16Vec4::from(v)*s).into()}
#[no_mangle] pub extern "C" fn mid_u16vec4_dot(a:CU16Vec4,b:CU16Vec4)->u32{U16Vec4::from(a).dot(U16Vec4::from(b))}
#[no_mangle] pub extern "C" fn mid_u16vec4_min(a:CU16Vec4,b:CU16Vec4)->CU16Vec4{U16Vec4::from(a).min(U16Vec4::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_u16vec4_max(a:CU16Vec4,b:CU16Vec4)->CU16Vec4{U16Vec4::from(a).max(U16Vec4::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_u16vec4_clamp(v:CU16Vec4,lo:CU16Vec4,hi:CU16Vec4)->CU16Vec4{U16Vec4::from(v).clamp(U16Vec4::from(lo),U16Vec4::from(hi)).into()}
#[no_mangle] pub extern "C" fn mid_u16vec4_length_sq(v:CU16Vec4)->u32{U16Vec4::from(v).length_sq()}
#[no_mangle] pub extern "C" fn mid_u16vec4_min_element(v:CU16Vec4)->u16{U16Vec4::from(v).min_element()}
#[no_mangle] pub extern "C" fn mid_u16vec4_max_element(v:CU16Vec4)->u16{U16Vec4::from(v).max_element()}
#[no_mangle] pub extern "C" fn mid_u16vec4_element_sum(v:CU16Vec4)->u16{U16Vec4::from(v).element_sum()}
#[no_mangle] pub extern "C" fn mid_u16vec4_wrapping_add(a:CU16Vec4,b:CU16Vec4)->CU16Vec4{U16Vec4::from(a).wrapping_add(U16Vec4::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_u16vec4_wrapping_sub(a:CU16Vec4,b:CU16Vec4)->CU16Vec4{U16Vec4::from(a).wrapping_sub(U16Vec4::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_u16vec4_saturating_add(a:CU16Vec4,b:CU16Vec4)->CU16Vec4{U16Vec4::from(a).saturating_add(U16Vec4::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_u16vec4_saturating_sub(a:CU16Vec4,b:CU16Vec4)->CU16Vec4{U16Vec4::from(a).saturating_sub(U16Vec4::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_u16vec4_cmpeq(a:CU16Vec4,b:CU16Vec4)->BVec4{U16Vec4::from(a).cmpeq(U16Vec4::from(b))}
#[no_mangle] pub extern "C" fn mid_u16vec4_cmpne(a:CU16Vec4,b:CU16Vec4)->BVec4{U16Vec4::from(a).cmpne(U16Vec4::from(b))}
#[no_mangle] pub extern "C" fn mid_u16vec4_cmpge(a:CU16Vec4,b:CU16Vec4)->BVec4{U16Vec4::from(a).cmpge(U16Vec4::from(b))}
#[no_mangle] pub extern "C" fn mid_u16vec4_cmpgt(a:CU16Vec4,b:CU16Vec4)->BVec4{U16Vec4::from(a).cmpgt(U16Vec4::from(b))}
#[no_mangle] pub extern "C" fn mid_u16vec4_cmple(a:CU16Vec4,b:CU16Vec4)->BVec4{U16Vec4::from(a).cmple(U16Vec4::from(b))}
#[no_mangle] pub extern "C" fn mid_u16vec4_cmplt(a:CU16Vec4,b:CU16Vec4)->BVec4{U16Vec4::from(a).cmplt(U16Vec4::from(b))}
