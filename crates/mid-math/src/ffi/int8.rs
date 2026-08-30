// crates/mid-math/src/ffi/int8.rs
//! C-ABI types and #[no_mangle] exports for i8/u8 integer vector types.
//!
//! Mirrors ffi/int32.rs's curated subset (not the full Rust API — no
//! checked_*/wrapping_mul/signum/select/splat/as_XXX conversions), plus
//! the 6 cmpXX comparisons (-> BVec2/3/4, matching what int32.rs's own
//! FFI does NOT yet expose — see ffi/bvec.rs for the BVec2/3/4 type itself).
//! dot/length_sq/distance_sq widen one step (i8->i16, u8->u16) to
//! match the underlying Rust methods' own overflow-safety widening —
//! verified directly against i8vec*.rs/u8vec*.rs source, not assumed
//! from the i32/u32 FFI pattern (which does NOT widen, since i32 dot doesn't
//! need to). Two real asymmetries also verified against source and encoded
//! here, not assumed: u8vec3 has no cross() (unlike UVec3/u32, which
//! does); u8vec4 has no distance_sq() (unlike every other type here).

use crate::{I8Vec2, I8Vec3, I8Vec4, U8Vec2, U8Vec3, U8Vec4, BVec2, BVec3, BVec4};

// =============================================================================
//  C types
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)] #[repr(C)]
pub struct CI8Vec2 { pub x: i8, pub y: i8 }
impl From<I8Vec2>  for CI8Vec2 { #[inline(always)] fn from(v: I8Vec2)  -> Self { Self { x: v.x, y: v.y } } }
impl From<CI8Vec2> for I8Vec2  { #[inline(always)] fn from(v: CI8Vec2) -> Self { I8Vec2::new(v.x, v.y) } }

#[derive(Debug, Clone, Copy, PartialEq, Eq)] #[repr(C)]
pub struct CI8Vec3 { pub x: i8, pub y: i8, pub z: i8 }
impl From<I8Vec3>  for CI8Vec3 { #[inline(always)] fn from(v: I8Vec3)  -> Self { Self { x: v.x, y: v.y, z: v.z } } }
impl From<CI8Vec3> for I8Vec3  { #[inline(always)] fn from(v: CI8Vec3) -> Self { I8Vec3::new(v.x, v.y, v.z) } }

#[derive(Debug, Clone, Copy, PartialEq, Eq)] #[repr(C)]
pub struct CI8Vec4 { pub x: i8, pub y: i8, pub z: i8, pub w: i8 }
impl From<I8Vec4>  for CI8Vec4 { #[inline(always)] fn from(v: I8Vec4)  -> Self { Self { x: v.x, y: v.y, z: v.z, w: v.w } } }
impl From<CI8Vec4> for I8Vec4  { #[inline(always)] fn from(v: CI8Vec4) -> Self { I8Vec4::new(v.x, v.y, v.z, v.w) } }

#[derive(Debug, Clone, Copy, PartialEq, Eq)] #[repr(C)]
pub struct CU8Vec2 { pub x: u8, pub y: u8 }
impl From<U8Vec2>  for CU8Vec2 { #[inline(always)] fn from(v: U8Vec2)  -> Self { Self { x: v.x, y: v.y } } }
impl From<CU8Vec2> for U8Vec2  { #[inline(always)] fn from(v: CU8Vec2) -> Self { U8Vec2::new(v.x, v.y) } }

#[derive(Debug, Clone, Copy, PartialEq, Eq)] #[repr(C)]
pub struct CU8Vec3 { pub x: u8, pub y: u8, pub z: u8 }
impl From<U8Vec3>  for CU8Vec3 { #[inline(always)] fn from(v: U8Vec3)  -> Self { Self { x: v.x, y: v.y, z: v.z } } }
impl From<CU8Vec3> for U8Vec3  { #[inline(always)] fn from(v: CU8Vec3) -> Self { U8Vec3::new(v.x, v.y, v.z) } }

#[derive(Debug, Clone, Copy, PartialEq, Eq)] #[repr(C)]
pub struct CU8Vec4 { pub x: u8, pub y: u8, pub z: u8, pub w: u8 }
impl From<U8Vec4>  for CU8Vec4 { #[inline(always)] fn from(v: U8Vec4)  -> Self { Self { x: v.x, y: v.y, z: v.z, w: v.w } } }
impl From<CU8Vec4> for U8Vec4  { #[inline(always)] fn from(v: CU8Vec4) -> Self { U8Vec4::new(v.x, v.y, v.z, v.w) } }

// =============================================================================
//  Exports
// =============================================================================

// ── Exports — I8Vec2 ────────────────────────────────────────────────────────────────
#[no_mangle] pub extern "C" fn mid_i8vec2_new(x:i8, y:i8)->CI8Vec2{I8Vec2::new(x, y).into()}
#[no_mangle] pub extern "C" fn mid_i8vec2_add(a:CI8Vec2,b:CI8Vec2)->CI8Vec2{(I8Vec2::from(a)+I8Vec2::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_i8vec2_sub(a:CI8Vec2,b:CI8Vec2)->CI8Vec2{(I8Vec2::from(a)-I8Vec2::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_i8vec2_mul(a:CI8Vec2,b:CI8Vec2)->CI8Vec2{(I8Vec2::from(a)*I8Vec2::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_i8vec2_scale(v:CI8Vec2,s:i8)->CI8Vec2{(I8Vec2::from(v)*s).into()}
#[no_mangle] pub extern "C" fn mid_i8vec2_dot(a:CI8Vec2,b:CI8Vec2)->i16{I8Vec2::from(a).dot(I8Vec2::from(b))}
#[no_mangle] pub extern "C" fn mid_i8vec2_min(a:CI8Vec2,b:CI8Vec2)->CI8Vec2{I8Vec2::from(a).min(I8Vec2::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_i8vec2_max(a:CI8Vec2,b:CI8Vec2)->CI8Vec2{I8Vec2::from(a).max(I8Vec2::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_i8vec2_clamp(v:CI8Vec2,lo:CI8Vec2,hi:CI8Vec2)->CI8Vec2{I8Vec2::from(v).clamp(I8Vec2::from(lo),I8Vec2::from(hi)).into()}
#[no_mangle] pub extern "C" fn mid_i8vec2_abs(v:CI8Vec2)->CI8Vec2{I8Vec2::from(v).abs().into()}
#[no_mangle] pub extern "C" fn mid_i8vec2_neg(v:CI8Vec2)->CI8Vec2{(-I8Vec2::from(v)).into()}
#[no_mangle] pub extern "C" fn mid_i8vec2_length_sq(v:CI8Vec2)->i16{I8Vec2::from(v).length_sq()}
#[no_mangle] pub extern "C" fn mid_i8vec2_distance_sq(a:CI8Vec2,b:CI8Vec2)->i16{I8Vec2::from(a).distance_sq(I8Vec2::from(b))}
#[no_mangle] pub extern "C" fn mid_i8vec2_min_element(v:CI8Vec2)->i8{I8Vec2::from(v).min_element()}
#[no_mangle] pub extern "C" fn mid_i8vec2_max_element(v:CI8Vec2)->i8{I8Vec2::from(v).max_element()}
#[no_mangle] pub extern "C" fn mid_i8vec2_element_sum(v:CI8Vec2)->i8{I8Vec2::from(v).element_sum()}
#[no_mangle] pub extern "C" fn mid_i8vec2_wrapping_add(a:CI8Vec2,b:CI8Vec2)->CI8Vec2{I8Vec2::from(a).wrapping_add(I8Vec2::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_i8vec2_wrapping_sub(a:CI8Vec2,b:CI8Vec2)->CI8Vec2{I8Vec2::from(a).wrapping_sub(I8Vec2::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_i8vec2_saturating_add(a:CI8Vec2,b:CI8Vec2)->CI8Vec2{I8Vec2::from(a).saturating_add(I8Vec2::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_i8vec2_saturating_sub(a:CI8Vec2,b:CI8Vec2)->CI8Vec2{I8Vec2::from(a).saturating_sub(I8Vec2::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_i8vec2_cmpeq(a:CI8Vec2,b:CI8Vec2)->BVec2{I8Vec2::from(a).cmpeq(I8Vec2::from(b))}
#[no_mangle] pub extern "C" fn mid_i8vec2_cmpne(a:CI8Vec2,b:CI8Vec2)->BVec2{I8Vec2::from(a).cmpne(I8Vec2::from(b))}
#[no_mangle] pub extern "C" fn mid_i8vec2_cmpge(a:CI8Vec2,b:CI8Vec2)->BVec2{I8Vec2::from(a).cmpge(I8Vec2::from(b))}
#[no_mangle] pub extern "C" fn mid_i8vec2_cmpgt(a:CI8Vec2,b:CI8Vec2)->BVec2{I8Vec2::from(a).cmpgt(I8Vec2::from(b))}
#[no_mangle] pub extern "C" fn mid_i8vec2_cmple(a:CI8Vec2,b:CI8Vec2)->BVec2{I8Vec2::from(a).cmple(I8Vec2::from(b))}
#[no_mangle] pub extern "C" fn mid_i8vec2_cmplt(a:CI8Vec2,b:CI8Vec2)->BVec2{I8Vec2::from(a).cmplt(I8Vec2::from(b))}

// ── Exports — I8Vec3 ────────────────────────────────────────────────────────────────
#[no_mangle] pub extern "C" fn mid_i8vec3_new(x:i8, y:i8, z:i8)->CI8Vec3{I8Vec3::new(x, y, z).into()}
#[no_mangle] pub extern "C" fn mid_i8vec3_add(a:CI8Vec3,b:CI8Vec3)->CI8Vec3{(I8Vec3::from(a)+I8Vec3::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_i8vec3_sub(a:CI8Vec3,b:CI8Vec3)->CI8Vec3{(I8Vec3::from(a)-I8Vec3::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_i8vec3_mul(a:CI8Vec3,b:CI8Vec3)->CI8Vec3{(I8Vec3::from(a)*I8Vec3::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_i8vec3_scale(v:CI8Vec3,s:i8)->CI8Vec3{(I8Vec3::from(v)*s).into()}
#[no_mangle] pub extern "C" fn mid_i8vec3_dot(a:CI8Vec3,b:CI8Vec3)->i16{I8Vec3::from(a).dot(I8Vec3::from(b))}
#[no_mangle] pub extern "C" fn mid_i8vec3_cross(a:CI8Vec3,b:CI8Vec3)->CI8Vec3{I8Vec3::from(a).cross(I8Vec3::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_i8vec3_min(a:CI8Vec3,b:CI8Vec3)->CI8Vec3{I8Vec3::from(a).min(I8Vec3::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_i8vec3_max(a:CI8Vec3,b:CI8Vec3)->CI8Vec3{I8Vec3::from(a).max(I8Vec3::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_i8vec3_clamp(v:CI8Vec3,lo:CI8Vec3,hi:CI8Vec3)->CI8Vec3{I8Vec3::from(v).clamp(I8Vec3::from(lo),I8Vec3::from(hi)).into()}
#[no_mangle] pub extern "C" fn mid_i8vec3_abs(v:CI8Vec3)->CI8Vec3{I8Vec3::from(v).abs().into()}
#[no_mangle] pub extern "C" fn mid_i8vec3_neg(v:CI8Vec3)->CI8Vec3{(-I8Vec3::from(v)).into()}
#[no_mangle] pub extern "C" fn mid_i8vec3_length_sq(v:CI8Vec3)->i16{I8Vec3::from(v).length_sq()}
#[no_mangle] pub extern "C" fn mid_i8vec3_distance_sq(a:CI8Vec3,b:CI8Vec3)->i16{I8Vec3::from(a).distance_sq(I8Vec3::from(b))}
#[no_mangle] pub extern "C" fn mid_i8vec3_min_element(v:CI8Vec3)->i8{I8Vec3::from(v).min_element()}
#[no_mangle] pub extern "C" fn mid_i8vec3_max_element(v:CI8Vec3)->i8{I8Vec3::from(v).max_element()}
#[no_mangle] pub extern "C" fn mid_i8vec3_element_sum(v:CI8Vec3)->i8{I8Vec3::from(v).element_sum()}
#[no_mangle] pub extern "C" fn mid_i8vec3_wrapping_add(a:CI8Vec3,b:CI8Vec3)->CI8Vec3{I8Vec3::from(a).wrapping_add(I8Vec3::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_i8vec3_wrapping_sub(a:CI8Vec3,b:CI8Vec3)->CI8Vec3{I8Vec3::from(a).wrapping_sub(I8Vec3::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_i8vec3_saturating_add(a:CI8Vec3,b:CI8Vec3)->CI8Vec3{I8Vec3::from(a).saturating_add(I8Vec3::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_i8vec3_saturating_sub(a:CI8Vec3,b:CI8Vec3)->CI8Vec3{I8Vec3::from(a).saturating_sub(I8Vec3::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_i8vec3_cmpeq(a:CI8Vec3,b:CI8Vec3)->BVec3{I8Vec3::from(a).cmpeq(I8Vec3::from(b))}
#[no_mangle] pub extern "C" fn mid_i8vec3_cmpne(a:CI8Vec3,b:CI8Vec3)->BVec3{I8Vec3::from(a).cmpne(I8Vec3::from(b))}
#[no_mangle] pub extern "C" fn mid_i8vec3_cmpge(a:CI8Vec3,b:CI8Vec3)->BVec3{I8Vec3::from(a).cmpge(I8Vec3::from(b))}
#[no_mangle] pub extern "C" fn mid_i8vec3_cmpgt(a:CI8Vec3,b:CI8Vec3)->BVec3{I8Vec3::from(a).cmpgt(I8Vec3::from(b))}
#[no_mangle] pub extern "C" fn mid_i8vec3_cmple(a:CI8Vec3,b:CI8Vec3)->BVec3{I8Vec3::from(a).cmple(I8Vec3::from(b))}
#[no_mangle] pub extern "C" fn mid_i8vec3_cmplt(a:CI8Vec3,b:CI8Vec3)->BVec3{I8Vec3::from(a).cmplt(I8Vec3::from(b))}

// ── Exports — I8Vec4 ────────────────────────────────────────────────────────────────
#[no_mangle] pub extern "C" fn mid_i8vec4_new(x:i8, y:i8, z:i8, w:i8)->CI8Vec4{I8Vec4::new(x, y, z, w).into()}
#[no_mangle] pub extern "C" fn mid_i8vec4_add(a:CI8Vec4,b:CI8Vec4)->CI8Vec4{(I8Vec4::from(a)+I8Vec4::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_i8vec4_sub(a:CI8Vec4,b:CI8Vec4)->CI8Vec4{(I8Vec4::from(a)-I8Vec4::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_i8vec4_mul(a:CI8Vec4,b:CI8Vec4)->CI8Vec4{(I8Vec4::from(a)*I8Vec4::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_i8vec4_scale(v:CI8Vec4,s:i8)->CI8Vec4{(I8Vec4::from(v)*s).into()}
#[no_mangle] pub extern "C" fn mid_i8vec4_dot(a:CI8Vec4,b:CI8Vec4)->i16{I8Vec4::from(a).dot(I8Vec4::from(b))}
#[no_mangle] pub extern "C" fn mid_i8vec4_min(a:CI8Vec4,b:CI8Vec4)->CI8Vec4{I8Vec4::from(a).min(I8Vec4::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_i8vec4_max(a:CI8Vec4,b:CI8Vec4)->CI8Vec4{I8Vec4::from(a).max(I8Vec4::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_i8vec4_clamp(v:CI8Vec4,lo:CI8Vec4,hi:CI8Vec4)->CI8Vec4{I8Vec4::from(v).clamp(I8Vec4::from(lo),I8Vec4::from(hi)).into()}
#[no_mangle] pub extern "C" fn mid_i8vec4_abs(v:CI8Vec4)->CI8Vec4{I8Vec4::from(v).abs().into()}
#[no_mangle] pub extern "C" fn mid_i8vec4_neg(v:CI8Vec4)->CI8Vec4{(-I8Vec4::from(v)).into()}
#[no_mangle] pub extern "C" fn mid_i8vec4_length_sq(v:CI8Vec4)->i16{I8Vec4::from(v).length_sq()}
#[no_mangle] pub extern "C" fn mid_i8vec4_distance_sq(a:CI8Vec4,b:CI8Vec4)->i16{I8Vec4::from(a).distance_sq(I8Vec4::from(b))}
#[no_mangle] pub extern "C" fn mid_i8vec4_min_element(v:CI8Vec4)->i8{I8Vec4::from(v).min_element()}
#[no_mangle] pub extern "C" fn mid_i8vec4_max_element(v:CI8Vec4)->i8{I8Vec4::from(v).max_element()}
#[no_mangle] pub extern "C" fn mid_i8vec4_element_sum(v:CI8Vec4)->i8{I8Vec4::from(v).element_sum()}
#[no_mangle] pub extern "C" fn mid_i8vec4_wrapping_add(a:CI8Vec4,b:CI8Vec4)->CI8Vec4{I8Vec4::from(a).wrapping_add(I8Vec4::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_i8vec4_wrapping_sub(a:CI8Vec4,b:CI8Vec4)->CI8Vec4{I8Vec4::from(a).wrapping_sub(I8Vec4::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_i8vec4_saturating_add(a:CI8Vec4,b:CI8Vec4)->CI8Vec4{I8Vec4::from(a).saturating_add(I8Vec4::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_i8vec4_saturating_sub(a:CI8Vec4,b:CI8Vec4)->CI8Vec4{I8Vec4::from(a).saturating_sub(I8Vec4::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_i8vec4_cmpeq(a:CI8Vec4,b:CI8Vec4)->BVec4{I8Vec4::from(a).cmpeq(I8Vec4::from(b))}
#[no_mangle] pub extern "C" fn mid_i8vec4_cmpne(a:CI8Vec4,b:CI8Vec4)->BVec4{I8Vec4::from(a).cmpne(I8Vec4::from(b))}
#[no_mangle] pub extern "C" fn mid_i8vec4_cmpge(a:CI8Vec4,b:CI8Vec4)->BVec4{I8Vec4::from(a).cmpge(I8Vec4::from(b))}
#[no_mangle] pub extern "C" fn mid_i8vec4_cmpgt(a:CI8Vec4,b:CI8Vec4)->BVec4{I8Vec4::from(a).cmpgt(I8Vec4::from(b))}
#[no_mangle] pub extern "C" fn mid_i8vec4_cmple(a:CI8Vec4,b:CI8Vec4)->BVec4{I8Vec4::from(a).cmple(I8Vec4::from(b))}
#[no_mangle] pub extern "C" fn mid_i8vec4_cmplt(a:CI8Vec4,b:CI8Vec4)->BVec4{I8Vec4::from(a).cmplt(I8Vec4::from(b))}

// ── Exports — U8Vec2 ────────────────────────────────────────────────────────────────
#[no_mangle] pub extern "C" fn mid_u8vec2_new(x:u8, y:u8)->CU8Vec2{U8Vec2::new(x, y).into()}
#[no_mangle] pub extern "C" fn mid_u8vec2_add(a:CU8Vec2,b:CU8Vec2)->CU8Vec2{(U8Vec2::from(a)+U8Vec2::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_u8vec2_sub(a:CU8Vec2,b:CU8Vec2)->CU8Vec2{(U8Vec2::from(a)-U8Vec2::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_u8vec2_mul(a:CU8Vec2,b:CU8Vec2)->CU8Vec2{(U8Vec2::from(a)*U8Vec2::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_u8vec2_scale(v:CU8Vec2,s:u8)->CU8Vec2{(U8Vec2::from(v)*s).into()}
#[no_mangle] pub extern "C" fn mid_u8vec2_dot(a:CU8Vec2,b:CU8Vec2)->u16{U8Vec2::from(a).dot(U8Vec2::from(b))}
#[no_mangle] pub extern "C" fn mid_u8vec2_min(a:CU8Vec2,b:CU8Vec2)->CU8Vec2{U8Vec2::from(a).min(U8Vec2::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_u8vec2_max(a:CU8Vec2,b:CU8Vec2)->CU8Vec2{U8Vec2::from(a).max(U8Vec2::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_u8vec2_clamp(v:CU8Vec2,lo:CU8Vec2,hi:CU8Vec2)->CU8Vec2{U8Vec2::from(v).clamp(U8Vec2::from(lo),U8Vec2::from(hi)).into()}
#[no_mangle] pub extern "C" fn mid_u8vec2_length_sq(v:CU8Vec2)->u16{U8Vec2::from(v).length_sq()}
#[no_mangle] pub extern "C" fn mid_u8vec2_distance_sq(a:CU8Vec2,b:CU8Vec2)->u16{U8Vec2::from(a).distance_sq(U8Vec2::from(b))}
#[no_mangle] pub extern "C" fn mid_u8vec2_min_element(v:CU8Vec2)->u8{U8Vec2::from(v).min_element()}
#[no_mangle] pub extern "C" fn mid_u8vec2_max_element(v:CU8Vec2)->u8{U8Vec2::from(v).max_element()}
#[no_mangle] pub extern "C" fn mid_u8vec2_element_sum(v:CU8Vec2)->u8{U8Vec2::from(v).element_sum()}
#[no_mangle] pub extern "C" fn mid_u8vec2_wrapping_add(a:CU8Vec2,b:CU8Vec2)->CU8Vec2{U8Vec2::from(a).wrapping_add(U8Vec2::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_u8vec2_wrapping_sub(a:CU8Vec2,b:CU8Vec2)->CU8Vec2{U8Vec2::from(a).wrapping_sub(U8Vec2::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_u8vec2_saturating_add(a:CU8Vec2,b:CU8Vec2)->CU8Vec2{U8Vec2::from(a).saturating_add(U8Vec2::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_u8vec2_saturating_sub(a:CU8Vec2,b:CU8Vec2)->CU8Vec2{U8Vec2::from(a).saturating_sub(U8Vec2::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_u8vec2_cmpeq(a:CU8Vec2,b:CU8Vec2)->BVec2{U8Vec2::from(a).cmpeq(U8Vec2::from(b))}
#[no_mangle] pub extern "C" fn mid_u8vec2_cmpne(a:CU8Vec2,b:CU8Vec2)->BVec2{U8Vec2::from(a).cmpne(U8Vec2::from(b))}
#[no_mangle] pub extern "C" fn mid_u8vec2_cmpge(a:CU8Vec2,b:CU8Vec2)->BVec2{U8Vec2::from(a).cmpge(U8Vec2::from(b))}
#[no_mangle] pub extern "C" fn mid_u8vec2_cmpgt(a:CU8Vec2,b:CU8Vec2)->BVec2{U8Vec2::from(a).cmpgt(U8Vec2::from(b))}
#[no_mangle] pub extern "C" fn mid_u8vec2_cmple(a:CU8Vec2,b:CU8Vec2)->BVec2{U8Vec2::from(a).cmple(U8Vec2::from(b))}
#[no_mangle] pub extern "C" fn mid_u8vec2_cmplt(a:CU8Vec2,b:CU8Vec2)->BVec2{U8Vec2::from(a).cmplt(U8Vec2::from(b))}

// ── Exports — U8Vec3 ────────────────────────────────────────────────────────────────
#[no_mangle] pub extern "C" fn mid_u8vec3_new(x:u8, y:u8, z:u8)->CU8Vec3{U8Vec3::new(x, y, z).into()}
#[no_mangle] pub extern "C" fn mid_u8vec3_add(a:CU8Vec3,b:CU8Vec3)->CU8Vec3{(U8Vec3::from(a)+U8Vec3::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_u8vec3_sub(a:CU8Vec3,b:CU8Vec3)->CU8Vec3{(U8Vec3::from(a)-U8Vec3::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_u8vec3_mul(a:CU8Vec3,b:CU8Vec3)->CU8Vec3{(U8Vec3::from(a)*U8Vec3::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_u8vec3_scale(v:CU8Vec3,s:u8)->CU8Vec3{(U8Vec3::from(v)*s).into()}
#[no_mangle] pub extern "C" fn mid_u8vec3_dot(a:CU8Vec3,b:CU8Vec3)->u16{U8Vec3::from(a).dot(U8Vec3::from(b))}
#[no_mangle] pub extern "C" fn mid_u8vec3_min(a:CU8Vec3,b:CU8Vec3)->CU8Vec3{U8Vec3::from(a).min(U8Vec3::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_u8vec3_max(a:CU8Vec3,b:CU8Vec3)->CU8Vec3{U8Vec3::from(a).max(U8Vec3::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_u8vec3_clamp(v:CU8Vec3,lo:CU8Vec3,hi:CU8Vec3)->CU8Vec3{U8Vec3::from(v).clamp(U8Vec3::from(lo),U8Vec3::from(hi)).into()}
#[no_mangle] pub extern "C" fn mid_u8vec3_length_sq(v:CU8Vec3)->u16{U8Vec3::from(v).length_sq()}
#[no_mangle] pub extern "C" fn mid_u8vec3_distance_sq(a:CU8Vec3,b:CU8Vec3)->u16{U8Vec3::from(a).distance_sq(U8Vec3::from(b))}
#[no_mangle] pub extern "C" fn mid_u8vec3_min_element(v:CU8Vec3)->u8{U8Vec3::from(v).min_element()}
#[no_mangle] pub extern "C" fn mid_u8vec3_max_element(v:CU8Vec3)->u8{U8Vec3::from(v).max_element()}
#[no_mangle] pub extern "C" fn mid_u8vec3_element_sum(v:CU8Vec3)->u8{U8Vec3::from(v).element_sum()}
#[no_mangle] pub extern "C" fn mid_u8vec3_wrapping_add(a:CU8Vec3,b:CU8Vec3)->CU8Vec3{U8Vec3::from(a).wrapping_add(U8Vec3::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_u8vec3_wrapping_sub(a:CU8Vec3,b:CU8Vec3)->CU8Vec3{U8Vec3::from(a).wrapping_sub(U8Vec3::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_u8vec3_saturating_add(a:CU8Vec3,b:CU8Vec3)->CU8Vec3{U8Vec3::from(a).saturating_add(U8Vec3::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_u8vec3_saturating_sub(a:CU8Vec3,b:CU8Vec3)->CU8Vec3{U8Vec3::from(a).saturating_sub(U8Vec3::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_u8vec3_cmpeq(a:CU8Vec3,b:CU8Vec3)->BVec3{U8Vec3::from(a).cmpeq(U8Vec3::from(b))}
#[no_mangle] pub extern "C" fn mid_u8vec3_cmpne(a:CU8Vec3,b:CU8Vec3)->BVec3{U8Vec3::from(a).cmpne(U8Vec3::from(b))}
#[no_mangle] pub extern "C" fn mid_u8vec3_cmpge(a:CU8Vec3,b:CU8Vec3)->BVec3{U8Vec3::from(a).cmpge(U8Vec3::from(b))}
#[no_mangle] pub extern "C" fn mid_u8vec3_cmpgt(a:CU8Vec3,b:CU8Vec3)->BVec3{U8Vec3::from(a).cmpgt(U8Vec3::from(b))}
#[no_mangle] pub extern "C" fn mid_u8vec3_cmple(a:CU8Vec3,b:CU8Vec3)->BVec3{U8Vec3::from(a).cmple(U8Vec3::from(b))}
#[no_mangle] pub extern "C" fn mid_u8vec3_cmplt(a:CU8Vec3,b:CU8Vec3)->BVec3{U8Vec3::from(a).cmplt(U8Vec3::from(b))}

// ── Exports — U8Vec4 ────────────────────────────────────────────────────────────────
#[no_mangle] pub extern "C" fn mid_u8vec4_new(x:u8, y:u8, z:u8, w:u8)->CU8Vec4{U8Vec4::new(x, y, z, w).into()}
#[no_mangle] pub extern "C" fn mid_u8vec4_add(a:CU8Vec4,b:CU8Vec4)->CU8Vec4{(U8Vec4::from(a)+U8Vec4::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_u8vec4_sub(a:CU8Vec4,b:CU8Vec4)->CU8Vec4{(U8Vec4::from(a)-U8Vec4::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_u8vec4_mul(a:CU8Vec4,b:CU8Vec4)->CU8Vec4{(U8Vec4::from(a)*U8Vec4::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_u8vec4_scale(v:CU8Vec4,s:u8)->CU8Vec4{(U8Vec4::from(v)*s).into()}
#[no_mangle] pub extern "C" fn mid_u8vec4_dot(a:CU8Vec4,b:CU8Vec4)->u16{U8Vec4::from(a).dot(U8Vec4::from(b))}
#[no_mangle] pub extern "C" fn mid_u8vec4_min(a:CU8Vec4,b:CU8Vec4)->CU8Vec4{U8Vec4::from(a).min(U8Vec4::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_u8vec4_max(a:CU8Vec4,b:CU8Vec4)->CU8Vec4{U8Vec4::from(a).max(U8Vec4::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_u8vec4_clamp(v:CU8Vec4,lo:CU8Vec4,hi:CU8Vec4)->CU8Vec4{U8Vec4::from(v).clamp(U8Vec4::from(lo),U8Vec4::from(hi)).into()}
#[no_mangle] pub extern "C" fn mid_u8vec4_length_sq(v:CU8Vec4)->u16{U8Vec4::from(v).length_sq()}
#[no_mangle] pub extern "C" fn mid_u8vec4_min_element(v:CU8Vec4)->u8{U8Vec4::from(v).min_element()}
#[no_mangle] pub extern "C" fn mid_u8vec4_max_element(v:CU8Vec4)->u8{U8Vec4::from(v).max_element()}
#[no_mangle] pub extern "C" fn mid_u8vec4_element_sum(v:CU8Vec4)->u8{U8Vec4::from(v).element_sum()}
#[no_mangle] pub extern "C" fn mid_u8vec4_wrapping_add(a:CU8Vec4,b:CU8Vec4)->CU8Vec4{U8Vec4::from(a).wrapping_add(U8Vec4::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_u8vec4_wrapping_sub(a:CU8Vec4,b:CU8Vec4)->CU8Vec4{U8Vec4::from(a).wrapping_sub(U8Vec4::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_u8vec4_saturating_add(a:CU8Vec4,b:CU8Vec4)->CU8Vec4{U8Vec4::from(a).saturating_add(U8Vec4::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_u8vec4_saturating_sub(a:CU8Vec4,b:CU8Vec4)->CU8Vec4{U8Vec4::from(a).saturating_sub(U8Vec4::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_u8vec4_cmpeq(a:CU8Vec4,b:CU8Vec4)->BVec4{U8Vec4::from(a).cmpeq(U8Vec4::from(b))}
#[no_mangle] pub extern "C" fn mid_u8vec4_cmpne(a:CU8Vec4,b:CU8Vec4)->BVec4{U8Vec4::from(a).cmpne(U8Vec4::from(b))}
#[no_mangle] pub extern "C" fn mid_u8vec4_cmpge(a:CU8Vec4,b:CU8Vec4)->BVec4{U8Vec4::from(a).cmpge(U8Vec4::from(b))}
#[no_mangle] pub extern "C" fn mid_u8vec4_cmpgt(a:CU8Vec4,b:CU8Vec4)->BVec4{U8Vec4::from(a).cmpgt(U8Vec4::from(b))}
#[no_mangle] pub extern "C" fn mid_u8vec4_cmple(a:CU8Vec4,b:CU8Vec4)->BVec4{U8Vec4::from(a).cmple(U8Vec4::from(b))}
#[no_mangle] pub extern "C" fn mid_u8vec4_cmplt(a:CU8Vec4,b:CU8Vec4)->BVec4{U8Vec4::from(a).cmplt(U8Vec4::from(b))}
