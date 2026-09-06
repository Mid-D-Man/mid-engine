// crates/mid-math/src/ffi/wide_float_avx2.rs
//! C-ABI types and #[no_mangle] exports for the AVX2-only additive
//! wide/float SIMD types: f32x8, Vec3x8.
//!
//! "FFI option 1" from the AVX2 rework's design pass -- see
//! wide_int_avx2.rs's header for the full reasoning (same reasoning
//! applies here: always compiled on x86/x86_64, runtime-dispatched
//! internally, no link-error risk on a non-AVX2 CPU anymore). No
//! QuatX8/Rotor3x8-equivalent exists in mid-math (confirmed against
//! source), so there's no third type to add here alongside f32x8/Vec3x8.
//!
//! Gated `#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]` at
//! the module level (see ffi/mod.rs).
//!
//! Scope mirrors wide_float.rs exactly, applied to the wider types:
//! construction, splat, core arithmetic, dot/cross/normalize/lerp.
//! Deliberately excludes, matching wide_float.rs's own narrower scope
//! for Vec3x4 (not just what's missing from f32x8's/Vec3x8's own
//! larger API -- these are the same omissions wide_float.rs already
//! made for the x4 tier, mirrored for consistency): cmpXX/blend/
//! is_finite/is_nan, min/max/select/length_lt/mul_elem/scale/
//! scale_uniform/madd/length/length_sq on Vec3x8, and div on f32x8
//! (f32x8 does have a real Div impl, unlike f32x4 -- omitted anyway
//! for parity with wide_float.rs's own choice not to expose division
//! at the x4 tier, recip()+mul being the established idiom instead).
//!
//! C representation, same reasoning as wide_float.rs:
//!   Cf32x8  -- a bare `v: [f32; 8]` array (8 independent packed
//!              scalar lanes, not a vector).
//!   CVec3x8 -- `v: [CVec3; 8]`, an AoS array of the EXISTING CVec3
//!              type from ffi/float32.rs (same type wide_float.rs
//!              reuses for CVec3x4) -- Vec3x8's internal SoA
//!              {lo,hi}: Vec3x4 halves are a platform-specific
//!              implementation detail, never exposed across this FFI
//!              boundary.
//!
//! dot() returns Cf32x8 (8 independent per-lane dot products),
//! matching Vec3x8::dot -> f32x8 exactly. lerp takes a Cf32x8 `t`
//! (per-lane blend factor, not a single scalar broadcast).

use crate::{f32x8, Vec3x8};
use super::float32::CVec3;

// ═══════════════════════════════════════════════════════════════════════════
//  C types
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Copy, PartialEq)] #[repr(C)]
pub struct Cf32x8 { pub v: [f32; 8] }
impl From<f32x8>  for Cf32x8 { #[inline(always)] fn from(w: f32x8)  -> Self { Self { v: w.to_array() } } }
impl From<Cf32x8> for f32x8  { #[inline(always)] fn from(c: Cf32x8) -> Self { f32x8::from_array(c.v) } }

#[derive(Debug, Clone, Copy, PartialEq)] #[repr(C)]
pub struct CVec3x8 { pub v: [CVec3; 8] }
impl From<Vec3x8> for CVec3x8 {
    #[inline(always)]
    fn from(w: Vec3x8) -> Self { Self { v: w.to_array().map(CVec3::from) } }
}
impl From<CVec3x8> for Vec3x8 {
    #[inline(always)]
    fn from(c: CVec3x8) -> Self {
        let arr: [crate::Vec3; 8] = c.v.map(crate::Vec3::from);
        Vec3x8::from_slice(&arr)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Exports — f32x8
// ═══════════════════════════════════════════════════════════════════════════

#[no_mangle] pub extern "C" fn mid_f32x8_new(v0:f32,v1:f32,v2:f32,v3:f32,v4:f32,v5:f32,v6:f32,v7:f32)->Cf32x8{f32x8::new(v0,v1,v2,v3,v4,v5,v6,v7).into()}
#[no_mangle] pub extern "C" fn mid_f32x8_splat(v:f32)->Cf32x8{f32x8::splat(v).into()}
#[no_mangle] pub extern "C" fn mid_f32x8_add(a:Cf32x8,b:Cf32x8)->Cf32x8{(f32x8::from(a)+f32x8::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_f32x8_sub(a:Cf32x8,b:Cf32x8)->Cf32x8{(f32x8::from(a)-f32x8::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_f32x8_mul(a:Cf32x8,b:Cf32x8)->Cf32x8{(f32x8::from(a)*f32x8::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_f32x8_min(a:Cf32x8,b:Cf32x8)->Cf32x8{f32x8::from(a).min(f32x8::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_f32x8_max(a:Cf32x8,b:Cf32x8)->Cf32x8{f32x8::from(a).max(f32x8::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_f32x8_clamp(v:Cf32x8,lo:Cf32x8,hi:Cf32x8)->Cf32x8{f32x8::from(v).clamp(f32x8::from(lo),f32x8::from(hi)).into()}
#[no_mangle] pub extern "C" fn mid_f32x8_abs(v:Cf32x8)->Cf32x8{f32x8::from(v).abs().into()}
#[no_mangle] pub extern "C" fn mid_f32x8_sqrt(v:Cf32x8)->Cf32x8{f32x8::from(v).sqrt().into()}
#[no_mangle] pub extern "C" fn mid_f32x8_recip(v:Cf32x8)->Cf32x8{f32x8::from(v).recip().into()}
#[no_mangle] pub extern "C" fn mid_f32x8_recip_sqrt(v:Cf32x8)->Cf32x8{f32x8::from(v).recip_sqrt().into()}
#[no_mangle] pub extern "C" fn mid_f32x8_mul_add(a:Cf32x8,b:Cf32x8,c:Cf32x8)->Cf32x8{f32x8::from(a).mul_add(f32x8::from(b),f32x8::from(c)).into()}

// ═══════════════════════════════════════════════════════════════════════════
//  Exports — Vec3x8
// ═══════════════════════════════════════════════════════════════════════════

#[no_mangle] pub extern "C" fn mid_vec3x8_new(v0:CVec3,v1:CVec3,v2:CVec3,v3:CVec3,v4:CVec3,v5:CVec3,v6:CVec3,v7:CVec3)->CVec3x8{CVec3x8{v:[v0,v1,v2,v3,v4,v5,v6,v7]}}
#[no_mangle] pub extern "C" fn mid_vec3x8_splat(v:CVec3)->CVec3x8{Vec3x8::splat(crate::Vec3::from(v)).into()}
#[no_mangle] pub extern "C" fn mid_vec3x8_add(a:CVec3x8,b:CVec3x8)->CVec3x8{(Vec3x8::from(a)+Vec3x8::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_vec3x8_sub(a:CVec3x8,b:CVec3x8)->CVec3x8{(Vec3x8::from(a)-Vec3x8::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_vec3x8_dot(a:CVec3x8,b:CVec3x8)->Cf32x8{Vec3x8::from(a).dot(Vec3x8::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_vec3x8_cross(a:CVec3x8,b:CVec3x8)->CVec3x8{Vec3x8::from(a).cross(Vec3x8::from(b)).into()}
#[no_mangle] pub extern "C" fn mid_vec3x8_normalize(v:CVec3x8)->CVec3x8{Vec3x8::from(v).normalize().into()}
#[no_mangle] pub extern "C" fn mid_vec3x8_normalize_precise(v:CVec3x8)->CVec3x8{Vec3x8::from(v).normalize_precise().into()}
#[no_mangle] pub extern "C" fn mid_vec3x8_lerp(a:CVec3x8,b:CVec3x8,t:Cf32x8)->CVec3x8{Vec3x8::from(a).lerp(Vec3x8::from(b),f32x8::from(t)).into()}
