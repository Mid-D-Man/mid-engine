// crates/mid-math/src/ffi/fixed.rs
//! C-ABI exports for deterministic fixed-point math.
//!
//! Fixed-point values are represented as raw i64 at the C boundary.
//! The FRAC parameter is baked into the function names (8, 12, 16 bits).
//!
//! FixedVec2/Vec3 variants expose raw structs of i64 fields.

use crate::fixed::{
    Fixed8,  Fixed12,  Fixed16,
    
    Fixed8Vec2,  Fixed12Vec2,  Fixed16Vec2,
    Fixed8Vec3,  Fixed12Vec3,  Fixed16Vec3,
};


// ═══════════════════════════════════════════════════════════════════════════
//  C scalar types (transparent i64)
// ═══════════════════════════════════════════════════════════════════════════

/// Fixed<8>  raw i64.
pub type CFixed8  = i64;
/// Fixed<12> raw i64.
pub type CFixed12 = i64;
/// Fixed<16> raw i64.
pub type CFixed16 = i64;

// ═══════════════════════════════════════════════════════════════════════════
//  C vector types
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct CFixed8Vec2  { pub x: i64, pub y: i64 }
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct CFixed12Vec2 { pub x: i64, pub y: i64 }
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct CFixed16Vec2 { pub x: i64, pub y: i64 }

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct CFixed8Vec3  { pub x: i64, pub y: i64, pub z: i64 }
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct CFixed12Vec3 { pub x: i64, pub y: i64, pub z: i64 }
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct CFixed16Vec3 { pub x: i64, pub y: i64, pub z: i64 }

// ═══════════════════════════════════════════════════════════════════════════
//  Fixed<8> scalar exports
// ═══════════════════════════════════════════════════════════════════════════

#[no_mangle] pub extern "C" fn mid_fixed8_from_f32(v: f32)  -> CFixed8 { Fixed8::from_f32(v).to_raw() }
#[no_mangle] pub extern "C" fn mid_fixed8_from_f64(v: f64)  -> CFixed8 { Fixed8::from_f64(v).to_raw() }
#[no_mangle] pub extern "C" fn mid_fixed8_from_i32(v: i32)  -> CFixed8 { Fixed8::from_i32(v).to_raw() }
#[no_mangle] pub extern "C" fn mid_fixed8_to_f32(v: CFixed8) -> f32    { Fixed8::from_raw(v).to_f32() }
#[no_mangle] pub extern "C" fn mid_fixed8_to_f64(v: CFixed8) -> f64    { Fixed8::from_raw(v).to_f64() }
#[no_mangle] pub extern "C" fn mid_fixed8_to_i32(v: CFixed8) -> i32    { Fixed8::from_raw(v).to_i32_trunc() }
#[no_mangle] pub extern "C" fn mid_fixed8_add(a: CFixed8, b: CFixed8) -> CFixed8 {
    (Fixed8::from_raw(a) + Fixed8::from_raw(b)).to_raw()
}
#[no_mangle] pub extern "C" fn mid_fixed8_sub(a: CFixed8, b: CFixed8) -> CFixed8 {
    (Fixed8::from_raw(a) - Fixed8::from_raw(b)).to_raw()
}
#[no_mangle] pub extern "C" fn mid_fixed8_mul(a: CFixed8, b: CFixed8) -> CFixed8 {
    Fixed8::from_raw(a).fixed_mul(Fixed8::from_raw(b)).to_raw()
}
#[no_mangle] pub extern "C" fn mid_fixed8_div(a: CFixed8, b: CFixed8) -> CFixed8 {
    Fixed8::from_raw(a).fixed_div(Fixed8::from_raw(b)).to_raw()
}
#[no_mangle] pub extern "C" fn mid_fixed8_abs(v: CFixed8) -> CFixed8 {
    Fixed8::from_raw(v).abs().to_raw()
}
#[no_mangle] pub extern "C" fn mid_fixed8_floor(v: CFixed8) -> CFixed8 {
    Fixed8::from_raw(v).floor().to_raw()
}
#[no_mangle] pub extern "C" fn mid_fixed8_ceil(v: CFixed8) -> CFixed8 {
    Fixed8::from_raw(v).ceil().to_raw()
}
#[no_mangle] pub extern "C" fn mid_fixed8_lerp(a: CFixed8, b: CFixed8, t: CFixed8) -> CFixed8 {
    Fixed8::from_raw(a).lerp(Fixed8::from_raw(b), Fixed8::from_raw(t)).to_raw()
}
#[no_mangle] pub extern "C" fn mid_fixed8_clamp(v: CFixed8, lo: CFixed8, hi: CFixed8) -> CFixed8 {
    Fixed8::from_raw(v).clamp(Fixed8::from_raw(lo), Fixed8::from_raw(hi)).to_raw()
}
#[no_mangle] pub extern "C" fn mid_fixed8_saturating_add(a: CFixed8, b: CFixed8) -> CFixed8 {
    Fixed8::from_raw(a).saturating_add(Fixed8::from_raw(b)).to_raw()
}
#[no_mangle] pub extern "C" fn mid_fixed8_saturating_sub(a: CFixed8, b: CFixed8) -> CFixed8 {
    Fixed8::from_raw(a).saturating_sub(Fixed8::from_raw(b)).to_raw()
}

// ═══════════════════════════════════════════════════════════════════════════
//  Fixed<12> scalar exports
// ═══════════════════════════════════════════════════════════════════════════

#[no_mangle] pub extern "C" fn mid_fixed12_from_f32(v: f32)   -> CFixed12 { Fixed12::from_f32(v).to_raw() }
#[no_mangle] pub extern "C" fn mid_fixed12_from_f64(v: f64)   -> CFixed12 { Fixed12::from_f64(v).to_raw() }
#[no_mangle] pub extern "C" fn mid_fixed12_from_i32(v: i32)   -> CFixed12 { Fixed12::from_i32(v).to_raw() }
#[no_mangle] pub extern "C" fn mid_fixed12_to_f32(v: CFixed12) -> f32     { Fixed12::from_raw(v).to_f32() }
#[no_mangle] pub extern "C" fn mid_fixed12_to_f64(v: CFixed12) -> f64     { Fixed12::from_raw(v).to_f64() }
#[no_mangle] pub extern "C" fn mid_fixed12_to_i32(v: CFixed12) -> i32     { Fixed12::from_raw(v).to_i32_trunc() }
#[no_mangle] pub extern "C" fn mid_fixed12_add(a: CFixed12, b: CFixed12) -> CFixed12 {
    (Fixed12::from_raw(a) + Fixed12::from_raw(b)).to_raw()
}
#[no_mangle] pub extern "C" fn mid_fixed12_sub(a: CFixed12, b: CFixed12) -> CFixed12 {
    (Fixed12::from_raw(a) - Fixed12::from_raw(b)).to_raw()
}
#[no_mangle] pub extern "C" fn mid_fixed12_mul(a: CFixed12, b: CFixed12) -> CFixed12 {
    Fixed12::from_raw(a).fixed_mul(Fixed12::from_raw(b)).to_raw()
}
#[no_mangle] pub extern "C" fn mid_fixed12_div(a: CFixed12, b: CFixed12) -> CFixed12 {
    Fixed12::from_raw(a).fixed_div(Fixed12::from_raw(b)).to_raw()
}
#[no_mangle] pub extern "C" fn mid_fixed12_abs(v: CFixed12) -> CFixed12 { Fixed12::from_raw(v).abs().to_raw() }
#[no_mangle] pub extern "C" fn mid_fixed12_floor(v: CFixed12) -> CFixed12 { Fixed12::from_raw(v).floor().to_raw() }
#[no_mangle] pub extern "C" fn mid_fixed12_ceil(v: CFixed12) -> CFixed12  { Fixed12::from_raw(v).ceil().to_raw() }
#[no_mangle] pub extern "C" fn mid_fixed12_lerp(a: CFixed12, b: CFixed12, t: CFixed12) -> CFixed12 {
    Fixed12::from_raw(a).lerp(Fixed12::from_raw(b), Fixed12::from_raw(t)).to_raw()
}
#[no_mangle] pub extern "C" fn mid_fixed12_clamp(v: CFixed12, lo: CFixed12, hi: CFixed12) -> CFixed12 {
    Fixed12::from_raw(v).clamp(Fixed12::from_raw(lo), Fixed12::from_raw(hi)).to_raw()
}
#[no_mangle] pub extern "C" fn mid_fixed12_saturating_add(a: CFixed12, b: CFixed12) -> CFixed12 {
    Fixed12::from_raw(a).saturating_add(Fixed12::from_raw(b)).to_raw()
}
#[no_mangle] pub extern "C" fn mid_fixed12_saturating_sub(a: CFixed12, b: CFixed12) -> CFixed12 {
    Fixed12::from_raw(a).saturating_sub(Fixed12::from_raw(b)).to_raw()
}

// ═══════════════════════════════════════════════════════════════════════════
//  Fixed<16> scalar exports
// ═══════════════════════════════════════════════════════════════════════════

#[no_mangle] pub extern "C" fn mid_fixed16_from_f32(v: f32)   -> CFixed16 { Fixed16::from_f32(v).to_raw() }
#[no_mangle] pub extern "C" fn mid_fixed16_from_f64(v: f64)   -> CFixed16 { Fixed16::from_f64(v).to_raw() }
#[no_mangle] pub extern "C" fn mid_fixed16_from_i32(v: i32)   -> CFixed16 { Fixed16::from_i32(v).to_raw() }
#[no_mangle] pub extern "C" fn mid_fixed16_to_f32(v: CFixed16) -> f32     { Fixed16::from_raw(v).to_f32() }
#[no_mangle] pub extern "C" fn mid_fixed16_to_f64(v: CFixed16) -> f64     { Fixed16::from_raw(v).to_f64() }
#[no_mangle] pub extern "C" fn mid_fixed16_to_i32(v: CFixed16) -> i32     { Fixed16::from_raw(v).to_i32_trunc() }
#[no_mangle] pub extern "C" fn mid_fixed16_add(a: CFixed16, b: CFixed16) -> CFixed16 {
    (Fixed16::from_raw(a) + Fixed16::from_raw(b)).to_raw()
}
#[no_mangle] pub extern "C" fn mid_fixed16_sub(a: CFixed16, b: CFixed16) -> CFixed16 {
    (Fixed16::from_raw(a) - Fixed16::from_raw(b)).to_raw()
}
#[no_mangle] pub extern "C" fn mid_fixed16_mul(a: CFixed16, b: CFixed16) -> CFixed16 {
    Fixed16::from_raw(a).fixed_mul(Fixed16::from_raw(b)).to_raw()
}
#[no_mangle] pub extern "C" fn mid_fixed16_div(a: CFixed16, b: CFixed16) -> CFixed16 {
    Fixed16::from_raw(a).fixed_div(Fixed16::from_raw(b)).to_raw()
}
#[no_mangle] pub extern "C" fn mid_fixed16_abs(v: CFixed16) -> CFixed16  { Fixed16::from_raw(v).abs().to_raw() }
#[no_mangle] pub extern "C" fn mid_fixed16_floor(v: CFixed16) -> CFixed16 { Fixed16::from_raw(v).floor().to_raw() }
#[no_mangle] pub extern "C" fn mid_fixed16_ceil(v: CFixed16) -> CFixed16  { Fixed16::from_raw(v).ceil().to_raw() }
#[no_mangle] pub extern "C" fn mid_fixed16_lerp(a: CFixed16, b: CFixed16, t: CFixed16) -> CFixed16 {
    Fixed16::from_raw(a).lerp(Fixed16::from_raw(b), Fixed16::from_raw(t)).to_raw()
}
#[no_mangle] pub extern "C" fn mid_fixed16_clamp(v: CFixed16, lo: CFixed16, hi: CFixed16) -> CFixed16 {
    Fixed16::from_raw(v).clamp(Fixed16::from_raw(lo), Fixed16::from_raw(hi)).to_raw()
}
#[no_mangle] pub extern "C" fn mid_fixed16_saturating_add(a: CFixed16, b: CFixed16) -> CFixed16 {
    Fixed16::from_raw(a).saturating_add(Fixed16::from_raw(b)).to_raw()
}
#[no_mangle] pub extern "C" fn mid_fixed16_saturating_sub(a: CFixed16, b: CFixed16) -> CFixed16 {
    Fixed16::from_raw(a).saturating_sub(Fixed16::from_raw(b)).to_raw()
}

// ═══════════════════════════════════════════════════════════════════════════
//  Fixed16Vec2 exports (representative — pattern same for 8 and 12)
// ═══════════════════════════════════════════════════════════════════════════

#[no_mangle] pub extern "C" fn mid_fixed16_vec2_from_f32(x: f32, y: f32) -> CFixed16Vec2 {
    let v = Fixed16Vec2::from_f32(x, y);
    CFixed16Vec2 { x: v.x.to_raw(), y: v.y.to_raw() }
}
#[no_mangle] pub extern "C" fn mid_fixed16_vec2_from_i32(x: i32, y: i32) -> CFixed16Vec2 {
    let v = Fixed16Vec2::from_i32(x, y);
    CFixed16Vec2 { x: v.x.to_raw(), y: v.y.to_raw() }
}
#[no_mangle] pub extern "C" fn mid_fixed16_vec2_to_f32(v: CFixed16Vec2, ox: *mut f32, oy: *mut f32) {
    let fv = Fixed16Vec2::from_raw(v.x, v.y);
    unsafe { *ox = fv.x.to_f32(); *oy = fv.y.to_f32(); }
}
#[no_mangle] pub extern "C" fn mid_fixed16_vec2_add(a: CFixed16Vec2, b: CFixed16Vec2) -> CFixed16Vec2 {
    let r = Fixed16Vec2::from_raw(a.x,a.y) + Fixed16Vec2::from_raw(b.x,b.y);
    CFixed16Vec2 { x: r.x.to_raw(), y: r.y.to_raw() }
}
#[no_mangle] pub extern "C" fn mid_fixed16_vec2_sub(a: CFixed16Vec2, b: CFixed16Vec2) -> CFixed16Vec2 {
    let r = Fixed16Vec2::from_raw(a.x,a.y) - Fixed16Vec2::from_raw(b.x,b.y);
    CFixed16Vec2 { x: r.x.to_raw(), y: r.y.to_raw() }
}
#[no_mangle] pub extern "C" fn mid_fixed16_vec2_scale(v: CFixed16Vec2, s: CFixed16) -> CFixed16Vec2 {
    let r = Fixed16Vec2::from_raw(v.x,v.y).scale(Fixed16::from_raw(s));
    CFixed16Vec2 { x: r.x.to_raw(), y: r.y.to_raw() }
}
#[no_mangle] pub extern "C" fn mid_fixed16_vec2_dot(a: CFixed16Vec2, b: CFixed16Vec2) -> CFixed16 {
    Fixed16Vec2::from_raw(a.x,a.y).dot(Fixed16Vec2::from_raw(b.x,b.y)).to_raw()
}
#[no_mangle] pub extern "C" fn mid_fixed16_vec2_lerp(a: CFixed16Vec2, b: CFixed16Vec2, t: CFixed16) -> CFixed16Vec2 {
    let r = Fixed16Vec2::from_raw(a.x,a.y).lerp(Fixed16Vec2::from_raw(b.x,b.y), Fixed16::from_raw(t));
    CFixed16Vec2 { x: r.x.to_raw(), y: r.y.to_raw() }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Fixed16Vec3 exports
// ═══════════════════════════════════════════════════════════════════════════

#[no_mangle] pub extern "C" fn mid_fixed16_vec3_from_f32(x: f32, y: f32, z: f32) -> CFixed16Vec3 {
    let v = Fixed16Vec3::from_f32(x, y, z);
    CFixed16Vec3 { x: v.x.to_raw(), y: v.y.to_raw(), z: v.z.to_raw() }
}
#[no_mangle] pub extern "C" fn mid_fixed16_vec3_from_i32(x: i32, y: i32, z: i32) -> CFixed16Vec3 {
    let v = Fixed16Vec3::from_i32(x, y, z);
    CFixed16Vec3 { x: v.x.to_raw(), y: v.y.to_raw(), z: v.z.to_raw() }
}
#[no_mangle] pub extern "C" fn mid_fixed16_vec3_to_f32(
    v: CFixed16Vec3, ox: *mut f32, oy: *mut f32, oz: *mut f32,
) {
    let fv = Fixed16Vec3::from_raw(v.x, v.y, v.z);
    unsafe { *ox = fv.x.to_f32(); *oy = fv.y.to_f32(); *oz = fv.z.to_f32(); }
}
#[no_mangle] pub extern "C" fn mid_fixed16_vec3_add(a: CFixed16Vec3, b: CFixed16Vec3) -> CFixed16Vec3 {
    let r = Fixed16Vec3::from_raw(a.x,a.y,a.z) + Fixed16Vec3::from_raw(b.x,b.y,b.z);
    CFixed16Vec3 { x: r.x.to_raw(), y: r.y.to_raw(), z: r.z.to_raw() }
}
#[no_mangle] pub extern "C" fn mid_fixed16_vec3_sub(a: CFixed16Vec3, b: CFixed16Vec3) -> CFixed16Vec3 {
    let r = Fixed16Vec3::from_raw(a.x,a.y,a.z) - Fixed16Vec3::from_raw(b.x,b.y,b.z);
    CFixed16Vec3 { x: r.x.to_raw(), y: r.y.to_raw(), z: r.z.to_raw() }
}
#[no_mangle] pub extern "C" fn mid_fixed16_vec3_scale(v: CFixed16Vec3, s: CFixed16) -> CFixed16Vec3 {
    let r = Fixed16Vec3::from_raw(v.x,v.y,v.z).scale(Fixed16::from_raw(s));
    CFixed16Vec3 { x: r.x.to_raw(), y: r.y.to_raw(), z: r.z.to_raw() }
}
#[no_mangle] pub extern "C" fn mid_fixed16_vec3_dot(a: CFixed16Vec3, b: CFixed16Vec3) -> CFixed16 {
    Fixed16Vec3::from_raw(a.x,a.y,a.z).dot(Fixed16Vec3::from_raw(b.x,b.y,b.z)).to_raw()
}
#[no_mangle] pub extern "C" fn mid_fixed16_vec3_cross(a: CFixed16Vec3, b: CFixed16Vec3) -> CFixed16Vec3 {
    let r = Fixed16Vec3::from_raw(a.x,a.y,a.z).cross(Fixed16Vec3::from_raw(b.x,b.y,b.z));
    CFixed16Vec3 { x: r.x.to_raw(), y: r.y.to_raw(), z: r.z.to_raw() }
}
#[no_mangle] pub extern "C" fn mid_fixed16_vec3_lerp(
    a: CFixed16Vec3, b: CFixed16Vec3, t: CFixed16,
) -> CFixed16Vec3 {
    let r = Fixed16Vec3::from_raw(a.x,a.y,a.z)
                .lerp(Fixed16Vec3::from_raw(b.x,b.y,b.z), Fixed16::from_raw(t));
    CFixed16Vec3 { x: r.x.to_raw(), y: r.y.to_raw(), z: r.z.to_raw() }
}
#[no_mangle] pub extern "C" fn mid_fixed16_vec3_manhattan_distance(
    a: CFixed16Vec3, b: CFixed16Vec3,
) -> CFixed16 {
    Fixed16Vec3::from_raw(a.x,a.y,a.z)
        .manhattan_distance(Fixed16Vec3::from_raw(b.x,b.y,b.z)).to_raw()
}

// ── Fixed8Vec2 / Fixed12Vec2 ───────────────────────────────────────────────

#[no_mangle] pub extern "C" fn mid_fixed8_vec2_from_f32(x: f32, y: f32) -> CFixed8Vec2 {
    let v = Fixed8Vec2::from_f32(x, y);
    CFixed8Vec2 { x: v.x.to_raw(), y: v.y.to_raw() }
}
#[no_mangle] pub extern "C" fn mid_fixed8_vec2_to_f32(v: CFixed8Vec2, ox: *mut f32, oy: *mut f32) {
    let fv = Fixed8Vec2::from_raw(v.x, v.y);
    unsafe { *ox = fv.x.to_f32(); *oy = fv.y.to_f32(); }
}
#[no_mangle] pub extern "C" fn mid_fixed8_vec2_add(a: CFixed8Vec2, b: CFixed8Vec2) -> CFixed8Vec2 {
    let r = Fixed8Vec2::from_raw(a.x,a.y) + Fixed8Vec2::from_raw(b.x,b.y);
    CFixed8Vec2 { x: r.x.to_raw(), y: r.y.to_raw() }
}
#[no_mangle] pub extern "C" fn mid_fixed8_vec2_dot(a: CFixed8Vec2, b: CFixed8Vec2) -> CFixed8 {
    Fixed8Vec2::from_raw(a.x,a.y).dot(Fixed8Vec2::from_raw(b.x,b.y)).to_raw()
}

#[no_mangle] pub extern "C" fn mid_fixed12_vec2_from_f32(x: f32, y: f32) -> CFixed12Vec2 {
    let v = Fixed12Vec2::from_f32(x, y);
    CFixed12Vec2 { x: v.x.to_raw(), y: v.y.to_raw() }
}
#[no_mangle] pub extern "C" fn mid_fixed12_vec2_to_f32(v: CFixed12Vec2, ox: *mut f32, oy: *mut f32) {
    let fv = Fixed12Vec2::from_raw(v.x, v.y);
    unsafe { *ox = fv.x.to_f32(); *oy = fv.y.to_f32(); }
}
#[no_mangle] pub extern "C" fn mid_fixed12_vec2_add(a: CFixed12Vec2, b: CFixed12Vec2) -> CFixed12Vec2 {
    let r = Fixed12Vec2::from_raw(a.x,a.y) + Fixed12Vec2::from_raw(b.x,b.y);
    CFixed12Vec2 { x: r.x.to_raw(), y: r.y.to_raw() }
}
#[no_mangle] pub extern "C" fn mid_fixed12_vec2_dot(a: CFixed12Vec2, b: CFixed12Vec2) -> CFixed12 {
    Fixed12Vec2::from_raw(a.x,a.y).dot(Fixed12Vec2::from_raw(b.x,b.y)).to_raw()
}

// ── Fixed8Vec3 / Fixed12Vec3 ───────────────────────────────────────────────

#[no_mangle] pub extern "C" fn mid_fixed8_vec3_from_f32(x: f32, y: f32, z: f32) -> CFixed8Vec3 {
    let v = Fixed8Vec3::from_f32(x, y, z);
    CFixed8Vec3 { x: v.x.to_raw(), y: v.y.to_raw(), z: v.z.to_raw() }
}
#[no_mangle] pub extern "C" fn mid_fixed8_vec3_to_f32(v: CFixed8Vec3, ox: *mut f32, oy: *mut f32, oz: *mut f32) {
    let fv = Fixed8Vec3::from_raw(v.x, v.y, v.z);
    unsafe { *ox = fv.x.to_f32(); *oy = fv.y.to_f32(); *oz = fv.z.to_f32(); }
}
#[no_mangle] pub extern "C" fn mid_fixed8_vec3_add(a: CFixed8Vec3, b: CFixed8Vec3) -> CFixed8Vec3 {
    let r = Fixed8Vec3::from_raw(a.x,a.y,a.z) + Fixed8Vec3::from_raw(b.x,b.y,b.z);
    CFixed8Vec3 { x: r.x.to_raw(), y: r.y.to_raw(), z: r.z.to_raw() }
}
#[no_mangle] pub extern "C" fn mid_fixed8_vec3_dot(a: CFixed8Vec3, b: CFixed8Vec3) -> CFixed8 {
    Fixed8Vec3::from_raw(a.x,a.y,a.z).dot(Fixed8Vec3::from_raw(b.x,b.y,b.z)).to_raw()
}

#[no_mangle] pub extern "C" fn mid_fixed12_vec3_from_f32(x: f32, y: f32, z: f32) -> CFixed12Vec3 {
    let v = Fixed12Vec3::from_f32(x, y, z);
    CFixed12Vec3 { x: v.x.to_raw(), y: v.y.to_raw(), z: v.z.to_raw() }
}
#[no_mangle] pub extern "C" fn mid_fixed12_vec3_to_f32(v: CFixed12Vec3, ox: *mut f32, oy: *mut f32, oz: *mut f32) {
    let fv = Fixed12Vec3::from_raw(v.x, v.y, v.z);
    unsafe { *ox = fv.x.to_f32(); *oy = fv.y.to_f32(); *oz = fv.z.to_f32(); }
}
#[no_mangle] pub extern "C" fn mid_fixed12_vec3_add(a: CFixed12Vec3, b: CFixed12Vec3) -> CFixed12Vec3 {
    let r = Fixed12Vec3::from_raw(a.x,a.y,a.z) + Fixed12Vec3::from_raw(b.x,b.y,b.z);
    CFixed12Vec3 { x: r.x.to_raw(), y: r.y.to_raw(), z: r.z.to_raw() }
}
#[no_mangle] pub extern "C" fn mid_fixed12_vec3_dot(a: CFixed12Vec3, b: CFixed12Vec3) -> CFixed12 {
    Fixed12Vec3::from_raw(a.x,a.y,a.z).dot(Fixed12Vec3::from_raw(b.x,b.y,b.z)).to_raw()
  }
