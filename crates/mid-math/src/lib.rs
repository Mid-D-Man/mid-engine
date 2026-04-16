// crates/mid-math/src/lib.rs

//! mid-math — SIMD-aligned math primitives for Mid Engine.
//!
//! All types are `#[repr(C)]` — safe to hand raw pointers to Unity,
//! Unreal, Godot, or any C/C++ host without a copy or translation layer.
//!
//! Storage strategy:
//! - Vec3 / Vec4 / Quat : 16-byte aligned (SSE2-friendly)
//! - Vec2               : 8 bytes, tight packing for UV / 2D physics
//! - Mat3               : 36 bytes column-major
//! - Mat4               : 64 bytes column-major, 16-byte aligned

pub mod vec;
pub mod quat;
pub mod mat;
pub mod ffi;

#[cfg(test)]
mod tests;

pub use vec::{Vec2, Vec3, Vec4};
pub use quat::Quat;
pub use mat::{Mat3, Mat4};

// ── Global constants ───────────────────────────────────────────────────────

pub const PI:        f32 = std::f32::consts::PI;
pub const TAU:       f32 = std::f32::consts::TAU;
pub const FRAC_PI_2: f32 = std::f32::consts::FRAC_PI_2;
pub const DEG2RAD:   f32 = PI / 180.0;
pub const RAD2DEG:   f32 = 180.0 / PI;
/// Epsilon for approximate float comparisons.
pub const EPSILON:   f32 = 1e-6;

// ── Scalar utilities ───────────────────────────────────────────────────────
// #[inline(always)] on every utility — these are called in tight loops
// (lerp inside ECS update, smoothstep in animation curves).
// Removing the call overhead here matches what glam does for its scalar ops.

/// Linear interpolation: `a + (b − a) × t`
#[inline(always)] pub fn lerp(a: f32, b: f32, t: f32) -> f32 { a + (b - a) * t }

/// Inverse lerp: what `t` produces `v` between `a` and `b`?
#[inline(always)] pub fn inverse_lerp(a: f32, b: f32, v: f32) -> f32 {
    if (b - a).abs() < EPSILON { 0.0 } else { (v - a) / (b - a) }
}

/// Remap `v` from `[in_min, in_max]` to `[out_min, out_max]`.
#[inline(always)] pub fn remap(v: f32, in_min: f32, in_max: f32, out_min: f32, out_max: f32) -> f32 {
    lerp(out_min, out_max, inverse_lerp(in_min, in_max, v))
}

/// Smooth Hermite interpolation — no derivative discontinuity at edges.
#[inline(always)] pub fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Clamp `v` to `[min, max]`.
#[inline(always)] pub fn clamp(v: f32, min: f32, max: f32) -> f32 { v.clamp(min, max) }

/// Clamp `v` to `[0, 1]`.
#[inline(always)] pub fn saturate(v: f32) -> f32 { v.clamp(0.0, 1.0) }

/// Degrees → radians.
#[inline(always)] pub fn to_radians(deg: f32) -> f32 { deg * DEG2RAD }

/// Radians → degrees.
#[inline(always)] pub fn to_degrees(rad: f32) -> f32 { rad * RAD2DEG }

/// Float approximate equality using `EPSILON`.
#[inline(always)] pub fn approx_eq(a: f32, b: f32) -> bool { (a - b).abs() < EPSILON }
