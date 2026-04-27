// crates/mid-math/src/lib.rs
//! mid-math — SIMD-aligned math primitives for Mid Engine.
//!
//! Internal types use SIMD storage on supported platforms:
//!   - x86 / x86_64 : Vec3/Vec4/Quat backed by __m128 (SSE2)
//!   - aarch64       : NEON stubs (scalar for now, full impl TODO)
//!   - wasm32/64     : v128 stubs (scalar for now, full impl TODO)
//!   - everything else: pure scalar fallback
//!
//! FFI callers use the #[repr(C)] types in `ffi::types`:
//!   CVec2, CVec3, CVec4, CQuat, CMat3, CMat4
//!
//! All internal → C and C → internal conversions are zero-cost.

#![cfg_attr(not(feature = "std"), no_std)]

// ── Shared arch helpers ───────────────────────────────────────────────────────

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub(crate) mod sse2;

// neon.rs / wasm.rs are stubs until the full implementations land.
// They live in f32/neon/ and f32/wasm/ respectively.

// ── Deref helpers for SIMD-backed types ──────────────────────────────────────

pub mod deref;

// ── Type modules ─────────────────────────────────────────────────────────────

pub mod f32;
pub mod ffi;

// Mat3 is always scalar — no SIMD benefit for 3×3.
// It is re-exported here from f32/mat3.rs.

// ── Public re-exports ─────────────────────────────────────────────────────────
//
// Users import from crate root: `use mid_math::{Vec3, Mat4, ...}`

pub use f32::Vec2;
pub use f32::{Vec3, Vec4, Quat, Mat4};
pub use f32::Mat3;

// ── Constants ─────────────────────────────────────────────────────────────────

pub mod constants;
pub use constants::*;

// ── Scalar utilities ──────────────────────────────────────────────────────────

/// Linear interpolation: `a + (b − a) × t`
#[inline(always)] pub fn lerp(a:f32, b:f32, t:f32) -> f32 { a + (b-a)*t }

/// Inverse lerp: what `t` produces `v` between `a` and `b`?
#[inline(always)] pub fn inverse_lerp(a:f32, b:f32, v:f32) -> f32 {
    let d = b - a;
    if d.abs() < constants::EPSILON { 0.0 } else { (v-a)/d }
}

/// Remap `v` from `[in_min, in_max]` to `[out_min, out_max]`.
#[inline(always)] pub fn remap(v:f32, in_min:f32, in_max:f32, out_min:f32, out_max:f32) -> f32 {
    lerp(out_min, out_max, inverse_lerp(in_min, in_max, v))
}

/// Smooth Hermite interpolation — no derivative discontinuity at edges.
#[inline(always)] pub fn smoothstep(edge0:f32, edge1:f32, x:f32) -> f32 {
    let t = ((x-edge0)/(edge1-edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0*t)
}

/// Clamp `v` to `[min, max]`.
#[inline(always)] pub fn clamp(v:f32, min:f32, max:f32) -> f32 { v.clamp(min, max) }

/// Clamp `v` to `[0, 1]`.
#[inline(always)] pub fn saturate(v:f32) -> f32 { v.clamp(0.0, 1.0) }

/// Degrees → radians.
#[inline(always)] pub fn to_radians(deg:f32) -> f32 { deg * constants::DEG2RAD }

/// Radians → degrees.
#[inline(always)] pub fn to_degrees(rad:f32) -> f32 { rad * constants::RAD2DEG }

/// Float approximate equality using EPSILON.
#[inline(always)] pub fn approx_eq(a:f32, b:f32) -> bool { (a-b).abs() < constants::EPSILON }

// ── Test modules ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests;
