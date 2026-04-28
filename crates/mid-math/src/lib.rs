// crates/mid-math/src/lib.rs
// REMOVED: #![cfg_attr(not(feature = "std"), no_std)]
// Reason: f32::sin/cos/sqrt/etc live in std, not core.
// no_std support requires libm — add as optional feature later (same as glam).

pub(crate) mod sse2;

pub mod deref;
pub mod f32;
pub mod ffi;
pub mod constants;

pub use constants::*;

pub use f32::Vec2;
pub use f32::mat3::Mat3;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub use f32::sse2::{Vec3, Vec4, Quat, Mat4};

#[cfg(target_arch = "aarch64")]
pub use f32::neon::{Vec3, Vec4, Quat, Mat4};

#[cfg(all(
    any(target_arch = "wasm32", target_arch = "wasm64"),
    target_feature = "simd128",
))]
pub use f32::wasm::{Vec3, Vec4, Quat, Mat4};

#[cfg(not(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(
        any(target_arch = "wasm32", target_arch = "wasm64"),
        target_feature = "simd128",
    ),
)))]
pub use f32::scalar::{Vec3, Vec4, Quat, Mat4};

// ── Scalar utilities ──────────────────────────────────────────────────────────

#[inline(always)] pub fn lerp(a: f32, b: f32, t: f32) -> f32 { a + (b - a) * t }

#[inline(always)] pub fn inverse_lerp(a: f32, b: f32, v: f32) -> f32 {
    let d = b - a;
    if d.abs() < constants::EPSILON { 0.0 } else { (v - a) / d }
}

#[inline(always)] pub fn remap(v: f32, in_min: f32, in_max: f32,
                                out_min: f32, out_max: f32) -> f32 {
    lerp(out_min, out_max, inverse_lerp(in_min, in_max, v))
}

#[inline(always)] pub fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[inline(always)] pub fn clamp(v: f32, min: f32, max: f32) -> f32 { v.clamp(min, max) }
#[inline(always)] pub fn saturate(v: f32) -> f32 { v.clamp(0.0, 1.0) }
#[inline(always)] pub fn to_radians(deg: f32) -> f32 { deg * constants::DEG2RAD }
#[inline(always)] pub fn to_degrees(rad: f32) -> f32 { rad * constants::RAD2DEG }
#[inline(always)] pub fn approx_eq(a: f32, b: f32) -> bool {
    (a - b).abs() < constants::EPSILON
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests;
