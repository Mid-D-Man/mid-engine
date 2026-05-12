// crates/mid-math/src/lib.rs

// Gate the sse2 helper module — only exists on x86/x86_64.
// All callers (f32/sse2/*) are themselves gated behind the same cfg,
// so this is safe and eliminates the "not found in scope" errors.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub(crate) mod sse2;

pub mod bvec;
pub mod deref;
pub mod f32;
pub mod f64;
pub mod ffi;
pub mod constants;
pub mod int32;
pub mod int64;
pub mod wide;
// pub mod fixed; // Commented out as this module has not been started yet
pub mod curves;

pub use constants::*;

// ── Bool masks ────────────────────────────────────────────────────────────────
pub use bvec::{BVec2, BVec3, BVec4};

// ── Integer vectors (i32 / u32) ───────────────────────────────────────────────
pub use int32::{IVec2, IVec3, IVec4, UVec2, UVec3, UVec4};

// ── Integer vectors (i64 / u64) ───────────────────────────────────────────────
pub use int64::{I64Vec2, I64Vec3, I64Vec4, U64Vec2, U64Vec3, U64Vec4};

// ── f32 types ─────────────────────────────────────────────────────────────────
pub use f32::Vec2;
pub use f32::Mat2;
pub use f32::Mat3;
pub use f32::Affine3;
// pub use f32::AxisAngle; // Commented out, unresolved import in `f32`

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

// ── f64 types ─────────────────────────────────────────────────────────────────
pub use f64::{DVec2, DVec3, DVec4, DQuat, DMat2, DMat3, DMat4, DAffine3};
pub use f64::DEPSILON;

// ── Wide SIMD — integer ───────────────────────────────────────────────────────
pub use wide::int::{IMask4, IMask8, IMask16};
#[allow(non_camel_case_types)]
pub use wide::int::{i32x4, u32x4, i16x8, u16x8, i8x16, u8x16};

// ── Wide SIMD — float ─────────────────────────────────────────────────────────
pub use wide::float::Mask4;
#[allow(non_camel_case_types)]
pub use wide::float::f32x4;
pub use wide::float::Vec3x4;
pub use wide::float::QuatX4;

#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    target_feature = "avx2",
))]
pub use wide::float::Vec3x8;

// ── Fixed-point ───────────────────────────────────────────────────────────────
// pub use fixed::{Fixed, Fixed8, Fixed12, Fixed16, FixedVec2, FixedVec3}; // Commented out with the module

// ── Curves ────────────────────────────────────────────────────────────────────
pub use curves::Interpolate;
pub use curves::catmull_rom::{CatmullRom, CatmullRomAlpha};
pub use curves::hermite::HermiteSpline;
pub use curves::hermite::HermiteKey;
pub use curves::kochanek_bartels::{KochanekBartels, TcbKey};
pub use curves::quadratic_bezier::QuadraticBezier;
pub use curves::cubic_bezier::CubicBezier;
pub use curves::cardinal::CardinalSpline;
pub use curves::bspline::BSpline;

// ── Scalar utilities ──────────────────────────────────────────────────────────
#[inline(always)] pub fn lerp(a: f32, b: f32, t: f32) -> f32 { a + (b - a) * t }
#[inline(always)] pub fn inverse_lerp(a: f32, b: f32, v: f32) -> f32 {
    let d = b - a;
    if d.abs() < constants::EPSILON { 0.0 } else { (v - a) / d }
}
#[inline(always)] pub fn remap(v: f32, in_min: f32, in_max: f32, out_min: f32, out_max: f32) -> f32 {
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
#[inline(always)] pub fn approx_eq(a: f32, b: f32) -> bool { (a - b).abs() < constants::EPSILON }

#[cfg(test)]
mod tests;
