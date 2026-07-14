// crates/mid-math/src/lib.rs
#![cfg_attr(feature = "coresimd", feature(portable_simd))]

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub(crate) mod sse2;

#[cfg(target_arch = "aarch64")]
pub(crate) mod neon;

#[cfg(all(
    any(target_arch = "wasm32", target_arch = "wasm64"),
    target_feature = "simd128",
))]
pub(crate) mod wasm;

// ── Storage / low-precision types ─────────────────────────────────────────────
pub mod storage;

// ── Core math ─────────────────────────────────────────────────────────────────
pub mod bvec;
pub mod deref;
pub mod f32;
pub mod f64;
pub mod ffi;
pub mod constants;
pub mod int8;
pub mod int16;
pub mod int32;
pub mod int64;
pub mod wide;
pub mod curves;
pub mod fixed;

// ── Supplementary systems ─────────────────────────────────────────────────────
pub mod color;
pub mod helpers;
pub mod ran_gen;
pub mod string;
pub mod noise;
pub mod camera;
pub mod geom;

// ── Small-vector storage ──────────────────────────────────────────────────────
pub mod mid_vec;

pub use constants::*;

// ── Storage — low-precision boundary types ────────────────────────────────────
// Rule: store compressed → unpack to f32 only for arithmetic.
// BitMask* : compact bit-packed booleans (NOT SIMD masks — those are in `wide`).

// f16 — IEEE half-precision (GPU normals, HDR, bone transforms)
#[allow(non_camel_case_types)]
pub use storage::f16;
pub use storage::{
    f32x4_to_f16x4, f16x4_to_f32x4,
    f32x8_to_f16x8, f16x8_to_f32x8,
    f32_slice_to_f16, f16_slice_to_f32,
};

// bf16 — bfloat16 (ML training; same exponent range as f32, no overflow risk)
#[allow(non_camel_case_types)]
pub use storage::bf16;
pub use storage::{
    f32x4_to_bf16x4, bf16x4_to_f32x4,
    f32x8_to_bf16x8, bf16x8_to_f32x8,
    f32_slice_to_bf16, bf16_slice_to_f32,
};

// f8 — 8-bit floats (ML weights / activations / gradients)
pub use storage::{F8Norm, F8E4M3, F8E5M2};
pub use storage::{
    f32x4_to_f8e4m3x4, f8e4m3x4_to_f32x4,
    f32x4_to_f8e5m2x4, f8e5m2x4_to_f32x4,
};

// f4 — 4-bit floats (ultra-compressed ML weights; two values per byte)
pub use storage::{F4E2M1, F4E3M0, F4E2M1Pair, F4E3M0Pair};
pub use storage::{
    f32x8_to_f4e2m1x4pairs, f4e2m1x4pairs_to_f32x8,
    f32x8_to_f4e3m0x4pairs, f4e3m0x4pairs_to_f32x8,
    f32_slice_to_f4e2m1_packed, f4e2m1_packed_to_f32_slice,
};

// Storage masks — 1 bit per boolean (ECS queries, bone flags, visibility)
pub use storage::{
    BitMask8, BitMask16, BitMask32, BitMask64,
    BitMask128, BitMask256,
};

// ── Bool masks ────────────────────────────────────────────────────────────────
pub use bvec::{BVec2, BVec3, BVec4};

// ── Integer vectors (i8 / u8) ─────────────────────────────────────────────────
pub use int8::{I8Vec2, I8Vec3, I8Vec4, U8Vec2, U8Vec3, U8Vec4};

// ── Integer vectors (i16 / u16) ───────────────────────────────────────────────
pub use int16::{I16Vec2, I16Vec3, I16Vec4, U16Vec2, U16Vec3, U16Vec4};

// ── Integer vectors (i32 / u32) ───────────────────────────────────────────────
pub use int32::{IVec2, IVec3, IVec4, UVec2, UVec3, UVec4};

// ── Integer vectors (i64 / u64) ───────────────────────────────────────────────
pub use int64::{I64Vec2, I64Vec3, I64Vec4, U64Vec2, U64Vec3, U64Vec4};

// ── f32 types ─────────────────────────────────────────────────────────────────
pub use f32::Vec2;
pub use f32::Mat2;
pub use f32::Mat3;
pub use f32::Affine2;
pub use f32::Affine3;
pub use f32::DualQuat;

// Platform dispatch: SSE2 → NEON → WASM → coresimd → scalar
//
// NOTE: these conditions must mirror f32/mod.rs's internal aliasing exactly.
// They used to omit `not(feature = "force-scalar")`/`feature = "force-scalar"`
// entirely, which meant force-scalar had NO EFFECT on the actual public
// Vec3/Vec4/Quat/Mat4 types (this block bypasses f32::mod's internal alias
// with a direct path to f32::sse2/neon/wasm/scalar) — only Mat2 was ever
// actually forced to scalar, since it re-exports via the correctly-gated
// `pub use f32::Mat2;` above instead of a separate direct-path export like
// this block. Confirmed via `mem::align_of`: Mat2 was 4 (scalar) under
// force-scalar while Vec3/Vec4/Quat/Mat4 stayed 16 (still SSE2) — this is
// why "scalar" CI benchmark runs looked like SSE2 for everything except Mat2.
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), not(feature = "force-scalar")))]
pub use f32::sse2::{Vec3, Vec4, Quat, Mat4};

#[cfg(all(target_arch = "aarch64", not(feature = "force-scalar")))]
pub use f32::neon::{Vec3, Vec4, Quat, Mat4};

#[cfg(all(
    any(target_arch = "wasm32", target_arch = "wasm64"),
    target_feature = "simd128",
    not(feature = "force-scalar"),
))]
pub use f32::wasm::{Vec3, Vec4, Quat, Mat4};

#[cfg(all(
    feature = "coresimd",
    not(feature = "force-scalar"),
    not(any(
        target_arch = "x86",
        target_arch = "x86_64",
        target_arch = "aarch64",
        all(any(target_arch = "wasm32", target_arch = "wasm64"), target_feature = "simd128"),
    )),
))]
pub use f32::coresimd::{Vec3, Vec4, Quat, Mat4};

#[cfg(any(
    feature = "force-scalar",
    not(any(
        target_arch = "x86",
        target_arch = "x86_64",
        target_arch = "aarch64",
        all(any(target_arch = "wasm32", target_arch = "wasm64"), target_feature = "simd128"),
        feature = "coresimd",
    )),
))]
pub use f32::scalar::{Vec3, Vec4, Quat, Mat4};

// ── f64 types ─────────────────────────────────────────────────────────────────
pub use f64::{
    DVec2, DVec3, DVec4, DQuat,
    DMat2, DMat3, DMat4,
    DAffine2, DAffine3,
    DDualQuat,
    DEPSILON,
};

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

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), target_feature = "avx2"))]
pub use wide::float::Vec3x8;

// ── Curves ────────────────────────────────────────────────────────────────────
pub use curves::{
    Interpolate, QuadraticBezier, CubicBezier, CatmullRom, CatmullRomAlpha,
    HermiteSpline, HermiteKey, KochanekBartels, TcbKey, CardinalSpline, BSpline,
};

// ── Fixed-point ───────────────────────────────────────────────────────────────
pub use fixed::{
    Fixed, FixedVec2, FixedVec3,
    Fixed8,  Fixed12,  Fixed16,
    Fixed8Vec2,  Fixed12Vec2,  Fixed16Vec2,
    Fixed8Vec3,  Fixed12Vec3,  Fixed16Vec3,
    FixedVec2_8,  FixedVec2_12,  FixedVec2_16,
    FixedVec3_8,  FixedVec3_12,  FixedVec3_16,
};

// ── Color ─────────────────────────────────────────────────────────────────────
pub use color::{
    Color32, Rgb, Rgba, Hsv, Hsl, Rgbe, LogLuv32, YCbCr, YCbCrStandard,
    srgb_to_linear, linear_to_srgb,
};

// ── Helpers ───────────────────────────────────────────────────────────────────
pub use helpers::angle::{Radians, Degrees};
pub use helpers::euler::{EulerRot, QuatExt};
pub use helpers::rotor::Rotor3;
pub use helpers::spatial::{SpatialVelocity, SpatialForce, SpatialInertia};
pub use helpers::tangent::{TangentFrame, PackedTangent};
pub use helpers::octahedral::{
    encode_octahedral, decode_octahedral,
    encode_octahedral_snorm8,  decode_octahedral_snorm8,
    encode_octahedral_snorm16, decode_octahedral_snorm16,
};

// ── RNG ───────────────────────────────────────────────────────────────────────
pub use ran_gen::prng::Xorshift64;
pub use ran_gen::pcg::Pcg32;

// ── String hashing ────────────────────────────────────────────────────────────
pub use string::StringId;

// ── Noise ─────────────────────────────────────────────────────────────────────
pub use noise::{
    Perlin, Simplex, Value, Worley,
    Fbm, DomainWarp,
    NoiseSource2d, NoiseSource3d, NoiseSource4d,
    worley::{DistanceMode, DistanceMetric},
};

// ── Camera ────────────────────────────────────────────────────────────────────
pub use camera::frustum::{
    Frustum, Visibility,
    FRUSTUM_LEFT, FRUSTUM_RIGHT, FRUSTUM_BOTTOM,
    FRUSTUM_TOP,  FRUSTUM_NEAR,  FRUSTUM_FAR,
};
pub use camera::{
    PerspectiveParams, unproject, unproject_separate, picking_ray,
    perspective_infinite_rh, perspective_infinite_rh_gl, perspective_reversed_z_rh,
    perspective_infinite_lh, perspective_reversed_z_lh,
    perspective_decompose, perspective_resize,
    csm_split_depths, sub_frustum_corners,
};

// ── Geometry ──────────────────────────────────────────────────────────────────
pub use geom::barycentric::{
    BarycentricCoords, Triangle2, Triangle3,
    signed_area_2d, triangle_area_3d,
};

// ── MidVec — inline small-vector ─────────────────────────────────────────────
pub use mid_vec::MidVec;

// ── Scalar utilities ──────────────────────────────────────────────────────────

/// Linear interpolation between `a` and `b` at parameter `t`.
#[inline(always)]
pub fn lerp(a: f32, b: f32, t: f32) -> f32 { a + (b - a) * t }

/// Inverse of `lerp`: returns the `t` such that `lerp(a, b, t) == v`.
/// Returns `0.0` if `a == b`.
#[inline(always)]
pub fn inverse_lerp(a: f32, b: f32, v: f32) -> f32 {
    let d = b - a;
    if d.abs() < constants::EPSILON { 0.0 } else { (v - a) / d }
}

/// Remap `v` from `[in_min, in_max]` to `[out_min, out_max]`.
#[inline(always)]
pub fn remap(v: f32, in_min: f32, in_max: f32, out_min: f32, out_max: f32) -> f32 {
    lerp(out_min, out_max, inverse_lerp(in_min, in_max, v))
}

/// Cubic Hermite smoothstep. Returns `0` below `edge0`, `1` above `edge1`.
/// `3t² − 2t³` curve — C1 continuous.
#[inline(always)]
pub fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Quintic smootherstep (Ken Perlin's improved version).
/// `6t⁵ − 15t⁴ + 10t³` — C2 continuous, zero first and second derivative at edges.
#[inline(always)]
pub fn smootherstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

/// Step function: `0.0` if `x < edge`, `1.0` otherwise.
#[inline(always)]
pub fn step(edge: f32, x: f32) -> f32 {
    if x < edge { 0.0 } else { 1.0 }
}

/// Sign of `x`: `+1.0`, `-1.0`, or `0.0`. Delegates to `f32::signum`.
#[inline(always)]
pub fn sign(x: f32) -> f32 { x.signum() }

/// Fractional part of `x`: `x - floor(x)`. Always in `[0, 1)`.
#[inline(always)]
pub fn fract(x: f32) -> f32 { x - x.floor() }

/// Ping-pong `t` between `0` and `length`.
#[inline(always)]
pub fn ping_pong(t: f32, length: f32) -> f32 {
    if length <= 0.0 { return 0.0; }
    let t = (t - length * (t / length).floor()).abs();
    if t > length { 2.0 * length - t } else { t }
}

/// Shortest signed delta from angle `current` to angle `target` (radians).
/// Result is in `(-π, π]`.
#[inline(always)]
pub fn delta_angle(current: f32, target: f32) -> f32 {
    let mut d = (target - current) % constants::TAU;
    if d > constants::PI       { d -= constants::TAU; }
    else if d < -constants::PI { d += constants::TAU; }
    d
}

/// Move `current` toward `target` by at most `max_delta`. Never overshoots.
#[inline(always)]
pub fn move_towards(current: f32, target: f32, max_delta: f32) -> f32 {
    let diff = target - current;
    if diff.abs() <= max_delta { target } else { current + diff.signum() * max_delta }
}

/// `x²`.
#[inline(always)] pub fn pow2(x: f32) -> f32 { x * x }
/// `x³`.
#[inline(always)] pub fn pow3(x: f32) -> f32 { x * x * x }

#[inline(always)] pub fn clamp(v: f32, min: f32, max: f32) -> f32 { v.clamp(min, max) }
#[inline(always)] pub fn saturate(v: f32) -> f32 { v.clamp(0.0, 1.0) }
#[inline(always)] pub fn to_radians(deg: f32) -> f32 { deg * constants::DEG2RAD }
#[inline(always)] pub fn to_degrees(rad: f32) -> f32 { rad * constants::RAD2DEG }
#[inline(always)] pub fn approx_eq(a: f32, b: f32) -> bool { (a - b).abs() < constants::EPSILON }

#[cfg(test)]
mod tests;
