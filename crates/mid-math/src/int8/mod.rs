// crates/mid-math/src/int8/mod.rs
//! Narrow signed/unsigned 8-bit integer vector types.
//!
//! All scalar — these are coordinate/attribute vectors, not batch SIMD types.
//! The wide SIMD equivalents (i8x16 / u8x16) live in crate::wide::int.
//!
//! Dot products return i16/u16 (widened) because i8*i8 overflows i8.
//!
//! Layout:
//!   I8Vec2 / U8Vec2 — 2 bytes, align 1
//!   I8Vec3 / U8Vec3 — 3 bytes, align 1
//!   I8Vec4 / U8Vec4 — 4 bytes, align 1

mod i8vec2;
mod i8vec3;
mod i8vec4;
mod u8vec2;
mod u8vec3;
mod u8vec4;

pub use i8vec2::I8Vec2;
pub use i8vec3::I8Vec3;
pub use i8vec4::I8Vec4;
pub use u8vec2::U8Vec2;
pub use u8vec3::U8Vec3;
pub use u8vec4::U8Vec4;
