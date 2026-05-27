// crates/mid-math/src/int16/mod.rs
//! Narrow signed/unsigned 16-bit integer vector types.
//!
//! All scalar — coordinate/attribute vectors, not batch SIMD types.
//! Wide SIMD equivalents (i16x8 / u16x8) live in crate::wide::int.
//!
//! Dot products return i32/u32 (widened) because i16*i16 overflows i16.
//!
//! Layout:
//!   I16Vec2 / U16Vec2 — 4 bytes, align 2
//!   I16Vec3 / U16Vec3 — 6 bytes, align 2
//!   I16Vec4 / U16Vec4 — 8 bytes, align 2

mod i16vec2;
mod i16vec3;
mod i16vec4;
mod u16vec2;
mod u16vec3;
mod u16vec4;

pub use i16vec2::I16Vec2;
pub use i16vec3::I16Vec3;
pub use i16vec4::I16Vec4;
pub use u16vec2::U16Vec2;
pub use u16vec3::U16Vec3;
pub use u16vec4::U16Vec4;
