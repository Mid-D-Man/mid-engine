// crates/mid-math/src/int64/mod.rs
//! Integer vector types — i64 and u64.
//!
//! Phase 3B. Mirrors int32 layout exactly but with 64-bit elements.
//! Used for: large world coordinates, nanosecond timestamps, entity IDs
//! that exceed u32::MAX, physics accumulator precision.
//!
//! Layout (no padding — matches Rust primitive alignment):
//!   I64Vec2 / U64Vec2 — 16 bytes, align 8
//!   I64Vec3 / U64Vec3 — 24 bytes, align 8   ← no padding, intentional
//!   I64Vec4 / U64Vec4 — 32 bytes, align 8

mod i64vec2;
mod i64vec3;
mod i64vec4;
mod u64vec2;
mod u64vec3;
mod u64vec4;

pub use i64vec2::I64Vec2;
pub use i64vec3::I64Vec3;
pub use i64vec4::I64Vec4;
pub use u64vec2::U64Vec2;
pub use u64vec3::U64Vec3;
pub use u64vec4::U64Vec4;
