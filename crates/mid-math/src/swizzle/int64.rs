// crates/mid-math/src/swizzle/int64.rs
//! Swizzle impls for the i64/u64 vector family.
//!
//! All 6 types (`I64Vec2/3/4`, `U64Vec2/3/4`) are always-scalar and
//! canonical — no backend split, no `#[cfg(...)]` needed. See `int8.rs` for
//! the shared reasoning (same structure, just widened).

crate::impl_vec2_swizzle!(crate::int64::I64Vec2, crate::int64::I64Vec3, crate::int64::I64Vec4);
crate::impl_vec3_swizzle!(crate::int64::I64Vec3, crate::int64::I64Vec2, crate::int64::I64Vec4);
crate::impl_vec4_swizzle!(crate::int64::I64Vec4, crate::int64::I64Vec2, crate::int64::I64Vec3);

crate::impl_vec2_swizzle!(crate::int64::U64Vec2, crate::int64::U64Vec3, crate::int64::U64Vec4);
crate::impl_vec3_swizzle!(crate::int64::U64Vec3, crate::int64::U64Vec2, crate::int64::U64Vec4);
crate::impl_vec4_swizzle!(crate::int64::U64Vec4, crate::int64::U64Vec2, crate::int64::U64Vec3);
