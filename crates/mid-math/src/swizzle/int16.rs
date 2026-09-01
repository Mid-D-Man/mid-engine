// crates/mid-math/src/swizzle/int16.rs
//! Swizzle impls for the i16/u16 vector family.
//!
//! All 6 types (`I16Vec2/3/4`, `U16Vec2/3/4`) are always-scalar and
//! canonical — no backend split, no `#[cfg(...)]` needed. See `int8.rs` for
//! the shared reasoning (same structure, just widened).

crate::impl_vec2_swizzle!(crate::int16::I16Vec2, crate::int16::I16Vec3, crate::int16::I16Vec4);
crate::impl_vec3_swizzle!(crate::int16::I16Vec3, crate::int16::I16Vec2, crate::int16::I16Vec4);
crate::impl_vec4_swizzle!(crate::int16::I16Vec4, crate::int16::I16Vec2, crate::int16::I16Vec3);

crate::impl_vec2_swizzle!(crate::int16::U16Vec2, crate::int16::U16Vec3, crate::int16::U16Vec4);
crate::impl_vec3_swizzle!(crate::int16::U16Vec3, crate::int16::U16Vec2, crate::int16::U16Vec4);
crate::impl_vec4_swizzle!(crate::int16::U16Vec4, crate::int16::U16Vec2, crate::int16::U16Vec3);
