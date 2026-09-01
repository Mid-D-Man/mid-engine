// crates/mid-math/src/swizzle/int32.rs
//! Swizzle impls for the i32/u32 vector family.
//!
//! All 6 types (`IVec2/3/4`, `UVec2/3/4` — no `32` in the name, per this
//! crate's own naming: i32/u32 are the "plain" int types) are always-scalar
//! and canonical. No `#[cfg(...)]` needed. See `int8.rs` for the shared
//! reasoning.

crate::impl_vec2_swizzle!(crate::int32::IVec2, crate::int32::IVec3, crate::int32::IVec4);
crate::impl_vec3_swizzle!(crate::int32::IVec3, crate::int32::IVec2, crate::int32::IVec4);
crate::impl_vec4_swizzle!(crate::int32::IVec4, crate::int32::IVec2, crate::int32::IVec3);

crate::impl_vec2_swizzle!(crate::int32::UVec2, crate::int32::UVec3, crate::int32::UVec4);
crate::impl_vec3_swizzle!(crate::int32::UVec3, crate::int32::UVec2, crate::int32::UVec4);
crate::impl_vec4_swizzle!(crate::int32::UVec4, crate::int32::UVec2, crate::int32::UVec3);
