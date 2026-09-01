// crates/mid-math/src/swizzle/int8.rs
//! Swizzle impls for the i8/u8 vector family.
//!
//! All 6 types (`I8Vec2/3/4`, `U8Vec2/3/4`) are always-scalar and canonical —
//! no backend split, so no `#[cfg(...)]` needed anywhere in this file, unlike
//! `f32.rs`/`f64.rs`. Each signed/unsigned pair swizzles within itself (an
//! `I8Vec3` swizzling up to a 4-component result produces `I8Vec4`, never
//! `U8Vec4`) — there's no cross-signedness swizzling, same as there's no
//! cross-signedness anything else in this crate's int types.

crate::impl_vec2_swizzle!(crate::int8::I8Vec2, crate::int8::I8Vec3, crate::int8::I8Vec4);
crate::impl_vec3_swizzle!(crate::int8::I8Vec3, crate::int8::I8Vec2, crate::int8::I8Vec4);
crate::impl_vec4_swizzle!(crate::int8::I8Vec4, crate::int8::I8Vec2, crate::int8::I8Vec3);

crate::impl_vec2_swizzle!(crate::int8::U8Vec2, crate::int8::U8Vec3, crate::int8::U8Vec4);
crate::impl_vec3_swizzle!(crate::int8::U8Vec3, crate::int8::U8Vec2, crate::int8::U8Vec4);
crate::impl_vec4_swizzle!(crate::int8::U8Vec4, crate::int8::U8Vec2, crate::int8::U8Vec3);
