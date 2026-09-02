// crates/mid-math/src/features/mint_conversions/vectors.rs
//! mint conversions for every Vec2/3/4-family type, across every numeric
//! family (f32/f64 + all 8 narrow int families) — mirrors glam's own
//! `impl_vec_types!` macro exactly (same field-by-field construction, same
//! `Point2`/`Point3`/`Vector2`/`Vector3`/`Vector4` coverage — no `Point4`,
//! matching mint's own lack of one).
//!
//! One invocation per family, using each family's crate-root public alias
//! (`crate::Vec2`, `crate::DVec2`, `crate::I8Vec2`, ...) — unlike swizzle,
//! mint conversions don't need one impl per backend: a caller only ever
//! holds `crate::Vec3` (whichever concrete backend that resolves to for
//! their build), so targeting the public alias is both correct and
//! sufficient regardless of which backend is active.

macro_rules! impl_mint_vectors {
    ($t:ty, $vec2:ty, $vec3:ty, $vec4:ty) => {
        impl From<mint::Point2<$t>> for $vec2 {
            fn from(v: mint::Point2<$t>) -> Self {
                Self::new(v.x, v.y)
            }
        }
        impl From<$vec2> for mint::Point2<$t> {
            fn from(v: $vec2) -> Self {
                Self { x: v.x, y: v.y }
            }
        }
        impl From<mint::Vector2<$t>> for $vec2 {
            fn from(v: mint::Vector2<$t>) -> Self {
                Self::new(v.x, v.y)
            }
        }
        impl From<$vec2> for mint::Vector2<$t> {
            fn from(v: $vec2) -> Self {
                Self { x: v.x, y: v.y }
            }
        }
        impl mint::IntoMint for $vec2 {
            type MintType = mint::Vector2<$t>;
        }

        impl From<mint::Point3<$t>> for $vec3 {
            fn from(v: mint::Point3<$t>) -> Self {
                Self::new(v.x, v.y, v.z)
            }
        }
        impl From<$vec3> for mint::Point3<$t> {
            fn from(v: $vec3) -> Self {
                Self { x: v.x, y: v.y, z: v.z }
            }
        }
        impl From<mint::Vector3<$t>> for $vec3 {
            fn from(v: mint::Vector3<$t>) -> Self {
                Self::new(v.x, v.y, v.z)
            }
        }
        impl From<$vec3> for mint::Vector3<$t> {
            fn from(v: $vec3) -> Self {
                Self { x: v.x, y: v.y, z: v.z }
            }
        }
        impl mint::IntoMint for $vec3 {
            type MintType = mint::Vector3<$t>;
        }

        impl From<mint::Vector4<$t>> for $vec4 {
            fn from(v: mint::Vector4<$t>) -> Self {
                Self::new(v.x, v.y, v.z, v.w)
            }
        }
        impl From<$vec4> for mint::Vector4<$t> {
            fn from(v: $vec4) -> Self {
                Self { x: v.x, y: v.y, z: v.z, w: v.w }
            }
        }
        impl mint::IntoMint for $vec4 {
            type MintType = mint::Vector4<$t>;
        }
    };
}

impl_mint_vectors!(f32, crate::Vec2, crate::Vec3, crate::Vec4);
impl_mint_vectors!(f64, crate::DVec2, crate::DVec3, crate::DVec4);
impl_mint_vectors!(i8, crate::I8Vec2, crate::I8Vec3, crate::I8Vec4);
impl_mint_vectors!(u8, crate::U8Vec2, crate::U8Vec3, crate::U8Vec4);
impl_mint_vectors!(i16, crate::I16Vec2, crate::I16Vec3, crate::I16Vec4);
impl_mint_vectors!(u16, crate::U16Vec2, crate::U16Vec3, crate::U16Vec4);
impl_mint_vectors!(i32, crate::IVec2, crate::IVec3, crate::IVec4);
impl_mint_vectors!(u32, crate::UVec2, crate::UVec3, crate::UVec4);
impl_mint_vectors!(i64, crate::I64Vec2, crate::I64Vec3, crate::I64Vec4);
impl_mint_vectors!(u64, crate::U64Vec2, crate::U64Vec3, crate::U64Vec4);
