// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-reflect.md, section "math.rs"
// ============================================================================
//! [`Reflect`](crate::Reflect) for `mid-math`'s vectors, quaternions and
//! affine transforms (feature `math`). Vectors and quaternions are maps of
//! their components (`{"x":1.0,"y":2.0,"z":3.0}`).

use mid_math::{Affine3, DAffine3, DQuat, DVec2, DVec3, DVec4, Quat, Vec2, Vec3, Vec4};

use crate::{impl_reflect_struct, Reflect, ReflectError, Ty, TypeInfo, TypeRegistry, Value};
use crate::{Field, Kind};

/// Components read through the type's accessors and built with its
/// constructor, because the fields of the SIMD-backed types are not plain
/// struct fields.
macro_rules! impl_reflect_components {
    ($name:literal => $ty:ty, $scalar:ty, $ctor:path, [$($c:ident),+]) => {
        impl Reflect for $ty {
            fn ty() -> Ty {
                Ty::Named($name.to_string())
            }

            fn info() -> Option<TypeInfo> {
                Some(TypeInfo {
                    name: $name.to_string(),
                    kind: Kind::Struct(vec![$(Field {
                        name: stringify!($c).to_string(),
                        ty: <$scalar as Reflect>::ty(),
                    }),+]),
                })
            }

            fn register_types(registry: &mut TypeRegistry) {
                registry.insert_named::<Self>();
            }

            fn to_value(&self) -> Value {
                Value::Map(vec![$((
                    stringify!($c).to_string(),
                    <$scalar as Reflect>::to_value(&self.$c),
                )),+])
            }

            fn from_value(value: &Value) -> Result<Self, ReflectError> {
                let entries = crate::private::expect_map(value, $name)?;
                crate::private::check_fields(entries, &[$(stringify!($c)),+])?;
                $(
                    let $c = <$scalar as Reflect>::from_value(
                        crate::private::field(entries, stringify!($c))?,
                    )
                    .map_err(|e| e.at_field(stringify!($c)))?;
                )+
                Ok($ctor($($c),+))
            }
        }
    };
}

impl_reflect_components!("mid_math::Vec2" => Vec2, f32, Vec2::new, [x, y]);
impl_reflect_components!("mid_math::Vec3" => Vec3, f32, Vec3::new, [x, y, z]);
impl_reflect_components!("mid_math::Vec4" => Vec4, f32, Vec4::new, [x, y, z, w]);
impl_reflect_components!("mid_math::Quat" => Quat, f32, Quat::from_xyzw, [x, y, z, w]);
impl_reflect_components!("mid_math::DVec2" => DVec2, f64, DVec2::new, [x, y]);
impl_reflect_components!("mid_math::DVec3" => DVec3, f64, DVec3::new, [x, y, z]);
impl_reflect_components!("mid_math::DVec4" => DVec4, f64, DVec4::new, [x, y, z, w]);
impl_reflect_components!("mid_math::DQuat" => DQuat, f64, DQuat::from_xyzw, [x, y, z, w]);

impl_reflect_struct!("mid_math::Affine3" => Affine3 {
    x_axis: Vec3,
    y_axis: Vec3,
    z_axis: Vec3,
    translation: Vec3,
});

impl_reflect_struct!("mid_math::DAffine3" => DAffine3 {
    x_axis: DVec3,
    y_axis: DVec3,
    z_axis: DVec3,
    translation: DVec3,
});
