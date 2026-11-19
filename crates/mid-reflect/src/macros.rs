// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-reflect.md, section "macros.rs"
// ============================================================================
//! The declarative macros that implement [`Reflect`](crate::Reflect) for
//! structs, newtypes and unit-variant enums. No procedural macro, so no
//! dependency.

/// Implements [`Reflect`](crate::Reflect) for a struct with named fields.
///
/// The type is registered under the given name, which stays the same when
/// the type moves or is renamed in Rust. Every field must be listed, and
/// each field's type must itself implement `Reflect`; a missing field is a
/// compile error.
///
/// ```
/// use mid_reflect::{impl_reflect_struct, Reflect, Value};
///
/// struct Health { current: u32, max: u32 }
/// impl_reflect_struct!("game::Health" => Health { current: u32, max: u32 });
///
/// let value = Health { current: 3, max: 10 }.to_value();
/// assert_eq!(value.to_json(), r#"{"current":3,"max":10}"#);
/// ```
#[macro_export]
macro_rules! impl_reflect_struct {
    ($name:literal => $ty:ty { $($field:ident : $fty:ty),* $(,)? }) => {
        impl $crate::Reflect for $ty {
            fn ty() -> $crate::Ty {
                $crate::Ty::Named($name.to_string())
            }

            fn info() -> ::core::option::Option<$crate::TypeInfo> {
                ::core::option::Option::Some($crate::TypeInfo {
                    name: $name.to_string(),
                    kind: $crate::Kind::Struct(::std::vec![$(
                        $crate::Field {
                            name: ::core::stringify!($field).to_string(),
                            ty: <$fty as $crate::Reflect>::ty(),
                        }
                    ),*]),
                })
            }

            fn register_types(registry: &mut $crate::TypeRegistry) {
                if registry.insert_named::<Self>() {
                    $( <$fty as $crate::Reflect>::register_types(registry); )*
                }
            }

            fn to_value(&self) -> $crate::Value {
                $crate::Value::Map(::std::vec![$(
                    (
                        ::core::stringify!($field).to_string(),
                        <$fty as $crate::Reflect>::to_value(&self.$field),
                    )
                ),*])
            }

            #[allow(unused_variables)]
            fn from_value(
                value: &$crate::Value,
            ) -> ::core::result::Result<Self, $crate::ReflectError> {
                let entries = $crate::private::expect_map(value, $name)?;
                $crate::private::check_fields(entries, &[$(::core::stringify!($field)),*])?;
                ::core::result::Result::Ok(Self {$(
                    $field: <$fty as $crate::Reflect>::from_value(
                        $crate::private::field(entries, ::core::stringify!($field))?,
                    )
                    .map_err(|e| e.at_field(::core::stringify!($field)))?,
                )*})
            }
        }
    };
}

/// Implements [`Reflect`](crate::Reflect) for a tuple struct with one field.
/// Its value is the inner value, with no wrapper.
///
/// ```
/// use mid_reflect::{impl_reflect_newtype, Reflect};
///
/// struct Name(String);
/// impl_reflect_newtype!("game::Name" => Name(String));
///
/// assert_eq!(Name("rock".into()).to_value().to_json(), r#""rock""#);
/// ```
#[macro_export]
macro_rules! impl_reflect_newtype {
    ($name:literal => $ty:ident($inner:ty)) => {
        impl $crate::Reflect for $ty {
            fn ty() -> $crate::Ty {
                $crate::Ty::Named($name.to_string())
            }

            fn info() -> ::core::option::Option<$crate::TypeInfo> {
                ::core::option::Option::Some($crate::TypeInfo {
                    name: $name.to_string(),
                    kind: $crate::Kind::Newtype(<$inner as $crate::Reflect>::ty()),
                })
            }

            fn register_types(registry: &mut $crate::TypeRegistry) {
                if registry.insert_named::<Self>() {
                    <$inner as $crate::Reflect>::register_types(registry);
                }
            }

            fn to_value(&self) -> $crate::Value {
                <$inner as $crate::Reflect>::to_value(&self.0)
            }

            fn from_value(
                value: &$crate::Value,
            ) -> ::core::result::Result<Self, $crate::ReflectError> {
                <$inner as $crate::Reflect>::from_value(value).map(Self)
            }
        }
    };
}

/// Implements [`Reflect`](crate::Reflect) for an enum whose variants all
/// carry no data. Its value is the variant's name as a string.
///
/// ```
/// use mid_reflect::{impl_reflect_enum, Reflect};
///
/// enum Mode { Idle, Run }
/// impl_reflect_enum!("game::Mode" => Mode { Idle, Run });
///
/// assert_eq!(Mode::Run.to_value().to_json(), r#""Run""#);
/// ```
#[macro_export]
macro_rules! impl_reflect_enum {
    ($name:literal => $ty:ident { $($variant:ident),+ $(,)? }) => {
        impl $crate::Reflect for $ty {
            fn ty() -> $crate::Ty {
                $crate::Ty::Named($name.to_string())
            }

            fn info() -> ::core::option::Option<$crate::TypeInfo> {
                ::core::option::Option::Some($crate::TypeInfo {
                    name: $name.to_string(),
                    kind: $crate::Kind::Enum(::std::vec![
                        $(::core::stringify!($variant).to_string()),+
                    ]),
                })
            }

            fn register_types(registry: &mut $crate::TypeRegistry) {
                registry.insert_named::<Self>();
            }

            fn to_value(&self) -> $crate::Value {
                match self {
                    $( $ty::$variant => $crate::Value::Str(
                        ::core::stringify!($variant).to_string()
                    ), )+
                }
            }

            fn from_value(
                value: &$crate::Value,
            ) -> ::core::result::Result<Self, $crate::ReflectError> {
                match $crate::private::expect_variant(value, $name)? {
                    $( ::core::stringify!($variant) => ::core::result::Result::Ok($ty::$variant), )+
                    other => ::core::result::Result::Err(
                        $crate::private::unknown_variant($name, other)
                    ),
                }
            }
        }
    };
}
