// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-reflect.md, section "reflect.rs"
// ============================================================================
//! The [`Reflect`] trait and its implementations for the primitive types,
//! `String`, `Option<T>`, `Vec<T>` and `[T; N]`.

use crate::error::{ErrorKind, ReflectError};
use crate::registry::TypeRegistry;
use crate::ty::{Ty, TypeInfo};
use crate::value::Value;

/// A type that can describe itself and be turned into and built from a
/// [`Value`].
///
/// Structs, newtypes and enums get an implementation from
/// [`impl_reflect_struct!`](crate::impl_reflect_struct),
/// [`impl_reflect_newtype!`](crate::impl_reflect_newtype) and
/// [`impl_reflect_enum!`](crate::impl_reflect_enum).
pub trait Reflect: Sized + Send + Sync + 'static {
    /// The type as written at a use site (a field's type, an element's type).
    fn ty() -> Ty;

    /// The description of a named type; `None` for the primitives and the
    /// containers, which have no name of their own.
    fn info() -> Option<TypeInfo> {
        None
    }

    /// Registers this type, if it is named, and every named type it
    /// contains. Does nothing for the primitives.
    fn register_types(_registry: &mut TypeRegistry) {}

    /// This value as a [`Value`].
    fn to_value(&self) -> Value;

    /// Builds a value of this type. Strict: a struct needs exactly its
    /// fields, a number must fit.
    fn from_value(value: &Value) -> Result<Self, ReflectError>;
}

/// Support for the macros. Not part of the interface.
#[doc(hidden)]
pub mod private {
    use super::*;

    /// The entries of a map, or a type mismatch naming `name`.
    pub fn expect_map<'a>(
        value: &'a Value,
        name: &str,
    ) -> Result<&'a [(String, Value)], ReflectError> {
        match value {
            Value::Map(entries) => Ok(entries),
            other => Err(ReflectError::mismatch(name, other)),
        }
    }

    /// Fails on the first entry whose key is not in `known`.
    pub fn check_fields(entries: &[(String, Value)], known: &[&str]) -> Result<(), ReflectError> {
        for (key, _) in entries {
            if !known.contains(&key.as_str()) {
                return Err(ReflectError::new(ErrorKind::UnknownField(key.clone())));
            }
        }
        Ok(())
    }

    /// The entry named `name`, or a missing-field error.
    pub fn field<'a>(
        entries: &'a [(String, Value)],
        name: &str,
    ) -> Result<&'a Value, ReflectError> {
        entries
            .iter()
            .find(|(k, _)| k == name)
            .map(|(_, v)| v)
            .ok_or_else(|| ReflectError::new(ErrorKind::MissingField(name.to_string())))
    }

    /// The variant name in `value`, or a mismatch.
    pub fn expect_variant<'a>(value: &'a Value, name: &str) -> Result<&'a str, ReflectError> {
        match value {
            Value::Str(s) => Ok(s),
            other => Err(ReflectError::mismatch(name, other)),
        }
    }

    /// An unknown-variant error.
    pub fn unknown_variant(ty: &str, found: &str) -> ReflectError {
        ReflectError::new(ErrorKind::UnknownVariant {
            ty: ty.to_string(),
            found: found.to_string(),
        })
    }
}

impl Reflect for bool {
    fn ty() -> Ty {
        Ty::Bool
    }

    fn to_value(&self) -> Value {
        Value::Bool(*self)
    }

    fn from_value(value: &Value) -> Result<Self, ReflectError> {
        value
            .as_bool()
            .ok_or_else(|| ReflectError::mismatch("bool", value))
    }
}

impl Reflect for String {
    fn ty() -> Ty {
        Ty::Str
    }

    fn to_value(&self) -> Value {
        Value::Str(self.clone())
    }

    fn from_value(value: &Value) -> Result<Self, ReflectError> {
        value
            .as_str()
            .map(str::to_string)
            .ok_or_else(|| ReflectError::mismatch("String", value))
    }
}

fn out_of_range(expected: &'static str, found: String) -> ReflectError {
    ReflectError::new(ErrorKind::OutOfRange { expected, found })
}

/// An integer-valued float, as an `i128`, if it is one and is not absurdly
/// large.
fn whole_float(f: f64) -> Option<i128> {
    if f.is_finite() && f.fract() == 0.0 && f.abs() < 1.0e30 {
        Some(f as i128)
    } else {
        None
    }
}

macro_rules! impl_signed {
    ($($t:ty, $bits:expr, $name:literal;)*) => {$(
        impl Reflect for $t {
            fn ty() -> Ty {
                Ty::Int { bits: $bits, signed: true }
            }

            fn to_value(&self) -> Value {
                Value::Int(*self as i64)
            }

            fn from_value(value: &Value) -> Result<Self, ReflectError> {
                match value {
                    Value::Int(i) => <$t>::try_from(*i)
                        .map_err(|_| out_of_range($name, i.to_string())),
                    Value::UInt(u) => <$t>::try_from(*u)
                        .map_err(|_| out_of_range($name, u.to_string())),
                    Value::Float(f) => whole_float(*f)
                        .ok_or_else(|| ReflectError::mismatch("an integer", value))
                        .and_then(|i| {
                            <$t>::try_from(i).map_err(|_| out_of_range($name, i.to_string()))
                        }),
                    other => Err(ReflectError::mismatch($name, other)),
                }
            }
        }
    )*};
}

macro_rules! impl_unsigned {
    ($($t:ty, $bits:expr, $name:literal;)*) => {$(
        impl Reflect for $t {
            fn ty() -> Ty {
                Ty::Int { bits: $bits, signed: false }
            }

            fn to_value(&self) -> Value {
                Value::from_u64(*self as u64)
            }

            fn from_value(value: &Value) -> Result<Self, ReflectError> {
                match value {
                    Value::Int(i) => <$t>::try_from(*i)
                        .map_err(|_| out_of_range($name, i.to_string())),
                    Value::UInt(u) => <$t>::try_from(*u)
                        .map_err(|_| out_of_range($name, u.to_string())),
                    Value::Float(f) => whole_float(*f)
                        .ok_or_else(|| ReflectError::mismatch("an integer", value))
                        .and_then(|i| {
                            <$t>::try_from(i).map_err(|_| out_of_range($name, i.to_string()))
                        }),
                    other => Err(ReflectError::mismatch($name, other)),
                }
            }
        }
    )*};
}

impl_signed! {
    i8, 8, "i8";
    i16, 16, "i16";
    i32, 32, "i32";
    i64, 64, "i64";
    isize, 64, "isize";
}

impl_unsigned! {
    u8, 8, "u8";
    u16, 16, "u16";
    u32, 32, "u32";
    u64, 64, "u64";
    usize, 64, "usize";
}

/// The special floats written as strings by the JSON module.
fn special_float(text: &str) -> Option<f64> {
    match text {
        "NaN" => Some(f64::NAN),
        "Infinity" => Some(f64::INFINITY),
        "-Infinity" => Some(f64::NEG_INFINITY),
        _ => None,
    }
}

fn float_of(value: &Value, expected: &str) -> Result<f64, ReflectError> {
    match value {
        Value::Float(f) => Ok(*f),
        Value::Int(i) => Ok(*i as f64),
        Value::UInt(u) => Ok(*u as f64),
        Value::Str(s) => special_float(s).ok_or_else(|| ReflectError::mismatch(expected, value)),
        other => Err(ReflectError::mismatch(expected, other)),
    }
}

impl Reflect for f64 {
    fn ty() -> Ty {
        Ty::Float { bits: 64 }
    }

    fn to_value(&self) -> Value {
        Value::Float(*self)
    }

    fn from_value(value: &Value) -> Result<Self, ReflectError> {
        float_of(value, "f64")
    }
}

impl Reflect for f32 {
    fn ty() -> Ty {
        Ty::Float { bits: 32 }
    }

    /// Stored as the `f64` nearest the shortest decimal text of the `f32`,
    /// so `0.1f32` becomes `0.1` and not `0.10000000149011612`.
    fn to_value(&self) -> Value {
        let wide = if self.is_finite() {
            self.to_string()
                .parse::<f64>()
                .unwrap_or_else(|_| f64::from(*self))
        } else {
            f64::from(*self)
        };
        Value::Float(wide)
    }

    fn from_value(value: &Value) -> Result<Self, ReflectError> {
        let wide = float_of(value, "f32")?;
        let narrow = wide as f32;
        if wide.is_finite() && !narrow.is_finite() {
            return Err(out_of_range("f32", wide.to_string()));
        }
        Ok(narrow)
    }
}

impl<T: Reflect> Reflect for Option<T> {
    fn ty() -> Ty {
        Ty::Option(Box::new(T::ty()))
    }

    fn register_types(registry: &mut TypeRegistry) {
        T::register_types(registry);
    }

    fn to_value(&self) -> Value {
        match self {
            Some(inner) => inner.to_value(),
            None => Value::Null,
        }
    }

    fn from_value(value: &Value) -> Result<Self, ReflectError> {
        match value {
            Value::Null => Ok(None),
            other => T::from_value(other).map(Some),
        }
    }
}

impl<T: Reflect> Reflect for Vec<T> {
    fn ty() -> Ty {
        Ty::List(Box::new(T::ty()))
    }

    fn register_types(registry: &mut TypeRegistry) {
        T::register_types(registry);
    }

    fn to_value(&self) -> Value {
        Value::List(self.iter().map(Reflect::to_value).collect())
    }

    fn from_value(value: &Value) -> Result<Self, ReflectError> {
        let Value::List(items) = value else {
            return Err(ReflectError::mismatch(Self::ty().to_string(), value));
        };
        items
            .iter()
            .enumerate()
            .map(|(i, item)| T::from_value(item).map_err(|e| e.at_index(i)))
            .collect()
    }
}

impl<T: Reflect, const N: usize> Reflect for [T; N] {
    fn ty() -> Ty {
        Ty::Array(Box::new(T::ty()), N)
    }

    fn register_types(registry: &mut TypeRegistry) {
        T::register_types(registry);
    }

    fn to_value(&self) -> Value {
        Value::List(self.iter().map(Reflect::to_value).collect())
    }

    fn from_value(value: &Value) -> Result<Self, ReflectError> {
        let Value::List(items) = value else {
            return Err(ReflectError::mismatch(Self::ty().to_string(), value));
        };
        if items.len() != N {
            return Err(ReflectError::new(ErrorKind::BadLength {
                expected: N,
                found: items.len(),
            }));
        }
        let built: Vec<T> = items
            .iter()
            .enumerate()
            .map(|(i, item)| T::from_value(item).map_err(|e| e.at_index(i)))
            .collect::<Result<_, _>>()?;
        <[T; N]>::try_from(built).map_err(|rest| {
            ReflectError::new(ErrorKind::BadLength {
                expected: N,
                found: rest.len(),
            })
        })
    }
}
