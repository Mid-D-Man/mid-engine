// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-reflect.md, section "ty.rs"
// ============================================================================
//! Descriptions of types: [`Ty`] for a type as written at a use site, and
//! [`TypeInfo`] for a named struct, newtype or enum.

use std::fmt;

use crate::value::Value;

/// A type as written where it is used: a field's type, an element's type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Ty {
    /// `bool`.
    Bool,
    /// An integer of `bits` width.
    Int {
        /// Width in bits (8, 16, 32, 64; `isize`/`usize` count as 64).
        bits: u8,
        /// Whether it is signed.
        signed: bool,
    },
    /// A float of `bits` width (32 or 64).
    Float {
        /// Width in bits.
        bits: u8,
    },
    /// `String`.
    Str,
    /// `Option<T>`.
    Option(Box<Ty>),
    /// `Vec<T>`.
    List(Box<Ty>),
    /// `[T; N]`.
    Array(Box<Ty>, usize),
    /// A struct, newtype or enum, by its registered name.
    Named(String),
}

impl fmt::Display for Ty {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Ty::Bool => write!(f, "bool"),
            Ty::Int { bits, signed } => write!(f, "{}{bits}", if *signed { 'i' } else { 'u' }),
            Ty::Float { bits } => write!(f, "f{bits}"),
            Ty::Str => write!(f, "String"),
            Ty::Option(inner) => write!(f, "Option<{inner}>"),
            Ty::List(inner) => write!(f, "Vec<{inner}>"),
            Ty::Array(inner, n) => write!(f, "[{inner}; {n}]"),
            Ty::Named(name) => write!(f, "{name}"),
        }
    }
}

/// One field of a struct.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Field {
    /// The field's name.
    pub name: String,
    /// The field's type.
    pub ty: Ty,
}

/// What kind of named type this is.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Kind {
    /// A struct with named fields; its value is a map in field order.
    Struct(Vec<Field>),
    /// A struct with one unnamed field; its value is the inner value.
    Newtype(Ty),
    /// An enum with only unit variants; its value is the variant's name.
    Enum(Vec<String>),
}

/// A named type: what the registry stores about a registered type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypeInfo {
    /// The stable name the type is registered under.
    pub name: String,
    /// Its shape.
    pub kind: Kind,
}

impl TypeInfo {
    /// The description as a [`Value`], the form a schema is exported in:
    /// `{"kind":"struct","fields":[{"name":"x","type":"f32"}]}`.
    pub fn to_value(&self) -> Value {
        match &self.kind {
            Kind::Struct(fields) => Value::map([
                ("kind", Value::from("struct")),
                (
                    "fields",
                    Value::List(
                        fields
                            .iter()
                            .map(|f| {
                                Value::map([
                                    ("name", Value::from(f.name.as_str())),
                                    ("type", Value::from(f.ty.to_string())),
                                ])
                            })
                            .collect(),
                    ),
                ),
            ]),
            Kind::Newtype(inner) => Value::map([
                ("kind", Value::from("newtype")),
                ("type", Value::from(inner.to_string())),
            ]),
            Kind::Enum(variants) => Value::map([
                ("kind", Value::from("enum")),
                (
                    "variants",
                    Value::List(variants.iter().map(|v| Value::from(v.as_str())).collect()),
                ),
            ]),
        }
    }
}
