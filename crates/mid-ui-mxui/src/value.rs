// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-ui-mxui.md, section "value.rs"
// ============================================================================
//! The source-neutral value tree the schema reader consumes.

/// A parsed value, independent of where it came from. The dixscript adapter
/// produces these from `DixValue`; a host can also build them by hand.
#[derive(Debug, Clone, PartialEq)]
pub enum MxValue {
    /// Absent or explicit null.
    Null,
    /// A boolean.
    Bool(bool),
    /// An integer of any width.
    Int(i64),
    /// A floating-point number of any width.
    Float(f64),
    /// A string.
    Str(String),
    /// A hex color string such as `#1E1E2E`.
    Color(String),
    /// A value of an enum declared in the file.
    Enum {
        /// Name of the enum type, for example `UiAnchor`.
        type_name: String,
        /// Name of the chosen variant, for example `Center`.
        field: String,
    },
    /// An ordered list.
    Array(Vec<MxValue>),
    /// Named fields in source order.
    Object(Vec<(String, MxValue)>),
}

impl MxValue {
    /// Builds an [`MxValue::Object`] from `(name, value)` pairs.
    pub fn object<I, K>(entries: I) -> Self
    where
        I: IntoIterator<Item = (K, MxValue)>,
        K: Into<String>,
    {
        MxValue::Object(entries.into_iter().map(|(k, v)| (k.into(), v)).collect())
    }

    /// Builds an [`MxValue::Enum`].
    pub fn enum_value(type_name: impl Into<String>, field: impl Into<String>) -> Self {
        MxValue::Enum { type_name: type_name.into(), field: field.into() }
    }

    /// Short lower-case name of this value's type, used in error messages.
    pub fn type_name(&self) -> &'static str {
        match self {
            MxValue::Null => "null",
            MxValue::Bool(_) => "bool",
            MxValue::Int(_) => "int",
            MxValue::Float(_) => "float",
            MxValue::Str(_) => "string",
            MxValue::Color(_) => "color",
            MxValue::Enum { .. } => "enum",
            MxValue::Array(_) => "array",
            MxValue::Object(_) => "object",
        }
    }

    /// The numeric value when this is an [`MxValue::Int`] or [`MxValue::Float`].
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            MxValue::Int(i) => Some(*i as f64),
            MxValue::Float(f) => Some(*f),
            _ => None,
        }
    }
}
