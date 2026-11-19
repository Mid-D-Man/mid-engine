// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-reflect.md, section "error.rs"
// ============================================================================
//! Errors from reading and writing values: what went wrong, and where.

use std::fmt;

use crate::value::Value;

/// One step of a path into a value: a field name or a list index.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PathSeg {
    /// A field of a map (a struct).
    Field(String),
    /// An element of a list.
    Index(usize),
}

/// What went wrong, without the location.
#[derive(Debug, Clone, PartialEq)]
pub enum ErrorKind {
    /// A value of one shape was found where another was expected.
    TypeMismatch {
        /// What was expected, as written in a schema (`f32`, `Vec<Name>`).
        expected: String,
        /// The shape of the value that was found.
        found: &'static str,
    },
    /// A number does not fit the type it was read into.
    OutOfRange {
        /// The type that was being read.
        expected: &'static str,
        /// The offending number, as text.
        found: String,
    },
    /// A struct needs a field that the value does not have.
    MissingField(String),
    /// A value has a field the struct does not have.
    UnknownField(String),
    /// A string names no variant of the enum.
    UnknownVariant {
        /// The enum's registered name.
        ty: String,
        /// The text that was found.
        found: String,
    },
    /// A fixed-size array was given the wrong number of elements.
    BadLength {
        /// The array's length.
        expected: usize,
        /// The number of elements found.
        found: usize,
    },
    /// A path could not be followed, or could not be parsed.
    BadPath(String),
    /// No type is registered under the name.
    UnknownType(String),
    /// A value of a different Rust type was passed than the one registered.
    WrongType(String),
}

impl fmt::Display for ErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ErrorKind::TypeMismatch { expected, found } => {
                write!(f, "expected {expected}, found {found}")
            }
            ErrorKind::OutOfRange { expected, found } => {
                write!(f, "{found} does not fit {expected}")
            }
            ErrorKind::MissingField(name) => write!(f, "missing field `{name}`"),
            ErrorKind::UnknownField(name) => write!(f, "unknown field `{name}`"),
            ErrorKind::UnknownVariant { ty, found } => {
                write!(f, "`{found}` is not a variant of {ty}")
            }
            ErrorKind::BadLength { expected, found } => {
                write!(f, "expected {expected} elements, found {found}")
            }
            ErrorKind::BadPath(message) => write!(f, "bad path: {message}"),
            ErrorKind::UnknownType(name) => write!(f, "no type registered as `{name}`"),
            ErrorKind::WrongType(name) => write!(f, "value is not a {name}"),
        }
    }
}

/// An error from reading or writing a value, with the path to where it
/// happened (empty at the top level).
#[derive(Debug, Clone, PartialEq)]
pub struct ReflectError {
    /// What went wrong.
    pub kind: ErrorKind,
    /// Where, from the outermost value inward.
    pub path: Vec<PathSeg>,
}

impl ReflectError {
    /// An error with no location yet.
    pub fn new(kind: ErrorKind) -> Self {
        Self {
            kind,
            path: Vec::new(),
        }
    }

    /// A [`ErrorKind::TypeMismatch`] for `found`.
    pub fn mismatch(expected: impl Into<String>, found: &Value) -> Self {
        Self::new(ErrorKind::TypeMismatch {
            expected: expected.into(),
            found: found.kind_name(),
        })
    }

    /// The same error one level further out, inside field `name`.
    pub fn at_field(mut self, name: &str) -> Self {
        self.path.insert(0, PathSeg::Field(name.to_string()));
        self
    }

    /// The same error one level further out, inside element `index`.
    pub fn at_index(mut self, index: usize) -> Self {
        self.path.insert(0, PathSeg::Index(index));
        self
    }

    /// The path as text: `a.b[2]`. Empty at the top level.
    pub fn path_string(&self) -> String {
        path_to_string(&self.path)
    }
}

/// Renders path steps the way [`crate::Value::get_path`] reads them.
pub(crate) fn path_to_string(path: &[PathSeg]) -> String {
    let mut out = String::new();
    for seg in path {
        match seg {
            PathSeg::Field(name) => {
                if !out.is_empty() {
                    out.push('.');
                }
                out.push_str(name);
            }
            PathSeg::Index(i) => {
                out.push('[');
                out.push_str(&i.to_string());
                out.push(']');
            }
        }
    }
    out
}

impl fmt::Display for ReflectError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.path.is_empty() {
            write!(f, "{}", self.kind)
        } else {
            write!(f, "at {}: {}", self.path_string(), self.kind)
        }
    }
}

impl std::error::Error for ReflectError {}
