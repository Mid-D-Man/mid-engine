// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-ui-mxui.md, section "error.rs"
// ============================================================================
//! Load errors, each carrying the path of the offending value.

use core::fmt;

/// What went wrong while reading a `.mxui` value.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum MxErrorKind {
    /// The source text failed to parse or compile; carries the parser's message.
    Source(String),
    /// The file has no `root` entry.
    MissingRoot,
    /// `root` is neither an object nor an array holding exactly one object.
    BadRoot(String),
    /// A value had the wrong type for its field.
    WrongType {
        /// What the field accepts.
        expected: &'static str,
        /// What was found.
        found: &'static str,
    },
    /// An enum value belongs to a different enum than the field requires.
    WrongEnum {
        /// The enum type the field requires.
        expected: &'static str,
        /// The enum type that was found.
        found: String,
    },
    /// An enum value names a variant the enum does not have.
    UnknownVariant {
        /// The enum type being read.
        enum_name: &'static str,
        /// The variant that was found.
        found: String,
    },
    /// An enum field carries an integer but the file declares no enum with
    /// the name the field needs, so the integer cannot be decoded.
    EnumNotDeclared(&'static str),
    /// A node has a field the schema does not define.
    UnknownField(String),
    /// A hex color string is malformed.
    BadColor(String),
    /// An inset array has a length other than 2 or 4.
    BadInsets(usize),
    /// A sizing object is not `{ fill = weight }`.
    BadSizing,
    /// Two nodes use the same `id`.
    DuplicateName(String),
    /// Two different binding names hash to the same key.
    BindCollision {
        /// The name seen first.
        first: String,
        /// The name that collided with it.
        second: String,
    },
    /// The tree has more nodes than a node id can address.
    TreeFull,
}

/// A load error with the path of the value that caused it, for example
/// `root.children[1].margin`.
#[derive(Debug, Clone, PartialEq)]
pub struct MxError {
    /// Path of the offending value; empty when the error is not tied to one.
    pub path: String,
    /// What went wrong.
    pub kind: MxErrorKind,
}

impl MxError {
    pub(crate) fn new(path: impl Into<String>, kind: MxErrorKind) -> Self {
        Self { path: path.into(), kind }
    }

    // An error on `field` of the node being read. The path is relative to the
    // node; `prefixed` makes it absolute once the node's position is known.
    pub(crate) fn field(field: &str, kind: MxErrorKind) -> Self {
        Self { path: field.to_string(), kind }
    }

    pub(crate) fn prefixed(mut self, node_path: &str) -> Self {
        self.path = if self.path.is_empty() {
            node_path.to_string()
        } else {
            format!("{node_path}.{}", self.path)
        };
        self
    }
}

impl fmt::Display for MxError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if !self.path.is_empty() {
            write!(f, "{}: ", self.path)?;
        }
        match &self.kind {
            MxErrorKind::Source(m) => write!(f, "source error: {m}"),
            MxErrorKind::MissingRoot => write!(f, "no `root` entry"),
            MxErrorKind::BadRoot(found) => {
                write!(f, "`root` must be one node, found {found}")
            }
            MxErrorKind::WrongType { expected, found } => {
                write!(f, "expected {expected}, found {found}")
            }
            MxErrorKind::WrongEnum { expected, found } => {
                write!(f, "expected a {expected} value, found {found}")
            }
            MxErrorKind::UnknownVariant { enum_name, found } => {
                write!(f, "{enum_name} has no variant `{found}`")
            }
            MxErrorKind::EnumNotDeclared(name) => {
                write!(f, "the file declares no enum `{name}`, so an integer here cannot be decoded")
            }
            MxErrorKind::UnknownField(name) => write!(f, "unknown field `{name}`"),
            MxErrorKind::BadColor(s) => write!(f, "malformed color `{s}`"),
            MxErrorKind::BadInsets(n) => {
                write!(f, "insets take a number or an array of 2 or 4, found {n} elements")
            }
            MxErrorKind::BadSizing => write!(f, "a size is a number or `{{ fill = weight }}`"),
            MxErrorKind::DuplicateName(n) => write!(f, "duplicate node id `{n}`"),
            MxErrorKind::BindCollision { first, second } => {
                write!(f, "binding names `{first}` and `{second}` hash to the same key")
            }
            MxErrorKind::TreeFull => write!(f, "too many nodes"),
        }
    }
}

impl std::error::Error for MxError {}
