// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-reflect.md, section "value.rs"
// ============================================================================
//! [`Value`]: a small, self-describing tree that any reflected type can be
//! turned into and built from, and the path syntax for reaching into one.

use crate::error::{path_to_string, ErrorKind, PathSeg, ReflectError};

/// A self-describing value. A struct is a [`Value::Map`] in field order.
///
/// Integers that fit an `i64` are always [`Value::Int`]; [`Value::UInt`]
/// holds only what is larger, so equal numbers compare equal.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    /// No value (`None`, JSON `null`).
    Null,
    /// A boolean.
    Bool(bool),
    /// An integer that fits an `i64`.
    Int(i64),
    /// An integer larger than `i64::MAX`.
    UInt(u64),
    /// A floating point number. May be non-finite.
    Float(f64),
    /// Text.
    Str(String),
    /// An ordered sequence.
    List(Vec<Value>),
    /// Named entries, in order. Keys are unique in values that were parsed
    /// or produced by a reflected type.
    Map(Vec<(String, Value)>),
}

impl Value {
    /// An unsigned integer in canonical form: [`Value::Int`] when it fits.
    pub fn from_u64(n: u64) -> Value {
        match i64::try_from(n) {
            Ok(i) => Value::Int(i),
            Err(_) => Value::UInt(n),
        }
    }

    /// A map from `(key, value)` pairs, keeping their order.
    pub fn map<K, I>(entries: I) -> Value
    where
        K: Into<String>,
        I: IntoIterator<Item = (K, Value)>,
    {
        Value::Map(entries.into_iter().map(|(k, v)| (k.into(), v)).collect())
    }

    /// The shape of this value, for error messages.
    pub fn kind_name(&self) -> &'static str {
        match self {
            Value::Null => "null",
            Value::Bool(_) => "bool",
            Value::Int(_) | Value::UInt(_) => "integer",
            Value::Float(_) => "float",
            Value::Str(_) => "string",
            Value::List(_) => "list",
            Value::Map(_) => "map",
        }
    }

    /// The boolean, if this is one.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Value::Bool(b) => Some(*b),
            _ => None,
        }
    }

    /// The integer, if this is one that fits an `i64`.
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Value::Int(i) => Some(*i),
            _ => None,
        }
    }

    /// The integer, if this is a non-negative one.
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Value::Int(i) => u64::try_from(*i).ok(),
            Value::UInt(u) => Some(*u),
            _ => None,
        }
    }

    /// The number as a float, if this is an integer or a float.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Value::Int(i) => Some(*i as f64),
            Value::UInt(u) => Some(*u as f64),
            Value::Float(f) => Some(*f),
            _ => None,
        }
    }

    /// The text, if this is a string.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }

    /// The elements, if this is a list.
    pub fn as_list(&self) -> Option<&[Value]> {
        match self {
            Value::List(items) => Some(items),
            _ => None,
        }
    }

    /// The entries, if this is a map.
    pub fn as_map(&self) -> Option<&[(String, Value)]> {
        match self {
            Value::Map(entries) => Some(entries),
            _ => None,
        }
    }

    /// The entry named `key`, if this is a map that has one.
    pub fn get(&self, key: &str) -> Option<&Value> {
        match self {
            Value::Map(entries) => entries.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    /// The value at `path` (`translation.x`, `items[2].name`). An empty path
    /// is this value itself.
    pub fn get_path(&self, path: &str) -> Result<&Value, ReflectError> {
        let segs = parse_path(path)?;
        let mut current = self;
        for (depth, seg) in segs.iter().enumerate() {
            current = step(current, seg).map_err(|e| with_prefix(e, &segs[..depth]))?;
        }
        Ok(current)
    }

    /// Replaces the value at `path`. The path must already exist: a typo in
    /// a field name is an error, not a new field. An empty path replaces
    /// this value.
    pub fn set_path(&mut self, path: &str, new: Value) -> Result<(), ReflectError> {
        let segs = parse_path(path)?;
        let mut current = self;
        for (depth, seg) in segs.iter().enumerate() {
            current = step_mut(current, seg).map_err(|e| with_prefix(e, &segs[..depth]))?;
        }
        *current = new;
        Ok(())
    }
}

impl From<bool> for Value {
    fn from(b: bool) -> Value {
        Value::Bool(b)
    }
}

impl From<i64> for Value {
    fn from(i: i64) -> Value {
        Value::Int(i)
    }
}

impl From<u64> for Value {
    fn from(u: u64) -> Value {
        Value::from_u64(u)
    }
}

impl From<f64> for Value {
    fn from(f: f64) -> Value {
        Value::Float(f)
    }
}

impl From<&str> for Value {
    fn from(s: &str) -> Value {
        Value::Str(s.to_string())
    }
}

impl From<String> for Value {
    fn from(s: String) -> Value {
        Value::Str(s)
    }
}

fn bad_path(message: String, at: &[PathSeg]) -> ReflectError {
    let mut error = ReflectError::new(ErrorKind::BadPath(message));
    error.path = at.to_vec();
    error
}

/// Puts the steps already walked in front of an error found one step later.
fn with_prefix(mut error: ReflectError, walked: &[PathSeg]) -> ReflectError {
    let mut path = walked.to_vec();
    path.append(&mut error.path);
    error.path = path;
    error
}

fn step<'a>(value: &'a Value, seg: &PathSeg) -> Result<&'a Value, ReflectError> {
    match (value, seg) {
        (Value::Map(entries), PathSeg::Field(name)) => entries
            .iter()
            .find(|(k, _)| k == name)
            .map(|(_, v)| v)
            .ok_or_else(|| bad_path(format!("no field `{name}`"), &[])),
        (Value::List(items), PathSeg::Index(i)) => items.get(*i).ok_or_else(|| {
            bad_path(
                format!("index {i} is out of range for {} elements", items.len()),
                &[],
            )
        }),
        (other, PathSeg::Field(name)) => Err(bad_path(
            format!("cannot read field `{name}` of a {}", other.kind_name()),
            &[],
        )),
        (other, PathSeg::Index(i)) => Err(bad_path(
            format!("cannot read index {i} of a {}", other.kind_name()),
            &[],
        )),
    }
}

fn step_mut<'a>(value: &'a mut Value, seg: &PathSeg) -> Result<&'a mut Value, ReflectError> {
    match (value, seg) {
        (Value::Map(entries), PathSeg::Field(name)) => entries
            .iter_mut()
            .find(|(k, _)| k == name)
            .map(|(_, v)| v)
            .ok_or_else(|| bad_path(format!("no field `{name}`"), &[])),
        (Value::List(items), PathSeg::Index(i)) => {
            let len = items.len();
            items.get_mut(*i).ok_or_else(|| {
                bad_path(format!("index {i} is out of range for {len} elements"), &[])
            })
        }
        (other, PathSeg::Field(name)) => Err(bad_path(
            format!("cannot read field `{name}` of a {}", other.kind_name()),
            &[],
        )),
        (other, PathSeg::Index(i)) => Err(bad_path(
            format!("cannot read index {i} of a {}", other.kind_name()),
            &[],
        )),
    }
}

/// Parses `a.b[2].c` into steps. A name is any run of characters other than
/// `.`, `[` and `]`; an index is digits in brackets. The empty string is the
/// empty path.
pub(crate) fn parse_path(path: &str) -> Result<Vec<PathSeg>, ReflectError> {
    let fail = |message: &str| {
        Err(ReflectError::new(ErrorKind::BadPath(format!(
            "{message} in `{path}`"
        ))))
    };
    let mut segs = Vec::new();
    let mut rest = path;
    let mut first = true;
    while !rest.is_empty() {
        if let Some(after) = rest.strip_prefix('[') {
            let Some(close) = after.find(']') else {
                return fail("unclosed `[`");
            };
            let digits = &after[..close];
            if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
                return fail("an index must be digits");
            }
            let Ok(index) = digits.parse::<usize>() else {
                return fail("index too large");
            };
            segs.push(PathSeg::Index(index));
            rest = &after[close + 1..];
        } else {
            if let Some(after) = rest.strip_prefix('.') {
                if first {
                    return fail("a path cannot start with `.`");
                }
                rest = after;
            } else if !first {
                return fail("expected `.` or `[`");
            }
            let end = rest.find(['.', '[', ']']).unwrap_or(rest.len());
            if end == 0 {
                return fail("empty field name");
            }
            segs.push(PathSeg::Field(rest[..end].to_string()));
            rest = &rest[end..];
            if rest.starts_with(']') {
                return fail("unexpected `]`");
            }
        }
        first = false;
    }
    Ok(segs)
}

/// Renders steps as path text; the inverse of parsing for valid paths.
pub fn path_string(path: &[PathSeg]) -> String {
    path_to_string(path)
}
