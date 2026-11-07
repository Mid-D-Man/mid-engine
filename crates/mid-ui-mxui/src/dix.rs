// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-ui-mxui.md, section "dix.rs"
// ============================================================================
//! The DixScript v1.0.0 adapter (feature `dixscript`).

use dixscript::Runtime::{DixLoadOptions, DixLoader, DixValue};
use mid_ui::Real;

use crate::error::{MxError, MxErrorKind};
use crate::load::{build, Loaded};
use crate::value::MxValue;

/// Converts a loaded DixScript value into the source-neutral tree. Object
/// fields come out of a hash map in arbitrary order, so they are sorted by
/// name to keep errors and results deterministic. Dates, timestamps, blobs
/// and regexes become strings; the UI schema has no use for them, so they
/// fail as a type error where a field is used.
pub fn from_dix(value: &DixValue) -> MxValue {
    match value {
        DixValue::Null => MxValue::Null,
        DixValue::Bool(b) => MxValue::Bool(*b),
        DixValue::Int(i) => MxValue::Int(i64::from(*i)),
        DixValue::Long(l) => MxValue::Int(*l),
        DixValue::Float(f) => MxValue::Float(f64::from(*f)),
        DixValue::Double(d) => MxValue::Float(*d),
        DixValue::String(s)
        | DixValue::Date(s)
        | DixValue::Timestamp(s)
        | DixValue::Blob(s)
        | DixValue::Regex(s) => MxValue::Str(s.clone()),
        DixValue::HexColor(s) => MxValue::Color(s.clone()),
        DixValue::Enum { enum_name, field_name, .. } => {
            MxValue::Enum { type_name: enum_name.clone(), field: field_name.clone() }
        }
        DixValue::Array(items) | DixValue::Tuple(items) => {
            MxValue::Array(items.iter().map(from_dix).collect())
        }
        DixValue::Object(map) => {
            let mut entries: Vec<(String, MxValue)> =
                map.iter().map(|(k, v)| (k.clone(), from_dix(v))).collect();
            entries.sort_by(|a, b| a.0.cmp(&b.0));
            MxValue::Object(entries)
        }
    }
}

/// Parses `.mxui` source text and builds its tree from the `root` entry.
///
/// `root` may be a group array holding one node (`root::` followed by a single
/// item) or a single object. Imports in the source resolve through the file
/// system, which does not exist on wasm, so keep each file self-contained
/// there.
pub fn load_str<R: Real>(source: &str) -> Result<Loaded<R>, MxError> {
    let loader = DixLoader::new_silent();
    let data = loader
        .load_from_str(source, &DixLoadOptions::new())
        .map_err(|message| MxError::new("", MxErrorKind::Source(message)))?;
    let root = data
        .get_value("root")
        .ok_or_else(|| MxError::new("root", MxErrorKind::MissingRoot))?;
    let node = match root {
        DixValue::Array(items) if items.len() == 1 => from_dix(&items[0]),
        DixValue::Object(_) => from_dix(root),
        other => {
            let found = match other {
                DixValue::Array(items) => format!("an array of {}", items.len()),
                _ => from_dix(other).type_name().to_string(),
            };
            return Err(MxError::new("root", MxErrorKind::BadRoot(found)));
        }
    };
    build(&node)
}
