// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-ui-mxui.md, section "lib.rs"
// ============================================================================
//! mid-ui-mxui — loads `.mxui` UI files into a [`mid_ui::Tree`].
//!
//! A `.mxui` file is DixScript v1.0.0 whose `root` entry describes the widget
//! tree. Loading is two steps. The schema reader ([`build`]) works on a
//! source-neutral [`MxValue`] tree and has no dependencies beyond `mid-ui`.
//! With the `dixscript` feature (on by default), [`load_str`] parses source
//! text with the published `dixscript` crate and feeds the schema reader. The
//! feature is the only place that depends on DixScript, so moving to a newer
//! DixScript touches one module.

mod bind;
mod error;
mod load;
mod value;

#[cfg(feature = "dixscript")]
mod dix;

pub use bind::BindKey;
pub use error::{MxError, MxErrorKind};
pub use load::{build, Bind, EnumRef, Loaded, NodeMeta};
pub use value::MxValue;

#[cfg(feature = "dixscript")]
pub use dix::{from_dix, load_str};
