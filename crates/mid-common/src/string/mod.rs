// crates/mid-common/src/string/mod.rs
//! String utilities for Mid Engine.
//!
//! Inspired by Blender's blenlib string module (BLI_string_ref.hh,
//! BLI_string_search.hh, BLI_string_utils.hh).
//!
//! Modules:
//!   str_ref   — StrRef<'a> (non-owning slice) + NulStr<'a> (FFI, null-terminated)
//!   fixed_str — FixedStr<N> (stack string, null-terminated, FFI safe)
//!   search    — Fuzzy string search (Damerau-Levenshtein)
//!   utils     — uniquename, flip_side_name, and other name-manipulation tools

pub mod str_ref;
pub mod fixed_str;
pub mod search;
pub mod utils;

pub use str_ref::{StrRef, NulStr};
pub use fixed_str::FixedStr;
pub use search::{StringSearch, SearchItem, damerau_levenshtein_distance, fuzzy_match_score};
pub use utils::{uniquename, flip_side_name, SideChar};
