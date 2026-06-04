// crates/mid-common/src/lib.rs
//! mid-common — shared types, traits, and utilities for the Mid Engine workspace.
//!
//! All engine crates depend on this. Keep it lean:
//! no rendering, no platform IO, no heavy algorithms.
//!
//! # Modules
//! - `error`  — engine error types
//! - `traits` — shared interfaces (Update, Fixed, etc.)
//! - `types`  — primitive shared types (EntityId, Tick, etc.)
//! - `string` — string utilities: StrRef, NulStr, FixedStr, StringSearch, uniquename

#![no_std]
extern crate alloc;

pub mod error;
pub mod traits;
pub mod types;
pub mod string;

// ── Flat re-exports — most-used types available at crate root ─────────────────

pub use error::MidError;
pub use string::{
    StrRef,
    NulStr,
    FixedStr,
    StringSearch,
    SearchItem,
    damerau_levenshtein_distance,
    fuzzy_match_score,
    uniquename,
    flip_side_name,
    SideChar,
};

// Macro re-exports (defined with #[macro_export], so they're already at crate root,
// but document them here for discoverability)
//   nul_str!(b"...\0")  — create a NulStr from a byte literal
