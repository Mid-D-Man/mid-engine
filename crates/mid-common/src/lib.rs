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
//! - `string` — StrRef, NulStr, FixedStr, StringSearch, uniquename, flip_side_name
//! - `ffi`    — C-ABI exports (CFixedStr32/64/256, MidStringSearch, utility fns)

#![no_std]
extern crate alloc;

pub mod error;
pub mod traits;
pub mod types;
pub mod string;
pub mod ffi;

// ── Flat re-exports ────────────────────────────────────────────────────────────

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

pub use ffi::{
    CFixedStr32,
    CFixedStr64,
    CFixedStr256,
    CSearchResult,
    MidStringSearch,
};
