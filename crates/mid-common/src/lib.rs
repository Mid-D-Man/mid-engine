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

// `pub use error::MidError;` intentionally NOT here yet -- error.rs is
// still an empty auto-generated stub (see its own doc comment), no
// `MidError` type exists to re-export. Confirmed via grep across the
// whole workspace: nothing depends on `MidError` except this one broken
// line, same situation as `mid-ecs::World` was. Add it back once
// error.rs actually defines it.

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
