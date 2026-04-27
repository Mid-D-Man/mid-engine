// crates/mid-math/src/ffi/mod.rs
//! FFI module — C-compatible types and exported functions.
//!
//! C callers include this via a generated header (cbindgen or hand-maintained).
//! The header exposes only CVec2/3/4, CQuat, CMat3, CMat4 and the
//! mid_* functions below.

pub mod types;
pub mod exports;

pub use types::{CVec2, CVec3, CVec4, CQuat, CMat3, CMat4};
