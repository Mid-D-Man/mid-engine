// crates/mid-common/src/ffi/mod.rs
//! C-ABI boundary layer for mid-common.
//!
//! Follows the same convention as mid-math/src/ffi:
//! C structs are defined here, Rust types live in their own modules,
//! and #[no_mangle] functions bridge them.

pub mod string;

pub use string::{
    CFixedStr32,
    CFixedStr64,
    CFixedStr256,
    CSearchResult,
    MidStringSearch,
};
