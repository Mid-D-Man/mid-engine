// crates/mid-log/src/lib.rs

//! mid-log — Non-blocking, tiered logger for Mid Engine.
//!
//! ## APIs
//!
//! ### Printf (ergonomic, ~250–500 ns for float-heavy messages)
//! ```rust,no_run
//! use mid_log::{mid_info, level::Tier};
//! mid_info!(Tier::High, "entity {} at ({:.2}, {:.2})", id, x, y);
//! ```
//!
//! ### Structured KV (~45–65 ns for the same data — no format!())
//! ```rust,no_run
//! use mid_log::{mid_kvinfo, level::Tier};
//! mid_kvinfo!(Tier::High, "entity update"; "id" => id, "x" => x, "y" => y);
//! ```

pub mod level;
pub mod entry;
pub mod filter;
pub mod buffer;
pub mod color;
pub mod format;
pub mod frame;
pub mod writer;
pub mod logger;
pub mod macros;
pub mod assert;
pub mod ratelimit;
pub mod console_buffer;
pub mod ffi;
pub mod kv;

#[cfg(test)]
mod tests;
