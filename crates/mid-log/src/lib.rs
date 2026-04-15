// crates/mid-log/src/lib.rs

//! mid-log — Non-blocking, tiered logger for Mid Engine.
//!
//! ## Rust face (game thread / engine code)
//! ```rust
//! use mid_log::{mid_info, mid_warn, level::Tier};
//!
//! mid_log::logger::MidLogger::init();
//! # let (id, x, y) = (1u32, 1.0_f32, 2.0_f32);
//! mid_info!(Tier::High, "Player {} spawned at ({}, {})", id, x, y);
//! mid_warn!(Tier::Low,  "Buffer overflow prevented in UDP stream");
//! ```
//!
//! ## C face (Unity / Unreal / Godot / any C host)
//! Include `headers/mid_log.h` and link against `libmid_log.dylib`.
//! ```c
//! mid_log_init();
//! mid_log_info_c(MID_TIER_HIGH, "C host: logger initialized");
//! mid_log_shutdown();
//! ```

pub mod level;
pub mod entry;
pub mod buffer;
pub mod writer;
pub mod logger;
pub mod macros;
pub mod ffi;
#[cfg(test)]
mod tests;
