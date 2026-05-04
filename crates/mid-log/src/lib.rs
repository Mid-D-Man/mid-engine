// crates/mid-log/src/lib.rs

//! mid-log — Non-blocking, tiered logger for Mid Engine.
//!
//! ## Quick start
//! ```rust,no_run
//! use mid_log::{mid_info, mid_warn, level::Tier};
//! use mid_log::logger::{MidLogger, InitConfig};
//! use mid_log::level::LogLevel;
//! use mid_log::format::FormatConfig;
//!
//! MidLogger::init_full(InitConfig {
//!     min_level: LogLevel::Info,
//!     format: FormatConfig { show_frame: true, ..Default::default() },
//!     ..Default::default()
//! });
//!
//! mid_log::frame::set_frame(0);
//! mid_info!(Tier::High, "Engine started");
//! mid_warn!(Tier::Low, "Something looks off: {}", 42);
//! ```
//!
//! ## Inline coloring
//! ```rust,no_run
//! use mid_log::{mid_warn, level::Tier, color::{Color, paint}};
//! let hp = 5u32;
//! mid_warn!(Tier::High, "HP: {} ({})", paint(hp, Color::Red), paint("critical", Color::Bold));
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

#[cfg(test)]
mod tests;
