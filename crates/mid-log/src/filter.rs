// crates/mid-log/src/filter.rs

//! Runtime log-level filter.
//!
//! A single `AtomicU8` stores the minimum level. The macros check
//! this *before* calling `format!()`, so filtered-out log calls
//! never allocate a String. This makes mid-log competitive with
//! `tracing`'s lazy-formatting approach on the disabled-level path.
//!
//! ## Hot path cost (disabled level)
//! One `AtomicU8::load(Relaxed)` + one comparison branch.
//! On x86_64 this is a single `movzx` + `cmp` + `jl` = ~1 cycle.
//! Equivalent to tracing's callsite check.
//!
//! ## Thread safety
//! `Relaxed` ordering is correct here. We don't need the filter change
//! to be immediately visible to all threads — a few extra log entries
//! between the set and the next load are acceptable. Using `SeqCst`
//! would add a memory fence for no practical benefit.

use std::sync::atomic::{AtomicU8, Ordering};
use crate::level::LogLevel;

/// Global minimum log level. Default: Trace (log everything).
static MIN_LEVEL: AtomicU8 = AtomicU8::new(LogLevel::Trace as u8);

/// Returns `true` if `level` passes the current filter.
///
/// Called from every log macro before `format!()`. Keep this `#[inline(always)]`
/// so the branch is visible to the optimizer at the call site — it can then
/// eliminate dead format strings entirely in release builds.
#[inline(always)]
pub fn is_enabled(level: LogLevel) -> bool {
    (level as u8) >= MIN_LEVEL.load(Ordering::Relaxed)
}

/// Set the minimum log level globally.
///
/// Log entries below this level are silently discarded *before* any
/// string formatting occurs. Call this once at startup:
///
/// ```rust
/// # use mid_log::filter::set_min_level;
/// # use mid_log::level::LogLevel;
/// set_min_level(LogLevel::Info); // silence TRACE in production
/// ```
pub fn set_min_level(level: LogLevel) {
    MIN_LEVEL.store(level as u8, Ordering::Relaxed);
}

/// Returns the current minimum log level.
pub fn get_min_level() -> LogLevel {
    LogLevel::from_u8(MIN_LEVEL.load(Ordering::Relaxed))
  }
