// crates/mid-log/src/format.rs

//! Runtime format configuration — controls which fields appear in each log line.
//!
//! Each flag is an individual `AtomicBool` so the IO thread can read the full
//! configuration with no locking — just five `Relaxed` loads per entry.
//!
//! The `FormatSnapshot` bundles all five into a single struct that the
//! `format_entry()` function receives, avoiding repeated atomic loads inside
//! the format loop.
//!
//! ## Log line structure (all fields enabled, colors on)
//!
//! ```text
//! HH:MM:SS.mmm [LEVEL][TIER] [T:thread_name] [F:12345] message body  (module  file:line)
//! ```
//!
//! ## Defaults
//!
//! Timestamp and source location are on by default. Thread and frame are off —
//! enable them when debugging multi-threaded or timing-sensitive issues.

use std::sync::atomic::{AtomicBool, Ordering};

// ── Global format flags ───────────────────────────────────────────────────────

static SHOW_TIMESTAMP:  AtomicBool = AtomicBool::new(true);
static SHOW_SOURCE_LOC: AtomicBool = AtomicBool::new(true);
static SHOW_MODULE:     AtomicBool = AtomicBool::new(false);
static SHOW_THREAD:     AtomicBool = AtomicBool::new(false);
static SHOW_FRAME:      AtomicBool = AtomicBool::new(false);

// ── Public API ────────────────────────────────────────────────────────────────

/// Configures which fields appear in each formatted log line.
///
/// Apply via [`set_format()`]. All fields can also be toggled individually
/// via the `set_show_*` functions.
///
/// # Example
/// ```rust,no_run
/// use mid_log::format::{FormatConfig, set_format};
///
/// // Editor / debug build: show everything.
/// set_format(&FormatConfig {
///     show_timestamp:  true,
///     show_source_loc: true,
///     show_module:     true,
///     show_thread:     true,
///     show_frame:      true,
/// });
///
/// // Shipping build: minimal output.
/// set_format(&FormatConfig {
///     show_timestamp:  true,
///     show_source_loc: false,
///     show_module:     false,
///     show_thread:     false,
///     show_frame:      false,
/// });
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct FormatConfig {
    /// Show `HH:MM:SS.mmm` before each entry.
    pub show_timestamp:  bool,
    /// Show `(file:line)` at the end of each entry.
    pub show_source_loc: bool,
    /// Show the Rust module path alongside source location.
    pub show_module:     bool,
    /// Show `[T:thread_name]` in each entry. Useful for multi-threaded debugging.
    pub show_thread:     bool,
    /// Show `[F:n]` in each entry. Updated via `set_frame()`.
    pub show_frame:      bool,
}

impl Default for FormatConfig {
    fn default() -> Self {
        Self {
            show_timestamp:  true,
            show_source_loc: true,
            show_module:     false,
            show_thread:     false,
            show_frame:      false,
        }
    }
}

/// Apply a complete format configuration atomically.
///
/// Changes take effect on the IO thread's next log entry.
pub fn set_format(config: &FormatConfig) {
    SHOW_TIMESTAMP .store(config.show_timestamp,  Ordering::Relaxed);
    SHOW_SOURCE_LOC.store(config.show_source_loc, Ordering::Relaxed);
    SHOW_MODULE    .store(config.show_module,      Ordering::Relaxed);
    SHOW_THREAD    .store(config.show_thread,      Ordering::Relaxed);
    SHOW_FRAME     .store(config.show_frame,       Ordering::Relaxed);
}

/// Returns the current format configuration.
pub fn get_format() -> FormatConfig {
    FormatConfig {
        show_timestamp:  SHOW_TIMESTAMP .load(Ordering::Relaxed),
        show_source_loc: SHOW_SOURCE_LOC.load(Ordering::Relaxed),
        show_module:     SHOW_MODULE    .load(Ordering::Relaxed),
        show_thread:     SHOW_THREAD    .load(Ordering::Relaxed),
        show_frame:      SHOW_FRAME     .load(Ordering::Relaxed),
    }
}

// ── Individual toggles ────────────────────────────────────────────────────────

/// Show or hide the `HH:MM:SS.mmm` timestamp prefix.
pub fn set_show_timestamp(v: bool)  { SHOW_TIMESTAMP .store(v, Ordering::Relaxed); }
/// Show or hide the `file:line` source location suffix.
pub fn set_show_source_loc(v: bool) { SHOW_SOURCE_LOC.store(v, Ordering::Relaxed); }
/// Show or hide the Rust module path in the source location.
pub fn set_show_module(v: bool)     { SHOW_MODULE    .store(v, Ordering::Relaxed); }
/// Show or hide the `[T:thread_name]` badge.
pub fn set_show_thread(v: bool)     { SHOW_THREAD    .store(v, Ordering::Relaxed); }
/// Show or hide the `[F:n]` frame counter badge.
pub fn set_show_frame(v: bool)      { SHOW_FRAME     .store(v, Ordering::Relaxed); }

// ── IO thread snapshot ────────────────────────────────────────────────────────

/// All five format flags bundled for use by the IO thread.
///
/// Taken once per `format_entry()` call to avoid five separate atomic loads
/// scattered through the format logic.
#[derive(Debug, Clone, Copy)]
pub(crate) struct FormatSnapshot {
    pub show_timestamp:  bool,
    pub show_source_loc: bool,
    pub show_module:     bool,
    pub show_thread:     bool,
    pub show_frame:      bool,
}

impl FormatSnapshot {
    /// Read all format flags in one go.
    pub(crate) fn take() -> Self {
        Self {
            show_timestamp:  SHOW_TIMESTAMP .load(Ordering::Relaxed),
            show_source_loc: SHOW_SOURCE_LOC.load(Ordering::Relaxed),
            show_module:     SHOW_MODULE    .load(Ordering::Relaxed),
            show_thread:     SHOW_THREAD    .load(Ordering::Relaxed),
            show_frame:      SHOW_FRAME     .load(Ordering::Relaxed),
        }
    }
}
