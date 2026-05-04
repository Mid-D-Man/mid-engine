// crates/mid-log/src/entry.rs

//! A single log entry placed into the channel.

use std::time::{SystemTime, UNIX_EPOCH};
use crate::level::{LogLevel, Tier};

/// A log entry. Produced on the calling thread, consumed by the IO thread.
///
/// All fields except `message` are zero-copy (`&'static str`, `u32`, `u64`).
/// `message` is heap-allocated after the level filter check, so filtered
/// entries never allocate. `thread` is the only field that may allocate
/// on the calling thread — and only once per unique thread name.
#[derive(Debug, Clone)]
pub struct LogEntry {
    pub level:     LogLevel,
    pub tier:      Tier,
    pub message:   String,
    /// Unix timestamp in milliseconds.
    pub timestamp: u64,
    /// Source file path — from `file!()`, `&'static str`, zero-cost.
    pub file:      &'static str,
    /// Source line number — from `line!()`.
    pub line:      u32,
    /// Rust module path — from `module_path!()`, `&'static str`, zero-cost.
    pub module:    &'static str,
    /// Name of the thread that produced this entry.
    /// Captured via `std::thread::current().name()` at construction time.
    /// `<unnamed>` for threads without an explicit name.
    pub thread:    String,
    /// Game frame counter at the time of logging.
    /// Zero when `set_frame()` has never been called.
    pub frame:     u64,
}

impl LogEntry {
    pub fn new(
        level:   LogLevel,
        tier:    Tier,
        message: String,
        file:    &'static str,
        line:    u32,
        module:  &'static str,
    ) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        // Thread name: captured once per entry. For hot-path threads
        // (e.g. the physics thread logging every frame), the thread name
        // is fetched from the OS handle — a single atomic read on most
        // platforms since pthread stores the name in the thread struct.
        let thread = std::thread::current()
            .name()
            .unwrap_or("<unnamed>")
            .to_owned();

        let frame = crate::frame::current_frame();

        LogEntry { level, tier, message, timestamp, file, line, module, thread, frame }
    }

    /// Format the timestamp as `HH:MM:SS.mmm` (UTC, no date).
    ///
    /// Computed from raw Unix milliseconds — no external crate required.
    pub fn format_time(&self) -> String {
        let total_secs = self.timestamp / 1_000;
        let ms         = self.timestamp % 1_000;
        let s          = total_secs % 60;
        let m          = (total_secs / 60) % 60;
        let h          = (total_secs / 3_600) % 24;
        format!("{:02}:{:02}:{:02}.{:03}", h, m, s, ms)
    }
}
