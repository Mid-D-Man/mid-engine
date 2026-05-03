// crates/mid-log/src/entry.rs

//! A single log entry placed into the ring buffer.

use std::time::{SystemTime, UNIX_EPOCH};
use crate::level::{LogLevel, Tier};

/// A log entry. Placed into the crossbeam channel by the calling thread.
/// The IO thread drains and formats these.
///
/// String is heap-allocated — allocation happens on the producer side
/// (game thread) after the level filter check, so filtered entries never
/// allocate. The IO thread is never the bottleneck.
#[derive(Debug, Clone)]
pub struct LogEntry {
    pub level:     LogLevel,
    pub tier:      Tier,
    pub message:   String,
    /// Unix timestamp in milliseconds.
    pub timestamp: u64,
    /// Source file path — from `file!()`, zero-cost `&'static str`.
    pub file:      &'static str,
    /// Source line number — from `line!()`.
    pub line:      u32,
    /// Module path — from `module_path!()`, zero-cost `&'static str`.
    pub module:    &'static str,
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

        LogEntry { level, tier, message, timestamp, file, line, module }
    }

    /// Format the timestamp portion as `HH:MM:SS.mmm` (UTC).
    ///
    /// Avoids pulling in `chrono` — computed from raw Unix ms.
    /// Good enough for log output; not a calendar library.
    pub fn format_time(&self) -> String {
        let total_secs = self.timestamp / 1_000;
        let ms         = self.timestamp % 1_000;
        let s          = total_secs % 60;
        let m          = (total_secs / 60) % 60;
        let h          = (total_secs / 3_600) % 24;
        format!("{:02}:{:02}:{:02}.{:03}", h, m, s, ms)
    }
}
