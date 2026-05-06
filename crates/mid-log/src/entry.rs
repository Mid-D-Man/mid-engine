// crates/mid-log/src/entry.rs

//! A single log entry placed into the channel.

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use crate::level::{LogLevel, Tier};

// ── Thread-name cache ─────────────────────────────────────────────────────────
//
// Previously every LogEntry::new() called:
//   std::thread::current().name().unwrap_or("<unnamed>").to_owned()
//
// That is a heap allocation (malloc + memcpy) on every single log call — the
// dominant non-format!() cost on the hot path (~50–100 ns per call).
//
// Solution: capture the name once per thread into an Arc<str>. Subsequent log
// calls clone the Arc, which is a single atomic fetch_add(1, Relaxed) — ~2 ns.
//
// The Arc is never dropped as long as the thread is alive; when the thread
// exits the TLS destructor drops it, decrementing the refcount.
thread_local! {
    static THREAD_NAME: Arc<str> = {
        let name = std::thread::current()
            .name()
            .unwrap_or("<unnamed>");
        Arc::from(name)
    };
}

/// A log entry. Produced on the calling thread, consumed by the IO thread.
///
/// All fields except `message` are zero-copy or near-zero-copy:
/// - `file`, `module`: `&'static str`
/// - `line`, `frame`: scalar integers
/// - `timestamp`: one vDSO call
/// - `thread`: `Arc<str>` — atomic refcount clone after the first call on each thread
/// - `message`: allocated by `format!()` only after the level-filter check passes
#[derive(Debug, Clone)]
pub struct LogEntry {
    pub level:     LogLevel,
    pub tier:      Tier,
    pub message:   String,
    /// Unix timestamp in milliseconds.
    pub timestamp: u64,
    /// Source file path — `file!()`, `&'static str`, zero-cost.
    pub file:      &'static str,
    /// Source line number — `line!()`.
    pub line:      u32,
    /// Rust module path — `module_path!()`, `&'static str`, zero-cost.
    pub module:    &'static str,
    /// Name of the thread that produced this entry.
    ///
    /// Shared via `Arc<str>`. The first log call on a given thread allocates the
    /// name once; every subsequent call on that thread pays only an atomic
    /// refcount increment (~2 ns).
    pub thread:    Arc<str>,
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

        // Arc::clone = fetch_add(1, Relaxed) ≈ 2 ns.
        // Replaces the previous to_owned() path (~50–100 ns malloc).
        let thread = THREAD_NAME.with(Arc::clone);

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
