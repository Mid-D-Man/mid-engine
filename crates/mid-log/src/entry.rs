// crates/mid-log/src/entry.rs

//! A single log entry placed into the channel.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use crate::level::{LogLevel, Tier};

// ── Coarse timestamp ──────────────────────────────────────────────────────────
//
// SystemTime::now() costs ~20–40 ns per call (vDSO on Linux, still a context-
// switch equivalent on macOS). At 128 Hz with thousands of entities logging,
// this is measurable.
//
// Fix: the IO thread calls tick_coarse_timestamp() once per drain cycle,
// writing the real clock into a global AtomicU64. LogEntry::new() reads this
// AtomicU64 (~2 ns) instead of calling the clock directly.
//
// Granularity trade-off: the timestamp is at most one drain-cycle stale
// (~0–2 ms under typical load). For a game engine logger this is completely
// acceptable — nobody debugging frame-level issues needs sub-millisecond
// timestamp precision on log entries.
//
// Cold start: the AtomicU64 starts at 0. LogEntry::new() falls back to
// SystemTime::now() exactly once, on the very first log before the IO thread
// has run. After that the fast path is always taken.
pub(crate) static COARSE_TS_MS: AtomicU64 = AtomicU64::new(0);

/// Refresh the coarse timestamp. Called by the IO thread at the top of each
/// drain cycle. One real clock call shared across all concurrent logging threads.
#[inline]
pub(crate) fn tick_coarse_timestamp() {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0);
    COARSE_TS_MS.store(now, Ordering::Relaxed);
}

// ── Thread-name cache ─────────────────────────────────────────────────────────
//
// Previously every LogEntry::new() called:
//   std::thread::current().name().unwrap_or("<unnamed>").to_owned()
//
// That is a heap allocation (malloc + memcpy) on every single log call.
//
// Fix: capture the name once per thread into an Arc<str>. Subsequent log
// calls clone the Arc = one atomic fetch_add(1, Relaxed) = ~2 ns.
//
// Note on E0716: Thread::name() borrows from the Thread value.
// Binding `thread` as a named local keeps the Thread alive through Arc::from.
thread_local! {
    static THREAD_NAME: Arc<str> = {
        let thread = std::thread::current();
        let name   = thread.name().unwrap_or("<unnamed>");
        Arc::from(name)
    };
}

/// A log entry. Produced on the calling thread, consumed by the IO thread.
///
/// Hot-path cost after optimisation:
/// - `file`, `module`: `&'static str` zero-cost
/// - `line`, `frame`:  scalar copy
/// - `timestamp`:      AtomicU64 load (~2 ns, vs ~20–40 ns vDSO)
/// - `thread`:         Arc::clone atomic refcount (~2 ns, vs ~50–100 ns malloc)
/// - `message`:        allocated by `format!()` only after level-filter passes
#[derive(Debug, Clone)]
pub struct LogEntry {
    pub level:     LogLevel,
    pub tier:      Tier,
    pub message:   String,
    /// Unix timestamp in milliseconds — sourced from `COARSE_TS_MS`.
    pub timestamp: u64,
    pub file:      &'static str,
    pub line:      u32,
    pub module:    &'static str,
    /// Thread name shared via `Arc<str>`.
    /// First log call on a thread allocates once; every subsequent call
    /// is an atomic refcount increment (~2 ns).
    pub thread:    Arc<str>,
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
        // Prefer the coarse timestamp written by the IO thread (~2 ns).
        // Fall back to the real clock only on the very first entry (COARSE == 0).
        let timestamp = {
            let coarse = COARSE_TS_MS.load(Ordering::Relaxed);
            if coarse > 0 {
                coarse
            } else {
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map(|d| d.as_millis() as u64)
                    .unwrap_or(0)
            }
        };

        // Arc::clone = fetch_add(1, Relaxed) ≈ 2 ns.
        let thread = THREAD_NAME.with(Arc::clone);

        let frame = crate::frame::current_frame();

        LogEntry { level, tier, message, timestamp, file, line, module, thread, frame }
    }

    /// Format the timestamp as `HH:MM:SS.mmm` (UTC, no date).
    pub fn format_time(&self) -> String {
        let total_secs = self.timestamp / 1_000;
        let ms         = self.timestamp % 1_000;
        let s          = total_secs % 60;
        let m          = (total_secs / 60) % 60;
        let h          = (total_secs / 3_600) % 24;
        format!("{:02}:{:02}:{:02}.{:03}", h, m, s, ms)
    }
            }
