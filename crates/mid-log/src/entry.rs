// crates/mid-log/src/entry.rs

//! A single log entry placed into the channel.

use std::borrow::Cow;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::kv::KvPair;
use crate::level::{LogLevel, Tier};

// ── Coarse timestamp ──────────────────────────────────────────────────────────
//
// The IO thread calls tick_coarse_timestamp() once per drain cycle.
// LogEntry::new() reads this AtomicU64 (~2 ns) instead of a vDSO clock
// call (~20–40 ns), saving ~18–38 ns per log entry.
//
// Granularity: at most one drain cycle stale, typically sub-millisecond.
// Acceptable for game engine logging — no one needs sub-ms timestamp
// precision on individual log entries.
pub(crate) static COARSE_TS_MS: AtomicU64 = AtomicU64::new(0);

/// Refresh the coarse timestamp. Called by the IO thread at the top of
/// each drain cycle. One real clock read shared across all logging threads.
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
// Thread::name().to_owned() was ~50–100 ns per log call (malloc + memcpy).
// Arc<str> captured once per thread; subsequent calls pay only an atomic
// refcount increment (~2 ns).
//
// Note on E0716: Thread::current() returns a temporary. We bind it to a
// named local `t` so Thread::name() can borrow from it through Arc::from().
thread_local! {
    static THREAD_NAME: Arc<str> = {
        let t    = std::thread::current();
        let name = t.name().unwrap_or("<unnamed>");
        Arc::from(name)
    };
}

/// A log entry placed in the channel by the calling thread.
///
/// ## Allocation profile per entry
///
/// | Field       | Printf path                     | KV path                         |
/// |-------------|---------------------------------|---------------------------------|
/// | `message`   | `Cow::Owned(format!(...))` — 1× | `Cow::Borrowed("static")` — 0× |
/// | `kvs`       | `Vec::new()` — 0×               | `vec![...]` — 1×                |
/// | `thread`    | `Arc::clone` — 0× (atomic inc)  | `Arc::clone` — 0× (atomic inc)  |
/// | `timestamp` | `AtomicU64::load` — 0× (~2 ns)  | same                            |
///
/// For printf the total is 1 allocation (the formatted String).
/// For KV the total is 1 allocation (the Vec<KvPair>); the message is borrowed.
#[derive(Debug, Clone)]
pub struct LogEntry {
    pub level:  LogLevel,
    pub tier:   Tier,

    /// Log message.
    /// - Printf: `Cow::Owned(format!(...))` — the formatted string.
    /// - KV:     `Cow::Borrowed("static literal")` — zero allocation.
    pub message:   Cow<'static, str>,

    /// Structured key-value pairs. Empty for printf-style entries.
    /// The IO thread formats these after the message on the log line.
    pub kvs:       Vec<KvPair>,

    /// Unix timestamp in milliseconds, sourced from `COARSE_TS_MS`.
    pub timestamp: u64,
    pub file:      &'static str,
    pub line:      u32,
    pub module:    &'static str,
    pub thread:    Arc<str>,
    pub frame:     u64,
}

impl LogEntry {
    /// Construct a **printf-style** entry.
    ///
    /// `message` is the already-formatted `String` from `format!(…)`.
    /// `kvs` should be `Vec::new()` — pass an empty Vec, not `vec![…]`.
    pub fn new(
        level:   LogLevel,
        tier:    Tier,
        message: Cow<'static, str>,
        kvs:     Vec<KvPair>,
        file:    &'static str,
        line:    u32,
        module:  &'static str,
    ) -> Self {
        let timestamp = {
            let coarse = COARSE_TS_MS.load(Ordering::Relaxed);
            if coarse > 0 {
                coarse // fast path: atomic load ~2 ns
            } else {
                // Cold start — IO thread hasn't ticked yet.
                // This path runs at most once per process lifetime.
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map(|d| d.as_millis() as u64)
                    .unwrap_or(0)
            }
        };

        let thread = THREAD_NAME.with(Arc::clone);
        let frame  = crate::frame::current_frame();

        LogEntry { level, tier, message, kvs, timestamp, file, line, module, thread, frame }
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
