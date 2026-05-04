// crates/mid-log/src/console_buffer.rs

//! In-game console ring buffer — a second consumer of log entries.
//!
//! The IO thread pushes a copy of every entry (after level filtering) into
//! a fixed-size circular buffer. The game's render thread (or ImGui overlay,
//! or in-game terminal) calls `drain_recent()` or `snapshot()` to retrieve
//! them for display.
//!
//! ## Design
//!
//! - Capacity is set at `init_console_buffer(capacity)` and is fixed thereafter.
//! - When the buffer is full, the oldest entry is silently overwritten (ring).
//! - `drain_recent()` returns entries added since the last drain call.
//!   Uses a generation cursor so multiple consumers can each track their
//!   own read position — just clone the `ConsoleReader` they receive.
//! - `snapshot()` returns all currently buffered entries in order, oldest first.
//! - This is entirely separate from the IO thread's stderr/file sink.
//!   The IO thread writes to both in the same pass.
//!
//! ## Thread safety
//!
//! A single `Mutex<ConsoleBufferInner>` guards the ring. The IO thread holds
//! the lock only for the duration of one `push()` call (~microsecond).
//! The render thread holds it only for the duration of `drain_recent()` or
//! `snapshot()`. Contention is negligible at game frame rates.
//!
//! ## Usage
//! ```rust,no_run
//! use mid_log::console_buffer::{init_console_buffer, ConsoleReader};
//!
//! // At init — before MidLogger::init():
//! init_console_buffer(512);
//!
//! // On the render thread — create one reader per consumer:
//! let mut reader = ConsoleReader::new();
//!
//! // Each frame — get new entries since last call:
//! for entry in reader.drain_recent() {
//!     // render entry in ImGui / in-game terminal
//!     println!("[{}] {}", entry.level.as_str(), entry.message);
//! }
//! ```

use std::sync::{Mutex, OnceLock};
use crate::entry::LogEntry;

// ── Inner ring ────────────────────────────────────────────────────────────────

struct ConsoleBufferInner {
    /// Fixed-capacity ring storage.
    ring:       Vec<Option<LogEntry>>,
    /// Next write position (wraps modulo `capacity`).
    write_head: usize,
    /// Total entries ever written — monotonically increasing.
    /// Used by `ConsoleReader` to compute how many new entries exist.
    total:      u64,
    capacity:   usize,
}

impl ConsoleBufferInner {
    fn new(capacity: usize) -> Self {
        Self {
            ring:       (0..capacity).map(|_| None).collect(),
            write_head: 0,
            total:      0,
            capacity,
        }
    }

    fn push(&mut self, entry: LogEntry) {
        self.ring[self.write_head] = Some(entry);
        self.write_head = (self.write_head + 1) % self.capacity;
        self.total += 1;
    }

    /// Return up to `count` most-recent entries in chronological order.
    fn recent(&self, count: usize) -> Vec<LogEntry> {
        let count  = count.min(self.capacity).min(self.total as usize);
        let total  = self.total as usize;
        let mut out = Vec::with_capacity(count);

        // Oldest of the `count` entries is at index:
        // (write_head - count + capacity) % capacity
        for i in 0..count {
            let idx = (self.write_head + self.capacity - count + i) % self.capacity;
            if let Some(ref e) = self.ring[idx] {
                out.push(e.clone());
            }
        }
        let _ = total; // suppress unused warning
        out
    }

    /// Return all currently buffered entries, oldest first.
    fn snapshot(&self) -> Vec<LogEntry> {
        let count = (self.total as usize).min(self.capacity);
        self.recent(count)
    }
}

// ── Global buffer ─────────────────────────────────────────────────────────────

static CONSOLE_BUFFER: OnceLock<Mutex<ConsoleBufferInner>> = OnceLock::new();

/// Initialize the in-game console buffer with the given capacity.
///
/// Must be called **before** `MidLogger::init*()`. Calling it after init
/// is safe but entries produced before this call will not be buffered.
///
/// `capacity`: number of entries retained. When full, oldest are overwritten.
/// Typical values: 256 (minimal), 512 (default), 2048 (verbose debugging).
///
/// Calling this more than once is a no-op — capacity is set once.
pub fn init_console_buffer(capacity: usize) {
    let capacity = capacity.max(8); // minimum sane value
    let _ = CONSOLE_BUFFER.set(Mutex::new(ConsoleBufferInner::new(capacity)));
}

/// Push an entry into the console buffer. Called by the IO thread.
///
/// No-op if `init_console_buffer()` was never called.
pub(crate) fn push(entry: &LogEntry) {
    if let Some(buf) = CONSOLE_BUFFER.get() {
        if let Ok(mut guard) = buf.lock() {
            guard.push(entry.clone());
        }
    }
}

/// Returns `true` if the console buffer has been initialized.
pub fn is_initialized() -> bool {
    CONSOLE_BUFFER.get().is_some()
}

/// Returns all currently buffered entries, oldest first.
///
/// Does not advance any read cursor — calling this repeatedly returns
/// the same entries (plus any new ones). Use `ConsoleReader::drain_recent()`
/// if you want incremental reads.
pub fn snapshot() -> Vec<LogEntry> {
    CONSOLE_BUFFER
        .get()
        .and_then(|b| b.lock().ok())
        .map(|g| g.snapshot())
        .unwrap_or_default()
}

// ── ConsoleReader — per-consumer incremental reader ───────────────────────────

/// A stateful reader that tracks how many entries it has already seen.
///
/// Create one per consumer (e.g. one for the ImGui overlay, one for an
/// in-game terminal). Each `ConsoleReader` independently tracks its own
/// read position — they do not interfere with each other.
///
/// # Example
/// ```rust,no_run
/// use mid_log::console_buffer::ConsoleReader;
///
/// let mut reader = ConsoleReader::new();
///
/// // Game loop:
/// loop {
///     for entry in reader.drain_recent() {
///         // Display in overlay
///         println!("{}", entry.message);
///     }
/// }
/// ```
pub struct ConsoleReader {
    /// Total entries seen by this reader. Compared against `buffer.total`
    /// to compute how many new entries exist.
    last_seen: u64,
}

impl ConsoleReader {
    /// Create a new reader starting from the current buffer head.
    ///
    /// Entries produced before this call are not returned by `drain_recent()`.
    /// Use `snapshot()` if you need existing entries.
    pub fn new() -> Self {
        let last_seen = CONSOLE_BUFFER
            .get()
            .and_then(|b| b.lock().ok())
            .map(|g| g.total)
            .unwrap_or(0);
        Self { last_seen }
    }

    /// Return entries produced since the last call to `drain_recent()`.
    ///
    /// Entries are returned in chronological order. If more new entries
    /// exist than the buffer's capacity, only the most recent `capacity`
    /// entries are returned (oldest were overwritten).
    pub fn drain_recent(&mut self) -> Vec<LogEntry> {
        let guard = match CONSOLE_BUFFER.get().and_then(|b| b.lock().ok()) {
            Some(g) => g,
            None    => return Vec::new(),
        };

        let new_count = (guard.total - self.last_seen) as usize;
        if new_count == 0 {
            return Vec::new();
        }

        self.last_seen = guard.total;
        guard.recent(new_count)
    }

    /// Reset this reader to the current buffer head.
    ///
    /// After calling this, `drain_recent()` will only return entries
    /// produced after the reset point.
    pub fn reset(&mut self) {
        self.last_seen = CONSOLE_BUFFER
            .get()
            .and_then(|b| b.lock().ok())
            .map(|g| g.total)
            .unwrap_or(0);
    }
}

impl Default for ConsoleReader {
    fn default() -> Self { Self::new() }
  }
