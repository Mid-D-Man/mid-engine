// crates/mid-log/src/frame.rs

//! Game frame counter — set once per tick, read by every log entry.
//!
//! The counter is a single `AtomicU64`. The game loop calls `set_frame(n)`
//! at the top of each tick. Every `LogEntry` reads it at construction time
//! with `Relaxed` ordering — one atomic load, zero contention.
//!
//! This is the single most useful debugging field a game logger can have.
//! When you have 10,000 log entries from a crash, knowing which frame each
//! came from reduces a 20-minute investigation to a 2-minute one.
//!
//! ## Usage
//! ```rust,no_run
//! use mid_log::frame::set_frame;
//! use mid_log::format::{set_format, FormatConfig};
//!
//! // Enable frame numbers in output:
//! set_format(&FormatConfig { show_frame: true, ..Default::default() });
//!
//! // Call once per game tick:
//! let mut frame_n = 0u64;
//! loop {
//!     set_frame(frame_n);
//!     // ... game tick ...
//!     frame_n += 1;
//! }
//! ```

use std::sync::atomic::{AtomicU64, Ordering};

static FRAME: AtomicU64 = AtomicU64::new(0);

/// Set the current frame number. Call once at the top of each game tick.
///
/// Thread-safe. The value is visible to all threads on their next
/// `LogEntry::new()` call.
#[inline]
pub fn set_frame(n: u64) {
    FRAME.store(n, Ordering::Relaxed);
}

/// Returns the current frame number.
#[inline]
pub fn current_frame() -> u64 {
    FRAME.load(Ordering::Relaxed)
}
