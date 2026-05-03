// crates/mid-log/src/buffer.rs

//! MPSC channel wiring.
//!
//! Switched from rtrb (SPSC) to crossbeam-channel (MPSC) so multiple
//! Rust threads can log simultaneously without a Mutex on the producer.
//! crossbeam-channel's unbounded sender is lock-free on the fast path
//! (no contention) and degrades gracefully under concurrent load.
//!
//! Capacity discipline:
//!   - Unbounded sender: never blocks, never drops. Memory grows if the
//!     IO thread falls behind. Acceptable for a game engine logger where
//!     log bursts are short-lived.
//!   - If you need bounded/drop behaviour (embedded targets, strict memory
//!     budgets), switch to crossbeam_channel::bounded(4096) here.

use crossbeam_channel::{unbounded, Receiver, Sender};
use crate::entry::LogEntry;

pub type LogSender   = Sender<LogEntry>;
pub type LogReceiver = Receiver<LogEntry>;

/// Create a matched sender/receiver pair.
pub fn create() -> (LogSender, LogReceiver) {
    unbounded()
}
