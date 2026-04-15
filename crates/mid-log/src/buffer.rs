//! SPSC ring buffer wiring.
//!
//! One producer  — game thread (hot path, zero blocking).
//! One consumer  — IO thread (background drain).

use rtrb::{Producer, Consumer, RingBuffer};
use crate::entry::LogEntry;

/// Capacity in entries. 4096 entries gives plenty of headroom
/// at 128 Hz before the IO thread would ever fall behind.
pub const CAPACITY: usize = 4_096;

/// Create the matched producer/consumer pair.
pub fn create() -> (Producer<LogEntry>, Consumer<LogEntry>) {
    RingBuffer::new(CAPACITY)
}
