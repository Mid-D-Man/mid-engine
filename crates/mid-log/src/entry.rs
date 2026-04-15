//! A single log entry placed into the ring buffer.

use std::time::{SystemTime, UNIX_EPOCH};
use crate::level::{LogLevel, Tier};

/// A log entry. Sized to fit cleanly in the ring buffer.
/// String is heap-allocated — acceptable because allocation
/// happens on the producer side (game thread), not on the
/// IO thread doing the actual write.
#[derive(Debug, Clone)]
pub struct LogEntry {
    pub level:      LogLevel,
    pub tier:       Tier,
    pub message:    String,
    /// Unix timestamp in milliseconds.
    pub timestamp:  u64,
}

impl LogEntry {
    pub fn new(level: LogLevel, tier: Tier, message: String) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        LogEntry { level, tier, message, timestamp }
    }
}
