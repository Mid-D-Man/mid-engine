//! Log levels and tier metadata.

/// Log level — ordered least to most severe.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum LogLevel {
    /// Per-frame firehose. Enable only in transient debug sessions.
    Trace = 0,
    Info  = 1,
    Warn  = 2,
    /// Non-fatal. Stack trace behind the 'backtrace' feature flag.
    Error = 3,
    /// Immediate memory dump + shutdown.
    Fatal = 4,
}

/// Which engine tier produced this log entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier {
    /// Engine internals — [LOW]
    Low,
    /// Gameplay logic — [HIGH]
    High,
}

impl std::fmt::Display for Tier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Tier::Low  => write!(f, "LOW"),
            Tier::High => write!(f, "HIGH"),
        }
    }
}
