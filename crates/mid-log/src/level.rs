//! Log levels and tier metadata.

/// Log level — ordered least to most severe.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum LogLevel {
    /// Per-frame firehose. Transient debugging only.
    Trace = 0,
    Info  = 1,
    Warn  = 2,
    /// Non-fatal. Stack trace behind the `backtrace` feature flag.
    Error = 3,
    /// Triggers immediate flush + shutdown.
    Fatal = 4,
}

impl LogLevel {
    pub fn as_str(self) -> &'static str {
        match self {
            LogLevel::Trace => "TRACE",
            LogLevel::Info  => "INFO ",
            LogLevel::Warn  => "WARN ",
            LogLevel::Error => "ERROR",
            LogLevel::Fatal => "FATAL",
        }
    }
}

impl std::fmt::Display for LogLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Engine tier — every log entry is tagged so you know exactly where it came from.
///
/// | Tier | C constant      | Description                        |
/// |------|-----------------|------------------------------------|
/// | Low  | MID_TIER_LOW  0 | Engine internals — physics, net, ECS |
/// | Mid  | MID_TIER_MID  1 | Engine-adjacent — scripting, tools |
/// | High | MID_TIER_HIGH 2 | Gameplay logic — player, AI, events|
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier {
    /// Engine internals. Printed as [LOW ].
    Low,
    /// Engine-adjacent systems. Printed as [MID ].
    Mid,
    /// Gameplay logic. Printed as [HIGH].
    High,
}

impl Tier {
    pub fn as_str(self) -> &'static str {
        match self {
            Tier::Low  => "LOW ",
            Tier::Mid  => "MID ",
            Tier::High => "HIGH",
        }
    }

    /// Convert from the C ABI tier constant.
    /// 0 = Low, 1 = Mid, 2+ = High.
    pub fn from_u8(v: u8) -> Self {
        match v {
            0 => Tier::Low,
            1 => Tier::Mid,
            _ => Tier::High,
        }
    }
}

impl std::fmt::Display for Tier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}
