// crates/mid-log/src/ratelimit.rs

//! Rate limiting — suppresses duplicate log entries from hot paths.
//!
//! Without rate limiting, a broken system that logs every frame buries
//! everything else. This module detects repeated `(file, line)` pairs
//! within a configurable time window and suppresses duplicates, emitting
//! a "repeated N times" summary when the window expires.
//!
//! ## Design
//!
//! Rate limiting runs entirely on the IO thread — no synchronization with
//! the calling thread. The IO thread owns a `RateLimiter` in its local
//! state and calls `check()` before formatting each entry.
//!
//! The `RateLimiter` maintains a `HashMap<SiteKey, SiteState>` keyed on
//! `(file, line)`. Each site tracks:
//! - The first message seen in the current window.
//! - The count of suppressed duplicates.
//! - The timestamp of the window start.
//!
//! When a site's window expires, the suppression summary is emitted and
//! the site resets.
//!
//! ## Configuration
//!
//! ```rust,no_run
//! use mid_log::ratelimit::{set_rate_limit_config, RateLimitConfig};
//! use std::time::Duration;
//!
//! set_rate_limit_config(RateLimitConfig {
//!     enabled:        true,
//!     window:         Duration::from_secs(1),
//!     max_per_window: 5,   // allow up to 5 identical logs per second
//! });
//! ```
//!
//! Disable entirely:
//! ```rust,no_run
//! use mid_log::ratelimit::{set_rate_limit_config, RateLimitConfig};
//! set_rate_limit_config(RateLimitConfig { enabled: false, ..Default::default() });
//! ```

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use crate::entry::LogEntry;
use crate::level::{LogLevel, Tier};

// ── Configuration ─────────────────────────────────────────────────────────────

/// Rate limit configuration.
#[derive(Debug, Clone)]
pub struct RateLimitConfig {
    /// Whether rate limiting is active. Default: `true`.
    pub enabled: bool,
    /// How long a suppression window lasts. Default: 1 second.
    pub window: Duration,
    /// How many entries from the same `(file, line)` are allowed before
    /// suppression begins. Default: 5.
    pub max_per_window: u32,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            enabled:        true,
            window:         Duration::from_secs(1),
            max_per_window: 5,
        }
    }
}

static RATE_LIMIT_CONFIG: OnceLock<Mutex<RateLimitConfig>> = OnceLock::new();

fn get_config_store() -> &'static Mutex<RateLimitConfig> {
    RATE_LIMIT_CONFIG.get_or_init(|| Mutex::new(RateLimitConfig::default()))
}

/// Replace the rate limit configuration.
///
/// The IO thread picks up the change on its next `check()` call.
pub fn set_rate_limit_config(config: RateLimitConfig) {
    if let Ok(mut g) = get_config_store().lock() {
        *g = config;
    }
}

/// Returns a clone of the current rate limit configuration.
pub fn get_rate_limit_config() -> RateLimitConfig {
    get_config_store().lock().ok()
        .map(|g| g.clone())
        .unwrap_or_default()
}

// ── Per-site state ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct SiteKey {
    file: &'static str,
    line: u32,
}

struct SiteState {
    /// The first message in the current window (kept for the summary).
    first_message:    String,
    /// Number of times this site fired in the current window (including #1).
    count:            u32,
    /// Number of entries that were fully suppressed (count > max_per_window).
    suppressed_count: u32,
    /// When this window started.
    window_start:     Instant,
}

// ── RateLimiter — IO thread local ─────────────────────────────────────────────

/// State owned by the IO thread. Not `Send` — lives entirely on the IO thread
/// and is therefore safe to use without locking.
pub(crate) struct RateLimiter {
    sites: HashMap<SiteKey, SiteState>,
}

/// Decision returned by `RateLimiter::check()`.
pub(crate) enum RateDecision {
    /// Emit this entry normally.
    Allow,
    /// Suppress this entry — it is a duplicate within the window.
    Suppress,
    /// This entry is a duplicate, but the window just expired.
    /// Emit a summary of the previous window and then emit this entry.
    WindowExpired {
        summary: LogEntry,
    },
}

impl RateLimiter {
    pub(crate) fn new() -> Self {
        Self { sites: HashMap::new() }
    }

    /// Check whether `entry` should be emitted, suppressed, or preceded
    /// by a suppression summary.
    ///
    /// Must be called on the IO thread only.
    pub(crate) fn check(
        &mut self,
        entry:  &LogEntry,
        config: &RateLimitConfig,
    ) -> RateDecision {
        if !config.enabled {
            return RateDecision::Allow;
        }

        let key = SiteKey { file: entry.file, line: entry.line };
        let now = Instant::now();

        if let Some(state) = self.sites.get_mut(&key) {
            let window_age = now.duration_since(state.window_start);

            if window_age >= config.window {
                // Window expired — emit summary then reset.
                let summary_entry = if state.suppressed_count > 0 {
                    Some(Self::make_summary(entry, state))
                } else {
                    None
                };
                // Reset the site for the new window.
                *state = SiteState {
                    first_message:    entry.message.clone(),
                    count:            1,
                    suppressed_count: 0,
                    window_start:     now,
                };
                return match summary_entry {
                    Some(s) => RateDecision::WindowExpired { summary: s },
                    None    => RateDecision::Allow,
                };
            }

            // Within the same window.
            state.count += 1;
            if state.count > config.max_per_window {
                state.suppressed_count += 1;
                return RateDecision::Suppress;
            }
            RateDecision::Allow
        } else {
            // First time we've seen this site.
            self.sites.insert(key, SiteState {
                first_message:    entry.message.clone(),
                count:            1,
                suppressed_count: 0,
                window_start:     now,
            });
            RateDecision::Allow
        }
    }

    /// Flush any active suppression windows that have expired since the
    /// last call to `check()`. Called periodically by the IO thread
    /// (e.g. on every drain cycle) to ensure summaries are emitted even
    /// when a site goes quiet after a burst.
    pub(crate) fn flush_expired(
        &mut self,
        config: &RateLimitConfig,
    ) -> Vec<LogEntry> {
        if !config.enabled {
            return Vec::new();
        }

        let now = Instant::now();
        let mut summaries = Vec::new();

        self.sites.retain(|_key, state| {
            if now.duration_since(state.window_start) >= config.window
                && state.suppressed_count > 0
            {
                summaries.push(Self::make_summary_static(state));
                false // remove the site — window is done
            } else {
                true  // keep — window still active
            }
        });

        summaries
    }

    fn make_summary(entry: &LogEntry, state: &SiteState) -> LogEntry {
        LogEntry {
            level:     entry.level,
            tier:      entry.tier,
            message:   format!(
                "... previous message repeated {} more time(s) (suppressed)",
                state.suppressed_count,
            ),
            timestamp: crate::entry::LogEntry::new(
                LogLevel::Trace, Tier::Low, String::new(), "", 0, "",
            ).timestamp,
            file:      entry.file,
            line:      entry.line,
            module:    entry.module,
            thread:    entry.thread.clone(),
            frame:     entry.frame,
        }
    }

    fn make_summary_static(state: &SiteState) -> LogEntry {
        // We don't have the original entry here — use a generic summary.
        LogEntry {
            level:   LogLevel::Warn,
            tier:    Tier::Low,
            message: format!(
                "... a previous log site was suppressed {} time(s) in the last window",
                state.suppressed_count,
            ),
            timestamp: SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0),
            file:   "<rate-limiter>",
            line:   0,
            module: "<rate-limiter>",
            thread: String::from("<io-thread>"),
            frame:  crate::frame::current_frame(),
        }
    }
}

use std::time::SystemTime;
