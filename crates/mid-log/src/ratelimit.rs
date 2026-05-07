// crates/mid-log/src/ratelimit.rs

//! Rate limiting — suppresses duplicate log entries from hot paths.

use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant, SystemTime};

use crate::entry::LogEntry;
use crate::level::{LogLevel, Tier};

// ── Configuration ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct RateLimitConfig {
    pub enabled:        bool,
    pub window:         Duration,
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

pub fn set_rate_limit_config(config: RateLimitConfig) {
    if let Ok(mut g) = get_config_store().lock() {
        *g = config;
    }
}

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
    first_message:    String,
    count:            u32,
    suppressed_count: u32,
    window_start:     Instant,
}

// ── RateLimiter ───────────────────────────────────────────────────────────────

pub(crate) struct RateLimiter {
    sites: HashMap<SiteKey, SiteState>,
}

pub(crate) enum RateDecision {
    Allow,
    Suppress,
    WindowExpired { summary: LogEntry },
}

impl RateLimiter {
    pub(crate) fn new() -> Self {
        Self { sites: HashMap::new() }
    }

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
                let summary_entry = if state.suppressed_count > 0 {
                    Some(Self::make_summary(entry, state))
                } else {
                    None
                };
                *state = SiteState {
                    first_message:    entry.message.as_ref().to_owned(),
                    count:            1,
                    suppressed_count: 0,
                    window_start:     now,
                };
                return match summary_entry {
                    Some(s) => RateDecision::WindowExpired { summary: s },
                    None    => RateDecision::Allow,
                };
            }

            state.count += 1;
            if state.count > config.max_per_window {
                state.suppressed_count += 1;
                return RateDecision::Suppress;
            }
            RateDecision::Allow
        } else {
            self.sites.insert(key, SiteState {
                first_message:    entry.message.as_ref().to_owned(),
                count:            1,
                suppressed_count: 0,
                window_start:     now,
            });
            RateDecision::Allow
        }
    }

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
                false
            } else {
                true
            }
        });
        summaries
    }

    fn make_summary(entry: &LogEntry, state: &SiteState) -> LogEntry {
        LogEntry {
            level:     entry.level,
            tier:      entry.tier,
            message:   Cow::Owned(format!(
                "... previous message repeated {} more time(s) (suppressed)",
                state.suppressed_count,
            )),
            kvs:       Vec::new(),
            timestamp: crate::entry::LogEntry::new(
                LogLevel::Trace, Tier::Low,
                Cow::Borrowed(""),
                Vec::new(), "", 0, "",
            ).timestamp,
            file:      entry.file,
            line:      entry.line,
            module:    entry.module,
            thread:    Arc::clone(&entry.thread),
            frame:     entry.frame,
        }
    }

    fn make_summary_static(state: &SiteState) -> LogEntry {
        LogEntry {
            level:   LogLevel::Warn,
            tier:    Tier::Low,
            message: Cow::Owned(format!(
                "... a previous log site was suppressed {} time(s) in the last window",
                state.suppressed_count,
            )),
            kvs:     Vec::new(),
            timestamp: SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0),
            file:   "<rate-limiter>",
            line:   0,
            module: "<rate-limiter>",
            thread: Arc::from("<io-thread>"),
            frame:  crate::frame::current_frame(),
        }
    }
}
