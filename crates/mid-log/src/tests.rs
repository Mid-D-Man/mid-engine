//! Unit tests for mid-log.
//!
//! These are the tests that the CI workflow runs under `--mid-log`.
//! They verify the Rust face of the logger — init, log, shutdown,
//! level ordering, tier display, and ring buffer behaviour.

#[cfg(test)]
mod tests {
    use crate::level::{LogLevel, Tier};
    use crate::entry::LogEntry;
    use crate::logger::MidLogger;

    // ── Level ─────────────────────────────────────────────────────────────

    #[test]
    fn level_ordering_is_correct() {
        assert!(LogLevel::Trace < LogLevel::Info);
        assert!(LogLevel::Info  < LogLevel::Warn);
        assert!(LogLevel::Warn  < LogLevel::Error);
        assert!(LogLevel::Error < LogLevel::Fatal);
    }

    #[test]
    fn level_as_str_returns_fixed_width_labels() {
        // All labels are 5 chars so log columns align cleanly.
        assert_eq!(LogLevel::Trace.as_str(), "TRACE");
        assert_eq!(LogLevel::Info .as_str(), "INFO ");
        assert_eq!(LogLevel::Warn .as_str(), "WARN ");
        assert_eq!(LogLevel::Error.as_str(), "ERROR");
        assert_eq!(LogLevel::Fatal.as_str(), "FATAL");
    }

    #[test]
    fn level_display_matches_as_str() {
        assert_eq!(format!("{}", LogLevel::Info),  "INFO ");
        assert_eq!(format!("{}", LogLevel::Error), "ERROR");
    }

    // ── Tier ──────────────────────────────────────────────────────────────

    #[test]
    fn tier_as_str_fixed_width() {
        assert_eq!(Tier::Low .as_str(), "LOW ");
        assert_eq!(Tier::High.as_str(), "HIGH");
    }

    #[test]
    fn tier_from_u8_zero_is_low() {
        assert_eq!(Tier::from_u8(0), Tier::Low);
    }

    #[test]
    fn tier_from_u8_nonzero_is_high() {
        assert_eq!(Tier::from_u8(1),   Tier::High);
        assert_eq!(Tier::from_u8(255), Tier::High);
    }

    #[test]
    fn tier_display_matches_as_str() {
        assert_eq!(format!("{}", Tier::Low),  "LOW ");
        assert_eq!(format!("{}", Tier::High), "HIGH");
    }

    // ── LogEntry ──────────────────────────────────────────────────────────

    #[test]
    fn log_entry_stores_fields_correctly() {
        let entry = LogEntry::new(
            LogLevel::Warn,
            Tier::Low,
            "buffer near capacity".to_string(),
        );
        assert_eq!(entry.level,   LogLevel::Warn);
        assert_eq!(entry.tier,    Tier::Low);
        assert_eq!(entry.message, "buffer near capacity");
        assert!(entry.timestamp > 0, "timestamp should be set");
    }

    #[test]
    fn log_entry_timestamp_increases_monotonically() {
        let a = LogEntry::new(LogLevel::Info, Tier::High, "a".into());
        // Sleep briefly to ensure timestamp advances.
        std::thread::sleep(std::time::Duration::from_millis(2));
        let b = LogEntry::new(LogLevel::Info, Tier::High, "b".into());
        assert!(b.timestamp >= a.timestamp, "timestamp should be non-decreasing");
    }

    // ── Ring buffer ───────────────────────────────────────────────────────

    #[test]
    fn buffer_capacity_is_power_of_two() {
        // rtrb requires power-of-two capacity internally.
        // Verify our chosen constant is a power of two.
        let cap = crate::buffer::CAPACITY;
        assert!(cap > 0 && (cap & (cap - 1)) == 0,
            "CAPACITY={} must be a power of two", cap);
    }

    #[test]
    fn buffer_create_returns_paired_producer_consumer() {
        let (mut prod, mut cons) = crate::buffer::create();
        let entry = LogEntry::new(LogLevel::Info, Tier::High, "ring buffer test".into());
        assert!(prod.push(entry).is_ok(), "push into empty buffer should succeed");
        assert!(cons.pop().is_ok(),        "pop from non-empty buffer should succeed");
    }

    #[test]
    fn buffer_empty_pop_returns_err() {
        let (_prod, mut cons) = crate::buffer::create();
        assert!(cons.pop().is_err(), "pop from empty buffer should return Err");
    }

    // ── Logger lifecycle ──────────────────────────────────────────────────

    #[test]
    fn logger_init_succeeds_on_first_call() {
        // NOTE: OnceLock means only one test process can init successfully.
        // If this test runs in a process that already inited, the return
        // value is false — but the logger is still usable. We accept both.
        let _ = MidLogger::init();
        // After either outcome, get() must return Some.
        assert!(MidLogger::get().is_some(), "logger must be accessible after init");
    }

    #[test]
    fn logger_get_returns_none_or_some() {
        // In a fresh process this is None until init; in our test binary
        // another test may have already called init. Either is valid.
        let result = MidLogger::get();
        // Just assert we don't panic — the type is Option<&'static MidLogger>.
        let _ = result;
    }

    #[test]
    fn logger_log_does_not_panic() {
        MidLogger::init();
        if let Some(logger) = MidLogger::get() {
            // These must complete without panicking even under test concurrency.
            logger.log(LogLevel::Trace, Tier::Low,  "trace from test".into());
            logger.log(LogLevel::Info,  Tier::High, "info from test".into());
            logger.log(LogLevel::Warn,  Tier::Low,  "warn from test".into());
            logger.log(LogLevel::Error, Tier::High, "error from test".into());
        }
    }

    #[test]
    fn logger_accepts_empty_message() {
        MidLogger::init();
        if let Some(logger) = MidLogger::get() {
            logger.log(LogLevel::Info, Tier::Low, String::new());
        }
    }

    #[test]
    fn logger_accepts_unicode_message() {
        MidLogger::init();
        if let Some(logger) = MidLogger::get() {
            logger.log(LogLevel::Info, Tier::High, "🦀 Rust + 🎮 Mid Engine".into());
        }
    }

    #[test]
    fn logger_handles_very_long_message() {
        MidLogger::init();
        if let Some(logger) = MidLogger::get() {
            let long_msg = "x".repeat(65_536);
            logger.log(LogLevel::Warn, Tier::Low, long_msg);
        }
    }

    // ── Macro API ─────────────────────────────────────────────────────────

    #[test]
    fn macros_do_not_panic_before_init() {
        // Macros silently no-op if logger is not yet initialised.
        // This is intentional — game code should not crash if the logger
        // is called before MidLogger::init().
        crate::mid_trace!(Tier::Low,  "trace before init");
        crate::mid_info! (Tier::High, "info before init");
        crate::mid_warn! (Tier::Low,  "warn before init");
        crate::mid_error!(Tier::High, "error before init");
    }

    #[test]
    fn macros_accept_format_args() {
        MidLogger::init();
        crate::mid_info!(Tier::High, "player {} spawned at ({:.1}, {:.1})", 42, 1.0, 2.5);
        crate::mid_warn!(Tier::Low,  "buffer {}% full", 87);
    }

    // ── FFI helpers ───────────────────────────────────────────────────────

    #[test]
    fn ffi_init_returns_one_or_zero() {
        // First call returns 1, subsequent calls return 0.
        // The logger state from other tests in this binary means we might
        // see 0 here — that is correct behaviour, not a failure.
        let result = crate::ffi::mid_log_init();
        assert!(result == 0 || result == 1);
    }

    #[test]
    fn ffi_log_with_null_message_does_not_panic() {
        crate::ffi::mid_log_init();
        // Safety: we are intentionally passing null to test the null guard.
        unsafe {
            crate::ffi::mid_log_info_c(0, std::ptr::null());
        }
    }

    #[test]
    fn ffi_log_valid_message_does_not_panic() {
        crate::ffi::mid_log_init();
        let msg = std::ffi::CString::new("ffi test message").unwrap();
        unsafe {
            crate::ffi::mid_log_trace_c(0, msg.as_ptr());
            crate::ffi::mid_log_info_c (1, msg.as_ptr());
            crate::ffi::mid_log_warn_c (0, msg.as_ptr());
            crate::ffi::mid_log_error_c(1, msg.as_ptr());
        }
    }

    #[test]
    fn ffi_tier_zero_maps_to_low() {
        assert_eq!(Tier::from_u8(0), Tier::Low);
    }

    #[test]
    fn ffi_tier_one_maps_to_high() {
        assert_eq!(Tier::from_u8(1), Tier::High);
    }
  }
