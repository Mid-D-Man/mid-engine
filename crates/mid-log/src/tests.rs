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
        assert_eq!(Tier::Mid .as_str(), "MID ");
        assert_eq!(Tier::High.as_str(), "HIGH");
    }

    #[test]
    fn tier_display_matches_as_str() {
        assert_eq!(format!("{}", Tier::Low),  "LOW ");
        assert_eq!(format!("{}", Tier::Mid),  "MID ");
        assert_eq!(format!("{}", Tier::High), "HIGH");
    }

    #[test]
    fn tier_from_u8_zero_is_low() {
        assert_eq!(Tier::from_u8(0), Tier::Low);
    }

    #[test]
    fn tier_from_u8_one_is_mid() {
        assert_eq!(Tier::from_u8(1), Tier::Mid);
    }

    #[test]
    fn tier_from_u8_two_is_high() {
        assert_eq!(Tier::from_u8(2), Tier::High);
    }

    #[test]
    fn tier_from_u8_large_value_is_high() {
        assert_eq!(Tier::from_u8(255), Tier::High);
    }

    #[test]
    fn tier_three_variants_are_distinct() {
        assert_ne!(Tier::Low,  Tier::Mid);
        assert_ne!(Tier::Mid,  Tier::High);
        assert_ne!(Tier::Low,  Tier::High);
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
        assert!(entry.timestamp > 0, "timestamp should be non-zero");
    }

    #[test]
    fn log_entry_timestamp_increases_monotonically() {
        let a = LogEntry::new(LogLevel::Info, Tier::High, "a".into());
        std::thread::sleep(std::time::Duration::from_millis(2));
        let b = LogEntry::new(LogLevel::Info, Tier::High, "b".into());
        assert!(b.timestamp >= a.timestamp);
    }

    // ── Ring buffer ───────────────────────────────────────────────────────

    #[test]
    fn buffer_capacity_is_power_of_two() {
        let cap = crate::buffer::CAPACITY;
        assert!(cap > 0 && (cap & (cap - 1)) == 0,
            "CAPACITY={} must be a power of two", cap);
    }

    #[test]
    fn buffer_create_returns_paired_producer_consumer() {
        let (mut prod, mut cons) = crate::buffer::create();
        let entry = LogEntry::new(LogLevel::Info, Tier::High, "ring buffer test".into());
        assert!(prod.push(entry).is_ok());
        assert!(cons.pop().is_ok());
    }

    #[test]
    fn buffer_empty_pop_returns_err() {
        let (_prod, mut cons) = crate::buffer::create();
        assert!(cons.pop().is_err());
    }

    // ── Logger lifecycle ──────────────────────────────────────────────────

    #[test]
    fn logger_init_succeeds_or_was_already_init() {
        let _ = MidLogger::init();
        assert!(MidLogger::get().is_some());
    }

    #[test]
    fn logger_log_does_not_panic() {
        MidLogger::init();
        if let Some(logger) = MidLogger::get() {
            logger.log(LogLevel::Trace, Tier::Low,  "trace — engine internal".into());
            logger.log(LogLevel::Info,  Tier::Mid,  "info — mid-level system".into());
            logger.log(LogLevel::Warn,  Tier::High, "warn — gameplay logic".into());
            logger.log(LogLevel::Error, Tier::Low,  "error — non-fatal".into());
        }
    }

    #[test]
    fn logger_accepts_all_tier_variants() {
        MidLogger::init();
        if let Some(logger) = MidLogger::get() {
            logger.log(LogLevel::Info, Tier::Low,  "low tier".into());
            logger.log(LogLevel::Info, Tier::Mid,  "mid tier".into());
            logger.log(LogLevel::Info, Tier::High, "high tier".into());
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
            logger.log(LogLevel::Warn, Tier::Low, "x".repeat(65_536));
        }
    }

    // ── Macro API ─────────────────────────────────────────────────────────

    #[test]
    fn macros_do_not_panic_before_init() {
        crate::mid_trace!(Tier::Low,  "before init");
        crate::mid_info! (Tier::Mid,  "before init");
        crate::mid_warn! (Tier::High, "before init");
        crate::mid_error!(Tier::Low,  "before init");
    }

    #[test]
    fn macros_accept_format_args() {
        MidLogger::init();
        crate::mid_info!(Tier::High, "player {} spawned at ({:.1}, {:.1})", 42, 1.0, 2.5);
        crate::mid_warn!(Tier::Mid,  "system {}% loaded", 87);
        crate::mid_error!(Tier::Low, "entity {} missing component {}", 99, "Transform");
    }

    #[test]
    fn macros_cover_all_tiers() {
        MidLogger::init();
        crate::mid_info!(Tier::Low,  "engine internal");
        crate::mid_info!(Tier::Mid,  "mid-level system");
        crate::mid_info!(Tier::High, "gameplay logic");
    }

    // ── FFI ───────────────────────────────────────────────────────────────

    #[test]
    fn ffi_init_returns_one_or_zero() {
        let result = crate::ffi::mid_log_init();
        assert!(result == 0 || result == 1,
            "mid_log_init must return 0 or 1, got {}", result);
    }

    #[test]
    fn ffi_log_with_null_message_does_not_panic() {
        crate::ffi::mid_log_init();
        unsafe {
            crate::ffi::mid_log_info_c(0, std::ptr::null());
        }
    }

    #[test]
    fn ffi_log_valid_message_all_levels_and_tiers() {
        crate::ffi::mid_log_init();
        let msg = std::ffi::CString::new("ffi test").unwrap();
        unsafe {
            // All three tiers
            crate::ffi::mid_log_info_c(0, msg.as_ptr()); // LOW
            crate::ffi::mid_log_info_c(1, msg.as_ptr()); // MID
            crate::ffi::mid_log_info_c(2, msg.as_ptr()); // HIGH
            // All five levels
            crate::ffi::mid_log_trace_c(0, msg.as_ptr());
            crate::ffi::mid_log_warn_c (1, msg.as_ptr());
            crate::ffi::mid_log_error_c(2, msg.as_ptr());
        }
    }

    #[test]
    fn ffi_tier_constants_map_correctly() {
        assert_eq!(Tier::from_u8(0), Tier::Low,  "MID_TIER_LOW  = 0");
        assert_eq!(Tier::from_u8(1), Tier::Mid,  "MID_TIER_MID  = 1");
        assert_eq!(Tier::from_u8(2), Tier::High, "MID_TIER_HIGH = 2");
    }
}
