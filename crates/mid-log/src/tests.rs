// crates/mid-log/src/tests.rs

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
        println!(
            "  Trace({}) < Info({}) < Warn({}) < Error({}) < Fatal({})",
            LogLevel::Trace as u8, LogLevel::Info as u8,
            LogLevel::Warn  as u8, LogLevel::Error as u8,
            LogLevel::Fatal as u8,
        );
    }

    #[test]
    fn level_as_str_returns_fixed_width_labels() {
        let cases = [
            (LogLevel::Trace, "TRACE"),
            (LogLevel::Info,  "INFO "),
            (LogLevel::Warn,  "WARN "),
            (LogLevel::Error, "ERROR"),
            (LogLevel::Fatal, "FATAL"),
        ];
        for (level, expected) in cases {
            assert_eq!(level.as_str(), expected);
            println!("  {:?}.as_str() = {:?}  (len={})", level, level.as_str(), level.as_str().len());
        }
    }

    #[test]
    fn level_display_matches_as_str() {
        for level in [LogLevel::Info, LogLevel::Error] {
            let displayed = format!("{}", level);
            assert_eq!(displayed, level.as_str());
            println!("  Display({:?}) = {:?}", level, displayed);
        }
    }

    // ── Tier ──────────────────────────────────────────────────────────────

    #[test]
    fn tier_as_str_fixed_width() {
        let cases = [(Tier::Low, "LOW "), (Tier::Mid, "MID "), (Tier::High, "HIGH")];
        for (tier, expected) in cases {
            assert_eq!(tier.as_str(), expected);
            println!("  {:?}.as_str() = {:?}  (len={})", tier, tier.as_str(), tier.as_str().len());
        }
    }

    #[test]
    fn tier_display_matches_as_str() {
        for tier in [Tier::Low, Tier::Mid, Tier::High] {
            let displayed = format!("{}", tier);
            assert_eq!(displayed, tier.as_str());
            println!("  Display({:?}) = {:?}", tier, displayed);
        }
    }

    #[test]
    fn tier_from_u8_zero_is_low() {
        let t = Tier::from_u8(0);
        assert_eq!(t, Tier::Low);
        println!("  from_u8(0) = {:?}", t);
    }

    #[test]
    fn tier_from_u8_one_is_mid() {
        let t = Tier::from_u8(1);
        assert_eq!(t, Tier::Mid);
        println!("  from_u8(1) = {:?}", t);
    }

    #[test]
    fn tier_from_u8_two_is_high() {
        let t = Tier::from_u8(2);
        assert_eq!(t, Tier::High);
        println!("  from_u8(2) = {:?}", t);
    }

    #[test]
    fn tier_from_u8_large_value_is_high() {
        let t = Tier::from_u8(255);
        assert_eq!(t, Tier::High);
        println!("  from_u8(255) = {:?}  (all values >= 2 map to High)", t);
    }

    #[test]
    fn tier_three_variants_are_distinct() {
        assert_ne!(Tier::Low,  Tier::Mid);
        assert_ne!(Tier::Mid,  Tier::High);
        assert_ne!(Tier::Low,  Tier::High);
        println!("  Low != Mid != High — all three variants are distinct");
    }

    // ── LogEntry ──────────────────────────────────────────────────────────

    #[test]
    fn log_entry_stores_fields_correctly() {
        let entry = LogEntry::new(LogLevel::Warn, Tier::Low, "buffer near capacity".to_string());
        assert_eq!(entry.level,   LogLevel::Warn);
        assert_eq!(entry.tier,    Tier::Low);
        assert_eq!(entry.message, "buffer near capacity");
        assert!(entry.timestamp > 0, "timestamp should be non-zero");
        println!(
            "  entry: level={:?}  tier={:?}  msg={:?}  timestamp={}",
            entry.level, entry.tier, entry.message, entry.timestamp,
        );
    }

    #[test]
    fn log_entry_timestamp_increases_monotonically() {
        let a = LogEntry::new(LogLevel::Info, Tier::High, "a".into());
        std::thread::sleep(std::time::Duration::from_millis(2));
        let b = LogEntry::new(LogLevel::Info, Tier::High, "b".into());
        assert!(b.timestamp >= a.timestamp);
        println!(
            "  a.timestamp={}  b.timestamp={}  delta={}ms",
            a.timestamp, b.timestamp, b.timestamp - a.timestamp,
        );
    }

    // ── Ring buffer ───────────────────────────────────────────────────────

    #[test]
    fn buffer_capacity_is_power_of_two() {
        let cap = crate::buffer::CAPACITY;
        assert!(cap > 0 && (cap & (cap - 1)) == 0,
            "CAPACITY={} must be a power of two", cap);
        println!("  CAPACITY={} ({} bits set)", cap, cap.count_ones());
    }

    #[test]
    fn buffer_create_returns_paired_producer_consumer() {
        let (mut prod, mut cons) = crate::buffer::create();
        let entry = LogEntry::new(LogLevel::Info, Tier::High, "ring buffer test".into());
        let msg = entry.message.clone();
        assert!(prod.push(entry).is_ok());
        let popped = cons.pop().expect("should pop the pushed entry");
        assert_eq!(popped.message, msg);
        println!("  pushed {:?} → popped {:?}", msg, popped.message);
    }

    #[test]
    fn buffer_empty_pop_returns_err() {
        let (_prod, mut cons) = crate::buffer::create();
        let result = cons.pop();
        assert!(result.is_err());
        println!("  pop on empty buffer = Err (correct — no blocking)");
    }

    // ── Logger lifecycle ──────────────────────────────────────────────────

    #[test]
    fn logger_init_succeeds_or_was_already_init() {
        let _ = MidLogger::init();
        let is_some = MidLogger::get().is_some();
        assert!(is_some);
        println!("  MidLogger::get().is_some() = {}", is_some);
    }

    #[test]
    fn logger_log_does_not_panic() {
        MidLogger::init();
        if let Some(logger) = MidLogger::get() {
            logger.log(LogLevel::Trace, Tier::Low,  "trace — engine internal".into());
            logger.log(LogLevel::Info,  Tier::Mid,  "info  — mid-level system".into());
            logger.log(LogLevel::Warn,  Tier::High, "warn  — gameplay logic".into());
            logger.log(LogLevel::Error, Tier::Low,  "error — non-fatal".into());
            println!("  logged Trace/Info/Warn/Error without panic");
        }
    }

    #[test]
    fn logger_accepts_all_tier_variants() {
        MidLogger::init();
        if let Some(logger) = MidLogger::get() {
            for tier in [Tier::Low, Tier::Mid, Tier::High] {
                logger.log(LogLevel::Info, tier, format!("tier {:?}", tier));
            }
            println!("  logged to Low, Mid, High tiers — all accepted");
        }
    }

    #[test]
    fn logger_accepts_empty_message() {
        MidLogger::init();
        if let Some(logger) = MidLogger::get() {
            logger.log(LogLevel::Info, Tier::Low, String::new());
            println!("  empty string message accepted without panic");
        }
    }

    #[test]
    fn logger_accepts_unicode_message() {
        MidLogger::init();
        if let Some(logger) = MidLogger::get() {
            let msg = "🦀 Rust + 🎮 Mid Engine".to_string();
            logger.log(LogLevel::Info, Tier::High, msg.clone());
            println!("  unicode message accepted: {:?}", msg);
        }
    }

    #[test]
    fn logger_handles_very_long_message() {
        MidLogger::init();
        if let Some(logger) = MidLogger::get() {
            let msg = "x".repeat(65_536);
            logger.log(LogLevel::Warn, Tier::Low, msg);
            println!("  65536-byte message accepted without panic or blocking");
        }
    }

    // ── Macro API ─────────────────────────────────────────────────────────

    #[test]
    fn macros_do_not_panic_before_init() {
        crate::mid_trace!(Tier::Low,  "before init");
        crate::mid_info! (Tier::Mid,  "before init");
        crate::mid_warn! (Tier::High, "before init");
        crate::mid_error!(Tier::Low,  "before init");
        println!("  all macros silently no-op when logger is not yet init");
    }

    #[test]
    fn macros_accept_format_args() {
        MidLogger::init();
        crate::mid_info!(Tier::High, "player {} spawned at ({:.1}, {:.1})", 42, 1.0, 2.5);
        crate::mid_warn!(Tier::Mid,  "system {}% loaded", 87);
        crate::mid_error!(Tier::Low, "entity {} missing component {}", 99, "Transform");
        println!("  format args: player 42, 87%, entity 99 — all accepted");
    }

    #[test]
    fn macros_cover_all_tiers() {
        MidLogger::init();
        crate::mid_info!(Tier::Low,  "engine internal");
        crate::mid_info!(Tier::Mid,  "mid-level system");
        crate::mid_info!(Tier::High, "gameplay logic");
        println!("  macros accepted Tier::Low, Mid, High without panic");
    }

    // ── FFI ───────────────────────────────────────────────────────────────

    #[test]
    fn ffi_init_returns_one_or_zero() {
        let result = crate::ffi::mid_log_init();
        assert!(result == 0 || result == 1,
            "mid_log_init must return 0 or 1, got {}", result);
        println!("  mid_log_init() = {} (0=already init, 1=fresh init)", result);
    }

    #[test]
    fn ffi_log_with_null_message_does_not_panic() {
        crate::ffi::mid_log_init();
        unsafe { crate::ffi::mid_log_info_c(0, std::ptr::null()); }
        println!("  null *const c_char → early return, no panic");
    }

    #[test]
    fn ffi_log_valid_message_all_levels_and_tiers() {
        crate::ffi::mid_log_init();
        let msg = std::ffi::CString::new("ffi test").unwrap();
        unsafe {
            crate::ffi::mid_log_info_c (0, msg.as_ptr()); // LOW
            crate::ffi::mid_log_info_c (1, msg.as_ptr()); // MID
            crate::ffi::mid_log_info_c (2, msg.as_ptr()); // HIGH
            crate::ffi::mid_log_trace_c(0, msg.as_ptr());
            crate::ffi::mid_log_warn_c (1, msg.as_ptr());
            crate::ffi::mid_log_error_c(2, msg.as_ptr());
        }
        println!("  6 FFI calls (3 tiers × 2 levels subset) — all accepted");
    }

    #[test]
    fn ffi_tier_constants_map_correctly() {
        let cases = [(0u8, Tier::Low, "MID_TIER_LOW"), (1, Tier::Mid, "MID_TIER_MID"), (2, Tier::High, "MID_TIER_HIGH")];
        for (v, expected, name) in cases {
            let got = Tier::from_u8(v);
            assert_eq!(got, expected, "{} = {}", name, v);
            println!("  {} ({}) → {:?}", name, v, got);
        }
    }
}
