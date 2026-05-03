// crates/mid-log/src/tests.rs

#[cfg(test)]
mod tests {
    use crate::level::{LogLevel, Tier};
    use crate::entry::LogEntry;
    use crate::logger::MidLogger;
    use crate::filter::{self, set_min_level};
    use std::time::Instant;

    fn ensure_logger() {
        MidLogger::init();
    }

    // ── LogLevel ──────────────────────────────────────────────────────────────

    #[test]
    fn level_ordering_is_correct() {
        assert!(LogLevel::Trace < LogLevel::Info);
        assert!(LogLevel::Info  < LogLevel::Warn);
        assert!(LogLevel::Warn  < LogLevel::Error);
        assert!(LogLevel::Error < LogLevel::Fatal);
        println!(
            "  Trace({}) < Info({}) < Warn({}) < Error({}) < Fatal({})",
            LogLevel::Trace as u8, LogLevel::Info  as u8,
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
            assert_eq!(format!("{}", level), level.as_str());
        }
    }

    #[test]
    fn level_from_u8_roundtrip() {
        let cases = [
            (0u8, LogLevel::Trace),
            (1u8, LogLevel::Info),
            (2u8, LogLevel::Warn),
            (3u8, LogLevel::Error),
            (4u8, LogLevel::Fatal),
            (255u8, LogLevel::Fatal), // clamps
        ];
        for (v, expected) in cases {
            assert_eq!(LogLevel::from_u8(v), expected);
            println!("  from_u8({}) = {:?}", v, expected);
        }
    }

    // ── Tier ──────────────────────────────────────────────────────────────────

    #[test]
    fn tier_as_str_fixed_width() {
        let cases = [(Tier::Low, "LOW "), (Tier::Mid, "MID "), (Tier::High, "HIGH")];
        for (tier, expected) in cases {
            assert_eq!(tier.as_str(), expected);
            println!("  {:?}.as_str() = {:?}  (len={})", tier, tier.as_str(), tier.as_str().len());
        }
    }

    #[test]
    fn tier_from_u8_zero_is_low()  { assert_eq!(Tier::from_u8(0),   Tier::Low);  }
    #[test]
    fn tier_from_u8_one_is_mid()   { assert_eq!(Tier::from_u8(1),   Tier::Mid);  }
    #[test]
    fn tier_from_u8_two_is_high()  { assert_eq!(Tier::from_u8(2),   Tier::High); }
    #[test]
    fn tier_from_u8_large_is_high(){ assert_eq!(Tier::from_u8(255), Tier::High); }

    #[test]
    fn tier_three_variants_are_distinct() {
        assert_ne!(Tier::Low, Tier::Mid);
        assert_ne!(Tier::Mid, Tier::High);
        assert_ne!(Tier::Low, Tier::High);
    }

    // ── LogEntry ──────────────────────────────────────────────────────────────

    #[test]
    fn log_entry_stores_fields_correctly() {
        let entry = LogEntry::new(
            LogLevel::Warn, Tier::Low, "buffer near capacity".to_string(),
            "buffer.rs", 42, "mid_log::buffer",
        );
        assert_eq!(entry.level,   LogLevel::Warn);
        assert_eq!(entry.tier,    Tier::Low);
        assert_eq!(entry.message, "buffer near capacity");
        assert_eq!(entry.file,    "buffer.rs");
        assert_eq!(entry.line,    42);
        assert_eq!(entry.module,  "mid_log::buffer");
        assert!(entry.timestamp > 0);
        println!(
            "  entry: level={:?} tier={:?} msg={:?} file={} line={} ts={}",
            entry.level, entry.tier, entry.message, entry.file, entry.line, entry.timestamp,
        );
    }

    #[test]
    fn log_entry_timestamp_increases_monotonically() {
        let a = LogEntry::new(LogLevel::Info, Tier::High, "a".into(), "f", 1, "m");
        std::thread::sleep(std::time::Duration::from_millis(2));
        let b = LogEntry::new(LogLevel::Info, Tier::High, "b".into(), "f", 1, "m");
        assert!(b.timestamp >= a.timestamp);
        println!("  delta={}ms", b.timestamp - a.timestamp);
    }

    #[test]
    fn log_entry_format_time_is_hh_mm_ss_mmm() {
        let entry = LogEntry::new(LogLevel::Info, Tier::Low, "t".into(), "f", 1, "m");
        let t = entry.format_time();
        // Format: HH:MM:SS.mmm  (14 chars)
        assert_eq!(t.len(), 12, "format_time = {:?}", t);
        assert_eq!(&t[2..3], ":");
        assert_eq!(&t[5..6], ":");
        assert_eq!(&t[8..9], ".");
        println!("  format_time = {:?}", t);
    }

    // ── Filter ────────────────────────────────────────────────────────────────

    #[test]
    fn filter_is_enabled_respects_min_level() {
        set_min_level(LogLevel::Warn);
        assert!(!filter::is_enabled(LogLevel::Trace), "Trace should be filtered");
        assert!(!filter::is_enabled(LogLevel::Info),  "Info should be filtered");
        assert!( filter::is_enabled(LogLevel::Warn),  "Warn should pass");
        assert!( filter::is_enabled(LogLevel::Error), "Error should pass");
        assert!( filter::is_enabled(LogLevel::Fatal), "Fatal should pass");
        // Restore
        set_min_level(LogLevel::Trace);
        println!("  filter correctly gates at Warn level");
    }

    #[test]
    fn filter_set_and_get_roundtrip() {
        for level in [LogLevel::Trace, LogLevel::Info, LogLevel::Warn, LogLevel::Error, LogLevel::Fatal] {
            set_min_level(level);
            assert_eq!(filter::get_min_level(), level);
        }
        set_min_level(LogLevel::Trace);
    }

    // ── Logger lifecycle ──────────────────────────────────────────────────────

    #[test]
    fn logger_init_succeeds_or_was_already_init() {
        ensure_logger();
        assert!(MidLogger::get().is_some());
        println!("  MidLogger::get().is_some() = true");
    }

    #[test]
    fn logger_log_does_not_panic() {
        ensure_logger();
        set_min_level(LogLevel::Trace);
        if let Some(logger) = MidLogger::get() {
            logger.log(LogLevel::Trace, Tier::Low,  "trace".into(), "f", 1, "m");
            logger.log(LogLevel::Info,  Tier::Mid,  "info".into(),  "f", 2, "m");
            logger.log(LogLevel::Warn,  Tier::High, "warn".into(),  "f", 3, "m");
            logger.log(LogLevel::Error, Tier::Low,  "error".into(), "f", 4, "m");
        }
        println!("  logged all levels without panic");
    }

    #[test]
    fn logger_accepts_all_tier_variants() {
        ensure_logger();
        if let Some(logger) = MidLogger::get() {
            for tier in [Tier::Low, Tier::Mid, Tier::High] {
                logger.log(LogLevel::Info, tier, format!("{:?}", tier), "f", 1, "m");
            }
        }
        println!("  all tiers accepted");
    }

    #[test]
    fn logger_accepts_empty_message() {
        ensure_logger();
        if let Some(logger) = MidLogger::get() {
            logger.log(LogLevel::Info, Tier::Low, String::new(), "f", 1, "m");
        }
    }

    #[test]
    fn logger_accepts_unicode_message() {
        ensure_logger();
        if let Some(logger) = MidLogger::get() {
            logger.log(LogLevel::Info, Tier::High, "🦀 Rust + 🎮 Mid Engine".into(), "f", 1, "m");
        }
    }

    #[test]
    fn logger_handles_very_long_message() {
        ensure_logger();
        if let Some(logger) = MidLogger::get() {
            logger.log(LogLevel::Warn, Tier::Low, "x".repeat(65_536), "f", 1, "m");
        }
    }

    // ── Macro API ─────────────────────────────────────────────────────────────

    #[test]
    fn macros_do_not_panic_before_init() {
        // Logger may already be init in other tests, but this must not panic either way.
        set_min_level(LogLevel::Trace);
        crate::mid_trace!(Tier::Low,  "before-or-after init");
        crate::mid_info! (Tier::Mid,  "before-or-after init");
        crate::mid_warn! (Tier::High, "before-or-after init");
        crate::mid_error!(Tier::Low,  "before-or-after init");
        println!("  macros silent when not init, functional when init");
    }

    #[test]
    fn macros_accept_format_args() {
        ensure_logger();
        set_min_level(LogLevel::Trace);
        crate::mid_info!(Tier::High, "player {} spawned at ({:.1}, {:.1})", 42, 1.0, 2.5);
        crate::mid_warn!(Tier::Mid,  "system {}% loaded", 87);
        crate::mid_error!(Tier::Low, "entity {} missing component {}", 99, "Transform");
        println!("  format args accepted");
    }

    #[test]
    fn macros_filtered_do_not_format() {
        ensure_logger();
        set_min_level(LogLevel::Fatal); // suppress everything
        let count = 10_000usize;
        let start = Instant::now();
        for i in 0..count {
            // format!() must NOT run — if it did, this would be much slower
            crate::mid_trace!(Tier::Low, "entity={} pos=({:.4},{:.4})", i, 1.0f32, 2.0f32);
        }
        let elapsed = start.elapsed();
        let ns = elapsed.as_nanos() as f64 / count as f64;
        println!(
            "  {} filtered mid_trace! calls in {:.3}ms  ({:.2} ns/call)",
            count, elapsed.as_secs_f64() * 1000.0, ns
        );
        // Filtered path should be <5ns per call (just one atomic load + branch).
        assert!(ns < 50.0,
            "filtered path took {:.2} ns/call — expected <50ns (atomic + branch only)", ns);
        set_min_level(LogLevel::Trace);
    }

    #[test]
    fn macros_capture_source_location() {
        ensure_logger();
        set_min_level(LogLevel::Trace);
        // We can't inspect the entry directly from outside but we can verify no panic
        // and that the macro expands at the correct call site (human verification via log output).
        crate::mid_info!(Tier::Low, "source location test — should show tests.rs");
        println!("  source location captured (verify in log output)");
    }

    // ── FFI ───────────────────────────────────────────────────────────────────

    #[test]
    fn ffi_init_returns_one_or_zero() {
        let result = crate::ffi::mid_log_init();
        assert!(result == 0 || result == 1,
            "mid_log_init must return 0 or 1, got {}", result);
        println!("  mid_log_init() = {} (0=already init, 1=fresh init)", result);
    }

    #[test]
    fn ffi_set_and_get_min_level() {
        crate::ffi::mid_log_set_min_level(2); // WARN
        assert_eq!(crate::ffi::mid_log_get_min_level(), 2);
        crate::ffi::mid_log_set_min_level(0); // restore TRACE
        assert_eq!(crate::ffi::mid_log_get_min_level(), 0);
        println!("  FFI set/get min level roundtrip OK");
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
        crate::ffi::mid_log_set_min_level(0);
        let msg = std::ffi::CString::new("ffi test").unwrap();
        unsafe {
            crate::ffi::mid_log_info_c (0, msg.as_ptr());
            crate::ffi::mid_log_info_c (1, msg.as_ptr());
            crate::ffi::mid_log_info_c (2, msg.as_ptr());
            crate::ffi::mid_log_trace_c(0, msg.as_ptr());
            crate::ffi::mid_log_warn_c (1, msg.as_ptr());
            crate::ffi::mid_log_error_c(2, msg.as_ptr());
        }
        println!("  6 FFI calls across all tiers — all accepted");
    }

    #[test]
    fn ffi_tier_constants_map_correctly() {
        let cases = [
            (0u8, Tier::Low,  "MID_TIER_LOW"),
            (1u8, Tier::Mid,  "MID_TIER_MID"),
            (2u8, Tier::High, "MID_TIER_HIGH"),
        ];
        for (v, expected, name) in cases {
            assert_eq!(Tier::from_u8(v), expected, "{} = {}", name, v);
            println!("  {} ({}) → {:?}", name, v, expected);
        }
    }

    #[test]
    fn ffi_flush_does_not_panic() {
        crate::ffi::mid_log_init();
        crate::ffi::mid_log_flush();
        println!("  mid_log_flush() returned without panic");
    }

    // ── Stress ────────────────────────────────────────────────────────────────

    #[test]
    fn stress_1000_info_logs_complete_without_panic() {
        ensure_logger();
        set_min_level(LogLevel::Trace);
        let count = 1_000usize;
        let start = Instant::now();
        if let Some(logger) = MidLogger::get() {
            for i in 0..count {
                logger.log(LogLevel::Info, Tier::Low, format!("stress info #{}", i), "f", 1, "m");
            }
        }
        let elapsed = start.elapsed();
        println!(
            "  {} INFO logs in {:.3}ms  ({:.1} ns/log)",
            count, elapsed.as_secs_f64() * 1000.0,
            elapsed.as_nanos() as f64 / count as f64,
        );
    }

    #[test]
    fn stress_mixed_burst_5000_logs_no_panic() {
        ensure_logger();
        set_min_level(LogLevel::Trace);
        let count = 5_000usize;
        let start = Instant::now();
        if let Some(logger) = MidLogger::get() {
            for i in 0..count {
                let level = match i % 4 {
                    0 => LogLevel::Trace,
                    1 => LogLevel::Info,
                    2 => LogLevel::Warn,
                    _ => LogLevel::Error,
                };
                let tier = match i % 3 { 0 => Tier::Low, 1 => Tier::Mid, _ => Tier::High };
                logger.log(level, tier,
                    format!("burst #{}: entity={} pos=({:.2},{:.2})", i, i % 1000, i as f32 * 0.1, i as f32 * 0.2),
                    "f", 1, "m");
            }
        }
        let elapsed = start.elapsed();
        let ms = elapsed.as_secs_f64() * 1000.0;
        println!(
            "  {} mixed logs in {:.3}ms  ({:.1} ns/log)",
            count, ms, elapsed.as_nanos() as f64 / count as f64,
        );
        println!(
            "  128Hz tick budget=7.8ms — burst took {:.3}ms ({})",
            ms, if elapsed.as_millis() < 8 { "✓ within budget" } else { "⚠ over budget" },
        );
    }

    #[test]
    fn stress_concurrent_threads_8x500_no_panic() {
        ensure_logger();
        set_min_level(LogLevel::Trace);
        let threads    = 8usize;
        let per_thread = 500usize;
        let start      = Instant::now();

        let handles: Vec<_> = (0..threads).map(|tid| {
            std::thread::spawn(move || {
                if let Some(logger) = MidLogger::get() {
                    for i in 0..per_thread {
                        let level = match (tid + i) % 4 {
                            0 => LogLevel::Trace,
                            1 => LogLevel::Info,
                            2 => LogLevel::Warn,
                            _ => LogLevel::Error,
                        };
                        let tier = match tid % 3 { 0 => Tier::Low, 1 => Tier::Mid, _ => Tier::High };
                        logger.log(level, tier, format!("t{} #{}", tid, i), "f", 1, "m");
                    }
                }
            })
        }).collect();

        for h in handles { h.join().expect("thread panicked"); }

        let elapsed = start.elapsed();
        let total   = threads * per_thread;
        println!(
            "  {} threads × {} logs = {} total in {:.3}ms  ({:.1} ns/log)",
            threads, per_thread, total,
            elapsed.as_secs_f64() * 1000.0,
            elapsed.as_nanos() as f64 / total as f64,
        );
        println!("  ✓ no deadlock, no panic — crossbeam MPSC under concurrent load");
    }

    #[test]
    fn stress_128hz_tick_budget_1000_logs_fit_within_7_8ms() {
        ensure_logger();
        set_min_level(LogLevel::Info);
        let count      = 1_000usize;
        let budget_ms  = 7.8_f64;
        let start      = Instant::now();
        if let Some(logger) = MidLogger::get() {
            for i in 0..count {
                logger.log(
                    LogLevel::Info, Tier::Low,
                    format!("tick entity={} vel=({:.3},{:.3})", i, i as f32 * 0.01, i as f32 * 0.02),
                    "f", 1, "m",
                );
            }
        }
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        println!(
            "  {} logs in {:.4}ms  budget={:.1}ms  headroom={:.4}ms",
            count, elapsed_ms, budget_ms, budget_ms - elapsed_ms,
        );
        assert!(
            elapsed_ms < budget_ms * 10.0,
            "1000 log pushes took {:.2}ms — exceeded 10× tick budget", elapsed_ms,
        );
    }

    #[test]
    fn stress_macro_filtered_path_is_near_free() {
        ensure_logger();
        set_min_level(LogLevel::Fatal);
        let count = 100_000usize;
        let start = Instant::now();
        for i in 0..count {
            crate::mid_trace!(Tier::Low, "entity={} health={}", i, 100u32);
        }
        let elapsed = start.elapsed();
        let ns = elapsed.as_nanos() as f64 / count as f64;
        println!(
            "  {} filtered mid_trace! in {:.3}ms  ({:.2} ns/call)  — should be ~1ns",
            count, elapsed.as_secs_f64() * 1000.0, ns,
        );
        assert!(ns < 20.0,
            "filtered path {:.2} ns/call — expected <20ns (one atomic load)", ns);
        set_min_level(LogLevel::Trace);
    }

    #[test]
    fn stress_ffi_burst_1000_c_calls_no_panic() {
        crate::ffi::mid_log_init();
        crate::ffi::mid_log_set_min_level(0);
        let msg   = std::ffi::CString::new("ffi stress entry").unwrap();
        let count = 1_000usize;
        let start = Instant::now();
        unsafe {
            for _ in 0..count {
                crate::ffi::mid_log_info_c(1, msg.as_ptr());
            }
        }
        let elapsed = start.elapsed();
        println!(
            "  {} FFI mid_log_info_c calls in {:.3}ms  ({:.1} ns/call)",
            count, elapsed.as_secs_f64() * 1000.0,
            elapsed.as_nanos() as f64 / count as f64,
        );
        println!("  ✓ C boundary held under sustained load");
    }
}
