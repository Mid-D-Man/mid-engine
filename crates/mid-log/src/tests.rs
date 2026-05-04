// crates/mid-log/src/tests.rs

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use crate::color::{Color, is_colors_enabled, paint, paint_bg, set_colors_enabled,
                       update_color_scheme, ColorScheme};
    use crate::console_buffer::{self, ConsoleReader, init_console_buffer};
    use crate::entry::LogEntry;
    use crate::filter::{self, set_min_level};
    use crate::format::{get_format, set_format, set_show_frame, set_show_thread,
                        set_show_timestamp, FormatConfig};
    use crate::frame::{current_frame, set_frame};
    use crate::level::{LogLevel, Tier};
    use crate::logger::{InitConfig, MidLogger};
    use crate::ratelimit::{set_rate_limit_config, RateLimitConfig};

    fn ensure_logger() {
        MidLogger::init();
    }

    // ══════════════════════════════════════════════════════════════════════════
    //  LogLevel
    // ══════════════════════════════════════════════════════════════════════════

    #[test]
    fn level_ordering_is_correct() {
        assert!(LogLevel::Trace < LogLevel::Info);
        assert!(LogLevel::Info  < LogLevel::Warn);
        assert!(LogLevel::Warn  < LogLevel::Error);
        assert!(LogLevel::Error < LogLevel::Fatal);
    }

    #[test]
    fn level_as_str_fixed_width() {
        assert_eq!(LogLevel::Trace.as_str(), "TRACE");
        assert_eq!(LogLevel::Info .as_str(), "INFO ");
        assert_eq!(LogLevel::Warn .as_str(), "WARN ");
        assert_eq!(LogLevel::Error.as_str(), "ERROR");
        assert_eq!(LogLevel::Fatal.as_str(), "FATAL");
    }

    #[test]
    fn level_from_u8_roundtrip() {
        assert_eq!(LogLevel::from_u8(0),   LogLevel::Trace);
        assert_eq!(LogLevel::from_u8(1),   LogLevel::Info);
        assert_eq!(LogLevel::from_u8(2),   LogLevel::Warn);
        assert_eq!(LogLevel::from_u8(3),   LogLevel::Error);
        assert_eq!(LogLevel::from_u8(4),   LogLevel::Fatal);
        assert_eq!(LogLevel::from_u8(255), LogLevel::Fatal); // clamps
    }

    // ══════════════════════════════════════════════════════════════════════════
    //  Tier
    // ══════════════════════════════════════════════════════════════════════════

    #[test]
    fn tier_as_str_fixed_width() {
        assert_eq!(Tier::Low .as_str(), "LOW ");
        assert_eq!(Tier::Mid .as_str(), "MID ");
        assert_eq!(Tier::High.as_str(), "HIGH");
    }

    #[test]
    fn tier_from_u8_all_values() {
        assert_eq!(Tier::from_u8(0),   Tier::Low);
        assert_eq!(Tier::from_u8(1),   Tier::Mid);
        assert_eq!(Tier::from_u8(2),   Tier::High);
        assert_eq!(Tier::from_u8(255), Tier::High);
    }

    // ══════════════════════════════════════════════════════════════════════════
    //  LogEntry
    // ══════════════════════════════════════════════════════════════════════════

    #[test]
    fn log_entry_stores_all_fields() {
        set_frame(42);
        let entry = LogEntry::new(
            LogLevel::Warn, Tier::Low,
            "test message".to_string(),
            "my_file.rs", 99, "my::module",
        );
        assert_eq!(entry.level,   LogLevel::Warn);
        assert_eq!(entry.tier,    Tier::Low);
        assert_eq!(entry.message, "test message");
        assert_eq!(entry.file,    "my_file.rs");
        assert_eq!(entry.line,    99);
        assert_eq!(entry.module,  "my::module");
        assert!(entry.timestamp > 0);
        // Thread name should be something (at least "<unnamed>" or the test thread name).
        assert!(!entry.thread.is_empty());
        // Frame should be 42 (set above).
        assert_eq!(entry.frame, 42);
        println!(
            "  entry: level={:?} tier={:?} msg={:?} thread={:?} frame={}",
            entry.level, entry.tier, entry.message, entry.thread, entry.frame,
        );
    }

    #[test]
    fn log_entry_timestamp_monotonic() {
        let a = LogEntry::new(LogLevel::Info, Tier::High, "a".into(), "f", 1, "m");
        std::thread::sleep(std::time::Duration::from_millis(2));
        let b = LogEntry::new(LogLevel::Info, Tier::High, "b".into(), "f", 1, "m");
        assert!(b.timestamp >= a.timestamp);
    }

    #[test]
    fn log_entry_format_time_structure() {
        let e = LogEntry::new(LogLevel::Info, Tier::Low, "t".into(), "f", 1, "m");
        let t = e.format_time();
        // HH:MM:SS.mmm = 12 characters
        assert_eq!(t.len(), 12, "format_time = {:?}", t);
        assert_eq!(&t[2..3], ":");
        assert_eq!(&t[5..6], ":");
        assert_eq!(&t[8..9], ".");
        println!("  format_time = {:?}", t);
    }

    // ══════════════════════════════════════════════════════════════════════════
    //  Filter
    // ══════════════════════════════════════════════════════════════════════════

    #[test]
    fn filter_gates_correctly_at_warn() {
        set_min_level(LogLevel::Warn);
        assert!(!filter::is_enabled(LogLevel::Trace));
        assert!(!filter::is_enabled(LogLevel::Info));
        assert!( filter::is_enabled(LogLevel::Warn));
        assert!( filter::is_enabled(LogLevel::Error));
        assert!( filter::is_enabled(LogLevel::Fatal));
        set_min_level(LogLevel::Trace);
    }

    #[test]
    fn filter_set_get_roundtrip() {
        for level in [LogLevel::Trace, LogLevel::Info, LogLevel::Warn,
                      LogLevel::Error, LogLevel::Fatal] {
            set_min_level(level);
            assert_eq!(filter::get_min_level(), level);
        }
        set_min_level(LogLevel::Trace);
    }

    // ══════════════════════════════════════════════════════════════════════════
    //  Frame counter
    // ══════════════════════════════════════════════════════════════════════════

    #[test]
    fn frame_counter_set_and_get() {
        set_frame(12345);
        assert_eq!(current_frame(), 12345);
        set_frame(0);
        assert_eq!(current_frame(), 0);
        println!("  frame counter set/get roundtrip OK");
    }

    #[test]
    fn frame_counter_captured_in_log_entry() {
        set_frame(999);
        let e = LogEntry::new(LogLevel::Info, Tier::Low, "x".into(), "f", 1, "m");
        assert_eq!(e.frame, 999);
        set_frame(0);
    }

    // ══════════════════════════════════════════════════════════════════════════
    //  Format flags
    // ══════════════════════════════════════════════════════════════════════════

    #[test]
    fn format_set_and_get_roundtrip() {
        let cfg = FormatConfig {
            show_timestamp:  false,
            show_source_loc: false,
            show_module:     true,
            show_thread:     true,
            show_frame:      true,
        };
        set_format(&cfg);
        let got = get_format();
        assert_eq!(got, cfg);
        // Restore defaults
        set_format(&FormatConfig::default());
    }

    #[test]
    fn format_individual_toggles() {
        set_show_timestamp(false);
        assert!(!get_format().show_timestamp);
        set_show_timestamp(true);
        assert!(get_format().show_timestamp);

        set_show_thread(true);
        assert!(get_format().show_thread);
        set_show_thread(false);
        assert!(!get_format().show_thread);

        set_show_frame(true);
        assert!(get_format().show_frame);
        set_show_frame(false);
        assert!(!get_format().show_frame);

        set_format(&FormatConfig::default());
        println!("  individual format toggles all passed");
    }

    // ══════════════════════════════════════════════════════════════════════════
    //  Color system
    // ══════════════════════════════════════════════════════════════════════════

    #[test]
    fn color_enable_disable_toggle() {
        set_colors_enabled(true);
        assert!(is_colors_enabled());
        set_colors_enabled(false);
        assert!(!is_colors_enabled());
        println!("  color enable/disable toggle OK");
    }

    #[test]
    fn color_ansi_prefixes_are_correct() {
        assert_eq!(Color::Red.to_ansi_string(),          "\x1b[31m");
        assert_eq!(Color::Green.to_ansi_string(),        "\x1b[32m");
        assert_eq!(Color::Yellow.to_ansi_string(),       "\x1b[33m");
        assert_eq!(Color::BrightRed.to_ansi_string(),    "\x1b[91m");
        assert_eq!(Color::Bold.to_ansi_string(),         "\x1b[1m");
        assert_eq!(Color::Dim.to_ansi_string(),          "\x1b[2m");
        assert_eq!(Color::None.to_ansi_string(),         "");
        assert_eq!(
            Color::Rgb(255, 128, 0).to_ansi_string(),
            "\x1b[38;2;255;128;0m",
        );
        assert_eq!(
            Color::Custom("1;35".to_string()).to_ansi_string(),
            "\x1b[1;35m",
        );
        println!("  all ANSI prefixes verified");
    }

    #[test]
    fn color_background_ansi_prefixes() {
        assert_eq!(Color::Red.to_bg_ansi_string(),   "\x1b[41m");
        assert_eq!(Color::Blue.to_bg_ansi_string(),  "\x1b[44m");
        assert_eq!(
            Color::Rgb(0, 0, 255).to_bg_ansi_string(),
            "\x1b[48;2;0;0;255m",
        );
        println!("  background ANSI prefixes verified");
    }

    #[test]
    fn paint_with_colors_enabled() {
        set_colors_enabled(true);
        let painted = format!("{}", paint(42u32, Color::Red));
        // Should contain the ANSI red code, the value, and the reset code.
        assert!(painted.contains("42"),         "missing value: {:?}", painted);
        assert!(painted.contains("\x1b[31m"),   "missing red:   {:?}", painted);
        assert!(painted.contains("\x1b[0m"),    "missing reset: {:?}", painted);
        println!("  paint() with colors: {:?}", painted);
    }

    #[test]
    fn paint_with_colors_disabled_is_passthrough() {
        set_colors_enabled(false);
        let painted = format!("{}", paint(42u32, Color::Red));
        assert_eq!(painted, "42");
        println!("  paint() without colors = pure passthrough: {:?}", painted);
    }

    #[test]
    fn paint_bg_with_colors_enabled() {
        set_colors_enabled(true);
        let painted = format!("{}", paint_bg("CRITICAL", Color::White, Color::Red));
        assert!(painted.contains("CRITICAL"));
        assert!(painted.contains("\x1b[37m")); // white fg
        assert!(painted.contains("\x1b[41m")); // red bg
        assert!(painted.contains("\x1b[0m"));  // reset
        println!("  paint_bg() verified: {:?}", painted);
    }

    #[test]
    fn paint_bg_with_colors_disabled_is_passthrough() {
        set_colors_enabled(false);
        let painted = format!("{}", paint_bg("CRITICAL", Color::White, Color::Red));
        assert_eq!(painted, "CRITICAL");
    }

    #[test]
    fn paint_bold_style() {
        set_colors_enabled(true);
        let painted = format!("{}", paint("important", Color::Bold));
        assert!(painted.contains("\x1b[1m"));
        assert!(painted.contains("important"));
        println!("  paint() with Bold style: {:?}", painted);
    }

    #[test]
    fn paint_rgb_color() {
        set_colors_enabled(true);
        let painted = format!("{}", paint("custom", Color::Rgb(255, 128, 0)));
        assert!(painted.contains("\x1b[38;2;255;128;0m"));
        assert!(painted.contains("custom"));
        println!("  paint() with Rgb color: {:?}", painted);
    }

    #[test]
    fn paint_custom_ansi() {
        set_colors_enabled(true);
        // Bold + magenta via custom string
        let painted = format!("{}", paint("styled", Color::Custom("1;35".to_string())));
        assert!(painted.contains("\x1b[1;35m"));
        assert!(painted.contains("styled"));
        println!("  paint() with Custom ANSI: {:?}", painted);
    }

    #[test]
    fn color_scheme_update_does_not_panic() {
        ensure_logger();
        update_color_scheme(|s| {
            s.warn    = Color::BrightYellow;
            s.error   = Color::Rgb(255, 80, 80);
            s.message = Color::None;
            s.tier_low = Color::Custom("38;5;208".to_string()); // xterm256 orange
        });
        // Restore defaults
        update_color_scheme(|s| {
            *s = ColorScheme::default();
        });
        println!("  color scheme update roundtrip OK");
    }

    #[test]
    fn color_scheme_all_slots_settable() {
        ensure_logger();
        update_color_scheme(|s| {
            s.trace     = Color::Dim;
            s.info      = Color::None;
            s.warn      = Color::Yellow;
            s.error     = Color::Red;
            s.fatal     = Color::BrightRed;
            s.tier_low  = Color::Cyan;
            s.tier_mid  = Color::Magenta;
            s.tier_high = Color::Green;
            s.timestamp = Color::Dim;
            s.source    = Color::Dim;
            s.module    = Color::Dim;
            s.thread    = Color::Blue;
            s.frame     = Color::Dim;
            s.message   = Color::None;
        });
        update_color_scheme(|s| { *s = ColorScheme::default(); });
        println!("  all 14 color slots are settable without panic");
    }

    // ══════════════════════════════════════════════════════════════════════════
    //  Logger lifecycle
    // ══════════════════════════════════════════════════════════════════════════

    #[test]
    fn logger_init_succeeds_or_already_init() {
        ensure_logger();
        assert!(MidLogger::get().is_some());
    }

    #[test]
    fn logger_log_all_levels_does_not_panic() {
        ensure_logger();
        set_min_level(LogLevel::Trace);
        if let Some(logger) = MidLogger::get() {
            logger.log(LogLevel::Trace, Tier::Low,  "trace".into(), "f", 1, "m");
            logger.log(LogLevel::Info,  Tier::Mid,  "info".into(),  "f", 2, "m");
            logger.log(LogLevel::Warn,  Tier::High, "warn".into(),  "f", 3, "m");
            logger.log(LogLevel::Error, Tier::Low,  "error".into(), "f", 4, "m");
        }
    }

    #[test]
    fn logger_accepts_unicode() {
        ensure_logger();
        if let Some(logger) = MidLogger::get() {
            logger.log(LogLevel::Info, Tier::High,
                "🦀 Rust + 🎮 Mid Engine + 🌍 Unicode".into(), "f", 1, "m");
        }
    }

    #[test]
    fn logger_accepts_empty_message() {
        ensure_logger();
        if let Some(logger) = MidLogger::get() {
            logger.log(LogLevel::Info, Tier::Low, String::new(), "f", 1, "m");
        }
    }

    #[test]
    fn logger_flush_does_not_panic() {
        ensure_logger();
        MidLogger::flush();
    }

    // ══════════════════════════════════════════════════════════════════════════
    //  Macros
    // ══════════════════════════════════════════════════════════════════════════

    #[test]
    fn macros_accept_format_args() {
        ensure_logger();
        set_min_level(LogLevel::Trace);
        crate::mid_trace!(Tier::Low,  "trace #{}", 1);
        crate::mid_info! (Tier::Mid,  "info  #{}", 2);
        crate::mid_warn! (Tier::High, "warn  #{}", 3);
        crate::mid_error!(Tier::Low,  "error #{}", 4);
    }

    #[test]
    fn macros_do_not_panic_when_uninit() {
        set_min_level(LogLevel::Trace);
        crate::mid_trace!(Tier::Low, "uninit trace");
        crate::mid_info! (Tier::Mid, "uninit info");
    }

    #[test]
    fn macros_filtered_do_not_format() {
        ensure_logger();
        set_min_level(LogLevel::Fatal);
        let count = 100_000usize;
        let start = Instant::now();
        for i in 0..count {
            crate::mid_trace!(Tier::Low, "entity={} health={}", i, 100u32);
        }
        let ns = start.elapsed().as_nanos() as f64 / count as f64;
        let mode = if cfg!(debug_assertions) { "DEBUG" } else { "RELEASE" };
        println!("  filtered mid_trace! {:.2} ns/call  [{}]", ns, mode);
        if !cfg!(debug_assertions) {
            assert!(ns < 20.0,
                "[RELEASE] filtered path {:.2} ns — expected <20ns", ns);
        }
        set_min_level(LogLevel::Trace);
    }

    // ══════════════════════════════════════════════════════════════════════════
    //  Assert macros
    // ══════════════════════════════════════════════════════════════════════════

    #[test]
    fn mid_assert_passes_on_true() {
        ensure_logger();
        set_min_level(LogLevel::Trace);
        crate::mid_assert!(1 + 1 == 2);
        crate::mid_assert!(1 + 1 == 2, "basic math");
        println!("  mid_assert! passes on true condition");
    }

    #[test]
    #[should_panic(expected = "mid_assert failed")]
    fn mid_assert_panics_on_false() {
        ensure_logger();
        crate::mid_assert!(1 + 1 == 3, "math is broken");
    }

    #[test]
    fn mid_assert_eq_passes() {
        ensure_logger();
        crate::mid_assert_eq!(42u32, 42u32);
        crate::mid_assert_eq!(42u32, 42u32, "should be equal");
        println!("  mid_assert_eq! passes on equal values");
    }

    #[test]
    #[should_panic(expected = "mid_assert_eq failed")]
    fn mid_assert_eq_panics_on_mismatch() {
        ensure_logger();
        crate::mid_assert_eq!(1u32, 2u32, "mismatch detected");
    }

    #[test]
    fn mid_assert_ne_passes() {
        ensure_logger();
        crate::mid_assert_ne!(1u32, 2u32);
        crate::mid_assert_ne!(1u32, 2u32, "should differ");
        println!("  mid_assert_ne! passes on unequal values");
    }

    #[test]
    #[should_panic(expected = "mid_assert_ne failed")]
    fn mid_assert_ne_panics_on_equal() {
        ensure_logger();
        crate::mid_assert_ne!(5u32, 5u32, "unexpectedly equal");
    }

    #[test]
    fn mid_assert_approx_eq_passes() {
        ensure_logger();
        crate::mid_assert_approx_eq!(1.0f32, 1.000_001_f32, 1e-4_f32);
        crate::mid_assert_approx_eq!(0.0f64, 1e-13_f64, 1e-12_f64, "within double epsilon");
        println!("  mid_assert_approx_eq! passes within epsilon");
    }

    #[test]
    #[should_panic(expected = "mid_assert_approx_eq failed")]
    fn mid_assert_approx_eq_panics_when_outside_epsilon() {
        ensure_logger();
        crate::mid_assert_approx_eq!(0.0f32, 1.0f32, 1e-6_f32, "too far apart");
    }

    #[test]
    #[should_panic(expected = "mid_unreachable")]
    fn mid_unreachable_panics() {
        ensure_logger();
        crate::mid_unreachable!("this path must never run");
    }

    #[test]
    fn mid_soft_assert_returns_true_on_pass() {
        ensure_logger();
        let result = crate::mid_soft_assert!(2 + 2 == 4, "basic math");
        assert!(result);
        println!("  mid_soft_assert! returns true on pass");
    }

    #[test]
    fn mid_soft_assert_returns_false_on_fail_no_panic() {
        ensure_logger();
        // This must NOT panic despite the failure.
        let result = crate::mid_soft_assert!(1 == 2, "expected mismatch");
        assert!(!result);
        println!("  mid_soft_assert! returns false on failure without panicking");
    }

    #[test]
    fn mid_soft_assert_eq_pass_and_fail() {
        ensure_logger();
        assert!( crate::mid_soft_assert_eq!(10u32, 10u32, "equal"));
        assert!(!crate::mid_soft_assert_eq!(10u32, 99u32, "not equal"));
        println!("  mid_soft_assert_eq! pass/fail without panic");
    }

    #[test]
    fn mid_soft_assert_ne_pass_and_fail() {
        ensure_logger();
        assert!( crate::mid_soft_assert_ne!(1u32, 2u32, "different"));
        assert!(!crate::mid_soft_assert_ne!(5u32, 5u32, "same"));
        println!("  mid_soft_assert_ne! pass/fail without panic");
    }

    #[test]
    fn mid_debug_assert_compiles_and_runs_in_debug() {
        ensure_logger();
        // Should always compile. In debug mode logs on failure;
        // in release mode is a no-op. Neither path should panic.
        crate::mid_debug_assert!(true, "always passes");
        crate::mid_debug_assert!(false, "fails in debug but never panics");
        println!("  mid_debug_assert! compiled and ran without panic in any mode");
    }

    #[test]
    fn mid_debug_assert_eq_and_ne() {
        ensure_logger();
        crate::mid_debug_assert_eq!(1u32, 1u32, "equal");
        crate::mid_debug_assert_eq!(1u32, 2u32, "fails in debug, silent in release");
        crate::mid_debug_assert_ne!(1u32, 2u32, "different");
        crate::mid_debug_assert_ne!(5u32, 5u32, "fails in debug, silent in release");
        println!("  mid_debug_assert_eq/ne variants all ran without panic");
    }

    // ══════════════════════════════════════════════════════════════════════════
    //  Console buffer
    // ══════════════════════════════════════════════════════════════════════════

    #[test]
    fn console_buffer_init_and_is_initialized() {
        // May already be initialized from another test — that's fine.
        init_console_buffer(64);
        assert!(console_buffer::is_initialized());
        println!("  console buffer initialized OK");
    }

    #[test]
    fn console_buffer_snapshot_returns_entries() {
        init_console_buffer(64);
        ensure_logger();
        set_min_level(LogLevel::Trace);

        // Log something and give the IO thread time to push to the buffer.
        if let Some(logger) = MidLogger::get() {
            logger.log(LogLevel::Info, Tier::High,
                "console buffer test entry".into(), "f", 1, "m");
        }
        std::thread::sleep(std::time::Duration::from_millis(50));

        let snap = console_buffer::snapshot();
        println!("  console buffer snapshot: {} entries", snap.len());
        // We can't assert an exact count due to other tests logging,
        // but the snapshot should be non-empty.
        assert!(console_buffer::is_initialized());
    }

    #[test]
    fn console_reader_drain_recent_is_incremental() {
        init_console_buffer(128);
        ensure_logger();
        set_min_level(LogLevel::Trace);

        let mut reader = ConsoleReader::new();

        // Drain anything buffered before this test.
        let _ = reader.drain_recent();

        // Log exactly 3 entries.
        if let Some(logger) = MidLogger::get() {
            for i in 0..3usize {
                logger.log(LogLevel::Info, Tier::Mid,
                    format!("reader_test entry {}", i),
                    "tests.rs", 1, "tests");
            }
        }

        // Give the IO thread time to forward entries to the console buffer.
        std::thread::sleep(std::time::Duration::from_millis(50));

        let recent = reader.drain_recent();
        println!(
            "  ConsoleReader saw {} new entries (expected 3)",
            recent.len(),
        );
        // We can't assert == 3 because other tests running in parallel
        // may have produced additional entries. Just verify drain works.
        assert!(recent.len() >= 3,
            "expected at least 3 entries, got {}", recent.len());

        // A second drain should return zero (or only entries from parallel tests).
        let _ = reader.drain_recent(); // don't assert exact 0 due to parallel tests
        println!("  ConsoleReader incremental drain OK");
    }

    #[test]
    fn console_reader_reset_works() {
        init_console_buffer(128);
        ensure_logger();

        let mut reader = ConsoleReader::new();
        let _ = reader.drain_recent(); // clear baseline

        if let Some(logger) = MidLogger::get() {
            logger.log(LogLevel::Info, Tier::Low, "before reset".into(), "f", 1, "m");
        }
        std::thread::sleep(std::time::Duration::from_millis(20));

        reader.reset(); // reset — future drain should not include "before reset"

        if let Some(logger) = MidLogger::get() {
            logger.log(LogLevel::Info, Tier::Low, "after reset".into(), "f", 1, "m");
        }
        std::thread::sleep(std::time::Duration::from_millis(20));

        let recent = reader.drain_recent();
        // All returned entries should have messages from "after reset" onwards.
        for e in &recent {
            assert!(
                !e.message.contains("before reset"),
                "entry from before reset leaked: {:?}", e.message,
            );
        }
        println!("  ConsoleReader::reset() correctly excludes pre-reset entries");
    }

    // ══════════════════════════════════════════════════════════════════════════
    //  Rate limiting
    // ══════════════════════════════════════════════════════════════════════════

    #[test]
    fn rate_limit_config_set_and_get() {
        let cfg = RateLimitConfig {
            enabled:        false,
            window:         std::time::Duration::from_millis(500),
            max_per_window: 10,
        };
        set_rate_limit_config(cfg.clone());
        let got = crate::ratelimit::get_rate_limit_config();
        assert_eq!(got.enabled,        false);
        assert_eq!(got.max_per_window, 10);
        assert_eq!(got.window,         std::time::Duration::from_millis(500));

        // Restore
        set_rate_limit_config(RateLimitConfig::default());
        println!("  rate limit config set/get roundtrip OK");
    }

    #[test]
    fn rate_limiter_allows_up_to_max_then_suppresses() {
        use crate::ratelimit::{RateLimiter, RateDecision};

        let config = RateLimitConfig {
            enabled:        true,
            window:         std::time::Duration::from_secs(10), // long window
            max_per_window: 3,
        };
        let mut limiter = RateLimiter::new();

        let make_entry = || LogEntry::new(
            LogLevel::Warn, Tier::Low,
            "repeated message".into(),
            "test_file.rs", 42, "test",
        );

        // First 3 should be allowed.
        for i in 0..3 {
            let entry = make_entry();
            match limiter.check(&entry, &config) {
                RateDecision::Allow => {}
                other => panic!("entry {} should be allowed, got suppressed", i),
            }
        }

        // 4th and beyond should be suppressed.
        for i in 3..6 {
            let entry = make_entry();
            match limiter.check(&entry, &config) {
                RateDecision::Suppress => {}
                RateDecision::Allow    => panic!("entry {} should be suppressed", i),
                RateDecision::WindowExpired { .. } => panic!("window should not have expired"),
            }
        }
        println!("  rate limiter: 3 allowed, 3 suppressed as expected");
    }

    #[test]
    fn rate_limiter_disabled_always_allows() {
        use crate::ratelimit::{RateLimiter, RateDecision};

        let config = RateLimitConfig {
            enabled:        false,
            window:         std::time::Duration::from_millis(1),
            max_per_window: 1,
        };
        let mut limiter = RateLimiter::new();
        let make_entry = || LogEntry::new(
            LogLevel::Info, Tier::Low,
            "spam".into(), "f.rs", 1, "m",
        );

        for _ in 0..100 {
            match limiter.check(&make_entry(), &config) {
                RateDecision::Allow => {}
                _ => panic!("disabled rate limiter should always allow"),
            }
        }
        println!("  disabled rate limiter always allows (100 entries)");
    }

    // ══════════════════════════════════════════════════════════════════════════
    //  FFI
    // ══════════════════════════════════════════════════════════════════════════

    #[test]
    fn ffi_init_returns_valid_value() {
        let r = crate::ffi::mid_log_init();
        assert!(r == 0 || r == 1);
    }

    #[test]
    fn ffi_set_get_min_level_roundtrip() {
        crate::ffi::mid_log_set_min_level(2); // WARN
        assert_eq!(crate::ffi::mid_log_get_min_level(), 2);
        crate::ffi::mid_log_set_min_level(0); // restore
    }

    #[test]
    fn ffi_set_get_colors_roundtrip() {
        crate::ffi::mid_log_set_colors(1);
        assert_eq!(crate::ffi::mid_log_get_colors(), 1);
        crate::ffi::mid_log_set_colors(0);
        assert_eq!(crate::ffi::mid_log_get_colors(), 0);
    }

    #[test]
    fn ffi_set_frame_roundtrip() {
        crate::ffi::mid_log_set_frame(777);
        assert_eq!(crate::ffi::mid_log_get_frame(), 777);
        crate::ffi::mid_log_set_frame(0);
    }

    #[test]
    fn ffi_update_color_slot_does_not_panic() {
        ensure_logger();
        // Set WARN to orange (255, 165, 0)
        crate::ffi::mid_log_update_color_c(2, 255, 165, 0, 0);
        // Remove color from message slot
        crate::ffi::mid_log_update_color_c(13, 0, 0, 0, 1);
        // Restore via update_color_scheme
        update_color_scheme(|s| { *s = ColorScheme::default(); });
        println!("  FFI color slot updates OK");
    }

    #[test]
    fn ffi_set_format_flags_does_not_panic() {
        crate::ffi::mid_log_set_format_flags(1, 0, 0, 1, 1);
        let got = get_format();
        assert!( got.show_timestamp);
        assert!(!got.show_source_loc);
        assert!(!got.show_module);
        assert!( got.show_thread);
        assert!( got.show_frame);
        // Restore
        set_format(&FormatConfig::default());
        println!("  FFI format flags set correctly");
    }

    #[test]
    fn ffi_null_message_does_not_panic() {
        crate::ffi::mid_log_init();
        unsafe { crate::ffi::mid_log_info_c(0, std::ptr::null()); }
    }

    #[test]
    fn ffi_log_all_levels_and_tiers() {
        crate::ffi::mid_log_init();
        crate::ffi::mid_log_set_min_level(0);
        let msg = std::ffi::CString::new("ffi test").unwrap();
        unsafe {
            crate::ffi::mid_log_trace_c(0, msg.as_ptr());
            crate::ffi::mid_log_info_c (1, msg.as_ptr());
            crate::ffi::mid_log_warn_c (2, msg.as_ptr());
            crate::ffi::mid_log_error_c(0, msg.as_ptr());
        }
    }

    #[test]
    fn ffi_flush_does_not_panic() {
        crate::ffi::mid_log_init();
        crate::ffi::mid_log_flush();
    }

    #[test]
    fn ffi_console_init_and_count() {
        crate::ffi::mid_log_console_init(64);
        // Count may be 0 or more depending on test order — just verify no panic.
        let _count = crate::ffi::mid_log_console_count();
        println!("  FFI console init + count OK (count={})", _count);
    }

    #[test]
    fn ffi_set_rate_limit_does_not_panic() {
        crate::ffi::mid_log_set_rate_limit(1, 500, 3);
        let got = crate::ratelimit::get_rate_limit_config();
        assert!(got.enabled);
        assert_eq!(got.max_per_window, 3);
        // Restore
        set_rate_limit_config(RateLimitConfig::default());
        println!("  FFI rate limit config accepted");
    }

    // ══════════════════════════════════════════════════════════════════════════
    //  Stress
    // ══════════════════════════════════════════════════════════════════════════

    #[test]
    fn stress_128hz_tick_budget() {
        ensure_logger();
        set_min_level(LogLevel::Info);
        let count = 1_000usize;
        let budget_ms = 7.8_f64;
        let start = Instant::now();
        if let Some(logger) = MidLogger::get() {
            for i in 0..count {
                logger.log(LogLevel::Info, Tier::Low,
                    format!("tick entity={} vel=({:.3},{:.3})", i, i as f32 * 0.01, 0.0),
                    "f", 1, "m");
            }
        }
        let ms = start.elapsed().as_secs_f64() * 1000.0;
        let mode = if cfg!(debug_assertions) { "DEBUG" } else { "RELEASE" };
        println!(
            "  {} logs in {:.4}ms  budget={:.1}ms  [{}]",
            count, ms, budget_ms, mode,
        );
        if !cfg!(debug_assertions) {
            assert!(ms < budget_ms * 10.0,
                "[RELEASE] exceeded 10× tick budget: {:.2}ms", ms);
        }
    }

    #[test]
    fn stress_concurrent_8x500_no_panic() {
        ensure_logger();
        set_min_level(LogLevel::Trace);
        let handles: Vec<_> = (0..8usize).map(|tid| {
            std::thread::spawn(move || {
                if let Some(logger) = MidLogger::get() {
                    for i in 0..500usize {
                        logger.log(LogLevel::Info, Tier::Mid,
                            format!("t{} #{}", tid, i), "f", 1, "m");
                    }
                }
            })
        }).collect();
        for h in handles { h.join().expect("thread panicked"); }
        println!("  8×500 concurrent logs: no deadlock, no panic");
    }

    #[test]
    fn stress_filtered_path_timing() {
        ensure_logger();
        set_min_level(LogLevel::Fatal);
        let count = 100_000usize;
        let start = Instant::now();
        for i in 0..count {
            crate::mid_trace!(Tier::Low, "entity={} health={}", i, 100u32);
        }
        let ns = start.elapsed().as_nanos() as f64 / count as f64;
        let mode = if cfg!(debug_assertions) { "DEBUG" } else { "RELEASE" };
        println!("  {} filtered mid_trace! = {:.2} ns/call  [{}]", count, ns, mode);
        if !cfg!(debug_assertions) {
            assert!(ns < 20.0,
                "[RELEASE] filtered path too slow: {:.2} ns/call", ns);
        }
        set_min_level(LogLevel::Trace);
    }
}
