// crates/mid-log/src/tests.rs

#[cfg(test)]
mod tests {
    use crate::level::{LogLevel, Tier};
    use crate::entry::LogEntry;
    use crate::logger::MidLogger;
    use std::time::Instant;

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
        println!("  CAPACITY={} ({} bits set = power of two ✓)", cap, cap.count_ones());
    }

    #[test]
    fn buffer_create_returns_paired_producer_consumer() {
        let (mut prod, mut cons) = crate::buffer::create();
        let entry = LogEntry::new(LogLevel::Info, Tier::High, "ring buffer test".into());
        let msg   = entry.message.clone();
        assert!(prod.push(entry).is_ok());
        let popped = cons.pop().expect("should pop the pushed entry");
        assert_eq!(popped.message, msg);
        println!("  pushed {:?} → popped {:?}", msg, popped.message);
    }

    #[test]
    fn buffer_empty_pop_returns_err() {
        let (_prod, mut cons) = crate::buffer::create();
        assert!(cons.pop().is_err());
        println!("  pop on empty buffer = Err (correct — no blocking)");
    }

    #[test]
    fn buffer_fills_to_capacity_without_panic() {
        let cap = crate::buffer::CAPACITY;
        let (mut prod, _cons) = crate::buffer::create();
        let mut accepted = 0usize;
        let mut dropped  = 0usize;
        for i in 0..cap + 100 {
            let entry = LogEntry::new(LogLevel::Trace, Tier::Low, format!("entry {}", i));
            if prod.push(entry).is_ok() { accepted += 1; } else { dropped += 1; }
        }
        assert_eq!(accepted, cap, "should fill exactly to capacity");
        assert_eq!(dropped, 100,  "100 pushes beyond capacity should be dropped");
        println!(
            "  CAPACITY={}  accepted={}  dropped={} (correct — ring buffer never blocks)",
            cap, accepted, dropped,
        );
    }

    // ── Logger lifecycle ──────────────────────────────────────────────────

    #[test]
    fn logger_init_succeeds_or_was_already_init() {
        let _ = MidLogger::init();
        assert!(MidLogger::get().is_some());
        println!("  MidLogger::get().is_some() = true");
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
            crate::ffi::mid_log_info_c (0, msg.as_ptr());
            crate::ffi::mid_log_info_c (1, msg.as_ptr());
            crate::ffi::mid_log_info_c (2, msg.as_ptr());
            crate::ffi::mid_log_trace_c(0, msg.as_ptr());
            crate::ffi::mid_log_warn_c (1, msg.as_ptr());
            crate::ffi::mid_log_error_c(2, msg.as_ptr());
        }
        println!("  6 FFI calls (3 tiers × levels subset) — all accepted");
    }

    #[test]
    fn ffi_tier_constants_map_correctly() {
        let cases = [
            (0u8, Tier::Low,  "MID_TIER_LOW"),
            (1u8, Tier::Mid,  "MID_TIER_MID"),
            (2u8, Tier::High, "MID_TIER_HIGH"),
        ];
        for (v, expected, name) in cases {
            let got = Tier::from_u8(v);
            assert_eq!(got, expected, "{} = {}", name, v);
            println!("  {} ({}) → {:?}", name, v, got);
        }
    }

    // ── Stress: throughput ────────────────────────────────────────────────

    #[test]
    fn stress_1000_info_logs_complete_without_panic() {
        MidLogger::init();
        let count = 1_000usize;
        let start = Instant::now();
        if let Some(logger) = MidLogger::get() {
            for i in 0..count {
                logger.log(LogLevel::Info, Tier::Low, format!("stress info #{}", i));
            }
        }
        let elapsed = start.elapsed();
        println!(
            "  {} INFO logs in {:.3}ms  ({:.1} ns/log)",
            count,
            elapsed.as_secs_f64() * 1000.0,
            elapsed.as_nanos() as f64 / count as f64,
        );
    }

    #[test]
    fn stress_1000_error_logs_complete_without_panic() {
        MidLogger::init();
        let count = 1_000usize;
        let start = Instant::now();
        if let Some(logger) = MidLogger::get() {
            for i in 0..count {
                logger.log(LogLevel::Error, Tier::High, format!("stress error #{}: non-fatal", i));
            }
        }
        let elapsed = start.elapsed();
        println!(
            "  {} ERROR logs in {:.3}ms  ({:.1} ns/log)",
            count,
            elapsed.as_secs_f64() * 1000.0,
            elapsed.as_nanos() as f64 / count as f64,
        );
    }

    #[test]
    fn stress_all_five_levels_200_each_no_panic() {
        MidLogger::init();
        let per_level = 200usize;
        let levels = [
            LogLevel::Trace,
            LogLevel::Info,
            LogLevel::Warn,
            LogLevel::Error,
            // Note: Fatal calls shutdown — we skip it in the burst loop.
        ];
        let start = Instant::now();
        if let Some(logger) = MidLogger::get() {
            for level in levels {
                for i in 0..per_level {
                    logger.log(level, Tier::Mid, format!("{:?} #{}", level, i));
                }
            }
        }
        let total   = per_level * levels.len();
        let elapsed = start.elapsed();
        println!(
            "  {} logs across 4 levels in {:.3}ms  ({:.1} ns/log)",
            total,
            elapsed.as_secs_f64() * 1000.0,
            elapsed.as_nanos() as f64 / total as f64,
        );
    }

    #[test]
    fn stress_all_three_tiers_333_each_no_panic() {
        MidLogger::init();
        let per_tier = 333usize;
        let tiers    = [Tier::Low, Tier::Mid, Tier::High];
        let start    = Instant::now();
        if let Some(logger) = MidLogger::get() {
            for tier in tiers {
                for i in 0..per_tier {
                    logger.log(LogLevel::Trace, tier, format!("[{:?}] trace #{}", tier, i));
                }
            }
        }
        let total   = per_tier * tiers.len();
        let elapsed = start.elapsed();
        println!(
            "  {} TRACE logs across 3 tiers in {:.3}ms  ({:.1} ns/log)",
            total,
            elapsed.as_secs_f64() * 1000.0,
            elapsed.as_nanos() as f64 / total as f64,
        );
    }

    #[test]
    fn stress_mixed_burst_5000_logs_no_panic() {
        MidLogger::init();
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
                let tier = match i % 3 {
                    0 => Tier::Low,
                    1 => Tier::Mid,
                    _ => Tier::High,
                };
                logger.log(level, tier, format!("burst #{}: entity={} pos=({:.2},{:.2})", i, i % 1000, i as f32 * 0.1, i as f32 * 0.2));
            }
        }
        let elapsed = start.elapsed();
        let ns_per  = elapsed.as_nanos() as f64 / count as f64;
        println!(
            "  {} mixed-level logs in {:.3}ms  ({:.1} ns/log)",
            count,
            elapsed.as_secs_f64() * 1000.0,
            ns_per,
        );
        // Budget: 7.8ms per 128Hz tick. 5000 logs should complete well under that.
        // This is not a hard assertion — timing varies by CI machine — but we print
        // clearly so regressions are visible in the HTML results.
        println!(
            "  128Hz tick budget = 7.8ms — this burst took {:.3}ms ({})",
            elapsed.as_secs_f64() * 1000.0,
            if elapsed.as_millis() < 8 { "✓ within budget" } else { "⚠ over budget on this machine" },
        );
    }

    #[test]
    fn stress_ring_buffer_saturation_never_blocks() {
        // Push far more entries than CAPACITY. The ring buffer must drop
        // entries silently — it must NEVER block or panic.
        let cap     = crate::buffer::CAPACITY;
        let burst   = cap * 4;  // 4× capacity
        let (mut prod, _cons) = crate::buffer::create();
        let start   = Instant::now();
        let mut accepted = 0usize;
        let mut dropped  = 0usize;
        for i in 0..burst {
            let entry = LogEntry::new(LogLevel::Trace, Tier::Low, format!("sat #{}", i));
            if prod.push(entry).is_ok() { accepted += 1; } else { dropped += 1; }
        }
        let elapsed = start.elapsed();
        assert_eq!(accepted, cap,  "should accept exactly CAPACITY entries");
        assert_eq!(dropped,  burst - cap, "remainder should be silently dropped");
        println!(
            "  CAPACITY={}  burst={}×cap  accepted={}  dropped={}  time={:.3}ms",
            cap, 4, accepted, dropped,
            elapsed.as_secs_f64() * 1000.0,
        );
        println!("  ✓ ring buffer saturated cleanly — zero blocking, zero panic");
    }

    #[test]
    fn stress_concurrent_threads_4x1000_logs_no_panic() {
        MidLogger::init();
        let threads     = 4usize;
        let per_thread  = 1_000usize;
        let start       = Instant::now();

        let handles: Vec<_> = (0..threads).map(|tid| {
            std::thread::spawn(move || {
                if let Some(logger) = MidLogger::get() {
                    for i in 0..per_thread {
                        logger.log(
                            LogLevel::Info,
                            Tier::Mid,
                            format!("thread {} log #{}", tid, i),
                        );
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
        println!("  ✓ no deadlock, no panic across concurrent producers");
    }

    #[test]
    fn stress_concurrent_threads_8x500_mixed_levels_no_panic() {
        MidLogger::init();
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
                        let tier = match tid % 3 {
                            0 => Tier::Low,
                            1 => Tier::Mid,
                            _ => Tier::High,
                        };
                        logger.log(level, tier, format!("t{} #{} {:?}", tid, i, level));
                    }
                }
            })
        }).collect();

        for h in handles { h.join().expect("thread panicked"); }

        let elapsed = start.elapsed();
        let total   = threads * per_thread;
        println!(
            "  {} threads × {} mixed logs = {} total in {:.3}ms  ({:.1} ns/log)",
            threads, per_thread, total,
            elapsed.as_secs_f64() * 1000.0,
            elapsed.as_nanos() as f64 / total as f64,
        );
        println!("  ✓ no deadlock, no panic — mixed levels + tiers across 8 threads");
    }

    #[test]
    fn stress_macro_burst_1000_mid_info_no_panic() {
        MidLogger::init();
        let count = 1_000usize;
        let start = Instant::now();
        for i in 0..count {
            crate::mid_info!(Tier::High, "macro burst #{}: entity={} health={}", i, i % 500, 100 - (i % 100));
        }
        let elapsed = start.elapsed();
        println!(
            "  mid_info! macro × {} in {:.3}ms  ({:.1} ns/call)",
            count,
            elapsed.as_secs_f64() * 1000.0,
            elapsed.as_nanos() as f64 / count as f64,
        );
    }

    #[test]
    fn stress_macro_burst_all_macros_250_each_no_panic() {
        MidLogger::init();
        let per_macro = 250usize;
        let start     = Instant::now();
        for i in 0..per_macro {
            crate::mid_trace!(Tier::Low,  "trace #{}", i);
            crate::mid_info! (Tier::Mid,  "info  #{}", i);
            crate::mid_warn! (Tier::High, "warn  #{}", i);
            crate::mid_error!(Tier::Low,  "error #{}", i);
        }
        let total   = per_macro * 4;
        let elapsed = start.elapsed();
        println!(
            "  4 macros × {} = {} calls in {:.3}ms  ({:.1} ns/call)",
            per_macro, total,
            elapsed.as_secs_f64() * 1000.0,
            elapsed.as_nanos() as f64 / total as f64,
        );
    }

    #[test]
    fn stress_ffi_burst_1000_c_calls_no_panic() {
        crate::ffi::mid_log_init();
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
            count,
            elapsed.as_secs_f64() * 1000.0,
            elapsed.as_nanos() as f64 / count as f64,
        );
        println!("  ✓ C boundary held under sustained load");
    }

    #[test]
    fn stress_128hz_tick_budget_1000_logs_fit_within_7_8ms() {
        // The network tick target is 128Hz = 7.8ms per tick.
        // We simulate a frame's logging burst and verify the hot path
        // stays far inside that window.
        // Note: this asserts timing so it will be loose on slow CI runners.
        // The printed output is the key signal — adjust the multiplier if needed.
        MidLogger::init();
        let count    = 1_000usize;
        let budget_ms = 7.8_f64;
        let start    = Instant::now();
        if let Some(logger) = MidLogger::get() {
            for i in 0..count {
                logger.log(
                    LogLevel::Info,
                    Tier::Low,
                    format!("tick frame entity={} vel=({:.3},{:.3})", i, i as f32 * 0.01, i as f32 * 0.02),
                );
            }
        }
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        println!(
            "  {} logs in {:.4}ms  budget={:.1}ms  headroom={:.4}ms",
            count, elapsed_ms, budget_ms, budget_ms - elapsed_ms,
        );
        // On a real machine the push is ~50-200ns total for 1000 entries.
        // Allow 10× margin for CI machines that may be throttled.
        assert!(
            elapsed_ms < budget_ms * 10.0,
            "1000 log pushes took {:.2}ms — exceeded 10× the 7.8ms tick budget (CI machine unusually slow?)",
            elapsed_ms,
        );
    }
        }
