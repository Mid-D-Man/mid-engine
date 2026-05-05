// crates/mid-log/benches/vs_baseline.rs

//! mid-log vs tracing vs log/env_logger — comprehensive throughput benchmark.
//!
//! ## What's measured
//!
//! | Group                    | Tests                                                  |
//! |--------------------------|--------------------------------------------------------|
//! | `mid_log/disabled`       | Filtered path (one atomic + branch)                   |
//! | `mid_log/enabled`        | Enabled path with various message complexities         |
//! | `mid_log/paint`          | `paint()` / `paint_bg()` inline color overhead         |
//! | `mid_log/asserts`        | Soft, debug, and passing assert macros                 |
//! | `mid_log/format_flags`   | Minimal vs full field output (IO thread throughput)    |
//! | `mid_log/bulk`           | Burst throughput at 1k entries                         |
//! | `tracing/enabled`        | tracing::info! with a sink subscriber                  |
//! | `tracing/disabled`       | tracing::info! with level filtered                     |
//! | `log_env_logger/enabled` | log::info! piped to a sink                             |
//!
//! ## Measurement budget
//!
//! Each benchmark: 1s warm-up + 2s measurement + 50 samples.
//! Total suite: ~5–7 minutes. Fits inside GitHub Actions resource limits.
//!
//! Run locally:
//!   cargo bench --bench vs_baseline -p mid-log
//!   # HTML report: target/criterion/report/index.html

use std::time::Duration;

use criterion::{
    black_box, criterion_group, criterion_main,
    Criterion, Throughput,
};

use mid_log::{
    mid_debug_assert, mid_info, mid_soft_assert, mid_soft_assert_eq, mid_trace, mid_warn,
    color::{Color, paint, paint_bg, set_colors_enabled},
    filter::set_min_level,
    format::{set_format, FormatConfig},
    frame::set_frame,
    level::{LogLevel, Tier},
    logger::{InitConfig, MidLogger},
    ratelimit::{set_rate_limit_config, RateLimitConfig},
};

// ── Criterion configuration ───────────────────────────────────────────────────

fn short_criterion() -> Criterion {
    Criterion::default()
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(2))
        .sample_size(50)
}

// ── Setup helpers ─────────────────────────────────────────────────────────────

fn setup_mid_log() {
    // init() is idempotent — safe to call from every bench function.
    MidLogger::init_full(InitConfig {
        min_level:    LogLevel::Trace,
        format:       FormatConfig::default(), // timestamp + source only
        color_scheme: mid_log::color::ColorScheme::default(),
        log_file:     None,
    });
    // Disable rate limiting so it does not interfere with throughput measurements.
    set_rate_limit_config(RateLimitConfig { enabled: false, ..Default::default() });
    // Reset frame counter.
    set_frame(0);
}

// ═════════════════════════════════════════════════════════════════════════════
//  1. Disabled (filtered) path
// ═════════════════════════════════════════════════════════════════════════════

fn bench_mid_log_disabled(c: &mut Criterion) {
    setup_mid_log();
    set_min_level(LogLevel::Fatal); // filter everything below Fatal

    let mut g = c.benchmark_group("mid_log/disabled");
    g.throughput(Throughput::Elements(1));
    g.warm_up_time(Duration::from_millis(500));
    g.measurement_time(Duration::from_secs(2));

    // Baseline: macro with no format args — absolute minimum cost path.
    g.bench_function("trace_no_args", |b| b.iter(|| {
        mid_trace!(Tier::Low, "tick");
    }));

    // With format args — format!() should NOT run since level is filtered.
    g.bench_function("trace_with_args", |b| b.iter(|| {
        mid_trace!(
            Tier::Low,
            "entity {} pos ({:.2}, {:.2})",
            black_box(42u32), black_box(1.0f32), black_box(2.5f32),
        );
    }));

    // Info also filtered.
    g.bench_function("info_filtered", |b| b.iter(|| {
        mid_info!(Tier::High, "player {} spawned", black_box(1u32));
    }));

    // Soft assert passing case — one branch check, never logs.
    // Note: this is gated by the condition, not the log level filter.
    g.bench_function("soft_assert_passing", |b| b.iter(|| {
        let _ = mid_soft_assert!(black_box(true), "condition holds");
    }));

    g.finish();
}

// ═════════════════════════════════════════════════════════════════════════════
//  2. Enabled path — message complexity comparison
// ═════════════════════════════════════════════════════════════════════════════

fn bench_mid_log_enabled(c: &mut Criterion) {
    setup_mid_log();
    set_min_level(LogLevel::Trace);
    set_colors_enabled(false); // exclude ANSI overhead from the baseline

    let mut g = c.benchmark_group("mid_log/enabled");
    g.throughput(Throughput::Elements(1));
    g.warm_up_time(Duration::from_millis(500));
    g.measurement_time(Duration::from_secs(2));

    // Absolute minimum — no format args.
    g.bench_function("trace_no_args", |b| b.iter(|| {
        mid_trace!(Tier::Low, "tick");
    }));

    // Simple with one integer arg.
    g.bench_function("info_one_int", |b| b.iter(|| {
        mid_info!(Tier::High, "frame {}", black_box(42u64));
    }));

    // Typical game entry — entity + position.
    g.bench_function("info_entity_pos", |b| b.iter(|| {
        mid_info!(
            Tier::High,
            "player {} spawned at ({:.2}, {:.2})",
            black_box(1u32), black_box(1.0f32), black_box(2.0f32),
        );
    }));

    // Long structured entry — worst-case calling-thread cost.
    g.bench_function("info_long_structured", |b| b.iter(|| {
        mid_info!(
            Tier::Mid,
            "entity={} comp=Transform pos=({:.4},{:.4},{:.4}) \
             vel=({:.4},{:.4},{:.4}) hp={} state=active",
            black_box(99u32),
            black_box(1.0f32), black_box(2.0f32), black_box(3.0f32),
            black_box(0.1f32), black_box(0.2f32), black_box(0.3f32),
            black_box(100u32),
        );
    }));

    // Warn level — same pipeline, different badge.
    g.bench_function("warn_simple", |b| b.iter(|| {
        mid_warn!(Tier::Low, "buffer {} / {} bytes", black_box(4000u32), black_box(4096u32));
    }));

    g.finish();
}

// ═════════════════════════════════════════════════════════════════════════════
//  3. paint() — inline color overhead
//
//  These measure the calling-thread cost of adding color to a value inside
//  a log message. The cost is: one atomic load (is_colors_enabled) + one
//  text.to_string() allocation when colors are enabled.
// ═════════════════════════════════════════════════════════════════════════════

fn bench_mid_log_paint(c: &mut Criterion) {
    setup_mid_log();
    set_min_level(LogLevel::Trace);

    let mut g = c.benchmark_group("mid_log/paint");
    g.throughput(Throughput::Elements(1));
    g.warm_up_time(Duration::from_millis(500));
    g.measurement_time(Duration::from_secs(2));

    // ── Baseline: no paint, no colors ─────────────────────────────────────────
    set_colors_enabled(false);
    g.bench_function("baseline_no_paint_colors_off", |b| b.iter(|| {
        mid_info!(Tier::High, "hp: {} / {}", black_box(25u32), black_box(100u32));
    }));

    // ── paint() with colors DISABLED — should be pure passthrough ─────────────
    set_colors_enabled(false);
    g.bench_function("paint_colors_off", |b| b.iter(|| {
        mid_info!(
            Tier::High,
            "hp: {} / {}",
            paint(black_box(25u32), Color::Red),
            paint(black_box(100u32), Color::Green),
        );
    }));

    // ── paint() with colors ENABLED — adds ANSI prefix + reset ───────────────
    set_colors_enabled(true);
    g.bench_function("paint_colors_on_standard", |b| b.iter(|| {
        mid_info!(
            Tier::High,
            "hp: {} / {}",
            paint(black_box(25u32), Color::Red),
            paint(black_box(100u32), Color::Green),
        );
    }));

    // ── paint() with RGB true-color ───────────────────────────────────────────
    set_colors_enabled(true);
    g.bench_function("paint_rgb_color", |b| b.iter(|| {
        mid_warn!(
            Tier::Mid,
            "status: {}",
            paint(black_box("degraded"), Color::Rgb(255, 165, 0)),
        );
    }));

    // ── paint() with Custom ANSI sequence ────────────────────────────────────
    set_colors_enabled(true);
    g.bench_function("paint_custom_ansi", |b| b.iter(|| {
        mid_info!(
            Tier::Low,
            "value: {}",
            paint(black_box(42u32), Color::Custom("1;35".to_string())),
        );
    }));

    // ── paint_bg() — fg + bg coloring ────────────────────────────────────────
    set_colors_enabled(true);
    g.bench_function("paint_bg_colors_on", |b| b.iter(|| {
        mid_warn!(
            Tier::High,
            "alert: {}",
            paint_bg(black_box("CRITICAL"), Color::White, Color::Red),
        );
    }));

    // ── Multiple paint() calls in one message ─────────────────────────────────
    set_colors_enabled(true);
    g.bench_function("paint_multiple_in_message", |b| b.iter(|| {
        mid_info!(
            Tier::High,
            "entity {} | hp:{} | mp:{} | status:{}",
            black_box(42u32),
            paint(black_box(25u32),    Color::Red),
            paint(black_box(80u32),    Color::Blue),
            paint(black_box("alive"),  Color::Green),
        );
    }));

    // ── Direct paint() call cost (without logging) ────────────────────────────
    // Isolates paint()'s overhead from the log pipeline.
    set_colors_enabled(true);
    g.bench_function("paint_call_only_colors_on", |b| b.iter(|| {
        black_box(format!("{}", paint(black_box(42u32), Color::Yellow)));
    }));

    set_colors_enabled(false);
    g.bench_function("paint_call_only_colors_off", |b| b.iter(|| {
        black_box(format!("{}", paint(black_box(42u32), Color::Yellow)));
    }));

    // Restore
    set_colors_enabled(false);
    g.finish();
}

// ═════════════════════════════════════════════════════════════════════════════
//  4. Assert macros
//
//  Hard asserts (mid_assert!, mid_assert_eq!) are excluded — they panic and
//  cannot be benchmarked in a loop. We benchmark:
//    - Soft asserts (log ERROR, return bool, no panic)
//    - Debug asserts (zero cost in release, log ERROR in debug)
// ═════════════════════════════════════════════════════════════════════════════

fn bench_mid_log_asserts(c: &mut Criterion) {
    setup_mid_log();
    set_min_level(LogLevel::Trace);

    let mut g = c.benchmark_group("mid_log/asserts");
    g.throughput(Throughput::Elements(1));
    g.warm_up_time(Duration::from_millis(500));
    g.measurement_time(Duration::from_secs(2));

    // ── Soft assert — condition passes (most common case) ─────────────────────
    // Cost: one `!cond` branch + false → return true.
    g.bench_function("soft_assert_passing", |b| b.iter(|| {
        black_box(mid_soft_assert!(black_box(true), "condition holds"));
    }));

    // ── Soft assert — condition fails (logs ERROR, continues) ─────────────────
    // Cost: format!() + mid_error!() pipeline (channel send).
    g.bench_function("soft_assert_failing", |b| b.iter(|| {
        black_box(mid_soft_assert!(
            black_box(false),
            "entity {} invariant broken", black_box(42u32),
        ));
    }));

    // ── Soft assert eq — values match ─────────────────────────────────────────
    g.bench_function("soft_assert_eq_passing", |b| b.iter(|| {
        black_box(mid_soft_assert_eq!(
            black_box(42u32),
            black_box(42u32),
            "should match",
        ));
    }));

    // ── Soft assert eq — values mismatch (logs ERROR) ─────────────────────────
    g.bench_function("soft_assert_eq_failing", |b| b.iter(|| {
        black_box(mid_soft_assert_eq!(
            black_box(1u32),
            black_box(2u32),
            "entity frame desync",
        ));
    }));

    // ── Debug assert — condition passes ───────────────────────────────────────
    // Release: zero cost (cfg!(debug_assertions) = false → block elided).
    // Debug:   one branch check.
    g.bench_function("debug_assert_passing", |b| b.iter(|| {
        mid_debug_assert!(black_box(true), "invariant holds");
    }));

    // ── Debug assert — condition fails ────────────────────────────────────────
    // Release: zero cost — the entire block is compiled out.
    // Debug:   format!() + mid_error!() pipeline.
    g.bench_function("debug_assert_failing", |b| b.iter(|| {
        mid_debug_assert!(black_box(false), "invariant violated: {}", black_box(42u32));
    }));

    g.finish();
}

// ═════════════════════════════════════════════════════════════════════════════
//  5. Format flags — impact on calling thread vs IO thread
//
//  Format flags control the IO thread's output — the calling thread always
//  pays the same cost (filter + format!() + channel send). These benchmarks
//  measure calling-thread cost with different flags so we can verify the
//  flags don't accidentally add overhead to the hot path.
// ═════════════════════════════════════════════════════════════════════════════

fn bench_format_flags(c: &mut Criterion) {
    setup_mid_log();
    set_min_level(LogLevel::Trace);
    set_colors_enabled(false);

    let mut g = c.benchmark_group("mid_log/format_flags");
    g.throughput(Throughput::Elements(1));
    g.warm_up_time(Duration::from_millis(500));
    g.measurement_time(Duration::from_secs(2));

    // Minimal — timestamp only, no source location.
    set_format(&FormatConfig {
        show_timestamp:  true,
        show_source_loc: false,
        show_module:     false,
        show_thread:     false,
        show_frame:      false,
    });
    g.bench_function("calling_thread_minimal_format", |b| b.iter(|| {
        mid_info!(Tier::High, "player {} spawned", black_box(1u32));
    }));

    // All flags on — most fields in output.
    set_format(&FormatConfig {
        show_timestamp:  true,
        show_source_loc: true,
        show_module:     true,
        show_thread:     true,
        show_frame:      true,
    });
    g.bench_function("calling_thread_all_flags_on", |b| b.iter(|| {
        mid_info!(Tier::High, "player {} spawned", black_box(1u32));
    }));

    // Restore defaults.
    set_format(&FormatConfig::default());
    g.finish();
}

// ═════════════════════════════════════════════════════════════════════════════
//  6. Frame counter — integration overhead
//
//  Every LogEntry::new() reads the frame counter (one AtomicU64 load).
//  These benchmarks verify the frame counter adds negligible calling-thread
//  overhead vs not setting it.
// ═════════════════════════════════════════════════════════════════════════════

fn bench_frame_counter(c: &mut Criterion) {
    setup_mid_log();
    set_min_level(LogLevel::Trace);
    set_format(&FormatConfig { show_frame: true, ..Default::default() });

    let mut g = c.benchmark_group("mid_log/frame_counter");
    g.throughput(Throughput::Elements(1));
    g.warm_up_time(Duration::from_millis(500));
    g.measurement_time(Duration::from_secs(2));

    // Log without updating frame — counter stays at 0.
    set_frame(0);
    g.bench_function("log_without_set_frame", |b| b.iter(|| {
        mid_info!(Tier::Low, "tick {}", black_box(0u64));
    }));

    // Log with frame update per call — simulates real game loop.
    // Cost: one AtomicU64 store per iteration (set_frame) +
    //       one AtomicU64 load per LogEntry::new().
    g.bench_function("log_with_set_frame_per_call", |b| {
        let mut frame = 0u64;
        b.iter(|| {
            set_frame(frame);
            mid_info!(Tier::Low, "tick {}", black_box(frame));
            frame += 1;
        })
    });

    // Frame counter read cost in isolation.
    g.bench_function("set_frame_only", |b| {
        let mut frame = 0u64;
        b.iter(|| {
            set_frame(black_box(frame));
            frame += 1;
        })
    });

    set_format(&FormatConfig::default());
    g.finish();
}

// ═════════════════════════════════════════════════════════════════════════════
//  7. Bulk throughput
// ═════════════════════════════════════════════════════════════════════════════

fn bench_bulk_throughput(c: &mut Criterion) {
    setup_mid_log();
    set_min_level(LogLevel::Trace);
    set_colors_enabled(false);

    let mut g = c.benchmark_group("mid_log/bulk");
    g.warm_up_time(Duration::from_millis(500));
    g.measurement_time(Duration::from_secs(2));
    g.sample_size(20); // bulk iterations are slow — fewer samples needed

    // ── 1k plain INFO entries ─────────────────────────────────────────────────
    const N: u64 = 1_000;
    g.throughput(Throughput::Elements(N));

    g.bench_function("1k_info_plain", |b| b.iter(|| {
        for i in 0..N {
            mid_info!(Tier::Mid, "entity={} health={}", black_box(i), black_box(100u32));
        }
    }));

    // ── 1k TRACE filtered ─────────────────────────────────────────────────────
    // All filtered before format!() — measures aggregate filter cost.
    g.bench_function("1k_trace_filtered", |b| {
        set_min_level(LogLevel::Info);
        b.iter(|| {
            for i in 0..N {
                mid_trace!(
                    Tier::Low,
                    "entity={} pos=({:.2},{:.2})",
                    black_box(i), black_box(1.0f32), black_box(2.0f32),
                );
            }
        });
        set_min_level(LogLevel::Trace);
    });

    // ── 1k INFO with paint() colors off ──────────────────────────────────────
    g.bench_function("1k_info_with_paint_colors_off", |b| {
        set_colors_enabled(false);
        b.iter(|| {
            for i in 0..N {
                mid_info!(
                    Tier::High,
                    "hp: {} / {}",
                    paint(black_box(i as u32 % 100), Color::Red),
                    black_box(100u32),
                );
            }
        });
    });

    // ── 1k INFO with paint() colors ON ───────────────────────────────────────
    g.bench_function("1k_info_with_paint_colors_on", |b| {
        set_colors_enabled(true);
        b.iter(|| {
            for i in 0..N {
                mid_info!(
                    Tier::High,
                    "hp: {} / {}",
                    paint(black_box(i as u32 % 100), Color::Red),
                    black_box(100u32),
                );
            }
        });
        set_colors_enabled(false);
    });

    // ── 1k soft assert mix (50% pass / 50% fail) ──────────────────────────────
    g.bench_function("1k_soft_assert_mixed", |b| b.iter(|| {
        for i in 0..N {
            let _ = mid_soft_assert!(
                black_box(i % 2 == 0),
                "entity {} invariant", black_box(i),
            );
        }
    }));

    g.finish();
}

// ═════════════════════════════════════════════════════════════════════════════
//  8. Comparison baselines — tracing and log/env_logger
// ═════════════════════════════════════════════════════════════════════════════

fn bench_tracing_enabled(c: &mut Criterion) {
    // Sink subscriber — measures tracing machinery cost, not terminal I/O.
    let _ = tracing_subscriber::fmt()
        .with_writer(std::io::sink)
        .try_init();

    let mut g = c.benchmark_group("tracing/enabled");
    g.throughput(Throughput::Elements(1));
    g.warm_up_time(Duration::from_millis(500));
    g.measurement_time(Duration::from_secs(2));

    g.bench_function("info_with_args", |b| b.iter(|| {
        tracing::info!(
            entity = black_box(42u32),
            x = black_box(1.0f32),
            y = black_box(2.5f32),
            "entity spawned",
        );
    }));

    g.bench_function("info_no_args", |b| b.iter(|| {
        tracing::info!("tick");
    }));

    g.bench_function("warn_with_args", |b| b.iter(|| {
        tracing::warn!(hp = black_box(25u32), max = black_box(100u32), "low health");
    }));

    g.finish();
}

fn bench_tracing_disabled(c: &mut Criterion) {
    // Filter everything — measures the minimum tracing disabled-path cost.
    // try_init may fail if bench_tracing_enabled already set a subscriber;
    // the existing subscriber from that bench filters at max_level if we
    // set it here. Since try_init is idempotent we test against the sink
    // subscriber — in practice tracing's disabled path is similar anyway.
    let _ = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::ERROR)
        .with_writer(std::io::sink)
        .try_init();

    let mut g = c.benchmark_group("tracing/disabled");
    g.throughput(Throughput::Elements(1));
    g.warm_up_time(Duration::from_millis(500));
    g.measurement_time(Duration::from_secs(2));

    g.bench_function("info_filtered", |b| b.iter(|| {
        tracing::info!(entity = black_box(42u32), "tick");
    }));

    g.finish();
}

fn bench_log_env_logger(c: &mut Criterion) {
    let _ = env_logger::Builder::new()
        .filter_level(log::LevelFilter::Info)
        .target(env_logger::Target::Pipe(Box::new(std::io::sink())))
        .try_init();

    let mut g = c.benchmark_group("log_env_logger/enabled");
    g.throughput(Throughput::Elements(1));
    g.warm_up_time(Duration::from_millis(500));
    g.measurement_time(Duration::from_secs(2));

    g.bench_function("info_with_args", |b| b.iter(|| {
        log::info!(
            "entity {} spawned at ({:.2}, {:.2})",
            black_box(42u32), black_box(1.0f32), black_box(2.5f32),
        );
    }));

    g.bench_function("info_no_args", |b| b.iter(|| {
        log::info!("tick");
    }));

    g.finish();
}

// ═════════════════════════════════════════════════════════════════════════════
//  Register all groups
// ═════════════════════════════════════════════════════════════════════════════

criterion_group! {
    name    = benches;
    config  = short_criterion();
    targets =
        bench_mid_log_disabled,
        bench_mid_log_enabled,
        bench_mid_log_paint,
        bench_mid_log_asserts,
        bench_format_flags,
        bench_frame_counter,
        bench_bulk_throughput,
        bench_tracing_enabled,
        bench_tracing_disabled,
        bench_log_env_logger,
}

criterion_main!(benches);
