// crates/mid-log/benches/vs_baseline.rs

//! mid-log vs slog vs fast_log vs tracing vs env_logger
//!
//! ## Architecture comparison
//!
//! | Logger     | Hot path                        | IO thread     |
//! |------------|---------------------------------|---------------|
//! | mid-log    | format!() + channel send        | recv() drain  |
//! | slog       | structured args + channel send  | drain trait   |
//! | fast_log   | format!() + channel send        | recv() drain  |
//! | tracing    | callsite check + subscriber     | in-subscriber |
//! | env_logger | format!() + mutex + write       | none (sync)   |
//!
//! ## What to look for
//!
//! - Disabled path: should be ~1ns for all loggers (filter check only)
//! - Enabled path: dominated by format!() cost (~200–800ns)
//! - slog with async: closest architectural peer to mid-log
//! - fast_log: same pattern as mid-log, useful sanity check
//!
//! ## CI note
//!
//! without_plots() is active unless CRITERION_PLOTS=1 is set.
//! 1s measurement, 10 samples. Suite completes in ~7 minutes.

use std::time::Duration;

use criterion::{
    black_box, criterion_group, criterion_main,
    Criterion, Throughput,
};

use mid_log::{
    mid_debug_assert, mid_info, mid_soft_assert, mid_trace,
    color::{Color, paint, set_colors_enabled},
    filter::set_min_level,
    format::FormatConfig,
    frame::set_frame,
    level::{LogLevel, Tier},
    logger::{InitConfig, MidLogger},
    ratelimit::{set_rate_limit_config, RateLimitConfig},
};

use slog::{o, Logger, Drain, info as slog_info};

// ── Criterion config ──────────────────────────────────────────────────────────

fn make_criterion() -> Criterion {
    let c = Criterion::default()
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(1))
        .sample_size(10);

    if std::env::var("CRITERION_PLOTS").is_err() {
        c.without_plots()
    } else {
        c
    }
}

// ── Setup ─────────────────────────────────────────────────────────────────────

fn setup_mid_log() {
    MidLogger::init_full(InitConfig {
        min_level:    LogLevel::Trace,
        format:       FormatConfig::default(),
        color_scheme: mid_log::color::ColorScheme::default(),
        log_file:     None,
    });
    set_rate_limit_config(RateLimitConfig { enabled: false, ..Default::default() });
    set_frame(0);
    set_colors_enabled(false);
}

fn setup_slog() -> Logger {
    // Async drain — slog's equivalent of mid-log's background IO thread.
    // Discard output to isolate channel-send cost from I/O cost.
    let drain = slog_async::Async::new(
        slog::Discard
    )
    .build()
    .fuse();
    Logger::root(drain, o!())
}

// A custom appender for fast_log that discards all records.
struct FastLogDiscardAppender;

impl fast_log::appender::LogAppender for FastLogDiscardAppender {
    fn do_logs(&mut self, _records: &[fast_log::appender::FastLogRecord]) {}
}

fn setup_fast_log() {
    // fast_log with a custom log receiver that discards output.
    // This measures the hot path (channel send) without I/O.
    fast_log::init(
        fast_log::Config::new()
            .level(log::LevelFilter::Trace)
            .chan_len(Some(100_000))
            .custom(FastLogDiscardAppender)
    ).ok();
}

// ═════════════════════════════════════════════════════════════════════════════
//  Group 1: hot_path — the core comparison across all loggers
// ═════════════════════════════════════════════════════════════════════════════

fn bench_hot_path(c: &mut Criterion) {
    setup_mid_log();
    let slog_logger = setup_slog();

    let mut g = c.benchmark_group("hot_path");
    g.throughput(Throughput::Elements(1));

    // ── mid-log: disabled ─────────────────────────────────────────────────────
    set_min_level(LogLevel::Fatal);
    g.bench_function("mid_log/disabled", |b| b.iter(|| {
        mid_trace!(
            Tier::Low,
            "entity {} pos ({:.2}, {:.2})",
            black_box(42u32), black_box(1.0f32), black_box(2.5f32),
        );
    }));

    // ── mid-log: enabled ──────────────────────────────────────────────────────
    set_min_level(LogLevel::Trace);
    g.bench_function("mid_log/enabled", |b| b.iter(|| {
        mid_info!(
            Tier::High,
            "player {} spawned at ({:.2}, {:.2})",
            black_box(1u32), black_box(1.0f32), black_box(2.0f32),
        );
    }));

    // ── slog: disabled (below drain level) ───────────────────────────────────
    // slog's disabled path is a static level check.
    g.bench_function("slog/disabled", |b| b.iter(|| {
        // slog::trace! — filtered by the async drain's min level.
        // We set drain to Discard at Info level, so trace is filtered.
        slog::trace!(slog_logger, "entity pos";
            "entity" => black_box(42u32),
            "x" => black_box(1.0f32),
            "y" => black_box(2.5f32),
        );
    }));

    // ── slog: enabled (async drain, discards output) ──────────────────────────
    g.bench_function("slog/enabled", |b| b.iter(|| {
        slog_info!(slog_logger, "player spawned";
            "id" => black_box(1u32),
            "x"  => black_box(1.0f32),
            "y"  => black_box(2.0f32),
        );
    }));

    // ── fast_log: enabled ─────────────────────────────────────────────────────
    setup_fast_log();
    g.bench_function("fast_log/enabled", |b| b.iter(|| {
        log::info!(
            "player {} spawned at ({:.2}, {:.2})",
            black_box(1u32), black_box(1.0f32), black_box(2.0f32),
        );
    }));

    // ── tracing: enabled (sink, no I/O) ──────────────────────────────────────
    let _ = tracing_subscriber::fmt()
        .with_writer(std::io::sink)
        .try_init();

    g.bench_function("tracing/enabled", |b| b.iter(|| {
        tracing::info!(
            id = black_box(1u32),
            x  = black_box(1.0f32),
            y  = black_box(2.0f32),
            "player spawned",
        );
    }));

    // ── env_logger: enabled (sync, sink) ─────────────────────────────────────
    let _ = env_logger::Builder::new()
        .filter_level(log::LevelFilter::Info)
        .target(env_logger::Target::Pipe(Box::new(std::io::sink())))
        .try_init();

    g.bench_function("env_logger/enabled", |b| b.iter(|| {
        log::info!(
            "player {} spawned at ({:.2}, {:.2})",
            black_box(1u32), black_box(1.0f32), black_box(2.0f32),
        );
    }));

    g.finish();
}

// ═════════════════════════════════════════════════════════════════════════════
//  Group 2: mid_log/paint
// ═════════════════════════════════════════════════════════════════════════════

fn bench_paint(c: &mut Criterion) {
    setup_mid_log();
    set_min_level(LogLevel::Trace);

    let mut g = c.benchmark_group("mid_log/paint");
    g.throughput(Throughput::Elements(1));

    set_colors_enabled(false);
    g.bench_function("baseline_no_paint", |b| b.iter(|| {
        mid_info!(Tier::High, "hp: {} / {}", black_box(25u32), black_box(100u32));
    }));

    set_colors_enabled(false);
    g.bench_function("paint_colors_off", |b| b.iter(|| {
        mid_info!(Tier::High, "hp: {} / {}",
            paint(black_box(25u32), Color::Red),
            paint(black_box(100u32), Color::Green),
        );
    }));

    set_colors_enabled(true);
    g.bench_function("paint_colors_on", |b| b.iter(|| {
        mid_info!(Tier::High, "hp: {} / {}",
            paint(black_box(25u32), Color::Red),
            paint(black_box(100u32), Color::Green),
        );
    }));

    set_colors_enabled(false);
    g.finish();
}

// ═════════════════════════════════════════════════════════════════════════════
//  Group 3: mid_log/asserts
// ═════════════════════════════════════════════════════════════════════════════

fn bench_asserts(c: &mut Criterion) {
    setup_mid_log();
    set_min_level(LogLevel::Trace);

    let mut g = c.benchmark_group("mid_log/asserts");
    g.throughput(Throughput::Elements(1));

    g.bench_function("soft_assert_passing", |b| b.iter(|| {
        black_box(mid_soft_assert!(black_box(true), "ok"));
    }));

    g.bench_function("soft_assert_failing", |b| b.iter(|| {
        black_box(mid_soft_assert!(black_box(false), "entity {} broken", black_box(42u32)));
    }));

    g.bench_function("debug_assert_failing", |b| b.iter(|| {
        mid_debug_assert!(black_box(false), "invariant: {}", black_box(99u32));
    }));

    g.finish();
}

// ═════════════════════════════════════════════════════════════════════════════
//  Group 4: mid_log/bulk
// ═════════════════════════════════════════════════════════════════════════════

fn bench_bulk(c: &mut Criterion) {
    setup_mid_log();

    let mut g = c.benchmark_group("mid_log/bulk");
    g.sample_size(10);

    const N: u64 = 1_000;
    g.throughput(Throughput::Elements(N));

    set_min_level(LogLevel::Trace);
    g.bench_function("1k_info_enabled", |b| b.iter(|| {
        for i in 0..N {
            mid_info!(Tier::Mid, "entity={} health={}", black_box(i), black_box(100u32));
        }
    }));

    set_min_level(LogLevel::Info);
    g.bench_function("1k_trace_filtered", |b| b.iter(|| {
        for i in 0..N {
            mid_trace!(Tier::Low, "entity={} pos=({:.2},{:.2})",
                black_box(i), black_box(1.0f32), black_box(2.0f32));
        }
    }));

    set_min_level(LogLevel::Trace);
    g.finish();
}

// ═════════════════════════════════════════════════════════════════════════════

criterion_group! {
    name    = benches;
    config  = make_criterion();
    targets =
        bench_hot_path,
        bench_paint,
        bench_asserts,
        bench_bulk,
}
criterion_main!(benches);
