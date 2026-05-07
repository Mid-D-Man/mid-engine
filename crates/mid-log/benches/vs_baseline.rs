// crates/mid-log/benches/vs_baseline.rs

//! mid-log benchmarks — printf vs KV vs external loggers
//!
//! ## Build #12 targets
//!
//! | Path                   | Expected     | Notes                              |
//! |------------------------|--------------|------------------------------------|
//! | mid_log disabled       | ~300 ps      | Single atomic load                 |
//! | mid_log printf enabled | ~650 ns      | format!() + channel send           |
//! | mid_log KV enabled     | ~100–150 ns  | No format!(), Vec alloc + send     |
//! | slog/async enabled     | ~340 ns      | Structured args + channel send     |
//!
//! ## Measurement config
//!
//! 20 samples × 2s measurement = 100ms/sample. Doubles run time vs Build #11
//! (~14 min total) but gives enough data to trust the variance figures.

use std::time::Duration;

use criterion::{
    black_box, criterion_group, criterion_main,
    Criterion, Throughput,
};

use mid_log::{
    mid_debug_assert, mid_info, mid_soft_assert, mid_trace,
    mid_kvinfo, mid_kvtrace,
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
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(2))
        .sample_size(20);    // was 10 — 20 samples × 2s = 100ms/sample, same density

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
    let drain = slog_async::Async::new(slog::Discard)
        .build()
        .fuse();
    Logger::root(drain, o!())
}

struct FastLogDiscardAppender;

impl fast_log::appender::LogAppender for FastLogDiscardAppender {
    fn do_logs(&mut self, _records: &[fast_log::appender::FastLogRecord]) {}
}

fn setup_fast_log() {
    fast_log::init(
        fast_log::Config::new()
            .level(log::LevelFilter::Trace)
            .chan_len(Some(100_000))
            .custom(FastLogDiscardAppender)
    ).ok();
}

// ═══════════════════════════════════════════════════════════════════════════
//  Group 1: hot_path — printf API vs external loggers
// ═══════════════════════════════════════════════════════════════════════════

fn bench_hot_path(c: &mut Criterion) {
    setup_mid_log();
    let slog_logger = setup_slog();

    let mut g = c.benchmark_group("hot_path");
    g.throughput(Throughput::Elements(1));

    // mid-log: disabled
    set_min_level(LogLevel::Fatal);
    g.bench_function("mid_log/disabled", |b| b.iter(|| {
        mid_trace!(
            Tier::Low,
            "entity {} pos ({:.2}, {:.2})",
            black_box(42u32), black_box(1.0f32), black_box(2.5f32),
        );
    }));

    // mid-log: printf enabled
    set_min_level(LogLevel::Trace);
    g.bench_function("mid_log/enabled", |b| b.iter(|| {
        mid_info!(
            Tier::High,
            "player {} spawned at ({:.2}, {:.2})",
            black_box(1u32), black_box(1.0f32), black_box(2.0f32),
        );
    }));

    // slog: disabled
    g.bench_function("slog/disabled", |b| b.iter(|| {
        slog::trace!(slog_logger, "entity pos";
            "entity" => black_box(42u32),
            "x" => black_box(1.0f32),
            "y" => black_box(2.5f32),
        );
    }));

    // slog: enabled
    g.bench_function("slog/enabled", |b| b.iter(|| {
        slog_info!(slog_logger, "player spawned";
            "id" => black_box(1u32),
            "x"  => black_box(1.0f32),
            "y"  => black_box(2.0f32),
        );
    }));

    // fast_log: enabled
    setup_fast_log();
    g.bench_function("fast_log/enabled", |b| b.iter(|| {
        log::info!(
            "player {} spawned at ({:.2}, {:.2})",
            black_box(1u32), black_box(1.0f32), black_box(2.0f32),
        );
    }));

    // tracing: enabled
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

    // env_logger: enabled
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

// ═══════════════════════════════════════════════════════════════════════════
//  Group 2: kv_vs_printf — the core question
//
//  Both encode the same logical data: a player spawn with id, x, y.
//  Printf calls format!() with floats. KV sends raw typed scalars.
//  Expected: KV should be ~4–6× faster for this pattern.
// ═══════════════════════════════════════════════════════════════════════════

fn bench_kv(c: &mut Criterion) {
    setup_mid_log();
    set_min_level(LogLevel::Trace);

    let mut g = c.benchmark_group("mid_log/kv_vs_printf");
    g.throughput(Throughput::Elements(1));

    // Printf — same message, float args
    g.bench_function("printf/enabled", |b| b.iter(|| {
        mid_info!(
            Tier::High,
            "player {} spawned at ({:.2}, {:.2})",
            black_box(1u32), black_box(1.0f32), black_box(2.0f32),
        );
    }));

    // KV — same data, no format!()
    g.bench_function("kv/enabled", |b| b.iter(|| {
        mid_kvinfo!(
            Tier::High,
            "player spawned";
            "id" => black_box(1u32),
            "x"  => black_box(1.0f32),
            "y"  => black_box(2.0f32),
        );
    }));

    // KV — static message only, zero allocation
    g.bench_function("kv/static_msg_only", |b| b.iter(|| {
        mid_kvinfo!(Tier::Mid, "frame tick complete");
    }));

    // KV — disabled path
    set_min_level(LogLevel::Fatal);
    g.bench_function("kv/disabled", |b| b.iter(|| {
        mid_kvtrace!(
            Tier::Low,
            "entity pos";
            "id" => black_box(42u32),
            "x"  => black_box(1.0f32),
            "y"  => black_box(2.5f32),
        );
    }));
    set_min_level(LogLevel::Trace);

    g.finish();
}

// ═══════════════════════════════════════════════════════════════════════════
//  Group 3: paint
// ═══════════════════════════════════════════════════════════════════════════

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

// ═══════════════════════════════════════════════════════════════════════════
//  Group 4: asserts
// ═══════════════════════════════════════════════════════════════════════════

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

// ═══════════════════════════════════════════════════════════════════════════
//  Group 5: bulk
// ═══════════════════════════════════════════════════════════════════════════

fn bench_bulk(c: &mut Criterion) {
    setup_mid_log();

    let mut g = c.benchmark_group("mid_log/bulk");
    g.sample_size(10); // keep bulk at 10 — it's slow by design

    const N: u64 = 1_000;
    g.throughput(Throughput::Elements(N));

    // Printf bulk
    set_min_level(LogLevel::Trace);
    g.bench_function("1k_printf_enabled", |b| b.iter(|| {
        for i in 0..N {
            mid_info!(Tier::Mid, "entity={} health={}", black_box(i), black_box(100u32));
        }
    }));

    // KV bulk — expect ~4× faster
    g.bench_function("1k_kv_enabled", |b| b.iter(|| {
        for i in 0..N {
            mid_kvinfo!(Tier::Mid, "entity update";
                "id"     => black_box(i),
                "health" => black_box(100u32),
            );
        }
    }));

    // Filtered bulk — both should be ~1 ns/entry
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

// ═══════════════════════════════════════════════════════════════════════════

criterion_group! {
    name    = benches;
    config  = make_criterion();
    targets =
        bench_hot_path,
        bench_kv,
        bench_paint,
        bench_asserts,
        bench_bulk,
}
criterion_main!(benches);
