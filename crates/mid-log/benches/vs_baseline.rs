// crates/mid-log/benches/vs_baseline.rs

//! mid-log vs tracing vs log/env_logger — throughput comparison.
//!
//! ## Why this file is lean
//!
//! Criterion spends ~85 seconds per benchmark function in CI when accounting
//! for warmup, sample analysis, and HTML generation. The previous 39-function
//! version hit the 20-minute runner limit.
//!
//! This file has 10 benchmark functions. With `without_plots()` and tight
//! measurement settings the suite completes in ~5–8 minutes.
//!
//! ## What each group answers
//!
//! | Group           | Question                                               |
//! |-----------------|--------------------------------------------------------|
//! | `hot_path`      | How does mid-log compare to tracing and env_logger?    |
//! | `mid_log/paint` | What does inline `paint()` add to the calling thread?  |
//! | `mid_log/asserts`| What do soft/debug asserts cost?                      |
//! | `mid_log/bulk`  | What is sustained throughput at 1k entries/burst?      |
//!
//! ## Running locally with full detail
//!
//!   cargo bench --bench vs_baseline -p mid-log
//!
//! HTML report lives at: target/criterion/report/index.html
//! (only generated when run locally — `without_plots()` is skipped via env var)

use std::time::Duration;

use criterion::{
    black_box, criterion_group, criterion_main,
    Criterion, Throughput,
};

use mid_log::{
    mid_debug_assert, mid_info, mid_soft_assert, mid_trace,
    color::{Color, paint, set_colors_enabled},
    filter::set_min_level,
    format::{set_format, FormatConfig},
    frame::set_frame,
    level::{LogLevel, Tier},
    logger::{InitConfig, MidLogger},
    ratelimit::{set_rate_limit_config, RateLimitConfig},
};

// ── Criterion configuration ───────────────────────────────────────────────────

fn make_criterion() -> Criterion {
    let c = Criterion::default()
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(1))
        .sample_size(10);

    // Skip HTML generation in CI — this alone saves 3–5 minutes per run.
    // Set CRITERION_PLOTS=1 locally to re-enable.
    if std::env::var("CRITERION_PLOTS").is_err() {
        c.without_plots()
    } else {
        c
    }
}

// ── One-time logger setup ─────────────────────────────────────────────────────

fn setup() {
    MidLogger::init_full(InitConfig {
        min_level:    LogLevel::Trace,
        format:       FormatConfig::default(),
        color_scheme: mid_log::color::ColorScheme::default(),
        log_file:     None,
    });
    // Disable rate limiting — it would skew throughput numbers.
    set_rate_limit_config(RateLimitConfig { enabled: false, ..Default::default() });
    set_frame(0);
    set_colors_enabled(false);
}

// ═════════════════════════════════════════════════════════════════════════════
//  Group 1: hot_path — the core comparison
//
//  All four implementations doing the same logical operation so the
//  numbers are directly comparable.
// ═════════════════════════════════════════════════════════════════════════════

fn bench_hot_path(c: &mut Criterion) {
    setup();

    let mut g = c.benchmark_group("hot_path");
    g.throughput(Throughput::Elements(1));

    // ── mid-log: disabled (filtered) path ─────────────────────────────────────
    // One AtomicU8 load + comparison branch. format!() never runs.
    // This is the production cost when level is below min_level.
    set_min_level(LogLevel::Fatal);
    g.bench_function("mid_log/disabled", |b| b.iter(|| {
        mid_trace!(
            Tier::Low,
            "entity {} pos ({:.2}, {:.2})",
            black_box(42u32), black_box(1.0f32), black_box(2.5f32),
        );
    }));

    // ── mid-log: enabled path ─────────────────────────────────────────────────
    // filter check + format!() + LogEntry::new() + channel send.
    set_min_level(LogLevel::Trace);
    g.bench_function("mid_log/enabled", |b| b.iter(|| {
        mid_info!(
            Tier::High,
            "player {} spawned at ({:.2}, {:.2})",
            black_box(1u32), black_box(1.0f32), black_box(2.0f32),
        );
    }));

    // ── tracing: enabled (sink subscriber, no I/O cost) ───────────────────────
    let _ = tracing_subscriber::fmt()
        .with_writer(std::io::sink)
        .try_init();

    g.bench_function("tracing/enabled", |b| b.iter(|| {
        tracing::info!(
            entity = black_box(42u32),
            x = black_box(1.0f32),
            y = black_box(2.5f32),
            "entity spawned",
        );
    }));

    // ── log/env_logger: enabled (sink, no I/O cost) ───────────────────────────
    let _ = env_logger::Builder::new()
        .filter_level(log::LevelFilter::Info)
        .target(env_logger::Target::Pipe(Box::new(std::io::sink())))
        .try_init();

    g.bench_function("log_env_logger/enabled", |b| b.iter(|| {
        log::info!(
            "player {} spawned at ({:.2}, {:.2})",
            black_box(1u32), black_box(1.0f32), black_box(2.0f32),
        );
    }));

    g.finish();
}

// ═════════════════════════════════════════════════════════════════════════════
//  Group 2: mid_log/paint — inline color overhead
//
//  paint() always allocates a String for the text on the calling thread.
//  The question: how much does it add vs a plain format!() argument?
// ═════════════════════════════════════════════════════════════════════════════

fn bench_paint(c: &mut Criterion) {
    setup();
    set_min_level(LogLevel::Trace);

    let mut g = c.benchmark_group("mid_log/paint");
    g.throughput(Throughput::Elements(1));

    // ── Baseline: same message, no paint() ───────────────────────────────────
    set_colors_enabled(false);
    g.bench_function("baseline_no_paint", |b| b.iter(|| {
        mid_info!(
            Tier::High,
            "hp: {} / {}",
            black_box(25u32), black_box(100u32),
        );
    }));

    // ── paint() with colors OFF — passthrough (only to_string() cost) ─────────
    set_colors_enabled(false);
    g.bench_function("paint_colors_off", |b| b.iter(|| {
        mid_info!(
            Tier::High,
            "hp: {} / {}",
            paint(black_box(25u32), Color::Red),
            paint(black_box(100u32), Color::Green),
        );
    }));

    // ── paint() with colors ON — ANSI prefix + reset added ────────────────────
    set_colors_enabled(true);
    g.bench_function("paint_colors_on", |b| b.iter(|| {
        mid_info!(
            Tier::High,
            "hp: {} / {}",
            paint(black_box(25u32), Color::Red),
            paint(black_box(100u32), Color::Green),
        );
    }));

    set_colors_enabled(false);
    g.finish();
}

// ═════════════════════════════════════════════════════════════════════════════
//  Group 3: mid_log/asserts
//
//  Soft assert: returns bool, never panics.
//  Debug assert: zero cost in release (block compiled out via #[cfg]).
// ═════════════════════════════════════════════════════════════════════════════

fn bench_asserts(c: &mut Criterion) {
    setup();
    set_min_level(LogLevel::Trace);

    let mut g = c.benchmark_group("mid_log/asserts");
    g.throughput(Throughput::Elements(1));

    // ── soft_assert: condition passes — one branch, returns true, no log ──────
    g.bench_function("soft_assert_passing", |b| b.iter(|| {
        black_box(mid_soft_assert!(black_box(true), "condition holds"));
    }));

    // ── soft_assert: condition fails — formats + logs ERROR + returns false ────
    g.bench_function("soft_assert_failing", |b| b.iter(|| {
        black_box(mid_soft_assert!(
            black_box(false),
            "entity {} invariant", black_box(42u32),
        ));
    }));

    // ── debug_assert: should be 0 ns in release (compiled out) ───────────────
    // In debug: condition check + format!() + mid_error!() on failure.
    // In release: the #[cfg(debug_assertions)] block disappears entirely.
    g.bench_function("debug_assert_failing", |b| b.iter(|| {
        mid_debug_assert!(black_box(false), "invariant: {}", black_box(99u32));
    }));

    g.finish();
}

// ═════════════════════════════════════════════════════════════════════════════
//  Group 4: mid_log/bulk — sustained throughput
// ═════════════════════════════════════════════════════════════════════════════

fn bench_bulk(c: &mut Criterion) {
    setup();

    let mut g = c.benchmark_group("mid_log/bulk");
    g.sample_size(10); // bulk iterations are inherently slower

    const N: u64 = 1_000;
    g.throughput(Throughput::Elements(N));

    // ── 1k enabled INFO entries ───────────────────────────────────────────────
    set_min_level(LogLevel::Trace);
    set_colors_enabled(false);
    g.bench_function("1k_info_enabled", |b| b.iter(|| {
        for i in 0..N {
            mid_info!(
                Tier::Mid,
                "entity={} health={}",
                black_box(i), black_box(100u32),
            );
        }
    }));

    // ── 1k TRACE filtered — aggregate filter cost ─────────────────────────────
    set_min_level(LogLevel::Info);
    g.bench_function("1k_trace_filtered", |b| b.iter(|| {
        for i in 0..N {
            mid_trace!(
                Tier::Low,
                "entity={} pos=({:.2},{:.2})",
                black_box(i), black_box(1.0f32), black_box(2.0f32),
            );
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
