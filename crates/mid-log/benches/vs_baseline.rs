// crates/mid-log/benches/vs_baseline.rs

//! mid-log vs tracing vs log/env_logger — throughput benchmark.
//!
//! Measures the *calling thread* cost only (what the game loop pays).
//! The IO thread runs concurrently and is not part of the measurement.
//!
//! Run: cargo bench --bench vs_baseline -p mid-log
//! HTML: target/criterion/report/index.html
//!
//! ## Criterion time budget
//!
//! Default Criterion settings run each benchmark for 5 seconds + 3 seconds
//! warmup. With 12+ functions that's 96+ seconds minimum. This file caps
//! measurement at 2 seconds + 1 second warmup so the full suite completes
//! in ~5 minutes, well within GitHub Actions' resource window.
//!
//! If you need higher-precision numbers locally, remove the
//! `.measurement_time()` / `.warm_up_time()` overrides.
//!
//! ## Expected results (release build)
//!
//! | Path                   | Expected        |
//! |------------------------|-----------------|
//! | mid-log disabled       | ~1–5 ns         |
//! | tracing disabled       | ~1–5 ns         |
//! | mid-log enabled        | ~50–200 ns      |
//! | tracing enabled        | ~100–400 ns     |
//! | log/env_logger enabled | ~200–600 ns     |

use std::time::Duration;

use criterion::{
    black_box, criterion_group, criterion_main,
    Criterion, Throughput,
};

use mid_log::{mid_info, mid_trace, level::Tier, filter::set_min_level};

// ── Criterion configuration ───────────────────────────────────────────────────

/// Build a Criterion instance with shortened measurement times so the
/// full suite fits inside CI resource limits (~5 minutes total).
fn short_criterion() -> Criterion {
    Criterion::default()
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(2))
        .sample_size(50) // default is 100; 50 is enough for stable medians
}

// ── mid-log setup ─────────────────────────────────────────────────────────────

fn setup_mid_log() {
    // init() is idempotent — safe to call multiple times across bench functions.
    mid_log::logger::MidLogger::init();
}

// ── Benchmarks ────────────────────────────────────────────────────────────────

fn bench_mid_log_enabled(c: &mut Criterion) {
    setup_mid_log();
    set_min_level(mid_log::level::LogLevel::Trace);

    let mut g = c.benchmark_group("mid_log/enabled");
    g.throughput(Throughput::Elements(1));
    g.warm_up_time(Duration::from_millis(500));
    g.measurement_time(Duration::from_secs(2));

    g.bench_function("trace_simple", |b| b.iter(|| {
        mid_trace!(
            Tier::Low,
            "entity {} pos ({:.2}, {:.2})",
            black_box(42u32), black_box(1.0f32), black_box(2.5f32),
        );
    }));

    g.bench_function("info_simple", |b| b.iter(|| {
        mid_info!(
            Tier::High,
            "player {} spawned at ({}, {})",
            black_box(1u32), black_box(1.0f32), black_box(2.0f32),
        );
    }));

    g.bench_function("info_no_args", |b| b.iter(|| {
        mid_info!(Tier::Low, "tick");
    }));

    g.bench_function("info_long_message", |b| b.iter(|| {
        mid_info!(
            Tier::Mid,
            "entity={} component=Transform pos=({:.4},{:.4},{:.4}) \
             vel=({:.4},{:.4},{:.4}) health={} status=active",
            black_box(99u32),
            black_box(1.0f32), black_box(2.0f32), black_box(3.0f32),
            black_box(0.1f32), black_box(0.2f32), black_box(0.3f32),
            black_box(100u32),
        );
    }));

    g.finish();
}

fn bench_mid_log_disabled(c: &mut Criterion) {
    setup_mid_log();
    // Disable all levels — only the atomic check runs.
    set_min_level(mid_log::level::LogLevel::Fatal);

    let mut g = c.benchmark_group("mid_log/disabled");
    g.throughput(Throughput::Elements(1));
    g.warm_up_time(Duration::from_millis(500));
    g.measurement_time(Duration::from_secs(2));

    g.bench_function("trace_filtered", |b| b.iter(|| {
        mid_trace!(
            Tier::Low,
            "entity {} pos ({:.2}, {:.2})",
            black_box(42u32), black_box(1.0f32), black_box(2.5f32),
        );
    }));

    g.bench_function("info_filtered", |b| b.iter(|| {
        mid_info!(Tier::High, "player {} spawned", black_box(1u32));
    }));

    g.finish();
}

fn bench_tracing_enabled(c: &mut Criterion) {
    // Set up a no-op subscriber (formats but discards output) to measure
    // tracing machinery cost without I/O. try_init() is idempotent.
    let _ = tracing_subscriber::fmt()
        .with_writer(std::io::sink)
        .try_init();

    let mut g = c.benchmark_group("tracing/enabled");
    g.throughput(Throughput::Elements(1));
    g.warm_up_time(Duration::from_millis(500));
    g.measurement_time(Duration::from_secs(2));

    g.bench_function("info_simple", |b| b.iter(|| {
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

    g.finish();
}

fn bench_tracing_disabled(c: &mut Criterion) {
    // Filter everything — measures the minimum tracing overhead.
    // try_init() may fail if already set from bench_tracing_enabled;
    // that's fine — the subscriber from there already discards everything.
    let _ = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::ERROR) // suppress INFO and below
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
    // Pipe output to sink so we measure logger machinery, not terminal I/O.
    let _ = env_logger::Builder::new()
        .filter_level(log::LevelFilter::Info)
        .target(env_logger::Target::Pipe(Box::new(std::io::sink())))
        .try_init();

    let mut g = c.benchmark_group("log_env_logger/enabled");
    g.throughput(Throughput::Elements(1));
    g.warm_up_time(Duration::from_millis(500));
    g.measurement_time(Duration::from_secs(2));

    g.bench_function("info_simple", |b| b.iter(|| {
        log::info!(
            "entity {} spawned at ({}, {})",
            black_box(42u32), black_box(1.0f32), black_box(2.5f32),
        );
    }));

    g.finish();
}

fn bench_bulk_throughput(c: &mut Criterion) {
    setup_mid_log();
    set_min_level(mid_log::level::LogLevel::Trace);

    let mut g = c.benchmark_group("mid_log/bulk");
    g.warm_up_time(Duration::from_millis(500));
    g.measurement_time(Duration::from_secs(2));
    g.sample_size(20); // bulk benchmarks are slow per iteration; fewer samples

    const N: u64 = 1_000; // reduced from 10k — keeps each iteration <100ms
    g.throughput(Throughput::Elements(N));

    g.bench_function("1k_info_burst", |b| b.iter(|| {
        for i in 0..N {
            mid_info!(
                Tier::Mid,
                "entity={} health={}",
                black_box(i), black_box(100u32),
            );
        }
    }));

    g.bench_function("1k_trace_filtered", |b| {
        set_min_level(mid_log::level::LogLevel::Info); // filter out TRACE
        b.iter(|| {
            for i in 0..N {
                mid_trace!(
                    Tier::Low,
                    "entity={} pos=({:.2},{:.2})",
                    black_box(i), black_box(1.0f32), black_box(2.0f32),
                );
            }
        });
        set_min_level(mid_log::level::LogLevel::Trace); // restore
    });

    g.finish();
}

// ── Register groups ───────────────────────────────────────────────────────────

criterion_group! {
    name    = benches;
    config  = short_criterion();
    targets =
        bench_mid_log_enabled,
        bench_mid_log_disabled,
        bench_tracing_enabled,
        bench_tracing_disabled,
        bench_log_env_logger,
        bench_bulk_throughput,
}
criterion_main!(benches);
