// crates/mid-log/benches/vs_baseline.rs

//! mid-log vs tracing vs log/env_logger — throughput benchmark.
//!
//! Measures the *calling thread* cost only (i.e. what the game loop pays).
//! The IO thread runs concurrently and is not included in the measurement.
//!
//! Run: cargo bench --bench vs_baseline -p mid-log
//! HTML: target/criterion/report/index.html
//!
//! ## What we're measuring
//!
//! - **mid-log (enabled)**  : level check + format!() + channel send
//! - **mid-log (disabled)** : level check only (one atomic load + branch)
//! - **tracing (enabled)**  : tracing::info!() with subscriber configured
//! - **tracing (disabled)** : tracing::info!() with max_level_off
//! - **log/env_logger**     : log::info!() with env_logger at info level
//!
//! ## Expected results
//!
//! | Path                   | Expected cost     |
//! |------------------------|-------------------|
//! | mid-log disabled       | ~1 ns             |
//! | tracing disabled       | ~1–2 ns           |
//! | mid-log enabled        | ~50–150 ns        |
//! | tracing enabled        | ~100–300 ns       |
//! | log/env_logger enabled | ~200–500 ns       |
//!
//! mid-log's enabled path pays for: 1 atomic + format!() + channel send.
//! tracing pays for: callsite check + span overhead + subscriber dispatch.
//! The disabled paths should be comparable (both are an atomic + branch).

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};

use mid_log::{mid_info, mid_trace, level::Tier, filter::set_min_level};

// ── mid-log setup ─────────────────────────────────────────────────────────────

fn setup_mid_log() {
    mid_log::logger::MidLogger::init();
}

// ── Benchmarks ────────────────────────────────────────────────────────────────

fn bench_mid_log_enabled(c: &mut Criterion) {
    setup_mid_log();
    set_min_level(mid_log::level::LogLevel::Trace);

    let mut g = c.benchmark_group("mid_log/enabled");
    g.throughput(Throughput::Elements(1));

    g.bench_function("trace_simple", |b| b.iter(|| {
        mid_trace!(Tier::Low, "entity {} pos ({:.2}, {:.2})", black_box(42u32), black_box(1.0f32), black_box(2.5f32));
    }));

    g.bench_function("info_simple", |b| b.iter(|| {
        mid_info!(Tier::High, "player {} spawned at ({}, {})", black_box(1u32), black_box(1.0f32), black_box(2.0f32));
    }));

    g.bench_function("info_no_args", |b| b.iter(|| {
        mid_info!(Tier::Low, "tick");
    }));

    g.bench_function("info_long_message", |b| b.iter(|| {
        mid_info!(Tier::Mid,
            "entity={} component=Transform pos=({:.4},{:.4},{:.4}) vel=({:.4},{:.4},{:.4}) health={} status=active",
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
    // Disable all levels — only the atomic check runs
    set_min_level(mid_log::level::LogLevel::Fatal);

    let mut g = c.benchmark_group("mid_log/disabled");
    g.throughput(Throughput::Elements(1));

    g.bench_function("trace_filtered", |b| b.iter(|| {
        mid_trace!(Tier::Low, "entity {} pos ({:.2}, {:.2})", black_box(42u32), black_box(1.0f32), black_box(2.5f32));
    }));

    g.bench_function("info_filtered", |b| b.iter(|| {
        mid_info!(Tier::High, "player {} spawned", black_box(1u32));
    }));

    g.finish();
}

fn bench_tracing_enabled(c: &mut Criterion) {
    // Set up a no-op subscriber (formats but discards output) so we measure
    // the tracing machinery cost, not I/O.
    use tracing_subscriber::fmt;
    use tracing_subscriber::prelude::*;
    let _ = tracing_subscriber::registry()
        .with(fmt::layer().with_writer(std::io::sink))
        .try_init();

    let mut g = c.benchmark_group("tracing/enabled");
    g.throughput(Throughput::Elements(1));

    g.bench_function("info_simple", |b| b.iter(|| {
        tracing::info!(
            entity = black_box(42u32),
            x = black_box(1.0f32),
            y = black_box(2.5f32),
            "entity spawned"
        );
    }));

    g.bench_function("info_no_args", |b| b.iter(|| {
        tracing::info!("tick");
    }));

    g.finish();
}

fn bench_tracing_disabled(c: &mut Criterion) {
    // max_level_off: tracing compiles out all callsite checks below OFF.
    // This simulates the fastest possible disabled path.
    // In practice, use the `max_level_off` feature flag in Cargo.toml.
    // Here we just set a subscriber that filters everything.
    use tracing_subscriber::EnvFilter;
    let _ = tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::new("off"))
        .with_writer(std::io::sink)
        .try_init();

    let mut g = c.benchmark_group("tracing/disabled");
    g.throughput(Throughput::Elements(1));

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

    g.bench_function("info_simple", |b| b.iter(|| {
        log::info!("entity {} spawned at ({}, {})", black_box(42u32), black_box(1.0f32), black_box(2.5f32));
    }));

    g.finish();
}

fn bench_bulk_throughput(c: &mut Criterion) {
    setup_mid_log();
    set_min_level(mid_log::level::LogLevel::Trace);

    let mut g = c.benchmark_group("mid_log/bulk");

    const N: u64 = 10_000;
    g.throughput(Throughput::Elements(N));

    g.bench_function("10k_info_burst", |b| b.iter(|| {
        for i in 0..N {
            mid_info!(Tier::Mid, "entity={} health={}", black_box(i), black_box(100u32));
        }
    }));

    g.bench_function("10k_trace_filtered", |b| {
        set_min_level(mid_log::level::LogLevel::Info);
        b.iter(|| {
            for i in 0..N {
                mid_trace!(Tier::Low, "entity={} pos=({:.2},{:.2})", black_box(i), black_box(1.0f32), black_box(2.0f32));
            }
        });
        set_min_level(mid_log::level::LogLevel::Trace);
    });

    g.finish();
}

criterion_group!(
    benches,
    bench_mid_log_enabled,
    bench_mid_log_disabled,
    bench_tracing_enabled,
    bench_tracing_disabled,
    bench_log_env_logger,
    bench_bulk_throughput,
);
criterion_main!(benches);
