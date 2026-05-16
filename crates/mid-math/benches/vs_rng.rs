// crates/mid-math/benches/vs_rng.rs
//! RNG benchmarks: Xorshift64 vs PCG32, head-to-head.
//!
//! Groups:
//!   rng/throughput      — raw generation: next_u32/u64, f32, f64
//!   rng/range           — range_u32, range_f32 (Lemire vs modulo)
//!   rng/bool_p          — weighted coin flip
//!   rng/advance         — PCG advance() (O(log n) skip)
//!   rng/bulk_1m         — 1 million values throughput comparison

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use mid_math::{Pcg32, Xorshift64};

// ── Raw throughput ────────────────────────────────────────────────────────────

fn bench_throughput(c: &mut Criterion) {
    let mut g = c.benchmark_group("rng/throughput");

    let mut xs = Xorshift64::new(0xDEAD_BEEF_CAFE_1234);
    let mut pg = Pcg32::new(0xDEAD_BEEF_CAFE_1234, 1);

    g.bench_function("xorshift64/next_u64", |b| {
        b.iter(|| black_box(black_box(&mut xs).next_u64()))
    });
    g.bench_function("pcg32/next_u32", |b| {
        b.iter(|| black_box(black_box(&mut pg).next_u32()))
    });
    g.bench_function("pcg32/next_u64", |b| {
        b.iter(|| black_box(black_box(&mut pg).next_u64()))
    });
    g.bench_function("xorshift64/f32", |b| {
        b.iter(|| black_box(black_box(&mut xs).f32()))
    });
    g.bench_function("pcg32/f32", |b| {
        b.iter(|| black_box(black_box(&mut pg).f32()))
    });
    g.bench_function("xorshift64/f64", |b| {
        b.iter(|| black_box(black_box(&mut xs).f64()))
    });
    g.bench_function("pcg32/f64", |b| {
        b.iter(|| black_box(black_box(&mut pg).f64()))
    });
    g.finish();
}

// ── Range functions ───────────────────────────────────────────────────────────

fn bench_range(c: &mut Criterion) {
    let mut g = c.benchmark_group("rng/range");

    let mut xs = Xorshift64::new(12345);
    let mut pg = Pcg32::new(12345, 1);

    g.bench_function("xorshift64/range_u32_0_100", |b| {
        b.iter(|| black_box(black_box(&mut xs).range_u32(0, 100)))
    });
    g.bench_function("pcg32/range_u32_0_100", |b| {
        b.iter(|| black_box(black_box(&mut pg).range_u32(0, 100)))
    });
    g.bench_function("xorshift64/range_u32_power_of_2", |b| {
        b.iter(|| black_box(black_box(&mut xs).range_u32(0, 256)))
    });
    g.bench_function("pcg32/range_u32_power_of_2", |b| {
        b.iter(|| black_box(black_box(&mut pg).range_u32(0, 256)))
    });
    g.bench_function("xorshift64/range_f32", |b| {
        b.iter(|| black_box(black_box(&mut xs).range_f32(-1.0, 1.0)))
    });
    g.bench_function("pcg32/range_f32", |b| {
        b.iter(|| black_box(black_box(&mut pg).range_f32(-1.0, 1.0)))
    });
    g.bench_function("xorshift64/range_f64", |b| {
        b.iter(|| black_box(black_box(&mut xs).range_f64(0.0, 100.0)))
    });
    g.bench_function("pcg32/range_f64", |b| {
        b.iter(|| black_box(black_box(&mut pg).range_f64(0.0, 100.0)))
    });
    g.finish();
}

// ── Bool with probability ─────────────────────────────────────────────────────

fn bench_bool_p(c: &mut Criterion) {
    let mut g = c.benchmark_group("rng/bool_p");

    let mut xs = Xorshift64::new(999);
    let mut pg = Pcg32::new(999, 1);

    for p in [0.1_f32, 0.5, 0.9] {
        g.bench_with_input(
            BenchmarkId::new("xorshift64", p),
            &p,
            |b, &p| b.iter(|| black_box(black_box(&mut xs).bool_p(black_box(p)))),
        );
        g.bench_with_input(
            BenchmarkId::new("pcg32", p),
            &p,
            |b, &p| b.iter(|| black_box(black_box(&mut pg).bool_p(black_box(p)))),
        );
    }
    g.finish();
}

// ── PCG advance (O(log n) skip) ───────────────────────────────────────────────

fn bench_advance(c: &mut Criterion) {
    let mut g = c.benchmark_group("rng/advance");

    for &delta in &[1u64, 1_000, 1_000_000, u64::MAX / 2] {
        g.bench_with_input(
            BenchmarkId::new("pcg32_advance", delta),
            &delta,
            |b, &d| {
                b.iter(|| {
                    let mut pg = Pcg32::new(42, 1);
                    black_box(&mut pg).advance(black_box(d));
                    black_box(pg)
                })
            },
        );
    }
    g.finish();
}

// ── Bulk 1M throughput ────────────────────────────────────────────────────────

fn bench_bulk_1m(c: &mut Criterion) {
    let mut g = c.benchmark_group("rng/bulk_1m");
    g.sample_size(20);

    g.bench_function("xorshift64/1m_u64", |b| {
        b.iter(|| {
            let mut rng = Xorshift64::new(0xABC);
            let mut sum = 0u64;
            for _ in 0..1_000_000 {
                sum = sum.wrapping_add(rng.next_u64());
            }
            black_box(sum)
        })
    });

    g.bench_function("pcg32/1m_u32", |b| {
        b.iter(|| {
            let mut rng = Pcg32::new(0xABC, 1);
            let mut sum = 0u64;
            for _ in 0..1_000_000 {
                sum = sum.wrapping_add(rng.next_u32() as u64);
            }
            black_box(sum)
        })
    });

    g.bench_function("xorshift64/1m_f32", |b| {
        b.iter(|| {
            let mut rng = Xorshift64::new(0xABC);
            let mut sum = 0.0f32;
            for _ in 0..1_000_000 {
                sum += rng.f32();
            }
            black_box(sum)
        })
    });

    g.bench_function("pcg32/1m_f32", |b| {
        b.iter(|| {
            let mut rng = Pcg32::new(0xABC, 1);
            let mut sum = 0.0f32;
            for _ in 0..1_000_000 {
                sum += rng.f32();
            }
            black_box(sum)
        })
    });

    g.bench_function("xorshift64/1m_range_u32_0_1000", |b| {
        b.iter(|| {
            let mut rng = Xorshift64::new(0xABC);
            let mut sum = 0u64;
            for _ in 0..1_000_000 {
                sum = sum.wrapping_add(rng.range_u32(0, 1000) as u64);
            }
            black_box(sum)
        })
    });

    g.bench_function("pcg32/1m_range_u32_0_1000", |b| {
        b.iter(|| {
            let mut rng = Pcg32::new(0xABC, 1);
            let mut sum = 0u64;
            for _ in 0..1_000_000 {
                sum = sum.wrapping_add(rng.range_u32(0, 1000) as u64);
            }
            black_box(sum)
        })
    });

    // Independent streams — PCG exclusive
    g.bench_function("pcg32/8_independent_streams_125k_each", |b| {
        b.iter(|| {
            let mut rngs: Vec<Pcg32> = (0..8).map(|seq| Pcg32::new(0xABC, seq)).collect();
            let mut sum = 0u64;
            for _ in 0..125_000 {
                for rng in &mut rngs {
                    sum = sum.wrapping_add(rng.next_u32() as u64);
                }
            }
            black_box(sum)
        })
    });

    g.finish();
}

criterion_group!(
    benches,
    bench_throughput,
    bench_range,
    bench_bool_p,
    bench_advance,
    bench_bulk_1m,
);
criterion_main!(benches);
