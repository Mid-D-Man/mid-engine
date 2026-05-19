// crates/mid-math/benches/noise.rs
//! Benchmarks for coherent noise generators.
//!
//! Purpose: establish throughput baselines for all noise types
//! so Phase 2 SIMD vectorisation has concrete before/after numbers.
//!
//! Groups:
//!   noise_scalar_2d     — single-sample 2D throughput per type
//!   noise_scalar_3d     — single-sample 3D throughput per type
//!   noise_scalar_4d     — single-sample 4D throughput per type
//!   fbm_2d              — fBm octave cost at 1/2/4/6/8 octaves
//!   fbm_3d              — same for 3D
//!   domain_warp_2d      — single vs double warp
//!   worley_modes_2d     — F1/F2/F2-F1 and Euclidean/Manhattan/Chebyshev
//!   noise_batch_100k    — sustained throughput: 100k samples, all types
//!
//! Run: cargo bench --bench noise -p mid-math
//! HTML: target/criterion/report/index.html

use criterion::{
    black_box, criterion_group, criterion_main,
    BatchSize, Criterion, Throughput,
};

use mid_math::noise::{
    DomainWarp, Fbm, NoiseSource2d, NoiseSource3d, NoiseSource4d,
    Perlin, Simplex, Value, Worley,
    worley::{DistanceMode, DistanceMetric},
};

// ─────────────────────────────────────────────────────────────────────────────
// Group 1: single-sample 2D
// ─────────────────────────────────────────────────────────────────────────────

fn bench_noise_scalar_2d(c: &mut Criterion) {
    let mut g = c.benchmark_group("noise_scalar_2d");

    let perlin  = Perlin::new();
    let simplex = Simplex::new();
    let value   = Value::new();
    let worley  = Worley::new().with_mode(DistanceMode::F1);

    g.bench_function("Perlin/2d",  |b| b.iter(|| perlin.sample_2d(black_box(1.23), black_box(4.56))));
    g.bench_function("Simplex/2d", |b| b.iter(|| simplex.sample_2d(black_box(1.23), black_box(4.56))));
    g.bench_function("Value/2d",   |b| b.iter(|| value.sample_2d(black_box(1.23), black_box(4.56))));
    g.bench_function("Worley_F1/2d", |b| b.iter(|| worley.sample_2d(black_box(1.23), black_box(4.56))));

    g.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// Group 2: single-sample 3D
// ─────────────────────────────────────────────────────────────────────────────

fn bench_noise_scalar_3d(c: &mut Criterion) {
    let mut g = c.benchmark_group("noise_scalar_3d");

    let perlin  = Perlin::new();
    let simplex = Simplex::new();
    let value   = Value::new();
    let worley  = Worley::new().with_mode(DistanceMode::F1);

    g.bench_function("Perlin/3d",    |b| b.iter(|| perlin.sample_3d(black_box(1.0), black_box(2.0), black_box(3.0))));
    g.bench_function("Simplex/3d",   |b| b.iter(|| simplex.sample_3d(black_box(1.0), black_box(2.0), black_box(3.0))));
    g.bench_function("Value/3d",     |b| b.iter(|| value.sample_3d(black_box(1.0), black_box(2.0), black_box(3.0))));
    g.bench_function("Worley_F1/3d", |b| b.iter(|| worley.sample_3d(black_box(1.0), black_box(2.0), black_box(3.0))));

    g.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// Group 3: single-sample 4D
// ─────────────────────────────────────────────────────────────────────────────

fn bench_noise_scalar_4d(c: &mut Criterion) {
    let mut g = c.benchmark_group("noise_scalar_4d");

    let perlin  = Perlin::new();
    let simplex = Simplex::new();
    let value   = Value::new();

    g.bench_function("Perlin/4d",  |b| b.iter(|| perlin.sample_4d(black_box(1.0), black_box(2.0), black_box(3.0), black_box(4.0))));
    g.bench_function("Simplex/4d", |b| b.iter(|| simplex.sample_4d(black_box(1.0), black_box(2.0), black_box(3.0), black_box(4.0))));
    g.bench_function("Value/4d",   |b| b.iter(|| value.sample_4d(black_box(1.0), black_box(2.0), black_box(3.0), black_box(4.0))));

    g.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// Group 4: fBm — octave cost (2D)
// ─────────────────────────────────────────────────────────────────────────────

fn bench_fbm_2d(c: &mut Criterion) {
    let mut g = c.benchmark_group("fbm_2d");

    for oct in [1u32, 2, 4, 6, 8] {
        let fbm = Fbm::new(Simplex::new())
            .octaves(oct)
            .lacunarity(2.0)
            .gain(0.5)
            .frequency(1.0);

        g.bench_function(format!("Simplex_{}_octaves", oct), |b| {
            b.iter(|| fbm.sample_2d(black_box(1.23), black_box(4.56)))
        });
    }

    // Perlin base at the common 6-octave setting for comparison.
    let perlin_fbm = Fbm::new(Perlin::new())
        .octaves(6).lacunarity(2.0).gain(0.5).frequency(1.0);
    g.bench_function("Perlin_6_octaves", |b| {
        b.iter(|| perlin_fbm.sample_2d(black_box(1.23), black_box(4.56)))
    });

    g.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// Group 5: fBm — octave cost (3D)
// ─────────────────────────────────────────────────────────────────────────────

fn bench_fbm_3d(c: &mut Criterion) {
    let mut g = c.benchmark_group("fbm_3d");

    for oct in [1u32, 2, 4, 6, 8] {
        let fbm = Fbm::new(Simplex::new())
            .octaves(oct).lacunarity(2.0).gain(0.5).frequency(1.0);
        g.bench_function(format!("Simplex_{}_octaves", oct), |b| {
            b.iter(|| fbm.sample_3d(black_box(1.0), black_box(2.0), black_box(3.0)))
        });
    }

    g.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// Group 6: Domain warp
// ─────────────────────────────────────────────────────────────────────────────

fn bench_domain_warp_2d(c: &mut Criterion) {
    let mut g = c.benchmark_group("domain_warp_2d");

    let base_fbm = || {
        Fbm::new(Simplex::new())
            .octaves(4).lacunarity(2.0).gain(0.5).frequency(1.0)
    };

    let single = DomainWarp::new(Simplex::new())
        .with_fbm(base_fbm()).warp_scale(1.0).double_warp(false);
    let double = DomainWarp::new(Simplex::new())
        .with_fbm(base_fbm()).warp_scale(1.0).double_warp(true);

    g.bench_function("single_warp", |b| {
        b.iter(|| single.sample_2d(black_box(1.23), black_box(4.56)))
    });
    g.bench_function("double_warp", |b| {
        b.iter(|| double.sample_2d(black_box(1.23), black_box(4.56)))
    });

    g.finish();
}

fn bench_domain_warp_3d(c: &mut Criterion) {
    let mut g = c.benchmark_group("domain_warp_3d");

    let base_fbm = || {
        Fbm::new(Simplex::new())
            .octaves(4).lacunarity(2.0).gain(0.5).frequency(1.0)
    };

    let single = DomainWarp::new(Simplex::new())
        .with_fbm(base_fbm()).warp_scale(1.0).double_warp(false);
    let double = DomainWarp::new(Simplex::new())
        .with_fbm(base_fbm()).warp_scale(1.0).double_warp(true);

    g.bench_function("single_warp", |b| {
        b.iter(|| single.sample_3d(black_box(1.0), black_box(2.0), black_box(3.0)))
    });
    g.bench_function("double_warp", |b| {
        b.iter(|| double.sample_3d(black_box(1.0), black_box(2.0), black_box(3.0)))
    });

    g.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// Group 7: Worley distance modes and metrics
// ─────────────────────────────────────────────────────────────────────────────

fn bench_worley_modes_2d(c: &mut Criterion) {
    let mut g = c.benchmark_group("worley_modes_2d");

    // Distance modes (Euclidean metric).
    for mode in [DistanceMode::F1, DistanceMode::F2, DistanceMode::F2MinusF1, DistanceMode::F1PlusF2] {
        let name = format!("{:?}", mode);
        let w = Worley::new().with_mode(mode).with_metric(DistanceMetric::Euclidean);
        g.bench_function(format!("mode_{}", name), |b| {
            b.iter(|| w.sample_2d(black_box(1.23), black_box(4.56)))
        });
    }

    // Distance metrics (F1 mode).
    for metric in [
        DistanceMetric::Euclidean,
        DistanceMetric::Manhattan,
        DistanceMetric::Chebyshev,
        DistanceMetric::Minkowski,
    ] {
        let name = format!("{:?}", metric);
        let w = Worley::new().with_mode(DistanceMode::F1).with_metric(metric);
        g.bench_function(format!("metric_{}", name), |b| {
            b.iter(|| w.sample_2d(black_box(1.23), black_box(4.56)))
        });
    }

    g.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// Group 8: 100k batch — sustained throughput (Phase 2 regression target)
// ─────────────────────────────────────────────────────────────────────────────

fn bench_noise_batch_100k(c: &mut Criterion) {
    const N: usize = 100_000;
    let mut g = c.benchmark_group("noise_batch_100k");
    g.throughput(Throughput::Elements(N as u64));

    // Pre-generate input coordinates once — vary across the unit grid.
    let coords: Vec<(f32, f32)> = (0..N)
        .map(|i| {
            let x = (i % 1000) as f32 * 0.01;
            let y = (i / 1000) as f32 * 0.1;
            (x, y)
        })
        .collect();

    // Perlin 2D batch.
    {
        let perlin = Perlin::new();
        let coords = coords.clone();
        g.bench_function("Perlin_2d", |b| {
            b.iter_batched(
                || coords.clone(),
                |pts| {
                    let mut sum = 0.0f32;
                    for (x, y) in pts { sum += perlin.sample_2d(black_box(x), black_box(y)); }
                    black_box(sum)
                },
                BatchSize::LargeInput,
            )
        });
    }

    // Simplex 2D batch.
    {
        let simplex = Simplex::new();
        let coords = coords.clone();
        g.bench_function("Simplex_2d", |b| {
            b.iter_batched(
                || coords.clone(),
                |pts| {
                    let mut sum = 0.0f32;
                    for (x, y) in pts { sum += simplex.sample_2d(black_box(x), black_box(y)); }
                    black_box(sum)
                },
                BatchSize::LargeInput,
            )
        });
    }

    // Value 2D batch.
    {
        let value = Value::new();
        let coords = coords.clone();
        g.bench_function("Value_2d", |b| {
            b.iter_batched(
                || coords.clone(),
                |pts| {
                    let mut sum = 0.0f32;
                    for (x, y) in pts { sum += value.sample_2d(black_box(x), black_box(y)); }
                    black_box(sum)
                },
                BatchSize::LargeInput,
            )
        });
    }

    // Worley F1 2D batch.
    {
        let worley = Worley::new().with_mode(DistanceMode::F1);
        let coords = coords.clone();
        g.bench_function("Worley_F1_2d", |b| {
            b.iter_batched(
                || coords.clone(),
                |pts| {
                    let mut sum = 0.0f32;
                    for (x, y) in pts { sum += worley.sample_2d(black_box(x), black_box(y)); }
                    black_box(sum)
                },
                BatchSize::LargeInput,
            )
        });
    }

    // fBm Simplex 6-octave 2D batch — terrain heightmap workload.
    {
        let fbm = Fbm::new(Simplex::new())
            .octaves(6).lacunarity(2.0).gain(0.5).frequency(1.0);
        let coords = coords.clone();
        g.bench_function("Fbm_Simplex_6oct_2d", |b| {
            b.iter_batched(
                || coords.clone(),
                |pts| {
                    let mut sum = 0.0f32;
                    for (x, y) in pts { sum += fbm.sample_2d(black_box(x), black_box(y)); }
                    black_box(sum)
                },
                BatchSize::LargeInput,
            )
        });
    }

    g.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// Group 9: seeded construction cost
//
// Measures how expensive it is to build a generator from a seed.
// Relevant when creating per-chunk noise generators at runtime.
// ─────────────────────────────────────────────────────────────────────────────

fn bench_noise_construction(c: &mut Criterion) {
    let mut g = c.benchmark_group("noise_construction");

    g.bench_function("Perlin_from_seed",  |b| b.iter(|| Perlin::from_seed(black_box(42u64))));
    g.bench_function("Simplex_from_seed", |b| b.iter(|| Simplex::from_seed(black_box(42u64))));
    g.bench_function("Value_from_seed",   |b| b.iter(|| Value::from_seed(black_box(42u64))));
    g.bench_function("Worley_from_seed",  |b| b.iter(|| Worley::from_seed(black_box(42u64))));

    g.finish();
}

// ─────────────────────────────────────────────────────────────────────────────

criterion_group!(
    benches,
    bench_noise_scalar_2d,
    bench_noise_scalar_3d,
    bench_noise_scalar_4d,
    bench_fbm_2d,
    bench_fbm_3d,
    bench_domain_warp_2d,
    bench_domain_warp_3d,
    bench_worley_modes_2d,
    bench_noise_batch_100k,
    bench_noise_construction,
);
criterion_main!(benches);
