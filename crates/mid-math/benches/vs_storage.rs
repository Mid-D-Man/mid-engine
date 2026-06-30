// crates/mid-math/benches/vs_storage.rs
//! Benchmarks for the low-precision storage types: f16, bf16, F8*, F4*, BitMask*.
//!
//! These types have no public-crate equivalent to benchmark against directly
//! (half-rs and the `float8` crate are the closest references, but mid-math's
//! versions are dependency-free reimplementations). This bench instead
//! measures:
//!   1. Absolute conversion cost (pack f32→storage, unpack storage→f32)
//!   2. Batch vs scalar throughput (does the x4/x8 batching actually help?)
//!   3. BitMask query cost (the ECS-critical path: matches/iter_ones)
//!
//! Run: cargo bench --bench vs_storage -p mid-math
//! HTML report: target/criterion/report/index.html

use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion, Throughput};
use mid_math::storage::*;

// ═════════════════════════════════════════════════════════════════════════════
// f16
// ═════════════════════════════════════════════════════════════════════════════

fn bench_f16(c: &mut Criterion) {
    let mut g = c.benchmark_group("f16/scalar");
    let v = 3.14159f32;
    let h = f16::from_f32(v);

    g.bench_function("from_f32",      |b| b.iter(|| f16::from_f32(black_box(v))));
    g.bench_function("to_f32",        |b| b.iter(|| black_box(h).to_f32()));
    g.bench_function("roundtrip",     |b| b.iter(|| f16::from_f32(black_box(v)).to_f32()));
    g.bench_function("add",           |b| b.iter(|| black_box(h) + black_box(h)));
    g.bench_function("abs_neg",       |b| b.iter(|| (-black_box(h)).abs()));
    g.finish();

    let mut g = c.benchmark_group("f16/batch");
    let src4 = [1.0f32, 2.0, 3.0, 4.0];
    let src8 = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

    g.throughput(Throughput::Elements(4));
    g.bench_function("x4_pack",   |b| b.iter(|| f32x4_to_f16x4(black_box(src4))));
    g.bench_function("x4_scalar", |b| b.iter(|| {
        let s = black_box(src4);
        [f16::from_f32(s[0]), f16::from_f32(s[1]), f16::from_f32(s[2]), f16::from_f32(s[3])]
    }));

    g.throughput(Throughput::Elements(8));
    g.bench_function("x8_pack",   |b| b.iter(|| f32x8_to_f16x8(black_box(src8))));
    g.bench_function("x8_scalar", |b| b.iter(|| src8.map(f16::from_f32)));

    g.finish();

    let mut g = c.benchmark_group("f16/slice_1k");
    let src: Vec<f32> = (0..1024).map(|i| i as f32 * 0.1).collect();
    g.throughput(Throughput::Elements(1024));
    g.bench_function("pack",   |b| b.iter_batched(
        || vec![f16::ZERO; 1024],
        |mut dst| { f32_slice_to_f16(black_box(&src), &mut dst); dst },
        BatchSize::SmallInput,
    ));
    g.bench_function("pack_scalar_loop", |b| b.iter_batched(
        || vec![f16::ZERO; 1024],
        |mut dst| {
            for (d, &s) in dst.iter_mut().zip(black_box(&src)) { *d = f16::from_f32(s); }
            dst
        },
        BatchSize::SmallInput,
    ));
    g.finish();
}

// ═════════════════════════════════════════════════════════════════════════════
// bf16
// ═════════════════════════════════════════════════════════════════════════════

fn bench_bf16(c: &mut Criterion) {
    let mut g = c.benchmark_group("bf16/scalar");
    let v = 3.14159f32;
    let h = bf16::from_f32(v);

    g.bench_function("from_f32",  |b| b.iter(|| bf16::from_f32(black_box(v))));
    g.bench_function("to_f32",    |b| b.iter(|| black_box(h).to_f32())); // should be ~free
    g.bench_function("roundtrip", |b| b.iter(|| bf16::from_f32(black_box(v)).to_f32()));
    g.finish();

    let mut g = c.benchmark_group("bf16/batch");
    let src8 = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    g.throughput(Throughput::Elements(8));
    g.bench_function("x8_pack",   |b| b.iter(|| f32x8_to_bf16x8(black_box(src8))));
    g.bench_function("x8_unpack", |b| {
        let packed = f32x8_to_bf16x8(src8);
        b.iter(|| bf16x8_to_f32x8(black_box(packed)))
    });
    g.finish();
}

// ═════════════════════════════════════════════════════════════════════════════
// f8 — F8Norm, F8E4M3, F8E5M2
// ═════════════════════════════════════════════════════════════════════════════

fn bench_f8(c: &mut Criterion) {
    let mut g = c.benchmark_group("f8norm/scalar");
    let v = 0.6543f32;
    g.bench_function("from_f32", |b| b.iter(|| F8Norm::from_f32(black_box(v))));
    g.bench_function("to_f32",   |b| b.iter(|| black_box(F8Norm::from_f32(v)).to_f32()));
    g.bench_function("lerp", |b| {
        let a = F8Norm::ZERO;
        let bb = F8Norm::ONE;
        let t = F8Norm::HALF;
        b.iter(|| F8Norm::lerp(black_box(a), black_box(bb), black_box(t)))
    });
    g.finish();

    let mut g = c.benchmark_group("f8e4m3/scalar");
    let v = 2.75f32;
    g.bench_function("from_f32_normal",     |b| b.iter(|| F8E4M3::from_f32(black_box(v))));
    g.bench_function("from_f32_subnormal",  |b| b.iter(|| F8E4M3::from_f32(black_box(0.003))));
    g.bench_function("from_f32_overflow",   |b| b.iter(|| F8E4M3::from_f32(black_box(9999.0))));
    g.bench_function("to_f32",              |b| b.iter(|| black_box(F8E4M3::from_f32(v)).to_f32()));
    g.finish();

    let mut g = c.benchmark_group("f8e5m2/scalar");
    g.bench_function("from_f32_normal",    |b| b.iter(|| F8E5M2::from_f32(black_box(v))));
    g.bench_function("from_f32_overflow",  |b| b.iter(|| F8E5M2::from_f32(black_box(1e8))));
    g.bench_function("to_f32",             |b| b.iter(|| black_box(F8E5M2::from_f32(v)).to_f32()));
    g.finish();

    let mut g = c.benchmark_group("f8/batch_x4");
    let src4 = [0.5f32, -1.0, 2.0, -0.25];
    g.throughput(Throughput::Elements(4));
    g.bench_function("e4m3_pack",   |b| b.iter(|| f32x4_to_f8e4m3x4(black_box(src4))));
    g.bench_function("e4m3_unpack", |b| {
        let p = f32x4_to_f8e4m3x4(src4);
        b.iter(|| f8e4m3x4_to_f32x4(black_box(p)))
    });
    g.bench_function("e5m2_pack",   |b| b.iter(|| f32x4_to_f8e5m2x4(black_box(src4))));
    g.bench_function("e5m2_unpack", |b| {
        let p = f32x4_to_f8e5m2x4(src4);
        b.iter(|| f8e5m2x4_to_f32x4(black_box(p)))
    });
    g.finish();
}

// ═════════════════════════════════════════════════════════════════════════════
// f4 — F4E2M1, F4E3M0, packed pairs
// ═════════════════════════════════════════════════════════════════════════════

fn bench_f4(c: &mut Criterion) {
    let mut g = c.benchmark_group("f4e2m1/scalar");
    let v = 2.75f32;
    g.bench_function("from_f32", |b| b.iter(|| F4E2M1::from_f32(black_box(v))));
    g.bench_function("to_f32",   |b| b.iter(|| black_box(F4E2M1::from_f32(v)).to_f32()));
    g.finish();

    let mut g = c.benchmark_group("f4e3m0/scalar");
    g.bench_function("from_f32", |b| b.iter(|| F4E3M0::from_f32(black_box(v))));
    g.bench_function("to_f32",   |b| b.iter(|| black_box(F4E3M0::from_f32(v)).to_f32()));
    g.finish();

    let mut g = c.benchmark_group("f4/pair_pack");
    g.bench_function("e2m1_pair_new", |b| {
        let a = F4E2M1::from_f32(1.5);
        let bb = F4E2M1::from_f32(-2.0);
        b.iter(|| F4E2M1Pair::new(black_box(a), black_box(bb)))
    });
    g.bench_function("e2m1_pair_unpack", |b| {
        let p = F4E2M1Pair::new(F4E2M1::from_f32(1.5), F4E2M1::from_f32(-2.0));
        b.iter(|| black_box(p).to_f32x2())
    });
    g.finish();

    // This is the key throughput number: 8 weights → 4 bytes, the ML use case.
    let mut g = c.benchmark_group("f4/batch_x8_pairs");
    let weights = [1.5f32, -3.0, 0.5, -1.0, 2.0, 0.0, -0.5, 6.0];
    g.throughput(Throughput::Elements(8));
    g.bench_function("e2m1_pack_8_to_4bytes", |b| {
        b.iter(|| f32x8_to_f4e2m1x4pairs(black_box(weights)))
    });
    g.bench_function("e2m1_unpack_4bytes_to_8", |b| {
        let packed = f32x8_to_f4e2m1x4pairs(weights);
        b.iter(|| f4e2m1x4pairs_to_f32x8(black_box(packed)))
    });
    g.finish();

    // Slice API — what an actual weight-loading path would call.
    let mut g = c.benchmark_group("f4/slice_1k_weights");
    let weights_1k: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.01).sin() * 4.0).collect();
    g.throughput(Throughput::Elements(1024));
    g.bench_function("pack_to_512_bytes", |b| b.iter_batched(
        || vec![0u8; 512],
        |mut dst| { f32_slice_to_f4e2m1_packed(black_box(&weights_1k), &mut dst); dst },
        BatchSize::SmallInput,
    ));
    g.bench_function("unpack_from_512_bytes", |b| {
        let mut packed = vec![0u8; 512];
        f32_slice_to_f4e2m1_packed(&weights_1k, &mut packed);
        b.iter_batched(
            || vec![0.0f32; 1024],
            |mut dst| { f4e2m1_packed_to_f32_slice(black_box(&packed), &mut dst); dst },
            BatchSize::SmallInput,
        )
    });
    g.finish();
}

// ═════════════════════════════════════════════════════════════════════════════
// BitMask — the ECS-critical path
// ═════════════════════════════════════════════════════════════════════════════

fn bench_bitmask(c: &mut Criterion) {
    let mut g = c.benchmark_group("bitmask64/core_ops");
    let entity = BitMask64::from_indices(&[0, 1, 2, 5, 10, 31, 63]);
    let query  = BitMask64::from_indices(&[0, 1, 2]);

    g.bench_function("matches",     |b| b.iter(|| black_box(entity).matches(black_box(query))));
    g.bench_function("intersection",|b| b.iter(|| black_box(entity).intersection(black_box(query))));
    g.bench_function("union",       |b| b.iter(|| black_box(entity) | black_box(query)));
    g.bench_function("count_ones",  |b| b.iter(|| black_box(entity).count_ones()));
    g.bench_function("get_bit",     |b| b.iter(|| black_box(entity).get(black_box(31))));
    g.bench_function("set_bit", |b| b.iter_batched(
        || entity,
        |mut m| { m.set(black_box(40)); m },
        BatchSize::SmallInput,
    ));
    g.finish();

    let mut g = c.benchmark_group("bitmask64/iteration");
    let sparse = BitMask64::from_indices(&[0, 32, 63]);
    let dense  = BitMask64::from_indices(&(0..64).step_by(2).collect::<Vec<_>>());
    g.bench_function("iter_ones_sparse_3bits", |b| b.iter(|| {
        let mut sum = 0usize;
        for i in black_box(sparse).iter_ones() { sum += i; }
        sum
    }));
    g.bench_function("iter_ones_dense_32bits", |b| b.iter(|| {
        let mut sum = 0usize;
        for i in black_box(dense).iter_ones() { sum += i; }
        sum
    }));
    g.finish();

    // ECS archetype query simulated at scale: 10,000 entities checked against
    // one query mask. This is the real-world hot loop.
    let mut g = c.benchmark_group("bitmask64/ecs_query_10k");
    let entities: Vec<BitMask64> = (0..10_000)
        .map(|i| BitMask64::from_indices(&[0, 1, (i % 60) as usize]))
        .collect();
    let physics_query = BitMask64::from_indices(&[0, 1]);
    g.throughput(Throughput::Elements(10_000));
    g.bench_function("scan_all", |b| b.iter(|| {
        let mut matched = 0usize;
        for &e in black_box(&entities) {
            if e.matches(physics_query) { matched += 1; }
        }
        matched
    }));
    g.finish();

    // Wide masks — check the multi-word path doesn't regress vs scalar.
    let mut g = c.benchmark_group("bitmask256/core_ops");
    let wide_entity = BitMask256::from_indices(&[0, 64, 128, 192, 255]);
    let wide_query  = BitMask256::from_indices(&[0, 64]);
    g.bench_function("matches",    |b| b.iter(|| black_box(wide_entity).matches(black_box(wide_query))));
    g.bench_function("count_ones", |b| b.iter(|| black_box(wide_entity).count_ones()));
    g.bench_function("iter_ones",  |b| b.iter(|| {
        let mut sum = 0usize;
        for i in black_box(wide_entity).iter_ones() { sum += i; }
        sum
    }));
    g.finish();
}

criterion_group!(
    storage_benches,
    bench_f16,
    bench_bf16,
    bench_f8,
    bench_f4,
    bench_bitmask,
);
criterion_main!(storage_benches);
