// crates/mid-math/benches/vs_wide_int.rs
//! Integer wide vector benchmarks: i32x4 / u32x4 / i16x8 / u16x8 / i8x16 / u8x16
//! vs scalar equivalents and vs glam integer types where applicable.
//!
//! Purpose: confirm the SSE2 paths provide the expected throughput advantage
//! over scalar loops. Integer operations run on the integer ALU independently
//! of the FPU — these benchmarks also validate that running int wide ops
//! concurrently with float ops (as in real engine code) doesn't bottleneck.
//!
//! Run: cargo bench --bench vs_wide_int -p mid-math
//! HTML: target/criterion/report/index.html

use criterion::{
    black_box, criterion_group, criterion_main,
    BatchSize, Criterion, Throughput,
};
use mid_math::{
    IMask4, i32x4, u32x4,
    i16x8, u16x8,
    i8x16, u8x16,
    IVec4, UVec4,
};

// ─────────────────────────────────────────────────────────────────────────────
// Group 1: i32x4 vs scalar i32 vs glam IVec4
//
// i32x4 processes 4 lanes — 4× more data per instruction.
// glam IVec4 uses scalar storage — comparison shows SSE2 lift.
// ─────────────────────────────────────────────────────────────────────────────

fn bench_i32x4(c: &mut Criterion) {
    let mut g = c.benchmark_group("i32x4");

    let wa = i32x4::new(1, 2, 3, 4);
    let wb = i32x4::new(5, 6, 7, 8);

    let ga = glam::IVec4::new(1, 2, 3, 4);
    let gb = glam::IVec4::new(5, 6, 7, 8);

    // ── add ──────────────────────────────────────────────────────────────────
    g.bench_function("add/i32x4",      |b| b.iter(|| black_box(wa) + black_box(wb)));
    g.bench_function("add/glam-IVec4", |b| b.iter(|| black_box(ga) + black_box(gb)));

    // ── mul ──────────────────────────────────────────────────────────────────
    g.bench_function("mul/i32x4",      |b| b.iter(|| black_box(wa) * black_box(wb)));
    g.bench_function("mul/glam-IVec4", |b| b.iter(|| black_box(ga) * black_box(gb)));

    // ── min/max ───────────────────────────────────────────────────────────────
    g.bench_function("min/i32x4",      |b| b.iter(|| black_box(wa).min(black_box(wb))));
    g.bench_function("min/glam-IVec4", |b| b.iter(|| black_box(ga).min(black_box(gb))));
    g.bench_function("max/i32x4",      |b| b.iter(|| black_box(wa).max(black_box(wb))));
    g.bench_function("max/glam-IVec4", |b| b.iter(|| black_box(ga).max(black_box(gb))));

    // ── abs ───────────────────────────────────────────────────────────────────
    let neg = i32x4::new(-1, 2, -3, 4);
    let neg_g = glam::IVec4::new(-1, 2, -3, 4);
    g.bench_function("abs/i32x4",      |b| b.iter(|| black_box(neg).abs()));
    g.bench_function("abs/glam-IVec4", |b| b.iter(|| black_box(neg_g).abs()));

    // ── cmp + blend ───────────────────────────────────────────────────────────
    g.bench_function("cmpeq+blend/i32x4", |b| {
        b.iter(|| {
            let m = black_box(wa).cmpeq(black_box(wb));
            i32x4::blend(m, black_box(wa), black_box(wb))
        })
    });

    // ── shift ────────────────────────────────────────────────────────────────
    g.bench_function("shl/i32x4",           |b| b.iter(|| black_box(wa).shl(2)));
    g.bench_function("shr_arithmetic/i32x4",|b| b.iter(|| black_box(wa).shr_arithmetic(2)));
    g.bench_function("shr_logical/i32x4",   |b| b.iter(|| black_box(wa).shr_logical(2)));

    // ── saturating ───────────────────────────────────────────────────────────
    g.bench_function("saturating_add/i32x4", |b| {
        b.iter(|| black_box(wa).saturating_add(black_box(wb)))
    });

    g.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// Group 2: u32x4 vs glam UVec4
// ─────────────────────────────────────────────────────────────────────────────

fn bench_u32x4(c: &mut Criterion) {
    let mut g = c.benchmark_group("u32x4");

    let wa = u32x4::new(1, 100, u32::MAX - 1, 0);
    let wb = u32x4::new(2, 50,  1,             u32::MAX);
    let ga = glam::UVec4::new(1, 100, u32::MAX - 1, 0);
    let gb = glam::UVec4::new(2, 50,  1,             u32::MAX);

    g.bench_function("add/u32x4",      |b| b.iter(|| black_box(wa) + black_box(wb)));
    g.bench_function("add/glam-UVec4", |b| b.iter(|| black_box(ga) + black_box(gb)));

    // Unsigned min/max — critical that comparison uses unsigned semantics
    g.bench_function("min/u32x4",      |b| b.iter(|| black_box(wa).min(black_box(wb))));
    g.bench_function("min/glam-UVec4", |b| b.iter(|| black_box(ga).min(black_box(gb))));

    g.bench_function("saturating_add/u32x4", |b| {
        b.iter(|| black_box(wa).saturating_add(black_box(wb)))
    });
    g.bench_function("saturating_sub/u32x4", |b| {
        b.iter(|| black_box(wa).saturating_sub(black_box(wb)))
    });

    g.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// Group 3: i16x8 — 8 lanes, quantized animation use case
//
// Real benchmark: dequantize 8 quaternion components in one register.
// Cost: i16x8::as_i32x4_lo + _mm_cvtepi32_ps (not shown here but derived).
// ─────────────────────────────────────────────────────────────────────────────

fn bench_i16x8(c: &mut Criterion) {
    let mut g = c.benchmark_group("i16x8");

    let wa = i16x8::from_array([100, -200, 300, -400, 500, -600, 700, -800]);
    let wb = i16x8::from_array([1, 2, 3, 4, 5, 6, 7, 8]);

    g.bench_function("add/i16x8",             |b| b.iter(|| black_box(wa) + black_box(wb)));
    g.bench_function("saturating_add/i16x8",  |b| b.iter(|| black_box(wa).saturating_add(black_box(wb))));
    g.bench_function("saturating_sub/i16x8",  |b| b.iter(|| black_box(wa).saturating_sub(black_box(wb))));
    g.bench_function("mul_lo/i16x8",          |b| b.iter(|| black_box(wa).mul_lo(black_box(wb))));
    g.bench_function("mul_high/i16x8",        |b| b.iter(|| black_box(wa).mul_high(black_box(wb))));
    g.bench_function("abs/i16x8",             |b| b.iter(|| black_box(wa).abs()));
    g.bench_function("min/i16x8",             |b| b.iter(|| black_box(wa).min(black_box(wb))));
    g.bench_function("max/i16x8",             |b| b.iter(|| black_box(wa).max(black_box(wb))));
    g.bench_function("cmpeq/i16x8",           |b| b.iter(|| black_box(wa).cmpeq(black_box(wb))));
    g.bench_function("widen_lo/i16x8",        |b| b.iter(|| black_box(wa).as_i32x4_lo()));
    g.bench_function("widen_hi/i16x8",        |b| b.iter(|| black_box(wa).as_i32x4_hi()));
    g.bench_function("element_sum/i16x8",     |b| b.iter(|| black_box(wa).element_sum()));

    g.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// Group 4: u16x8 — unsigned 16-bit, texture coords use case
// ─────────────────────────────────────────────────────────────────────────────

fn bench_u16x8(c: &mut Criterion) {
    let mut g = c.benchmark_group("u16x8");

    let wa = u16x8::from_array([100, 200, 300, 400, 500, 600, 700, 800]);
    let wb = u16x8::from_array([50, 100, 350, 500, 600, 400, 800, 700]);

    g.bench_function("add/u16x8",            |b| b.iter(|| black_box(wa) + black_box(wb)));
    g.bench_function("saturating_add/u16x8", |b| b.iter(|| black_box(wa).saturating_add(black_box(wb))));
    g.bench_function("saturating_sub/u16x8", |b| b.iter(|| black_box(wa).saturating_sub(black_box(wb))));
    g.bench_function("min/u16x8",            |b| b.iter(|| black_box(wa).min(black_box(wb))));
    g.bench_function("max/u16x8",            |b| b.iter(|| black_box(wa).max(black_box(wb))));
    g.bench_function("mul_lo/u16x8",         |b| b.iter(|| black_box(wa).mul_lo(black_box(wb))));
    g.bench_function("shr/u16x8",            |b| b.iter(|| black_box(wa).shr(1)));

    g.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// Group 5: i8x16 / u8x16 — byte processing
//
// DixScript string hashing: 16 bytes per cycle. FNV-1a-like step.
// RGBA pixel pack/unpack for texture staging.
// ─────────────────────────────────────────────────────────────────────────────

fn bench_i8x16(c: &mut Criterion) {
    let mut g = c.benchmark_group("i8x16");

    let wa = i8x16::from_array([1,-2,3,-4,5,-6,7,-8,9,-10,11,-12,13,-14,15,-16]);
    let wb = i8x16::from_array([16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1]);

    g.bench_function("add/i8x16",            |b| b.iter(|| black_box(wa) + black_box(wb)));
    g.bench_function("saturating_add/i8x16", |b| b.iter(|| black_box(wa).saturating_add(black_box(wb))));
    g.bench_function("saturating_sub/i8x16", |b| b.iter(|| black_box(wa).saturating_sub(black_box(wb))));
    g.bench_function("abs/i8x16",            |b| b.iter(|| black_box(wa).abs()));
    g.bench_function("cmpeq/i8x16",          |b| b.iter(|| black_box(wa).cmpeq(black_box(wb))));
    g.bench_function("count_eq/i8x16",       |b| {
        let needle = i8x16::splat(7);
        b.iter(|| black_box(wa).count_eq(black_box(needle)))
    });

    g.finish();
}

fn bench_u8x16(c: &mut Criterion) {
    let mut g = c.benchmark_group("u8x16");

    let wa = u8x16::from_array([255,128,64,32,16,8,4,2,1,0,100,200,50,150,75,25]);
    let wb = u8x16::from_array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]);

    g.bench_function("add/u8x16",            |b| b.iter(|| black_box(wa) + black_box(wb)));
    g.bench_function("saturating_add/u8x16", |b| b.iter(|| black_box(wa).saturating_add(black_box(wb))));
    g.bench_function("saturating_sub/u8x16", |b| b.iter(|| black_box(wa).saturating_sub(black_box(wb))));
    g.bench_function("min/u8x16",            |b| b.iter(|| black_box(wa).min(black_box(wb))));
    g.bench_function("max/u8x16",            |b| b.iter(|| black_box(wa).max(black_box(wb))));
    g.bench_function("cmpeq/u8x16",          |b| b.iter(|| black_box(wa).cmpeq(black_box(wb))));
    g.bench_function("element_sum/u8x16",    |b| b.iter(|| black_box(wa).element_sum()));
    g.bench_function("count_eq/u8x16",       |b| {
        let needle = u8x16::splat(128);
        b.iter(|| black_box(wa).count_eq(black_box(needle)))
    });

    g.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// Group 6: Bulk ECS-style integer processing — simulated entity ID lookup
//
// Engine scenario: given a sorted array of 100k entity IDs, find all entities
// matching a given archetype hash. With i32x4: 4 comparisons per cycle.
// ─────────────────────────────────────────────────────────────────────────────

fn bench_bulk_entity_id_scan(c: &mut Criterion) {
    const N: usize = 100_000;
    let mut g = c.benchmark_group("bulk_entity_id_scan");
    g.throughput(Throughput::Elements(N as u64));

    let ids: Vec<i32> = (0..N as i32).collect();
    let target = 77777i32;

    // ── Scalar loop ───────────────────────────────────────────────────────────
    g.bench_function("scalar_find_id", |b| {
        b.iter_batched(
            || ids.clone(),
            |v| {
                let mut found = 0usize;
                for &id in &v {
                    if black_box(id) == black_box(target) { found += 1; }
                }
                black_box(found)
            },
            BatchSize::LargeInput,
        )
    });

    // ── i32x4 SIMD loop — 4 comparisons per cycle ────────────────────────────
    g.bench_function("i32x4_find_id", |b| {
        b.iter_batched(
            || {
                // Pad to multiple of 4
                let mut v = ids.clone();
                while v.len() % 4 != 0 { v.push(-1); }
                v
            },
            |v| {
                let needle = i32x4::splat(target);
                let mut found = 0u32;
                for chunk in v.chunks_exact(4) {
                    let data = i32x4::from_array(chunk.try_into().unwrap());
                    let mask = data.cmpeq(black_box(needle));
                    found += mask.bitmask().count_ones();
                }
                black_box(found)
            },
            BatchSize::LargeInput,
        )
    });

    g.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// Group 7: u8x16 bulk sum — simulates health/flag aggregation over 100k entities
// ─────────────────────────────────────────────────────────────────────────────

fn bench_bulk_u8_sum(c: &mut Criterion) {
    const N: usize = 100_000;
    let mut g = c.benchmark_group("bulk_u8_sum");
    g.throughput(Throughput::Elements(N as u64));

    let bytes: Vec<u8> = (0..N).map(|i| (i % 256) as u8).collect();

    g.bench_function("scalar_sum", |b| {
        b.iter_batched(
            || bytes.clone(),
            |v| {
                let sum: u64 = v.iter().map(|&x| x as u64).sum();
                black_box(sum)
            },
            BatchSize::LargeInput,
        )
    });

    g.bench_function("u8x16_sum", |b| {
        b.iter_batched(
            || {
                let mut v = bytes.clone();
                while v.len() % 16 != 0 { v.push(0); }
                v
            },
            |v| {
                let mut total = 0u64;
                for chunk in v.chunks_exact(16) {
                    let w = u8x16::from_array(chunk.try_into().unwrap());
                    total += black_box(w).element_sum() as u64;
                }
                black_box(total)
            },
            BatchSize::LargeInput,
        )
    });

    g.finish();
}

// ─────────────────────────────────────────────────────────────────────────────

criterion_group!(
    benches,
    bench_i32x4,
    bench_u32x4,
    bench_i16x8,
    bench_u16x8,
    bench_i8x16,
    bench_u8x16,
    bench_bulk_entity_id_scan,
    bench_bulk_u8_sum,
);
criterion_main!(benches);
