// crates/mid-math/benches/vs_wide_int.rs
//! Integer wide vector benchmarks: i32x4/u32x4/i16x8/u16x8/i8x16/u8x16
//! (SSE2/NEON/scalar tier) and i32x8/u32x8/i16x16/u16x16/i8x32/u8x32
//! (AVX2 tier, additive) vs scalar equivalents, vs glam integer types
//! where applicable, and vs the `wide` crate (crates.io) — the one
//! comparison target that shares mid-math's own concept here: N packed
//! scalar lanes, not a component vector. glam's IVec4/UVec4 stay as a
//! comparison too (shows the scalar-storage-vs-SIMD lift), but they're
//! a different concept — one vector's x/y/z/w, not four packed
//! instances. See docs/platform-optimization.md §9 for why nalgebra and
//! ultraviolet aren't compared here: neither has a stable-Rust integer
//! wide type. ultraviolet's `int.rs` is narrow IVec2/3/4/UVec2/3/4 only
//! (same category as glam, checked directly against its published
//! source — no x4/x8 batch concept for integers, only for its float
//! Vec3x4/Vec3x8 rotor types). nalgebra's SIMD story goes through
//! `simba`, whose `wide`-crate integration (`WideF32x4` etc.) is
//! float-only; simba's integer SIMD types exist ONLY behind Rust's
//! nightly-only `#![feature(portable_simd)]`, which this project's
//! stable-only toolchain policy rules out.
//!
//! `wide` crate comparisons are scoped to operations verified directly
//! against its published source (`splat`, `new([T;N])`, `min`, `max`,
//! `abs`, `saturating_add`, `saturating_sub`, and the `Add`/`Sub`/`Mul`
//! operators) — its shift-by-vector and equality-comparison APIs have a
//! different calling convention that wasn't worth guessing at without a
//! compiler to check against.
//!
//! Purpose: confirm the SSE2/AVX2 paths provide the expected throughput
//! advantage over scalar loops. Integer operations run on the integer
//! ALU independently of the FPU — these benchmarks also validate that
//! running int wide ops concurrently with float ops (as in real engine
//! code) doesn't bottleneck.
//!
//! Run: cargo bench --bench vs_wide_int -p mid-math
//! HTML: target/criterion/report/index.html

use criterion::{
    black_box, criterion_group, criterion_main,
    BatchSize, Criterion, Throughput,
};
use mid_math::{
    i32x4, u32x4,
    i16x8, u16x8,
    i8x16, u8x16,
};

#[cfg(target_feature = "avx2")]
use mid_math::{i32x8, u32x8, i16x16, u16x16, i8x32, u8x32};

// ─────────────────────────────────────────────────────────────────────────────
// Group 1: i32x4 vs scalar i32 vs glam IVec4 vs wide::i32x4
//
// i32x4 processes 4 lanes — 4× more data per instruction.
// glam IVec4 uses scalar storage — comparison shows SSE2 lift.
// wide::i32x4 is the apples-to-apples comparison: same concept (4
// packed i32 lanes), different crate, its own SSE2/NEON/scalar dispatch.
// ─────────────────────────────────────────────────────────────────────────────

fn bench_i32x4(c: &mut Criterion) {
    let mut g = c.benchmark_group("i32x4");

    let wa = i32x4::new(1, 2, 3, 4);
    let wb = i32x4::new(5, 6, 7, 8);

    let ga = glam::IVec4::new(1, 2, 3, 4);
    let gb = glam::IVec4::new(5, 6, 7, 8);

    let da = wide::i32x4::new([1, 2, 3, 4]);
    let db = wide::i32x4::new([5, 6, 7, 8]);

    // ── add ──────────────────────────────────────────────────────────────────
    g.bench_function("add/i32x4",      |b| b.iter(|| black_box(wa) + black_box(wb)));
    g.bench_function("add/glam-IVec4", |b| b.iter(|| black_box(ga) + black_box(gb)));
    g.bench_function("add/wide-i32x4", |b| b.iter(|| black_box(da) + black_box(db)));

    // ── mul ──────────────────────────────────────────────────────────────────
    g.bench_function("mul/i32x4",      |b| b.iter(|| black_box(wa) * black_box(wb)));
    g.bench_function("mul/glam-IVec4", |b| b.iter(|| black_box(ga) * black_box(gb)));
    g.bench_function("mul/wide-i32x4", |b| b.iter(|| black_box(da) * black_box(db)));

    // ── min/max ───────────────────────────────────────────────────────────────
    g.bench_function("min/i32x4",      |b| b.iter(|| black_box(wa).min(black_box(wb))));
    g.bench_function("min/glam-IVec4", |b| b.iter(|| black_box(ga).min(black_box(gb))));
    g.bench_function("min/wide-i32x4", |b| b.iter(|| black_box(da).min(black_box(db))));
    g.bench_function("max/i32x4",      |b| b.iter(|| black_box(wa).max(black_box(wb))));
    g.bench_function("max/glam-IVec4", |b| b.iter(|| black_box(ga).max(black_box(gb))));
    g.bench_function("max/wide-i32x4", |b| b.iter(|| black_box(da).max(black_box(db))));

    // ── abs ───────────────────────────────────────────────────────────────────
    let neg = i32x4::new(-1, 2, -3, 4);
    let neg_g = glam::IVec4::new(-1, 2, -3, 4);
    let neg_d = wide::i32x4::new([-1, 2, -3, 4]);
    g.bench_function("abs/i32x4",      |b| b.iter(|| black_box(neg).abs()));
    g.bench_function("abs/glam-IVec4", |b| b.iter(|| black_box(neg_g).abs()));
    g.bench_function("abs/wide-i32x4", |b| b.iter(|| black_box(neg_d).abs()));

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
    g.bench_function("saturating_add/wide-i32x4", |b| {
        b.iter(|| black_box(da).saturating_add(black_box(db)))
    });

    g.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// Group 2: u32x4 vs glam UVec4 vs wide::u32x4
// ─────────────────────────────────────────────────────────────────────────────

fn bench_u32x4(c: &mut Criterion) {
    let mut g = c.benchmark_group("u32x4");

    let wa = u32x4::new(1, 100, u32::MAX - 1, 0);
    let wb = u32x4::new(2, 50,  1,             u32::MAX);
    let ga = glam::UVec4::new(1, 100, u32::MAX - 1, 0);
    let gb = glam::UVec4::new(2, 50,  1,             u32::MAX);
    let da = wide::u32x4::new([1, 100, u32::MAX - 1, 0]);
    let db = wide::u32x4::new([2, 50,  1,             u32::MAX]);

    g.bench_function("add/u32x4",      |b| b.iter(|| black_box(wa) + black_box(wb)));
    g.bench_function("add/glam-UVec4", |b| b.iter(|| black_box(ga) + black_box(gb)));
    g.bench_function("add/wide-u32x4", |b| b.iter(|| black_box(da) + black_box(db)));

    // Unsigned min/max — critical that comparison uses unsigned semantics
    g.bench_function("min/u32x4",      |b| b.iter(|| black_box(wa).min(black_box(wb))));
    g.bench_function("min/glam-UVec4", |b| b.iter(|| black_box(ga).min(black_box(gb))));
    g.bench_function("min/wide-u32x4", |b| b.iter(|| black_box(da).min(black_box(db))));

    g.bench_function("saturating_add/u32x4", |b| {
        b.iter(|| black_box(wa).saturating_add(black_box(wb)))
    });
    g.bench_function("saturating_add/wide-u32x4", |b| {
        b.iter(|| black_box(da).saturating_add(black_box(db)))
    });
    g.bench_function("saturating_sub/u32x4", |b| {
        b.iter(|| black_box(wa).saturating_sub(black_box(wb)))
    });
    g.bench_function("saturating_sub/wide-u32x4", |b| {
        b.iter(|| black_box(da).saturating_sub(black_box(db)))
    });

    g.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// Group 3: i16x8 vs wide::i16x8 — 8 lanes, quantized animation use case
//
// Real benchmark: dequantize 8 quaternion components in one register.
// Cost: i16x8::as_i32x4_lo + _mm_cvtepi32_ps (not shown here but derived).
// ─────────────────────────────────────────────────────────────────────────────

fn bench_i16x8(c: &mut Criterion) {
    let mut g = c.benchmark_group("i16x8");

    let wa = i16x8::from_array([100, -200, 300, -400, 500, -600, 700, -800]);
    let wb = i16x8::from_array([1, 2, 3, 4, 5, 6, 7, 8]);
    let da = wide::i16x8::new([100, -200, 300, -400, 500, -600, 700, -800]);
    let db = wide::i16x8::new([1, 2, 3, 4, 5, 6, 7, 8]);

    g.bench_function("add/i16x8",             |b| b.iter(|| black_box(wa) + black_box(wb)));
    g.bench_function("add/wide-i16x8",        |b| b.iter(|| black_box(da) + black_box(db)));
    g.bench_function("saturating_add/i16x8",  |b| b.iter(|| black_box(wa).saturating_add(black_box(wb))));
    g.bench_function("saturating_add/wide-i16x8", |b| b.iter(|| black_box(da).saturating_add(black_box(db))));
    g.bench_function("saturating_sub/i16x8",  |b| b.iter(|| black_box(wa).saturating_sub(black_box(wb))));
    g.bench_function("mul_lo/i16x8",          |b| b.iter(|| black_box(wa).mul_lo(black_box(wb))));
    g.bench_function("mul_high/i16x8",        |b| b.iter(|| black_box(wa).mul_high(black_box(wb))));
    g.bench_function("abs/i16x8",             |b| b.iter(|| black_box(wa).abs()));
    g.bench_function("abs/wide-i16x8",        |b| b.iter(|| black_box(da).abs()));
    g.bench_function("min/i16x8",             |b| b.iter(|| black_box(wa).min(black_box(wb))));
    g.bench_function("min/wide-i16x8",        |b| b.iter(|| black_box(da).min(black_box(db))));
    g.bench_function("max/i16x8",             |b| b.iter(|| black_box(wa).max(black_box(wb))));
    g.bench_function("cmpeq/i16x8",           |b| b.iter(|| black_box(wa).cmpeq(black_box(wb))));
    g.bench_function("widen_lo/i16x8",        |b| b.iter(|| black_box(wa).as_i32x4_lo()));
    g.bench_function("widen_hi/i16x8",        |b| b.iter(|| black_box(wa).as_i32x4_hi()));
    g.bench_function("element_sum/i16x8",     |b| b.iter(|| black_box(wa).element_sum()));

    g.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// Group 4: u16x8 vs wide::u16x8 — unsigned 16-bit, texture coords use case
// ─────────────────────────────────────────────────────────────────────────────

fn bench_u16x8(c: &mut Criterion) {
    let mut g = c.benchmark_group("u16x8");

    let wa = u16x8::from_array([100, 200, 300, 400, 500, 600, 700, 800]);
    let wb = u16x8::from_array([50, 100, 350, 500, 600, 400, 800, 700]);
    let da = wide::u16x8::new([100, 200, 300, 400, 500, 600, 700, 800]);
    let db = wide::u16x8::new([50, 100, 350, 500, 600, 400, 800, 700]);

    g.bench_function("add/u16x8",            |b| b.iter(|| black_box(wa) + black_box(wb)));
    g.bench_function("add/wide-u16x8",       |b| b.iter(|| black_box(da) + black_box(db)));
    g.bench_function("saturating_add/u16x8", |b| b.iter(|| black_box(wa).saturating_add(black_box(wb))));
    g.bench_function("saturating_sub/u16x8", |b| b.iter(|| black_box(wa).saturating_sub(black_box(wb))));
    g.bench_function("min/u16x8",            |b| b.iter(|| black_box(wa).min(black_box(wb))));
    g.bench_function("min/wide-u16x8",       |b| b.iter(|| black_box(da).min(black_box(db))));
    g.bench_function("max/u16x8",            |b| b.iter(|| black_box(wa).max(black_box(wb))));
    g.bench_function("mul_lo/u16x8",         |b| b.iter(|| black_box(wa).mul_lo(black_box(wb))));
    g.bench_function("shr/u16x8",            |b| b.iter(|| black_box(wa).shr(1)));

    g.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// Group 5: i8x16 / u8x16 vs wide — byte processing
//
// DixScript string hashing: 16 bytes per cycle. FNV-1a-like step.
// RGBA pixel pack/unpack for texture staging.
// ─────────────────────────────────────────────────────────────────────────────

fn bench_i8x16(c: &mut Criterion) {
    let mut g = c.benchmark_group("i8x16");

    let wa = i8x16::from_array([1,-2,3,-4,5,-6,7,-8,9,-10,11,-12,13,-14,15,-16]);
    let wb = i8x16::from_array([16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1]);
    let da = wide::i8x16::new([1,-2,3,-4,5,-6,7,-8,9,-10,11,-12,13,-14,15,-16]);
    let db = wide::i8x16::new([16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1]);

    g.bench_function("add/i8x16",            |b| b.iter(|| black_box(wa) + black_box(wb)));
    g.bench_function("add/wide-i8x16",       |b| b.iter(|| black_box(da) + black_box(db)));
    g.bench_function("saturating_add/i8x16", |b| b.iter(|| black_box(wa).saturating_add(black_box(wb))));
    g.bench_function("saturating_sub/i8x16", |b| b.iter(|| black_box(wa).saturating_sub(black_box(wb))));
    g.bench_function("abs/i8x16",            |b| b.iter(|| black_box(wa).abs()));
    g.bench_function("abs/wide-i8x16",       |b| b.iter(|| black_box(da).abs()));
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
    let da = wide::u8x16::new([255,128,64,32,16,8,4,2,1,0,100,200,50,150,75,25]);
    let db = wide::u8x16::new([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]);

    g.bench_function("add/u8x16",            |b| b.iter(|| black_box(wa) + black_box(wb)));
    g.bench_function("add/wide-u8x16",       |b| b.iter(|| black_box(da) + black_box(db)));
    g.bench_function("saturating_add/u8x16", |b| b.iter(|| black_box(wa).saturating_add(black_box(wb))));
    g.bench_function("saturating_sub/u8x16", |b| b.iter(|| black_box(wa).saturating_sub(black_box(wb))));
    g.bench_function("min/u8x16",            |b| b.iter(|| black_box(wa).min(black_box(wb))));
    g.bench_function("min/wide-u8x16",       |b| b.iter(|| black_box(da).min(black_box(db))));
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
// Group 6-11: AVX2 additive types — i32x8/u32x8/i16x16/u16x16/i8x32/u8x32
//
// No glam comparison — glam has no 8/16/32-wide integer type at any
// width (checked directly against its published source, same result as
// the narrow-vs-wide investigation that started this work: glam's
// int vecs top out at IVec4). wide::TYPE is the only apples-to-apples
// comparison available. Compiled and benched only when target_feature
// avx2 is active — see bench_avx2_types below for the dispatch.
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(target_feature = "avx2")]
fn bench_i32x8(c: &mut Criterion) {
    let mut g = c.benchmark_group("i32x8");

    let wa = i32x8::new(1, 2, 3, 4, 5, 6, 7, 8);
    let wb = i32x8::new(10, 20, 30, 40, 50, 60, 70, 80);
    let da = wide::i32x8::new([1, 2, 3, 4, 5, 6, 7, 8]);
    let db = wide::i32x8::new([10, 20, 30, 40, 50, 60, 70, 80]);

    g.bench_function("add/i32x8",      |b| b.iter(|| black_box(wa) + black_box(wb)));
    g.bench_function("add/wide-i32x8", |b| b.iter(|| black_box(da) + black_box(db)));
    g.bench_function("mul/i32x8",      |b| b.iter(|| black_box(wa) * black_box(wb)));
    g.bench_function("mul/wide-i32x8", |b| b.iter(|| black_box(da) * black_box(db)));
    g.bench_function("min/i32x8",      |b| b.iter(|| black_box(wa).min(black_box(wb))));
    g.bench_function("min/wide-i32x8", |b| b.iter(|| black_box(da).min(black_box(db))));
    g.bench_function("max/i32x8",      |b| b.iter(|| black_box(wa).max(black_box(wb))));
    g.bench_function("max/wide-i32x8", |b| b.iter(|| black_box(da).max(black_box(db))));

    let neg = i32x8::new(-1, 2, -3, 4, -5, 6, -7, 8);
    let neg_d = wide::i32x8::new([-1, 2, -3, 4, -5, 6, -7, 8]);
    g.bench_function("abs/i32x8",      |b| b.iter(|| black_box(neg).abs()));
    g.bench_function("abs/wide-i32x8", |b| b.iter(|| black_box(neg_d).abs()));

    g.bench_function("cmpeq+blend/i32x8", |b| {
        b.iter(|| {
            let m = black_box(wa).cmpeq(black_box(wb));
            i32x8::blend(m, black_box(wa), black_box(wb))
        })
    });
    g.bench_function("saturating_add/i32x8", |b| b.iter(|| black_box(wa).saturating_add(black_box(wb))));
    g.bench_function("saturating_add/wide-i32x8", |b| b.iter(|| black_box(da).saturating_add(black_box(db))));

    g.finish();
}

#[cfg(target_feature = "avx2")]
fn bench_u32x8(c: &mut Criterion) {
    let mut g = c.benchmark_group("u32x8");

    let wa = u32x8::new(1, 100, u32::MAX - 1, 0, 5, 6, 7, 8);
    let wb = u32x8::new(2, 50,  1,             u32::MAX, 1, 2, 3, 4);
    let da = wide::u32x8::new([1, 100, u32::MAX - 1, 0, 5, 6, 7, 8]);
    let db = wide::u32x8::new([2, 50,  1,             u32::MAX, 1, 2, 3, 4]);

    g.bench_function("add/u32x8",      |b| b.iter(|| black_box(wa) + black_box(wb)));
    g.bench_function("add/wide-u32x8", |b| b.iter(|| black_box(da) + black_box(db)));
    g.bench_function("min/u32x8",      |b| b.iter(|| black_box(wa).min(black_box(wb))));
    g.bench_function("min/wide-u32x8", |b| b.iter(|| black_box(da).min(black_box(db))));
    g.bench_function("saturating_add/u32x8", |b| b.iter(|| black_box(wa).saturating_add(black_box(wb))));
    g.bench_function("saturating_add/wide-u32x8", |b| b.iter(|| black_box(da).saturating_add(black_box(db))));
    g.bench_function("saturating_sub/u32x8", |b| b.iter(|| black_box(wa).saturating_sub(black_box(wb))));

    g.finish();
}

#[cfg(target_feature = "avx2")]
fn bench_i16x16(c: &mut Criterion) {
    let mut g = c.benchmark_group("i16x16");

    let arr_a = [100i16, -200, 300, -400, 500, -600, 700, -800, 1, 2, 3, 4, 5, 6, 7, 8];
    let arr_b = [1i16, 2, 3, 4, 5, 6, 7, 8, 100, -200, 300, -400, 500, -600, 700, -800];
    let wa = i16x16::from_array(arr_a);
    let wb = i16x16::from_array(arr_b);
    let da = wide::i16x16::new(arr_a);
    let db = wide::i16x16::new(arr_b);

    g.bench_function("add/i16x16",             |b| b.iter(|| black_box(wa) + black_box(wb)));
    g.bench_function("add/wide-i16x16",        |b| b.iter(|| black_box(da) + black_box(db)));
    g.bench_function("saturating_add/i16x16",  |b| b.iter(|| black_box(wa).saturating_add(black_box(wb))));
    g.bench_function("mul_lo/i16x16",          |b| b.iter(|| black_box(wa).mul_lo(black_box(wb))));
    g.bench_function("abs/i16x16",             |b| b.iter(|| black_box(wa).abs()));
    g.bench_function("abs/wide-i16x16",        |b| b.iter(|| black_box(da).abs()));
    g.bench_function("min/i16x16",             |b| b.iter(|| black_box(wa).min(black_box(wb))));
    g.bench_function("min/wide-i16x16",        |b| b.iter(|| black_box(da).min(black_box(db))));
    g.bench_function("cmpeq/i16x16",           |b| b.iter(|| black_box(wa).cmpeq(black_box(wb))));
    // Widen — the fix this pass: dedicated _mm256_cvtepi16_epi32, not the
    // per-128-bit-lane unpacklo/unpackhi shuffle trick. See avx2/i16x16.rs.
    g.bench_function("widen_lo/i16x16",        |b| b.iter(|| black_box(wa).as_i32x8_lo()));
    g.bench_function("widen_hi/i16x16",        |b| b.iter(|| black_box(wa).as_i32x8_hi()));
    g.bench_function("element_sum/i16x16",     |b| b.iter(|| black_box(wa).element_sum()));

    g.finish();
}

#[cfg(target_feature = "avx2")]
fn bench_u16x16(c: &mut Criterion) {
    let mut g = c.benchmark_group("u16x16");

    let arr_a = [100u16, 200, 300, 400, 500, 600, 700, 800, 1, 2, 3, 4, 5, 6, 7, 8];
    let arr_b = [50u16, 100, 350, 500, 600, 400, 800, 700, 8, 7, 6, 5, 4, 3, 2, 1];
    let wa = u16x16::from_array(arr_a);
    let wb = u16x16::from_array(arr_b);
    let da = wide::u16x16::new(arr_a);
    let db = wide::u16x16::new(arr_b);

    g.bench_function("add/u16x16",            |b| b.iter(|| black_box(wa) + black_box(wb)));
    g.bench_function("add/wide-u16x16",       |b| b.iter(|| black_box(da) + black_box(db)));
    g.bench_function("saturating_add/u16x16", |b| b.iter(|| black_box(wa).saturating_add(black_box(wb))));
    g.bench_function("min/u16x16",            |b| b.iter(|| black_box(wa).min(black_box(wb))));
    g.bench_function("min/wide-u16x16",       |b| b.iter(|| black_box(da).min(black_box(db))));
    g.bench_function("mul_lo/u16x16",         |b| b.iter(|| black_box(wa).mul_lo(black_box(wb))));
    g.bench_function("shr/u16x16",            |b| b.iter(|| black_box(wa).shr(1)));
    // Widen — dedicated _mm256_cvtepu16_epi32, see avx2/u16x16.rs.
    g.bench_function("widen_lo/u16x16",       |b| b.iter(|| black_box(wa).as_u32x8_lo()));
    g.bench_function("widen_hi/u16x16",       |b| b.iter(|| black_box(wa).as_u32x8_hi()));

    g.finish();
}

#[cfg(target_feature = "avx2")]
fn bench_i8x32(c: &mut Criterion) {
    let mut g = c.benchmark_group("i8x32");

    let arr_a: [i8; 32] = core::array::from_fn(|i| ((i as i32 * 7 - 100) % 127) as i8);
    let arr_b: [i8; 32] = core::array::from_fn(|i| ((i as i32 * 3 + 5) % 100) as i8);
    let wa = i8x32::from_array(arr_a);
    let wb = i8x32::from_array(arr_b);
    let da = wide::i8x32::new(arr_a);
    let db = wide::i8x32::new(arr_b);

    g.bench_function("add/i8x32",            |b| b.iter(|| black_box(wa) + black_box(wb)));
    g.bench_function("add/wide-i8x32",       |b| b.iter(|| black_box(da) + black_box(db)));
    g.bench_function("saturating_add/i8x32", |b| b.iter(|| black_box(wa).saturating_add(black_box(wb))));
    g.bench_function("abs/i8x32",            |b| b.iter(|| black_box(wa).abs()));
    g.bench_function("abs/wide-i8x32",       |b| b.iter(|| black_box(da).abs()));
    g.bench_function("cmpeq/i8x32",          |b| b.iter(|| black_box(wa).cmpeq(black_box(wb))));
    g.bench_function("count_eq/i8x32",       |b| {
        let needle = i8x32::splat(7);
        b.iter(|| black_box(wa).count_eq(black_box(needle)))
    });
    // Widen — dedicated _mm256_cvtepi8_epi16, see avx2/i8x32.rs.
    g.bench_function("widen_lo/i8x32",       |b| b.iter(|| black_box(wa).as_i16x16_lo()));
    g.bench_function("widen_hi/i8x32",       |b| b.iter(|| black_box(wa).as_i16x16_hi()));
    // shuffle_bytes — per-16-byte-half semantics, see avx2/i8x32.rs doc comment.
    g.bench_function("shuffle_bytes/i8x32",  |b| b.iter(|| black_box(wa).shuffle_bytes(black_box(wb))));

    g.finish();
}

#[cfg(target_feature = "avx2")]
fn bench_u8x32(c: &mut Criterion) {
    let mut g = c.benchmark_group("u8x32");

    let arr_a: [u8; 32] = core::array::from_fn(|i| (i * 7) as u8);
    let arr_b: [u8; 32] = core::array::from_fn(|i| (i * 3 + 1) as u8);
    let wa = u8x32::from_array(arr_a);
    let wb = u8x32::from_array(arr_b);
    let da = wide::u8x32::new(arr_a);
    let db = wide::u8x32::new(arr_b);

    g.bench_function("add/u8x32",            |b| b.iter(|| black_box(wa) + black_box(wb)));
    g.bench_function("add/wide-u8x32",       |b| b.iter(|| black_box(da) + black_box(db)));
    g.bench_function("saturating_add/u8x32", |b| b.iter(|| black_box(wa).saturating_add(black_box(wb))));
    g.bench_function("min/u8x32",            |b| b.iter(|| black_box(wa).min(black_box(wb))));
    g.bench_function("min/wide-u8x32",       |b| b.iter(|| black_box(da).min(black_box(db))));
    g.bench_function("cmpeq/u8x32",          |b| b.iter(|| black_box(wa).cmpeq(black_box(wb))));
    g.bench_function("element_sum/u8x32",    |b| b.iter(|| black_box(wa).element_sum()));
    // Widen — dedicated _mm256_cvtepu8_epi16, see avx2/u8x32.rs.
    g.bench_function("widen_lo/u8x32",       |b| b.iter(|| black_box(wa).as_u16x16_lo()));
    g.bench_function("widen_hi/u8x32",       |b| b.iter(|| black_box(wa).as_u16x16_hi()));

    g.finish();
}

/// Dispatch wrapper so `criterion_group!` below doesn't need a
/// conditionally-assembled function list — the AVX2 bench functions
/// themselves are `#[cfg(target_feature = "avx2")]`-gated (they use
/// `mid_math::i32x8` etc., which don't exist otherwise), so this
/// wrapper is the single always-present entry point criterion calls.
fn bench_avx2_types(c: &mut Criterion) {
    #[cfg(target_feature = "avx2")]
    {
        bench_i32x8(c);
        bench_u32x8(c);
        bench_i16x16(c);
        bench_u16x16(c);
        bench_i8x32(c);
        bench_u8x32(c);
    }
    #[cfg(not(target_feature = "avx2"))]
    {
        let _ = c;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Group 12: Bulk ECS-style integer processing — simulated entity ID lookup
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
// Group 13: u8x16 bulk sum — simulates health/flag aggregation over 100k entities
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
    bench_avx2_types,
    bench_bulk_entity_id_scan,
    bench_bulk_u8_sum,
);
criterion_main!(benches);
