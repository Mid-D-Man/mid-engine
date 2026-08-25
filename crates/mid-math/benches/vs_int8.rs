// crates/mid-math/benches/vs_int8.rs
//! Integer vector benchmark: mid-math I8Vec/U8Vec vs glam I8Vec/U8Vec.
//!
//! Mirrors vs_int32.rs exactly in structure/scope. Both libraries are
//! scalar i8/u8 only — no SIMD for integers (see docs/platform-
//! optimization.md §4/§9 for why narrow int vecs stay scalar).
//!
//! One real API difference from vs_int32.rs, not a bug: `dot()` returns
//! a WIDENED type here. mid-math's I8Vec4::dot returns i16 (deliberate
//! overflow-safety choice — summing four i8*i8 products can exceed i8's
//! range); glam's I8Vec4::dot returns i8 (accepts wrapping). Both
//! benched as-is with their real return types — this widening
//! difference doesn't affect the benchmark itself, just noting it since
//! it's the one place these two libraries genuinely disagree on API
//! shape rather than just implementation.
//!
//! Operations benchmarked per type: add, sub, mul (element-wise), scale
//! (scalar), dot, min, max, clamp, abs (I8Vec only), cross (I8Vec3/
//! U8Vec3 only), wrapping_add, saturating_add (mid-math extras — glam
//! doesn't expose these, same note as vs_int32.rs).
//!
//! Run: cargo bench --bench vs_int8 -p mid-math
//! HTML report: target/criterion/report/index.html

use criterion::{black_box, criterion_group, criterion_main, Criterion};

// ── mid-math ──────────────────────────────────────────────────────────────────
use mid_math::{I8Vec2, I8Vec3, I8Vec4, U8Vec2, U8Vec3, U8Vec4};

// ── glam ─────────────────────────────────────────────────────────────────────
use glam::{
    I8Vec2 as GI8Vec2, I8Vec3 as GI8Vec3, I8Vec4 as GI8Vec4,
    U8Vec2 as GU8Vec2, U8Vec3 as GU8Vec3, U8Vec4 as GU8Vec4,
};

// ─────────────────────────────────────────────────────────────────────────────
// I8Vec2
// ─────────────────────────────────────────────────────────────────────────────

fn bench_i8vec2(c: &mut Criterion) {
    let mut g = c.benchmark_group("i8vec2");

    let mm_a = I8Vec2::new(3, -7);
    let mm_b = I8Vec2::new(-2, 5);
    let mm_lo = I8Vec2::new(-10, -10);
    let mm_hi = I8Vec2::new(10, 10);

    let gl_a = GI8Vec2::new(3, -7);
    let gl_b = GI8Vec2::new(-2, 5);
    let gl_lo = GI8Vec2::new(-10, -10);
    let gl_hi = GI8Vec2::new(10, 10);

    g.bench_function("add/mid-math", |b| b.iter(|| black_box(mm_a) + black_box(mm_b)));
    g.bench_function("add/glam",     |b| b.iter(|| black_box(gl_a) + black_box(gl_b)));

    g.bench_function("sub/mid-math", |b| b.iter(|| black_box(mm_a) - black_box(mm_b)));
    g.bench_function("sub/glam",     |b| b.iter(|| black_box(gl_a) - black_box(gl_b)));

    g.bench_function("mul/mid-math", |b| b.iter(|| black_box(mm_a) * black_box(mm_b)));
    g.bench_function("mul/glam",     |b| b.iter(|| black_box(gl_a) * black_box(gl_b)));

    g.bench_function("scale/mid-math", |b| b.iter(|| black_box(mm_a) * 3i8));
    g.bench_function("scale/glam",     |b| b.iter(|| black_box(gl_a) * 3i8));

    g.bench_function("dot/mid-math", |b| b.iter(|| black_box(mm_a).dot(black_box(mm_b))));
    g.bench_function("dot/glam",     |b| b.iter(|| black_box(gl_a).dot(black_box(gl_b))));

    g.bench_function("min/mid-math", |b| b.iter(|| black_box(mm_a).min(black_box(mm_b))));
    g.bench_function("min/glam",     |b| b.iter(|| black_box(gl_a).min(black_box(gl_b))));

    g.bench_function("max/mid-math", |b| b.iter(|| black_box(mm_a).max(black_box(mm_b))));
    g.bench_function("max/glam",     |b| b.iter(|| black_box(gl_a).max(black_box(gl_b))));

    g.bench_function("clamp/mid-math", |b| b.iter(|| black_box(mm_a).clamp(mm_lo, mm_hi)));
    g.bench_function("clamp/glam",     |b| b.iter(|| black_box(gl_a).clamp(gl_lo, gl_hi)));

    g.bench_function("abs/mid-math", |b| b.iter(|| black_box(mm_a).abs()));
    g.bench_function("abs/glam",     |b| b.iter(|| black_box(gl_a).abs()));

    g.bench_function("wrapping_add/mid-math",   |b| b.iter(|| black_box(mm_a).wrapping_add(black_box(mm_b))));
    g.bench_function("saturating_add/mid-math", |b| b.iter(|| black_box(mm_a).saturating_add(black_box(mm_b))));

    g.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// I8Vec3
// ─────────────────────────────────────────────────────────────────────────────

fn bench_i8vec3(c: &mut Criterion) {
    let mut g = c.benchmark_group("i8vec3");

    let mm_a = I8Vec3::new(1, -2, 3);
    let mm_b = I8Vec3::new(-4, 5, -6);
    let mm_lo = I8Vec3::new(-10, -10, -10);
    let mm_hi = I8Vec3::new(10, 10, 10);

    let gl_a = GI8Vec3::new(1, -2, 3);
    let gl_b = GI8Vec3::new(-4, 5, -6);
    let gl_lo = GI8Vec3::new(-10, -10, -10);
    let gl_hi = GI8Vec3::new(10, 10, 10);

    g.bench_function("add/mid-math", |b| b.iter(|| black_box(mm_a) + black_box(mm_b)));
    g.bench_function("add/glam",     |b| b.iter(|| black_box(gl_a) + black_box(gl_b)));

    g.bench_function("sub/mid-math", |b| b.iter(|| black_box(mm_a) - black_box(mm_b)));
    g.bench_function("sub/glam",     |b| b.iter(|| black_box(gl_a) - black_box(gl_b)));

    g.bench_function("mul/mid-math", |b| b.iter(|| black_box(mm_a) * black_box(mm_b)));
    g.bench_function("mul/glam",     |b| b.iter(|| black_box(gl_a) * black_box(gl_b)));

    g.bench_function("scale/mid-math", |b| b.iter(|| black_box(mm_a) * 3i8));
    g.bench_function("scale/glam",     |b| b.iter(|| black_box(gl_a) * 3i8));

    g.bench_function("dot/mid-math", |b| b.iter(|| black_box(mm_a).dot(black_box(mm_b))));
    g.bench_function("dot/glam",     |b| b.iter(|| black_box(gl_a).dot(black_box(gl_b))));

    g.bench_function("cross/mid-math", |b| b.iter(|| black_box(mm_a).cross(black_box(mm_b))));
    g.bench_function("cross/glam",     |b| b.iter(|| black_box(gl_a).cross(black_box(gl_b))));

    g.bench_function("min/mid-math", |b| b.iter(|| black_box(mm_a).min(black_box(mm_b))));
    g.bench_function("min/glam",     |b| b.iter(|| black_box(gl_a).min(black_box(gl_b))));

    g.bench_function("max/mid-math", |b| b.iter(|| black_box(mm_a).max(black_box(mm_b))));
    g.bench_function("max/glam",     |b| b.iter(|| black_box(gl_a).max(black_box(gl_b))));

    g.bench_function("clamp/mid-math", |b| b.iter(|| black_box(mm_a).clamp(mm_lo, mm_hi)));
    g.bench_function("clamp/glam",     |b| b.iter(|| black_box(gl_a).clamp(gl_lo, gl_hi)));

    g.bench_function("abs/mid-math", |b| b.iter(|| black_box(mm_a).abs()));
    g.bench_function("abs/glam",     |b| b.iter(|| black_box(gl_a).abs()));

    g.bench_function("wrapping_add/mid-math",   |b| b.iter(|| black_box(mm_a).wrapping_add(black_box(mm_b))));
    g.bench_function("saturating_add/mid-math", |b| b.iter(|| black_box(mm_a).saturating_add(black_box(mm_b))));

    g.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// I8Vec4
// ─────────────────────────────────────────────────────────────────────────────

fn bench_i8vec4(c: &mut Criterion) {
    let mut g = c.benchmark_group("i8vec4");

    let mm_a = I8Vec4::new(1, -2, 3, -4);
    let mm_b = I8Vec4::new(-5, 6, -7, 8);
    let mm_lo = I8Vec4::new(-10, -10, -10, -10);
    let mm_hi = I8Vec4::new(10, 10, 10, 10);

    let gl_a = GI8Vec4::new(1, -2, 3, -4);
    let gl_b = GI8Vec4::new(-5, 6, -7, 8);
    let gl_lo = GI8Vec4::new(-10, -10, -10, -10);
    let gl_hi = GI8Vec4::new(10, 10, 10, 10);

    g.bench_function("add/mid-math", |b| b.iter(|| black_box(mm_a) + black_box(mm_b)));
    g.bench_function("add/glam",     |b| b.iter(|| black_box(gl_a) + black_box(gl_b)));

    g.bench_function("sub/mid-math", |b| b.iter(|| black_box(mm_a) - black_box(mm_b)));
    g.bench_function("sub/glam",     |b| b.iter(|| black_box(gl_a) - black_box(gl_b)));

    g.bench_function("mul/mid-math", |b| b.iter(|| black_box(mm_a) * black_box(mm_b)));
    g.bench_function("mul/glam",     |b| b.iter(|| black_box(gl_a) * black_box(gl_b)));

    g.bench_function("scale/mid-math", |b| b.iter(|| black_box(mm_a) * 3i8));
    g.bench_function("scale/glam",     |b| b.iter(|| black_box(gl_a) * 3i8));

    g.bench_function("dot/mid-math", |b| b.iter(|| black_box(mm_a).dot(black_box(mm_b))));
    g.bench_function("dot/glam",     |b| b.iter(|| black_box(gl_a).dot(black_box(gl_b))));

    g.bench_function("min/mid-math", |b| b.iter(|| black_box(mm_a).min(black_box(mm_b))));
    g.bench_function("min/glam",     |b| b.iter(|| black_box(gl_a).min(black_box(gl_b))));

    g.bench_function("max/mid-math", |b| b.iter(|| black_box(mm_a).max(black_box(mm_b))));
    g.bench_function("max/glam",     |b| b.iter(|| black_box(gl_a).max(black_box(gl_b))));

    g.bench_function("clamp/mid-math", |b| b.iter(|| black_box(mm_a).clamp(mm_lo, mm_hi)));
    g.bench_function("clamp/glam",     |b| b.iter(|| black_box(gl_a).clamp(gl_lo, gl_hi)));

    g.bench_function("abs/mid-math", |b| b.iter(|| black_box(mm_a).abs()));
    g.bench_function("abs/glam",     |b| b.iter(|| black_box(gl_a).abs()));

    g.bench_function("wrapping_add/mid-math",   |b| b.iter(|| black_box(mm_a).wrapping_add(black_box(mm_b))));
    g.bench_function("saturating_add/mid-math", |b| b.iter(|| black_box(mm_a).saturating_add(black_box(mm_b))));

    g.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// U8Vec2
// ─────────────────────────────────────────────────────────────────────────────

fn bench_u8vec2(c: &mut Criterion) {
    let mut g = c.benchmark_group("u8vec2");

    let mm_a = U8Vec2::new(10, 3);
    let mm_b = U8Vec2::new(2, 7);
    let mm_lo = U8Vec2::new(1, 1);
    let mm_hi = U8Vec2::new(20, 20);

    let gl_a = GU8Vec2::new(10, 3);
    let gl_b = GU8Vec2::new(2, 7);
    let gl_lo = GU8Vec2::new(1, 1);
    let gl_hi = GU8Vec2::new(20, 20);

    g.bench_function("add/mid-math", |b| b.iter(|| black_box(mm_a) + black_box(mm_b)));
    g.bench_function("add/glam",     |b| b.iter(|| black_box(gl_a) + black_box(gl_b)));

    g.bench_function("mul/mid-math", |b| b.iter(|| black_box(mm_a) * black_box(mm_b)));
    g.bench_function("mul/glam",     |b| b.iter(|| black_box(gl_a) * black_box(gl_b)));

    g.bench_function("scale/mid-math", |b| b.iter(|| black_box(mm_a) * 3u8));
    g.bench_function("scale/glam",     |b| b.iter(|| black_box(gl_a) * 3u8));

    g.bench_function("dot/mid-math", |b| b.iter(|| black_box(mm_a).dot(black_box(mm_b))));
    g.bench_function("dot/glam",     |b| b.iter(|| black_box(gl_a).dot(black_box(gl_b))));

    g.bench_function("min/mid-math", |b| b.iter(|| black_box(mm_a).min(black_box(mm_b))));
    g.bench_function("min/glam",     |b| b.iter(|| black_box(gl_a).min(black_box(gl_b))));

    g.bench_function("max/mid-math", |b| b.iter(|| black_box(mm_a).max(black_box(mm_b))));
    g.bench_function("max/glam",     |b| b.iter(|| black_box(gl_a).max(black_box(gl_b))));

    g.bench_function("clamp/mid-math", |b| b.iter(|| black_box(mm_a).clamp(mm_lo, mm_hi)));
    g.bench_function("clamp/glam",     |b| b.iter(|| black_box(gl_a).clamp(gl_lo, gl_hi)));

    g.bench_function("wrapping_add/mid-math",   |b| b.iter(|| black_box(mm_a).wrapping_add(black_box(mm_b))));
    g.bench_function("saturating_add/mid-math", |b| b.iter(|| black_box(mm_a).saturating_add(black_box(mm_b))));

    g.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// U8Vec3
// ─────────────────────────────────────────────────────────────────────────────

fn bench_u8vec3(c: &mut Criterion) {
    let mut g = c.benchmark_group("u8vec3");

    let mm_a = U8Vec3::new(10, 3, 7);
    let mm_b = U8Vec3::new(2, 7, 1);
    let mm_lo = U8Vec3::new(1, 1, 1);
    let mm_hi = U8Vec3::new(20, 20, 20);

    let gl_a = GU8Vec3::new(10, 3, 7);
    let gl_b = GU8Vec3::new(2, 7, 1);
    let gl_lo = GU8Vec3::new(1, 1, 1);
    let gl_hi = GU8Vec3::new(20, 20, 20);

    g.bench_function("add/mid-math", |b| b.iter(|| black_box(mm_a) + black_box(mm_b)));
    g.bench_function("add/glam",     |b| b.iter(|| black_box(gl_a) + black_box(gl_b)));

    g.bench_function("mul/mid-math", |b| b.iter(|| black_box(mm_a) * black_box(mm_b)));
    g.bench_function("mul/glam",     |b| b.iter(|| black_box(gl_a) * black_box(gl_b)));

    g.bench_function("scale/mid-math", |b| b.iter(|| black_box(mm_a) * 3u8));
    g.bench_function("scale/glam",     |b| b.iter(|| black_box(gl_a) * 3u8));

    g.bench_function("dot/mid-math", |b| b.iter(|| black_box(mm_a).dot(black_box(mm_b))));
    g.bench_function("dot/glam",     |b| b.iter(|| black_box(gl_a).dot(black_box(gl_b))));

    g.bench_function("min/mid-math", |b| b.iter(|| black_box(mm_a).min(black_box(mm_b))));
    g.bench_function("min/glam",     |b| b.iter(|| black_box(gl_a).min(black_box(gl_b))));

    g.bench_function("max/mid-math", |b| b.iter(|| black_box(mm_a).max(black_box(mm_b))));
    g.bench_function("max/glam",     |b| b.iter(|| black_box(gl_a).max(black_box(gl_b))));

    g.bench_function("clamp/mid-math", |b| b.iter(|| black_box(mm_a).clamp(mm_lo, mm_hi)));
    g.bench_function("clamp/glam",     |b| b.iter(|| black_box(gl_a).clamp(gl_lo, gl_hi)));

    g.bench_function("wrapping_add/mid-math",   |b| b.iter(|| black_box(mm_a).wrapping_add(black_box(mm_b))));
    g.bench_function("saturating_add/mid-math", |b| b.iter(|| black_box(mm_a).saturating_add(black_box(mm_b))));

    g.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// U8Vec4
// ─────────────────────────────────────────────────────────────────────────────

fn bench_u8vec4(c: &mut Criterion) {
    let mut g = c.benchmark_group("u8vec4");

    let mm_a = U8Vec4::new(10, 3, 7, 1);
    let mm_b = U8Vec4::new(2, 7, 1, 5);
    let mm_lo = U8Vec4::new(1, 1, 1, 1);
    let mm_hi = U8Vec4::new(20, 20, 20, 20);

    let gl_a = GU8Vec4::new(10, 3, 7, 1);
    let gl_b = GU8Vec4::new(2, 7, 1, 5);
    let gl_lo = GU8Vec4::new(1, 1, 1, 1);
    let gl_hi = GU8Vec4::new(20, 20, 20, 20);

    g.bench_function("add/mid-math", |b| b.iter(|| black_box(mm_a) + black_box(mm_b)));
    g.bench_function("add/glam",     |b| b.iter(|| black_box(gl_a) + black_box(gl_b)));

    g.bench_function("mul/mid-math", |b| b.iter(|| black_box(mm_a) * black_box(mm_b)));
    g.bench_function("mul/glam",     |b| b.iter(|| black_box(gl_a) * black_box(gl_b)));

    g.bench_function("scale/mid-math", |b| b.iter(|| black_box(mm_a) * 3u8));
    g.bench_function("scale/glam",     |b| b.iter(|| black_box(gl_a) * 3u8));

    g.bench_function("dot/mid-math", |b| b.iter(|| black_box(mm_a).dot(black_box(mm_b))));
    g.bench_function("dot/glam",     |b| b.iter(|| black_box(gl_a).dot(black_box(gl_b))));

    g.bench_function("min/mid-math", |b| b.iter(|| black_box(mm_a).min(black_box(mm_b))));
    g.bench_function("min/glam",     |b| b.iter(|| black_box(gl_a).min(black_box(gl_b))));

    g.bench_function("max/mid-math", |b| b.iter(|| black_box(mm_a).max(black_box(mm_b))));
    g.bench_function("max/glam",     |b| b.iter(|| black_box(gl_a).max(black_box(gl_b))));

    g.bench_function("clamp/mid-math", |b| b.iter(|| black_box(mm_a).clamp(mm_lo, mm_hi)));
    g.bench_function("clamp/glam",     |b| b.iter(|| black_box(gl_a).clamp(gl_lo, gl_hi)));

    g.bench_function("wrapping_add/mid-math",   |b| b.iter(|| black_box(mm_a).wrapping_add(black_box(mm_b))));
    g.bench_function("saturating_add/mid-math", |b| b.iter(|| black_box(mm_a).saturating_add(black_box(mm_b))));

    g.finish();
}

// ─────────────────────────────────────────────────────────────────────────────

criterion_group!(
    benches,
    bench_i8vec2,
    bench_i8vec3,
    bench_i8vec4,
    bench_u8vec2,
    bench_u8vec3,
    bench_u8vec4,
);
criterion_main!(benches);
