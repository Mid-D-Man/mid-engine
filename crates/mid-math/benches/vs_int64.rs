// crates/mid-math/benches/vs_int64.rs
//! Integer vector benchmark: mid-math I64Vec/U64Vec vs glam I64Vec/U64Vec.
//!
//! Both libraries are scalar i64/u64 only — no SIMD for 64-bit integers.
//! The goal is confirming we have no implementation overhead vs glam.
//! Any gap > 5% here is pure struct layout or missed inlining, not a
//! fundamental algorithmic difference.
//!
//! Operations benchmarked per type:
//!   add, sub, mul (element-wise), scale (scalar), dot, min, max,
//!   clamp, abs (IVec only), cross (IVec3/UVec3 only).
//!
//! Note: glam has no saturating_add/wrapping_add exposed on I64Vec/U64Vec
//! in all versions — mid-math extras are regression-only where missing.
//!
//! Run: cargo bench --bench vs_int64 -p mid-math
//! HTML report: target/criterion/report/index.html

use criterion::{black_box, criterion_group, criterion_main, Criterion};

// ── mid-math ──────────────────────────────────────────────────────────────────
use mid_math::{I64Vec2, I64Vec3, I64Vec4, U64Vec2, U64Vec3, U64Vec4};

// ── glam ─────────────────────────────────────────────────────────────────────
use glam::{
    I64Vec2 as GI64Vec2, I64Vec3 as GI64Vec3, I64Vec4 as GI64Vec4,
    U64Vec2 as GU64Vec2, U64Vec3 as GU64Vec3, U64Vec4 as GU64Vec4,
};

// ─────────────────────────────────────────────────────────────────────────────
// I64Vec2
// ─────────────────────────────────────────────────────────────────────────────

fn bench_i64vec2(c: &mut Criterion) {
    let mut g = c.benchmark_group("i64vec2");

    let mm_a = I64Vec2::new(3, -7);
    let mm_b = I64Vec2::new(-2, 5);
    let mm_lo = I64Vec2::new(-10, -10);
    let mm_hi = I64Vec2::new(10, 10);

    let gl_a = GI64Vec2::new(3, -7);
    let gl_b = GI64Vec2::new(-2, 5);
    let gl_lo = GI64Vec2::new(-10, -10);
    let gl_hi = GI64Vec2::new(10, 10);

    g.bench_function("add/mid-math", |b| b.iter(|| black_box(mm_a) + black_box(mm_b)));
    g.bench_function("add/glam",     |b| b.iter(|| black_box(gl_a) + black_box(gl_b)));

    g.bench_function("sub/mid-math", |b| b.iter(|| black_box(mm_a) - black_box(mm_b)));
    g.bench_function("sub/glam",     |b| b.iter(|| black_box(gl_a) - black_box(gl_b)));

    g.bench_function("mul/mid-math", |b| b.iter(|| black_box(mm_a) * black_box(mm_b)));
    g.bench_function("mul/glam",     |b| b.iter(|| black_box(gl_a) * black_box(gl_b)));

    g.bench_function("scale/mid-math", |b| b.iter(|| black_box(mm_a) * 3i64));
    g.bench_function("scale/glam",     |b| b.iter(|| black_box(gl_a) * 3i64));

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

    // mid-math extras
    g.bench_function("wrapping_add/mid-math",   |b| b.iter(|| black_box(mm_a).wrapping_add(black_box(mm_b))));
    g.bench_function("saturating_add/mid-math", |b| b.iter(|| black_box(mm_a).saturating_add(black_box(mm_b))));

    g.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// I64Vec3
// ─────────────────────────────────────────────────────────────────────────────

fn bench_i64vec3(c: &mut Criterion) {
    let mut g = c.benchmark_group("i64vec3");

    let mm_a = I64Vec3::new(1, -2, 3);
    let mm_b = I64Vec3::new(-4, 5, -6);
    let mm_lo = I64Vec3::new(-10, -10, -10);
    let mm_hi = I64Vec3::new(10, 10, 10);

    let gl_a = GI64Vec3::new(1, -2, 3);
    let gl_b = GI64Vec3::new(-4, 5, -6);
    let gl_lo = GI64Vec3::new(-10, -10, -10);
    let gl_hi = GI64Vec3::new(10, 10, 10);

    g.bench_function("add/mid-math", |b| b.iter(|| black_box(mm_a) + black_box(mm_b)));
    g.bench_function("add/glam",     |b| b.iter(|| black_box(gl_a) + black_box(gl_b)));

    g.bench_function("sub/mid-math", |b| b.iter(|| black_box(mm_a) - black_box(mm_b)));
    g.bench_function("sub/glam",     |b| b.iter(|| black_box(gl_a) - black_box(gl_b)));

    g.bench_function("mul/mid-math", |b| b.iter(|| black_box(mm_a) * black_box(mm_b)));
    g.bench_function("mul/glam",     |b| b.iter(|| black_box(gl_a) * black_box(gl_b)));

    g.bench_function("scale/mid-math", |b| b.iter(|| black_box(mm_a) * 3i64));
    g.bench_function("scale/glam",     |b| b.iter(|| black_box(gl_a) * 3i64));

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
// I64Vec4
// ─────────────────────────────────────────────────────────────────────────────

fn bench_i64vec4(c: &mut Criterion) {
    let mut g = c.benchmark_group("i64vec4");

    let mm_a = I64Vec4::new(1, -2, 3, -4);
    let mm_b = I64Vec4::new(-5, 6, -7, 8);
    let mm_lo = I64Vec4::new(-10, -10, -10, -10);
    let mm_hi = I64Vec4::new(10, 10, 10, 10);

    let gl_a = GI64Vec4::new(1, -2, 3, -4);
    let gl_b = GI64Vec4::new(-5, 6, -7, 8);
    let gl_lo = GI64Vec4::new(-10, -10, -10, -10);
    let gl_hi = GI64Vec4::new(10, 10, 10, 10);

    g.bench_function("add/mid-math", |b| b.iter(|| black_box(mm_a) + black_box(mm_b)));
    g.bench_function("add/glam",     |b| b.iter(|| black_box(gl_a) + black_box(gl_b)));

    g.bench_function("sub/mid-math", |b| b.iter(|| black_box(mm_a) - black_box(mm_b)));
    g.bench_function("sub/glam",     |b| b.iter(|| black_box(gl_a) - black_box(gl_b)));

    g.bench_function("mul/mid-math", |b| b.iter(|| black_box(mm_a) * black_box(mm_b)));
    g.bench_function("mul/glam",     |b| b.iter(|| black_box(gl_a) * black_box(gl_b)));

    g.bench_function("scale/mid-math", |b| b.iter(|| black_box(mm_a) * 3i64));
    g.bench_function("scale/glam",     |b| b.iter(|| black_box(gl_a) * 3i64));

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
// U64Vec2
// ─────────────────────────────────────────────────────────────────────────────

fn bench_u64vec2(c: &mut Criterion) {
    let mut g = c.benchmark_group("u64vec2");

    let mm_a = U64Vec2::new(10, 3);
    let mm_b = U64Vec2::new(2, 7);
    let mm_lo = U64Vec2::new(1, 1);
    let mm_hi = U64Vec2::new(20, 20);

    let gl_a = GU64Vec2::new(10, 3);
    let gl_b = GU64Vec2::new(2, 7);
    let gl_lo = GU64Vec2::new(1, 1);
    let gl_hi = GU64Vec2::new(20, 20);

    g.bench_function("add/mid-math", |b| b.iter(|| black_box(mm_a) + black_box(mm_b)));
    g.bench_function("add/glam",     |b| b.iter(|| black_box(gl_a) + black_box(gl_b)));

    g.bench_function("mul/mid-math", |b| b.iter(|| black_box(mm_a) * black_box(mm_b)));
    g.bench_function("mul/glam",     |b| b.iter(|| black_box(gl_a) * black_box(gl_b)));

    g.bench_function("scale/mid-math", |b| b.iter(|| black_box(mm_a) * 3u64));
    g.bench_function("scale/glam",     |b| b.iter(|| black_box(gl_a) * 3u64));

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
// U64Vec3
// ─────────────────────────────────────────────────────────────────────────────

fn bench_u64vec3(c: &mut Criterion) {
    let mut g = c.benchmark_group("u64vec3");

    let mm_a = U64Vec3::new(10, 3, 7);
    let mm_b = U64Vec3::new(2, 7, 1);
    let mm_lo = U64Vec3::new(1, 1, 1);
    let mm_hi = U64Vec3::new(20, 20, 20);

    let gl_a = GU64Vec3::new(10, 3, 7);
    let gl_b = GU64Vec3::new(2, 7, 1);
    let gl_lo = GU64Vec3::new(1, 1, 1);
    let gl_hi = GU64Vec3::new(20, 20, 20);

    g.bench_function("add/mid-math", |b| b.iter(|| black_box(mm_a) + black_box(mm_b)));
    g.bench_function("add/glam",     |b| b.iter(|| black_box(gl_a) + black_box(gl_b)));

    g.bench_function("mul/mid-math", |b| b.iter(|| black_box(mm_a) * black_box(mm_b)));
    g.bench_function("mul/glam",     |b| b.iter(|| black_box(gl_a) * black_box(gl_b)));

    g.bench_function("scale/mid-math", |b| b.iter(|| black_box(mm_a) * 3u64));
    g.bench_function("scale/glam",     |b| b.iter(|| black_box(gl_a) * 3u64));

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
// U64Vec4
// ─────────────────────────────────────────────────────────────────────────────

fn bench_u64vec4(c: &mut Criterion) {
    let mut g = c.benchmark_group("u64vec4");

    let mm_a = U64Vec4::new(10, 3, 7, 1);
    let mm_b = U64Vec4::new(2, 7, 1, 5);
    let mm_lo = U64Vec4::new(1, 1, 1, 1);
    let mm_hi = U64Vec4::new(20, 20, 20, 20);

    let gl_a = GU64Vec4::new(10, 3, 7, 1);
    let gl_b = GU64Vec4::new(2, 7, 1, 5);
    let gl_lo = GU64Vec4::new(1, 1, 1, 1);
    let gl_hi = GU64Vec4::new(20, 20, 20, 20);

    g.bench_function("add/mid-math", |b| b.iter(|| black_box(mm_a) + black_box(mm_b)));
    g.bench_function("add/glam",     |b| b.iter(|| black_box(gl_a) + black_box(gl_b)));

    g.bench_function("mul/mid-math", |b| b.iter(|| black_box(mm_a) * black_box(mm_b)));
    g.bench_function("mul/glam",     |b| b.iter(|| black_box(gl_a) * black_box(gl_b)));

    g.bench_function("scale/mid-math", |b| b.iter(|| black_box(mm_a) * 3u64));
    g.bench_function("scale/glam",     |b| b.iter(|| black_box(gl_a) * 3u64));

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
    bench_i64vec2,
    bench_i64vec3,
    bench_i64vec4,
    bench_u64vec2,
    bench_u64vec3,
    bench_u64vec4,
);
criterion_main!(benches);
