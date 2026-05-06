// crates/mid-math/benches/vs_int32.rs
//! Integer vector benchmark: mid-math IVec/UVec vs glam IVec/UVec.
//!
//! Both libraries are scalar i32/u32 only — no SIMD for integers.
//! The goal is confirming we have no implementation overhead vs glam.
//! Any gap here is pure struct layout or missed inlining, not a
//! fundamental algorithmic difference.
//!
//! Operations benchmarked per type:
//!   add, sub, mul (element-wise), scale (scalar), dot, min, max,
//!   clamp, abs (IVec only), cross (IVec3/UVec3 only).
//!
//! Note: glam has no saturating_add/wrapping_add benches here because
//! glam does not expose those — they are mid-math extras.
//!
//! Run: cargo bench --bench vs_int32 -p mid-math
//! HTML report: target/criterion/report/index.html

use criterion::{black_box, criterion_group, criterion_main, Criterion};

// ── mid-math ──────────────────────────────────────────────────────────────────
use mid_math::{IVec2, IVec3, IVec4, UVec2, UVec3, UVec4};

// ── glam ─────────────────────────────────────────────────────────────────────
use glam::{
    IVec2 as GIVec2, IVec3 as GIVec3, IVec4 as GIVec4,
    UVec2 as GUVec2, UVec3 as GUVec3, UVec4 as GUVec4,
};

// ─────────────────────────────────────────────────────────────────────────────
// IVec2
// ─────────────────────────────────────────────────────────────────────────────

fn bench_ivec2(c: &mut Criterion) {
    let mut g = c.benchmark_group("ivec2");

    let mm_a = IVec2::new(3, -7);
    let mm_b = IVec2::new(-2, 5);
    let mm_lo = IVec2::new(-10, -10);
    let mm_hi = IVec2::new(10, 10);

    let gl_a = GIVec2::new(3, -7);
    let gl_b = GIVec2::new(-2, 5);
    let gl_lo = GIVec2::new(-10, -10);
    let gl_hi = GIVec2::new(10, 10);

    g.bench_function("add/mid-math", |b| b.iter(|| black_box(mm_a) + black_box(mm_b)));
    g.bench_function("add/glam",     |b| b.iter(|| black_box(gl_a) + black_box(gl_b)));

    g.bench_function("sub/mid-math", |b| b.iter(|| black_box(mm_a) - black_box(mm_b)));
    g.bench_function("sub/glam",     |b| b.iter(|| black_box(gl_a) - black_box(gl_b)));

    g.bench_function("mul/mid-math", |b| b.iter(|| black_box(mm_a) * black_box(mm_b)));
    g.bench_function("mul/glam",     |b| b.iter(|| black_box(gl_a) * black_box(gl_b)));

    g.bench_function("scale/mid-math", |b| b.iter(|| black_box(mm_a) * 3i32));
    g.bench_function("scale/glam",     |b| b.iter(|| black_box(gl_a) * 3i32));

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

    // mid-math extras — no glam equivalent, regression-only
    g.bench_function("wrapping_add/mid-math",   |b| b.iter(|| black_box(mm_a).wrapping_add(black_box(mm_b))));
    g.bench_function("saturating_add/mid-math", |b| b.iter(|| black_box(mm_a).saturating_add(black_box(mm_b))));

    g.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// IVec3
// ─────────────────────────────────────────────────────────────────────────────

fn bench_ivec3(c: &mut Criterion) {
    let mut g = c.benchmark_group("ivec3");

    let mm_a = IVec3::new(1, -2, 3);
    let mm_b = IVec3::new(-4, 5, -6);
    let mm_lo = IVec3::new(-10, -10, -10);
    let mm_hi = IVec3::new(10, 10, 10);

    let gl_a = GIVec3::new(1, -2, 3);
    let gl_b = GIVec3::new(-4, 5, -6);
    let gl_lo = GIVec3::new(-10, -10, -10);
    let gl_hi = GIVec3::new(10, 10, 10);

    g.bench_function("add/mid-math", |b| b.iter(|| black_box(mm_a) + black_box(mm_b)));
    g.bench_function("add/glam",     |b| b.iter(|| black_box(gl_a) + black_box(gl_b)));

    g.bench_function("sub/mid-math", |b| b.iter(|| black_box(mm_a) - black_box(mm_b)));
    g.bench_function("sub/glam",     |b| b.iter(|| black_box(gl_a) - black_box(gl_b)));

    g.bench_function("mul/mid-math", |b| b.iter(|| black_box(mm_a) * black_box(mm_b)));
    g.bench_function("mul/glam",     |b| b.iter(|| black_box(gl_a) * black_box(gl_b)));

    g.bench_function("scale/mid-math", |b| b.iter(|| black_box(mm_a) * 3i32));
    g.bench_function("scale/glam",     |b| b.iter(|| black_box(gl_a) * 3i32));

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
// IVec4
// ─────────────────────────────────────────────────────────────────────────────

fn bench_ivec4(c: &mut Criterion) {
    let mut g = c.benchmark_group("ivec4");

    let mm_a = IVec4::new(1, -2, 3, -4);
    let mm_b = IVec4::new(-5, 6, -7, 8);
    let mm_lo = IVec4::new(-10, -10, -10, -10);
    let mm_hi = IVec4::new(10, 10, 10, 10);

    let gl_a = GIVec4::new(1, -2, 3, -4);
    let gl_b = GIVec4::new(-5, 6, -7, 8);
    let gl_lo = GIVec4::new(-10, -10, -10, -10);
    let gl_hi = GIVec4::new(10, 10, 10, 10);

    g.bench_function("add/mid-math", |b| b.iter(|| black_box(mm_a) + black_box(mm_b)));
    g.bench_function("add/glam",     |b| b.iter(|| black_box(gl_a) + black_box(gl_b)));

    g.bench_function("sub/mid-math", |b| b.iter(|| black_box(mm_a) - black_box(mm_b)));
    g.bench_function("sub/glam",     |b| b.iter(|| black_box(gl_a) - black_box(gl_b)));

    g.bench_function("mul/mid-math", |b| b.iter(|| black_box(mm_a) * black_box(mm_b)));
    g.bench_function("mul/glam",     |b| b.iter(|| black_box(gl_a) * black_box(gl_b)));

    g.bench_function("scale/mid-math", |b| b.iter(|| black_box(mm_a) * 3i32));
    g.bench_function("scale/glam",     |b| b.iter(|| black_box(gl_a) * 3i32));

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
// UVec2
// ─────────────────────────────────────────────────────────────────────────────

fn bench_uvec2(c: &mut Criterion) {
    let mut g = c.benchmark_group("uvec2");

    let mm_a = UVec2::new(10, 3);
    let mm_b = UVec2::new(2, 7);
    let mm_lo = UVec2::new(1, 1);
    let mm_hi = UVec2::new(20, 20);

    let gl_a = GUVec2::new(10, 3);
    let gl_b = GUVec2::new(2, 7);
    let gl_lo = GUVec2::new(1, 1);
    let gl_hi = GUVec2::new(20, 20);

    g.bench_function("add/mid-math", |b| b.iter(|| black_box(mm_a) + black_box(mm_b)));
    g.bench_function("add/glam",     |b| b.iter(|| black_box(gl_a) + black_box(gl_b)));

    g.bench_function("mul/mid-math", |b| b.iter(|| black_box(mm_a) * black_box(mm_b)));
    g.bench_function("mul/glam",     |b| b.iter(|| black_box(gl_a) * black_box(gl_b)));

    g.bench_function("scale/mid-math", |b| b.iter(|| black_box(mm_a) * 3u32));
    g.bench_function("scale/glam",     |b| b.iter(|| black_box(gl_a) * 3u32));

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
// UVec3
// ─────────────────────────────────────────────────────────────────────────────

fn bench_uvec3(c: &mut Criterion) {
    let mut g = c.benchmark_group("uvec3");

    let mm_a = UVec3::new(10, 3, 7);
    let mm_b = UVec3::new(2, 7, 1);
    let mm_lo = UVec3::new(1, 1, 1);
    let mm_hi = UVec3::new(20, 20, 20);

    let gl_a = GUVec3::new(10, 3, 7);
    let gl_b = GUVec3::new(2, 7, 1);
    let gl_lo = GUVec3::new(1, 1, 1);
    let gl_hi = GUVec3::new(20, 20, 20);

    g.bench_function("add/mid-math", |b| b.iter(|| black_box(mm_a) + black_box(mm_b)));
    g.bench_function("add/glam",     |b| b.iter(|| black_box(gl_a) + black_box(gl_b)));

    g.bench_function("mul/mid-math", |b| b.iter(|| black_box(mm_a) * black_box(mm_b)));
    g.bench_function("mul/glam",     |b| b.iter(|| black_box(gl_a) * black_box(gl_b)));

    g.bench_function("scale/mid-math", |b| b.iter(|| black_box(mm_a) * 3u32));
    g.bench_function("scale/glam",     |b| b.iter(|| black_box(gl_a) * 3u32));

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
// UVec4
// ─────────────────────────────────────────────────────────────────────────────

fn bench_uvec4(c: &mut Criterion) {
    let mut g = c.benchmark_group("uvec4");

    let mm_a = UVec4::new(10, 3, 7, 1);
    let mm_b = UVec4::new(2, 7, 1, 5);
    let mm_lo = UVec4::new(1, 1, 1, 1);
    let mm_hi = UVec4::new(20, 20, 20, 20);

    let gl_a = GUVec4::new(10, 3, 7, 1);
    let gl_b = GUVec4::new(2, 7, 1, 5);
    let gl_lo = GUVec4::new(1, 1, 1, 1);
    let gl_hi = GUVec4::new(20, 20, 20, 20);

    g.bench_function("add/mid-math", |b| b.iter(|| black_box(mm_a) + black_box(mm_b)));
    g.bench_function("add/glam",     |b| b.iter(|| black_box(gl_a) + black_box(gl_b)));

    g.bench_function("mul/mid-math", |b| b.iter(|| black_box(mm_a) * black_box(mm_b)));
    g.bench_function("mul/glam",     |b| b.iter(|| black_box(gl_a) * black_box(gl_b)));

    g.bench_function("scale/mid-math", |b| b.iter(|| black_box(mm_a) * 3u32));
    g.bench_function("scale/glam",     |b| b.iter(|| black_box(gl_a) * 3u32));

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
    bench_ivec2,
    bench_ivec3,
    bench_ivec4,
    bench_uvec2,
    bench_uvec3,
    bench_uvec4,
);
criterion_main!(benches);
