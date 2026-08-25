// crates/mid-math/benches/vs_int16.rs
//! Integer vector benchmark: mid-math I16Vec/U16Vec vs glam I16Vec/U16Vec.
//!
//! Mirrors vs_int32.rs exactly in structure/scope. Both libraries are
//! scalar i16/u16 only — no SIMD for integers.
//!
//! Same `dot()` widening difference as vs_int8.rs, not an exception:
//! mid-math's I16Vec4::dot returns i32 (widened, overflow-safe);
//! glam's returns i16 (not widened — checked its source directly,
//! glam never widens dot() at any integer width, mid-math always does).
//! Both benched with their real return types.
//!
//! Operations benchmarked per type: add, sub, mul (element-wise), scale
//! (scalar), dot, min, max, clamp, abs (I16Vec only), cross (I16Vec3/
//! U16Vec3 only), wrapping_add, saturating_add (mid-math extras).
//!
//! Run: cargo bench --bench vs_int16 -p mid-math
//! HTML report: target/criterion/report/index.html

use criterion::{black_box, criterion_group, criterion_main, Criterion};

// ── mid-math ──────────────────────────────────────────────────────────────────
use mid_math::{I16Vec2, I16Vec3, I16Vec4, U16Vec2, U16Vec3, U16Vec4};

// ── glam ─────────────────────────────────────────────────────────────────────
use glam::{
    I16Vec2 as GI16Vec2, I16Vec3 as GI16Vec3, I16Vec4 as GI16Vec4,
    U16Vec2 as GU16Vec2, U16Vec3 as GU16Vec3, U16Vec4 as GU16Vec4,
};

// ─────────────────────────────────────────────────────────────────────────────
// I16Vec2
// ─────────────────────────────────────────────────────────────────────────────

fn bench_i16vec2(c: &mut Criterion) {
    let mut g = c.benchmark_group("i16vec2");

    let mm_a = I16Vec2::new(300, -700);
    let mm_b = I16Vec2::new(-200, 500);
    let mm_lo = I16Vec2::new(-1000, -1000);
    let mm_hi = I16Vec2::new(1000, 1000);

    let gl_a = GI16Vec2::new(300, -700);
    let gl_b = GI16Vec2::new(-200, 500);
    let gl_lo = GI16Vec2::new(-1000, -1000);
    let gl_hi = GI16Vec2::new(1000, 1000);

    g.bench_function("add/mid-math", |b| b.iter(|| black_box(mm_a) + black_box(mm_b)));
    g.bench_function("add/glam",     |b| b.iter(|| black_box(gl_a) + black_box(gl_b)));

    g.bench_function("sub/mid-math", |b| b.iter(|| black_box(mm_a) - black_box(mm_b)));
    g.bench_function("sub/glam",     |b| b.iter(|| black_box(gl_a) - black_box(gl_b)));

    g.bench_function("mul/mid-math", |b| b.iter(|| black_box(mm_a) * black_box(mm_b)));
    g.bench_function("mul/glam",     |b| b.iter(|| black_box(gl_a) * black_box(gl_b)));

    g.bench_function("scale/mid-math", |b| b.iter(|| black_box(mm_a) * 3i16));
    g.bench_function("scale/glam",     |b| b.iter(|| black_box(gl_a) * 3i16));

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
// I16Vec3
// ─────────────────────────────────────────────────────────────────────────────

fn bench_i16vec3(c: &mut Criterion) {
    let mut g = c.benchmark_group("i16vec3");

    let mm_a = I16Vec3::new(100, -200, 300);
    let mm_b = I16Vec3::new(-400, 500, -600);
    let mm_lo = I16Vec3::new(-1000, -1000, -1000);
    let mm_hi = I16Vec3::new(1000, 1000, 1000);

    let gl_a = GI16Vec3::new(100, -200, 300);
    let gl_b = GI16Vec3::new(-400, 500, -600);
    let gl_lo = GI16Vec3::new(-1000, -1000, -1000);
    let gl_hi = GI16Vec3::new(1000, 1000, 1000);

    g.bench_function("add/mid-math", |b| b.iter(|| black_box(mm_a) + black_box(mm_b)));
    g.bench_function("add/glam",     |b| b.iter(|| black_box(gl_a) + black_box(gl_b)));

    g.bench_function("sub/mid-math", |b| b.iter(|| black_box(mm_a) - black_box(mm_b)));
    g.bench_function("sub/glam",     |b| b.iter(|| black_box(gl_a) - black_box(gl_b)));

    g.bench_function("mul/mid-math", |b| b.iter(|| black_box(mm_a) * black_box(mm_b)));
    g.bench_function("mul/glam",     |b| b.iter(|| black_box(gl_a) * black_box(gl_b)));

    g.bench_function("scale/mid-math", |b| b.iter(|| black_box(mm_a) * 3i16));
    g.bench_function("scale/glam",     |b| b.iter(|| black_box(gl_a) * 3i16));

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
// I16Vec4
// ─────────────────────────────────────────────────────────────────────────────

fn bench_i16vec4(c: &mut Criterion) {
    let mut g = c.benchmark_group("i16vec4");

    let mm_a = I16Vec4::new(100, -200, 300, -400);
    let mm_b = I16Vec4::new(-500, 600, -700, 800);
    let mm_lo = I16Vec4::new(-1000, -1000, -1000, -1000);
    let mm_hi = I16Vec4::new(1000, 1000, 1000, 1000);

    let gl_a = GI16Vec4::new(100, -200, 300, -400);
    let gl_b = GI16Vec4::new(-500, 600, -700, 800);
    let gl_lo = GI16Vec4::new(-1000, -1000, -1000, -1000);
    let gl_hi = GI16Vec4::new(1000, 1000, 1000, 1000);

    g.bench_function("add/mid-math", |b| b.iter(|| black_box(mm_a) + black_box(mm_b)));
    g.bench_function("add/glam",     |b| b.iter(|| black_box(gl_a) + black_box(gl_b)));

    g.bench_function("sub/mid-math", |b| b.iter(|| black_box(mm_a) - black_box(mm_b)));
    g.bench_function("sub/glam",     |b| b.iter(|| black_box(gl_a) - black_box(gl_b)));

    g.bench_function("mul/mid-math", |b| b.iter(|| black_box(mm_a) * black_box(mm_b)));
    g.bench_function("mul/glam",     |b| b.iter(|| black_box(gl_a) * black_box(gl_b)));

    g.bench_function("scale/mid-math", |b| b.iter(|| black_box(mm_a) * 3i16));
    g.bench_function("scale/glam",     |b| b.iter(|| black_box(gl_a) * 3i16));

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
// U16Vec2
// ─────────────────────────────────────────────────────────────────────────────

fn bench_u16vec2(c: &mut Criterion) {
    let mut g = c.benchmark_group("u16vec2");

    let mm_a = U16Vec2::new(1000, 300);
    let mm_b = U16Vec2::new(200, 700);
    let mm_lo = U16Vec2::new(100, 100);
    let mm_hi = U16Vec2::new(2000, 2000);

    let gl_a = GU16Vec2::new(1000, 300);
    let gl_b = GU16Vec2::new(200, 700);
    let gl_lo = GU16Vec2::new(100, 100);
    let gl_hi = GU16Vec2::new(2000, 2000);

    g.bench_function("add/mid-math", |b| b.iter(|| black_box(mm_a) + black_box(mm_b)));
    g.bench_function("add/glam",     |b| b.iter(|| black_box(gl_a) + black_box(gl_b)));

    g.bench_function("mul/mid-math", |b| b.iter(|| black_box(mm_a) * black_box(mm_b)));
    g.bench_function("mul/glam",     |b| b.iter(|| black_box(gl_a) * black_box(gl_b)));

    g.bench_function("scale/mid-math", |b| b.iter(|| black_box(mm_a) * 3u16));
    g.bench_function("scale/glam",     |b| b.iter(|| black_box(gl_a) * 3u16));

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
// U16Vec3
// ─────────────────────────────────────────────────────────────────────────────

fn bench_u16vec3(c: &mut Criterion) {
    let mut g = c.benchmark_group("u16vec3");

    let mm_a = U16Vec3::new(1000, 300, 700);
    let mm_b = U16Vec3::new(200, 700, 100);
    let mm_lo = U16Vec3::new(100, 100, 100);
    let mm_hi = U16Vec3::new(2000, 2000, 2000);

    let gl_a = GU16Vec3::new(1000, 300, 700);
    let gl_b = GU16Vec3::new(200, 700, 100);
    let gl_lo = GU16Vec3::new(100, 100, 100);
    let gl_hi = GU16Vec3::new(2000, 2000, 2000);

    g.bench_function("add/mid-math", |b| b.iter(|| black_box(mm_a) + black_box(mm_b)));
    g.bench_function("add/glam",     |b| b.iter(|| black_box(gl_a) + black_box(gl_b)));

    g.bench_function("mul/mid-math", |b| b.iter(|| black_box(mm_a) * black_box(mm_b)));
    g.bench_function("mul/glam",     |b| b.iter(|| black_box(gl_a) * black_box(gl_b)));

    g.bench_function("scale/mid-math", |b| b.iter(|| black_box(mm_a) * 3u16));
    g.bench_function("scale/glam",     |b| b.iter(|| black_box(gl_a) * 3u16));

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
// U16Vec4
// ─────────────────────────────────────────────────────────────────────────────

fn bench_u16vec4(c: &mut Criterion) {
    let mut g = c.benchmark_group("u16vec4");

    let mm_a = U16Vec4::new(1000, 300, 700, 100);
    let mm_b = U16Vec4::new(200, 700, 100, 500);
    let mm_lo = U16Vec4::new(100, 100, 100, 100);
    let mm_hi = U16Vec4::new(2000, 2000, 2000, 2000);

    let gl_a = GU16Vec4::new(1000, 300, 700, 100);
    let gl_b = GU16Vec4::new(200, 700, 100, 500);
    let gl_lo = GU16Vec4::new(100, 100, 100, 100);
    let gl_hi = GU16Vec4::new(2000, 2000, 2000, 2000);

    g.bench_function("add/mid-math", |b| b.iter(|| black_box(mm_a) + black_box(mm_b)));
    g.bench_function("add/glam",     |b| b.iter(|| black_box(gl_a) + black_box(gl_b)));

    g.bench_function("mul/mid-math", |b| b.iter(|| black_box(mm_a) * black_box(mm_b)));
    g.bench_function("mul/glam",     |b| b.iter(|| black_box(gl_a) * black_box(gl_b)));

    g.bench_function("scale/mid-math", |b| b.iter(|| black_box(mm_a) * 3u16));
    g.bench_function("scale/glam",     |b| b.iter(|| black_box(gl_a) * 3u16));

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
    bench_i16vec2,
    bench_i16vec3,
    bench_i16vec4,
    bench_u16vec2,
    bench_u16vec3,
    bench_u16vec4,
);
criterion_main!(benches);
