// crates/mid-math/benches/vs_fixed.rs
//! Fixed-point arithmetic benchmarks.
//!
//! Groups:
//!   fixed/scalar_ops    — Fixed16 add/sub/mul/div vs f32 equivalent
//!   fixed/vec2_ops      — FixedVec2 ops: dot, scale, lerp, perp_dot
//!   fixed/vec3_ops      — FixedVec3 ops: dot, cross, scale, lerp
//!   fixed/boundary      — from_f32 / to_f32 conversion cost
//!   fixed/checked       — checked_mul / saturating_add overhead
//!   fixed/bulk_100k     — 100k entity position integration throughput

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use mid_math::{Fixed16, Fixed16Vec2, Fixed16Vec3, Fixed12};

// ── Scalar ops ────────────────────────────────────────────────────────────────

fn bench_scalar_ops(c: &mut Criterion) {
    let mut g = c.benchmark_group("fixed/scalar_ops");

    let a = Fixed16::from_f32(3.75);
    let b = Fixed16::from_f32(1.25);
    let fa = 3.75_f32;
    let fb = 1.25_f32;

    // Fixed vs f32 — shows determinism cost
    g.bench_function("fixed16_add", |bench| {
        bench.iter(|| black_box(black_box(a) + black_box(b)))
    });
    g.bench_function("f32_add_baseline", |bench| {
        bench.iter(|| black_box(black_box(fa) + black_box(fb)))
    });
    g.bench_function("fixed16_sub", |bench| {
        bench.iter(|| black_box(black_box(a) - black_box(b)))
    });
    g.bench_function("fixed16_mul", |bench| {
        bench.iter(|| black_box(black_box(a).fixed_mul(black_box(b))))
    });
    g.bench_function("f32_mul_baseline", |bench| {
        bench.iter(|| black_box(black_box(fa) * black_box(fb)))
    });
    g.bench_function("fixed16_div", |bench| {
        bench.iter(|| black_box(black_box(a).fixed_div(black_box(b))))
    });
    g.bench_function("f32_div_baseline", |bench| {
        bench.iter(|| black_box(black_box(fa) / black_box(fb)))
    });
    g.bench_function("fixed16_lerp", |bench| {
        let t = Fixed16::from_f32(0.5);
        bench.iter(|| black_box(black_box(a).lerp(black_box(b), black_box(t))))
    });
    g.bench_function("fixed16_floor", |bench| {
        bench.iter(|| black_box(black_box(a).floor()))
    });
    g.bench_function("fixed16_ceil", |bench| {
        bench.iter(|| black_box(black_box(a).ceil()))
    });
    g.bench_function("fixed16_abs", |bench| {
        bench.iter(|| black_box(black_box(-a).abs()))
    });
    g.finish();
}

// ── FixedVec2 ops ─────────────────────────────────────────────────────────────

fn bench_vec2_ops(c: &mut Criterion) {
    let mut g = c.benchmark_group("fixed/vec2_ops");

    let a = Fixed16Vec2::from_f32(3.0, 4.0);
    let b = Fixed16Vec2::from_f32(1.5, -2.0);
    let s = Fixed16::from_f32(2.5);
    let t = Fixed16::from_f32(0.5);

    g.bench_function("add", |bench| {
        bench.iter(|| black_box(black_box(a) + black_box(b)))
    });
    g.bench_function("sub", |bench| {
        bench.iter(|| black_box(black_box(a) - black_box(b)))
    });
    g.bench_function("dot", |bench| {
        bench.iter(|| black_box(black_box(a).dot(black_box(b))))
    });
    g.bench_function("length_sq", |bench| {
        bench.iter(|| black_box(black_box(a).length_sq()))
    });
    g.bench_function("scale", |bench| {
        bench.iter(|| black_box(black_box(a).scale(black_box(s))))
    });
    g.bench_function("lerp", |bench| {
        bench.iter(|| black_box(black_box(a).lerp(black_box(b), black_box(t))))
    });
    g.bench_function("perp_dot", |bench| {
        bench.iter(|| black_box(black_box(a).perp_dot(black_box(b))))
    });
    g.bench_function("neg", |bench| {
        bench.iter(|| black_box(-black_box(a)))
    });
    g.bench_function("manhattan_distance", |bench| {
        bench.iter(|| black_box(black_box(a).manhattan_distance(black_box(b))))
    });
    g.finish();
}

// ── FixedVec3 ops ─────────────────────────────────────────────────────────────

fn bench_vec3_ops(c: &mut Criterion) {
    let mut g = c.benchmark_group("fixed/vec3_ops");

    let a = Fixed16Vec3::from_f32(1.0, 2.0, 3.0);
    let b = Fixed16Vec3::from_f32(4.0, 5.0, 6.0);
    let s = Fixed16::from_f32(2.0);
    let t = Fixed16::from_f32(0.3);

    g.bench_function("add", |bench| {
        bench.iter(|| black_box(black_box(a) + black_box(b)))
    });
    g.bench_function("dot", |bench| {
        bench.iter(|| black_box(black_box(a).dot(black_box(b))))
    });
    g.bench_function("cross", |bench| {
        bench.iter(|| black_box(black_box(a).cross(black_box(b))))
    });
    g.bench_function("length_sq", |bench| {
        bench.iter(|| black_box(black_box(a).length_sq()))
    });
    g.bench_function("scale", |bench| {
        bench.iter(|| black_box(black_box(a).scale(black_box(s))))
    });
    g.bench_function("lerp", |bench| {
        bench.iter(|| black_box(black_box(a).lerp(black_box(b), black_box(t))))
    });
    g.bench_function("mul_elem", |bench| {
        bench.iter(|| black_box(black_box(a).mul_elem(black_box(b))))
    });
    g.bench_function("manhattan_distance", |bench| {
        bench.iter(|| black_box(black_box(a).manhattan_distance(black_box(b))))
    });
    g.finish();
}

// ── Boundary conversion ───────────────────────────────────────────────────────

fn bench_boundary(c: &mut Criterion) {
    let mut g = c.benchmark_group("fixed/boundary");

    g.bench_function("from_f32", |b| {
        b.iter(|| black_box(Fixed16::from_f32(black_box(3.14159_f32))))
    });
    g.bench_function("to_f32", |b| {
        let f = Fixed16::from_f32(3.14159);
        b.iter(|| black_box(black_box(f).to_f32()))
    });
    g.bench_function("from_i32", |b| {
        b.iter(|| black_box(Fixed16::from_i32(black_box(42_i32))))
    });
    g.bench_function("to_i32_trunc", |b| {
        let f = Fixed16::from_f32(7.99);
        b.iter(|| black_box(black_box(f).to_i32_trunc()))
    });
    g.bench_function("vec3_from_f32", |b| {
        b.iter(|| black_box(Fixed16Vec3::from_f32(
            black_box(1.0_f32), black_box(2.0_f32), black_box(3.0_f32)
        )))
    });
    g.bench_function("vec3_to_vec3", |b| {
        let fv = Fixed16Vec3::from_f32(1.5, 2.5, 3.5);
        b.iter(|| black_box(black_box(fv).to_vec3()))
    });
    g.finish();
}

// ── Checked / saturating ops ─────────────────────────────────────────────────

fn bench_checked(c: &mut Criterion) {
    let mut g = c.benchmark_group("fixed/checked");

    let a = Fixed12::from_f32(100.0);
    let b = Fixed12::from_f32(3.0);

    g.bench_function("checked_mul_ok", |bench| {
        bench.iter(|| black_box(black_box(a).checked_mul(black_box(b))))
    });
    g.bench_function("checked_div_ok", |bench| {
        bench.iter(|| black_box(black_box(a).checked_div(black_box(b))))
    });
    g.bench_function("saturating_add", |bench| {
        bench.iter(|| black_box(black_box(a).saturating_add(black_box(b))))
    });
    g.bench_function("saturating_mul", |bench| {
        bench.iter(|| black_box(black_box(a).saturating_mul(black_box(b))))
    });
    g.finish();
}

// ── Bulk 100k integration ─────────────────────────────────────────────────────

fn bench_bulk_100k(c: &mut Criterion) {
    let mut g = c.benchmark_group("fixed/bulk_100k");
    g.sample_size(20);

    let positions: Vec<Fixed16Vec3> = (0..100_000)
        .map(|i| Fixed16Vec3::from_f32(i as f32 * 0.01, 0.0, 0.0))
        .collect();
    let velocities: Vec<Fixed16Vec3> = (0..100_000)
        .map(|_| Fixed16Vec3::from_f32(0.016, 0.0, 0.0)) // ~60Hz dt
        .collect();

    g.bench_function("integrate_positions_100k", |b| {
        b.iter(|| {
            let mut out = positions.clone();
            for (p, v) in out.iter_mut().zip(black_box(&velocities)) {
                *p = *p + *v;
            }
            black_box(out)
        })
    });

    g.bench_function("dot_products_100k", |b| {
        b.iter(|| {
            let mut sum = Fixed16::ZERO;
            for (a, b_val) in black_box(&positions).iter().zip(&velocities) {
                sum = sum + a.dot(*b_val);
            }
            black_box(sum)
        })
    });

    g.bench_function("cross_products_100k", |b| {
        b.iter(|| {
            let mut last = Fixed16Vec3::ZERO;
            for (a, b_val) in black_box(&positions).iter().zip(&velocities) {
                last = a.cross(*b_val);
            }
            black_box(last)
        })
    });

    g.finish();
}

criterion_group!(
    benches,
    bench_scalar_ops,
    bench_vec2_ops,
    bench_vec3_ops,
    bench_boundary,
    bench_checked,
    bench_bulk_100k,
);
criterion_main!(benches);
