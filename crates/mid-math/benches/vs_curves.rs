// crates/mid-math/benches/vs_curves.rs
//! Criterion benchmarks for mid-math curve evaluation.
//!
//! Groups:
//!   curves/bezier_quad_evaluate   — QuadraticBezier::evaluate (f32 scalar, Vec3)
//!   curves/bezier_cubic_evaluate  — CubicBezier::evaluate (f32 scalar, Vec3)
//!   curves/catmull_rom_evaluate   — CatmullRom::evaluate (4-pt, 8-pt; Uniform, Centripetal)
//!   curves/hermite_evaluate       — HermiteSpline::evaluate (2-key, 4-key)
//!   curves/bspline_evaluate       — BSpline::evaluate (4-pt, 8-pt)
//!   curves/cardinal_evaluate      — CardinalSpline (t=0 vs t=0.5)
//!   curves/sample_bulk            — sample_uniform 1000 points, each curve type

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use mid_math::{
    BSpline, CardinalSpline, CatmullRom, CatmullRomAlpha, CubicBezier,
    HermiteKey, HermiteSpline, QuadraticBezier, Vec3,
};

// ── Test data helpers ─────────────────────────────────────────────────────────

fn make_vec3s(n: usize) -> Vec<Vec3> {
    (0..n)
        .map(|i| Vec3::new(i as f32, (i as f32).sin(), (i as f32 * 0.3).cos()))
        .collect()
}

fn make_hermite_keys(n: usize) -> Vec<HermiteKey<Vec3>> {
    (0..n)
        .map(|i| {
            let p = Vec3::new(i as f32, (i as f32).sin(), 0.0);
            let t = Vec3::new(1.0, (i as f32).cos(), 0.0);
            HermiteKey::smooth(p, t)
        })
        .collect()
}

// ── Quadratic Bézier ──────────────────────────────────────────────────────────

fn bench_bezier_quad(c: &mut Criterion) {
    let mut g = c.benchmark_group("curves/bezier_quad_evaluate");

    let p0 = Vec3::new(0.0, 0.0, 0.0);
    let p1 = Vec3::new(1.0, 2.0, 0.5);
    let p2 = Vec3::new(3.0, 0.0, 0.0);
    let b  = QuadraticBezier::new(p0, p1, p2);

    g.bench_function("vec3_t05",  |x| x.iter(|| b.evaluate(black_box(0.5))));
    g.bench_function("vec3_t025", |x| x.iter(|| b.evaluate(black_box(0.25))));

    let bf = QuadraticBezier::new(0.0f32, 2.0f32, 4.0f32);
    g.bench_function("f32_t05",   |x| x.iter(|| bf.evaluate(black_box(0.5))));

    g.finish();
}

// ── Cubic Bézier ─────────────────────────────────────────────────────────────

fn bench_bezier_cubic(c: &mut Criterion) {
    let mut g = c.benchmark_group("curves/bezier_cubic_evaluate");

    let pts = make_vec3s(4);
    let b   = CubicBezier::new(pts[0], pts[1], pts[2], pts[3]);

    g.bench_function("vec3_t05",      |x| x.iter(|| b.evaluate(black_box(0.5))));
    g.bench_function("vec3_tangent",  |x| x.iter(|| b.tangent(black_box(0.5))));
    g.bench_function("vec3_split",    |x| x.iter(|| b.split(black_box(0.4))));
    g.bench_function("vec3_arc_len",  |x| x.iter(|| b.arc_length(black_box(32))));

    let bf = CubicBezier::new(0.0f32, 1.0f32, 2.0f32, 3.0f32);
    g.bench_function("f32_t05",       |x| x.iter(|| bf.evaluate(black_box(0.5))));

    g.finish();
}

// ── Catmull-Rom ───────────────────────────────────────────────────────────────

fn bench_catmull_rom(c: &mut Criterion) {
    let mut g = c.benchmark_group("curves/catmull_rom_evaluate");

    let pts4 = make_vec3s(4);
    let pts8 = make_vec3s(8);

    for (label, pts, t) in &[
        ("uniform_4pt",       &pts4, 1.5f32),
        ("centripetal_4pt",   &pts4, 1.5),
        ("uniform_8pt",       &pts8, 3.5),
        ("centripetal_8pt",   &pts8, 3.5),
    ] {
        let alpha = if label.contains("centripetal") {
            CatmullRomAlpha::Centripetal
        } else {
            CatmullRomAlpha::Uniform
        };
        let cr = CatmullRom::with_alpha(pts.clone(), alpha);
        let t  = *t;
        g.bench_function(*label, |x| x.iter(|| cr.evaluate(black_box(t))));
    }

    g.finish();
}

// ── Hermite ───────────────────────────────────────────────────────────────────

fn bench_hermite(c: &mut Criterion) {
    let mut g = c.benchmark_group("curves/hermite_evaluate");

    let keys2 = make_hermite_keys(2);
    let keys4 = make_hermite_keys(4);
    let keys8 = make_hermite_keys(8);

    let sp2 = HermiteSpline::new(keys2);
    let sp4 = HermiteSpline::new(keys4);
    let sp8 = HermiteSpline::new(keys8);

    g.bench_function("2key_t05",        |x| x.iter(|| sp2.evaluate(black_box(0.5))));
    g.bench_function("4key_t15",        |x| x.iter(|| sp4.evaluate(black_box(1.5))));
    g.bench_function("8key_t35",        |x| x.iter(|| sp8.evaluate(black_box(3.5))));
    g.bench_function("4key_velocity",   |x| x.iter(|| sp4.velocity(black_box(1.5))));

    g.finish();
}

// ── B-Spline ──────────────────────────────────────────────────────────────────

fn bench_bspline(c: &mut Criterion) {
    let mut g = c.benchmark_group("curves/bspline_evaluate");

    let pts4  = make_vec3s(4);
    let pts8  = make_vec3s(8);
    let pts16 = make_vec3s(16);

    let sp4  = BSpline::new(pts4);
    let sp8  = BSpline::new(pts8);
    let sp16 = BSpline::new(pts16);

    g.bench_function("4pt_t05",   |x| x.iter(|| sp4.evaluate(black_box(0.5))));
    g.bench_function("8pt_t25",   |x| x.iter(|| sp8.evaluate(black_box(2.5))));
    g.bench_function("16pt_t65",  |x| x.iter(|| sp16.evaluate(black_box(6.5))));
    g.bench_function("8pt_tangent", |x| x.iter(|| sp8.tangent(black_box(2.5))));

    g.finish();
}

// ── Cardinal ──────────────────────────────────────────────────────────────────

fn bench_cardinal(c: &mut Criterion) {
    let mut g = c.benchmark_group("curves/cardinal_evaluate");

    let pts = make_vec3s(6);

    for tension in [0.0f32, 0.5, 1.0] {
        let sp = CardinalSpline::new(pts.clone(), tension);
        g.bench_with_input(
            BenchmarkId::new("tension", tension),
            &tension,
            |x, _| x.iter(|| sp.evaluate(black_box(2.5))),
        );
    }

    g.finish();
}

// ── Bulk sample_uniform ───────────────────────────────────────────────────────

fn bench_sample_bulk(c: &mut Criterion) {
    let mut g = c.benchmark_group("curves/sample_bulk_1000pts");
    const N: usize = 1000;

    // Cubic Bézier
    {
        let pts = make_vec3s(4);
        let b   = CubicBezier::new(pts[0], pts[1], pts[2], pts[3]);
        let mut buf = vec![Vec3::ZERO; N + 1];
        g.bench_function("cubic_bezier", |x| {
            x.iter(|| { b.sample_uniform(black_box(N), &mut buf); black_box(&buf); })
        });
    }

    // Catmull-Rom centripetal 8 points
    {
        let cr  = CatmullRom::with_alpha(make_vec3s(8), CatmullRomAlpha::Centripetal);
        let mut buf = vec![Vec3::ZERO; N + 1];
        g.bench_function("catmull_rom_centripetal_8pt", |x| {
            x.iter(|| { cr.sample_uniform(black_box(N), &mut buf); black_box(&buf); })
        });
    }

    // Hermite 8 keys
    {
        let sp  = HermiteSpline::new(make_hermite_keys(8));
        let mut buf = vec![Vec3::ZERO; N + 1];
        g.bench_function("hermite_8key", |x| {
            x.iter(|| { sp.sample_uniform(black_box(N), &mut buf); black_box(&buf); })
        });
    }

    // B-Spline 8 points
    {
        let sp  = BSpline::new(make_vec3s(8));
        let mut buf = vec![Vec3::ZERO; N + 1];
        g.bench_function("bspline_8pt", |x| {
            x.iter(|| { sp.sample_uniform(black_box(N), &mut buf); black_box(&buf); })
        });
    }

    // Cardinal tension=0 8 points
    {
        let sp  = CardinalSpline::new(make_vec3s(8), 0.0);
        let mut buf = vec![Vec3::ZERO; N + 1];
        g.bench_function("cardinal_t0_8pt", |x| {
            x.iter(|| { sp.sample_uniform(black_box(N), &mut buf); black_box(&buf); })
        });
    }

    g.finish();
}

// ── Registration ─────────────────────────────────────────────────────────────

criterion_group!(
    benches,
    bench_bezier_quad,
    bench_bezier_cubic,
    bench_catmull_rom,
    bench_hermite,
    bench_bspline,
    bench_cardinal,
    bench_sample_bulk,
);
criterion_main!(benches);
