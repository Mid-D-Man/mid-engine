//! vs_glam.rs — Direct comparison of mid-math against glam.
//!
//! This is NOT our stress test suite. This is a head-to-head benchmark
//! using criterion's statistical framework. Criterion controls iteration
//! count automatically and reports mean ± confidence interval.
//!
//! Run:
//!   cargo bench --bench vs_glam -p mid-math
//!
//! HTML report: target/criterion/report/index.html
//!
//! What "glam" is: the reference SIMD game math library used by Bevy.
//! https://crates.io/crates/glam
//!
//! Layout notes for fair comparison:
//! - mid-math Vec3 = 16 bytes, align(16) — matches glam::Vec3A
//! - glam::Vec3    = 12 bytes, tight     — NOT comparable to our Vec3
//! - glam::Vec4    = 16 bytes, align(16) — matches our Vec4
//! - glam::Mat4    = 64 bytes, align(16) — matches our Mat4
//! - glam::Affine3A is glam's optimised TRS type — compared vs our inverse_trs

use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion, Throughput};
use mid_math::{to_radians, Mat4, Quat, Vec3, Vec4};

// ── Helpers ──────────────────────────────────────────────────────────────────

fn mid_quat(angle_deg: f32, axis: Vec3) -> Quat {
    Quat::from_axis_angle(axis, to_radians(angle_deg))
}

fn glam_quat(angle_deg: f32, axis: glam::Vec3) -> glam::Quat {
    glam::Quat::from_axis_angle(axis, angle_deg.to_radians())
}

fn mid_trs(tx: f32, angle_deg: f32, sx: f32) -> Mat4 {
    Mat4::from_trs(
        Vec3::new(tx, 0.0, 0.0),
        mid_quat(angle_deg, Vec3::Y),
        Vec3::new(sx, sx, sx),
    )
}

fn glam_trs(tx: f32, angle_deg: f32, sx: f32) -> glam::Mat4 {
    glam::Mat4::from_scale_rotation_translation(
        glam::Vec3::splat(sx),
        glam_quat(angle_deg, glam::Vec3::Y),
        glam::Vec3::new(tx, 0.0, 0.0),
    )
}

fn glam_affine3a(tx: f32, angle_deg: f32, sx: f32) -> glam::Affine3A {
    glam::Affine3A::from_scale_rotation_translation(
        glam::Vec3::splat(sx),
        glam_quat(angle_deg, glam::Vec3::Y),
        glam::Vec3::new(tx, 0.0, 0.0),
    )
}

// ── Vec3 ─────────────────────────────────────────────────────────────────────
//
// Fair comparison:
//   mid-math Vec3 (16 B, align 16)  vs  glam Vec3A (16 B, align 16)
//   Do NOT compare against glam::Vec3 (12 B, unaligned) — different layout.

fn bench_vec3(c: &mut Criterion) {
    let mut g = c.benchmark_group("vec3");

    let am = Vec3::new(1.0, 2.0, 3.0);
    let bm = Vec3::new(4.0, 5.0, 6.0);
    let ag = glam::Vec3A::new(1.0, 2.0, 3.0);
    let bg = glam::Vec3A::new(4.0, 5.0, 6.0);

    // Add
    g.bench_function("add/mid-math",
        |b| b.iter(|| black_box(am) + black_box(bm)));
    g.bench_function("add/glam-Vec3A",
        |b| b.iter(|| black_box(ag) + black_box(bg)));

    // Dot
    g.bench_function("dot/mid-math",
        |b| b.iter(|| black_box(am).dot(black_box(bm))));
    g.bench_function("dot/glam-Vec3A",
        |b| b.iter(|| black_box(ag).dot(black_box(bg))));

    // Cross
    g.bench_function("cross/mid-math",
        |b| b.iter(|| black_box(am).cross(black_box(bm))));
    g.bench_function("cross/glam-Vec3A",
        |b| b.iter(|| black_box(ag).cross(black_box(bg))));

    // Normalize
    g.bench_function("normalize/mid-math",
        |b| b.iter(|| black_box(am).normalize()));
    g.bench_function("normalize/glam-Vec3A",
        |b| b.iter(|| black_box(ag).normalize()));

    // Lerp
    g.bench_function("lerp/mid-math",
        |b| b.iter(|| black_box(am).lerp(black_box(bm), black_box(0.5_f32))));
    g.bench_function("lerp/glam-Vec3A",
        |b| b.iter(|| black_box(ag).lerp(black_box(bg), black_box(0.5_f32))));

    g.finish();
}

// ── Vec4 ─────────────────────────────────────────────────────────────────────

fn bench_vec4(c: &mut Criterion) {
    let mut g = c.benchmark_group("vec4");

    let am = Vec4::new(1.0, 2.0, 3.0, 4.0);
    let bm = Vec4::new(5.0, 6.0, 7.0, 8.0);
    let ag = glam::Vec4::new(1.0, 2.0, 3.0, 4.0);
    let bg = glam::Vec4::new(5.0, 6.0, 7.0, 8.0);

    g.bench_function("dot/mid-math",
        |b| b.iter(|| black_box(am).dot(black_box(bm))));
    g.bench_function("dot/glam",
        |b| b.iter(|| black_box(ag).dot(black_box(bg))));

    g.bench_function("normalize/mid-math",
        |b| b.iter(|| black_box(am).normalize()));
    g.bench_function("normalize/glam",
        |b| b.iter(|| black_box(ag).normalize()));

    g.finish();
}

// ── Quat ─────────────────────────────────────────────────────────────────────

fn bench_quat(c: &mut Criterion) {
    let mut g = c.benchmark_group("quat");

    let q1m = mid_quat(45.0, Vec3::Y);
    let q2m = mid_quat(30.0, Vec3::new(1.0, 1.0, 0.0).normalize());
    let vm  = Vec3::new(1.0, 0.0, 0.0);

    let q1g = glam_quat(45.0, glam::Vec3::Y);
    let q2g = glam_quat(30.0, glam::Vec3::new(1.0, 1.0, 0.0).normalize());
    let vg  = glam::Vec3::new(1.0, 0.0, 0.0);

    // Multiplication
    g.bench_function("mul/mid-math",
        |b| b.iter(|| black_box(q1m) * black_box(q2m)));
    g.bench_function("mul/glam",
        |b| b.iter(|| black_box(q1g) * black_box(q2g)));

    // Rotate Vec3 (the sandwich product)
    g.bench_function("rotate/mid-math",
        |b| b.iter(|| black_box(q1m).rotate(black_box(vm))));
    g.bench_function("rotate/glam",
        |b| b.iter(|| black_box(q1g) * black_box(vg)));

    // Slerp (trig-bound, both should be similar)
    g.bench_function("slerp/mid-math",
        |b| b.iter(|| black_box(q1m).slerp(black_box(q2m), black_box(0.5_f32))));
    g.bench_function("slerp/glam",
        |b| b.iter(|| black_box(q1g).slerp(black_box(q2g), black_box(0.5_f32))));

    // Nlerp (cheaper alternative, small angles)
    g.bench_function("nlerp/mid-math",
        |b| b.iter(|| black_box(q1m).nlerp(black_box(q2m), black_box(0.5_f32))));
    // glam uses lerp for the cheap path
    g.bench_function("nlerp/glam-lerp",
        |b| b.iter(|| black_box(q1g).lerp(black_box(q2g), black_box(0.5_f32))));

    g.finish();
}

// ── Mat4 ─────────────────────────────────────────────────────────────────────

fn bench_mat4(c: &mut Criterion) {
    let mut g = c.benchmark_group("mat4");

    let am  = mid_trs(1.0, 45.0, 2.0);
    let bm  = mid_trs(0.5, 30.0, 1.5);
    let pm  = Vec3::new(1.0, 2.0, 3.0);

    let ag  = glam_trs(1.0, 45.0, 2.0);
    let bg  = glam_trs(0.5, 30.0, 1.5);
    let pg  = glam::Vec3::new(1.0, 2.0, 3.0);

    // affine3a: glam's optimised TRS type — direct counterpart for inverse_trs
    let aff = glam_affine3a(1.0, 45.0, 2.0);

    // Multiply
    g.bench_function("mul/mid-math",
        |b| b.iter(|| black_box(am) * black_box(bm)));
    g.bench_function("mul/glam",
        |b| b.iter(|| black_box(ag) * black_box(bg)));

    // Transform point (w=1)
    g.bench_function("transform_point/mid-math",
        |b| b.iter(|| black_box(am).transform_point(black_box(pm))));
    g.bench_function("transform_point/glam",
        |b| b.iter(|| black_box(ag).transform_point3(black_box(pg))));

    // General inverse (Cramer's rule / cofactor expansion)
    g.bench_function("inverse_general/mid-math",
        |b| b.iter(|| black_box(am).inverse()));
    g.bench_function("inverse_general/glam-Mat4",
        |b| b.iter(|| black_box(ag).inverse()));

    // TRS inverse — our fast-path vs glam's Affine3A
    // Note: glam::Affine3A::inverse() is glam's optimised TRS inverse,
    // equivalent in purpose to our Mat4::inverse_trs().
    g.bench_function("inverse_trs/mid-math",
        |b| b.iter(|| black_box(am).inverse_trs()));
    g.bench_function("inverse_trs/glam-Affine3A",
        |b| b.iter(|| black_box(aff).inverse()));

    // Scalar fallback (always available, used in tests to verify SSE2)
    g.bench_function("inverse_trs_scalar/mid-math",
        |b| b.iter(|| black_box(am).inverse_trs_scalar()));

    g.finish();
}

// ── 100k entity transform (bulk throughput) ───────────────────────────────────
//
// This matches the scale of our stress test benchmarks.
// iter_batched separates the clone cost from the measured loop.

fn bench_100k_entity_transforms(c: &mut Criterion) {
    const N: usize = 100_000;
    let mut g = c.benchmark_group("100k_entity_transforms");
    g.throughput(Throughput::Elements(N as u64));

    let tm = mid_trs(1.0, 45.0, 1.0);
    let tg = glam_trs(1.0, 45.0, 1.0);

    let pos_m: Vec<Vec3>       = (0..N).map(|i| Vec3::new(i as f32 * 0.01, 0.0, 0.0)).collect();
    let pos_g: Vec<glam::Vec3> = (0..N).map(|i| glam::Vec3::new(i as f32 * 0.01, 0.0, 0.0)).collect();

    g.bench_function("mid-math", |b| {
        b.iter_batched(
            || pos_m.clone(),
            |mut pos| {
                for p in pos.iter_mut() {
                    *p = tm.transform_point(black_box(*p));
                }
                black_box(pos)
            },
            BatchSize::LargeInput,
        )
    });

    g.bench_function("glam", |b| {
        b.iter_batched(
            || pos_g.clone(),
            |mut pos| {
                for p in pos.iter_mut() {
                    *p = tg.transform_point3(black_box(*p));
                }
                black_box(pos)
            },
            BatchSize::LargeInput,
        )
    });

    g.finish();
}

// ── 5k inverse_trs (bulk — our stress test scale) ────────────────────────────

fn bench_5k_inverse_trs(c: &mut Criterion) {
    const N: usize = 5_000;
    let mut g = c.benchmark_group("5k_inverse_trs");
    g.throughput(Throughput::Elements(N as u64));

    let mats_m: Vec<Mat4> = (0..N)
        .map(|i| mid_trs(i as f32 * 0.1, i as f32, 1.0 + i as f32 * 0.001))
        .collect();
    let mats_aff: Vec<glam::Affine3A> = (0..N)
        .map(|i| glam_affine3a(i as f32 * 0.1, i as f32, 1.0 + i as f32 * 0.001))
        .collect();

    g.bench_function("mid-math-sse2", |b| {
        b.iter_batched(
            || mats_m.clone(),
            |mats| {
                for m in &mats { black_box(m.inverse_trs()); }
                black_box(mats)
            },
            BatchSize::LargeInput,
        )
    });

    g.bench_function("mid-math-scalar", |b| {
        b.iter_batched(
            || mats_m.clone(),
            |mats| {
                for m in &mats { black_box(m.inverse_trs_scalar()); }
                black_box(mats)
            },
            BatchSize::LargeInput,
        )
    });

    g.bench_function("glam-Affine3A", |b| {
        b.iter_batched(
            || mats_aff.clone(),
            |mats| {
                for m in &mats { black_box(m.inverse()); }
                black_box(mats)
            },
            BatchSize::LargeInput,
        )
    });

    g.finish();
}

criterion_group!(
    benches,
    bench_vec3,
    bench_vec4,
    bench_quat,
    bench_mat4,
    bench_100k_entity_transforms,
    bench_5k_inverse_trs,
);
criterion_main!(benches);
