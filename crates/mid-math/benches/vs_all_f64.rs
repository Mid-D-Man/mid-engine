// crates/mid-math/benches/vs_all_f64.rs
//! f64 benchmark: mid-math DVec3/DQuat/DMat4/DAffine3 vs nalgebra f64 types.
//!
//! We only compare against nalgebra here because:
//!   - glam has no native f64 Vec3 (only DVec3 which is scalar-only like ours)
//!   - ultraviolet has no f64 support
//!   - nalgebra is the main f64-capable competitor
//!
//! The goal is to confirm our scalar f64 types have no pathological overhead
//! vs the industry reference, and to establish baseline numbers for when
//! we add AVX2 fast paths later.
//!
//! Run: cargo bench --bench vs_all_f64 -p mid-math
//! HTML report: target/criterion/report/index.html

use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion, Throughput};

// ── mid-math f64 ──────────────────────────────────────────────────────────────
use mid_math::{DAffine3, DMat4, DQuat, DVec3};

// ── nalgebra f64 ──────────────────────────────────────────────────────────────
use nalgebra::{
    Isometry3, Matrix4, Point3, Translation3, Unit, UnitQuaternion, Vector3,
};

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

fn mid_trs(tx: f64, angle_deg: f64, sx: f64) -> DMat4 {
    DMat4::from_trs(
        DVec3::new(tx, 0.0, 0.0),
        DQuat::from_axis_angle(DVec3::Y, angle_deg.to_radians()),
        DVec3::new(sx, sx, sx),
    )
}

fn mid_affine(tx: f64, angle_deg: f64, sx: f64) -> DAffine3 {
    DAffine3::from_trs(
        DVec3::new(tx, 0.0, 0.0),
        DQuat::from_axis_angle(DVec3::Y, angle_deg.to_radians()),
        DVec3::new(sx, sx, sx),
    )
}

fn na_trs(tx: f64, angle_deg: f64, sx: f64) -> Matrix4<f64> {
    let iso = Isometry3::from_parts(
        Translation3::new(tx, 0.0, 0.0),
        UnitQuaternion::from_axis_angle(&Vector3::y_axis(), angle_deg.to_radians()),
    );
    iso.to_homogeneous() * Matrix4::new_scaling(sx)
}

// ─────────────────────────────────────────────────────────────────────────────
// Group 1: DVec3 arithmetic
// ─────────────────────────────────────────────────────────────────────────────

fn bench_dvec3(c: &mut Criterion) {
    let mut g = c.benchmark_group("dvec3");

    let mm_a = DVec3::new(1.0, 2.0, 3.0);
    let mm_b = DVec3::new(4.0, 5.0, 6.0);
    let na_a = Vector3::new(1.0f64, 2.0, 3.0);
    let na_b = Vector3::new(4.0f64, 5.0, 6.0);

    // add
    g.bench_function("add/mid-math-f64",  |b| b.iter(|| black_box(mm_a) + black_box(mm_b)));
    g.bench_function("add/nalgebra-f64",  |b| b.iter(|| black_box(na_a) + black_box(na_b)));

    // dot
    g.bench_function("dot/mid-math-f64",  |b| b.iter(|| black_box(mm_a).dot(black_box(mm_b))));
    g.bench_function("dot/nalgebra-f64",  |b| b.iter(|| black_box(na_a).dot(&black_box(na_b))));

    // cross
    g.bench_function("cross/mid-math-f64", |b| b.iter(|| black_box(mm_a).cross(black_box(mm_b))));
    g.bench_function("cross/nalgebra-f64", |b| b.iter(|| black_box(na_a).cross(&black_box(na_b))));

    // normalize
    g.bench_function("normalize/mid-math-f64", |b| b.iter(|| black_box(mm_a).normalize()));
    g.bench_function("normalize/nalgebra-f64", |b| b.iter(|| black_box(na_a).normalize()));

    // lerp
    g.bench_function("lerp/mid-math-f64", |b| b.iter(|| black_box(mm_a).lerp(black_box(mm_b), 0.5)));
    g.bench_function("lerp/nalgebra-f64", |b| b.iter(|| black_box(na_a).lerp(&black_box(na_b), 0.5)));

    g.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// Group 2: DQuat operations
// ─────────────────────────────────────────────────────────────────────────────

fn bench_dquat(c: &mut Criterion) {
    let mut g = c.benchmark_group("dquat");

    let mm_q1 = DQuat::from_axis_angle(DVec3::Y, 45.0f64.to_radians());
    let mm_q2 = DQuat::from_axis_angle(
        DVec3::new(1.0, 1.0, 0.0).normalize(), 30.0f64.to_radians());
    let mm_v  = DVec3::new(1.0, 0.0, 0.0);

    let na_q1 = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 45.0f64.to_radians());
    let na_q2 = UnitQuaternion::from_axis_angle(
        &Unit::new_normalize(Vector3::new(1.0f64, 1.0, 0.0)),
        30.0f64.to_radians());
    let na_v = Vector3::new(1.0f64, 0.0, 0.0);

    // mul
    g.bench_function("mul/mid-math-f64",      |b| b.iter(|| black_box(mm_q1) * black_box(mm_q2)));
    g.bench_function("mul/nalgebra-f64",       |b| b.iter(|| black_box(na_q1) * black_box(na_q2)));

    // rotate
    g.bench_function("rotate/mid-math-f64",   |b| b.iter(|| black_box(mm_q1).rotate(black_box(mm_v))));
    g.bench_function("rotate/nalgebra-f64",   |b| b.iter(|| black_box(na_q1) * black_box(na_v)));

    // slerp
    g.bench_function("slerp/mid-math-f64",    |b| b.iter(|| black_box(mm_q1).slerp(black_box(mm_q2), 0.5)));
    g.bench_function("slerp/nalgebra-f64",    |b| b.iter(|| black_box(na_q1).slerp(&black_box(na_q2), 0.5)));

    // nlerp
    g.bench_function("nlerp/mid-math-f64",    |b| b.iter(|| black_box(mm_q1).nlerp(black_box(mm_q2), 0.5)));
    g.bench_function("nlerp/nalgebra-f64",    |b| b.iter(|| black_box(na_q1).nlerp(&black_box(na_q2), 0.5)));

    g.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// Group 3: DMat4 operations
// ─────────────────────────────────────────────────────────────────────────────

fn bench_dmat4(c: &mut Criterion) {
    let mut g = c.benchmark_group("dmat4");

    let mm_a  = mid_trs(1.0, 45.0, 2.0);
    let mm_b  = mid_trs(0.5, 30.0, 1.5);
    let mm_p  = DVec3::new(1.0, 2.0, 3.0);

    let na_a  = na_trs(1.0, 45.0, 2.0);
    let na_b  = na_trs(0.5, 30.0, 1.5);
    let na_p  = Point3::new(1.0f64, 2.0, 3.0);

    // mul
    g.bench_function("mul/mid-math-f64",              |b| b.iter(|| black_box(mm_a) * black_box(mm_b)));
    g.bench_function("mul/nalgebra-f64",               |b| b.iter(|| black_box(na_a) * black_box(na_b)));

    // transform_point
    g.bench_function("transform_point/mid-math-f64",  |b| b.iter(|| black_box(mm_a).transform_point(black_box(mm_p))));
    g.bench_function("transform_point/nalgebra-f64",  |b| b.iter(|| black_box(na_a).transform_point(&black_box(na_p))));

    // inverse general
    g.bench_function("inverse_general/mid-math-f64",  |b| b.iter(|| black_box(mm_a).inverse()));
    g.bench_function("inverse_general/nalgebra-f64",  |b| b.iter(|| black_box(na_a).try_inverse()));

    // inverse TRS fast path
    g.bench_function("inverse_trs/mid-math-f64",      |b| b.iter(|| black_box(mm_a).inverse_trs()));
    g.bench_function("inverse_trs/nalgebra-isometry",  |b| {
        let iso = Isometry3::<f64>::from_parts(
            Translation3::new(1.0, 0.0, 0.0),
            UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 45.0f64.to_radians()),
        );
        b.iter(|| black_box(iso).inverse())
    });

    g.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// Group 4: DAffine3 operations
// ─────────────────────────────────────────────────────────────────────────────

fn bench_daffine3(c: &mut Criterion) {
    let mut g = c.benchmark_group("daffine3");

    let mm_a = mid_affine(1.0, 45.0, 2.0);
    let mm_b = mid_affine(0.5, 30.0, 1.5);
    let mm_p = DVec3::new(1.0, 2.0, 3.0);

    // nalgebra Isometry3 is the closest equivalent for TRS-only work
    let na_iso_a = Isometry3::<f64>::from_parts(
        Translation3::new(1.0, 0.0, 0.0),
        UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 45.0f64.to_radians()),
    );
    let na_iso_b = Isometry3::<f64>::from_parts(
        Translation3::new(0.5, 0.0, 0.0),
        UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 30.0f64.to_radians()),
    );
    let na_p = Point3::new(1.0f64, 2.0, 3.0);

    // inverse
    g.bench_function("inverse/mid-math-DAffine3",     |b| b.iter(|| black_box(mm_a).inverse()));
    g.bench_function("inverse/nalgebra-Isometry3",    |b| b.iter(|| black_box(na_iso_a).inverse()));

    // mul/compose
    g.bench_function("mul/mid-math-DAffine3",         |b| b.iter(|| black_box(mm_a) * black_box(mm_b)));
    g.bench_function("mul/nalgebra-Isometry3",        |b| b.iter(|| black_box(na_iso_a) * black_box(na_iso_b)));

    // transform_point
    g.bench_function("transform_point/mid-math-DAffine3", |b| b.iter(|| black_box(mm_a).transform_point(black_box(mm_p))));
    g.bench_function("transform_point/nalgebra-Isometry3",|b| b.iter(|| black_box(na_iso_a) * black_box(na_p)));

    g.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// Group 5: 100k entity bulk transforms (f64)
//
// f64 bulk transforms are slower than f32 by definition (no SIMD, 2× data).
// This establishes the baseline for future AVX2 work and tells us whether
// f64 physics simulation is viable within a 16.6ms frame budget.
// ─────────────────────────────────────────────────────────────────────────────

fn bench_100k_f64(c: &mut Criterion) {
    const N: usize = 100_000;
    let mut g = c.benchmark_group("100k_entity_transforms_f64");
    g.throughput(Throughput::Elements(N as u64));

    let mm_t  = mid_trs(1.0, 45.0, 1.0);
    let na_t  = na_trs(1.0, 45.0, 1.0);

    let pos_mm: Vec<DVec3>        = (0..N).map(|i| DVec3::new(i as f64 * 0.01, 0.0, 0.0)).collect();
    let pos_na: Vec<Point3<f64>>  = (0..N).map(|i| Point3::new(i as f64 * 0.01, 0.0, 0.0)).collect();

    g.bench_function("mid-math-DMat4", |b| {
        b.iter_batched(
            || pos_mm.clone(),
            |mut p| {
                for v in p.iter_mut() { *v = mm_t.transform_point(black_box(*v)); }
                black_box(p)
            },
            BatchSize::LargeInput,
        )
    });

    g.bench_function("nalgebra-Matrix4", |b| {
        b.iter_batched(
            || pos_na.clone(),
            |mut p| {
                for v in p.iter_mut() { *v = black_box(na_t).transform_point(&black_box(*v)); }
                black_box(p)
            },
            BatchSize::LargeInput,
        )
    });

    g.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// Group 6: 5k inverse general (f64) — physics solver proxy
// ─────────────────────────────────────────────────────────────────────────────

fn bench_5k_inverse_f64(c: &mut Criterion) {
    const N: usize = 5_000;
    let mut g = c.benchmark_group("5k_inverse_general_f64");
    g.throughput(Throughput::Elements(N as u64));

    let mats_mm: Vec<DMat4>          = (0..N).map(|i| mid_trs(i as f64 * 0.1, i as f64, 1.0 + i as f64 * 0.001)).collect();
    let mats_na: Vec<Matrix4<f64>>   = (0..N).map(|i| na_trs(i as f64 * 0.1, i as f64, 1.0 + i as f64 * 0.001)).collect();

    g.bench_function("mid-math-f64", |b| b.iter_batched(
        || mats_mm.clone(),
        |m| { for v in &m { black_box(v.inverse()); } black_box(m) },
        BatchSize::LargeInput,
    ));
    g.bench_function("nalgebra-f64", |b| b.iter_batched(
        || mats_na.clone(),
        |m| { for v in &m { black_box(v.try_inverse()); } black_box(m) },
        BatchSize::LargeInput,
    ));

    g.finish();
}

// ─────────────────────────────────────────────────────────────────────────────

criterion_group!(
    benches,
    bench_dvec3,
    bench_dquat,
    bench_dmat4,
    bench_daffine3,
    bench_100k_f64,
    bench_5k_inverse_f64,
);
criterion_main!(benches);
