// crates/mid-math/benches/vs_all_f64.rs
//! f64 benchmark: mid-math DVec2/DVec3/DVec4/DQuat/DMat2/DMat3/DMat4/DAffine3
//! vs nalgebra f64 types.
//!
//! We only compare against nalgebra here because:
//!   - glam has no native f64 Vec3 (only DVec3 which is scalar-only like ours)
//!   - ultraviolet has no f64 support
//!   - nalgebra is the main f64-capable competitor
//!
//! Coverage note: DVec3/DMat2/DMat3/DMat4/DAffine2/DAffine3/DDualQuat are
//! always-scalar on every platform (see f64/mod.rs) — only DVec2/DVec4/DQuat
//! have a dispatched SIMD backend (sse2/neon/wasm), so those three groups
//! are the ones where target-cpu should actually move the needle. Everything
//! else here establishes scalar baselines, same purpose as before.
//!
//! DAffine2 and DDualQuat aren't covered yet — narrower usage, add if needed.
//!
//! Run: cargo bench --bench vs_all_f64 -p mid-math
//! HTML report: target/criterion/report/index.html

use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion, Throughput};

// ── mid-math f64 ──────────────────────────────────────────────────────────────
use mid_math::{DAffine3, DMat2, DMat3, DMat4, DQuat, DVec2, DVec3, DVec4};

// ── nalgebra f64 ──────────────────────────────────────────────────────────────
use nalgebra::{
    Isometry3, Matrix2, Matrix3, Matrix4, Point3, Translation3, Unit, UnitQuaternion,
    Vector2, Vector3, Vector4,
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
// Group 1b: DVec2 arithmetic — has a real SIMD backend (sse2/neon/wasm),
// unlike most of this file. Previously had zero bench coverage at all.
// ─────────────────────────────────────────────────────────────────────────────

fn bench_dvec2(c: &mut Criterion) {
    let mut g = c.benchmark_group("dvec2");

    let mm_a = DVec2::new(1.0, 2.0);
    let mm_b = DVec2::new(3.0, 4.0);
    let na_a = Vector2::new(1.0f64, 2.0);
    let na_b = Vector2::new(3.0f64, 4.0);

    g.bench_function("add/mid-math-f64", |b| b.iter(|| black_box(mm_a) + black_box(mm_b)));
    g.bench_function("add/nalgebra-f64", |b| b.iter(|| black_box(na_a) + black_box(na_b)));

    g.bench_function("dot/mid-math-f64", |b| b.iter(|| black_box(mm_a).dot(black_box(mm_b))));
    g.bench_function("dot/nalgebra-f64", |b| b.iter(|| black_box(na_a).dot(&black_box(na_b))));

    g.bench_function("normalize/mid-math-f64", |b| b.iter(|| black_box(mm_a).normalize()));
    g.bench_function("normalize/nalgebra-f64", |b| b.iter(|| black_box(na_a).normalize()));

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
// Group 2b: DVec4 arithmetic — real SIMD backend, previously zero coverage.
// ─────────────────────────────────────────────────────────────────────────────

fn bench_dvec4(c: &mut Criterion) {
    let mut g = c.benchmark_group("dvec4");

    let mm_a = DVec4::new(1.0, 2.0, 3.0, 4.0);
    let mm_b = DVec4::new(5.0, 6.0, 7.0, 8.0);
    let na_a = Vector4::new(1.0f64, 2.0, 3.0, 4.0);
    let na_b = Vector4::new(5.0f64, 6.0, 7.0, 8.0);

    g.bench_function("dot/mid-math-f64", |b| b.iter(|| black_box(mm_a).dot(black_box(mm_b))));
    g.bench_function("dot/nalgebra-f64", |b| b.iter(|| black_box(na_a).dot(&black_box(na_b))));

    g.bench_function("normalize/mid-math-f64", |b| b.iter(|| black_box(mm_a).normalize()));
    g.bench_function("normalize/nalgebra-f64", |b| b.iter(|| black_box(na_a).normalize()));

    g.bench_function("lerp/mid-math-f64", |b| b.iter(|| black_box(mm_a).lerp(black_box(mm_b), 0.5)));
    g.bench_function("lerp/nalgebra-f64", |b| b.iter(|| black_box(na_a).lerp(&black_box(na_b), 0.5)));

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
// Group 4b: DMat2 operations — always-scalar (see f64/mod.rs), scalar
// baseline only, no target-cpu movement expected here.
// ─────────────────────────────────────────────────────────────────────────────

fn bench_dmat2(c: &mut Criterion) {
    let mut g = c.benchmark_group("dmat2");

    let mm_a = DMat2::from_angle(0.5);
    let mm_b = DMat2::from_angle(1.0);
    let mm_v = DVec2::new(1.0, 2.0);

    let na_a = Matrix2::new(0.5f64.cos(), -0.5f64.sin(), 0.5f64.sin(), 0.5f64.cos());
    let na_b = Matrix2::new(1.0f64.cos(), -1.0f64.sin(), 1.0f64.sin(), 1.0f64.cos());
    let na_v = Vector2::new(1.0f64, 2.0);

    g.bench_function("mul/mid-math-f64", |b| b.iter(|| black_box(mm_a) * black_box(mm_b)));
    g.bench_function("mul/nalgebra-f64", |b| b.iter(|| black_box(na_a) * black_box(na_b)));

    g.bench_function("determinant/mid-math-f64", |b| b.iter(|| black_box(mm_a).determinant()));
    g.bench_function("determinant/nalgebra-f64", |b| b.iter(|| black_box(na_a).determinant()));

    g.bench_function("inverse/mid-math-f64", |b| b.iter(|| black_box(mm_a).inverse()));
    g.bench_function("inverse/nalgebra-f64", |b| b.iter(|| black_box(na_a).try_inverse()));

    g.bench_function("mul_vec2/mid-math-f64", |b| b.iter(|| black_box(mm_a).mul_vec2(black_box(mm_v))));
    g.bench_function("mul_vec2/nalgebra-f64", |b| b.iter(|| black_box(na_a) * black_box(na_v)));

    g.bench_function("transpose/mid-math-f64", |b| b.iter(|| black_box(mm_a).transpose()));
    g.bench_function("transpose/nalgebra-f64", |b| b.iter(|| black_box(na_a).transpose()));

    g.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// Group 4c: DMat3 operations — always-scalar, scalar baseline only.
//
// from_rotation_z and normal_matrix are mid-math-only here (no nalgebra
// comparison): nalgebra doesn't expose a directly equivalent one-call API
// for either without pulling in Rotation3/fixed_view plumbing that's easy
// to get subtly wrong without a compiler on hand to check it against — add
// a nalgebra side for these two later if it's worth the exact API dig.
// ─────────────────────────────────────────────────────────────────────────────

fn bench_dmat3(c: &mut Criterion) {
    let mut g = c.benchmark_group("dmat3");

    let mm_a = DMat3::from_rotation_z(0.5);
    let mm_b = DMat3::from_rotation_z(1.0);
    let mm_v = DVec3::new(1.0, 2.0, 3.0);

    let na_rot = |angle: f64| {
        UnitQuaternion::from_axis_angle(&Vector3::z_axis(), angle).to_rotation_matrix().into_inner()
    };
    let na_a: Matrix3<f64> = na_rot(0.5);
    let na_b: Matrix3<f64> = na_rot(1.0);
    let na_v = Vector3::new(1.0f64, 2.0, 3.0);

    g.bench_function("mul/mid-math-f64", |b| b.iter(|| black_box(mm_a) * black_box(mm_b)));
    g.bench_function("mul/nalgebra-f64", |b| b.iter(|| black_box(na_a) * black_box(na_b)));

    g.bench_function("transform/mid-math-f64", |b| b.iter(|| black_box(mm_a).transform(black_box(mm_v))));
    g.bench_function("transform/nalgebra-f64", |b| b.iter(|| black_box(na_a) * black_box(na_v)));

    g.bench_function("transpose/mid-math-f64", |b| b.iter(|| black_box(mm_a).transpose()));
    g.bench_function("transpose/nalgebra-f64", |b| b.iter(|| black_box(na_a).transpose()));

    g.bench_function("determinant/mid-math-f64", |b| b.iter(|| black_box(mm_a).determinant()));
    g.bench_function("determinant/nalgebra-f64", |b| b.iter(|| black_box(na_a).determinant()));

    g.bench_function("inverse/mid-math-f64", |b| b.iter(|| black_box(mm_a).inverse()));
    g.bench_function("inverse/nalgebra-f64", |b| b.iter(|| black_box(na_a).try_inverse()));

    g.bench_function("from_rotation_z/mid-math-f64", |b| b.iter(|| DMat3::from_rotation_z(black_box(0.5))));

    g.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// Group 5b: 100k quat slerp/nlerp bulk (f64) — DQuat has a real SIMD backend;
// this is the bulk-throughput counterpart to bench_dquat's single-call numbers,
// matching the shape of the f32 100k_quat_slerp group. Previously missing.
// ─────────────────────────────────────────────────────────────────────────────

fn bench_100k_quat_slerp_f64(c: &mut Criterion) {
    const N: usize = 100_000;
    let mut g = c.benchmark_group("100k_quat_slerp_f64");
    g.throughput(Throughput::Elements(N as u64));

    let mm_q1 = DQuat::from_axis_angle(DVec3::Y, 45.0f64.to_radians());
    let mm_q2 = DQuat::from_axis_angle(DVec3::new(1.0, 1.0, 0.0).normalize(), 30.0f64.to_radians());
    let na_q1 = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 45.0f64.to_radians());
    let na_q2 = UnitQuaternion::from_axis_angle(
        &Unit::new_normalize(Vector3::new(1.0f64, 1.0, 0.0)), 30.0f64.to_radians());

    g.bench_function("mid-math-slerp-f64", |b| b.iter_batched(
        || (0..N).map(|_| mm_q1).collect::<Vec<_>>(),
        |v| { for q in &v { black_box(q.slerp(black_box(mm_q2), 0.5)); } black_box(v) },
        BatchSize::LargeInput,
    ));
    g.bench_function("nalgebra-slerp-f64", |b| b.iter_batched(
        || (0..N).map(|_| na_q1).collect::<Vec<_>>(),
        |v| { for q in &v { black_box(q.slerp(&black_box(na_q2), 0.5)); } black_box(v) },
        BatchSize::LargeInput,
    ));
    g.bench_function("mid-math-nlerp-f64", |b| b.iter_batched(
        || (0..N).map(|_| mm_q1).collect::<Vec<_>>(),
        |v| { for q in &v { black_box(q.nlerp(black_box(mm_q2), 0.5)); } black_box(v) },
        BatchSize::LargeInput,
    ));
    g.bench_function("nalgebra-nlerp-f64", |b| b.iter_batched(
        || (0..N).map(|_| na_q1).collect::<Vec<_>>(),
        |v| { for q in &v { black_box(q.nlerp(&black_box(na_q2), 0.5)); } black_box(v) },
        BatchSize::LargeInput,
    ));

    g.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// Group 5c: 1M entity bulk transforms (f64) — scale counterpart to
// bench_100k_f64, matching f32's 100k/1m pairing. Previously missing.
// ─────────────────────────────────────────────────────────────────────────────

fn bench_1m_f64(c: &mut Criterion) {
    const N: usize = 1_000_000;
    let mut g = c.benchmark_group("1m_entity_transforms_f64");
    g.sample_size(10);
    g.throughput(Throughput::Elements(N as u64));

    let mm_t = mid_trs(1.0, 45.0, 1.0);
    let na_t = na_trs(1.0, 45.0, 1.0);

    let pos_mm: Vec<DVec3>       = (0..N).map(|i| DVec3::new(i as f64 * 0.001, 0.0, 0.0)).collect();
    let pos_na: Vec<Point3<f64>> = (0..N).map(|i| Point3::new(i as f64 * 0.001, 0.0, 0.0)).collect();

    g.bench_function("mid-math-DMat4", |b| b.iter_batched(
        || pos_mm.clone(),
        |mut p| { for v in p.iter_mut() { *v = mm_t.transform_point(black_box(*v)); } black_box(p) },
        BatchSize::LargeInput,
    ));
    g.bench_function("nalgebra-Matrix4", |b| b.iter_batched(
        || pos_na.clone(),
        |mut p| { for v in p.iter_mut() { *v = black_box(na_t).transform_point(&black_box(*v)); } black_box(p) },
        BatchSize::LargeInput,
    ));

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
    bench_dvec2,
    bench_dvec3,
    bench_dvec4,
    bench_dquat,
    bench_dmat2,
    bench_dmat3,
    bench_dmat4,
    bench_daffine3,
    bench_100k_f64,
    bench_100k_quat_slerp_f64,
    bench_1m_f64,
    bench_5k_inverse_f64,
);
criterion_main!(benches);
