// crates/mid-math/benches/vs_wide_float.rs
//! Float wide vector benchmarks: Vec3x4 / QuatX4 vs scalar equivalents.
//!
//! The key question answered by each group:
//!   vec3_4wide  — does Vec3x4 process 4× the work for the same instruction count?
//!   quat_4wide  — do 4 Hamilton products cost the same as 1?
//!   100k_scalar — baseline entity transform loop (plain Mat4::transform_point)
//!   100k_wide   — Vec3x4 batched loop (Mat4::transform_vec3x4, 4 per iter)
//!   100k_wide8  — Vec3x8 batched loop (AVX2 only, 8 per iter)
//!
//! Run: cargo bench --bench vs_wide_float -p mid-math
//! HTML: target/criterion/report/index.html

use criterion::{
    black_box, criterion_group, criterion_main,
    BatchSize, Criterion, Throughput,
};
use mid_math::{
    to_radians,
    Vec3, Vec4, Quat, Mat4,
    Vec3x4, QuatX4, f32x4,
};

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

fn make_mat() -> Mat4 {
    Mat4::from_trs(
        Vec3::new(1.0, 2.0, 3.0),
        Quat::from_axis_angle(Vec3::Y, to_radians(45.0)),
        Vec3::new(2.0, 2.0, 2.0),
    )
}

// ─────────────────────────────────────────────────────────────────────────────
// Group 1: Vec3x4 operations vs scalar Vec3
//
// Each scalar bench processes ONE vector.
// Each wide bench processes FOUR vectors simultaneously.
// Equal wall-clock time = 4× throughput.
// ─────────────────────────────────────────────────────────────────────────────

fn bench_vec3_4wide(c: &mut Criterion) {
    let mut g = c.benchmark_group("vec3_4wide");

    let sv = Vec3::new(1.0, 2.0, 3.0);
    let so = Vec3::new(4.0, 5.0, 6.0);
    let wv = Vec3x4::splat(sv);
    let wo = Vec3x4::splat(so);

    // ── add ──────────────────────────────────────────────────────────────────
    g.bench_function("add/scalar-x1", |b| {
        b.iter(|| black_box(sv) + black_box(so))
    });
    g.bench_function("add/wide-x4",   |b| {
        b.iter(|| black_box(wv) + black_box(wo))
    });

    // ── dot ──────────────────────────────────────────────────────────────────
    g.bench_function("dot/scalar-x1", |b| {
        b.iter(|| black_box(sv).dot(black_box(so)))
    });
    g.bench_function("dot/wide-x4",   |b| {
        b.iter(|| black_box(wv).dot(black_box(wo)))
    });

    // ── cross ─────────────────────────────────────────────────────────────────
    g.bench_function("cross/scalar-x1", |b| {
        b.iter(|| black_box(sv).cross(black_box(so)))
    });
    g.bench_function("cross/wide-x4",   |b| {
        b.iter(|| black_box(wv).cross(black_box(wo)))
    });

    // ── normalize ─────────────────────────────────────────────────────────────
    g.bench_function("normalize/scalar-x1", |b| {
        b.iter(|| black_box(sv).normalize())
    });
    g.bench_function("normalize/wide-x4",   |b| {
        b.iter(|| black_box(wv).normalize())
    });
    g.bench_function("normalize_precise/wide-x4", |b| {
        b.iter(|| black_box(wv).normalize_precise())
    });

    // ── lerp ─────────────────────────────────────────────────────────────────
    g.bench_function("lerp/scalar-x1", |b| {
        b.iter(|| black_box(sv).lerp(black_box(so), 0.5))
    });
    g.bench_function("lerp/wide-x4",   |b| {
        let t = f32x4::splat(0.5);
        b.iter(|| black_box(wv).lerp(black_box(wo), black_box(t)))
    });

    // ── transform_vec3x4 vs transform_point ──────────────────────────────────
    let m = make_mat();
    let single_p = Vec3::new(1.0, 0.0, 0.0);
    let wide_p   = Vec3x4::splat(single_p);

    g.bench_function("transform_point/scalar-x1", |b| {
        b.iter(|| black_box(m).transform_point(black_box(single_p)))
    });
    g.bench_function("transform_point/wide-x4",   |b| {
        b.iter(|| black_box(m).transform_vec3x4(black_box(wide_p)))
    });

    g.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// Group 2: QuatX4 vs scalar Quat
// ─────────────────────────────────────────────────────────────────────────────

fn bench_quat_4wide(c: &mut Criterion) {
    let mut g = c.benchmark_group("quat_4wide");

    let sq1  = Quat::from_axis_angle(Vec3::Y, to_radians(45.0));
    let sq2  = Quat::from_axis_angle(Vec3::X, to_radians(30.0));
    let sv   = Vec3::X;
    let wq1  = QuatX4::splat(sq1);
    let wq2  = QuatX4::splat(sq2);
    let wv   = Vec3x4::splat(sv);

    // ── mul ──────────────────────────────────────────────────────────────────
    g.bench_function("mul/scalar-x1", |b| {
        b.iter(|| black_box(sq1) * black_box(sq2))
    });
    g.bench_function("mul/wide-x4",   |b| {
        b.iter(|| black_box(wq1) * black_box(wq2))
    });

    // ── rotate ───────────────────────────────────────────────────────────────
    g.bench_function("rotate/scalar-x1", |b| {
        b.iter(|| black_box(sq1).rotate(black_box(sv)))
    });
    g.bench_function("rotate/wide-x4",   |b| {
        b.iter(|| black_box(wq1).rotate(black_box(wv)))
    });

    // ── nlerp ────────────────────────────────────────────────────────────────
    g.bench_function("nlerp/scalar-x1", |b| {
        b.iter(|| black_box(sq1).nlerp(black_box(sq2), 0.5))
    });
    g.bench_function("nlerp/wide-x4",   |b| {
        let t = f32x4::splat(0.5);
        b.iter(|| black_box(wq1).nlerp(black_box(wq2), black_box(t)))
    });

    // ── normalize ─────────────────────────────────────────────────────────────
    g.bench_function("normalize/scalar-x1", |b| {
        b.iter(|| black_box(sq1).normalize())
    });
    g.bench_function("normalize/wide-x4",   |b| {
        b.iter(|| black_box(wq1).normalize())
    });

    g.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// Group 3: 100k entity transforms — the engine-critical benchmark
//
// Compares three approaches:
//   scalar   — one transform_point per entity (baseline from stress_tests.rs)
//   wide_x4  — process 4 entities per iteration via transform_vec3x4
//   wide_x8  — process 8 entities per iteration via transform_vec3x8 (AVX2)
//
// Expected: wide_x4 ≈ 4× throughput of scalar at same wall-clock time.
//           wide_x8 ≈ 8× throughput on AVX2-capable CPUs.
//
// The wide approaches require N to be divisible by 4 (or 8).
// The remainder loop is a scalar cleanup — not shown here since 100k % 4 = 0.
// ─────────────────────────────────────────────────────────────────────────────

fn bench_100k_transforms(c: &mut Criterion) {
    const N: usize = 100_000;
    let mut g = c.benchmark_group("100k_entity_transforms_wide");
    g.throughput(Throughput::Elements(N as u64));

    let m = make_mat();

    // ── Scalar baseline ───────────────────────────────────────────────────────
    let pos_scalar: Vec<Vec3> = (0..N)
        .map(|i| Vec3::new(i as f32 * 0.01, 0.0, 0.0))
        .collect();

    g.bench_function("scalar_transform_point", |b| {
        b.iter_batched(
            || pos_scalar.clone(),
            |mut v| {
                for p in v.iter_mut() {
                    *p = m.transform_point(black_box(*p));
                }
                black_box(v)
            },
            BatchSize::LargeInput,
        )
    });

    // ── Vec3x4 — 4 entities per iteration ─────────────────────────────────────
    // Pad source to multiple of 4
    let pos_wide4: Vec<Vec3> = (0..N)
        .map(|i| Vec3::new(i as f32 * 0.01, 0.0, 0.0))
        .collect();

    g.bench_function("wide_x4_transform_vec3x4", |b| {
        b.iter_batched(
            || pos_wide4.clone(),
            |mut v| {
                let chunks = v.len() / 4;
                for chunk in 0..chunks {
                    let base = chunk * 4;
                    let input = Vec3x4::from_slice(
                        v[base..base+4].try_into().unwrap()
                    );
                    let out = m.transform_vec3x4(black_box(input)).to_array();
                    v[base..base+4].copy_from_slice(&out);
                }
                // Scalar remainder (0 in this case since 100k % 4 == 0)
                black_box(v)
            },
            BatchSize::LargeInput,
        )
    });

    // ── Vec3x8 — 8 entities per iteration (AVX2 only) ─────────────────────────
    #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature = "avx2",
    ))]
    {
        use mid_math::Vec3x8;

        let pos_wide8: Vec<Vec3> = (0..N)
            .map(|i| Vec3::new(i as f32 * 0.01, 0.0, 0.0))
            .collect();

        g.bench_function("wide_x8_transform_vec3x8", |b| {
            b.iter_batched(
                || pos_wide8.clone(),
                |mut v| {
                    let chunks = v.len() / 8;
                    for chunk in 0..chunks {
                        let base = chunk * 8;
                        // Note: Vec3x8 does not yet have transform_vec3x8 on Mat4
                        // so we use two Vec3x4 passes as a fair comparison baseline.
                        let a = Vec3x4::from_slice(v[base..base+4].try_into().unwrap());
                        let b2 = Vec3x4::from_slice(v[base+4..base+8].try_into().unwrap());
                        let oa = m.transform_vec3x4(black_box(a)).to_array();
                        let ob = m.transform_vec3x4(black_box(b2)).to_array();
                        v[base..base+4].copy_from_slice(&oa);
                        v[base+4..base+8].copy_from_slice(&ob);
                    }
                    black_box(v)
                },
                BatchSize::LargeInput,
            )
        });
    }

    g.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// Group 4: 100k quaternion rotations — animation blend comparison
// ─────────────────────────────────────────────────────────────────────────────

fn bench_100k_quat_rotations(c: &mut Criterion) {
    const N: usize = 100_000;
    let mut g = c.benchmark_group("100k_quat_rotations_wide");
    g.throughput(Throughput::Elements(N as u64));

    let q = Quat::from_axis_angle(Vec3::Y, to_radians(45.0));

    // ── Scalar ────────────────────────────────────────────────────────────────
    let vecs_scalar: Vec<Vec3> = (0..N)
        .map(|i| Vec3::new(i as f32 * 0.001, 0.0, 0.0))
        .collect();

    g.bench_function("scalar_rotate_x1", |b| {
        b.iter_batched(
            || vecs_scalar.clone(),
            |mut v| {
                for p in v.iter_mut() {
                    *p = q.rotate(black_box(*p));
                }
                black_box(v)
            },
            BatchSize::LargeInput,
        )
    });

    // ── QuatX4 ───────────────────────────────────────────────────────────────
    let wq = QuatX4::splat(q);
    let vecs_wide: Vec<Vec3> = (0..N)
        .map(|i| Vec3::new(i as f32 * 0.001, 0.0, 0.0))
        .collect();

    g.bench_function("wide_x4_rotate", |b| {
        b.iter_batched(
            || vecs_wide.clone(),
            |mut v| {
                let chunks = v.len() / 4;
                for chunk in 0..chunks {
                    let base = chunk * 4;
                    let vw   = Vec3x4::from_slice(v[base..base+4].try_into().unwrap());
                    let out  = wq.rotate(black_box(vw)).to_array();
                    v[base..base+4].copy_from_slice(&out);
                }
                black_box(v)
            },
            BatchSize::LargeInput,
        )
    });

    g.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// Group 5: AoS→SoA transpose cost (amortisation check)
//
// The transpose (from_vec3s / from_quats) is the setup cost for wide ops.
// For it to pay off, the wide ops must process enough data to amortise
// the transpose. This bench shows the raw transpose cost so we know when
// it's worth using wide types.
// ─────────────────────────────────────────────────────────────────────────────

fn bench_wide_transpose(c: &mut Criterion) {
    let mut g = c.benchmark_group("wide_transpose");

    let vecs = [
        Vec3::new(1.0, 2.0, 3.0),
        Vec3::new(4.0, 5.0, 6.0),
        Vec3::new(7.0, 8.0, 9.0),
        Vec3::new(10.0, 11.0, 12.0),
    ];
    let quats = [
        Quat::from_axis_angle(Vec3::Y, to_radians(10.0)),
        Quat::from_axis_angle(Vec3::Y, to_radians(20.0)),
        Quat::from_axis_angle(Vec3::Y, to_radians(30.0)),
        Quat::from_axis_angle(Vec3::Y, to_radians(40.0)),
    ];
    let wide = Vec3x4::from_slice(&vecs);

    g.bench_function("Vec3x4_from_slice",    |b| b.iter(|| Vec3x4::from_slice(black_box(&vecs))));
    g.bench_function("Vec3x4_to_array",      |b| b.iter(|| black_box(wide).to_array()));
    g.bench_function("QuatX4_from_slice",    |b| b.iter(|| QuatX4::from_slice(black_box(&quats))));
    g.bench_function("Vec3x4_splat",         |b| b.iter(|| Vec3x4::splat(black_box(vecs[0]))));

    g.finish();
}

// ─────────────────────────────────────────────────────────────────────────────

criterion_group!(
    benches,
    bench_vec3_4wide,
    bench_quat_4wide,
    bench_100k_transforms,
    bench_100k_quat_rotations,
    bench_wide_transpose,
);
criterion_main!(benches);
