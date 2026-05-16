// crates/mid-math/benches/vs_helpers.rs
//! Helper type benchmarks: DualQuat, Rotor3, SpatialVelocity/Force, TangentFrame, Angle.
//!
//! Groups:
//!   helpers/dual_quat   — construction, transform, blend
//!   helpers/rotor       — construction, rotate, compose, nlerp
//!   helpers/spatial     — cross product, dot (power), inertia mul
//!   helpers/tangent     — from_triangle, transform_normal, pack/unpack
//!   helpers/angle       — Radians/Degrees trig and conversion
//!   helpers/bulk_100k   — skinning blend throughput (4-bone DLB)

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use mid_math::{
    Degrees, DualQuat, PackedTangent, Quat, Radians, Rotor3,
    SpatialForce, SpatialInertia, SpatialVelocity, TangentFrame, Vec2, Vec3,
};

// ── DualQuat ──────────────────────────────────────────────────────────────────

fn bench_dual_quat(c: &mut Criterion) {
    let mut g = c.benchmark_group("helpers/dual_quat");

    let rot   = Quat::from_axis_angle(Vec3::new(0.0, 1.0, 0.0), 1.047); // 60°
    let trans = Vec3::new(1.0, 2.0, 3.0);
    let dq    = DualQuat::from_rotation_translation(rot, trans);
    let dq2   = DualQuat::from_rotation_translation(
        Quat::from_axis_angle(Vec3::new(1.0, 0.0, 0.0), 0.5),
        Vec3::new(-1.0, 0.5, 2.0),
    );
    let point = Vec3::new(0.5, 1.0, -0.5);

    g.bench_function("from_rotation_translation", |b| {
        b.iter(|| black_box(DualQuat::from_rotation_translation(
            black_box(rot), black_box(trans)
        )))
    });
    g.bench_function("from_rotation_only", |b| {
        b.iter(|| black_box(DualQuat::from_rotation(black_box(rot))))
    });
    g.bench_function("transform_point", |b| {
        b.iter(|| black_box(black_box(dq).transform_point(black_box(point))))
    });
    g.bench_function("transform_vector", |b| {
        b.iter(|| black_box(black_box(dq).transform_vector(black_box(point))))
    });
    g.bench_function("normalize", |b| {
        b.iter(|| black_box(black_box(dq).normalize()))
    });
    g.bench_function("mul_compose", |b| {
        b.iter(|| black_box(black_box(dq) * black_box(dq2)))
    });
    g.bench_function("blend2", |b| {
        b.iter(|| black_box(DualQuat::blend2(
            black_box(dq), black_box(0.6_f32),
            black_box(dq2), black_box(0.4_f32),
        )))
    });
    g.bench_function("blend4", |b| {
        let dq3 = DualQuat::from_rotation(Quat::from_euler(0.1, 0.2, 0.3));
        let dq4 = DualQuat::from_translation(Vec3::new(0.5, 0.5, 0.5));
        b.iter(|| black_box(DualQuat::blend4(black_box([
            (dq, 0.4), (dq2, 0.3), (dq3, 0.2), (dq4, 0.1),
        ]))))
    });
    g.bench_function("extract_translation", |b| {
        b.iter(|| black_box(black_box(dq).translation()))
    });
    g.finish();
}

// ── Rotor3 ────────────────────────────────────────────────────────────────────

fn bench_rotor(c: &mut Criterion) {
    let mut g = c.benchmark_group("helpers/rotor");

    let axis  = Vec3::new(0.0, 1.0, 0.0).normalize();
    let r1    = Rotor3::from_axis_angle(axis, 1.047);
    let r2    = Rotor3::from_axis_angle(Vec3::new(1.0, 0.0, 0.0), 0.5);
    let v     = Vec3::new(1.0, 0.0, 0.0);
    let from  = Vec3::new(1.0, 0.0, 0.0);
    let to    = Vec3::new(0.0, 1.0, 0.0);

    g.bench_function("from_axis_angle", |b| {
        b.iter(|| black_box(Rotor3::from_axis_angle(black_box(axis), black_box(1.047_f32))))
    });
    g.bench_function("from_vec_to_vec", |b| {
        b.iter(|| black_box(Rotor3::from_vec_to_vec(black_box(from), black_box(to))))
    });
    g.bench_function("rotate_vec3", |b| {
        b.iter(|| black_box(black_box(r1).rotate(black_box(v))))
    });
    g.bench_function("geometric_product", |b| {
        b.iter(|| black_box(black_box(r1) * black_box(r2)))
    });
    g.bench_function("nlerp", |b| {
        b.iter(|| black_box(black_box(r1).nlerp(black_box(r2), black_box(0.5_f32))))
    });
    g.bench_function("normalize", |b| {
        b.iter(|| black_box(black_box(r1).normalize()))
    });
    g.bench_function("to_quat", |b| {
        b.iter(|| black_box(black_box(r1).to_quat()))
    });
    g.bench_function("from_quat", |b| {
        let q = r1.to_quat();
        b.iter(|| black_box(Rotor3::from_quat(black_box(q))))
    });
    g.bench_function("reverse", |b| {
        b.iter(|| black_box(black_box(r1).reverse()))
    });
    g.finish();
}

// ── Spatial vectors ───────────────────────────────────────────────────────────

fn bench_spatial(c: &mut Criterion) {
    let mut g = c.benchmark_group("helpers/spatial");

    let vel = SpatialVelocity::new(
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(1.0, 0.0, 0.5),
    );
    let vel2 = SpatialVelocity::new(
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
    );
    let force = SpatialForce::new(
        Vec3::new(0.0, 0.0, 5.0),
        Vec3::new(10.0, 0.0, 0.0),
    );
    let inertia = SpatialInertia {
        mass:    10.0,
        com:     Vec3::new(0.0, 0.1, 0.0),
        inertia: [1.0, 2.0, 1.5, 0.0, 0.0, 0.0],
    };

    g.bench_function("vel_cross_vel", |b| {
        b.iter(|| black_box(black_box(vel).cross_vel(black_box(vel2))))
    });
    g.bench_function("vel_cross_force", |b| {
        b.iter(|| black_box(black_box(vel).cross_force(black_box(force))))
    });
    g.bench_function("vel_dot_force", |b| {
        b.iter(|| black_box(black_box(vel).dot_force(black_box(force))))
    });
    g.bench_function("inertia_mul_vel", |b| {
        b.iter(|| black_box(black_box(inertia).mul_vel(black_box(vel))))
    });
    g.bench_function("force_add", |b| {
        let f2 = SpatialForce::new(Vec3::new(1.0, 2.0, 3.0), Vec3::new(4.0, 5.0, 6.0));
        b.iter(|| black_box(black_box(force) + black_box(f2)))
    });
    g.bench_function("vel_scale", |b| {
        b.iter(|| black_box(black_box(vel).scale(black_box(2.0_f32))))
    });
    g.finish();
}

// ── TangentFrame ──────────────────────────────────────────────────────────────

fn bench_tangent(c: &mut Criterion) {
    let mut g = c.benchmark_group("helpers/tangent");

    let n  = Vec3::new(0.0, 1.0, 0.0);
    let t  = Vec3::new(1.0, 0.0, 0.0);
    let tf = TangentFrame::from_normal_tangent(n, t, 1.0);

    let p0  = Vec3::new(0.0, 0.0, 0.0);
    let p1  = Vec3::new(1.0, 0.0, 0.0);
    let p2  = Vec3::new(0.0, 0.0, 1.0);
    let uv0 = Vec2::new(0.0, 0.0);
    let uv1 = Vec2::new(1.0, 0.0);
    let uv2 = Vec2::new(0.0, 1.0);

    let sample_normal = Vec3::new(0.1, 0.95, 0.05).normalize();

    g.bench_function("from_normal_tangent", |b| {
        b.iter(|| black_box(TangentFrame::from_normal_tangent(
            black_box(n), black_box(t), black_box(1.0_f32)
        )))
    });
    g.bench_function("from_triangle", |b| {
        b.iter(|| black_box(TangentFrame::from_triangle(
            black_box(p0), black_box(p1), black_box(p2),
            black_box(uv0), black_box(uv1), black_box(uv2),
            black_box(n),
        )))
    });
    g.bench_function("transform_normal", |b| {
        b.iter(|| black_box(black_box(tf).transform_normal(black_box(sample_normal))))
    });
    g.bench_function("to_tangent_space", |b| {
        b.iter(|| black_box(black_box(tf).to_tangent_space(black_box(sample_normal))))
    });
    g.bench_function("orthogonalise", |b| {
        b.iter(|| black_box(black_box(tf).orthogonalise()))
    });
    g.bench_function("pack", |b| {
        b.iter(|| black_box(black_box(tf).pack()))
    });
    g.bench_function("unpack", |b| {
        let packed = PackedTangent { tangent: t, handedness: 1.0 };
        b.iter(|| black_box(TangentFrame::unpack(black_box(packed), black_box(n))))
    });
    g.bench_function("to_mat3", |b| {
        b.iter(|| black_box(black_box(tf).to_mat3()))
    });
    g.finish();
}

// ── Angle ─────────────────────────────────────────────────────────────────────

fn bench_angle(c: &mut Criterion) {
    let mut g = c.benchmark_group("helpers/angle");

    let rad = Radians::new(1.047); // 60°
    let deg = Degrees::new(60.0);

    g.bench_function("radians_sin", |b| {
        b.iter(|| black_box(black_box(rad).sin()))
    });
    g.bench_function("radians_cos", |b| {
        b.iter(|| black_box(black_box(rad).cos()))
    });
    g.bench_function("radians_sin_cos", |b| {
        b.iter(|| black_box(black_box(rad).sin_cos()))
    });
    g.bench_function("radians_to_degrees", |b| {
        b.iter(|| black_box(black_box(rad).to_degrees()))
    });
    g.bench_function("degrees_to_radians", |b| {
        b.iter(|| black_box(black_box(deg).to_radians()))
    });
    g.bench_function("radians_wrap", |b| {
        let big = Radians::new(7.5);
        b.iter(|| black_box(black_box(big).wrap()))
    });
    g.bench_function("degrees_wrap", |b| {
        let big = Degrees::new(450.0);
        b.iter(|| black_box(black_box(big).wrap()))
    });
    g.bench_function("radians_lerp", |b| {
        let a = Radians::new(0.0);
        let b_val = Radians::PI;
        b.iter(|| black_box(black_box(a).lerp(black_box(b_val), black_box(0.5_f32))))
    });
    g.bench_function("atan2", |b| {
        b.iter(|| black_box(Radians::atan2(black_box(1.0_f32), black_box(1.0_f32))))
    });
    g.finish();
}

// ── Bulk: 4-bone DLB skinning 100k vertices ───────────────────────────────────

fn bench_bulk_skinning_100k(c: &mut Criterion) {
    let mut g = c.benchmark_group("helpers/bulk_100k");
    g.sample_size(20);

    // Build 4 bone transforms
    let bones: [DualQuat; 4] = [
        DualQuat::from_rotation_translation(
            Quat::from_axis_angle(Vec3::new(0.0, 1.0, 0.0), 0.3),
            Vec3::new(0.0, 1.0, 0.0),
        ),
        DualQuat::from_rotation_translation(
            Quat::from_axis_angle(Vec3::new(1.0, 0.0, 0.0), 0.5),
            Vec3::new(0.0, 2.0, 0.0),
        ),
        DualQuat::from_rotation_translation(
            Quat::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), 0.2),
            Vec3::new(0.0, 3.0, 0.0),
        ),
        DualQuat::from_rotation(Quat::from_euler(0.1, 0.2, 0.3)),
    ];

    // 100k vertices, each with 4 bone weights (normalized)
    let weights: Vec<[f32; 4]> = (0..100_000)
        .map(|i| {
            let t = i as f32 / 100_000.0;
            [0.5 - t * 0.2, 0.3 + t * 0.1, 0.1 + t * 0.05, 0.1 + t * 0.05]
        })
        .collect();
    let vertices: Vec<Vec3> = (0..100_000)
        .map(|i| Vec3::new(i as f32 * 0.001, 0.0, 0.0))
        .collect();

    g.bench_function("dlb_4bone_100k_vertices", |b| {
        b.iter(|| {
            let mut output = Vec::with_capacity(100_000);
            for (vert, w) in black_box(&vertices).iter().zip(black_box(&weights)) {
                let blended = DualQuat::blend4([
                    (bones[0], w[0]),
                    (bones[1], w[1]),
                    (bones[2], w[2]),
                    (bones[3], w[3]),
                ]);
                output.push(blended.transform_point(*vert));
            }
            black_box(output)
        })
    });

    g.bench_function("rotor_rotate_100k", |b| {
        let r = Rotor3::from_axis_angle(Vec3::new(0.0, 1.0, 0.0), 0.785);
        b.iter(|| {
            let mut last = Vec3::ZERO;
            for &v in black_box(&vertices) {
                last = black_box(r).rotate(v);
            }
            black_box(last)
        })
    });

    g.finish();
}

criterion_group!(
    benches,
    bench_dual_quat,
    bench_rotor,
    bench_spatial,
    bench_tangent,
    bench_angle,
    bench_bulk_skinning_100k,
);
criterion_main!(benches);
