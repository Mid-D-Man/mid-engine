// crates/mid-math/benches/vs_all.rs
//! Four-way benchmark: mid-math vs glam vs nalgebra vs ultraviolet.
//!
//! PURPOSE: find what each library does faster and why, so we can steal
//! the implementation strategy while keeping mid-math's FFI ABI.
//!
//! Layout comparison (critical for FFI mandate):
//!   mid-math Vec3   : 16 B align(16)  __m128 backed — FFI via CVec3
//!   glam     Vec3A  : 16 B align(16)  __m128 backed
//!   nalgebra Vector3: 12 B align(4)   scalar, no padding
//!   ultraviolet Vec3: 12 B align(4)   scalar storage, SIMD in bulk ops
//!
//! Reading the results:
//!   Any library beating mid-math = implementation technique to study.
//!   Any library mid-math beats   = confirmation we're doing it right.
//!
//! Run: cargo bench --bench vs_all -p mid-math
//! HTML report: target/criterion/report/index.html

use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion, Throughput};

// ── mid-math ──────────────────────────────────────────────────────────────────
use mid_math::{to_radians, Mat4, Quat, Vec3, Vec4};

// ── glam ─────────────────────────────────────────────────────────────────────
use glam::{Mat4 as GMat4, Quat as GQuat, Vec3A as GVec3, Vec4 as GVec4};

// ── nalgebra ──────────────────────────────────────────────────────────────────
use nalgebra::{Matrix4, Point3, UnitQuaternion, Vector3, Vector4};

// ── ultraviolet ───────────────────────────────────────────────────────────────
use ultraviolet::{Mat4 as UMat4, Rotor3, Vec3 as UVec3, Vec4 as UVec4};

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

fn mid_trs(tx: f32, angle_deg: f32, sx: f32) -> Mat4 {
    Mat4::from_trs(
        Vec3::new(tx, 0.0, 0.0),
        Quat::from_axis_angle(Vec3::Y, to_radians(angle_deg)),
        Vec3::new(sx, sx, sx),
    )
}

fn glam_trs(tx: f32, angle_deg: f32, sx: f32) -> GMat4 {
    GMat4::from_scale_rotation_translation(
        glam::Vec3::splat(sx),
        GQuat::from_rotation_y(angle_deg.to_radians()),
        glam::Vec3::new(tx, 0.0, 0.0),
    )
}

fn na_trs(tx: f32, angle_deg: f32, sx: f32) -> Matrix4<f32> {
    use nalgebra::{Isometry3, Scale3, Translation3};
    let iso = Isometry3::from_parts(
        Translation3::new(tx, 0.0, 0.0),
        UnitQuaternion::from_axis_angle(&Vector3::y_axis(), angle_deg.to_radians()),
    );
    iso.to_homogeneous() * Matrix4::new_scaling(sx)
}

fn uv_trs(tx: f32, angle_deg: f32, sx: f32) -> UMat4 {
    let t = UMat4::from_translation(ultraviolet::Vec3::new(tx, 0.0, 0.0));
    let r = UMat4::from_rotation_y(angle_deg.to_radians());
    let s = UMat4::from_nonuniform_scale(ultraviolet::Vec3::splat(sx));
    t * r * s
}

// ─────────────────────────────────────────────────────────────────────────────
// Group 1: Vec3 arithmetic
// ─────────────────────────────────────────────────────────────────────────────

fn bench_vec3(c: &mut Criterion) {
    let mut g = c.benchmark_group("vec3");

    // add ─────────────────────────────────────────────────────────────────────
    let mm_a = Vec3::new(1.0, 2.0, 3.0);
    let mm_b = Vec3::new(4.0, 5.0, 6.0);
    let gl_a = GVec3::new(1.0, 2.0, 3.0);
    let gl_b = GVec3::new(4.0, 5.0, 6.0);
    let na_a = Vector3::new(1.0f32, 2.0, 3.0);
    let na_b = Vector3::new(4.0f32, 5.0, 6.0);
    let uv_a = UVec3::new(1.0, 2.0, 3.0);
    let uv_b = UVec3::new(4.0, 5.0, 6.0);

    g.bench_function("add/mid-math",   |b| b.iter(|| black_box(mm_a) + black_box(mm_b)));
    g.bench_function("add/glam",       |b| b.iter(|| black_box(gl_a) + black_box(gl_b)));
    g.bench_function("add/nalgebra",   |b| b.iter(|| black_box(na_a) + black_box(na_b)));
    g.bench_function("add/ultraviolet",|b| b.iter(|| black_box(uv_a) + black_box(uv_b)));

    // dot ─────────────────────────────────────────────────────────────────────
    g.bench_function("dot/mid-math",    |b| b.iter(|| black_box(mm_a).dot(black_box(mm_b))));
    g.bench_function("dot/glam",        |b| b.iter(|| black_box(gl_a).dot(black_box(gl_b))));
    g.bench_function("dot/nalgebra",    |b| b.iter(|| black_box(na_a).dot(&black_box(na_b))));
    g.bench_function("dot/ultraviolet", |b| b.iter(|| black_box(uv_a).dot(black_box(uv_b))));

    // cross ───────────────────────────────────────────────────────────────────
    g.bench_function("cross/mid-math",    |b| b.iter(|| black_box(mm_a).cross(black_box(mm_b))));
    g.bench_function("cross/glam",        |b| b.iter(|| black_box(gl_a).cross(black_box(gl_b))));
    g.bench_function("cross/nalgebra",    |b| b.iter(|| black_box(na_a).cross(&black_box(na_b))));
    g.bench_function("cross/ultraviolet", |b| b.iter(|| black_box(uv_a).cross(black_box(uv_b))));

    // normalize ───────────────────────────────────────────────────────────────
    g.bench_function("normalize/mid-math",    |b| b.iter(|| black_box(mm_a).normalize()));
    g.bench_function("normalize/glam",        |b| b.iter(|| black_box(gl_a).normalize()));
    g.bench_function("normalize/nalgebra",    |b| b.iter(|| black_box(na_a).normalize()));
    g.bench_function("normalize/ultraviolet", |b| b.iter(|| black_box(uv_a).normalized()));

    // lerp ────────────────────────────────────────────────────────────────────
    g.bench_function("lerp/mid-math",    |b| b.iter(|| black_box(mm_a).lerp(black_box(mm_b), 0.5)));
    g.bench_function("lerp/glam",        |b| b.iter(|| black_box(gl_a).lerp(black_box(gl_b), 0.5)));
    g.bench_function("lerp/nalgebra",    |b| b.iter(|| black_box(na_a).lerp(&black_box(na_b), 0.5)));
    g.bench_function("lerp/ultraviolet", |b| b.iter(|| {
        let t = 0.5f32;
        black_box(uv_a) + (black_box(uv_b) - black_box(uv_a)) * t
    }));

    g.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// Group 2: Vec4
// ─────────────────────────────────────────────────────────────────────────────

fn bench_vec4(c: &mut Criterion) {
    let mut g = c.benchmark_group("vec4");

    let mm = Vec4::new(1.0, 2.0, 3.0, 4.0);
    let gl = GVec4::new(1.0, 2.0, 3.0, 4.0);
    let na = Vector4::new(1.0f32, 2.0, 3.0, 4.0);
    let uv = UVec4::new(1.0, 2.0, 3.0, 4.0);

    g.bench_function("dot/mid-math",    |b| b.iter(|| black_box(mm).dot(black_box(mm))));
    g.bench_function("dot/glam",        |b| b.iter(|| black_box(gl).dot(black_box(gl))));
    g.bench_function("dot/nalgebra",    |b| b.iter(|| black_box(na).dot(&black_box(na))));
    g.bench_function("dot/ultraviolet", |b| b.iter(|| black_box(uv).dot(black_box(uv))));

    g.bench_function("normalize/mid-math",    |b| b.iter(|| black_box(mm).normalize()));
    g.bench_function("normalize/glam",        |b| b.iter(|| black_box(gl).normalize()));
    g.bench_function("normalize/nalgebra",    |b| b.iter(|| black_box(na).normalize()));
    g.bench_function("normalize/ultraviolet", |b| b.iter(|| black_box(uv).normalized()));

    g.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// Group 3: Rotation (Quat vs UnitQuat vs Rotor3)
//
// Note: ultraviolet uses Rotor3 (geometric-algebra rotor), not a quaternion.
// Rotors are ~equivalent in cost but encode the rotation differently.
// This tells us whether the quaternion representation matters for speed.
// ─────────────────────────────────────────────────────────────────────────────

fn bench_rotation(c: &mut Criterion) {
    let mut g = c.benchmark_group("rotation");

    // mid-math quat
    let mm_q1 = Quat::from_axis_angle(Vec3::Y, to_radians(45.0));
    let mm_q2 = Quat::from_axis_angle(Vec3::new(1.0, 1.0, 0.0).normalize(), to_radians(30.0));
    let mm_v  = Vec3::new(1.0, 0.0, 0.0);

    // glam quat
    let gl_q1 = GQuat::from_rotation_y(45_f32.to_radians());
    let gl_q2 = GQuat::from_axis_angle(glam::Vec3::new(1.0, 1.0, 0.0).normalize(), 30_f32.to_radians());
    let gl_v  = glam::Vec3::X;

    // nalgebra unit quaternion
    let na_q1 = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 45_f32.to_radians());
    let na_q2 = UnitQuaternion::from_axis_angle(
        &Unit::new_normalize(Vector3::new(1.0f32, 1.0, 0.0)),
        30_f32.to_radians(),
    );
    let na_v = Vector3::new(1.0f32, 0.0, 0.0);

    // ultraviolet rotor (geometric algebra — no gimbal lock, same benefit as quat)
    let uv_r1 = Rotor3::from_rotation_between(UVec3::unit_x(), UVec3::unit_y());
    let uv_r2 = Rotor3::from_rotation_between(UVec3::unit_y(), UVec3::unit_z());
    let uv_v  = UVec3::unit_x();

    // mul ─────────────────────────────────────────────────────────────────────
    g.bench_function("mul/mid-math-quat",     |b| b.iter(|| black_box(mm_q1) * black_box(mm_q2)));
    g.bench_function("mul/glam-quat",         |b| b.iter(|| black_box(gl_q1) * black_box(gl_q2)));
    g.bench_function("mul/nalgebra-unitquat", |b| b.iter(|| black_box(na_q1) * black_box(na_q2)));
    g.bench_function("mul/ultraviolet-rotor", |b| b.iter(|| black_box(uv_r1) * black_box(uv_r2)));

    // rotate vec ──────────────────────────────────────────────────────────────
    g.bench_function("rotate/mid-math-quat",     |b| b.iter(|| black_box(mm_q1).rotate(black_box(mm_v))));
    g.bench_function("rotate/glam-quat",         |b| b.iter(|| black_box(gl_q1) * black_box(gl_v)));
    g.bench_function("rotate/nalgebra-unitquat", |b| b.iter(|| black_box(na_q1) * black_box(na_v)));
    g.bench_function("rotate/ultraviolet-rotor", |b| b.iter(|| black_box(uv_r1) * black_box(uv_v)));

    // slerp ───────────────────────────────────────────────────────────────────
    g.bench_function("slerp/mid-math-quat",     |b| b.iter(|| black_box(mm_q1).slerp(black_box(mm_q2), 0.5)));
    g.bench_function("slerp/glam-quat",         |b| b.iter(|| black_box(gl_q1).slerp(black_box(gl_q2), 0.5)));
    g.bench_function("slerp/nalgebra-unitquat", |b| b.iter(|| black_box(na_q1).slerp(&black_box(na_q2), 0.5)));
    g.bench_function("slerp/ultraviolet-rotor", |b| b.iter(|| Rotor3::slerp(black_box(uv_r1), black_box(uv_r2), 0.5)));

    // nlerp ───────────────────────────────────────────────────────────────────
    g.bench_function("nlerp/mid-math-quat",     |b| b.iter(|| black_box(mm_q1).nlerp(black_box(mm_q2), 0.5)));
    g.bench_function("nlerp/glam-quat",         |b| b.iter(|| black_box(gl_q1).lerp(black_box(gl_q2), 0.5)));
    // nalgebra nlerp:
    g.bench_function("nlerp/nalgebra-unitquat", |b| b.iter(|| black_box(na_q1).nlerp(&black_box(na_q2), 0.5)));

    g.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// Group 4: Mat4 operations
//
// This is the most important group — mat4 mul and inverse are where
// mid-math currently loses to glam by 2.5× and 3× respectively.
// Seeing nalgebra and ultraviolet numbers tells us whether glam's approach
// is the only path or if there's something better.
// ─────────────────────────────────────────────────────────────────────────────

fn bench_mat4(c: &mut Criterion) {
    let mut g = c.benchmark_group("mat4");

    let mm_a  = mid_trs(1.0, 45.0, 2.0);
    let mm_b  = mid_trs(0.5, 30.0, 1.5);
    let mm_p  = Vec3::new(1.0, 2.0, 3.0);

    let gl_a  = glam_trs(1.0, 45.0, 2.0);
    let gl_b  = glam_trs(0.5, 30.0, 1.5);
    let gl_p  = glam::Vec3::new(1.0, 2.0, 3.0);

    let na_a  = na_trs(1.0, 45.0, 2.0);
    let na_b  = na_trs(0.5, 30.0, 1.5);
    let na_p  = Point3::new(1.0f32, 2.0, 3.0);

    let uv_a  = uv_trs(1.0, 45.0, 2.0);
    let uv_b  = uv_trs(0.5, 30.0, 1.5);
    let uv_p  = UVec3::new(1.0, 2.0, 3.0);

    // mul ─────────────────────────────────────────────────────────────────────
    // KEY GAP: mid-math 17.6 ns vs glam 7.0 ns. What does nalgebra/uv cost?
    g.bench_function("mul/mid-math",    |b| b.iter(|| black_box(mm_a) * black_box(mm_b)));
    g.bench_function("mul/glam",        |b| b.iter(|| black_box(gl_a) * black_box(gl_b)));
    g.bench_function("mul/nalgebra",    |b| b.iter(|| black_box(na_a) * black_box(na_b)));
    g.bench_function("mul/ultraviolet", |b| b.iter(|| black_box(uv_a) * black_box(uv_b)));

    // transform point ─────────────────────────────────────────────────────────
    g.bench_function("transform_point/mid-math",    |b| b.iter(|| black_box(mm_a).transform_point(black_box(mm_p))));
    g.bench_function("transform_point/glam",        |b| b.iter(|| black_box(gl_a).transform_point3(black_box(gl_p))));
    g.bench_function("transform_point/nalgebra",    |b| b.iter(|| black_box(na_a).transform_point(&black_box(na_p))));
    g.bench_function("transform_point/ultraviolet", |b| b.iter(|| black_box(uv_a).transform_point3(black_box(uv_p))));

    // inverse (general) ───────────────────────────────────────────────────────
    // KEY GAP: mid-math 40 ns vs glam 13 ns. Does nalgebra/uv close this gap
    // with a different algorithm, or is glam's shuffle approach uniquely fast?
    g.bench_function("inverse_general/mid-math",    |b| b.iter(|| black_box(mm_a).inverse()));
    g.bench_function("inverse_general/glam",        |b| b.iter(|| black_box(gl_a).inverse()));
    g.bench_function("inverse_general/nalgebra",    |b| b.iter(|| black_box(na_a).try_inverse()));
    g.bench_function("inverse_general/ultraviolet", |b| b.iter(|| black_box(uv_a).inversed()));

    // inverse trs ─────────────────────────────────────────────────────────────
    g.bench_function("inverse_trs/mid-math",        |b| b.iter(|| black_box(mm_a).inverse_trs()));
    g.bench_function("inverse_trs/glam-affine3a",   |b| {
        let aff = glam::Affine3A::from_mat4(gl_a);
        b.iter(|| black_box(aff).inverse())
    });
    g.bench_function("inverse_trs/nalgebra-isometry", |b| {
        // nalgebra Isometry3 inverse is just conjugate — O(1), the fastest possible
        let iso = nalgebra::Isometry3::<f32>::from_parts(
            nalgebra::Translation3::new(1.0, 0.0, 0.0),
            UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 45_f32.to_radians()),
        );
        b.iter(|| black_box(iso).inverse())
    });

    g.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// Group 5: 100k entity bulk transforms
//
// The engine-critical case. Measures sustained throughput, not single-call
// latency. Cache effects, SIMD pipelining, and memory layout all matter here.
// This is where ultraviolet's SoA wide-SIMD approach might shine.
// ─────────────────────────────────────────────────────────────────────────────

fn bench_100k_entities(c: &mut Criterion) {
    const N: usize = 100_000;
    let mut g = c.benchmark_group("100k_entity_transforms");
    g.throughput(Throughput::Elements(N as u64));

    let mm_t  = mid_trs(1.0, 45.0, 1.0);
    let gl_t  = glam_trs(1.0, 45.0, 1.0);
    let na_t  = na_trs(1.0, 45.0, 1.0);
    let uv_t  = uv_trs(1.0, 45.0, 1.0);

    let pos_mm: Vec<Vec3>          = (0..N).map(|i| Vec3::new(i as f32 * 0.01, 0.0, 0.0)).collect();
    let pos_gl: Vec<glam::Vec3>    = (0..N).map(|i| glam::Vec3::new(i as f32 * 0.01, 0.0, 0.0)).collect();
    let pos_na: Vec<Point3<f32>>   = (0..N).map(|i| Point3::new(i as f32 * 0.01, 0.0, 0.0)).collect();
    let pos_uv: Vec<UVec3>         = (0..N).map(|i| UVec3::new(i as f32 * 0.01, 0.0, 0.0)).collect();

    g.bench_function("mid-math", |b| {
        b.iter_batched(
            || pos_mm.clone(),
            |mut p| {
                for v in p.iter_mut() { *v = mm_t.transform_point(black_box(*v)); }
                black_box(p)
            },
            BatchSize::LargeInput,
        )
    });

    g.bench_function("glam", |b| {
        b.iter_batched(
            || pos_gl.clone(),
            |mut p| {
                for v in p.iter_mut() { *v = gl_t.transform_point3(black_box(*v)); }
                black_box(p)
            },
            BatchSize::LargeInput,
        )
    });

    g.bench_function("nalgebra", |b| {
        b.iter_batched(
            || pos_na.clone(),
            |mut p| {
                for v in p.iter_mut() { *v = black_box(na_t).transform_point(&black_box(*v)); }
                black_box(p)
            },
            BatchSize::LargeInput,
        )
    });

    g.bench_function("ultraviolet", |b| {
        b.iter_batched(
            || pos_uv.clone(),
            |mut p| {
                for v in p.iter_mut() { *v = black_box(uv_t).transform_point3(black_box(*v)); }
                black_box(p)
            },
            BatchSize::LargeInput,
        )
    });

    g.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// Group 6: 5k bulk inverse
// ─────────────────────────────────────────────────────────────────────────────

fn bench_5k_inverse(c: &mut Criterion) {
    const N: usize = 5_000;
    let mut g = c.benchmark_group("5k_inverse_general");
    g.throughput(Throughput::Elements(N as u64));

    let mats_mm: Vec<Mat4>         = (0..N).map(|i| mid_trs(i as f32 * 0.1, i as f32, 1.0 + i as f32 * 0.001)).collect();
    let mats_gl: Vec<GMat4>        = (0..N).map(|i| glam_trs(i as f32 * 0.1, i as f32, 1.0 + i as f32 * 0.001)).collect();
    let mats_na: Vec<Matrix4<f32>> = (0..N).map(|i| na_trs(i as f32 * 0.1, i as f32, 1.0 + i as f32 * 0.001)).collect();
    let mats_uv: Vec<UMat4>        = (0..N).map(|i| uv_trs(i as f32 * 0.1, i as f32, 1.0 + i as f32 * 0.001)).collect();

    g.bench_function("mid-math",    |b| b.iter_batched(|| mats_mm.clone(), |m| { for v in &m { black_box(v.inverse()); } black_box(m) }, BatchSize::LargeInput));
    g.bench_function("glam",        |b| b.iter_batched(|| mats_gl.clone(), |m| { for v in &m { black_box(v.inverse()); } black_box(m) }, BatchSize::LargeInput));
    g.bench_function("nalgebra",    |b| b.iter_batched(|| mats_na.clone(), |m| { for v in &m { black_box(v.try_inverse()); } black_box(m) }, BatchSize::LargeInput));
    g.bench_function("ultraviolet", |b| b.iter_batched(|| mats_uv.clone(), |m| { for v in &m { black_box(v.inversed()); } black_box(m) }, BatchSize::LargeInput));

    g.finish();
}

// ─────────────────────────────────────────────────────────────────────────────

criterion_group!(
    benches,
    bench_vec3,
    bench_vec4,
    bench_rotation,
    bench_mat4,
    bench_100k_entities,
    bench_5k_inverse,
);
criterion_main!(benches);
