// crates/mid-math/benches/vs_all.rs
//! Comprehensive benchmark: mid-math vs glam vs nalgebra vs ultraviolet.
//!
//! ── Build history ────────────────────────────────────────────────────────────
//!
//!  Build 5 (baseline before SSE2 work):
//!    vec3/normalize    4.97 ns  glam 2.92 ns
//!    rotation/nlerp    5.40 ns  glam 4.19 ns
//!    mat4/mul         19.74 ns  glam 6.91 ns
//!
//!  Build 6 (rsqrt_nr + tree-form mat4_mul_col):
//!    vec3/normalize    4.08 ns  (-18%)
//!    rotation/nlerp    8.22 ns  REGRESSION
//!    mat4/mul         19.77 ns  (0%)
//!
//!  Build 7 (sequential accumulation + normalize_fast + remove vec3 guard):
//!    vec3/normalize    2.62 ns  WE BEAT GLAM (2.92 ns)
//!    rotation/nlerp    6.44 ns  still slow
//!    mat4/mul         19.78 ns  UNCHANGED — root cause [[f32;4];4] storage
//!
//!  Build 8 (Vec4 field storage, dot4_into_m128):
//!    mat4/mul          ~7 ns    TARGET: parity glam
//!    mat4/transform_pt ~3.9 ns  improved
//!    rotation/nlerp    ~4.2 ns  dot4_into_m128 fix
//!
//! Run: cargo bench --bench vs_all -p mid-math

use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion, Throughput};

// ── mid-math ──────────────────────────────────────────────────────────────────
use mid_math::{to_radians, Affine3, Mat2, Mat3, Mat4, Quat, Vec2, Vec3, Vec4};

// ── glam ─────────────────────────────────────────────────────────────────────
use glam::{
    Affine3A as GAffine3A, Mat2 as GMat2, Mat3 as GMat3,
    Mat4 as GMat4, Quat as GQuat, Vec2 as GVec2, Vec3 as GVec3A,
    Vec3A as GVec3, Vec4 as GVec4,
};

// ── nalgebra ──────────────────────────────────────────────────────────────────
use nalgebra::{Matrix2, Matrix3, Matrix4, Point3, Unit, UnitQuaternion, Vector2, Vector3, Vector4};

// ── ultraviolet ───────────────────────────────────────────────────────────────
use ultraviolet::{Mat2 as UMat2, Mat3 as UMat3, Mat4 as UMat4, Rotor3, Slerp, Vec2 as UVec2, Vec3 as UVec3, Vec4 as UVec4};

// ─────────────────────────────────────────────────────────────────────────────
// Shared helpers
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
    use nalgebra::{Isometry3, Translation3};
    let iso = Isometry3::from_parts(
        Translation3::new(tx, 0.0, 0.0),
        UnitQuaternion::from_axis_angle(&Vector3::y_axis(), angle_deg.to_radians()),
    );
    iso.to_homogeneous() * Matrix4::new_scaling(sx)
}

fn uv_trs(tx: f32, angle_deg: f32, sx: f32) -> UMat4 {
    let t = UMat4::from_translation(ultraviolet::Vec3::new(tx, 0.0, 0.0));
    let r = UMat4::from_rotation_y(angle_deg.to_radians());
    let s = UMat4::from_nonuniform_scale(ultraviolet::Vec3::broadcast(sx));
    t * r * s
}

// ─────────────────────────────────────────────────────────────────────────────
// Group 1: Vec3
// ─────────────────────────────────────────────────────────────────────────────

fn bench_vec3(c: &mut Criterion) {
    let mut g = c.benchmark_group("vec3");

    let mm_a = Vec3::new(1.0, 2.0, 3.0);
    let mm_b = Vec3::new(4.0, 5.0, 6.0);
    let gl_a = GVec3::new(1.0, 2.0, 3.0);
    let gl_b = GVec3::new(4.0, 5.0, 6.0);
    let na_a = Vector3::new(1.0f32, 2.0, 3.0);
    let na_b = Vector3::new(4.0f32, 5.0, 6.0);
    let uv_a = UVec3::new(1.0, 2.0, 3.0);
    let uv_b = UVec3::new(4.0, 5.0, 6.0);

    g.bench_function("add/mid-math",    |b| b.iter(|| black_box(mm_a) + black_box(mm_b)));
    g.bench_function("add/glam",        |b| b.iter(|| black_box(gl_a) + black_box(gl_b)));
    g.bench_function("add/nalgebra",    |b| b.iter(|| black_box(na_a) + black_box(na_b)));
    g.bench_function("add/ultraviolet", |b| b.iter(|| black_box(uv_a) + black_box(uv_b)));

    g.bench_function("dot/mid-math",    |b| b.iter(|| black_box(mm_a).dot(black_box(mm_b))));
    g.bench_function("dot/glam",        |b| b.iter(|| black_box(gl_a).dot(black_box(gl_b))));
    g.bench_function("dot/nalgebra",    |b| b.iter(|| black_box(na_a).dot(&black_box(na_b))));
    g.bench_function("dot/ultraviolet", |b| b.iter(|| black_box(uv_a).dot(black_box(uv_b))));

    g.bench_function("cross/mid-math",    |b| b.iter(|| black_box(mm_a).cross(black_box(mm_b))));
    g.bench_function("cross/glam",        |b| b.iter(|| black_box(gl_a).cross(black_box(gl_b))));
    g.bench_function("cross/nalgebra",    |b| b.iter(|| black_box(na_a).cross(&black_box(na_b))));
    g.bench_function("cross/ultraviolet", |b| b.iter(|| black_box(uv_a).cross(black_box(uv_b))));

    // Build 7: 2.62 ns — WE BEAT GLAM (2.92 ns). Guard removal + rsqrt_nr.
    g.bench_function("normalize/mid-math",    |b| b.iter(|| black_box(mm_a).normalize()));
    g.bench_function("normalize/glam",        |b| b.iter(|| black_box(gl_a).normalize()));
    g.bench_function("normalize/nalgebra",    |b| b.iter(|| black_box(na_a).normalize()));
    g.bench_function("normalize/ultraviolet", |b| b.iter(|| black_box(uv_a).normalized()));

    g.bench_function("normalize_or_zero/mid-math", |b| b.iter(|| black_box(mm_a).normalize_or_zero()));
    g.bench_function("normalize_or_zero/glam",     |b| b.iter(|| black_box(gl_a).normalize_or_zero()));

    g.bench_function("lerp/mid-math",    |b| b.iter(|| black_box(mm_a).lerp(black_box(mm_b), 0.5)));
    g.bench_function("lerp/glam",        |b| b.iter(|| black_box(gl_a).lerp(black_box(gl_b), 0.5)));
    g.bench_function("lerp/nalgebra",    |b| b.iter(|| black_box(na_a).lerp(&black_box(na_b), 0.5)));
    g.bench_function("lerp/ultraviolet", |b| b.iter(|| {
        let t = 0.5f32;
        black_box(uv_a) + (black_box(uv_b) - black_box(uv_a)) * t
    }));

    g.bench_function("length/mid-math",    |b| b.iter(|| black_box(mm_a).length()));
    g.bench_function("length/glam",        |b| b.iter(|| black_box(gl_a).length()));
    g.bench_function("length/nalgebra",    |b| b.iter(|| black_box(na_a).norm()));
    g.bench_function("length/ultraviolet", |b| b.iter(|| black_box(uv_a).mag()));

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

    g.bench_function("lerp/mid-math", |b| b.iter(|| black_box(mm).lerp(black_box(mm), 0.5)));
    g.bench_function("lerp/glam",     |b| b.iter(|| black_box(gl).lerp(black_box(gl), 0.5)));

    g.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// Group 3: Rotation
// ─────────────────────────────────────────────────────────────────────────────

fn bench_rotation(c: &mut Criterion) {
    let mut g = c.benchmark_group("rotation");

    let mm_q1 = Quat::from_axis_angle(Vec3::Y, to_radians(45.0));
    let mm_q2 = Quat::from_axis_angle(Vec3::new(1.0, 1.0, 0.0).normalize(), to_radians(30.0));
    let mm_v  = Vec3::new(1.0, 0.0, 0.0);

    let gl_q1 = GQuat::from_rotation_y(45_f32.to_radians());
    let gl_q2 = GQuat::from_axis_angle(glam::Vec3::new(1.0, 1.0, 0.0).normalize(), 30_f32.to_radians());
    let gl_v  = glam::Vec3::X;

    let na_q1 = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 45_f32.to_radians());
    let na_q2 = UnitQuaternion::from_axis_angle(
        &Unit::new_normalize(Vector3::new(1.0f32, 1.0, 0.0)),
        30_f32.to_radians(),
    );
    let na_v = Vector3::new(1.0f32, 0.0, 0.0);

    let uv_r1 = Rotor3::from_rotation_between(UVec3::unit_x(), UVec3::unit_y());
    let uv_r2 = Rotor3::from_rotation_between(UVec3::unit_y(), UVec3::unit_z());
    let uv_v  = UVec3::unit_x();

    g.bench_function("mul/mid-math-quat",     |b| b.iter(|| black_box(mm_q1) * black_box(mm_q2)));
    g.bench_function("mul/glam-quat",         |b| b.iter(|| black_box(gl_q1) * black_box(gl_q2)));
    g.bench_function("mul/nalgebra-unitquat", |b| b.iter(|| black_box(na_q1) * black_box(na_q2)));
    g.bench_function("mul/ultraviolet-rotor", |b| b.iter(|| black_box(uv_r1) * black_box(uv_r2)));

    // mid-math 1.9× faster than glam — architectural win from our rotate impl.
    g.bench_function("rotate/mid-math-quat",     |b| b.iter(|| black_box(mm_q1).rotate(black_box(mm_v))));
    g.bench_function("rotate/glam-quat",         |b| b.iter(|| black_box(gl_q1) * black_box(gl_v)));
    g.bench_function("rotate/nalgebra-unitquat", |b| b.iter(|| black_box(na_q1) * black_box(na_v)));
    g.bench_function("rotate/ultraviolet-rotor", |b| b.iter(|| black_box(uv_r1) * black_box(uv_v)));

    g.bench_function("slerp/mid-math-quat",     |b| b.iter(|| black_box(mm_q1).slerp(black_box(mm_q2), 0.5)));
    g.bench_function("slerp/glam-quat",         |b| b.iter(|| black_box(gl_q1).slerp(black_box(gl_q2), 0.5)));
    g.bench_function("slerp/nalgebra-unitquat", |b| b.iter(|| black_box(na_q1).slerp(&black_box(na_q2), 0.5)));
    g.bench_function("slerp/ultraviolet-rotor", |b| b.iter(|| black_box(uv_r1).slerp(black_box(uv_r2), 0.5)));

    // Build 8: dot4_into_m128 eliminates scalar round-trip. Target: ~4.2 ns.
    g.bench_function("nlerp/mid-math-quat",     |b| b.iter(|| black_box(mm_q1).nlerp(black_box(mm_q2), 0.5)));
    g.bench_function("nlerp/glam-quat",         |b| b.iter(|| black_box(gl_q1).lerp(black_box(gl_q2), 0.5)));
    g.bench_function("nlerp/nalgebra-unitquat", |b| b.iter(|| black_box(na_q1).nlerp(&black_box(na_q2), 0.5)));

    g.bench_function("from_axis_angle/mid-math", |b| b.iter(|| {
        Quat::from_axis_angle(black_box(Vec3::Y), black_box(0.785_f32))
    }));
    g.bench_function("from_axis_angle/glam", |b| b.iter(|| {
        GQuat::from_axis_angle(black_box(glam::Vec3::Y), black_box(0.785_f32))
    }));

    g.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// Group 4: Mat4
// ─────────────────────────────────────────────────────────────────────────────

fn bench_mat4(c: &mut Criterion) {
    let mut g = c.benchmark_group("mat4");

    let mm_a = mid_trs(1.0, 45.0, 2.0);
    let mm_b = mid_trs(0.5, 30.0, 1.5);
    let mm_p = Vec3::new(1.0, 2.0, 3.0);

    let gl_a = glam_trs(1.0, 45.0, 2.0);
    let gl_b = glam_trs(0.5, 30.0, 1.5);
    let gl_p = glam::Vec3::new(1.0, 2.0, 3.0);

    let na_a = na_trs(1.0, 45.0, 2.0);
    let na_b = na_trs(0.5, 30.0, 1.5);
    let na_p = Point3::new(1.0f32, 2.0, 3.0);

    let uv_a = uv_trs(1.0, 45.0, 2.0);
    let uv_b = uv_trs(0.5, 30.0, 1.5);
    let uv_p = UVec3::new(1.0, 2.0, 3.0);

    // Build 8: Vec4 field storage → target ~7 ns (parity glam).
    g.bench_function("mul/mid-math",    |b| b.iter(|| black_box(mm_a) * black_box(mm_b)));
    g.bench_function("mul/glam",        |b| b.iter(|| black_box(gl_a) * black_box(gl_b)));
    g.bench_function("mul/nalgebra",    |b| b.iter(|| black_box(na_a) * black_box(na_b)));
    g.bench_function("mul/ultraviolet", |b| b.iter(|| black_box(uv_a) * black_box(uv_b)));

    g.bench_function("transform_point/mid-math",    |b| b.iter(|| black_box(mm_a).transform_point(black_box(mm_p))));
    g.bench_function("transform_point/glam",        |b| b.iter(|| black_box(gl_a).transform_point3(black_box(gl_p))));
    g.bench_function("transform_point/nalgebra",    |b| b.iter(|| black_box(na_a).transform_point(&black_box(na_p))));
    g.bench_function("transform_point/ultraviolet", |b| b.iter(|| black_box(uv_a).transform_point3(black_box(uv_p))));

    g.bench_function("transpose/mid-math", |b| b.iter(|| black_box(mm_a).transpose()));
    g.bench_function("transpose/glam",     |b| b.iter(|| black_box(gl_a).transpose()));

    g.bench_function("inverse_general/mid-math",    |b| b.iter(|| black_box(mm_a).inverse()));
    g.bench_function("inverse_general/glam",        |b| b.iter(|| black_box(gl_a).inverse()));
    g.bench_function("inverse_general/nalgebra",    |b| b.iter(|| black_box(na_a).try_inverse()));
    g.bench_function("inverse_general/ultraviolet", |b| b.iter(|| black_box(uv_a).inversed()));

    g.bench_function("inverse_trs/mid-math",          |b| b.iter(|| black_box(mm_a).inverse_trs()));
    g.bench_function("inverse_trs/glam-affine3a",     |b| {
        let aff = GAffine3A::from_mat4(gl_a);
        b.iter(|| black_box(aff).inverse())
    });
    g.bench_function("inverse_trs/nalgebra-isometry", |b| {
        let iso = nalgebra::Isometry3::<f32>::from_parts(
            nalgebra::Translation3::new(1.0, 0.0, 0.0),
            UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 45_f32.to_radians()),
        );
        b.iter(|| black_box(iso).inverse())
    });

    g.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// Group 5: Mat4 construction
// ─────────────────────────────────────────────────────────────────────────────

fn bench_mat4_construction(c: &mut Criterion) {
    let mut g = c.benchmark_group("mat4_construction");

    let t   = Vec3::new(1.0, 2.0, 3.0);
    let q   = Quat::from_axis_angle(Vec3::Y, to_radians(45.0));
    let s   = Vec3::new(2.0, 2.0, 2.0);
    let eye = Vec3::new(0.0, 5.0, 10.0);
    let ctr = Vec3::ZERO;
    let up  = Vec3::Y;

    g.bench_function("from_trs/mid-math", |b| b.iter(|| {
        Mat4::from_trs(black_box(t), black_box(q), black_box(s))
    }));
    g.bench_function("from_trs/glam", |b| b.iter(|| {
        GMat4::from_scale_rotation_translation(
            black_box(glam::Vec3::splat(2.0)),
            black_box(GQuat::from_rotation_y(0.785)),
            black_box(glam::Vec3::new(1.0, 2.0, 3.0)),
        )
    }));
    g.bench_function("from_trs/nalgebra", |b| b.iter(|| {
        use nalgebra::{Isometry3, Translation3};
        let iso = Isometry3::from_parts(
            Translation3::new(1.0_f32, 2.0, 3.0),
            UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 0.785_f32),
        );
        black_box(iso.to_homogeneous() * Matrix4::new_scaling(2.0))
    }));

    g.bench_function("look_at_rh/mid-math", |b| b.iter(|| {
        Mat4::look_at_rh(black_box(eye), black_box(ctr), black_box(up))
    }));
    g.bench_function("look_at_rh/glam", |b| b.iter(|| {
        GMat4::look_at_rh(
            black_box(glam::Vec3::new(0.0, 5.0, 10.0)),
            black_box(glam::Vec3::ZERO),
            black_box(glam::Vec3::Y),
        )
    }));

    g.bench_function("perspective_rh/mid-math", |b| b.iter(|| {
        Mat4::perspective_rh(black_box(0.785_f32), black_box(16.0/9.0), black_box(0.1_f32), black_box(1000.0_f32))
    }));
    g.bench_function("perspective_rh/glam", |b| b.iter(|| {
        GMat4::perspective_rh(black_box(0.785_f32), black_box(16.0/9.0_f32), black_box(0.1_f32), black_box(1000.0_f32))
    }));

    g.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// Group 6: Affine3
// ─────────────────────────────────────────────────────────────────────────────

fn bench_affine3(c: &mut Criterion) {
    let mut g = c.benchmark_group("affine3");

    let mm_t = Vec3::new(1.0, 2.0, 3.0);
    let mm_q = Quat::from_axis_angle(Vec3::Y, to_radians(45.0));
    let mm_s = Vec3::new(2.0, 2.0, 2.0);
    let mm_p = Vec3::new(5.0, 0.0, 0.0);

    let mm_a1 = Affine3::from_trs(mm_t, mm_q, mm_s);
    let mm_a2 = Affine3::from_trs(Vec3::new(-1.0, 0.0, 1.0), mm_q, Vec3::new(1.5, 1.5, 1.5));

    let gl_a1 = GAffine3A::from_scale_rotation_translation(
        glam::Vec3::splat(2.0), GQuat::from_rotation_y(0.785), glam::Vec3::new(1.0, 2.0, 3.0),
    );
    let gl_a2 = GAffine3A::from_scale_rotation_translation(
        glam::Vec3::splat(1.5), GQuat::from_rotation_y(0.785), glam::Vec3::new(-1.0, 0.0, 1.0),
    );

    g.bench_function("mul/mid-math-affine3", |b| b.iter(|| black_box(mm_a1) * black_box(mm_a2)));
    g.bench_function("mul/glam-affine3a",    |b| b.iter(|| black_box(gl_a1) * black_box(gl_a2)));

    g.bench_function("inverse/mid-math-affine3", |b| b.iter(|| black_box(mm_a1).inverse()));
    g.bench_function("inverse/glam-affine3a",    |b| b.iter(|| black_box(gl_a1).inverse()));

    g.bench_function("transform_point/mid-math-affine3", |b| b.iter(|| black_box(mm_a1).transform_point(black_box(mm_p))));
    g.bench_function("transform_point/glam-affine3a",    |b| b.iter(|| black_box(gl_a1).transform_point3(black_box(glam::Vec3::new(5.0, 0.0, 0.0)))));

    g.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// Group 7: Mat4 vs matrixmultiply
// ─────────────────────────────────────────────────────────────────────────────

fn bench_mat4_vs_matrixmultiply(c: &mut Criterion) {
    let mut g = c.benchmark_group("mat4_vs_matrixmultiply");

    let mm_a = mid_trs(1.0, 45.0, 2.0);
    let mm_b = mid_trs(0.5, 30.0, 1.5);
    let gl_a = glam_trs(1.0, 45.0, 2.0);
    let gl_b = glam_trs(0.5, 30.0, 1.5);

    // Build 8: Mat4 is repr(C) over four Vec4 fields = 64 bytes = [f32;16].
    // Transmute of the whole Mat4 is safe — layout identical to old [[f32;4];4].
    let a_flat: [f32; 16] = unsafe { core::mem::transmute(mm_a) };
    let b_flat: [f32; 16] = unsafe { core::mem::transmute(mm_b) };

    g.bench_function("mid-math-mat4-mul", |b| b.iter(|| black_box(mm_a) * black_box(mm_b)));
    g.bench_function("glam-mat4-mul",     |b| b.iter(|| black_box(gl_a) * black_box(gl_b)));

    g.bench_function("matrixmultiply-sgemm-4x4", |b| {
        b.iter(|| {
            let mut c_flat = [0.0f32; 16];
            unsafe {
                matrixmultiply::sgemm(
                    4, 4, 4, 1.0,
                    a_flat.as_ptr(), 1, 4,
                    b_flat.as_ptr(), 1, 4,
                    0.0,
                    c_flat.as_mut_ptr(), 1, 4,
                );
            }
            black_box(c_flat)
        })
    });

    g.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// Group 8: Chain of 8 Mat4 multiplies
// ─────────────────────────────────────────────────────────────────────────────

fn bench_chain_mat4(c: &mut Criterion) {
    let mut g = c.benchmark_group("chain_mat4_8");
    g.throughput(Throughput::Elements(8));

    let mm_mats: Vec<Mat4> = (0..8)
        .map(|i| mid_trs(i as f32 * 0.5, i as f32 * 15.0, 1.0 + i as f32 * 0.05))
        .collect();
    let gl_mats: Vec<GMat4> = (0..8)
        .map(|i| glam_trs(i as f32 * 0.5, i as f32 * 15.0, 1.0 + i as f32 * 0.05))
        .collect();

    g.bench_function("mid-math", |b| b.iter(|| {
        let mut m = black_box(mm_mats[0]);
        for i in 1..8 { m = black_box(m) * black_box(mm_mats[i]); }
        black_box(m)
    }));
    g.bench_function("glam", |b| b.iter(|| {
        let mut m = black_box(gl_mats[0]);
        for i in 1..8 { m = black_box(m) * black_box(gl_mats[i]); }
        black_box(m)
    }));

    g.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// Group 9: 100k entity transforms
// ─────────────────────────────────────────────────────────────────────────────

fn bench_100k_entities(c: &mut Criterion) {
    const N: usize = 100_000;
    let mut g = c.benchmark_group("100k_entity_transforms");
    g.throughput(Throughput::Elements(N as u64));

    let mm_t = mid_trs(1.0, 45.0, 1.0);
    let gl_t = glam_trs(1.0, 45.0, 1.0);
    let na_t = na_trs(1.0, 45.0, 1.0);
    let uv_t = uv_trs(1.0, 45.0, 1.0);

    let pos_mm: Vec<Vec3>        = (0..N).map(|i| Vec3::new(i as f32 * 0.01, 0.0, 0.0)).collect();
    let pos_gl: Vec<glam::Vec3>  = (0..N).map(|i| glam::Vec3::new(i as f32 * 0.01, 0.0, 0.0)).collect();
    let pos_na: Vec<Point3<f32>> = (0..N).map(|i| Point3::new(i as f32 * 0.01, 0.0, 0.0)).collect();
    let pos_uv: Vec<UVec3>       = (0..N).map(|i| UVec3::new(i as f32 * 0.01, 0.0, 0.0)).collect();

    g.bench_function("mid-math", |b| b.iter_batched(
        || pos_mm.clone(),
        |mut p| { for v in p.iter_mut() { *v = mm_t.transform_point(black_box(*v)); } black_box(p) },
        BatchSize::LargeInput,
    ));
    g.bench_function("glam", |b| b.iter_batched(
        || pos_gl.clone(),
        |mut p| { for v in p.iter_mut() { *v = gl_t.transform_point3(black_box(*v)); } black_box(p) },
        BatchSize::LargeInput,
    ));
    g.bench_function("nalgebra", |b| b.iter_batched(
        || pos_na.clone(),
        |mut p| { for v in p.iter_mut() { *v = black_box(na_t).transform_point(&black_box(*v)); } black_box(p) },
        BatchSize::LargeInput,
    ));
    g.bench_function("ultraviolet", |b| b.iter_batched(
        || pos_uv.clone(),
        |mut p| { for v in p.iter_mut() { *v = black_box(uv_t).transform_point3(black_box(*v)); } black_box(p) },
        BatchSize::LargeInput,
    ));

    g.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// Group 10: 1M entity transforms
// ─────────────────────────────────────────────────────────────────────────────

fn bench_1m_entities(c: &mut Criterion) {
    const N: usize = 1_000_000;
    let mut g = c.benchmark_group("1m_entity_transforms");
    g.throughput(Throughput::Elements(N as u64));
    g.sample_size(10);

    let mm_t = mid_trs(1.0, 45.0, 1.0);
    let gl_t = glam_trs(1.0, 45.0, 1.0);

    let pos_mm: Vec<Vec3>       = (0..N).map(|i| Vec3::new(i as f32 * 0.001, 0.0, 0.0)).collect();
    let pos_gl: Vec<glam::Vec3> = (0..N).map(|i| glam::Vec3::new(i as f32 * 0.001, 0.0, 0.0)).collect();

    g.bench_function("mid-math", |b| b.iter_batched(
        || pos_mm.clone(),
        |mut p| { for v in p.iter_mut() { *v = mm_t.transform_point(black_box(*v)); } black_box(p) },
        BatchSize::LargeInput,
    ));
    g.bench_function("glam", |b| b.iter_batched(
        || pos_gl.clone(),
        |mut p| { for v in p.iter_mut() { *v = gl_t.transform_point3(black_box(*v)); } black_box(p) },
        BatchSize::LargeInput,
    ));

    g.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// Group 11: 100k quaternion slerp batch
// ─────────────────────────────────────────────────────────────────────────────

fn bench_100k_quat_slerp(c: &mut Criterion) {
    const N: usize = 100_000;
    let mut g = c.benchmark_group("100k_quat_slerp");
    g.throughput(Throughput::Elements(N as u64));

    let quats_mm: Vec<(Quat, Quat)> = (0..N)
        .map(|i| {
            let a = Quat::from_axis_angle(Vec3::Y, to_radians(i as f32 * 0.01));
            let b = Quat::from_axis_angle(Vec3::new(1.0, 0.0, 0.0).normalize(), to_radians(i as f32 * 0.02));
            (a, b)
        })
        .collect();

    let quats_gl: Vec<(GQuat, GQuat)> = (0..N)
        .map(|i| {
            let a = GQuat::from_rotation_y(to_radians(i as f32 * 0.01));
            let b = GQuat::from_rotation_x(to_radians(i as f32 * 0.02));
            (a, b)
        })
        .collect();

    g.bench_function("mid-math-slerp", |b| b.iter_batched(
        || quats_mm.clone(),
        |pairs| { let mut acc = Quat::IDENTITY; for (a, b) in &pairs { acc = black_box(*a).slerp(black_box(*b), 0.5); } black_box(acc) },
        BatchSize::LargeInput,
    ));
    g.bench_function("glam-slerp", |b| b.iter_batched(
        || quats_gl.clone(),
        |pairs| { let mut acc = GQuat::IDENTITY; for (a, b) in &pairs { acc = black_box(*a).slerp(black_box(*b), 0.5); } black_box(acc) },
        BatchSize::LargeInput,
    ));
    g.bench_function("mid-math-nlerp", |b| b.iter_batched(
        || quats_mm.clone(),
        |pairs| { let mut acc = Quat::IDENTITY; for (a, b) in &pairs { acc = black_box(*a).nlerp(black_box(*b), 0.5); } black_box(acc) },
        BatchSize::LargeInput,
    ));
    g.bench_function("glam-nlerp", |b| b.iter_batched(
        || quats_gl.clone(),
        |pairs| { let mut acc = GQuat::IDENTITY; for (a, b) in &pairs { acc = black_box(*a).lerp(black_box(*b), 0.5); } black_box(acc) },
        BatchSize::LargeInput,
    ));

    g.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// Group 12: 5k bulk inverse
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
// Group 13: Mat2
//
// 2×2 operations: mul, determinant, inverse, from_angle.
// These are always scalar (no SIMD benefit at 2 floats), but establish a
// correctness and regression baseline.
// glam: Mat2. nalgebra: Matrix2<f32>. ultraviolet: Mat2.
// ─────────────────────────────────────────────────────────────────────────────

fn bench_mat2(c: &mut Criterion) {
    let mut g = c.benchmark_group("mat2");

    let mm_a = Mat2::from_angle(0.785_f32);
    let mm_b = Mat2::from_scale(Vec2::new(2.0, 1.5));
    let mm_v = Vec2::new(1.0, 2.0);

    let gl_a = GMat2::from_angle(0.785_f32);
    let gl_b = GMat2::from_scale_angle(glam::Vec2::new(2.0, 1.5), 0.0);
    let gl_v = GVec2::new(1.0, 2.0);

    let na_a = Matrix2::new(
        0.785_f32.cos(), -0.785_f32.sin(),
        0.785_f32.sin(),  0.785_f32.cos(),
    );
    let na_b: Matrix2<f32> = Matrix2::new_scaling(2.0);

    // ultraviolet Mat2 has no from_rotation constructor; build it manually.
    // ultraviolet Mat2 has no from_rotation constructor; build it manually.
    let (sin_uv, cos_uv) = (0.785_f32).sin_cos();
    let uv_a = UMat2::new(
        ultraviolet::Vec2::new(cos_uv,  sin_uv),
        ultraviolet::Vec2::new(-sin_uv, cos_uv),
    );
    let uv_b = UMat2::new(
        ultraviolet::Vec2::new(2.0, 0.0),
        ultraviolet::Vec2::new(0.0, 1.5),
    );

    // Matrix multiply — 2×2 = 8 scalar multiplies + 4 adds
    g.bench_function("mul/mid-math",    |b| b.iter(|| black_box(mm_a).mul_mat2(black_box(mm_b))));
    g.bench_function("mul/glam",        |b| b.iter(|| black_box(gl_a) * black_box(gl_b)));
    g.bench_function("mul/nalgebra",    |b| b.iter(|| black_box(na_a) * black_box(na_b)));
    g.bench_function("mul/ultraviolet", |b| b.iter(|| black_box(uv_a) * black_box(uv_b)));

    // Determinant
    g.bench_function("determinant/mid-math", |b| b.iter(|| black_box(mm_a).determinant()));
    g.bench_function("determinant/glam",     |b| b.iter(|| black_box(gl_a).determinant()));
    g.bench_function("determinant/nalgebra", |b| b.iter(|| black_box(na_a).determinant()));

    // Inverse (2×2 adjugate — 5 muls + 1 div)
    g.bench_function("inverse/mid-math", |b| b.iter(|| black_box(mm_a).inverse()));
    g.bench_function("inverse/glam",     |b| b.iter(|| black_box(gl_a).inverse()));
    g.bench_function("inverse/nalgebra", |b| b.iter(|| black_box(na_a).try_inverse()));

    // Transform vector (mat × vec2)
    g.bench_function("mul_vec2/mid-math", |b| b.iter(|| black_box(mm_a).mul_vec2(black_box(mm_v))));
    g.bench_function("mul_vec2/glam",     |b| b.iter(|| black_box(gl_a) * black_box(gl_v)));

    // Transpose
    g.bench_function("transpose/mid-math", |b| b.iter(|| black_box(mm_a).transpose()));
    g.bench_function("transpose/glam",     |b| b.iter(|| black_box(gl_a).transpose()));

    // Construction: from_angle (sin+cos → 4 element fills)
    g.bench_function("from_angle/mid-math", |b| b.iter(|| Mat2::from_angle(black_box(0.785_f32))));
    g.bench_function("from_angle/glam",     |b| b.iter(|| GMat2::from_angle(black_box(0.785_f32))));

    g.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// Group 14: Mat3
//
// 3×3 operations: mul, transform, transpose, inverse, from_rotation_z.
// Always scalar. Important for 2D transform hierarchies and normal matrices.
// glam: Mat3. nalgebra: Matrix3<f32>. ultraviolet: Mat3.
// ─────────────────────────────────────────────────────────────────────────────

fn bench_mat3(c: &mut Criterion) {
    let mut g = c.benchmark_group("mat3");

    // Rotation matrices so inverse is well-conditioned
    let angle = 0.785_f32;
    let mm_a = Mat3::from_rotation_z(angle);
    let mm_b = Mat3::from_scale(Vec3::new(2.0, 1.5, 1.0));
    let mm_v = Vec3::new(1.0, 2.0, 3.0);

    let gl_a = GMat3::from_rotation_z(angle);
    let gl_b = GMat3::from_scale(glam::Vec2::new(2.0, 1.5));
    let gl_v = glam::Vec3::new(1.0, 2.0, 0.0);

    let na_a = Matrix3::new_rotation(angle);
    let na_b: Matrix3<f32> = Matrix3::new_scaling(2.0);
    let na_v = Vector3::new(1.0f32, 2.0, 3.0);

    let uv_a = UMat3::from_rotation_z(angle);
    let uv_v = UVec3::new(1.0, 2.0, 3.0);

    // Matrix multiply — 3×3 = 27 scalar multiplies + 18 adds
    g.bench_function("mul/mid-math",    |b| b.iter(|| black_box(mm_a) * black_box(mm_b)));
    g.bench_function("mul/glam",        |b| b.iter(|| black_box(gl_a) * black_box(gl_b)));
    g.bench_function("mul/nalgebra",    |b| b.iter(|| black_box(na_a) * black_box(na_b)));
    g.bench_function("mul/ultraviolet", |b| b.iter(|| black_box(uv_a) * black_box(uv_a)));

    // Transform vector (3×3 × Vec3 — no translation)
    // glam::Mat3 transforms a Vec3 via mul_vec3 (or the * operator).
    g.bench_function("transform/mid-math",    |b| b.iter(|| black_box(mm_a).transform(black_box(mm_v))));
    g.bench_function("transform/glam",        |b| b.iter(|| black_box(gl_a).mul_vec3(black_box(gl_v))));
    g.bench_function("transform/nalgebra",    |b| b.iter(|| black_box(na_a) * black_box(na_v)));
    g.bench_function("transform/ultraviolet", |b| b.iter(|| black_box(uv_a) * black_box(uv_v)));

    // Transpose
    g.bench_function("transpose/mid-math", |b| b.iter(|| black_box(mm_a).transpose()));
    g.bench_function("transpose/glam",     |b| b.iter(|| black_box(gl_a).transpose()));
    g.bench_function("transpose/nalgebra", |b| b.iter(|| black_box(na_a).transpose()));

    // Determinant
    g.bench_function("determinant/mid-math", |b| b.iter(|| black_box(mm_a).determinant()));
    g.bench_function("determinant/glam",     |b| b.iter(|| black_box(gl_a).determinant()));
    g.bench_function("determinant/nalgebra", |b| b.iter(|| black_box(na_a).determinant()));

    // Inverse (3×3 cofactor — more expensive than 2×2)
    g.bench_function("inverse/mid-math", |b| b.iter(|| black_box(mm_a).inverse()));
    g.bench_function("inverse/glam",     |b| b.iter(|| black_box(gl_a).inverse()));
    g.bench_function("inverse/nalgebra", |b| b.iter(|| black_box(na_a).try_inverse()));

    // Construction: from_rotation_z (sin+cos → 9 element fills)
    g.bench_function("from_rotation_z/mid-math", |b| b.iter(|| Mat3::from_rotation_z(black_box(0.785_f32))));
    g.bench_function("from_rotation_z/glam",     |b| b.iter(|| GMat3::from_rotation_z(black_box(0.785_f32))));

    // Normal matrix (inverse-transpose of upper-3×3 — used in shaders every frame)
    // glam::Mat3::inverse() returns Self (not Option), so no .map() needed.
    let mm_model = mid_trs(1.0, 45.0, 2.0);
    let gl_model = glam_trs(1.0, 45.0, 2.0);
    g.bench_function("normal_matrix/mid-math", |b| b.iter(|| Mat3::normal_matrix(black_box(&mm_model))));
    g.bench_function("normal_matrix/glam",     |b| b.iter(|| {
        let m3 = GMat3::from_mat4(black_box(gl_model));
        black_box(m3.inverse().transpose())
    }));

    g.finish();
}

// ─────────────────────────────────────────────────────────────────────────────

criterion_group!(
    benches,
    bench_vec3,
    bench_vec4,
    bench_rotation,
    bench_mat4,
    bench_mat4_construction,
    bench_affine3,
    bench_mat4_vs_matrixmultiply,
    bench_chain_mat4,
    bench_100k_entities,
    bench_1m_entities,
    bench_100k_quat_slerp,
    bench_5k_inverse,
    bench_mat2,
    bench_mat3,
);
criterion_main!(benches);
