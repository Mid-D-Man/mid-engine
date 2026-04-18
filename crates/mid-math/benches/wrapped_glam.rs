//! wrapped_glam.rs — Measures the true cost of wrapping glam behind mid-math types.
//!
//! Three implementations benchmarked side by side for every operation:
//!
//!   A) mid-math pure    — our own implementation, no glam involved
//!   B) glam native      — glam called directly, no mid-math types
//!   C) glam wrapped     — mid-math types at the boundary, glam for compute
//!
//! "Wrapped" means:
//!   1. Accept mid-math types (e.g. Vec3, Mat4)
//!   2. Convert to glam types at the call site
//!   3. Call glam's SIMD implementation
//!   4. Convert result back to mid-math types
//!   5. Return mid-math type to caller
//!
//! Two conversion strategies are compared:
//!   - field-copy   : safe, explicit, LLVM may or may not eliminate
//!   - transmute    : unsafe, zero-copy if layouts match, experimental
//!
//! Reading the results:
//!   wrapped ≈ glam-native   → conversion is zero-cost, LLVM eliminated it
//!   wrapped > glam-native   → conversion overhead is real
//!   pure-mid-math vs wrapped → remaining gap = SIMD work left in Option A
//!
//! Run: cargo bench --bench wrapped_glam -p mid-math
//!
//! Layout compatibility assumptions (all verified by size_of/align_of tests):
//!   mid-math Vec3  = { x,y,z,_pad } 16 B align(16)  ↔  glam Vec3A { x,y,z }+pad 16 B align(16)
//!   mid-math Vec4  = { x,y,z,w }    16 B align(16)  ↔  glam Vec4  { x,y,z,w }   16 B align(16)
//!   mid-math Quat  = { x,y,z,w }    16 B align(16)  ↔  glam Quat  { x,y,z,w }   16 B align(16)
//!   mid-math Mat4  = { cols[4][4] }  64 B align(16)  ↔  glam Mat4  column-major   64 B align(16)

use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion, Throughput};
use mid_math::{to_radians, Mat4, Quat, Vec3, Vec4};

// ─────────────────────────────────────────────────────────────────────────────
// Safe field-copy conversions (Strategy 1)
//
// These are the "real world Option C" conversions — no unsafe.
// If LLVM is doing its job they compile to register moves, cost ≈ 0.
// The bench below will tell you if that assumption holds.
// ─────────────────────────────────────────────────────────────────────────────

#[inline(always)]
fn vec3_to_glam(v: Vec3) -> glam::Vec3A {
    glam::Vec3A::new(v.x, v.y, v.z)
}

#[inline(always)]
fn glam_to_vec3(v: glam::Vec3A) -> Vec3 {
    Vec3::new(v.x, v.y, v.z)
}

#[inline(always)]
fn vec4_to_glam(v: Vec4) -> glam::Vec4 {
    glam::Vec4::new(v.x, v.y, v.z, v.w)
}

#[inline(always)]
fn glam_to_vec4(v: glam::Vec4) -> Vec4 {
    Vec4::new(v.x, v.y, v.z, v.w)
}

#[inline(always)]
fn quat_to_glam(q: Quat) -> glam::Quat {
    glam::Quat::from_xyzw(q.x, q.y, q.z, q.w)
}

#[inline(always)]
fn mat4_to_glam(m: Mat4) -> glam::Mat4 {
    // cols is [[f32;4];4] column-major — matches glam's column layout exactly.
    glam::Mat4::from_cols_array_2d(&m.cols)
}

#[inline(always)]
fn glam_to_mat4(m: glam::Mat4) -> Mat4 {
    Mat4::from_cols(
        m.x_axis.to_array(),
        m.y_axis.to_array(),
        m.z_axis.to_array(),
        m.w_axis.to_array(),
    )
}

// ─────────────────────────────────────────────────────────────────────────────
// Transmute conversions (Strategy 2) — unsafe, experimental
//
// Relies on layout compatibility listed in the module doc.
// If Strategy 2 is faster than Strategy 1: LLVM is not eliminating the
// field copies above and there is real measurable overhead.
// If they are the same speed: LLVM is already handling it and the safe
// path is preferred.
//
// Safety: caller guarantees the source value is a valid instance of the
// destination type. Valid here because both sides are f32-only structs
// with the same alignment and field positions.
// ─────────────────────────────────────────────────────────────────────────────

/// # Safety
/// Vec3 and Vec3A have identical memory layout: 16 bytes, align(16), floats at [0,4,8].
#[inline(always)]
unsafe fn vec3_to_glam_transmute(v: Vec3) -> glam::Vec3A {
    std::mem::transmute(v)
}

/// # Safety
/// See vec3_to_glam_transmute.
#[inline(always)]
unsafe fn glam_to_vec3_transmute(v: glam::Vec3A) -> Vec3 {
    std::mem::transmute(v)
}

/// # Safety
/// Mat4 and glam::Mat4 have identical memory layout: 64 bytes, align(16), column-major f32 data.
#[inline(always)]
unsafe fn mat4_to_glam_transmute(m: Mat4) -> glam::Mat4 {
    std::mem::transmute(m)
}

/// # Safety
/// See mat4_to_glam_transmute.
#[inline(always)]
unsafe fn glam_to_mat4_transmute(m: glam::Mat4) -> Mat4 {
    std::mem::transmute(m)
}

// ─────────────────────────────────────────────────────────────────────────────
// TRS construction helpers
// ─────────────────────────────────────────────────────────────────────────────

fn mid_trs(tx: f32, angle_deg: f32, sx: f32) -> Mat4 {
    Mat4::from_trs(
        Vec3::new(tx, 0.0, 0.0),
        Quat::from_axis_angle(Vec3::Y, to_radians(angle_deg)),
        Vec3::new(sx, sx, sx),
    )
}

fn glam_trs(tx: f32, angle_deg: f32, sx: f32) -> glam::Mat4 {
    glam::Mat4::from_scale_rotation_translation(
        glam::Vec3::splat(sx),
        glam::Quat::from_rotation_y(angle_deg.to_radians()),
        glam::Vec3::new(tx, 0.0, 0.0),
    )
}

// ─────────────────────────────────────────────────────────────────────────────
// Group 0: Conversion cost in isolation
//
// The most important group. Measures just the round-trip cost with no compute.
// If this is ~0ns: LLVM sees through the conversion entirely. Proceed with
// confidence that Option C has zero FFI overhead for these types.
// If this is measurable: there is a real cost per call site.
// ─────────────────────────────────────────────────────────────────────────────

fn bench_conversion_cost(c: &mut Criterion) {
    let mut g = c.benchmark_group("conversion_roundtrip");

    let vm  = Vec3::new(1.0, 2.0, 3.0);
    let v4m = Vec4::new(1.0, 2.0, 3.0, 4.0);
    let mm  = mid_trs(1.0, 45.0, 2.0);

    g.bench_function("vec3_field_copy",
        |b| b.iter(|| glam_to_vec3(vec3_to_glam(black_box(vm)))));

    g.bench_function("vec3_transmute",
        |b| b.iter(|| unsafe { glam_to_vec3_transmute(vec3_to_glam_transmute(black_box(vm))) }));

    g.bench_function("vec4_field_copy",
        |b| b.iter(|| glam_to_vec4(vec4_to_glam(black_box(v4m)))));

    g.bench_function("mat4_field_copy",
        |b| b.iter(|| glam_to_mat4(mat4_to_glam(black_box(mm)))));

    g.bench_function("mat4_transmute",
        |b| b.iter(|| unsafe { glam_to_mat4_transmute(mat4_to_glam_transmute(black_box(mm))) }));

    g.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// Group 1: Vec3 operations
// ─────────────────────────────────────────────────────────────────────────────

fn bench_vec3(c: &mut Criterion) {
    let mut g = c.benchmark_group("vec3_wrapped");

    let am = Vec3::new(1.0, 2.0, 3.0);
    let bm = Vec3::new(4.0, 5.0, 6.0);
    let ag = vec3_to_glam(am);
    let bg = vec3_to_glam(bm);

    // Add
    g.bench_function("add/mid-math-pure",
        |b| b.iter(|| black_box(am) + black_box(bm)));
    g.bench_function("add/glam-native",
        |b| b.iter(|| black_box(ag) + black_box(bg)));
    g.bench_function("add/glam-field-copy-wrapped",
        |b| b.iter(|| glam_to_vec3(vec3_to_glam(black_box(am)) + vec3_to_glam(black_box(bm)))));
    g.bench_function("add/glam-transmute-wrapped",
        |b| b.iter(|| unsafe {
            glam_to_vec3_transmute(vec3_to_glam_transmute(black_box(am)) + vec3_to_glam_transmute(black_box(bm)))
        }));

    // Dot
    g.bench_function("dot/mid-math-pure",
        |b| b.iter(|| black_box(am).dot(black_box(bm))));
    g.bench_function("dot/glam-native",
        |b| b.iter(|| black_box(ag).dot(black_box(bg))));
    g.bench_function("dot/glam-field-copy-wrapped",
        |b| b.iter(|| vec3_to_glam(black_box(am)).dot(vec3_to_glam(black_box(bm)))));

    // Normalize
    g.bench_function("normalize/mid-math-pure",
        |b| b.iter(|| black_box(am).normalize()));
    g.bench_function("normalize/glam-native",
        |b| b.iter(|| black_box(ag).normalize()));
    g.bench_function("normalize/glam-field-copy-wrapped",
        |b| b.iter(|| glam_to_vec3(vec3_to_glam(black_box(am)).normalize())));
    g.bench_function("normalize/glam-transmute-wrapped",
        |b| b.iter(|| unsafe {
            glam_to_vec3_transmute(vec3_to_glam_transmute(black_box(am)).normalize())
        }));

    // Cross
    g.bench_function("cross/mid-math-pure",
        |b| b.iter(|| black_box(am).cross(black_box(bm))));
    g.bench_function("cross/glam-native",
        |b| b.iter(|| black_box(ag).cross(black_box(bg))));
    g.bench_function("cross/glam-field-copy-wrapped",
        |b| b.iter(|| glam_to_vec3(vec3_to_glam(black_box(am)).cross(vec3_to_glam(black_box(bm))))));

    g.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// Group 2: Mat4 operations
// ─────────────────────────────────────────────────────────────────────────────

fn bench_mat4(c: &mut Criterion) {
    let mut g = c.benchmark_group("mat4_wrapped");

    let am  = mid_trs(1.0, 45.0, 2.0);
    let bm  = mid_trs(0.5, 30.0, 1.5);
    let pm  = Vec3::new(1.0, 2.0, 3.0);
    let ag  = glam_trs(1.0, 45.0, 2.0);
    let bg  = glam_trs(0.5, 30.0, 1.5);
    let pg  = glam::Vec3::new(1.0, 2.0, 3.0);

    // Multiply
    g.bench_function("mul/mid-math-pure",
        |b| b.iter(|| black_box(am) * black_box(bm)));
    g.bench_function("mul/glam-native",
        |b| b.iter(|| black_box(ag) * black_box(bg)));
    g.bench_function("mul/glam-field-copy-wrapped",
        |b| b.iter(|| glam_to_mat4(mat4_to_glam(black_box(am)) * mat4_to_glam(black_box(bm)))));
    g.bench_function("mul/glam-transmute-wrapped",
        |b| b.iter(|| unsafe {
            glam_to_mat4_transmute(mat4_to_glam_transmute(black_box(am)) * mat4_to_glam_transmute(black_box(bm)))
        }));

    // Transform point
    g.bench_function("transform_point/mid-math-pure",
        |b| b.iter(|| black_box(am).transform_point(black_box(pm))));
    g.bench_function("transform_point/glam-native",
        |b| b.iter(|| black_box(ag).transform_point3(black_box(pg))));
    g.bench_function("transform_point/glam-field-copy-wrapped",
        |b| b.iter(|| {
            let r = mat4_to_glam(black_box(am)).transform_point3a(vec3_to_glam(black_box(pm)));
            glam_to_vec3(r)
        }));

    // General inverse
    g.bench_function("inverse_general/mid-math-pure",
        |b| b.iter(|| black_box(am).inverse()));
    g.bench_function("inverse_general/glam-native",
        |b| b.iter(|| black_box(ag).inverse()));
    g.bench_function("inverse_general/glam-field-copy-wrapped",
        |b| b.iter(|| {
            // glam returns Mat4 directly (no Option). We re-wrap to match our API shape.
            let inv = mat4_to_glam(black_box(am)).inverse();
            Some(glam_to_mat4(inv))
        }));
    g.bench_function("inverse_general/glam-transmute-wrapped",
        |b| b.iter(|| unsafe {
            let inv = mat4_to_glam_transmute(black_box(am)).inverse();
            Some(glam_to_mat4_transmute(inv))
        }));

    // TRS inverse — mid-math fast path vs glam Mat4::inverse
    // Note: glam::Affine3A::inverse is faster than Mat4::inverse for TRS but
    // requires constructing from TRS components, not from a raw Mat4.
    // This measures the fair apples-to-apples: given a Mat4, invert it.
    g.bench_function("inverse_trs/mid-math-pure",
        |b| b.iter(|| black_box(am).inverse_trs()));
    g.bench_function("inverse_trs/glam-mat4-wrapped",
        |b| b.iter(|| unsafe {
            // Transmute to avoid field-copy overhead — measures compute cost only.
            glam_to_mat4_transmute(mat4_to_glam_transmute(black_box(am)).inverse())
        }));

    g.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// Group 3: 100k entity bulk transforms
//
// The most engine-relevant benchmark. Does the per-element conversion overhead
// compound over 100k entities, or does it vanish in the loop?
// ─────────────────────────────────────────────────────────────────────────────

fn bench_100k_wrapped(c: &mut Criterion) {
    const N: usize = 100_000;
    let mut g = c.benchmark_group("100k_entity_wrapped");
    g.throughput(Throughput::Elements(N as u64));

    let tm = mid_trs(1.0, 45.0, 1.0);
    let tg = glam_trs(1.0, 45.0, 1.0);

    let pos_m: Vec<Vec3>      = (0..N).map(|i| Vec3::new(i as f32 * 0.01, 0.0, 0.0)).collect();
    let pos_g: Vec<glam::Vec3> = (0..N).map(|i| glam::Vec3::new(i as f32 * 0.01, 0.0, 0.0)).collect();

    // A: pure mid-math
    g.bench_function("mid-math-pure", |b| {
        b.iter_batched(
            || pos_m.clone(),
            |mut pos| {
                for p in pos.iter_mut() { *p = tm.transform_point(black_box(*p)); }
                black_box(pos)
            },
            BatchSize::LargeInput,
        )
    });

    // B: glam native (reference ceiling)
    g.bench_function("glam-native", |b| {
        b.iter_batched(
            || pos_g.clone(),
            |mut pos| {
                for p in pos.iter_mut() { *p = tg.transform_point3(black_box(*p)); }
                black_box(pos)
            },
            BatchSize::LargeInput,
        )
    });

    // C1: glam wrapped, convert mat once, convert each position (field-copy)
    g.bench_function("glam-wrapped-field-copy", |b| {
        b.iter_batched(
            || pos_m.clone(),
            |mut pos| {
                let tg_wrapped = mat4_to_glam(tm);
                for p in pos.iter_mut() {
                    let pg = vec3_to_glam(*p);
                    *p = glam_to_vec3(tg_wrapped.transform_point3a(pg));
                }
                black_box(pos)
            },
            BatchSize::LargeInput,
        )
    });

    // C2: glam wrapped, transmute (zero-copy conversions, theoretical minimum)
    g.bench_function("glam-wrapped-transmute", |b| {
        b.iter_batched(
            || pos_m.clone(),
            |mut pos| {
                let tg_wrapped = unsafe { mat4_to_glam_transmute(tm) };
                for p in pos.iter_mut() {
                    let pg = unsafe { vec3_to_glam_transmute(*p) };
                    *p = unsafe { glam_to_vec3_transmute(tg_wrapped.transform_point3a(pg)) };
                }
                black_box(pos)
            },
            BatchSize::LargeInput,
        )
    });

    g.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// Group 4: 5k inverse TRS bulk
// ─────────────────────────────────────────────────────────────────────────────

fn bench_5k_inverse_trs(c: &mut Criterion) {
    const N: usize = 5_000;
    let mut g = c.benchmark_group("5k_inverse_trs_wrapped");
    g.throughput(Throughput::Elements(N as u64));

    let mats_m: Vec<Mat4> = (0..N)
        .map(|i| mid_trs(i as f32 * 0.1, i as f32, 1.0 + i as f32 * 0.001))
        .collect();
    let mats_g: Vec<glam::Mat4> = (0..N)
        .map(|i| glam_trs(i as f32 * 0.1, i as f32, 1.0 + i as f32 * 0.001))
        .collect();
    let mats_aff: Vec<glam::Affine3A> = (0..N)
        .map(|i| glam::Affine3A::from_scale_rotation_translation(
            glam::Vec3::splat(1.0 + i as f32 * 0.001),
            glam::Quat::from_rotation_y((i as f32).to_radians()),
            glam::Vec3::new(i as f32 * 0.1, 0.0, 0.0),
        ))
        .collect();

    // mid-math TRS inverse (SSE2 on x86_64)
    g.bench_function("mid-math-inverse_trs", |b| {
        b.iter_batched(
            || mats_m.clone(),
            |mats| {
                for m in &mats { black_box(m.inverse_trs()); }
                black_box(mats)
            },
            BatchSize::LargeInput,
        )
    });

    // glam Mat4 general inverse (their best SIMD inverse for a raw Mat4)
    g.bench_function("glam-mat4-inverse", |b| {
        b.iter_batched(
            || mats_g.clone(),
            |mats| {
                for m in &mats { black_box(m.inverse()); }
                black_box(mats)
            },
            BatchSize::LargeInput,
        )
    });

    // glam Affine3A inverse (glam's fastest TRS-specific path, for reference)
    g.bench_function("glam-affine3a-inverse", |b| {
        b.iter_batched(
            || mats_aff.clone(),
            |mats| {
                for m in &mats { black_box(m.inverse()); }
                black_box(mats)
            },
            BatchSize::LargeInput,
        )
    });

    // glam Mat4 inverse wrapped in mid-math types (field-copy)
    g.bench_function("glam-mat4-inverse-field-copy-wrapped", |b| {
        b.iter_batched(
            || mats_m.clone(),
            |mats| {
                for m in &mats {
                    black_box(glam_to_mat4(mat4_to_glam(*m).inverse()));
                }
                black_box(mats)
            },
            BatchSize::LargeInput,
        )
    });

    // glam Mat4 inverse wrapped in mid-math types (transmute)
    g.bench_function("glam-mat4-inverse-transmute-wrapped", |b| {
        b.iter_batched(
            || mats_m.clone(),
            |mats| {
                for m in &mats {
                    unsafe {
                        black_box(glam_to_mat4_transmute(mat4_to_glam_transmute(*m).inverse()));
                    }
                }
                black_box(mats)
            },
            BatchSize::LargeInput,
        )
    });

    g.finish();
}

criterion_group!(
    benches,
    bench_conversion_cost,
    bench_vec3,
    bench_mat4,
    bench_100k_wrapped,
    bench_5k_inverse_trs,
);
criterion_main!(benches);
