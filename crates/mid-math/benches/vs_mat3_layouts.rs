// crates/mid-math/benches/vs_mat3_layouts.rs
//! Experiment: does the storage layout matter for 2D TRS matrix math, or
//! does it all come out in the wash once the compiler inlines everything?
//!
//! Three representations compete for the SAME logical operation
//! (compose two transforms, apply to a point, invert):
//!
//!   1. `Mat3`     — general 3×3, column-major `[[f32;3];3]` (36 bytes).
//!                   Cofactor-expansion inverse. Handles shear, any 3×3.
//!   2. `Affine2`  — TRS-specialized, 2×`Vec2` + `Vec2` translation (24 bytes).
//!                   Fast inverse via `M⁻¹ = S⁻¹ × Rᵀ` — assumes no shear.
//!   3. `f32 flat` — hand-written scalar math over a raw `[f32; 6]`
//!                   (`[m00,m01,m10,m11,tx,ty]`), no struct, no methods.
//!                   The theoretical floor: what if there were no
//!                   abstraction at all?
//!
//! ## Why this is a fair fight and why it isn't
//! `Affine2` and the flat array both encode the SAME mathematical
//! restriction (TRS only, no shear) — they SHOULD win on `inverse()`
//! against `Mat3` by algorithm alone, independent of layout. The
//! interesting number is `Affine2` vs `f32 flat`: if they're
//! statistically indistinguishable, the `Vec2`-based struct costs nothing
//! over hand-rolled scalar code and there's no reason to ever drop to raw
//! arrays. If `f32 flat` wins meaningfully, that's a real signal that
//! `Vec2`'s abstraction (even at `#[inline(always)]`) is leaking overhead.
//!
//! `Mat3` is included as the "general case tax" reference point — by
//! design it should lose on `inverse()` (different algorithm complexity)
//! but should be roughly competitive on `mul`/`transform_point` since
//! those don't exploit the TRS-only assumption as heavily.
//!
//! Run: cargo bench --bench vs_mat3_layouts -p mid-math

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use mid_math::{Affine2, Mat3, Vec2, Vec3};

// ═════════════════════════════════════════════════════════════════════════════
// f32-flat reference implementation
//
// Layout: [m00, m01, m10, m11, tx, ty]
//   | m00 m10 tx |
//   | m01 m11 ty |
//   |  0   0   1 |   (implicit, never stored)
//
// Mirrors Affine2's exact field layout (x_axis=[m00,m01], y_axis=[m10,m11],
// translation=[tx,ty]) so the algorithms are byte-for-byte the same ops,
// just without the Vec2 wrapper.
// ═════════════════════════════════════════════════════════════════════════════

type FlatAffine = [f32; 6];

#[inline(always)]
fn flat_identity() -> FlatAffine { [1.0, 0.0, 0.0, 1.0, 0.0, 0.0] }

#[inline(always)]
fn flat_from_parts(m00: f32, m01: f32, m10: f32, m11: f32, tx: f32, ty: f32) -> FlatAffine {
    [m00, m01, m10, m11, tx, ty]
}

/// Compose: equivalent to Affine2::mul — applies `rhs` then `self`.
#[inline(always)]
fn flat_mul(a: FlatAffine, b: FlatAffine) -> FlatAffine {
    let [a00, a01, a10, a11, atx, aty] = a;
    let [b00, b01, b10, b11, btx, bty] = b;
    [
        a00 * b00 + a10 * b01,
        a01 * b00 + a11 * b01,
        a00 * b10 + a10 * b11,
        a01 * b10 + a11 * b11,
        a00 * btx + a10 * bty + atx,
        a01 * btx + a11 * bty + aty,
    ]
}

#[inline(always)]
fn flat_transform_point(m: FlatAffine, px: f32, py: f32) -> (f32, f32) {
    let [m00, m01, m10, m11, tx, ty] = m;
    (m00 * px + m10 * py + tx, m01 * px + m11 * py + ty)
}

/// Fast TRS-only inverse — identical algorithm to Affine2::inverse,
/// just unrolled over raw floats instead of Vec2 operators.
#[inline(always)]
fn flat_inverse(m: FlatAffine) -> FlatAffine {
    let [m00, m01, m10, m11, tx, ty] = m;
    let sx2 = m00 * m00 + m01 * m01;
    let sy2 = m10 * m10 + m11 * m11;
    let isx = if sx2 < f32::EPSILON { 0.0 } else { 1.0 / sx2 };
    let isy = if sy2 < f32::EPSILON { 0.0 } else { 1.0 / sy2 };

    let inv00 = m00 * isx; let inv01 = m10 * isy;
    let inv10 = m01 * isx; let inv11 = m11 * isy;

    let inv_tx = -(inv00 * tx + inv10 * ty);
    let inv_ty = -(inv01 * tx + inv11 * ty);

    [inv00, inv01, inv10, inv11, inv_tx, inv_ty]
}

// ═════════════════════════════════════════════════════════════════════════════
// Construction
// ═════════════════════════════════════════════════════════════════════════════

fn bench_construction(c: &mut Criterion) {
    let mut g = c.benchmark_group("mat3_layouts/construction");

    g.bench_function("Mat3::from_cols",    |b| b.iter(|| {
        Mat3::from_cols(
            black_box(Vec3::new(1.0, 0.5, 0.0)),
            black_box(Vec3::new(-0.5, 1.0, 0.0)),
            black_box(Vec3::new(10.0, 20.0, 1.0)),
        )
    }));

    g.bench_function("Affine2::new_vec2",  |b| b.iter(|| {
        Affine2 {
            x_axis: black_box(Vec2::new(1.0, 0.5)),
            y_axis: black_box(Vec2::new(-0.5, 1.0)),
            translation: black_box(Vec2::new(10.0, 20.0)),
        }
    }));

    g.bench_function("flat_from_parts",    |b| b.iter(|| {
        flat_from_parts(
            black_box(1.0), black_box(0.5),
            black_box(-0.5), black_box(1.0),
            black_box(10.0), black_box(20.0),
        )
    }));

    g.finish();
}

// ═════════════════════════════════════════════════════════════════════════════
// Compose (mul) — the per-frame hierarchy-flatten hot path
// ═════════════════════════════════════════════════════════════════════════════

fn bench_compose(c: &mut Criterion) {
    let mut g = c.benchmark_group("mat3_layouts/compose");

    let a3 = Mat3::from_cols(Vec3::new(1.0, 0.2, 0.0), Vec3::new(-0.2, 1.0, 0.0), Vec3::new(5.0, 3.0, 1.0));
    let b3 = Mat3::from_cols(Vec3::new(0.9, -0.1, 0.0), Vec3::new(0.1, 0.9, 0.0), Vec3::new(-2.0, 1.0, 1.0));

    let a2 = Affine2 { x_axis: Vec2::new(1.0, 0.2), y_axis: Vec2::new(-0.2, 1.0), translation: Vec2::new(5.0, 3.0) };
    let b2 = Affine2 { x_axis: Vec2::new(0.9, -0.1), y_axis: Vec2::new(0.1, 0.9), translation: Vec2::new(-2.0, 1.0) };

    let af = flat_from_parts(1.0, 0.2, -0.2, 1.0, 5.0, 3.0);
    let bf = flat_from_parts(0.9, -0.1, 0.1, 0.9, -2.0, 1.0);

    g.bench_function("Mat3",    |b| b.iter(|| black_box(a3).mul_mat3(black_box(b3))));
    g.bench_function("Affine2", |b| b.iter(|| black_box(a2) * black_box(b2)));
    g.bench_function("f32_flat",|b| b.iter(|| flat_mul(black_box(af), black_box(bf))));

    g.finish();
}

// ═════════════════════════════════════════════════════════════════════════════
// Transform a point — the single most common per-vertex / per-entity call
// ═════════════════════════════════════════════════════════════════════════════

fn bench_transform_point(c: &mut Criterion) {
    let mut g = c.benchmark_group("mat3_layouts/transform_point");

    let m3 = Mat3::from_cols(Vec3::new(1.0, 0.2, 0.0), Vec3::new(-0.2, 1.0, 0.0), Vec3::new(5.0, 3.0, 1.0));
    let m2 = Affine2 { x_axis: Vec2::new(1.0, 0.2), y_axis: Vec2::new(-0.2, 1.0), translation: Vec2::new(5.0, 3.0) };
    let mf = flat_from_parts(1.0, 0.2, -0.2, 1.0, 5.0, 3.0);

    let p3 = Vec3::new(2.0, 3.0, 1.0); // homogeneous point for Mat3
    let p2 = Vec2::new(2.0, 3.0);

    g.bench_function("Mat3_mul_vec3",       |b| b.iter(|| black_box(m3).mul_vec3(black_box(p3))));
    g.bench_function("Affine2_transform",   |b| b.iter(|| black_box(m2).transform_point(black_box(p2))));
    g.bench_function("f32_flat_transform",  |b| b.iter(|| flat_transform_point(black_box(mf), black_box(2.0), black_box(3.0))));

    g.finish();
}

// ═════════════════════════════════════════════════════════════════════════════
// Inverse — where algorithm complexity (not just layout) diverges
// ═════════════════════════════════════════════════════════════════════════════

fn bench_inverse(c: &mut Criterion) {
    let mut g = c.benchmark_group("mat3_layouts/inverse");

    let m3 = Mat3::from_cols(Vec3::new(1.0, 0.2, 0.0), Vec3::new(-0.2, 1.0, 0.0), Vec3::new(5.0, 3.0, 1.0));
    let m2 = Affine2 { x_axis: Vec2::new(1.0, 0.2), y_axis: Vec2::new(-0.2, 1.0), translation: Vec2::new(5.0, 3.0) };
    let mf = flat_from_parts(1.0, 0.2, -0.2, 1.0, 5.0, 3.0);

    g.bench_function("Mat3_general_cofactor",   |b| b.iter(|| black_box(m3).try_inverse()));
    g.bench_function("Affine2_trs_specialized", |b| b.iter(|| black_box(m2).inverse()));
    g.bench_function("f32_flat_trs_specialized",|b| b.iter(|| flat_inverse(black_box(mf))));

    g.finish();
}

// ═════════════════════════════════════════════════════════════════════════════
// Batch throughput — 10,000 entity transforms composed + applied, the actual
// scene-graph-flatten workload this matters for in practice.
// ═════════════════════════════════════════════════════════════════════════════

fn bench_batch_10k(c: &mut Criterion) {
    let mut g = c.benchmark_group("mat3_layouts/batch_10k_compose_and_transform");
    g.throughput(criterion::Throughput::Elements(10_000));

    let parent3 = Mat3::from_cols(Vec3::new(1.0, 0.1, 0.0), Vec3::new(-0.1, 1.0, 0.0), Vec3::new(100.0, 50.0, 1.0));
    let children3: Vec<Mat3> = (0..10_000).map(|i| {
        let a = i as f32 * 0.001;
        Mat3::from_cols(Vec3::new(a.cos(), a.sin(), 0.0), Vec3::new(-a.sin(), a.cos(), 0.0), Vec3::new(i as f32 * 0.1, 0.0, 1.0))
    }).collect();

    let parent2 = Affine2 { x_axis: Vec2::new(1.0, 0.1), y_axis: Vec2::new(-0.1, 1.0), translation: Vec2::new(100.0, 50.0) };
    let children2: Vec<Affine2> = (0..10_000).map(|i| {
        let a = i as f32 * 0.001;
        Affine2 { x_axis: Vec2::new(a.cos(), a.sin()), y_axis: Vec2::new(-a.sin(), a.cos()), translation: Vec2::new(i as f32 * 0.1, 0.0) }
    }).collect();

    let parentf = flat_from_parts(1.0, 0.1, -0.1, 1.0, 100.0, 50.0);
    let childrenf: Vec<FlatAffine> = (0..10_000).map(|i| {
        let a = i as f32 * 0.001;
        flat_from_parts(a.cos(), a.sin(), -a.sin(), a.cos(), i as f32 * 0.1, 0.0)
    }).collect();

    g.bench_function("Mat3", |b| b.iter(|| {
        let mut sum = Vec3::ZERO;
        for &child in black_box(&children3) {
            let world = parent3.mul_mat3(child);
            sum = sum + world.mul_vec3(Vec3::new(1.0, 1.0, 1.0));
        }
        sum
    }));

    g.bench_function("Affine2", |b| b.iter(|| {
        let mut sum = Vec2::ZERO;
        for &child in black_box(&children2) {
            let world = parent2 * child;
            sum = sum + world.transform_point(Vec2::new(1.0, 1.0));
        }
        sum
    }));

    g.bench_function("f32_flat", |b| b.iter(|| {
        let mut sum = (0.0f32, 0.0f32);
        for &child in black_box(&childrenf) {
            let world = flat_mul(parentf, child);
            let p = flat_transform_point(world, 1.0, 1.0);
            sum = (sum.0 + p.0, sum.1 + p.1);
        }
        sum
    }));

    g.finish();
}

criterion_group!(
    mat3_layout_benches,
    bench_construction,
    bench_compose,
    bench_transform_point,
    bench_inverse,
    bench_batch_10k,
);
criterion_main!(mat3_layout_benches);
