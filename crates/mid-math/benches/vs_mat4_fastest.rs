// crates/mid-math/benches/vs_mat4_fastest.rs
//! Who has the fastest 4×4 matrix multiply?
//!
//! ── Build 8 update ───────────────────────────────────────────────────────────
//!
//! mid-math storage changed from [[f32;4];4] to four Vec4 fields.
//! Expected result: mid-math latency drops from ~17.6 ns to ~7 ns (parity glam).
//! Bench labels updated: "current-storage" → "vec4-field-layout".
//!
//! Remaining gap after storage fix:
//!   mat4/mul ~7 ns (SSE2, no FMA) vs cglm ~3.5 ns (FMA via -march=native C)
//!   Next target: OPT-7 AVX2 two-column-per-ymm approach → ~3.5 ns.
//!
//! Run: cargo bench --bench vs_mat4_fastest -p mid-math

use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion, Throughput};

use mid_math::{to_radians, Mat4, Quat, Vec3};
use glam::{Mat4 as GMat4, Quat as GQuat};
use faer::Mat as FMat;
use vek::mat::column_major::Mat4 as VMat4Core;
type VMat4 = VMat4Core<f32>;
use nalgebra::{DMatrix, Matrix4};
use ultraviolet::Mat4 as UMat4;

// ─────────────────────────────────────────────────────────────────────────────
// Shared construction helpers
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
    use nalgebra::{Isometry3, Translation3, Unit, UnitQuaternion, Vector3};
    let iso = Isometry3::from_parts(
        Translation3::new(tx, 0.0, 0.0),
        UnitQuaternion::from_axis_angle(
            &Unit::new_normalize(Vector3::y()),
            angle_deg.to_radians(),
        ),
    );
    iso.to_homogeneous() * Matrix4::new_scaling(sx)
}

fn uv_trs(tx: f32, angle_deg: f32, sx: f32) -> UMat4 {
    let t = UMat4::from_translation(ultraviolet::Vec3::new(tx, 0.0, 0.0));
    let r = UMat4::from_rotation_y(angle_deg.to_radians());
    let s = UMat4::from_nonuniform_scale(ultraviolet::Vec3::broadcast(sx));
    t * r * s
}

/// Convert mid-math Mat4 to vek column-major Mat4<f32>.
///
/// Build 8: Mat4 is repr(C) over four Vec4 fields = 64 bytes.
/// Transmute to [f32;16] is safe — layout identical to [[f32;4];4].
fn mid_to_vek(m: &Mat4) -> VMat4 {
    let flat: [f32; 16] = unsafe { core::mem::transmute(*m) };
    VMat4Core::from_col_array(flat)
}

/// Convert mid-math Mat4 to faer Mat<f32> (4×4, column-major).
///
/// Build 8: use to_array() on each Vec4 field — no .cols indexing.
fn mid_to_faer(m: &Mat4) -> FMat<f32> {
    let cols = [
        m.x_axis.to_array(),
        m.y_axis.to_array(),
        m.z_axis.to_array(),
        m.w_axis.to_array(),
    ];
    FMat::from_fn(4, 4, |row, col| cols[col][row])
}

// ─────────────────────────────────────────────────────────────────────────────
// Group 1: 4×4 latency
// ─────────────────────────────────────────────────────────────────────────────

fn bench_4x4_latency(c: &mut Criterion) {
    let mut g = c.benchmark_group("mat4_4x4_latency");

    let mm_a = mid_trs(1.0, 45.0, 2.0);
    let mm_b = mid_trs(0.5, 30.0, 1.5);

    // Build 8: Vec4 field storage — expect ~7 ns (parity glam SSE2).
    // After AVX2 OPT-7: ~3.5 ns.
    g.bench_function("mid-math/vec4-field-layout", |b| {
        b.iter(|| black_box(mm_a) * black_box(mm_b))
    });

    let gl_a = glam_trs(1.0, 45.0, 2.0);
    let gl_b = glam_trs(0.5, 30.0, 1.5);
    g.bench_function("glam/vec4-field-layout", |b| {
        b.iter(|| black_box(gl_a) * black_box(gl_b))
    });

    let na_a = na_trs(1.0, 45.0, 2.0);
    let na_b = na_trs(0.5, 30.0, 1.5);
    g.bench_function("nalgebra/smatrix4", |b| {
        b.iter(|| black_box(na_a) * black_box(na_b))
    });

    let uv_a = uv_trs(1.0, 45.0, 2.0);
    let uv_b = uv_trs(0.5, 30.0, 1.5);
    g.bench_function("ultraviolet/mat4", |b| {
        b.iter(|| black_box(uv_a) * black_box(uv_b))
    });

    let vk_a = mid_to_vek(&mm_a);
    let vk_b = mid_to_vek(&mm_b);
    g.bench_function("vek/repr-simd-mat4", |b| {
        b.iter(|| black_box(vk_a) * black_box(vk_b))
    });

    // Build 8: transmute whole Mat4 (Vec4 fields) — layout identical to old cols.
    let a_flat: [f32; 16] = unsafe { core::mem::transmute(mm_a) };
    let b_flat: [f32; 16] = unsafe { core::mem::transmute(mm_b) };
    let mut c_flat_lat = [0.0f32; 16];
    g.bench_function("matrixmultiply/sgemm-4x4", |b| {
        b.iter(|| {
            unsafe {
                matrixmultiply::sgemm(
                    4, 4, 4, 1.0,
                    a_flat.as_ptr(), 1, 4,
                    b_flat.as_ptr(), 1, 4,
                    0.0,
                    c_flat_lat.as_mut_ptr(), 1, 4,
                );
            }
            black_box(c_flat_lat[0])
        })
    });

    let fa = mid_to_faer(&mm_a);
    let fb = mid_to_faer(&mm_b);
    g.bench_function("faer/alloc-4x4", |b| {
        b.iter(|| {
            let fc: FMat<f32> = black_box(fa.as_ref()) * black_box(fb.as_ref());
            black_box(fc)
        })
    });

    g.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// Group 2: 4×4 throughput — 100k matrix pairs
// ─────────────────────────────────────────────────────────────────────────────

fn bench_4x4_throughput_100k(c: &mut Criterion) {
    const N: usize = 100_000;
    let mut g = c.benchmark_group("mat4_4x4_throughput_100k");
    g.throughput(Throughput::Elements(N as u64));

    let mm_pairs: Vec<(Mat4, Mat4)> = (0..N)
        .map(|i| (
            mid_trs(i as f32 * 0.01, i as f32 * 0.1, 1.001),
            mid_trs(-(i as f32) * 0.005, i as f32 * 0.05 + 1.0, 0.999),
        ))
        .collect();

    let gl_pairs: Vec<(GMat4, GMat4)> = (0..N)
        .map(|i| (
            glam_trs(i as f32 * 0.01, i as f32 * 0.1, 1.001),
            glam_trs(-(i as f32) * 0.005, i as f32 * 0.05 + 1.0, 0.999),
        ))
        .collect();

    let vk_pairs: Vec<(VMat4, VMat4)> = mm_pairs.iter()
        .map(|(a, b)| (mid_to_vek(a), mid_to_vek(b)))
        .collect();

    let na_pairs: Vec<(Matrix4<f32>, Matrix4<f32>)> = (0..N)
        .map(|i| (
            na_trs(i as f32 * 0.01, i as f32 * 0.1, 1.001),
            na_trs(-(i as f32) * 0.005, i as f32 * 0.05 + 1.0, 0.999),
        ))
        .collect();

    let uv_pairs: Vec<(UMat4, UMat4)> = (0..N)
        .map(|i| (
            uv_trs(i as f32 * 0.01, i as f32 * 0.1, 1.001),
            uv_trs(-(i as f32) * 0.005, i as f32 * 0.05 + 1.0, 0.999),
        ))
        .collect();

    g.bench_function("mid-math", |b| b.iter_batched(
        || mm_pairs.clone(),
        |pairs| { let mut acc = Mat4::IDENTITY; for (a, b) in &pairs { acc = black_box(*a) * black_box(*b); } black_box(acc) },
        BatchSize::LargeInput,
    ));
    g.bench_function("glam", |b| b.iter_batched(
        || gl_pairs.clone(),
        |pairs| { let mut acc = GMat4::IDENTITY; for (a, b) in &pairs { acc = black_box(*a) * black_box(*b); } black_box(acc) },
        BatchSize::LargeInput,
    ));
    g.bench_function("vek/repr-simd", |b| b.iter_batched(
        || vk_pairs.clone(),
        |pairs| { let mut acc = VMat4::identity(); for (a, b) in &pairs { acc = black_box(*a) * black_box(*b); } black_box(acc) },
        BatchSize::LargeInput,
    ));
    g.bench_function("nalgebra/smatrix4", |b| b.iter_batched(
        || na_pairs.clone(),
        |pairs| { let mut acc = Matrix4::<f32>::identity(); for (a, b) in &pairs { acc = black_box(*a) * black_box(*b); } black_box(acc) },
        BatchSize::LargeInput,
    ));
    g.bench_function("ultraviolet", |b| b.iter_batched(
        || uv_pairs.clone(),
        |pairs| { let mut acc = UMat4::identity(); for (a, b) in &pairs { acc = black_box(*a) * black_box(*b); } black_box(acc) },
        BatchSize::LargeInput,
    ));

    g.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// Group 3: Matrix multiply crossover
// ─────────────────────────────────────────────────────────────────────────────

fn bench_crossover(c: &mut Criterion) {
    const SIZES: &[usize] = &[4, 32, 128, 512, 1024];

    for &n in SIZES {
        let mut g = c.benchmark_group(format!("matmul_crossover_{n}x{n}"));
        g.throughput(Throughput::Elements((n * n) as u64));
        if n >= 256 { g.sample_size(20); }
        if n >= 512 { g.sample_size(10); }

        let fa: FMat<f32> = FMat::from_fn(n, n, |r, c_idx| {
            (r as f32 * 0.001 + c_idx as f32 * 0.0007 + 0.01).sin()
        });
        let fb: FMat<f32> = FMat::from_fn(n, n, |r, c_idx| {
            (r as f32 * 0.0005 + c_idx as f32 * 0.001 + 0.5).cos()
        });
        g.bench_function("faer/alloc", |b| {
            b.iter(|| { let fc: FMat<f32> = black_box(fa.as_ref()) * black_box(fb.as_ref()); black_box(fc) })
        });

        let a_flat: Vec<f32> = (0..n * n).map(|i| (i as f32 * 0.001).sin()).collect();
        let b_flat: Vec<f32> = (0..n * n).map(|i| (i as f32 * 0.0007 + 0.5).cos()).collect();
        let mut c_flat: Vec<f32> = vec![0.0f32; n * n];
        g.bench_function("matrixmultiply/sgemm", |b| {
            b.iter(|| {
                unsafe {
                    matrixmultiply::sgemm(
                        n, n, n, 1.0,
                        a_flat.as_ptr(), 1, n as isize,
                        b_flat.as_ptr(), 1, n as isize,
                        0.0,
                        c_flat.as_mut_ptr(), 1, n as isize,
                    );
                }
                black_box(c_flat[0])
            })
        });

        if n <= 512 {
            let na_a: DMatrix<f32> = DMatrix::from_fn(n, n, |r, c_idx| {
                (r as f32 * 0.001 + c_idx as f32 * 0.0007).sin()
            });
            let na_b: DMatrix<f32> = DMatrix::from_fn(n, n, |r, c_idx| {
                (r as f32 * 0.0005 + c_idx as f32 * 0.001).cos()
            });
            let mut na_c: DMatrix<f32> = DMatrix::zeros(n, n);
            g.bench_function("nalgebra/dmatrix", |b| {
                b.iter(|| { na_c.gemm(1.0f32, &na_a, &na_b, 0.0f32); black_box(na_c[(0, 0)]) })
            });
        }

        if n == 4 {
            let gl_a = glam_trs(1.0, 45.0, 2.0);
            let gl_b = glam_trs(0.5, 30.0, 1.5);
            g.bench_function("glam/mat4-game-winner", |b| {
                b.iter(|| black_box(gl_a) * black_box(gl_b))
            });
            let mm_a = mid_trs(1.0, 45.0, 2.0);
            let mm_b = mid_trs(0.5, 30.0, 1.5);
            // Build 8: label updated from "mat4-pre-fix" to "mat4-post-fix"
            g.bench_function("mid-math/mat4-post-fix", |b| {
                b.iter(|| black_box(mm_a) * black_box(mm_b))
            });
        }

        g.finish();
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Group 4: faer peak GFLOPS
// ─────────────────────────────────────────────────────────────────────────────

fn bench_faer_peak_throughput(c: &mut Criterion) {
    const BENCH_SIZES: &[usize] = &[64, 256, 512];

    for &n in BENCH_SIZES {
        let flops = 2_u64 * n as u64 * n as u64 * n as u64;
        let mut g = c.benchmark_group(format!("faer_peak_throughput_{n}x{n}"));
        g.throughput(Throughput::Elements(flops));
        if n >= 256 { g.sample_size(20); }

        let fa: FMat<f32> = FMat::from_fn(n, n, |r, c_idx| {
            (r as f32 * 0.01 + c_idx as f32 * 0.007 + 0.01).sin()
        });
        let fb: FMat<f32> = FMat::from_fn(n, n, |r, c_idx| {
            (r as f32 * 0.005 + c_idx as f32 * 0.01 + 0.5).cos()
        });
        g.bench_function("faer/alloc", |b| {
            b.iter(|| { let fc: FMat<f32> = black_box(fa.as_ref()) * black_box(fb.as_ref()); black_box(fc) })
        });

        let a_flat: Vec<f32> = (0..n * n).map(|i| (i as f32 * 0.01).sin()).collect();
        let b_flat: Vec<f32> = (0..n * n).map(|i| (i as f32 * 0.007 + 0.5).cos()).collect();
        let mut c_flat: Vec<f32> = vec![0.0f32; n * n];
        g.bench_function("matrixmultiply/sgemm", |b| {
            b.iter(|| {
                unsafe {
                    matrixmultiply::sgemm(
                        n, n, n, 1.0,
                        a_flat.as_ptr(), 1, n as isize,
                        b_flat.as_ptr(), 1, n as isize,
                        0.0,
                        c_flat.as_mut_ptr(), 1, n as isize,
                    );
                }
                black_box(c_flat[0])
            })
        });

        g.finish();
    }
}

// ─────────────────────────────────────────────────────────────────────────────

criterion_group!(
    benches,
    bench_4x4_latency,
    bench_4x4_throughput_100k,
    bench_crossover,
    bench_faer_peak_throughput,
);
criterion_main!(benches);
