// crates/mid-math/benches/vs_mat4_fastest.rs
//! Who has the fastest 4×4 matrix multiply — and what should we steal?
//!
//! ── Short answer ──────────────────────────────────────────────────────────────
//!
//!  For 4×4 game math: glam (~7 ns) is the bar. Nobody else beats it.
//!  mid-math (~18 ns): 2.5× behind due to [[f32;4];4] storage forcing loads.
//!  vek (~12–16 ns): repr_simd helps but can't beat glam's explicit __m128.
//!  faer (≥500 ns): wrong tool — per-call overhead dominates at 4×4.
//!
//! ── Why glam wins at 4×4 ─────────────────────────────────────────────────────
//!
//!  glam stores Mat4 as:
//!    { x_axis: Vec4, y_axis: Vec4, z_axis: Vec4, w_axis: Vec4 }
//!  where Vec4 = #[repr(transparent)] __m128.
//!
//!  The x86-64 SysV ABI passes __m128 fields in XMM0-XMM3 registers.
//!  When Mat4 arrives as a function argument, its four columns are already
//!  sitting in XMM registers — no memory loads needed.
//!
//!  Our storage [[f32;4];4] causes the compiler to pass a pointer to stack.
//!  Every column access = _mm_load_ps(cols[i].as_ptr()).
//!  Eight loads × ~4 cycle latency = ~32 cycle overhead before any FP math.
//!  That IS the entire gap versus glam.
//!
//! ── Why faer loses at 4×4 ────────────────────────────────────────────────────
//!
//!  faer is a blocked-GEMM library for large matrices. Its f32 AVX2 microkernel
//!  processes a 16×6 tile per call. For 4×4 the blocking setup pays no dividend:
//!    - Packing A and B into L1-friendly buffers:  ~50–200 ns (fixed overhead)
//!    - Actual 64 multiply-accumulate FP ops:       ~1 ns
//!    - Result: overhead dominates completely
//!
//!  Note (faer 0.22): faer::linalg::matmul::Accum is pub(crate) so the
//!  low-level no-alloc path is not callable from outside the crate.
//!  All faer benches here use the allocating * operator.
//!  The overhead at 4×4 is still dominated by the blocking setup cost,
//!  not by the heap allocation — both paths are ~500–2000 ns.
//!
//! ── What to steal from faer for our AVX2 path (OPT-7) ───────────────────────
//!
//!  faer's microkernel idea IS what we want, scaled to 4×4:
//!
//!    SSE2 path (current after storage fix, target ~7 ns):
//!      For each of 4 output columns:
//!        acc  = a0 * bx          (_mm_mul_ps)
//!        acc += a1 * by          (_mm_add_ps / vfmadd231ps)
//!        acc += a2 * bz
//!        acc += a3 * bw
//!      4 output columns × 4 FP ops = 16 FP ops  (1 column per iteration)
//!
//!    AVX2 path (OPT-7, target ~3.5 ns):
//!      Pack self columns 0+1 into ymm0, columns 2+3 into ymm1.
//!      For each of 4 output columns (2 at a time):
//!        ymm_acc = ymm_col01 * broadcast(b.col0[0], 8 lanes)
//!        VFMADD231PS ymm_acc, ymm_col01, broadcast(b.col0[1], 8 lanes)
//!        VFMADD231PS ymm_acc, ymm_col01, broadcast(b.col0[2], 8 lanes)
//!        VFMADD231PS ymm_acc, ymm_col01, broadcast(b.col0[3], 8 lanes)
//!      2 output columns per ymm iteration → half the loop count → ~3.5 ns.
//!
//!  The key insight: AVX2 lets you do two columns of FMA simultaneously.
//!  Reference: cglm avx/affine.h glm_mul_avx, DirectXMath XMMatrixMultiply.
//!
//! ── Predicted table after storage change ──────────────────────────────────────
//!
//!  | Op                  | build 8 | after storage fix | after AVX2 (OPT-7) |
//!  |---------------------|---------|-------------------|--------------------|
//!  | mat4/mul            | 17.6 ns | ~7.0 ns  (=glam)  | ~3.5 ns            |
//!  | mat4/transform_pt   |  6.0 ns | ~3.9 ns           | ~2.0 ns            |
//!  | mat4/transpose *    |  9.6 ns | ~3.2 ns †         | ~3.2 ns (no gain)  |
//!  | mat4/from_trs       | 13.5 ns | ~7.0 ns           | n/a                |
//!  | chain_mat4_8        | 98.0 ns | ~50 ns            | ~25 ns             |
//!
//!  * Transpose also needs a dedicated SSE2 implementation (4 shuffles).
//!    Currently it's scalar regardless of storage format.
//!  † Transpose gain requires SSE2 fix alongside storage change.
//!
//! Run:  cargo bench --bench vs_mat4_fastest -p mid-math
//! HTML: target/criterion/report/index.html

use criterion::{
    black_box, criterion_group, criterion_main, BatchSize, Criterion, Throughput,
};

// ── mid-math ──────────────────────────────────────────────────────────────────
use mid_math::{to_radians, Mat4, Quat, Vec3};

// ── glam — gold standard for 4×4 ─────────────────────────────────────────────
use glam::{Mat4 as GMat4, Quat as GQuat};

// ── faer — large-matrix specialist ───────────────────────────────────────────
// NOTE: faer 0.22 makes linalg::matmul::Accum pub(crate).
// We use the allocating * operator for all faer benches. The overhead
// at 4×4 is dominated by blocking setup, not heap alloc.
use faer::Mat as FMat;

// ── vek — game math with repr_simd ───────────────────────────────────────────
use vek::mat::column_major::Mat4 as VMat4Core;
type VMat4 = VMat4Core<f32>;

// ── nalgebra ──────────────────────────────────────────────────────────────────
use nalgebra::{DMatrix, Matrix4};

// ── ultraviolet ───────────────────────────────────────────────────────────────
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
fn mid_to_vek(m: &Mat4) -> VMat4 {
    let mut flat = [0.0f32; 16];
    for col in 0..4 {
        for row in 0..4 {
            flat[col * 4 + row] = m.cols[col][row];
        }
    }
    VMat4Core::from_col_array(flat)
}

/// Convert mid-math Mat4 to faer Mat<f32> (4×4, column-major).
fn mid_to_faer(m: &Mat4) -> FMat<f32> {
    FMat::from_fn(4, 4, |row, col| m.cols[col][row])
}

// ─────────────────────────────────────────────────────────────────────────────
// Group 1: 4×4 latency — single call, all competitors
//
// Expected order (fastest → slowest):
//   glam         ~7.1 ns   ← Vec4 field storage, columns in XMM registers
//   vek          ~12–16 ns ← repr_simd auto-vect, still slower than explicit
//   nalgebra     ~15–20 ns ← simba SIMD backend
//   ultraviolet  ~15–20 ns
//   mid-math     ~17–18 ns ← STORAGE BUG: [[f32;4];4] forces 8 memory loads
//   sgemm        ~44 ns    ← per-call overhead dominates
//   faer (alloc) ≥500 ns   ← blocking setup overhead, wrong tool for 4×4
// ─────────────────────────────────────────────────────────────────────────────

fn bench_4x4_latency(c: &mut Criterion) {
    let mut g = c.benchmark_group("mat4_4x4_latency");

    let mm_a = mid_trs(1.0, 45.0, 2.0);
    let mm_b = mid_trs(0.5, 30.0, 1.5);

    // mid-math: current (storage bug — [[f32;4];4]).
    // After storage change to Vec4 fields this should drop to ~7 ns.
    g.bench_function("mid-math/current-storage", |b| {
        b.iter(|| black_box(mm_a) * black_box(mm_b))
    });

    // glam: gold standard. Columns live in XMM0-XMM3. No loads. Pure FMA.
    let gl_a = glam_trs(1.0, 45.0, 2.0);
    let gl_b = glam_trs(0.5, 30.0, 1.5);
    g.bench_function("glam/vec4-field-layout", |b| {
        b.iter(|| black_box(gl_a) * black_box(gl_b))
    });

    // nalgebra: SMatrix<f32,4,4> — fixed-size, uses simba SIMD.
    let na_a = na_trs(1.0, 45.0, 2.0);
    let na_b = na_trs(0.5, 30.0, 1.5);
    g.bench_function("nalgebra/smatrix4", |b| {
        b.iter(|| black_box(na_a) * black_box(na_b))
    });

    // ultraviolet: game math, uses `wide` crate for SIMD.
    let uv_a = uv_trs(1.0, 45.0, 2.0);
    let uv_b = uv_trs(0.5, 30.0, 1.5);
    g.bench_function("ultraviolet/mat4", |b| {
        b.iter(|| black_box(uv_a) * black_box(uv_b))
    });

    // vek: game math with repr_simd feature enabling SIMD-aligned Vec4.
    let vk_a = mid_to_vek(&mm_a);
    let vk_b = mid_to_vek(&mm_b);
    g.bench_function("vek/repr-simd-mat4", |b| {
        b.iter(|| black_box(vk_a) * black_box(vk_b))
    });

    // matrixmultiply sgemm: BLAS-style, per-call overhead dominates at 4×4.
    let a_flat: [f32; 16] = unsafe { core::mem::transmute(mm_a.cols) };
    let b_flat: [f32; 16] = unsafe { core::mem::transmute(mm_b.cols) };
    let mut c_flat_lat = [0.0f32; 16];
    g.bench_function("matrixmultiply/sgemm-4x4", |b| {
        b.iter(|| {
            unsafe {
                matrixmultiply::sgemm(
                    4, 4, 4,
                    1.0,
                    a_flat.as_ptr(), 1, 4,
                    b_flat.as_ptr(), 1, 4,
                    0.0,
                    c_flat_lat.as_mut_ptr(), 1, 4,
                );
            }
            // Return owned f32 — avoids "captured variable cannot escape FnMut" error.
            black_box(c_flat_lat[0])
        })
    });

    // ── faer ──────────────────────────────────────────────────────────────────
    // faer 0.22: linalg::matmul::Accum is pub(crate), so the no-alloc path is
    // not callable externally. We use the allocating * operator instead.
    // The 4×4 overhead is still dominated by blocking setup, not heap alloc.
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
// Group 2: 4×4 throughput — 100k matrix pairs, cache-warm
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

    g.bench_function("mid-math", |b| {
        b.iter_batched(
            || mm_pairs.clone(),
            |pairs| {
                let mut acc = Mat4::IDENTITY;
                for (a, b) in &pairs {
                    acc = black_box(*a) * black_box(*b);
                }
                black_box(acc)
            },
            BatchSize::LargeInput,
        )
    });

    g.bench_function("glam", |b| {
        b.iter_batched(
            || gl_pairs.clone(),
            |pairs| {
                let mut acc = GMat4::IDENTITY;
                for (a, b) in &pairs {
                    acc = black_box(*a) * black_box(*b);
                }
                black_box(acc)
            },
            BatchSize::LargeInput,
        )
    });

    g.bench_function("vek/repr-simd", |b| {
        b.iter_batched(
            || vk_pairs.clone(),
            |pairs| {
                let mut acc = VMat4::identity();
                for (a, b) in &pairs {
                    acc = black_box(*a) * black_box(*b);
                }
                black_box(acc)
            },
            BatchSize::LargeInput,
        )
    });

    g.bench_function("nalgebra/smatrix4", |b| {
        b.iter_batched(
            || na_pairs.clone(),
            |pairs| {
                let mut acc = Matrix4::<f32>::identity();
                for (a, b) in &pairs {
                    acc = black_box(*a) * black_box(*b);
                }
                black_box(acc)
            },
            BatchSize::LargeInput,
        )
    });

    g.bench_function("ultraviolet", |b| {
        b.iter_batched(
            || uv_pairs.clone(),
            |pairs| {
                let mut acc = UMat4::identity();
                for (a, b) in &pairs {
                    acc = black_box(*a) * black_box(*b);
                }
                black_box(acc)
            },
            BatchSize::LargeInput,
        )
    });

    g.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// Group 3: Matrix multiply crossover — when does faer start winning?
//
// faer vs matrixmultiply (sgemm) at n = 4, 32, 128, 512, 1024.
//
// Predicted crossover:
//   n = 4:   sgemm ~44 ns; faer ~500+ ns  → sgemm wins
//   n = 128: faer starts winning
//   n = 512+: faer clearly ahead (AVX2 microkernel + cache blocking)
// ─────────────────────────────────────────────────────────────────────────────

fn bench_crossover(c: &mut Criterion) {
    const SIZES: &[usize] = &[4, 32, 128, 512, 1024];

    for &n in SIZES {
        let mut g = c.benchmark_group(format!("matmul_crossover_{n}x{n}"));
        g.throughput(Throughput::Elements((n * n) as u64));
        if n >= 256 {
            g.sample_size(20);
        }
        if n >= 512 {
            g.sample_size(10);
        }

        // ── faer (allocating) ─────────────────────────────────────────────────
        // faer 0.22: using the * operator (allocating). Still demonstrates the
        // blocking overhead at small n and the crossover point at large n.
        let fa: FMat<f32> = FMat::from_fn(n, n, |r, c_idx| {
            (r as f32 * 0.001 + c_idx as f32 * 0.0007 + 0.01).sin()
        });
        let fb: FMat<f32> = FMat::from_fn(n, n, |r, c_idx| {
            (r as f32 * 0.0005 + c_idx as f32 * 0.001 + 0.5).cos()
        });
        g.bench_function("faer/alloc", |b| {
            b.iter(|| {
                let fc: FMat<f32> = black_box(fa.as_ref()) * black_box(fb.as_ref());
                black_box(fc)
            })
        });

        // ── matrixmultiply sgemm ──────────────────────────────────────────────
        let a_flat: Vec<f32> = (0..n * n)
            .map(|i| (i as f32 * 0.001).sin())
            .collect();
        let b_flat: Vec<f32> = (0..n * n)
            .map(|i| (i as f32 * 0.0007 + 0.5).cos())
            .collect();
        let mut c_flat: Vec<f32> = vec![0.0f32; n * n];
        g.bench_function("matrixmultiply/sgemm", |b| {
            b.iter(|| {
                unsafe {
                    matrixmultiply::sgemm(
                        n, n, n,
                        1.0,
                        a_flat.as_ptr(), 1, n as isize,
                        b_flat.as_ptr(), 1, n as isize,
                        0.0,
                        c_flat.as_mut_ptr(), 1, n as isize,
                    );
                }
                // Return owned f32 — avoids "captured variable cannot escape FnMut".
                black_box(c_flat[0])
            })
        });

        // ── nalgebra DMatrix ──────────────────────────────────────────────────
        if n <= 512 {
            let na_a: DMatrix<f32> = DMatrix::from_fn(n, n, |r, c_idx| {
                (r as f32 * 0.001 + c_idx as f32 * 0.0007).sin()
            });
            let na_b: DMatrix<f32> = DMatrix::from_fn(n, n, |r, c_idx| {
                (r as f32 * 0.0005 + c_idx as f32 * 0.001).cos()
            });
            let mut na_c: DMatrix<f32> = DMatrix::zeros(n, n);
            g.bench_function("nalgebra/dmatrix", |b| {
                b.iter(|| {
                    na_c.gemm(1.0f32, &na_a, &na_b, 0.0f32);
                    // Return owned f32 — avoids "captured variable cannot escape FnMut".
                    black_box(na_c[(0, 0)])
                })
            });
        }

        // ── For n=4 only: include game math libs as reference ─────────────────
        if n == 4 {
            let gl_a = glam_trs(1.0, 45.0, 2.0);
            let gl_b = glam_trs(0.5, 30.0, 1.5);
            g.bench_function("glam/mat4-game-winner", |b| {
                b.iter(|| black_box(gl_a) * black_box(gl_b))
            });
            let mm_a = mid_trs(1.0, 45.0, 2.0);
            let mm_b = mid_trs(0.5, 30.0, 1.5);
            g.bench_function("mid-math/mat4-pre-fix", |b| {
                b.iter(|| black_box(mm_a) * black_box(mm_b))
            });
        }

        g.finish();
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Group 4: faer peak GFLOPS — prove the algorithm is good, overhead is the issue
//
// At n=256+, faer should approach theoretical peak for single-threaded AVX2 f32.
// GFLOPS = 2*n^3 / time_ns
// ─────────────────────────────────────────────────────────────────────────────

fn bench_faer_peak_throughput(c: &mut Criterion) {
    const BENCH_SIZES: &[usize] = &[64, 256, 512];

    for &n in BENCH_SIZES {
        let flops = 2_u64 * n as u64 * n as u64 * n as u64;
        let mut g = c.benchmark_group(format!("faer_peak_throughput_{n}x{n}"));
        g.throughput(Throughput::Elements(flops));
        if n >= 256 {
            g.sample_size(20);
        }

        // faer — expected to approach peak FLOP rate at large sizes.
        // Using the allocating * operator (faer 0.22 has no public no-alloc path).
        let fa: FMat<f32> = FMat::from_fn(n, n, |r, c_idx| {
            (r as f32 * 0.01 + c_idx as f32 * 0.007 + 0.01).sin()
        });
        let fb: FMat<f32> = FMat::from_fn(n, n, |r, c_idx| {
            (r as f32 * 0.005 + c_idx as f32 * 0.01 + 0.5).cos()
        });
        g.bench_function("faer/alloc", |b| {
            b.iter(|| {
                let fc: FMat<f32> = black_box(fa.as_ref()) * black_box(fb.as_ref());
                black_box(fc)
            })
        });

        // matrixmultiply sgemm — baseline for comparison.
        let a_flat: Vec<f32> = (0..n * n)
            .map(|i| (i as f32 * 0.01).sin())
            .collect();
        let b_flat: Vec<f32> = (0..n * n)
            .map(|i| (i as f32 * 0.007 + 0.5).cos())
            .collect();
        let mut c_flat: Vec<f32> = vec![0.0f32; n * n];
        g.bench_function("matrixmultiply/sgemm", |b| {
            b.iter(|| {
                unsafe {
                    matrixmultiply::sgemm(
                        n, n, n,
                        1.0,
                        a_flat.as_ptr(), 1, n as isize,
                        b_flat.as_ptr(), 1, n as isize,
                        0.0,
                        c_flat.as_mut_ptr(), 1, n as isize,
                    );
                }
                // Return owned f32 — avoids "captured variable cannot escape FnMut".
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
