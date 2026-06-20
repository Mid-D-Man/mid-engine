// crates/mid-math/benches/vs_mat4_asm.rs
//! mat4 multiply: AVX intrinsic vs explicit asm! with pinned low ymm registers.
//!
//! Purpose
//! -------
//! Determine whether forcing ymm0–ymm9 (vs LLVM's free assignment which may
//! pick ymm8+) changes wall-clock time.  Also includes a 4-way independent
//! throughput batch to match cglm's measurement methodology (interleaved ops
//! with no cross-iteration data dependency).
//!
//! Run
//! ---
//!   cargo bench --bench vs_mat4_asm -p mid-math
//!
//! To inspect LLVM's asm for the intrinsic path:
//!   RUSTFLAGS='-C target-cpu=native --emit=asm' cargo build -p mid-math --release
//!   grep -A 40 'avx.*mat4\|mat4.*mul' target/release/deps/mid_math-*.s | head -80

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use mid_math::{to_radians, Mat4, Quat, Vec3};

// ── helpers ───────────────────────────────────────────────────────────────────

fn make(tx: f32, angle_deg: f32, sx: f32) -> Mat4 {
    Mat4::from_trs(
        Vec3::new(tx, 0.0, 0.0),
        Quat::from_axis_angle(Vec3::Y, to_radians(angle_deg)),
        Vec3::splat(sx),
    )
}

// ── explicit asm! path ────────────────────────────────────────────────────────
//
// Same algorithm as avx/mat4.rs col_pair() but with explicit ymm0-ymm9 names:
//
//   ymm0 = lhs col pair [col0|col1]          ymm2 = [col0|col0] broadcast
//   ymm1 = lhs col pair [col2|col3]          ymm3 = [col1|col1] broadcast
//   ymm4 = [col2|col2] broadcast             ymm5 = [col3|col3] broadcast
//   ymm6 = rhs col pair (scratch)            ymm7 = permuted row (scratch)
//   ymm8 = accumulator for out col pair 0,1
//   ymm9 = accumulator for out col pair 2,3
//
// Mat4 is 16-byte aligned (repr(C) of Vec4 fields); 32-byte vmovaps would
// fault — vmovups is used throughout.
//
// Compiled only when AVX + FMA are active (typically via -C target-cpu=native
// on CI; this bench is meaningless on plain SSE2).

#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    target_feature = "avx",
    target_feature = "fma",
))]
#[inline(never)]
unsafe fn mat4_mul_lowreg(a: *const Mat4, b: *const Mat4, out: *mut Mat4) {
    core::arch::asm!(
        // ── Load LHS column pairs ────────────────────────────────────────────
        "vmovups ymm0, [{a}]",           // ymm0 = [col0 | col1]
        "vmovups ymm1, [{a} + 32]",      // ymm1 = [col2 | col3]

        // ── Hoist: broadcast each LHS column to both 128-bit halves ──────────
        // vperm2f128 dst, src1, src2, imm:
        //   imm=0x00 → both halves = src1 low  (col0)
        //   imm=0x11 → both halves = src1 high (col1)
        "vperm2f128 ymm2, ymm0, ymm0, 0x00",  // [col0 | col0]
        "vperm2f128 ymm3, ymm0, ymm0, 0x11",  // [col1 | col1]
        "vperm2f128 ymm4, ymm1, ymm1, 0x00",  // [col2 | col2]
        "vperm2f128 ymm5, ymm1, ymm1, 0x11",  // [col3 | col3]

        // ── Output column pair (0, 1) ─────────────────────────────────────────
        // vpermilps operates independently on each 128-bit half:
        //   imm=0x00 → broadcast lane 0 within each half (row 0 of each rhs col)
        //   imm=0x55 → broadcast lane 1 (row 1), 0xAA → row 2, 0xFF → row 3
        "vmovups ymm6, [{b}]",
        "vpermilps ymm7, ymm6, 0x00",
        "vmulps ymm8, ymm2, ymm7",
        "vpermilps ymm7, ymm6, 0x55",
        "vfmadd231ps ymm8, ymm3, ymm7",
        "vpermilps ymm7, ymm6, 0xAA",
        "vfmadd231ps ymm8, ymm4, ymm7",
        "vpermilps ymm7, ymm6, 0xFF",
        "vfmadd231ps ymm8, ymm5, ymm7",

        // ── Output column pair (2, 3) ─────────────────────────────────────────
        "vmovups ymm6, [{b} + 32]",
        "vpermilps ymm7, ymm6, 0x00",
        "vmulps ymm9, ymm2, ymm7",
        "vpermilps ymm7, ymm6, 0x55",
        "vfmadd231ps ymm9, ymm3, ymm7",
        "vpermilps ymm7, ymm6, 0xAA",
        "vfmadd231ps ymm9, ymm4, ymm7",
        "vpermilps ymm7, ymm6, 0xFF",
        "vfmadd231ps ymm9, ymm5, ymm7",

        // ── Store ─────────────────────────────────────────────────────────────
        "vmovups [{out}], ymm8",
        "vmovups [{out} + 32], ymm9",

        // Transition guard: zero upper halves to avoid SSE→AVX penalty on exit
        "vzeroupper",

        a   = in(reg) a,
        b   = in(reg) b,
        out = in(reg) out,
        // Declare every modified YMM register as clobbered.
        out("ymm0") _, out("ymm1") _, out("ymm2") _, out("ymm3") _,
        out("ymm4") _, out("ymm5") _, out("ymm6") _, out("ymm7") _,
        out("ymm8") _, out("ymm9") _,
        options(nostack),
    );
}

// ── benchmarks ────────────────────────────────────────────────────────────────

fn bench_mat4_asm(c: &mut Criterion) {
    let mut g = c.benchmark_group("mat4_asm");

    let a = make(1.0, 45.0, 2.0);
    let b = make(0.5, 30.0, 1.5);

    // ── 1. Current mid-math Mul<Mat4> (dispatches to avx/mat4.rs when AVX+FMA) ─
    g.throughput(Throughput::Elements(1));
    g.bench_function("mid-math/avx-intrinsic", |bench| {
        bench.iter(|| black_box(black_box(a) * black_box(b)))
    });

    // ── 2. Explicit asm! version: same algorithm, ymm0-ymm9 pinned ────────────
    #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature = "avx",
        target_feature = "fma",
    ))]
    {
        let mut out = Mat4::ZERO;
        g.bench_function("mid-math/asm-lowreg", |bench| {
            bench.iter(|| unsafe {
                let aa = black_box(a);
                let bb = black_box(b);
                mat4_mul_lowreg(&aa, &bb, &mut out);
                black_box(out)
            })
        });
    }

    // ── 3. 4-way independent throughput ──────────────────────────────────────
    //
    // Four multiplies with no data dependency between them.  Saturates the AVX
    // execution units and reveals maximum throughput — analogous to cglm's
    // "4-way interleaved" measurement method that produces their ~4 ns figure.
    // Criterion reports per-element ns by dividing total time by Elements(4).
    {
        let a0 = make(1.0, 45.0, 2.0); let b0 = make(2.0, 90.0, 1.0);
        let a1 = make(0.5, 30.0, 1.5); let b1 = make(1.2, 75.0, 1.8);
        let a2 = make(1.5, 60.0, 1.2); let b2 = make(0.7, 20.0, 2.5);
        let a3 = make(0.8, 15.0, 0.9); let b3 = make(1.8, 10.0, 1.1);

        g.throughput(Throughput::Elements(4));
        g.bench_function("mid-math/4x-independent", |bench| {
            bench.iter(|| {
                let c0 = black_box(a0) * black_box(b0);
                let c1 = black_box(a1) * black_box(b1);
                let c2 = black_box(a2) * black_box(b2);
                let c3 = black_box(a3) * black_box(b3);
                black_box((c0, c1, c2, c3))
            })
        });
    }

    // ── 4. glam reference ─────────────────────────────────────────────────────
    {
        use glam::{Mat4 as GMat4, Quat as GQuat, Vec3 as GVec3};
        let ga = GMat4::from_scale_rotation_translation(
            GVec3::splat(2.0),
            GQuat::from_rotation_y(45f32.to_radians()),
            GVec3::new(1.0, 0.0, 0.0),
        );
        let gb = GMat4::from_scale_rotation_translation(
            GVec3::splat(1.5),
            GQuat::from_rotation_y(30f32.to_radians()),
            GVec3::new(0.5, 0.0, 0.0),
        );
        g.throughput(Throughput::Elements(1));
        g.bench_function("glam", |bench| {
            bench.iter(|| black_box(black_box(ga) * black_box(gb)))
        });
    }

    g.finish();
}

criterion_group!(benches, bench_mat4_asm);
criterion_main!(benches);
