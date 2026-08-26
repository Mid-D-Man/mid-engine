// crates/mid-math/benches/vs_wide_float.rs
//! Float wide vector benchmarks: f32x4 / Vec3x4 / QuatX4 / Vec3x8 vs
//! scalar equivalents, vs glam, and vs the wide-vector crates that
//! share the actual same concept — `wide` (bare N-packed-lanes) and
//! `ultraviolet` (SoA Vec3x4/Vec3x8/Rotor3x4/Rotor3x8 — the one other
//! crate on crates.io with genuine 4-and-8-wide vector/rotor batch
//! types, not just a per-instance component vector). glam 0.33,
//! wide 1.6, and ultraviolet 0.10 were already dev-dependencies before
//! this pass — no version bumps needed.
//!
//! The key question answered by each group:
//!   f32x4       — does mid-math's own bare 4-lane f32 type (previously
//!                 unbenched — only ever used inline as lerp/nlerp's
//!                 `t` param) hold up against wide::f32x4, the true
//!                 apples-to-apples same-concept type?
//!   vec3_4wide  — does Vec3x4 process 4× the work for the same
//!                 instruction count, and how does that compare to
//!                 glam::Vec3A (per-instance) and ultraviolet::Vec3x4
//!                 (the other crate's SoA wide vector)?
//!   quat_4wide  — do 4 Hamilton products cost the same as 1, vs
//!                 glam::Quat and ultraviolet::Rotor3x4 (geometric-
//!                 algebra rotor, not a quaternion, but the same role)?
//!   vec3_8wide  — Vec3x8 (AVX2-only) op-level coverage, vs
//!                 ultraviolet::Vec3x8 (no glam row — glam has no
//!                 8-wide type at all).
//!   100k_scalar — baseline entity transform loop (plain Mat4::transform_point)
//!   100k_wide   — Vec3x4 batched loop (Mat4::transform_vec3x4, 4 per iter)
//!   100k_wide8  — Vec3x8 batched loop (AVX2 only, 8 per iter)
//!
//! Comparison methodology — each point below was verified directly
//! against the relevant crate's published source before use, not
//! recalled from memory:
//!
//!   * glam::Vec3A (not plain Vec3) is the fair per-instance comparison
//!     — matches this crate's own vs_glam.rs convention exactly, since
//!     Vec3A is glam's SIMD-aligned type.
//!   * glam has no `nlerp` method. `.lerp()` on glam::Quat already *is*
//!     the unnormalized-lerp operation — confirmed against this
//!     crate's own vs_glam.rs, which already ships an
//!     `nlerp/glam-lerp` row on that basis.
//!   * ultraviolet's Rotor3 construction API is angle+plane
//!     (`from_rotation_xz`/`from_rotation_yz`), not axis-angle —
//!     confirmed directly against ultraviolet's rotor.rs, no
//!     `from_axis_angle` exists there. Exact rotation semantics don't
//!     matter for a wall-clock benchmark, only realistic non-degenerate
//!     operand values do — `from_rotation_xz`/`yz` stand in for
//!     "around Y" / "around X" purely for that reason. Passing a
//!     splatted f32x4 angle straight to the x4 constructor broadcasts
//!     to all 4 lanes directly, so no separate rotor-splat step is
//!     needed the way Vec3x4 needs `Vec3x4::splat()`.
//!   * `normalize_precise` (not the fast `normalize` row) is where the
//!     ultraviolet Vec3x4/Vec3x8 comparison lands: ultraviolet's only
//!     `.normalized()` path is full `mag()` + divide, the same
//!     algorithm as mid-math's normalize_precise — not the rsqrt +
//!     Newton-Raphson fast path mid-math's plain `.normalize()` uses.
//!     Putting it under the fast row would silently compare two
//!     different algorithms under the same label.
//!   * ultraviolet re-exports its own internal `wide` crate version
//!     (0.7.x) as `ultraviolet::f32x4`/`ultraviolet::f32x8` — a
//!     *different* type from this workspace's own `wide = "1.6"` dev-
//!     dependency despite the identical name, confirmed by reading
//!     ultraviolet's Cargo.toml directly. Never pass one where the
//!     other is expected; this file always calls through
//!     `ultraviolet::f32x4::splat(...)` for ultraviolet operands.
//!   * `ultraviolet` is NOT available under `wasm32` — it sits in this
//!     crate's `[target.'cfg(not(target_arch = "wasm32"))'.dev-dependencies]`
//!     block (confirmed directly in Cargo.toml), unlike glam and wide
//!     which are unconditional dev-dependencies. Every ultraviolet
//!     comparison row in this file is grouped into its own
//!     `#[cfg(not(target_arch = "wasm32"))]` block for that reason —
//!     see vec3_4wide/quat_4wide below. vec3_8wide needs no separate
//!     wasm32 gate: its AVX2 gate already excludes wasm32 by
//!     construction.
//!   * glam::Vec4 has no `recip_sqrt`/rsqrt method (checked directly) —
//!     that row is mid-math + wide only in the f32x4 group.
//!
//! No QuatX8/Rotor3x8 group: mid-math has no 8-wide quaternion type
//! yet (AVX2 tier only has Vec3x8) — nothing of ours to bench there.
//! ultraviolet *does* have Rotor3x8 (confirmed in rotor.rs), noted here
//! for whenever mid-math grows the type.
//!
//! Run: cargo bench --bench vs_wide_float -p mid-math
//! HTML: target/criterion/report/index.html

use criterion::{
    black_box, criterion_group, criterion_main,
    BatchSize, Criterion, Throughput,
};
use mid_math::{
    to_radians,
    Vec3, Quat, Mat4,
    Vec3x4, QuatX4, f32x4,
};

#[cfg(not(target_arch = "wasm32"))]
use ultraviolet::Lerp;

#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    target_feature = "avx2",
))]
use mid_math::Vec3x8;

#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    target_feature = "avx2",
))]
#[cfg(target_arch = "x86")]
use core::arch::x86::_mm256_set1_ps;
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    target_feature = "avx2",
))]
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::_mm256_set1_ps;

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
// Group 0: bare f32x4 vs wide::f32x4 vs glam::Vec4
//
// mid-math's own f32x4 is the same concept `wide::f32x4` is: 4
// independent packed f32 lanes, not a math vector — the same
// relationship wide::i32x4 had to mid-math's own i32x4 in
// vs_wide_int.rs. glam::Vec4 is a secondary, different-concept
// baseline (4 packed floats used as one vector's x/y/z/w) — same call
// vs_wide_int.rs already made keeping glam::IVec4 alongside wide::i32x4.
// This type had ZERO bench coverage before this pass.
// ─────────────────────────────────────────────────────────────────────────────

fn bench_f32x4(c: &mut Criterion) {
    let mut g = c.benchmark_group("f32x4");

    let wa = f32x4::new(1.0, 2.0, 3.0, 4.0);
    let wb = f32x4::new(5.0, 6.0, 7.0, 8.0);
    let da = wide::f32x4::new([1.0, 2.0, 3.0, 4.0]);
    let db = wide::f32x4::new([5.0, 6.0, 7.0, 8.0]);
    let ga = glam::Vec4::new(1.0, 2.0, 3.0, 4.0);
    let gb = glam::Vec4::new(5.0, 6.0, 7.0, 8.0);

    // ── add ──────────────────────────────────────────────────────────────────
    g.bench_function("add/f32x4",      |b| b.iter(|| black_box(wa) + black_box(wb)));
    g.bench_function("add/wide-f32x4", |b| b.iter(|| black_box(da) + black_box(db)));
    g.bench_function("add/glam-Vec4",  |b| b.iter(|| black_box(ga) + black_box(gb)));

    // ── mul ──────────────────────────────────────────────────────────────────
    g.bench_function("mul/f32x4",      |b| b.iter(|| black_box(wa) * black_box(wb)));
    g.bench_function("mul/wide-f32x4", |b| b.iter(|| black_box(da) * black_box(db)));
    g.bench_function("mul/glam-Vec4",  |b| b.iter(|| black_box(ga) * black_box(gb)));

    // ── min / max ────────────────────────────────────────────────────────────
    g.bench_function("min/f32x4",      |b| b.iter(|| black_box(wa).min(black_box(wb))));
    g.bench_function("min/wide-f32x4", |b| b.iter(|| black_box(da).min(black_box(db))));
    g.bench_function("min/glam-Vec4",  |b| b.iter(|| black_box(ga).min(black_box(gb))));
    g.bench_function("max/f32x4",      |b| b.iter(|| black_box(wa).max(black_box(wb))));
    g.bench_function("max/wide-f32x4", |b| b.iter(|| black_box(da).max(black_box(db))));
    g.bench_function("max/glam-Vec4",  |b| b.iter(|| black_box(ga).max(black_box(gb))));

    // ── abs ──────────────────────────────────────────────────────────────────
    let neg   = f32x4::new(-1.0, 2.0, -3.0, 4.0);
    let neg_d = wide::f32x4::new([-1.0, 2.0, -3.0, 4.0]);
    let neg_g = glam::Vec4::new(-1.0, 2.0, -3.0, 4.0);
    g.bench_function("abs/f32x4",      |b| b.iter(|| black_box(neg).abs()));
    g.bench_function("abs/wide-f32x4", |b| b.iter(|| black_box(neg_d).abs()));
    g.bench_function("abs/glam-Vec4",  |b| b.iter(|| black_box(neg_g).abs()));

    // ── sqrt / recip ─────────────────────────────────────────────────────────
    g.bench_function("sqrt/f32x4",      |b| b.iter(|| black_box(wa).sqrt()));
    g.bench_function("sqrt/wide-f32x4", |b| b.iter(|| black_box(da).sqrt()));
    g.bench_function("sqrt/glam-Vec4",  |b| b.iter(|| black_box(ga).sqrt()));
    g.bench_function("recip/f32x4",      |b| b.iter(|| black_box(wa).recip()));
    g.bench_function("recip/wide-f32x4", |b| b.iter(|| black_box(da).recip()));
    g.bench_function("recip/glam-Vec4",  |b| b.iter(|| black_box(ga).recip()));
    // recip_sqrt — mid-math + wide only; glam::Vec4 has no recip_sqrt /
    // rsqrt method (checked directly against glam's source).
    g.bench_function("recip_sqrt/f32x4",      |b| b.iter(|| black_box(wa).recip_sqrt()));
    g.bench_function("recip_sqrt/wide-f32x4", |b| b.iter(|| black_box(da).recip_sqrt()));

    // ── mul_add ──────────────────────────────────────────────────────────────
    g.bench_function("mul_add/f32x4",      |b| b.iter(|| black_box(wa).mul_add(black_box(wb), black_box(wa))));
    g.bench_function("mul_add/wide-f32x4", |b| b.iter(|| black_box(da).mul_add(black_box(db), black_box(da))));
    g.bench_function("mul_add/glam-Vec4",  |b| b.iter(|| black_box(ga).mul_add(black_box(gb), black_box(ga))));

    // ── clamp ────────────────────────────────────────────────────────────────
    let lo   = f32x4::splat(0.0);
    let hi   = f32x4::splat(10.0);
    let lo_d = wide::f32x4::splat(0.0);
    let hi_d = wide::f32x4::splat(10.0);
    let lo_g = glam::Vec4::splat(0.0);
    let hi_g = glam::Vec4::splat(10.0);
    g.bench_function("clamp/f32x4",      |b| b.iter(|| black_box(wa).clamp(black_box(lo), black_box(hi))));
    g.bench_function("clamp/wide-f32x4", |b| b.iter(|| black_box(da).clamp(black_box(lo_d), black_box(hi_d))));
    g.bench_function("clamp/glam-Vec4",  |b| b.iter(|| black_box(ga).clamp(black_box(lo_g), black_box(hi_g))));

    // ── cmpeq + blend — mid-math's own idiom, same shape as
    //    vs_wide_int.rs's cmpeq+blend/i32x4 row. wide::f32x4/glam::Vec4
    //    use different mask calling conventions (simd_eq / BVec4
    //    select) — not benched here for the same reason vs_wide_int.rs
    //    gave for skipping wide's shift/equality ops: an unverified
    //    calling convention isn't worth benching blind.
    g.bench_function("cmpeq+blend/f32x4", |b| {
        b.iter(|| {
            let m = black_box(wa).cmpeq(black_box(wb));
            f32x4::blend(m, black_box(wa), black_box(wb))
        })
    });

    g.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// Group 1: Vec3x4 operations vs scalar Vec3, glam::Vec3A, ultraviolet::Vec3x4
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

    // glam — Vec3A is glam's SIMD-aligned Vec3, the fair comparison
    // (matches this crate's own vs_glam.rs convention exactly).
    let gv = glam::Vec3A::new(1.0, 2.0, 3.0);
    let go = glam::Vec3A::new(4.0, 5.0, 6.0);

    // ── add ──────────────────────────────────────────────────────────────────
    g.bench_function("add/scalar-x1", |b| {
        b.iter(|| black_box(sv) + black_box(so))
    });
    g.bench_function("add/wide-x4",   |b| {
        b.iter(|| black_box(wv) + black_box(wo))
    });
    g.bench_function("add/glam-Vec3A", |b| {
        b.iter(|| black_box(gv) + black_box(go))
    });

    // ── dot ──────────────────────────────────────────────────────────────────
    g.bench_function("dot/scalar-x1", |b| {
        b.iter(|| black_box(sv).dot(black_box(so)))
    });
    g.bench_function("dot/wide-x4",   |b| {
        b.iter(|| black_box(wv).dot(black_box(wo)))
    });
    g.bench_function("dot/glam-Vec3A", |b| {
        b.iter(|| black_box(gv).dot(black_box(go)))
    });

    // ── cross ─────────────────────────────────────────────────────────────────
    g.bench_function("cross/scalar-x1", |b| {
        b.iter(|| black_box(sv).cross(black_box(so)))
    });
    g.bench_function("cross/wide-x4",   |b| {
        b.iter(|| black_box(wv).cross(black_box(wo)))
    });
    g.bench_function("cross/glam-Vec3A", |b| {
        b.iter(|| black_box(gv).cross(black_box(go)))
    });

    // ── normalize ─────────────────────────────────────────────────────────────
    g.bench_function("normalize/scalar-x1", |b| {
        b.iter(|| black_box(sv).normalize())
    });
    g.bench_function("normalize/wide-x4",   |b| {
        b.iter(|| black_box(wv).normalize())
    });
    g.bench_function("normalize/glam-Vec3A", |b| {
        b.iter(|| black_box(gv).normalize())
    });
    // normalize_precise gets the ultraviolet comparison below (same
    // full sqrt+divide algorithm), not this fast rsqrt+NR row — see
    // file header.
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
    g.bench_function("lerp/glam-Vec3A", |b| {
        b.iter(|| black_box(gv).lerp(black_box(go), 0.5))
    });

    // ── ultraviolet::Vec3x4 — the true wide-vector apples-to-apples
    //    target (same SoA x/y/z-as-lanes layout as mid-math's own
    //    Vec3x4). Not available under wasm32 — see file header.
    #[cfg(not(target_arch = "wasm32"))]
    {
        let uv = ultraviolet::Vec3x4::splat(ultraviolet::Vec3::new(1.0, 2.0, 3.0));
        let uo = ultraviolet::Vec3x4::splat(ultraviolet::Vec3::new(4.0, 5.0, 6.0));

        g.bench_function("add/ultraviolet-Vec3x4", |b| {
            b.iter(|| black_box(uv) + black_box(uo))
        });
        g.bench_function("dot/ultraviolet-Vec3x4", |b| {
            b.iter(|| black_box(uv).dot(black_box(uo)))
        });
        g.bench_function("cross/ultraviolet-Vec3x4", |b| {
            b.iter(|| black_box(uv).cross(black_box(uo)))
        });
        // paired with normalize_precise, not normalize — see file header.
        g.bench_function("normalize_precise/ultraviolet-Vec3x4", |b| {
            b.iter(|| black_box(uv).normalized())
        });
        g.bench_function("lerp/ultraviolet-Vec3x4", |b| {
            let t = ultraviolet::f32x4::splat(0.5);
            b.iter(|| black_box(uv).lerp(black_box(uo), black_box(t)))
        });
    }

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
// Group 2: QuatX4 vs scalar Quat, glam::Quat, ultraviolet::Rotor3x4
// ─────────────────────────────────────────────────────────────────────────────

fn bench_quat_4wide(c: &mut Criterion) {
    let mut g = c.benchmark_group("quat_4wide");

    let sq1  = Quat::from_axis_angle(Vec3::Y, to_radians(45.0));
    let sq2  = Quat::from_axis_angle(Vec3::X, to_radians(30.0));
    let sv   = Vec3::X;
    let wq1  = QuatX4::splat(sq1);
    let wq2  = QuatX4::splat(sq2);
    let wv   = Vec3x4::splat(sv);

    let gq1 = glam::Quat::from_axis_angle(glam::Vec3::Y, to_radians(45.0));
    let gq2 = glam::Quat::from_axis_angle(glam::Vec3::X, to_radians(30.0));
    let gv  = glam::Vec3::X;

    // ── mul ──────────────────────────────────────────────────────────────────
    g.bench_function("mul/scalar-x1", |b| {
        b.iter(|| black_box(sq1) * black_box(sq2))
    });
    g.bench_function("mul/wide-x4",   |b| {
        b.iter(|| black_box(wq1) * black_box(wq2))
    });
    g.bench_function("mul/glam", |b| {
        b.iter(|| black_box(gq1) * black_box(gq2))
    });

    // ── rotate ───────────────────────────────────────────────────────────────
    g.bench_function("rotate/scalar-x1", |b| {
        b.iter(|| black_box(sq1).rotate(black_box(sv)))
    });
    g.bench_function("rotate/wide-x4",   |b| {
        b.iter(|| black_box(wq1).rotate(black_box(wv)))
    });
    g.bench_function("rotate/glam", |b| {
        b.iter(|| black_box(gq1) * black_box(gv))
    });

    // ── nlerp ────────────────────────────────────────────────────────────────
    // glam has no `nlerp` — `.lerp()` on glam::Quat already is the
    // unnormalized-lerp operation. Confirmed against, and matches,
    // this crate's own vs_glam.rs `nlerp/glam-lerp` row.
    g.bench_function("nlerp/scalar-x1", |b| {
        b.iter(|| black_box(sq1).nlerp(black_box(sq2), 0.5))
    });
    g.bench_function("nlerp/wide-x4",   |b| {
        let t = f32x4::splat(0.5);
        b.iter(|| black_box(wq1).nlerp(black_box(wq2), black_box(t)))
    });
    g.bench_function("nlerp/glam-lerp", |b| {
        b.iter(|| black_box(gq1).lerp(black_box(gq2), 0.5))
    });

    // ── normalize ─────────────────────────────────────────────────────────────
    g.bench_function("normalize/scalar-x1", |b| {
        b.iter(|| black_box(sq1).normalize())
    });
    g.bench_function("normalize/wide-x4",   |b| {
        b.iter(|| black_box(wq1).normalize())
    });
    g.bench_function("normalize/glam", |b| {
        b.iter(|| black_box(gq1).normalize())
    });

    // ── ultraviolet::Rotor3x4 — geometric-algebra rotor, not a
    //    quaternion, but the same functional role: mul = compose,
    //    `rotor * vec` = rotate, `.lerp()` via ultraviolet's Lerp
    //    trait. Not available under wasm32 — see file header.
    #[cfg(not(target_arch = "wasm32"))]
    {
        // ultraviolet's Rotor3 has no from_axis_angle — its API is
        // angle+plane. from_rotation_xz/yz stand in for "around Y" /
        // "around X" purely to get realistic operands; see file header.
        let uq1 = ultraviolet::Rotor3x4::from_rotation_xz(ultraviolet::f32x4::splat(to_radians(45.0)));
        let uq2 = ultraviolet::Rotor3x4::from_rotation_yz(ultraviolet::f32x4::splat(to_radians(30.0)));
        let uv  = ultraviolet::Vec3x4::splat(ultraviolet::Vec3::unit_x());

        g.bench_function("mul/ultraviolet-Rotor3x4", |b| {
            b.iter(|| black_box(uq1) * black_box(uq2))
        });
        g.bench_function("rotate/ultraviolet-Rotor3x4", |b| {
            b.iter(|| black_box(uq1) * black_box(uv))
        });
        g.bench_function("nlerp/ultraviolet-lerp", |b| {
            let t = ultraviolet::f32x4::splat(0.5);
            b.iter(|| black_box(uq1).lerp(black_box(uq2), black_box(t)))
        });
        g.bench_function("normalize/ultraviolet-Rotor3x4", |b| {
            b.iter(|| black_box(uq1).normalized())
        });
    }

    g.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// Group 3: Vec3x8 (AVX2-only) vs ultraviolet::Vec3x8
//
// Mid-math's Vec3x8 had no direct op-level bench coverage before this
// pass — it only ever appeared inside 100k_entity_transforms_wide, via
// a two-Vec3x4-passes workaround (Mat4 has no transform_vec3x8 yet —
// a real gap, flagged there and here, out of scope for a bench-only
// pass). No glam row: glam has no 8-wide vector type at all. No
// separate wasm32 gate needed here beyond the AVX2 gate — AVX2 and
// wasm32 are mutually exclusive by construction.
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    target_feature = "avx2",
))]
fn bench_vec3_8wide(c: &mut Criterion) {
    let mut g = c.benchmark_group("vec3_8wide");

    let sv8: [Vec3; 8] = core::array::from_fn(|i| Vec3::new(1.0 + i as f32, 2.0, 3.0));
    let so8: [Vec3; 8] = core::array::from_fn(|i| Vec3::new(4.0, 5.0 + i as f32, 6.0));
    let wv = Vec3x8::from_slice(&sv8);
    let wo = Vec3x8::from_slice(&so8);

    let uv8: [ultraviolet::Vec3; 8] =
        core::array::from_fn(|i| ultraviolet::Vec3::new(1.0 + i as f32, 2.0, 3.0));
    let uo8: [ultraviolet::Vec3; 8] =
        core::array::from_fn(|i| ultraviolet::Vec3::new(4.0, 5.0 + i as f32, 6.0));
    let uv: ultraviolet::Vec3x8 = uv8.into();
    let uo: ultraviolet::Vec3x8 = uo8.into();

    // ── add ──────────────────────────────────────────────────────────────────
    g.bench_function("add/vec3x8", |b| {
        b.iter(|| black_box(wv) + black_box(wo))
    });
    g.bench_function("add/ultraviolet-Vec3x8", |b| {
        b.iter(|| black_box(uv) + black_box(uo))
    });

    // ── dot ──────────────────────────────────────────────────────────────────
    g.bench_function("dot/vec3x8", |b| {
        b.iter(|| black_box(wv).dot(black_box(wo)))
    });
    g.bench_function("dot/ultraviolet-Vec3x8", |b| {
        b.iter(|| black_box(uv).dot(black_box(uo)))
    });

    // ── cross ─────────────────────────────────────────────────────────────────
    g.bench_function("cross/vec3x8", |b| {
        b.iter(|| black_box(wv).cross(black_box(wo)))
    });
    g.bench_function("cross/ultraviolet-Vec3x8", |b| {
        b.iter(|| black_box(uv).cross(black_box(uo)))
    });

    // ── normalize / normalize_precise ───────────────────────────────────────
    // ultraviolet's single `.normalized()` pairs with normalize_precise,
    // not the fast row — same reasoning as vec3_4wide, see file header.
    g.bench_function("normalize/vec3x8", |b| {
        b.iter(|| black_box(wv).normalize())
    });
    g.bench_function("normalize_precise/vec3x8", |b| {
        b.iter(|| black_box(wv).normalize_precise())
    });
    g.bench_function("normalize_precise/ultraviolet-Vec3x8", |b| {
        b.iter(|| black_box(uv).normalized())
    });

    // ── lerp ─────────────────────────────────────────────────────────────────
    // Vec3x8::lerp takes a raw __m256 directly (unlike Vec3x4, which
    // wraps it in f32x4) — confirmed directly against avx2/vec3x8.rs.
    g.bench_function("lerp/vec3x8", |b| {
        let t = unsafe { _mm256_set1_ps(0.5) };
        b.iter(|| black_box(wv).lerp(black_box(wo), black_box(t)))
    });
    g.bench_function("lerp/ultraviolet-Vec3x8", |b| {
        let t = ultraviolet::f32x8::splat(0.5);
        b.iter(|| black_box(uv).lerp(black_box(uo), black_box(t)))
    });

    g.finish();
}

/// Dispatch wrapper so `criterion_group!` below doesn't need a
/// conditionally-assembled function list — mirrors vs_wide_int.rs's
/// `bench_avx2_types` wrapper exactly.
fn bench_vec3_8wide_dispatch(c: &mut Criterion) {
    #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature = "avx2",
    ))]
    {
        bench_vec3_8wide(c);
    }
    #[cfg(not(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature = "avx2",
    )))]
    {
        let _ = c;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Group 4: 100k entity transforms — the engine-critical benchmark
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
// Group 5: 100k quaternion rotations — animation blend comparison
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
// Group 6: AoS→SoA transpose cost (amortisation check)
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
    bench_f32x4,
    bench_vec3_4wide,
    bench_quat_4wide,
    bench_vec3_8wide_dispatch,
    bench_100k_transforms,
    bench_100k_quat_rotations,
    bench_wide_transpose,
);
criterion_main!(benches);
