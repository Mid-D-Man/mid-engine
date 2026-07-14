// crates/mid-math/benches/vs_mid_vec.rs
//! Criterion benchmarks: `Vec<T>` vs. `MidVec<T, N>` for the shapes this
//! crate actually needs — curve control points / keyframes (typically
//! 4-16 elements) and CSM cascade splits (typically 3-8 elements).
//!
//! Groups:
//!   mid_vec/construct_push        — push-one-at-a-time from empty, no
//!                                   pre-reserve (the realistic pattern
//!                                   for incrementally-parsed control
//!                                   points)
//!   mid_vec/construct_and_drop    — isolates the alloc/dealloc round trip
//!                                   at sizes at/under and over N
//!   mid_vec/bulk_collect          — FromIterator/collect from a
//!                                   known-size iterator (Vec's collect
//!                                   specialisation is a tough baseline —
//!                                   see the doc comment on
//!                                   `bench_bulk_collect` below)
//!   mid_vec/hermite_keys          — same push/collect shape, but with the
//!                                   actual `HermiteKey<Vec3>` element type
//!                                   `curves::hermite` stores
//!   mid_vec/array_of_curves       — many independent small curves (as in
//!                                   a per-bone animation rig), each
//!                                   holding 8 control points; sums every
//!                                   point across all curves
//!   mid_vec/csm_like_small        — very small (3-8 element) f32
//!                                   collections shaped like
//!                                   `camera::csm_split_depths`'s return
//!                                   value
//!
//! Honest framing, from actually running this on a throwaway crate before
//! writing it against the real one: `MidVec` clearly wins
//! `construct_push` (skips 1-2 of `Vec`'s early reallocation cycles by
//! getting `N` elements "for free") and `construct_and_drop` at/under `N`
//! (no allocation at all). It does *not* reliably beat `Vec` at
//! `bulk_collect` from an already-known-size iterator, because `Vec`'s own
//! `collect()` also only allocates once there, and `MidVec` still pays a
//! small per-element tag-check tax on top. That's an inherent trade-off of
//! any tagged inline/heap container, not a bug — see the module doc on
//! `mid_vec::mod` for the design rationale. Numbers on your hardware will
//! differ from whatever this prints in CI; look at the relative shape
//! (which groups favour which container), not the absolute nanoseconds.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use mid_math::{HermiteKey, MidVec, Vec3};

/// Matches the inline capacity this crate's own curve types would use —
/// see the discussion in the `curves` module docs.
const CURVE_N: usize = 8;
/// Matches a generous upper bound on shadow cascade count.
const CSM_N: usize = 8;

fn make_vec3(i: usize) -> Vec3 {
    Vec3::new(i as f32, (i as f32).sin(), (i as f32 * 0.3).cos())
}

fn make_hermite_key(i: usize) -> HermiteKey<Vec3> {
    let p = Vec3::new(i as f32, (i as f32).sin(), 0.0);
    let t = Vec3::new(1.0, (i as f32).cos(), 0.0);
    HermiteKey::smooth(p, t)
}

// ── construct via push loop, no pre-reserve ─────────────────────────────────

fn bench_construct_push(c: &mut Criterion) {
    let mut g = c.benchmark_group("mid_vec/construct_push");
    for &n in &[2usize, 4, 8, 16, 32] {
        g.bench_with_input(BenchmarkId::new("vec", n), &n, |b, &n| {
            b.iter(|| {
                let mut v: Vec<Vec3> = Vec::new();
                for i in 0..n {
                    v.push(black_box(make_vec3(i)));
                }
                black_box(v)
            })
        });
        g.bench_with_input(BenchmarkId::new("mid_vec", n), &n, |b, &n| {
            b.iter(|| {
                let mut v: MidVec<Vec3, CURVE_N> = MidVec::new();
                for i in 0..n {
                    v.push(black_box(make_vec3(i)));
                }
                black_box(v)
            })
        });
    }
    g.finish();
}

// ── construct + immediately drop: isolates alloc/dealloc round-trip cost ───

fn bench_construct_and_drop(c: &mut Criterion) {
    let mut g = c.benchmark_group("mid_vec/construct_and_drop");
    for &n in &[4usize, 8, 16] {
        g.bench_with_input(BenchmarkId::new("vec", n), &n, |b, &n| {
            b.iter(|| {
                let mut v: Vec<Vec3> = Vec::with_capacity(n);
                for i in 0..n {
                    v.push(make_vec3(i));
                }
                black_box(&v);
                drop(v);
            })
        });
        g.bench_with_input(BenchmarkId::new("mid_vec", n), &n, |b, &n| {
            b.iter(|| {
                let mut v: MidVec<Vec3, CURVE_N> = MidVec::new();
                for i in 0..n {
                    v.push(make_vec3(i));
                }
                black_box(&v);
                drop(v);
            })
        });
    }
    g.finish();
}

// ── bulk build via FromIterator/collect ─────────────────────────────────────
//
// `Vec::collect()` for a `TrustedLen`/`ExactSizeIterator` source allocates
// exactly once and writes via a raw-pointer fast path with no per-element
// branching at all. `MidVec` also allocates at most once here (`reserve`
// is called up front from the iterator's size hint), but every write still
// goes through the heap/inline tag check. So this group is the fairest
// "worst case" for `MidVec` — it does not have its usual advantage
// (skipping an allocation `Vec` would also skip).

fn bench_bulk_collect(c: &mut Criterion) {
    let mut g = c.benchmark_group("mid_vec/bulk_collect");
    for &n in &[8usize, 64, 512] {
        g.bench_with_input(BenchmarkId::new("vec", n), &n, |b, &n| {
            b.iter(|| black_box((0..n).map(make_vec3).collect::<Vec<_>>()))
        });
        g.bench_with_input(BenchmarkId::new("mid_vec", n), &n, |b, &n| {
            b.iter(|| black_box((0..n).map(make_vec3).collect::<MidVec<_, CURVE_N>>()))
        });
    }
    g.finish();
}

// ── same shapes, but with the actual HermiteKey<Vec3> element type ─────────

fn bench_hermite_keys(c: &mut Criterion) {
    let mut g = c.benchmark_group("mid_vec/hermite_keys");
    for &n in &[4usize, 8, 16] {
        g.bench_with_input(BenchmarkId::new("push_vec", n), &n, |b, &n| {
            b.iter(|| {
                let mut v: Vec<HermiteKey<Vec3>> = Vec::new();
                for i in 0..n {
                    v.push(black_box(make_hermite_key(i)));
                }
                black_box(v)
            })
        });
        g.bench_with_input(BenchmarkId::new("push_mid_vec", n), &n, |b, &n| {
            b.iter(|| {
                let mut v: MidVec<HermiteKey<Vec3>, CURVE_N> = MidVec::new();
                for i in 0..n {
                    v.push(black_box(make_hermite_key(i)));
                }
                black_box(v)
            })
        });
    }
    g.finish();
}

// ── the real target scenario: many independent small curves, each holding
//    CURVE_N control points, summed every frame — what a per-bone /
//    per-object animation-curve evaluation loop actually looks like. ───────

fn bench_array_of_curves(c: &mut Criterion) {
    let mut g = c.benchmark_group("mid_vec/array_of_curves_iterate_sum");
    for &curve_count in &[100usize, 1_000, 10_000] {
        // Vec<Vec<Vec3>>: one separate heap allocation per curve.
        let vec_of_vecs: Vec<Vec<Vec3>> =
            (0..curve_count).map(|_| (0..CURVE_N).map(make_vec3).collect()).collect();
        g.bench_with_input(BenchmarkId::new("vec_of_vec", curve_count), &vec_of_vecs, |b, data| {
            b.iter(|| {
                let mut acc = Vec3::new(0.0, 0.0, 0.0);
                for curve in data {
                    for p in curve {
                        acc += *p;
                    }
                }
                black_box(acc)
            })
        });

        // Vec<MidVec<Vec3, N>>: control points live inline, directly
        // inside each element of the outer Vec — no per-curve heap
        // indirection to chase.
        let vec_of_midvecs: Vec<MidVec<Vec3, CURVE_N>> = (0..curve_count)
            .map(|_| (0..CURVE_N).map(make_vec3).collect::<MidVec<_, CURVE_N>>())
            .collect();
        g.bench_with_input(
            BenchmarkId::new("vec_of_mid_vec", curve_count),
            &vec_of_midvecs,
            |b, data| {
                b.iter(|| {
                    let mut acc = Vec3::new(0.0, 0.0, 0.0);
                    for curve in data {
                        for p in curve.iter() {
                            acc += *p;
                        }
                    }
                    black_box(acc)
                })
            },
        );
    }
    g.finish();
}

// ── very small f32 collections shaped like camera::csm_split_depths ────────

fn bench_csm_like_small(c: &mut Criterion) {
    let mut g = c.benchmark_group("mid_vec/csm_like_small");
    for &cascades in &[3usize, 4, 6, 8] {
        g.bench_with_input(BenchmarkId::new("vec", cascades), &cascades, |b, &n| {
            b.iter(|| {
                let mut splits: Vec<f32> = Vec::with_capacity(n);
                for i in 0..n {
                    splits.push(black_box(i as f32 / n as f32));
                }
                black_box(splits)
            })
        });
        g.bench_with_input(BenchmarkId::new("mid_vec", cascades), &cascades, |b, &n| {
            b.iter(|| {
                let mut splits: MidVec<f32, CSM_N> = MidVec::new();
                for i in 0..n {
                    splits.push(black_box(i as f32 / n as f32));
                }
                black_box(splits)
            })
        });
    }
    g.finish();
}

criterion_group!(
    benches,
    bench_construct_push,
    bench_construct_and_drop,
    bench_bulk_collect,
    bench_hermite_keys,
    bench_array_of_curves,
    bench_csm_like_small,
);
criterion_main!(benches);
