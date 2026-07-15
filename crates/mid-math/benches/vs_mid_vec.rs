// crates/mid-math/benches/vs_mid_vec.rs
//! Criterion benchmarks: `MidVec<T, N>` against every other small-buffer /
//! growable-container strategy worth knowing about, for the shapes this
//! crate actually needs — curve control points / keyframes (typically
//! 4-16 elements) and CSM cascade splits (typically 3-8 elements).
//!
//! Containers compared (not all appear in every group — see the per-group
//! notes below for why):
//!   vec               `std::vec::Vec<T>` — the baseline everything is
//!                     measured against.
//!   vecdeque          `std::collections::VecDeque<T>` — the "ring vec":
//!                     heap-only like `Vec`, no small-buffer optimisation,
//!                     but a different growth/indexing strategy (power-of-
//!                     two ring, no memmove on front ops). Included to
//!                     check whether the ring-buffer approach itself buys
//!                     anything at these sizes — it has no inline storage,
//!                     so it's bucketed with `vec` (pre-sized, 1 alloc) in
//!                     `construct_and_drop` rather than with the small-
//!                     buffer types.
//!   mid_vec           This crate's own `MidVec<T, N>`.
//!   smallvec          `smallvec::SmallVec<[T; N]>`, `union` feature
//!                     enabled — same union+`MaybeUninit` storage strategy
//!                     as `MidVec`, no `T: Default` bound. The closest
//!                     structural relative `MidVec` has.
//!   tinyvec           `tinyvec::TinyVec<[T; N]>` — auto-spilling
//!                     `Inline(ArrayVec)`/`Heap(Vec)` enum. Requires
//!                     `T: Default` (`tinyvec::Array::Item: Default`).
//!   tinyvec_arrayvec  `tinyvec::ArrayVec<[T; N]>` — fixed-capacity, never
//!                     spills, `push` panics past `N`. Only benched at
//!                     sizes `<= N`, since that's the only regime it can
//!                     run in at all.
//!
//! Groups:
//!   mid_vec/construct_push        — push-one-at-a-time from empty, no
//!                                   pre-reserve (the realistic pattern
//!                                   for incrementally-parsed control
//!                                   points)
//!   mid_vec/construct_and_drop    — isolates the alloc/dealloc round trip
//!                                   at sizes at/under and over N.
//!                                   `tinyvec_arrayvec` is excluded here —
//!                                   it never allocates, so there's no
//!                                   round trip for this group to isolate,
//!                                   and it's `Copy` whenever `T: Copy`, so
//!                                   `drop()` on it is a no-op.
//!   mid_vec/bulk_collect          — FromIterator/collect from a
//!                                   known-size iterator (Vec's collect
//!                                   specialisation is a tough baseline —
//!                                   see the doc comment on
//!                                   `bench_bulk_collect` below).
//!                                   `tinyvec_arrayvec` is excluded here:
//!                                   every size in this group is well past
//!                                   `N`, so it isn't a type that could
//!                                   legally hold the result.
//!   mid_vec/hermite_keys          — same push/collect shape, but with the
//!                                   actual `HermiteKey<Vec3>` element type
//!                                   `curves::hermite` stores
//!   mid_vec/array_of_curves       — many independent small curves (as in
//!                                   a per-bone animation rig), each
//!                                   holding exactly `CURVE_N` control
//!                                   points (so this is the one group
//!                                   where `tinyvec_arrayvec` fits every
//!                                   curve exactly, with zero spare
//!                                   capacity); sums every point across
//!                                   all curves
//!   mid_vec/csm_like_small        — very small (3-8 element) f32
//!                                   collections shaped like
//!                                   `camera::csm_split_depths`'s return
//!                                   value; every size here is `<= CSM_N`,
//!                                   so `tinyvec_arrayvec` runs throughout
//!
//! Honest framing, from actually running this on a throwaway crate before
//! writing it against the real one: `MidVec` clearly wins
//! `construct_push` (skips 1-2 of `Vec`'s early reallocation cycles by
//! getting `N` elements "for free") and `construct_and_drop` at/under `N`
//! (no allocation at all). It does *not* reliably beat `Vec` at
//! `bulk_collect` from an already-known-size iterator, because `Vec`'s own
//! `collect()` also only allocates once there, and every tagged
//! inline/heap container (`MidVec`, `smallvec`, `tinyvec`) still pays a
//! small per-element tag-check tax on top — that's an inherent trade-off
//! of the design, not a bug specific to one implementation. Against
//! `smallvec` specifically (the closest relative — same union storage, no
//! `Default` bound), expect the two to track each other closely
//! everywhere; a persistent, reproducible gap there is the signal worth
//! investigating, not noise. Numbers on your hardware will differ from
//! whatever this prints in CI; look at the relative shape (which
//! containers win which groups), not the absolute nanoseconds.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use mid_math::{HermiteKey, MidVec, Vec3};
use smallvec::SmallVec;
use std::collections::VecDeque;
use tinyvec::{ArrayVec, TinyVec};

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
        g.bench_with_input(BenchmarkId::new("vecdeque", n), &n, |b, &n| {
            b.iter(|| {
                let mut v: VecDeque<Vec3> = VecDeque::new();
                for i in 0..n {
                    v.push_back(black_box(make_vec3(i)));
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
        g.bench_with_input(BenchmarkId::new("smallvec", n), &n, |b, &n| {
            b.iter(|| {
                let mut v: SmallVec<[Vec3; CURVE_N]> = SmallVec::new();
                for i in 0..n {
                    v.push(black_box(make_vec3(i)));
                }
                black_box(v)
            })
        });
        g.bench_with_input(BenchmarkId::new("tinyvec", n), &n, |b, &n| {
            b.iter(|| {
                let mut v: TinyVec<[Vec3; CURVE_N]> = TinyVec::new();
                for i in 0..n {
                    v.push(black_box(make_vec3(i)));
                }
                black_box(v)
            })
        });
        if n <= CURVE_N {
            g.bench_with_input(BenchmarkId::new("tinyvec_arrayvec", n), &n, |b, &n| {
                b.iter(|| {
                    let mut v: ArrayVec<[Vec3; CURVE_N]> = ArrayVec::new();
                    for i in 0..n {
                        v.push(black_box(make_vec3(i)));
                    }
                    black_box(v)
                })
            });
        }
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
        g.bench_with_input(BenchmarkId::new("vecdeque", n), &n, |b, &n| {
            b.iter(|| {
                let mut v: VecDeque<Vec3> = VecDeque::with_capacity(n);
                for i in 0..n {
                    v.push_back(make_vec3(i));
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
        g.bench_with_input(BenchmarkId::new("smallvec", n), &n, |b, &n| {
            b.iter(|| {
                let mut v: SmallVec<[Vec3; CURVE_N]> = SmallVec::new();
                for i in 0..n {
                    v.push(make_vec3(i));
                }
                black_box(&v);
                drop(v);
            })
        });
        g.bench_with_input(BenchmarkId::new("tinyvec", n), &n, |b, &n| {
            b.iter(|| {
                let mut v: TinyVec<[Vec3; CURVE_N]> = TinyVec::new();
                for i in 0..n {
                    v.push(make_vec3(i));
                }
                black_box(&v);
                drop(v);
            })
        });
        // `tinyvec_arrayvec` deliberately excluded from this group: it
        // never allocates at all (fixed-capacity, stack-only), so there is
        // no alloc/dealloc round trip for this group to isolate — and
        // since its backing storage is `Copy` whenever `T: Copy`, `drop()`
        // on it is a literal no-op the compiler warns about. It's already
        // covered at these same sizes in `construct_push`.
    }
    g.finish();
}

// ── bulk build via FromIterator/collect ─────────────────────────────────────
//
// `Vec::collect()` for a `TrustedLen`/`ExactSizeIterator` source allocates
// exactly once and writes via a raw-pointer fast path with no per-element
// branching at all. The tagged inline/heap containers (`MidVec`,
// `smallvec`, `tinyvec`) also allocate at most once here (`reserve` is
// called up front from the iterator's size hint), but every write still
// goes through the heap/inline tag check. So this group is the fairest
// "worst case" for all three of them — none has its usual advantage
// (skipping an allocation `Vec` would also skip). `vecdeque` has no
// inline storage either, but its ring layout means `collect()` still
// can't use `Vec`'s specialised raw-pointer fast path, so it's an open
// question whether it tracks `vec` or the tagged containers more closely.

fn bench_bulk_collect(c: &mut Criterion) {
    let mut g = c.benchmark_group("mid_vec/bulk_collect");
    for &n in &[8usize, 64, 512] {
        g.bench_with_input(BenchmarkId::new("vec", n), &n, |b, &n| {
            b.iter(|| black_box((0..n).map(make_vec3).collect::<Vec<_>>()))
        });
        g.bench_with_input(BenchmarkId::new("vecdeque", n), &n, |b, &n| {
            b.iter(|| black_box((0..n).map(make_vec3).collect::<VecDeque<_>>()))
        });
        g.bench_with_input(BenchmarkId::new("mid_vec", n), &n, |b, &n| {
            b.iter(|| black_box((0..n).map(make_vec3).collect::<MidVec<_, CURVE_N>>()))
        });
        g.bench_with_input(BenchmarkId::new("smallvec", n), &n, |b, &n| {
            b.iter(|| black_box((0..n).map(make_vec3).collect::<SmallVec<[Vec3; CURVE_N]>>()))
        });
        g.bench_with_input(BenchmarkId::new("tinyvec", n), &n, |b, &n| {
            b.iter(|| black_box((0..n).map(make_vec3).collect::<TinyVec<[Vec3; CURVE_N]>>()))
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
        g.bench_with_input(BenchmarkId::new("push_vecdeque", n), &n, |b, &n| {
            b.iter(|| {
                let mut v: VecDeque<HermiteKey<Vec3>> = VecDeque::new();
                for i in 0..n {
                    v.push_back(black_box(make_hermite_key(i)));
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
        g.bench_with_input(BenchmarkId::new("push_smallvec", n), &n, |b, &n| {
            b.iter(|| {
                let mut v: SmallVec<[HermiteKey<Vec3>; CURVE_N]> = SmallVec::new();
                for i in 0..n {
                    v.push(black_box(make_hermite_key(i)));
                }
                black_box(v)
            })
        });
        g.bench_with_input(BenchmarkId::new("push_tinyvec", n), &n, |b, &n| {
            b.iter(|| {
                let mut v: TinyVec<[HermiteKey<Vec3>; CURVE_N]> = TinyVec::new();
                for i in 0..n {
                    v.push(black_box(make_hermite_key(i)));
                }
                black_box(v)
            })
        });
        if n <= CURVE_N {
            g.bench_with_input(BenchmarkId::new("push_tinyvec_arrayvec", n), &n, |b, &n| {
                b.iter(|| {
                    let mut v: ArrayVec<[HermiteKey<Vec3>; CURVE_N]> = ArrayVec::new();
                    for i in 0..n {
                        v.push(black_box(make_hermite_key(i)));
                    }
                    black_box(v)
                })
            });
        }
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

        // Vec<VecDeque<Vec3>>: same per-curve heap indirection as vec_of_vec,
        // but each curve is a ring buffer instead of a contiguous slice.
        let vec_of_vecdeques: Vec<VecDeque<Vec3>> =
            (0..curve_count).map(|_| (0..CURVE_N).map(make_vec3).collect()).collect();
        g.bench_with_input(
            BenchmarkId::new("vec_of_vecdeque", curve_count),
            &vec_of_vecdeques,
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

        // Vec<SmallVec<[Vec3; N]>>: same inline-per-element idea as
        // vec_of_mid_vec, servo's implementation instead of ours.
        let vec_of_smallvecs: Vec<SmallVec<[Vec3; CURVE_N]>> = (0..curve_count)
            .map(|_| (0..CURVE_N).map(make_vec3).collect::<SmallVec<[Vec3; CURVE_N]>>())
            .collect();
        g.bench_with_input(
            BenchmarkId::new("vec_of_smallvec", curve_count),
            &vec_of_smallvecs,
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

        // Vec<TinyVec<[Vec3; N]>>: auto-spilling enum variant.
        let vec_of_tinyvecs: Vec<TinyVec<[Vec3; CURVE_N]>> = (0..curve_count)
            .map(|_| (0..CURVE_N).map(make_vec3).collect::<TinyVec<[Vec3; CURVE_N]>>())
            .collect();
        g.bench_with_input(
            BenchmarkId::new("vec_of_tinyvec", curve_count),
            &vec_of_tinyvecs,
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

        // Vec<ArrayVec<[Vec3; N]>>: every curve holds exactly N points, so
        // this is the one spot the fixed-capacity type fits with zero
        // spare capacity and zero risk of the panic-on-overflow path.
        let vec_of_arrayvecs: Vec<ArrayVec<[Vec3; CURVE_N]>> = (0..curve_count)
            .map(|_| (0..CURVE_N).map(make_vec3).collect::<ArrayVec<[Vec3; CURVE_N]>>())
            .collect();
        g.bench_with_input(
            BenchmarkId::new("vec_of_tinyvec_arrayvec", curve_count),
            &vec_of_arrayvecs,
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
        g.bench_with_input(BenchmarkId::new("vecdeque", cascades), &cascades, |b, &n| {
            b.iter(|| {
                let mut splits: VecDeque<f32> = VecDeque::with_capacity(n);
                for i in 0..n {
                    splits.push_back(black_box(i as f32 / n as f32));
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
        g.bench_with_input(BenchmarkId::new("smallvec", cascades), &cascades, |b, &n| {
            b.iter(|| {
                let mut splits: SmallVec<[f32; CSM_N]> = SmallVec::new();
                for i in 0..n {
                    splits.push(black_box(i as f32 / n as f32));
                }
                black_box(splits)
            })
        });
        g.bench_with_input(BenchmarkId::new("tinyvec", cascades), &cascades, |b, &n| {
            b.iter(|| {
                let mut splits: TinyVec<[f32; CSM_N]> = TinyVec::new();
                for i in 0..n {
                    splits.push(black_box(i as f32 / n as f32));
                }
                black_box(splits)
            })
        });
        g.bench_with_input(BenchmarkId::new("tinyvec_arrayvec", cascades), &cascades, |b, &n| {
            b.iter(|| {
                let mut splits: ArrayVec<[f32; CSM_N]> = ArrayVec::new();
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
