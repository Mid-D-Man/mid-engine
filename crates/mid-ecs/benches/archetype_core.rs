//! Criterion benchmarks for the Archetype Core's own operations —
//! `mid-ecs` only, no `bevy_ecs` dependency. Complements, doesn't
//! replace, `benches/ecs-vs-bevy-ecs`: that crate is isolated in its
//! own workspace member specifically because `bevy_ecs` needs rustc
//! 1.95+ (see its own header comment and root `Cargo.toml`'s
//! explanatory block), which locks it out of any toolchain below that
//! — including the rustc-1.91 sandbox this project has been developed
//! against for most of its life. This suite exists so mid-ecs's own
//! operations can be measured and regression-guarded on *any*
//! toolchain this workspace already supports, without waiting for a
//! real CI run against the newer toolchain just to get a number back.
//!
//! Run: `cargo bench -p mid-ecs --bench archetype_core`
//! Report: `target/criterion/report/index.html` (same `html_reports`
//! convention as `mid-collections`'s and `mid-math`'s own benches).
//!
//! Sizes: same `[100, 1_000, 10_000, 100_000]` sweep as
//! `mid-collections/benches/sparse_set.rs`, bracketing
//! `docs/architecture.md`'s stated "100,000+ entities per core" target.
//!
//! GROUPS:
//!
//! - `spawn_insert_bundle`: raw two-component entity creation
//!   throughput, matching `benches/ecs-vs-bevy-ecs`'s own `spawn` group
//!   shape (`spawn()` into the empty archetype, then `insert_bundle`
//!   migrates into the final one — the same real, honest architectural
//!   step that bench's own header comment already documents as
//!   non-apples-to-apples against bevy's single-call `spawn(bundle)`).
//!   **Not yet investigated** — real CI shows mid-ecs ~2.9-5x slower
//!   than `bevy_ecs` here across three platforms, same order as
//!   `structural_churn` below, no root-cause pass done yet.
//! - `query_static_single_component`: dense iteration over one
//!   archetype-tracked component. **Real, profiled fix, not just a
//!   baseline**: `Archetypes::iter` used to be `Option::into_iter()
//!   .flat_map(|_| archetypes_with(...).flat_map(|_| entities.zip
//!   (column)))` — a chain of `flat_map` adaptors, each its own
//!   `Iterator` with its own `next()`. Found *because* fixing the
//!   two-component group below first (independently) left it running
//!   ~16x *faster* than this one on the same data — a two-component
//!   query beating a one-component query is a correctness-of-reasoning
//!   signal on its own. Replaced with `Iter1`, a hand-written flat
//!   state machine shaped like `bevy_ecs`'s own
//!   `QueryIterationCursor::next` (real source read directly,
//!   `Mid-D-Man/bevy`, `query/iter.rs`) — see `archetype.rs`'s `Iter1`
//!   doc comment for the full writeup. 145.20µs → 7.2131µs at
//!   N=10,000, zero `unsafe` added.
//! - `query2_static_two_components`: dense iteration over two
//!   archetype-tracked components on the same entities. **Two real,
//!   separate, sequential fixes**, both in `archetype.rs`'s `Iter2`
//!   doc comment: (1) `iter2` used to re-resolve its second column
//!   with a full `locations`+`archetypes`+column-lookup+downcast
//!   *per entity* — fixed to resolve once per archetype, ~22% real
//!   win. (2) that still left a `flat_map`/`filter`/`zip` combinator
//!   chain as the per-item cost — replaced with `Iter2`, same
//!   hand-written-state-machine treatment as `Iter1` above. 152.34µs
//!   → 9.1605µs at N=10,000, landing within noise of `bevy_ecs`'s own
//!   real CI number (9.3882µs) for the same workload. **This group's
//!   ratio against `query_static_single_component` is the direct,
//!   ongoing regression guard for both fixes** — see
//!   `raw_slice_ceiling` below for what that ratio should realistically
//!   be, and don't expect it to hit 1.0x.
//! - `structural_churn_insert_remove`: repeated single-component
//!   insert+remove on already-populated entities, forcing an archetype
//!   migration each time — same shape as `benches/ecs-vs-bevy-ecs`'s
//!   own `structural_churn` group, mirrored here so the tradeoff it
//!   tests (safe `Box<dyn Any>`-boxing move per migrated component)
//!   can be watched on every toolchain, not just the one that can also
//!   build `bevy_ecs`. **Not yet investigated** — same real CI gap
//!   (~2.9-4.3x) as `spawn_insert_bundle` above, no root-cause pass
//!   done yet; this is a genuinely different operation (structural
//!   migration, not dense iteration) and there's no evidence yet it
//!   shares a cause with the query-iteration gap that's now fixed.
//! - `raw_slice_ceiling`: zero ECS abstraction at all — two plain
//!   `Vec<T>`s, summed with a bare indexed loop, `one_field_sum` vs
//!   `two_field_sum`. Added to answer a real question the
//!   `query2_static`/`query_static` ratio raised once both were fixed:
//!   is the remaining gap between them a real inefficiency, or just
//!   the expected cost of reading one more field? **Real finding**: at
//!   N≥1,000 this ceiling's own two-field/one-field ratio is ~1.0x
//!   (reading a second `f32` from a second array costs, within noise,
//!   nothing extra here — memory-latency-bound, not compute-bound),
//!   while `query2_static`/`query_static`'s ratio is ~1.28-1.35x,
//!   stable across N (confirmed with `--sample-size 50` specifically
//!   to rule out noise). So roughly 25-30 points of that ratio *isn't*
//!   explained by "one more field" — a real, modest, still-open
//!   residual cost, candidate causes not yet checked (an extra slice
//!   reference competing for registers; the loop body's slightly
//!   larger tuple construction). Not urgent — `query2_static` is
//!   already within noise of `bevy_ecs`'s own absolute numbers — but
//!   not claimed as fully closed either.

use criterion::{
    black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion, Throughput,
};
use mid_ecs::World;

const SIZES: [usize; 4] = [100, 1_000, 10_000, 100_000];

#[derive(Clone, Copy)]
struct Position {
    x: f32,
    y: f32,
    z: f32,
}

#[derive(Clone, Copy)]
struct Velocity {
    dx: f32,
    dy: f32,
    dz: f32,
}

struct Marker;

fn populated_world(n: usize) -> World {
    let mut world = World::new();
    for _ in 0..n {
        let e = world.spawn();
        world.insert_bundle(
            e,
            (
                Position {
                    x: 1.0,
                    y: 2.0,
                    z: 3.0,
                },
                Velocity {
                    dx: 0.1,
                    dy: 0.2,
                    dz: 0.3,
                },
            ),
        );
    }
    world
}

fn bench_spawn_insert_bundle(c: &mut Criterion) {
    let mut group = c.benchmark_group("spawn_insert_bundle");
    for &n in &SIZES {
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::new("mid-ecs", n), &n, |b, &n| {
            b.iter(|| {
                let world = populated_world(n);
                black_box(world);
            });
        });
    }
    group.finish();
}

fn bench_query_static_single_component(c: &mut Criterion) {
    let mut group = c.benchmark_group("query_static_single_component");
    for &n in &SIZES {
        group.throughput(Throughput::Elements(n as u64));
        let world = populated_world(n);
        group.bench_with_input(BenchmarkId::new("mid-ecs", n), &n, |b, _| {
            b.iter(|| {
                let mut sum = 0.0f32;
                for (_, pos) in world.query_static::<Position>() {
                    sum += pos.x;
                }
                black_box(sum);
            });
        });
    }
    group.finish();
}

fn bench_query2_static_two_components(c: &mut Criterion) {
    let mut group = c.benchmark_group("query2_static_two_components");
    for &n in &SIZES {
        group.throughput(Throughput::Elements(n as u64));
        let world = populated_world(n);
        group.bench_with_input(BenchmarkId::new("mid-ecs", n), &n, |b, _| {
            b.iter(|| {
                let mut sum = 0.0f32;
                for (_, pos, vel) in world.query2_static::<Position, Velocity>() {
                    sum += pos.x + vel.dx;
                }
                black_box(sum);
            });
        });
    }
    group.finish();
}

// ── TEMPORARY, real-CI inlining-regression diagnostic ──────────────
// See crates/mid-ecs/src/diag_inline.rs's own doc comment for the full
// story: `query2_static_two_components` above runs ~4x slower than
// bevy_ecs on real CI (rustc 1.98.0) but within noise of it on this
// sandbox's rustc 1.91.1. `#[inline(always)]` on Iter2::next made
// things *worse* here, not better, so this compares three identical
// copies of the same logic — no attribute (matches the group above),
// #[inline(never)], #[inline(always)] — to see, on whichever toolchain
// actually shows the regression, which extreme (if either) the
// no-attribute default already resembles. Delete this function and
// its criterion_group! entries together with diag_inline.rs once the
// investigation concludes.
fn bench_query2_static_two_components_diag_inlining(c: &mut Criterion) {
    let mut group = c.benchmark_group("query2_static_two_components_diag_inlining");
    for &n in &SIZES {
        group.throughput(Throughput::Elements(n as u64));
        let world = populated_world(n);
        group.bench_with_input(BenchmarkId::new("inline_never", n), &n, |b, _| {
            b.iter(|| {
                let mut sum = 0.0f32;
                for (_, pos, vel) in world.query2_static_diag_never::<Position, Velocity>() {
                    sum += pos.x + vel.dx;
                }
                black_box(sum);
            });
        });
        group.bench_with_input(BenchmarkId::new("inline_always", n), &n, |b, _| {
            b.iter(|| {
                let mut sum = 0.0f32;
                for (_, pos, vel) in world.query2_static_diag_always::<Position, Velocity>() {
                    sum += pos.x + vel.dx;
                }
                black_box(sum);
            });
        });
    }
    group.finish();
}

fn bench_structural_churn(c: &mut Criterion) {
    let mut group = c.benchmark_group("structural_churn_insert_remove");
    for &n in &SIZES {
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::new("mid-ecs", n), &n, |b, &n| {
            b.iter_batched(
                || {
                    let mut world = World::new();
                    let entities: Vec<_> = (0..n)
                        .map(|_| {
                            let e = world.spawn();
                            world.insert_static(
                                e,
                                Position {
                                    x: 0.0,
                                    y: 0.0,
                                    z: 0.0,
                                },
                            );
                            e
                        })
                        .collect();
                    (world, entities)
                },
                |(mut world, entities)| {
                    for e in &entities {
                        world.insert_static(*e, Marker);
                        world.remove_static::<Marker>(*e);
                    }
                    black_box(world);
                },
                BatchSize::LargeInput,
            );
        });
    }
    group.finish();
}

fn bench_raw_slice_ceiling(c: &mut Criterion) {
    // Zero ECS abstraction at all -- two plain `Vec<Position>`/
    // `Vec<Velocity>`, summed with a bare indexed loop. Added to check
    // a real question the query2_static/query_static ratio raised
    // (see docs/benching-standards.md and this bench file's own doc
    // comment): once both queries resolve their columns once per
    // archetype instead of doing anything redundant, is the remaining
    // ~1.27-1.35x ratio between them a real inefficiency, or just the
    // structurally-expected cost of reading one more field and doing
    // one more float add per item? This group is that floor, with no
    // archetype/column/entity machinery in the way at all.
    let mut group = c.benchmark_group("raw_slice_ceiling");
    for &n in &SIZES {
        group.throughput(Throughput::Elements(n as u64));
        let positions: Vec<Position> = (0..n)
            .map(|_| Position {
                x: 1.0,
                y: 2.0,
                z: 3.0,
            })
            .collect();
        let velocities: Vec<Velocity> = (0..n)
            .map(|_| Velocity {
                dx: 0.1,
                dy: 0.2,
                dz: 0.3,
            })
            .collect();
        group.bench_with_input(BenchmarkId::new("one_field_sum", n), &n, |b, _| {
            b.iter(|| {
                let mut sum = 0.0f32;
                for p in &positions {
                    sum += p.x;
                }
                black_box(sum);
            });
        });
        group.bench_with_input(BenchmarkId::new("two_field_sum", n), &n, |b, _| {
            b.iter(|| {
                let mut sum = 0.0f32;
                let len = positions.len();
                for i in 0..len {
                    sum += positions[i].x + velocities[i].dx;
                }
                black_box(sum);
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_spawn_insert_bundle,
    bench_query_static_single_component,
    bench_query2_static_two_components,
    bench_query2_static_two_components_diag_inlining,
    bench_structural_churn,
    bench_raw_slice_ceiling
);
criterion_main!(benches);
