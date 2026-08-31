//! Criterion benchmarks for the Sparse Shell's own operations —
//! `World::insert`/`get`/`remove`/`query`/`query2` (`component.rs`'s
//! `SparseShell`), as distinct from `archetype_core.rs`'s Archetype
//! Core (`_static` suffix) benches. First real bench coverage this
//! storage system has ever had — see `component.rs`'s own doc comments
//! on `SparseShell::iter` and `World::query2` for the two places that
//! explicitly say "revisit if this ever shows up in a real profile —
//! it hasn't been built for one." This is that profile.
//!
//! Run: `cargo bench -p mid-ecs --bench sparse_shell`
//!
//! Sizes: same `[100, 1_000, 10_000, 100_000]` sweep as
//! `archetype_core.rs` and `mid-collections/benches/sparse_set.rs`.
//!
//! Not included here: a `raw_slice_ceiling`-style floor group.
//! `mid_collections::SparseSet` — what `SparseShell` is built directly
//! on top of, one per component type — already has its own dedicated
//! bench (`crates/mid-collections/benches/sparse_set.rs`, `SparseSet`
//! vs `HashMap`) that serves the same purpose; no need to duplicate it
//! here.
//!
//! GROUPS:
//!
//! - `insert_two_components`: `World::spawn()` + two `World::insert`
//!   calls per entity. Sparse Shell's own equivalent of
//!   `archetype_core.rs`'s `spawn_insert_bundle` — but a structurally
//!   different operation, not just the same one on different storage:
//!   there's no archetype migration here at all, each `insert` is
//!   independent (outer `SparseSet<ComponentId, _>` lookup-or-create,
//!   then inner `SparseSet<Entity, T>::insert`), so this doesn't pay
//!   `archetype_core.rs`'s `Box<dyn Any>`-per-migrated-component cost.
//! - `query_single_component`: dense iteration over one component via
//!   `World::query::<T>()`. **What this actually measures**:
//!   `SparseShell::iter`'s own documented, deliberate choice to return
//!   `Box<dyn Iterator<...>>` rather than a concrete type — one vtable
//!   dispatch per `.next()` call. Real, open question this bench
//!   exists to answer: does that cost show up at real scale the way
//!   the Archetype Core's old combinator-chain cost did (`Iter1`/
//!   `Iter2` in `archetype.rs`), or is a single boxed hop cheap enough
//!   in practice that de-boxing wouldn't be worth the code it costs?
//!   Not assumed either way going in.
//! - `query2_two_components`: dense iteration over two components via
//!   `World::query2::<A, B>()`. Same *shape* of v1 simplification as
//!   the Archetype Core's old `iter2` (drives off `A`, looks up `B`
//!   per entity) — but not necessarily the same *magnitude*:
//!   `SparseShell::get`'s per-entity lookup is a direct one-hop
//!   `SparseSet<Entity, T>::get`, not the location+archetype+column
//!   chain the old Archetype Core version paid. The ratio against
//!   `query_single_component` at the same N is the real signal here,
//!   same regression-guard idea as `archetype_core.rs`'s own
//!   two-component overhead table.
//! - `remove_insert_churn`: repeated single-component insert+remove on
//!   already-populated entities. Sparse Shell's own equivalent of
//!   `archetype_core.rs`'s `structural_churn_insert_remove` — again a
//!   structurally different operation on this storage, not the same
//!   cost measured twice: no migration, no `Box<dyn Any>` boxing, just
//!   `SparseSet<Entity, T>::insert`/`remove` directly. Real question:
//!   how much of the Archetype Core's own churn gap against `bevy_ecs`
//!   (`benches/ecs-vs-bevy-ecs`, ~2.9-5x, not yet investigated) is
//!   inherent to *any* insert/remove, versus specific to the Archetype
//!   Core's migration machinery? This group is the mid-ecs-internal
//!   half of answering that; `benches/ecs-vs-bevy-ecs`'s own
//!   `sparse_shell` group (bevy's `#[component(storage = "SparseSet")]`
//!   opt-in) is the cross-engine half.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
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
        world.insert(
            e,
            Position {
                x: 1.0,
                y: 2.0,
                z: 3.0,
            },
        );
        world.insert(
            e,
            Velocity {
                dx: 0.1,
                dy: 0.2,
                dz: 0.3,
            },
        );
    }
    world
}

fn bench_insert_two_components(c: &mut Criterion) {
    let mut group = c.benchmark_group("insert_two_components");
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

fn bench_query_single_component(c: &mut Criterion) {
    let mut group = c.benchmark_group("query_single_component");
    for &n in &SIZES {
        group.throughput(Throughput::Elements(n as u64));
        let world = populated_world(n);
        group.bench_with_input(BenchmarkId::new("mid-ecs", n), &n, |b, _| {
            b.iter(|| {
                let mut sum = 0.0f32;
                for (_, pos) in world.query::<Position>() {
                    sum += pos.x;
                }
                black_box(sum);
            });
        });
    }
    group.finish();
}

fn bench_query2_two_components(c: &mut Criterion) {
    let mut group = c.benchmark_group("query2_two_components");
    for &n in &SIZES {
        group.throughput(Throughput::Elements(n as u64));
        let world = populated_world(n);
        group.bench_with_input(BenchmarkId::new("mid-ecs", n), &n, |b, _| {
            b.iter(|| {
                let mut sum = 0.0f32;
                for (_, pos, vel) in world.query2::<Position, Velocity>() {
                    sum += pos.x + vel.dx;
                }
                black_box(sum);
            });
        });
    }
    group.finish();
}

fn bench_remove_insert_churn(c: &mut Criterion) {
    let mut group = c.benchmark_group("remove_insert_churn");
    for &n in &SIZES {
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::new("mid-ecs", n), &n, |b, &n| {
            b.iter_batched(
                || {
                    let mut world = World::new();
                    let entities: Vec<_> = (0..n)
                        .map(|_| {
                            let e = world.spawn();
                            world.insert(
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
                        world.insert(*e, Marker);
                        world.remove::<Marker>(*e);
                    }
                    black_box(world);
                },
                criterion::BatchSize::LargeInput,
            );
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_insert_two_components,
    bench_query_single_component,
    bench_query2_two_components,
    bench_remove_insert_churn
);
criterion_main!(benches);
