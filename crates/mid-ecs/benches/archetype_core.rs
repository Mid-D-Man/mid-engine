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
//! - `query_static_single_component`: dense iteration over one
//!   archetype-tracked component. The baseline the two-component group
//!   below is read against.
//! - `query2_static_two_components`: dense iteration over two
//!   archetype-tracked components on the same entities. **This is the
//!   direct regression guard for a real, profiled fix** — `Archetypes::
//!   iter2` used to resolve its second component with a full
//!   `locations` + `archetypes` + column-`SparseSet` + `dyn Column`
//!   downcast lookup *per entity*, even when every entity being
//!   iterated shares the exact same archetype and therefore the exact
//!   same answer. Isolated via a standalone timing probe (`mid-ecs`
//!   only, N=10,000, single archetype): the two-component query cost
//!   6.5x what the single-column query cost over the identical
//!   entities before the fix. Fixed to resolve both columns once per
//!   matching archetype instead (see `archetype.rs`'s `iter2` doc
//!   comment for the full writeup). If this group's ratio against
//!   `query_static_single_component` ever creeps back up toward that
//!   old ~6.5x figure, that's this exact regression coming back.
//! - `structural_churn_insert_remove`: repeated single-component
//!   insert+remove on already-populated entities, forcing an archetype
//!   migration each time — same shape as `benches/ecs-vs-bevy-ecs`'s
//!   own `structural_churn` group, mirrored here so the tradeoff it
//!   tests (safe `Box<dyn Any>`-boxing move per migrated component)
//!   can be watched on every toolchain, not just the one that can also
//!   build `bevy_ecs`.

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

criterion_group!(
    benches,
    bench_spawn_insert_bundle,
    bench_query_static_single_component,
    bench_query2_static_two_components,
    bench_structural_churn
);
criterion_main!(benches);
