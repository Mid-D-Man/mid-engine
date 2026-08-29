//! Direct comparison of mid-ecs's storage engine against bevy_ecs's, on
//! equivalent real workloads.
//!
//! Run: `cargo bench -p ecs-vs-bevy-ecs --bench vs_bevy_ecs`
//! HTML report: `target/criterion/report/index.html`
//!
//! NOTE ON VERIFICATION: bevy_ecs 0.19.1 declares `rust-version =
//! "1.95.0"` (see this crate's own Cargo.toml). Real CI's
//! `dtolnay/rust-toolchain@stable` clears that; the sandbox this was
//! *written* in (apt's rustc-1.91) does not, so the mid-ecs half below
//! was compiled, run, and its numbers are real -- but the bevy_ecs half
//! could only be grounded by reading bevy_ecs's real source directly
//! (`world/mod.rs`, `world/entity_access/world_mut.rs`, `query/state.rs`
//! in `Mid-D-Man/bevy`) and was NOT locally compiled. Confirm this
//! actually builds on the next CI run before trusting bevy_ecs's numbers
//! specifically.
//!
//! WORKLOAD DESIGN: three groups, each a real, meaningfully different
//! stress on a storage engine, not just "spawn a lot of entities" three
//! times over:
//!
//! - `spawn`: raw entity + two-component creation throughput. Note this
//!   isn't perfectly apples-to-apples -- bevy's `World::spawn(bundle)`
//!   places an entity directly into its final archetype in one step;
//!   mid-ecs's closest equivalent is `World::spawn()` (into the empty
//!   archetype) then `World::insert_bundle(e, bundle)` (one migration
//!   into the final archetype) -- a real, honest architectural
//!   difference this benchmark exists to actually measure, not hide.
//! - `dense_query_iteration`: the hottest, most common real operation
//!   (matches `GlobalTransform`'s own "hottest, most-iterated component"
//!   framing in docs/mid-ecs.md) -- iterate every entity's two
//!   components and touch both.
//! - `structural_churn`: repeated single-component insert+remove on
//!   already-populated entities, forcing an archetype migration each
//!   time. Directly tests the tradeoff docs/mid-ecs.md's own top-level
//!   doc comment calls out: mid-ecs's safe `Box<dyn Any>`-boxing move
//!   per migrated component vs. bevy's unsafe raw-pointer table move --
//!   a "no profiled need here" claim this benchmark can actually check.

use bevy_ecs::prelude::{Component, World as BevyWorld};
use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion};
use mid_ecs::World as MidWorld;

const N: usize = 10_000;

// ── mid-ecs side ────────────────────────────────────────────────────────

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

// ── bevy_ecs side (separate types -- bevy's `Component` derive adds
// storage/registration machinery mid-ecs's plain structs don't carry,
// so these can't be the same types as above even though the shape is
// identical) ─────────────────────────────────────────────────────────

#[derive(Component, Clone, Copy)]
struct BevyPosition {
    x: f32,
    y: f32,
    z: f32,
}

#[derive(Component, Clone, Copy)]
struct BevyVelocity {
    dx: f32,
    dy: f32,
    dz: f32,
}

#[derive(Component)]
struct BevyMarker;

fn bench_spawn(c: &mut Criterion) {
    let mut g = c.benchmark_group("spawn_n_entities_two_components");

    g.bench_function("mid-ecs", |b| {
        b.iter_batched(
            MidWorld::new,
            |mut world| {
                for _ in 0..N {
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
                black_box(world);
            },
            BatchSize::LargeInput,
        );
    });

    g.bench_function("bevy_ecs", |b| {
        b.iter_batched(
            BevyWorld::new,
            |mut world| {
                for _ in 0..N {
                    world.spawn((
                        BevyPosition {
                            x: 1.0,
                            y: 2.0,
                            z: 3.0,
                        },
                        BevyVelocity {
                            dx: 0.1,
                            dy: 0.2,
                            dz: 0.3,
                        },
                    ));
                }
                black_box(world);
            },
            BatchSize::LargeInput,
        );
    });

    g.finish();
}

fn bench_dense_query_iteration(c: &mut Criterion) {
    let mut mid_world = MidWorld::new();
    for _ in 0..N {
        let e = mid_world.spawn();
        mid_world.insert_bundle(
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

    let mut bevy_world = BevyWorld::new();
    bevy_world.spawn_batch((0..N).map(|_| {
        (
            BevyPosition {
                x: 1.0,
                y: 2.0,
                z: 3.0,
            },
            BevyVelocity {
                dx: 0.1,
                dy: 0.2,
                dz: 0.3,
            },
        )
    }));
    let mut bevy_query = bevy_world.query::<(&BevyPosition, &BevyVelocity)>();

    let mut g = c.benchmark_group("dense_query_iteration");

    g.bench_function("mid-ecs", |b| {
        b.iter(|| {
            let mut sum = 0.0f32;
            for (_, pos, vel) in mid_world.query2_static::<Position, Velocity>() {
                sum += pos.x + vel.dx;
            }
            black_box(sum);
        });
    });

    g.bench_function("bevy_ecs", |b| {
        b.iter(|| {
            let mut sum = 0.0f32;
            for (pos, vel) in bevy_query.iter(&bevy_world) {
                sum += pos.x + vel.dx;
            }
            black_box(sum);
        });
    });

    g.finish();
}

fn bench_structural_churn(c: &mut Criterion) {
    let mut g = c.benchmark_group("structural_churn_insert_remove");

    g.bench_function("mid-ecs", |b| {
        b.iter_batched(
            || {
                let mut world = MidWorld::new();
                let entities: Vec<_> = (0..N)
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

    g.bench_function("bevy_ecs", |b| {
        b.iter_batched(
            || {
                let mut world = BevyWorld::new();
                let entities: Vec<_> = (0..N)
                    .map(|_| {
                        world
                            .spawn(BevyPosition {
                                x: 0.0,
                                y: 0.0,
                                z: 0.0,
                            })
                            .id()
                    })
                    .collect();
                (world, entities)
            },
            |(mut world, entities)| {
                for e in &entities {
                    world.entity_mut(*e).insert(BevyMarker);
                    world.entity_mut(*e).remove::<BevyMarker>();
                }
                black_box(world);
            },
            BatchSize::LargeInput,
        );
    });

    g.finish();
}

criterion_group!(
    benches,
    bench_spawn,
    bench_dense_query_iteration,
    bench_structural_churn
);
criterion_main!(benches);
