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
//! WORKLOAD DESIGN: four groups, each a real, meaningfully different
//! stress on a storage engine, not just "spawn a lot of entities" four
//! times over:
//!
//! - `spawn`: raw entity + two-component creation throughput. Note this
//!   isn't perfectly apples-to-apples -- bevy's `World::spawn(bundle)`
//!   places an entity directly into its final archetype in one step;
//!   mid-ecs's closest equivalent is `World::spawn()` (into the empty
//!   archetype) then `World::insert_bundle(e, bundle)` (one migration
//!   into the final archetype) -- a real, honest architectural
//!   difference this benchmark exists to actually measure, not hide.
//! - `query_static_single_component`: one-component dense iteration,
//!   added after `dense_query_iteration`'s own fix (below) turned up a
//!   real, separate bug in the single-column path -- see
//!   `crates/mid-ecs/src/archetype.rs`'s `Iter1` doc comment. This
//!   group is the same-machine, same-run confirmation that fix holds
//!   up against `bevy_ecs` directly, not just against mid-ecs's own
//!   prior numbers in `crates/mid-ecs/benches/archetype_core.rs`.
//! - `dense_query_iteration`: the hottest, most common real operation
//!   (matches `GlobalTransform`'s own "hottest, most-iterated component"
//!   framing in docs/mid-ecs.md) -- iterate every entity's two
//!   components and touch both. **Real history worth keeping**: this
//!   group measured mid-ecs at ~18-21x slower than `bevy_ecs` across
//!   three real CI platforms (ubuntu/macos/arm) before a two-part fix
//!   (`crates/mid-ecs/src/archetype.rs`'s `Iter1`/`Iter2` doc comments
//!   have the full writeup) replaced a `flat_map`/`filter`/`zip`
//!   combinator-adaptor chain with a hand-written state machine shaped
//!   like `bevy_ecs`'s own `QueryIterationCursor::next` (real source
//!   read directly, `Mid-D-Man/bevy`, `query/iter.rs`). Zero `unsafe`
//!   added -- the combinator-chain overhead itself was the dominant
//!   cost, not bounds-checking. Same-crate internal bench (N=10,000):
//!   152.34µs -> 9.1605µs, landing within noise of `bevy_ecs`'s own
//!   9.3882µs from the CI run that first surfaced this gap. This group
//!   should confirm that holds on real CI, not just in the sandbox that
//!   found and fixed it.
//! - `raw_slice_ceiling`: no ECS abstraction at all -- two plain
//!   `Vec<Position>`/`Vec<Velocity>`, iterated with a bare `for i in
//!   0..len` loop, for both "engines" (bevy_ecs's own storage isn't
//!   involved either -- this is the same workload, not bevy-specific).
//!   Added alongside the `dense_query_iteration` fix specifically to
//!   answer the question that fix's own numbers raise: once mid-ecs and
//!   `bevy_ecs` are within noise of each other, are they *both* still
//!   paying some shared, engine-agnostic floor (allocation pattern,
//!   memory layout, whatever this specific runner's cache/branch
//!   predictor does with this exact loop shape), or has either one
//!   actually reached the real ceiling? A real interpretability anchor
//!   for every group above, not a claim about either engine on its own.
//! - `structural_churn`: repeated single-component insert+remove on
//!   already-populated entities, forcing an archetype migration each
//!   time. Directly tests the tradeoff docs/mid-ecs.md's own top-level
//!   doc comment calls out: mid-ecs's safe `Box<dyn Any>`-boxing move
//!   per migrated component vs. bevy's unsafe raw-pointer table move --
//!   a "no profiled need here" claim this benchmark can actually check.
//!   **Not yet investigated**: this group's ~2.9-4.3x gap (real CI,
//!   three platforms) is a separate operation from the two fixed above
//!   (structural migration, not dense iteration) and has had no
//!   equivalent root-cause pass yet -- next real target, not assumed to
//!   have the same cause as the query-iteration gap did.

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

fn bench_query_static_single_component(c: &mut Criterion) {
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
    let mut bevy_query = bevy_world.query::<&BevyPosition>();

    let mut g = c.benchmark_group("query_static_single_component");

    g.bench_function("mid-ecs", |b| {
        b.iter(|| {
            let mut sum = 0.0f32;
            for (_, pos) in mid_world.query_static::<Position>() {
                sum += pos.x;
            }
            black_box(sum);
        });
    });

    g.bench_function("bevy_ecs", |b| {
        b.iter(|| {
            let mut sum = 0.0f32;
            for pos in bevy_query.iter(&bevy_world) {
                sum += pos.x;
            }
            black_box(sum);
        });
    });

    g.finish();
}

fn bench_raw_slice_ceiling(c: &mut Criterion) {
    // No `World`, no `Entity`, no archetype/table lookup at all on
    // either side -- see this file's own header doc comment for why
    // this group exists. Same two arrays feed both "engines" below;
    // there's nothing engine-specific left to differ on.
    let positions: Vec<Position> = (0..N)
        .map(|_| Position {
            x: 1.0,
            y: 2.0,
            z: 3.0,
        })
        .collect();
    let velocities: Vec<Velocity> = (0..N)
        .map(|_| Velocity {
            dx: 0.1,
            dy: 0.2,
            dz: 0.3,
        })
        .collect();

    let mut g = c.benchmark_group("raw_slice_ceiling");

    g.bench_function("mid-ecs", |b| {
        b.iter(|| {
            let mut sum = 0.0f32;
            let len = positions.len();
            for i in 0..len {
                sum += positions[i].x + velocities[i].dx;
            }
            black_box(sum);
        });
    });

    g.bench_function("bevy_ecs", |b| {
        b.iter(|| {
            let mut sum = 0.0f32;
            for (pos, vel) in positions.iter().zip(velocities.iter()) {
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
    bench_query_static_single_component,
    bench_dense_query_iteration,
    bench_raw_slice_ceiling,
    bench_structural_churn
);
criterion_main!(benches);
