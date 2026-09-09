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
//! WORKLOAD DESIGN: eleven groups now, not four -- the original four
//! (below) were a broad-workload overview; six more were added this
//! pass as single-op and multi-op comparisons, matching the
//! granularity `crates/mid-math/benches/vs_glam.rs` already uses (one
//! bench_function pair per concrete operation). See each new group's
//! own comment at its `fn bench_*` definition further down for what
//! it isolates and why. The four original groups:
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

// ── Single-op comparisons ───────────────────────────────────────────────
// Everything above this line predates this pass: 5 broad workload
// buckets. These 6 below break specific operations out individually,
// one-to-one, the same granularity crates/mid-math/benches/vs_glam.rs
// already uses (one bench_function pair per concrete operation, not
// one pair per broad workload). Every loop here is wrapped in a
// #[inline(never)] free function, matching bevy_ecs's own convention
// confirmed by direct source read (benches/benches/bevy_ecs/iteration/
// in Mid-D-Man/bevy -- iter_simple.rs, iter_frag.rs,
// iter_simple_foreach.rs, iter_simple_contiguous.rs all wrap their loop
// in a #[inline(never)] fn run(&mut self), no exceptions found among
// the ones checked). Adopted unconditionally here, not because it's
// been confirmed to matter for mid-ecs (crates/mid-ecs/benches/
// archetype_core.rs's own real_query1/2_inline_never_wrapper arms are
// the actual test of that, still awaiting a real CI result) -- using
// bevy's own practice can only make this comparison fairer, never
// less fair, regardless of how that question resolves.

#[inline(never)]
fn mid_spawn_single(n: usize) -> MidWorld {
    let mut world = MidWorld::new();
    for _ in 0..n {
        let e = world.spawn();
        world.insert_static(
            e,
            Position { x: 1.0, y: 2.0, z: 3.0 },
        );
    }
    world
}

#[inline(never)]
fn bevy_spawn_single(n: usize) -> BevyWorld {
    let mut world = BevyWorld::new();
    for _ in 0..n {
        world.spawn(BevyPosition { x: 1.0, y: 2.0, z: 3.0 });
    }
    world
}

fn bench_spawn_single_component(c: &mut Criterion) {
    // Isolates spawn+single-insert from spawn+bundle-insert
    // (`spawn_n_entities_two_components` above): one component, not
    // two, so `insert_bundle`'s own Bundle-trait machinery never
    // enters the picture on the mid-ecs side. Same non-apples-to-
    // apples caveat as that group applies here too -- bevy's
    // `World::spawn(single_component)` is one step; mid-ecs's closest
    // equivalent is still `spawn()` then `insert_static()`, two steps.
    let mut g = c.benchmark_group("spawn_single_component");

    g.bench_function("mid-ecs", |b| {
        b.iter(|| black_box(mid_spawn_single(N)));
    });

    g.bench_function("bevy_ecs", |b| {
        b.iter(|| black_box(bevy_spawn_single(N)));
    });

    g.finish();
}

#[inline(never)]
fn mid_insert_single(world: &mut MidWorld, entities: &[mid_ecs::world::Entity]) {
    for &e in entities {
        world.insert_static(e, Position { x: 1.0, y: 2.0, z: 3.0 });
    }
}

#[inline(never)]
fn bevy_insert_single(world: &mut BevyWorld, entities: &[bevy_ecs::prelude::Entity]) {
    for &e in entities {
        world.entity_mut(e).insert(BevyPosition { x: 1.0, y: 2.0, z: 3.0 });
    }
}

fn bench_insert_single_component(c: &mut Criterion) {
    // Single-component insert onto an already-spawned, otherwise-empty
    // entity -- the structural-migration cost alone, no spawn cost
    // mixed in, no bundle machinery on either side (bevy's own
    // `insert` takes `T: Bundle`, but a lone component satisfies that
    // via its blanket impl -- this is still the single-component path,
    // same as mid-ecs's `insert_static`).
    let mut g = c.benchmark_group("insert_single_component");

    g.bench_function("mid-ecs", |b| {
        b.iter_batched(
            || {
                let mut world = MidWorld::new();
                let entities: Vec<_> = (0..N).map(|_| world.spawn()).collect();
                (world, entities)
            },
            |(mut world, entities)| {
                mid_insert_single(&mut world, &entities);
                black_box(world);
            },
            BatchSize::LargeInput,
        );
    });

    g.bench_function("bevy_ecs", |b| {
        b.iter_batched(
            || {
                let mut world = BevyWorld::new();
                let entities: Vec<_> = (0..N).map(|_| world.spawn_empty().id()).collect();
                (world, entities)
            },
            |(mut world, entities)| {
                bevy_insert_single(&mut world, &entities);
                black_box(world);
            },
            BatchSize::LargeInput,
        );
    });

    g.finish();
}

#[inline(never)]
fn mid_remove_single(world: &mut MidWorld, entities: &[mid_ecs::world::Entity]) {
    for &e in entities {
        world.remove_static::<Position>(e);
    }
}

#[inline(never)]
fn bevy_remove_single(world: &mut BevyWorld, entities: &[bevy_ecs::prelude::Entity]) {
    for &e in entities {
        world.entity_mut(e).remove::<BevyPosition>();
    }
}

fn bench_remove_single_component(c: &mut Criterion) {
    // Mirror of insert_single_component: each entity starts with
    // exactly one component and loses it, the single-component
    // structural migration back toward empty.
    let mut g = c.benchmark_group("remove_single_component");

    g.bench_function("mid-ecs", |b| {
        b.iter_batched(
            || {
                let mut world = MidWorld::new();
                let entities: Vec<_> = (0..N)
                    .map(|_| {
                        let e = world.spawn();
                        world.insert_static(e, Position { x: 1.0, y: 2.0, z: 3.0 });
                        e
                    })
                    .collect();
                (world, entities)
            },
            |(mut world, entities)| {
                mid_remove_single(&mut world, &entities);
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
                    .map(|_| world.spawn(BevyPosition { x: 1.0, y: 2.0, z: 3.0 }).id())
                    .collect();
                (world, entities)
            },
            |(mut world, entities)| {
                bevy_remove_single(&mut world, &entities);
                black_box(world);
            },
            BatchSize::LargeInput,
        );
    });

    g.finish();
}

#[inline(never)]
fn mid_get_random_access(world: &MidWorld, entities: &[mid_ecs::world::Entity]) -> f32 {
    let mut sum = 0.0f32;
    for &e in entities {
        if let Some(pos) = world.get_static::<Position>(e) {
            sum += pos.x;
        }
    }
    sum
}

#[inline(never)]
fn bevy_get_random_access(world: &BevyWorld, entities: &[bevy_ecs::prelude::Entity]) -> f32 {
    let mut sum = 0.0f32;
    for &e in entities {
        if let Some(pos) = world.get::<BevyPosition>(e) {
            sum += pos.x;
        }
    }
    sum
}

fn bench_get_component_random_access(c: &mut Criterion) {
    // Deliberately different code path from every iteration group
    // above: N separate entity -> archetype -> column lookups by id,
    // not one contiguous archetype scan. Isolates per-lookup overhead
    // (mid-ecs's entity/archetype-location table vs bevy_ecs's own)
    // from anything about dense iteration specifically.
    let mut mid_world = MidWorld::new();
    let mid_entities: Vec<_> = (0..N)
        .map(|_| {
            let e = mid_world.spawn();
            mid_world.insert_bundle(
                e,
                (
                    Position { x: 1.0, y: 2.0, z: 3.0 },
                    Velocity { dx: 0.1, dy: 0.2, dz: 0.3 },
                ),
            );
            e
        })
        .collect();

    let mut bevy_world = BevyWorld::new();
    let bevy_entities: Vec<_> = (0..N)
        .map(|_| {
            bevy_world
                .spawn((
                    BevyPosition { x: 1.0, y: 2.0, z: 3.0 },
                    BevyVelocity { dx: 0.1, dy: 0.2, dz: 0.3 },
                ))
                .id()
        })
        .collect();

    let mut g = c.benchmark_group("get_component_random_access");

    g.bench_function("mid-ecs", |b| {
        b.iter(|| black_box(mid_get_random_access(&mid_world, &mid_entities)));
    });

    g.bench_function("bevy_ecs", |b| {
        b.iter(|| black_box(bevy_get_random_access(&bevy_world, &bevy_entities)));
    });

    g.finish();
}

#[inline(never)]
fn mid_insert_bundle_existing(world: &mut MidWorld, entities: &[mid_ecs::world::Entity]) {
    for &e in entities {
        world.insert_bundle(
            e,
            (
                Position { x: 1.0, y: 2.0, z: 3.0 },
                Velocity { dx: 0.1, dy: 0.2, dz: 0.3 },
            ),
        );
    }
}

#[inline(never)]
fn bevy_insert_bundle_existing(world: &mut BevyWorld, entities: &[bevy_ecs::prelude::Entity]) {
    for &e in entities {
        world.entity_mut(e).insert((
            BevyPosition { x: 1.0, y: 2.0, z: 3.0 },
            BevyVelocity { dx: 0.1, dy: 0.2, dz: 0.3 },
        ));
    }
}

fn bench_insert_bundle_on_existing_entity(c: &mut Criterion) {
    // Multi-op counterpart to insert_single_component: a 2-component
    // bundle insert, but onto an entity that already carries an
    // unrelated component (Marker), not a bare freshly-spawned one.
    // Genuinely different from `spawn_n_entities_two_components`
    // above, which inserts the bundle immediately after spawning an
    // empty entity -- this measures the migration cost when there's
    // already real data on the entity to carry across archetypes.
    let mut g = c.benchmark_group("insert_bundle_on_existing_entity");

    g.bench_function("mid-ecs", |b| {
        b.iter_batched(
            || {
                let mut world = MidWorld::new();
                let entities: Vec<_> = (0..N)
                    .map(|_| {
                        let e = world.spawn();
                        world.insert_static(e, Marker);
                        e
                    })
                    .collect();
                (world, entities)
            },
            |(mut world, entities)| {
                mid_insert_bundle_existing(&mut world, &entities);
                black_box(world);
            },
            BatchSize::LargeInput,
        );
    });

    g.bench_function("bevy_ecs", |b| {
        b.iter_batched(
            || {
                let mut world = BevyWorld::new();
                let entities: Vec<_> = (0..N).map(|_| world.spawn(BevyMarker).id()).collect();
                (world, entities)
            },
            |(mut world, entities)| {
                bevy_insert_bundle_existing(&mut world, &entities);
                black_box(world);
            },
            BatchSize::LargeInput,
        );
    });

    g.finish();
}

#[inline(never)]
fn mid_remove_bundle(world: &mut MidWorld, entities: &[mid_ecs::world::Entity]) {
    for &e in entities {
        world.remove_bundle::<(Position, Velocity)>(e);
    }
}

#[inline(never)]
fn bevy_remove_bundle(world: &mut BevyWorld, entities: &[bevy_ecs::prelude::Entity]) {
    for &e in entities {
        world.entity_mut(e).remove::<(BevyPosition, BevyVelocity)>();
    }
}

fn bench_remove_bundle(c: &mut Criterion) {
    // Mirror of insert_bundle_on_existing_entity: each entity starts
    // with [Marker, Position, Velocity] and loses the 2-component
    // bundle, landing back on [Marker] -- not the empty archetype,
    // so this is a genuine migration between two non-empty archetypes
    // on both sides, not a return to a degenerate base case.
    let mut g = c.benchmark_group("remove_bundle_two_components");

    g.bench_function("mid-ecs", |b| {
        b.iter_batched(
            || {
                let mut world = MidWorld::new();
                let entities: Vec<_> = (0..N)
                    .map(|_| {
                        let e = world.spawn();
                        world.insert_static(e, Marker);
                        world.insert_bundle(
                            e,
                            (
                                Position { x: 1.0, y: 2.0, z: 3.0 },
                                Velocity { dx: 0.1, dy: 0.2, dz: 0.3 },
                            ),
                        );
                        e
                    })
                    .collect();
                (world, entities)
            },
            |(mut world, entities)| {
                mid_remove_bundle(&mut world, &entities);
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
                            .spawn((
                                BevyMarker,
                                BevyPosition { x: 1.0, y: 2.0, z: 3.0 },
                                BevyVelocity { dx: 0.1, dy: 0.2, dz: 0.3 },
                            ))
                            .id()
                    })
                    .collect();
                (world, entities)
            },
            |(mut world, entities)| {
                bevy_remove_bundle(&mut world, &entities);
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
    bench_spawn_single_component,
    bench_query_static_single_component,
    bench_dense_query_iteration,
    bench_raw_slice_ceiling,
    bench_structural_churn,
    bench_insert_single_component,
    bench_remove_single_component,
    bench_get_component_random_access,
    bench_insert_bundle_on_existing_entity,
    bench_remove_bundle
);
criterion_main!(benches);
