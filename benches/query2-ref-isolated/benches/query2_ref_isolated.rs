// NOTICE: See this crate's own Cargo.toml for why it exists and what
// question it's answering. TEMPORARY -- delete once answered, same as
// `benches/iter1-isolated`.
//
// Deliberately kept to exactly two groups, mirroring
// `benches/iter1-isolated/benches/iter1_isolated.rs`'s own two groups
// byte-for-byte in structure (same `populated_world` shape, same loop
// body style) -- `query2_static_ref` and `raw_slice_ceiling`'s
// `two_field_sum`, and nothing else lives in this crate.

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

// Same two-component archetype as archetype_core.rs's own
// populated_world and as iter1-isolated's own copy of it.
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

/// The real, unmodified `World::query2_static_ref` -- 2 columns, no
/// `Entity`, 16-byte item -- with nothing else from `archetype_core.rs`
/// or its diagnostic history sharing this bench binary.
fn bench_query2_static_ref(c: &mut Criterion) {
    let mut group = c.benchmark_group("query2_static_ref_two_components");
    for &n in &SIZES {
        group.throughput(Throughput::Elements(n as u64));
        let world = populated_world(n);
        group.bench_with_input(BenchmarkId::new("mid-ecs", n), &n, |b, _| {
            b.iter(|| {
                let mut sum = 0.0f32;
                for (pos, vel) in world.query2_static_ref::<Position, Velocity>() {
                    sum += pos.x + vel.dx;
                }
                black_box(sum);
            });
        });
    }
    group.finish();
}

/// Floor: zero ECS abstraction, both fields, same shape as
/// `archetype_core.rs`'s own `raw_slice_ceiling — two_field_sum`.
fn bench_raw_slice_ceiling(c: &mut Criterion) {
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
        group.bench_with_input(BenchmarkId::new("two_field_sum", n), &n, |b, _| {
            b.iter(|| {
                let mut sum = 0.0f32;
                for i in 0..positions.len() {
                    sum += positions[i].x + velocities[i].dx;
                }
                black_box(sum);
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_query2_static_ref, bench_raw_slice_ceiling);
criterion_main!(benches);
