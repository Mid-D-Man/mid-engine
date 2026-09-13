// NOTICE: See this crate's own Cargo.toml for why it exists and what
// question it's answering. TEMPORARY -- delete once answered, same as
// mid-ecs's own diag_*.rs modules.
//
// Deliberately kept to exactly two groups, mirroring
// crates/mid-ecs/benches/archetype_core.rs's `query_static_single_component`
// and `raw_slice_ceiling — one_field_sum` byte-for-byte (same
// populated_world shape, same loop body) -- nothing else lives in this
// crate. If `query_static` comes back close to the floor here, the
// ~4x gap seen in archetype_core.rs as of Archetype Core builds
// #15-#20 is specific to that file's own accumulated size/contents,
// not to `Iter1`/`query_static` itself. If it doesn't, the gap is real
// regardless of what else is compiled alongside it, and isolating it
// like this was never going to show a difference.

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
// populated_world -- Velocity is otherwise unused here, but keeping it
// means query_static::<Position> is resolving one column out of a
// real two-column archetype, exactly as it does in the file this is
// isolating the question from, not a simplified single-component
// world that would also change what's being measured.
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
        group.bench_with_input(BenchmarkId::new("one_field_sum", n), &n, |b, _| {
            b.iter(|| {
                let mut sum = 0.0f32;
                for p in &positions {
                    sum += p.x;
                }
                black_box(sum);
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_query_static_single_component,
    bench_raw_slice_ceiling
);
criterion_main!(benches);
