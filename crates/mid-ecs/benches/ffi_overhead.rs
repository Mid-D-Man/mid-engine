// ============================================================================
// NOTICE: Full documentation, design decisions, and results for this file
// live in docs/mid-ecs.md, section "FFI test and bench"
// ============================================================================
//! Criterion benchmarks for the `extern "C"` surface in `ffi.rs`, measured
//! against the equivalent Rust-side call on the *same* world, in the same
//! process. `mid-ecs` only, no `bevy_ecs` dependency, so it runs on any
//! toolchain this workspace supports.
//!
//! What this measures, and what it does not: each `ffi` variant calls the
//! `extern "C"` function directly from Rust. That includes everything the
//! function itself does (null checks, `catch_unwind`, status-code
//! plumbing, the count-then-fill double call) and excludes the ABI
//! crossing from a separately compiled C program, which is a real but
//! separate cost, left for a C-timed pass. Inside one binary the optimizer
//! can still see through the boundary in places a real C caller never
//! allows, so read the ratios as a lower bound on FFI overhead.
//!
//! Run: `cargo bench -p mid-ecs --bench ffi_overhead`
//!
//! GROUPS (ids are `group/variant/param` where the group sweeps a size, or
//! `group/variant` where it doesn't; `scripts/bench_mid_ecs_ffi.py` reads
//! both):
//!
//! - `lifecycle_spawn_despawn`: spawn N entities then despawn all N, then
//!   free the world. `rust` vs `ffi`, swept over N. Pure per-call cost
//!   times N.
//! - `span_read`: read one archetype's component column and sum it, swept
//!   over N. `typed_query` (the idiomatic Rust read) is the floor;
//!   `rust_span`/`ffi_span` add the raw-span step both languages share;
//!   `*_call_only` stops after getting the span, so its cost is the call
//!   itself and should not grow with N.
//! - `archetypes_matching`: enumerate the archetypes holding a component,
//!   swept over K, the number of archetypes (fragmentation), not entities.
//!   `rust_collect`, `ffi_count_then_fill` (the documented two-call idiom)
//!   and `ffi_count_only`.
//! - `change_rows`: sum the values changed since a tracker last ran, with
//!   1 row in 10 changed, swept over N. `rust_query_changed` against
//!   `ffi_changed_rows` (enumerate archetypes, take each one's row list,
//!   read the span at those rows).
//! - `resource_access`: single-call reads and writes of one resource,
//!   `rust_*` vs `ffi_*`. Not swept.
//!
//! Every group asserts, once at setup, that its variants compute the same
//! answer, so a variant that quietly measures the wrong thing fails
//! instead of producing a number.

use criterion::{
    black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion, Throughput,
};
use mid_collections::FfiSpan;
use mid_ecs::ffi::*;
use mid_ecs::{ArchetypeId, ChangeTracker, ComponentId, ResourceId, World};

const SIZES: [usize; 4] = [100, 1_000, 10_000, 100_000];
const FRAGMENTATION: [usize; 3] = [1, 4, 16];

type Health = MidEcsTestHealthStatic;

#[derive(Clone, Copy)]
struct Tag<const I: usize>;

/// An FFI handle plus the ids every group needs, freed on drop.
struct Fixture {
    handle: *mut MidEcsWorld,
    health_id: u32,
}

impl Fixture {
    fn new(world: World) -> Self {
        let health_id = world
            .lookup_ffi_static_component_id("FfiHealthStatic")
            .expect("registered by the builder")
            .as_u32();
        Self {
            handle: MidEcsWorld::from_world(world),
            health_id,
        }
    }

    fn world(&self) -> &World {
        // SAFETY: `handle` is live until `drop`, and no `&mut` alias exists
        // while a benchmark body runs.
        unsafe { &*self.handle }.world()
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        // SAFETY: freed exactly once, here.
        unsafe { mid_ecs_world_free(self.handle) };
    }
}

fn health_world(n: usize) -> World {
    let mut world = World::new();
    world.register_ffi_static_component::<Health>("FfiHealthStatic");
    for hp in 0..n as u32 {
        let e = world.spawn();
        world.insert_static(e, Health { hp });
    }
    world
}

/// The one archetype holding `Health` in a `health_world`.
fn populated_archetype(world: &World, health: ComponentId) -> ArchetypeId {
    let mut found = world.archetypes_with_static_component(health);
    let only = found.next().expect("one archetype holds Health");
    assert!(
        found.next().is_none(),
        "health_world has a single archetype"
    );
    only
}

/// # Safety
/// `span` must describe `count` valid `Health` elements or be empty.
unsafe fn sum_span(span: &FfiSpan) -> u64 {
    if span.count == 0 {
        return 0;
    }
    std::slice::from_raw_parts(span.ptr as *const Health, span.count)
        .iter()
        .map(|h| h.hp as u64)
        .sum()
}

fn empty_span() -> FfiSpan {
    FfiSpan::empty()
}

fn bench_lifecycle(c: &mut Criterion) {
    let mut group = c.benchmark_group("lifecycle_spawn_despawn");
    for &n in &SIZES {
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::new("rust", n), &n, |b, &n| {
            b.iter_batched(
                World::new,
                |mut world| {
                    let mut entities = Vec::with_capacity(n);
                    for _ in 0..n {
                        entities.push(world.spawn());
                    }
                    for e in entities {
                        black_box(world.despawn(e));
                    }
                },
                BatchSize::LargeInput,
            );
        });
        group.bench_with_input(BenchmarkId::new("ffi", n), &n, |b, &n| {
            b.iter_batched(
                || mid_ecs_world_new(),
                |handle| {
                    let mut entities = Vec::with_capacity(n);
                    // SAFETY: `handle` is live until the free below.
                    unsafe {
                        for _ in 0..n {
                            entities.push(mid_ecs_world_spawn(handle));
                        }
                        for e in entities {
                            black_box(mid_ecs_world_despawn(handle, e));
                        }
                        mid_ecs_world_free(handle);
                    }
                },
                BatchSize::LargeInput,
            );
        });
    }
    group.finish();
}

fn bench_span_read(c: &mut Criterion) {
    let mut group = c.benchmark_group("span_read");
    for &n in &SIZES {
        group.throughput(Throughput::Elements(n as u64));
        let fx = Fixture::new(health_world(n));
        let health = ComponentId::from_u32(fx.health_id);
        let archetype = populated_archetype(fx.world(), health);
        let a = archetype.as_u32();
        let expected: u64 = (0..n as u64).sum();

        let rust_span = || {
            fx.world()
                .static_component_raw_span(archetype, health)
                .expect("registered and present")
        };
        let ffi_span = || {
            let mut span = empty_span();
            // SAFETY: `handle` is live, `span` is valid for one write.
            let status = unsafe {
                mid_ecs_world_static_component_raw_span(fx.handle, a, fx.health_id, &mut span)
            };
            assert_eq!(status, MidEcsStatus::Ok as i32);
            span
        };
        // SAFETY: both spans point into `fx`'s live column.
        unsafe {
            assert_eq!(sum_span(&rust_span()), expected);
            assert_eq!(sum_span(&ffi_span()), expected);
        }
        assert_eq!(
            fx.world()
                .query_static::<Health>()
                .map(|(_, h)| h.hp as u64)
                .sum::<u64>(),
            expected
        );

        group.bench_with_input(BenchmarkId::new("typed_query", n), &n, |b, _| {
            b.iter(|| {
                black_box(
                    fx.world()
                        .query_static::<Health>()
                        .map(|(_, h)| h.hp as u64)
                        .sum::<u64>(),
                )
            });
        });
        group.bench_with_input(BenchmarkId::new("rust_span", n), &n, |b, _| {
            // SAFETY: the span points into `fx`'s live column.
            b.iter(|| black_box(unsafe { sum_span(&rust_span()) }));
        });
        group.bench_with_input(BenchmarkId::new("ffi_span", n), &n, |b, _| {
            // SAFETY: as above.
            b.iter(|| black_box(unsafe { sum_span(&ffi_span()) }));
        });
        group.bench_with_input(BenchmarkId::new("rust_span_call_only", n), &n, |b, _| {
            b.iter(|| black_box(rust_span().count));
        });
        group.bench_with_input(BenchmarkId::new("ffi_span_call_only", n), &n, |b, _| {
            b.iter(|| black_box(ffi_span().count));
        });
    }
    group.finish();
}

/// Adds `Tag<I>` to `e`, moving it to the archetype `{Health, Tag<I>}`.
macro_rules! tag_entity {
    ($world:expr, $e:expr, $i:expr; $($n:literal),*) => {
        match $i {
            $( $n => { $world.insert_static($e, Tag::<$n>); } )*
            other => panic!("FRAGMENTATION sweep exceeds the tag list: {other}"),
        }
    };
}

fn fragmented_world(k: usize) -> World {
    let mut world = World::new();
    world.register_ffi_static_component::<Health>("FfiHealthStatic");
    for i in 0..k {
        let e = world.spawn();
        world.insert_static(e, Health { hp: i as u32 });
        tag_entity!(world, e, i; 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    }
    world
}

fn bench_archetypes_matching(c: &mut Criterion) {
    let mut group = c.benchmark_group("archetypes_matching");
    for &k in &FRAGMENTATION {
        let fx = Fixture::new(fragmented_world(k));
        let with = [fx.health_id];
        let with_ids = [ComponentId::from_u32(fx.health_id)];

        let ffi_count = || -> i32 {
            // SAFETY: `handle` is live; `with` outlives the call; NULL
            // buffer with capacity 0 asks for the count.
            unsafe {
                mid_ecs_world_archetypes_matching_static(
                    fx.handle,
                    with.as_ptr(),
                    1,
                    std::ptr::null(),
                    0,
                    std::ptr::null(),
                    0,
                    std::ptr::null_mut(),
                    0,
                )
            }
        };
        let ffi_fill = |buf: &mut Vec<u32>| {
            let count = ffi_count();
            assert!(count >= 0);
            buf.clear();
            buf.resize(count as usize, 0);
            // SAFETY: as above; `buf` is valid for `count` elements.
            let written = unsafe {
                mid_ecs_world_archetypes_matching_static(
                    fx.handle,
                    with.as_ptr(),
                    1,
                    std::ptr::null(),
                    0,
                    std::ptr::null(),
                    0,
                    buf.as_mut_ptr(),
                    buf.len(),
                )
            };
            assert_eq!(written, count);
        };
        let rust_collect = || -> Vec<u32> {
            fx.world()
                .archetypes_matching_static(&with_ids, &[], &[])
                .map(|id| id.as_u32())
                .collect()
        };

        let mut scratch = Vec::new();
        ffi_fill(&mut scratch);
        let mut want = rust_collect();
        want.sort_unstable();
        scratch.sort_unstable();
        // {Health} itself plus one archetype per tag, each populated.
        assert_eq!(want, scratch);
        assert_eq!(want.len(), k + 1);

        group.bench_with_input(BenchmarkId::new("rust_collect", k), &k, |b, _| {
            b.iter(|| black_box(rust_collect()));
        });
        group.bench_with_input(BenchmarkId::new("ffi_count_then_fill", k), &k, |b, _| {
            let mut buf = Vec::new();
            b.iter(|| {
                ffi_fill(&mut buf);
                black_box(buf.len())
            });
        });
        group.bench_with_input(BenchmarkId::new("ffi_count_only", k), &k, |b, _| {
            b.iter(|| black_box(ffi_count()));
        });
    }
    group.finish();
}

fn bench_change_rows(c: &mut Criterion) {
    let mut group = c.benchmark_group("change_rows");
    for &n in &SIZES {
        group.throughput(Throughput::Elements(n as u64));
        let mut world = health_world(n);
        world.increment_change_tick();
        let mut tracker = ChangeTracker::new();
        tracker.update(&world);
        let last_run = world.change_tick().get();
        world.increment_change_tick();
        let victims: Vec<_> = world
            .query_static::<Health>()
            .map(|(e, _)| e)
            .step_by(10)
            .collect();
        for e in &victims {
            world.get_static_mut::<Health>(*e).expect("alive").hp += 1;
        }
        let fx = Fixture::new(world);
        let health = ComponentId::from_u32(fx.health_id);
        let a = populated_archetype(fx.world(), health).as_u32();

        let rust_sum = || -> u64 {
            fx.world()
                .query_changed::<Health>(&tracker)
                .map(|(_, h)| h.hp as u64)
                .sum()
        };
        let ffi_sum = |rows: &mut Vec<u32>| -> u64 {
            let mut span = empty_span();
            // SAFETY (all three calls): `handle` is live; buffers are NULL
            // or valid for the capacity passed; the span points into the
            // live column and is read before any mutation.
            unsafe {
                let status =
                    mid_ecs_world_static_component_raw_span(fx.handle, a, fx.health_id, &mut span);
                assert_eq!(status, MidEcsStatus::Ok as i32);
                let count = mid_ecs_world_static_component_changed_rows(
                    fx.handle,
                    a,
                    fx.health_id,
                    last_run,
                    std::ptr::null_mut(),
                    0,
                );
                assert!(count >= 0);
                rows.clear();
                rows.resize(count as usize, 0);
                let written = mid_ecs_world_static_component_changed_rows(
                    fx.handle,
                    a,
                    fx.health_id,
                    last_run,
                    rows.as_mut_ptr(),
                    rows.len(),
                );
                assert_eq!(written, count);
                let values = std::slice::from_raw_parts(span.ptr as *const Health, span.count);
                rows.iter().map(|&r| values[r as usize].hp as u64).sum()
            }
        };

        let mut rows = Vec::new();
        let expected = rust_sum();
        assert_eq!(ffi_sum(&mut rows), expected);
        assert_eq!(rows.len(), victims.len(), "exactly the mutated rows");

        group.bench_with_input(BenchmarkId::new("rust_query_changed", n), &n, |b, _| {
            b.iter(|| black_box(rust_sum()));
        });
        group.bench_with_input(BenchmarkId::new("ffi_changed_rows", n), &n, |b, _| {
            let mut rows = Vec::new();
            b.iter(|| black_box(ffi_sum(&mut rows)));
        });
    }
    group.finish();
}

fn bench_resource_access(c: &mut Criterion) {
    let mut world = World::new();
    world.register_ffi_resource::<MidEcsTestTime>("FfiTime");
    world.insert_resource(MidEcsTestTime {
        delta: 0.016,
        frame: 0,
    });
    let id: ResourceId = world
        .lookup_ffi_resource_id("FfiTime")
        .expect("registered above");
    let id = id.as_u32();
    let handle = MidEcsWorld::from_world(world);

    let rust_read = || {
        // SAFETY: `handle` is live for this whole function.
        unsafe { &*handle }
            .world()
            .get_resource::<MidEcsTestTime>()
            .expect("inserted")
            .frame
    };
    let ffi_read = || {
        let mut span = empty_span();
        // SAFETY: `handle` is live, `span` valid for one write, and the
        // span points at one live `MidEcsTestTime`.
        unsafe {
            let status = mid_ecs_world_resource_raw_span(handle, id, &mut span);
            assert_eq!(status, MidEcsStatus::Ok as i32);
            (*(span.ptr as *const MidEcsTestTime)).frame
        }
    };
    assert_eq!(rust_read(), 0);
    assert_eq!(ffi_read(), 0);

    let mut group = c.benchmark_group("resource_access");
    group.bench_function("rust_read", |b| b.iter(|| black_box(rust_read())));
    group.bench_function("ffi_read", |b| b.iter(|| black_box(ffi_read())));
    group.bench_function("rust_write", |b| {
        let mut frame = 0u32;
        b.iter(|| {
            frame = frame.wrapping_add(1);
            // SAFETY: `handle` is live; no other reference is held.
            let time = unsafe { &mut *handle }
                .world_mut()
                .get_resource_mut::<MidEcsTestTime>()
                .expect("inserted");
            time.frame = black_box(frame);
        });
    });
    group.bench_function("ffi_write", |b| {
        let mut frame = 0u32;
        b.iter(|| {
            frame = frame.wrapping_add(1);
            let value = MidEcsTestTime {
                delta: 0.016,
                frame: black_box(frame),
            };
            let mut bytes = [0u8; 8];
            bytes[..4].copy_from_slice(&value.delta.to_ne_bytes());
            bytes[4..].copy_from_slice(&value.frame.to_ne_bytes());
            // SAFETY: `handle` is live; `bytes` is valid for 8 reads and
            // 8 is the registered type's size.
            let status =
                unsafe { mid_ecs_world_resource_write(handle, id, bytes.as_ptr(), bytes.len()) };
            black_box(status);
        });
    });
    group.finish();

    // SAFETY: freed exactly once, after every closure using it is done.
    unsafe { mid_ecs_world_free(handle) };
}

criterion_group!(
    benches,
    bench_lifecycle,
    bench_span_read,
    bench_archetypes_matching,
    bench_change_rows,
    bench_resource_access
);
criterion_main!(benches);
