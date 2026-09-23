// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-ecs.md, section "diag_alloc_count.rs"
// ============================================================================
//! Diagnostic, not a permanent part of the crate's public shape —
//! kept for reference rather than deleted, matching this project's own
//! precedent (`benches/iter1-isolated`, `diag_query2_unchecked.rs`).
//! The question it was built to answer is now answered (see
//! `docs/mid-ecs.md`); it stays because re-running it is a real,
//! sandbox-valid way to check "did this change actually remove an
//! allocation" before spending a real CI run on the timing question.
//!
//! `benches/ecs-vs-bevy-ecs/benches/vs_bevy_ecs.rs`'s real CI timing
//! numbers are the ones that matter, but they're not usable for root-
//! causing anything on this sandbox — the whole Iter1/Iter2
//! investigation already established that LTO-sensitive codegen makes
//! sandbox *timing* numbers meaningless here. Allocation *counts* don't
//! have that problem: counting real `alloc`/`dealloc` calls through a
//! `#[global_allocator]` wrapper is deterministic regardless of
//! optimization level or which rustc built it — same code, same
//! allocations, every time. This exists to get a real, sandbox-valid
//! signal on one question: is `Column::swap_remove_and_forget`/
//! `push_any`'s (since replaced by `Column::move_row_to`)
//! `Box<dyn Any>`-per-moved-component cost (`archetype.rs`'s
//! own doc comment, `scratch.rs`'s whole reason for existing) actually
//! the dominant allocation source behind any of the real, measured
//! mid-ecs-vs-bevy_ecs gaps — or something else entirely.
//!
//! Mirrors seven of `vs_bevy_ecs.rs`'s real workloads exactly (same
//! N=10,000, same component shapes, same call sequence), each split
//! into an uncounted setup phase and a counted measured phase.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use mid_ecs::world::World;

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

const N: usize = 10_000;

// ── Counting allocator ──────────────────────────────────────────────────
// Wraps `System` and only counts while `COUNTING` is set, so setup
// (world/entity construction) doesn't pollute the numbers for the one
// operation actually being measured.

struct CountingAllocator;

static COUNTING: AtomicBool = AtomicBool::new(false);
static ALLOC_COUNT: AtomicU64 = AtomicU64::new(0);
static DEALLOC_COUNT: AtomicU64 = AtomicU64::new(0);
static ALLOC_BYTES: AtomicU64 = AtomicU64::new(0);

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
            ALLOC_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        }
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if COUNTING.load(Ordering::Relaxed) {
            DEALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            // A realloc is neither a fresh alloc nor a dealloc in the
            // usual sense, but it's still a real call into the
            // allocator -- count it on the alloc side so it isn't
            // silently invisible; bytes tracked as the net new size.
            ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
            ALLOC_BYTES.fetch_add(new_size as u64, Ordering::Relaxed);
        }
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

fn reset_and_start_counting() {
    ALLOC_COUNT.store(0, Ordering::Relaxed);
    DEALLOC_COUNT.store(0, Ordering::Relaxed);
    ALLOC_BYTES.store(0, Ordering::Relaxed);
    COUNTING.store(true, Ordering::Relaxed);
}

struct Measurement {
    label: &'static str,
    allocs: u64,
    deallocs: u64,
    bytes: u64,
}

fn stop_counting_and_report(label: &'static str) -> Measurement {
    COUNTING.store(false, Ordering::Relaxed);
    Measurement {
        label,
        allocs: ALLOC_COUNT.load(Ordering::Relaxed),
        deallocs: DEALLOC_COUNT.load(Ordering::Relaxed),
        bytes: ALLOC_BYTES.load(Ordering::Relaxed),
    }
}

fn print_measurement(m: &Measurement) {
    println!(
        "{:<34} allocs={:>7}  deallocs={:>7}  bytes={:>9}  allocs/op={:>6.3}",
        m.label,
        m.allocs,
        m.deallocs,
        m.bytes,
        m.allocs as f64 / N as f64
    );
}

// ── Scenario 1: remove_bundle_two_components ────────────────────────────
// Mirror of vs_bevy_ecs.rs's real setup: each entity starts with
// [Marker (sparse), Position, Velocity (both archetype-tracked)], then
// loses the (Position, Velocity) bundle -- the one scenario that
// actually exercised Column::swap_remove_and_forget/push_any's boxing (both since replaced by move_row_to),
// per this file's own module doc comment reasoning.
fn measure_remove_bundle_two_components() -> Measurement {
    let mut world = World::new();
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

    reset_and_start_counting();
    for &e in &entities {
        std::hint::black_box(world.remove_bundle::<(Position, Velocity)>(e));
    }
    stop_counting_and_report("remove_bundle_two_components")
}

// ── Scenario 2: insert_bundle_on_existing_entity ────────────────────────
// Mirror of vs_bevy_ecs.rs's real setup: each entity starts with only
// [Marker] (sparse, so the *archetype-tracked* "from" table has zero
// columns), then gains the (Position, Velocity) bundle.
fn measure_insert_bundle_on_existing_entity() -> Measurement {
    let mut world = World::new();
    let entities: Vec<_> = (0..N)
        .map(|_| {
            let e = world.spawn();
            world.insert_static(e, Marker);
            e
        })
        .collect();

    reset_and_start_counting();
    for &e in &entities {
        world.insert_bundle(
            e,
            (
                Position { x: 1.0, y: 2.0, z: 3.0 },
                Velocity { dx: 0.1, dy: 0.2, dz: 0.3 },
            ),
        );
    }
    stop_counting_and_report("insert_bundle_on_existing_entity")
}

// ── Scenario 3: spawn_n_entities_two_components ─────────────────────────
// Mirror of vs_bevy_ecs.rs's real setup: brand-new entity each
// iteration, spawn then insert_bundle -- the "from" archetype is
// always the empty one, same zero-columns case as scenario 2.
fn measure_spawn_n_entities_two_components() -> Measurement {
    // `World::new()` is `iter_batched`'s uncounted setup closure in the
    // real bench (`MidWorld::new` passed directly, not part of the
    // timed routine) -- excluded here the same way, so this scenario's
    // count is measuring the same thing the real bench times.
    let mut world = World::new();
    reset_and_start_counting();
    for _ in 0..N {
        let e = world.spawn();
        world.insert_bundle(
            e,
            (
                Position { x: 1.0, y: 2.0, z: 3.0 },
                Velocity { dx: 0.1, dy: 0.2, dz: 0.3 },
            ),
        );
    }
    let m = stop_counting_and_report("spawn_n_entities_two_components");
    std::hint::black_box(&world);
    m
}

// ── Scenarios 4-7: the single-component (`insert_static`/`remove_static`)
// groups: spawn_single_component, insert_single_component,
// remove_single_component, structural_churn_insert_remove. Same call
// sequences as vs_bevy_ecs.rs; added when those groups became the next
// real gaps after the bundle paths reached parity.

fn measure_spawn_single() -> Measurement {
    let mut world = World::new();
    reset_and_start_counting();
    for _ in 0..N {
        let e = world.spawn();
        world.insert_static(e, Position { x: 1.0, y: 2.0, z: 3.0 });
    }
    std::hint::black_box(&world);
    stop_counting_and_report("spawn_single_component")
}

fn measure_insert_single() -> Measurement {
    let mut world = World::new();
    let entities: Vec<_> = (0..N).map(|_| world.spawn()).collect();
    reset_and_start_counting();
    for &e in &entities {
        world.insert_static(e, Position { x: 1.0, y: 2.0, z: 3.0 });
    }
    std::hint::black_box(&world);
    stop_counting_and_report("insert_single_component")
}

fn measure_remove_single() -> Measurement {
    let mut world = World::new();
    let entities: Vec<_> = (0..N)
        .map(|_| {
            let e = world.spawn();
            world.insert_static(e, Position { x: 1.0, y: 2.0, z: 3.0 });
            e
        })
        .collect();
    reset_and_start_counting();
    for &e in &entities {
        std::hint::black_box(world.remove_static::<Position>(e));
    }
    stop_counting_and_report("remove_single_component")
}

fn measure_structural_churn() -> Measurement {
    let mut world = World::new();
    let entities: Vec<_> = (0..N)
        .map(|_| {
            let e = world.spawn();
            world.insert_static(e, Position { x: 0.0, y: 0.0, z: 0.0 });
            e
        })
        .collect();
    reset_and_start_counting();
    for &e in &entities {
        world.insert_static(e, Marker);
        world.remove_static::<Marker>(e);
    }
    std::hint::black_box(&world);
    stop_counting_and_report("structural_churn_insert_remove")
}

fn main() {
    println!("N = {N}, one measurement pass each (no criterion iterations --");
    println!("allocation counts don't need repeated sampling the way timing does).");
    println!();

    let remove = measure_remove_bundle_two_components();
    let insert_existing = measure_insert_bundle_on_existing_entity();
    let spawn = measure_spawn_n_entities_two_components();

    print_measurement(&remove);
    print_measurement(&insert_existing);
    print_measurement(&spawn);

    print_measurement(&measure_spawn_single());
    print_measurement(&measure_insert_single());
    print_measurement(&measure_remove_single());
    print_measurement(&measure_structural_churn());
}
