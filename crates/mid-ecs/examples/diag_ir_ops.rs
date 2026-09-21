// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-ecs.md, section "diag_ir_ops.rs"
// ============================================================================
//! Diagnostic, kept for reference like `diag_alloc_count.rs`: a
//! sandbox-valid way to count *instructions* per operation, for the
//! compute-shaped `ecs-vs-bevy-ecs` gaps that allocation counts can't
//! see (`get_component_random_access`, `insert_bundle_on_existing_entity`,
//! `spawn_n_entities_two_components`, `remove_bundle_two_components`).
//!
//! Instruction counts under callgrind are deterministic for a given
//! binary, unlike timing, so they can rank "which change removed work"
//! without a CI round trip. They are NOT a substitute for real CI
//! timing: fewer instructions is necessary evidence, not sufficient.
//!
//! Usage (the op's cost is the difference between the two runs):
//!
//! ```text
//! cargo build --release --example diag_ir_ops
//! valgrind --tool=callgrind --callgrind-out-file=a.out \
//!     target/release/examples/diag_ir_ops insert 0
//! valgrind --tool=callgrind --callgrind-out-file=b.out \
//!     target/release/examples/diag_ir_ops insert 1
//! # Ir/op = (total(b) - total(a)) / ops, ops = 10_000 (get: 200_000)
//! ```
//!
//! Modes: `get`, `insert`, `remove`, `spawn`. Second argument `0` runs
//! setup only, `1` runs setup plus the measured operation. Workloads
//! mirror `vs_bevy_ecs.rs` (same N, component shapes and call
//! sequence).

use std::hint::black_box;

use mid_ecs::world::{Entity, World};

#[allow(dead_code)] // only `x` is read; the rest mirror the bench shape
#[derive(Clone, Copy)]
struct Position {
    x: f32,
    y: f32,
    z: f32,
}

#[allow(dead_code)]
#[derive(Clone, Copy)]
struct Velocity {
    dx: f32,
    dy: f32,
    dz: f32,
}

struct Marker;

const N: usize = 10_000;
const GET_ROUNDS: usize = 20;

fn bundle() -> (Position, Velocity) {
    (
        Position { x: 1.0, y: 2.0, z: 3.0 },
        Velocity { dx: 0.1, dy: 0.2, dz: 0.3 },
    )
}

#[inline(never)]
fn op_get(world: &World, entities: &[Entity]) -> f32 {
    let mut sum = 0.0;
    for &e in entities {
        if let Some(p) = world.get_static::<Position>(e) {
            sum += p.x;
        }
    }
    sum
}

#[inline(never)]
fn op_insert(world: &mut World, entities: &[Entity]) {
    for &e in entities {
        world.insert_bundle(e, bundle());
    }
}

#[inline(never)]
fn op_remove(world: &mut World, entities: &[Entity]) {
    for &e in entities {
        world.remove_bundle::<(Position, Velocity)>(e);
    }
}

#[inline(never)]
fn op_spawn(world: &mut World) {
    for _ in 0..N {
        let e = world.spawn();
        world.insert_bundle(e, bundle());
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (Some(mode), Some(run)) = (args.get(1), args.get(2)) else {
        eprintln!("usage: diag_ir_ops <get|insert|remove|spawn> <0|1>");
        std::process::exit(2);
    };
    let run_op = run == "1";
    let mut world = World::new();
    match mode.as_str() {
        "get" => {
            let entities: Vec<_> = (0..N)
                .map(|_| {
                    let e = world.spawn();
                    world.insert_bundle(e, bundle());
                    e
                })
                .collect();
            if run_op {
                for _ in 0..GET_ROUNDS {
                    black_box(op_get(&world, &entities));
                }
            }
        }
        "insert" => {
            let entities: Vec<_> = (0..N)
                .map(|_| {
                    let e = world.spawn();
                    world.insert_static(e, Marker);
                    e
                })
                .collect();
            if run_op {
                op_insert(&mut world, &entities);
            }
        }
        "remove" => {
            let entities: Vec<_> = (0..N)
                .map(|_| {
                    let e = world.spawn();
                    world.insert_static(e, Marker);
                    world.insert_bundle(e, bundle());
                    e
                })
                .collect();
            if run_op {
                op_remove(&mut world, &entities);
            }
        }
        "spawn" => {
            if run_op {
                op_spawn(&mut world);
            }
        }
        other => {
            eprintln!("unknown mode `{other}`");
            std::process::exit(2);
        }
    }
    black_box(&world);
}
