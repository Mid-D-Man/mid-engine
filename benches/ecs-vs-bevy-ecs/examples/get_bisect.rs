// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-ecs.md, section "get_bisect.rs"
// ============================================================================
//! Diagnostic, kept for reference. Bisects *why* `get_component_random_access`
//! in `vs_bevy_ecs.rs` measures ~36ns per mid-ecs lookup when the isolated
//! `mid-ecs/examples/diag_ir_ops.rs` binary measures ~7.5ns for the same
//! code on the same CI toolchain (`ecs-vs-bevy-ecs` build #25 vs `diag Ir/op
//! A/B` builds #1/#2).
//!
//! The isolated binary has nothing but mid-ecs in it; the real bench has
//! criterion, every other group, and bevy_ecs in one fat-LTO unit. This
//! binary adds back exactly one of those at a time. It contains mid-ecs
//! AND bevy_ecs (both worlds built, same order and shapes as the bench's
//! `bench_get_component_random_access`), and NO criterion.
//!
//! Modes:
//! - `time`: prints, best-of-7 over 300 passes of 10,000 lookups,
//!   `mid_alone_ns` (mid world built, bevy world not yet built),
//!   `mid_bevy_alive_ns` (same loop after the bevy world exists), and
//!   `bevy_ns`. All ns per lookup.
//! - `mid-ir <0|1>` / `bevy-ir <0|1>`: both worlds built; `1` also runs
//!   20 passes of that engine's lookup loop. For callgrind differentials
//!   (op cost = total(1) - total(0), over 20 * 10,000 lookups).
//!
//! Driven by `scripts/diag_get_bisect.py` /
//! `.github/workflows/diag-get-bisect.yml`.

use std::hint::black_box;
use std::time::Instant;

use bevy_ecs::prelude::{Component, World as BevyWorld};
use mid_ecs::World as MidWorld;

#[allow(dead_code)]
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

#[allow(dead_code)]
#[derive(Component, Clone, Copy)]
struct BevyPosition {
    x: f32,
    y: f32,
    z: f32,
}

#[allow(dead_code)]
#[derive(Component, Clone, Copy)]
struct BevyVelocity {
    dx: f32,
    dy: f32,
    dz: f32,
}

const N: usize = 10_000;
const PASSES: usize = 300;
const IR_ROUNDS: usize = 20;

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

fn build_mid() -> (MidWorld, Vec<mid_ecs::world::Entity>) {
    let mut world = MidWorld::new();
    let entities: Vec<_> = (0..N)
        .map(|_| {
            let e = world.spawn();
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
}

fn build_bevy() -> (BevyWorld, Vec<bevy_ecs::prelude::Entity>) {
    let mut world = BevyWorld::new();
    let entities: Vec<_> = (0..N)
        .map(|_| {
            world
                .spawn((
                    BevyPosition { x: 1.0, y: 2.0, z: 3.0 },
                    BevyVelocity { dx: 0.1, dy: 0.2, dz: 0.3 },
                ))
                .id()
        })
        .collect();
    (world, entities)
}

fn best_ns_per_lookup(mut pass: impl FnMut() -> f32) -> f64 {
    let mut best = f64::MAX;
    for _ in 0..7 {
        let start = Instant::now();
        for _ in 0..PASSES {
            black_box(pass());
        }
        let ns = start.elapsed().as_nanos() as f64 / (PASSES as f64 * N as f64);
        best = best.min(ns);
    }
    best
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let Some(mode) = args.get(1) else {
        eprintln!("usage: get_bisect time | mid-ir <0|1> | bevy-ir <0|1>");
        std::process::exit(2);
    };
    let run_op = args.get(2).is_some_and(|r| r == "1");

    match mode.as_str() {
        "time" => {
            let (mid_world, mid_entities) = build_mid();
            let mid_alone = best_ns_per_lookup(|| {
                mid_get_random_access(black_box(&mid_world), black_box(&mid_entities))
            });
            let (bevy_world, bevy_entities) = build_bevy();
            let mid_alive = best_ns_per_lookup(|| {
                mid_get_random_access(black_box(&mid_world), black_box(&mid_entities))
            });
            let bevy = best_ns_per_lookup(|| {
                bevy_get_random_access(black_box(&bevy_world), black_box(&bevy_entities))
            });
            println!(
                "time mid_alone_ns={mid_alone:.3} mid_bevy_alive_ns={mid_alive:.3} bevy_ns={bevy:.3}"
            );
        }
        "mid-ir" | "bevy-ir" => {
            let (mid_world, mid_entities) = build_mid();
            let (bevy_world, bevy_entities) = build_bevy();
            if run_op {
                for _ in 0..IR_ROUNDS {
                    if mode == "mid-ir" {
                        black_box(mid_get_random_access(&mid_world, &mid_entities));
                    } else {
                        black_box(bevy_get_random_access(&bevy_world, &bevy_entities));
                    }
                }
            }
            black_box((&mid_world, &bevy_world));
        }
        other => {
            eprintln!("unknown mode `{other}`");
            std::process::exit(2);
        }
    }
}
