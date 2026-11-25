# ECS requirements of a C game: Tower of Babylon and mid-ecs

## Overview

Tower of Babylon (`MidManStudio/Tower-Of-Babylon`, C99 with raylib) chose flecs on 2026-11-25 for its monsters, the G.V.E.X. specimens (`docs/design/10-gvex-dossier.md` in that repository). The game cannot use `mid-ecs` for this today, and the decision was to keep a list, while the game uses flecs, of what `mid-ecs` would need in order to reach the same use. This file is that list. It is a set of requirements with evidence, not a plan to switch, and it adds to `docs/research-ecs-comparison.md` (which is written from the editor's needs) rather than repeating it.

Evidence tags follow that document, plus two:

- **[spike]** run on 2026-11-25 in a sandbox with flecs 4.2.0 (the repository at `7d1bc7d`, 2026-11-24): the amalgamated `distr/flecs.c` and `distr/flecs.h`, gcc, `-std=gnu99 -O2`.
- **[earlier]** found when the game's ECS options were first looked at, in a reading of `crates/mid-ecs` and its C ABI that was not repeated for this file. Treat these as leads to re-check before anyone relies on them.

## What the game does with an ECS

The dossier lists eleven creature kinds in tiers X0 to X6. Read as data, each is a tier, a sense kit (sight, hearing, smell, wall-scan, vibration), a body, a behaviour and variants of one species. Three behaviours are relations between creatures: a Prime Specimen governs a swarm, a boss spawns subordinates, and a creature divides into more of itself when its body is hit. The game is multiplayer: one process on a phone runs the server's world of creatures and, in the same process, the client's picture of what the server said.

## Requirements

| # | Requirement | The game's use | flecs | mid-ecs |
|---|---|---|---|---|
| R1 | Several independent worlds in one process, usable from C | the host phone holds the server world and a client view world | **[spike]** two `ecs_init()` worlds in one program, a system in one did not touch the other | multiple worlds exist in Rust **[src]** (the tick-wraparound test in `world.rs` runs two, and `docs/mid-editor.md` notes that handles from two worlds collide); through the C ABI **[gap]** not checked |
| R2 | Components defined from C: a name, a size and an alignment, no Rust type | the game's structs (`Pos`, `Health`, `Senses`) are C structs | `ECS_COMPONENT` **[spike]** | the C side looks up components that Rust registered, by name **[doc]**; a component defined by C bytes **[earlier]** is not built. This is gap 11 in `research-ecs-comparison.md` (run-time components, size L) |
| R3 | Systems and a schedule callable from C, with phases and ordering | `sense`, `chase`, `attack`, `divide`, `detonate` as systems over the right components | `ECS_SYSTEM` with `EcsOnUpdate` **[spike]** (a signature string needs the query DSL addon) | systems and the schedule exist in Rust **[doc: `mid-ecs-schedule.md`]**; from the C ABI **[earlier]** there are none |
| R4 | Relationships between entities with a lookup in both directions and cleanup rules | `governs` (Prime to colony), `spawned_by` (brood to boss), `divided_from`, later a team | pairs **[doc]** | `ChildOf` and hierarchy **[doc: `mid-ecs-hierarchy.md`]**; custom relations are gap 8 (a `Relationship` trait over the hooks, size M). The game mostly needs "who is my target" and "who are my members", not wildcard queries, which `research-ecs-comparison.md` marks as not recommended |
| R5 | Observers or hooks callable from C | react when a creature's health reaches 0: divide, detonate, drop an item | observers **[doc]** | hooks and events exist in Rust **[doc: `mid-ecs-events.md`]**; from C **[earlier]** none |
| R6 | Prefabs or templates with overrides | a variant of a species is the species with a few changed values | prefabs **[doc]** | gap 9 (required components or templates, size M to L) |
| R7 | Cheap "what changed" and a way to keep a short history of selected components | send only the creatures that changed to each player; judge a shot against where a creature was a few ticks ago (a ring of the last 64 ticks, as the game already keeps for players) | change detection on queries **[doc]**; the history ring is the game's own code, a system that copies `Pos` | change detection `Added` and `Changed` **[doc]**; the history is the same kind of own code |
| R8 | A C-only build that works for the native targets (including `armv7-linux-androideabi` with the NDK's compiler) and, later, for the web through emscripten | the game's web build is emscripten C and cannot link a Rust library at all; native multiplayer is the Rust driver | C, no dependencies **[doc]**; native build works **[spike]**; Android and emscripten **[gap]** not tried by the game yet | a Rust library with a C ABI; the web target checked was `wasm32-unknown-unknown`, which is not linkable from emscripten **[earlier]** |
| R9 | A small footprint, a custom set of parts | a phone, an APK | a custom build with `FLECS_SYSTEM`, `FLECS_PIPELINE`, `FLECS_TIMER`, `FLECS_MODULE`, `FLECS_PARSER`, `FLECS_QUERY_DSL`, `FLECS_LOG` and `FLECS_OS_API_IMPL` compiled in 10 seconds on one core to a 1.4 MB object **[spike]**; the source is 4 MB for `flecs.c` and 1.2 MB for `flecs.h` | size of the C-ABI static library not measured |

## What it would take, in order of what blocks the game most

1. **R2 and R3 together** are the wall. A game written in C cannot define a creature's data or its behaviour through the current C ABI, so it would have to write that part in Rust, and the game has no Rust in it apart from the networking library.
2. **R8** is the second: a library the web build cannot link means two code paths for the same creatures, one of them without an ECS.
3. **R1** may already be satisfied in Rust and only need exposing.
4. **R4, R5, R6** are in `research-ecs-comparison.md` as gaps 8 and 9 and the hooks work, and are the same work an editor needs.

## Notes from using flecs (to be extended as the game uses it)

- A system declared with a signature string (`ECS_SYSTEM(world, Fn, EcsOnUpdate, Pos)`) needs `FLECS_QUERY_DSL` in a custom build; without it the program aborted at the declaration with no message **[spike]**.
- `ECS_COMPONENT` and `ECS_SYSTEM` declare local variables, so they cannot sit in an `if` without braces, and the component id is per world **[spike]**.
- Still to record while the game uses it: entity counts per floor, time per system on a phone, the cost of a snapshot with creatures in it, and anything flecs makes awkward.

## Sources

- flecs repository at `7d1bc7d` (2026-11-24), `distr/flecs.c` and `distr/flecs.h`, version 4.2.0; the spike program was a two-world test of about 25 lines and is not kept.
- `docs/research-ecs-comparison.md` in this repository, for every **[doc]** and **[src]** entry about mid-ecs and flecs.
- `crates/mid-ecs/src/world.rs` and `docs/mid-editor.md`, for the two-world evidence in R1.
- Tower of Babylon: `docs/design/10-gvex-dossier.md`, `docs/gvex_session.md`.
