# mid-ecs reflect

## Overview

The bridge between `mid-reflect` and `mid-ecs` (roadmap T1b). It reads, replaces, adds and removes
an entity's components by registered name, through `Value`s. A name and a value go in, a name and a
value come out, and the caller never names a Rust type. That is what an inspector, a scene loader
and the remote protocol need.

Source: `crates/mid-ecs/src/reflect.rs`, behind the `reflect` feature of `mid-ecs`, plus
`World::static_component_types` in `world.rs`, which is always compiled. The shape follows Bevy's
`ReflectComponent` (read from the `Mid-D-Man/bevy` fork, `bevy_ecs/src/reflect/component.rs`): a
table of function pointers per component type, attached to the type's registration as type data,
with a constructor for hand-written tables. The differences are listed under "Decisions".

## Decisions

- **The bridge lives in `mid-ecs`, behind a feature, not in `mid-reflect`.** `mid-reflect` has no
  dependencies and builds on Rust 1.75. `mid-ecs` does not build there (`mid-collections` uses
  `usize::is_multiple_of`, and `rayon` needs 1.80). Keeping the bridge out of `mid-reflect` keeps
  that crate usable on its own. Bevy makes the same split: `ReflectComponent` is in `bevy_ecs`.
- **The feature is off by default.** A `mid-ecs` build without `reflect` has no new dependency and
  no new code. With it, `mid-reflect` comes in with its `math` feature, because the transform
  components hold `mid-math` types.
- **The enumeration is always compiled.** `World::static_component_types(entity)` yields the
  `TypeId` of every Archetype Core component the entity holds, in the order of its archetype's
  signature. It is empty for a dead entity. It checks `is_alive` itself: the archetype lookup is
  by index only, so a stale handle whose slot was reused would otherwise read the new occupant's
  components (a test covers this).
- **Archetype Core only.** The tables call `insert_static`, `get_static`, `get_static_mut`,
  `has_static` and `remove_static`. Sparse Shell components are neither enumerated nor reachable
  by name. A type lives in one storage, and the engine's own components live in the Core.
- **`insert` adds or replaces.** `insert_static` is a no-op when the component is already there, so
  the table checks first and replaces through `get_static_mut`. Bevy's `insert` also replaces.
- **`apply` needs the component to be present.** A missing one is `ComponentReflectError::Missing`
  where Bevy panics. Dead entities are `NotAlive`.
- **Replacement fires no hooks and stamps the value as changed; adding fires the add hooks.** That
  follows from the calls above, and it matches how `get_static_mut` already behaves.
- **A bad value changes nothing, not even the change tick.** The value is parsed before the
  mutable access that stamps the tick. A test asserts both.
- **`ChildOf` goes through the hierarchy.** Applying or inserting it calls `World::set_parent`
  (which enables the hierarchy, refuses cycles and dead entities, and moves a child between
  parents), and removing it calls `remove_parent`. A raw `insert_static` would skip the cycle
  check. Refusals come back as `ComponentReflectError::Hierarchy`.
- **`Children` is read-only.** Apply, insert and remove all return `Refused`, because only the
  hierarchy hooks may write it. `Children::from_value` fails too, with the new
  `ErrorKind::Unsupported` in `mid-reflect`, so the registry's own `apply` and `construct` cannot
  be used to build one either.
- **An entity is its packed `u64`** (`Entity::as_ffi`), registered as `mid_ecs::Entity`. A handle
  read back from a value is checked against the world's generations like any other. Handles with a
  high generation exceed 2^53, so a JavaScript client reading the JSON would lose precision. Rust
  readers and `mid-cli` are not affected; the remote protocol will have to decide (a string, or a
  pair).
- **`reflect_entity` lists components in registration order,** not archetype order, so the list is
  stable from run to run. The cost is one pass over the registry per entity. It also reports how
  many components the entity holds that are not registered as components (`unreflected`), so an
  inspector can say so.
- **Stable names.** `mid_ecs::LocalTransform`, `mid_ecs::GlobalTransform`,
  `mid_ecs::GlobalTransformLWC`, `mid_ecs::ChildOf`, `mid_ecs::Children` and `mid_ecs::Entity`,
  chosen the way `mid-reflect` names its own types.
- **The type data carries the name,** so an error can say which component without the registry.
- **No C counterpart.** Reflection is for Rust-side tools and the remote protocol. The C surface
  already reads and writes spans by registered name.

## API

| Item | What it does |
|---|---|
| `World::static_component_types(entity)` | The `TypeId`s of the entity's Archetype Core components (always compiled) |
| `ReflectComponent::of::<T>()` | Type data for a plain Archetype Core component |
| `ReflectComponent::new(name, contains, get, apply, insert, remove)` | Type data with hand-written operations |
| `ReflectComponent::{name, contains, get, apply, insert, remove, set_path}` | The operations, on a `World` and an `Entity` |
| `register_component::<T>(&mut registry)` | Registers `T` and its contents, attaches `ReflectComponent::of::<T>()` |
| `register_component_with::<T>(&mut registry, data)` | The same with hand-written type data |
| `register_core_components(&mut registry)` | `LocalTransform`, `GlobalTransform`, `GlobalTransformLWC`, `ChildOf`, `Children` |
| `World::reflect_entity(registry, entity)` | Every registered component with its value, plus the unreflected count |
| `World::{reflect_component, has_reflected, apply_reflected, insert_reflected, remove_reflected, set_reflected_path}` | The same operations by registered name |
| `ComponentReflectError` | `NotAlive`, `UnknownComponent`, `NotAComponent`, `Missing`, `Refused`, `Value`, `Hierarchy` |

## Modules

### `reflect.rs`

**What it does:** the `Reflect` implementations for `Entity`, `LocalTransform`, `GlobalTransform`,
`GlobalTransformLWC`, `ChildOf` and `Children`; the error type; `ReflectComponent` and its two
constructors; the registration functions; and the `World` methods that look a component up by
registered name.

**Decisions:** the ones above. Also: the operations are plain `fn` pointers, as in Bevy, so the type
data is `Clone` and `Send + Sync` with no boxing, and a game's own components with special rules
use `ReflectComponent::new` the way `ChildOf` and `Children` do.

**Tests:** 35 in `reflect::tests`. They cover each `Reflect` implementation, registration
(idempotence, a name conflict), reading, every error, `apply` and its change stamp, `insert` as add
and as replace and its hook, `remove`, `set_path`, the hierarchy rules through reflection (insert,
move, cycle through insert and through apply, dead parent, remove, `Children` refused), and a
transform inserted from JSON feeding `propagate_transforms`.

### `world.rs`: `static_component_types`

**What it does:** the enumeration described above.

**Tests:** five in `world::tests`: the listed types, removal and other entities, a dead entity, a
Sparse Shell component not listed, and a stale handle whose slot was reused.

## Not built yet

- **Entity remapping.** Bevy's `ReflectComponent::map_entities` rewrites the entity references in a
  component when a scene is loaded into another world. `ChildOf` holds one. Needed before scenes.
- **`take` and `copy`.** Removing a component and returning its value, and copying one between
  entities or worlds. Bevy has both.
- **Resources by name** (Bevy's `ReflectResource`) and **Sparse Shell components.**
- **Events when a reflected edit happens.** The editor's undo (E1) will want a before and after
  value; `apply` does not return the old one yet.
- **Reflection for the rest of the engine's components,** as they appear (`mid-nodes` names, the
  physics components, `mid-anim` players).

## Verification status

Sandbox, rustc 1.97.1, x86_64. Real CI: the maintainer reported the `mid-ecs 32-bit` workflow
(i686 tests and the `armv7-linux-androideabi` check) passing, and the `mid-ecs` test workflow
failing on its `wasm32` check, which the fix below addresses.

| | before | after |
|---|---|---|
| `mid-ecs` lib tests, default features | 1,193 | 1,198 (5 in `world.rs`) |
| `mid-ecs` lib tests, `--features reflect` | not applicable | 1,233 (35 in `reflect.rs`) |
| `mid-ecs` doctests | 40 | 40, and 41 with `reflect` (the module example) |
| lib tests with `--features reflect,scratch-arena` | not recorded | 1,238 |
| `mid-reflect` tests | 63 without `math`, 69 with | 64 without, 70 with (plus 4 doctests) |
| `mid-app` tests | not measured before | 49 (the crate is not changed) |

- Ten hand-made mutants of the bridge and the enumeration (apply without the presence check, insert
  that never replaces, `ChildOf` apply bypassing the hierarchy, `Children` apply allowed, the
  unreflected count fixed at zero, `set_path` that skips the write, `remove` that always reports
  true, the alive check removed from the enumeration, a wrong entity value, reversed registration
  order): all ten caught. Two survived the first version of the tests (the `ChildOf` bypass, which
  the cycle test did not reach through `apply`, and the missing alive check, which a plain dead
  entity does not expose); each got a test.
- `rustfmt` is clean on every file this change touches. `cargo fmt --check` on the whole crate
  still reports the same differences as on unmodified main, none in the new code. Clippy with
  `--no-deps` reports nothing in `reflect.rs`; it stops on two existing errors in `resource.rs`
  (a mutable borrow from an immutable input), which are on main as well.
- `mid-reflect` builds and passes on Rust 1.75 (74 tests with `math`, checked directly). `mid-ecs`
  does not build on 1.75 for the two reasons under "Decisions", before and after this change, so the
  floor check for the bridge itself is rustc 1.97 only.
- `cargo check --target wasm32-unknown-unknown` passes for `mid-collections`, `mid-ecs`, `mid-ecs`
  with `reflect`, `mid-reflect` with `math`, and `mid-app` with the feature. `cargo tree` for that
  target still shows no `rayon`, and shows `mid-reflect` when `reflect` is on. The first delivery
  was never checked on `wasm32`; see "Fixes and Problems".
- Not run in the sandbox: the C smoke test (no C surface changed) and the three benchmark crates
  (nothing they use changed).

## CI and Workflows

- `mid-ecs-test.yml`: the unit-test step now runs with `--features mid-ecs/reflect`, so it covers
  the default tests and the bridge in one run with the same output format; the build step and the
  `wasm32` step each gain a `cargo check` with the feature.
- `mid-ecs-32bit.yml` (new): runs `mid-ecs` (with `reflect`), `mid-app` and `mid-reflect` on
  `i686-unknown-linux-gnu`, and checks them for `armv7-linux-androideabi` without the NDK. It is
  the check the roadmap asked for on the phone's architecture, never done for the ECS before. It
  runs on a manual dispatch or on a commit or pull request title with `--mid-ecs-32bit` or
  `--mid-all`. It is separate from the editor workflow on purpose: a 32-bit failure in the ECS
  should not stop an APK build.

## Fixes and Problems

### The bridge's dependency was gated out of wasm32
The first delivery added `mid-reflect` to `crates/mid-ecs/Cargo.toml` just above
`[dev-dependencies]`. That position is inside the `[target.'cfg(not(target_arch = "wasm32"))'.dependencies]`
table that holds `rayon`, so the dependency existed on every target except wasm32, while the
`reflect` feature and `reflect.rs` still compiled there. Native and 32-bit builds passed; the
`wasm32` step of `mid-ecs-test.yml` failed with `unresolved import mid_reflect`. The sandbox had no
`wasm32` target when the change was built, so the check was never run. The dependency now sits in
the main `[dependencies]` table, and the `wasm32` checks above were run before delivering the fix.

### A mutation detector that counted failures as compile errors
The first mutation run printed "compile error" for eight of ten mutants. The detector matched
cargo's `error: test failed` line, which is what a killed mutant produces. Only the two mutants
that passed were real survivors, and they were the two that got new tests.
