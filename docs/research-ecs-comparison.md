# ECS comparison: mid-ecs, bevy_ecs and flecs

## Overview

First pass of the research the maintainer asked for on 2026-11-23: where `mid-ecs` stands next to
`bevy_ecs` and `flecs` (through its Rust binding), as input for what ECS work comes next, for the
editor, and for the remote protocol (`docs/roadmap.md`, Decision 8). The UI research (Slint, Iced,
Dioxus Native, egui, and the reference repos the maintainer named) is the next document.

This is reading, not measuring. No benchmark was run and no code changed. The only numbers quoted are
real CI numbers already recorded in `docs/mid-ecs.md`; the two build checks mentioned (`wasm32`,
and rustc 1.75) were run earlier for other work and are marked where used. No flecs number exists in
this project yet.

## Method and evidence levels

| Source | Commit | What was read |
|---|---|---|
| `SanderMertens/flecs` (the C library, version 4.2) | `85188ed`, 2026-11-19 | The manuals in `docs/` (Queries, Relationships, ComponentTraits, HierarchiesManual, Systems, FlecsRemoteApi, EntitiesComponents, ObserversManual, PrefabsManual by heading), and `include/flecs.h` and `include/flecs/private/api_defines.h` for the id layout and limits |
| `Indra-db/flecs-ecs-rs` (crate `flecs_ecs` 0.2.2, wrapping flecs 4.1) | `bf956db`, 2026-07-18 | `README.md`, `flecs_ecs/Cargo.toml` (features, versions), the bundled `flecs.h` version |
| `Mid-D-Man/bevy` (`bevy_ecs` 0.20.0-dev) | `f3b5346`, 2026-09-16 | `crates/bevy_ecs/src` (entity, storage, component, query, relationship, hierarchy, lifecycle, observer, schedule, name, entity_disabling, system/commands) and `crates/bevy_remote/src` |
| `Mid-D-Man/mid-engine` | `31ad880` and `3bbb31f`, 2026-11-22 and 2026-11-23 | `crates/mid-ecs/src`, `crates/mid-collections/src/generational_index.rs`, `docs/mid-ecs*.md` |

Every claim below carries one tag:

- **[src]** read in the source of that project.
- **[doc]** stated in that project's own documentation or README. For flecs this covers most of the
  engine's internals, because only the id layout and limits were read in its C source.
- **[ci]** a real CI figure recorded in `docs/mid-ecs.md`.
- **[gap]** the mid-ecs source was searched for it and nothing was found.

Two version notes. The Rust binding wraps flecs 4.1 while the flecs repository is at 4.2, so a 4.2
feature (the `Parent` hierarchy storage) is not necessarily in the binding. And the binding's README
lists the meta, JSON and script addons as "what's next" while its `Cargo.toml` already has features
for them, so that README line looks stale; how complete the addons are was not checked.

## Summary

| Area | mid-ecs | bevy_ecs | flecs |
|---|---|---|---|
| API | Rust, plus a C ABI by registered name [doc] | Rust only | C core; the Rust binding is alpha [doc] |
| Entity id | 32-bit index and 32-bit generation, packed into a `u64` [src] | The same layout, `repr(C, align(8))` [src] | 32-bit index, 16-bit generation, 8 flag bits in a `u64`; pairs share the id space [src] |
| Storage | Archetype Core tables, and the Sparse Shell; a type uses one, chosen by the first call that uses it [src] | Table or SparseSet, declared per component type by a constant [src] | Archetype tables; `Sparse` and `DontFragment` traits move a component out of them [doc] |
| Runtime-defined components | Only through the C side, by name [doc] | `ComponentDescriptor` may stand for something that is not a Rust type [src] | Components are entities, so any id can be one [doc] |
| Queries | Typed tuples with filters (`With`, `Without`, `Or`), `Added`, `Changed`, `Has`, `Mut`, `Ref`; none built at run time [src] | Typed, plus `QueryBuilder` for run-time queries; `Has`, `AnyOf`, `Allow` [src] | A query language, builders, cached and uncached, variables, traversal, sorting, grouping [doc] |
| Relationships | `ChildOf` and `Children` only [src] | A `Relationship` trait, one to many, built on hooks; `ChildOf` is one [src] | First-class pairs: any relation, wildcards, transitive, exclusive, symmetric, cleanup traits [doc] |
| Change detection | A pair of ticks per row for each column [src] | A pair of ticks per row, stored next to each column [src] | A counter per table and component, cached queries only [doc] |
| Hooks and observers | Component hooks, observers with `On<E>`, entity-targeted triggers [src] | Lifecycle events `Add`, `Insert`, `Discard`, `Remove`, `Despawn`; observers; hooks [src] | Observers on `OnAdd`, `OnSet`, `OnRemove`, custom events, `yield_existing`, propagation [doc] |
| Scheduling | Graph with labels, run conditions, ambiguity detection, single and rayon executors [src] | The same family, single and multi-threaded executors [src] | Systems are entities in pipelines with phases, automatic sync points, worker threads [doc] |
| Deferred work | A command queue; a deferred spawn returns a slot, not an `Entity` [src] | `Commands`; an `Entity` is allocated at once [src] | Operations are staged while the world is read only inside `progress` [doc] |
| Names and paths | None [gap] | A `Name` component with a cached hash; no name index in `name.rs` [src] | Names, hierarchical paths and `lookup` [doc] |
| Disabling | None [gap] | `Disabled` and `DefaultQueryFilters` [src] | `Disabled`, and `CanToggle` for one component [doc] |
| Templates | None [gap] | Required components (`#[require]`) [src] | Prefabs, `IsA` inheritance, overrides, variants [doc] |
| Reflection | `mid-reflect` plus the bridge in `mid-ecs` [src] | `bevy_reflect`, `ReflectComponent` [src] | The meta addon with a JSON serializer [doc] |
| Remote access | None yet | `bevy_remote`: JSON-RPC 2.0 over HTTP, 20 methods [src] | A REST API, the explorer, statistics, alerts, shell sessions [doc] |
| Platforms | Pure Rust; `cargo check` for `wasm32` passes (run 2026-11-22) | Pure Rust, including `wasm32` and Android | C core built by `cc`; `wasm32-unknown-unknown` "in progress" in the binding README [doc] |
| Rust floor | Does not build on 1.75 today (checked 2026-11-22: `mid-collections` and `rayon`) | `rust-version` 1.95 [src] | `rust-version` 1.97, edition 2024 [src] |

## Findings by area

### Entity ids

Bevy and mid-ecs use the same layout: index in the low 32 bits, generation in the high 32. Bevy's
`Entity` is `repr(C, align(8))` with the field order fixed so that it is a `u64` in memory
(`crates/bevy_ecs/src/entity/mod.rs`); mid-ecs packs `(generation << 32) | index`
(`mid-collections/src/generational_index.rs`, `as_ffi`) and its source comment says the layout was
taken from Bevy.

Flecs spends the 64 bits differently: a 32-bit entity index (`ECS_ENTITY_MASK`), a 16-bit generation
(`ECS_GENERATION_MASK`, bits 32 to 47) and 8 flag bits in the top byte (`ECS_ID_FLAGS_MASK`)
[src: `api_defines.h`]. The flag bits are what let a relationship pair live in the same space as an
entity. A 16-bit generation wraps after 65,536 reuses of a slot; mid-ecs and Bevy's 32 bits do not
in practice.

Implication: nothing to change. If a remote protocol sends entities as JSON numbers, a generation
above 2^21 pushes the value past 2^53, where a JavaScript client loses precision. Bevy's protocol
sends the number as is (`bevy_remote/src/lib.rs` shows `4294967298`); the choice is open for mid.

### Storage

Bevy decides storage per component type, once, in the type: `Component::STORAGE_TYPE` is `Table`
(fast iteration, slower add and remove, the default) or `SparseSet` (the reverse)
(`component/mod.rs`) [src]. Flecs stores everything in archetype tables unless a trait says
otherwise: `Sparse` stores a component outside the tables, with stable pointers and non-movable
types, and still lets queries work; `DontFragment` is the same sparse storage that also stops the
component from splitting tables [doc: `ComponentTraits.md`].

Mid-ecs chooses by which API a type is first used with (`insert_static` for the Archetype Core,
`insert` for the Sparse Shell) and refuses to mix them: a runtime check in `world.rs` panics if a
type is used with both [src]. The cost of that split is in the filters: they "name archetype-tracked
components only", so `With` of a Shell-only type matches nothing and `Without` matches everything
[src: `filter.rs` header]. Bevy and flecs let a sparse component take part in a query like any
other.

Implication: this is the one structural difference that limits the editor. An inspector or a
remote query that says "every entity with component X" cannot be answered across both storages
today. A cross-storage query, or the flecs approach of a trait that moves a component but keeps it
queryable, is the way out; it is also the biggest piece of work in this list.

### Component identity and run-time components

Bevy has a `ComponentDescriptor`, whose own documentation says it can describe a component that is
not a Rust type, and `World::register_component_with_descriptor` (`component/info.rs`,
`world/mod.rs`) [src]. In flecs a component is an entity carrying the builtin
`Component` component, so defining one at run time is the normal path [doc:
`Relationships.md`, "Relationship performance"]. Mid-ecs registers Rust types, and the C side
registers by name with spans [doc].

Implication: a scripting layer, or the editor's "add component" for something defined in data, needs
run-time components. The reflection registry already names types; what is missing is a component
whose layout comes from a description, stored in the same tables.

### The archetype graph

All three move entities between tables along cached edges. Flecs stores edges by direct array index
for component ids below `FLECS_HI_COMPONENT_ID` (256 by default) and in a hash map above it, and a
pair id is always in the hash map, which the manual puts at "usually 5-10% of the total cost of an
operation" [doc: `Relationships.md`]. Mid-ecs has `edge_for_insert` and cached add and remove edges
in `archetype.rs` [src].

### Queries

Flecs queries are the largest difference in kind. A query can be cached (it keeps the list of
matching tables) or uncached; terms can be components, tags, pairs, wildcards, with operators
(and, or, not, optional, and-from, or-from, not-from), sources, variables (`$this`, `$parent`),
traversal up a relationship, member value matching, sorting and grouping
[doc: `Queries.md`, `FlecsQueryLanguage.md`]. The default is cached for a query attached to an entity
(every system), uncached otherwise, and some features, such as `group_by` and `order_by`, force
caching. The Rust binding has a feature that defaults queries to uncached to save memory.

Bevy keeps typed queries but adds `QueryBuilder` for queries assembled at run time
(`query/builder.rs`), and a `QueryState` that caches matched tables and archetypes behind an
archetype generation counter (`query/state.rs`) [src]. Its term set is wider than mid-ecs's: `Has`,
`AnyOf`, `Allow`, `Or`, `Added`, `Changed`.

Mid-ecs has typed tuples of up to twelve terms with `With`, `Without`, `Or`, `Has`, `Mut`, `Ref`,
`Added` and `Changed`, and its filter runs once per archetype while the matched list is built, so a
filter costs nothing per row [src: `filter.rs`, `query_data.rs`]. It has no query assembled at run
time in Rust; the C surface does have `archetypes_matching_static(with, without, any_of)` by
component id [doc: `docs/mid-ecs.md`, FFI section].

Implication: a run-time query is the piece the editor and the remote protocol need most (the
explorer in the screenshot the maintainer sent has a query panel, and both remote protocols below
have a query method). The existing archetype matching by id is a foundation; what is missing is a
Rust-side query built from names, returning entities and reflected values.

### Relationships and hierarchies

Mid-ecs and Bevy share the design: a component on the child points at the parent
(`ChildOf`), the parent holds a collection (`Children`), and hooks keep the two in step
[src: `hierarchy.rs`; `bevy_ecs/src/relationship/mod.rs`]. Bevy generalizes it with `Relationship`
and `RelationshipTarget` traits, one-to-many only, with a `linked_spawn` option that despawns the
targets with the source. Mid-ecs has the specific case only.

Flecs makes relationships the base of the model. A relationship is a pair of two ids encoded in one
64-bit id, so adding one costs what adding a component costs; it can have wildcards, be transitive,
exclusive, symmetric or reflexive, and carries cleanup rules [doc: `Relationships.md`,
`ComponentTraits.md`]. The price is fragmentation: many relationship targets mean many tables, and
the manual says so and offers `DontFragment` as the fix. Version 4.2 also adds a second hierarchy
storage: `ChildOf` for large unstructured hierarchies (a scene with thousands of children) and
`Parent` for small structured ones (a prefab with dozens) [doc: `HierarchiesManual.md`, "Hierarchy
storage"].

Implication: the cheap step is Bevy's, a `Relationship` trait over the hooks mid-ecs already has,
which would let `ChildOf` be one case and give custom relations. Pairs with wildcards are a
different project and are not recommended now.

### Change detection

Mid-ecs and Bevy track ticks per row: mid-ecs keeps a `ComponentTicks` per row for each component,
row-aligned with the column (`Table::ticks` in `archetype.rs`); Bevy keeps `added_ticks` and
`changed_ticks` arrays next to the column (`storage/table/column.rs`) [src]. Flecs counts per table and component and says it
"is less granular than per entity tracking" but "can be used to skip entities in bulk"; it works for
cached queries only and does not see a write through `ensure` without a `modified` call
[doc: `Queries.md`, "Change Detection"].

Implication: a table-level counter next to the per-row ticks would let a consumer such as the
editor's drawable list skip a whole table cheaply; today `EditorWorld` rebuilds its list on a flag.

### Hooks, observers and events

Bevy's lifecycle events are `Add`, `Insert`, `Discard`, `Remove` and `Despawn`, each generic over a
bundle (`lifecycle.rs`) [src]. Flecs observers cover `OnAdd`, `OnSet`, `OnRemove`, multi-event and
multi-term observers, monitors, `yield_existing` (run for what already matches), custom events,
event propagation up a relationship and entity observers [doc: `ObserversManual.md`]. Mid-ecs has
component hooks and observers with `On<E>`, including triggers aimed at an entity
(`World::trigger_targets`) [src: `observer.rs`, `hooks.rs`]. Event propagation and `yield_existing`
are the two flecs features mid-ecs does not have.

### Systems, scheduling and deferred work

The three are close in shape. Mid-ecs and Bevy build a graph of systems with labels, order
constraints, run conditions and ambiguity detection (Bevy: 38 mentions in `schedule/schedule.rs`;
mid-ecs: `schedule/graph.rs`, `condition.rs`), with a single-thread and a multi-thread executor
(mid-ecs on the rayon pool; falls back to single-thread on `wasm32`) [src, doc:
`docs/mid-ecs-schedule-parallel.md`]. Flecs systems are entities with a query, placed in pipeline
phases; sync points are inserted by analysing which commands can occur and which components systems
read; threads come from `set_threads`; timers, rates and tick sources are built in
[doc: `Systems.md`].

Deferral differs in one visible way. Bevy's `Commands` holds a reference to the entity allocator and
calls `alloc()`, so `spawn_empty` returns a real `Entity` at once
(`system/commands/mod.rs:216`; the allocator is concurrent and can be used remotely,
`entity/remote_allocator.rs`). Mid-ecs's `Commands::spawn` pushes a spawn onto the queue and returns a
`Target::Slot`; the `Entity` does not exist until the queue is applied (`commands.rs`, `push_spawn`)
[src]. Code that must hand the new id to something else in the same system cannot do it.

Implication: a reserved-entity allocator is a contained change in `mid-collections` and `commands.rs`
and removes a real limitation; it also unlocks spawning from other threads.

### Names, paths, disabling and templates

None of these exist in mid-ecs [gap]; each is planned elsewhere or not planned.

- **Names.** Flecs entities have names and hierarchical paths with `lookup` (`Sun.Earth`), and the
  REST API addresses entities by path (`GET entity/Sun/Earth`) [doc]. Bevy has `Name`, a component
  holding a string with a precomputed hash; `name.rs` has no name-to-entity index [src]. The
  editor's `Name` is a stand-in for the one `mid-nodes` will define (`docs/mid-nodes.md`, N3, with its
  open question of whether names are unique among siblings).
- **Disabling.** Bevy's `Disabled` marker hides an entity from queries unless the query mentions it,
  through `DefaultQueryFilters` (`entity_disabling.rs`). The explorer's selected-entity panel has a
  Disable button. In mid-ecs this can be done where filters already run, once per archetype, so the
  per-row cost stays zero; a design that changed the iterators would touch the inlining behaviour
  that `docs/mid-ecs.md` records as fragile.
- **Templates.** Bevy has required components (`#[require(B)]` inserts `B` with `A`, recursively)
  and a `template.rs`; flecs has prefabs with `IsA` inheritance, overrides and variants
  [src, doc]. The editor's Add Component and any scene format want one of these.

### Reflection, serialization and remote access

Mid-ecs now reads and writes components by name through `Value`s
(`docs/mid-ecs-reflect.md`). That is the base both remote protocols stand on, so their shapes matter.

**Bevy Remote Protocol** (`bevy_remote/src/lib.rs`, `builtin_methods.rs`) [src]: JSON-RPC 2.0 over
HTTP. Methods: `world.get_components`, `world.query`, `world.spawn_entity`,
`world.insert_components`, `world.remove_components`, `world.despawn_entity`,
`world.reparent_entities`, `world.list_components`, `world.mutate_components`, the resource
equivalents (`get`, `insert`, `remove`, `mutate`, `list`), `world.trigger_event`,
`world.write_message`, `registry.schema`, and three streaming `+watch` methods
(`get_components+watch`, `list_components+watch`, `observe+watch`). An entity is its `u64`, a
component is its reflected type path.

**Flecs REST API** (`docs/FlecsRemoteApi.md`) [doc]: `GET entity/<path>`, `PUT` and `DELETE` for an
entity and for a component, `GET query?expr=` with options (`offset`, `limit`, `values`,
`type_info`, `query_plan`, `query_profile`), `GET world`, `GET type_info`, `GET components`,
`GET queries`, `GET tables`, `GET stats/world`, `GET stats/pipeline`, `GET commands/capture`, `PUT
action` and `PUT script`, plus shell sessions (`PUT /shell/session`). The default port is 27750. It
can run over an embedded server, a third-party one, direct calls, or an `emscripten` build, which
exports `flecs_explorer_request`. This is what the explorer in the maintainer's screenshot talks to.

The explorer's left rail has icons that look like an entity list, a search (query), a chart
(statistics), a table grid (tables), an announcement (alerts), an info mark and a manual. Each maps to
an endpoint above; none of them is in the editor.

Implication: a mid protocol can be Bevy-shaped (JSON-RPC 2.0, one method per reflected operation, the
types named by registry name) at very little cost, because `reflect_entity`, `apply_reflected`,
`insert_reflected`, `remove_reflected` and `set_reflected_path` are already those methods. What it
lacks is a query, a statistics call, and a change feed (`+watch`), which the tick storage can drive.
Flecs adds two ideas worth keeping in mind: addressing by path, and a statistics and tables view for
the tool.

### FFI and platforms

Flecs is C first and its Rust binding is a wrapper: `World` is `!Send` and `!Sync`, aliasing of a
component in one query is checked at run time, and the README says the Rust API matches the C
library's performance except where safety overhead was added, naming `get` as the example [doc]. The
binding builds the C library with `cc`, so a C compiler is needed for every target, and
its README lists `wasm32-unknown-unknown` support as in progress. The C core exports a REST entry
point for `emscripten`, so a browser path exists through that toolchain.

Mid-ecs is Rust first with a C ABI by registered name: component types are registered from Rust and
C refers to them by name and id, reads spans, and since the later passes writes through them
[doc: `docs/mid-ecs.md`, FFI]. Bevy has no C surface.

For the Android work this matters in one way: a flecs build for `armv7-linux-androideabi` needs the
NDK's C compiler through `cc`; it is likely to work and has not been tried.

### Safety

Mid-ecs allows `unsafe` in scoped places (`WorldCell` is the first outside the FFI) and has a Miri
workflow over those modules [doc]. Bevy's ECS is heavily `unsafe` and heavily tested. Flecs is C;
the binding adds checks (mutable aliasing, optional read and write locks on component access, on by
default through `flecs_safety_locks`) and a threading model where a world stays on its thread.

### Performance evidence

Only mid-ecs against bevy_ecs has numbers, all from real CI and recorded in `docs/mid-ecs.md`
[ci]:

- Structural changes are at or near parity after the work on `insert_single`, `remove_single` and
  `spawn_bundle`: build 28, `spawn_single_component` 1.00x, `insert_single_component` 0.79x,
  `remove_single_component` 0.57x, `structural_churn_insert_remove` 0.82x; `spawn_bundle_direct`
  1.18x on build 30.
- The dense two-column read loop is the known gap: `dense_query_iteration` is about 4x on the
  older x86_64 runners, 6.0x on the AMD EPYC 9V45 (build 31), and 0.94x on macOS (Apple M1) and
  1.25x on aarch64 (Neoverse-N2). The recorded explanation is the compiler declining to inline
  `next()` in a large binary, so the ratio depends on the bench binary and the CPU as much as on
  the iterator. That work is parked until the ECS is feature complete.
- Flecs numbers: none. The flecs Rust binding has no comparison against Bevy in its repository
  (checked: no Bevy reference outside its workspace `Cargo.toml`).

## What mid-ecs has that the others do not

Kept to what the sources show:

- A transform type for large worlds in the ECS crate itself, `GlobalTransformLWC` (`transform.rs`),
  next to `GlobalTransform`. Bevy keeps transforms outside `bevy_ecs` (in `bevy_transform`, not read
  for this document), and flecs ships transform components as a separate module (the explorer
  screenshot shows `Position3`, `Rotation3` and `Transform3`); whether either has a large-world
  variant was not checked.
- A C ABI designed as part of each feature, by registered name, with the write path included.
- Per-use storage without a declaration, with a runtime check that stops a type from being in both.
  This is a trade (see Storage), but it needs no derive or trait on the user's type.
- Reflection, a bridge, a play-mode snapshot and a hierarchy and inspector in an editor core already
  exist on top of it.

## Gaps, ranked

Sizes are rough: S is days of work, M a pass of its own, L several.

| # | Gap | Who has it | What it unlocks | Size |
|---|---|---|---|---|
| 1 | Run-time query by names, returning entities and values | Bevy `QueryBuilder`; flecs query language | The editor's query panel, the remote `world.query`, scripting | M |
| 2 | Statistics: entity, archetype and table counts, per-system time | Flecs `stats` and `tables` endpoints | The explorer's right-hand panels, the Console tab, a remote `stats` call | S |
| 3 | A remote protocol over the reflect API (JSON-RPC 2.0, stdio first) | `bevy_remote`; flecs REST | The command line's world commands, a desktop editor attached to a game, `+watch` | M |
| 4 | Names and paths | Flecs; Bevy has `Name` only | Addressing in the protocol, the inspector title, scenes (also `mid-nodes` N3) | M |
| 5 | Reserved entity ids in `Commands` | Bevy; flecs | Handing a new id on within one system, spawning from other threads | M |
| 6 | Disabling an entity | Bevy; flecs | The explorer's Disable button, Unity-style active toggles | S to M |
| 7 | Entity remapping and cloning | Bevy `map_entities`, `clone_entities` | Scene files, restoring a despawned entity after play, undo | M |
| 8 | A `Relationship` trait over the hooks | Bevy | Custom relations next to `ChildOf`, `linked_spawn` cleanup | M |
| 9 | Required components or templates | Bevy `#[require]`; flecs prefabs | Add Component, spawning a configured object, a scene format | M to L |
| 10 | Queries across both storages | Bevy and flecs | "Every entity with X" regardless of storage | L |
| 11 | Run-time components | Bevy descriptors; flecs | Components defined in data or by a script | L |
| 12 | Observer propagation, `yield_existing` | Flecs | Rarely needed first | M |
| 13 | A table-level change counter | Flecs | Cheap bulk skipping for tools | S |

Not recommended now: flecs-style pairs with wildcard queries (a different storage design), a
query-string parser in the core (the reflection names and a builder cover the need), and making
components entities (it would end the Rust-typed model that the rest of the engine relies on).

Suggested order, from the editor's needs: 2 and 1 first (small, and they feed the tool), then 3, then
4 together with `mid-nodes` N3, then 5 and 7 (they make play-mode restore and scenes complete), then
6. The ranking changes if the maintainer prefers to finish ECS parity before tooling; items 5, 8 and
10 are the parity ones.

## A three-way benchmark, designed and not built

The performance comparison the maintainer asked about needs flecs in the same harness as the
existing `benches/ecs-vs-bevy-ecs`.

- **Where.** A new crate beside it, outside the workspace's normal build, on the same CI pattern as
  the Bevy comparison (rustc 1.99 on CI satisfies both `bevy_ecs` 1.95 and `flecs_ecs` 1.97; the
  workspace floor is not involved). It runs on `ubuntu-latest`, macOS and the ARM Linux runner, as
  build 30 did, and writes the CPU into the summary header.
- **Workloads**, the same data and the same row counts in all three: spawn 10,000 entities with two
  components; iterate two components and write one; add and remove a component on every entity;
  random access `get`; a query with a filter; spawn and propagate a parent-child tree. Each engine
  gets its own idiomatic fast path (`Iter2` or the mutable query for mid-ecs, `QueryState` for Bevy,
  a cached query with `each` for flecs).
- **Settings that change the answer**, to record in every result: flecs query caching on, the Rust
  binding's `flecs_safety_locks` off for one run and on for another, flecs built by `cc` at the
  profile the workflow states, no LTO for the engines (the `docs/mid-ecs.md` finding that LTO changes
  the iteration results), single thread, criterion or the existing harness for both.
- **Controls**: a raw-slice loop as the floor, as `ecs-vs-bevy-ecs` already has, so a ratio can be
  read against the hardware.
- **Reading the result**: a ratio inside one run is meaningful; across runs only on the same CPU
  line, as the recorded builds 25 and 29 showed.

## Questions for the maintainer

1. Should relationships stay a hierarchy only, or become a trait like Bevy's (gap 8), or is the
   flecs pair model something to design toward later?
2. Names and paths: unique among siblings (the open `mid-nodes` question), and written with a dot
   like flecs or a slash?
3. Protocol shape: JSON-RPC 2.0 like Bevy's, a REST path model like flecs, or both (the second as
   a thin layer over the first)?
4. Entity numbers on the wire: plain numbers, or strings, given the 2^53 limit in JavaScript?
5. Is gap 10 (queries across both storages) worth doing before the editor needs a general query, or
   should a query be allowed to see only the Archetype Core at first?
6. Build the three-way harness now, or after the next ECS feature work?


## Addendum: blaze-ecs (2026-11-25)

The maintainer found `blaze-ecs` as a further reference. It was read once and in less depth than the
three libraries above. It has no benchmark (the crate has no `benches` directory and no benchmark
dependency [src]), so the finding "Performance evidence" stands unchanged.

| Source | Commit | What was read |
|---|---|---|
| `indexable-inc/blaze` | `2bbf1bc8`, 2026-11-25 | `crates/blaze-ecs` (17,558 lines of Rust in `src`, of which `ui` is 2,697), `crates/blaze-ecs-wasm` (1,005), the manifests, the public API list of `world.rs`, the headers of `id.rs`, `query.rs`, `pipeline.rs`, `reactive.rs`, `explorer.rs`, `ui/mod.rs` |

What it is: a Flecs-4.2-equivalent archetype ECS in Rust, `#![no_std]`, `#![forbid(unsafe_code)]` and
"sans-IO" (no sockets, no ambient I/O), with an allocator type parameter on its types [src]. An entity
is a 64-bit handle of a 32-bit slot and a 32-bit generation, with generation 0 meaning null [src].
Tables are archetypes of safe structure-of-arrays columns with cached add and remove edges between
tables [doc]. `DontFragment` tags and pairs are stored sparsely, with no table moves [src]. The crate
cannot be used as a dependency of this project: its collections crate `blaze-own` turns on many unstable
features, `blaze-ecs` uses `register_tool`, its rust-version is 1.101, and it depends on six crates of
the blaze workspace (data, int, json, math, own, tab) [src]. Its license is MIT OR Apache-2.0 [src,
workspace manifest]. It can be read and its ideas taken.

What it shows for each of the gaps ranked above:

| Gap | In blaze-ecs |
|---|---|
| 1 Run-time query by names | `QuerySpec` holds `With` and `Without` terms over ids (components, tags, exact and wildcard pairs), prefab filtering and a cascade order, built at run time; the Explorer endpoint `/api/query` takes `with` and `without` [src] |
| 2 Statistics | `live_count`, `table_count`, `component_count`, and a `stats` map in the world snapshot [src]; system timing not found in the API list read [gap] |
| 3 Remote protocol | A pure function `handle_request(world, method, path, body)` over a REST route set: `/api/world`, `/api/world.tab`, `/api/entity`, `/api/entity/{slot}` with component, tag and parent sub-routes, `/api/query` [src]; the routes follow the Flecs Explorer API in spirit [doc], entities are addressed by slot, not by path |
| 4 Names and paths | `spawn_named`, `entity_by_name`, `set_name` [src]; a path syntax was not seen in the API list [gap] |
| 5 Reserved ids in commands | No deferred command queue found: no `defer` or `Commands` match in `world.rs`, `pipeline.rs` or `fetch.rs` [gap] |
| 6 Disabling | Not found in `world.rs` [gap] |
| 7 Cloning and remapping | No clone found [gap]; `instantiate(prefab)` exists [src] |
| 8 Relationships | First-class `(Rel, Target)` pairs, with `ChildOf` cascade, and the traits `Exclusive`, `Transitive`, `Symmetric`, `OnDeleteCascade`, `Traversable`, `Acyclic`, `OrderedChildren`; a typed `Relation` trait with `relate` and `unrelate` [src] |
| 9 Required components and templates | Prefabs with `IsA` inheritance, `instantiate` and `get_inherited` [src]; required components in the Bevy sense not seen [gap] |
| 10 Queries across both storages | `DontFragment` exists [src]; whether queries match it was not checked [gap] |
| 11 Run-time components | `register_component(name)` and a variant with a schema, still tied to a Rust type [src] |
| 12 Observer propagation | `OnAdd`, `OnSet` and `OnRemove` observers with exact and wildcard pair filters, a capped event buffer and callbacks [src]; propagation and replay of existing matches not seen [gap] |
| 13 Table-level change counter | Each column keeps a changed tick per row and a `max_tick`, which lets a query skip a whole table in constant time; the world has `changed_since` and `each_changed` [src] |

Also relevant to the UI work. `reactive.rs` (886 lines) holds `Signal`, `Memo` and effects with
glitch-free push-pull propagation, written as the runtime target for UI components in the style of Solid
and Svelte runes, plus incremental collection helpers [src, doc]. The crate's own Explorer UI mirrors a
remote world into a client-side world and renders it from Rust compiled to wasm [doc]; see the addendum
in `docs/research-ui-comparison.md`.

Ideas to take, as proposals: per-row change ticks with a column maximum (gap 13); the shape of a pure
request handler over the world for the remote protocol decision; deterministic world summaries and an
invariant check as test oracles (`snapshot_tab` and `verify_invariants` exist [src]); signals for the
`.mxui` bindings. Not read: the table move code, how `DontFragment` interacts with queries, and
pipeline execution beyond its header.

## Sources

- flecs: `docs/Queries.md` (caching, change detection), `docs/Relationships.md`,
  `docs/ComponentTraits.md` (`Sparse`, `DontFragment`), `docs/HierarchiesManual.md`,
  `docs/Systems.md` (staging, sync points, threading), `docs/FlecsRemoteApi.md`,
  `docs/EntitiesComponents.md` (names), `include/flecs.h`, `include/flecs/private/api_defines.h`.
- flecs_ecs: `README.md`, `flecs_ecs/Cargo.toml`, `flecs_ecs_sys/src/flecs.h`.
- Bevy fork: `crates/bevy_ecs/src/entity/mod.rs`, `entity/remote_allocator.rs`,
  `component/mod.rs`, `component/info.rs`, `world/mod.rs`, `storage/table/column.rs`,
  `query/state.rs`, `query/builder.rs`, `query/filter.rs`, `query/fetch.rs`, `relationship/mod.rs`,
  `hierarchy.rs`, `lifecycle.rs`, `observer/mod.rs`, `schedule/schedule.rs`,
  `schedule/executor/`, `name.rs`, `entity_disabling.rs`, `system/commands/mod.rs`;
  `crates/bevy_remote/src/lib.rs`, `builtin_methods.rs`.
- mid-engine: `crates/mid-ecs/src/` (`world.rs`, `archetype.rs`, `filter.rs`, `commands.rs`,
  `hierarchy.rs`, `observer.rs`, `hooks.rs`, `schedule/`, `reflect.rs`, `transform.rs`),
  `crates/mid-collections/src/generational_index.rs`, `docs/mid-ecs.md`,
  `docs/mid-ecs-schedule-parallel.md`, `docs/mid-ecs-reflect.md`, `docs/mid-nodes.md`.
