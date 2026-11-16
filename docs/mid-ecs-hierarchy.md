# mid-ecs hierarchy

## Overview

A parent/child hierarchy for `mid-ecs`: `ChildOf` (an entity's parent) and
`Children` (a parent's children, in the order they were added), kept in step by
component hooks. Setting or removing a `ChildOf` is all it takes; the parent's
`Children` follows, an empty one is removed, and despawning a parent despawns
everything below it.

Built on what already existed: component hooks (`docs/mid-ecs.md`, "Component
hooks"), `Commands`, and the Archetype Core. Source: `crates/mid-ecs/src/hierarchy.rs`,
with the `Commands` side in `commands.rs`, the C surface in `ffi.rs`, and
`HierarchyPlugin` in `mid-app`. The design follows Bevy's relationship hooks
(read from the `Mid-D-Man/bevy` fork: `bevy_ecs/src/relationship/mod.rs` and
`hierarchy.rs`), adapted to mid-ecs's hook system.

## Decisions

- **Archetype Core for both components.** They are then queryable like any
  other (`query_data::<(Entity, &Children)>()`, `With`/`Without<ChildOf>` for
  roots), which the transform propagation this is meant for needs. The Sparse
  Shell has no query. A `ChildOf` inserted through the Sparse Shell
  (`World::insert`) is rejected and removed, because the hierarchy reads only
  the Archetype Core. The cost of the Archetype Core: `insert_static` never
  overwrites, so moving a child is a removal and an insert.
- **Hooks, opt-in per world.** `World::enable_hierarchy()` registers the four
  hooks and is idempotent. A world without any hooks pays nothing on structural
  changes; with the hierarchy on, every structural change checks the hook table.
  That cost is **not measured**. The `World` hierarchy methods and the
  `Commands` methods enable it themselves; `mid_app::HierarchyPlugin` does it for
  an `App`. A `ChildOf` inserted by hand before enabling has no `Children` behind
  it (a test pins this down).
- **Despawn is recursive** once the hierarchy is on. `Children`'s despawn hook
  queues one command that despawns the whole subtree, deepest and last first. It
  is one command, not one per level, because the hook-command loop gives up after
  1,000 rounds and a per-level scheme would panic on a chain deeper than that (a
  test despawns a 2,500-deep chain).
- **Cycles are refused.** `set_parent` and `add_children` return a
  `HierarchyError` (`NotAlive`, `SelfParent`, `Cycle`) and change nothing. A
  `ChildOf` inserted by hand that points at itself, at a dead entity, or at a
  descendant is removed as soon as the operation that inserted it finishes. A
  walk up the hierarchy is bounded at 2^20 levels, so a corrupted hierarchy cannot
  loop forever; deeper than that counts as a cycle.
- **Order.** `Children` is in the order children were added. Moving a child puts
  it last under the new parent. Setting the parent a child already has is a no-op
  and keeps its place.
- **Removal searches from the back.** Despawning a subtree takes the newest
  child out first, so each removal is at the end of its list; removing an
  arbitrary child is still O(children).
- **`Children` has no public constructor**, so it cannot be inserted by hand. An
  entity with no children has no `Children`.
- **Losing `Children` detaches.** Removing the `Children` component from a live
  parent detaches its children (they stay alive).
- **The hierarchy components are plain components.** Observers see their
  lifecycle (`Add`, `Remove` and `Despawn` of `ChildOf` and `Children`; tested).
  Change detection on them was not tested.
- **Observer propagation follows `ChildOf`.** An event type with
  `Event::PROPAGATE` set, triggered at an entity, is delivered at each ancestor in
  turn (`docs/mid-ecs-events.md`, "Propagation: the design"). It reads only `ChildOf`
  through `World::parent`, so it works in a world that never enabled the hierarchy,
  and it stops at a loop made by raw inserts, which only such a world can have.

## API

On `World`: `enable_hierarchy`, `set_parent`, `add_child`, `add_children`
(all or nothing), `spawn_child`, `remove_parent`, `remove_children`,
`clear_children` (detach, keep alive), `despawn_children` (despawn everything
below, keep the parent), `parent`, `children`, `root_of`, `is_ancestor_of`,
`ancestors` (nearest first), `descendants` (depth first, in child order).

On `EntityCommands`: `set_parent`, `add_child`, `add_children`, `remove_parent`,
`clear_children`, `despawn_children`, and `with_children`, which gives a
`ChildSpawner` whose `spawn` and `spawn_bundle` queue children in order and work
under a parent that is itself a queued spawn. Refused links do nothing, like every
other command.

Components and types: `ChildOf(pub Entity)`, `Children` (derefs to `[Entity]`),
`HierarchyError`, `Ancestors`, `Descendants`, `ChildSpawner`.

C (`mid_ecs.h`): `mid_ecs_world_enable_hierarchy`, `_set_parent`, `_remove_parent`,
`_parent`, `_children` (the null-buffer-queries-the-count idiom, capacity in
elements), `_clear_children`, `_despawn_children`, plus a new status
`MID_ECS_CYCLE` (-12). A self parent returns `MID_ECS_INVALID_ARGUMENT`, a dead
entity `MID_ECS_NOT_ALIVE`, no parent `MID_ECS_NOT_FOUND`.
`mid_ecs_world_despawn` is now recursive when the hierarchy is on. There is no C
`Commands` counterpart.

## Modules

### `hierarchy.rs`

**What it does:** the components, the four hooks (`ChildOf` insert and discard,
`Children` discard and despawn), the `World` methods, and the two iterators.

**Decisions:**
- Hook commands re-check before acting: the link command confirms the child still
  points at the parent, and the subtree despawn skips a root that was moved to
  another parent by something else queued in the same flush.
- The `ChildOf` insert hook queues the append to the parent's `Children` instead
  of writing it directly, so creating `Children` on a parent that has none and
  appending to an existing one share one path.

**Tests (41):** building and order; enabling (twice, before-versus-after raw
inserts, bundles); `add_children` all-or-nothing; moving, same-parent no-op,
removal, `remove_children`, `clear_children`, removing `Children` by hand;
refusals changing nothing; cycles at any depth; invalid raw inserts and a Sparse
Shell `ChildOf` being taken out again; despawning a child, a subtree, a middle
node, `despawn_children`; a 2,500-deep chain and a 20,000-wide parent;
two hook-ordering cases (a link whose child loses its parent in the same flush,
and a child moved away by another hook before the subtree despawn); traversal
(`descendants`, `ancestors`, `root_of`, `is_ancestor_of`, dead entities,
querying `Children`); the `Commands` methods and `with_children` (queued and
existing parents, nested); observers seeing the lifecycle; and a random history
(8 seeds, 600 steps each) compared with an independent model after every step.
The random test counts what it reached and fails if it did not reach enough
(1,163 links, 511 re-parentings, 83 refused cycles, 46 real subtree despawns, 60
refused raw inserts in the calibration run).

### `commands.rs` (additions)

`Command::Link`, the one command that resolves two targets (either may be a
queued spawn), the `EntityCommands` methods above, and `ChildSpawner`.

### `ffi.rs` (additions)

The seven functions above, the `Cycle` status, and the note on
`mid_ecs_world_despawn`. **Tests (10 Rust, 12 C checks):** null handling, order
and round trips, the buffer conventions (null, too small writes nothing, exact,
roomy leaves the tail alone), distinct error codes that change nothing, parent
and remove-parent reporting, recursive despawn through C, the counts returned by
`clear_children` and `despawn_children`, idempotent enabling.

### `mid-app`

`HierarchyPlugin` (`docs/mid-app.md`), with two tests.

## Not built yet

- `mid-nodes` on top of the hierarchy and of `LocalTransform` composition (the
  composition is built: `docs/mid-ecs-transform.md`).
- Query-side helpers (`iter_descendants`, `iter_ancestors` on `Query`): the
  traversal is on `World` only.
- Inserting a child at an index, and reordering children.
- A C `Commands` counterpart.
- Custom relationships (Bevy's generic `Relationship` trait). Only parent/child.
- Enabling the hierarchy automatically in `App::new` (a deliberate omission, see
  the decisions).

## Verification status

Sandbox, rustc 1.97.1, one core. Real CI has not run this.

| | before | after |
|---|---|---|
| `mid-ecs` lib tests | 1,092 | 1,143 (41 hierarchy, 10 FFI) |
| with `--features scratch-arena` | not recorded | 1,148 |
| `mid-ecs` doctests | not recorded | 38 (two new) |
| `mid-app` tests | 44 | 46 |
| C smoke test checks | 287 | 299 |

- The C smoke test passes with `gcc -Wall -Wextra` (no warnings) against the shared
  and the static library, and under valgrind with no errors and no leaks.
- 25 hand-made mutants of `hierarchy.rs` and `commands.rs` (missing removal,
  dropped re-checks, wrong order, cycles allowed, wrong iteration order, swapped
  arguments, and so on): 24 caught. The one survivor was equivalent, a
  self-parent check made redundant by the cycle walk, and was deleted. Two other
  mutants survived the first version of the tests, which exposed two untested
  guards; each got a test.
- `cargo check` passes for `mid-ecs` and `mid-app` on `wasm32-unknown-unknown`
  and for `headless-server`. Clippy (`--no-deps`) reports the same 3,147 warning
  lines on this tree as on unmodified main, none in `hierarchy.rs`.
- The three benchmark crates were not rebuilt (they pull in Bevy); the change to
  `World` is one private field and additive methods.
- Not measured: the cost of a world with the hierarchy on, and the cost of
  despawning large subtrees.

## CI and Workflows

No workflow changed. `mid-ecs-test.yml` runs the new lib tests, and
`mid-ecs-ffi-test.yml` compiles and runs the C smoke test, so the new C checks run
there, under valgrind too.

## Fixes and Problems

### A mutation harness left a mutant behind
The first attempt to run the mutants in the background was killed when its shell
call ended, and left its first mutant applied to `hierarchy.rs`. A clean-baseline
check showed 11 unmutated failures, the leftover was removed, and the harness now
refuses to run unless the baseline passes. No mutation result from before that
check was used.

### The random test did not reach its interesting cases
The first version of the random-history test passed while reaching 1 refused cycle
and 17 real subtree despawns in 4,000 steps: despawns removed whole trees faster
than links built them. The generator was rebalanced (links common, despawns rare,
population topped up) and the test now fails if it does not reach a minimum of each
case.
