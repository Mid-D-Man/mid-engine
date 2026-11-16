# mid-ecs transform propagation

## Overview

`LocalTransform` (a pose relative to the parent) and the pass that composes it down the
hierarchy into `GlobalTransform` and `GlobalTransformLWC`: `World::propagate_transforms()`,
and `mid_app::TransformPlugin`, which runs it every frame in `PostUpdate`.

This is the piece `docs/mid-ecs.md` ("Large World Coordinates: GlobalTransform") left as
"not yet decided", and the one `mid-nodes` waits on. It builds on the hierarchy
(`docs/mid-ecs-hierarchy.md`) and on the two-component decision (`f32` `GlobalTransform` by
default, `f64` `GlobalTransformLWC` opt-in), which it does not change. Source:
`crates/mid-ecs/src/transform.rs`, and `TransformPlugin` in `crates/mid-app/src/app.rs`.

The model is Bevy's `Transform`/`GlobalTransform` pair, read from the `Mid-D-Man/bevy` fork
(`crates/bevy_transform/src/systems.rs`: `propagate_parent_transforms`,
`propagate_recursive`, `sync_simple_transforms`). The code is not a port: the two differ in
the ways listed under "Decisions".

## Decisions

- **`LocalTransform` is translation, rotation and scale, always `f32`.** The parts a game
  edits (turn something without taking a matrix apart), and what `docs/mid-ecs.md` already
  said about it: small in magnitude, `f32` whatever the world size. `to_affine()` gives
  scale, then rotation, then translation. Its size was not measured.
- **One serial pass, top down, from the roots, with an explicit stack.** A root is an
  entity with no `ChildOf` that has a `LocalTransform` or `Children`. A chain far deeper
  than the call stack is fine (a 5,000 deep one is tested). Each entity is composed once, as
  `parent * local`, so what it gets is the product of the locals above it, root first.
- **Four kinds of node, by what the entity has:**
  - *Local transform* (`LocalTransform`): composed, and the result is written to each
    global it has. Nothing is inserted for it: with neither global the result is carried to
    the children and not stored.
  - *Driven* (a global, no `LocalTransform`): left alone. Its global is the truth, set by
    hand or by physics, and its children compose onto that value. With both globals the
    `GlobalTransformLWC` is the one used.
  - *Grouping* (nothing): passes its parent's transform to its children unchanged, as if it
    were the identity.
  - A `LocalTransform` with no global is the first kind with nothing to write.
- **Precision is sticky downwards.** Composition is in `f32` until an entity with a
  `GlobalTransformLWC` (or one below it) is reached, and in `f64` from there on. An `f32`
  global below an `f64` one is written the truncated result, and what is below *it* still
  composes in `f64`, not from the truncated value. An `f64` entity below an `f32` parent
  promotes the parent (exact). The `f64` composition promotes the local pose with the
  local's own `f32` rounding already in it, which is right: a pose is small and `f32`.
- **A global is written only if its value changed.** The new value is compared with the
  stored one first, so `Changed<GlobalTransform>` and `Changed<GlobalTransformLWC>` pick out
  the entities that actually moved. Without the comparison every entity would be stamped
  every call and downstream change detection (culling, rendering) would see everything as
  changed.
- **Every call recomputes everything.** No dirty flags and no skipping of unchanged
  subtrees; the comparison above is what keeps the results honest, not what keeps the cost
  down. This is the obvious first thing to measure, and the performance work is paused, so
  nothing was measured here.
- **It enables the hierarchy.** It walks `Children`, which only a world with the hierarchy on
  maintains, so `propagate_transforms` calls `enable_hierarchy()` (idempotent), as the other
  `World` hierarchy methods do. `TransformPlugin` does the same when it is built. A
  `ChildOf` inserted by hand before the hierarchy was enabled has no `Children` behind it and
  is not reached (the hierarchy doc says the same).
- **No required components.** Bevy gives an entity with a `Transform` a `GlobalTransform`
  automatically. mid-ecs has no such mechanism, so the caller inserts both. An entity with a
  `LocalTransform` and no global gets no result stored, and says nothing about it.
- **It runs in `PostUpdate`, after `Physics`.** `Update` writes the local poses, `Physics`
  writes the driven globals, and `PostUpdate` composes both in the same frame. Physics runs
  on the fixed timestep, so in a frame with no step the driven globals are the previous
  step's.
- **`TransformPlugin` is a plugin, not part of `App::new`,** for the reason `HierarchyPlugin`
  is: it turns the hierarchy on. Adding it twice adds the system once (a marker resource).
- **Entities no root reaches are not visited, and nothing panics.** A loop of `ChildOf`
  made by raw inserts in a world that never enabled the hierarchy has no root. Bevy asserts
  on a malformed hierarchy; here such an entity keeps what it had.
- **No C counterpart.** The transform types have no FFI-span registration yet (see the
  module notes in `transform.rs`: it needs `zerocopy` on the `mid-math` types or a wrapper),
  so a C caller has nothing to compose. Both come together as one follow-up.

### The same as Bevy, and different

The same: `global = parent_global * local`; a root's global is its own local; the pass is
top down from roots; it belongs after the systems that move things and before the ones
that read the result (`PostUpdate`).

Different:
- A node with no transform of its own **does not break the chain.** In Bevy a child of an
  entity without a `Transform` or `GlobalTransform` is not visited (the queries require
  both), so its subtree stays stale. Here a grouping node is the identity and a driven node
  is its global.
- **No dirty tracking.** Bevy propagates a changed flag down, marks and skips unchanged
  trees (`mark_dirty_trees`), and parallelizes across roots. Here: full recompute, serial,
  compare before writing.
- **Two precisions** (`GlobalTransform` and `GlobalTransformLWC`), which Bevy does not have.
- **Entities need both parts inserted** (no required components), and a loop in the
  hierarchy is skipped, not an assertion failure.

## API

`mid_ecs::LocalTransform { translation: Vec3, rotation: Quat, scale: Vec3 }` with
`IDENTITY`, `new`, `from_translation`, `from_rotation`, `from_scale`, `to_affine`, and
`Default`. `World::propagate_transforms(&mut self)`. `mid_app::TransformPlugin`.

## Modules

### `transform.rs` (additions)

`LocalTransform`; a private `Frame` (nothing above, an `f32` affine, or an `f64` one) with
`to_f32` and `to_f64`; `promote` (an `Affine3` to a `DAffine3`, axis by axis); and on
`World`, `propagate_transforms` (finds the roots, runs the stack) and the private
`compose_one` (one entity: its kind, its frame, its writes). `DAffine3::as_affine3` from
`mid-math` does the truncation. Nothing in `mid-math` changed.

**Tests:** 28 new (41 in the file with the 13 that were there):
`LocalTransform` identity, default, constructors and the order of `to_affine`; a root's
global is its local; a child adds its translation, a parent's rotation turns the child's
offset, a parent's scale scales it and multiplies; a three deep chain equals the product of
its locals root first (exact); a second call changes nothing (`ChangeTracker`); only the
entities that moved are marked changed, for both globals; a node with no transform passes its
parent's on; a driven global is kept and its children follow it; with both globals and no
local the `LWC` one is the source; a local with no global writes nothing and still carries
its children; an `f64` child below a far parent keeps `.75` that an `f32` could not; an
`f32` global below it is the truncated result; below an `f32` node under an `f64` ancestor
the composition stays in `f64`; an `f64` child below an `f32` parent promotes it; both
globals written; an `LWC` global has the `f32` pose's axes, checked axis by axis without
going through `promote`; a 5,000 deep chain; a 1,000 wide fan-out; reparenting and
`remove_parent`; the hierarchy gets enabled, and an empty world is fine; separate trees;
an entity made before its parent; and ten seeds of random forests with every kind of node
(80 nodes each) compared with a model that composes in creation order. The random test
fails if it reaches too few of its cases; a run reaches 130 roots, 283 nodes at depth 4 or
more, 106 `f64` nodes below an `f64` parent and 112 `f32` nodes below one in 800 nodes,
and the minimums are 80, 150, 60 and 60. One doctest.

### `app.rs` (additions)

`TransformPlugin` (hierarchy on, one system in `PostUpdate` that calls
`World::propagate_transforms`, guarded by a marker resource). **Tests (3, in `tests.rs`):**
it composes locals down the hierarchy each frame, including after a local is edited; what
`Update` and `Physics` wrote in a frame is composed the same frame (a local edited in
`Update`, a driven global set in `Physics` on a fixed step); it turns the hierarchy on and
can be added twice.

## Not built yet

- **Dirty tracking and skipping unchanged subtrees**, and composing roots in parallel.
  Only worth doing after measuring what a full pass costs.
- **Inserting a `GlobalTransform` for an entity that has a `LocalTransform` and none**, and
  a one call "spawn with a transform".
- **A C surface and FFI-span registration** of `LocalTransform`, `GlobalTransform` and
  `GlobalTransformLWC`.
- **`mid-camera`** and its per-frame `to_view_relative` driver (`docs/mid-ecs.md`).
- Helpers: composing one entity's transform on demand, propagating a single subtree, and
  look-at constructors.
- **`mid-nodes`**, which this was written for.

## Verification status

Sandbox, rustc 1.97.1, one core, applied to a fresh clone of main at `2449e48` (which
carries the observer propagation change, `docs/mid-ecs-events.md`). Real CI has not run this.

| | main (`2449e48`) | after |
|---|---|---|
| `mid-ecs` lib tests | 1,165 | 1,193 (28 transform) |
| with `--features scratch-arena` | 1,170 | 1,198 |
| `mid-ecs` doctests | 39 | 40 |
| `ffi::` | 113 | 113 |
| `mid-app` tests | 45, plus one doctest | 48, plus one doctest |

- `cargo check` passes for `mid-ecs` and `mid-app` on `wasm32-unknown-unknown` and for
  `headless-server`; `rustfmt --check` is clean on the changed files; `cargo clippy --no-deps
  --all-targets` on `mid-ecs` and `mid-app` reports 3,147 warning lines, the same as before,
  none in the new code.
- **Mutation testing, 25 mutants of `transform.rs`,** a clean baseline checked before each
  batch and the file restored after each mutant: both product orders swapped; every local a
  root, every `Children` holder a root, either root source dropped; duplicate roots
  not removed; the `f64` stickiness off, and `f64` for an `LWC` entity off; a grouping node
  resetting to the identity; a driven node ignored (each global), or the wrong one
  preferred; children handed the parent's frame; each global always, or never, written; each
  write skipped; the hierarchy not enabled; the truncation dropping its value; `promote`
  swapping two axes, or dropping the translation; `to_affine` swapping scale and
  translation; the identity wrongly widened. 24 killed. The survivor, duplicate roots not
  removed, is equivalent: a root visited twice composes the same frames and writes the same
  values, so only the cost differs; the removal of duplicates stays as the guard against
  that cost and is not testable without a counter.
- Not verified: real CI; the cost of a pass (nothing was timed); the cost of the `PostUpdate`
  system in a frame where nothing moved; the wasm32 tests were type-checked, not run; the
  three bench crates were not rebuilt (nothing they use changed); the C surface does not
  exist.

## CI and Workflows

No workflow changed. `mid-ecs-test.yml` and `mid-app` tests run the new tests.

## Fixes and Problems

### Three mutants survived the first tests, each exposing a gap
- *Every `Children` holder as a root* survived because a parent usually has a lower index
  than its child, so the correct visit came last and covered the wrong one. A child made
  before its parent did not; it got a test.
- *The `LWC` global always rewritten* survived because the change-detection test watched only
  the `f32` global. It got an `LWC` twin.
- *`promote` with two axes swapped* survived because the random model calls `promote` too,
  so the same mistake sat on both sides. The axis-by-axis test above does not go through it.

### A `mid-app` test used the wrong clock
The test for `Physics` ran one frame of 1 ms, which does not make a whole fixed step, so
`Physics` did not run and the driven global was never set. The other tests in the file use
`one_step()` for that reason; this one does now.
