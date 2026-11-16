# mid-nodes

**Status: design only. Nothing is built, and nothing here has been verified.** Every decision
below is a proposal written on 2026-11-16 for the maintainer to confirm or change. No crate
exists yet.

## Overview

`mid-nodes` is the "scene graph / node system" named last in the build order since the first
architecture doc, with no definition beyond that. This document gives it one.

Proposed scope: **a scene and authoring layer on top of `mid-ecs`, where a node is an entity.**
It adds what an ECS leaves out and a scene needs: names and path lookup, standard bundles for
the common kinds of node, a description of a tree of nodes that can be instantiated into a
world and saved from one, and later instancing with overrides. It does not draw, simulate,
play animation or lay out UI.

## What the repo already says about it

- `docs/architecture.md`: "Scene graph / node system", last in the build order, not started.
- `crates/mid-anim/src/lib.rs`: sampled tracks are to be applied to a joint or scene node
  "once `mid-nodes` exists". That reasoning is now out of date (see "Corrections made with this
  document").
- `docs/mid-collections.md`: the intrusive lists are "relevant once a job system or scene graph
  exists", with `mid-nodes` named as a possible consumer.
- `docs/roadmap.md`: suggests a structural study of Blender's module layout for scene-graph
  design, as its own follow-up. Not done.
- `docs/mid-ecs.md`, "The Ubel Stratum Bridge": a design vision, explicitly not decided, in
  which gameplay code sees `Actor`-like objects lowered to ECS accesses. If that ever lands it
  would sit above or beside this layer; it is not a requirement here.

## What already exists to build on

Verified against current main (`5dc9525`):

- **The hierarchy** (`docs/mid-ecs-hierarchy.md`): `ChildOf` and `Children`, `set_parent`,
  `add_children`, recursive despawn of a subtree as one command, cycle refusal, `Commands`
  with `with_children`, and a C API.
- **Transforms** (`docs/mid-ecs-transform.md`): `LocalTransform`, `GlobalTransform`,
  `GlobalTransformLWC`, and `World::propagate_transforms`, run each frame by
  `mid_app::TransformPlugin`.
- **Events that bubble up the hierarchy** (`docs/mid-ecs-events.md`): an event with
  `PROPAGATE` set climbs from a node to its ancestors, so node-level events need nothing new.
- **Bundles:** tuples of up to 8 Archetype Core components spawn atomically with
  `World::spawn_bundle`, so a "node bundle" can be an ordinary tuple today. There is no
  required-components mechanism, so nothing is added automatically; a constructor does it.
- **Component hooks, observers, change detection, `Commands`:** all built.

## What it needs that does not exist

- **A description of a component** (its name and fields) and a way to read and write one as a
  value: reflection (`docs/roadmap.md`, Decision 7). Without it a scene can hold only the
  component types this crate knows by name, which is not a scene format.
- **A file format and a loader.** DixScript is the natural candidate (`mid-ui-mxui` already
  loads UI trees from it), but the core of this crate must not depend on it.
- **Names.** No `Name` component exists.

## Decisions (all proposed)

### N1. A node is an entity; there is no separate node object

Nodes are entities with a standard set of components. `mid-nodes` adds no object model, no
virtual methods and no second tree. **Why:** the engine is ECS-only by mandate, the hierarchy
and transforms already live there, and a parallel node tree would have to be kept in step with
it (the exact problem `mid-ui`'s separate arena has, accepted there because `mid-ui` must work
without an ECS). **Alternative:** Godot-style node objects with behaviour. Rejected: it
reintroduces the object model the ECS replaced.

### N2. A separate crate, depending on `mid-ecs`

Not inside `mid-ecs`, which stays generic, and not folded into `mid-app`. Same precedent as
`mid-app` and `mid-time`. Reflection-dependent parts (N2 onward in "Phases") depend on
`mid-reflect` too. A `NodesPlugin` for `mid-app` registers what the crate needs.

### N3. `Name` is a component, and names are not required to be unique

`Name(String)`. Path lookup (`"Player/Arm/Gun"`) walks `Children` in order and returns the first
match at each level. The ECS does not enforce unique sibling names, because enforcement would
need a hook on every rename and every reparent, and a lookup that silently picks the first of
two equal names is a known and documented behaviour. A lint ("no duplicate sibling names in
this scene") can live in the scene tooling instead. **Needs your call:** Godot forbids
duplicate sibling names; do you want that?

### N4. Node kinds are bundles, not types

`Node3D` is a tuple bundle of `Name`, `LocalTransform` and `GlobalTransform`; a far-world
variant uses `GlobalTransformLWC`. Later kinds (a camera node, a mesh node) are bundles plus
the components their own crates add. A "kind" is a name for a recognisable set of components,
nothing more. **Not decided:** 2D nodes. `GlobalTransform` is a 3D affine; a 2D transform type
is deferred until a 2D renderer says what it needs. Until then a 2D node is a 3D node with
`z = 0`.

### N5. A scene is data: a tree of node descriptions

Illustrative shape, not a committed API:

```
SceneNode { name, components: Vec<(TypeName, Value)>, children: Vec<SceneNode>, instance_of: Option<SceneRef> }
```

`World::instantiate(&scene, parent) -> Entity` creates the entities in depth-first order,
sets components through the reflection registry, and returns the root. Despawning the root
removes the instance (recursive despawn already does that). The inverse, saving a live subtree
as a scene, falls out of reflection plus the hierarchy.

### N6. The scene file loader is a separate optional crate

Following the `mid-ui` / `mid-ui-mxui` split: the core takes a source-neutral value tree, and
a DixScript adapter crate turns a `.mdix` scene into one. The core never depends on DixScript,
which also keeps it testable in the sandbox (DixScript-Rust needs Rust 1.80 and is only
type-checked there, never run).

### N7. Instancing with overrides comes last

A scene that contains another scene, with per-instance overrides of component values, is the
prefab feature. It needs N5 working first and a decision about how overrides are addressed
(by node path and component name is the proposal). It is a phase of its own.

### N8. `mid-ui` stays separate

`mid-ui`'s tree is its own arena by design (`docs/mid-ui.md`). Widgets are not nodes. An
optional binding crate (the "`mid-ecs` binding" that doc already lists as not built) could
mirror a UI tree into nodes later. **Needs your call:** if you want UI screens to be scenes
(instantiated, saved, edited like any other), that is a different and larger decision than
this document makes.

## Phases

| Phase | What | Needs | Done when |
|---|---|---|---|
| N1 | `Name`, node bundles, `find_by_path`, `NodesPlugin` | hierarchy and transform (built) | Tests pass in the sandbox; the bundles spawn and compose through `TransformPlugin` |
| N2 | Scene tree data and `instantiate`, plus the inverse | `mid-reflect` | A subtree round-trips to a scene and back with equal component values |
| N3 | Scene file loader (DixScript, optional crate) | N2 | A fixture `.mdix` scene loads and instantiates |
| N4 | Instancing with overrides | N3 | A scene instanced twice with different overrides |
| N5 | Groups (a marker for "all enemies"), anything else a real scene needs | use | Driven by a real scene, not guessed |

Phase 1 is small and has no unmet dependency, so it can be built before reflection.

## Not built, on purpose

- Rendering, mesh or sprite nodes (no renderer).
- Animation playback (that is `mid-anim`'s consumer side; applying a sampled `Track` to a
  `LocalTransform` is a thin system and does not need this crate).
- Physics bodies as nodes (`mid-physics` is not wired to the ECS).
- A C API (comes with each phase, scoped per feature, per the working rule).
- UI nodes (N8).

## Open questions for the maintainer

1. Confirm N1 (nodes are entities), or say you want something else.
2. Names unique among siblings or not (N3).
3. 2D nodes: wait for a 2D renderer (proposed), or define a 2D transform now (N4).
4. Should UI screens ever be scenes (N8)?
5. Scene format: DixScript for scene files (proposed), or something smaller for the first
   version?
6. Is the Blender structural study (named in the roadmap) wanted before N2?

## Corrections made with this document

- `crates/mid-anim/src/lib.rs` said that applying a track to a node waits on hierarchy
  composition being `mid-nodes`'s job. The hierarchy and the composition now exist in
  `mid-ecs`. The comment now says wiring a track to a `LocalTransform` no longer waits on
  `mid-nodes`.
- `docs/roadmap.md` and `docs/architecture.md`: the build order now puts `nodes` after the
  tools and UI tracks, not last behind `physics` and `anim`.

## Verification status

None. This is a design document. No code, no tests, no CI.
