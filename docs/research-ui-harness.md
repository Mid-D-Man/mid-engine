# The harness UI stack: UI as entities, read in depth

## Overview

Sixth research document of the UI series, and the deeper read behind the addendum in
`docs/research-ui-comparison.md`. The maintainer answered the question "should `mid-ui` nodes stay
outside the ECS?" with the expectation that the ECS should perform better, and answered "should `.mxui`
have prefabs with inheritance?" with yes. The only UI stack read in this series that puts nodes in an
ECS is the harness stack of `indexable-inc/blaze`, so this document reads it for what it does, what it
costs and what it leaves unsaid about speed.

This is reading, not measuring. Evidence tags: **[src]** read in source or a manifest, **[doc]** stated
in a project's own documentation, **[gap]** searched for and not found. Reasoning is marked as such.

## Method

| Source | Commit | What was read |
|---|---|---|
| `indexable-inc/blaze`, `crates/harness/` | `2bbf1bc8`, 2026-11-25 | `harness-native-ui-layout` (`lib.rs`, `solve.rs` header and function list, `typeset.rs`), the heads of `harness-native-ui-input`, `-state-api`, `-motion`, `prims.flecs` (the prefab list and first prefabs), the crate descriptions and sizes |
| `andrewgazelka/den` | not readable | the clone failed with a credentials prompt, so the repository is private or moved [gap]; the design document that `solve.rs` cites, `docs/ui-model.md`, is not in the blaze repository either [gap] |

Not read: the renderer, shaders, text shaper, scroll-physics and timeline code, and the app crate. A
search of the harness manifests for a benchmark found none [gap].

## Findings

### One pass per root: gather, a pure solver, write back where changed

The layout module is "a single system" that runs on each root node, meaning a node whose parent is not
a node. It gathers the subtree into a flat list of `Item` values, calls the pure solver, and writes the
`Laid` component back only where the result differs; a root whose inputs are what they were at its last
solve is left alone [src, `lib.rs`]. Overlays and anchored nodes are laid out as roots of their own and
placed against a frame, with one frame of lag while the anchor moves [src]. The solver itself is
"pure over an index tree so it needs no world" [src, `solve.rs`]. The three files total 1,932 lines:
`solve.rs` 1,394, `lib.rs` 363 and `typeset.rs` 175 [src].

The consequence for the question asked: the layout arithmetic does not run on ECS storage at all. The
ECS supplies inputs and stores outputs, and the work between is on a contiguous array. The cost of the
ECS route is therefore the gather and the write-back, plus the change check that skips a clean root.
Whether that cost is small or large was not measured by anyone in what was read [gap]. Reasoning, not a
finding: this shape makes the solver reusable by a host with no ECS, which is how `mid-ui` is meant to be
usable everywhere.

### What the solver covers

The model is SwiftUI's: a parent proposes room and each child picks its size within it. Per axis it
runs `natural` bottom-up (content sizes), then `assign` top-down (Fixed, Percent, Fit and Grow by weight
with a minimum and a maximum, shrinking down to the floor, lowest `LayoutPriority` first), width before
height, then `place` [src, `solve.rs` header]. On top of flex it has: backgrounds and overlays out of the
flow, grids with columns sized by their widest cell, adaptive columns, lazy rows that say which range
is in view, first and last baseline alignment, `ViewThatFits`, safe-area insets, aspect ratio, and
scroll extents with overlay scrollers [src]. Text is the one thing it cannot do: leaves take their size
from an `Intrinsic` component that the host's typesetter writes [src].

The text loop is explicit. Widths do not depend on heights (except through `Aspect`), but a text's height
depends on the width it is given, so the host measures each text at the width it was last laid out at,
and the frame alternates typesetting and layout until no height moves [src, `solve.rs`, `typeset.rs`].
The typesetter's side is a trait with three methods: the width of a string on one line, the height of a
string wrapped at a width, and the first and last baselines [src]. That is a worked example of the
`TextMeasure` seam of Decision 11.

Against `mid-ui` today (anchors, row and column flow, `Fill` weights, no min, max, intrinsic sizes or
wrap [doc]), this is the missing half: the model that text-driven and content-driven sizing needs.

### Input: a collect system, a step system, state on the app entity

The input pass gathers every hoverable node with its laid box and draw key, then once a frame resolves
the topmost hit by layer, `Z`, depth and creation order, skipping nodes that are on their way out
[src, `harness-native-ui-input`]. The result is written as relations on the app entity: `(Hovered,
node)`, `(Pressed, node)` and `(Focused, node)`; letting go over the same pressable tags it `Activated`
so that feature systems can observe the action [src]. The pointer shape is resolved in the same step,
from the hit's cursor or its nearest ancestor's [src].

Reasoning, not a finding: holding hover, press and focus as pairs on one entity means a hover change
touches one entity, not two nodes' archetypes. In `mid-ecs`, adding or removing a component on a node
moves it between tables, and structural changes are the cost the ECS work has been tuning [doc,
`docs/research-ecs-comparison.md`]. Toggle, selection and open state are different: they are relations on
the node itself, `(On, Set)`, `(Selected, Set)`, `(Open, Set)`, each exclusive [src, `-state-api`].

### Prefabs with inheritance are the vocabulary

The primitives are prefabs in `prims.flecs`, about 45 of them, each built from others: `Row`, `Column`,
`ZStack`, `Spacer`, `Box` (a `Column` with a `Plate`), `Words`, `Line`, `Lead`, `Icon`, `Button` (a `Row`
with a `Wash`), `ButtonPrimary`, `ButtonQuiet`, `IconButton`, `Toggle` (a `Button`), `Field` (a `Box`),
`Scroll`, `List`, `Overlay`, `Divider`, `Bubble`, `Label`, `Grid`, `GridRow`, `AdaptiveGrid`,
`LazyColumn`, `LazyGrid`, `LazyList`, `ViewThatFits`, `TextArea` (a `Scroll`), and a family of table
prefabs from `TableHeadRow` to `TableRecord` [src]. An instance states only what differs, for example
`title : ui.Words { Text: {s: "Hello"} }`. A prefab's properties are small components: `Flex` with an
axis, gap and alignment, `Size` with a kind per axis (Fit, Grow, Fixed, Percent) and a value, minimum
and maximum, `Padding`, looks such as `Fill`, `Edge`, `Shape`, `Lift` and `Frost`, and interaction marks
such as `Pressable`, `Focusable`, `Editable` and `Cursor` [src, prefab comments].

This is the closest example read to what the maintainer described for `.mxui`: the HTML and CSS of the
UI, with prefabs playing the part of classes and instances stating only their differences. Decision 17
records the yes to prefabs with inheritance.

### Motion and scripting

Animation is generic: visibility springs, transitions, matched geometry and implicit animation of layout
inputs, with all persistent state as components on the nodes, and the spring tuning taken from the node's
own `Motion` or from a token prefab [src, `-motion`]. Timelines apply cues to members of nodes through
reflection [doc, crate description]. `mid-anim` and the reflect bridge are the nearest existing pieces in
this project.

### What the stack does not offer

- **Portability.** It is a macOS app (AppKit, Metal) built only with blaze, and the world is a vendored
  C fork of Flecs [doc]. The ECS typed store builds for wasm32 [doc], but nothing in the UI crates
  targets Android or 32-bit ARM [gap].
- **Speed numbers.** None found [gap]. The design shows where work is done; it does not show how long it
  takes.
- **An editor-grade widget set.** Text areas, tables, lists, lazy lists and scrollers exist, but this was
  not read as a widget kit for an inspector with numeric drag fields and a hierarchy tree [gap].

## What this means for the node-model question

Three shapes are now visible.

| Shape | What it is | Shown by | Cost named by the reading |
|---|---|---|---|
| Standalone tree | `mid-ui`'s arena `Tree` with generational ids | `mid-ui` today | No ECS integration; reflection, queries and change detection must be rebuilt for UI |
| Nodes as entities, solver pure | Gather a root's subtree, solve on an array, write back where changed | the harness stack | Gather and write-back per root; structural changes when state is components on nodes |
| Pure solver with an ECS adapter | The solver and the tree live in `mid-ui`; a binding crate maps components to nodes and back | reasoning from the above | A mapping layer to keep in step |

The maintainer's expectation was that the ECS should perform better. What the reading supports is
narrower. Iteration over many entities with the same components is the case ECS storage is built for, but
the project's own CI currently records `mid-ecs` behind `bevy_ecs` on dense iteration: the ratio is about
4x on older x86_64 runners, 6.0x on the AMD EPYC 9V45, 1.25x on aarch64 (Neoverse-N2), and 0.94x on Apple
M1, the one case where it is ahead; the recorded cause is the compiler declining to inline, and the
performance work is parked until the ECS is feature complete [doc, `docs/research-ecs-comparison.md`]. That
is a gap against another ECS. The baseline that matters for a layout pass is a flat array of items, which
the harness solver uses and which no ECS iteration can beat, so an ECS node model starts at or behind
that baseline, not ahead of it. The ECS gains that remain are the ones the harness stack actually uses:
change detection to skip clean roots, relations for state, prefabs, reflection for inspection and
scripting, and systems that can run in parallel. Those are integration and scheduling gains, not faster
arithmetic. Whether they outweigh the gather, the write-back and the current dense-iteration gap for the
node counts an editor has is unmeasured here, so the roadmap puts a benchmark ahead of the decision
(`docs/mid-ui-roadmap.md`, D1) and builds the solver so that either answer stays open.

## Gaps, ranked

1. **The solver's size model** (intrinsic, min, max, wrap, text-driven sizes) in `mid-ui`; it does not need
   the ECS and serves every route.
2. **A benchmark of the three shapes** on the same screen, on the phone and on CI.
3. **Component vocabulary and prefab syntax** for `.mxui`, with inheritance.
4. **State as relations on one entity** for hover, press and focus, which depends on `mid-ecs` pairs.
   `mid-ecs` has `ChildOf` and `Children`; the other relations, names and paths, and templates are in the
   gaps of the ECS comparison.
5. **A text measure trait** with the three methods above and a rule on hinted advances.
6. **Lazy rows** in the solver for the hierarchy panel.

## Questions for the maintainer

1. Is the benchmark of the three shapes accepted as the way to settle the node model, with the pure
   solver built first so that no route is closed?
2. Should hover, press and focus live as relations on one entity (needs relations in `mid-ecs`), or as
   fields of a UI state resource?
3. Should the primitive vocabulary start from this stack's list (rows, columns, boxes, words, buttons,
   fields, scrolls, lists, tables) or from the explorer screenshot's needs alone?

## Sources

- blaze: `crates/harness/harness-native-ui-layout/src/{lib,solve,typeset}.rs`,
  `crates/harness/harness-native-ui-input/src/lib.rs`, `crates/harness/harness-native-ui-state-api/src/lib.rs`,
  `crates/harness/harness-native-ui-motion/src/lib.rs`, `prims.flecs`, the crate manifests.
- mid-engine: `docs/mid-ui.md`, `docs/research-ui-comparison.md`, `docs/research-ecs-comparison.md`.
