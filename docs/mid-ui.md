# mid-ui

## Overview

Standalone UI core for Mid Engine: a widget tree, anchor/margin layout with
row and column flow, pointer hit testing, and a backend-agnostic draw-command
list. Generic over `f32` and `f64`. It has no renderer, no input loop, no ECS
and no file format yet; a host owns the tree, runs layout, asks whether the UI
claims the pointer, and hands the draw list to a backend.

Depends on `mid-math` only (`Color32`, and `Vec2`/`DVec2` conversions).

## Decisions

- **Architecture: standalone core (option A).** The tree is mid-ui's own
  arena, not ECS entities. A `mid-ecs` binding, if wanted, is a separate
  optional crate on top. Reasons: usable from raylib, FFI, web and any other
  host with one tree and no sync step, and the wasm build never carries
  `mid-ecs`. Not measured against an ECS-backed layout; benchmark both before
  reversing this.
- **f32 and f64 through a `Real` trait, not separate `D` types.** `mid-math`
  splits precision into separate structs (`Vec2`/`DVec2`) with no shared scalar
  trait, so following that split would mean writing the layout engine twice.
  The core is written once against `Real` and tested at both precisions.
- **mid-ui owns its 2D UI geometry (`Point`, `Size`, `Rect`, `Insets`).**
  `mid_geom::Rect` is `f32`-only and `#[repr(C)]` for the FFI surface, so
  making it generic would break that contract. This also means `Insets` lives
  here and not in `mid-geom` as first proposed. Conversions exist for
  `Point` <-> `Vec2`/`DVec2`; a `Rect` conversion to `mid_geom::Rect` is not
  built (no `mid-geom` dependency yet).
- **Draw list is the only render seam.** `DrawCmd` is `#[non_exhaustive]`;
  text and image commands arrive with their own crates.
- **Explicit stacks everywhere.** Layout, hit testing and drawing never recurse,
  so a deep or hostile tree cannot overflow the stack (tested at 50,000 levels).
- **No panics on bad numbers.** NaN and infinity propagate through layout;
  negative sizes clamp to zero.
- **wasm32-clean now.** No `std::fs`, no clocks, no threads. `cargo check -p
  mid-ui --target wasm32-unknown-unknown` passes.

## Modules

### `lib.rs`

**What it does:** crate docs, module list, flat re-exports (`Color32` is
re-exported from `mid-math`). Holds the one doctest.

### `real.rs`

**What it does:** the `Real` trait (`ZERO`, `ONE`, `HALF`, conversions,
`min`/`max`/`non_neg`), implemented for `f32` and `f64`.

**Decisions:**
- `min`/`max` return `self` when `other` is NaN, so one bad input does not
  poison a comparison.

**Tests:** unit tests in the file; `tests/f64_parity.rs` runs one scenario at
both precisions (bit-identical for power-of-two weights, within 1e-3 for
non-dyadic ones) and checks an f64-only precision case at 2^24 + 1.

### `geom.rs`

**What it does:** `Point`, `Size`, `Insets`, `Rect` (origin plus extent),
`contains`, `intersect`, `inset`, and the `mid-math` conversions.

**Decisions:**
- `contains` is half-open (left/top inside, right/bottom outside), so adjacent
  rectangles never both claim a shared edge; a zero-size rectangle contains
  nothing.
- `inset` and `intersect` clamp to zero size instead of going negative.

**Tests:** unit tests in the file.

### `style.rs`

**What it does:** `Anchor` (3x3 grid), `Flow` (`Stack`, `Row`, `Column`),
`Sizing` (`Fixed`, `Fill(weight)`), and `Style` (anchor, sizes, margin,
padding, gap, flow, visible, interactive, clip, background, radius).

**Decisions:**
- Default style is `Fill` on both axes, visible, non-interactive, transparent,
  so a bare node occupies its parent and neither draws nor blocks input.
- In a row or column the anchor only acts on the cross axis.
- Layout and paint properties share one `Style` for now; the `.mxui` style
  blocks map onto it directly.

**Tests:** unit tests in the file.

### `tree.rs`

**What it does:** `Tree` arena with generational `NodeId`s: `add_child`,
`remove` (whole subtree), `parent`, `children`, `style`/`style_mut`, `rect`.

**Decisions:**
- Child order is paint order (later is on top).
- A removed slot's generation is bumped, so a stale id never resolves to a
  different node that reused the slot.
- The root is permanent and cannot be removed.
- Hidden nodes keep the rectangle from before they were hidden; layout skips
  them.

**Tests:** `tests/tree.rs` (ordering, subtree removal, stale ids, slot reuse,
add/remove churn).

### `layout.rs`

**What it does:** `compute(tree, viewport)`. The root is placed as the only
child of a stack the size of the viewport; every other node is placed inside
its parent's padded content area. `Stack` places each child by its own anchor;
`Row`/`Column` place children in order with `gap`, give `Fixed` children their
size, and share what is left among `Fill` children by weight. The cross axis
uses the anchor.

**Decisions:**
- Margins and gaps are subtracted before leftover space is shared.
- Zero, negative and NaN weights get no main-axis space.
- When fixed children overflow, `Fill` children get zero and the overflow is
  not clipped or shrunk.
- A single pass with a reused scratch buffer; no dirty tracking yet (full
  relayout is linear in node count).

**Tests:** `tests/layout.rs` (22 tests: anchor grid, margins, padding, row and
column flow, weights, hidden children, overflow, nesting, relayout, deep tree,
non-finite input).

### `hit.rs`

**What it does:** `hit_test(tree, point)` returns the topmost interactive node;
`claims_pointer` is the UI-versus-app seam (host checks it before routing a
pointer event to a game world).

**Decisions:**
- Topmost means last in paint order. Non-interactive nodes are transparent to
  the pointer. A clipping node removes its whole subtree from hit testing
  outside its own rectangle.
- Rectangular hit areas only; rounded corners are not hit-tested.

**Tests:** `tests/hit_draw.rs`.

### `draw.rs`

**What it does:** `DrawCmd` (`FillRect`, `PushClip`, `PopClip`) and
`build_draw_list`, which appends to a caller-owned buffer.

**Decisions:**
- Pre-order paint; a node's own fill comes before its children, and a
  clipping node's `PushClip` follows its own fill.
- Transparent and zero-size fills are skipped. `PushClip`/`PopClip` are always
  balanced; the backend intersects nested clips.

**Tests:** `tests/hit_draw.rs`.

## Not built yet

DixScript `.mxui` loader, text (shaping, atlas, line breaking), images and
vector paths, pointer/keyboard event dispatch and focus, data binding by key,
animation (planned on `mid-anim`'s `Track`), flexbox wrap and min/max sizes,
a per-node user tag, a `mid_geom::Rect` conversion, the optional `mid-ecs`
binding crate, and the wasm demo (`web/mid-ui-wasm` plus a playground page,
built the way msx's is).

## Verification status

Sandbox only, rustc 1.97.1: 62 tests pass (12 unit, 49 integration, 1
doctest), clippy reports nothing for mid-ui, `cargo check -p mid-ui --target
wasm32-unknown-unknown` passes, and 12 hand-made mutants (hit order, clip
pruning, half-open edge, weight share, cross-axis factor, leftover clamp, missing
`PopClip`, draw order, generation bump, hidden skip, gap count, stack anchor)
were each caught by at least one test. Not yet run on real CI.

## CI and Workflows

- `.github/workflows/mid-ui-test.yml`: `workflow_dispatch` only. Job
  `run-tests` runs fmt and clippy (both `continue-on-error`), builds, runs
  `cargo test -p mid-ui` and writes a pass/fail count to the step summary. Job
  `wasm32-check` runs `cargo check -p mid-ui --target wasm32-unknown-unknown`.
  Cache keys are scoped to mid-ui and the runner OS, and `~/.cargo/bin/` is not
  cached (see `docs/RUST_AND_CRATE_GUIDELINES.md` section 7).

## Fixes and Problems

### Verification environment

- `mid-math` does not compile on rustc 1.75 (`f32::from_bits` used in a
  `const` context, stable as const only in newer toolchains), so mid-ui cannot
  be checked on the 1.75 floor the rest of the workspace sometimes assumes.
  A bare `cargo build -p mid-ui` on 1.75 also fails earlier, resolving
  `edition2024` crates pulled in by bench members. Sandbox checks used rustc
  1.97.1 against the real workspace manifests.
