# mid-ui

## Overview

Standalone UI core for Mid Engine: a widget tree, anchor/margin layout with
row and column flow, content-driven sizes with minimum and maximum limits,
pointer hit testing, and a backend-agnostic draw-command list. Generic over `f32` and `f64`. It has no renderer, no input loop, no ECS
and no file format of its own; a host owns the tree, runs layout, asks whether the UI
claims the pointer, and hands the draw list to a backend.

Depends on `mid-math` only (`Color32`, and `Vec2`/`DVec2` conversions).

The `.mxui` file loader is a separate crate, `mid-ui-mxui` (see
`docs/mid-ui-mxui.md`), so this core never depends on DixScript.
A browser playground built on both lives in `web/mid-ui-wasm` and `web/ui`
(see `docs/mid-ui-wasm.md`).

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
re-exported from `mid-math`). Holds the one doctest. `solve` is a private
module; `measure` and `lazy` are public, and `ContentKey`, `Measure`,
`NoMeasure` and `LazyRows` are re-exported at the root.

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
`Sizing` (`Fixed`, `Fill(weight)`, `Fit`), and `Style` (anchor, sizes, margin,
padding, gap, flow, visible, interactive, clip, background, radius, and since
A1 `min_width`, `min_height`, `max_width`, `max_height` and `aspect`).

**Decisions:**
- Default style is `Fill` on both axes, visible, non-interactive, transparent,
  so a bare node occupies its parent and neither draws nor blocks input.
- In a row or column the anchor only acts on the cross axis.
- Layout and paint properties share one `Style` for now; the `.mxui` style
  blocks map onto it directly.
- `Fill` keeps the name and the meaning of a grow weight. `Fit` was added as a
  variant, so an exhaustive `match` on `Sizing` outside this crate would need a
  new arm; none exists in the workspace (the `.mxui` loader and the editor only
  construct values).
- The new limits default to no limit (`min` zero, `max` `None`), so a style
  written before A1, with `..Style::default()`, behaves as it did. A minimum
  wins over a maximum when they conflict, as in CSS.
- `aspect` is width over height and applies only where one side is `Fit` and
  the other is settled: `height: Fit` takes the width over the ratio, and
  `width: Fit` over a `Fixed` height takes the height times the ratio. Zero,
  negative and non-finite ratios are ignored. Both sides `Fit` uses the width
  from the content and derives the height.

**Tests:** unit tests in the file, and `tests/size_model.rs` for the new
fields.

### `tree.rs`

**What it does:** `Tree` arena with generational `NodeId`s: `add_child`,
`remove` (whole subtree), `parent`, `children`, `style`/`style_mut`, `rect`,
and since A1 `content`/`set_content`, the optional `ContentKey` a measure is
asked about for a leaf.

**Decisions:**
- Child order is paint order (later is on top).
- A removed slot's generation is bumped, so a stale id never resolves to a
  different node that reused the slot.
- The root is permanent and cannot be removed.
- Hidden nodes keep the rectangle from before they were hidden; layout skips
  them.
- The content key belongs to the node, not to its style, because it names what
  the host draws there and is not a layout rule. A stale id is refused by
  `set_content`, and a node in a reused slot starts without content.

**Tests:** `tests/tree.rs` (ordering, subtree removal, stale ids, slot reuse,
add/remove churn), and the content key in `tests/size_model.rs`.

### `layout.rs`

**What it does:** `compute(tree, viewport)` and, since A1,
`compute_with(tree, viewport, &measure)`. `compute` is `compute_with` with
`NoMeasure`, so every leaf's content has no size. The root is placed as the only
child of a stack the size of the viewport; every other node is placed inside its
parent's padded content area. `Stack` places each child by its own anchor;
`Row`/`Column` place children in order with `gap`, give `Fixed` children their
size, `Fit` children their content size, and share what is left among `Fill`
children by weight. The cross axis uses the anchor.

The file is now the driver only: it gathers the visible part of the tree
breadth first into a flat list, calls the pure solver in `solve.rs`, and writes
the rectangles back. Hidden nodes and their subtrees are not gathered and keep
their old rectangles. The measure is asked about a node only while it has no
visible children.

**Decisions:**
- Margins and gaps are subtracted before leftover space is shared.
- Zero, negative and NaN weights get no main-axis space (they keep their
  minimum, which is zero unless set).
- When fixed children overflow, `Fill` children get zero and the overflow is
  not clipped. Since A1, `Fit` children in an overflowing row or column give up
  room down to their floors (see `solve.rs`).
- The result for a tree that uses none of the A1 features is the one the old
  single pass gave, number for number: the 22 layout tests, the 16 hit and draw
  tests and the editor and `.mxui` suites pass unchanged, and the stack and
  cross-axis formulas keep their original order of operations because the
  last bit can differ otherwise.
- No dirty tracking yet; a full relayout is linear in node count.

**Tests:** `tests/layout.rs` (22 tests: anchor grid, margins, padding, row and
column flow, weights, hidden children, overflow, nesting, relayout, deep tree,
non-finite input) and `tests/size_model.rs`.

### `solve.rs`

**What it does:** the pure size solver. Input is a flat list of nodes in
breadth-first order (the visible children of node `i` are entries
`first[i] .. first[i] + count[i]`), the viewport and a `Measure`; output is one
`Rect` per entry. It follows the proposal model of the harness solver
(`docs/research-ui-harness.md`): a parent proposes room and a child picks a
size inside it, with a bottom-up pass that finds what each node wants and a
top-down pass that assigns sizes and positions, run for the width axis and then
the height axis.

- **Natural size and floor.** A leaf with content wants its `line_width` (its
  `min_content_width` is its floor) plus padding; a leaf without content wants
  its padding. A container wants the sum of its children's outer sizes on the
  main axis plus gaps, or the largest on the other axes, plus padding. A
  child's outer size is its margin plus its size held inside its own minimum
  and maximum.
- **Heights come after widths.** A text leaf's height is `wrapped_height` at the
  width it was assigned minus its padding, so one pass already gives every
  text the height of its own width. This replaces the explicit loop of the
  harness stack (measure at the last width, solve, repeat until no height
  moves), which exists there because its solver cannot call the typesetter; the
  `Measure` seam of `docs/mid-ui-contracts.md` lets the solver ask directly.
  The loop's stopping condition is tested instead: after one pass no text
  height moves when re-measured, and a second pass changes nothing.
- **Size rules.** `Fixed(v)` is `v`; `Fill` takes the available space (on a
  row's or column's main axis, its weighted share); `Fit` is
  `max(floor, min(natural, available))`. Every result is then held inside the
  node's minimum and maximum, the minimum winning.
- **Main axis.** Fixed and fit sizes, margins and gaps are taken off first. If
  something is left it is shared among the `Fill` children by weight; a child
  that would leave its minimum or maximum is fixed at that bound and the rest is
  shared again, until nobody is outside (each round fixes at least one child).
  If nothing is left, the `Fit` children shrink toward their floors in
  proportion to how much each can give, `Fixed` children never shrink, and
  `Fill` children get their minimum, which is zero by default. If even the
  floors do not fit, the row overflows.
- **Heights do not shrink** below their content (the height floor is the
  natural height), so a short column overflows instead of squeezing text.
- **Answers from a host are cleaned:** a measure result that is negative or not
  finite counts as zero.
- **No recursion.** Both passes are loops over the breadth-first list, so a
  20,000 level chain solves without touching the call stack (tested). Work
  arrays are reused per container.
- **Not built:** percent sizes, layout priority among shrinking children, grids,
  baseline alignment, wrapping rows, overlays out of flow, `ViewThatFits`. The
  harness solver has them; none was in A1.

**Tests:** `tests/size_model.rs`.

### `measure.rs`

**What it does:** the seam between the solver and whatever knows how big
content is. `ContentKey(u64)` is an opaque handle a host sets on a leaf.
`Measure<R>` has three methods: `line_width(key)`, `min_content_width(key)`
and `wrapped_height(key, max_width)`. `NoMeasure` answers zero to all three.

**Decisions:**
- The trait takes a key, not a string and a font. The text crate's
  four-method `TextMeasure` of `docs/mid-ui-contracts.md` (section 4) is
  adapted to this trait by a thin wrapper that maps a key to a text and a
  style; the baselines method is not needed until baseline alignment exists.
- The answers have the rules of the contract: pure, finite, not negative. The
  solver enforces the last two by counting anything else as zero.

**Tests:** `tests/size_model.rs`, against a fake measure with a fixed
character width and greedy wrapping, so the expected numbers are exact.

### `lazy.rs`

**What it does:** `LazyRows { count, row_height, gap }` for a long list of
uniform rows, such as the hierarchy panel: `extent()` (the height of the whole
list), `row_top(i)`, and `visible(scroll, view)`, the range of rows with any
part inside the window.

**Decisions:**
- It is a pure function of numbers, not a node in the tree: the host creates
  nodes only for the rows in the range, and the tree never holds the other tens
  of thousands. Attaching it to a scroller node comes with the widget kit (A6).
- The scroll position is clamped into the list, so overscroll shows the last
  rows; a negative or NaN scroll counts as the top. A row that ends exactly at
  the window's top edge, or a gap alone, does not count. A window with no height
  shows no rows. Unusable rows (none, zero, negative or non-finite height) show
  nothing.

**Tests:** `tests/lazy.rs`.

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

text (shaping, atlas, line breaking), images and
vector paths, pointer/keyboard event dispatch and focus, data binding by key,
animation (planned on `mid-anim`'s `Track`), flexbox wrap, percent sizes,
layout priority, baseline alignment, a lazy list attached to a scroller node,
a per-node user tag, a `mid_geom::Rect` conversion, and the optional `mid-ecs`
binding crate.

## Verification status

Sandbox only, rustc 1.97.1: 111 tests pass (12 unit, 98 integration, 1
doctest; 62 before A1), clippy with `--all-targets --no-deps` reports nothing
for mid-ui, `cargo check -p mid-ui --target wasm32-unknown-unknown` passes, and
the 12 earlier hand-made mutants (hit order, clip pruning, half-open edge,
weight share, cross-axis factor, leftover clamp, missing `PopClip`, draw order,
generation bump, hidden skip, gap count, stack anchor) were each caught by at
least one test. Those 12 were not re-run after A1 rewrote the layout pass, because the code they targeted in
`layout.rs` no longer has that shape (the hit, draw and tree code is unchanged). Instead six equivalents in the
rewritten solver (weight share, anchor on the cross axis, root anchor, gap count, margin before a child, padding
in the content origin) were run against only the 62 older tests and were each caught. For A1, 21 more hand-made mutants (a `Fit` that ignores its
floor or never shrinks, a maximum that beats a minimum, a main axis that takes the
maximum, dropped gaps, padding left out of the wrap width, an undivided shrink
share, a swapped violation sign, held space ignored, a multiplied and a divided
aspect ratio, a wrong column axis, uncleaned measure answers, a zero-weight
`Fill` that ignores its minimum, content that beats children, hidden nodes not
skipped, a stale id accepted by `set_content`, a row ending at the window edge
counted, overscroll not clamped, a missing round-up, a list extent without gaps)
were each caught by a failing test; one survived at first (the window-edge case
needs a gap between rows) and the test was strengthened until it was caught.
The downstream crates `mid-ui-mxui`, `mid-editor` and `mid-editor-ffi` build and
pass, except one `mid-ui-mxui` e2e test that already failed on `main` (see
`docs/mid-ui-mxui.md`). Not run: rustfmt, rustc 1.75 (see below), real CI, the phone.

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
