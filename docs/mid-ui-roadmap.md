# mid-ui: roadmap from the research to a UI the editor can use

Status: **provisional**. Written 2026-11-25 as the plan that sits on top of the research documents:
`docs/research-ui-comparison.md`, `docs/research-ui-text.md`, `docs/research-ui-hinting.md`,
`docs/research-ui-raster.md`, `docs/research-ui-harness.md` and, for the ECS side,
`docs/research-ecs-comparison.md`. Statements about libraries keep the tags of those documents: **[src]**
read in source, **[doc]** stated in documentation, **[gap]** searched for and not found, **[run]** measured
in the research. *Reasoned* means it follows from those but was not tested. *Decided* means the maintainer
chose it (Decisions 14 to 17 in `docs/roadmap.md`). *Leaning* and *open* mark what is not settled.

The plan has two tracks and one side track. **Track B** is the design phase: contracts and decisions that
define the data every later crate consumes, so they come first. **Track A** is engineering in slices, each
needing named earlier slices and each ending in something that can be checked. **Track P** runs beside
them and blocks nothing until a slice says it does.

## Where things stand

**See `docs/mid-ui-status.md`, which is kept current**: what exists, what is missing, the path from here to
an editor on its own UI, and the next five actions. In short, as of 2026-11-25: `mid-ui` is a layout, hit
test and draw-list core with no text, events or widgets; the editor draws through its own paint list and
a C host (raylib); the research is finished and Decisions 14 to 17 are recorded; no UI crate beyond those
exists yet.

## What is decided

| Decision | What | Recorded |
|---|---|---|
| 14 | `mid-ui` is painted by backend crates in this repository, using `mid-math`; CPU backend first in order of work, GPU first in a shipped build; two draw vocabularies stay; text is read from the candidates and written here; `.mxui` grows expressions and bindings, as HTML and CSS are for a page; soft keyboard by a helper and an in-UI keypad; an immediate-mode authoring layer on top of the retained tree | `docs/roadmap.md` |
| 15 | Dev-dependency oracles allowed in tests; hinting wanted; bundled and system fonts; `mid-math` scalar on 32-bit ARM for now and an armv7 backend later on nightly; `.mxui` syntax worked on as the rest goes; FFI and wasm testing for every new UI crate | `docs/roadmap.md` |
| 16 | Text is Latin first and built for every script | `docs/roadmap.md` |
| 17 | `.mxui` has prefabs with inheritance; the node model leans toward ECS and is settled by a benchmark | `docs/roadmap.md` |

## The rules that fix the order

1. **Data shapes before the code that uses them.** The pixel format, the paint list, the text types and the
   measure contract are consumed by the rasterizer, the text crate, the layout and the backends. They are
   designed first (B1), small, and kept small.
2. **Pure cores before adapters, adapters before hosts.** The solver, the rasterizer and the text crate are
   pure Rust with no I/O. ECS binding, FFI and wasm exports are thin layers over them.
3. **CPU before GPU** (Decision 14). The CPU output is the reference the GPU output is compared with.
4. **Unhinted before hinted, Latin before the rest,** with the text types carrying direction, script and
   clusters from the first line (Decisions 15 and 16).
5. **Every slice is done when** it has: tests through the public interface, hand-made mutants that the tests
   catch, a C API shaped like `mid-editor-ffi` where it has a user outside Rust, a wasm32 check and a
   test job like `mid-ui-wasm-test.yml`, and a docs section per source file under the docs lint. This is
   Decision 15 made into a checklist.
6. **The phone and real CI are the authority.** A sandbox x86_64 number says nothing about a 32-bit phone.
7. **The solver is built so that the node-model question stays open** (D1): the same pure solver serves a
   standalone tree, an ECS adapter and nodes as entities.

---

## Track B: the design phase

Suggested order: **B1 first**, because every slice in Track A consumes it. **B2 runs alongside A1** and ends
in a benchmark, not a debate. **B3 starts when A1 has the size model**, because the vocabulary of `.mxui`
is the vocabulary of the solver. **B4 before A5.** **B5 last**, because it waits on ECS decisions that are
already open.

### B1. The contracts

Four small written contracts, each a page, each with the types spelled out.

- **Pixel format and color.** Premultiplied `RGBA8` is the default proposal; how `Color32` and its color
  spaces map to it (D4). Output integer-exact on every architecture, so golden images work (D14).
- **The paint list.** The editor's `PaintItem` (five kinds, 36 bytes, `repr(C)`) and `mid-ui`'s `DrawCmd`
  stay two vocabularies (Decision 14); B1 states how one converts to the other, which one the CPU backend
  consumes first (D3), and adds what is missing: a glyph run command with positioned glyph ids, a
  custom-content slot for the viewport, a clip kind beyond rectangles if wanted.
- **Text types.** Glyph ids and clusters, a direction and a script on every run, no ASCII or
  one-glyph-per-character assumption (Decision 16).
- **The measure contract.** A trait of three methods as in the harness stack, the width of a string on one
  line, the height wrapped at a width, the first and last baselines [src, `docs/research-ui-harness.md`], and
  the rule on hinted against unhinted advances: the measure returns what the renderer draws (D9).

*Deliverable:* `docs/mid-ui-contracts.md`, reviewed by the maintainer. *Blocks:* A1 (the `Intrinsic` shape
only), A2, A3, A4.

*Status:* written and answered on 2026-11-26. The measure has four methods, not three (the minimum content
width was added); the blend space is an option; `mid-math` gained rounded color helpers for it.

### B2. The node model, settled by a benchmark

Three shapes: a standalone tree, nodes as entities with a pure solver, a pure solver with an ECS adapter
(`docs/research-ui-harness.md`). Current standing: *leaning* toward ECS (maintainer, 2026-11-25), pending
measurement. The reading says the layout arithmetic is on an array in every shape, and the ECS cost is the
gather, the write-back and, today, the dense-iteration gap against `bevy_ecs` on most CI runners.

*Deliverable:* a benchmark crate that lays out the same editor screen in the three shapes, on x86_64 CI and
on the phone, with node counts of 200, 1,000 and 5,000, recording the CPU line in the header (the pattern of
`benches/ecs-vs-bevy-ecs`). *Blocks:* A8 only. A1 does not wait for it.

### B3. `.mxui` v1

The HTML and CSS of the UI for the engine and for anything that uses `mid-ui` (Decision 14). To settle, in
order: the component vocabulary (the solver's inputs: size per axis with kind, value, min and max; flex;
padding; aspect; looks; interaction marks); prefab inheritance, decided yes (Decision 17); expressions and
bindings, their scope and where the evaluator lives outside the core crate (D13); the relation to DixScript
2.0, which targets Rust 1.75 or lower; error paths kept as `mid-ui-mxui` has them today. The primitive list
starts from the explorer screenshot's needs and is checked against the harness stack's vocabulary (D12).

*Deliverable:* `docs/mid-ui-mxui-v1.md`, a grammar and worked examples. *Blocks:* A6, A7.

### B4. Input

Pointer and touch events, hit order, focus, capture, scrolling with touch inertia, text input through the
keypad and the soft keyboard helper (Java first, D11), the composition range for an OS keyboard. The
references are Parley's editor for composition [src] and the harness input pass for hit order and state [src].
Hover, press and focus as relations on one entity or as fields of a UI state resource is open (D8).

*Deliverable:* `docs/mid-ui-input.md`. *Blocks:* A5, A12.

### B5. The ECS contract

How the UI reads and writes the world: reflection-driven rows (as the editor does today), names and paths,
run-time queries, statistics, the remote protocol. This waits on the six questions at the end of
`docs/research-ecs-comparison.md`, which are open (D10), and can use `blaze-ecs` as a further reference
(`docs/research-ecs-comparison.md`, addendum).

*Deliverable:* `docs/mid-ui-ecs.md`. *Blocks:* A8.

---

## Track A: engineering slices

Each slice lists what it needs, what it gives and when it is done beyond the checklist in rule 5.

### A1. The solver's size model (roadmap U1, part of U3)

Needs the `Intrinsic` shape from B1. Adds to `mid-ui`: intrinsic sizes for leaves, minimum and maximum,
`Fit`, `Grow` and `Fixed` per axis, wrap driven by a width, aspect ratio, and lazy row ranges for the
hierarchy panel, following the proposal model of the harness solver [src]. Stays pure, generic over `f32`
and `f64`, without recursion, and `wasm32`-clean, as today. *Done when* the text loop (measure at the last
width, solve, repeat until no height moves) is tested with a fake measure, and the existing 62 tests and 12
mutants still hold.

*Status:* built on 2026-11-26. Two differences from the wording above. The text loop is not needed as a loop:
the solver asks the `Measure` of `docs/mid-ui-contracts.md` directly, widths before heights, so one pass already
ends where the loop would; the loop's stopping condition is what the tests check. `Percent` sizes and layout
priority were left out, and the lazy row ranges are a pure function (`LazyRows`), not yet a node in the tree.
The 62 earlier tests pass unchanged, and six mutants of the rewritten layout code are caught by those
older tests alone; there are now 111 tests and 21 more mutants, all caught.

### A2. The coverage rasterizer

Needs the pixel format from B1. A new pure crate depending on `mid-math` only: closed-form coverage for
rectangles, rounded rectangles and thick lines, a path coverage rasterizer for glyph outlines, an axis-aligned
clip stack, blending into premultiplied `RGBA8`, scalar first (Decision 15). *Done when* its output matches
`tiny-skia` within a stated tolerance on a fixed shape set (the oracle is a dev-dependency, Decision 15) and
is integer-exact across x86_64 and wasm32.

### A3. The CPU backend, first pixels (backend crates are new, Decision 14)

Needs A2 and the paint list from B1. Draws the editor's paint list into a framebuffer, with a C API shaped
like `mid-editor-ffi` and a wasm export writing into `ImageData`. *Done when* the editor's self-test scenes
render to golden images in CI and in the wasm test, and the 1,000 and 5,000 cube scenes are timed against the
raylib host on the phone through the Console (the probe of `docs/research-ui-raster.md`).

### A4. Text, tier T1 (roadmap U1, built properly)

Needs A2 and the text types from B1. A font reader for `cmap`, `head`, `hhea`, `hmtx`, `maxp`, `loca`, `glyf`
with composites and `kern`; outlines; glyph masks through A2; a glyph cache; the measure of B1 implemented over it.
Oracles in tests: Skrifa for outlines and advances, fontdue and zeno for coverage (Decision 15). *Done when*
the bundled DejaVu renders printable ASCII within tolerance of the oracles, the measure and the drawing agree to
the pixel, and the Latin glyph probe of `docs/research-ui-text.md` has run on x86_64 CI and the phone.

### A5. Events, focus, scrolling, touch (roadmap U2)

Needs B4 and A1. Hit order, focus, capture, scroll physics with touch inertia, the keypad as an in-UI widget.
*Done when* the editor's panels scroll and take numeric input by touch on the phone.

### A6. The widget kit v1 (roadmap U3)

Needs A1, A4, A5, B3. The eight pieces the explorer screenshot needs: scrolling with row virtualization, a text
field, a numeric drag field, a fold, a button, a tab or rail item, a popup list, a splitter
(`docs/research-ui-comparison.md`). *Done when* a screen with the explorer's layout runs on the CPU backend.

### A7. The `.mxui` v1 loader (roadmap U4, first half)

Needs B3 and A1. Prefabs with inheritance, expressions, bindings by key. The `mid-ui-mxui` schema reader and its
DixScript adapter are the starting point; its source-neutral split stays so that the DixScript 2.0 move touches one
module. *Done when* the explorer screen of A6 is written in `.mxui` and its error paths are tested.

### A8. The ECS connection

Needs B2's result and B5. Whichever node model B2 picks, the reflection-driven inspector rows and the hierarchy come
from the world, with change detection driving invalidation. *Done when* the editor's inspector and hierarchy are
drawn by the new stack from a live `mid-ecs` world.

### A9. The editor on its own stack (roadmap U4 and E1's enabler)

Needs A3, A6, A7, A8. The editor stops converting to a host paint list for its panels and draws them through the
backend. *Done when* the Android APK runs the editor panels on the CPU backend and the Console reports frame times
no worse than the raylib host at the same scenes, or the difference is explained.

### A10. Hinting, then system fonts, then the soft keyboard helper

Needs A4 (hinting, system fonts) and A5 (helper). Hinting: H1, a light vertical-only hinter for Latin, after the
unhinted pipeline is correct; the interpreter and PostScript hints later (D5). System fonts: a fallback chain and
Android discovery, referenced on Fontique's Android backend [src]. The helper: one small class compiled by the
build script and loaded from bytes in memory, the route Slint takes, which leaves `android:hasCode="false"`
alone [src, not tried here].

### A11. The GPU backend

Needs A3 as the reference and the custom-content slot of B1. `wgpu` with Vulkan is known to run on the phone
through the canvas app (maintainer's report), so the GPU path is viable; frame cost is unmeasured. Priority in a
shipped build, order of work after the CPU backend (Decision 14). *Done when* the GPU output matches the CPU
reference within a stated tolerance on the golden scenes, and the 3D viewport shares the pass.

### A12. Later, in no fixed order

Text tiers T2 to T4 (ligatures and marks, bidi and joining, Indic and CJK), images, gradients and shadows,
the accessibility tree, animation on `mid-anim` tracks, UI over the remote protocol, an armv7 SIMD path for the
rasterizer.

---

## Track P: beside the others

- **P1. `mid-math` on 32-bit ARM.** Scalar now. A hand-written armv7 backend later, on nightly, with `vpadd` where
  the aarch64 code uses `vaddv`, and estimates with Newton steps where it uses `vdiv` and `vsqrt`
  (`docs/research-ui-text.md`). Blocks nothing; helps A2 and A4 on the phone.
- **P2. The ECS gaps the UI needs,** in the order of `docs/research-ecs-comparison.md`: run-time queries by names,
  statistics, the remote protocol, names and paths, then relations beyond `ChildOf`. Blocks B5 and A8.
- **P3. The editor APK with the world-backed editor on the phone.** Never run yet; the 32-bit ARM link with
  `mid-ecs` and `rayon` is the biggest unknown and gives the phone baseline numbers that A3 and A9 are compared
  with. Do first, because it also tests the build pipeline every UI slice relies on.
- **P4. Documentation upkeep.** The docs site has no pages for the UI crates (`web/site/src/SUMMARY.md`); each
  slice adds its pages, as rule 5 says.

---

## Order at a glance

| Step | Needs | Gives | Roadmap rows |
|---|---|---|---|
| P3 editor APK run | existing code | phone baseline, pipeline proof | M2, H3 |
| B1 contracts | research | shared types | U1 |
| A1 solver size model | B1 | text-driven layout | U1, U3 |
| A2 rasterizer | B1 | coverage, blending | none yet |
| A3 CPU backend | A2, B1 | first pixels, golden images | none yet |
| B2 benchmark | A1 | node-model answer | none |
| A4 text T1 | A2, B1 | Latin text, measure | U1 |
| B3 `.mxui` v1 | A1 | grammar | U4 |
| B4 input | A1 | event model | U2 |
| A5 events and scroll | B4, A1 | touch UI | U2 |
| A6 widget kit | A1, A4, A5, B3 | explorer screen | U3 |
| A7 `.mxui` loader | B3, A1 | declarative screens | U4 |
| B5, P2 | ECS decisions | world contract | T1b |
| A8 ECS connection | B2, B5 | live inspector | U4, T1b |
| A9 editor on the stack | A3, A6, A7, A8 | M3 path | U4, E1 |
| A10 hinting, fonts, helper | A4, A5 | small-size quality, input | M5 |
| A11 GPU backend | A3 | speed, shaders | Decision 12, step 2 |

## Milestones

- **UI-M1 first pixels:** the editor's paint list drawn by own code, golden images in CI and the wasm test, timed on the phone (A3).
- **UI-M2 Latin text:** text drawn and measured by own code, layout driven by it (A1, A4).
- **UI-M3 touch:** scrolling and numeric entry by touch (A5).
- **UI-M4 the explorer screen:** a screen like the screenshot, written in `.mxui`, on the CPU backend (A6, A7).
- **UI-M5 live editor panels:** inspector and hierarchy from a live world on the new stack (A8, A9).
- **UI-M6 GPU:** the GPU backend matches the CPU reference and carries the 3D viewport (A11).

## Decisions needed from the maintainer

| # | Decision | Blocks | Default I would take if told to proceed (provisional) |
|---|---|---|---|
| D1 | The node model: standalone tree, nodes as entities with a pure solver, or a pure solver with an ECS adapter | A8, A6's shape | Leaning ECS (maintainer, 2026-11-25). Build the pure solver first; run the B2 benchmark; decide on its result |
| D2 | `.mxui` prefabs with inheritance | B3 | DECIDED yes (2026-11-25) |
| D3 | Which draw vocabulary the CPU backend consumes first | A3 | The editor's `PaintItem`, because it is flat, has lines and text, and the hosts already speak it |
| D4 | Pixel format and color mapping | A2, A3 | Premultiplied `RGBA8`; `Color32` converted at the edge. Blend space: both, sRGB by default and linear light as an option (answered 2026-11-26) |
| D5 | The first hinting step | A10 | H1, a light vertical-only hinter for Latin, after the unhinted pipeline |
| D6 | One coverage rasterizer for shapes and glyphs, or two | A2, A4 | One |
| D7 | Where the text code lives, and the crate names | A4 | A crate of its own, depending on `mid-math` only; names open |
| D8 | Hover, press, focus as relations on one entity or as fields of a UI state resource | B4, A5 | A UI state resource first; relations when `mid-ecs` has them |
| D9 | Hinted against unhinted advances in the measure contract | B1, A1, A4 | The measure returns what the renderer draws |
| D10 | The six ECS questions (relationships, names and paths, protocol shape, entity numbers on the wire, first run-time query scope, three-way benchmark) | B5, P2 | Needs the maintainer's call; the UI does not depend on all six |
| D11 | The soft keyboard helper language | A10 | DECIDED Kotlin, not Java (2026-11-26); the work is not reached yet |
| D12 | The primitive vocabulary: the explorer's needs alone, or also the harness stack's list | B3 | The explorer's needs, checked against that list |
| D13 | The expression language: scope, where the evaluator lives, relation to DixScript 2.0 | B3, A7 | Small, outside the core crates, with a clean seam so DixScript 2.0 can supply it later |
| D14 | Integer-exact output on every architecture | A2, golden images | Yes |
| D15 | Run the B2 benchmark before D1 is final | B2, A8 | Yes |

## Parked

Kept so they are not lost, not scheduled: LCD subpixel rendering, color glyphs and emoji, a general path renderer
with strokes and dashes for vector content, a GPU glyph atlas, sharing the rasterizer with msx, the Windows,
Linux and macOS hosts beyond the Android one, a web host through the browser's canvas beyond the playground, and the
interpreter and PostScript hinting engines.
