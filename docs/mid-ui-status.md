# mid-ui: where it stands, and the road to an editor on its own UI

Status: **kept current**. First written 2026-11-25, at the end of the research phase; last updated 2026-11-26 (B1 written). The plan is in
`docs/mid-ui-roadmap.md`; this document says what exists, what does not, and what to do next. Evidence tags
as in the roadmap: **[src]**, **[doc]**, **[gap]**, **[run]**, *reasoned*.

## 0. The short answer

`mid-ui` is a small, tested core (layout, hit testing, a draw list) with a DixScript loader beside it. It has
no text, no events, no widgets and no renderer of its own. The editor does not paint through it: it lays
out four regions with `mid-ui`, converts to its own paint list and has a C host (raylib) draw that. The plan
is to paint through backend crates in this repository, CPU first, with text and input written here, and to
move the editor onto that stack. The research is finished and Decisions 14 to 17 are recorded; no code of
the new stack has been written. The first piece of evidence the plan needs, the world-backed editor on the
phone, has not been collected.

## 1. What exists

| Piece | State |
|---|---|
| `mid-ui` | 1,018 lines in eight files: an arena `Tree`, `layout::compute`, `hit_test`, `build_draw_list`; generic over `f32` and `f64`; no recursion; 62 tests and 12 hand-made mutants caught; `cargo check` for `wasm32` passes; does not build on Rust 1.75 because of `mid-math` [doc] |
| `mid-ui-mxui` | 801 lines: a DixScript loader into a tree with strict fields, error paths and binding keys kept as metadata; 29 reader tests plus 9 fixture tests and 8 end-to-end tests; 18 mutants caught [doc] |
| `mid-ui-wasm` and `web/ui` | A browser playground that paints the draw list with Canvas 2D; 10 logic tests and 12 `node --test` tests [doc] |
| `mid-editor` | 4,304 lines on a real `mid-ecs` world: panels, reflection-driven inspector rows, hierarchy paging, play mode with snapshot and restore; 127 tests; its own paint list of five kinds, 36 bytes each [doc] |
| `mid-editor-ffi` | 302 lines, 8 tests: the C API the host calls [doc] |
| The C host | `apps/editor-android/main.c`, raylib 6.0, one font atlas loaded at 64 pixels and scaled at draw time [src] |
| On the phone | Only the earlier flat-scene editor ran on the Galaxy A13 4G (`armeabi-v7a`); at 5,000 cubes zoomed out it ran at 8.9 frames per second, and the host's rectangle submission, not the core, was the limit [doc]. The world-backed editor has not been built into an APK or run there |
| `mid-math` on 32-bit ARM | Scalar. NEON is aarch64 only [src] |
| The canvas app | A separate paint app on egui and `wgpu` (Vulkan) over `android-activity`, which runs smoothly on the phone by the maintainer's report; adapter and frame time unknown |
| Research | Six documents in `docs/research-ui-*.md` and the addenda in the ECS and UI comparisons |
| Decisions | 14 (backend crates, order of work), 15 (oracles, hinting, fonts, FFI and wasm), 16 (script scope), 17 (prefabs, node-model leaning) in `docs/roadmap.md` |

## 2. What the plan needs that does not exist

| Need | State | Slice |
|---|---|---|
| Written contracts: pixel format, paint list, text types, measure | written 2026-11-26 as `docs/mid-ui-contracts.md`, proposed, five open points for the maintainer | B1 |
| Intrinsic, minimum and maximum sizes, wrap, lazy rows in the solver | not started | A1 |
| A coverage rasterizer and blending | not started | A2 |
| A CPU backend with a C API and a wasm export | not started | A3 |
| Text: font reader, outlines, glyph masks, cache, measure | not started | A4 |
| Events, focus, scroll physics, touch inertia | tap, drag and pinch exist in the editor; the rest not started | A5 |
| Widget kit (eight pieces) | not started | A6 |
| `.mxui` v1: prefabs with inheritance, expressions, bindings | the loader exists without them | A7 |
| The ECS connection with change-driven invalidation | reflection rows exist in the editor; the rest not started | A8 |
| Hinting, system fonts, the soft keyboard helper | not started | A10 |
| A GPU backend | not started | A11 |

## 3. From source to a UI the editor owns, today and as it should be

**Today:** editor core, then `mid-ui` for four regions, then the editor's own paint list, then
`mid-editor-ffi`, then the C host, which draws rounded rectangles, lines, clips and text with raylib. Text is
measured by a host callback and drawn from a bitmap atlas.

**As it should be:** editor core and `.mxui` screens, then the solver with text-driven sizes, then one paint
list, then a backend crate (CPU first, GPU after) that rasterizes shapes and glyphs itself, with a C API for
hosts and a wasm export for the browser. The host shrinks to a window, input and a framebuffer.

The distance between the two is Track A of the roadmap: contracts, solver, rasterizer, CPU backend, text,
input, widgets, loader, ECS connection, then the editor's move.

## 4. The roadmap

`docs/mid-ui-roadmap.md`: Track B (contracts and design), Track A (slices A1 to A12), Track P (beside them),
an order-at-a-glance table, six milestones and the decisions needed. The first move is not a design: it is
collecting the phone baseline (P3).

## 5. How far (reasoned)

| Area | Built | Partly | Not started |
|---|---|---|---|
| Layout core | anchors, flow, fill weights, hit test | | size model, lazy rows |
| Draw and paint | a draw list; a paint list; a host | | an own rasterizer, an own backend |
| Text | a measure callback, a bundled font | | everything in tiers T1 to T4 |
| Input | tap, drag, pinch | | focus, scroll, keyboard |
| Widgets | | editor rows and buttons drawn by the editor core | a kit |
| Declarative UI | a loader | | prefabs, expressions, bindings |
| ECS connection | reflection rows, hierarchy paging | | change-driven invalidation, remote protocol |
| Platforms | desktop host, an Android spike, a wasm playground | | an own backend on any of them |

## 6. What is settled and what is open

**Settled.** Backend crates here, using `mid-math`; CPU first in order of work and GPU first in a shipped
build; two draw vocabularies; text read from the candidates and written here; Latin first and built for every
script; hinting wanted; bundled and system fonts; dev-dependency oracles in tests; `.mxui` as the HTML and CSS of
the UI with prefabs and inheritance; FFI and wasm testing for every new UI crate; `mid-math` scalar on 32-bit ARM
for now.

**Open.** The node model (leaning ECS, to be settled by a benchmark); the pixel format and color mapping; which
vocabulary the CPU backend consumes first; where the text code lives; how hinting starts; the soft keyboard helper's
language; the expression language and its relation to DixScript 2.0; the six ECS questions. The full table is in
the roadmap, section "Decisions needed from the maintainer".

## 7. Next five actions

1. **P3: run the Editor Android APK workflow** with the world-backed editor on the phone and send the Console
   numbers. It tests the 32-bit ARM link with `mid-ecs` and `rayon` for the first time and gives the baseline
   that A3 and A9 are compared with.
2. **B1: review the contracts** (`docs/mid-ui-contracts.md`, written 2026-11-26): pixel format, paint list
   relation, text types, measure contract. Section 5 of that document lists five open points with defaults;
   A2 to A4 wait on its answers.
3. **A1: the solver's size model,** with a fake measure, pure and test-first. It does not wait on the node-model
   decision.
4. **A2: the coverage rasterizer** skeleton with the `tiny-skia` oracle in dev-dependencies, once the pixel
   format is written.
5. **B2: the node-model benchmark** crate, once A1 gives it a solver to run in three shapes.

## 8. How this was measured, and what was not

Line counts and test counts come from the repository at `33b549b` and from the test sections of the docs they
cite. The only measurement made during the research is the FreeType hinting experiment in
`docs/research-ui-hinting.md` [run], on x86_64, about geometry and not speed. Nothing about speed of any
candidate was measured, and no candidate was built or run on the phone. Not verified: the world-backed editor on
the phone, the 32-bit ARM link with the ECS, the soft keyboard on Android, `mid-cli` on macOS, Windows and
Android, and file access on the phone's storage.
