# UI comparison: mid-ui, Slint, Iced, Dioxus Native, egui and the reference repos

## Overview

Second pass of the research the maintainer asked for on 2026-11-21 to 23, after
`docs/research-ecs-comparison.md`. The subject is `mid-ui`. The question is what it should become so
that it runs natively on the hardware (not only painted through a host), on desktop and mobile with
the web later, fast, easy to use, aware of the ECS, able to carry shaders, and driven by DixScript.
The editor is the first customer: the look asked for is the flecs explorer combined with the Unity
editor (`docs/mid-editor.md`, "Decisions").

This is reading, not measuring. No candidate was built, run or benchmarked, nothing ran on the
phone, and no code changed. No UI number from CI exists in this project, so none is quoted. Where a
size or a count appears (lines in a file, a version, a feature name) it was read from the source and
carries a tag.

## Method and evidence levels

| Source | Commit | What was read |
|---|---|---|
| `slint-ui/slint` (1.20.0, rust-version 1.95) | `4d1f229`, 2026-11-23 | `LICENSE.md`; the guides for backends and renderers and for Android; the manifests and READMEs of `internal/renderers/{software,femtovg,skia,anyrender}`; `internal/backends/android-activity` (`Cargo.toml`, `build.rs`, the Java directory); the headers of `internal/core/properties.rs` and the size of `layout.rs`; the Cargo features of the `winit` backend; the CI file for 32-bit ARM lines |
| `iced-rs/iced` (0.15.0-dev, rust-version 1.93) | `98bd48f`, 2026-11-19 | `README.md`; `core/src/{widget,renderer,text,backend}.rs`, `core/src/layout/`; the file list of `widget/src`; the crate list; manifests; a search of manifests, markdown and the changelog for Android and accessibility |
| `DioxusLabs/dioxus` (0.8.0-alpha.3, rust-version 1.85) | `0a7242a`, 2026-11-22 | `README.md`; `packages/vdom/src/mutations.rs`; the package list; the manifest of `packages/core` |
| `DioxusLabs/blitz` (0.3.0-beta.4, rust-version 1.91) | `29a439e`, 2026-11-23 | `README.md`, `HOWTO_WASM.md`; the workspace manifest; the manifests of `dioxus-native`, `blitz-shell`, `accesskit_xplat` |
| `DioxusLabs/anyrender` | `ecf014e`, 2026-11-24 | `README.md`; `crates/anyrender/src/lib.rs` (the three traits) |
| `emilk/egui` (0.37.0, rust-version 1.95) | `ea6b094`, 2026-11-23 | `README.md`, `ARCHITECTURE.md`; `crates/egui/src/context.rs`; `epaint` (`lib.rs`, `mesh.rs`, shape list, manifest); `egui_glow/src/painter.rs`; the READMEs of `egui_inspection` and `egui_kittest`; the container list; `scroll_area.rs` signatures |
| `Mid-D-Man/msx` | `490b204`, 2026-11-24 | `README.md`; the manifests of the render crates, the viewer and `msx-text`; the doc comments of `msx-text` and `msx-render-gpu`; `render/msx-render-core/src/traits.rs` |
| `mxross-canvas` (the uploaded zip, 79 files) | the commit named in the zip's comment, `dd5b068` | the workspace and Android manifests; `crates/mxross-android/src/{lib,gpu,ui}.rs`; the CI workflow |
| `Mid-D-Man/mid-engine` | `1d1d999`, 2026-11-24 | `crates/mid-ui/src/draw.rs`; `docs/mid-ui.md`, `docs/mid-ui-mxui.md`, `docs/mid-ui-wasm.md`, `docs/mid-editor.md`, `docs/architecture.md` (mandates), `docs/roadmap.md` (Decisions 11 to 13, the build table); `crates/mid-editor/src/paint.rs` by its constants |

Every claim below carries one tag:

- **[src]** read in the source or manifest of that project.
- **[doc]** stated in that project's own documentation or README, not checked against its code.
- **[gap]** searched for and not found, where the search is named. A gap means "not found in what was
  read", not "does not exist".

There is no **[ci]** tag in this document, because no CI figure for UI exists yet.

Version notes. `egui` on main is 0.37.0 and its entry point is `Context::run_ui` [src], while the
canvas app pins 0.34, so code read in the canvas app is an older API. Iced pins `cosmic-text` to a
git revision of a fork [src]. Blitz pins `winit` to an exact beta, `=0.31.0-beta.4` [src]. Every
project was read at its main branch, not at the release the maintainer would take from crates.io.
Not read: Slint's standard widgets (only the directory of widget style families was listed),
`egui_extras`, Iced's runtime and `Task` model beyond the file list, Freya (named in the Dioxus
README), and Slint's Qt backend. The performance of every candidate is unknown.

## Summary

| Area | mid-ui | Slint | Iced | Dioxus Native (Blitz) | egui |
|---|---|---|---|---|---|
| Model | Retained arena tree with generational ids, one `Style` struct [src] | A declarative `.slint` language with a compiler and an interpreter crate, a retained item tree, a property binding engine [src] | Elm style: messages, a `Widget` trait with `layout`, `draw`, `update`, state in a tree diffed between views [src] | A DOM with CSS (Stylo), driven by Dioxus's virtual DOM through a stream of mutations [src] | Immediate: the UI function runs every frame and returns shapes [doc] |
| What the library hands to the host | A draw list of three commands [src] | Its own renderers: software, FemtoVG, Skia, Vello, and Qt [doc] | Its own renderers: `wgpu` and `tiny-skia`; the workspace also has a `vello_cpu` crate [doc, src] | Draw calls into `anyrender`, with Vello (GPU and CPU), Skia and `tiny-skia` backends [src] | Textured triangle meshes with a clip rectangle each, plus texture changes [src] |
| Layout | Anchors, row and column flow, fill weights; no min, max or text-driven sizes [doc] | An own layout engine, `layout.rs`, 3,709 lines; contents not read [src] | `Limits` (min, max, compression) passed down, with a flex module [src] | Taffy: flexbox, grid, block [doc] | Immediate; sizes of some things are remembered for the next frame [doc] |
| Text | A host callback and a bundled ASCII font [doc] | Parley, Fontique and Swash; the software renderer limited to western scripts [src, doc] | `cosmic-text` from a git fork [src] | Parley and Skrifa [src] | `epaint` with `harfrust`, `skrifa`, `unicode-segmentation` [src] |
| Android | A raylib host, `armeabi-v7a`, portrait only [doc] | A backend on `android-activity` that depends on the Skia renderer; `armv7-linux-androideabi` is in its docs.rs targets [src] | Not in the platform list [doc]; window calls documented as unsupported on Android [src] | `android-activity` in the shell crate; a browser app build for Android is advertised [src, doc] | `eframe` lists Android [doc]; the canvas app drives it by hand [src] |
| Without a GPU | Whatever the host does | A software renderer that runs in `no_std` and can render line by line [doc] | `tiny-skia` fallback [doc] | CPU paths exist (`vello_cpu` and Skia raster features; `anyrender_tiny_skia` in the workspace dependencies) [src] | No CPU painter among the crates `ARCHITECTURE.md` lists (glow, wgpu) [gap]; `epaint` has `vello_cpu` as a dependency, purpose not checked [src] |
| GPU floor | None in the core | FemtoVG needs OpenGL; Skia is "heavy" on disk; Vello needs compute shaders and has no fallback [doc] | The `wgpu` GL path needs OpenGL ES 3.0 or newer [src] | Default feature is `vello-gpu` [src] | `egui_glow` requires OpenGL 2.0 or newer [src] |
| Accessibility | None | AccessKit through the `winit` backend feature [src] | None found in manifests, markdown or the changelog [gap] | AccessKit, with an Android adapter crate [src] | AccessKit [src] |
| Custom GPU content in the UI | None | `unstable-wgpu-29` and `unstable-wgpu-30` features [src] | A `Shader` widget [src] | A `wgpu_texture` example [doc] | `Primitive::Callback` [src] |
| Test tooling | Unit and mutation tests, a Canvas 2D playground [doc] | A testing backend directory [src] | `test`, `selector`, `tester`, `devtools` crates [src] | A test harness crate and a `wpt` directory [src] | `egui_kittest`, `egui_inspection` [src] |
| Rust floor | `mid-ui` does not check on 1.75 today (`mid-math`) [doc] | 1.95 [src] | 1.93 [src] | Blitz 1.91, Dioxus 1.85 [src] | 1.95 [src] |
| License | Not checked | GPLv3, a royalty-free license that excludes embedded, or a commercial license [doc] | MIT text in `LICENSE` [src] | Apache 2.0 or MIT [doc] | Apache 2.0 and MIT files [src] |

## Findings by area

### What mid-ui is today

`mid-ui` is a standalone core, about 1,000 lines in eight files: an arena `Tree`, `layout::compute`,
`hit_test`, and `build_draw_list` [src, `docs/mid-ui.md`]. It depends on `mid-math` only, is generic
over `f32` and `f64` through a `Real` trait, uses explicit stacks so a 50,000-level tree cannot
overflow, and passes `cargo check` for `wasm32` [doc]. It has no text, images, event dispatch, focus,
animation, min or max sizes, or wrap [doc]. `mid-ui-mxui` loads DixScript documents into a tree,
with strict fields, error paths such as `root.children[2].margin`, and binding keys hashed with
FNV-1a, kept as metadata only [doc]. The browser playground paints the draw list with Canvas 2D and
draws text itself as a stand-in, because the draw list carries none [doc, `docs/mid-ui-wasm.md`].

The editor does not paint `mid-ui`'s draw list. It lays out its four regions with `mid-ui`, then
converts to its own paint list, a flat `repr(C)` record of 36 bytes with five kinds (rectangle, clip
push, clip pop, line, text), drawn by the C host [doc, `docs/mid-editor.md`; the five kind constants
in `paint.rs` [src]]. Buttons, rows, text and hit testing are done by the editor core, not by
`mid-ui`. This is the state the roadmap names as the reason for U1 to U3.

### The render seam

The five libraries differ most here, and the difference decides who owns the window and the GPU.

| Library | What crosses the seam |
|---|---|
| egui | `Context::run_ui(RawInput, ...)` returns a `FullOutput`; `tessellate` turns shapes into `ClippedPrimitive` values, each a `Mesh` of `Vertex { pos, uv, color }` (20 bytes) with a clip rectangle, or a `PaintCallback` [src]. The core has no window and no GL. |
| Iced | A `Renderer` trait with `start_layer`, `end_layer`, `start_transformation`, `fill_quad`, and in the text half `fill_text` [src]. Widgets call it from `draw`. |
| anyrender (Blitz, Slint's Vello option) | `PaintScene`: `push_layer`, `push_clip_layer`, `push_mask_layer`, `pop_layer`, `stroke`, `fill`, `draw_glyphs`, and a CSS-style box shadow call. Two more traits stand beside it: `WindowRenderer` and `ImageRenderer` [src]. |
| Slint | Not a command list. Renderers receive items and own the drawing: software, FemtoVG, Skia, Vello, Qt. The backend (window, input) is a separate choice from the renderer, and both are selected at run time by name, for example `winit-software` [doc]. |
| msx | A `Renderer` trait, `render(&Scene, &mut RenderTarget)`, with SVG, CPU (`tiny-skia`) and GPU (`wgpu`, `lyon`) implementations [src]. |

The reading: `mid-ui`'s flat, pointer-free list sits at the same level as Iced's `fill_quad` and
anyrender's `fill`, and one level above egui's triangles. The editor's `PaintItem` is already richer
than `DrawCmd` (lines and text), so two draw vocabularies exist in the repository today. Every
multi-platform candidate keeps exactly one such seam and puts everything platform-specific behind it.
Every candidate also has an escape hatch for content it does not draw itself: egui's
`Primitive::Callback`, Iced's `Shader`, Slint's wgpu features, Blitz's texture example [src, doc].
`mid-ui` has none, and the editor's 3D viewport avoids the need only because it is projected on the
CPU into the same list (Decision 12, step 1).

### Does the library paint itself

| Option | Shown by | For | Against |
|---|---|---|---|
| Draw list and a host per platform (today) | egui and `mid-ui` | No GPU or window assumptions in the core; the C host already runs on the phone [doc] | Every host re-implements text, clipping, rounded rectangles; the host's line and rectangle submission was the limit at 5000 cubes [doc, `docs/mid-editor.md`] |
| A CPU rasterizer as a backend crate | Slint's software renderer, `tiny-skia` in Iced and msx, `vello_cpu` in Blitz [doc, src] | No driver or GPU level to depend on; partial and line-by-line rendering exist in Slint's [doc]; runs wherever a framebuffer exists | Speed on 32-bit ARM is unmeasured; Slint's software renderer lacks rotation, scaling and `inner-shadow` [doc] |
| A GPU backend crate | Iced (`wgpu`), Blitz (Vello), the canvas app (`wgpu`, Vulkan only), msx (`wgpu` offscreen) [src] | One shader language, custom shaders, the 3D viewport in the same pass | `wgpu` breaks its API about every three months (msx's own note, [src]); the three projects in the maintainer's reach use 26 (msx, per its doc comment), 29 (canvas) and 30 (Blitz) [src]; the phone's GPU facts are unverified |

All three fit the mandates, because a backend is one layer above the core crates. The maintainer's own
canvas app is the template for the native route: `android-activity` without `winit`, a `wgpu`
surface, touch events turned into the UI library's events by hand [src, `gpu.rs`, `lib.rs`].

### Layout

`mid-ui` has anchors, row and column flow with gaps, and `Fill(weight)` [doc]. What its doc lists as
missing is what the candidates build their layouts on: Iced passes `Limits` with a minimum, a
maximum, and two flags (compress fluid lengths to intrinsic size, allow intrinsic size past the
maximum) [src]; Blitz delegates to Taffy for flexbox, grid and block [doc]; egui avoids the problem
by laying out as it draws, and says plainly that this makes centering and sizing harder, with a
one-frame delay for windows and grids [doc]. Slint has a 3,709-line layout module [src] that was not
read. The common ingredient is intrinsic size: a leaf (text above all) reports what it needs, and
the parent shares the rest. That is the same text seam as Decision 11, so text measuring and layout
are one gap, not two.

### State, events and reactivity

| Library | How state reaches the UI |
|---|---|
| Slint | Properties with automatic dependency tracking; a binding re-evaluates when what it read changes. `properties.rs` is 1,657 lines [src] |
| Iced | Messages and an update function; the view is rebuilt and widget state in a tree is diffed [src] |
| Dioxus | A virtual DOM. Renderers implement `WriteMutations`, a stack machine: `push_id`, `child`, `pop`, `create_element`, `create_text`, `append_children`, `replace_with`, `insert_after`, `insert_before`, `set_attribute`, `set_text`, `add_event_listener`, `remove_event_listener`, `remove` [src]. The Blitz workspace has a `dioxus-native-dom` package for this role; its code was not read |
| egui | Nothing is kept; the application state is the state [doc] |
| mid-ui | Retained tree, binding keys as metadata, no dispatch yet [doc] |

Two readings for the ECS. First, `mid-ecs` keeps a pair of change ticks per column and has `Changed`
[doc, `docs/research-ecs-comparison.md`]; a UI binding layer could mark a node dirty from a changed
component, which is how a retained tree stays cheap against a live world. Nothing of this was
written, and no candidate does it. Second, the mutation stream is a small, closed set of operations
that map onto what a world already offers (spawn with a parent, set a component, despawn), so a
`mid-ui` document fed by such a stream, or an ECS-backed document in the style of `dioxus-native-dom` (not read),
is a shape that exists elsewhere. The `docs/mid-ui.md` decision to keep the tree off the ECS stands
until both are benchmarked, as that decision itself says.

### Text and input

| Library | Text stack | Soft keyboard on Android |
|---|---|---|
| mid-ui, editor | A host callback measures; DejaVu Sans, ASCII only [doc] | Unchecked [doc] |
| Slint | Parley, Fontique, Swash; software renderer "limited to western scripts" [src, doc] | One Java file, `SlintAndroidJavaHelper.java`, compiled and dexed by `build.rs` through the `android-build` crate [src] |
| Iced | `cosmic-text` [src]; `input_method.rs` exists in the core [src] | Android not supported [doc] |
| Blitz | Parley, Skrifa [src] | An AccessKit Android adapter that embeds a dex [src] |
| egui | `harfrust`, `skrifa`, `unicode-segmentation` [src] | Through `egui-winit` in `eframe` [doc], not checked on a phone; the canvas app avoided it (below) |
| msx | `ttf-parser` outlines from an embedded DejaVu subset, no shaping; text becomes paths [src] | Not applicable |

The canvas app records the Android constraint that matters to the editor. Under `NativeActivity`
the OS keyboard cannot be shown for a view that is not an `EditText`, the upstream issues
`rust-mobile/android-activity#44` and `#140` are named there as unresolved, and `game-activity` was
rejected there because it needs Gradle for its `.aar`. The app built a numeric keypad inside egui
instead [src, `ui.rs` comments]. Slint's route, a small Java helper built by the Cargo build script,
is the only evidence read of a Rust UI getting a real soft keyboard on Android, and it conflicts
with the editor's current decision of "No Gradle, no Java" (`docs/mid-editor.md`). That decision
needs the maintainer, not more reading (question 6).

### Widgets and the explorer screenshot

The screenshot needs: a left icon rail; a tree with a dropdown, a search field, folds and colored
dots; a viewport; an inspector with a header (path, name, number, Disable, delete), a Components
dropdown, groups by module path with folds, editable numeric fields, an Add Component button; and a
status bar. What was verified to exist in the candidates is thin, because widget libraries were not
read in full:

- egui: `CollapsingHeader`, `ScrollArea` with `show_rows` (rows drawn from a range, which is the
  virtualization a hierarchy of thousands of entities needs), `ComboBox`, `Popup`, `Panel`, `Resize`,
  a `scroll_physics.rs` [src, file and signature names only].
- Iced: `table`, `pane_grid` (dockable panels), `scrollable`, `text_input`, `text_editor`,
  `combo_box`, `responsive`, `sticky`, `shader` [src, file names only].
- Blitz: "HTML form controls" as a goal, extensibility by custom widgets [doc].
- Slint: widgets live in a standard library that was not read [gap].

The smallest kit that gives the screenshot is therefore: scrolling with row virtualization, a text
field, a numeric drag field, a fold, a button, a tab or rail item, a popup list, and a splitter.
`mid-ui` has none of the eight, and the editor draws its own buttons and rows. That list is the practical content of roadmap U3.

### Shaders and unique content

Shader-capable UI means two different things. One is a widget whose fill or effect is a shader. msx
is the only reading here that does it: `Def::Shader` WGSL runs only in its GPU renderer, and text is
converted to paths exactly so that gradients and shader fills apply to it [src, doc]. msx also has
signed-distance-field and Gaussian-splat elements and a DixScript front end with QuickFuncs as
parametric generators [doc], which is the maintainer's own prior art for "unique stuff". The other is
an embedded viewport, which is the escape hatch listed above.

### Platforms and floors

The Rust floors of the candidates (1.85 to 1.95) are above the 1.75 that some libraries in the
workspace keep [src, doc]. They can only live in host or backend crates, not in a core crate, under
the dependency mandate ("zero-to-minimal external dependencies, every core crate, no exceptions",
`docs/architecture.md`). No candidate has been run on 32-bit ARM Android with this project's phone.
The nearest evidence is manifest-level: Slint lists `armv7-linux-androideabi` in its Android backend
docs.rs targets [src]; its CI cross-checks `armv7-unknown-linux-gnueabihf` for the C++ library, which
is Linux, not Android [src]; the canvas app's workflow builds `armv7-linux-androideabi` with
`cargo-apk` [src]. Web: egui says all of its code is Wasm friendly [doc]; Blitz documents a
`wasm32-unknown-unknown` build with `anyrender_vello_gpu` and WebGL [src]; Slint has a web guide
[src]; `mid-ui` passes `cargo check` for `wasm32` and has the Canvas 2D page [doc].

What the canvas app shows about the phone's GPU is limited: it creates `wgpu` with Vulkan only and
default limits [src]. There is no record in the zip of it running on the Galaxy A13 4G, or of the
adapter it found (question 7). Slint's Android backend depends on its Skia renderer, which the docs
call heavy in disk footprint [src, doc]. The editor's own finding on this phone, that raylib's
default OpenGL ES 2 shaders are `mediump` and placed geometry a pixel or two off, is the one hard
fact about its GPU path [doc].

### Accessibility and testing

AccessKit is shared by egui, Slint (behind a feature) and Blitz [src]. Iced has none in the places
searched [gap]. `mid-ui` has no accessibility tree. In Bevy's build order accessibility sits beside
input (`docs/roadmap.md`).

egui's testing is worth noting for the remote protocol. `egui_kittest` queries the AccessKit tree
and clicks by label. `egui_inspection` defines a wire protocol with four requests (read the tree,
inject input events, take a screenshot, resize) that is "strictly request and response", so it maps
onto a TCP socket or a unary RPC [src]. The ECS comparison already proposed a JSON-RPC style remote
over the reflect API; one transport could carry world methods and UI methods. The editor's
self-test, which taps by reading labels out of the paint list [doc], is the same idea in the
repository today.

### Licensing

Slint's three choices are GPLv3, a royalty-free license that requires disclosing the use and excludes
embedded systems, and a commercial license [doc]. The others are MIT or Apache 2.0 [src, doc]. The
license of `mid-engine` itself was not found at the repository root, so whether code may be vendored
into a host crate is open (question 8). Reading and re-implementing an idea needs no answer.

## What mid-ui already has

Facts from `docs/mid-ui.md` and `docs/mid-ui-mxui.md`, not compared against the others:

- A core with no renderer, no window, no ECS and no file format, usable by any host through one tree.
- `f32` and `f64` from one implementation, tested at both.
- No recursion in layout, hit testing, drawing or loading (50,000 levels, tested on a small stack).
- A DixScript loader with strict fields, error paths, enum decoding and key-collision checks, split
  into a source-neutral reader and a DixScript adapter so the 2.0 move touches one module.
- A browser playground that paints the draw list with Canvas 2D.
- Twelve hand-made mutants caught by the tests [doc].

## What to take from each

Proposals for the maintainer, not decisions.

| From | Idea | Where it lands in mid-ui |
|---|---|---|
| Slint | Declarative markup with named properties and bindings; backend and renderer chosen separately by name; a CPU renderer that works line by line | `.mxui` plus `BindKey`; the host and backend crates; a CPU backend crate |
| Slint | A Java helper for the soft keyboard | Question 6 |
| Iced | Constraint passing with `Limits`; widget state in a tree diffed between views; `pane_grid` for dockable panels; a `Shader` slot; headless test crates | Layout with min, max, intrinsic; node state that survives a rebuild; the editor's panel arrangement; the viewport slot |
| Dioxus and Blitz | A mutation stream as the only door between app logic and a retained document; `anyrender` as one draw seam with CPU and GPU backends picked by feature | A world-fed document; the shape of the draw seam |
| egui | `RawInput` in and a plain output out, with no window in the core; row virtualization; a paint callback; AccessKit tests; a remote inspection protocol | The editor `frame` already has this shape; the hierarchy list; the viewport slot; U2 tests; the remote protocol |
| msx | DixScript-defined documents with generators; a `Renderer` trait; SDF primitives; text as paths for shader fills | `.mxui` authoring; the backend trait; shader-capable widgets |
| mxross-canvas | `android-activity` with no `winit`; touch turned into UI events by hand; an in-UI keypad | The native host if the maintainer wants one without raylib |

## Gaps, ranked

Ranked for the editor that has to run on the phone first. Each names where the way is shown.

1. **Text in the seam.** A measuring callback exists in the editor, not in `mid-ui`; the draw list
   has no text command. Decision 11 (all five stacks above; the choice of shaper is the open part).
2. **Events, focus, scrolling with touch** (roadmap U2). Hit testing exists; dispatch, capture,
   focus and scroll physics do not. egui and Iced both have scrolling code to read next.
3. **Row virtualization** for the hierarchy. egui's `show_rows`.
4. **A custom-content slot** for the viewport and shaders. egui, Iced, Slint, Blitz.
5. **Intrinsic, min and max sizes, and wrap.** Iced `Limits`, Taffy.
6. **The widget kit** from the explorer screenshot (U3): the eight pieces listed above.
7. **Keyed binding to reflected values,** with change-driven invalidation from `mid-ecs`.
8. **A soft keyboard decision** and path on Android.
9. **One draw vocabulary.** `DrawCmd` has three commands, `PaintItem` five; lines, text, images,
   gradients, shadows, paths are missing from the first. `anyrender`'s `PaintScene` is the reference.
10. **An accessibility tree.** AccessKit.
11. **UI over the remote protocol.** The `egui_inspection` shape.
12. **A CPU rasterizer backend,** then a GPU backend (Decision 12, step 2).
13. **Animation** on `mid-anim` tracks (already planned in `docs/mid-ui.md`).

## A comparison probe, designed and not built

The question that reading cannot answer is cost on the phone. Probe the same screen three ways.

- **The screen.** The explorer layout from the screenshot with a tree of N rows (1,000 and 10,000),
  a search field, folds, an inspector with 20 numeric fields, a status bar.
- **The three builds.** `mid-ui` with the editor host (raylib); egui through the canvas app's pattern
  (`android-activity`, `egui-wgpu`, armv7); Slint with its Android backend. Iced is left out until
  something shows it runs on Android; Blitz waits for the same.
- **What to record.** APK size, cold start, frame time idle and while scrolling, memory, whether text
  entry works, and the CI time. Put the CPU line and, for `wgpu`, the adapter name and backend into
  the output header, as the ECS probe does.
- **Controls.** A floor that draws N plain rectangles with each backend, so a ratio is read against
  the hardware.
- **How to read it.** The phone and real CI are the authority. A sandbox x86_64 number says nothing
  about a 32-bit phone, as the ECS work already found.

A second, smaller probe: feed `mid-ui`'s draw list into `anyrender`'s `tiny-skia` and `vello_cpu`
backends on CI and compare to the host's own painting. It answers whether the seam can be shared
with an existing backend without touching the core.

## Questions for the maintainer

1. Does `mid-ui` paint itself through backend crates it owns, or stay a draw-list producer with a
   host per platform?
2. If it paints itself, CPU first (no driver assumptions) or GPU first (Vulkan only is what the canvas
   app tried)?
3. One draw vocabulary for `mid-ui` and the editor's paint list, or two?
4. Text: keep the host callback and the host's fonts until U3, or choose a shaping stack now (Parley,
   `harfrust` with `skrifa`, or `cosmic-text`) in a host-layer crate?
5. Should `.mxui` stay data with names bound in Rust, or grow expressions and bindings like
   Slint's `.slint`?
6. Android soft keyboard: accept a small Java helper built by the build script, or keep "no Java"
   and write an in-UI keypad like the canvas app?
7. Did the canvas app ever run on the Galaxy A13 4G, and if so, which adapter, which backend, and
   what frame time?
8. May host crates vendor MIT or Apache code from these projects, or is reading and
   re-implementing the rule? What license does `mid-engine` use?
9. Is an immediate-mode authoring layer wanted for tools (egui style, generated from reflection)
   on top of the retained tree, or is one retained model enough?


## Answers and consequences (2026-11-24)

The maintainer answered the nine questions. They are recorded as Decision 14 in `docs/roadmap.md`.

| # | Answer | What follows |
|---|---|---|
| 1 | Backend crates owned here, using `mid-math` | The core stays renderer-free; backends sit one layer up; vectors, `Affine2`, `Mat3`, `Color32` and curves come from `mid-math` |
| 2 | CPU backend implemented first; in a shipped build the GPU backend takes priority | The backend seam has to fit both; the CPU output is the reference the GPU output is compared with |
| 3 | Two draw vocabularies | Which one the CPU backend consumes first is open; the editor converts today |
| 4 | Read the candidates' repositories, take what is needed, write an own stack | Gap 1 becomes a reading task: font parsing, shaping, line breaking, glyph coverage or atlas; no third-party shaper by default |
| 5 | `.mxui` grows expressions and bindings, as HTML and CSS are for a page, for any user of `mid-ui` | Needs an expression language spec; the evaluator lives outside the core crate (dependency mandate); the link to DixScript 2.0 is open |
| 6 | Both a native helper and an in-UI keypad; asked why not Kotlin | See below |
| 7 | The canvas app runs smoothly on the phone; details not remembered | `wgpu` with Vulkan is known to run there; adapter, limits and frame time are unknown, and a status line in the app would record them |
| 8 | Open source, with MidManStudio keeping ownership | Read and re-implement; vendoring is a separate decision; Slint's GPLv3 and royalty-free terms (see Licensing) are a reason to read its code only |
| 9 | An immediate-mode authoring layer is wanted | Needs stable node identity across frames (egui's `Id`, the binding keys of `.mxui`) and a diff of the emitted nodes into the retained tree; the Dioxus mutation stream is the same mechanism and could serve bindings too (proposal, not read in code) |

### Checks made for the answers

- **Slint's helper route** [src, `internal/backends/android-activity/javahelper.rs`]: the dex bytes are
  embedded in the native library with `include_bytes!` and loaded at run time through
  `InMemoryDexClassLoader`, with a `DexClassLoader` fallback that writes the dex to the code cache.
  This path needs no `classes.dex` in the APK, so `android:hasCode="false"` in the editor's manifest
  does not have to change by this route. It was not tried in this repository.
- **Java or Kotlin.** Both go through the same dexer. For Java: a JDK is already a stated
  requirement of `scripts/build_editor_android.sh` [src], and the only precedent read, Slint's, is one
  Java file compiled by its build script [src]. For Kotlin: a Kotlin compiler is an extra build
  tool, and the compiler's null-check helper classes have to be dexed in or switched off by flags
  [gap, not verified here]. The runner-image release notes that were reachable list Java versions and
  Android build-tools but showed no Kotlin line [gap, partial notes]. The lean: Java for a helper that
  only does text input, Kotlin if the helper grows to cover more of the Android API surface. The
  Rust or C side should call a small interface so that the language can change.
- **`mid-math` on the phone.** SIMD is selected for x86, x86_64, aarch64 and wasm32 with `simd128`;
  `armv7` is in none of those lists, so the 32-bit ARM phone appears to run the scalar backend
  [src, `crates/mid-math/src/lib.rs`, cfg lists]. A CPU backend there is scalar until an `armv7`
  backend exists.


## Addendum: the harness UI stack in the blaze repository (2026-11-25, first pass)

Found while reading `blaze-ecs` (see the addendum in `docs/research-ecs-comparison.md`).
`crates/harness/` in `indexable-inc/blaze` (`2bbf1bc8`) holds a native macOS app and its UI stack,
described as copied from `github.com/andrewgazelka/den` [doc, crate descriptions]. This pass read the
crate descriptions, sizes, the doc comment of the layout crate and the head of `prims.flecs`. The
`den` repository was not cloned [gap], and the renderer, shaders, text shaper, scroll physics and
motion code were not read.

| Crate | Role | Rust lines |
|---|---|---|
| `harness-native` | The app: AppKit and Metal, Slang shaders, a Flecs world of devices and objects | 40,816 |
| `harness-native-render` | The Metal renderer, a text shaper and glyph atlas | 19,816 |
| `harness-native-scroll-physics` | macOS scroll physics, said to match a default `NSScrollView` | 2,505 |
| `harness-native-shaders` | The renderer's shaders in Rust | 2,421 |
| `harness-native-ui-view` | Views as trees of entities, built from prefabs | 1,986 |
| `harness-native-ui-layout` | One flex layout pass, "the Clay model", over node trees | 1,939 |
| `harness-native-ecs` | A typed store over a vendored Flecs fork; builds for wasm32 | 1,887 |
| `harness-native-ui-motion`, `-timeline` | Springs over components and relationships; timeline cues applied through reflection | 626, 92 |
| `harness-native-ui-text-api`, `-input`, `-draw`, `-state-api` | Text and picture content, hit testing, the draw pass, one-of state relations | 445, 386, 285, 43 |

What it shows [src unless marked]:

- **UI as ECS.** A node is an entity. The primitives (`ui.Row`, `ui.Column`, `ui.Words` and others) are
  Flecs prefabs declared in script files, `prims.flecs` being 758 lines. An instance states only what
  differs from its prefab, as `title : ui.Words { Text: {s: "Hello"} }`, and the prefabs are made of
  small components. This is the closest analogue read to `.mxui` as the HTML and CSS of the UI.
- **Passes as systems.** Layout reads `Flex`, `Size`, `Padding`, `Aspect` and `Intrinsic` and writes
  `Laid`. Draw reads the looks (`Fill`, `Edge`, `Shape`, `Lift`, `Frost`, `TextStyle`) and the content
  (`Text`, `Icon`, `Picture`). Input reads `Pressable`, `Focusable`, `Editable` and `Cursor`, sets
  `Hovered`, `Pressed` and `Focused`, and emits `Activated`. A layout root whose inputs are unchanged
  since its last solve is left as it is.
- **A size model with intrinsic sizes.** An axis is `Fit`, `Grow`, `Fixed` or `Percent`, with a value,
  a minimum and a maximum. Text sizes arrive as an `Intrinsic` component that the host's typesetter
  writes, which is the text measuring seam of Decision 11 expressed as data in the world.
- **Motion and scripting through reflection.** Springs run over components and relationships, and a
  timeline sets members of nodes through Flecs reflection, which is what the reflect bridge of this
  project offers [doc, crate descriptions].
- **Scope.** The stack is macOS only (AppKit, Metal), builds only with blaze, and sits on a vendored C
  Flecs fork [doc]. It is a reference for architecture, not a candidate to depend on. Nothing in it
  targets Android or 32-bit ARM.

Relation to the decisions. `docs/mid-ui.md` keeps `mid-ui`'s tree off the ECS until it is
benchmarked. This stack is the first example read of the other choice: nodes as entities, passes as
systems, prefabs in a declarative script. Reasoning, not a finding: following that route in
`mid-ecs` would need what the ECS comparison ranks as missing (names and paths, templates, run-time
queries), and it fits the immediate-mode authoring layer and the bindings of Decision 14 less
directly than a retained tree does.

Questions for the maintainer from this addendum:

1. Should `mid-ui` nodes stay outside the ECS, or become entities with the passes as systems?
2. Should `.mxui` have prefabs with inheritance, so that an instance states only what differs?

## Sources

- Slint: `LICENSE.md`; `docs/astro/src/content/docs/guide/backends-and-renderers/backends_and_renderers.mdx`,
  `.../platforms/mobile/android.mdx`; `internal/renderers/software/{README.md,Cargo.toml}`,
  `internal/renderers/femtovg/Cargo.toml`; `internal/backends/android-activity/{Cargo.toml,build.rs,java/}`;
  `internal/backends/winit/Cargo.toml`; `internal/core/{properties.rs,layout.rs}`;
  `.github/workflows/ci.yaml`.
- Iced: `README.md`; `core/src/{widget.rs,renderer.rs,text.rs,backend.rs}`, `core/src/layout/`;
  `widget/src/` (file list); `Cargo.toml`; `LICENSE`; `CHANGELOG.md` (search).
- Dioxus: `README.md`; `packages/vdom/src/mutations.rs`; `packages/core/Cargo.toml`.
- Blitz: `README.md`, `HOWTO_WASM.md`; `Cargo.toml`; `packages/{dioxus-native,blitz-shell,accesskit_xplat}/Cargo.toml`.
- anyrender: `README.md`; `crates/anyrender/src/lib.rs`.
- egui: `README.md`, `ARCHITECTURE.md`; `crates/egui/src/context.rs`, `crates/egui/src/containers/`;
  `crates/epaint/{Cargo.toml,src/lib.rs,src/mesh.rs,src/shapes/shape.rs}`;
  `crates/egui_glow/src/painter.rs`; `crates/egui_inspection/README.md`; `crates/egui_kittest/README.md`.
- msx: `README.md`; `render/*/Cargo.toml`; `apps/msx-viewer/Cargo.toml`;
  `primitives/msx-text/{Cargo.toml,src/lib.rs}`; `render/msx-render-gpu/src/lib.rs`;
  `render/msx-render-core/src/traits.rs`.
- mxross-canvas (zip): `Cargo.toml`; `crates/mxross-android/{Cargo.toml,src/lib.rs,src/gpu.rs,src/ui.rs}`;
  `.github/workflows/cross-platform-build.yml`.
- mid-engine: `crates/mid-ui/src/draw.rs`; `crates/mid-editor/src/paint.rs`; `docs/mid-ui.md`,
  `docs/mid-ui-mxui.md`, `docs/mid-ui-wasm.md`, `docs/mid-editor.md`, `docs/architecture.md`,
  `docs/roadmap.md`, `docs/research-ecs-comparison.md`.
