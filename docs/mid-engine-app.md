# The engine app: a phone-first editor

**Status: the first version is built and tested in the sandbox and has not run on a phone.** It is
described in `docs/mid-editor.md`. This document is the design around it. First written on 2026-11-16
as a tools app (a console, an entity tree, an inspector). Rewritten on 2026-11-18, after the maintainer
said what the app is, and again on 2026-11-19, after the maintainer answered the open questions and
pointed at the Android build in the `Tower-of-Babylon` repository. "The engine app" is a working
title; the crates are `mid-editor` and `mid-editor-ffi` and the host is `apps/editor-android`.

## Overview

The app is an editor in the sense of the Unity editor: a **viewport** that shows the scene, a
**hierarchy** of the entities in it, an **inspector** for the selected entity's components, and
**project** panels. It runs on an Android phone, because a laptop is not available, and its
interface is built on `mid-ui`. These three points come from the maintainer. Everything else in
this document is a proposal.

Answered by the maintainer on 2026-11-19:

- The editor has an **edit mode and a play mode**, as in Unity.
- The viewport is **3D, with a 2D mode**, as in Unity.
- It goes **straight to native Android**. The web comes later.
- It reaches the phone as a **workflow-built download**, "GitHub workflows all the way", with
  scripts where needed.
- The phone is a **Samsung Galaxy A13, the 4G model**.

Still assumptions:

- The 2D mode is, for now, a top-down orthographic view of the same scene. Real 2D nodes wait, as
  `docs/mid-nodes.md` proposes.
- The first version is usable when a project opens, the scene shows in the viewport, a tap in the
  viewport or the hierarchy selects an entity, the inspector edits its components, a gizmo moves
  it, undo works, and the scene saves.

Constraints that shape it:

- The interface is `mid-ui`, not a borrowed toolkit.
- The phone cannot build the engine, so everything the app needs is built by a workflow, and a
  change reaches the phone through a pushed commit.
- The phone is a Galaxy A13 4G: an Exynos 850 with a Mali-G52 GPU, an entry-level part, and a
  32-bit ARM userspace (it installs `armeabi-v7a` APKs only, per the maintainer's repositories). The
  viewport has to be cheap, and every Rust crate in the editor path has to be right on 32 bits.
- Core crates keep the dependency mandate (`docs/architecture.md`, "Technical Mandates"). A host
  or a backend is one layer up and may use external code, as `tools/mdix-compiler` does.
- As much as possible must be testable without the phone.

## What the editor is made of

| Panel | Shows | Needs |
|---|---|---|
| **Viewport** | The scene, a camera, the selection, a grid, gizmos | The viewport draw list, a camera, picking, input |
| **Hierarchy** | The entities as a tree (`ChildOf`, `Children`), with names where `mid-nodes` provides them | The tree widget, reparenting by drag, selection |
| **Inspector** | The selected entity's components as a property grid | `mid-reflect` and its ECS bridge, field widgets, edit commands |
| **Project** | The project's files and scenes | A way to read and write files (Decision 13), a list widget |
| **Console** | Log output and a command line | Text, a text field, `mid-log` output |

On a phone the panels cannot all show at once. See "The phone layout".

## Architecture: core, two draw lists, a host

```
 host (per platform)              editor core (platform independent)
 --------------------             ----------------------------------------------
 window or page, event loop  -->  input messages (touch, pointer, key, text)
 text measuring, text entry  <->  mid-ui tree: panels, widgets, layout (.mxui)
 file access                 <->  editor logic: selection, edit commands, undo
 backend that paints         <--  UI draw list + viewport draw list
                                  embedded world: mid-ecs, mid-app, mid-reflect,
                                  mid-nodes; the viewport builder projects it
```

1. **Editor core.** Platform-independent and testable in plain tests. It holds the world being
   edited, `mid-ui`'s tree and the screens, the editor's own logic, and a builder that turns the
   world into a viewport draw list for the current camera. Its inputs are messages; its outputs
   are two draw lists.
2. **Draw lists.** `mid-ui`'s `DrawCmd` list already exists and is the only render seam for the
   interface. The viewport gets a second list of screen-space primitives (lines, filled polygons,
   circles, text through the host). Whether that is the same type as `DrawCmd` or a separate one
   is open; sharing one set of primitives means each backend paints one thing.
3. **Host.** What runs the core and paints its output: a browser page, or an Android activity. A
   host turns platform events into input messages, measures and enters text, reads and writes
   files, and has a backend for the lists.

## The viewport

The proposal is a viewport that needs no GPU pipeline in its first version.

- **CPU projection.** The viewport builder projects the world with `mid-math`'s camera math and
  each entity's `GlobalTransform`, and emits lines and polygons in screen space: a ground grid,
  wireframe and flat-shaded shapes from `mid-geom` (boxes, spheres and capsules), bounds,
  the camera and light icons, selection outlines, and gizmo handles. `mid-geom` already has
  frustum culling, so off-screen entities are skipped.
- **Camera.** An orbit camera (turn, pan, zoom) driven by touch.
- **Picking.** A tap becomes a ray, and `mid-geom`'s raycasting picks an entity.
- **Gizmos.** Translate, rotate and scale handles, sized for fingers. Android's accessibility
  guidance recommends touch targets of at least 48 by 48 dp, about 9 mm.
- **Cost.** The work per frame is the entity count times the primitives per entity, on the CPU,
  and the A13's cores are small ones. This is **not measured**. A software backend and a bench
  against a scene of a few thousand entities answer it early.
- **What it cannot do.** Textures, lighting and meshes from files. They need the asset, image,
  mesh, shader and material layers that a GPU renderer needs (Decision 12), and they come later.

Bevy's own path is the reason not to copy its structure. In the fork (`Mid-D-Man/bevy`, commit
`f3b5346`), `bevy_render` has 26 other Bevy crates as non-optional dependencies, including
`asset`, `image`, `mesh`, `shader`, `material`, `camera` and `window`. Drawing a 2D sprite
(`sprite_render`, layer 12) needs `render` and `core_pipeline` on top. The editor widget set
(`feathers`, layer 14) needs `render`, `ui_render` and `scene`. A viewport that is lines and
polygons skips all of it.

## What `mid-ui` needs

| Need | In `mid-ui` today | Plan |
|---|---|---|
| Tree, layout (anchors, margins, row and column flow), hit testing | Built, tested (62 tests, sandbox only per its doc) | Use as is |
| Rectangles and clipping | `FillRect`, `PushClip`, `PopClip` | Use as is |
| `.mxui` screens | Loader built; node names, `kind`, `text` and binding declarations are read and kept | Use as is |
| **Text** | Not built | U1: `DrawCmd::Text` and a measuring trait the host implements (Decision 11) |
| **Events, focus, scrolling, touch gestures, text input** | Not built; hit testing only says whether the UI claims the pointer | U2. An editor needs drag, long press, two-finger pan and pinch, and a soft keyboard |
| **Widgets** (button, label, field, list, tabs, tree, property grid, splitter or drawer) | Not built | U3 |
| Data binding | Declarations are read, not acted on | With U3 or U4 |
| Images and icons | Not built | Not needed for v0; text, rectangles and primitives suffice |
| Animation | Not built (planned on `mid-anim`'s `Track`) | Later |

Text is the first blocker, as before: every panel is mostly text. Bevy's order for the same
stack is the guide (`docs/roadmap.md`, "Bevy's layers, recomputed"): text, then focus and input,
then widgets, then an editor kit.

## The phone layout

- **Portrait** (the A13's panel is 1080 by 2408 pixels): the viewport on top, a tab strip below
  it, and one panel at a time (Hierarchy, Inspector, Project, Console). The divider can be
  dragged.
- **Landscape:** the viewport on the left, the panels on the right.
- **Touch conventions (proposed):** one finger selects or orbits, two fingers pan and zoom, a
  long press opens the context actions, a drag on a gizmo handle moves the selection.
- The layout is `.mxui` and `mid-ui` flow layout, so a portrait and a landscape version are two
  screens over the same panels.

## Hosts

**Native Android first, by the maintainer's decision.** The host is a C file with raylib 6.0 and the
Rust core behind a C API (`docs/mid-editor.md`, `docs/mid-editor-ffi.md`). It follows the build the
maintainer already uses in `Tower-of-Babylon` (and `TerminalViscosity`): the NDK's `clang`, raylib built
from source for `armeabi-v7a`, a `NativeActivity` with no Java, `aapt`, `zipalign` and a debug key,
all in a workflow that is run by hand and leaves the APK as a downloadable artifact. A Tauri app
(`midnote`) already installs and runs on the phone, which is the evidence that a 32-bit ARM APK
built this way works there.

What was learned from those repositories and from raylib 6.0's source:

- **`InitWindow(0, 0)`** gives the display's native resolution, with no letterbox bars.
- **Rotation is not handled.** raylib 6.0's Android backend ignores `APP_CMD_CONFIG_CHANGED`, so the
  manifest locks portrait. The core lays out landscape (tested); following a rotation needs a
  change to raylib's Android backend or another native glue.
- **The default ES2 shader is `mediump`**, which put geometry a pixel or two off on this phone's GPU
  (`Tower-of-Babylon/docs/phone-rendering.md`). The host loads a highp copy.
- **Touch** arrives as the set of points that are down (`GetTouchPointCount`, `GetTouchPointId`,
  `GetTouchPosition`), and the host turns that into down, move and up events for the core.
- **raylib's `fopen` is redirected to the APK's assets** for reading, so the font and, later, project
  files bundled in the APK open with ordinary paths.

**Later hosts.** A web page (the existing Canvas 2D backend for the interface, a Canvas 2D painter
for the paint list, DOM input for the soft keyboard) and, if wanted, a WebView app. They consume the
same paint list, so the core does not change.

**Engine in wasm.** `mid-ecs` and `mid-app` pass `cargo check` for `wasm32-unknown-unknown`, and the
code splits so that on wasm32 the parallel batches run one after another on the calling thread
(`docs/mid-ecs.md`, "Parallel iteration"). The wasm32 tests have been type-checked, not run. It
matters for the web host only.

**32-bit ARM.** `mid-math` picks its scalar backend there (no SIMD backend exists for 32-bit ARM),
which the workflow's scalar-backend test job also exercises. `mid-ecs` has not been built or run for
a 32-bit target. That is the next thing to check before the ECS joins the editor.

## Where the engine runs

**Embedded first.** A viewport and a play mode need the world in the editor's own process, as in
Unity. So the editor embeds `mid-ecs` and `mid-app`, and the inspector reads and writes the world
in process through `mid-reflect`, with no protocol in between.

**Remote, later.** The remote protocol (`docs/roadmap.md`, Decision 8) still serves the command
line's world commands, and an editor attached to an engine running elsewhere (a CI job, a cloud
shell, a desktop). If it is exposed over a network:

- A token is required on any network transport; stdio needs none.
- A network transport binds to localhost by default.
- TLS comes from whatever fronts the endpoint (a tunnel or a proxy), not from the engine.
- Writes can be turned off entirely for a read-only endpoint.

## Project storage

The maintainer's direction is GitHub workflows for everything that reaches the phone, with scripts
where a workflow is not enough. For a project, that suggests:

- **Reading.** A project folder in the repository is bundled into the APK as assets by the build
  script, the way the font is, and opens with ordinary paths. This needs no storage permission.
- **Writing.** raylib's Android backend writes files to the app's private internal storage, so a
  saved scene lands there.
- **Getting edits back into the repository** is not designed. It is the open question below: a script
  that pulls the files off the phone, an in-app export that triggers a workflow, or something else.

The byte seam in `docs/roadmap.md`, Decision 13, is where reading and writing meet.

## Edits, undo and play mode

- **Edit commands.** Every change the editor makes goes through a command that knows how to
  apply and revert itself (set a field, spawn, despawn, reparent), written against `mid-reflect`'s
  `Value`. Undo and redo are a stack of those. Doing this from the start is cheaper than adding it
  after the inspector exists.
- **Play mode.** The world steps with `mid-app`'s schedule. Entering play mode snapshots the
  edited world as a `Value` tree, and leaving restores it. The snapshot depends on reflection
  covering every component in the scene, which is a requirement on component authors.

## Testing without a phone

Most of the editor is testable where the engine is tested. The screens are loaded, laid out and
turned into draw lists in plain tests, and a draw list can be compared with a stored one. The
viewport builder is a pure function from a world and a camera to a list, so the same applies. A
software backend paints a list into a buffer that a test can compare with a stored image. The
web host's painting functions are already tested in node against a recording fake
(`web/ui/render.test.mjs`). What cannot be tested here is each host, the pixels a real backend
produces, the cost on the phone, and anything about Android.

## Phases

The identifiers match `docs/roadmap.md`. "Built" means built and tested in the sandbox, not run on a
phone.

| Phase | What | Status |
|---|---|---|
| T1b | The ECS bridge for `mid-reflect` | Built (`docs/mid-ecs-reflect.md`), sandbox only. The editor uses it for its inspector and for play-mode restore (`docs/mid-editor.md`) |
| U1 | Text | Built as a seam: the core asks the host to measure text, and the host draws it with a bundled font |
| U2 | Events, focus, scrolling, touch gestures, text input | Partly built: tap, drag and pinch. No scrolling, focus or text entry |
| U3 | Widget kit | Not built as widgets. The panels draw their own buttons and rows |
| V1 | The viewport draw list | Built (the paint list). The host paints it with raylib; there is no separate software backend |
| V2 | The viewport builder | Built: grid, wireframe cubes, culling, 2D view. A bench exists (the 5000-cube chip), unrun on the phone |
| V3 | Picking and selection | Built, with the hierarchy and viewport selection in step |
| V4 | Gizmos | Not built |
| H1 | The web host | Not started, by decision |
| H2 | An Android APK spike | Built: the script and the workflow, unrun on GitHub |
| H3 | The native host | Built as the same host; unrun on a phone |
| E1 | Edit commands, undo and redo | Not built |
| E2 | Scenes load and save; play mode | Play mode is a demo (the cubes bob and turn). No scenes |

## Facts checked

Public facts, looked up on 2026-11-18. None was run by this project.

| Fact | Where |
|---|---|
| raylib has an Android platform target (`PLATFORM_ANDROID`: arm, i686, arm64, x86_64), built with the NDK, and its wiki has a guide for it | raylib's `src/Makefile` and the wiki page "Working for Android (on Linux)" |
| The `raylib-sys` crate's build script has an Android path that reads `ANDROID_NDK_HOME` and handles `aarch64-linux-android` and `armv7-linux-androideabi` | `raylib-sys` 5.5.1, `build.rs` |
| `raylib-rs` 6.0.0-rc.2 binds raylib 6.0 and states a minimum Rust of 1.85 | its README on docs.rs |
| GitHub's Ubuntu 24.04 runner image (20250720) lists an Android SDK and NDKs 26.3, 27.3 (default) and 28.2, with `ANDROID_NDK_ROOT` set; `sdkmanager` is preinstalled | the `actions/runner-images` release notes and issue 13467. The versions change, so the current image list is the thing to check |
| The Galaxy A13 4G (SM-A135F) has an Exynos 850 with a Mali-G52 (single-core configuration in the benchmark listings), a 1080 by 2408 panel, and 3 to 6 GB of memory. A device report for the same chip in the Galaxy M13 shows OpenGL ES 3.2. The A13 5G is a different chip (Dimensity 700, Mali-G57 MP2) | GSMArena, NotebookCheck, UL Benchmarks and AIDA64 listings. The maintainer's exact model is not recorded |
| The Galaxy A13 installs only `armeabi-v7a` (32-bit ARM) APKs; the build that runs there uses NDK r26d, API 29, build-tools 34.0.0 and no Gradle | `Tower-of-Babylon/docs/android.md` and `scripts/build_android.sh`, and `TerminalViscosity/docs/ANDROID.md`; `midnote` (Tauri) is the app known to install and run |
| raylib 6.0 on Android: `InitWindow(0, 0)` uses the display size; `APP_CMD_CONFIG_CHANGED` does nothing; touch points are readable by index and id | raylib 6.0 `src/platforms/rcore_android.c` (read, not run) |
| raylib's default ES2 vertex shader is `mediump`, and that misplaced geometry on this phone's GPU | `Tower-of-Babylon/docs/phone-rendering.md` |
| Android's accessibility guidance recommends touch targets of at least 48 by 48 dp (about 9 mm), 8 dp apart | Android Accessibility Help, "Touch target size", and the developer guide "Make apps more accessible" |
| In the Bevy fork at `f3b5346`, `bevy_render` has 26 non-optional Bevy dependencies; `bevy_android` has none | the fork's `Cargo.toml` files, computed the same way as `docs/roadmap.md`'s layer table |

## Not built, on purpose

Textured or lit 3D, meshes from files, images and icons, animation, a GPU backend, multi-user
editing, a plugin system for the editor, and accessibility beyond what `mid-ui` provides (Bevy's
`a11y` is a layer 4 crate and has no equivalent here yet).

## Open questions for the maintainer

1. How do edits get from the phone back into the repository (see "Project storage")?
2. Rotation: patch raylib's Android backend, use another native glue, or stay portrait?
3. Does the first real scene format use `mid-reflect`'s JSON, or DixScript (`docs/mid-nodes.md`)?
4. One shared set of draw primitives for `mid-ui`'s `DrawCmd` and the editor's paint list, or two?
   Today the editor converts `mid-ui`'s rectangles into its own list.
5. A name for the app. Nothing is called anything yet; the Android package is
   `com.midmanstudio.mideditor` and the label is "Mid Editor".

## Verification status

- Built and tested in the sandbox: `mid-editor` (77 tests), `mid-editor-ffi` (8), the C host on a desktop
  software renderer (eight screenshots, looked at), and the packaging half of the Android script (30
  checks with the real `aapt`, `zipalign` and `apksigner`). Details are in `docs/mid-editor.md`.
- Not verified: the NDK build, raylib's Android build, the armv7 Rust build, the APK on the phone,
  frame time and cost of the CPU-projected viewport on the phone, and `mid-ecs` on a 32-bit target.
