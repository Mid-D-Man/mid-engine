# The engine app: a phone-first editor

**Status: design only. Nothing here is built or verified, apart from what `mid-ui` and its web
playground already do and the public facts listed under "Facts checked".** First written on
2026-11-16 as a tools app (a console, an entity tree, an inspector). Rewritten on 2026-11-18,
after the maintainer said what the app is. "The engine app" is a working title; no crate,
directory or name exists yet.

## Overview

The app is an editor in the sense of the Unity editor: a **viewport** that shows the scene, a
**hierarchy** of the entities in it, an **inspector** for the selected entity's components, and
**project** panels. It runs on an Android phone, because a laptop is not available, and its
interface is built on `mid-ui`. These three points come from the maintainer. Everything else in
this document is a proposal.

Assumptions that are written down so they can be corrected:

- The editor edits scenes and can also run them (a play mode, as in Unity). Whether play mode is
  in the first version is open.
- The viewport shows a 3D scene, because the transform and geometry crates are 3D. A 2D scene is
  shown through the same view, from above. Real 2D nodes wait, as `docs/mid-nodes.md` proposes.
- The first version is usable when a project opens, the scene shows in the viewport, a tap in the
  viewport or the hierarchy selects an entity, the inspector edits its components, a gizmo moves
  it, undo works, and the scene saves.

Constraints that shape it:

- The interface is `mid-ui`, not a borrowed toolkit.
- The phone cannot build the engine, so everything the app needs is built by a workflow, and a
  change reaches the phone through a pushed commit.
- The phone is a Galaxy A13. The 4G model has an Exynos 850 with a Mali-G52 GPU, an entry-level
  part (see "Facts checked"; the exact model is not recorded). The viewport has to be cheap.
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

| | Host | What it needs | Trade-offs |
|---|---|---|---|
| A | **A web page** | A wasm build of the core, the existing Canvas 2D backend for the UI list, a Canvas 2D painter for the viewport list, DOM input for events and the soft keyboard, `fetch` for files | The fastest loop available on a phone: push, the site deploys, reload. No Android build chain. The engine must run inside wasm (below). A page served over HTTPS cannot open an insecure WebSocket except to localhost |
| B | **An Android WebView app** | A small Android project that hosts the page, and a CI job that builds the APK | Installable and offline. Adds the Android build chain. A viewport inside a WebView is as fast as the WebView allows |
| C | **A native Android app** | The core as a native library built with the NDK, an activity, a surface, input and text entry from the platform, a backend that paints the two lists, file access | Best fit eventually. Needs the most unverified work (below) |

**Proposal:** A first, because it gives the editor core a real screen and a real touch loop
soonest, and a spike for C in parallel, because the Android build chain is the largest unknown
and only a workflow is needed to find out. B is the cheap way to an installable app if C is late.
The core and the screens do not change between hosts.

**Engine in wasm.** `mid-ecs` and `mid-app` pass `cargo check` for `wasm32-unknown-unknown`, and
the code splits so that on wasm32 the parallel batches run one after another on the calling
thread (`docs/mid-ecs.md`, "Parallel iteration"). The wasm32 tests have been type-checked, not
run, so whether a world runs in a page is unverified.

**Native Android.** Decision 12 covers what paints. The documented route for raylib is its
Android platform target, built with the NDK and a `NativeActivity`. For a Rust core, the binding
`raylib-sys` has an Android path in its build script. Soft-keyboard text entry on that route has not
been checked.

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

A project is a directory of scene files and assets. Where it lives on the phone is open.
App-private storage is the simplest. Shared storage goes through Android's storage access
framework, which a native app reaches through the platform. Getting a project in and out of the
phone, and of the GitHub repository it is worked from, is a real question: see "Open questions".
The file access itself is the byte seam in Decision 13.

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

The identifiers match `docs/roadmap.md`.

| Phase | What | Needs | Done when |
|---|---|---|---|
| T1b | The ECS bridge for `mid-reflect` | an `mid-ecs` API to enumerate an entity's components | An entity's components are read and written by name from the registry |
| U1 | Text in `mid-ui` | `mid-ui` | A `.mxui` screen with text lays out and draws in the browser |
| U2 | Events, focus, scrolling, touch gestures, text input | U1 | A screen can be tapped, dragged, scrolled and typed into |
| U3 | Widget kit, including tree, property grid, tabs and a splitter | U2 | The panels' widgets exist |
| V1 | The viewport draw list and a software backend | nothing | A list of primitives paints into a buffer that a test compares |
| V2 | The viewport builder: camera, grid, shapes | V1, `mid-math`, `mid-geom` | A scene projects to a list; a bench says what it costs |
| V3 | Picking and selection | V2, `mid-nodes` phase 1 | A tap selects an entity, and hierarchy and inspector follow |
| V4 | Gizmos | V3, U2 | A drag on a handle moves the selected entity |
| H1 | The web host | U1, input messages | The editor page runs on the phone with an embedded world |
| H2 | An Android APK spike | nothing from the engine | A CI-built APK shows a `mid-ui` draw list and reports touch |
| H3 | The native host | H2, V1, U2 | The same editor runs natively |
| E1 | Edit commands, undo and redo | T1b | Every inspector edit can be undone |
| E2 | Scenes load and save; play mode | `mid-nodes`, scene format | The scene round-trips; play mode enters and leaves |

## Facts checked

Public facts, looked up on 2026-11-18. None was run by this project.

| Fact | Where |
|---|---|
| raylib has an Android platform target (`PLATFORM_ANDROID`: arm, i686, arm64, x86_64), built with the NDK, and its wiki has a guide for it | raylib's `src/Makefile` and the wiki page "Working for Android (on Linux)" |
| The `raylib-sys` crate's build script has an Android path that reads `ANDROID_NDK_HOME` and handles `aarch64-linux-android` and `armv7-linux-androideabi` | `raylib-sys` 5.5.1, `build.rs` |
| `raylib-rs` 6.0.0-rc.2 binds raylib 6.0 and states a minimum Rust of 1.85 | its README on docs.rs |
| GitHub's Ubuntu 24.04 runner image (20250720) lists an Android SDK and NDKs 26.3, 27.3 (default) and 28.2, with `ANDROID_NDK_ROOT` set; `sdkmanager` is preinstalled | the `actions/runner-images` release notes and issue 13467. The versions change, so the current image list is the thing to check |
| The Galaxy A13 4G (SM-A135F) has an Exynos 850 with a Mali-G52 (single-core configuration in the benchmark listings), a 1080 by 2408 panel, and 3 to 6 GB of memory. A device report for the same chip in the Galaxy M13 shows OpenGL ES 3.2. The A13 5G is a different chip (Dimensity 700, Mali-G57 MP2) | GSMArena, NotebookCheck, UL Benchmarks and AIDA64 listings. The maintainer's exact model is not recorded |
| Android's accessibility guidance recommends touch targets of at least 48 by 48 dp (about 9 mm), 8 dp apart | Android Accessibility Help, "Touch target size", and the developer guide "Make apps more accessible" |
| In the Bevy fork at `f3b5346`, `bevy_render` has 26 non-optional Bevy dependencies; `bevy_android` has none | the fork's `Cargo.toml` files, computed the same way as `docs/roadmap.md`'s layer table |

## Not built, on purpose

Textured or lit 3D, meshes from files, images and icons, animation, a GPU backend, multi-user
editing, a plugin system for the editor, and accessibility beyond what `mid-ui` provides (Bevy's
`a11y` is a layer 4 crate and has no equivalent here yet).

## Open questions for the maintainer

1. Is play mode part of the first version, or is the first version an editor that only edits
   and saves?
2. Is a 3D-capable viewport right (a projected 3D scene, wireframe and flat shapes first), or
   should the first viewport be a true 2D view?
3. Web host first with an Android spike in parallel (proposed), or straight to the native app?
4. How should a project get in and out of the phone, and into the repository it is worked from:
   exported archives, files in app storage that a workflow pulls, or something else?
5. Which exact model is the phone, 4G or 5G? The GPU differs.
6. Text: let the host draw and measure text (proposed, through the measuring seam), or build text
   from the start?
7. One shared set of draw primitives for the interface and the viewport, or two lists?
8. A name for the app. Nothing is called anything yet.

## Verification status

None. This is a design document. No code, no tests, no CI. The sandbox can build and test the
core, the screens, the viewport builder and a software backend; it cannot test Android, a
WebView, a native host or a phone's GPU.
