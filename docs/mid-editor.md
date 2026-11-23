# mid-editor: the editor core, its C API and the Android host

## Overview

This is the first runnable part of the engine app (`docs/mid-engine-app.md`): an editor shell with a
3D viewport, a 2D view, edit and play modes, a hierarchy, an inspector, a project panel and a
console, built to run on an Android phone and to be tested without one.

It has four parts:

- **`crates/mid-editor`**, the core. Safe Rust (`#![forbid(unsafe_code)]`), no windowing and no
  renderer. It holds the scene as a real `mid-ecs` world, lays the screen out with `mid-ui`,
  projects the world's boxes on the CPU with `mid-math`, turns touch events into taps, drags and
  pinches, and produces one flat paint list per frame.
- **`crates/mid-editor-ffi`**, a C API over the core (`include/mid_editor.h`), documented in
  `docs/mid-editor-ffi.md`. It is the only unsafe code, and every entry point catches panics.
- **`apps/editor-android`**, the host: a C file that opens a raylib window, feeds touches to the
  core, and paints the core's list. It is the same host on a desktop (for the self-test) and on
  Android.
- **`scripts/build_editor_android.sh` and `.github/workflows/editor-android.yml`**, which build the
  APK with no Gradle and no Java, the way the maintainer's other repositories already do.

**Status.** The first version of the app was built, packaged by the workflow and run on the
maintainer's phone (a Galaxy A13 4G, 60 fps with 12 cubes). Since then the core was connected to a
real world (see "`world.rs`"); that change is built and tested in the sandbox only, so the phone
has not seen it. See "Verification status".

### How to get the app

1. In the repository on GitHub, open **Actions**, pick **Editor Android APK**, press **Run workflow**.
2. When the run ends, open it and download the artifact **mid-editor-armv7-debug-apk** from the
   bottom of the page. It is a zip with `mid-editor-armv7-debug.apk` inside.
3. Install the APK on the phone (the browser or file manager needs "install unknown apps" once).
4. Later runs are signed with the same debug key (kept in the Actions cache), so they update the
   installed app in place. If the cache is ever cleared, uninstall the old app first.

The same workflow runs the Rust tests (host, scalar math backend, 32-bit x86) and the desktop
self-test first, and attaches the self-test's screenshots as **editor-host-selftest-screenshots**.
Set `run_tests` to `off` to go straight to the APK.

### What the app shows

- A top bar with **Edit** and **Play**, and **3D** and **2D**.
- The viewport. 3D: a perspective grid and twelve colored wireframe cubes; one finger orbits, two
  fingers zoom and pan, a tap selects a cube (yellow outline). 2D: a top-down orthographic view of
  the same scene. Frame statistics sit in the corner.
- A tab strip and a panel: **Hierarchy** (the world's entity tree, children indented under their
  parent; tap a row to select; drag up or down in the panel to scroll), **Inspector** (the selected
  entity's name and number, then each of its components and their fields, read through reflection;
  drag to scroll), **Project** (placeholder), **Console** (frame time, the time the core itself
  spent, paint item count, screen size, the last touch, the log, and chips that change the scene to
  12, 200, 1000 or 5000 cubes).
- **Play** runs the world's systems, which make the cubes bob and turn, and children move with
  their parent; **Edit** puts the world back as it was and holds still.

The 5000-cube chip is the bench that was missing: it shows what the CPU-projected viewport costs on
the phone (`docs/mid-engine-app.md`, "The viewport").

## Decisions

All proposals, for the maintainer to confirm or change.

- **A C host with raylib 6.0 and a Rust core behind a C API,** not `raylib-rs`, not a WebView, and
  not `cargo-apk`. It is the pipeline `Tower-of-Babylon` and `TerminalViscosity` already use, so
  the same NDK, the same `aapt` packaging and the same debug-key habit apply. It also matches the
  engine's FFI-first mandate.
- **`armeabi-v7a` only.** The maintainer's repositories record that this phone installs 32-bit ARM
  APKs (`Tower-of-Babylon/docs/android.md`). `midnote`, a Tauri app, already installs and runs on the
  phone, which is the evidence that the chain works. Every Rust crate in the editor path therefore
  has to be right on 32-bit, and the workflow tests on 32-bit x86 for that reason.
- **Portrait only.** raylib 6.0's Android backend does nothing on `APP_CMD_CONFIG_CHANGED`
  (`src/platforms/rcore_android.c`), so a rotation would leave the window size stale. The core lays
  out landscape too (tested); the host cannot follow a rotation yet.
- **`InitWindow(0, 0)`.** In raylib 6.0 a width and height of 0 means the display's own resolution,
  with no letterbox bars. The editor draws at native pixels.
- **A highp shader on OpenGL ES 2.** raylib's default ES2 shaders are `mediump`. On this phone's
  GPU that put geometry a pixel or two off (`Tower-of-Babylon/docs/phone-rendering.md`). The host
  loads raylib's default shaders with only the precision changed, as that game does.
- **Units follow the screen.** One UI unit is the short side divided by 411, so a unit is about one
  Android dp on this phone. Buttons are 40 units and rows 48, in line with the 48 dp guidance.
- **DejaVu Sans is bundled** for text, with its license (`apps/editor-android/assets/fonts`). The
  host measures text for the core through a callback, which is the text seam of
  `docs/roadmap.md`, Decision 11.
- **The scene is a `mid-ecs` world.** `EditorWorld` owns a `mid-app` `App` with `TransformPlugin`, so
  the hierarchy, the transform composition and the schedule are the engine's own, not a copy. The
  core's own code stays safe Rust; the dependencies are `mid-ecs` (feature `reflect`), `mid-app` and
  `mid-reflect`. The cost is that the editor path now includes the ECS on 32-bit ARM, which the
  `mid-ecs 32-bit` workflow checks and which the phone has not run.
- **Three editor components.** `Name`, `BoxShape` (half size and outline color) and `Bob` (the play
  animation) live in the editor, registered as `mid_editor::*`. `Name` stands in for the one
  `mid-nodes` will define (`docs/mid-nodes.md`, N3) and is replaced by it then.
- **Selection is an `Entity`.** It is cleared when the entity dies, and an inspector never draws a
  dead one.
- **The hierarchy is always fully expanded,** roots in the order they were created, each followed by
  its children in the order of its `Children`. It scrolls in whole rows. There are no folds yet.
- **The inspector reads through reflection and does not edit.** Every registered component an
  entity holds becomes a header line and one line per field; a struct of plain numbers (a vector, a
  quaternion) is one line; an entity reference shows as the entity's name; a color shows as hex.
  The two derived transforms (`GlobalTransform`, `GlobalTransformLWC`) show as one line with their
  position, because their matrices are large, read only and would fill the few rows a phone panel
  has. Components that are not registered are counted on a last line. Editing waits for E1.
- **Play keeps a snapshot, and Edit restores it.** Entering play records every registered component
  of every entity except the two transform results (recomputed afterwards). Leaving play puts back
  each value, removes components play added, despawns entities play spawned, then re-attaches
  parents child by child in each parent's original order, after detaching every parent first so a
  half-restored tree cannot look like a cycle. An entity despawned during play is counted as lost
  and not brought back: that needs entity remapping (`docs/mid-ecs-reflect.md`, "Not built yet").
- **Design direction.** The maintainer sent a screenshot of the flecs explorer and asked for that
  combined with the Unity editor. What it shows, and where this editor stands:

  | In the flecs explorer | Here |
  |---|---|
  | An entity tree with a dropdown and a search field | The tree is there; no search, no dropdown |
  | A selected entity with its name and number | The inspector title, `name  #n` |
  | Components grouped by module, each group folding | Components listed flat, with no module groups or folds |
  | Editable numeric fields per component | Read-only text |
  | An Add Component button | Not built |
  | A left icon rail for tools (entities, query, charts, tables) | Not built |
  | A status bar with the version | The version is in the top bar |

  The Unity half (scene and game views, play controls, component foldouts) points the same way:
  folds, editing and a rail are the next steps for the panels, and they wait for `mid-ui` to have
  text, events and widgets (roadmap U1 to U3) rather than being hand-drawn here.
- **2D is a top-down orthographic view of the ground plane.** There are no 2D nodes yet.
- **No Gradle, no Java.** `android:hasCode="false"`, a `NativeActivity`, and the APK assembled with
  `aapt`, `zipalign` and `apksigner`.

## Modules

The C API (`crates/mid-editor-ffi` and its header) is in `docs/mid-editor-ffi.md`.

### `lib.rs`

**What it does:** the crate documentation, the module list (`camera`, `components`, `editor`,
`input`, `inspect`, `paint`, `scene`, `stats`, `text`, `viewport`, `world`) and the re-exports
(`Editor`, `Mode`, `Tab`, `Phase`, `PaintItem`, `PaintList`, `TextMeasure`, `FixedWidth`). It
forbids `unsafe`.

**Tests:** through the other modules.

### `paint.rs`

**What it does:** the paint list. `PaintItem` is a 36-byte `repr(C)` record: a kind (rectangle, clip
push, clip pop, line, text), a color as `0xRRGGBBAA`, four coordinates, one extra number (radius,
width or text size) and a range into a text pool. `PaintList` holds the items and the pool, with
`rect`, `push_clip`, `pop_clip`, `line` and `text`, and `text_of` to read a text item back.

**Decisions:** a flat list with no pointers, so a C host reads it directly. A text item holds a
range into one pool, so the list needs no per-item allocation.

**Tests:** 5 unit tests, including the layout the C header declares.

### `text.rs`

**What it does:** `TextMeasure`, the trait a host fills in, `FixedWidth` for tests, and `fit`, which
shortens text with `...` to fit a width.

**Tests:** 4 unit tests.

### `stats.rs`

**What it does:** a rolling window over the last 60 frame times (average, worst, frames per second)
and a moving average of the core's own cost per frame.

**Decisions:** bad values (NaN, negative, infinite) are ignored, and a pause is recorded as at most a
second, so one stall does not wreck the display.

**Tests:** 5 unit tests.

### `scene.rs`

**What it does:** the demo scene. `spawn_demo(world, n)` spawns `n` cubes on a square grid, 2.5
units apart, each with a name, a local and a composed transform, a `BoxShape` and a `Bob`. The first
cubes form a small hierarchy: `Cube.001` and `Cube.002` hang under `Cube`, and `Cube.003` under
`Cube.002`, so the tree and the composition have something to show. A child's local position is its
grid position minus its parent's, so building the hierarchy moves nothing. `PALETTE` and `EDGES`
(the twelve edges of a box, as pairs of corner indices) are here too.

**Decisions:** the cube palette must never contain the selection color (see "Fixes and Problems").
Only the first four cubes are nested, so the 200, 1000 and 5000 cube scenes cost what a flat scene
does apart from the world itself.

**Tests:** 7 unit tests: the grid, the names, the nesting, small counts, composition onto the grid,
the palette and the edges.

### `components.rs`

**What it does:** `Name`, `Rgba`, `BoxShape` and `Bob`, with their reflection, and `registry()`,
which returns a registry that knows the engine's components (`register_core_components`) and these.

**Decisions:** the names are `mid_editor::Name` and so on, so a scene file written later does not
clash with `mid-nodes`. `Rgba` is its own type so the inspector can show a color as hex without
guessing from a number.

**Tests:** 2 unit tests: every component is registered with type data, and each round-trips through
a value.

### `world.rs`

**What it does:** `EditorWorld`, the world the editor shows. It owns the `App`, the registry, the
boxes to draw, the hierarchy rows and the play-mode snapshot.
- `Drawable` is one box read from the world: the entity, its composed transform, its half size, its
  half extents in world units, and its color. `center`, `radius` and `corners` serve the viewport.
- `refresh` composes transforms and rebuilds the boxes and the root list only when something
  changed, so an Edit frame with nothing happening costs a flag test. `world_mut` marks everything
  stale.
- `hierarchy_rows(first, count)` walks the tree in order without building a full list, so a window
  of a 5000-entity scene is cheap to ask for.
- `enter_play`, `step(dt)` and `exit_play` run play mode. `step` runs one `App` update, which moves
  every `Bob` entity and composes transforms. `exit_play` returns a `RestoreReport` (restored,
  removed, lost, failed).
- `inspect(entity)` gives the inspector rows.

**Decisions:** those under "Decisions" above. Also: a mismatch of handles is possible because every
`World` allocates entity numbers from zero, so an `Entity` from one world can equal one in another;
nothing here mixes worlds, and the viewport tests use handles with a number the demo does not use.

**Tests:** 21 unit tests: rows, paging, tree order, drawables and their scale, play, restore in each
of its cases (values, spawned entities, added and removed components, a reparented tree, a despawned
child), the snapshot being taken once, and the inspector's rows.

### `inspect.rs`

**What it does:** `rows(world, registry, entity)` reads an entity through `World::reflect_entity` and
turns each component into `Row`s: a header line, then the fields at one more level of indent. It
follows the registered type information (`Kind`, `Ty`), so a new component shows up with no code
here. A value is one line when it is a number, text, a boolean, a list of such, an entity, a color,
or a struct of at most four plain numbers; anything else gets a line of its own and its fields
below it, to a depth of six.

**Tests:** 12 unit tests: dead entities, headers and fields, parents shown by name, children listed,
vectors on one line, nested structs, a derived transform on one line, colors, an unregistered
component counted, a dead reference shown as `dead`, long lists cut with a count, and the depth
bound.

### `camera.rs`

**What it does:** the cameras. `Orbit` (yaw, pitch, distance, target) turns, zooms and pans along the
ground. `Ortho` is the top-down 2D camera. `Perspective` projects a world point or line to the screen,
cutting a line at the near plane. `Area` is a screen rectangle.

**Decisions:** the view matrix comes from `mid-math`'s `Mat4::look_at_rh` and `transform_point`.
`transform_point` exists on every `mid-math` backend (scalar, SSE2, NEON, `coresimd`);
`transform_point3` exists only on the scalar one, so the code uses the first.

**Tests:** 11 unit tests.

### `viewport.rs`

**What it does:** `Viewport` holds the view (3D or 2D) and both cameras. `paint` draws the grid, the
boxes and the selection into a paint list; `pick` finds the nearest box under a touch and returns
its `Entity`; `drag` and `pinch` move the active camera. It reads `Drawable`s from the world, so a
box is drawn with its full composed transform (scale and turn included); the 2D view draws each box
as an axis-aligned rectangle of its world extents.

**Decisions:**
- Cubes behind the camera, past 300 units or off screen are skipped, and a cube smaller than a few
  pixels is drawn as a dot, so a large scene stays cheap.
- Picking uses a touch radius of 20 units at least, so a finger can hit a small cube.

**Tests:** 14 unit tests, including a scaled box picked by its scaled extent.

### `input.rs`

**What it does:** `Tracker` turns raw pointer events (down, move, up, cancel, with an id) into
gestures: `Tap`, `Drag` (with where it started), `Pinch` (a ratio and the midpoint's movement) and
`DragEnd`.

**Decisions:**
- A touch is a tap when it lifts within 0.6 seconds and has not moved past the slop (10 units).
- Two fingers down means no tap, and lifting one of two never becomes a tap or a new drag.
- At most five pointers are tracked. `clear` forgets them all, for a host that loses track.

**Tests:** 10 unit tests.

### `editor.rs`

**What it does:** `Editor`, the whole core. It builds the `mid-ui` tree for portrait or landscape,
paints the top bar, viewport, tab strip and panel, keeps the hotspots for the buttons and rows, and
routes gestures: a tap on a button or row acts, a tap in the viewport picks, a drag that began in the
viewport orbits or pans, a drag that began in the panel scrolls the list, a pinch in the viewport
zooms. `frame` advances time, steps the world in play mode (or refreshes it in edit mode) and
rebuilds the paint list, and times its own work. Switching to Play snapshots the world; switching
to Edit restores it and logs what was restored.

**Decisions:**
- All text is ASCII, because the bundled font is loaded with the ASCII set.
- A drag keeps controlling the viewport after the finger leaves it.
- `cancel_pointers` exists because a lost touch-up would otherwise leave a phantom finger.
- The panel scrolls in pixels internally and shows whole rows. Dragging in the viewport still
  orbits and a drag in the panel never does. Changing tab, or changing the cube count, puts the
  scroll back at the top. A tap on a row after scrolling selects that row.
- Changing the cube count while playing builds the new world already playing, so Edit still
  restores it.

**Tests:** `tests/editor.rs`, 33 tests through the public interface: the layouts, every tab,
selection from the hierarchy and from the viewport, orbit, pinch, 2D, play and edit, the cube-count
chips, the statistics, text that must fit the screen at several sizes, indentation, scrolling to the
end and back, selecting after a scroll, the inspector's components and scroll, play and restore with
the selection kept, and a randomized run (400 events and frames at five screen sizes) checking every
paint value is finite, clips are balanced and text is valid. Three unit tests in `editor.rs` cover a
selection whose entity dies, which the public interface cannot cause.

### `apps/editor-android/main.c`

**What it does:** the host. It opens a raylib window (`InitWindow(0, 0)` on Android), loads the font
and, on OpenGL ES 2, the highp shader, creates the editor with a measure callback, and each frame
turns raylib's touch points into down, move and up events, calls `mid_editor_frame`, and paints the
list: rounded rectangles, scissor clips, thick lines and text. With `-DEDITOR_SELFTEST` it instead runs
a scripted session on a desktop, tapping by label, and writes screenshots.

**Decisions:**
- The host is thin on purpose. It owns no editor state; the core decides everything.
- Touch is derived from the set of points raylib reports each frame (a new id is a down, a changed
  position a move, a vanished id an up).
- The self-test taps by reading the paint list for a label, so it exercises the text pool through
  the C ABI too.

**Tests:** compiled with `-Wall -Wextra` for the desktop and `-Werror=implicit-function-declaration`
for Android; run through `scripts/editor_host_selftest.sh`.

### `apps/editor-android/AndroidManifest.xml`

**What it does:** a `NativeActivity` with `android:hasCode="false"`, library name `main`, portrait,
OpenGL ES 2, no permissions. The script fills in the SDK levels and the version.

**Tests:** the script check reads the built APK with `aapt dump badging`.

### `scripts/build_editor_android.sh`

**What it does:** five stages, each runnable alone: `raylib` (raylib 6.0 from source for Android
armv7), `editor` (`cargo rustc` of `mid-editor-ffi` as a static library for
`armv7-linux-androideabi`, then the header check), `compile` (the host to an object file), `link`
(`libmain.so`, then checks that it exports `ANativeActivity_onCreate` and the editor core, needs
only phone libraries, and is a 32-bit ARM ELF file) and `package` (manifest, assets, `aapt`,
`zipalign`, a debug signature).

**Decisions:**
- It follows `Tower-of-Babylon/scripts/build_android.sh`: NDK r26d, API 29, build-tools 34.0.0, and
  the lessons recorded there (the Rust stage runs as a child process so a failure names its line;
  `llvm-nm`'s exit status is ignored because the NDK's is older than rustc's LLVM and complains
  about the bitcode in Rust static libraries).
- The Rust stage resolves the whole workspace with a fresh lockfile, because the repository commits
  none. The editor crates have no outside dependencies, so there is nothing to pin.
- `EDITOR_SKIP_ELF_CHECK` and `EDITOR_EXTRA_NEEDED_OK` exist only for the script's own test.

**Tests:** `scripts/tests/editor_android_script_check.sh`; `shellcheck` is clean.

### `scripts/editor_host_selftest.sh`

**What it does:** builds raylib for the desktop, the Rust library and the real host, runs the scripted
session under Xvfb with Mesa's software renderer, and checks that eight screenshots were written:
the initial state, a hierarchy selection, the inspector, the console, 1000 cubes, the 2D view, play
mode and an orbited camera.

**Decisions:** the screenshots are the review artifact. They show what the host actually drew, which
a passing test does not.

**Tests:** it is the test. `shellcheck` is clean.

### `scripts/tests/editor_android_script_check.sh`

**What it does:** runs the build script against a stand-in NDK (the host's `gcc` behind the NDK's
compiler names, ignoring the Android-only link flags), a tiny library standing in for raylib's
entry point, a Rust stage built for the host's own triple, and the real `aapt`, `zipalign` and
`apksigner`. It checks the success path (30 checks: the APK exists, its package, version, SDK levels,
launcher activity, native code, the font at the path the host opens, a valid signature, the same key
on a second run, version defaults) and the failures (no NDK, no SDK, a missing tool, an unknown
stage, a header function the library lacks and the reverse, a library the phone does not have, code
that is not 32-bit ARM).

**Tests:** it is the test.

## Verification status

Sandbox, rustc 1.97.1, x86_64.

- `mid-editor`: 127 tests (94 unit, 33 integration), `mid-editor-ffi` 8 and `mid-ui` 62, all passing
  with the default and the scalar `mid-math` backend. `mid-ecs` passes 1,240 tests (1,276 with
  `reflect`, including the two new `World::entities` tests), `mid-app` 49.
- Mutation checks over the new code: 20 hand-made mutants (restore skipping its detach pass, extras
  not despawned, added components kept, paging that never counts, children reversed, extents
  ignoring scale, corners ignoring the transform, play entered twice, the panel not scrolling, scroll
  kept across tabs, a count change not entering play, a dead selection kept, play that never steps,
  Edit that does not restore, selecting a dead entity, an inspector drawing one, a wrong dead-entity
  label, a derived transform expanded, and picking by the unscaled extent). All are caught. The first run missed four, which got
  tests; one more (an unclamped scroll value) cannot be seen from outside and was left.
- Cost, measured in the sandbox in release mode on x86_64, not on the phone, with the default
  camera, so only cubes in view cost anything to draw (milliseconds per call):

  | Cubes | Edit frame, old / new | Play frame, old / new | Enter play | Exit play |
  |---|---|---|---|---|
  | 200 | 0.009 / 0.009 | 0.011 / 0.028 | 0.56 | 0.27 |
  | 1000 | 0.023 / 0.023 | 0.031 / 0.114 | 2.5 | 1.9 |
  | 5000 | 0.082 / 0.083 | 0.116 / 0.616 | 12.5 | 10.8 |

  "Old" is the flat scene. The edit frame is unchanged because the boxes are cached. A play frame
  costs more, mainly the schedule, the transform composition and the rebuild of the box list; the
  phone's earlier numbers say the host's drawing, not the core, was what limited it. Entering and
  leaving play at 5000 cubes is a one-time pause that will be several times longer on the phone.
  Each of those timings includes two frames of the tap that triggers them.
- **On the phone, the earlier flat-scene version** (Galaxy A13 4G, screen reported as 1080 by 2338,
  one unit 2.63 pixels, shader line `highp ES2`): the first workflow run built an APK that installed
  and ran, and tap, drag and pinch worked. Measured from the maintainer's screenshots, in 3D play
  mode:

  | Scene | fps | Frame ms | Core ms | Paint items |
  |---|---|---|---|---|
  | 12 cubes (also in edit) | 60 | 16.7 | 0.23 | 213 |
  | 200 | 58.8 | 17.0 | 0.54 | 2,457 |
  | 1000 | 37.1 | 26.9 | 1.59 | 11,877 |
  | 5000, zoomed out | 8.9 | 112.0 | 7.91 | 56,073 |
  | 5000, zoomed in | 55.8 | 17.9 | 1.5 | 5,175 |
  | 5000 in 2D | 11.3 | 88.5 | 1.39 | 25,123 |

  The core costs about 0.14 microseconds per paint item; the rest of the frame is the host drawing,
  about 2 microseconds per item (3.5 in 2D, with five items per cube). So raylib's line and
  rectangle submission is the limit, not the projection and not the ECS. The usual fixes, when they
  are needed: batch the lines into one vertex buffer, merge distant cubes, or the GPU backend step
  (`docs/roadmap.md`, Decision 12).
- The desktop self-test ran again against the world-backed core (raylib 6.0 on a software renderer,
  the real C host and header, eight screenshots written, three looked at: the hierarchy with
  indentation and a selection, the inspector, and play mode at 1000 cubes with turning boxes). The
  script check of the Android packaging was not rerun; the host and the C API did not change.
- **Not verified:** any of the world-backed editor on the phone, including the Rust build and link
  for 32-bit ARM with `mid-ecs` and `rayon` in the editor's static library; how the new panels look
  on the screen; the 32-bit x86 job of `editor-android.yml` with the new dependencies; frame time on
  the phone in play mode at 1000 and 5000 cubes.

## CI and Workflows

- `.github/workflows/editor-android.yml`, run by hand from the Actions tab. Jobs: `rust-tests` (host,
  scalar backend, 32-bit x86), `host-selftest` (the desktop session, screenshots kept), and `apk`
  (Java 17, the Android SDK with build-tools 34.0.0 and platform 34, NDK r26d, Rust with the armv7
  target, the debug-key cache, then `scripts/build_editor_android.sh`; the APK and the logs are kept).
  The version name is `0.1.<run number>` and the version code is the run number.
- `scripts/tests/editor_android_script_check.sh` is not in the workflow; it needs a raylib source
  directory for its headers and the Ubuntu packaging tools, and is run by hand.

## Not built yet

- Editing the world: edit commands, undo and redo, gizmos, text entry. The inspector only reads.
- Folds in the hierarchy, a search field, component groups, an Add Component button and a tool
  rail (see "Design direction").
- Fixes for what the phone showed: a black band about 70 pixels high above the top bar (the status
  bar's area is not drawn) and the navigation bar covering the bottom of the console (the host needs
  a bottom inset or immersive mode). Only the Console tab was seen on the phone; the others have not
  been. Portrait only, because raylib 6.0's Android backend ignores a configuration change.
- Restoring an entity that play despawned (needs entity remapping in the ECS bridge).
- A real `Name` from `mid-nodes`, and components for something other than cubes.
- Scenes and project files; the project panel is a placeholder.
- Rotation (a patch to raylib's Android backend, or a different native glue).
- Textures and lighting (the GPU backend step, `docs/roadmap.md`, Decision 12).

## Fixes and Problems

### `scene.rs`

- **A cube color matched the selection color.** The fourth palette color was the same yellow as the
  selection outline, so that cube would have looked selected all the time. A unit test that counted
  selection-colored lines caught it. The palette now has an orange there, and a test checks that no
  palette color equals the selection color.

### `viewport.rs` and `world.rs`

- **Handles from two worlds collide.** The first test of a selection that is not among the boxes
  used `World::new().spawn()` for a stranger. That handle is the same as the demo world's first
  cube, because every world numbers its entities from zero, so the cube was drawn as selected and the
  test failed. The test now takes a handle with a number the demo does not use.
- **A test premise about the panel.** Three of the first integration tests for scrolling assumed
  twelve rows fit the panel, that `Name` is the first inspector row, and that a cube count chip does
  not step the world. On the phone-sized screen about five rows fit, `LocalTransform` comes first,
  and a new world already plays when the chip is tapped. The code was right and the tests were
  changed.

### `camera.rs`

- **`Mat4::transform_point3` is on one backend only.** The first version used it and compiled on
  the scalar backend alone. `transform_point` exists on all four, so the code uses that. This is the
  kind of difference 32-bit ARM would have exposed on the first CI run.

### `input.rs`

- **A pointer that never lifts.** A test that left two fingers down made the next tap do nothing,
  because the tracker still had a second finger. A real host can lose a touch-up too, so the tracker
  has `clear` and the core and the C API have `cancel_pointers`.

### `scripts/build_editor_android.sh`

- **The library check refused the host's libraries.** The script's own test linked against the PC's
  `libm.so.6`, which the phone-library whitelist rightly rejects. A test-only override
  (`EDITOR_EXTRA_NEEDED_OK`) was added, and the failure path is still tested with a library that is
  not on the list.
- **The Rust stage's child process printed "done".** It reported an APK path before any APK existed.
  Only the parent reports now.
