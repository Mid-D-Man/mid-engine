# mid-editor: the editor core, its C API and the Android host

## Overview

This is the first runnable part of the engine app (`docs/mid-engine-app.md`): an editor shell with a
3D viewport, a 2D view, edit and play modes, a hierarchy, an inspector, a project panel and a
console, built to run on an Android phone and to be tested without one.

It has four parts:

- **`crates/mid-editor`**, the core. Safe Rust (`#![forbid(unsafe_code)]`), no windowing and no
  renderer. It lays the screen out with `mid-ui`, projects a scene on the CPU with `mid-math`,
  turns touch events into taps, drags and pinches, and produces one flat paint list per frame.
- **`crates/mid-editor-ffi`**, a C API over the core (`include/mid_editor.h`), documented in
  `docs/mid-editor-ffi.md`. It is the only unsafe code, and every entry point catches panics.
- **`apps/editor-android`**, the host: a C file that opens a raylib window, feeds touches to the
  core, and paints the core's list. It is the same host on a desktop (for the self-test) and on
  Android.
- **`scripts/build_editor_android.sh` and `.github/workflows/editor-android.yml`**, which build the
  APK with no Gradle and no Java, the way the maintainer's other repositories already do.

**Status.** Built and tested in the sandbox: the core, the C API, the host on a desktop software
renderer, and the packaging half of the Android script. **Not run:** anything with the Android NDK,
raylib's Android build, the Rust build for 32-bit ARM, the APK on a phone. The first workflow run
is the first time those steps happen. See "Verification status".

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
- A tab strip and a panel: **Hierarchy** (tap a row to select), **Inspector** (the selected cube's
  position, rotation and size), **Project** (placeholder), **Console** (frame time, the time the core
  itself spent, paint item count, screen size, the last touch, the log, and chips that change the
  scene to 12, 200, 1000 or 5000 cubes).
- **Play** makes the cubes bob and turn; **Edit** puts them back and holds still.

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
- **A demo scene, not the ECS.** The scene is a flat list of boxes in the core. Connecting the real
  world is next: the ECS bridge exists (`docs/mid-ecs-reflect.md`), the editor does not use it yet.
- **2D is a top-down orthographic view of the ground plane.** There are no 2D nodes yet.
- **No Gradle, no Java.** `android:hasCode="false"`, a `NativeActivity`, and the APK assembled with
  `aapt`, `zipalign` and `apksigner`.

## Modules

The C API (`crates/mid-editor-ffi` and its header) is in `docs/mid-editor-ffi.md`.

### `lib.rs`

**What it does:** the crate documentation, the module list and the re-exports (`Editor`, `Mode`,
`Tab`, `Phase`, `PaintItem`, `PaintList`, `TextMeasure`, `FixedWidth`). It forbids `unsafe`.

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

**What it does:** the demo scene. `Scene::demo(n)` lays `n` cubes on a square grid, `Scene::pose(t)`
bobs and turns them for play-mode time `t` (and puts them back at `t == 0`), `Scene::corners` gives a
box's eight corners after its yaw, and `EDGES` lists the twelve edges.

**Decisions:** the cube palette must never contain the selection color (see "Fixes and Problems").

**Tests:** 7 unit tests.

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
cubes and the selection into a paint list; `pick` finds the nearest cube under a touch; `drag` and
`pinch` move the active camera.

**Decisions:**
- Cubes behind the camera, past 300 units or off screen are skipped, and a cube smaller than a few
  pixels is drawn as a dot, so a large scene stays cheap.
- Picking uses a touch radius of 20 units at least, so a finger can hit a small cube.

**Tests:** 13 unit tests.

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
viewport orbits or pans, a pinch there zooms. `frame` advances time, steps play mode and rebuilds the
paint list, and times its own work.

**Decisions:**
- All text is ASCII, because the bundled font is loaded with the ASCII set.
- A drag keeps controlling the viewport after the finger leaves it.
- `cancel_pointers` exists because a lost touch-up would otherwise leave a phantom finger.

**Tests:** `tests/editor.rs`, 23 tests through the public interface: the layouts, every tab, selection
from the hierarchy and from the viewport, orbit, pinch, 2D, play and edit, the cube-count chips, the
statistics, text that must fit the screen at several sizes, and a randomized run (400 events and
frames at five screen sizes) checking every paint value is finite, clips are balanced and text is
valid.

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

- `mid-editor`: 77 tests (54 unit, 23 integration). `mid-editor-ffi`: 8 tests. `mid-ui` with them: 147
  tests pass on rustc 1.97.1 and on 1.75.0, and on the default and the scalar `mid-math` backend.
  `clippy --no-deps --all-targets` and `rustfmt` are clean.
- The desktop self-test ran: raylib 6.0 (desktop), the Rust core through the C header, the C host, a
  scripted session, eight screenshots, all looked at.
- The script check: 30 of 30, with the real packaging tools.
- **Not verified:** the NDK build of raylib; the armv7 Rust build and link; the highp shader and
  OpenGL ES 2 rendering; touch on the phone; the APK installing; frame time on the phone; the 32-bit
  x86 test job and the workflow itself (parsed as YAML, with its commands run locally, but never run
  on GitHub).

## CI and Workflows

- `.github/workflows/editor-android.yml`, run by hand from the Actions tab. Jobs: `rust-tests` (host,
  scalar backend, 32-bit x86), `host-selftest` (the desktop session, screenshots kept), and `apk`
  (Java 17, the Android SDK with build-tools 34.0.0 and platform 34, NDK r26d, Rust with the armv7
  target, the debug-key cache, then `scripts/build_editor_android.sh`; the APK and the logs are kept).
  The version name is `0.1.<run number>` and the version code is the run number.
- `scripts/tests/editor_android_script_check.sh` is not in the workflow; it needs a raylib source
  directory for its headers and the Ubuntu packaging tools, and is run by hand.

## Not built yet

- The real world: the editor reading and editing a `World` through the ECS bridge (T1b, built in
  `docs/mid-ecs-reflect.md`), so the hierarchy and inspector show and edit the ECS.
- Editing: edit commands, undo and redo, gizmos, text entry.
- Scenes and project files; the project panel is a placeholder.
- Rotation (a patch to raylib's Android backend, or a different native glue).
- Textures and lighting (the GPU backend step, `docs/roadmap.md`, Decision 12).

## Fixes and Problems

### `scene.rs`

- **A cube color matched the selection color.** The fourth palette color was the same yellow as the
  selection outline, so that cube would have looked selected all the time. A unit test that counted
  selection-colored lines caught it. The palette now has an orange there, and a test checks that no
  palette color equals the selection color.

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
