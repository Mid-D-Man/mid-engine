# mid-editor-ffi: the C API of the editor core

## Overview

`mid-editor-ffi` exposes `mid-editor` (`docs/mid-editor.md`) to C. It is a `staticlib` (and an `rlib`
for its own tests), declared in `include/mid_editor.h`. It exists so that the core can stay safe
Rust: this crate is the only place with `unsafe`, and it has no `[lints]` table for that reason, like
the other FFI crates in the workspace.

A host creates an editor with `mid_editor_new`, feeds it `mid_editor_pointer` events and
`mid_editor_resize`, calls `mid_editor_frame` once per frame, reads the paint list with
`mid_editor_paint_items` and `mid_editor_paint_text`, and frees it with `mid_editor_free`. The Android
host (`apps/editor-android/main.c`) is the user of this API.

## Modules

### `lib.rs`

**What it does:** the C API. `mid_editor_new` takes the screen size and a text-measure callback;
`mid_editor_resize`, `mid_editor_pointer`, `mid_editor_cancel_pointers` and `mid_editor_frame` feed
the core; `mid_editor_paint_items` and `mid_editor_paint_text` expose the list; `mid_editor_set_version`
and `mid_editor_log` pass text in; `mid_editor_panic_count` and `mid_editor_version` report.

**Decisions:**
- Every entry point runs its body in `catch_unwind` and counts a panic instead of letting it cross
  into C. A caught panic is also written to the console log.
- Every function accepts NULL and does nothing. A bad phase, a non-UTF-8 string or a bad size is
  ignored, never a crash.
- A measure callback that returns a non-finite or negative width falls back to an estimate.

**Tests:** 3 unit tests and `tests/ffi.rs` (5 tests): a full session through the C API, NULL and bad
input everywhere, the version string, and two checks that read `include/mid_editor.h` against the
source (every declared function is exported and the reverse; the enum values and the item layout match).

### `include/mid_editor.h`

**What it does:** the hand-written header. It declares the opaque editor type, the phase and paint
kind constants, the 36-byte `MidPaintItem`, the measure callback type and every function.

**Decisions:** the build script checks the header against the built library's symbols in both
directions, so a function cannot be added on one side and forgotten on the other.

**Tests:** the two header tests in `tests/ffi.rs`, and the script check.

## Verification status

- 8 tests (3 unit, 5 integration) pass on rustc 1.97.1 and 1.75.0, on the default and the scalar
  `mid-math` backend. `clippy --no-deps --all-targets` and `rustfmt` are clean.
- The real C host drives it through the header in the desktop self-test (`docs/mid-editor.md`).
- Not verified: the library built for `armv7-linux-androideabi` and linked into `libmain.so`.

## CI and Workflows

Tested by the `rust-tests` job of `.github/workflows/editor-android.yml` (host, scalar backend and
32-bit x86), and built for armv7 by its `apk` job.

## Fixes and Problems

None recorded for this crate. The build script's header check (`scripts/build_editor_android.sh`)
guards against the header and the library drifting apart.
