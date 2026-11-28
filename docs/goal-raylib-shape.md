# Goal: a raylib shaped way to use mid-engine

Status: **goal**, set by the maintainer on 2026-11-27 (Decision 19 in `docs/roadmap.md`). Nothing is built for
it yet and no design is fixed; the ladder in section 5 and the code in section 6 are proposals. Evidence tags
as elsewhere: **[src]** read in a source tree (named), **[doc]** stated in a document of this repository,
**[run]** counted or measured in the sandbox, *reasoned* follows from what was read.

## 1. The goal

A project, in Rust or through the C API, depends on mid-engine and uses it as a library: the program owns
`main` and its loop, opens a window with one call, draws, reads input and plays sound with plain function
calls, and needs no editor, no command line tool and no engine-owned application loop. The way raylib is used.

The editor and the `mid` command line stay what they are. This goal adds a second way in, for people who want
to write a game as code.

## 2. What the raylib shape is

Read from raylib at commit `40b9da5` (2026-11-26, v6.5-dev) [src, `raylib/src`, `raylib/examples`].

- **One header, plain functions.** `raylib.h` declares 620 `RLAPI` functions [run, counted]. Plain C99 means
  bindings are easy, and many exist [src, header comment].
- **Modules a program does not have to know about:** core (window, input, timing), shapes, textures, text,
  models and audio, plus `rlgl`, a graphics abstraction that can be used alone, and `raymath`. Sizes in lines
  [run]: `rcore` 4,717, `rshapes` 4,577, `rtextures` 5,971, `rtext` 3,101, `rmodels` 7,397, `raudio` 2,940,
  `rlgl` 5,430.
- **No dependencies for the user.** The header lists none to install; the libraries it needs are bundled
  [src, header comment]. Platform backends sit in `src/platforms`: Android, desktop through GLFW, SDL, RGFW or
  Win32, a direct-rendering mode for boards without a desktop, and the web. A software renderer, `rlsw`, is
  optional for machines with no GPU [src].
- **The program owns the loop.** The basic window example is about thirty lines: `InitWindow`, `SetTargetFPS`,
  then `while (!WindowShouldClose())` around `BeginDrawing`, `ClearBackground`, a draw call and `EndDrawing`,
  then `CloseWindow` [src, `examples/core/core_basic_window.c`].
- **Immediate mode with a default of everything.** One font, one white texture, one shader and one render
  batch are created at start, so a draw call works with no setup [src, header notes].

As properties, this is what makes the shape (*reasoned*):

1. the program owns `main` and the loop;
2. one dependency and one initialization call;
3. drawing works with no scene, no world and no asset;
4. every part beyond the core is optional and costs nothing when unused;
5. the whole surface is plain functions, so any language can call it.

## 3. Where mid-engine stands against those properties

| Property | State today |
|---|---|
| The program owns the loop | **Mostly there for logic.** `mid-app` has `App::update()`, `update_with(delta)` and `run_until(closure)`, so a host drives the loop [src, `crates/mid-app/src/app.rs`]. There is no window to drive it from |
| One dependency, one init call | **No.** There is no umbrella crate; a project names each crate [src, `crates` listing, no `mid-engine` crate] |
| Drawing with no setup | **No.** `mid-ui` produces a draw list and `mid-editor` a paint list; nothing paints either on its own [doc, `docs/mid-ui.md`, `docs/roadmap.md` Decision 14] |
| Optional parts | **Yes.** Each crate is a separate library in the workspace, so a project takes only what it names [src, `crates` listing] |
| Plain functions for any language | **Partly.** C exports found by counting `extern "C" fn` [run]: `mid-math` 1,340, `mid-ecs` 69, `mid-net` 51, `mid-geom` 44, `mid-log` 21, `mid-editor-ffi` 15, and none in `mid-app` or `mid-ui` |

What exists as libraries, usable today as pure code: math, the ECS and its hierarchy, the app and plugin
schedule, networking, logging, time, collections and arenas, geometry, early physics and animation sampling
[src, crate listing and `docs/roadmap.md`]. `mid-platform` is not a window layer: it holds synchronization
primitives and hashers [src, `crates/mid-platform/Cargo.toml`].

What does not exist: a window, input, rendering, audio, images, assets and an umbrella crate.
`docs/roadmap.md` lists windowing, rendering and audio as missing [doc]. The one program that owns a window today
is the editor's C host, `apps/editor-android`, built on raylib 6.0 [doc, Decision 10], and the maintainer's own
raylib game uses `mid-net` beside raylib [doc]. That is the raylib shape in reverse: raylib is the window and
mid-engine is the library underneath.

## 4. What the shape would need

| Part | Exists | Missing |
|---|---|---|
| Entry point | | One crate, with features, that re-exports the others and a small prelude |
| Window and loop | `App::update` | Opening a window; an event pump; desktop, Android and web backends |
| Input | `mid-editor` takes pointer messages | Keyboard, mouse, touch and gamepad as plain state a program can poll |
| 2D drawing | the paint list and the contracts of `docs/mid-ui-contracts.md` | A CPU backend (A3), a GPU backend (A11), and a draw API of shapes, text and images that fills the paint list |
| Text and fonts | a measure seam, a bundled font in the editor | The text stack of Decisions 14 to 16 |
| Images and assets | a byte seam (Decision 13) | Image decoding, a loader, handles |
| 3D, models, cameras | `mid-math` camera helpers, `mid-anim` sampling | Meshes, materials, a renderer, model loading |
| Audio | | Everything |
| Timing | `mid-time` | A frame limiter tied to the window |
| C API | per crate, as counted above | The same surface for the new window and drawing crates |

## 5. A ladder (proposal)

Each rung is usable on its own and keeps the program in charge of the loop.

- **R0, today.** Libraries only. A project brings its own window (raylib, SDL) and uses mid-engine for the
  simulation, math and networking. This already works and is how the maintainer's raylib game is built.
- **R1, one entry crate.** An umbrella crate that re-exports the others behind features and a prelude. No
  window. It is small and removes the "name each crate" cost. It inherits the highest Rust floor of what it
  includes; `mid-ecs` does not build on rustc 1.75 today [doc, `docs/mid-ecs-reflect.md`], so the floor needs a decision.
- **R2, a window and input.** A crate that opens a window, pumps events, exposes input state and shows a CPU
  framebuffer. The CPU backend of A3 is the first thing it can present, which is the same role `rlsw` plays for
  raylib. Desktop first, Android second, since the editor already builds there.
- **R3, a 2D draw API.** Immediate calls for shapes, text and images that write into the same paint list the UI
  uses, so one renderer serves game drawing and UI.
- **R4, the GPU path, audio and 3D.** The GPU backend of A11, an audio crate, then models and cameras.

An illustration of the shape at R2 and R3. The names are not designed.

```rust
use mid::prelude::*;

fn main() {
    let mut window = Window::open(800, 450, "basic window");
    while !window.should_close() {
        let mut frame = window.begin_frame();
        frame.clear(Color32::WHITE);
        frame.text("Congrats! You created your first window!", 190.0, 200.0, 20.0, Color32::GRAY);
    }
}
```

## 6. Constraints that already stand

- **FFI first.** Every crate is meant to be FFI-compatible out of the box [doc, `README.md`]; each new rung carries a C API, as
  each new UI crate does (Decision 15).
- **Dependencies.** Core crates keep the dependency mandate, with hosts and backends one layer up, where a
  dependency is allowed if it stays confined there [doc, `docs/roadmap.md` Decision 12]. A window layer is such
  a layer, so an existing library is permitted in it; the open point is whether to use one.
- **Open source, owned by MidManStudio** [doc, Decision 14, point 8]. Reading other libraries and
  re-implementing is the working rule; vendoring needs its own decision.
- **The phone.** It is 32-bit ARM, and raylib's Android backend ignores a rotation [doc, Decision 10]. Any
  Android rung has to be tried there.
- **Scope of the UI plan.** The order A1 to A12 of `docs/mid-ui-roadmap.md` is unchanged. R2 and R3 reuse A3
  and A11; they do not replace them.

## 7. Questions for the maintainer

| # | Question, in plain words | Default taken here |
|---|---|---|
| 1 | One umbrella crate with features, or several? What name? | One crate with features. The name is open: `mid` is already the command line binary |
| 2 | The window and input layer: write thin backends per platform, or sit on an existing library such as winit or SDL, confined to that layer? | Not decided. An existing library is faster to ship on desktop; own backends keep the layer free of dependencies; Android already has a working own pipeline |
| 3 | Should immediate draw calls (rectangles, text, images) feed the same paint list and backends as the UI? | Yes, so there is one renderer |
| 4 | Is 3D part of this goal now, or after 2D works? | After 2D |
| 5 | Audio: own crate or an existing library? | Later; no default |

## 8. What this does not change

The editor, the `mid` command line, and the order of A1 to A12. Nothing here is built; the first step on
this ladder that is not already planned is R1.

## Sources

Read for this document: raylib `src/raylib.h`, `src/platforms`, `examples/core/core_basic_window.c` and the line
counts of its module files; `Cargo.toml`, `crates/mid-app/src/app.rs`, `crates/mid-platform/Cargo.toml`,
`docs/roadmap.md` (Decisions 10, 13, 14 and 15), `docs/mid-ui-contracts.md`, and the `extern "C" fn` counts per
crate.
