# The engine app

**Status: design only. Nothing is built, and nothing here has been verified.** Written on
2026-11-16 as proposals for the maintainer to confirm or change. "The engine app" is a working
title; no crate, directory or name exists yet.

## Overview

An application, used on an Android phone, that controls the engine: a console, a tree of the
entities in a running world, an inspector to read and edit their components, and the status of
the project. Its interface is drawn with `mid-ui`.

**This is an assumption about what was asked for, and is written down so it can be corrected.**
It is not a game, and not yet a visual scene editor: a viewport that draws a scene needs a
renderer, which does not exist (`docs/roadmap.md`, Decision 12), and comes at milestone M4.

Constraints that shape it:

- The interface must be `mid-ui`, not a borrowed toolkit.
- The phone cannot build the engine, so everything the app needs must be built by a workflow.
- The engine does not have to run on the phone: the app can be a client of an engine running
  elsewhere, which is what the remote protocol is for (`docs/roadmap.md`, Decision 8).
- It is the second client of that protocol; the first is the command line (`docs/mid-cli.md`).

## Architecture: shared logic, thin shell

Three layers, so that only the bottom one changes per platform.

1. **App core.** The `mid-ui` tree, the screens (`.mxui` files, loaded by `mid-ui-mxui`), the
   app's own logic, and a client of the remote protocol. Platform-independent. Built as
   WebAssembly first, because that is what already exists (`web/mid-ui-wasm`).
2. **Backend.** Turns a `mid-ui` draw list into pixels. The draw list is already the only
   render seam (`docs/mid-ui.md`), so the backend is replaceable: Canvas 2D exists in
   `web/ui/render.js`; a native one would come later.
3. **Shell.** What hosts the core and the backend: a browser tab or installed web app, an
   Android WebView, or later a native activity.

The data source is either a **remote engine** (over the protocol) or an **embedded engine**
(`mid-ecs` and `mid-app` pass `cargo check` for `wasm32-unknown-unknown`, so a world inside
the page is plausible; it has not been run).

## What exists in `mid-ui`, and what the app needs

| Need | In `mid-ui` today | Plan |
|---|---|---|
| Tree, layout (anchors, margins, row and column flow), hit testing | Built, tested (62 tests, sandbox only per its doc) | Use as is |
| Rectangles and clipping | `FillRect`, `PushClip`, `PopClip` | Use as is |
| `.mxui` screens | Loader built; node names, `kind`, `text` and binding declarations are read and kept | Use as is |
| **Text** | Not built | U1: `DrawCmd::Text` and a measuring trait the host implements (Decision 11) |
| **Events, focus, scrolling, text input** | Not built; hit testing only says whether the UI claims the pointer | U2 |
| **Widgets** (button, label, field, list, tabs, tree, property grid) | Not built | U3 |
| Data binding | Declarations are read, not acted on | With U3 or U4 |
| Images and icons | Not built | Not needed for v0; text and rectangles suffice |
| Animation | Not built (planned on `mid-anim`'s `Track`) | Later |

Text is the first blocker: a console, an entity tree and an inspector are nearly all text.
Bevy's order for the same stack is the useful guide (`docs/roadmap.md`, "Bevy's layers,
recomputed"): text, then focus and input, then widgets, then an editor kit.

## The first screens

1. **Console.** Runs `mid rpc` style calls and shows results; a history list and a text field.
2. **Entities.** The hierarchy as a tree (`ChildOf` and `Children`, over `world.query` and
   `world.reparent_entities` equivalents), with names where `mid-nodes` provides them.
3. **Inspector.** A property grid for the selected entity's components, driven by the schema
   `mid-reflect` describes; edits go back through the protocol.
4. **Project.** Test counts, CI results, the roadmap's status table, read from `mid ci`
   output; no engine needed.
5. **Settings.** The endpoint, and the token for it.

Screens 2 and 3 need reflection and the protocol. Screens 1, 4 and 5 can come first.

## Getting it onto the phone: three ways

All three share the core and the backend; they differ in the shell. Everything about Android
below is **unverified**.

| | Shell | What it needs | Trade-offs |
|---|---|---|---|
| A | **A web page, installable as a web app** | A manifest and a service worker; HTTPS (the existing pages are served over it) | No separate build chain. Sandboxed by the browser. A page served over HTTPS cannot open an insecure WebSocket except to localhost, so the engine endpoint needs TLS or a tunnel |
| B | **An Android WebView app (an APK)** | A small Android project that hosts the same page and assets, and a CI job that builds it (believed to be possible on the hosted runners, not checked) | Works offline with bundled assets. Adds an Android build chain. Installed by hand |
| C | **A native app** | A Rust library built with the Android NDK, an activity, a window and input layer, a drawing backend, text rendering, and a soft keyboard for text input | Best fit eventually; needs most of milestone M4's backend work first |

**Proposal:** A, then B, then C. A reaches the phone soonest and risks least; B follows when
the page works and an installed app is wanted; C waits for the runtime work. The core and the
screens do not change between them.

## Where the engine runs

Not decided, and nothing is built or assumed. The app asks for an endpoint. Candidates: a
cloud development environment or CI job with a forwarded port, a small server, a tunnel to a
desktop when one is available, or an engine embedded in the page. The remote protocol must
therefore be safe to expose:

- A token is required on any network transport; stdio needs none.
- A network transport binds to localhost by default.
- TLS comes from whatever fronts the endpoint (a tunnel or a proxy), not from the engine.
- Writes (`insert`, `mutate`, `remove`, `despawn`, `reparent`, `trigger`) can be turned off
  entirely for a read-only endpoint.

## Testing without a phone

Most of the app can be tested where the engine is tested, because `mid-ui` has no renderer:
the screens are loaded, laid out and turned into draw lists in plain tests, and the draw list
can be compared with a stored one. The remote client is tested against a fake server. What
cannot be tested here is each shell, the backend's pixels, and anything about Android.

## Phases

| Phase | What | Needs | Done when |
|---|---|---|---|
| U1 | Text in `mid-ui` | `mid-ui` | A `.mxui` screen with text lays out and draws in the browser |
| U2 | Events, focus, scrolling, text input | U1 | A screen can be clicked, scrolled and typed into |
| U3 | Widget kit | U2 | The first five screens' widgets exist |
| U4 | The screens and the `/app/` page | U3, `mid-remote`, a WebSocket transport | The console and the entity tree work against a live world |
| U5 | WebView shell and a CI job that builds the APK | U4 | The APK installs on the phone and shows the same screens |

## Not built, on purpose

A viewport that draws a scene; images and icons; animation; a native shell; offline engine
state; multi-user editing; accessibility beyond what `mid-ui` provides (Bevy's `a11y` is a
layer 4 crate and has no equivalent here yet).

## Open questions for the maintainer

1. Is the assumed meaning right: a tools app (console, entities, inspector, project status),
   not a game and not a scene editor, at least at first?
2. A, then B, then C (proposed), or go straight to B?
3. Where would the engine run for you: cloud shell, a server, a tunnel to a desktop, or only
   embedded in the page for now?
4. Text rendering: let the browser draw text (proposed, via the measuring seam) for the first
   versions, with your own text crate later, or build text yourself from the start?
5. Do you want a name for the app now? Nothing is called anything yet.

## Verification status

None. This is a design document. No code, no tests, no CI. The sandbox can build and test the
core, the screens and the protocol client; it cannot test Android, a WebView, or a PWA.
