# mid-ui-wasm

## Overview

The browser playground for `mid-ui`: a page where a `.mxui` file is edited,
loaded, laid out and painted, with every step except the painting running in
WebAssembly. Three pieces:

- `web/mid-ui-wasm/`: a thin wasm-bindgen wrapper over `mid-ui` and
  `mid-ui-mxui`, built the way msx's `web/msx-wasm` is.
- `web/ui/index.html` and `web/ui/render.js`: the page and its Canvas 2D
  backend, published at `/ui/` by Deploy Site.
- The example files the page offers are the real fixtures from
  `crates/mid-ui-mxui/` (see `docs/mid-ui-mxui.md`), copied into the site
  at deploy time, so the playground and the tests never drift apart.

## Decisions

- **Standalone crate.** `web/mid-ui-wasm` has its own `[workspace]`, so it
  needs no change to the root `Cargo.toml` and its `cdylib` stays out of the
  engine's workspace-wide builds. Every cargo command for it runs inside that
  directory.
- **Logic is separate from bindings.** All behavior lives in `logic.rs` with no
  wasm-bindgen types, so it is tested natively; `lib.rs` is only the
  `#[wasm_bindgen]` shims. The `dixscript` feature (default on) gates the one
  constructor that parses source, so the logic tests run without DixScript.
- **JSON across the boundary, written by hand.** The draw list and node
  description cross as JSON strings built without serde, so the shipped module
  carries no JSON dependency. serde_json is a dev-dependency used only to check
  that the output parses.
- **The draw list is the only paint input.** `render.js` consumes it exactly
  as a raylib or software backend would. Text is the exception: the core has no
  text commands yet, so the page draws each node's `text` itself, on top, as a
  stand-in. Do not read the page's text as evidence the draw list carries any.
- **Painting is pure.** `render.js` exports functions over a canvas context,
  so they are tested in node against a recording fake.
- **Degrades gracefully.** In Deploy Site the wasm steps are
  `continue-on-error` and the assemble step tolerates a missing package; the
  page then shows a message instead of breaking the site.

## Modules

### `logic.rs`
`Core` (layout, draw list and node description as JSON, `hit_name`,
`claims_pointer`, `node_count`), `json_string`, `draw_list_json`,
`describe_json`.

- Draw list JSON: `{"t":"fill","x","y","w","h","r","c":[r,g,b,a]}`,
  `{"t":"clip","x","y","w","h"}`, `{"t":"pop"}`; an unknown command becomes
  `{"t":"unknown"}` because `DrawCmd` is `#[non_exhaustive]`.
- Node description JSON: nodes in paint order with `depth`, `id`, `kind`
  (`"Type.Variant"`), `text`, `bind`, rectangle, `visible`, `interactive`;
  undeclared fields are `null`.
- NaN and infinity are written as `0`, so the JSON always parses.
- `hit_name` returns `None` for no hit and `"(unnamed)"` for a hit on a node
  without an `id`.

**Tests:** `tests/logic.rs`, 10 tests, parsing the JSON with serde_json:
escaping, draw list shape, non-finite numbers, empty list, node description
and ordering, nesting depth, hit names, pointer claims, relayout.

### `lib.rs`
`Playground` (constructor from source, `layout`, `draw_list_json`,
`describe_json`, `hit_name`, `claims_pointer`, `node_count`) and the panic
hook. Loading errors are thrown as a message carrying the path of the problem.

### `render.js`
`paint`, `paintDrawList`, `paintLabels`, `paintOutline`, `roundedRectPath`,
`cssColor`, `toViewport`. Clip pushes and pops stay balanced even for an
unbalanced list. Rounded-rectangle radii are clamped to half the shorter side.

**Tests:** `web/ui/render.test.mjs`, 12 tests with `node --test`.

### `index.html`
Example picker, viewport size, a source editor that reloads 300 ms after the
last keystroke, the canvas, a pointer readout (hit node, whether the UI claims
the pointer) and a node list. Not tested; there is no browser in the authoring
environment.

## Size

Release build of the wrapper, raw `wasm32-unknown-unknown` output before
wasm-bindgen and wasm-opt:

| Build | wasm |
|---|---|
| default (with DixScript) | 3,740,209 bytes raw, 964,789 bytes gzip -9 |
| `--no-default-features` (reader only, no parser) | 103,657 bytes raw |

DixScript is almost all of the size. A game that ships a pre-built tree and
needs no parser at runtime would stay near the second number.

## Not built yet

Text drawing in the draw list, image nodes, touch and keyboard input on the
page, an in-page error location marker, links to `/ui/` from the other pages'
navigation (only this page lists itself), and an ECS-driven variant.

## Verification status

Sandbox, rustc 1.97.1 and node 22: 10 Rust logic tests and 12 node render
tests pass; 6 hand-made mutants of `render.js` and the earlier mutation runs
were caught. `cargo build --release --target wasm32-unknown-unknown` succeeds
with and without the `dixscript` feature. **The wasm module was never run**:
there is no wasm-bindgen CLI or browser here, so `wasm-pack`, the generated
glue, `index.html` and the DixScript path through the module are first
exercised by CI and by opening `/ui/` after a deploy. The Deploy Site wiring is
YAML-validated only.

## CI and Workflows

- `.github/workflows/mid-ui-wasm-test.yml`: `workflow_dispatch` only. Job
  `logic-tests` runs the native Rust tests (no DixScript) and the node render
  tests. Job `wasm-build` builds the package with wasm-pack, reports its raw and
  gzip sizes to the step summary and uploads it.
- `.github/workflows/deploy-site.yml`: new steps install wasm-pack, build
  `web/mid-ui-wasm`, run the node render tests, and the assemble step copies
  the page, the example files and the package into `dist/ui/`. All three new
  steps are `continue-on-error`.

## Fixes and Problems

None yet; first release.
