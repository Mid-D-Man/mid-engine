# mid-ui-mxui

## Overview

Loads `.mxui` files, which are DixScript v1.0.0 documents whose `root` entry
describes a widget tree, into a `mid_ui::Tree`. Generic over `f32` and `f64`.
Besides the tree it returns the node names, and the `kind`, `text` and binding
declarations each node carried, for later layers (text, widgets, data binding)
to use.

## Decisions

- **Two layers.** The schema reader (`load.rs`) works on a source-neutral
  `MxValue` tree and depends only on `mid-ui`. The DixScript adapter (`dix.rs`)
  is behind the `dixscript` feature, on by default. Reasons: the reader is fully
  testable without running DixScript, moving to DixScript 2.0 touches one
  module, and a host can build `MxValue` trees by hand.
- **`dixscript = "1.0.0"` with `default-features = false`.** That drops
  cloud-import (reqwest, tokio), rayon, toml and the DLM/encryption modules,
  none of which a UI loader uses. `load_from_str` is not feature-gated.
- **No `@IMPORTS` for shared enums.** In 1.0.0 imports resolve through
  `std::fs` (relative to the importing file) or cloud URLs, there is no
  `res://` scheme, and a browser has no file system. So every `.mxui` file
  declares its own enums. The reader requires the names `UiAnchor` and `UiFlow`
  for those two fields and accepts any enum for `kind`. Repeating the enum
  block per file is the cost; an import provider (DixScript 2.0 or an upstream
  hook) would remove it.
- **Strict fields.** An unknown field on a node is an error at its own path,
  so `widht` is caught instead of silently ignored.
- **`kind`, `text`, `bind` are metadata only.** The core draws no text and has
  no widget behavior yet; the loader keeps what the file says.
- **Binding keys are FNV-1a 32 hashes** of the name (`BindKey::of`, usable in
  `const`), computed at load. Two different names that hash alike are rejected
  at load rather than silently merged; the same name on many nodes is fine.
- **Object fields are sorted by name** in the adapter because DixScript hands
  objects back as a hash map. Order that matters (children) lives in arrays,
  which keep file order.
- **Numbers.** Ints and floats of any width become `i64`/`f64` and then the
  target scalar. A `1.5f` literal is a DixScript `f32` and widens exactly to
  `f64`; f64 users should write plain decimals, which DixScript reads as
  `f64`.
- **`build` is iterative.** Nesting depth in a file never grows the call stack
  (tested at 50,000 levels on a 128 KiB stack), and error paths are rebuilt
  from parent links only when an error happens, so a deep file is not
  quadratic. `from_dix`, the adapter's value conversion, is recursive, and
  its depth is bounded by what the DixScript parser accepts.

## The `.mxui` schema

`root` is one node: either a group array (`root::`) with exactly one item, or a
table. A node is an object; every field is optional.

| Field | Value |
|---|---|
| `id` | string; unique in the file |
| `kind` | any enum value (metadata) |
| `text` | string (metadata) |
| `bind` | string binding key (metadata, hashed) |
| `anchor` | `UiAnchor` value |
| `flow` | `UiFlow` value (`Stack`, `Row`, `Column`) |
| `width`, `height` | number (fixed) or `{ fill = weight }` |
| `margin`, `padding` | number, `[h, v]` or `[left, top, right, bottom]` |
| `gap` | number |
| `visible`, `interactive`, `clip` | bool |
| `background` | hex color: `#RGB`, `#RGBA`, `#RRGGBB`, `#RRGGBBAA` |
| `radius` | number |
| `children` | array of nodes, in paint order |

`examples/hud.mxui` is a complete file using QuickFuncs (`panel`, `label`,
`button`, `fill`) to keep each node to one line. QuickFuncs are only
convenience: a node is just an object, however it was produced.

## Modules

### `lib.rs`
Crate docs and re-exports.

### `value.rs`
`MxValue` (null, bool, int, float, string, color, enum, array, object) with
small constructors and `type_name`/`as_f64`.

### `bind.rs`
`BindKey`, the const FNV-1a 32 hash.
**Tests:** `tests/build.rs` checks the published reference values (`""`,
`"a"`, `"foobar"`) and the const form.

### `error.rs`
`MxError { path, kind }` and `MxErrorKind`, with paths like
`root.children[2].children[1].margin`. `Display` and `std::error::Error`.

### `load.rs`
`build::<R>(&MxValue) -> Result<Loaded<R>, MxError>`, `Loaded` (`tree`,
`names`, `meta`), `NodeMeta`, `Bind`, `EnumRef`. Reads each field, rejects
unknown ones, checks id uniqueness and bind-key collisions, adds children in
file order.
**Tests:** `tests/build.rs`, 21 tests: structure and order, every style field,
layout of a loaded tree, f32/f64 agreement, 50,000-deep file on a tiny stack,
metadata, bind keys and collisions, duplicate ids, sizing/insets/color/enum
forms and every error kind with its path.

### `dix.rs` (feature `dixscript`)
`from_dix` (`DixValue` to `MxValue`) and `load_str::<R>(&str)`, which loads
with `DixLoader::new_silent()`, reads `root`, and calls `build`. Dates,
timestamps, blobs and regexes become strings so they fail as type errors where
used. Imports in the source resolve through the file system, so keep files
self-contained, especially on wasm.
**Tests:** `tests/dixscript_e2e.rs` (5 tests) loads `examples/hud.mxui` and a
few small sources through the real parser.

## Not built yet

Hot reload (the watcher belongs to the host; dixscript ships a native
`HotReloadWatcher`), file loading helpers, a theme/token system, shared enum
files (see Decisions), widget behavior for `kind`, events, text and image
nodes, and a loader for DixScript 2.0.

## Verification status

- Schema reader: 21 tests pass (rustc 1.97.1 sandbox) and 12 hand-made
  mutants (child order, field targets, hash variant, short-color digits,
  shared bind names, duplicate ids, inset order, unknown fields, path order,
  fill weight) were each caught.
- DixScript adapter: **type-checked against the published `dixscript` 1.0.0
  (including the e2e test) but never run**; DixScript-Rust is not executed in
  the authoring environment, so the first real run is CI. The e2e tests rest on
  assumptions read from the grammar and dixscript's own tests, not observed:
  nested values come back inside `get_value("root")` (msx matches on the same
  shape); an enum literal inside a QuickFunc's returned object
  (`kind = UiKind.Panel`) parses (the tests only show enums passed as call
  arguments); a zero-argument call to a function whose parameters all have
  defaults (`fill()`) is accepted; `root:` as a table gives an object;
  `@DATA(` alone is a source error.
- `cargo check -p mid-ui-mxui --target wasm32-unknown-unknown` passes with and
  without the `dixscript` feature. Nothing was run on wasm; with the feature on
  it pulls `getrandom` 0.2, which has no browser backend enabled here, and
  whether the load path ever calls it is unknown.

## CI and Workflows

- `.github/workflows/mid-ui-mxui-test.yml`: `workflow_dispatch` only. Job
  `reader-tests` runs the reader tests without the DixScript feature (and
  clippy, `continue-on-error`). Job `dixscript-e2e` runs the whole suite with
  the feature on, including the e2e file. Job `wasm32-check` checks both
  feature settings. Cache keys are scoped to this crate and the runner OS and
  `~/.cargo/bin/` is not cached.

## Fixes and Problems

### Path building was quadratic in nesting depth
The first version of `load.rs` formatted a full path string for every node.
For a file nested N levels deep that is O(N squared) in time and copying.
Replaced with parent links and a `render` that runs only when an error needs a
path.
