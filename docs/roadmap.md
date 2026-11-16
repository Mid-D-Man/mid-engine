# Mid Engine Development Roadmap

Living roadmap. Supersedes `docs/architecture.md`'s "Build order"
section going forward — that section stays as the historical record
(it already does this once, see its own trailing note about the
superseded "Priority column" table); this doc is where build-order
changes get made from now on. Grounded in `docs/bevy-comparison.md`'s
analysis — read that first if the reasoning below needs the source
data.

## Current status (as of 2026-11-16, `mid-engine` @ `5dc9525`)

Checked by reading the repository (each crate's contents and doc), not by rebuilding
everything. Real CI: the `mid-ecs` and `mid-ecs-ffi` workflows pass on main (reported by
the maintainer on 2026-11-16, after the hierarchy, observer propagation and transform
deliveries). Anything below marked "sandbox only" has not run on real CI.

| Crate | Status |
|---|---|
| `mid-math` | Practically done: f32/f64 SIMD vectors, matrices, quaternions, curves, noise, color, camera math, fixed point, LWC view-space shift. The `const_bits` patch (builds on rustc 1.75) and the `mint` feature fix are in. Open: its tests do not compile under `--features force-scalar` (`tests/wide_tests.rs`, E0308, also on unmodified main), and `cargo clippy` reports 17 errors on rustc 1.97 |
| `mid-common` | Thin, unchanged: `string/` and `ffi/` are about 2,000 lines; `error.rs`, `traits.rs` and `types.rs` are 2 to 3 lines each |
| `mid-platform` | Phases 1 to 3 built (atomics, cells, poison, mutex, `Arc`/`Weak`, rwlock, once, lazy lock, barrier, hashers, `HashMap`/`HashSet`), about 5,700 lines; see `docs/mid-platform.md` |
| `mid-ptr` | A full port of `bevy_ptr`; nothing in `mid-ecs` calls into it (`docs/mid-ptr.md`) |
| `mid-alloc`, `mid-arena` | First pieces built (stack allocator; generational slot arena); see their docs |
| `mid-collections` | `SparseSet`, `GenerationalIndexAllocator`, `fx_hash` and `ffi_span` built; the rest is design only |
| `mid-geom` | Substantial: shapes, intersection tests, raycasting, frustum culling. Known gaps (OBB, capsule-vs-AABB, broadphase) wait for `mid-physics` to need them |
| `mid-net` | A facade plus subcrates (wire, reliable, transport, quinn and wasm backends, an FFI crate); `docs/mid-net.md` says what has run on real CI for each |
| `mid-ecs` | Archetype Core and Sparse Shell; queries with `Mut`, `Ref`, `Added`/`Changed`, `With`/`Without`/`Or`; `Commands`; change detection and tick wraparound; resources; system params with access checking and `WorldCell`; the schedule (graph, executor, parallel executor); messages, states, observers (including events bubbling up `ChildOf`); component hooks; removal tracking; `par_iter`; the entity hierarchy; `LocalTransform` composition into `GlobalTransform` and `GlobalTransformLWC`. A C API covers most of it. 1,193 lib tests (113 of them for the C API) and 40 doctests (sandbox) |
| `mid-app` | `App`, `Plugin`, `Main`; stages `PreUpdate`, `StateTransition`, `Update`, `Net`, `Physics` (fixed step), `PostUpdate`; `HierarchyPlugin`, `TransformPlugin`; 48 tests plus a doctest |
| `mid-time` | `Clock` and `FixedTimestep`, zero dependencies, wasm32-safe |
| `mid-physics` | Exists but small (about 160 lines): `RigidBody` and a semi-implicit Euler linear integrator. No angular motion, no collision, not wired into `mid-ecs` |
| `mid-anim` | Exists but small (about 140 lines): `Track<T>` keyframe sampling (Catmull-Rom). Not wired to anything |
| `mid-log`, `mid-trace` | `mid-log` good (lock-free SPSC, tiered levels, an Android logcat sink behind a feature); `mid-trace` a feature-gated no-op by default |
| `mid-ui` | Standalone UI core: widget tree (its own arena, not ECS), anchor/margin layout with row and column flow, hit testing, a draw list of `FillRect`, `PushClip`, `PopClip`. f32 and f64. No text, images, event dispatch, focus or animation. Sandbox only (62 tests) per its doc |
| `mid-ui-mxui` | The `.mxui` loader (DixScript v1.0.0 documents describing a widget tree). With `web/mid-ui-wasm` and `web/ui` it forms a browser playground with a Canvas 2D backend |
| `examples/`, `tools/` | `headless-server`, `custom-transport-demo`; `tools/mdix-compiler` |
| `mid-nodes`, `mid-camera` | Not started. No crates, no design docs until this one (`docs/mid-nodes.md`) |

**Does not exist at all yet:** reflection (a description of a component's name and fields),
component serialization, assets, input, windowing, text and fonts, images, rendering, audio,
scenes and nodes, a protocol for talking to a running world, a command line, and any app.

**Where that leaves us.** The ECS has reached the point where other crates can be built on
it without waiting for it: everything a program needs from it to run headless is there. What
is missing is everything between the ECS and a person. That gap is the subject of "The road
to the engine app", below. Remaining ECS items: reactivity; `startup` and `Last` stages and a
128 Hz `Net` tick only if something needs them (the `Net` stage is shaped by how the games
using `mid-net` behave); change detection on `ChildOf`/`Children` is untested; the query
iteration gap on x86_64 (`dense_query_iteration`, about 4x against `bevy_ecs`) is parked,
with performance work paused until the ECS is feature-complete.

## What Bevy's own graph confirms or complicates about that order

From `docs/bevy-comparison.md` §1:

- **Math and ECS are independent in Bevy's own graph** (same layer,
  neither depends on the other — both only need `platform`/`utils`/
  `tasks`/`ptr`/`reflect`). This doesn't invalidate mid-engine doing
  math before ecs — `mid-geom` and the eventual `GlobalTransform`
  work both need math regardless — it just means Bevy's graph isn't
  independent *evidence* that math must precede ecs. The two are
  parallel-buildable systems in both engines; mid-engine's ordering
  is a reasonable choice, not a forced one.
- **`bevy_app` (App/Plugin/schedule-runner) is built immediately
  after ecs+math, before anything else consumes it** — before assets,
  before rendering, before windowing. mid-engine has no equivalent
  crate at all right now (`docs/bevy-comparison.md` §2). This is a
  real, concrete gap, not a stylistic one: every crate mid-engine
  builds from here on (physics, anim, nodes, eventually rendering)
  will want a real place to register systems into and a real loop to
  run under, and right now that place doesn't exist —
  `examples/headless-server` hand-rolls its own bootstrap instead.
- **`bevy_time` is a standalone crate at layer 6**, not a data type
  folded into something bigger, because it has real per-frame systems
  (advance `Time<Fixed>`/`Time<Virtual>` once, several consumers read
  the result). mid-engine has no `mid-time` crate, and already has
  two independent signs it wants one: `mid-net/reliable.rs`'s
  caller-supplied `Timestamp(u64)` (`docs/mid-net.md`, "Platform
  Design Principles"), and the stated 60 Hz physics / 128 Hz network
  performance targets in `docs/architecture.md` that nothing in the
  current crate list actually owns a clock for yet.

## Bevy's layers, recomputed (2026-11-16)

`docs/bevy-comparison.md` was written against an older Bevy. This table was recomputed from
the `Cargo.toml` of every crate in the `Mid-D-Man/bevy` fork (62 crates; layer = one more than
the highest layer of its non-optional `bevy_*` dependencies). It answers "what is needed
before what" from Bevy's own structure.

| Layer | Bevy crates | mid-engine |
|---|---|---|
| 0 | `platform`, `ptr`, `math`, `macro_utils`, `android` | `mid-platform`, `mid-ptr`, `mid-math` built. `bevy_android` (platform glue for Android) has no equivalent yet |
| 1 | `utils`, `tasks`, `curve`, `shape`, `derive`, `encase_derive` | `mid-common` (thin); tasks live in `mid-ecs`'s parallel executor; curves in `mid-math`. No proc-macros, by the dependency mandate |
| 2 | `ecs`, **`reflect`**, `transform`, `color` | `mid-ecs` built, transform built, color in `mid-math`. **Reflection is missing** |
| 3 | `app`, `state` | `mid-app` and states built |
| 4 | `input`, `time`, `log`, `a11y` | `mid-time`, `mid-log` built. **Input is missing** |
| 5 | `window`, `diagnostic`, `clipboard`, `gilrs`, settings | **Windowing is missing**; `mid-trace` partly covers diagnostics |
| 6 | `asset` | **Missing** |
| 7 | `scene`, `animation`, `image`, `mesh`, `audio`, `shader` | `mid-anim` samples only; **scene (`mid-nodes`) missing**; the rest missing |
| 8 | `camera`, `text`, `material` | camera math in `mid-math`, `mid-camera` not started; **text missing** |
| 9 | `picking`, `input_focus`, `ui`, `sprite`, `gizmos`, `world_serialization` | `mid-ui` core (its own design) with hit testing; **focus missing**; serialization missing |
| 10 | `render`, `winit`, `gltf` | **Missing** |
| 11 to 12 | `ui_widgets`, `dev_tools`, `pbr`, `core_pipeline` and others | Missing |
| 13 | `remote` (the Bevy Remote Protocol) | Missing. See below |
| 14 | `feathers` (widgets for editors and utilities) | Missing |

Four things this table settles:

1. **Reflection sits next to the ECS in Bevy** (layer 2), and `asset`, `scene`,
   `world_serialization`, `remote`, `window`, `input`, `ui` and `picking` all depend on it,
   some only optionally. It is the foundation under every tool. mid-engine has no equivalent,
   and it is the biggest single gap between "an ECS" and "an engine you can look inside".
2. **The remote protocol is late in Bevy only by accident.** `bevy_remote` hard-depends on
   `app`, `ecs`, `reflect`, `tasks`, `log`, `color`, `platform`, `utils`, `derive` and, oddly,
   `dev_tools` (which drags in ui, animation, audio and more); rendering and assets are
   optional for it. What a remote protocol really needs is reflection and an app. mid-engine
   does not have to follow Bevy's order here, and a terminal client of a running world can
   exist long before any renderer does.
3. **Editor widgets come last.** `feathers` (editor widgets) sits on `ui_widgets`, `ui`,
   `text`, `scene`, `render` and `ui_render`; `ui` needs `text`, `input_focus` and `a11y`.
   For `mid-ui` the transferable lesson is the sequence text, then focus and input, then
   widgets, then the editor kit.
4. **Android is a platform crate plus a windowing backend in Bevy** (`bevy_android`,
   `bevy_winit`), not a special UI path.

## Decisions

Framed as decisions because more than one reasonable answer exists
for each — recording context/alternatives the way this workspace's
other docs already do (e.g. `docs/mid-collections.md`'s FFI-wrapper
tension section), not full ADR ceremony.

**Decisions 1 and 2 (below) are confirmed** — standalone crates
outside `mid-common` are fine; `mid-app` and `mid-time` don't need to
be folded into an existing crate to be built.

---

### Decision 1 — Add a `mid-app` crate (App/Plugin/schedule-runner), now

**Context:** No crate in the workspace owns "assemble systems into a
runnable program." `mid-ecs` has `World`/`query`, but nothing decides
*when* a system runs relative to others, and nothing drives the loop.
Bevy builds this layer immediately after ecs+math (layer 5, before
literally everything else) — not deferred until rendering or assets
exist.

**Recommendation: build it now**, once `mid-ecs`'s Sparse Shell +
Archetype Core are usable (they already are) — before `mid-physics`.
Minimal scope for a first pass: an `App` struct owning a `World`, a
fixed schedule with a small number of stages (something like
`PreUpdate`/`Update`/`PostUpdate`/`Net`/`Physics` given mid-net and
mid-physics both need deterministic ordering against a tick), a
`Plugin` trait, and a loop driver. FFI-ready per the existing
mandate, same as everything else.

**Alternative considered — defer until rendering/windowing exists.**
Rejected: Bevy itself didn't defer this, and mid-net + mid-ecs are
already at the point where "what actually calls `Connection::poll()`
once a tick, in what order relative to ECS systems" is a live
question `examples/headless-server` is currently answering by hand,
per-example, with no shared answer. Every crate built between now and
rendering (physics, anim, nodes) would otherwise repeat that same
ad-hoc bootstrap.

---

### Decision 2 — Add a standalone `mid-time` crate, not folded into `mid-common`

**Context:** `mid-net`'s `reliable.rs` already needed "time" and
solved it locally (`Timestamp(u64)`, caller-supplied, never queried
internally — load-bearing for wasm32 compatibility, per
`docs/mid-net.md`). `mid-physics` (60 Hz target) will need a real
fixed-timestep clock with the same wasm32-safety constraint. Bevy
keeps `bevy_time` separate from `bevy_utils`/`bevy_platform`
specifically because it's not a passive data type — it's a system
that runs every frame and multiple other systems read from.

**Recommendation:** standalone `mid-time` crate, not a few structs
added to `mid-common`. `mid-common`'s own scope (per
`docs/mid-collections.md` and its own thin `error.rs`/`traits.rs`)
is passive shared types — a crate with a real per-frame advance-the-
clock system doesn't fit that description, the same reasoning that
keeps it out of `bevy_utils` in Bevy. Build order: after `mid-app`
(Decision 1), before `mid-physics` — physics is the first real
consumer of a fixed timestep.

---

### Decision 3 — Defer a `bevy_platform`-equivalent (`mid-platform`?); name the trigger explicitly

**Context:** `bevy_platform` (no_std sync/hash/time/cell primitives)
is the one crate 55 of Bevy's 60 crates transitively depend on. Right
now exactly one place in mid-engine has needed this class of problem
— `mid-net/reliable.rs`'s hand-rolled `Timestamp` — and it solved it
locally rather than needing a shared crate.

**Recommendation: don't build this speculatively.** This matches
mid-engine's own established discipline, applied consistently
elsewhere in this workspace: `mid-collections`' pieces are built "the
moment `mid-ecs`'s real storage work started, nothing before it"
(`docs/mid-collections.md`); `mid-geom`'s remaining gaps are "driven
by `mid-physics`'s actual requirements rather than built
speculatively" (`docs/architecture.md`). Same rule here.

**Naming the trigger so it isn't silently forgotten** (the specific
ask behind this rule, per that same precedent): the day a **second**
crate needs a wasm32-safe `Instant` substitute, a faster-than-SipHash
hash map, or a no_std-safe mutex — most likely `mid-time` (Decision
2) or `mid-ecs`'s eventual parallel query work — that's the trigger
to extract a shared crate instead of a second hand-rolled local
solution. Until then, `mid-common` stays thin, correctly.

**Reopened:** the named trigger fired, twice, unnoticed until checked
directly. `mid-time` independently hand-rolled its own wasm32-safe
`Instant`, and `mid-alloc` independently hand-rolled `SpinLock<T>` —
exactly the first and third named conditions. `mid-platform` is now
being built in phases; see `docs/mid-platform.md` for the full
dependency-by-dependency breakdown (`bevy_platform` itself is not
zero-dependency, unlike `mid-ptr` — that doc works out which of its
real dependencies mid-engine's actual target matrix needs at all,
which can be hand-rolled, and which are deferred the same way this
decision itself was). Phase 1 (`sync::atomic`, `cell`, `sync::poison`,
`sync::mutex`, `sync::{Arc, Weak}`) and Phase 2 (`sync::rwlock`,
`sync::once`, `sync::lazy_lock`, `sync::barrier`) are both built now,
and so is Phase 3 (`hash`, `collections`). The `Instant` half of the
trigger is settled without moving code: `mid-time` keeps owning its
`Instant`, and `mid-platform` takes `mid-time` as an optional `std`-only
dependency when a platform feature (`thread::sleep`, `future::block_on`)
first needs one (`docs/mid-platform.md`, "Build order"). Nothing is built
for it yet, same trigger-based discipline this entry always used.

---

### Decision 4 — `mid-math`'s scope (color/noise/camera): no action now, same trigger-based deferral

**Context:** `docs/bevy-comparison.md` §4 — Bevy splits color into its
own crate (`bevy_color`) specifically so non-rendering consumers
don't pull in projection/quaternion code. `mid-math` currently bundles
color, noise, camera math, and curves alongside the core SIMD types
in one crate. The compile-boundary argument for splitting is weaker
here (no_std, zero external deps, nothing pulls in a heavy transitive
tree the way a real crate split would avoid in a glam-based engine).

**Recommendation:** no split now — there's no real crate on the other
side of that boundary yet (no `mid-render` to *not* want camera code,
no `mid-ui` to *not* want color code). Revisit once `mid-render`
exists and there's a concrete reason (build-time cost, or wanting
`mid-color`/`mid-camera` independently publishable) rather than
matching Bevy's split by default. Same trigger-based discipline as
Decision 3, applied to a different crate.

---

### Decision 5 — Add `[profile.wasm-release]` and a `[workspace.lints]` table now

**Context:** both are cheap, both are directly supported by things
that already exist in the workspace today (not speculative):
`mid-net-transport-wasm` already targets wasm32 and will eventually
need a real, size-optimized release build for browser deployment;
`mid-collections`, `mid-net`, and `mid-math` already have real
`unsafe`/FFI code that this workspace's docs already *claim* follows
a "document every unsafe block, unsafe only at FFI/SIMD boundaries"
discipline, currently enforced by convention, not tooling.

**Recommendation — do both now, low effort, matches
`docs/bevy-comparison.md` §6 directly:**

```toml
# root Cargo.toml
[profile.wasm-release]
inherits = "release"
opt-level = "z"
lto = "fat"
codegen-units = 1

[workspace.lints.clippy]
undocumented_unsafe_blocks = "warn"

[workspace.lints.rust]
unsafe_code = "deny"
unsafe_op_in_unsafe_fn = "warn"
```

`unsafe_code = "deny"` at the workspace root needs explicit
`#![allow(unsafe_code)]` added to whichever crates actually use it
today (`mid-math`'s SIMD intrinsics, `mid-net`'s `ffi.rs`,
`mid-collections`' `ffi_span.rs`, `mid-ecs`'s `ffi.rs`) — that's a
feature of doing this now, not friction: it makes "these are the
crates allowed to use `unsafe`" a real, greppable, compiler-enforced
list instead of an implicit convention. `missing_docs = "warn"` is
worth considering too but skipped from the recommendation above
since `mid-math`'s own known clippy debt (182 errors, per
`docs/mid-math.md`) means turning on new workspace-wide warnings
right now would bury the wasm-release/unsafe-code wins in noise —
add it separately, after that existing debt gets its own pass.

---

### Decision 6 (resolved) — No `mid-ptr` crate (`bevy_ptr` equivalent) right now

**Context:** `bevy_ptr` gives `bevy_ecs` two things: type-erased,
lifetime+alignment-tracked pointer wrappers (`Ptr`/`PtrMut`/
`ThinSlicePtr`) for reading component bytes during query iteration,
and an owning/move variant (`OwningPtr`/`MovingPtr`) that lets
archetype-migration move a component's raw bytes directly between
table columns with a single non-overlapping `unsafe` copy — no boxing,
no intermediate typed value.

**Checked directly against `mid-ecs`'s actual code before answering,
not assumed:**

- The FFI-facing half already exists, just under a different name and
  a different (equally valid) mechanism: `mid_collections::ffi_span::
  FfiSpan` is a `#[repr(C)]` `(ptr, stride, count)` view built on
  `zerocopy`'s `IntoBytes`/`Immutable`/`KnownLayout` bounds rather than
  `bevy_ptr`'s lifetime-tracked wrapper types + `Aligned`/`Unaligned`
  marker types — same job (a type-erased, C-safe view into a Rust
  array), different technique. Already real, already shared by both
  `mid-net` and `mid-ecs`.
- The internal, `OwningPtr`-style migration half was already
  considered and explicitly turned down — `crates/mid-ecs/src/
  archetype.rs`'s own module doc says so directly: *"Bevy's real
  row-migration (`Tables::move_row`) is a sorted merge-join... using
  raw pointers and `unsafe` non-overlapping copies... Copying that
  technique wholesale would mean importing a large amount of `unsafe`
  for a perf technique with no profiled need here — directly against
  this project's own established precedent (`SparseSet`,
  `GenerationalIndexAllocator`: zero `unsafe`, by choice, revisit only
  against a real profile)."* The accepted cost is spelled out too:
  "one heap allocation per moved component per structural change — not
  per frame, not per query" — deliberately kept off the hot path this
  whole Sparse-Shell-vs-Archetype-Core split exists to protect.

**Recommendation: no new crate.** A `mid-ptr` crate's entire point
would be exactly the raw-pointer migration technique `archetype.rs`
already weighed and rejected for lack of a profiled need — building
it now would quietly reopen a decision that's already made, reasoned,
and written down, not fill an actual gap. This is independent of
Decisions 1/2 above: "helper crates outside `mid-common` are fine" is
about where code lives, not about picking up new `unsafe` surface
without a profiled reason to. Same trigger as Decision 3: if
migration ever gets profiled as a real bottleneck, that's when this
gets revisited — not before.

**Reopened (2026-09-16):** built anyway, on direct instruction, after
the above was surfaced and the conflict was made explicit rather than
silently overridden. Neither of the two reasons above turned out to
be wrong — `FfiSpan` still covers the FFI-facing half, and
`archetype.rs`'s own rejection of the migration technique stands
un-revisited, still with no profiled bottleneck behind it. `mid-ptr`
now exists as a full, close port of upstream `bevy_ptr` (including
`MovingPtr` and its deconstruction macros — the exact half this
decision turned down), but nothing in `mid-ecs`/`archetype.rs` calls
into it. See `docs/mid-ptr.md` for the port itself, including a
second, independent MSRV-wall note this introduces. This entry is
left in place rather than rewritten so the original reasoning stays
on record.

---

## Decisions 7 to 13: proposed on 2026-11-16, not yet confirmed

These come from the plan below. Each says what it would unblock and what it costs. Decisions
7 to 10 are the ones that need an answer before the first build; 11 to 13 can wait.

---

### Decision 7 (proposed): Add a small `mid-reflect` crate: a registry that describes components

**Context:** `mid-ecs` knows a component by a dense `ComponentId` and can hand C a byte span
of it (`ffi_span`). It knows no name and no fields, and cannot read a component as a value.
Everything that looks inside a running world needs that description: a protocol to inspect and
edit it, a scene file, an inspector panel, and the component serialization that `docs/mid-ecs.md`'s
"Network Sync" section assumes. Bevy puts `bevy_reflect` at layer 2 for this reason.

**Recommendation:** a crate with a registry of `TypeInfo` (a stable name, size, alignment, a kind,
and fields with a name, kind and offset), read and write through a small `Value` tree, and
registration by declarative macro (`macro_rules!`), so no proc-macro dependency. Hand-written
registration is fine for engine types. A derive can come later as an optional, separate crate.

**Alternatives considered:** port `bevy_reflect` whole (trait objects, dynamic types, patching):
large, and most of it unneeded here. Generate the description from DixScript schemas (the
`@net` idea in `docs/mid-ecs.md`): possible later, cannot be run in the sandbox, and would make
DixScript a build dependency of the core, which the dependency mandate rules out. `serde`:
ruled out by the same mandate.

**Needs your call:** the crate name, and whether macro registration is acceptable.

---

### Decision 8 (proposed): A JSON-RPC remote protocol modeled on the Bevy Remote Protocol, over stdio first

**Context:** the Bevy Remote Protocol is JSON-RPC 2.0 with methods such as `world.query`,
`world.get_components`, `world.insert_components`, `world.mutate_components`,
`world.remove_components`, `world.spawn_entity`, `world.despawn_entity`,
`world.reparent_entities`, `world.list_components`, resource methods, `world.trigger_event`,
`registry.schema`, `rpc.discover`, and "and watch" variants (read from
`bevy_remote/src/builtin_methods.rs` in the fork). It needs reflection and an app, not a renderer.

**Recommendation:** `mid-remote`, the same shape and mostly the same method names, written with
hand-made JSON (as `web/mid-ui-wasm` already does, with no `serde`), acting on `Value` through
`mid-reflect`. Transports in order: lines over stdio (the command line and AI use it), then TCP,
then WebSocket (the app uses it). No async runtime.

**Alternatives considered:** carry it over `mid-net` (QUIC): too heavy for tooling, and its
browser path needs certificates. A binary protocol: faster, but neither a person nor an AI can
read it at a prompt.

**Needs:** Decision 7.

---

### Decision 9 (proposed): One command-line binary, `mid`, in `tools/mid-cli`

The whole engine usable from a terminal, designed for scripts and AI first. Full proposal in
`docs/mid-cli.md`: the same binary runs the development workflow (check, test, bench, docs,
applying a delivery tarball) and, through Decision 8, inspects and edits a live world. Standard
library only, `--json` on every command, no prompts when not a terminal.

---

### Decision 10 (proposed): The app is shared UI logic plus a thin shell per platform

Full proposal in `docs/mid-engine-app.md`. The app's screens are `.mxui` files driven by
`mid-ui`, whose draw list is already the only render seam, so the same screens run in a browser
(Canvas 2D backend, exists), in an Android WebView shell, and later in a native shell, with only
the shell and backend changing. Order: browser page installable as a PWA, then a WebView APK
built by CI, then a native shell. Nothing Android-specific has been built or verified.

---

### Decision 11 (proposed): Text enters `mid-ui` as a draw command and a measuring seam

`DrawCmd` is already `#[non_exhaustive]`, so `Text` can be added without breaking anyone. The
core would define a `TextMeasure` trait that the host implements (the browser backend uses the
canvas, a native one its own font code) so layout can size text before any glyph is drawn.
Shaping and a glyph atlas become their own crate later. Bevy's `text` needs `asset`, `image`
and `clipboard` first; this order avoids that dependency for the app's first screens.

---

### Decision 12 (proposed, deferred): No renderer yet, and a named trigger

A renderer is not on the path to the tools app, and it carries the largest unmade choices:
GPU API (and `wgpu` against the dependency mandate), a software rasterizer, or `raylib` as a
backend (`mid-ui` already names raylib as a supported host). **Trigger:** a `mid-nodes` scene
that has to be seen, i.e. when the app needs a viewport. Then decide with a measured spike.

---

### Decision 13 (proposed, deferred): No asset system until something loads an asset

Bevy's `asset` sits at layer 6 and `scene`, `text`, `image` and `ui` depend on it. Here the
first consumers (scene files, textures, fonts) do not exist yet. **Trigger:** the first
feature that needs to load a file on a platform with no file system (a browser, Android's asset
manager). The design then starts from "how does a wasm page load bytes", not from Bevy's `AssetServer`.

---

## The road to the engine app

### What "the engine app" is taken to mean here

An application, on an Android phone, that controls the engine: a console, a tree of entities
in a running world, an inspector to read and edit their components, and the status of the
project, all drawn with `mid-ui`. This is an assumption, written down so it can be corrected.
It is not a game and not (yet) a visual scene editor; a scene viewport comes at milestone M4.

Two facts about how the work is done shape the plan. Builds happen in CI, because the phone
cannot build the engine, so everything the app needs must be buildable by a workflow. And
the engine does not need to run on the phone for the app to be useful: the app can be a
client of an engine running elsewhere (a CI job, a cloud shell, a desktop), which is exactly
what a remote protocol is for.

### Milestones

| | Milestone | Done when | Needs |
|---|---|---|---|
| M1 | **Terminal engine** | From a terminal, `mid` runs the development workflow with JSON output, starts a headless world, and lists, reads, edits and queries entities and components in it, with schema discovery | Decisions 7, 8, 9 |
| M2 | **Tools UI in the browser** | Console, entity tree and inspector screens, written in `.mxui`, run at `/app/` against a live world; installable as a PWA on Android Chrome | M1, `mid-ui` text, events and widgets, a WebSocket transport reachable over HTTPS |
| M3 | **Android app (WebView shell)** | An APK built by CI hosts the same page and screens and installs on the phone | M2, an Android shell project, a CI job |
| M4 | **Content** | Scenes load and save, nodes exist, anim is wired to transforms, assets load, a 2D viewport draws | `mid-nodes`, scene format, minimal assets, input, a window or shell abstraction, a 2D renderer |
| M5 | **Native Android** | A native shell (no WebView) runs the same screens | M4's backend work |

M1 comes first because it has no unknown outside the ECS: reflection and the protocol are
built and verified here, in the sandbox, with no device and no GPU, and the command line is
useful to AI-driven work the day it exists.

### What is needed before what

Four tracks. An item can start when everything in its "needs first" column is done. Sizes are
relative: S is about one delivery, M a few, L many.

| Track | Item | Needs first | Size |
|---|---|---|---|
| Tools | T1 `mid-reflect` (Decision 7) | `mid-ecs` (done) | M |
| | T2 `mid` command line v0: development-workflow commands | nothing new | S |
| | T3 `mid-remote` core: methods on `Value`, stdio transport | T1 | M |
| | T4 `mid` world commands | T3 | S |
| | T5 TCP, then WebSocket transport | T3 | M |
| UI | U1 text seam in `mid-ui` (Decision 11) | `mid-ui` (done) | M |
| | U2 events, focus, scrolling, text input | U1 | L |
| | U3 widget kit: button, label, field, list, tabs, tree, property grid | U2 | M |
| | U4 screens in `.mxui` and the `/app/` page | U3, T3, T5 | M |
| | U5 Android WebView shell and a CI APK job | U4 | M |
| Content | C1 `mid-nodes` phase 1: names, path lookup, node bundles | hierarchy and transform (done) | S |
| | C2 scene tree and instantiation | C1, T1 | M |
| | C3 scene file loader (DixScript, optional crate) | C2 | M |
| | C4 wire `mid-anim` tracks to `LocalTransform` | transform (done) | S |
| | C5 minimal `mid-asset` (Decision 13) | a consumer | M |
| Runtime | R1 input as messages (touch, keyboard, pointer) | `mid-ecs` (done) | M |
| | R2 window or shell abstraction | R1 | M |
| | R3 2D renderer (Decision 12) | R2, U1 | L |
| | R4 physics: angular motion, collision | `mid-geom` gaps | L |
| | R5 `mid-camera` | R3 | M |

**Critical path to the app (M3):** T1, T3, T5 in one line; U1, U2, U3 in another; then U4 and
U5. The two lines are independent until U4. C4 and C1 are small and can be done whenever
convenient; they do not block the app.

### What to work on now, in order

1. **Confirm Decisions 7 to 10** (or change them). Nothing below should start on an
   unconfirmed answer.
2. **T2, the command line's development-workflow commands.** Small, no dependencies, and the
   first thing that makes AI-driven work cheaper.
3. **T1, `mid-reflect`.** The single biggest unlock: the protocol, scenes, the inspector and
   network sync all wait on it.
4. **T3 and T4, the remote protocol and the world commands.** Completes M1.
5. **U1, text in `mid-ui`,** in parallel with 3 and 4, since it is independent of the ECS.
6. **C1, `mid-nodes` phase 1,** once reflection has settled what a node description looks like.

### What not to do yet, and why

- **A renderer, GPU work, audio, a camera crate.** Not on the path to the tools app, and each
  carries large unmade choices (Decision 12).
- **Assets.** No consumer exists (Decision 13).
- **Physics collision.** `mid-geom`'s gaps are meant to be driven by `mid-physics`, and nothing
  needs it yet.
- **A scene file format before reflection.** It would be written twice.
- **A native Android shell.** It needs M4's backend work first, and the browser path reaches the
  phone sooner with far less risk.
- **ECS performance work.** Paused until the ECS is feature-complete, by decision.

### What is unverified

Everything about Android is unverified: that the Android SDK and NDK are available on the CI
runner (believed, not checked), that a WebView shell builds in a workflow, how a PWA reaches an
engine over HTTPS (a page served over HTTPS cannot open an insecure WebSocket except to
localhost, so the engine endpoint needs TLS or a tunnel), and whether a terminal on the phone
(Termux) could run the `mid` binary. `mid-ecs` and `mid-app` pass `cargo check` for
`wasm32-unknown-unknown`, so an engine inside the page is plausible, but it has not been run.

## Updated build order

```
math ✅ → common (thin) → geom ✅ → net ✅(core) → ecs ✅ → app ✅ → time ✅
    → TOOLS:    reflect → remote → mid CLI                      (M1)
    → UI:       text → events/focus → widgets → screens → shell (M2, M3)
    → CONTENT:  nodes → scenes → assets → anim wiring           (M4)
    → RUNTIME:  input → window → render2d → camera              (M4)
    → physics (angular, collision) when something needs it
    → (trigger-based) mid-platform extensions [Decision 3], mid-camera, native shell
```

Replaces the order in `docs/architecture.md` (`math → common → geom → net → ecs → physics →
anim → nodes`), which put `nodes` last and had nothing for tools. Bevy's own order (the
layers above) is the reason reflection and input move up.

## Open items carried forward

- `mid-math` tests do not compile under `--features force-scalar` (E0308 in `tests/wide_tests.rs`),
  also on unmodified main; `cargo clippy` on `mid-math` reports 17 errors on rustc 1.97
  (lint `mid-ecs` with `--no-deps`).
- The docs site (`web/site/src/SUMMARY.md`) has no pages for `mid-ui`, `mid-ui-mxui`, `mid-app`,
  `mid-time`, `mid-anim`, `mid-physics`, the hierarchy, observers or transforms docs, and the
  home page does not link `/ui/`.
- Check `.github/workflows` filenames if CI looks odd: the `mid-ui-mxui` workflow was once saved
  as `mid-ui-test.yml`.
- `Changed`/`Added` detection on `ChildOf`/`Children` is untested; the cost of a world with the
  hierarchy enabled and of despawning very large subtrees is unmeasured; the cost of a full
  transform pass is unmeasured.
- Real CI has not run `mid-ui` (per its doc).

## Explicitly out of scope for this pass

Blender, Unity, and Godot weren't part of this analysis — the method
used here (parsing `Cargo.toml` dependency graphs) is Rust/Cargo-
specific and doesn't extend to Blender's CMake-based module structure
or to Unity/Godot's closed/differently-organized source. A structural
pass over Blender specifically (its own module boundaries — likely
the most relevant of the three for `mid-nodes`/scene-graph design
later) would need its own methodology; flag if that's wanted as a
separate follow-up rather than folding it into this one.
