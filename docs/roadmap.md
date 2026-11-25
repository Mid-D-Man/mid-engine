# Mid Engine Development Roadmap

Living roadmap. Supersedes `docs/architecture.md`'s "Build order"
section going forward — that section stays as the historical record
(it already does this once, see its own trailing note about the
superseded "Priority column" table); this doc is where build-order
changes get made from now on. Grounded in `docs/bevy-comparison.md`'s
analysis — read that first if the reasoning below needs the source
data.

## Current status (as of 2026-11-18, `mid-engine` @ `2cc1f04`, with the `mid-reflect` and `mid-cli` delivery applied)

Checked by reading the repository (each crate's contents and doc), not by rebuilding
everything. Real CI: the `mid-ecs` and `mid-ecs-ffi` workflows pass on main (reported by
the maintainer on 2026-11-16, after the hierarchy, observer propagation and transform
deliveries). Anything below marked "sandbox only" has not run on real CI; that includes
`mid-reflect`, `mid-cli` and the editor.

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
| `mid-reflect` | Built (`docs/mid-reflect.md`): a `Value` tree with JSON in and out, the `Reflect` trait, three declarative registration macros, `TypeRegistry` with type data, and an optional `math` feature. No dependencies; builds on 1.75. 70 integration tests with `math` and 4 doctests; sandbox only. The bridge to `mid-ecs` is built (`docs/mid-ecs-reflect.md`, feature `reflect` of `mid-ecs`; sandbox only) |
| `tools/mid-cli` | Built (`docs/mid-cli.md`): the `mid` binary with `doctor`, `check`, `test`, `docs lint`, `apply` and `deliver`; its own workspace; 92 tests; sandbox only. The world commands wait for the ECS bridge and the remote protocol |
| `mid-editor`, `mid-editor-ffi`, `apps/editor-android` | Built (`docs/mid-editor.md`, `docs/mid-editor-ffi.md`): the editor core (panels on `mid-ui`, a CPU-projected 3D and 2D viewport, touch gestures, edit and play modes, a demo scene held as a real `mid-ecs` world), its C API, and a raylib 6.0 host with the Android build script and workflow. 127 core tests plus the C API's, 30 script checks and an eight-screenshot desktop session; the first version ran on the phone, the world-backed one is sandbox only. Not built: scene files, editing and undo, gizmos |
| `examples/`, `tools/` | `headless-server`, `custom-transport-demo`; `tools/mdix-compiler` |
| `mid-nodes`, `mid-camera` | Not started. No crates, no design docs until this one (`docs/mid-nodes.md`) |

**Does not exist at all yet:** scene files (reflection is bridged to the ECS, `docs/mid-ecs-reflect.md`,
but nothing saves or loads a world), assets, input, windowing, text and fonts, images, rendering, audio, scenes and
nodes, a protocol for talking to a running world, and any app.

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
| 2 | `ecs`, **`reflect`**, `transform`, `color` | `mid-ecs` built, transform built, color in `mid-math`. **Reflection is built** (`mid-reflect`) and bridged to the ECS behind the `reflect` feature of `mid-ecs` |
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
   some only optionally. It is the foundation under every tool. mid-engine had no equivalent
   until `mid-reflect`, which is built and now bridged to the ECS (`docs/mid-ecs-reflect.md`), and it was the biggest
   single gap between "an ECS" and "an engine that can be looked inside".
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

## Decisions 7 to 13: proposed on 2026-11-16, revised on 2026-11-18

On 2026-11-18 the maintainer said what "the app" is: a graphical editor on the lines of the Unity
editor (a viewport, a hierarchy, an inspector, project panels), used on an Android phone because a
laptop is not available, with its interface built on `mid-ui`. That replaced the first reading, a
tools app with a console and no viewport. It also brought forward the triggers named for
Decisions 12 and 13: a viewport needs a renderer, and a project panel and scene files need to
read and write files. Decisions 10 to 13 are rewritten below. The maintainer also asked for
`mid-reflect` and `mid-cli` to be started, so Decisions 7 and 9 are built and waiting for
confirmation. Decision 8 keeps its shape but is no longer on the editor's critical path. Nothing
here is confirmed unless it says so.

On 2026-11-19 the maintainer answered the app questions: the editor has an edit mode and a play mode,
the viewport is 3D with a 2D mode as in Unity, it goes straight to native Android with the web later,
it reaches the phone as a workflow-built download, and the phone is a Galaxy A13 4G. The maintainer
also pointed at the Android build in the `Tower-of-Babylon` repository. Decisions 10 and 12 below are
updated for that, and the first version of the editor was built (`docs/mid-editor.md`).

---

### Decision 7 (proposed, built): Add a small `mid-reflect` crate: a registry that describes components

**Status:** built as `crates/mid-reflect` (`docs/mid-reflect.md`) on the maintainer's instruction
to start it. The crate name and the macro approach were not explicitly confirmed. Two parts of the
recommendation changed in the build: there are no field offsets (`core::mem::offset_of!` needs
Rust 1.77 and the floor is 1.75), and only unit-variant enums are supported. A `TypeInfo` records
a stable name and a kind, not a size or an alignment. The bridge to `mid-ecs` (T1b) is built: `docs/mid-ecs-reflect.md`.

**Context:** `mid-ecs` knows a component by a dense `ComponentId` and can hand C a byte span
of it (`ffi_span`). It knows no name and no fields, and cannot read a component as a value.
Everything that looks inside a running world needs that description: a protocol to inspect and
edit it, a scene file, an inspector panel, and the component serialization that `docs/mid-ecs.md`'s
"Network Sync" section assumes. Bevy puts `bevy_reflect` at layer 2 for this reason.

**Recommendation:** a crate with a registry of `TypeInfo` (a stable name, a kind, and fields
with a name and a type), read and write through a small `Value` tree, and registration by
declarative macro (`macro_rules!`), so no proc-macro dependency. Hand-written registration is fine
for engine types. A derive can come later as an optional, separate crate.

**Alternatives considered:** port `bevy_reflect` whole (trait objects, dynamic types, patching):
large, and most of it unneeded here. Generate the description from DixScript schemas (the
`@net` idea in `docs/mid-ecs.md`): possible later, cannot be run in the sandbox, and would make
DixScript a build dependency of the core, which the dependency mandate rules out. `serde`:
ruled out by the same mandate.

**Needs a decision:** the crate name, and whether macro registration is acceptable.

---

### Decision 8 (proposed): A JSON-RPC remote protocol modeled on the Bevy Remote Protocol, over stdio first

**Revised 2026-11-18:** the editor embeds its world (Decision 10), so the protocol is not on the
editor's critical path. It serves the command line's world commands and, later, an editor
attached to an engine running elsewhere.

**Context:** the Bevy Remote Protocol is JSON-RPC 2.0 with methods such as `world.query`,
`world.get_components`, `world.insert_components`, `world.mutate_components`,
`world.remove_components`, `world.spawn_entity`, `world.despawn_entity`,
`world.reparent_entities`, `world.list_components`, resource methods, `world.trigger_event`,
`registry.schema`, `rpc.discover`, and "and watch" variants (read from
`bevy_remote/src/builtin_methods.rs` in the fork). It needs reflection and an app, not a renderer.

**Recommendation:** `mid-remote`, the same shape and mostly the same method names, written with
hand-made JSON (as `web/mid-ui-wasm` already does, with no `serde`), acting on `Value` through
`mid-reflect`. Transports in order: lines over stdio (the command line and AI use it), then TCP,
then WebSocket (a remote editor would use it). No async runtime.

**Alternatives considered:** carry it over `mid-net` (QUIC): too heavy for tooling, and its
browser path needs certificates. A binary protocol: faster, but neither a person nor an AI can
read it at a prompt.

**Needs:** Decision 7 and the ECS bridge (both built).

---

### Decision 9 (proposed, built): One command-line binary, `mid`, in `tools/mid-cli`

**Status:** phase 1 is built (`docs/mid-cli.md`). It depends on `mid-reflect` for `Value` and
JSON output, which changes the first design's "standard library only".

The whole engine usable from a terminal, designed for scripts and AI first. The same binary runs
the development workflow (check, test, documentation lint, applying a delivery tarball, building
one) and, through Decision 8, will inspect and edit a live world. `--json` on every command, no
prompts when not a terminal.

---

### Decision 10 (decided on 2026-11-19, native first): The editor is a platform-independent core plus a host per platform

**Decided by the maintainer:** straight to native Android, the web later, delivered by workflows.

**What was built:** the core (`mid-editor`), a C API (`mid-editor-ffi`) and a C host with raylib 6.0
(`apps/editor-android`), built for `armeabi-v7a` with the NDK, `aapt`, `zipalign` and a debug key and no
Gradle, by `scripts/build_editor_android.sh` and `.github/workflows/editor-android.yml`. The shape is
the one `Tower-of-Babylon` and `TerminalViscosity` already use for the same phone. The core takes input
messages and returns one paint list per frame; a host supplies events, text and files and paints the
list. A web host and a WebView app stay possible and do not change the core.

**Why this and not `raylib-rs` or `cargo-apk`:** the maintainer's pipeline is proven in structure for
this phone, matches the FFI-first mandate, and `raylib-rs` needs Rust 1.85 and a different build path.

**Known limits:** raylib 6.0's Android backend ignores a rotation, so the app is portrait only; the
phone is 32-bit ARM only, so `mid-ecs` has to be checked on a 32-bit target before it joins the editor.

**Needs a decision:** how a rotation is handled (a patch to raylib's Android backend or another native
glue), and whether a web host is wanted later.

---

### Decision 11 (proposed): Text enters `mid-ui` as a draw command and a measuring seam

Unchanged in shape. It is now needed by every panel, and by text entry (renaming an entity,
editing a value). `DrawCmd` is already `#[non_exhaustive]`, so `Text` can be added without
breaking anyone. The core defines a `TextMeasure` trait that the host implements (the web host
uses the canvas for measuring and a DOM input for typing and the soft keyboard; a native host
uses its own font code and the platform's keyboard), so layout can size text before any glyph is
drawn. Shaping and a glyph atlas become their own crate later. Bevy's `text` needs `asset`,
`image` and `clipboard` first; this order avoids that dependency for the editor's first screens.

---

### Decision 12 (proposed, revised): A draw-list viewport first, a GPU backend later

**Context:** the trigger named on 2026-11-16, an app that needs a viewport, has fired. The
constraints: the phone has an entry-level GPU (a Galaxy A13; see `docs/mid-engine-app.md`, "Facts
checked"); core crates keep the dependency mandate, with hosts and backends one layer up; every
build is a workflow; and the web host needs a backend that runs in a page. Bevy's own path is
heavy. In the fork (checked at `f3b5346`), `bevy_render` has 26 non-optional Bevy dependencies
(`asset`, `image`, `mesh`, `shader`, `material`, `camera` and `window` among them),
`sprite_render` sits at layer 12 and `feathers` at layer 14.

**Recommendation:** in two steps.

1. **A viewport draw list.** A small set of screen-space primitives (lines, filled polygons,
   circles, text through the host), a software backend for tests and as a fallback, and a Canvas
   2D painter in the web host. A builder projects the 3D world on the CPU with `mid-math`'s camera
   and `mid-geom`'s shapes into that list, so the scene shows as a grid, wireframe and flat
   shapes, with picking by `mid-geom`'s raycasting and gizmos drawn as primitives. It needs no
   asset, image, mesh, shader or material layer.
2. **A GPU backend later,** when textured or lit 3D is wanted. The trigger: a scene whose look
   cannot be judged from wireframe and flat shading. For the native Android host the candidates
   are:

| Option | For | Against |
|---|---|---|
| **raylib** as a host backend | **In use as the native host** (2026-11-19): a documented Android platform target; brings a window, a GLES context, input and text drawing; the maintainer's other repositories build it for this phone | An external C library (acceptable one layer up, not in a core crate); soft-keyboard text entry on Android is unchecked; a rotation is not handled |
| **An own GLES layer** | No dependency; shaped exactly for the engine | The largest option: context creation, an activity and glue by hand, and the same work again for the web |
| **`wgpu`** | Portable, with a web path | Against the dependency mandate unless confined to a backend crate; heavy; the least suited to an entry-level GPU |
| **Software only** | No dependency, deterministic, testable anywhere | Too slow for textured 3D on the phone's CPU; fine for the draw-list viewport |

**Status:** step 1 is built in `mid-editor` (a CPU-projected 3D viewport and a top-down 2D view, with
culling) and painted by the raylib host. The maintainer wants a 3D viewport with a 2D mode, which is
what was built. **Needs a decision:** only the later GPU option. **Unverified:** the cost of the CPU-projected viewport on
the phone, raylib's Android text input, and every Android build step.

---

### Decision 13 (proposed, revised): A byte seam now, asset loading when a loader needs it

**Context:** the trigger named on 2026-11-16, a feature that must load a file on a platform with
no file system, has fired. The editor's project panel lists and opens files, scenes load and
save, and the web host has only `fetch`. Bevy's `asset` sits at layer 6 and `image`, `scene`,
`text`, `render` and `ui` depend on it.

**Recommendation:** a small crate that is only a byte source: read and write bytes by path, with a
backend for a directory, one for files bundled in the Android APK and one for `fetch` in a page,
plus a table of path to type. No asset server, handles, dependency tracking, async loading or hot
reload until a consumer needs one. Scene files are `mid-reflect` `Value` trees as JSON, so they
need no loader (the scene format is open in `docs/mid-nodes.md`; DixScript is the mandate's named
home for scene data, one layer up). Typed loaders (images, fonts, meshes) come with the first
renderer step that needs one.

**Open:** where a project lives on the phone, and how it moves in and out of the repository it is
worked from (`docs/mid-engine-app.md`, "Project storage").


### Decision 14 (decided on 2026-11-24): `mid-ui` is painted by backend crates this repository owns

**Context:** `docs/research-ui-comparison.md` compared `mid-ui` with Slint, Iced, Dioxus Native
(Blitz), egui, msx and the canvas app by reading, and ended with nine questions. The maintainer
answered them on 2026-11-24.

**Decided:**

1. Backend crates in this repository paint `mid-ui`, one layer above the core crates, and use
   `mid-math` for the math (vectors, `Affine2`, `Mat3`, `Color32` and its color spaces, curves).
   The `mid-ui` core stays free of any renderer and window.
2. A CPU backend is implemented first. In a shipped build the GPU backend takes priority. The reason
   given is the hardware the maintainer has.
3. Two draw vocabularies stay: `DrawCmd` in `mid-ui` and the editor's paint list.
4. Text: read the text stacks of the candidates, take what is needed, and write an own stack. Which
   repositories are read first is open.
5. `.mxui` grows expressions and bindings, in the role HTML and CSS have for a web page, for the
   engine and for anything else that uses `mid-ui`.
6. Android soft keyboard: both a native helper and an in-UI keypad. The helper's language, Java or
   Kotlin, is open.
7. An immediate-mode authoring layer for tools sits on top of the retained tree.
8. The whole of `mid-engine` is open source with MidManStudio keeping ownership of it, on terms the
   maintainer sets. Working rule that follows from answer 4: the candidates are read and
   re-implemented, and vendoring any of their code needs its own decision.

**Evidence added with the answers:** the canvas app (egui with `wgpu`, Vulkan, on `android-activity`)
runs smoothly on the phone, by the maintainer's report; the adapter and the frame time are not
known. The reason this roadmap gave for a GPU backend later was an entry-level GPU, so the report is
new evidence for Decision 12, step 2, which stays as written until it is revised.

**Open:** the helper language; the script scope for text (the sources are read in `docs/research-ui-text.md`); which draw vocabulary the CPU backend
consumes first; the expression language (its scope, and whether DixScript 2.0 or an own evaluator
supplies it, given the dependency mandate for core crates); crate names and layout. On the phone
`mid-math` appears to run its scalar backend: SIMD is selected for x86, x86_64, aarch64 and wasm32
with `simd128`, and the `armv7` target is in none of those lists (`crates/mid-math/src/lib.rs`).


### Decision 15 (decided on 2026-11-25): text answers, `mid-math` on 32-bit ARM, FFI and wasm for the UI crates

**Context:** `docs/research-ui-text.md` ended with seven questions and a side finding on `mid-math`.
The maintainer answered some of them on 2026-11-25.

**Decided:**

1. Tests may use external crates as oracles (Skrifa, `ttf-parser`, HarfRust) as dev-dependencies. Core
   crates keep zero dependencies in `[dependencies]`.
2. Hinting is wanted.
3. Android fonts: both bundled fonts and system fonts, with a fallback chain.
4. `mid-math` stays scalar on 32-bit ARM for now. A hand-written armv7 backend will be built later; it
   needs a nightly toolchain, because the NEON intrinsics and the `neon` target feature are unstable
   on 32-bit ARM.
5. The `.mxui` syntax is worked on alongside the backend and text work, not as a separate phase.
6. The FFI surface and wasm testing of the new UI crates are handled as they are written. The
   existing shape to follow: `mid-editor-ffi` (a `staticlib`, `unsafe` only in the FFI crate, every
   entry point in `catch_unwind`, NULL-tolerant, a hand-written header checked against the exported
   symbols) and the wasm workflows `mid-ui-wasm-test.yml` and `mid-math-test-wasm.yml`. The exact
   shape per crate is still to be agreed when the crate exists.

**Open:** the script scope for text (the question was not clear and is asked again in plain terms);
where the text code lives; whether the CPU backend's shape rasterizer and the glyph rasterizer are one
piece of code; where the editing model lives; how hinting is done, since Skrifa's TrueType interpreter
(`glyf`, 11,989 lines) and automatic hinter (10,805 lines) show that this is a large piece
(`docs/research-ui-text.md`).

---

## The road to the engine app

### What "the engine app" is

An editor on the lines of the Unity editor, on an Android phone, with its interface built on
`mid-ui` (the maintainer, 2026-11-18): a viewport, a hierarchy, an inspector, project panels.
Detail is in `docs/mid-engine-app.md`. It replaced the first reading, a tools app with no
viewport, which is why the milestones below changed.

Three facts shape the plan. Builds happen in CI, because the phone cannot build the engine, so
everything the app needs is built by a workflow. The engine is embedded in the editor, because a
viewport and a play mode need the world local. And the phone's GPU is entry-level, so the first
viewport is drawn from lists of lines and polygons projected on the CPU, with a GPU backend later.

### Milestones

| | Milestone | Done when | Needs |
|---|---|---|---|
| M1 | **Terminal engine** | `mid` runs the development workflow with JSON output (built, sandbox only), starts a headless world, and lists, reads, edits and queries entities and components in it | T1b (built), then T3 and T4 |
| M2 | **Editor panels on the phone** | A native app on the phone shows the hierarchy, inspector and console over an embedded world, and edits go through undoable commands. **Built over a real `mid-ecs` world (sandbox only; the phone ran the earlier flat-scene version): hierarchy, inspector through reflection, selection by entity, play with restore. Not built: edits and undo** | T1b (built), U3, E1, and the app running on the phone |
| M3 | **Viewport v0** | The scene shows in the viewport (grid, wireframe and flat shapes), an orbit camera works by touch, a tap selects, a gizmo moves the selection, and hierarchy and inspector follow. **Built except the gizmo** | M2, V4, C1 |
| M4 | **Content** | Scenes load and save, the project panel lists files, play mode enters and leaves | C2, C3, C5, E2 |
| M5 | **Android app** | A CI-built APK installs on the phone and runs the editor natively, with files in app storage and a working soft keyboard. **The build exists and has not been run** | The first workflow run, then text entry |
| M6 | **Rich rendering** | Textured and lit 3D, sprites, animation on transforms | The GPU backend step of Decision 12, typed loaders, C4 |

M1 is nearly in hand and has no unknown outside the ECS. M2 comes next because it puts the editor
core in front of a real screen and a real touch loop without waiting for the Android build chain.

### What is needed before what

Seven tracks. An item can start when everything in its "needs first" column is done. Sizes are
relative: S is about one delivery, M a few, L many.

| Track | Item | Needs first | Size |
|---|---|---|---|
| Tools | T1 `mid-reflect` (Decision 7) | `mid-ecs` (done) | built, sandbox only |
| | T1b ECS bridge: enumerate an entity's components, then a component table on the registry | T1 | built, sandbox only (`docs/mid-ecs-reflect.md`) |
| | T2 `mid` command line, phase 1 | nothing | built, sandbox only |
| | T3 `mid-remote` core: methods on `Value`, stdio transport | T1b | M |
| | T4 `mid` world commands | T3 | S |
| | T5 TCP, then WebSocket transport | T3 | M, for a remote editor only |
| UI | U1 text seam (Decision 11) | `mid-ui` (done) | built as a measure callback and a bundled font, sandbox only |
| | U2 events, focus, scrolling, touch gestures, text input | U1 | partly built (tap, drag, pinch); focus, scrolling and text entry remain, L |
| | U3 widget kit: button, label, field, list, tabs, tree, property grid, splitter | U2 | M |
| | U4 editor screens in `.mxui` | U3, T1b | M |
| Viewport | V1 viewport draw list (Decision 12) | nothing | built as the paint list, painted by the host |
| | V2 viewport builder: camera, grid, shapes; a bench | V1 | built; the bench (the 5000-cube chip) has not run on the phone |
| | V3 picking and selection | V2 (C1 for names) | built |
| | V4 gizmos | V3, U2 | M |
| Hosts | H1 web host: page, input messages, embedded world | U1, R1 | later, by decision, M |
| | H2 Android APK spike: a CI-built APK that draws a list and reports touch | nothing | built (script and workflow), not yet run on GitHub |
| | H3 native Android host | H2, V1, U2 | built as the same host, not yet run on a phone |
| | H4 GPU backend (Decision 12, step 2) | H3 | L |
| Editor | E1 edit commands, undo and redo | T1b | M |
| | E2 scenes load and save; play mode | C2, C3 | M |
| Content | C1 `mid-nodes` phase 1: names, path lookup, node bundles | hierarchy and transform (done) | S |
| | C2 scene tree and instantiation | C1, T1 | M |
| | C3 scene file | C2 | M |
| | C4 wire `mid-anim` tracks to `LocalTransform` | transform (done) | S |
| | C5 byte seam (Decision 13) | nothing | S |
| Runtime | R1 input as messages (touch, keyboard, pointer) | `mid-ecs` (done) | M |
| | R4 physics: angular motion, collision | `mid-geom` gaps | L |
| | R5 `mid-camera` | C1 | M |

**Critical path to M3:** T1b feeds the inspector and the hierarchy (U4, E1). The UI line U1, U2,
U3 feeds U4. The viewport line V1 to V4 does not depend on the ECS and can start now. The host
line H1 needs U1 and input messages. The lines meet at U4 and V3. H2, the Android spike, needs
nothing and is the cheapest way to learn about the largest unknown. T3 to T5, the protocol, are
not on this path.

### What to work on now, in order

1. **Done: the `Editor Android APK` workflow built an app that installed and ran on the phone**
   (version name 0.1.1, a Galaxy A13 4G, `armeabi-v7a`). The numbers and the observed problems are in
   `docs/mid-editor.md`, "Verification status" and "Not built yet". What is left is the phone's view of
   the world-backed editor, which is item 2.
2. **Run the editor workflow again and look at it on the phone.** The editor now holds a real
   `mid-ecs` world (`docs/mid-editor.md`), which puts the ECS and `rayon` into the 32-bit ARM build
   for the first time. Check that the APK builds, that the hierarchy and inspector draw properly, and
   send the Console numbers in play mode at 1000 and 5000 cubes. The command line's world commands
   can start once the remote protocol exists.
3. **E1 and V4:** edit commands with undo, and gizmos, so the editor can change something.
4. **U2 and U3:** focus, scrolling and text entry, then the widget kit, so the panels stop drawing their
   own buttons.
5. **Decide the project path** (`docs/mid-engine-app.md`, "Project storage") and a rotation plan.
6. **C1, `mid-nodes` phase 1,** when convenient.
7. **Research.** The ECS comparison is written (`docs/research-ecs-comparison.md`: mid-ecs, bevy_ecs
   and flecs, thirteen gaps ranked, a three-way benchmark designed and not built). The UI comparison
   is written too (`docs/research-ui-comparison.md`: Slint, Iced, Dioxus Native, egui, msx and the
   canvas app, thirteen gaps ranked, a phone probe designed and not built, nine questions). The text
   comparison follows (`docs/research-ui-text.md`: Parley, HarfRust, Skrifa, cosmic-text, swash,
   fontdue and the egui text code sized by stage, five tiers of script support, eleven gaps ranked,
   a Latin glyph probe designed and not built, seven questions). All three are reading, not
   measuring.
   A fourth note, `docs/research-ecs-from-a-c-game.md`, lists what a C game needs from an ECS (Tower
   of Babylon chose flecs on 2026-11-25) and where `mid-ecs` stands against each item.

### What not to do yet, and why

- **A GPU renderer,** until the draw-list viewport has shown what it cannot do (Decision 12,
  step 2). Audio and a camera crate stay deferred.
- **An asset server.** A byte seam is enough (Decision 13).
- **Physics collision.** `mid-geom`'s gaps are meant to be driven by `mid-physics`, and nothing
  needs it yet.
- **A scene file format before nodes and reflection are settled.** It would be written twice.
- **A native-only commitment** before the APK spike has run.
- **ECS performance work.** Paused until the ECS is feature-complete, by decision.

### What is unverified

The editor has not run on the phone. Unverified: the NDK build and raylib's Android build in this
repository's workflow (the runner images list an Android SDK and NDK in their release notes, and the
maintainer's other repositories build the same way), the armv7 Rust build and link, the highp shader and
OpenGL ES 2 rendering, touch, the APK installing, and the cost of the viewport on the phone. The editor
crates were tested on rustc 1.97.1 and 1.75.0, on the default and the scalar math backend, but no crate
in the repository has run on a 32-bit target; the workflow adds a 32-bit x86 test job for that. `mid-ecs`
has not been checked on 32 bits at all. `mid-ecs` and `mid-app` pass `cargo check` for
`wasm32-unknown-unknown` and the wasm32 tests were type-checked, not run, which matters for a later web
host. Soft-keyboard text entry and file access on the phone's storage are unchecked. The public facts that
were checked are listed in `docs/mid-engine-app.md`, "Facts checked".

## Updated build order

```
math ✅ → common (thin) → geom ✅ → net ✅(core) → ecs ✅ → app ✅ → time ✅
    → TOOLS:     reflect ✅(core) → ECS bridge → remote → world commands; mid CLI ✅(phase 1)
    → UI:        text → events and touch → widgets → editor screens              (M2)
    → VIEWPORT:  draw list + software backend → projection → picking → gizmos     (M3)
    → HOSTS:     web host → Android APK spike → native host                       (M2, M5)
    → CONTENT:   nodes → scenes → byte seam → undo and play mode                  (M3, M4)
    → RUNTIME:   input → GPU backend, typed loaders, camera → anim wiring         (M6)
    → physics (angular, collision) when something needs it
    → (trigger-based) mid-platform extensions [Decision 3]
```

Replaces the order in `docs/architecture.md` (`math → common → geom → net → ecs → physics →
anim → nodes`), which put `nodes` last and had nothing for tools. Bevy's own order (the layers
above) is the reason reflection and input move up.

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
- Real CI has not run `mid-ui` (per its doc), or `mid-reflect` and `mid-cli` (their workflows,
  `mid-reflect-test.yml` and `mid-cli-test.yml`, are new).
- The docs site also has no pages for `mid-reflect`, `mid-cli` or the engine app docs.

## Explicitly out of scope for this pass

Blender, Unity, and Godot weren't part of this analysis — the method
used here (parsing `Cargo.toml` dependency graphs) is Rust/Cargo-
specific and doesn't extend to Blender's CMake-based module structure
or to Unity/Godot's closed/differently-organized source. A structural
pass over Blender specifically (its own module boundaries — likely
the most relevant of the three for `mid-nodes`/scene-graph design
later) would need its own methodology; flag if that's wanted as a
separate follow-up rather than folding it into this one.
