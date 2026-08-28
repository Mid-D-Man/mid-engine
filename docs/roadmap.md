# Mid Engine Development Roadmap

Living roadmap. Supersedes `docs/architecture.md`'s "Build order"
section going forward — that section stays as the historical record
(it already does this once, see its own trailing note about the
superseded "Priority column" table); this doc is where build-order
changes get made from now on. Grounded in `docs/bevy-comparison.md`'s
analysis — read that first if the reasoning below needs the source
data.

## Current status (as of 2026-08-27, `mid-engine` @ `3535e46`)

| Crate | Status |
|---|---|
| `mid-math` | Practically done — f32/f64 SIMD vectors/matrices/quaternions, curves, noise, color, camera math, fixed-point. LWC view-space-shift primitives added this pass. |
| `mid-common` | Thin — `string/`/`ffi/` substantial, `error.rs`/`traits.rs`/`types.rs` still 2-3 lines each |
| `mid-geom` | Substantial — shapes, intersection tests, raycasting, frustum culling. Known gaps (OBB, capsule-vs-AABB, broadphase) confirmed non-unique to mid-engine, see `docs/bevy-comparison.md` §4 |
| `mid-net` | In progress, substantial — wire/reliable/transport/connection/FFI all real and CI-verified; both `Transport` backends (quinn, wasm) written, quinn verified further than wasm |
| `mid-ecs` | In progress, started — `World`, Sparse Shell, Archetype Core (with real dynamic migration), `query`/`query2` all real; `sync`/FFI-for-component-data still open |
| `mid-collections` | Two pieces built (SparseSet, GenerationalIndexAllocator + FFI wrapper), rest is design-doc-only, built piecemeal as `mid-ecs` needs them |
| `mid-log` | Good — lock-free SPSC ring buffer, tiered levels, non-blocking |
| `mid-trace` | Exists, feature-gated no-op by default (`profile`/`tracy`/`perf`) |
| `mid-physics`, `mid-anim`, `mid-nodes`, `mid-camera` | Not started — no crate directories yet |

Stated build order going in: `math → common → geom → net → ecs →
physics → anim → nodes`.

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

---

## Updated build order

```
math ✅ → common (thin, fine as-is) → geom ✅ → net (continue) → ecs (continue)
    → mid-app (NEW — Decision 1)
    → mid-time (NEW — Decision 2)
    → physics → anim → nodes
    → (trigger-based, not scheduled) mid-platform [Decision 3], mid-render, mid-camera [depends on mid-render existing]
```

`mid-collections` and geometry gap-filling stay exactly as
`docs/architecture.md` already describes them — not phases, pulled in
reactively. Nothing here changes that.

## Immediate next steps (this pass's concrete output)

1. Land the `[profile.wasm-release]` + `[workspace.lints]` additions
   (Decision 5) — smallest, least risky change here, no design work
   needed.
2. Scope `mid-app`'s v1 (Decision 1): `App`, a small fixed schedule,
   `Plugin` trait, loop driver. Start once current `mid-ecs`/`mid-net`
   work reaches a stable point — doesn't need to wait for either to
   be "done," just not mid-refactor.
3. Scope `mid-time` (Decision 2) alongside it — the two are related
   enough (schedule needs a tick source) that designing them together
   avoids a rework later.
4. Look at the `tools/mdix-compiler` doc/manifest mismatch flagged in
   `docs/bevy-comparison.md` §7 — either uncomment the real dependency
   or correct the doc.

## Explicitly out of scope for this pass

Blender, Unity, and Godot weren't part of this analysis — the method
used here (parsing `Cargo.toml` dependency graphs) is Rust/Cargo-
specific and doesn't extend to Blender's CMake-based module structure
or to Unity/Godot's closed/differently-organized source. A structural
pass over Blender specifically (its own module boundaries — likely
the most relevant of the three for `mid-nodes`/scene-graph design
later) would need its own methodology; flag if that's wanted as a
separate follow-up rather than folding it into this one.
