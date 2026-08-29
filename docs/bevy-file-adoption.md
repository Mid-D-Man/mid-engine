# Bevy File Adoption Tracker

A living map of `Mid-D-Man/bevy`'s `bevy_ecs` (the file tree we read from
directly, per `docs/bevy-comparison.md`) against what mid-ecs actually
needs next. Where a Bevy file does essentially the same job we'd
otherwise build from scratch, adapt it — rename, relocate, minimal
changes. Where it doesn't, this doc says why, so the "why" doesn't have
to get re-derived every time someone goes looking.

**Real, current bottom line up front:** almost nothing here is adopt-
*today*-ready, and that's not a bad sign — it's an accurate reflection of
where mid-ecs actually is. mid-ecs is a storage engine. Nearly all of
`bevy_ecs`'s ~119,000 lines are the *runtime on top of* a storage engine
(systems, schedules, observers, events, hierarchy) — mid-ecs hasn't
started any of that yet. So most of this doc is a map for later, not a
todo list for now. The one genuine adopt-now candidate is called out
explicitly below.

## How a file/module gets judged here

Two independent gates, both real, both distinct from "is this a good
idea":

1. **External-crate gate** — does adapting this file require one of
   Bevy's own *internal* crates (`bevy_reflect`, `bevy_tasks`,
   `bevy_platform`, `bevy_ptr`, `bevy_utils`, `bevy_ecs_macros`) that
   mid-engine hasn't built? If yes: hold off, come back once that
   crate (or an equivalent) exists.
2. **Internal-prerequisite gate** — does it require some *other*
   mid-ecs concept that doesn't exist yet (a `System`/`Schedule`
   concept, an `Observer` concept, `Relationship`/hierarchy, an
   event/message bus, query filter machinery beyond two-tuple
   `query2`)? This is the gate that actually blocks most of the
   codebase right now, not gate 1 — see the real numbers below.

A file can clear gate 1 (touches none of the six) and still be
completely unusable today because of gate 2. `traversal.rs` and
`component/constants.rs` are both real examples of this, checked
directly (see the shortlist section).

## Real, measured footprint of the six hold-off crates

Checked directly against every file in `bevy_ecs/src` (154 files, one
`grep` per crate, not estimated):

| Crate | Files touching it | What it actually provides | Recommendation |
|---|---|---|---|
| `bevy_platform` | 48 / 154 | `no_std`-portable time, sync primitives, hashing (`HashMap`/`HashSet` type aliases over a fast hasher), collections that work the same in `std` and `no_std` builds | The most pervasive one. Worth hand-rolling a small `mid-platform`-style shim eventually (mid-collections is already `no_std` + `alloc`, so the appetite for this exists) — but it's genuinely a project of its own, not a quick add. Hold off. |
| `bevy_utils` | 48 / 154 | Grab-bag: `Duration`/`Instant` helpers, small macros, `default()` helper, a couple of `HashMap` conveniences | Mostly thin. Likely hand-rollable piecemeal, file by file, as each blocked file actually gets adopted — not worth building speculatively ahead of need. |
| `bevy_ptr` | 40 / 154 | `OwningPtr`, `Ptr`, `PtrMut` — type-erased raw-pointer wrappers used for the unsafe table/column memory management | Small, focused, and exactly the kind of thing mid-ecs's own Archetype Core could eventually want if it ever moves off `Box<dyn Any>` per-component boxing toward raw erased storage. Worth reading closely when that day comes; not needed for anything adoptable today. |
| `bevy_reflect` | 39 / 154 | Full runtime reflection (type info, dynamic get/set by field name, serialization hooks) | Given mid-engine's performance mandate, this may be a deliberate **skip**, not a hold-off — runtime reflection has a real cost and mid-ecs doesn't have an obvious consumer for it (no editor, no dynamic scripting layer yet). Revisit only if DixScript or a future editor tool genuinely needs it. |
| `bevy_ecs_macros` | 27 / 154 | The `#[derive(Component)]`/`#[derive(Bundle)]`/`#[derive(Resource)]`/`#[derive(Event)]` proc macros | This is really "does mid-ecs want a `Component` trait + derive macro at all" — a real, previously-deferred design question (see `docs/mid-ecs.md`'s `StorageClaims` section: the full `Component` trait redesign is accepted-deferred, not forgotten). Anything gated on this waits until that question gets revisited. |
| `bevy_tasks` | 10 / 154 | Task pool abstraction for parallel query iteration / async systems | Least pervasive. mid-ecs's own `rayon` dependency (already present, wasm-gated) is the more likely path here when parallel iteration actually gets built, not a `bevy_tasks` port. |

## Per-module verdict

| Module | Real size | Hold-off crates | Real prerequisite | Verdict |
|---|---|---|---|---|
| `storage/` | 3,720 ln | `bevy_platform`, `bevy_ptr`, `bevy_utils` (module-wide; **not** every file — see shortlist) | none for the clean files | Mixed — see shortlist for the one real adopt-now file (`thin_array_ptr.rs`). The rest (`table/`, `sparse_set.rs`) describes Bevy's Table+SparseSet-per-archetype model, which `docs/mid-ecs.md`'s own top doc already compares against and deliberately diverges from (Sparse Shell + Archetype Core as two hard-separated systems, not one unified per-component-storage-choice archetype). Reference, not a port target. |
| `query/` | 17,840 ln | all six, but concentrated in `fetch.rs`/`state.rs`/`iter.rs` | mid-ecs's own `Query`/`query2` (326 ln) is a v1 by design — see its own doc comment on why it doesn't need the smaller-side optimization yet | Mostly build-from-scratch-when-needed, matching mid-ecs's own stated "no real workload to justify more yet" philosophy. `access.rs` (1,960 ln, clean of the six, needs `fixedbitset` + a `System`/parallel-scheduling concept) is worth a real read once systems work starts — it's Bevy's read/write access-conflict tracker, the actual mechanism that makes parallel system execution sound. |
| `component/` | 3,980 ln | `bevy_reflect`, `bevy_platform`, `bevy_ptr`, `bevy_utils`, `bevy_ecs_macros` | `Component` trait + derive (see table above) | Hold off as a whole. `component/constants.rs` (16 ln, clean) looked like a shortlist candidate at a glance but is entirely Bevy's own observer/lifecycle marker IDs — not usable until an observer system exists. |
| `entity/` | 13,787 ln | all six | mostly a `System`/query-filter/observer prerequisite, not storage | mid-ecs's own `Entity` (packed `u64`, `GenerationalIndex`-backed, real, tested, real precedent already documented) is simpler than Bevy's and already done. Not a port target — mid-ecs's version is the *better fit* for the FFI mandate (Bevy's `Entity` is internally `NonMaxU32` + generation, not built with a C ABI in mind at all — confirmed, zero `extern "C"` anywhere in `bevy_ecs`). |
| `bundle/` | 3,055 ln | `bevy_platform`, `bevy_ptr`, `bevy_utils`, `bevy_ecs_macros` | `Component` trait (Bevy's `Bundle` is built on top of it) | mid-ecs already built its own, smaller `Bundle` (arity-8 tuple macro, `pub(crate)` + `#[allow(private_bounds)]` workaround — see `docs/mid-ecs.md`). Not a port target; already diverged on purpose, and it's what surfaced the `insert_bundle` archetype-migration bug fixed this session. |
| `world/` | 17,052 ln | all six | `Component` trait, `Resource`, mostly everything | Reference-only for API shape (this is exactly where the `spawn`/`query`/`entity_mut`/`insert`/`remove` signatures for the bench crate came from). Not adoptable as files — mid-ecs's `World` (1,425 ln) is already its own, much smaller thing by design. |
| `schedule/` | 16,014 ln | `bevy_tasks`, `bevy_platform`, `bevy_utils`, `bevy_ecs_macros` | a `System` concept doesn't exist in mid-ecs at all yet | Whole module on hold — this *is* "the scheduler," called out as not-yet-started in `docs/mid-ecs.md`'s own carryover list. `schedule/graph/tarjan_scc.rs` (282 ln, clean of the six, needs `smallvec` + the module's own `DiGraph`/`GraphNodeId` types) is a real, self-contained cycle-detection algorithm worth keeping in mind for whenever system-dependency-graph validation gets built — genuinely portable *logic*, just not portable as a standalone file today. |
| `system/` | 18,554 ln | all six | same as `schedule/` | Same as above — not started, whole module on hold. |
| `observer/` | 3,327 ln | `bevy_reflect`, `bevy_platform`, `bevy_ptr`, `bevy_utils` | `Component` trait + event/lifecycle machinery | Hold off, whole module. |
| `event/` + `message/` | 1,707 + 2,025 ln | `bevy_ptr`/`bevy_ecs_macros` (event); `bevy_reflect`/`bevy_tasks`/`bevy_platform`/`bevy_ecs_macros` (message) | an event/message bus concept, not built | Hold off. Smallest of the not-started subsystems by line count, if priority ordering comes up. |
| `relationship/` + `hierarchy.rs` | 3,296 + 1,128 ln | `bevy_reflect`, `bevy_platform`, `bevy_ptr`, `bevy_utils` (relationship); `bevy_reflect` (hierarchy) | `Component` trait, observer machinery | Hold off. This is real parent/child scene-graph support — genuinely useful eventually, not urgent. |
| `reflect/` | 2,016 ln | `bevy_reflect`, `bevy_utils`, `bevy_ecs_macros` | — | Matches the **skip** recommendation above, not hold-off. Revisit only if something concrete needs it. |
| `change_detection/` | 3,125 ln | `bevy_reflect`, `bevy_ptr`, `bevy_ecs_macros` | `Component` trait | Hold off. Real, wanted eventually (`Added<T>`/`Changed<T>` dirty-tracking matters for `mid-net`'s own sync story — see `SyncRegistry`'s doc comment on what it's a foundation *for*), but gated on `Component` first. |
| `archetype.rs` (top-level) | 1,002 ln | `bevy_platform` only | — | Reference-only, already read closely and cited directly in mid-ecs's own `archetype.rs` doc comments (`Table { columns: ImmutableSparseSet<ComponentId, Column>, entities: Vec<Entity> }` — verified against this exact file). Not a port target; mid-ecs's Archetype Core is deliberately a different, simpler shape. |

## Shortlist: what's actually adoptable right now

Checked each of these directly, not from the module-level table above
(module-level "touches X" doesn't mean every file in it does — see
"How a file/module gets judged" above).

- **`storage/thin_array_ptr.rs` (322 ln) — real adopt-now candidate.**
  Zero Bevy-crate coupling: only `core`/`alloc` plus one tiny in-crate
  debug-assertion trait. A hand-rolled `ManuallyDrop<Box<[T]>>`-shaped
  type with the capacity/length deliberately cut out for performance —
  exactly mid-collections' own stated territory (`mid-collections.md`'s
  design doc already lists "sparse sets, generational arenas,
  lock-free ring buffers, intrusive lists, hierarchical bitsets" as the
  piecemeal build list; this is a real, tested, working example of
  that same kind of primitive, free to read closely or adapt directly
  whenever mid-collections wants a fixed-capacity array type).
- **`never.rs` (39 ln) — keep in back pocket, not urgent.** A real,
  documented workaround for Rust 2024 edition's never-type-fallback
  change, needed only once mid-ecs has trait impls over function
  pointers/closures (a `System`-trait-shaped pattern) *and* is on
  edition 2024 itself (mid-ecs is on 2021 today). Zero Bevy coupling,
  trivially portable when that day comes — just not relevant yet.
- **`schedule/graph/tarjan_scc.rs` (282 ln) — logic is portable, file
  isn't, yet.** A genuinely standalone cycle-detection algorithm
  (Pierce's memory-efficient Tarjan's SCC variant), clean of the six,
  needs only `smallvec` plus the module's own `DiGraph`/`GraphNodeId`
  types. Real, valuable reference for whenever a scheduler needs to
  validate its system-dependency graph has no cycles — worth writing
  down here so nobody re-derives an SCC algorithm from scratch later
  when this one's sitting right there, verified and documented.
- Everything else checked directly during this pass
  (`component/constants.rs`, `traversal.rs`, `world/entity_fetch.rs`,
  `error/mod.rs`, `query/access.rs`) cleared the external-crate gate
  but failed the internal-prerequisite gate — real, checked, not
  adoptable today. Noted in the module table above rather than
  repeated here.

## External (non-Bevy-workspace) crates seen along the way

Surfaced while reading `Cargo.toml` and the files above — not a
commitment to add any of these, just the real, current list to weigh
hand-roll-vs-add against, per the "check first whether it's something
we can reasonably hand-roll" policy in `CONTRIBUTING.md`:

`bitflags` (2.3), `fixedbitset` (0.5), `thiserror` (2), `derive_more`
(2), `nonmax` (0.5.4), `arrayvec` (0.7.4), `smallvec` (1, with `union`
+ `const_generics` features), `indexmap` (2.5.0), `variadics_please`
(2.0), `log` (0.4), `bumpalo` (3), `slotmap` (1.0.7),
`concurrent-queue` (2.5.0, `portable-atomic`-gated on platforms
without full atomic support).

`slotmap` is already precedent, not a new consideration — mid-ecs's
own `Entity::as_ffi`/`from_ffi` packing is explicitly grounded in
`slotmap::KeyData`'s real design (see `docs/mid-ecs.md`), just
hand-rolled rather than taken as a dependency. `fixedbitset` is the
one most likely to come up for real soon, if `query/access.rs`-style
read/write conflict tracking gets built alongside a future scheduler.

## Updating this doc

When a hold-off item's prerequisite actually gets built (a `Component`
trait lands, a scheduler starts, etc.), come back here first — several
"hold off" rows above turn into real adopt candidates the moment their
one blocking prerequisite exists, and this doc is the place that
already did the legwork of finding them.
