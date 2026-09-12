# mid-ecs

Data-oriented ECS using Structure of Arrays (SoA) layout.

## Status

**Entity allocation and the Sparse Shell are both real now.** `World`
(`crates/mid-ecs/src/world.rs`) — `spawn`/`despawn`/`is_alive`,
generation-checked handles via `Entity`/
`mid_collections::GenerationalIndexAllocator`. `World::insert`/`get`/
`get_mut`/`remove`/`has` (`crates/mid-ecs/src/component.rs`'s
`SparseShell`) — any `T: 'static` attachable to any entity, no upfront
declaration, type-erased via a dense `ComponentId` (registered once per
type, `TypeId` only used at that registration step) rather than a
`TypeId`-keyed `HashMap` on the hot path — design grounded directly in
Bevy ECS's real `Components`/`ComponentId`/`Table` source, not invented
independently. 66/66 real tests passing across `mid-collections` +
`mid-ecs` (verified by temporarily stripping `rayon` and `criterion`
locally, same technique used every time these MSRV walls come up,
restored unchanged afterward).

`despawn` correctly removes every attached component *before* freeing
the entity's generational slot — load-bearing ordering, not incidental,
since `SparseSet` looks up purely by raw index and can't itself tell a
stale handle from a live one sharing a reused index. Every `World`
component method checks liveness first for the same reason. Both
properties have dedicated tests proving them directly
(`reused_slot_does_not_inherit_the_old_entitys_components`,
`stale_handle_cannot_read_the_live_entity_now_sharing_its_index`), not
just described in a comment.

One real self-caught inconsistency worth noting: `insert` initially had
a `debug_assert!` for the dead-entity case, on the reasoning that it's
"almost always a caller logic bug." The test written to prove the safe
fallback immediately panicked instead, in a debug/test build — which
was the actual bug: that `debug_assert!` directly contradicted this
codebase's own established convention everywhere else
(`SparseSet::remove`, `GenerationalIndexAllocator::deallocate`) of never
panicking on this class of misuse. Removed, not worked around.

`Archetype`, `Query`, `sync`, `ffi` are still stubs. The Archetype Core
(dense/table storage for stable, always-present components — the other
half of the Hybrid ECS Architecture below) doesn't exist yet; nothing in
`component.rs` is trying to be both.

**Query — real for one and two component types.**
`World::query<T>()` iterates every `(Entity, &T)` currently alive with a
`T` attached; `World::query2<A, B>()` intersects two — every
`(Entity, &A, &B)` for entities alive with *both*. 7 new tests (39/39 in
`mid-ecs` total, 73/73 across `mid-collections` + `mid-ecs`), all passing
on the actual first run — including the two that matter most:
`query_excludes_despawned_entities` and
`query2_excludes_a_despawned_entity_even_if_it_had_both`, proving
`despawn`'s component cleanup and `query`'s iteration agree with each
other, not just each independently claiming to be correct.

Deliberate v1 scope, not an oversight: `query2` always drives iteration
off its first type parameter and checks the second per-entity, rather
than picking whichever side is actually smaller — a real optimization
for a mismatched pair, but nothing in this workspace has a real query
shape yet that would justify the extra complexity over shipping the
correct, simpler version first. No `query3`+, and no generic tuple-based
`Query<T>` trait system (the shape real ECS crates converge on for
arbitrary arity) — both are natural follow-ons once real usage patterns
exist to design against, not before.

Organizational fix caught, not by review: `query`/`query2` were first
written directly on `World` in `world.rs`. Moved to `query.rs` shortly
after — the file that already existed specifically for this — where
they belonged from the start. Implementations unchanged, `World`'s
fields made `pub(crate)` so `query.rs` can reach them.

**Archetype Core — real, with full dynamic migration, not a simplified
static-at-spawn version.** `crates/mid-ecs/src/archetype.rs`:
`Archetypes` — dense SoA [`Table`]s (one contiguous `Vec<T>` per
component type, in lockstep by row), keyed by exact component-type-set
signature, with real migration when `World::insert_static`/
`remove_static` change an entity's set. `World::get_static`/
`get_static_mut`/`has_static` round out the API — named distinctly from
the Sparse Shell's `insert`/`get`/etc. since a component type has to
live in exactly one of the two systems, and there's no enforcement yet
beyond caller discipline (a `Component` trait fixing each type's
storage strategy once, matching where Bevy eventually landed, is a real
future refinement).

Grounded in Bevy ECS's real source, cloned and read directly (not
memory, not search-result excerpts) — `archetype.rs` (1002 lines),
`storage/table/{mod,column}.rs` (1428 lines). Confirmed, not assumed:
`Table` really is `{ columns: SparseSet<ComponentId, Column>, entities:
Vec<Entity> }`, the same shape `component.rs`'s `SparseShell` converged
on independently, now confirmed twice as the right structure for this
class of problem. `Edges`-style memoized add/remove transition caching
is real here too (`Archetype::add_edges`/`remove_edges`), same idea as
Bevy's, simpler storage (`HashMap` over a dedicated sparse-array type
this project doesn't have and didn't need to build for this alone).

Deliberate, stated divergence from Bevy, not an oversimplification:
Bevy's real row-migration is `unsafe`, raw-pointer, merge-join code with
change-detection ticks this project doesn't have yet. This module gets
the *same real capability* — genuine dynamic migration, any entity, any
component, at any time, no "components fixed at spawn" restriction —
through safe Rust instead: each migrated value is briefly boxed
(`Column::swap_remove_and_forget`/`push_any`) rather than raw-copied.
One heap allocation per moved component per structural change — not per
frame, not per query, only on the path this whole Sparse-Shell-vs-
Archetype-Core split exists specifically to keep rare. Zero `unsafe`,
matching `SparseSet`/`GenerationalIndexAllocator`'s own precedent
throughout this project.

Single-component structural changes only (not Bevy's general `Bundle`
trait for atomic multi-component changes) — deliberate, not a
limitation nobody noticed: a single-component add is always a strict
superset transition, a single-component remove always a strict subset,
which is exactly what lets every *other* column move unconditionally
with no merge-join needed to work out what's actually shared.

24 new tests (15 in `archetype.rs` + 9 `World`-level integration tests),
99/99 total across `mid-collections` + `mid-ecs`, all passing on the
actual first real run after two real, caught-by-clippy `Box::new(_)`
lint fixes and two doc-link fixes. The test that matters most:
`swap_remove_during_migration_fixes_up_the_swapped_entitys_row` —
proves that when a *middle* entity migrates out of a shared archetype,
the entity swapped into its old row stays correct not just for reading
afterward, but for *its own future migrations* too. Still wasm32-clean:
re-checked via `cargo tree --target wasm32-unknown-unknown` after —
`mid-ecs` still resolves to depending on only `mid-collections` for
that target, this module added no new dependencies.

## Target

100 000+ entities at 60 Hz physics on a single core.
Parallelised queries via rayon — on native. See Platform below.

## Platform

`mid-ecs` targets native *and* wasm32 (browser), same as the rest of the
workspace (`docs/architecture.md`'s core commitments). `rayon` — real
OS threads under the hood — doesn't work on `wasm32-unknown-unknown` the
way it needs to, so it's gated to non-wasm32 targets only in
`crates/mid-ecs/Cargo.toml`, under
`[target.'cfg(not(target_arch = "wasm32"))'.dependencies]` — the same
target-gating pattern already established in
`crates/mid-net/transport-wasm/Cargo.toml`.

Confirmed directly, not assumed: `cargo tree --target
wasm32-unknown-unknown -p mid-ecs` showed rayon's entire transitive tree
(`rayon-core`, `crossbeam-deque`/`epoch`/`utils`, `either`) resolving
into the wasm32 dependency graph *before* the gate existed, and cleanly
absent after — `mid-ecs` resolves to depending on only
`mid-collections` for that target. `.github/workflows/mid-ecs-test.yml`
now has a real "Check wasm32 build" step
(`cargo check --target wasm32-unknown-unknown`) proving the crate
actually compiles clean for that target on real CI, not just that its
dependency graph looks right — the sandbox this was developed in has no
wasm32 target installed at all, so dependency-graph resolution was as
far as local verification could go.

`query.rs`'s eventual rayon-based parallel iteration will need the
matching `#[cfg(not(target_arch = "wasm32"))]` split at the *code* level
too, once it's actually built — not needed yet, it's still a stub. FFI
work for `mid-ecs` is no longer saved for the end — see the FFI section
below for the real, incremental strategy and what's already built.

## FFI — built incrementally as we go, not saved for the end

The original plan deferred all FFI work until the ECS was otherwise
"done." Revisited: `mid-ecs`'s whole reason for existing separately from
being embedded directly in a monolithic engine is that its crates are
meant to be genuinely usable from *any* game engine or language, at
real performance — which means FFI correctness isn't a final coat of
paint, it's a core requirement that needs proving as each real piece
lands, the same way every other piece in this project has been proven
as it was built rather than asserted afterward.

**`World` lifecycle — real, tested, verified against actual compiled C.**
`crates/mid-ecs/src/ffi.rs`: `mid_ecs_world_new`/`free`/`spawn`/
`despawn`/`is_alive`/`entity_count`. Conventions copied directly from
`mid-net`'s real, already-proven `ffi.rs` — `MidEcsStatus` codes,
`ffi_guard`/`catch_unwind` on every function body, null-pointer checks
before every dereference, `unsafe extern "C" fn` + `# Safety` docs,
opaque heap handle for `World` (not `repr(C)` — nothing about it is
C-representable).

The one genuinely new piece: `Entity` can't cross the boundary as a
Rust value (its fields are deliberately private — only `World::spawn`
should ever produce one) and a two-field `repr(C)` struct would make
every caller's language agree on a layout for no real benefit. Instead,
`Entity::as_ffi`/`from_ffi` (thin wrappers over `mid_collections::
GenerationalIndex::as_ffi`/`from_ffi`, which do the real packing) hand
out one plain `u64` — directly grounded in `slotmap::KeyData::as_ffi`/
`from_ffi`'s real, shipped design (checked directly, not assumed),
including the property that matters most: reconstructing from a `u64`
that never actually came from a real `as_ffi()` call is still *safe* —
every `World` method re-validates the handle's generation against the
slot's current one regardless of where the value came from, so a bogus
handle just reads back as not alive. It can never alias a real, live
entity it wasn't issued for. Proven directly, not just documented:
`generational_index.rs`'s `from_ffi_on_a_bogus_value_is_safe_and_reads_as_not_alive`
and `ffi.rs`'s `bogus_packed_entity_is_safe_and_reads_as_not_alive`
both construct a genuinely bogus value and confirm exactly this.

**Verified the same way `mid-net`'s FFI was — real gcc, real C, real
memory, not just Rust-side `unsafe {}` blocks calling into themselves.**
`crates/mid-ecs/ffi-smoke-test/{mid_ecs.h, test.c}`: hand-written header
(matching `mid-net`'s own "hand-written, not cbindgen-generated, updated
by hand alongside `ffi.rs`" convention), 19 real checks. Compiled with
real gcc, linked against the real built `libmid_ecs.so` *and*
`libmid_ecs.a` separately, both run, both 19/19 — including the stale-
handle-after-slot-reuse case and the bogus-packed-`u64` case, proven
through actual C memory, not simulated. `.github/workflows/
mid-ecs-test.yml` now runs this on every CI trigger too, mirroring
`mid-net-test.yml`'s own FFI smoke test step exactly.

**Deliberately not covered yet in the `World`-lifecycle pass above, and
why it was genuinely harder, not just more of the same:** reading
component data (a `Position` column, say) from C. Every function in
that pass either passed a value by-value or went through an opaque
handle with no live pointer into mutable interior storage — nothing had
to reason about a pointer a *later* call could invalidate. Component
data lives in `Vec<T>`-backed columns (`SparseShell`'s `SparseSet`s,
`Archetypes`' `Table`s) that `insert`/`remove`/migration can reallocate
or move out from under a previously handed-out pointer. This is what
the "FFI span" idea (`docs/mid-collections.md`'s FFI wrapper section)
was actually for.

**Sparse Shell span access — real, tested, done for v1.**
`SparseShell::register_ffi<T>`/`raw_span`/`lookup_ffi_id`
(`crates/mid-ecs/src/component.rs`), thin-wrapped at `World::
register_ffi_component`/`component_raw_span`/`lookup_ffi_component_id`.
The real problem this had to solve, worked out rather than assumed:
producing a byte-erased `(ptr, stride, count)` view through a
`ComponentId` (an opaque `u32` at the FFI boundary) *without* the
caller knowing the concrete Rust type `T` — while `Box<dyn
ComponentColumn>`'s existing type-erasure mechanism only supports
downcasting when the caller already supplies `T` generically, which an
`extern "C"` function structurally can't do. Resolved with a type-erased
accessor function (`fn(&dyn Any) -> FfiSpan`), monomorphized once per
`T` at `register_ffi::<T>()` time (where `T` *is* known, generically)
and stored in a side table keyed by `ComponentId`, called later purely
non-generically — deliberately *not* baked into the base
`ComponentColumn` trait itself as a required method, since that would
force `IntoBytes + Immutable + KnownLayout` (from `zerocopy`) onto
every component type in the Sparse Shell, including plain Rust-only
types with no FFI intent (this crate's own `Position`/`Velocity` test
types, not `#[repr(C)]`, would have stopped compiling). Opting a type
in is explicit and per-type instead — real test coverage confirms both
halves: `register_ffi_before_any_insert_still_gives_a_valid_empty_span`
etc. exercise the opted-in path, while `Position`/`Velocity`'s own
existing tests keep passing completely untouched, proving the
restriction really is scoped to only what opts in.

A real bug caught by actually running the tests, not by review: the
first `raw_span` returned `None` for a type that was `register_ffi`'d
but had nothing inserted for it yet, since `columns` entries are only
created lazily on first `insert` — inconsistent with this same file's
own already-established convention (`SparseShell::iter<T>` already
treats "nothing inserted yet" as an *empty* result, not a *not-found*
one). Fixed to match the existing convention, not patched around it.

C-side `ComponentId` registration, scoped concretely: `register_ffi`
also records a plain string name, resolved later via `lookup_ffi_id`/
`lookup_ffi_component_id` — this is *not* C defining an entirely new,
Rust-unknown component layout (that would need a parallel byte-blob
storage mode this pass doesn't touch, a real, much larger undertaking
flagged rather than attempted); it's C obtaining the `ComponentId` for
a type Rust already opted in, by the name Rust gave it. `register_ffi`
itself is necessarily still a Rust-side, generic call — an `extern "C"`
function can't be generic over `T` — real, unavoidable one-time setup
glue, not an oversight.

**Archetype Core span access — real, tested, done for v1, and
genuinely harder than the Sparse Shell side above.** `Archetypes::
register_ffi<T>`/`raw_span`/`archetypes_with`/`lookup_ffi_id`
(`crates/mid-ecs/src/archetype.rs`), thin-wrapped at `World::
register_ffi_static_component`/`static_component_raw_span`/
`archetypes_with_static_component`/`lookup_ffi_static_component_id`.
Same type-erased-accessor mechanism as the Sparse Shell side (a
`fn(&dyn Any) -> FfiSpan`, monomorphized per `T` at registration time),
same reasoning for not baking it into the base `Column` trait — but one
real, unavoidable additional wrinkle: a component type here isn't one
stable thing to read. An entity's row lives in whichever archetype
currently matches its exact component set, so one type's data is
fragmented across every archetype containing it. `raw_span` is
necessarily per-`(ArchetypeId, ComponentId)`, not just per-`ComponentId`
the way the Sparse Shell's is; `archetypes_with` enumerates the
fragments.

A real, non-obvious distinction worked through and confirmed by
dedicated tests, not glossed over: unlike the Sparse Shell's own
"registered but nothing inserted yet" case (fixed to return an empty
span, not `None`, per the bug above), an archetype's signature simply
not including a given component is a *different*, *permanent* fact
about that specific archetype — `raw_span` correctly returns `None`
there (matching `Archetypes::has`'s own established `false`-not-panic
convention), while a real *empty-but-present* column (the component is
in the signature, every entity that had it has since migrated away)
correctly returns `Some` with `count == 0` — proven directly:
`raw_span_on_an_archetype_that_does_not_have_this_component_is_none`
and `raw_span_is_some_and_empty_after_every_entity_migrates_away` are
two separate tests because they're two genuinely separate cases, not
one case described two ways. Grounded in a real check of `ensure_column`
before trusting the "empty-but-present" case could even happen: columns
are only ever added to a table, never removed, once an archetype has
been created with a given signature — confirmed by reading that
function, not assumed.

A completely separate `ComponentId` name space from the Sparse Shell's
own `register_ffi`/`lookup_ffi_id`, matching `Archetypes`' own
already-established separate `ComponentId` numbering space from
`SparseShell`'s (see this doc's Sparse Shell section above) — the same
name string can resolve to a different `ComponentId` in each system,
proven directly by
`sparse_and_static_ffi_registrations_use_independent_name_spaces`.

**Not yet done, real next increment — flagged, not silently dropped:**

- **Entity correlation**, for *both* storage systems now. Neither
  `raw_span` can tell a caller *which* entity each element belongs to
  — `Entity` itself isn't `#[repr(C)]`/zerocopy-compatible today (its
  fields are deliberately private, see `Entity::as_ffi`/`from_ffi`
  above), so a zero-copy span over either system's dense entity array
  isn't possible without a real decision about how `Entity` should
  cross this specific boundary. This is now the single real blocker
  standing between "callable" and "actually usable" for the whole FFI-
  span mechanism.
- Actual `extern "C"` functions in `ffi.rs` exposing any of this, plus
  a real C smoke test and CI wiring, matching the rigor the `World`-
  lifecycle pass above already has. Nothing above has been proven
  against real compiled C yet — only real Rust-side tests so far, 147
  of them across `mid-collections` (49, `--features ffi`) + `mid-ecs`
  (98) combined as of this pass, none of them C.

## The Hybrid ECS Architecture: Static Core, Dynamic Shell

Mid Engine completely avoids the traditional Object-Oriented memory traps by splitting entity data into two highly optimized zones:

### 1. The Archetype Core (Heavy Logic)
* Components that remain static throughout an entity's lifecycle—like `Transform`, `Velocity`, or `PhysicsBody`—are packed into rigid Archetype tables.
* This guarantees perfect CPU cache locality.
* It allows our `mid-math` wide SIMD vectors to blast through positional updates without jumping around in memory, forming our high-performance "Inner Loops".

### 2. The Sparse Shell (Volatile Logic)
* Status effects or states that flicker on and off constantly—like `IsPoisoned`, `Disabled`, or `Hidden`—are managed using Sparse Sets or highly efficient Bitsets.
* The Sparse Shell is real now — `World::insert`/`get`/`get_mut`/`remove`/`has` (`crates/mid-ecs/src/component.rs`), any `T: 'static` attachable to any entity, backed by `mid_collections::SparseSet` per component type, keyed by a dense `ComponentId` rather than a `TypeId` hash (design grounded in Bevy ECS's own real `ComponentId` source — see `component.rs`'s doc comment). `despawn` correctly cleans up every attached component before freeing the entity's slot, closing the exact stale-handle gap `SparseSet` can't close on its own.
* **The "Stutter" Fix:** If you poison 1,000 goblins, the engine just flips a bitmask or adds a tiny entry in a sparse set. 
* Result: Zero memory is physically moved between archetype tables. The engine stays fast, and we avoid the memory-copying lag spikes that plague pure archetype architectures during massive state changes.
* For lightning-fast entity querying, the engine utilizes a `BitVec` layout (1 boolean into 1 bit), allowing us to filter hundreds of thousands of entities in microseconds using simple bitwise AND operations.

## Large World Coordinates: GlobalTransform

**Status: implemented.** `GlobalTransform` and `GlobalTransformLWC`
are real, tested types in `crates/mid-ecs/src/transform.rs` — both
usable with `World::insert_static`/`query_static` today. Not yet
built: `LocalTransform`/hierarchy composition, and `mid-camera`'s own
per-frame `to_view_relative` driver loop (see "Not yet decided" below
— unchanged, still real, still not started).

**The decision: two component types, `f32` default, `f64` opt-in —
not one type, and not `f64` everywhere.**

- `GlobalTransform` — `f32`, backed by `mid-math`'s existing `Affine3`.
  The default. Lives in the Archetype Core, same as any other static,
  every-frame-touched component (see "The Archetype Core" above).
- `GlobalTransformLWC` — `f64`, backed by `mid-math`'s `DAffine3`. Opt-in,
  for entities that actually travel far enough from world origin to
  need it (open-world terrain, distant structures, anything a camera
  might travel tens of kilometers to reach). A distinct archetype
  family from `GlobalTransform`, not the same component with a
  runtime-branching representation — a tagged union inside one
  Archetype Core column would break the homogeneous-`Vec<T>`-column
  assumption `component.rs`'s FFI-span mechanism (and every other
  system that reads a column) already depends on, and would cost
  exactly the cache/branch overhead this split exists to avoid.

**Why `f32` default, not `f64` default:** `DAffine3` is 96 bytes;
`Affine3` is 64 (16-byte aligned, SSE2-backed on x86/x86_64 — corrected
here from an earlier "48" figure in this doc that didn't match
`Affine3`'s own doc comment; checked directly against
`crates/mid-math/src/f32/affine3.rs` while implementing
`GlobalTransform`, not left as a quiet inconsistency). `f64` is still
1.5x the size, not 2x — the conclusion below is unchanged.
`GlobalTransform` is about the hottest, most-iterated
component this engine will ever have — read every frame for every
visible entity, exactly the access pattern the Archetype Core exists
to make cache-friendly. Most entities in most scenes (UI-anchored
objects, particle effects, interior/local gameplay) never travel far
enough from origin to need `f64` at all. Doubling the stride of the
hottest column engine-wide, to solve a problem only some entities
have, is in direct tension with this project's own performance
mandate — so the cost is opt-in, paid only by the entities that
actually need it.

**The pipeline this feeds into**, once built: `GlobalTransformLWC`
holds true world-space state in `f64`. Once per frame, for the
active camera, every visible `GlobalTransformLWC` gets passed through
`DAffine3::to_view_relative(camera_position)` (see `docs/mid-math.md`)
— composing the camera-relative shift and the `f64`→`f32` truncation
in one step, safe regardless of how far the entity is from world
origin, because only the shifted (small) translation gets truncated,
never the raw world-magnitude one. The result is a plain `f32`
`Affine3`, indistinguishable downstream from a `GlobalTransform`
entity's own data — rendering, culling, and anything else consuming
"the" transform for a draw call never needs to know or care which
storage precision an entity actually used. That narrow point (right
before GPU upload) is the only place the two component types'
consumers actually have to know both exist.

**Not yet decided:** how `LocalTransform`/hierarchy (parent-relative,
always small-magnitude, always `f32` regardless of world size)
composes into either `GlobalTransform` variant — that's the actual
"Integrating f64 global transform components into archetype storage
tables" implementation work, not yet started. `mid-camera` (planned,
not started — this engine's Cinemachine equivalent, sitting on top of
both this system and `mid-math`'s existing `camera/` module) is what
will eventually own "which entity is the active camera" and drive the
per-frame `to_view_relative` call above.

## The Ubel Stratum Bridge (The OOP Illusion)

**Design vision, not confirmed integration.** `docs/mid-collections.md`
is explicit about this: Ubel is *"a separate project... deliberately
not folded in here."* `mid-log.md`/`mid-net.md` both hedge the same
way (*"prepared for"*, *"a plausible future consumer"*). This section
previously read as settled architecture with no such hedge — a real
contradiction with those three docs, not just a tone mismatch, since
it described the bridge mechanism as if it already existed. Fixed to
match the same framing used everywhere else in this workspace: the
idea below is real and worth recording, but Ubel's actual integration
with `mid-ecs` specifically is not decided, designed in detail, or
started.

The vision, as discussed: gameplay code never touches Archetypes or
Bitsets directly.
* **HIGH Tier:** a gameplay programmer would interact with what reads
  like standard OOP classes (an `Actor`/`Entity` object).
* **LOW Tier:** if built, an Ubel compiler would lower that high-level
  code (`player.health -= 10`) into raw, memory-safe array accesses
  against `mid-ecs`'s own storage — Sparse Shell or Archetype Core,
  whichever the accessed component actually lives in.

Real open question this doc doesn't answer, and shouldn't pretend to:
whether that lowering targets this crate's own Rust API directly, or
goes through the FFI-span mechanism `component.rs`/`archetype.rs`
already expose — the latter would make Ubel just another FFI
consumer, no different in kind from any other non-Rust caller this
crate already supports, rather than a special-cased integration.

## Network Sync (Multiplayer-First)

The `sync` module marks components for `mid-net` replication.
This is the Multiplayer-First mandate in practice: networking is baked into the ECS from day one, not bolted on later.
* Components can be explicitly flagged for synchronization (e.g., `@net Transform`). 
* The engine automatically handles serialization via DixScript (`.mdix`) to sync state across the wire.
* Because data is stored contiguously in the Archetype Core, the network system can simply request a memory block and run a single SIMD pass over that memory to detect deltas, compress with MBFA-lite, encrypt, and ship the UDP packet.

## Modules

### `diag_query2_unchecked.rs`

Temporary diagnostic module, delete once the investigation it exists
for concludes (see its own NOTICE header). Real CI (rustc 1.98.0)
shows `query2_static_two_components` running 2.96x-4.57x slower than
`bevy_ecs` on an equivalent workload, in every run since the
Iter1/Iter2 rewrite. This sandbox's own rustc 1.91.1 has never
reproduced it — the same code measures within noise of `bevy_ecs`
here every time. An `#[inline(always)]`/`#[inline(never)]` three-way
diagnostic (`diag_inline.rs`) already ruled out inlining as the cause,
run for real on rustc 1.98.0.

Real source comparison against `Mid-D-Man/bevy` (`crates/bevy_ecs/src/
query/iter.rs` and `fetch.rs`, read directly) found a real structural
difference. Bevy's `QueryIterationCursor` is one generic struct,
parameterized over `D: QueryData`. A two-component query has no
hand-written "two-component" implementation at all — `D = (A, B)`,
`D::Fetch<'w> = (A::Fetch<'w>, B::Fetch<'w>)`, and `D::fetch` is
generated by a macro (`impl_tuple_query_data`, expanded once per arity
via `all_tuples!`) as `Some((A::fetch(...)?, B::fetch(...)?))` — the
same single-component fetch code, reused twice, composed through the
tuple impl. `&T`'s own `fetch` (`ReadFetch<T>`) reads through
`table.get_unchecked(table_row.index())`, `#[inline(always)]`, no
bounds check. mid-ecs's `Iter1` and `Iter2` (`archetype.rs`) are
instead two separately hand-written structs — meant to be the same
shape, but not literally the same code the compiler sees, and this
kind of divergence between two independently-authored near-duplicates
is exactly the sort of thing whose optimization treatment can differ
across a compiler version bump.

Four new variants, each isolating one candidate cause, matching what
was actually different between mid-ecs's approach and bevy's:

* `Iter1Unchecked`/`Iter2Unchecked` — the current `Iter1`/`Iter2`
  logic unchanged, except entity/column indexing goes through
  `get_unchecked` instead of safe indexing, directly matching bevy's
  own technique. Sound: `len` is already set to the min of every
  slice's length on each archetype advance, so `row < len` already
  proves every access in bounds — this was already true of the safe
  version's own reasoning (see `Iter2`'s doc comment in
  `archetype.rs`), `get_unchecked` just stops asking the compiler to
  re-derive it. This is the first `unsafe` in `mid-ecs`'s own query
  iteration path — agreed as a deliberate, scoped exception for this
  crate specifically, not a change to the workspace's general
  zero-unsafe default.
* `Iter2TwoTupleItem` — reads both `a_col` and `b_col` per item like
  the real `Iter2`, but combines them into one derived value before
  returning, so `Item` is a 2-tuple instead of a 3-tuple. Isolates the
  tuple/`Item`-shape question from the two-slice-fields question.
* `Iter2UnusedBCol` — same struct shape as the real `Iter2` (both
  `a_col` and `b_col` fields present, both kept current on every
  archetype advance), but the hot loop only ever reads `entities`/
  `a_col` — `b_col` is tracked for `len` only, never indexed per item.
  Isolates whether merely carrying the extra field matters,
  independent of whether it's read.

Each variant has a correctness test in `query.rs` cross-checking its
output against the real, safe `query_static`/`query2_static` — an
`unsafe` variant that's merely fast and wrong isn't useful data. All
pass, 176/176 total `mid-ecs` tests, sandbox rustc 1.91.1.

Not yet run on real CI. That's the actual next step — none of this
sandbox's numbers can confirm or rule out anything here, the same
asymmetry that opened this investigation in the first place. Exposed
via `World::query_static_diag_unchecked`/`query2_static_diag_unchecked`/
`query2_static_diag_two_tuple_item`/`query2_static_diag_unused_b_col`,
`#[doc(hidden)] pub` for the same reason as `diag_inline.rs`'s own
methods — `benches/archetype_core.rs` needs real public API to reach
them from outside the crate. Benching them is the next real
deliverable, not built yet.

Bench group `query2_static_diag_unchecked` added to
`archetype_core.rs`, covering all four variants. Real, same-sandbox-
session numbers, N=10,000, next to a same-session run of the real safe
`Iter1`/`Iter2` for direct comparison:

* Safe (current, real): `query_static` 7.291µs, `query2_static`
  10.265µs — ratio 1.41x (matches this sandbox's usual ~1.28-1.35x
  baseline within normal run-to-run variance).
* `Iter1Unchecked`: 7.222µs — indistinguishable from safe `Iter1`,
  consistent with `Iter1` already being at parity with `bevy_ecs`
  everywhere this has been measured.
* `Iter2Unchecked`: 29.044µs — **~2.83x slower than safe `Iter2`**,
  on this sandbox. `Iter2TwoTupleItem` (28.700µs) and `Iter2UnusedBCol`
  (28.706µs) land in the same place, all three roughly 4x the unchecked
  1-column number, at every N from 100 to 100,000 (ratio actually
  creeps up slightly with N: 3.51x at N=100 to 4.10x at N=100,000 — a
  per-item cost difference, not a one-time per-archetype-advance cost).

This is a real, surprising sandbox result, not a reason to abandon
these variants — `get_unchecked` should never make correct code
slower than the safe equivalent it replaces, on any toolchain that's
reasoning about it correctly, so this is itself informative. Worth
noting directly: 29.044µs is almost exactly the same number
`Iter2::next`'s own doc comment records for the earlier, reverted
`#[inline(always)]` attempt (~29µs) — on this same sandbox, two
completely different changes (forcing inlining, and switching to
`get_unchecked`) both land at close to the same, worse number. This
sandbox (rustc 1.91.1) has now shown *two* real cases where a change
that's well-motivated by bevy's own real source makes `Iter2`
specifically worse here — while the actual regression this whole
investigation exists to explain only shows up on rustc 1.98.0, never
here. That asymmetry cuts both ways: a sandbox result saying "worse"
carries exactly as little weight as one saying "better" would have.
Real CI is the only real answer here, same as every other toolchain-
sensitive question in this investigation so far.

Correctness is independent of this and already solid: all four
variants pass their dedicated tests in `query.rs`, cross-checked
against the real, safe `query_static`/`query2_static` output, 176/176
`mid-ecs` tests total.

### Real CI results (Archetype Core build #7, rustc 1.98.1)

These four variants ran on real CI for the first time. The result
overturns the sandbox reading above, which is exactly why the sandbox
numbers were never trusted on their own.

At N=10,000: safe `query2_static` 37.446µs against safe `query_static`
9.4228µs, ratio 3.97x, matching every prior real CI run of this
comparison. `Iter2Unchecked` (`query2_static_unchecked_2col`) measured
37.758µs and `Iter2UnusedBCol` (`unused_b_col`) measured 37.734µs,
both indistinguishable from the safe baseline. `get_unchecked` alone
does nothing on real CI either, same conclusion the sandbox reached,
for once matching it.

`Iter2TwoTupleItem` (`two_tuple_item`) measured 9.4864µs, matching
`query_static`'s single-column number almost exactly, at every N from
100 to 100,000. This is the opposite of what the sandbox showed
(28.700µs there, indistinguishable from the other two variants). One
variant returning a single owned value instead of two references
closes nearly the entire gap on the toolchain where the gap is real,
despite doing more work per item: its `combine` function reads all six
fields across both components (`Position`'s three plus `Velocity`'s
three) and constructs a new `Position` by value, against the baseline
loop's two field reads (`pos.x`, `vel.dx`). More arithmetic, less time.

This rules a specific thing in, not just several things out. Returning
`(Entity, &A, &B)` is not inherently slow. `Mid-D-Man/bevy`'s own
`D::Item` for a two-component query is exactly this shape (`(A::Item,
B::Item)` off the tuple macro, `(&Position, &Velocity)` for a plain
read query), and bevy pays no penalty for it (`ecs-vs-bevy-ecs` build
#10, same day: `query_static_single_component` 1.0x, `raw_slice_ceiling`
1.0x). Also checked and ruled out this pass: `mid-ecs`'s real `Iter2`
(`archetype.rs`) downcasts its `Box<dyn Any>` columns to `&[A]`/`&[B]`
once per archetype advance, same as every diagnostic variant, never
per item, so the downcast itself isn't a per-item cost candidate.

What's actually different, still untested until this pass: bevy's
tuple `QueryData::fetch` composes as `Some((A::fetch(...)?,
B::fetch(...)?))`, two independent single-component fetch calls glued
by `?`. `Iter2`, `Iter2Unchecked`, `Iter2TwoTupleItem`, and
`Iter2UnusedBCol` all instead read both columns inline in one
hand-written tuple literal. Whether returning two references together
is the cost, or whether it only becomes the cost when they're
constructed by one block instead of composed from two calls, hasn't
been separated yet.

### `Iter2Composed` (built this pass, not yet run on real CI)

Isolates exactly that. Same struct fields, same archetype-advance and
per-archetype downcast logic as `Iter2Unchecked`, same `Item =
(Entity, &A, &B)`. The only real change: two small functions,
`fetch_one::<A>`/`fetch_one::<B>`, each doing one
`get_unchecked`-and-wrap-in-`Some`, called from `next()` as `Some((...,
fetch_one(a_col, row)?, fetch_one(b_col, row)?))` instead of building
the tuple directly from `&self.a_col[row]`/`&self.b_col[row]` inline.
Mirrors bevy's own composition shape as closely as a hand-written
non-generic version reasonably can.

Two dedicated tests in `query.rs`
(`diag_query2_static_composed_matches_the_real_query2_static`,
`diag_query2_static_composed_empty_when_one_side_was_never_registered`),
cross-checked against real `query2_static` the same way every other
variant's tests are. 178/178 `mid-ecs` tests total now. Exposed via
`World::query2_static_diag_composed`, same `#[doc(hidden)] pub`
pattern as the other four. Bench arm `composed` added to the
`query2_static_diag_unchecked` group in `archetype_core.rs`.

Not run on real CI yet. Given `Iter2TwoTupleItem`'s result, the most
useful outcome to watch for is whether `Iter2Composed` also closes
most of the gap despite still returning two references, which would
point at the inline-tuple-literal construction itself as the real
cost rather than the reference-pair return value; or whether it stays
near the current 4x, which would point back at something specific to
carrying two live references out of `next()` together, and make
`Iter2TwoTupleItem`'s result closer to a lucky side effect of also
narrowing the return type rather than a real fix for a query API that
actually needs to hand back both components separately.

### Real CI result for `Iter2Composed` (Archetype Core build #8, rustc 1.98.1)

Second answer, not the first. `Iter2Composed`'s own bench arm measured
42.350µs at N=10,000, indistinguishable from the safe baseline
(42.505µs), `Iter2Unchecked` (42.271µs), and `Iter2UnusedBCol`
(42.335µs). Composing the fetch as two separate calls glued by `?`,
matching bevy's own macro shape exactly, changed nothing. That rules
out "how the fetch is composed" as the cost.

What actually separates the fast variant from the three slow ones,
looking at all four together:

| Variant | Struct holds both a_col and b_col? | Item | Real CI (N=10,000) |
|---|---|---|---|
| `Iter1Unchecked` | No, one column only | `&T` | 10.6µs, fast |
| `Iter2UnusedBCol` | Yes, b_col never read | `&A` | 42.3µs, slow |
| `Iter2TwoTupleItem` | Yes, b_col read via `combine` | `A` (owned) | 10.7µs, fast |
| `Iter2` / `Unchecked` / `Composed` | Yes | `(&A, &B)` | 42.3-42.5µs, slow |

`Iter2UnusedBCol` is the tell. Its `Item` is `(Entity, &A)`, the exact
same shape as the fast `Iter1Unchecked`, and it never reads `b_col` in
the hot loop. The only thing it does differently from `Iter1Unchecked`
is carry a second, unread, differently-typed slice field
(`b_col: &'a [B]`) that gets updated on every archetype advance. That
alone reproduces the full regression. Meanwhile `Iter2TwoTupleItem`
reads both columns every item and stays fast, because what crosses
back out of `next()` is owned, not a reference.

So it isn't "returning two references" (bevy does that and pays
nothing) and it isn't "how many columns get read" (`TwoTupleItem`
reads more than `UnusedBCol` and is faster). It's specifically: once
the iterator's own state holds two independently-typed live slice
references, returning any reference from `next()` pays a real cost
that returning an owned value does not, regardless of how much work
built that owned value or how the fetch that produced it was
structured.

### `Iter2RawPtr` (built this pass, not yet run on real CI)

Direct test of this. Same struct and archetype-advance logic as
`Iter2Unchecked`, but `a_col`/`b_col` are `*const A`/`*const B`
instead of `&'a [A]`/`&'a [B]`, with a `PhantomData<(&'a [A], &'a
[B])>` marker carrying the same lifetime obligation the slice fields
used to enforce directly. `next()` builds the returned item with
`&*self.a_col.add(row)`/`&*self.b_col.add(row)` instead of indexing a
stored slice. This is a hand-rolled version of the same idea as
`ThinSlicePtr`, not a dependency on it; `Mid-D-Man/bevy` is source-read
and porting reference only, never a direct dependency, so nothing here
imports `bevy_ptr`.

Two dedicated tests
(`diag_query2_static_raw_ptr_matches_the_real_query2_static`,
`diag_query2_static_raw_ptr_empty_when_one_side_was_never_registered`),
same cross-check pattern as every other variant. 180/180 `mid-ecs`
tests total now. Exposed via `World::query2_static_diag_raw_ptr`.
Bench arm `raw_ptr` added to the same group.

If this closes the gap on real CI, the natural next step is exactly
what was asked for: a small, focused, no-std-style crate mirroring
`ThinSlicePtr`'s actual shape (`NonNull<T>` plus debug-only length plus
a `PhantomData<&'a [T]>` marker, `get_unchecked` returning `&'a T`),
built as mid-engine's own, not a dependency on `bevy_ptr` itself, the
same relationship `Mid-D-Man/bevy` already has to everything else in
this project. Worth building once real CI says raw-pointer storage is
actually the fix and not one more thing that looks right for reasons
that don't hold up outside this sandbox, the same caution every other
diagnostic in this file has been given.

### Real CI result for `Iter2RawPtr` (Archetype Core build #9, rustc 1.98.1)

It didn't. 37.452µs at N=10,000, indistinguishable from every other
reference-returning variant (safe baseline 37.672µs, `Unchecked`
37.458µs, `UnusedBCol` 37.402µs, `Composed` 37.445µs). Storage
representation, slice vs raw pointer, changes nothing.

This makes sense in hindsight rather than being a dead end. `RawPtr`
and `Unchecked` both still return `Option<(Entity, &'a A, &'a B)>` from
`next()` — identical signatures. Converting a raw pointer back to
`&'a T` inside the function produces the exact same type crossing the
exact same boundary; whatever LLVM does with that signature doesn't
know or care whether the reference was built by indexing a slice or
adding to a pointer. The lever that was actually being pulled every
time (`Unchecked`'s safety, `Composed`'s call structure, `RawPtr`'s
storage) was never the one that mattered. The one variable that has
correlated with the result every single time is simpler: does `Item`
contain a reference, or not.

### `Iter2OwnedDirect` (built this pass, not yet run on real CI): isolating the one remaining confound

Before settling on "owned return is what matters," one thing needed
separating out. `Iter2TwoTupleItem`, the only fast variant found so
far, gets its owned value through `(self.combine)(a, b)` — a stored
`fn(&A, &B) -> A` pointer field, called indirectly. An indirect call
through a function pointer is its own kind of optimization barrier;
LLVM generally can't inline through one. Every slow variant
(`Unchecked`, `UnusedBCol`, `Composed`, `RawPtr`) calls nothing
indirectly. So "returns an owned value" and "goes through an opaque,
uninlinable call" have been riding together in the one data point that
worked, and it hadn't been checked which of the two actually explains
the speed.

`Iter2OwnedDirect` separates them: same owned-return shape as
`TwoTupleItem` (`Item = (Entity, A)`), same struct fields as
`Unchecked` (`a_col`/`b_col` as real `&'a [_]` slices, no pointer
games), but the combine step goes through `DiagCombine`, a small
trait with one method, called as `A::diag_combine(a, b)` rather than
through a stored function pointer. A trait method call like this is
fully known at compile time and monomorphized per concrete type, so
the compiler is free to inline it — there's no runtime indirection at
all, unlike a `fn` pointer field whose target isn't fixed until the
value is constructed.

`DiagCombine` is `pub`, not `pub(crate)` (re-exported hidden from
`lib.rs` as `mid_ecs::DiagCombine`, module itself stays private): the
bench is a separate crate linking against `mid-ecs` and needs to
implement it for its own `Position`/`Velocity` to call
`World::query2_static_diag_owned_direct` at all. Two dedicated tests
(`diag_query2_static_owned_direct_matches_manual_combine`,
`diag_query2_static_owned_direct_empty_when_one_side_was_never_registered`),
same cross-check discipline as every other variant. 182/182 `mid-ecs`
tests total now. Bench arm `owned_direct` added to the same group.

Two possible outcomes once this runs on real CI. Lands near 9.4µs,
matching `TwoTupleItem` and the 1-column baseline: confirms owned
return is genuinely what matters, independent of the function-pointer
question, and `TwoTupleItem`'s result wasn't a fluke of that
confound. Stays near 37µs, matching every reference-returning variant:
means the opaque call boundary itself was doing the real work in
`TwoTupleItem`, a stranger and more specific finding about this
toolchain's handling of that particular loop shape, not about owned
values at all — and would mean the `ThinSlice`/raw-pointer design
sketch proposed above is very unlikely to be the fix, whatever else it
might still be worth for other reasons.

### Real CI result for `Iter2OwnedDirect` (Archetype Core builds #10 and #11, rustc 1.98.1)

Second outcome. Build #10's absolute numbers were a false alarm (a
uniform ~31% drop across every group in that run, including
benchmarks nothing in this investigation touches — CI noise, caught
and re-run by the user rather than trusted). Build #11, same commit,
is back in the normal range and settles it: `owned_direct` measured
42.267µs at N=10,000, indistinguishable from every reference-returning
variant (safe baseline 42.793µs, `Unchecked` 42.276µs, `UnusedBCol`
42.250µs, `Composed` 42.260µs, `RawPtr` 42.279µs). `TwoTupleItem`
stayed fast at 10.712µs, matching the 1-column baseline (10.593µs).
Build #10's noisy absolute values still preserved this same relative
split (owned_direct 26.046µs next to a 26.0-26.1µs cluster, two_tuple_item
alone at 11.614µs), so it's two real CI runs agreeing on the ordering,
not one.

This rules out "owned return" outright. `Iter2OwnedDirect` returns
owned, same as `TwoTupleItem`, and it's exactly as slow as every
reference-returning variant. The one thing left standing, out of seven
variants tested, is that `TwoTupleItem` is the only one whose combine
step goes through a genuine indirect call (a stored `fn` pointer) that
the compiler can't inline through. Every other variant, whatever else
differs between them, fully inlines.

### Bevy's own benchmark convention (found this pass, checked directly against Mid-D-Man/bevy)

Went back to `Mid-D-Man/bevy` with a narrower question than before:
not what the fetch code does, but how bevy structures the benchmark
loop itself. Checked `benches/benches/bevy_ecs/iteration/` directly —
`iter_simple.rs`, `iter_frag.rs`, `iter_simple_foreach.rs`,
`iter_simple_contiguous.rs`, no exceptions found across the ones
checked. Every one wraps its loop in a `#[inline(never)] fn
run(&mut self)` on a small `Benchmark` struct, and `mod.rs` calls it as
`b.iter(move || bench.run())` — the criterion closure is a single call
to an opaque, never-inlined function that does the real work inside
itself. None of this project's benches do that anywhere; every loop,
in every variant above and in `benches/ecs-vs-bevy-ecs/benches/vs_bevy_ecs.rs`,
sits directly inside `b.iter(...)`, fully visible to the optimizer
alongside criterion's own harness code.

Checked whether this alone explains bevy's own speed before assuming
it explains mid-ecs's slowness: `vs_bevy_ecs.rs`'s `dense_query_iteration`
benches both engines the same way, loop inline in `b.iter()`, no
wrapper, for both sides. `bevy_ecs` still lands at parity there (its
own iteration doesn't need the wrapper to be fast). So this can't be
the whole explanation. But that doesn't rule out `mid-ecs`'s specific
`Iter2` needing it, independent of whatever bevy's implementation is
doing.

### Real, unmodified `query2_static`/`query_static`, wrapped to match bevy's convention (built this pass, not yet run on real CI)

Two new bench arms in the same group, `real_query1_inline_never_wrapper`
and `real_query2_inline_never_wrapper`. Neither touches archetype.rs
or adds a new Iter2 variant — both call the real, current, completely
unmodified `World::query_static`/`World::query2_static`, from inside a
`#[inline(never)] fn run(&mut self)` on a small wrapper struct, called
as `b.iter(|| black_box(bench.run()))`, matching bevy's own structure
exactly. Compiles clean, smoke-tested locally (`cargo bench -- --test`,
all arms report `Success` — sandbox numbers themselves stay
non-authoritative as always, this only confirms nothing panics).

This is the most direct test yet of what's actually been driving every
result in this file so far, and it uses zero diagnostic code: if
wrapping the real `query2_static` in a never-inlined function alone
closes most of the gap, a meaningful part of this entire investigation
has been chasing a benchmark-harness artifact rather than a real
runtime difference in `Iter2` itself — the fix would be to the bench,
not the iterator. If it stays near 37-43µs even wrapped, that rules
out harness structure too, and leaves indirect-call-specifically (not
owned return, not harness shape) as the one remaining explanation with
any real evidence behind it, which would need a real disassembly
comparison on rustc 1.98.1 to actually resolve.

### Real CI result for the `inline_never` wrapper test (Archetype Core builds #12 and #13, rustc 1.98.1)

Not a clean answer either way. The user ran this twice specifically to
check it wasn't a fluke, and both runs agree with each other, but not
in a way that confirms the wrapper hypothesis. What actually moved:

`query2_static_two_components` (real `Iter2`, no wrapper): 37.7µs
historical -> ~27.5-28.5µs both builds. `real_query2_inline_never_wrapper`
(same real `Iter2`, wrapped): ~27.5-28.9µs, indistinguishable from the
unwrapped version right next to it. So the wrapper itself changed
nothing — wrapped and unwrapped moved together.

But `query_static_single_component` (real `Iter1`, its own separate
top-level group, unrelated to this pass's diagnostic work) went the
other direction: ~9.4-11.6µs historical -> ~24.5-26.2µs both builds,
over *twice* as slow. And `raw_slice_ceiling` — zero mid-ecs code at
all, two plain `Vec`s — dropped from ~9.3-10.5µs to ~6.0-6.5µs, faster
in the opposite direction from `query_static_single_component`. Three
benchmarks that share no code moved three different amounts in two
different directions in the same run, twice.

That's not "the regression closed." The `query2_static_two_components`
vs `query_static_single_component` ratio only looks good (1.1-1.25x
this run, versus the usual ~4x) because the denominator got worse by
coincidence, not because the numerator got better for a real reason.
The number that actually matters was checked the same day and didn't
move: `ecs-vs-bevy-ecs` build #14, `dense_query_iteration`, still
3.99x (37.440µs vs bevy's fresh, independently-compiled 9.3733µs).
That comparison doesn't share this file's internal noise sources at
all, and it's unchanged. Treating builds #12/#13 as resolution would
mean trusting the one measurement that's easiest to fool and ignoring
the one that isn't.

Real, useful signal from this regardless: `archetype_core.rs`'s
internal "regression guard" ratio can swing this wide from
measurement conditions alone, with zero code change on either side of
the ratio. That's a concrete argument for the granular rework below —
a comparison with more independent, narrowly-scoped operations makes
a coincidence like this easier to catch (three unrelated numbers
moving inconsistently is a clearer tell than two numbers producing a
falsely-reassuring ratio).

### `benches/ecs-vs-bevy-ecs/benches/vs_bevy_ecs.rs`: single-op and multi-op rework

The four original groups (`spawn_n_entities_two_components`,
`query_static_single_component`, `dense_query_iteration`,
`raw_slice_ceiling`, `structural_churn_insert_remove` — five, not
four; that miscount predates this pass, fixed in the file's own doc
comment while here) were broad workload buckets. Six more added this
pass break specific operations out individually, matching
`crates/mid-math/benches/vs_glam.rs`'s own granularity: one
`bench_function` pair per concrete operation, not per broad workload.

`spawn_single_component`, `insert_single_component`,
`remove_single_component`: single-component versions of operations
the existing groups only measure bundled with a second component
(`spawn_n_entities_two_components`) or not at all (bare insert/remove
with no spawn or churn mixed in). `get_component_random_access`: a
genuinely different code path from every iteration group above, N
separate entity-by-id lookups rather than one contiguous archetype
scan. `insert_bundle_on_existing_entity`, `remove_bundle_two_components`:
multi-component structural moves on an entity that already carries
data, distinct from `spawn_n_entities_two_components`'s spawn-then-
immediately-bundle case and `structural_churn_insert_remove`'s
single-component churn.

Every new loop is wrapped in a `#[inline(never)]` free function,
matching bevy_ecs's own benchmark convention (confirmed by direct
source read this pass, see the `diag_query2_unchecked.rs` module
section above) — adopted unconditionally, not because builds #12/#13
confirmed it matters for mid-ecs specifically (they didn't confirm
anything either way), but because using bevy's own practice can only
make the comparison fairer, never less fair.

Verification: `ecs-vs-bevy-ecs` itself still can't be checked in this
sandbox at all (`bevy_ecs`'s `rust-version = "1.95.0"` blocks even
reaching this crate's own code on the sandbox's rustc 1.91, same wall
as always). The mid-ecs half of all six new functions was verified for
real instead: extracted into a standalone throwaway binary crate
depending on real `mid-ecs` directly (no `bevy_ecs` involved, so it
actually compiles here), with assertions checking each operation's
actual effect (entity count after spawn, `has_static` after
insert/remove, the summed value after random-access get, bundle
membership after insert/remove onto a non-empty entity) — all passed,
then deleted. The bevy_ecs half is grounded the same way the original
four groups already were: read directly against real source
(`World::spawn_empty`, `World::get`, `World::get_mut`,
`EntityWorldMut::insert`/`remove`, all confirmed in
`Mid-D-Man/bevy`'s `world/mod.rs` and `world/entity_access/world_mut.rs`),
not locally compiled. Confirm it actually builds on the next real CI
run before trusting bevy_ecs's numbers specifically, same standing
caveat the file's own header has always carried.

`.github/workflows/bench-vs-bevy-ecs.yml`'s own summary text hardcoded
"Three groups" — already wrong before this pass (there were five), now
updated to describe all eleven. `scripts/bench_vs_bevy_ecs.py` needed
no changes at all: it discovers groups by name from criterion's own
output and builds one table per group automatically, so the six new
groups just show up.

Checked `Mid-D-Man/bevy`'s actual `crates/bevy_ptr/src/lib.rs` directly
on the lead the user gave: bevy_ecs doesn't just happen to avoid this,
it has a dedicated crate (`bevy_ptr`, `#![no_std]`, one dependency,
`bevy_utils`) built specifically so its storage layer never holds
plain `&[T]` slice references across a fetch boundary. The relevant
type is `ThinSlicePtr<'a, T>`: a `NonNull<T>` plus a debug-only `len`
plus `PhantomData<&'a [T]>` for the borrow checker, with
`get_unchecked(&self, index) -> &'a T` doing the same
pointer-add-and-deref `Iter2RawPtr` does below. Confirmed this is not
a coincidence: `query/fetch.rs` imports it directly
(`use bevy_ptr::{ThinSlicePtr, UnsafeCellDeref};`), and `ReadFetch<'w,
T>`, the real `WorldQuery::Fetch` type for a plain `&T` query (its own
doc comment says so), stores its table column as exactly
`Option<ThinSlicePtr<'w, UnsafeCell<T>>>`, never a `&'w [T]`. A
2-component query composes two of these side by side, same shape
`Iter2UnusedBCol`/`Iter2Composed` tested, except bevy's two fields are
`ThinSlicePtr` and mid-ecs's are `&'a [_]`.


### Builds #11 and #13 (Archetype Core, rustc 1.98.1): the regression-guard ratio's own instability

Two real CI runs pasted back this pass, not adjacent — build #11 (03:36)
showed the regression-guard ratio at a "bad" 3.14-4.08x (flagged red by
the job summary itself); build #13 (12:51, same day) showed it at a
"good" 1.09-1.24x (flagged green, "consistent with the ~1.28-1.35x
currently-observed baseline"). Read at face value, that looks like the
gap closed on its own between the two runs. It didn't — the ratio's own
denominator moved, not its numerator.

**What actually moved, `build #13 / build #11`, at N=100,000:**

| Group | Build #11 | Build #13 | Ratio (13/11) |
|---|---|---|---|
| `query_static_single_component` (Iter1, safe, real) | 106.09µs | 262.06µs | **1.90x SLOWER** |
| `query_static_unchecked_1col` (Iter1Unchecked) | 106.01µs | 65.413µs | 0.617x (38% faster) |
| `two_tuple_item` (Iter2TwoTupleItem) | 106.39µs | 89.366µs | 0.840x (16% faster) |
| `query2_static_two_components` (Iter2, safe, real) | 433.32µs | 299.11µs | 0.690x (31% faster) |
| `query2_static_unchecked_2col` (Iter2Unchecked) | 423.22µs | 265.28µs | 0.627x (37% faster) |
| `raw_slice_ceiling — one_field_sum` (zero ECS code) | 105.76µs | 65.626µs | 0.620x (38% faster) |
| `spawn_insert_bundle` | 19.906ms | 12.234ms | 0.615x (39% faster) |
| `structural_churn_insert_remove` | 21.624ms | 14.708ms | 0.680x (32% faster) |

Every group in the binary got faster in build #13 — between 16% and
39% — except one: `query_static_single_component`, which got 90%
*slower*, enough on its own to land it almost exactly inside build
#13's "slow" cluster (262.06µs, next to `query2_static_unchecked_2col`'s
265.28µs) despite querying one column, not two. Its own nearest
sibling, `query_static_unchecked_1col` — same state machine, same
archetype-advance logic, the *only* line that differs is
`get_unchecked` instead of safe indexing — moved with the crowd (38%
faster). This is not "everything is noisy in the same direction," the
usual, dismissable pattern (see `structural_churn`/`spawn_insert_bundle`
above, which did move together with the crowd, consistent with a
faster run in general). One specific, safe, real, production function
moved alone, against 15+ others in the same binary, by a wide margin.

**The clean signal, unaffected by any of this:** `query_static_unchecked_1col`
vs `query2_static_unchecked_2col` — both diagnostics, both `get_unchecked`,
neither touched by whatever hit `query_static_single_component` — sits
at 3.99x (build #11) and 4.06x (build #13) at N=100,000, 3.33x/3.18x at
N=100. Stable, in lockstep, across a run where literally everything
else's absolute numbers moved by up to 39%. **This is the trustworthy
number, and it says unchanged: still ~4x, matching `ecs-vs-bevy-ecs`'s
own `dense_query_iteration` (3.99x as of build #14, a separate binary
sharing none of `archetype_core.rs`'s internal noise sources).** Build
#13's 1.24x reading is real data, correctly transcribed, and still
wrong to trust as "fixed" — its denominator broke, not its numerator.

**Cross-referencing this against `diag_inline.rs`'s own Never/Always/
Default group, already present in both these builds but not yet read
this way:** at N=100,000, `Default` (`query2_static_two_components`
itself) sits 1.1% from `Never` and 2.4% from `Always` in build #11; 0.2%
from `Never` and 10.7% from `Always` in build #13. Default tracks
Never, not Always, in both runs — the compiler is not silently
auto-inlining `Iter2::next` by default on rustc 1.98.1. That specific
theory (a hidden default-heuristic inline causing the gap) is ruled
out. What's *not* ruled out: build #13 shows `Always` measurably faster
than `Default`/`Never` (267.06µs vs ~298-299µs, ~11% at N=100,000,
~14-17% at smaller N) — the opposite of what the sandbox found when
`#[inline(always)]` was tried and reverted on `Iter2::next` itself (see
that method's own doc comment in `archetype.rs`, and this module's
header). Build #11 shows no such gap (all three within ~3.5% of each
other). One run showing an effect and one showing nothing is a lead,
not a finding — needs a third run to know if build #13's edge for
`Always` is real or noise.

**Built this pass, not yet run on real CI:** `Iter1Never`/`Iter1Always`
in `diag_inline.rs`, exposed as `World::query_static_diag_never`/
`_always`, benched as the new `query_static_single_component_diag_inlining`
group in `archetype_core.rs` — the exact same Never/Always/Default
treatment `Iter2` already has, applied to `Iter1` for the first time.
Motivation is the table above: `query_static_single_component` just
produced the single most specific, isolable anomaly this investigation
has seen (one real function, alone, moving hard against everything else
in its own binary), and until now nothing has ever tested whether
`Iter1`'s inline attribute matters the way `Iter2`'s might. If `Iter1`
shows the same Default≈Never pattern with `Always` pulling ahead, that's
a second, independent signal pointing at the same lever as `Iter2`'s —
and worth actually shipping `#[inline(always)]` on both real functions
if a third run confirms it, reversing the earlier sandbox-only-informed
revert (which never had real-CI `Iter2` data to check against — it does
now, partially, and this adds the `Iter1` half). If `Iter1` does *not*
show the pattern, `query_static_single_component`'s build #13 anomaly
has some other cause, and that's worth knowing too before spending more
time on the inlining line of investigation specifically.

Not a fix yet. Land this, run Archetype Core at least twice more (one
run already looked like a fluke this project — build #10 — and was
caught only by rerunning), and read `Iter1`'s new group the same way
the table above reads `query_static_single_component`'s.

### Builds #15 and #16: the `Iter1` attribute test came back negative, and something more useful came back positive instead

Landed the `Iter1Never`/`Iter1Always` diagnostic from the previous
section, then ran Archetype Core twice (builds #15, #16) specifically
to check the result wasn't a fluke, same discipline as everywhere else
in this investigation. It wasn't a fluke — the two runs agree with
each other to within 1-2% on nearly every group, a level of internal
consistency this suite hasn't shown since build #7. But the answer
itself is a clean negative on the question asked, and a clean positive
on a different, better question asked by accident.

**The negative result:** `query_static_single_component_diag_inlining`
— `inline_never`, `inline_always`, and `Default` (`query_static_single_component`
itself) — all landed within 1% of each other in both builds (e.g.
build #15 @ N=100,000: 377.18µs / 375.18µs / 375.25µs). No split at all,
in either direction. Pinning `Iter1::next`'s own inline attribute does
nothing, for either extreme — the same clean negative `Iter2Never`/
`Iter2Always` already gave for the 2-column case. **An attribute on
`next` itself, in isolation, is not the lever, for either arity.** That
line of investigation is closed.

**The accidental finding:** every benchmark that used to separate into
a "fast" 1-column cluster and a "slow" 2-column cluster has collapsed
into one cluster in both builds — including `query_static_single_component`
itself (375-377µs, up from its historical ~65-106µs) *and*
`query_static_unchecked_1col` (Iter1Unchecked — 375µs, also up from its
own historical ~65-106µs, and previously the single most reliable
"stays fast no matter what" reference point in this whole matrix).
Three things are still fast, unmoved, matching their own historical
numbers exactly: `raw_slice_ceiling` (~94µs, zero ECS code — nothing
for a binary-layout shift to grab onto), `two_tuple_item` (~94µs,
unchanged), and — this is the one worth sitting with —
`real_query1_inline_never_wrapper` (~94µs, unchanged), while its
structural twin `real_query2_inline_never_wrapper` sits at ~395-400µs,
matching `query2_static_two_components` exactly.

`RealQuery1::run`/`RealQuery2::run` (in `archetype_core.rs`'s
`bench_query2_static_diag_unchecked`) are byte-for-byte identical in
shape — same `#[inline(never)] fn run(&mut self) -> f32`, same for-loop
calling the real, unmodified `query_static`/`query2_static`, differing
only in `sum += pos.x` vs `sum += pos.x + vel.dx`. Wrapping the *whole
consuming loop* in an outer `#[inline(never)]` boundary reliably fixes
the 1-column case and reliably does not touch the 2-column case — this
is now the same result across at least three separate builds (#12/#13
per the previous session, #15/#16 this one). **"Just wrap the call
site" is ruled out as a sufficient fix for `query2_static` specifically
— whatever's different has to be something `Iter2::next` carries on its
own, not something fixable purely from outside it.**

**The likely explanation for the collapse itself, stated plainly since
it changes how to read the last several sections:** nothing in
`archetype.rs` changed between build #13 and build #15 — `Iter1::next`
is the exact same source it's been all session. What changed is that
`diag_inline.rs` and `query.rs` grew new code in the same compilation
unit `Iter1::next` lives in. The most likely read is that this shifted
enough about the compiled binary's layout to tip `Iter1::next` — which
build #13 already showed *can* tip, just not reliably before now — into
whatever regime `Iter2::next` sits in permanently. Shipping the next
diagnostic pushed the very thing it was trying to observe into a new,
now-stable state. Worth stating plainly rather than treating as a
side note: the regression-guard table's own baseline
(`query_static_single_component`) is not currently a safe thing to read
at face value, and won't be until this is actually understood, not just
patched around.

**Patched around in the meantime, because leaving it silent is worse
than a workaround:** `scripts/bench_mid_ecs_archetype_core.py` now
cross-checks `query_static_single_component` against `raw_slice_ceiling`'s
own floor at N≥1,000 (N=100 excluded — checked directly against build
#11's own healthy numbers, which still show a 2.6× baseline/floor ratio
at N=100 alone from ordinary fixed-per-call overhead, a false positive
this exclusion avoids) and refuses to print a clean ✅ — downgrading
every row to at least ⚠️ regardless of how good the raw ratio looks —
if that baseline is running >2.0× the floor. Verified against both
regimes directly: build #11's real numbers (healthy baseline, genuine
4.08× worst ratio) still correctly print the original 🔴 for the
original reason; build #15's real numbers (collapsed baseline, a
misleadingly clean 1.07-1.13×) now correctly print a drift warning and
downgrade every ✅ to ⚠️ instead of reporting a false pass. This doesn't
fix the underlying issue — it stops the summary from actively lying
about it while it's unresolved.

**Built and shipped this pass, not yet run on real CI:**
`Iter2ColdSplit` in `diag_query2_unchecked.rs` — `Iter2`'s exact logic
with the archetype-advance ("cold") branch physically moved into its
own `#[inline(never)] fn advance`, leaving `next`'s own body as just
the per-entity fast path and a call out to `advance` on the rare
branch. Different question than the already-closed one above: not "does
an attribute on `next` matter" (closed, no) and not "does wrapping the
*caller* matter" (closed for 2 columns, no) but "does the *cold path's
own size*, sitting physically inside `next`, affect how well the *hot*
path inlines at 2 columns' worth of state" — untested until now, and
the one remaining structural difference between `Iter2` and
`Iter1`/`Iter2TwoTupleItem` this investigation hasn't tried adjusting.
Three new tests (`diag_query2_static_cold_split_matches_the_real_query2_static`,
`_empty_when_one_side_was_never_registered`, and a new three-archetype
`_walks_multiple_matching_archetypes_and_skips_a_non_matching_one_between_them`
case — `two_archetype_world`'s own two archetypes weren't enough to be
confident the split between `next`/`advance` hands off state correctly
across more than one real archetype boundary) — 185/185 mid-ecs tests
pass, bench compiles clean. Benched as a new `cold_split` entry in the
existing `query2_static_diag_unchecked` group — no new bench group
needed, it slots in next to `composed`/`raw_ptr`/`owned_direct`.

Land it, run Archetype Core at least twice (same standing discipline),
and read `cold_split` against `two_tuple_item`/`raw_slice_ceiling` (the
still-reliable fast references) rather than against
`query_static_single_component` — which, per the section above, isn't
a safe comparison point right now regardless of what the automated
table says about it.

### Builds #17 and #18: `cold_split` is a clean negative, and the profile itself became a suspect

Ran Archetype Core twice more (builds #17, #18 — same commit,
triggered together, ~12-13% apart in absolute terms across every group
uniformly — the ordinary "different runner" pattern this project has
always treated as dismissable noise, not the isolated single-function
pattern builds #13/#15/#16 showed). `cold_split` landed in the slow
cluster in both — 376.37µs / 423.40µs at N=100,000, indistinguishable
from `composed`/`raw_ptr`/`owned_direct`/`unused_b_col`. **Clean
negative: the cold path's own size, factored out into its own
`#[inline(never)]` function, is not what's holding the hot path back.**
That closes this specific structural line of inquiry — outlining
doesn't help, at least not the way it was tried here. `query_static_single_component`
and `query_static_unchecked_1col` are still sitting in the collapsed
state builds #15/#16 first showed, in both #17 and #18 — four
consecutive builds now, so whatever tipped it during the `Iter1Never`/
`Iter1Always` addition looks like a settled new state, not a fluke.

At this point every source-level variant of `Iter2`'s own logic this
investigation has tried — safe, unchecked, raw-pointer, owned,
composed-fetch, attribute-pinned, cold-path-split — reproduces the same
number, with the sole exception of the one that adds a genuine indirect
call. That's a strong pattern in one direction (something about a fully
analyzable, fully inlinable 2-column loop specifically) but seven-plus
negative results without a working fix for the real API starts to look
like the wrong axis is being varied. Two things checked this pass that
aren't another `Iter2` source variant:

**Checked directly, not assumed: this workspace's own build profile.**
Root `Cargo.toml`'s `[profile.bench]` sets `codegen-units = 1` and
`lto = true` workspace-wide — added 2026-08-23, for a real, different,
already-fixed problem (`wide::i32x4::add` benching ~14x slower than
mid-math's own equivalent, a cross-crate inlining gap; see
docs/platform-optimization.md §9). Two things about it worth stating
plainly: (1) `benches/ecs-vs-bevy-ecs` is a workspace member, not a
separate workspace (checked directly, per its own Cargo.toml's
comment), so real `bevy_ecs` compiles under this exact same
codegen-units=1+LTO regime whenever that comparison runs — it isn't
somehow exempt. (2) Mid-D-Man/bevy's own root `Cargo.toml` (real
source, re-checked this pass) has neither a bare `[profile.release]`
nor any `[profile.bench]` at all — bevy's own benchmarks, the same
`benches/benches/bevy_ecs/iteration/*.rs` already read for the
`#[inline(never)]`-wrapper convention, run under cargo's plain,
unconfigured default: `codegen-units = 16`, no LTO. Bevy leans on
explicit `#[inline(always)]` (confirmed on `QueryIterationCursor::next`
itself) to guarantee its own hot path's codegen, not on whole-program
LTO to let the compiler figure it out. This project's workspace applies
LTO+single-codegen-unit to `Iter1`/`Iter2` as a side effect of fixing an
unrelated crate's problem, not as a deliberate choice for these two
functions specifically.

Searched before treating this as more than a coincidence: LTO making
one *specific* loop's own codegen measurably worse, independent of any
codegen-unit-reshuffling question, is real and already confirmed
upstream — rust-lang/rust#106609 ("LTO produces worse codegen for a
loop," with real before/after disassembly showing the LTO'd version
adds an extra live pointer the non-LTO'd version optimizes away) and
rust-lang/rust#146497 (a 2025 nalgebra/criterion reproduction showing
over 4000% degradation from `lto = "fat"` alone). Separately,
vortex-data/vortex#9259 (real PR, months old) hit the *other*
mechanism — `codegen-units = 16` reshuffling which functions share a
unit when unrelated code is added, moving benchmarks with no source
change on the branch that added them — and fixed it by moving *to*
codegen-units=1+lto=true, the setting this workspace already has. Two
distinct, independently-real mechanisms; this project's current
profile is already the known fix for one of them, and has never been
tested against the other for `Iter2`'s specific loop shape.

**Built and shipped, not yet run on real CI:** a `profile.bench-nolto`
entry in the root `Cargo.toml` (`inherits = "release"`, then every
field set explicitly: `opt-level = 3, lto = false, codegen-units = 16,
strip = false, debug = true` — matching bevy's own actual, unconfigured
numbers rather than guessing at them), plus a `profile` dispatch input
on this workflow (`bench`, the current default, or `bench-nolto`),
wired into the actual `cargo bench --profile` invocation and into the
cache key so the two don't share a `target/` cache entry. Verified
directly on this sandbox: `cargo build -p mid-ecs --profile bench-nolto`
and `cargo bench -p mid-ecs --bench archetype_core --profile bench-nolto`
both compile and run cleanly (rustc 1.91.1) — this only confirms the
plumbing works, not anything about the performance question itself,
which this sandbox has never been able to speak to for this
investigation either way.

Run Archetype Core once with `profile: bench-nolto` and once more with
the default `bench` on the same commit, and read the two side by side —
particularly `query2_static_two_components`, `query2_static_unchecked_2col`,
and (now that it's a clean negative under the current profile)
`cold_split` again under the other one. If any of them move
meaningfully between the two profiles while `raw_slice_ceiling` and
`two_tuple_item` don't, that's the profile, not the source, and the
seven-plus negative source-level results above stop being seven-plus
negative results and start being seven-plus results that were never
going to show anything because the actual lever was never in the
source to begin with.
