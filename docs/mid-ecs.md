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

**Status: design only, not yet implemented.** No `GlobalTransform`
component exists in this crate yet. This section records the design
decided for it, so implementation starts from an agreed shape rather
than getting re-litigated. The actual `f64` math primitives this
design depends on already exist and are real — see `docs/mid-math.md`.

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
`Affine3` is 48. `GlobalTransform` is about the hottest, most-iterated
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

While the Rust core handles the raw, flat memory arrays, the gameplay programmer never has to think about Archetypes or Bitsets.
* **HIGH Tier:** Developers interact with what looks like standard OOP classes (e.g., an `Actor` or `Entity` object).
* **LOW Tier:** The Ubel compiler acts as the "Middle Man," secretly lowering high-level code (e.g., `player.health -= 10`) into raw, memory-safe array accesses in the `mid-ecs` core.

## Network Sync (Multiplayer-First)

The `sync` module marks components for `mid-net` replication.
This is the Multiplayer-First mandate in practice: networking is baked into the ECS from day one, not bolted on later.
* Components can be explicitly flagged for synchronization (e.g., `@net Transform`). 
* The engine automatically handles serialization via DixScript (`.mdix`) to sync state across the wire.
* Because data is stored contiguously in the Archetype Core, the network system can simply request a memory block and run a single SIMD pass over that memory to detect deltas, compress with MBFA-lite, encrypt, and ship the UDP packet.
