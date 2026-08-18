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
work for `mid-ecs` (`ffi.rs`) is deliberately last in the build order —
the FFI-span data structure design (`docs/mid-collections.md`'s "FFI
wrapper" section) will inform it once that's reached.

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
