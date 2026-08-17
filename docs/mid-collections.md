# mid-collections

**Status: real crate now (`crates/mid-collections`), one piece built
(Sparse Set). Still a design doc for everything else — written to be
built against, not a backlog to clear in one pass.**

The intent, stated explicitly so a future session doesn't second-guess
it: build each of these **when the specific need shows up**, not
upfront. `mid-geom`'s own history is the model to follow here — its gaps
(OBB, capsule-vs-AABB, broadphase) are explicitly left to be "driven by
mid-physics's actual requirements rather than built speculatively" (see
`docs/architecture.md`). Same discipline applies here: this doc exists so
nobody has to re-derive *why* a given structure is the right call when
the moment comes, not to justify building all of it before there's a
consumer. Sparse Set is the proof this works as intended — it got built
the moment `mid-ecs`'s real storage work started, nothing before it.

Not part of the build order as a separate phase — these get pulled in
piecemeal *during* the `ecs` phase (math → common → geom → net → **ecs**
→ physics → anim → nodes), triggered by whatever `mid-ecs` actually
needs at each point, not inserted as their own step.

## Sparse Set — built (`crates/mid-collections/src/sparse_set.rs`)

**Status: real, tested, done for v1.** `SparseSet<I: SparseSetIndex, T>`,
18/18 real tests passing on real rustc 1.75 — zero dependencies, so
(unlike everything quinn-touching in this workspace) nothing about this
verification is "syntax-checked, never run": it was actually compiled and
actually executed, including a real bug the test suite itself caught on
first run (a test's own logic was wrong, not the structure — fixed,
reran, confirmed).

Two arrays: a sparse one indexed directly by key, pointing into a dense
one, and a dense one holding the real data plus a back-pointer (split
internally into two parallel `Vec`s — `dense_keys`/`dense_values` — so a
pure `values()` scan never strides past key data it doesn't need; EnTT
and Bevy's own sparse sets are both split this way for the same reason).
O(1) insert/remove/lookup, and iteration over live elements is a straight
contiguous memory scan — no tombstones to skip, no indirection per
element.

Researched against real source before building, not from memory: EnTT's
current `sparse_set.hpp` (paged sparse array, three deletion policies,
generation baked into the entity ID itself) and Bevy ECS's real
`SparseSet`/`SparseSetIndex` (their own hybrid Table+SparseSet storage is
the closest existing real-world parallel to this project's own Hybrid ECS
Architecture — see `docs/mid-ecs.md` — right down to using an identical
"status effect that shouldn't fragment tables" example in their own
docs).

Deliberate scope decisions made, each one a smaller cut than EnTT's full
design, and why:
- **No paging.** EnTT pages its sparse array to stay cheap at an
  arbitrarily large entity-ID space. Mid Engine's stated target is
  100 000+ entities (`docs/architecture.md`), not millions — a flat,
  growable `Vec<u32>` sparse array costs at most a few hundred KB per
  component type at that scale, which doesn't justify paging's
  complexity. Revisit only if a real use case needs a much larger sparse
  ID space.
- **`u32::MAX` sentinel instead of `Option<u32>`** for empty sparse
  slots — matches EnTT's `null` and Bevy's own explored `NonMaxUsize`
  (`bevyengine/bevy` PR #2104). `Option<u32>` isn't niche-optimized in
  Rust, so it costs a discriminant plus padding for no benefit here.
  Worth being honest that Bevy's own benchmark on the equivalent change
  was inconclusive at runtime — the memory case is the one that's
  clear-cut, and it's the one `docs/architecture.md`'s own "zero-copy"
  mandate actually cares about.
- **`swap_and_pop` only** — no tombstones, no order-preserving removal.
  EnTT supports three deletion policies; nothing in `mid-ecs` needs
  anything beyond the simplest one yet. Add a policy when something
  actually needs one.
- **Generic over a `SparseSetIndex` trait, not any concrete `Entity`
  type** — mirrors Bevy's own real trait of the same shape, for the same
  reason: the generational-arena piece below (which defines what an
  actual entity handle looks like) hasn't been built yet, so this
  structure has no business assuming one. `u32` keys work today; whatever
  `Entity` type the arena work produces just needs to implement the
  trait once it exists.
- **`insert` replaces on collision and returns the old value**, matching
  `HashMap::insert`'s convention, rather than EnTT's assert-it's-absent
  contract — safer by construction, costs nothing extra to implement this
  way instead.

This isn't one option among several for `mid-ecs`'s storage — it's the
standard technique, and now it's real. Everything else in this doc is
still secondary to it and still unbuilt.

**Benchmarked for real** (`crates/mid-collections/benches/sparse_set.rs`,
`cargo bench -p mid-collections`), vs `std::HashMap<u32, T>` as the
honest baseline — not a strawman, it's what anyone would reach for
without a specific reason not to. The predicted split going in was
"`get`/iterate should win decisively, `insert`/`remove` should be
closer" — real numbers came back better than that hedge across the
board, at every scale from 100 to 100,000 elements:

| Operation | SparseSet vs HashMap, 100 elements | ...at 100,000 elements |
|---|---|---|
| `insert` (sequential) | 2.25x faster | 4.6x faster |
| `get` (existing key)  | ~20x faster  | ~24.7x faster |
| `remove` (all)        | 2.5x faster  | 5.3x faster |
| `iterate` (`values()`)| 4.5x faster  | ~5x faster |

`get`/`iterate` winning decisively was expected — direct array indexing
and a contiguous scan against hashing-plus-probing. `insert`/`remove`
winning by a similar margin was the actual surprise: the sparse array's
lazy growth (`Vec::resize` calls as new indices push past the current
extent — the direct cost of the "no paging" decision above) was expected
to eat into that margin at small sizes, and it doesn't show up as more
than noise even at n=100.

## Generational-index arena — built (`crates/mid-collections/src/generational_index.rs`)

**Status: real, tested, done for v1.** `GenerationalIndex` +
`GenerationalIndexAllocator`, 16/16 real tests passing (34/34 across the
whole crate, alongside Sparse Set's 18) — same "actually compiled and
run" standard as Sparse Set, verified by temporarily stripping the
criterion dev-dependency locally (the same technique this project used
once already for mid-ecs's rayon dependency), confirming, then restoring
the real `Cargo.toml` unchanged.

Not from the C++ list — this is the Rust ecosystem's own well-established
solution (`slotmap`, `generational-arena`) to a problem that shows up the
moment entities can be despawned and their ID slots reused: a stale
handle held from before a despawn must not silently alias whatever new
entity got allocated into that same slot. Fix: pair each slot with a
generation counter: incremented on reuse, and every handle carries the
generation it was issued with. A handle whose generation doesn't match
the slot's current one is caught as dead, not aliased.

Rank this **above** most of the C++ list below for `mid-ecs`
specifically — it's the single most common correctness bug in a naive
ECS handle design, and it needs to be right from the entity-ID design
itself, not retrofitted after.

Design verified against real `slotmap` 1.0.7 source, not assumed or
derived from a blog post: the core trick — one `u32` generation per slot
where **even means vacant, odd means occupied** — is `slotmap`'s own
actual shipped design (`src/basic.rs`), confirmed by reading it directly.
Deliberate divergence from `slotmap`/`generational-arena`: **no value
storage per slot.** Checked `hecs` and `bevy_ecs`'s own real entity
allocators before deciding this, not assumed — both roll their own
value-less allocator rather than reusing a generic slotmap crate, because
entity component data lives in per-component storage (`SparseSet` today,
the Archetype Core later) keyed *by* the entity, not stored *in* the
allocator. Building the heavier value-storing version would have solved
a problem `mid-ecs` doesn't have. Everything else — LIFO free-list reuse,
`wrapping_add` on generation bump instead of a checked/panicking add,
`free_head` pointing past the end of the slot array meaning "grow
instead" with no separate sentinel needed — matches `slotmap`'s own
choices directly, not reinvented independently.

**Wired into `mid-ecs` for real, not just built and left standalone.**
`mid-ecs`'s `World` (`crates/mid-ecs/src/world.rs`) is a thin wrapper —
`World::spawn`/`despawn`/`is_alive` delegate straight through to a
`GenerationalIndexAllocator`, and `Entity` is a thin wrapper over
`GenerationalIndex` implementing `SparseSetIndex` itself, so it composes
directly with `SparseSet` with no adapter type needed once component
storage exists. This is `mid-ecs`'s first genuinely real behavior — no
more empty stub — verified the same way: rayon (`mid-ecs`'s own MSRV
wall) stripped temporarily, 12/12 real tests run against the actual
`World` API, restored unchanged. One real bug caught by that run, not by
review: a test meant to prove "despawning a handle from an unrelated
`World` fails" was itself wrong — two fresh `World::new()` allocators
both hand out `{index: 0, generation: 1}` as their first spawn, since
`GenerationalIndex` carries no per-allocator identity, so the "unrelated"
handle wasn't actually distinguishable from a legitimate one. Fixed by
testing the thing that's actually load-bearing instead: a *stale* handle
after its slot gets reused by a *new* entity must fail to despawn that
new entity.

**Explicitly not yet done, and why:** `World::despawn` does not remove
any component data, because there's no component storage attached to
entities yet. This is the exact place that removal will need to be
threaded through once `SparseSet`-backed component storage exists —
*before* the slot is freed and reused, or a reused slot's new entity
would silently inherit the old one's leftover components. Demonstrated
directly (not just described) in `generational_index.rs`'s own
`usable_as_a_real_sparse_set_key` test.

## Lock-free SPSC/MPMC ring buffer

Fixed-size array, atomic read/write cursors, no mutex. Real, current
relevance to `mid-net`, not hypothetical: `QuinnTransport`'s cross-task
channels (background datagram/accept/writer tasks talking to the sync
`Transport` methods) currently use `tokio::sync::mpsc`. A hand-rolled
ring buffer would let that path drop the `tokio::sync` dependency for
the pure in-process handoff, in the same spirit as every other
hand-rolled-over-heavy-crate call already made in this project (see
`docs/architecture.md`'s dependency mandate). Also the natural choice
for any future OS-thread-to-gameplay-thread handoff (input events,
render commands) once those subsystems exist.

## Intrusive lists

`next`/`prev` pointers stored directly inside the user's own struct
instead of a separate node wrapper. Zero allocation on insert/remove,
total pointer stability. No consumer yet — relevant once a job system or
scene graph exists (`mid-nodes`, not started), not before.

## Hierarchical bitset

An array of 64-bit words, each bit a flag ("does entity N have
component X"). Bitwise AND/OR/XOR checks thousands of entities per
instruction. Pairs directly with the sparse set above for archetype
queries ("give me every entity with Transform AND Velocity"), and later
for broad-phase culling in `mid-geom`/rendering. Build alongside the
sparse set, not before it — it's the query layer on top of that storage,
not a standalone structure.

## `std::hive` (formerly `plf::hive`, now C++23)

Bucketed storage: stable pointers/indices to elements even as others are
inserted or erased, O(1) erase without the shift-everything cost of a
`Vec::remove`. Real prior art, but no immediate `mid-ecs` consumer
identified yet the way sparse-set has one from day one — flagged here so
it's not forgotten, not scheduled.

## The FFI wrapper (`FfiBuf`-shaped, name not settled)

Not speculative — grounded in a real, already-visible cost in this
codebase. Every `PlayerState`/`PlayerEvent` getter in `mid-net`'s
`ffi.rs` hand-rolls its own null-check + buffer-size-check +
`catch_unwind` boundary, one function at a time (this is exactly what
the `dangerous_implicit_autorefs` fix earlier this project touched). A
shared wrapper owning that logic once, with individual FFI functions
becoming thin calls into it, would consolidate duplication that's
already real and growing with every new FFI function added.

Scope call: **build on `zerocopy`, don't reimplement alignment-checked
casting by hand.** Checked directly (not assumed): `zerocopy`'s
`FromBytes` family does real runtime alignment checking as part of the
trait itself — `ref_from_bytes`/`mut_from_prefix` return `Err` on
misaligned input, not UB — and it's maintained by Google/Amazon
engineers, used in the Linux kernel's own Rust bindings for this exact
byte-casting problem. What it doesn't have, and what this wrapper would
actually add: a sentinel/canary corruption check, and folding the
null-check + `catch_unwind` boundary into the type itself so it isn't
repeated at every call site. `bytemuck` is the lighter sibling with
similar but less rigorous guarantees — mentioned for completeness, not
the pick.

Debug/release split (from the same discussion, worth keeping): checks
active in debug builds, compiled away to a zero-cost pass-through in
release — the same "prove it during testing, pay nothing in production"
shape mid-net's own hot paths already follow.

## Explicitly out of scope for this doc

A separate project ("Ubel" — a language design with its own tiered
compilation/memory model) came up in the same conversation that produced
this doc. Deliberately not folded in here — different project, and this
doc only tracks what's grounded in mid-engine's own real code and real
needs.
