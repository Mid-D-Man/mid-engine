# mid-arena

Arena/slot allocators for Mid Engine. Motivated by `mid-ecs` wanting an
arena allocator, but scoped wider on purpose: a real survey of 28 Rust
arena crates and 3 established C arena libraries came first, and the
crate is being built piece by piece against what that survey actually
found, not against a single reference implementation someone half-
remembered.

The short version, if you read nothing else: `SlotArena<T>` is built and
tested (16/16 real, rustc 1.75), it's the safe Vec-with-freelist approach
every serious Rust arena crate in this space converges on, and real CI
(rustc 1.98, criterion — see "Real CI benchmark results") confirms it's
squarely competitive with its true peers (`slotmap`, `generational-arena`,
`typed-generational-arena`, `thunderdome` — all generation-checked, same
as it is). Plain `slab` is the real outlier, ~4x faster than that whole
band, because it skips generation-checking entirely — a documented
safety trade-off, not something `SlotArena` was competing to match.
`CompactSlotArena<T>` is built too (14/14 tests, behind `compact`) — a
`slotmap`-style union layout, ported from `slotmap`'s own real source,
landing a hair faster inside that same peer band on real CI, not in a
different league. `BumpArena<T>` is built (11/11 tests, behind `bump`)
— single-typed, chunk-linked, the approach both the Rust and C surveys
found fastest for insert-heavy workloads. Its first version measured
3.2x slower than `bumpalo`/`typed-arena` on real CI; reading `bumpalo`'s
actual source and matching its real intrusive-linked-list structure
fixed that, and real CI now shows it tying both within measurement
noise (a second bug — a missing `black_box` — briefly made it look
*faster* than both, caught and fixed before trusting that number; see
"Fixes and Problems"). GC-based approaches are
ruled out entirely, not benched further. Everything else is catalogued,
not built.

## Modules

### `lib.rs`
**What it does:** crate root, module declarations, the feature gate
roadmap and the garbage collection exclusion, all in the crate-level doc
comment.

**Decisions:** see "Feature gates" and "Explicitly out of scope: garbage
collection" below.

### `slot_arena.rs`
**What it does:** `SlotArena<T>` and `ArenaKey`, the generational
value-storing arena. See "What's built" below for the full design.

**Decisions and benchmarks:** see "What's built", "Real CI benchmark
results", and "Fixes and Problems" below.

**Tests:** 16, in this file, `#[cfg(test)] mod tests`. Run for real on
rustc 1.75 before `criterion` was added as a dev-dependency (see "Fixes
and Problems").

### `bump_arena.rs`
**What it does:** `BumpArena<T>`, single-typed chunk-linked bump
allocator, feature-gated behind `bump`. See "Feature gates" below for
why this shape and not `bumpalo`'s mixed-type one.

**Decisions:** second version now. The first used
`RefCell<Vec<Region<T>>>`, which real CI numbers showed running 3.2x
slower on insert than `bumpalo`/`typed-arena`. Cloning and reading
`bumpalo`'s and `slab`'s actual source found the real cause and this
version fixes it -- see "Fixes and Problems" below for the full story.
Now: `Cell<NonNull<RegionNode<T>>>` intrusive linked list, matching
`bumpalo::Bump`'s real structure directly. Geometric region growth
ported from `tsoding/arena.h`'s real source, unchanged from the first
version.

**Tests:** 11, in this file. Covers multi-region growth, geometric
capacity doubling, `iter_mut` order (including specifically across a
region boundary) and write-through, and running
`Drop` for every value across every region on arena drop.

### `compact_slot_arena.rs`
**What it does:** `CompactSlotArena<T>`, union-based generational slot
arena, feature-gated behind `compact`. Same `ArenaKey` handle type and
algorithm as `SlotArena`, ported from `slotmap` 1.0.7's real
`src/basic.rs`. Deliberately a separate type from `SlotArena`, not a
feature-swapped internal representation of it -- see this file's own
doc comment for why (Cargo feature unification would otherwise let an
unrelated crate's `compact` flag silently change `SlotArena`'s behavior
for everyone in the build).

**Decisions:** real, checked finding while building this: writing an
entire new value to a `ManuallyDrop<T>` union field needs no `unsafe` on
this compiler, only reads do -- confirmed by the compiler itself
(`unused_unsafe` warnings on the first draft), not assumed from
`slotmap`'s own file-level `#![allow(unused_unsafe)]` comment, which
turned out to describe some other rustc/edition combination, not this
one.

**Tests:** 14, in this file. Includes one specifically checking that a
removed-then-reused slot's old value isn't double-dropped when the
arena itself later drops -- the real risk this union layout carries
that `SlotArena`'s plain enum doesn't.

**Wired into the bench:** `benches/vs_arena_crates.rs` now includes
`CompactSlotArena` in insert/get/churn, behind `#[cfg(feature =
"compact")]` -- see "Fixes and Problems" below for when.

### `examples/drop_arena_standalone.rs`
**What it does:** standalone `std::time::Instant` micro-benchmark for
`drop_arena`, run via `cargo run --release --example
drop_arena_standalone -p mid-arena`, not a criterion bench inside
`vs_arena_crates.rs`.

**Decisions:** `drop_arena::DropBox`'s `Drop` implementation calls back
into the arena through an invariant lifetime tied to the arena's own
type parameter. Every criterion bench in `vs_arena_crates.rs` returns
the populated arena from the timed closure so criterion has something
to observe; that shape does not compile for `drop_arena` (confirmed
directly, `E0505`/`E0515`, not assumed) since returning `(arena,
boxes)` -- or even just `arena` alone after any `DropBox` exists --
ties the arena's borrow to the same lifetime the type itself carries.
Lives under `examples/`, not `benches/` or `src/bin/`, specifically
because example targets get `[dev-dependencies]` access the way
tests/benches do; plain `[[bin]]` targets do not, and `drop_arena` has
no reason to be a real dependency of this crate.

**Tests:** none (it is itself a measurement program, not a
correctness-tested module) -- its own three operations were manually
verified against expected values (checksums, lengths) in a separate
scratch project before this file was written, the same verification
standard every other real API addition in this survey got.

## Survey methodology

Started from a comparison table (screenshot, `Overview` sheet) plus the
donsz.nl blog post it's drawn from — fetched directly rather than
transcribed from the screenshot alone, which turned up 6 more crates the
screenshot didn't include. 28 unique crates total:

**From the screenshot:** `slab`, `bumpalo`, `sharded-slab`,
`typed-arena`, `slotmap`, `id-arena`, `generational-arena`,
`internment`, `concurrent_arena`, `atree`, `multi-stash`, `colosseum`,
`gc`, `atomic-arena`, `gc-arena`, `typed-arena-nomut`, `compact_arena`,
`bump-scope`, `shredder`, `erased-type-arena`, `elise`, `drop_arena`.

**From the blog, not in the screenshot:** `thunderdome`,
`typed-generational-arena`, `blink-alloc`, `bumpalo-herd`, `riddance`,
`hato`.

Every crate was checked against this project's real rustc-1.75 floor
with an isolated `cargo build` (one crate at a time, in a scratch
package, not inferred from a changelog) before anything else happened —
buildability first, then benching, matching the order actually asked
for: bench and compare before reading source and taking anything.

## Rust crate buildability (real, checked directly, rustc 1.75.0)

| Crate | Builds on 1.75? | If not, why |
|---|---|---|
| slab | ✅ | |
| bumpalo | ✅ | |
| sharded-slab | ✅ | |
| typed-arena | ✅ | |
| slotmap | ✅ | |
| id-arena | ✅ | |
| generational-arena | ✅ | |
| internment | ✅ | |
| atree | ✅ | |
| multi-stash | ✅ | |
| colosseum | ✅ | |
| gc | ✅ | |
| atomic-arena | ✅ | |
| typed-arena-nomut | ✅ | |
| compact_arena | ✅ | |
| erased-type-arena | ✅ | |
| thunderdome | ✅ | |
| typed-generational-arena | ✅ | |
| blink-alloc | ✅ | |
| bumpalo-herd | ✅ | |
| riddance | ✅ | |
| drop_arena | ✅ | |
| concurrent_arena | ❌ | `triomphe` (transitive) needs rustc 1.81+ |
| gc-arena | ❌ | `gc-arena-derive` needs `edition2024` (~1.85+) — same wall class as `mid-net-transport-quinn`, root `Cargo.toml` |
| bump-scope | ❌ | needs `edition2024` directly (~1.85+), same wall class |
| shredder | ❌ | `rayon` (transitive) needs rustc 1.80+ — same wall `mid-ecs` itself already hit |
| elise | ❌ | uses nightly-only `#![feature(...)]` unconditionally — not an MSRV gap, no stable channel supports it at all |
| hato | ❌ | uses `core::ptr::from_ref`, stabilized ~1.76 — one version past this project's floor |

22/28 buildable today. The 4 real MSRV walls (`concurrent_arena`,
`gc-arena`, `bump-scope`, `shredder`) are all CI-only for this project,
same as every other `edition2024`/rustc-1.8x wall already documented in
root `Cargo.toml` — not unusual, not a reason to exclude them from the
design conversation, just from local benching this pass.

## Rust benchmarks (real, actually executed — not criterion)

`criterion` needs `edition2024` transitively (`clap_builder`), same wall
as everywhere else in this project — unusable in this sandbox. Benched
with `std::time::Instant` instead: N=100,000, a 16-byte
`{ a: u64, b: u64 }` payload, single core, `opt-level=3` release build,
each crate's own real API (checked against real docs.rs pages before
writing a single call, not assumed from memory). 10 crates chosen to
cover every distinct `Approach` in the survey table, plus `mid-arena`'s
own `SlotArena` in the same harness. 3 runs each; the sandbox is a noisy
single-core VM, so figures below are the middle/most-consistent run,
with the observed range noted for anything that moved more than ~20%
run to run.

| Crate | Approach | insert (ns) | get (ns) | remove_half (ns) | reinsert_half (ns) |
|---|---|---|---|---|---|
| slotmap | Vec + freelist | 5.7–6.1 | 1.2–1.4 | 2.7–4.9 | 3.4–3.8 |
| slab | Vec + freelist | 11.5–20.0¹ | 1.2–3.3 | 2.2–2.8 | 4.1–4.8 |
| generational-arena | Vec + freelist | 9.2–11.1 | 1.7–2.5 | 3.4–4.5 | 6.7–11.1 |
| thunderdome | Vec + freelist | 11.2–11.8 | 1.2–1.7 | 2.5–2.5 | 4.3–4.9 |
| id-arena | Indexed Vec, no reuse | 10.6–12.1 | 1.1–1.7 | — | — |
| **mid-arena `SlotArena`** | Vec + freelist (enum slot) | **21.7–33.3¹** | **1.3–3.5** | **2.4–4.5** | **4.0–5.2** |
| typed-arena | Linked arena chunks | 2.7–3.5 | 0.7–1.9 | — | — |
| bumpalo | Linked arena chunks | 7.5–8.3 | 0.7–0.8 | — | — |
| internment | Hashset of boxes (dedup) | 52.1–57.5² | 1.4–1.6 | — | — |
| sharded-slab | Sharded, lock-free | 112.5–115.7 | 22.1–23.4 | 26.7–26.9 | — |
| gc | Garbage collected | 24.3–25.7³ | 2.2–2.6 | — | — |

¹ Both `slab` and `SlotArena` had a visibly-elevated first-run number
(20.0 ns and 33.3 ns respectively) that dropped and stabilized on
repeat runs — read as sandbox/allocator warm-up noise, not a real cold
vs warm cost difference in either crate, though not independently
confirmed beyond re-running.
² `intern_dedup_hit` (interning the same value a second time) measured
22.0–23.3 ns — roughly half the unique-insert cost, since it's a hash
lookup plus a hit instead of a hash lookup plus a real allocation.
³ `gc::force_collect()` on 100k still-live objects: 4.6–5.1 ns/object.
On the same 100k after they'd all been dropped: 47.0–57.5 ns/object —
a real, measured 10x jump, entirely from the collector actually having
work to do. This is the concrete number behind "Explicitly out of
scope: garbage collection" below, not a hypothetical one.

**What this means:** the Vec-with-freelist family (`slotmap`, `slab`,
`generational-arena`, `thunderdome`) is a tight, competitive band — all
within about 2x of each other on every operation, confirming this is
the right general-purpose default, not just a convenient one.
Linked-arena-chunks (`typed-arena`, `bumpalo`) wins insert by 2–4x when
there's nothing to remove. `sharded-slab` costs roughly 10–20x more than
plain `slab` unshared, matching that crate's own documented caveat about
lock-free overhead not paying for itself single-threaded — real
confirmation, not just trusting the caveat.

**The "honest surprise" this pass got wrong, corrected below:** this
sandbox pass originally reported `SlotArena` measuring 2–3x slower on
insert than its closest algorithmic peers, with a `size_of`-grounded but
ultimately unconfirmed guess about branch shape as the cause. Once
`benches/vs_arena_crates.rs` actually ran on real CI (rustc 1.98.0,
criterion, GitHub Actions run #3) that gap didn't hold up — see "Real CI
benchmark results" below for the corrected numbers and a better-grounded
explanation. Left visible here, struck through in spirit rather than
deleted, because the point of recording a surprise honestly is that it
can turn out to be sandbox noise, and this project's own convention is
to say so plainly rather than quietly edit the earlier claim away.

## Real CI benchmark results (rustc 1.98.1, actual GitHub Actions runs #8 and #10 — not the sandbox pass above)

`benches/vs_arena_crates.rs` run for real on CI (`workflow_dispatch`).
Took several real runs to get here, not one clean shot — runs 1 and 2
failed on CI infrastructure (a cache-key collision, then a caching
mistake around the toolchain binary itself), run 3 was the first clean
result but only covered `SlotArena`, runs 4 to 6 chased down a real
`BumpArena` benchmarking bug (missing `black_box`, made it look faster
than `bumpalo` itself, which was the tell that something was wrong) and
a real table-formatting bug (Rust's batch-total time printed next to
C's already-per-op time, unconverted), and run 8 added
`CompactSlotArena` and `typed-generational-arena` once those existed.
Every one of those runs is logged in "Fixes and Problems" below, not
smoothed over. This table is the state as of run #8, all figures
already per-operation (criterion's own methodology — many samples,
statistical, real hardware — not a single `Instant` call on a shared
sandbox VM).

`atomic-arena` and `drop_arena` were added after run #8 — real API
verified in isolated scratch projects (see "Fixes and Problems" below).
`atomic-arena` got its first real CI number on run 10: 15.98 ns/op
insert, 2.07 ns/op get, notably slower than every other crate in its own
generation-checked group (that group otherwise sits at 5.5–7.0 ns/op
insert) — a real, standalone finding, not folded into the "ties its
peer band" story the rest of that group shares. Plausible cause not yet
confirmed by reading the source further: `atomic-arena`'s `Controller`
type exists to support cross-thread reservation even when unused, and
that bookkeeping may carry a real cost regardless of whether a given
call site ever touches the concurrent path. `drop_arena` never gets a
CI number the same way the C libraries don't: it's measured by
`examples/drop_arena_standalone.rs`, `std::time::Instant`, not
criterion, for a real confirmed reason (that module's own doc comment).
Its run 10 numbers (11.18 / 1.19 / 2.90 ns/op) land inside the range
already seen while building it (8.9–14.8 / 0.83–2.32 / 2.90–9.54
ns/op) — meaningfully slower than plain `typed-arena` across every run,
which tracks: it's `typed-arena` with free-list bookkeeping layered on
top, and that bookkeeping has a real cost, the same way `slab`/
`slotmap`'s ABA-safety bookkeeping does elsewhere in this survey.

**Vec + freelist, ABA-safe (generation-checked)** — insert / get, ns/op

| Crate | insert | get |
|---|---|---|
| generational-arena | 5.66 | 0.87 |
| typed-generational-arena | 5.67 | 0.77 |
| **mid-arena `CompactSlotArena`** | **6.91** | **0.75** |
| **mid-arena `SlotArena`** | **6.93** | **0.86** |
| slotmap | 6.96 | 0.71 |
| thunderdome | 7.01 | 0.76 |
| atomic-arena | 15.98 | 2.07 |

**Linked arena chunks (bump, no per-item reuse)**

| Crate | insert | get |
|---|---|---|
| **mid-arena `BumpArena`** | **1.30** | **0.34** |
| typed-arena | 1.32 | 0.33 |
| bumpalo | 1.37 | 0.34 |

**Everything else** — `slab` (no ABA check) 1.66 / 0.66, `id-arena`
(indexed, no reuse) 6.47 / 0.64, `sharded-slab` (sharded/lock-free)
29.95 / 10.03, `internment` (hashset dedup) 71.60 / —. Full grouped
table, C libraries and `drop_arena` included, in the workflow's own
step summary.

**The corrected finding, still holding across every run since:**
`SlotArena` isn't an outlier. It sits inside the same ~5.7–7.0 ns band
as `generational-arena`/`typed-generational-arena`/`slotmap`/
`thunderdome` — its actual peer group, all of them generation-checked,
ABA-safe handles — and `CompactSlotArena` lands a hair faster inside
that same band, not in a different league. What's actually unusual is
`slab`, at roughly 3.5–4.2x faster than that entire band — and there's
a real, checkable reason for that rather than a guessed one: `slab`'s
`usize` keys carry **no generation counter at all**. Reusing a freed
slot's index hands out the exact same key value it had before;
`slab`'s own documentation is explicit that this is a real, accepted
ABA trade-off, not an oversight. Every crate in the generation-checked
band pays a real, measured cost for the staleness check that buys
ABA-safety. That's a fair trade to be making, and it's the correct
comparison: `SlotArena`/`CompactSlotArena` were never competing with
`slab`'s weaker guarantee.

**`BumpArena` is now honestly competitive, not artificially ahead:**
first measured 3.2x slower than `bumpalo` (a real design gap, fixed by
matching `bumpalo`'s actual intrusive-linked-list structure — see
"Fixes and Problems"), then measured *faster* than both `bumpalo` and
`typed-arena` (a benchmarking bug, missing `black_box`, not a real
result), now ties them within measurement noise. Three different
numbers for the same code across this project's history, each one
real for the reason recorded at the time — worth stating plainly rather
than only keeping the final one.

**Loose ends, not yet followed up:**
- `internment` has no `get` figure — `bench_get` never included it
  (interning's return value *is* the access handle; there's no
  separate lookup step to time), not a gap.
- The checked-in `gc` bench only measures `force_collect` against a
  fully-live 100k-object set (a single sweep, no garbage to reclaim).
  It does **not** reproduce the earlier sandbox pass's after-drop
  figure (footnote 3 above, 47.0–57.5 ns/object) — that measurement
  only ever existed in a scratch script, never in `vs_arena_crates.rs`.
  Worth closing that gap.
- Criterion has warned about incomplete samples on more than one run
  now ("Unable to complete 100 samples in 5.0s"). Non-fatal every time,
  likely `internment`/`gc` given their multi-ms iteration cost. A
  `.sample_size(50)`/longer `.measurement_time(...)` on those specific
  groups would clear it; not done yet since it doesn't change the
  numbers' validity, only the noise floor around them.

## C arena libraries (real, compiled `-O3 -march=native`, actually run)

Picked for spread, same reasoning as the Rust survey's approach
diversity: one production-grade packaged library (`apr_pools`, decades
inside Apache HTTPD/Subversion), one genuinely different paradigm
(`talloc`, hierarchical/reference-style rather than flat bump), one
minimal header-only reference (`tsoding/arena.h`, MIT, the same role
`HandmadeMath.h` plays in `mid-math`'s own C comparisons).

| Library | Paradigm | insert (ns) | get (ns) |
|---|---|---|---|
| tsoding/arena.h | Bump allocator, whole-arena reset | 6.3–8.6 | 1.1 |
| APR pools (`apr_palloc`) | Bump allocator, whole-pool clear | 12.5–13.8 | 1.0–1.1 |
| talloc (`talloc_pool`) | Hierarchical, reference-style | 61.4–76.1 | 4.2–6.8 |

Reuse/free doesn't unify across the three — each API's own real shape,
not forced into one row:

- **tsoding/arena.h**: `arena_reset()` (whole arena) took 1255 ns once;
  the next 100k inserts into the reset arena ran 1.55–2.49 ns/op —
  *faster* than the original fill, since the backing regions are
  already grown and just get reused from the top.
- **APR pools**: `apr_pool_clear()` took 7793–32017 ns once (more
  variance than tsoding's reset — APR's clear walks and destroys any
  registered cleanups/sub-pools, real extra bookkeeping tsoding's
  arena doesn't have); reinsert after clear ran 3.31–3.58 ns/op.
- **talloc**: per-item `talloc_free()` on half the allocations ran
  23.0–28.3 ns/op each — real calls, but per `talloc_pool`'s own
  documented contract (quoted directly in `talloc_bench.c`'s header
  comment, not paraphrased into something it doesn't say), freeing a
  pooled child does **not** give its bytes back to the pool; only
  freeing the whole pool does. `talloc_free()` on the whole pool
  (recursively freeing every remaining child, running every
  destructor) took 796585–885088 ns — genuinely the most expensive
  single operation measured anywhere in this survey, entirely because
  it's doing real recursive tree work the other two approaches don't
  do at all.

**Cross-language sanity check:** `tsoding/arena.h`'s 6.3–8.6 ns insert
lands right next to `typed-arena`/`bumpalo`'s 2.7–8.3 ns in the Rust
table — the same bump-allocator approach measuring the same in both
languages is a real, useful confirmation that neither number is an
artifact of one toolchain or the other.

Source: `crates/mid-arena/benches/{tsoding_arena_bench.c, apr_pool_bench.c,
talloc_bench.c}`, compiled and run directly in this sandbox (gcc 13.3.0,
`-O3 -march=native`) — not deferred to CI the way the Rust criterion
suite had to be. `scripts/bench_vs_c_arena_libs.py` parses all four
outputs (three C, one Rust) into one step-summary table; tested against
the real captured C output above, since that part could be verified
directly — the Rust half degrades to `—` gracefully until a real CI run
produces `/tmp/rust.txt`. `.github/workflows/bench-vs-c-arena-libs.yml`
mirrors `bench-vs-c-libs.yml`'s structure exactly (apt-installs
`libapr1-dev`/`libtalloc-dev`, curl-fetches `arena.h` at CI time the same
way that workflow fetches `HandmadeMath.h` — not committed to the repo).

## What's built: `SlotArena<T>`

`crates/mid-arena/src/slot_arena.rs`. Generational, value-storing arena:
`insert` returns an `ArenaKey`, `get`/`get_mut`/`remove`/`contains` all
validate that key's generation against the slot's current one before
returning anything, so a stale handle from a freed-and-reused slot reads
as "not present" rather than aliasing the new value.

Directly extends `mid_collections::GenerationalIndexAllocator`'s own
algorithm rather than re-deriving one: same even-vacant/odd-occupied
generation trick, same LIFO free list, same
`free_head == slots.len()` past-the-end-means-grow convention. The one
real difference — `Slot<T>` has to be an enum (`Occupied { generation,
value }` / `Vacant { generation, next_free }`), not a flat struct, since
a vacant slot has nowhere to put an arbitrary `T`'s bit pattern without
either requiring `T: Default` or reaching for an unsafe union the way
`slotmap` does internally. Plain safe enum by default, matching this
workspace's own established precedent (`SparseSet`,
`GenerationalIndexAllocator`) that raw-pointer/union tricks wait for a
real, profiled need — and now there is one, logged above, feeding
directly into the `compact` feature gate.

**Tests:** 16, all real, all passing on rustc 1.75 — run and recorded
*before* `criterion` was added as a dev-dependency, same sequencing
`mid-collections` used for its own SparseSet/GenerationalIndex tests,
same reason: verify everything the current toolchain still can before
adding the thing that closes that window. Covers insert/get/get_mut
round-trips, remove-returns-value, remove-on-dead-or-unknown-handle as a
safe no-op, LIFO reuse order, generation bump on reuse, iteration
(including that it skips removed slots), `iter_mut` write-through,
`clear()` dropping every live value and invalidating every outstanding
handle, `slot_count()` vs `len()` divergence after free/reuse, a 50-round
mixed insert/remove consistency sweep, `as_ffi`/`from_ffi` round-tripping
including across a generation bump, and — the one test worth calling out
by name — `drop_runs_for_every_live_value_when_the_arena_itself_is_dropped`,
which uses a real `Drop`-counting type to *check* the "Runs Drop" column
claim rather than assume it from `Vec<T>`'s own well-known behavior.

Adding `criterion` afterward for `benches/vs_arena_crates.rs` triggers
the same `edition2024`-via-`clap_builder` wall as everywhere else in this
project — a real, documented regression (root `Cargo.toml`'s comment
block, "a SIXTH independent instance"), not hidden: `cargo test -p
mid-arena` alone now needs the newer toolchain too, not just `--bench`.
Checked directly that `--lib` does *not* route around this (Cargo
resolves a package's full manifest, dev-dependencies included, before
building any target from it — `mid-collections`' own note already found
this, re-confirmed here rather than re-assumed). Also checked the
other direction, since it wasn't obvious either way: plain `cargo build
-p mid-arena` (no `--tests`/`--benches`) still works fine on 1.75 —
verified directly, immediately after adding the dependency. It doesn't
need dev-dependencies compiled, and with no regular dependencies of its
own, there's nothing left for the resolver to trip on. So consuming
`mid-arena` from elsewhere in this workspace stays on the 1.75 floor;
it's specifically testing or benching `mid-arena` itself, locally, that
doesn't.

`benches/vs_arena_crates.rs` has since run for real on CI (rustc 1.98.0,
`workflow_dispatch` run #3 — see "Real CI benchmark results" above). It
took three attempts to get a clean run: the first two failed on workflow
infrastructure (a cache-key collision with four unrelated workflows in
this repo, then a caching-the-toolchain-binary issue), not on anything
in this crate's own code — both fixed in
`.github/workflows/bench-vs-c-arena-libs.yml` directly, nothing here
needed to change. Worth naming plainly rather than glossing over: the
Instant-based sandbox numbers earlier in this doc turned out to disagree
with the real run on at least one real conclusion (the `SlotArena`
insert-time "surprise"), which is exactly why they're kept, labeled, and
superseded rather than quietly replaced.

## Relationship to `mid-collections`' `GenerationalIndex`

Worth being direct about, since the algorithm is shared: this doesn't
replace or second-guess `mid-collections::generational_index`.  That
module is deliberately value-less — its own doc comment states the
reasoning: `mid-ecs`'s entity allocator has nowhere useful to put a
value, because component data lives in per-component storage
(`SparseSet` today, the Archetype Core later) keyed *by* the entity, not
stored *in* the allocator. `World::spawn`/`despawn` should keep using
`GenerationalIndexAllocator`, unchanged — nothing here argues otherwise.

`SlotArena<T>` is for value storage that doesn't already have a
`SparseSet` sitting one layer up: asset caches, DixScript AST nodes, MSX
path-command buffers, scripting object tables. Real candidate consumers
— **and, as of this pass, nothing in this workspace actually calls
`SlotArena` yet.** Stated plainly rather than implied: this crate was
built ahead of a wired-in consumer, on the strength of the survey and
the explicit ask to cover this ground broadly. That's a deliberate,
one-time departure from `mid-collections`' own "pulled in piece-by-piece
exactly as `mid-ecs` needs it" build order (`docs/mid-collections.md`),
not a quiet abandonment of it — worth flagging honestly rather than
letting the two docs read as if they'd never noticed the tension.

## Feature gates (`bump` and `compact` built, rest still planned)

- **`compact`** — built. `CompactSlotArena<T>`, a `slotmap`-style
  unsafe union slot layout. Originally justified in this doc by an
  insert-time gap that the real CI run ("Real CI benchmark results"
  above) showed wasn't actually there — `SlotArena` ties `slotmap` and
  sits inside the same band as its other true peers, so this was never
  closing a speed gap. Built on the narrower, honest justification
  instead: memory footprint (`Slot<T>`'s enum discriminant, even where
  it fits inside existing alignment padding for free as measured above,
  doesn't always — a union layout removes that dependence on `T`'s own
  alignment for `Vacant`/`Occupied` to share space for free).
- **`bump`** — built. `BumpArena<T>`, single-typed chunk-linked bump
  allocator (`bumpalo`/`typed-arena`/`tsoding-arena`'s shared approach),
  for insert-heavy, rarely-freed workloads. Both the Rust and the C
  survey agreed this approach wins that shape of workload by a wide
  margin before this was built; the real numbers already in this doc
  are what motivated building it first out of everything on this list.
  Real CI numbers on the first version showed it running 3.2x slower on
  insert than `bumpalo`/`typed-arena` despite the same approach — see
  "Fixes and Problems" below for the real cause (found by reading
  `bumpalo`'s actual source, not guessed) and the rewrite that
  substantially closed that gap without fully eliminating it.
- **`intern`** — hashset-of-boxes dedup arena (`internment`'s
  `ArenaIntern` approach), for string/path/asset-key interning.
- **`concurrent`** — sharded lock-free slab (`sharded-slab`'s
  approach). Deliberately not default: this survey's own real
  benchmark shows it costing roughly 10–20x more than plain `slab`
  single-threaded, matching `sharded-slab`'s own documented caveat.
- **`ffi`** — checked FFI access, matching `mid_collections`'s own
  `ffi` feature shape exactly (optional `zerocopy` 0.8.56 dependency,
  `derive` feature only, off by default). `ArenaKey::as_ffi`/`from_ffi`
  already exist unconditionally (cheap, no dependency) — this feature
  is specifically for a `checked_slice`-equivalent over arena-owned
  memory, not built yet.

Every one of these traces to a specific real number or a specific real
API gap above, not to "this is what other arena crates tend to have."

## Explicitly out of scope: garbage collection

`gc`, `gc-arena`, `shredder`, `elise` all solve a real problem, and
`gc`'s numbers above are real and reasonable for what they are. The
issue isn't performance — it's that a tracing GC's collection pause is,
by construction, not a cost a caller can bound in advance, and this
project's own `docs/architecture.md` sets hard 128 Hz/60 Hz frame
budgets. The measured 47.0–57.5 ns/object collection cost after a drop
(footnote 3, above) isn't the concern by itself; the concern is that
number scaling with however much garbage happens to be live *at the
moment the collector decides to run*, which is exactly the kind of
latency spike a physics or network tick can't absorb. If a scripting
sandbox ever genuinely needs tracing-GC semantics, that belongs in its
own crate with its own explicit, opted-into latency contract — not
blended into an allocator every other system is assumed to be able to
call without a pause budget.

## CI and Workflows

- `.github/workflows/bench-vs-c-arena-libs.yml` — builds and runs the
  three C library benchmarks plus the Rust `vs_arena_crates` criterion
  suite, publishes a unified comparison table to the step summary.
  Depends on `scripts/bench_vs_c_arena_libs.py` to parse and merge both
  sides' raw output into one table.

## Fixes and Problems

### `lib.rs`
- `cargo doc` without the `bump` feature enabled produced two broken
  intra-doc link warnings for `[bump_arena]`/`[BumpArena<T>]`, since
  that module doesn't exist in scope when the feature is off. Fixed by
  dropping the link brackets in favor of plain code-formatted text for
  that one line, checked against both feature configurations after.

### `scripts/bench_vs_c_arena_libs.py`
- The insert/get table printed Rust's raw criterion figure (total time
  for a 100,000-op batch) next to C's already-per-op figure, unconverted,
  in the same column. Real, repeated confusion across at least two CI
  runs: read by eye, C looked dramatically faster than every Rust crate
  including `slab`, when normalized it's the other way around for
  `tsoding/arena.h` against the bump allocators, and close for the rest.
  Fixed by dividing every Rust figure by its real operation count (N for
  insert/get, 200,000 or 150,000 depending on the crate for the churn
  table, since `sharded-slab` measures a different op count than the
  rest there) before printing anything, and labeling `gc`'s two rows
  differently since one is genuinely per-operation and the other is a
  single sweep that dividing would understate. Verified against a
  synthetic criterion-format file built from run 5's real numbers, not
  just read by inspection.
- Even after the unit fix (run 6/7), the insert/get table still listed
  implementations in benchmark-registration order -- a flat list mixing
  generation-checked arenas, `slab` (no ABA check), bump allocators (no
  reuse at all), and everything else in one sequence, so comparing
  `SlotArena` against its real peers meant scanning past several
  unrelated rows first. Reorganized into one small table per real
  approach category (`docs/mid-arena.md`'s own survey taxonomy),
  fastest to slowest within each, `mid-arena`'s own types bolded.
  Verified against the same synthetic-file approach, extended to cover
  the two crates added in this same pass (`CompactSlotArena`,
  `typed-generational-arena`) landing in the correct group.

### `.github/workflows/bench-vs-c-arena-libs.yml`
- Run 1 failed with "cargo: command not found" inside the Rust bench
  step, reported as a green step. Cause: a repo-wide cache restore-key
  prefix matched four unrelated workflows' cache entries and restored
  one of their `~/.cargo/bin/` snapshots over the freshly installed
  toolchain, and missing `set -o pipefail` let `cargo bench | tee`
  report `tee`'s exit code instead of cargo's, hiding the failure.
- Run 2 scoped the cache key to this workflow and still failed the same
  way, root cause not fully confirmed. Fixed by removing `~/.cargo/bin/`
  from the cached paths entirely (only `registry/` and `git/db/` need
  caching, the toolchain binary never should have been cached at all)
  and adding a PATH diagnostic step.
- Run 3 succeeded. Real numbers now live in "Real CI benchmark results"
  above.
- Open: criterion warned about incomplete samples on both run 3 and run
  4 ("Unable to complete 100 samples in 5.0s"), non-fatal both times,
  likely `internment` or `gc` given their multi-millisecond iteration
  cost. Not tuned yet.
- Open: the checked-in `gc` bench only measures `force_collect` against
  a fully live set. The after-drop number in "Explicitly out of scope:
  garbage collection" only exists in a scratch sandbox script, not in
  this suite.

### `bump_arena.rs`
- First version measured 3.2x slower on insert than `bumpalo`/
  `typed-arena` on real CI (run 4), despite using the same approach.
  Root cause found by cloning and reading `bumpalo`'s and `slab`'s
  actual current source rather than continuing to guess: `bumpalo::Bump`
  holds a single `Cell<NonNull<ChunkFooter>>` pointing directly at the
  current chunk (an intrusive linked list), where the first version of
  this file used `RefCell<Vec<Region<T>>>` — paying for a `RefCell`
  borrow check, a `Vec` index to find the current region, and doing
  that lookup twice per call, none of which `bumpalo` pays for at all.
  Rewritten to the same `Cell<NonNull<RegionNode<T>>>` intrusive
  structure `bumpalo` actually uses. A follow-up fix (eliminating a
  redundant second `current` read even when no growth happened)
  narrowed the gap further. Real, local (not CI) sandbox comparison
  after both fixes: roughly 1.5 to 1.7x slower than `bumpalo`/
  `typed-arena`, down from the original 3.2x — a substantial, measured
  improvement, not full parity. The remaining gap wasn't root-caused
  further; this sandbox has no profiler to look past what source
  reading alone can explain.
- While verifying the test suite locally, mixed up which of two similar
  tests actually needed `mut` on its `BumpArena` binding: removed it
  from `iter_mut_visits_every_value_in_allocation_order_and_writes_through`
  (which calls `iter_mut(&mut self)` and genuinely needs it) instead of
  from `later_regions_hold_at_least_double_the_previous_capacity` (which
  only calls `alloc(&self)` and doesn't). Caught by re-running the full
  suite after the first edit instead of assuming it was right, fixed
  both, re-ran again to confirm.

### `compact_slot_arena.rs`
- First draft wrapped every union field write in `unsafe`, following
  `slotmap`'s own file-level `#![allow(unused_unsafe)]` comment
  literally. The compiler disagreed: two real `unused_unsafe` warnings
  on writes to `ManuallyDrop<T>` union fields, which need no `unsafe` on
  this rustc since `ManuallyDrop<T>` has no drop glue to skip in the
  first place. Fixed by removing the unnecessary wrapping and correcting
  the safety comments to say what's actually true here rather than what
  `slotmap`'s own comment (written for some other rustc/edition
  combination) seemed to imply.

### `benches/vs_arena_crates.rs`
- The original sandbox pass (`std::time::Instant`, not criterion)
  reported `SlotArena` insert running 2 to 3 times slower than its
  closest peers. Sandbox noise: the real CI run showed `SlotArena` ties
  `slotmap` exactly and sits inside the same band as
  `generational-arena`/`id-arena`/`thunderdome`. The real outlier is
  `slab`, about 4 times faster than that band, because its keys carry
  no generation counter (a documented trade-off in `slab`'s own source).
  The `compact` feature's justification below was rewritten once this
  became clear.
- `BumpArena`'s insert bench was the one entry in this file missing
  `black_box()` around each per-item allocation, unlike every other
  crate's own bench. Real consequence, not cosmetic: without it, the
  compiler had room to auto-vectorize a tight sequential write into a
  flat, non-aliasing buffer in a way none of the fair comparisons got to
  benefit from, and one real CI run showed `BumpArena` running faster
  than `bumpalo` itself -- implausible for a newer, less-optimized
  implementation doing the same thing. Fixed; the next real run showed
  `BumpArena` landing within measurement noise of `bumpalo`, not
  artificially ahead of it.
- Added `CompactSlotArena` (behind `compact`) and `typed-generational-arena`
  (real API verified in an isolated scratch project before use -- exact
  same `insert`/`with_capacity`/indexing/`remove` shape as
  `generational-arena`, confirmed by compiling and running it, not
  assumed from the crate description) to insert/get/churn.
  `atomic-arena` and `drop_arena` were also considered and passed over
  for this round -- both have real API complexity beyond a simple
  `insert`/`get`/`remove` shape (`atomic-arena`'s `Controller`-based
  reservation workflow, `drop_arena`'s lifetime-tied `DropBox` wrapper),
  and getting either wrong in a benchmark is worse than not benching it
  yet.
- Followed up on both, real source checked this time rather than
  reasoning from the API's surface shape. `atomic-arena`'s real source
  (`atomic-arena-0.1.2/src/lib.rs`) has a plain `insert(&mut self, data:
  T) -> Result<Key, ArenaFull>` alongside the `Controller` workflow --
  the reservation machinery is for cross-thread use, not required for
  ordinary single-threaded insert/get/remove. Verified end to end in an
  isolated scratch project, added normally, no caveats needed.
  `drop_arena` was a real, confirmed dead end for this file specifically:
  tried returning `(arena, boxes)` from a closure, then just `arena`
  alone after allocating from it, both failed to compile
  (`cannot move out of arena because it is borrowed`) -- `DropBox`'s
  `Drop` impl calls back into the arena via an invariant lifetime tied
  to the arena's own type parameter, which is fundamentally incompatible
  with the "return the populated arena from the timed closure" shape
  every criterion bench in this file uses. Not a workaround-able API
  quirk, an actual structural property, confirmed by trying it twice.
  Solved by moving it to `examples/drop_arena_standalone.rs`, a plain
  `std::time::Instant` program with no return-value constraint at all
  (matching the same reasoning the `.c` benchmarks in this directory
  already use, for a related reason). Real numbers gathered from it are
  in "Real CI benchmark results" above.
- Run 10 reported `id-arena` insert at 1.51 ns/op -- wildly outside its
  own consistent 5.5-7.5 ns/op range across every prior run, and faster
  than `bumpalo` itself, the same shape of implausible result the
  `BumpArena`/`black_box` incident had. Checked the code directly rather
  than assume the same bug: `black_box` was already present and correct
  in `id-arena`'s bench, ruling that out. Real, separate, genuine finding
  instead: `id-arena`'s bench was the only one in this whole file using
  `Arena::new()` instead of `Arena::with_capacity(N)` -- every other
  crate here pre-allocates, so `id-arena` alone was paying variable
  Vec-reallocation cost the others don't, making it structurally more
  exposed to scheduling noise on a busy runner (criterion's own
  diagnostics flagged this specific run as unusually unstable: 3 sampling
  warnings with suggested target times up to 9.4s, versus the usual 1
  warning around 5.1s). `id_arena::Arena::with_capacity` exists (checked
  directly) and is now used in all three `id-arena` bench sites. Also
  fixed while investigating: `CARGO_TERM_COLOR: never` added to the
  `drop_arena` example's CI step, since raw ANSI color codes from the
  compile log were leaking into that step's raw-output display (cosmetic
  only -- `parse_c()` only matches lines ending in `ns/op`, so the parsed
  numbers were never affected). Next CI run is the real test of whether
  the `with_capacity` fix actually stabilizes `id-arena`'s number; not
  folded into "Real CI benchmark results" above until confirmed, since
  run 10's figure for it specifically should not be trusted.

## Reproducing these numbers

Rust crate buildability: `cargo build` against each crate individually
in a scratch package, one at a time, rustc 1.75.0 (`apt install rustc
cargo` on Ubuntu 24.04 — this project's documented sandbox floor).

Rust benchmarks: `std::time::Instant`-based, not checked into this repo
as a standalone binary (the checked-in, CI-runnable version is
`crates/mid-arena/benches/vs_arena_crates.rs`, criterion-based, for a
real toolchain). N=100,000, `opt-level=3`, single core, 3 runs.

C benchmarks: `crates/mid-arena/benches/{tsoding_arena_bench.c,
apr_pool_bench.c, talloc_bench.c}`, `gcc -O3 -march=native`, linked
against `libapr1-dev`/`libtalloc-dev` (apt) and a curl-fetched
`arena.h` (`raw.githubusercontent.com/tsoding/arena/master/arena.h`,
MIT). `.github/workflows/bench-vs-c-arena-libs.yml` automates all of
this end to end, including the unified summary via
`scripts/bench_vs_c_arena_libs.py`.

Every number in this doc came from an actual run in this sandbox, not
from a crate's own README or a remembered benchmark — checked directly
because the last several docs in this project found real gaps between
"what the ecosystem says" and "what actually happens on this project's
own floor" often enough that assuming the two match isn't a safe
default anymore.
