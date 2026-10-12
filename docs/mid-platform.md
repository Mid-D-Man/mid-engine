# mid-platform

## Overview

`bevy_platform` (`Mid-D-Man/bevy`, real source read directly — `crates/bevy_platform/`,
version `0.20.0-dev`) is the crate 55 of Bevy's 60 crates transitively depend on
(`docs/bevy-comparison.md` §1). `docs/roadmap.md`'s Decision 3 deferred building a
`mid-platform` equivalent until a second crate needed a wasm32-safe `Instant`, a
no_std-safe mutex, or a faster-than-SipHash hash map — naming the trigger explicitly
so it wouldn't get silently missed.

**It already fired, twice, unnoticed, before this doc was written:**
- `mid-time` independently hand-rolled its own wasm32-safe `Instant` (native
  `std::time::Instant`, an `f64`-based one on `wasm32`).
- `mid-alloc` independently hand-rolled `SpinLock<T>`, ported directly from the
  real `spin` crate's algorithm and verified under real multi-threaded stress
  tests (`crates/mid-alloc/src/sync.rs`'s own test module — 8 real OS threads,
  2,000 increments each, zero lost updates).

So building this now isn't overriding Decision 3 — its own named condition for
revisiting has genuinely been met.

**What it isn't, unlike `mid-ptr`:** zero-dependency. `bevy_platform`'s real
manifest pulls in `spin`, `portable-atomic`, `portable-atomic-util`, `foldhash`,
`hashbrown`, `critical-section`, and (target/feature-gated) `web-time`,
`windows-sys`, `wasm-bindgen`, `wasm-bindgen-futures`, `js-sys`, `futures-channel`,
`futures-lite`, `async-io`, `serde`, `rayon`, `bytemuck`. `docs/architecture.md`'s
Technical Mandates say "zero-to-minimal external dependencies, every core crate,
no exceptions," with a real precedent of rejecting this project's own published
`dixscript` crate as a core dependency over its transitive count. This doc exists
to go through that list one dependency at a time and decide, concretely, which
ones mid-engine's real target matrix (native desktop + `wasm32-unknown-unknown`,
both of which have full `std` and full-width native atomics — no true bare-metal
/ no-OS target exists anywhere in this workspace's plans) actually needs, versus
which only exist for platforms bevy supports that mid-engine doesn't.

## Dependency-by-dependency verdict

| Dependency | What it's for in `bevy_platform` | Needed for mid-engine's real targets? | Verdict |
|---|---|---|---|
| `spin` | `Mutex`/`RwLock`/`Once`/`Barrier`/`LazyLock` fallback impls when `std` is off | Only the *mechanism* (spin-locking), not the crate itself | **Hand-roll.** `mid-alloc::sync::SpinLock` already proves the exact algorithm works, tested under real threads. mid-platform gets its own independent copy of the same shape (see "Why not just depend on mid-alloc" below) rather than a Cargo dependency on either `spin` or `mid-alloc`. |
| `foldhash` | The actual hash algorithm behind `DefaultHasher`/`RandomState`/`FixedState` in `hash.rs` | Yes, if we want a fast general-purpose hasher at all | **Hand-rolled — done (Phase 3, `hash.rs`).** A fold/multiply-based hash is well-trodden ground (FxHash-style algorithms are ~30 lines); built as its own benchmarked pass rather than bolted onto an earlier slice, see "Why a second, independent `FxHasher`" below. Deliberately no `RandomState` equivalent — see "hash.rs" under Modules. |
| `hashbrown` | The actual `HashMap`/`HashSet`/`HashTable` implementation (`collections/hash_map.rs` is 1,292 lines, `hash_set.rs` 1,083 — real API-surface work on top of the raw table, not a thin wrapper) | Only if we want a `std`-HashMap-equivalent API at all | **Hardest one — separate decision, not answered by this doc.** Reimplementing a correct, reasonably fast open-addressing hash table (SIMD group matching, tombstone-free Robin-Hood-style probing, raw-entry API) from scratch is a materially bigger undertaking than anything else on this list — closer in scope to a second `mid-ptr`-sized project than a one-file port. Not attempted in this pass. |
| `portable-atomic` | Atomic types on platforms lacking a native width (8/16/32/64/ptr) | **No** — only pulled in via `[target.'cfg(not(all(target_has_atomic = "8", "16", "32", "64", "ptr")))'.dependencies]`; x86_64, aarch64, and wasm32 (with atomics) all have full native width support | **Not needed at all.** `mid_platform::sync::atomic` can be a pure re-export of `core::sync::atomic` — verified directly against `sync/atomic.rs`'s own `#[cfg(target_has_atomic = "N")]` gating, not assumed. |
| `portable-atomic-util` | `Arc`/`Weak` built on `portable-atomic` when the platform lacks pointer-width atomics | **No**, same reasoning as above (`[target.'cfg(not(target_has_atomic = "ptr"))'.dependencies]`) | **Not needed.** `mid_platform::sync::Arc`/`Weak` can be a pure re-export of `alloc::sync::{Arc, Weak}`. |
| `critical-section` | Interrupt-disabling abstraction, `portable-atomic`'s own fallback mechanism on true bare-metal/no-OS targets | No — only relevant if `portable-atomic` itself is needed | **Not needed**, follows directly from the `portable-atomic` verdict above. |
| `web-time` | wasm32 `Instant` via `performance.now()` | No — `mid-time` already solved this with its own `f64`-based approach, no dependency | **Not needed**, already solved. |
| `windows-sys` | `dirs::windows::preferences_dir()` — OS "where does the user's app-preferences folder live" resolution | Not clearly — this is an application/tooling concern (Bevy uses it for its own editor tooling), not an engine-core primitive. `dirs::linux`/`dirs::macos` need zero external deps (pure `std::env` + XDG spec logic); only Windows needs the Win32 Shell API | **Defer entirely** — trigger-based, same discipline as Decision 3 itself: build when something (an asset-cache path, a save-file location) actually needs it, not speculatively. |
| `wasm-bindgen`, `wasm-bindgen-futures`, `js-sys`, `futures-channel` | The optional `web` feature — browser API / JS-future integration | Not yet — no async task executor exists anywhere in this workspace (`docs/bevy-comparison.md` §2 already notes "no App/Plugin/schedule-runner layer") | **Defer entirely**, trigger-based. |
| `futures-lite`, `async-io` | Optional `block_on` executor backends (`future.rs`'s fallback is a plain busy-spin poll loop and needs neither) | No | **Hand-roll the busy-spin fallback only** (already zero-dep in `bevy_platform` itself); skip the optional accelerated backends until something needs a real executor. |
| `serde` | Optional serialization support | No | **Defer**, trigger-based (add `serde` as an optional feature if/when a crate that already depends on it needs `mid-platform` types to serialize). |
| `rayon` | Optional parallel-iterator support for the hash collections | No — named in Decision 3 itself as a *possible future* trigger ("mid-ecs's eventual parallel query work"), not a current need | **Defer**, matches Decision 3's own reasoning exactly. |
| `bytemuck` | Optional cast methods on `collections::aligned_vec`'s `AlignedVec` (1,010 lines, mostly bytemuck-independent) | Not yet — no current consumer | **Defer the whole `AlignedVec` type**, not just the bytemuck feature — no trigger for it yet either. |

## Why not just depend on `mid-alloc` for `SpinLock`

The premise for prioritizing this crate at all is that it has no in-workspace
dependency — a true layer-0 crate, matching `bevy_platform`'s own position in
Bevy's graph. Depending on `mid-alloc` for its `SpinLock` would invert that.
`mid-platform` gets its own copy of the same algorithm instead: test-then-
test-and-set (attempt the real compare-exchange first, only fall back to a
cheap `Relaxed`-load spin-wait on contention), the same shape `mid-alloc`
already validated under real multi-threaded stress tests. Not a blind
duplication — grounded in code this same workspace already proved works,
just kept as an independent copy so each crate's dependency count stays
honest.

## Why a second, independent `FxHasher`

`mid-collections` already has a working FxHash-style hasher
(`crates/mid-collections/src/fx_hash.rs`: `FxHasher`, `FxBuildHasher`) —
the same fold/multiply algorithm this crate's Phase 3 was always going to
need. Same question `SpinLock` raised, same answer: `mid-platform` gets its
own independent copy rather than a Cargo dependency in either direction.
`mid-collections` is documented as a no-engine-deps layer-0 crate too
(`web/site/src/dependency-order.md`), so neither crate sits below the other;
a dependency either way would break the "layer-0, no in-workspace
dependency" premise this crate was prioritized on, for a ~30-line algorithm
that's cheaper to copy than to couple.

The copy is deliberately close to the original, not a redesign: same
multiplier, same rotate-5/xor/wrapping-multiply step, same 8/4/2/1-byte
`write` chunking, same sign-extension of narrow signed integer writes.
Two real differences, both intentional:
- `core::hash` instead of `std::hash`, so the module is usable under the
  `no_std` configuration with no feature split at all (`mid-collections`'s
  copy imports `std::hash`, though both traits live in `core` — a one-line
  fix there if it's ever wanted, not touched in this pass).
- `FxHasher::with_seed(u64)`, so `FixedHasher` can start from a fixed
  non-zero state (`bevy_platform::hash::FixedHasher`'s own role) instead of
  zero.

Naming: this crate follows `bevy_platform`'s names (`FixedHasher`,
`fixed_hash_one`), matching every other module here.
`mid-collections`'s `FxBuildHasher = BuildHasherDefault<FxHasher>` is a
different, already-established public name for the analogous thing in that
crate — not renamed or aliased to match, since changing an existing public
API wasn't asked for. Disclosed drift, same class as the doc-location and
versioning drift already noted elsewhere in this project.

**Open, not decided here:** whether `mid-collections` should eventually
switch to this crate's hasher (a dependency in the other direction, which
`mid-collections` doesn't have today) or keep its own copy forever, the
way `mid-alloc` keeps its `SpinLock`. Nothing forces the question until a
second consumer of one hasher actually wants a single source of truth.
Also spotted while checking, not fixed (out of scope): `fx_hash.rs`'s top
comment still names `crates/chemistry_core/src/fx_hash.rs` — a stale path
from where it was copied.

## Build order

**Phase 1 (this pass):** the pieces with the highest confidence and lowest
risk — no new concurrency-primitive design, either a pure re-export or a
straight port of an already-proven algorithm:
- `sync::atomic` — pure re-export of `core::sync::atomic`
- `cell::{SyncCell, SyncUnsafeCell}` — pure logic, no locking involved at all
- `sync::poison` — pure logic (`PoisonError`/`TryLockError`/`LockResult`/`TryLockResult`), needed by every lock type below
- `sync::mutex::Mutex` — `std::sync::Mutex` when available, otherwise mid-platform's own spin-based fallback (see above)
- `sync::{Arc, Weak}` — pure re-export of `alloc::sync::{Arc, Weak}`

**Phase 2 (done):** real new concurrency-primitive design work, each piece
grounded in real upstream source read fresh rather than from memory —
`Mid-D-Man/bevy`'s own `crates/bevy_platform/src/sync/` (confirms the real
API shape each type needs to match) and `spin` 0.10.0's real source from
crates.io (the actual algorithm each one's `no_std` fallback is grounded
in, since `bevy_platform` itself just re-exports `spin`'s types rather than
implementing anything — reading its files alone would not have been enough).
See "Modules" below for what each file actually does and why:
- `sync::rwlock::RwLock` — hand-rolled spin-based reader/writer lock, a simplified cut of `spin::RwLock`'s real algorithm (no upgradeable-guard mechanism — out of scope for this crate's `read`/`write`/`try_read`/`try_write` API surface)
- `sync::once::{Once, OnceLock, OnceState}` — double-checked locking built on top of `sync::mutex::Mutex`, not a from-scratch atomic state machine
- `sync::lazy_lock::LazyLock` — builds directly on `OnceLock`, plus `cell::SyncUnsafeCell` for the initializer slot
- `sync::barrier::Barrier` — ported from `spin::Barrier`'s real algorithm onto this crate's own `Mutex`

**Phase 2, explicitly deferred (not part of this pass):**
- `thread::sleep` — std path trivial; no_std fallback busy-spins on `Instant`, needs the `mid-time`-consolidation decision below settled first
- `future::block_on` — the busy-spin fallback itself is simple and zero-dep; mainly blocked on deciding whether `mid-platform` re-exports `mid-time`'s `Instant` or grows its own

**Phase 3, hasher half (done):** `hash.rs` — `FxHasher`, `FixedHasher`,
`fixed_hash_one`. Benchmarked against `rustc-hash` 1.1.0 and std's SipHash
(`benches/hash_bench.rs`, first run pending). No `RandomState` equivalent,
on purpose — see "hash.rs" under Modules.

**Phase 3, `HashMap`/`HashSet` half (open — needs an explicit decision, not
started):** `collections::hash_map`/`hash_set`. The options on the table,
none chosen:
1. A `std` passthrough only — `HashMap`/`HashSet` aliases over
   `std::collections` with `FixedHasher` as the default hasher, and no
   `no_std` fallback at all yet. Smallest, matches how every other type in
   this crate is built `std`-first; leaves the `no_std` configuration
   without a map until something needs one.
2. Depend on `hashbrown` directly (what `bevy_platform` itself does).
   Breaks this crate's zero-dependency premise, for the one thing on the
   list that's genuinely hard to hand-roll well.
3. A deliberately simple, safe, hand-rolled open-addressing map for the
   `no_std` path — slower than `hashbrown`, far easier to get right with no
   compiler in the working environment, measured against real workloads
   before anyone decides whether the gap matters.
4. A from-scratch SwissTable/`hashbrown`-equivalent — the "second
   `mid-ptr`-sized project" the dependency table above already names.
   Not recommended without a compiler and fuzzing in the loop.

**Not planned at all right now:** `dirs::*`, the `web`/async-executor features,
`serde`, `rayon`, `collections::aligned_vec` — all trigger-based deferrals per
the table above, named here so, like Decision 3 itself, they aren't silently
forgotten later.

## Modules

Phase 1's files (`sync/atomic.rs`, `cell.rs`, `sync/poison.rs`,
`sync/mutex.rs`, the `sync::{Arc, Weak}` re-exports) don't have their own
sections here yet — a disclosed gap from when this doc was first written
phase-by-phase rather than file-by-file, not fixed retroactively in this
pass; each Phase 2 file below gets one, matching
`docs/DOCUMENTATION_AND_COMMENTING_GUIDELINES.md`'s per-file convention
going forward.

### `sync/rwlock.rs`

**What it does:** `RwLock`/`RwLockReadGuard`/`RwLockWriteGuard`. `std`
passthrough when the `std` feature is on; otherwise a hand-rolled spin-based
reader/writer lock.

**Decisions:**
- Single `AtomicUsize` state (bit 0 = `WRITER`, each reader adds `2`) —
  `spin::RwLock`'s own real shape (source read directly, not from memory),
  minus its `UPGRADED` bit and `RwLockUpgradableGuard`/upgrade-downgrade
  methods. Left out deliberately: this crate's Phase 2 scope is
  `read`/`write`/`try_read`/`try_write` only, and a simpler correct design
  was prioritized over a more feature-complete port.
- `RwLockWriteGuard::drop` uses `fetch_and(!WRITER, ..)`, never a blind
  `store(0, ..)` — a real correctness requirement, not a style pick. See the
  file's own top doc comment for the exact race a blind store would open
  (a losing `try_read`'s speculative `fetch_add`/compensating `fetch_sub`
  pair can straddle a writer's unlock).
- Unfair to writers under continuous read pressure — same disclosed
  trade-off `spin::RwLock`'s own doc comment states for itself. No fairness
  mechanism exists to alleviate it (that's exactly what the omitted
  `UPGRADED`/upgradeable-guard machinery would have provided). Worth a real
  pass if a workload ever actually hits this; not built speculatively.

**Tests:** in-file `#[cfg(test)]` module — exclusive/shared access,
try-fails-while-held for both read and write, `Send`/`Sync` bounds, poison
no-ops, and one real-multi-thread stress test (4 writers × 1,000 increments
racing 4 readers, asserts the final count is exact).

### `sync/once.rs`

**What it does:** `Once`/`OnceLock`/`OnceState`. `std` passthrough when
available; otherwise a hand-rolled fallback.

**Decisions:**
- Built on top of `sync::mutex::Mutex`, not a from-scratch CAS state
  machine — real `spin::Once` (source read directly) hand-rolls its own
  four-state (`Incomplete`/`Running`/`Complete`/`Panicked`) atomic protocol;
  this type deliberately doesn't port that. With no compiler in this
  project's working environment to catch a subtle state-machine bug,
  reusing the crate's own already-tested `Mutex` for the actual mutual
  exclusion was judged more trustworthy than a new hand-rolled primitive —
  and since this crate's `Mutex` never poisons, there's one fewer state
  (`Panicked`) to model in the first place.
- Not a bare `Mutex<Option<T>>` either, despite that being simpler still: a
  separate `completed: AtomicBool`, checked with `Acquire` before ever
  touching the lock, keeps `get()` fully lock-free once initialized,
  matching `std::sync::OnceLock`'s real performance characteristic — routing
  every `get()` through the mutex would regress that for what's expected to
  be the hot path.
- `Once` is a thin `OnceLock<()>` wrapper — same composition real `spin`/
  `bevy_platform` both use.

**Tests:** in-file — closure-runs-exactly-once (both `get_or_init` directly
and via `Once::call_once`), `get`/`set`/`take` state transitions, poison
no-ops, `Send`/`Sync` bounds, and a real-multi-thread test (8 threads racing
`get_or_init`, asserts the initializer ran exactly once and every thread
observed the same result).

### `sync/lazy_lock.rs`

**What it does:** `LazyLock<T, F>`. `std` passthrough when available;
otherwise built directly on `OnceLock`.

**Decisions:**
- Does not replicate `std::sync::LazyLock`'s real internal layout (a
  hand-rolled union storing either the initializer or the result in the
  same memory, swapped via raw pointer writes) — that complexity exists in
  std purely for a memory-layout optimization, not for correctness. Instead:
  an `OnceLock<T>` plus this crate's own `cell::SyncUnsafeCell<Option<F>>`
  (Phase 1, already built and tested) to hold the initializer until
  consumed. No new unsafe algorithm gets invented for this type at all —
  every bit of synchronization it needs, `OnceLock::get_or_init` already
  provides; `force`'s own `unsafe` block is a single-line proof that
  `OnceLock`'s own exclusivity guarantee covers the `.take()` on `init` too.

**Tests:** in-file — initializer-runs-once (via both `Deref` and the
explicit `force` function), and a real-multi-thread test (8 threads racing
a `Deref`, asserts the initializer ran exactly once and every thread got
the same value).

### `sync/barrier.rs`

**What it does:** `Barrier`/`BarrierWaitResult`. `std` passthrough when
available; otherwise ported from `spin::Barrier`'s real algorithm.

**Decisions:**
- Lowest priority of Phase 2's four primitives (rarely used) and the
  simplest to get right: real `spin::Barrier` (source read directly) is
  itself built on top of `spin::Mutex<BarrierState>`, not a bespoke atomic
  protocol — a generation counter plus a thread count, guarded by a plain
  lock. Ported onto this crate's own `Mutex` instead of `spin::Mutex`, same
  "reuse an already-proven primitive from this crate" reasoning
  `sync/once.rs` already applied.

**Tests:** in-file — single-thread (`n = 1`) immediate-leader case, and a
real-multi-thread test (10 threads, barrier reused across two generations,
confirms exactly one leader per round and that the generation counter
actually unblocks the next round).

### `hash.rs`

**What it does:** `FxHasher`, `FixedHasher` (a deterministic `BuildHasher`
for it), and `fixed_hash_one`. No `std`/`no_std` split — pure `core`
integer arithmetic, one code path in every configuration.

**Decisions:**
- Independent copy of `mid-collections`'s `FxHasher`, not a dependency —
  see "Why a second, independent `FxHasher`" above.
- **No `RandomState`/HashDoS-resistant equivalent**, unlike
  `bevy_platform::hash`. Two reasons: FxHash's diffusion is weak by design,
  so randomizing the seed only defeats a *precomputed* attack, not an
  attacker who can observe outputs or timing — a false sense of security
  for an algorithm that never tried to be DoS-resistant; and getting real
  OS entropy with zero dependencies, under a `no_std` fallback that may
  have no OS to ask, is a separate hard problem not worth solving for it.
  A caller who needs HashDoS resistance should use `std`'s own
  `RandomState`/SipHash. This hasher is for trusted engine-internal keys
  (spatial-hash cells, entity ids), the same category `mid-collections`'s
  copy already targets.
- `FixedHasher` starts from `0x9E37_79B9_7F4A_7C15` (the 64-bit golden-ratio
  constant) rather than zero — a well-known odd constant, not a value the
  implementation depends on for any particular property.
- Matches `rustc-hash` 1.1.0 (read from its crates.io source) bit-for-bit
  on 64-bit targets from a zero seed, except for negative `i8`/`i16`/`i32`:
  `rustc-hash` zero-extends those, this sign-extends them (inherited from
  `mid-collections`). Unlike `rustc-hash`, this uses a `u64` state on every
  target, so its output is identical on 32-bit and 64-bit (wasm32 vs
  native) — `rustc-hash`'s isn't.
- `bevy_platform`'s `Hashed<V, S>`, `PassHash`/`NoOpHash` are not built —
  not needed by anything yet, trigger-based like everything else deferred
  here.

**Tests:** in-file — determinism, seed sensitivity, a real parity test
against the published `rustc-hash` 1.1.0 crate (a dev-dependency; covers
every `write` tail length 0–15 bytes plus each integer width), and a test
pinning the one known divergence (negative narrow signed ints).

### `ffi.rs`

**What it does:** C-compatible FFI exports. `std`-only. `Once`, `Barrier`,
and the fixed hasher — see "Fixes and Problems" below for the full
reasoning on scope and what's still open.

**Decisions:**
- Every convention (status enum shape, `ffi_guard`/`catch_unwind`, opaque
  `Box::into_raw`/`Box::from_raw` handles, null-safe defaults, `# Safety`
  doc sections on every pointer-taking fn) copied directly from
  `mid-ecs`'s/`mid-net`'s own real `ffi.rs` files, not reinvented.
- `Once`/`Barrier` picked for this first pass specifically because
  neither hands back a borrowed guard/reference a C caller would need to
  explicitly, correctly release — the harder problem `Mutex`/`RwLock`/
  `OnceLock`/`LazyLock`'s FFI surface all share, deliberately not
  attempted in this same pass.
- `Once`'s C surface takes a `void (*f)(void *ctx)` callback rather than
  trying to expose anything about the closure it wraps on the Rust side
  — the only way a C ABI can express "run this once," matching how
  `register_ffi_*`'s own genericity forced `mid-ecs`'s `ffi.rs` to leave
  those functions uncallable from pure C either.

**Tests:** in-file `#[cfg(test)]` — lifecycle, NULL-handling, and a real
multi-thread test for `Barrier` (four real OS threads via
`std::thread::spawn`, addresses passed as `usize` since raw pointers
aren't `Send`). Plus `ffi-smoke-test/test.c` — a real, separately
compiled C program, run against the actual compiled
`libmid_platform.so`/`.a`, whose own `Barrier` check spawns four real
`pthread`s (not just single-threaded C calls) via
`.github/workflows/mid-platform-test.yml`.

## Benchmarks

`benches/sync_bench.rs` — `sync::Mutex` and `sync::RwLock` against `spin`
(the real crate their `no_std` fallback algorithms were grounded in) and
`std::sync`'s own equivalents. Single-threaded uncontended `lock`/
`try_lock`/`read`/`write` throughput only — real multi-threaded correctness
is the test suite's job (see "Modules" above), not this bench's; see the
bench file's own doc comment for the full reasoning, including why
`cell::{SyncCell, SyncUnsafeCell}` and Once/OnceLock/LazyLock/Barrier aren't
separately benched. Comparison crates (`spin`) are `[dev-dependencies]`
only, never promoted to a real dependency — same pattern
`mid-arena`/`mid-alloc`'s own comparison benches already use.

Meant to run twice, matching `mid-platform-test.yml`'s own two-configuration
pattern: default features (std passthrough — mostly a sanity check that the
passthrough really is zero-cost) and `--no-default-features` (this crate's
own hand-rolled fallback — the run that actually answers something new).

**Real first-run results (Build #1, `rustc 1.98.1`, commit `25956a8f`):**
confirms exactly what this bench exists to check. Default features:
`mid-platform::Mutex`/`RwLock` track `std::sync`'s own numbers almost
exactly (43.98µs vs 43.98µs mutex lock/unlock; 48.27µs vs 48.59µs rwlock
read) — the passthrough really is zero-cost, not just zero-cost in theory.
`--no-default-features`: `mid-platform::Mutex`'s hand-rolled fallback
(21.78µs) lands within noise of real `spin::Mutex` (21.77µs) — no
performance left on the table. `RwLock`'s fallback (41.2µs read, 41.2µs
write) also tracks `spin::RwLock` closely (40.4µs / 41.5µs) and beats
`std::sync::RwLock`'s real OS-backed numbers (48.4µs / 47.3µs, unchanged
between runs since that baseline doesn't depend on this crate's own
features) — the simplified no-upgradeable-guard design isn't paying for
the machinery it left out. Full numbers in that run's own uploaded
artifact, not reproduced here — this paragraph is a summary, not the
record of truth.

Historical numbers age fast (a future change to either implementation
makes this paragraph stale) — treat this as "confirmed once, on Build #1,"
not a promise every future run reproduces exactly these figures.

`benches/hash_bench.rs` (Phase 3) — `FixedHasher` against `rustc-hash`
1.1.0 and std's default SipHash `RandomState`: raw hashing for `u64`, short
string, and 256-byte keys, plus each hasher as the `BuildHasher` of a real
`std::collections::HashMap` (insert + lookup). Runs once, under default
features — no `std`/`no_std` split to sweep. Since parity with `rustc-hash`
is already asserted bit-for-bit by `hash.rs`'s own tests, any real gap
between those two rows is codegen, not algorithm. **Not yet run** — no
numbers exist for this bench yet; same honesty as `sync_bench.rs` had
before its own first dispatch.

## CI and Workflows

- `.github/workflows/mid-platform-bench.yml` — `benches/sync_bench.rs`,
  run twice (default features, then `--no-default-features`), plus
  `benches/hash_bench.rs` once (default features only — no feature split
  for `hash.rs`), bash-grep summary pattern (see
  `docs/benching-standards.md`). `workflow_dispatch` only. Adding `criterion` as a dev-dependency gives this crate the same
  edition2024-via-`clap_builder` MSRV wall `mid-collections`'s and
  `mid-arena`'s own bench dev-dependencies already carry — see
  `docs/workspace-cargo.md`, "MSRV / toolchain walls" — so `cargo test -p
  mid-platform` now needs the newer toolchain too, not just `--bench`.
- `.github/workflows/mid-platform-test.yml` — build, clippy, fmt check, unit
  + doc-tests, run twice (default `std` features, then
  `--no-default-features` for the actual hand-rolled no_std fallback path).
  `workflow_dispatch` only, with a real parsed `$GITHUB_STEP_SUMMARY` (one
  combined table covering both feature configurations) — matches
  `mid-ecs-test.yml`'s established pattern; see
  `docs/RUST_AND_CRATE_GUIDELINES.md` §7. This workflow's first version
  wrongly ran on push/pull_request with a log-echo summary instead — fixed
  before it ever shipped with that mistake live, once `mid-ptr-test.yml`'s
  own identical mistake was caught and corrected.
  **Not replicated**: the HTML-report-plus-`gh-pages`-deploy half of
  `mid-ecs-test.yml`'s pattern — same reasoning as `mid-ptr-test.yml`, the
  whole `gh-pages` pipeline is being migrated to Cloudflare Pages.
  Now also runs `ffi-smoke-test/test.c` (real gcc, real link against the
  `libmid_platform.so` the default-features build step produces) after
  both Rust test runs — same "real C program, not just Rust tests"
  pattern `mid-ecs-test.yml` already uses, `continue-on-error: true`
  matching that same precedent. See "Fixes and Problems" below for what
  it covers.

## Fixes and Problems

### FFI — partially closed, started deliberately narrow

Started this pass rather than deferred further — the root `README.md`'s
own Design Mandates state "every crate exposes a strict `#[repr(C)]` FFI
boundary," and letting that stay unaddressed indefinitely is exactly what
happened with `mid-ptr` (see `docs/mid-ptr.md`'s own Fixes and Problems);
not repeating that here.

`src/ffi.rs` (`std`-only — a linkable `cdylib` needs a real OS underneath
it regardless) covers `Once`, `Barrier`, and the fixed hasher:
`mid_platform_once_{new,free,call,is_completed}`,
`mid_platform_barrier_{new,free,wait}`,
`mid_platform_fixed_hash_{bytes,u64}` (added with Phase 3's hasher, per the
"build FFI as each piece lands" rule — a pure function of its input, so no
guard/handle problem at all). Real, hand-written C header and a
real C program compiled with gcc and run against the actual compiled
`libmid_platform.so` — `ffi-smoke-test/mid_platform.h`,
`ffi-smoke-test/test.c` — wired into `mid-platform-test.yml`, same pattern
`mid-ecs`'s/`mid-net`'s own smoke tests already use. `test.c`'s barrier
check spawns four real `pthread`s, not just single-threaded calls, so the
smoke test exercises `Barrier` under genuine multi-thread contention
through the actual C ABI, not only Rust-side.

**Real mistake, caught and fixed (Build #3):** `Cargo.toml`'s `[lib]`
originally declared `crate-type = ["lib", "cdylib", "staticlib"]`
unconditionally, matching `mid-ecs`'s/`mid-net`'s own `[lib]` shape — this
broke `cargo build -p mid-platform --no-default-features` in real CI: a
`cdylib`/`staticlib` is a final linked artifact needing a `#[panic_handler]`
and unwind support, neither of which exist (or should exist) in the
no_std fallback path, and Cargo has no way to make `crate-type`
conditional on a feature within one manifest. `mid-ecs`/`mid-net` never
hit this because neither is no_std-capable in the first place —
mid-platform is the first crate in this workspace to need cdylib output
*and* have a real no_std configuration at the same time. Fixed by keeping
`crate-type = ["lib"]` (works for every configuration, matching this
crate's pre-FFI state) and building the cdylib the smoke test actually
needs on demand, only under default features, via `cargo rustc -p
mid-platform --lib --crate-type cdylib` in
`mid-platform-test.yml`'s own FFI smoke test step, instead of declaring it
in the manifest at all.

**Still open, deliberately not attempted in this same pass:** `Mutex`,
`RwLock`, `OnceLock`, `LazyLock`. Every one of them hands back a live,
borrowed guard/reference into memory this crate owns — a C caller would
need an opaque guard handle explicitly released with a paired unlock call
(C has no `Drop`), plus a real hazard if the underlying handle is freed
while a guard is still outstanding. `Once`/`Barrier` were picked for this
first pass specifically because neither has that problem at all
(`Barrier::wait` returns a plain `bool`; `Once::call_once` takes a
callback and returns nothing) — the same "narrowest real slice first,
name the harder piece and defer it explicitly" call `mid-ecs`'s own
`ffi.rs` made for component-data access in its first pass, not a new
principle invented here.
