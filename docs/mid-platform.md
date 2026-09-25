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
| `foldhash` | The actual hash algorithm behind `DefaultHasher`/`RandomState`/`FixedState` in `hash.rs` | Yes, if we want a fast general-purpose hasher at all | **Hand-roll**, deferred to its own phase — a fold/multiply-based hash is well-trodden ground (FxHash-style algorithms are ~30 lines), but it deserves its own benchmarked pass, not a rushed one bolted onto this crate's first slice. |
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

**Phase 3 (separate decision, not scoped here):** `hash.rs`'s fast-hasher
algorithm (deserves a benchmarked pass of its own) and the `collections::hash_map`/
`hash_set` question (depend on `hashbrown` directly, since reimplementing it is a
project-sized undertaking on its own — or accept a slower, simpler hand-rolled
map and measure whether it actually matters for mid-engine's real workloads).
Not decided in this doc.

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

**Not yet run** — `.github/workflows/mid-platform-bench.yml`'s first real
dispatch is what actually produces numbers; nothing here is a real result
yet, same honesty `bench-mid-alloc.yml`'s own header keeps for itself until
its own first run.

## CI and Workflows

- `.github/workflows/mid-platform-bench.yml` — `benches/sync_bench.rs`,
  run twice (default features, then `--no-default-features`), bash-grep
  summary pattern (see `docs/benching-standards.md`). `workflow_dispatch`
  only. Adding `criterion` as a dev-dependency gives this crate the same
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

## Fixes and Problems

### FFI — open gap, not yet fixed
- The root `README.md`'s own Design Mandates state "every crate exposes a
  strict `#[repr(C)]` FFI boundary." This crate currently has neither an FFI
  module nor a `cdylib`/`staticlib` `crate-type`. Same gap as `mid-ptr`
  (see `docs/mid-ptr.md`'s own Fixes and Problems) — flagged here rather
  than left silently missing, not fixed in the pass that built Phase 1.
