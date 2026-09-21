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

**Phase 2 (not started):** real new concurrency-primitive design work each
one deserves its own careful pass, not a rushed inclusion here:
- `sync::rwlock::RwLock` — needs a hand-rolled spin-based reader/writer lock (reader count + writer flag, fairness questions a plain mutex doesn't have)
- `sync::once::{Once, OnceLock}` — needs a hand-rolled spin-based once-cell (a 3-state atomic state machine; can build on `sync::mutex` rather than needing its own primitive, but still new design)
- `sync::lazy_lock::LazyLock` — builds directly on `OnceLock` once that exists
- `sync::barrier::Barrier` — lowest priority of this group, rarely used
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

## Fixes and Problems

*(none yet — Phase 1 is the initial build)*
