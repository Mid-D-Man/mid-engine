# mid-platform

Platform-agnostic primitives, `bevy_platform`-shaped but **hand-rolled and
dependency-free** — unlike `mid-ptr`, this is not a verbatim port. Real
`bevy_platform` pulls in `spin`, `portable-atomic`, `foldhash`, `hashbrown`,
and `critical-section`; none of mid-engine's real targets (native desktop,
`wasm32-unknown-unknown`) actually need what most of those exist for.

## Phase 1 (built)

- **`sync::atomic`** — a plain re-export of `core::sync::atomic`. Upstream's
  `portable-atomic` fallback only activates on targets lacking a native
  atomic width; none of this project's targets do.
- **`cell::{SyncCell, SyncUnsafeCell}`** — pure logic, no locking involved.
- **`sync::{Mutex, MutexGuard}`** — a thin `std::sync::Mutex` pass-through
  when the `std` feature (on by default) is enabled, and a self-contained
  spin-based fallback otherwise — the same algorithm `mid-alloc::SpinLock`
  already validated under real multi-threaded stress tests, kept as an
  independent copy rather than a dependency so this crate stays
  workspace-dependency-free.
- **`sync::{Arc, Weak}`** — a plain `alloc::sync` re-export.

## Phase 2 (not built yet)

`RwLock`, `Once`/`OnceLock`, `LazyLock`, and `Barrier` — each needs genuinely
new spin-based concurrency design, not just a port of already-proven logic,
so each deserves its own careful pass rather than being rushed in
alongside Phase 1.

## Phase 3 (not decided)

A fast hasher (hand-rollable, deserves a benchmarked pass of its own) and a
`HashMap`/`HashSet` (the hard one — reimplementing a real hash table is a
project-sized decision on its own, not resolved yet).

## Testing both configurations

This crate has real Cargo features (`std`, default-on; `alloc`), not just
an unconditional `no_std`. CI runs it twice: default features exercise the
`std` passthrough, `--no-default-features` exercises the actual hand-rolled
fallback — a genuinely different code path default features would never
even compile.

## FFI

Not yet exposed over a C boundary — same tracked gap as `mid-ptr`.

## Status

See the [Tests](/tests/) page, or run `mid-platform — Tests` from the
Actions tab yourself.
