# mid-platform

No-`std`-safe replacements for the pieces of `std::sync`/`std::cell` mid-engine
actually needs, `std`-backed on native targets and hand-rolled (spin-based,
no OS to ask for a real wait) wherever `std` is off. Zero external
dependencies — every fallback in this crate is a from-scratch port of a real,
proven algorithm rather than a `spin` dependency, so mid-engine's real target
matrix (native desktop, `wasm32-unknown-unknown`) gets the behavior without
the crate.

**Modules:**

- [`cell`](mid-platform/cell.md) — `SyncCell`, `SyncUnsafeCell`
- [`sync::atomic`](mid-platform/atomic.md) — plain re-export of `core::sync::atomic`
- [`sync::Mutex`](mid-platform/mutex.md) — `MutexGuard`
- [`sync::RwLock`](mid-platform/rwlock.md) — `RwLockReadGuard`, `RwLockWriteGuard`
- [`sync::Once` / `OnceLock`](mid-platform/once.md) — `OnceState`
- [`sync::LazyLock`](mid-platform/lazy-lock.md)
- [`sync::Barrier`](mid-platform/barrier.md) — `BarrierWaitResult`
- [`sync::Arc` / `Weak`](mid-platform/arc.md) — plain re-export of `alloc::sync`

**Status:** Phase 1 and Phase 2 both done (see `docs/mid-platform.md`). A
fast hasher and a `HashMap`/`HashSet` are a separate, later decision, not
started.
