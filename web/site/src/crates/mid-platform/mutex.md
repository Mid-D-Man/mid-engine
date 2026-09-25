# sync::Mutex

A `std`-shaped mutual-exclusion lock that also works with `std` off.

## What it does

`std::sync::Mutex` directly when the `std` feature is on. Otherwise a
hand-rolled spin-based fallback: "test, then test-and-set" — attempt the
real atomic compare-exchange first, and only on contention fall back to
spinning on a plain `Relaxed` load, which is cheaper than repeatedly
retrying the exclusive compare-exchange while the lock is genuinely held
elsewhere. This is an independent copy of the exact algorithm
`mid_alloc::sync::SpinLock` already validated under real multi-threaded
stress tests, not a dependency on that crate. Never actually poisons — a
panic while holding the guard just unwinds past `Drop` and releases the
lock normally; `is_poisoned()`/`clear_poison()` exist only to keep this
type a drop-in replacement for `std::sync::Mutex` call sites.

## Example usage

```rust
use mid_platform::sync::Mutex;

let counter = Mutex::new(0u32);
*counter.lock().unwrap() += 1;
assert_eq!(*counter.lock().unwrap(), 1);
```

## Status

Done — Phase 1. Full design write-up: `docs/mid-platform.md`, "Why not
just depend on mid-alloc for SpinLock" and "Build order" (a Phase 1 file —
predates this doc's per-module `### sync/<file>.rs` sections, which start
with Phase 2 below).

