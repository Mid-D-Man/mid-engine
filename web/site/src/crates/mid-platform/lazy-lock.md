# sync::LazyLock

A value that computes itself from a closure the first time it's actually
used, safely across any number of racing threads.

## What it does

`std::sync::LazyLock` directly when the `std` feature is on. Otherwise a
fallback built directly on top of this crate's own `OnceLock`, plus
`cell::SyncUnsafeCell` to hold the initializer closure until it's consumed.
No new unsafe synchronization exists in this type at all — every guarantee
it needs comes straight from `OnceLock::get_or_init`, which already
guarantees the initializer runs on at most one thread, at most once.
Dereferencing a `LazyLock` triggers initialization on first access and
returns the cached value on every access after that.

## Example usage

```rust
use mid_platform::sync::LazyLock;

static CONFIG: LazyLock<Vec<u32>> = LazyLock::new(|| {
    // Runs once, on whichever thread first dereferences CONFIG.
    (0..10).collect()
});

assert_eq!(CONFIG.len(), 10);
assert_eq!(CONFIG[0], 0);
```

## Status

Done — Phase 2. Full design write-up, including why `std::sync::LazyLock`'s
real union-based layout wasn't ported: `docs/mid-platform.md`,
"sync/lazy_lock.rs."
