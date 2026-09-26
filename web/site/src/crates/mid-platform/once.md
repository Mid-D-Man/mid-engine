# sync::Once / OnceLock

Run something exactly once, or lazily initialize a value exactly once,
safely across any number of racing threads.

## What it does

`std::sync::{Once, OnceLock, OnceState}` directly when the `std` feature is
on. Otherwise a hand-rolled fallback built on top of this same crate's own
`sync::Mutex` rather than a from-scratch atomic state machine — the actual
mutual exclusion during initialization is just the crate's already-tested
`Mutex`. A separate atomic completion flag, checked before ever touching
the lock, keeps `OnceLock::get()` fully lock-free once a value exists, so
reading an already-initialized value never pays a locking cost. `Once` is
a thin wrapper around `OnceLock<()>` — "has this run yet" is exactly what
an initialized-or-not unit cell already answers. Neither type ever
poisons; `OnceState::is_poisoned()` always returns `false`.

## Example usage

```rust
use mid_platform::sync::{Once, OnceLock};

// Run initialization exactly once, however many times/threads call it.
static INIT: Once = Once::new();
INIT.call_once(|| {
    // one-time setup
});

// Lazily compute and cache a value exactly once.
static CONFIG: OnceLock<u32> = OnceLock::new();
let value = CONFIG.get_or_init(|| expensive_computation());
assert_eq!(*value, *CONFIG.get().unwrap());

fn expensive_computation() -> u32 { 42 }
```

## Status

Done — Phase 2. Full design write-up, including why the four-state
`spin::Once`-style atomic protocol was deliberately not ported:
`docs/mid-platform.md`, "sync/once.rs."

Also callable from C: `mid_platform_once_{new,free,call,is_completed}` —
see `docs/mid-platform.md`, "ffi.rs."
