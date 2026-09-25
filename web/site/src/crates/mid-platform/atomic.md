# sync::atomic

A portable set of atomic types — the same ones every target this crate
actually supports (x86_64, aarch64, wasm32) already has natively.

## What it does

A plain re-export of `core::sync::atomic`'s `Atomic*` types and `Ordering`
— `AtomicBool`, `AtomicI8`..`AtomicI64`, `AtomicIsize`, `AtomicPtr`,
`AtomicU8`..`AtomicU64`, `AtomicUsize`. Upstream `bevy_platform` falls back
to the `portable-atomic` crate on targets missing a native atomic width;
none of mid-engine's real targets are missing one, so this module carries
no fallback dependency at all. If this crate is ever built for a target
that genuinely lacks full-width atomics, a `compile_error!` fails the build
loudly and explains why, rather than silently miscompiling.

## Example usage

```rust
use mid_platform::sync::atomic::{AtomicU32, Ordering};

let counter = AtomicU32::new(0);
counter.fetch_add(1, Ordering::SeqCst);
assert_eq!(counter.load(Ordering::SeqCst), 1);
```

## Status

Done — Phase 1. See `docs/mid-platform.md`, "Build order."
