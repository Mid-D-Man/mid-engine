# sync::Arc / Weak

Thread-safe reference counting — the same `Arc`/`Weak` every Rust
codebase already reaches for, made available with or without `std`.

## What it does

A plain re-export of `alloc::sync::{Arc, Weak}`, gated behind this crate's
`alloc` feature. Upstream `bevy_platform` needs `portable-atomic-util`
here only on targets lacking pointer-width native atomics; none of
mid-engine's real targets (x86_64, aarch64, wasm32) are missing that, so
this is zero-cost — no fallback implementation, no dependency, just the
real standard-library type.

## Example usage

```rust
use mid_platform::sync::Arc;

let shared = Arc::new(vec![1, 2, 3]);
let handle = Arc::clone(&shared);
assert_eq!(shared.len(), handle.len());
assert_eq!(Arc::strong_count(&shared), 2);
```

## Status

Done — Phase 1. Design write-up: `docs/mid-platform.md`'s dependency
table entry for `portable-atomic-util` (a Phase 1 file — predates this
doc's per-module `### sync/<file>.rs` sections, which start with Phase 2).
