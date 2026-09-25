# cell

`SyncCell<T>` and `SyncUnsafeCell<T>` — ways to make a type `Sync` without
locking, for the cases where locking would be the wrong tool.

## What it does

`SyncCell<T>` reimplements the currently-unstable `std::sync::Exclusive`:
it makes any `T` unconditionally `Sync` by only ever handing out `&mut T`
(through `get()`), never a shared `&T` unless `T` itself is already `Sync`
(`read()` requires `T: Sync`). Since `Sync` only allows multithreaded access
through a shared reference, and this type never allows unsynchronized
shared access to a non-`Sync` inner value, marking it `Sync` doesn't open
any real hole — mutable access is still exclusive by Rust's own borrow
rules.

`SyncUnsafeCell<T>` reimplements the currently-unstable
`std::cell::SyncUnsafeCell`: a plain `UnsafeCell<T>` that additionally
implements `Sync` whenever `T` does. `UnsafeCell` itself is never `Sync`,
specifically to stop accidental cross-thread misuse — this type opts back
in deliberately, for a caller who's going to provide their own
synchronization around the raw pointer it hands out (`get()`). It carries
none of the runtime overhead a lock would — no atomics, no spinning, just
the same interior-mutability primitive `UnsafeCell` already is.

## Example usage

```rust
use mid_platform::cell::{SyncCell, SyncUnsafeCell};

// SyncCell: exclusive access only, so it's Sync even for a !Sync T.
let mut counter = SyncCell::new(0u32);
*counter.get() += 1;

// SyncUnsafeCell: raw-pointer access, caller provides synchronization.
let cell = SyncUnsafeCell::new(0u32);
unsafe {
    *cell.get() += 1;
}
assert_eq!(unsafe { *cell.get() }, 1);
```

## Status

Done — Phase 1. See `docs/mid-platform.md`, "Build order."
