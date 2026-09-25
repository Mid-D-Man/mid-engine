# sync::RwLock

A `std`-shaped reader/writer lock — any number of concurrent readers, or
one exclusive writer, never both at once.

## What it does

`std::sync::RwLock` directly when the `std` feature is on. Otherwise a
hand-rolled spin-based fallback grounded in `spin::RwLock`'s real algorithm
(read fresh from its own source, not from memory): a single packed atomic
holds both the writer flag and the reader count, so `read()` and
`try_read()` never block each other, while `write()`/`try_write()` only
succeed when the lock is fully free. Simplified from the real `spin`
algorithm by leaving out its upgradeable-guard mechanism — this type's
surface is `read`/`write`/`try_read`/`try_write` only. Like `spin::RwLock`
itself, this is unfair to writers under continuous read pressure; there's
no fairness mechanism to prevent a steady stream of readers from starving
a waiting writer.

## Example usage

```rust
use mid_platform::sync::RwLock;

let data = RwLock::new(vec![1, 2, 3]);

// Many readers can hold the lock at once.
{
    let r1 = data.read().unwrap();
    let r2 = data.read().unwrap();
    assert_eq!(r1.len(), r2.len());
}

// Only one writer, and only once every reader has released.
data.write().unwrap().push(4);
assert_eq!(data.read().unwrap().len(), 4);
```

## Status

Done — Phase 2. Full design write-up, including exactly why the write
guard's unlock can't be a blind `store(0, ..)`: `docs/mid-platform.md`,
"sync/rwlock.rs."
