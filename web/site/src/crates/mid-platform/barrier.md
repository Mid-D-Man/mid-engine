# sync::Barrier

Blocks a fixed number of threads until they've all reached the same point,
then releases them all together.

## What it does

`std::sync::{Barrier, BarrierWaitResult}` directly when the `std` feature
is on. Otherwise a fallback ported from `spin::Barrier`'s real algorithm
(read fresh from its own source) onto this crate's own `sync::Mutex`: a
thread count and a generation counter behind a plain lock, not a bespoke
atomic protocol. Every thread that calls `wait()` blocks until the last
thread arrives; that last thread resets the count and bumps the
generation, which wakes everyone else. A barrier is reusable across
generations — calling `wait()` on it again after everyone's been released
works the same way a second time.

## Example usage

```rust
use mid_platform::sync::Barrier;
use std::sync::Arc;
use std::thread;

let barrier = Arc::new(Barrier::new(4));
let handles: Vec<_> = (0..4)
    .map(|_| {
        let barrier = Arc::clone(&barrier);
        thread::spawn(move || {
            // ... each thread's own work ...
            barrier.wait(); // blocks until all 4 threads reach this line
            // ... all 4 threads continue together from here ...
        })
    })
    .collect();
for h in handles {
    h.join().unwrap();
}
```

## Status

Done — Phase 2, the lowest-priority primitive of this group (rarely used).
Full design write-up: `docs/mid-platform.md`, "sync/barrier.rs."
