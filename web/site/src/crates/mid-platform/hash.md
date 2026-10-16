# hash

A fast, non-cryptographic hasher and a deterministic `BuildHasher` for it —
for `HashMap`/`HashSet` keys you control, not keys an attacker chooses.

## What it does

`FxHasher` is the fold/multiply hash rustc itself uses for its own compiler
data structures: rotate, xor in the next word, multiply by a fixed constant,
eight bytes at a time. It is fast because it does very little per word, and
weak for the same reason. `FixedHasher` is a `BuildHasher` that hands out
`FxHasher`s from the same fixed starting state every time, so the same key
hashes to the same value on every run and every platform (the state is
`u64` on every target, including wasm32). `fixed_hash_one` hashes a single
value with it.

One property to know about: the output ends in a multiplication, so its low
bits depend only on the low bits of the key. A table that picks buckets from
the low bits of the hash can collide heavily on keys that share them
(`i << 12`, aligned addresses). See the note on the
[`collections`](collections.md) page.

There is no `RandomState` equivalent, on purpose. A fold/multiply hash can't
be made DoS-resistant by randomizing its seed, so a randomized version would
only look safer than it is. Anything that hashes untrusted input should use
`std`'s own `RandomState`.

## Example usage

```rust
use std::collections::HashMap;
use mid_platform::hash::{fixed_hash_one, FixedHasher};

// As the hasher of a HashMap keyed by data the engine generates itself.
let mut cells: HashMap<(i32, i32), u32, FixedHasher> =
    HashMap::with_hasher(FixedHasher);
cells.insert((3, -4), 7);
assert_eq!(cells[&(3, -4)], 7);

// Hashing one value directly. Same result every run.
assert_eq!(fixed_hash_one("mid-engine"), fixed_hash_one("mid-engine"));
```

## Status

Done — Phase 3, hasher half. Matches the published `rustc-hash` 1.1.0 on
64-bit targets, apart from negative `i8`/`i16`/`i32` values. Benchmarks
written, first run pending. The maps that use it are on the
[`collections`](collections.md) page. Callable from C: `mid_platform_fixed_hash_{bytes,u64}`. Full design write-up, including why this is a second copy of
`mid-collections`'s hasher rather than a dependency on it:
`docs/mid-platform.md`, "hash.rs."
