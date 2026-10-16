# collections

`HashMap` and `HashSet` with the deterministic `FixedHasher` as the default
hasher, in two implementations behind one API: `std`'s own when `std` is on,
a safe-code table of this crate's when it is off.

## What it does

With the `std` feature (on by default), `collections::HashMap` and
`HashSet` are aliases for `std::collections`' types, with `FixedHasher`
filled in as the default hasher. `std`'s map is hashbrown underneath, so that
is hashbrown's speed with no dependency and no unsafe code written here.

Without `std` but with `alloc`, the same two names are aliases for
`OpenHashMap` and `OpenHashSet`: linear probing over a power-of-two table, up
to 3/4 full, removal by shifting the cluster back instead of leaving
tombstones, all in safe code. Both are always reachable under their own
names, so they can be tested and measured against `std`'s map.

Both implementations share the part of `std`'s map API they have in common,
with `std`'s signatures, so code written against `collections::HashMap`
builds in either configuration. Two consequences:

- There is no `new()`. `std` only provides it for `RandomState`. Use
  `HashMap::default()` (with the type known), `HashMap::with_hasher(FixedHasher)`,
  or `HashMap::with_capacity_and_hasher(n, FixedHasher)`.
- `FixedHasher` is deterministic and not DoS-resistant. Keys an attacker
  chooses belong in a `std` map with its own `RandomState`.

## Example usage

```rust
use mid_platform::collections::{HashMap, HashSet};
use mid_platform::hash::FixedHasher;

let mut counts: HashMap<&str, u32> = HashMap::with_hasher(FixedHasher);
for word in ["a", "b", "a"] {
    *counts.entry(word).or_insert(0) += 1;
}
assert_eq!(counts[&"a"], 2);

let seen: HashSet<u32> = (0..10).collect();
assert!(seen.contains(&3));
```

## Status

Done: Phase 3, map half. 35 tests plus 2 doctests, run in default features
and in the `alloc`-only configuration; 17 deliberate bugs each caught by at
least one test. Benchmarks written, first CI run pending.

One thing to watch: `FixedHasher`'s low output bits depend only on the low
bits of the key, and `std`'s map indexes by them, so keys that share their
low bits may collide heavily in the `std` alias. `OpenHashMap` is not exposed
to that. A short sandbox run pointed that way; the bench's `aligned` groups
will confirm or refute it on CI.

Not callable from C: a C ABI cannot carry a generic key and value type, and
no fixed-type shape has a user yet. Full design write-up:
`docs/mid-platform.md`, "collections/."
