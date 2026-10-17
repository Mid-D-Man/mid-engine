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
least one test. All three feature configurations pass on CI.

**One thing that was wrong, and the fix:** `FixedHasher` used to end in a plain
multiplication, so its low output bits depended only on the key's low bits.
CI showed that keys sharing their low bits (`i << 12`, aligned addresses) made
the `std` alias 8x to 17x slower than `std`'s own default map on insert,
lookup, and remove, and 18x to 49x slower than `OpenHashMap`. `FixedHasher`
now ends in a folded multiply (see the [`hash`](hash.md) page), so the alias
is expected to behave on those keys. That is a sandbox result until the next
CI bench run measures it, on sequential, `i << 12`, and `i << 32` keys.
`OpenHashMap` was never exposed to it: it indexes by the top bits after one
more multiply. Details and the pre-fix figures: `docs/mid-platform.md`,
"collections/" and "Benchmarks."
