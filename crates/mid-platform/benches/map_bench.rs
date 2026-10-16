// crates/mid-platform/benches/map_bench.rs
//! `collections::HashMap` against what a caller would otherwise reach for,
//! and `OpenHashMap` (the no_std fallback table) against `std`'s map.
//!
//! Three variants in every group, all `u64 -> u64`:
//! - `std RandomState`: `std::collections::HashMap` with its default SipHash
//!   hasher. What a caller gets with no help from this crate.
//! - `collections::HashMap (std + FixedHasher)`: this crate's alias under
//!   default features. Isolates what swapping the hasher is worth.
//! - `OpenHashMap (FixedHasher)`: the safe-code table that stands in for the
//!   alias when `std` is off. Same hasher as the row above, so the gap
//!   between the two rows is the table, not the hash function.
//!
//! Two key shapes, because they stress different things:
//! - `seq`: `0..N`. Friendly to a multiply hash; the case a fold/multiply
//!   hasher is built for.
//! - `aligned`: `i << 12`. Every key shares its low 12 bits. A fold/multiply
//!   hash ends in a multiplication, so its *low* output bits depend only on
//!   the low input bits, and a table that indexes by low hash bits sees
//!   these keys collide. `OpenHashMap` indexes by the *top* bits after one
//!   more multiply; whether `std`'s table copes is what this shape measures.
//!
//! Groups, per key shape: `insert_grow` (build from empty, includes every
//! rehash), `insert_presized`, `get_hit`, `get_miss`, `remove_all`
//! (clone excluded from the timing), `iterate`.
//!
//! Default features only: the comparison needs `std`. `OpenHashMap`'s code is
//! identical with `std` off, so there is no second configuration to run.
//!
//! Not measured: string keys, large values, mixed workloads, memory use.
//! Whatever shows up here is the first data, not a verdict.
//!
//! Run
//! ---
//!   cargo bench -p mid-platform --bench map_bench

use std::collections::HashMap as StdMap;
use std::time::Duration;

use criterion::{
    black_box, criterion_group, criterion_main, measurement::WallTime, BatchSize, BenchmarkGroup,
    Criterion,
};
use mid_platform::collections::{HashMap as AliasMap, OpenHashMap};
use mid_platform::hash::FixedHasher;

const N: u64 = 10_000;

type StdRandom = StdMap<u64, u64>;
type StdFixed = AliasMap<u64, u64>;
type Open = OpenHashMap<u64, u64>;

/// Just enough of a map's surface for one generic bench body.
trait Map: Clone {
    fn empty() -> Self;
    fn presized(entries: usize) -> Self;
    fn put(&mut self, key: u64, value: u64);
    fn fetch(&self, key: u64) -> Option<u64>;
    fn take(&mut self, key: u64) -> Option<u64>;
    fn sum_values(&self) -> u64;
}

impl Map for StdRandom {
    fn empty() -> Self {
        StdMap::new()
    }
    fn presized(entries: usize) -> Self {
        StdMap::with_capacity(entries)
    }
    fn put(&mut self, key: u64, value: u64) {
        self.insert(key, value);
    }
    fn fetch(&self, key: u64) -> Option<u64> {
        self.get(&key).copied()
    }
    fn take(&mut self, key: u64) -> Option<u64> {
        self.remove(&key)
    }
    fn sum_values(&self) -> u64 {
        self.values().fold(0, |acc, v| acc.wrapping_add(*v))
    }
}

impl Map for StdFixed {
    fn empty() -> Self {
        AliasMap::with_hasher(FixedHasher)
    }
    fn presized(entries: usize) -> Self {
        AliasMap::with_capacity_and_hasher(entries, FixedHasher)
    }
    fn put(&mut self, key: u64, value: u64) {
        self.insert(key, value);
    }
    fn fetch(&self, key: u64) -> Option<u64> {
        self.get(&key).copied()
    }
    fn take(&mut self, key: u64) -> Option<u64> {
        self.remove(&key)
    }
    fn sum_values(&self) -> u64 {
        self.values().fold(0, |acc, v| acc.wrapping_add(*v))
    }
}

impl Map for Open {
    fn empty() -> Self {
        OpenHashMap::with_hasher(FixedHasher)
    }
    fn presized(entries: usize) -> Self {
        OpenHashMap::with_capacity_and_hasher(entries, FixedHasher)
    }
    fn put(&mut self, key: u64, value: u64) {
        self.insert(key, value);
    }
    fn fetch(&self, key: u64) -> Option<u64> {
        self.get(&key).copied()
    }
    fn take(&mut self, key: u64) -> Option<u64> {
        self.remove(&key)
    }
    fn sum_values(&self) -> u64 {
        self.values().fold(0, |acc, v| acc.wrapping_add(*v))
    }
}

const STD_RANDOM: &str = "std RandomState";
const STD_FIXED: &str = "collections::HashMap (std + FixedHasher)";
const OPEN: &str = "OpenHashMap (FixedHasher)";

fn keys_seq() -> Vec<u64> {
    (0..N).collect()
}

fn keys_aligned() -> Vec<u64> {
    (0..N).map(|i| i << 12).collect()
}

/// Keys guaranteed absent from the sets above: odd, so never `i << 12`, and
/// beyond `N` so never `0..N`.
fn miss_keys(shape: &[u64]) -> Vec<u64> {
    shape.iter().map(|k| (k << 1) | 1).map(|k| k + N).collect()
}

fn filled<M: Map>(keys: &[u64]) -> M {
    let mut map = M::presized(keys.len());
    for (i, k) in keys.iter().enumerate() {
        map.put(*k, i as u64);
    }
    map
}

fn insert_grow<M: Map>(g: &mut BenchmarkGroup<'_, WallTime>, name: &str, keys: &[u64]) {
    g.bench_function(name, |b| {
        b.iter(|| {
            let mut map = M::empty();
            for (i, k) in keys.iter().enumerate() {
                map.put(black_box(*k), i as u64);
            }
            black_box(map)
        })
    });
}

fn insert_presized<M: Map>(g: &mut BenchmarkGroup<'_, WallTime>, name: &str, keys: &[u64]) {
    g.bench_function(name, |b| {
        b.iter(|| {
            let mut map = M::presized(keys.len());
            for (i, k) in keys.iter().enumerate() {
                map.put(black_box(*k), i as u64);
            }
            black_box(map)
        })
    });
}

fn lookups<M: Map>(
    g: &mut BenchmarkGroup<'_, WallTime>,
    name: &str,
    built_from: &[u64],
    probe: &[u64],
) {
    let map: M = filled(built_from);
    g.bench_function(name, |b| {
        b.iter(|| {
            let mut acc = 0u64;
            for k in probe {
                if let Some(v) = map.fetch(black_box(*k)) {
                    acc = acc.wrapping_add(v);
                }
            }
            black_box(acc)
        })
    });
}

fn remove_all<M: Map>(g: &mut BenchmarkGroup<'_, WallTime>, name: &str, keys: &[u64]) {
    let template: M = filled(keys);
    g.bench_function(name, |b| {
        b.iter_batched(
            || template.clone(),
            |mut map| {
                let mut acc = 0u64;
                for k in keys {
                    if let Some(v) = map.take(black_box(*k)) {
                        acc = acc.wrapping_add(v);
                    }
                }
                black_box((map, acc))
            },
            BatchSize::SmallInput,
        )
    });
}

fn iterate<M: Map>(g: &mut BenchmarkGroup<'_, WallTime>, name: &str, keys: &[u64]) {
    let map: M = filled(keys);
    g.bench_function(name, |b| b.iter(|| black_box(map.sum_values())));
}

fn bench_shape(c: &mut Criterion, shape: &str, keys: &[u64]) {
    let misses = miss_keys(keys);

    let mut g = c.benchmark_group(format!("insert_grow_{shape}_u64"));
    insert_grow::<StdRandom>(&mut g, STD_RANDOM, keys);
    insert_grow::<StdFixed>(&mut g, STD_FIXED, keys);
    insert_grow::<Open>(&mut g, OPEN, keys);
    g.finish();

    let mut g = c.benchmark_group(format!("insert_presized_{shape}_u64"));
    insert_presized::<StdRandom>(&mut g, STD_RANDOM, keys);
    insert_presized::<StdFixed>(&mut g, STD_FIXED, keys);
    insert_presized::<Open>(&mut g, OPEN, keys);
    g.finish();

    let mut g = c.benchmark_group(format!("get_hit_{shape}_u64"));
    lookups::<StdRandom>(&mut g, STD_RANDOM, keys, keys);
    lookups::<StdFixed>(&mut g, STD_FIXED, keys, keys);
    lookups::<Open>(&mut g, OPEN, keys, keys);
    g.finish();

    let mut g = c.benchmark_group(format!("get_miss_{shape}_u64"));
    lookups::<StdRandom>(&mut g, STD_RANDOM, keys, &misses);
    lookups::<StdFixed>(&mut g, STD_FIXED, keys, &misses);
    lookups::<Open>(&mut g, OPEN, keys, &misses);
    g.finish();

    let mut g = c.benchmark_group(format!("remove_all_{shape}_u64"));
    remove_all::<StdRandom>(&mut g, STD_RANDOM, keys);
    remove_all::<StdFixed>(&mut g, STD_FIXED, keys);
    remove_all::<Open>(&mut g, OPEN, keys);
    g.finish();

    let mut g = c.benchmark_group(format!("iterate_{shape}_u64"));
    iterate::<StdRandom>(&mut g, STD_RANDOM, keys);
    iterate::<StdFixed>(&mut g, STD_FIXED, keys);
    iterate::<Open>(&mut g, OPEN, keys);
    g.finish();
}

fn bench_maps(c: &mut Criterion) {
    bench_shape(c, "seq", &keys_seq());
    bench_shape(c, "aligned", &keys_aligned());
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(2))
        .sample_size(50);
    targets = bench_maps
}
criterion_main!(benches);
