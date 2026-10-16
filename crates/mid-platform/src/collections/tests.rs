// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-platform.md, section "collections/"
// ============================================================================
//! Tests for `collections`. Written against `alloc` only (no `std` types), so
//! the same file runs under both feature configurations.

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::format;
use alloc::rc::Rc;
use alloc::string::String;
use alloc::vec::Vec;
use core::cell::Cell;
use core::hash::{BuildHasher, Hasher};

use super::hash_map;
use super::open_map::{Entry, OpenHashMap};
use super::open_set::OpenHashSet;
use super::{HashMap, HashSet};
use crate::hash::FixedHasher;

// --- test hashers ----------------------------------------------------------

/// Every key hashes to the same value: one maximal cluster.
#[derive(Clone, Copy, Default)]
struct ConstState(u64);

struct ConstHasher(u64);

impl Hasher for ConstHasher {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, _bytes: &[u8]) {}
}

impl BuildHasher for ConstState {
    type Hasher = ConstHasher;
    fn build_hasher(&self) -> ConstHasher {
        ConstHasher(self.0)
    }
}

/// `hash(u64 key) == key`: all the entropy is in whichever bits the keys use.
#[derive(Clone, Copy, Default)]
struct IdentityState;

struct IdentityHasher(u64);

impl Hasher for IdentityHasher {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for byte in bytes {
            self.0 = (self.0 << 8) | u64::from(*byte);
        }
    }
    fn write_u64(&mut self, i: u64) {
        self.0 = i;
    }
}

impl BuildHasher for IdentityState {
    type Hasher = IdentityHasher;
    fn build_hasher(&self) -> IdentityHasher {
        IdentityHasher(0)
    }
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
}

// --- model test ------------------------------------------------------------

fn run_model<S: BuildHasher + Default>(seed: u64, key_space: u64, ops: usize) {
    let mut rng = Rng(seed);
    let mut map: OpenHashMap<u64, u64, S> = OpenHashMap::default();
    let mut model: BTreeMap<u64, u64> = BTreeMap::new();
    for step in 0..ops {
        let key = rng.next() % key_space;
        match rng.next() % 5 {
            0 | 1 => {
                let value = rng.next();
                assert_eq!(
                    map.insert(key, value),
                    model.insert(key, value),
                    "insert, step {step}"
                );
            }
            2 | 3 => {
                assert_eq!(map.remove(&key), model.remove(&key), "remove, step {step}");
            }
            _ => {
                assert_eq!(map.get(&key), model.get(&key), "get, step {step}");
            }
        }
        assert_eq!(map.len(), model.len(), "len, step {step}");
        if step % 97 == 0 {
            for (k, v) in &model {
                assert_eq!(map.get(k), Some(v), "reachable, step {step}");
            }
        }
    }
    for (k, v) in &model {
        assert_eq!(map.get(k), Some(v));
    }
    for absent in 0..key_space {
        if !model.contains_key(&absent) {
            assert!(!map.contains_key(&absent));
        }
    }
    let mut keys: Vec<u64> = map.keys().copied().collect();
    keys.sort_unstable();
    let model_keys: Vec<u64> = model.keys().copied().collect();
    assert_eq!(keys, model_keys);
}

#[test]
fn matches_btreemap_under_random_operations() {
    run_model::<FixedHasher>(0x1234_5678_9ABC_DEF1, 64, 20_000);
    run_model::<FixedHasher>(0x0FED_CBA9_8765_4321, 1024, 20_000);
    run_model::<FixedHasher>(0x5555_AAAA_5555_AAAB, 5, 5_000);
}

#[test]
fn matches_btreemap_when_every_key_collides() {
    for constant in [0, 1, 2, 3, 0x9E37_79B9_7F4A_7C15, u64::MAX, u64::MAX / 3] {
        let mut rng = Rng(constant ^ 0xDEAD_BEEF_0000_0001);
        let mut map: OpenHashMap<u64, u64, ConstState> =
            OpenHashMap::with_hasher(ConstState(constant));
        let mut model: BTreeMap<u64, u64> = BTreeMap::new();
        for step in 0..3_000 {
            let key = rng.next() % 40;
            if rng.next() % 2 == 0 {
                let value = rng.next();
                assert_eq!(
                    map.insert(key, value),
                    model.insert(key, value),
                    "step {step}"
                );
            } else {
                assert_eq!(map.remove(&key), model.remove(&key), "step {step}");
            }
            assert_eq!(map.len(), model.len());
        }
        for (k, v) in &model {
            assert_eq!(map.get(k), Some(v));
        }
    }
}

#[test]
fn matches_btreemap_with_an_identity_hasher() {
    run_model::<IdentityState>(0xCAFE_F00D_1234_5679, 300, 10_000);
}

#[test]
fn removal_from_one_cluster_leaves_every_other_entry_reachable() {
    // Six entries with one shared home fill a four-slot-load table (8 slots)
    // as a single run. For many homes (so the run wraps past the end of the
    // table for some), remove each member in turn.
    for constant in 0..48u64 {
        for victim in 0..6u64 {
            let mut map: OpenHashMap<u64, u64, ConstState> =
                OpenHashMap::with_hasher(ConstState(constant));
            for key in 0..6u64 {
                map.insert(key, key * 10);
            }
            assert_eq!(map.capacity(), 6);
            assert_eq!(map.remove(&victim), Some(victim * 10));
            assert!(!map.contains_key(&victim));
            for key in (0..6u64).filter(|k| *k != victim) {
                assert_eq!(
                    map.get(&key),
                    Some(&(key * 10)),
                    "home {constant}, victim {victim}"
                );
            }
            assert_eq!(map.len(), 5);
        }
    }
}

#[test]
fn identity_hasher_on_keys_with_equal_low_bits_still_spreads() {
    // Keys that differ only above bit 20. Indexing by the low bits of the
    // hash would put all of them in one cluster.
    let mut map: OpenHashMap<u64, u64, IdentityState> = OpenHashMap::default();
    for i in 0..1000u64 {
        map.insert(i << 20, i);
    }
    assert_eq!(map.len(), 1000);
    assert!(
        map.longest_cluster() < 64,
        "longest cluster {}",
        map.longest_cluster()
    );
    for i in 0..1000u64 {
        assert_eq!(map.get(&(i << 20)), Some(&i));
    }
}

// --- basics ----------------------------------------------------------------

#[test]
fn insert_get_remove() {
    let mut map: OpenHashMap<u32, &str> = OpenHashMap::default();
    assert!(map.is_empty());
    assert_eq!(map.get(&1), None);
    assert_eq!(map.remove(&1), None);
    assert_eq!(map.insert(1, "one"), None);
    assert_eq!(map.insert(2, "two"), None);
    assert_eq!(map.len(), 2);
    assert_eq!(map.get(&1), Some(&"one"));
    assert!(map.contains_key(&2));
    assert_eq!(map.get_key_value(&2), Some((&2, &"two")));
    *map.get_mut(&1).unwrap() = "uno";
    assert_eq!(map[&1], "uno");
    assert_eq!(map.remove(&1), Some("uno"));
    assert_eq!(map.remove_entry(&2), Some((2, "two")));
    assert!(map.is_empty());
}

#[test]
fn insert_replaces_the_value_and_keeps_len() {
    let mut map: OpenHashMap<u32, u32> = OpenHashMap::default();
    assert_eq!(map.insert(5, 1), None);
    assert_eq!(map.insert(5, 2), Some(1));
    assert_eq!(map.insert(5, 3), Some(2));
    assert_eq!(map.len(), 1);
    assert_eq!(map.get(&5), Some(&3));
}

#[test]
fn grows_from_empty_and_keeps_every_entry() {
    let mut map: OpenHashMap<u32, u32> = OpenHashMap::default();
    assert_eq!(map.capacity(), 0);
    for i in 0..5000u32 {
        map.insert(i, i.wrapping_mul(7));
        assert!(map.capacity() >= map.len());
    }
    assert_eq!(map.len(), 5000);
    for i in 0..5000u32 {
        assert_eq!(map.get(&i), Some(&i.wrapping_mul(7)));
    }
    assert_eq!(map.get(&5000), None);
}

#[test]
fn capacity_follows_the_three_quarters_rule() {
    let mut map: OpenHashMap<u32, u32> = OpenHashMap::default();
    map.insert(0, 0);
    assert_eq!(map.capacity(), 3);
    map.insert(1, 1);
    map.insert(2, 2);
    assert_eq!(map.capacity(), 3);
    map.insert(3, 3);
    assert_eq!(map.capacity(), 6);
}

#[test]
fn with_capacity_holds_that_many_without_growing() {
    for wanted in [0usize, 1, 2, 3, 4, 5, 6, 7, 100, 1000] {
        let mut map: OpenHashMap<usize, usize> =
            OpenHashMap::with_capacity_and_hasher(wanted, FixedHasher);
        let capacity = map.capacity();
        assert!(capacity >= wanted, "wanted {wanted}, got {capacity}");
        for i in 0..wanted {
            map.insert(i, i);
        }
        assert_eq!(map.capacity(), capacity, "grew while filling to {wanted}");
    }
}

#[test]
fn reserve_makes_room_for_additional_entries() {
    let mut map: OpenHashMap<u32, u32> = OpenHashMap::default();
    for i in 0..10 {
        map.insert(i, i);
    }
    map.reserve(500);
    let capacity = map.capacity();
    assert!(capacity >= 510);
    for i in 10..510 {
        map.insert(i, i);
    }
    assert_eq!(map.capacity(), capacity);
    for i in 0..510 {
        assert_eq!(map.get(&i), Some(&i));
    }
}

#[test]
fn clear_keeps_the_allocation_and_the_map_stays_usable() {
    let mut map: OpenHashMap<u32, u32> = OpenHashMap::default();
    for i in 0..100 {
        map.insert(i, i);
    }
    let capacity = map.capacity();
    map.clear();
    assert!(map.is_empty());
    assert_eq!(map.capacity(), capacity);
    assert_eq!(map.get(&5), None);
    assert_eq!(map.iter().count(), 0);
    for i in 0..100 {
        map.insert(i, i + 1);
    }
    assert_eq!(map.get(&5), Some(&6));
    assert_eq!(map.capacity(), capacity);
}

#[test]
fn borrowed_lookup_with_string_keys() {
    let mut map: OpenHashMap<String, u32> = OpenHashMap::default();
    map.insert(String::from("alpha"), 1);
    map.insert(String::from("beta"), 2);
    assert_eq!(map.get("alpha"), Some(&1));
    assert!(map.contains_key("beta"));
    assert_eq!(map.remove("alpha"), Some(1));
    assert_eq!(map.get("alpha"), None);
    assert_eq!(map["beta"], 2);
}

#[test]
#[should_panic(expected = "no entry found for key")]
fn indexing_a_missing_key_panics() {
    let map: OpenHashMap<u32, u32> = OpenHashMap::default();
    let _ = map[&1];
}

#[test]
fn same_inserts_iterate_in_the_same_order() {
    let build = || {
        let mut map: OpenHashMap<u32, u32> = OpenHashMap::default();
        for i in 0..200u32 {
            map.insert(i.wrapping_mul(2_654_435_761), i);
        }
        map.iter().map(|(k, v)| (*k, *v)).collect::<Vec<_>>()
    };
    assert_eq!(build(), build());
}

// --- clone, eq, debug, collect ---------------------------------------------

#[test]
fn clone_and_equality() {
    let a: OpenHashMap<u32, u32> = (0..50).map(|i| (i, i * 2)).collect();
    let b: OpenHashMap<u32, u32> = (0..50).rev().map(|i| (i, i * 2)).collect();
    assert!(a == b);
    let mut c = a.clone();
    assert!(a == c);
    c.insert(7, 0);
    assert!(a != c);
    c.insert(7, 14);
    assert!(a == c);
    c.remove(&7);
    assert!(a != c);
    c.insert(1000, 0);
    assert!(a != c, "same length, different keys");

    // `a` is a strict subset of `d`: every key of `a` is in `d` with the
    // same value, so only the length can tell them apart.
    let mut d = a.clone();
    d.insert(2000, 1);
    assert!(a != d);
    assert!(d != a);
}

#[test]
fn debug_formatting() {
    let mut map: OpenHashMap<u8, u8> = OpenHashMap::default();
    assert_eq!(format!("{map:?}"), "{}");
    map.insert(1, 2);
    assert_eq!(format!("{map:?}"), "{1: 2}");
}

#[test]
fn collect_and_extend() {
    let mut map: OpenHashMap<u32, u32> = (0..100).map(|i| (i, i)).collect();
    assert_eq!(map.len(), 100);
    map.extend((50..150).map(|i| (i, i + 1)));
    assert_eq!(map.len(), 150);
    assert_eq!(map[&10], 10);
    assert_eq!(map[&60], 61);
    assert_eq!(map[&149], 150);
}

// --- entry API -------------------------------------------------------------

#[test]
fn entry_or_insert_family() {
    let mut map: OpenHashMap<&str, u32> = OpenHashMap::default();
    for word in ["a", "b", "a", "c", "a", "b"] {
        *map.entry(word).or_insert(0) += 1;
    }
    assert_eq!(map.get(&"a"), Some(&3));
    assert_eq!(map.get(&"b"), Some(&2));
    assert_eq!(map.get(&"c"), Some(&1));

    map.entry("a").and_modify(|v| *v *= 10).or_insert(99);
    map.entry("z").and_modify(|v| *v *= 10).or_insert(99);
    assert_eq!(map.get(&"a"), Some(&30));
    assert_eq!(map.get(&"z"), Some(&99));

    assert_eq!(*map.entry("w").or_insert_with(|| 5), 5);
    assert_eq!(*map.entry("w").or_insert_with(|| 6), 5);
    assert_eq!(*map.entry("kk").or_insert_with_key(|k| k.len() as u32), 2);
    assert_eq!(*map.entry("n").or_default(), 0);
    assert_eq!(*map.entry("a").key(), "a");

    let mut lists: OpenHashMap<u8, Vec<u8>> = OpenHashMap::default();
    lists.entry(1).or_default().push(5);
    lists.entry(1).or_default().push(6);
    assert_eq!(lists[&1], [5, 6]);
}

#[test]
fn occupied_and_vacant_entries() {
    let mut map: OpenHashMap<&str, u32> = OpenHashMap::default();
    map.insert("a", 30);
    match map.entry("a") {
        Entry::Occupied(mut entry) => {
            assert_eq!(*entry.key(), "a");
            assert_eq!(*entry.get(), 30);
            assert_eq!(entry.insert(1), 30);
            *entry.get_mut() += 1;
            assert_eq!(entry.remove(), 2);
        }
        Entry::Vacant(_) => panic!("expected occupied"),
    }
    assert!(!map.contains_key(&"a"));

    match map.entry("q") {
        Entry::Vacant(entry) => {
            assert_eq!(*entry.key(), "q");
            assert_eq!(entry.into_key(), "q");
        }
        Entry::Occupied(_) => panic!("expected vacant"),
    }
    assert!(!map.contains_key(&"q"));

    map.insert("r", 7);
    match map.entry("r") {
        Entry::Occupied(entry) => assert_eq!(entry.remove_entry(), ("r", 7)),
        Entry::Vacant(_) => panic!("expected occupied"),
    }
    assert!(map.is_empty());

    match map.entry("s") {
        Entry::Vacant(entry) => *entry.insert(4) += 1,
        Entry::Occupied(_) => panic!("expected vacant"),
    }
    assert_eq!(map.get(&"s"), Some(&5));
}

#[test]
fn entries_survive_growth() {
    let mut map: OpenHashMap<u32, u32> = OpenHashMap::default();
    for i in 0..2000u32 {
        *map.entry(i % 500).or_insert(0) += 1;
    }
    assert_eq!(map.len(), 500);
    assert!(map.values().all(|v| *v == 4));
    // A vacant insert that has to grow the table first must hand back a
    // reference to the value's final slot.
    let mut small: OpenHashMap<u32, u32> = OpenHashMap::default();
    for i in 0..64u32 {
        let slot = small.entry(i).or_insert(i);
        *slot += 100;
    }
    for i in 0..64u32 {
        assert_eq!(small.get(&i), Some(&(i + 100)));
    }
}

// --- retain, iterators -----------------------------------------------------

#[test]
fn retain_filters_and_can_mutate() {
    let mut map: OpenHashMap<u32, u32> = (0..200).map(|i| (i, i)).collect();
    map.retain(|k, v| {
        *v += 1;
        k % 3 == 0
    });
    assert_eq!(map.len(), 67);
    for i in 0..200u32 {
        if i % 3 == 0 {
            assert_eq!(map.get(&i), Some(&(i + 1)));
        } else {
            assert!(!map.contains_key(&i));
        }
    }
    let mut empty: OpenHashMap<u32, u32> = OpenHashMap::default();
    empty.retain(|_, _| true);
    assert!(empty.is_empty());
    map.retain(|_, _| false);
    assert!(map.is_empty());
}

#[test]
fn iterators_report_exact_sizes_and_cover_everything() {
    let map: OpenHashMap<u32, u32> = (0..100).map(|i| (i, i * 2)).collect();
    assert_eq!(map.iter().len(), 100);
    assert_eq!(map.keys().len(), 100);
    assert_eq!(map.values().len(), 100);

    let mut iter = map.iter();
    iter.next();
    iter.next();
    assert_eq!(iter.size_hint(), (98, Some(98)));
    let fork = iter.clone();
    assert_eq!(iter.count(), 98);
    assert_eq!(fork.count(), 98);

    assert_eq!(map.keys().copied().sum::<u32>(), 4950);
    assert_eq!(map.values().copied().sum::<u32>(), 9900);
    assert_eq!((&map).into_iter().count(), 100);

    let mut owned: Vec<(u32, u32)> = map.clone().into_iter().collect();
    owned.sort_unstable();
    let expected: Vec<(u32, u32)> = (0..100).map(|i| (i, i * 2)).collect();
    assert_eq!(owned, expected);

    let mut keys: Vec<u32> = map.clone().into_keys().collect();
    keys.sort_unstable();
    assert_eq!(keys, (0..100).collect::<Vec<u32>>());
    assert_eq!(map.clone().into_values().sum::<u32>(), 9900);
    assert_eq!(map.clone().into_iter().len(), 100);

    let empty: OpenHashMap<u32, u32> = OpenHashMap::default();
    assert_eq!(empty.iter().count(), 0);
    assert_eq!(empty.into_iter().count(), 0);
}

#[test]
fn mutable_iteration() {
    let mut map: OpenHashMap<u32, u32> = (0..100).map(|i| (i, i)).collect();
    for (_, value) in map.iter_mut() {
        *value += 1;
    }
    for value in map.values_mut() {
        *value += 1;
    }
    for (_, value) in &mut map {
        *value += 1;
    }
    for i in 0..100u32 {
        assert_eq!(map[&i], i + 3);
    }
}

// --- drops -----------------------------------------------------------------

struct Tracked(Rc<Cell<usize>>);

impl Drop for Tracked {
    fn drop(&mut self) {
        self.0.set(self.0.get() + 1);
    }
}

#[test]
fn every_value_is_dropped_exactly_once() {
    let drops = Rc::new(Cell::new(0usize));
    let mut map: OpenHashMap<u32, Tracked> = OpenHashMap::default();
    for i in 0..100 {
        map.insert(i, Tracked(drops.clone()));
    }
    assert_eq!(drops.get(), 0);

    for i in 0..10 {
        assert!(map.remove(&i).is_some());
    }
    assert_eq!(drops.get(), 10);

    // Overwriting hands the old value back; dropping it counts once.
    map.insert(50, Tracked(drops.clone()));
    assert_eq!(drops.get(), 11);

    map.retain(|k, _| k % 2 == 0);
    assert_eq!(map.len(), 45);
    assert_eq!(drops.get(), 56);

    map.clear();
    assert_eq!(drops.get(), 101);
    drop(map);
    assert_eq!(drops.get(), 101);
}

#[test]
fn dropping_a_map_or_a_partly_consumed_iterator_drops_the_rest() {
    let drops = Rc::new(Cell::new(0usize));
    let mut map: OpenHashMap<u32, Tracked> = OpenHashMap::default();
    for i in 0..30 {
        map.insert(i, Tracked(drops.clone()));
    }
    drop(map);
    assert_eq!(drops.get(), 30);

    let mut map: OpenHashMap<u32, Tracked> = OpenHashMap::default();
    for i in 0..30 {
        map.insert(i, Tracked(drops.clone()));
    }
    let mut iter = map.into_iter();
    for _ in 0..5 {
        drop(iter.next());
    }
    assert_eq!(drops.get(), 35);
    drop(iter);
    assert_eq!(drops.get(), 60);
}

// --- sets ------------------------------------------------------------------

fn sorted<'a>(iter: impl Iterator<Item = &'a u64>) -> Vec<u64> {
    let mut values: Vec<u64> = iter.copied().collect();
    values.sort_unstable();
    values
}

#[test]
fn set_basics() {
    let mut set: OpenHashSet<u32> = OpenHashSet::default();
    assert!(set.is_empty());
    assert!(set.insert(7));
    assert!(!set.insert(7));
    assert!(set.insert(9));
    assert_eq!(set.len(), 2);
    assert!(set.contains(&7));
    assert_eq!(set.get(&9), Some(&9));
    assert_eq!(set.take(&9), Some(9));
    assert_eq!(set.take(&9), None);
    assert!(set.remove(&7));
    assert!(!set.remove(&7));
    assert!(set.is_empty());
}

#[test]
fn set_with_string_elements_and_borrowed_lookup() {
    let mut set: OpenHashSet<String> = OpenHashSet::default();
    set.insert(String::from("x"));
    assert!(set.contains("x"));
    assert!(!set.contains("y"));
    assert_eq!(set.get("x").map(String::as_str), Some("x"));
}

#[test]
fn set_algebra_matches_btreeset() {
    let mut rng = Rng(0xABCD_EF01_2345_6789);
    for _ in 0..50 {
        let a: BTreeSet<u64> = (0..40).map(|_| rng.next() % 60).collect();
        let b: BTreeSet<u64> = (0..40).map(|_| rng.next() % 60).collect();
        let sa: OpenHashSet<u64> = a.iter().copied().collect();
        let sb: OpenHashSet<u64> = b.iter().copied().collect();

        assert_eq!(sorted(sa.union(&sb)), sorted(a.union(&b)));
        assert_eq!(sorted(sa.intersection(&sb)), sorted(a.intersection(&b)));
        assert_eq!(sorted(sa.difference(&sb)), sorted(a.difference(&b)));
        assert_eq!(sa.is_disjoint(&sb), a.is_disjoint(&b));
        assert_eq!(sa.is_subset(&sb), a.is_subset(&b));
        assert_eq!(sa.is_superset(&sb), a.is_superset(&b));
    }
    let small: OpenHashSet<u32> = [1, 2].into_iter().collect();
    let big: OpenHashSet<u32> = [1, 2, 3].into_iter().collect();
    assert!(small.is_subset(&big));
    assert!(big.is_superset(&small));
    assert!(!big.is_subset(&small));
}

#[test]
fn set_equality_ignores_order_and_collect_dedups() {
    let a: OpenHashSet<u32> = (0..50).collect();
    let b: OpenHashSet<u32> = (0..50).rev().collect();
    assert!(a == b);
    let dup: OpenHashSet<u32> = [1, 1, 2, 2, 3].into_iter().collect();
    assert_eq!(dup.len(), 3);
    let mut c = a.clone();
    c.insert(1000);
    assert!(a != c);
    c.remove(&1000);
    c.remove(&3);
    assert!(a != c, "same length is not enough");
    c.insert(3000);
    assert!(a != c);

    let mut d = a.clone();
    d.insert(2000);
    assert!(a != d, "strict subset");
    assert!(d != a);
}

#[test]
fn set_retain_extend_iterate_debug() {
    let mut set: OpenHashSet<u32> = (0..100).collect();
    set.retain(|x| x % 10 == 0);
    assert_eq!(set.len(), 10);
    set.extend([5, 15, 10]);
    assert_eq!(set.len(), 12);
    assert_eq!(set.iter().len(), 12);
    assert_eq!((&set).into_iter().count(), 12);
    let mut all: Vec<u32> = set.clone().into_iter().collect();
    all.sort_unstable();
    assert_eq!(all, [0, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90]);
    assert_eq!(set.clone().into_iter().len(), 12);

    let mut one: OpenHashSet<u8> = OpenHashSet::default();
    assert_eq!(format!("{one:?}"), "{}");
    one.insert(4);
    assert_eq!(format!("{one:?}"), "{4}");
}

#[test]
fn set_of_unit_like_and_zero_sized_elements() {
    let mut set: OpenHashSet<()> = OpenHashSet::default();
    assert!(set.insert(()));
    assert!(!set.insert(()));
    assert_eq!(set.len(), 1);
}

// --- the HashMap / HashSet aliases -----------------------------------------

/// Written only against what both implementations share, so it builds in
/// whichever configuration is active.
fn exercise_common_subset(map: &mut HashMap<u32, u32>) {
    assert!(map.is_empty());
    for i in 0..300u32 {
        assert_eq!(map.insert(i, i + 1), None);
    }
    assert_eq!(map.len(), 300);
    assert_eq!(map.get(&7), Some(&8));
    *map.get_mut(&7).unwrap() = 70;
    assert_eq!(map[&7], 70);
    assert!(map.contains_key(&299));
    assert_eq!(map.remove(&7), Some(70));
    assert_eq!(map.remove_entry(&8), Some((8, 9)));
    match map.entry(7) {
        hash_map::Entry::Vacant(entry) => {
            entry.insert(1);
        }
        hash_map::Entry::Occupied(_) => panic!("7 was removed"),
    }
    *map.entry(7).or_insert(0) += 1;
    assert_eq!(map[&7], 2);
    map.retain(|k, _| k % 2 == 1);
    assert_eq!(map.len(), 150);
    assert_eq!(map.keys().count(), 150);
    assert_eq!(map.values().count(), 150);
    assert_eq!(map.iter().count(), 150);
    for (_, value) in map.iter_mut() {
        *value = 0;
    }
    assert!(map.values().all(|v| *v == 0));
    let extra: HashMap<u32, u32> = (0..10).map(|i| (1000 + i, i)).collect();
    map.extend(extra);
    assert_eq!(map.len(), 160);
    map.clear();
    assert!(map.is_empty());
}

#[test]
fn hashmap_alias_supports_the_shared_api() {
    let mut by_default: HashMap<u32, u32> = HashMap::default();
    exercise_common_subset(&mut by_default);
    let mut by_hasher: HashMap<u32, u32> = HashMap::with_hasher(FixedHasher);
    exercise_common_subset(&mut by_hasher);
    let mut by_capacity: HashMap<u32, u32> = HashMap::with_capacity_and_hasher(64, FixedHasher);
    exercise_common_subset(&mut by_capacity);
}

#[test]
fn hashset_alias_supports_the_shared_api() {
    let mut set: HashSet<u32> = HashSet::default();
    assert!(set.insert(1));
    assert!(!set.insert(1));
    assert!(set.contains(&1));
    assert!(set.remove(&1));
    let a: HashSet<u32> = (0..10).collect();
    let b: HashSet<u32> = (5..15).collect();
    assert_eq!(a.intersection(&b).count(), 5);
    assert_eq!(a.union(&b).count(), 15);
    assert_eq!(a.difference(&b).count(), 5);
    assert!(!a.is_disjoint(&b));
    assert_eq!(a.iter().count(), 10);
}

#[test]
fn hashmap_alias_matches_the_model() {
    let mut rng = Rng(0x7777_1111_3333_5555);
    let mut map: HashMap<u64, u64> = HashMap::default();
    let mut model: BTreeMap<u64, u64> = BTreeMap::new();
    for _ in 0..5_000 {
        let key = rng.next() % 200;
        if rng.next() % 2 == 0 {
            let value = rng.next();
            assert_eq!(map.insert(key, value), model.insert(key, value));
        } else {
            assert_eq!(map.remove(&key), model.remove(&key));
        }
    }
    assert_eq!(map.len(), model.len());
    for (k, v) in &model {
        assert_eq!(map.get(k), Some(v));
    }
}
