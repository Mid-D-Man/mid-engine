// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-platform.md, section "collections/"
// ============================================================================
//! The table underneath [`OpenHashMap`](super::OpenHashMap) and
//! [`OpenHashSet`](super::OpenHashSet): linear probing over a power-of-two
//! `Vec<Option<Bucket>>`, safe code only.
//!
//! Nothing in here knows about a hasher. Every operation takes a hash the
//! caller already computed, and every bucket stores its own hash, so growing
//! the table never needs the key's `Hash` impl again. That is what lets the
//! entry types hold a bare `&mut RawTable<K, V>` and keep `std`'s own
//! two-parameter `Entry<'a, K, V>` shape.

use alloc::vec::Vec;
use core::mem;

/// Smallest table that is ever allocated.
const MIN_SLOTS: usize = 4;

/// 2^64 / golden ratio, odd. Multiplying by it and taking the *top* bits of
/// the product spreads any hasher's output across the table, including a
/// hasher whose low bits are weak (an identity hasher, or `FxHasher` on keys
/// that share their low bits).
const FIBONACCI: u64 = 0x9E37_79B9_7F4A_7C15;

/// How many entries a table of `slots` slots may hold: 3/4 of them. Linear
/// probing degrades quickly past that point.
const fn usable(slots: usize) -> usize {
    slots - slots / 4
}

#[derive(Clone)]
pub(crate) struct Bucket<K, V> {
    pub(crate) hash: u64,
    pub(crate) key: K,
    pub(crate) value: V,
}

/// Result of [`RawTable::probe`].
pub(crate) enum Slot {
    /// The key is present, at this slot index.
    Found(usize),
    /// The key is absent; this is the first empty slot on its probe path.
    /// Only meaningful to [`RawTable::place`] if nothing changed in between.
    Empty(usize),
}

#[derive(Clone)]
pub(crate) struct RawTable<K, V> {
    slots: Vec<Option<Bucket<K, V>>>,
    len: usize,
    /// `64 - log2(slots.len())`. Meaningless while `slots` is empty; every
    /// path that reads it checks for the empty table first.
    shift: u32,
}

impl<K, V> RawTable<K, V> {
    pub(crate) const fn new() -> Self {
        Self {
            slots: Vec::new(),
            len: 0,
            shift: 63,
        }
    }

    pub(crate) fn with_capacity(entries: usize) -> Self {
        let mut table = Self::new();
        table.reserve(entries);
        table
    }

    pub(crate) fn len(&self) -> usize {
        self.len
    }

    /// Entries the table can hold before it has to grow.
    pub(crate) fn capacity(&self) -> usize {
        usable(self.slots.len())
    }

    pub(crate) fn slots(&self) -> &[Option<Bucket<K, V>>] {
        &self.slots
    }

    pub(crate) fn slots_mut(&mut self) -> &mut [Option<Bucket<K, V>>] {
        &mut self.slots
    }

    pub(crate) fn into_slots(self) -> Vec<Option<Bucket<K, V>>> {
        self.slots
    }

    pub(crate) fn bucket(&self, index: usize) -> &Bucket<K, V> {
        match &self.slots[index] {
            Some(bucket) => bucket,
            None => unreachable!("slot index came from a probe that found an entry"),
        }
    }

    pub(crate) fn bucket_mut(&mut self, index: usize) -> &mut Bucket<K, V> {
        match &mut self.slots[index] {
            Some(bucket) => bucket,
            None => unreachable!("slot index came from a probe that found an entry"),
        }
    }

    /// Ideal slot for `hash`. Only call on a non-empty table.
    #[inline]
    fn home(&self, hash: u64) -> usize {
        (hash.wrapping_mul(FIBONACCI) >> self.shift) as usize
    }

    /// Walks the probe path for `hash`. Stops at the first empty slot, which
    /// always exists because the table is never more than 3/4 full.
    #[inline]
    pub(crate) fn probe<F: Fn(&K) -> bool>(&self, hash: u64, eq: F) -> Slot {
        if self.slots.is_empty() {
            // `place` only trusts this index while `len < capacity()`, which
            // is false for an unallocated table, so it never gets used.
            return Slot::Empty(0);
        }
        let mask = self.slots.len() - 1;
        let mut index = self.home(hash);
        loop {
            match &self.slots[index] {
                None => return Slot::Empty(index),
                Some(bucket) => {
                    if bucket.hash == hash && eq(&bucket.key) {
                        return Slot::Found(index);
                    }
                }
            }
            index = (index + 1) & mask;
        }
    }

    #[inline]
    pub(crate) fn find<F: Fn(&K) -> bool>(&self, hash: u64, eq: F) -> Option<usize> {
        match self.probe(hash, eq) {
            Slot::Found(index) => Some(index),
            Slot::Empty(_) => None,
        }
    }

    /// First empty slot on the probe path for `hash`. The table must be
    /// allocated and below its load limit.
    fn find_empty(&self, hash: u64) -> usize {
        let mask = self.slots.len() - 1;
        let mut index = self.home(hash);
        while self.slots[index].is_some() {
            index = (index + 1) & mask;
        }
        index
    }

    /// Stores a bucket whose key is known to be absent and returns its slot.
    /// `empty_hint` is the index a [`probe`](Self::probe) of the same `hash`
    /// returned; it is used only when no growth is needed.
    pub(crate) fn place(&mut self, hash: u64, key: K, value: V, empty_hint: usize) -> usize {
        let index = if self.len < self.capacity() {
            empty_hint
        } else {
            self.grow();
            self.find_empty(hash)
        };
        self.slots[index] = Some(Bucket { hash, key, value });
        self.len += 1;
        index
    }

    fn grow(&mut self) {
        let slots = if self.slots.is_empty() {
            MIN_SLOTS
        } else {
            self.slots
                .len()
                .checked_mul(2)
                .expect("hash table capacity overflow")
        };
        self.resize(slots);
    }

    /// Makes room for `additional` more entries without another growth.
    pub(crate) fn reserve(&mut self, additional: usize) {
        let needed = self
            .len
            .checked_add(additional)
            .expect("hash table capacity overflow");
        if needed <= self.capacity() {
            return;
        }
        // Smallest power of two whose 3/4 holds `needed`: ceil(needed * 4 / 3).
        let slots = needed
            .checked_mul(4)
            .expect("hash table capacity overflow")
            .div_ceil(3)
            .checked_next_power_of_two()
            .expect("hash table capacity overflow")
            .max(MIN_SLOTS);
        self.resize(slots);
    }

    fn resize(&mut self, new_slots: usize) {
        debug_assert!(new_slots.is_power_of_two() && new_slots >= MIN_SLOTS);
        debug_assert!(usable(new_slots) >= self.len);
        let mut slots = Vec::with_capacity(new_slots);
        slots.resize_with(new_slots, || None);
        let old = mem::replace(&mut self.slots, slots);
        self.shift = 64 - new_slots.trailing_zeros();
        for bucket in old.into_iter().flatten() {
            let index = self.find_empty(bucket.hash);
            self.slots[index] = Some(bucket);
        }
    }

    /// Removes the entry at `index` and shifts later members of its cluster
    /// back so no probe path ever crosses a hole. No tombstones.
    pub(crate) fn remove_at(&mut self, index: usize) -> Bucket<K, V> {
        let removed = match self.slots[index].take() {
            Some(bucket) => bucket,
            None => unreachable!("slot index came from a probe that found an entry"),
        };
        self.len -= 1;
        let mask = self.slots.len() - 1;
        let mut hole = index;
        let mut next = index;
        loop {
            next = (next + 1) & mask;
            let home = match &self.slots[next] {
                None => break,
                Some(bucket) => self.home(bucket.hash),
            };
            // The entry at `next` may fill the hole only if the hole lies on
            // its probe path, i.e. at or after its home slot (cyclically).
            if (next.wrapping_sub(home) & mask) >= (next.wrapping_sub(hole) & mask) {
                self.slots[hole] = self.slots[next].take();
                hole = next;
            }
        }
        removed
    }

    pub(crate) fn clear(&mut self) {
        for slot in &mut self.slots {
            *slot = None;
        }
        self.len = 0;
    }

    /// Keeps the entries `keep` returns `true` for, by rebuilding the table
    /// at its current size. Entries are visited in slot order.
    pub(crate) fn retain<F: FnMut(&K, &mut V) -> bool>(&mut self, mut keep: F) {
        if self.len == 0 {
            return;
        }
        let mut slots = Vec::with_capacity(self.slots.len());
        slots.resize_with(self.slots.len(), || None);
        let old = mem::replace(&mut self.slots, slots);
        self.len = 0;
        for mut bucket in old.into_iter().flatten() {
            if keep(&bucket.key, &mut bucket.value) {
                let index = self.find_empty(bucket.hash);
                self.slots[index] = Some(bucket);
                self.len += 1;
            }
        }
    }

    /// Length of the longest run of occupied slots. Test-only: a measure of
    /// how well the home-slot function spreads a set of hashes.
    #[cfg(test)]
    pub(crate) fn longest_cluster(&self) -> usize {
        let mut longest = 0;
        let mut run = 0;
        for slot in &self.slots {
            if slot.is_some() {
                run += 1;
                longest = longest.max(run);
            } else {
                run = 0;
            }
        }
        longest
    }
}
