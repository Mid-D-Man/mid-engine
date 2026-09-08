// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-arena.md, section "unchecked_slot_arena.rs"
// ============================================================================
//! Non-generational, value-storing slot arena: [`UncheckedSlotArena<T>`]
//! is [`SlotArena<T>`](crate::SlotArena) with the generation field and
//! check removed entirely -- a bare `u32` index in, a bare `u32` index
//! back, no staleness detection at all. Mid-arena's own native
//! implementation of the "Vec + freelist, no ABA check" row in
//! `docs/mid-arena.md`'s own survey taxonomy, which until this pass
//! only had one occupant: `slab` itself, an external crate.
//!
//! # Why this exists: a real, measured ~4-5x gap that isn't a
//! micro-optimization opportunity
//!
//! `slab` (real source read this pass, exact pinned version 0.4.12
//! from `Cargo.toml`) measures 1.65 ns/op insert, 0.74 ns/op get on
//! CI -- against `SlotArena`'s own 7.76 / 0.99 on the same run. Its
//! `Entry<T>` is `enum Entry<T> { Vacant(usize), Occupied(T) }`: no
//! generation field in either variant, anywhere. Its `insert`/`get`
//! are otherwise the exact same shape `SlotArena` already uses --
//! bounds-checked `Vec` access via `.get()`/`.get_mut()`, LIFO free
//! list, no special inlining tricks (checked directly, not assumed --
//! see `docs/mid-arena.md`'s `#[inline(never)]` writeup for why that
//! specific lead was chased and ruled out first). The entire gap is
//! the generation field: one fewer `u32` read plus one fewer integer
//! compare, every single call, which at ~1-8 ns/op total is a large
//! fraction of the real cost. `generational-arena`/
//! `typed-generational-arena`/`thunderdome`/`slotmap` all cluster at
//! 6.4-8.2 ns/op insert on the same run *because* they all pay this
//! same real cost for the same real guarantee -- that's the practical
//! floor for ABA-safety, not a sign any of them (or `SlotArena`) is
//! poorly written. The only way to get `slab`'s number is to stop
//! paying for what `slab` doesn't do.
//!
//! # Why a separate type, not a feature flag that removes the check
//! from `SlotArena` itself
//!
//! Matches this crate's own already-established convention
//! (`SlotArena` vs [`CompactSlotArena`](crate::CompactSlotArena) vs
//! [`BumpArena`](crate::BumpArena) are three separate types for three
//! different tradeoffs, never one type with a runtime flag toggling
//! unsafe-adjacent behavior). Removing ABA-safety is a real,
//! meaningful change to what a handle *means* -- `ArenaKey` shrinks
//! from an index+generation pair to a bare index, so the choice has
//! to be visible in the type signature a caller writes down, not
//! buried in a constructor argument or a runtime bool.
//!
//! # What "unchecked" actually means here -- read before reaching for
//! this over `SlotArena`
//!
//! There is no memory-unsafety anywhere in this file -- every access
//! is still bounds-checked against `slots.len()`, exactly like
//! `SlotArena`. What's gone is *staleness* detection. Hold an index
//! past a [`remove`](UncheckedSlotArena::remove) of the value it
//! pointed at, and once that slot is reused by a later
//! [`insert`](UncheckedSlotArena::insert), the old index silently
//! reads, writes, or removes the *new* value at that slot -- no
//! panic, no `None`, just the wrong data, indistinguishable from the
//! right data. `slab`'s own doc comment undersells this ("it is
//! important to note that keys may be reused"); this module's own
//! test suite includes one that demonstrates the actual failure mode
//! directly (`stale_index_silently_aliases_the_reused_slot`) rather
//! than only describing it in prose. Reach for this only where the
//! index's lifetime is provably shorter than one insert/remove cycle
//! at that slot -- a tight per-frame scratch loop, an index handed
//! straight back within the same function, anything closer to
//! `Vec`-with-swap-remove territory than to a handle meant to outlive
//! the operation that produced it. `SlotArena` stays the default;
//! this is the deliberate opt-out, not a faster replacement for it.
//!
//! # Design
//!
//! Otherwise identical to `SlotArena`: same LIFO free list, same
//! `free_head == slots.len()` past-the-end-means-grow convention,
//! same plain safe enum for `Slot<T>` (no union -- see `SlotArena`'s
//! own doc comment for why that trade waits for a real, profiled
//! need). The only structural difference is `Slot<T>`'s `Vacant`
//! variant carrying a bare `next_free: u32` instead of
//! `{ generation: u32, next_free: u32 }`.

use alloc::vec::Vec;
use core::mem::replace;

enum Slot<T> {
    Occupied(T),
    Vacant(u32),
}

/// Non-generational, value-storing arena. See this module's doc
/// comment -- in particular "What 'unchecked' actually means here" --
/// before choosing this over [`SlotArena`](crate::SlotArena).
pub struct UncheckedSlotArena<T> {
    slots: Vec<Slot<T>>,
    /// Index into `slots` of the next slot to reuse.
    /// `free_head == slots.len()` means "nothing free, grow instead" --
    /// same convention as `SlotArena`.
    free_head: u32,
    live_count: usize,
}

impl<T> UncheckedSlotArena<T> {
    /// Creates an arena with nothing allocated yet.
    pub fn new() -> Self {
        Self {
            slots: Vec::new(),
            free_head: 0,
            live_count: 0,
        }
    }

    /// Creates an arena pre-sized for `capacity` live values before the
    /// next insert past that would reallocate.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            slots: Vec::with_capacity(capacity),
            free_head: 0,
            live_count: 0,
        }
    }

    /// Number of currently-live (inserted, not yet removed) values.
    #[inline]
    pub fn len(&self) -> usize {
        self.live_count
    }

    /// True if nothing is currently live.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.live_count == 0
    }

    /// Live values this arena can hold before the next insert past that
    /// reallocates.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.slots.capacity()
    }

    /// Total slots ever created (live + freed-but-not-yet-reused). Not
    /// the same as [`len`](Self::len) once anything has been removed.
    #[inline]
    pub fn slot_count(&self) -> usize {
        self.slots.len()
    }

    /// Inserts `value`, returning the raw index it now lives at. Either
    /// reuses the most recently freed slot (LIFO) or grows by one if
    /// nothing is free -- see this module's doc comment. Read "What
    /// 'unchecked' actually means here" before holding the returned
    /// index past a [`remove`](Self::remove) of it.
    pub fn insert(&mut self, value: T) -> u32 {
        let free_head = self.free_head;

        if let Some(slot) = self.slots.get_mut(free_head as usize) {
            let next_free = match slot {
                Slot::Vacant(next_free) => *next_free,
                Slot::Occupied(_) => unreachable!(
                    "free_head must always point at a Vacant slot -- \
                     insert/remove are the only writers of free_head \
                     and both uphold this"
                ),
            };
            *slot = Slot::Occupied(value);
            self.free_head = next_free;
            self.live_count += 1;
            free_head
        } else {
            debug_assert_eq!(
                free_head as usize,
                self.slots.len(),
                "free_head should never point past a single new slot beyond the end"
            );
            debug_assert!(
                self.slots.len() < u32::MAX as usize,
                "UncheckedSlotArena holds u32::MAX slots -- index would overflow"
            );
            self.slots.push(Slot::Occupied(value));
            self.free_head = free_head + 1;
            self.live_count += 1;
            free_head
        }
    }

    /// Removes and returns the value at `index`, if that slot is
    /// currently occupied. Removing an out-of-bounds or already-vacant
    /// index is a safe no-op returning `None` -- but a *stale* index
    /// (one whose slot has since been reused by a later `insert`)
    /// removes whatever is currently there instead, silently. See this
    /// module's doc comment.
    pub fn remove(&mut self, index: u32) -> Option<T> {
        let slot = self.slots.get_mut(index as usize)?;
        if matches!(slot, Slot::Vacant(_)) {
            return None;
        }

        let next_free = self.free_head;
        let old = replace(slot, Slot::Vacant(next_free));
        self.free_head = index;
        self.live_count -= 1;

        match old {
            Slot::Occupied(value) => Some(value),
            Slot::Vacant(_) => unreachable!("just checked above"),
        }
    }

    /// Whether `index` currently points at a live value. Same staleness
    /// caveat as every other accessor here -- see this module's doc
    /// comment.
    #[inline]
    pub fn contains(&self, index: u32) -> bool {
        self.get(index).is_some()
    }

    /// Immutable access to the value at `index`, or `None` if it's
    /// out of bounds or currently vacant. A *stale* index reads back
    /// whatever value currently occupies that slot, not `None` -- see
    /// this module's doc comment.
    #[inline]
    pub fn get(&self, index: u32) -> Option<&T> {
        match self.slots.get(index as usize)? {
            Slot::Occupied(value) => Some(value),
            Slot::Vacant(_) => None,
        }
    }

    /// Mutable counterpart to [`get`](Self::get). Same staleness
    /// caveat.
    #[inline]
    pub fn get_mut(&mut self, index: u32) -> Option<&mut T> {
        match self.slots.get_mut(index as usize)? {
            Slot::Occupied(value) => Some(value),
            Slot::Vacant(_) => None,
        }
    }

    /// Iterates over every live `(index, &value)` pair. Not
    /// necessarily insertion order once anything has been removed and
    /// its slot reused (reuse is LIFO) -- a straight index-order scan,
    /// skipping vacant slots.
    pub fn iter(&self) -> impl Iterator<Item = (u32, &T)> {
        self.slots.iter().enumerate().filter_map(|(i, slot)| match slot {
            Slot::Occupied(value) => Some((i as u32, value)),
            Slot::Vacant(_) => None,
        })
    }

    /// Mutable counterpart to [`iter`](Self::iter).
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (u32, &mut T)> {
        self.slots
            .iter_mut()
            .enumerate()
            .filter_map(|(i, slot)| match slot {
                Slot::Occupied(value) => Some((i as u32, value)),
                Slot::Vacant(_) => None,
            })
    }

    /// Drops every live value and resets to empty. Every previously
    /// issued index reads as dead afterward -- `slots` itself is
    /// cleared, so `get` on any old index misses outright.
    pub fn clear(&mut self) {
        self.slots.clear();
        self.free_head = 0;
        self.live_count = 0;
    }
}

impl<T> Default for UncheckedSlotArena<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn starts_empty() {
        let a: UncheckedSlotArena<u32> = UncheckedSlotArena::new();
        assert_eq!(a.len(), 0);
        assert!(a.is_empty());
        assert_eq!(a.slot_count(), 0);
    }

    #[test]
    fn insert_get_roundtrip() {
        let mut a = UncheckedSlotArena::new();
        let k = a.insert(42u32);
        assert_eq!(a.get(k), Some(&42));
        assert_eq!(a.len(), 1);
    }

    #[test]
    fn get_mut_writes_through() {
        let mut a = UncheckedSlotArena::new();
        let k = a.insert(1u32);
        *a.get_mut(k).unwrap() = 2;
        assert_eq!(a.get(k), Some(&2));
    }

    #[test]
    fn remove_returns_value_and_frees_slot() {
        let mut a = UncheckedSlotArena::new();
        let k = a.insert(7u32);
        assert_eq!(a.remove(k), Some(7));
        assert_eq!(a.get(k), None);
        assert!(a.is_empty());
    }

    #[test]
    fn remove_on_already_dead_or_out_of_bounds_index_is_a_safe_no_op() {
        let mut a: UncheckedSlotArena<u32> = UncheckedSlotArena::new();
        let k = a.insert(1);
        a.remove(k);
        assert_eq!(
            a.remove(k),
            None,
            "removing the already-vacant slot again must not panic"
        );
        assert_eq!(a.remove(999), None, "an out-of-bounds index must not panic");
    }

    #[test]
    fn stale_index_silently_aliases_the_reused_slot() {
        // This is not a bug -- it is the entire, documented tradeoff
        // this type makes in exchange for slab's real, measured speed.
        // This test exists so that tradeoff stays true and visible on
        // purpose, matching this module's own doc comment word for
        // word, not just described in prose and left to drift.
        let mut a = UncheckedSlotArena::new();
        let first = a.insert(100u32);
        assert_eq!(a.remove(first), Some(100));

        let second = a.insert(200u32);
        assert_eq!(
            second, first,
            "the freed slot's index is reused exactly -- there is no \
             generation to make it look different"
        );
        assert_eq!(
            a.get(first),
            Some(&200),
            "the 'stale' handle is not stale at all here -- it silently \
             reads the new value, which is the actual risk this type \
             carries"
        );
    }

    #[test]
    fn free_list_reuse_order_is_lifo() {
        let mut a = UncheckedSlotArena::new();
        let k0 = a.insert('a');
        let k1 = a.insert('b');
        let k2 = a.insert('c');

        a.remove(k0);
        a.remove(k1);
        a.remove(k2);

        let r1 = a.insert('x');
        let r2 = a.insert('y');
        let r3 = a.insert('z');
        assert_eq!(r1, k2);
        assert_eq!(r2, k1);
        assert_eq!(r3, k0);
    }

    #[test]
    fn iterate_visits_every_live_value_and_skips_removed_ones() {
        let mut a = UncheckedSlotArena::new();
        let k0 = a.insert(1u32);
        let _k1 = a.insert(2u32);
        let k2 = a.insert(3u32);
        a.remove(k0);

        let mut seen: Vec<u32> = a.iter().map(|(_, v)| *v).collect();
        seen.sort_unstable();
        assert_eq!(seen, [2, 3]);
        assert!(a.iter().any(|(k, _)| k == k2));
    }

    #[test]
    fn iter_mut_writes_through_to_every_live_value() {
        let mut a = UncheckedSlotArena::new();
        a.insert(1u32);
        a.insert(2u32);
        for (_, v) in a.iter_mut() {
            *v *= 10;
        }
        let mut seen: Vec<u32> = a.iter().map(|(_, v)| *v).collect();
        seen.sort_unstable();
        assert_eq!(seen, [10, 20]);
    }

    #[test]
    fn clear_drops_values_and_resets_indices() {
        let mut a = UncheckedSlotArena::new();
        let k0 = a.insert(1u32);
        let k1 = a.insert(2u32);
        a.clear();
        assert!(a.is_empty());
        assert_eq!(a.get(k0), None);
        assert_eq!(a.get(k1), None);
        let k2 = a.insert(3u32);
        assert_eq!(k2, 0);
    }

    #[test]
    fn slot_count_tracks_total_slots_not_just_live() {
        let mut a = UncheckedSlotArena::new();
        let k0 = a.insert(1u32);
        a.insert(2u32);
        a.insert(3u32);
        assert_eq!(a.slot_count(), 3);
        a.remove(k0);
        assert_eq!(a.slot_count(), 3, "freeing doesn't shrink slot_count");
        assert_eq!(a.len(), 2);
        a.insert(4u32);
        assert_eq!(a.slot_count(), 3, "reuse shouldn't grow it either");
        assert_eq!(a.len(), 3);
    }

    #[test]
    fn many_insert_remove_cycles_stay_consistent() {
        let mut a = UncheckedSlotArena::new();
        let mut live: Vec<(u32, u32)> = Vec::new();

        for round in 0u32..50 {
            let k = a.insert(round);
            live.push((k, round));
            if round % 3 == 0 && !live.is_empty() {
                let (dead_key, dead_val) = live.remove(0);
                assert_eq!(a.remove(dead_key), Some(dead_val));
            }
            assert_eq!(a.len(), live.len());
            for &(k, v) in &live {
                assert_eq!(a.get(k), Some(&v));
            }
        }
    }

    #[test]
    fn default_matches_new() {
        let a: UncheckedSlotArena<u32> = UncheckedSlotArena::default();
        assert!(a.is_empty());
    }

    #[test]
    fn drop_runs_for_every_live_value_when_the_arena_itself_is_dropped() {
        use core::cell::Cell;

        struct DropCounter<'a>(&'a Cell<u32>);
        impl<'a> Drop for DropCounter<'a> {
            fn drop(&mut self) {
                self.0.set(self.0.get() + 1);
            }
        }

        let count = Cell::new(0u32);
        {
            let mut a = UncheckedSlotArena::new();
            a.insert(DropCounter(&count));
            a.insert(DropCounter(&count));
            let k2 = a.insert(DropCounter(&count));
            a.remove(k2);
            assert_eq!(count.get(), 1);
        }
        assert_eq!(count.get(), 3);
    }
}
