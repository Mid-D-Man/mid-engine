//! Generational-index allocator: issues [`GenerationalIndex`] handles that
//! detect their own staleness after the slot they point at gets freed and
//! reused.
//!
//! Not from the C++ prior-art list `docs/mid-collections.md` otherwise
//! draws on -- this is the Rust ecosystem's own well-established answer
//! (`slotmap`, `generational-arena`) to a problem that shows up the
//! moment IDs can be freed and their slots reused: a stale handle held
//! from before a free must not silently alias whatever new thing got
//! allocated into that same slot. Fix: pair each slot with a generation
//! counter, bumped on every occupied↔vacant transition, and every handle
//! carries the generation it was issued with. A handle whose generation
//! no longer matches the slot's current one is caught as dead, not
//! aliased.
//!
//! # What this is for
//!
//! This is `mid-ecs`'s entity-ID allocator -- `World::spawn`/`despawn`
//! are a thin wrapper over this, see `docs/mid-ecs.md`. Deliberately
//! **does not store a value per slot** the way `slotmap::SlotMap<K, V>`
//! or `generational_arena::Arena<T>` do: real ECS implementations
//! (`hecs`, `bevy_ecs` -- checked their actual entity-allocation code,
//! not assumed) both roll their own value-less allocator rather than
//! reusing a generic value-storing slotmap crate, because entity
//! component data lives in per-component storage
//! ([`crate::SparseSet`] today, the Archetype Core later) keyed *by*
//! the entity, not stored *in* the allocator itself. Building the
//! heavier value-storing version here would solve a problem `mid-ecs`
//! doesn't have.
//!
//! # Design, verified against real `slotmap` 1.0.7 source, not assumed
//!
//! The core trick -- one `u32` generation counter per slot, where **even
//! means vacant and odd means occupied** -- is `slotmap`'s own real,
//! shipped design (`src/basic.rs`: `version: u32, // Even = vacant, odd
//! = occupied.`), not derived from a blog post or invented independently.
//! Bumping generation on *every* transition (not just on free) means a
//! stale key's generation can never coincidentally match the slot's
//! current one after a reuse, and it gives a free `is_occupied` check
//! (`generation % 2 == 1`) with no separate `bool` field needed.
//!
//! The free list itself needs no `u32::MAX`-style sentinel, for the same
//! reason `slotmap`'s doesn't: `free_head` pointing *past the end* of
//! `slots` (i.e. `free_head == slots.len()`) already means "nothing to
//! reuse, grow instead" for free, via a plain `Vec::get` returning
//! `None`. One divergence from `slotmap`, deliberate: it also unions
//! each slot's storage between the live value and the free-list pointer
//! (`unsafe`, to avoid paying for both at once). Nothing here stores a
//! value, so there's no such union to build in the first place -- this
//! stays plain, safe Rust throughout, no `unsafe`, matching
//! [`crate::SparseSet`]'s own precedent that raw-pointer tricks wait for
//! a real, profiled need rather than being built speculatively.
//!
//! Reuse order is LIFO (most-recently-freed slot is the next one
//! reused) -- also matching `slotmap`, and a reasonable default for
//! cache behavior (a just-freed slot's memory is more likely to still be
//! hot).
//!
//! `generation.wrapping_add(1)` on free, not a checked/panicking add:
//! copied deliberately from `slotmap`'s own choice. After exactly
//! 2^32 reuse cycles on one single slot, generation wraps back to a
//! value that could theoretically alias an ancient handle -- accepted
//! as-is here the same way `slotmap` accepts it in production, rather
//! than second-guessing a battle-tested crate's own considered trade-off.

use crate::sparse_set::SparseSetIndex;
use alloc::vec::Vec;

/// A handle from a [`GenerationalIndexAllocator`]. Valid until the slot
/// it points at is freed -- after that, every allocator method correctly
/// reports it as dead rather than aliasing whatever gets allocated into
/// that slot next.
///
/// Implements [`SparseSetIndex`] directly, so this can be used as a
/// [`crate::SparseSet`] key with no wrapper needed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GenerationalIndex {
    index: u32,
    generation: u32,
}

impl GenerationalIndex {
    /// The raw slot index. Not meaningful on its own without the
    /// matching generation -- use [`GenerationalIndexAllocator::is_alive`]
    /// to check validity, not this in isolation.
    #[inline]
    pub fn index(&self) -> u32 {
        self.index
    }

    /// The generation this handle was issued with. Always odd (see this
    /// module's doc comment) -- exposed mainly for debugging/FFI, not
    /// something callers need to interpret themselves.
    #[inline]
    pub fn generation(&self) -> u32 {
        self.generation
    }
}

impl SparseSetIndex for GenerationalIndex {
    #[inline]
    fn sparse_index(&self) -> u32 {
        self.index
    }
}

/// One slot's bookkeeping. No value storage -- see this module's doc
/// comment for why that's deliberate here, unlike `slotmap`/
/// `generational-arena`.
#[derive(Debug, Clone, Copy)]
struct Slot {
    /// Even = vacant, odd = occupied.
    generation: u32,
    /// Free-list link. Meaningful only while `generation` is even
    /// (vacant) -- while occupied this field is simply unused, not
    /// read by anything.
    next_free: u32,
}

/// Issues [`GenerationalIndex`] handles, recycling freed slots safely.
///
/// See this module's top-level doc comment for the full design
/// reasoning (verified against real `slotmap` source, not assumed).
pub struct GenerationalIndexAllocator {
    slots: Vec<Slot>,
    /// Index into `slots` of the next slot to reuse.
    /// `free_head == slots.len()` means "nothing free, grow instead" --
    /// no separate sentinel value needed, see module doc comment.
    free_head: u32,
    live_count: usize,
}

impl GenerationalIndexAllocator {
    /// Creates an allocator with nothing allocated yet.
    pub fn new() -> Self {
        Self {
            slots: Vec::new(),
            free_head: 0,
            live_count: 0,
        }
    }

    /// Creates an allocator pre-sized for `capacity` live handles before
    /// the next allocation past that would reallocate.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            slots: Vec::with_capacity(capacity),
            free_head: 0,
            live_count: 0,
        }
    }

    /// Number of currently-live (allocated, not yet freed) handles.
    #[inline]
    pub fn len(&self) -> usize {
        self.live_count
    }

    /// True if nothing is currently allocated.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.live_count == 0
    }

    /// Total slots ever created (live + freed-but-not-yet-reused). Not
    /// the same as [`len`](Self::len) once anything has been freed.
    #[inline]
    pub fn slot_count(&self) -> usize {
        self.slots.len()
    }

    /// Issues a new handle -- either reusing the most recently freed
    /// slot (LIFO) or growing by one if nothing is free.
    pub fn allocate(&mut self) -> GenerationalIndex {
        let free_head = self.free_head;

        if let Some(slot) = self.slots.get_mut(free_head as usize) {
            // Reusing a freed slot. `| 1`, not `+ 1`: the slot's
            // generation is guaranteed even (vacant) here, so both would
            // give the same result, but `| 1` states the actual intent
            // -- force it odd -- directly, matching slotmap's own choice
            // rather than leaning on the even-plus-one-is-odd identity.
            let generation = slot.generation | 1;
            self.free_head = slot.next_free;
            slot.generation = generation;
            self.live_count += 1;
            GenerationalIndex { index: free_head, generation }
        } else {
            // Free list exhausted -- grow by one.
            debug_assert_eq!(
                free_head as usize,
                self.slots.len(),
                "free_head should never point past a single new slot beyond the end"
            );
            debug_assert!(
                self.slots.len() < u32::MAX as usize,
                "GenerationalIndexAllocator holds u32::MAX slots -- index would overflow"
            );
            let generation = 1;
            self.slots.push(Slot { generation, next_free: 0 });
            self.free_head = free_head + 1;
            self.live_count += 1;
            GenerationalIndex { index: free_head, generation }
        }
    }

    /// Frees `key`'s slot, if it's still alive. Returns whether it
    /// actually was -- freeing an already-dead or never-issued handle is
    /// a safe no-op, not a panic, so a caller holding a possibly-stale
    /// handle doesn't need to check first.
    pub fn deallocate(&mut self, key: GenerationalIndex) -> bool {
        if !self.is_alive(key) {
            return false;
        }
        let index = key.index;
        let slot = &mut self.slots[index as usize];
        slot.next_free = self.free_head;
        // wrapping_add, not a checked add: see module doc comment --
        // matches slotmap's own considered choice, not reconsidered here.
        slot.generation = slot.generation.wrapping_add(1);
        self.free_head = index;
        self.live_count -= 1;
        true
    }

    /// Whether `key` still points at the slot it was issued for -- false
    /// if it's been freed (and possibly reused) since, or was never a
    /// real handle from this allocator at all (out-of-range index).
    #[inline]
    pub fn is_alive(&self, key: GenerationalIndex) -> bool {
        match self.slots.get(key.index as usize) {
            Some(slot) => slot.generation == key.generation,
            None => false,
        }
    }
}

impl Default for GenerationalIndexAllocator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn starts_empty() {
        let a = GenerationalIndexAllocator::new();
        assert_eq!(a.len(), 0);
        assert!(a.is_empty());
        assert_eq!(a.slot_count(), 0);
    }

    #[test]
    fn first_allocated_index_is_zero() {
        // free_head starts at 0, which must not off-by-one the very
        // first handle ever issued -- 0 is a real, valid index here,
        // same paranoia as SparseSet's own "index zero is valid" test.
        let mut a = GenerationalIndexAllocator::new();
        let k = a.allocate();
        assert_eq!(k.index(), 0);
        assert_eq!(k.generation(), 1, "first generation on a fresh slot must be odd (occupied), i.e. 1");
    }

    #[test]
    fn allocate_returns_distinct_indices_when_nothing_freed() {
        let mut a = GenerationalIndexAllocator::new();
        let ks: Vec<GenerationalIndex> = (0..10).map(|_| a.allocate()).collect();
        let mut indices: Vec<u32> = ks.iter().map(|k| k.index()).collect();
        indices.sort_unstable();
        indices.dedup();
        assert_eq!(indices.len(), 10, "all ten indices must be distinct");
        assert_eq!(a.len(), 10);
    }

    #[test]
    fn allocated_handle_is_alive() {
        let mut a = GenerationalIndexAllocator::new();
        let k = a.allocate();
        assert!(a.is_alive(k));
    }

    #[test]
    fn deallocate_then_not_alive() {
        let mut a = GenerationalIndexAllocator::new();
        let k = a.allocate();
        assert!(a.deallocate(k));
        assert!(!a.is_alive(k));
        assert_eq!(a.len(), 0);
    }

    #[test]
    fn deallocate_already_dead_returns_false_not_panic() {
        let mut a = GenerationalIndexAllocator::new();
        let k = a.allocate();
        assert!(a.deallocate(k));
        assert!(!a.deallocate(k), "second deallocate of the same handle must be a safe no-op");
    }

    #[test]
    fn deallocate_never_issued_handle_returns_false() {
        let mut a = GenerationalIndexAllocator::new();
        a.allocate(); // slot 0 exists, but...
        let fake = GenerationalIndex { index: 999, generation: 1 };
        assert!(!a.deallocate(fake));
        assert!(!a.is_alive(fake));
    }

    #[test]
    fn is_alive_out_of_range_index_is_false_not_panic() {
        let a = GenerationalIndexAllocator::new();
        let fake = GenerationalIndex { index: 0, generation: 1 };
        assert!(!a.is_alive(fake));
    }

    #[test]
    fn reallocate_reuses_freed_slot_with_bumped_generation() {
        // The actual correctness promise of this whole structure: a
        // stale handle from before the reuse must read as dead, even
        // though the raw index got reused.
        let mut a = GenerationalIndexAllocator::new();
        let first = a.allocate();
        assert!(a.deallocate(first));

        let second = a.allocate();
        assert_eq!(
            second.index(),
            first.index(),
            "the freed slot should be the one reused, not a fresh one"
        );
        assert_ne!(
            second.generation(),
            first.generation(),
            "the reused slot must carry a different generation"
        );
        assert!(a.is_alive(second));
        assert!(!a.is_alive(first), "the stale first handle must not alias the reused slot");
    }

    #[test]
    fn generation_climbs_by_one_on_every_transition() {
        let mut a = GenerationalIndexAllocator::new();
        let g1 = a.allocate().generation();
        assert_eq!(g1, 1);
        let k = GenerationalIndex { index: 0, generation: g1 };
        a.deallocate(k);
        let g2 = a.allocate().generation();
        assert_eq!(g2, 3, "vacant(0->1 occupied)->deallocate(1->2 vacant)->allocate(2->3 occupied)");
    }

    #[test]
    fn free_list_reuse_order_is_lifo() {
        let mut a = GenerationalIndexAllocator::new();
        let k0 = a.allocate();
        let k1 = a.allocate();
        let k2 = a.allocate();

        assert!(a.deallocate(k0));
        assert!(a.deallocate(k1));
        assert!(a.deallocate(k2));

        // Most-recently-freed (k2's slot) should come back first.
        let r1 = a.allocate();
        let r2 = a.allocate();
        let r3 = a.allocate();
        assert_eq!(r1.index(), k2.index());
        assert_eq!(r2.index(), k1.index());
        assert_eq!(r3.index(), k0.index());
    }

    #[test]
    fn slot_count_tracks_total_slots_not_just_live() {
        let mut a = GenerationalIndexAllocator::new();
        let k0 = a.allocate();
        a.allocate();
        a.allocate();
        assert_eq!(a.slot_count(), 3);
        a.deallocate(k0);
        assert_eq!(a.slot_count(), 3, "freeing doesn't shrink slot_count, the slot still exists");
        assert_eq!(a.len(), 2);
        a.allocate(); // reuses k0's freed slot
        assert_eq!(a.slot_count(), 3, "reuse shouldn't grow it either");
        assert_eq!(a.len(), 3);
    }

    #[test]
    fn many_allocate_deallocate_cycles_stay_consistent() {
        // Not a specific regression case -- just real exercise across a
        // mixed alloc/dealloc pattern, checking len() and is_alive()
        // stay correct throughout rather than only at the start/end.
        let mut a = GenerationalIndexAllocator::new();
        let mut live: Vec<GenerationalIndex> = Vec::new();

        for round in 0..50u32 {
            live.push(a.allocate());
            if round % 3 == 0 && !live.is_empty() {
                let dead = live.remove(0);
                assert!(a.deallocate(dead));
                assert!(!a.is_alive(dead));
            }
            assert_eq!(a.len(), live.len());
            for &k in &live {
                assert!(a.is_alive(k), "every handle still held live must read as alive");
            }
        }
    }

    #[test]
    fn default_matches_new() {
        let a = GenerationalIndexAllocator::default();
        assert!(a.is_empty());
    }

    #[test]
    fn implements_sparse_set_index() {
        let mut a = GenerationalIndexAllocator::new();
        let k = a.allocate();
        assert_eq!(k.sparse_index(), k.index());
    }

    #[test]
    fn usable_as_a_real_sparse_set_key() {
        // Not just a trait-bound compile check -- an actual SparseSet
        // keyed by real allocator-issued handles, exercising both
        // structures together the way mid-ecs's World eventually will.
        use crate::sparse_set::SparseSet;

        let mut allocator = GenerationalIndexAllocator::new();
        let mut healths: SparseSet<GenerationalIndex, u32> = SparseSet::new();

        let e1 = allocator.allocate();
        let e2 = allocator.allocate();
        healths.insert(e1, 100);
        healths.insert(e2, 80);

        allocator.deallocate(e1);
        let e3 = allocator.allocate(); // reuses e1's slot

        // The SparseSet doesn't know anything died -- it's still holding
        // e1's old entry under that raw index. This is exactly why
        // mid-ecs's real despawn path has to remove component data
        // itself using the entity handle *before* the allocator frees
        // it, not rely on the allocator to do that. Documented here as
        // the actual reason, not left implicit.
        assert_eq!(healths.get(e1), Some(&100));
        assert!(!allocator.is_alive(e1));
        assert_ne!(e3.generation(), e1.generation());
    }
}
