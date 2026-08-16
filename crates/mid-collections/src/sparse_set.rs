//! Sparse set: O(1) insert/remove/lookup, contiguous iteration over live
//! elements, no tombstones to skip.
//!
//! Two arrays, matching `docs/mid-collections.md`'s own description: a
//! **sparse** one indexed directly by key, pointing into a **dense** one
//! that holds the real data plus a back-pointer to the key it belongs to
//! (needed to fix up the sparse array after a swap-remove). The dense side
//! is implemented here as two parallel `Vec`s (`dense_keys`/`dense_values`)
//! rather than one `Vec<(I, T)>`, so a pure value iteration (`values()`,
//! `values_mut()` — the common case: "give me every `Poisoned` component
//! to tick down") never has to stride past key data it doesn't need. EnTT
//! and Bevy's own `SparseSet` both split it this way for the same reason.
//!
//! # What this is for
//!
//! This is the storage `mid-ecs`'s "Sparse Shell" is built on — the half
//! of the Hybrid ECS Architecture (`docs/mid-ecs.md`) that handles
//! volatile, frequently-toggled components (status effects, tags,
//! anything added and removed constantly) without the archetype-migration
//! cost the "Archetype Core" would pay for the same churn. Bevy ECS V2
//! hit the identical fork and landed on the same hybrid answer for the
//! same reason — their own docs use a `Bleeding` status effect as the
//! textbook case for sparse-set storage, "to not fragment tables."
//!
//! # Design decisions made building this, and why
//!
//! - **No paging.** EnTT's real `sparse_set.hpp` (checked directly, not
//!   assumed) allocates its sparse array in lazily-allocated fixed-size
//!   pages rather than one flat array, specifically to stay cheap when
//!   the entity ID space is arbitrarily large. Mid Engine's stated target
//!   is 100 000+ entities (`docs/architecture.md`), not the
//!   many-millions-of-sparse-IDs case paging is solving for — at this
//!   scale a flat, growable `Vec<u32>` sparse array costs at most a few
//!   hundred KB per component type, which doesn't justify paging's extra
//!   complexity. Revisit if a real use case needs a much larger sparse ID
//!   space than this.
//! - **`u32::MAX` sentinel, not `Option<u32>`, for empty sparse slots.**
//!   Both EnTT (a dedicated `null` entity value) and Bevy (explored a
//!   hand-rolled `NonMaxUsize` specifically for this, see bevyengine/bevy
//!   PR #2104) avoid `Option<u32>` here. `Option<u32>` isn't
//!   niche-optimized in Rust (`u32` has no spare bit pattern the way
//!   `NonZeroU32` does), so it costs a discriminant plus padding — twice
//!   the size of a bare sentinel `u32` for no runtime benefit. Worth
//!   being honest about the runtime cost, not just the memory one: Bevy's
//!   own benchmark on the equivalent change was inconclusive, mostly
//!   noise. The memory case is the one that's clear-cut, and it's the one
//!   that actually matters here — "zero-copy, minimize RAM-to-RAM
//!   movement" is one of this project's own stated mandates
//!   (`docs/architecture.md`).
//! - **`swap_and_pop` only, no tombstones.** EnTT supports three deletion
//!   policies (`swap_and_pop`, `in_place` tombstones, `swap_only`) for
//!   cases like preserving iteration order across removals. Nothing in
//!   `mid-ecs` needs that yet — the Sparse Shell's whole job is components
//!   that come and go arbitrarily, where iteration order was never a
//!   guarantee to begin with. Add a policy if and when something actually
//!   needs one; matches this whole doc's "build what's needed, not what's
//!   possible" discipline.
//! - **Generic over a `SparseSetIndex` key, not tied to any `Entity`
//!   type.** The generational-arena work that will define what an actual
//!   entity handle looks like (`docs/mid-collections.md`, ranked directly
//!   above this doc's older draft of this structure) hasn't been built
//!   yet. This mirrors Bevy's own real `SparseSetIndex` trait
//!   (`bevy_ecs::storage`) for the identical reason: the storage
//!   shouldn't need to know what an entity is, only how to get a raw
//!   index out of one. Whatever entity handle mid-ecs ends up with just
//!   needs to implement this trait once it exists.
//! - **`insert` replaces on collision instead of asserting.** EnTT's
//!   `push`/`try_emplace` is UB if you insert a key that's already
//!   present — the caller is required to know. Rust's own `HashMap`
//!   convention (return the old value, replace in place) is safer by
//!   construction and costs nothing extra here, so that's what this
//!   follows instead of blindly porting EnTT's contract.

use alloc::vec::Vec;
use core::fmt;

/// Marks an empty slot in the sparse array. See this module's doc comment
/// for why a sentinel instead of `Option<u32>`.
const EMPTY: u32 = u32::MAX;

/// A key type that can be stored in a [`SparseSet`].
///
/// Deliberately minimal — just "give me a raw `u32` index" — so the
/// sparse set itself never has to know anything about entities,
/// generations, or versioning. A future `Entity` handle (once the
/// generational-arena piece exists) implements this by returning its raw
/// index field; the generation itself is the arena's problem, not this
/// structure's.
///
/// `u32` rather than `usize`: Bevy made the identical call for the
/// identical reason (`bevyengine/bevy` PR #4723) — halves the memory cost
/// of every sparse slot and back-pointer, and no ECS this project is
/// aiming for needs more than ~4.29 billion live indices. Mid Engine's
/// own target is 100 000+ (`docs/architecture.md`), nowhere close.
pub trait SparseSetIndex: Copy {
    /// The raw dense-array index this key maps to. Must not return
    /// `u32::MAX` — that value is reserved as the empty-slot sentinel.
    fn sparse_index(&self) -> u32;
}

impl SparseSetIndex for u32 {
    #[inline]
    fn sparse_index(&self) -> u32 {
        *self
    }
}

/// Sparse-set storage keyed by anything implementing [`SparseSetIndex`].
///
/// See this module's top-level doc comment for the full design reasoning.
/// In short: O(1) insert/remove/lookup, and iterating [`values`](Self::values)
/// or [`iter`](Self::iter) is a straight scan over contiguous memory — no
/// gaps, no tombstones to skip.
pub struct SparseSet<I: SparseSetIndex, T> {
    /// Indexed by `key.sparse_index() as usize`. Holds the position of
    /// that key's data in `dense_keys`/`dense_values`, or `EMPTY`.
    sparse: Vec<u32>,
    /// Parallel to `dense_values`. The back-pointer needed to fix up
    /// `sparse` when a swap-remove moves the last element into a hole.
    dense_keys: Vec<I>,
    /// Parallel to `dense_keys`. The actual stored data, packed
    /// contiguously with no gaps.
    dense_values: Vec<T>,
}

impl<I: SparseSetIndex, T> SparseSet<I, T> {
    /// Creates an empty sparse set. Allocates nothing until the first
    /// [`insert`](Self::insert).
    pub fn new() -> Self {
        Self {
            sparse: Vec::new(),
            dense_keys: Vec::new(),
            dense_values: Vec::new(),
        }
    }

    /// Creates an empty sparse set with room for `capacity` dense
    /// elements before the next insert reallocates.
    ///
    /// Only pre-sizes the dense side. The sparse array grows lazily,
    /// keyed off the largest index actually inserted — pre-sizing it here
    /// would require knowing the key range up front, which callers
    /// usually don't.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            sparse: Vec::new(),
            dense_keys: Vec::with_capacity(capacity),
            dense_values: Vec::with_capacity(capacity),
        }
    }

    /// Number of elements currently stored.
    #[inline]
    pub fn len(&self) -> usize {
        self.dense_values.len()
    }

    /// True if nothing is stored.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.dense_values.is_empty()
    }

    /// Dense-side capacity — how many elements can be inserted before the
    /// next reallocation.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.dense_values.capacity()
    }

    /// Returns the dense position for `key`, if present.
    #[inline]
    fn dense_index_of(&self, key: I) -> Option<usize> {
        let index = key.sparse_index() as usize;
        match self.sparse.get(index) {
            Some(&slot) if slot != EMPTY => Some(slot as usize),
            _ => None,
        }
    }

    /// Whether `key` currently has a value stored.
    #[inline]
    pub fn contains(&self, key: I) -> bool {
        self.dense_index_of(key).is_some()
    }

    /// Looks up the value for `key`, if present.
    #[inline]
    pub fn get(&self, key: I) -> Option<&T> {
        self.dense_index_of(key).map(|pos| &self.dense_values[pos])
    }

    /// Looks up the value for `key` mutably, if present.
    #[inline]
    pub fn get_mut(&mut self, key: I) -> Option<&mut T> {
        let pos = self.dense_index_of(key)?;
        Some(&mut self.dense_values[pos])
    }

    /// Inserts `value` under `key`. If `key` was already present, its
    /// value is replaced in place (no structural change, no effect on
    /// dense ordering) and the old value is returned — matching
    /// `HashMap::insert`'s convention rather than asserting the key must
    /// be absent, see this module's doc comment for why.
    pub fn insert(&mut self, key: I, value: T) -> Option<T> {
        let raw = key.sparse_index();
        debug_assert!(
            raw != EMPTY,
            "SparseSetIndex::sparse_index() returned u32::MAX, which is reserved as the empty-slot sentinel"
        );
        let index = raw as usize;

        if index >= self.sparse.len() {
            self.sparse.resize(index + 1, EMPTY);
        }

        let slot = self.sparse[index];
        if slot != EMPTY {
            Some(core::mem::replace(&mut self.dense_values[slot as usize], value))
        } else {
            debug_assert!(
                self.dense_values.len() < EMPTY as usize,
                "SparseSet holds u32::MAX elements -- dense position would collide with the empty sentinel"
            );
            let dense_pos = self.dense_values.len() as u32;
            self.sparse[index] = dense_pos;
            self.dense_keys.push(key);
            self.dense_values.push(value);
            None
        }
    }

    /// Removes `key`'s value, if present, and returns it. O(1): swaps the
    /// removed element with whatever's last in the dense arrays and pops,
    /// rather than shifting everything after it the way `Vec::remove`
    /// would. This is why iteration order isn't guaranteed to match
    /// insertion order — see this module's doc comment.
    pub fn remove(&mut self, key: I) -> Option<T> {
        let index = key.sparse_index() as usize;
        let slot = *self.sparse.get(index)?;
        if slot == EMPTY {
            return None;
        }

        let removed_pos = slot as usize;
        let last_pos = self.dense_values.len() - 1;

        self.dense_keys.swap(removed_pos, last_pos);
        self.dense_values.swap(removed_pos, last_pos);

        // Whatever now sits at `removed_pos` needs its sparse entry
        // repointed to its new position. If we just removed the last
        // element, this is a harmless self-write to `key`'s own slot,
        // overwritten by EMPTY on the next line either way.
        let moved_key = self.dense_keys[removed_pos];
        self.sparse[moved_key.sparse_index() as usize] = removed_pos as u32;

        self.sparse[index] = EMPTY;
        self.dense_keys.pop();
        self.dense_values.pop()
    }

    /// Removes every element. Only walks the dense side to reset sparse
    /// slots (O(len), not O(sparse capacity)) — the sparse array stays
    /// allocated at its current extent so later re-inserts of
    /// previously-seen keys don't have to regrow it.
    pub fn clear(&mut self) {
        for key in self.dense_keys.drain(..) {
            self.sparse[key.sparse_index() as usize] = EMPTY;
        }
        self.dense_values.clear();
    }

    /// Iterates every live key, in dense (not insertion) order.
    pub fn keys(&self) -> impl Iterator<Item = I> + '_ {
        self.dense_keys.iter().copied()
    }

    /// Iterates every live value by reference — a straight contiguous
    /// scan, no gaps.
    pub fn values(&self) -> impl Iterator<Item = &T> + '_ {
        self.dense_values.iter()
    }

    /// Iterates every live value by mutable reference.
    pub fn values_mut(&mut self) -> impl Iterator<Item = &mut T> + '_ {
        self.dense_values.iter_mut()
    }

    /// Iterates `(key, &value)` pairs, in dense order.
    pub fn iter(&self) -> impl Iterator<Item = (I, &T)> + '_ {
        self.dense_keys.iter().copied().zip(self.dense_values.iter())
    }

    /// Iterates `(key, &mut value)` pairs, in dense order.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (I, &mut T)> + '_ {
        self.dense_keys.iter().copied().zip(self.dense_values.iter_mut())
    }
}

impl<I: SparseSetIndex, T> Default for SparseSet<I, T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<I: SparseSetIndex + fmt::Debug, T: fmt::Debug> fmt::Debug for SparseSet<I, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map().entries(self.iter()).finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn starts_empty() {
        let s: SparseSet<u32, u32> = SparseSet::new();
        assert_eq!(s.len(), 0);
        assert!(s.is_empty());
        assert!(!s.contains(0));
        assert_eq!(s.get(0), None);
    }

    #[test]
    fn insert_then_get() {
        let mut s = SparseSet::new();
        assert_eq!(s.insert(3u32, "three"), None);
        assert_eq!(s.get(3), Some(&"three"));
        assert!(s.contains(3));
        assert_eq!(s.len(), 1);
    }

    #[test]
    fn index_zero_is_a_valid_key() {
        // EMPTY is u32::MAX, not 0 -- index 0 must work like any other,
        // unlike designs that wrongly reserve 0 as a null sentinel.
        let mut s = SparseSet::new();
        s.insert(0u32, "zero");
        assert_eq!(s.get(0), Some(&"zero"));
        assert!(s.contains(0));
    }

    #[test]
    fn insert_on_existing_key_replaces_and_returns_old() {
        let mut s = SparseSet::new();
        s.insert(5u32, 100);
        let old = s.insert(5u32, 200);
        assert_eq!(old, Some(100));
        assert_eq!(s.get(5), Some(&200));
        assert_eq!(s.len(), 1, "replacing must not grow the dense array");
    }

    #[test]
    fn get_missing_key_returns_none() {
        let mut s: SparseSet<u32, i32> = SparseSet::new();
        s.insert(1, 10);
        assert_eq!(s.get(999), None);
        assert_eq!(s.get(0), None);
    }

    #[test]
    fn get_mut_actually_mutates() {
        let mut s = SparseSet::new();
        s.insert(2u32, 10);
        *s.get_mut(2).unwrap() += 5;
        assert_eq!(s.get(2), Some(&15));
    }

    #[test]
    fn remove_missing_key_returns_none_and_changes_nothing() {
        let mut s = SparseSet::new();
        s.insert(1u32, "a");
        assert_eq!(s.remove(999), None);
        assert_eq!(s.len(), 1);
        assert_eq!(s.remove(0), None, "never-inserted index below any real key");
    }

    #[test]
    fn remove_last_element() {
        let mut s = SparseSet::new();
        s.insert(1u32, "a");
        s.insert(2u32, "b");
        s.insert(3u32, "c");
        assert_eq!(s.remove(3), Some("c"));
        assert_eq!(s.len(), 2);
        assert!(!s.contains(3));
        assert_eq!(s.get(1), Some(&"a"));
        assert_eq!(s.get(2), Some(&"b"));
    }

    #[test]
    fn remove_middle_element_swaps_and_fixes_up_the_moved_key() {
        // The real correctness case for swap_and_pop: removing a
        // non-last element must fix up the sparse slot of whatever got
        // swapped into its place, not just pop and forget.
        let mut s = SparseSet::new();
        s.insert(10u32, "A");
        s.insert(20u32, "B");
        s.insert(30u32, "C");
        s.insert(40u32, "D");

        // dense order: [A, B, C, D]. Removing key 20 ("B", dense pos 1)
        // should swap D (dense pos 3, the last) into pos 1.
        assert_eq!(s.remove(20), Some("B"));

        assert_eq!(s.len(), 3);
        assert!(!s.contains(20));
        assert_eq!(s.get(10), Some(&"A"));
        assert_eq!(s.get(30), Some(&"C"));
        assert_eq!(s.get(40), Some(&"D"), "D must still be reachable after being moved");

        // Every remaining key's sparse entry must point at a dense
        // position that actually holds that key -- not just "some
        // position that happens to still be in bounds".
        for key in [10u32, 30, 40] {
            let pos = s.dense_index_of(key).unwrap();
            assert_eq!(s.dense_keys[pos], key);
        }

        // Iteration must yield exactly the surviving three, each with
        // its correct value, and nothing from the removed slot.
        let mut pairs: Vec<_> = s.iter().collect();
        pairs.sort_by_key(|&(k, _)| k);
        assert_eq!(pairs, vec![(10, &"A"), (30, &"C"), (40, &"D")]);
    }

    #[test]
    fn remove_first_element_repeatedly_drains_correctly() {
        let mut s = SparseSet::new();
        for i in 0..10u32 {
            s.insert(i, i * 10);
        }
        for i in 0..10u32 {
            assert_eq!(s.remove(i), Some(i * 10));
            assert_eq!(
                s.remove(i),
                None,
                "removing the same key a second time must be a no-op, not remove whatever swapped into its old slot"
            );
        }
        assert!(s.is_empty());
    }

    #[test]
    fn reinsert_after_remove_reuses_the_freed_sparse_slot() {
        let mut s = SparseSet::new();
        s.insert(7u32, "first");
        s.remove(7);
        assert!(!s.contains(7));
        assert_eq!(s.insert(7, "second"), None);
        assert_eq!(s.get(7), Some(&"second"));
        assert_eq!(s.len(), 1);
    }

    #[test]
    fn sparse_grows_lazily_for_large_sparse_indices() {
        let mut s = SparseSet::new();
        s.insert(100_000u32, "far");
        assert_eq!(s.get(100_000), Some(&"far"));
        assert_eq!(s.len(), 1, "one dense element regardless of how sparse the key space is");
        assert!(!s.contains(50_000), "untouched slots in between must read as absent");
    }

    #[test]
    fn clear_empties_and_resets_all_sparse_slots() {
        let mut s = SparseSet::new();
        for i in 0..5u32 {
            s.insert(i, i);
        }
        s.clear();
        assert!(s.is_empty());
        for i in 0..5u32 {
            assert!(!s.contains(i));
        }
        // Reinserting after clear must behave like a fresh insert.
        assert_eq!(s.insert(2, 99), None);
        assert_eq!(s.get(2), Some(&99));
    }

    #[test]
    fn values_and_values_mut_are_contiguous_no_gaps() {
        let mut s = SparseSet::new();
        s.insert(1u32, 1);
        s.insert(2u32, 2);
        s.insert(3u32, 3);
        s.remove(2);

        let mut vals: Vec<_> = s.values().copied().collect();
        vals.sort_unstable();
        assert_eq!(vals, vec![1, 3], "no gap/tombstone left where the removed element was");

        for v in s.values_mut() {
            *v *= 10;
        }
        let mut vals: Vec<_> = s.values().copied().collect();
        vals.sort_unstable();
        assert_eq!(vals, vec![10, 30]);
    }

    #[test]
    fn keys_matches_iter_and_values_pairwise() {
        let mut s = SparseSet::new();
        s.insert(5u32, "e");
        s.insert(6u32, "f");
        s.insert(7u32, "g");

        let from_keys: Vec<u32> = s.keys().collect();
        let from_iter: Vec<u32> = s.iter().map(|(k, _)| k).collect();
        assert_eq!(from_keys, from_iter);

        // and iter's values must be positionally consistent with values()
        let iter_vals: Vec<_> = s.iter().map(|(_, v)| *v).collect();
        let vals: Vec<_> = s.values().copied().collect();
        assert_eq!(iter_vals, vals);
    }

    #[test]
    fn default_matches_new() {
        let s: SparseSet<u32, i32> = SparseSet::default();
        assert!(s.is_empty());
    }

    #[test]
    fn debug_impl_does_not_panic_and_lists_entries() {
        let mut s = SparseSet::new();
        s.insert(1u32, "a");
        let text = alloc::format!("{s:?}");
        assert!(text.contains('1'));
        assert!(text.contains('a'));
    }

    /// A minimal stand-in for what a future generational `Entity` handle
    /// might look like, to prove `SparseSetIndex` genuinely decouples
    /// this structure from `u32` keys specifically -- not just compiling
    /// against the trait, but a second concrete type exercising the same
    /// insert/remove/iterate paths.
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    struct FakeEntity {
        index: u32,
        #[allow(dead_code)]
        generation: u16,
    }

    impl SparseSetIndex for FakeEntity {
        fn sparse_index(&self) -> u32 {
            self.index
        }
    }

    #[test]
    fn works_with_a_non_u32_key_type() {
        let mut s: SparseSet<FakeEntity, &str> = SparseSet::new();
        let e1 = FakeEntity { index: 3, generation: 1 };
        let e2 = FakeEntity { index: 8, generation: 4 };

        s.insert(e1, "one");
        s.insert(e2, "two");
        assert_eq!(s.get(e1), Some(&"one"));
        assert_eq!(s.remove(e1), Some("one"));
        assert_eq!(s.get(e2), Some(&"two"));
        assert!(!s.contains(e1));
    }
}
