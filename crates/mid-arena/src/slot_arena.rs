// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-arena.md, section "slot_arena.rs"
// ============================================================================
//! Generational, value-storing slot arena: [`SlotArena<T>`] issues
//! [`ArenaKey`] handles that detect their own staleness, the same way
//! `mid_collections::GenerationalIndex` does, extended to actually own
//! a `T` per slot.
//!
//! # Why this exists next to `mid-collections`' `GenerationalIndex`,
//! not instead of it
//!
//! `mid-collections::generational_index` is deliberately value-less --
//! its own doc comment states the reasoning directly: `mid-ecs`'s
//! entity allocator has nowhere useful to put a value, because entity
//! component data lives in per-component storage (`SparseSet` today,
//! the Archetype Core later) keyed *by* the entity, not stored *in* the
//! allocator. That reasoning is still correct, and this module doesn't
//! relitigate it -- `World::spawn`/`despawn` should keep using
//! `GenerationalIndexAllocator`, unchanged.
//!
//! `SlotArena<T>` is for everything *else* that wants stable,
//! ABA-safe, generational value storage and doesn't already have a
//! `SparseSet` sitting one layer up: asset caches, DixScript AST
//! nodes, MSX path-command buffers, scripting object tables. Real
//! candidate consumers, not yet wired to any -- see `docs/mid-arena.md`
//! "Relationship to mid-collections' GenerationalIndex" for the honest
//! version of this (nothing in this workspace calls `SlotArena` yet).
//!
//! # Design, directly extending `GenerationalIndexAllocator`'s own
//! verified-against-real-`slotmap`-source algorithm
//!
//! Same even-vacant/odd-occupied generation trick, same LIFO free
//! list, same `free_head == slots.len()` past-the-end-means-grow
//! convention -- copied deliberately from that module rather than
//! re-derived, so the two allocators behave identically at the
//! bookkeeping level and differ only in whether a slot carries a `T`.
//!
//! The one real difference: `Slot<T>` has to be an enum here, not a
//! flat struct -- a vacant slot has nowhere to put `T`'s bit pattern
//! without either requiring `T: Default` (a real API restriction
//! `slotmap`/`generational-arena` don't impose on their callers) or
//! reaching for an unsafe union the way `slotmap` itself does
//! internally (checked directly against `slotmap` 1.0.7's real
//! `src/basic.rs` while building the value-less allocator this one
//! extends -- see that module's doc comment). Plain safe enum picked
//! here for the same reason `GenerationalIndex` itself stays
//! unsafe-free: matches this workspace's own established precedent
//! (`SparseSet` and `GenerationalIndexAllocator` both) that
//! raw-pointer/union tricks wait for a real, profiled need rather than
//! being built speculatively. A `compact` feature carrying the union
//! layout is planned, not built -- see `docs/mid-arena.md` "Feature
//! gates".
//!
//! Real benchmark grounding for the approach itself, not just the
//! algorithm: `docs/mid-arena.md`'s survey actually ran `slab`,
//! `slotmap`, and `generational-arena` (the three real crates that
//! share this exact Vec-with-freelist approach) head to head against
//! seven other approaches, N=100,000, on this project's own rustc-1.75
//! floor. All three land in the same competitive band across
//! insert/get/remove/iterate -- confirming this approach, not just this
//! specific algorithm, is the right general-purpose default rather than
//! an untested pick.
//!
//! # Growth: eager prefill-and-double, not lazy grow-by-one -- an
//! untested-locally change, shipped for a real CI measurement on
//! purpose
//!
//! [`new`](SlotArena::new)/[`with_capacity`](SlotArena::with_capacity)
//! used to just reserve raw `Vec` capacity and grow by pushing one
//! `Occupied` slot per insert past that. This module now instead
//! mirrors `generational-arena` 0.2.9's *real* growth strategy exactly
//! (re-verified against its actual source directly before writing this,
//! not recalled from an earlier pass): [`reserve`](SlotArena::reserve)
//! eagerly pre-fills `additional` fresh `Vacant` placeholder slots,
//! linked into a real free list, before any insert touches them --
//! `new()` calls it with `DEFAULT_CAPACITY = 4` (their real constant),
//! and [`insert`](SlotArena::insert)'s growth path
//! ([`insert_slow_path`](SlotArena::insert_slow_path)) doubles by
//! calling `reserve(current_len)` and retrying, instead of pushing
//! exactly one new slot.
//!
//! Why this matters, and why it's real: on `generational-arena`'s own
//! insert benchmark shape (fresh arena, N sequential inserts), this
//! means their `#[inline(never)]`-marked "grow" branch is only ever
//! called ~`log2(N)` times, not N times -- the other ~N calls all hit
//! their small, always-inlined "pop from an already-there free list"
//! path. This crate's *previous* growth model took the grow branch on
//! literally every call in that same benchmark shape, which is exactly
//! why an earlier attempt at just the `#[inline(never)]` annotation (no
//! growth-strategy change) measured as a real regression here -- see
//! `docs/mid-arena.md`'s `#[inline(never)]` writeup for that full,
//! corrected investigation, including the mistake in the original
//! version of it.
//!
//! **This specific change could not be verified locally before
//! shipping, unlike everything else in this crate.** A from-scratch
//! reimplementation of this same prefill-and-double strategy measured
//! 60-87% *slower* in this sandbox (rustc 1.75, non-statistical
//! `std::time::Instant` timing) across four repeated runs -- the
//! working theory (checkable arithmetic, not confirmed by
//! disassembly) is that eagerly writing every slot twice (once as a
//! placeholder, once at real-insert time) roughly doubles total
//! memory-write volume for the run, and that this sandbox's older
//! LLVM (bundled with rustc 1.75) doesn't auto-vectorize the tight,
//! regular prefill loop in `reserve()` the way whatever LLVM ships
//! with the real CI's rustc 1.98 plausibly does -- which would make
//! the "extra" writes cheap there and not here. Not confirmed either
//! way. Shipped anyway, specifically to get a real measurement on real
//! CI, because local testing has now produced a *second* real,
//! source-grounded technique that regresses in this exact sandbox
//! while the crates that actually use it measure faster on the real
//! runner -- see `docs/mid-arena.md`'s "Fixes and Problems" entry for
//! this file for the full reasoning and the open decision if this
//! doesn't pan out on real CI either.
//!
//! One real, observable behavior change from this: `slot_count()`
//! immediately after `with_capacity(n)`/`new()` now reports `n`/`4`
//! rather than `0` -- every pre-filled slot counts as "created," even
//! before anything is inserted into it. Documented here and in the
//! affected tests, not left as a silent surprise.

use alloc::vec::Vec;
use core::mem::replace;

/// A handle from a [`SlotArena`]. Detects its own staleness after the
/// slot it points at is freed and reused -- see this module's doc
/// comment for the shared design with
/// `mid_collections::GenerationalIndex`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ArenaKey {
    index: u32,
    generation: u32,
}

impl ArenaKey {
    /// Builds a handle from raw parts. Not exposed outside this crate
    /// -- only an arena's own `insert`/`iter` should ever mint a real
    /// one. Exists so `compact_slot_arena.rs` (behind the `compact`
    /// feature) can issue the exact same handle type `SlotArena` does,
    /// instead of duplicating it, since the two share nothing else --
    /// unused (and so `#[allow(dead_code)]`'d) when `compact` is off,
    /// same as `compact_slot_arena.rs` itself not existing in that
    /// build.
    #[inline]
    #[cfg_attr(not(feature = "compact"), allow(dead_code))]
    pub(crate) fn new(index: u32, generation: u32) -> Self {
        Self { index, generation }
    }

    /// The raw slot index. Not meaningful alone without the matching
    /// generation -- use [`SlotArena::contains`] to check validity.
    #[inline]
    pub fn index(&self) -> u32 {
        self.index
    }

    /// The generation this handle was issued with. Always odd (see
    /// this module's doc comment).
    #[inline]
    pub fn generation(&self) -> u32 {
        self.generation
    }

    /// Packs this handle into a single `u64` -- `index` in the low 32
    /// bits, `generation` in the high 32 bits. Same layout, same
    /// reasoning as `mid_collections::GenerationalIndex::as_ffi`, not
    /// independently invented -- both trace back to
    /// `slotmap::KeyData::as_ffi`'s real, shipped bit layout.
    #[inline]
    pub fn as_ffi(self) -> u64 {
        (u64::from(self.generation) << 32) | u64::from(self.index)
    }

    /// Reconstructs a handle from a `u64` produced by
    /// [`as_ffi`](Self::as_ffi). Safe on any input, including a value
    /// that never came from a real `as_ffi()` call -- every real
    /// operation still validates the handle's generation against the
    /// slot's actual current one, so a bogus `value` just reads back
    /// as "not present," never as an alias of a real live slot. Same
    /// safety argument as `GenerationalIndex::from_ffi`, unchanged
    /// here.
    #[inline]
    pub fn from_ffi(value: u64) -> Self {
        Self {
            index: (value & 0xFFFF_FFFF) as u32,
            generation: (value >> 32) as u32,
        }
    }
}

enum Slot<T> {
    Occupied { generation: u32, value: T },
    Vacant { generation: u32, next_free: u32 },
}

/// Generational, value-storing arena. See this module's doc comment
/// for the full design and why it exists next to
/// `mid_collections::GenerationalIndexAllocator`.
pub struct SlotArena<T> {
    slots: Vec<Slot<T>>,
    /// Index into `slots` of the next slot to reuse.
    /// `free_head == slots.len()` means "nothing free, grow instead" --
    /// same sentinel-free convention as `GenerationalIndexAllocator`.
    free_head: u32,
    live_count: usize,
}

/// Matches `generational-arena` 0.2.9's real constant of the same
/// name exactly -- see this module's doc comment on growth.
/// `pub(crate)` so `compact_slot_arena.rs` shares this exact value
/// rather than risking a second copy drifting out of sync with it.
pub(crate) const DEFAULT_CAPACITY: usize = 4;

impl<T> SlotArena<T> {
    /// Creates an arena, eagerly pre-filled with `DEFAULT_CAPACITY`
    /// free slots -- see this module's doc comment on growth for why.
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_CAPACITY)
    }

    /// Creates an arena, eagerly pre-filled with `capacity` free slots
    /// (at least 1) ready for the first `capacity` inserts to reuse
    /// without growing -- see this module's doc comment on growth.
    pub fn with_capacity(capacity: usize) -> Self {
        let mut arena = Self {
            slots: Vec::new(),
            free_head: 0,
            live_count: 0,
        };
        arena.reserve(capacity.max(1));
        arena
    }

    /// Eagerly pre-fills `additional` fresh `Vacant` slots, linked into
    /// a real free list ready for immediate reuse. Mirrors
    /// `generational-arena::Arena::reserve`'s real shape (re-verified
    /// against its actual source directly, not recalled) adapted to
    /// this arena's own `free_head == slots.len()` sentinel convention
    /// -- their version chains a new batch onto whatever was already on
    /// the free list via a stored `Option<usize>` head; this one is
    /// only ever called when the free list is already empty (from
    /// `with_capacity` on a fresh arena, or from `insert_slow_path`
    /// exactly when nothing is free), so the new batch's tail simply
    /// terminates at the new total length, which already means "empty"
    /// under this arena's own convention.
    fn reserve(&mut self, additional: usize) {
        let start = self.slots.len();
        let new_len = start + additional;
        debug_assert!(
            new_len < u32::MAX as usize,
            "SlotArena holds u32::MAX slots -- index would overflow"
        );
        self.slots.reserve_exact(additional);
        for i in start..new_len {
            let next_free = if i == new_len - 1 {
                new_len as u32
            } else {
                (i + 1) as u32
            };
            self.slots.push(Slot::Vacant {
                generation: 0,
                next_free,
            });
        }
        self.free_head = start as u32;
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

    /// Backing `Vec`'s raw allocated capacity. Since
    /// [`reserve`](Self::reserve) always fills exactly what it
    /// allocates, this now tracks [`slot_count`](Self::slot_count)
    /// closely in practice (not guaranteed exactly equal --
    /// `Vec::reserve_exact` is "at least this much," not "exactly this
    /// much") -- kept as a distinct method rather than merged into
    /// `slot_count` since they answer different questions (raw backing
    /// allocation vs. logical slots created) even though the numbers
    /// now usually agree.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.slots.capacity()
    }

    /// Total slots ever created (live + freed-but-not-yet-reused).
    /// **Changed by the growth-strategy update documented in this
    /// module's doc comment: this is no longer 0 right after
    /// `new()`/`with_capacity(n)`** -- those now eagerly pre-fill
    /// `DEFAULT_CAPACITY`/`n` slots immediately, so `slot_count()`
    /// reports that count right away, before any insert. Still not the
    /// same as [`len`](Self::len) once anything has been removed.
    #[inline]
    pub fn slot_count(&self) -> usize {
        self.slots.len()
    }

    /// Inserts `value`, returning a handle that can later
    /// [`get`](Self::get)/[`get_mut`](Self::get_mut)/[`remove`](Self::remove)
    /// it. Either reuses the most recently freed slot (LIFO) or grows
    /// by doubling if nothing is free -- see this module's doc comment
    /// on growth.
    ///
    /// Split into this fast, always-inlined common case and
    /// [`insert_slow_path`](Self::insert_slow_path) for the growth
    /// case -- real technique read directly from
    /// `generational-arena`/`typed-generational-arena`'s own source
    /// (`insert`/`try_insert` marked `#[inline]`, the growth branch
    /// pulled into a separate `#[inline(never)] fn insert_slow_path`).
    /// Unlike the first time this split was tried in this file (see
    /// `docs/mid-arena.md`), the growth strategy now actually makes the
    /// "grow" branch rare rather than universal, which is the whole
    /// premise this split needs to be worth anything -- see this
    /// module's doc comment on growth for the real reasoning.
    #[inline]
    pub fn insert(&mut self, value: T) -> ArenaKey {
        let free_head = self.free_head;

        if let Some(slot) = self.slots.get_mut(free_head as usize) {
            let (generation, next_free) = match slot {
                // `| 1`, not `+ 1`: the slot's generation is guaranteed
                // even (vacant) here, so both give the same result, but
                // `| 1` states the actual intent directly, matching
                // `GenerationalIndexAllocator::allocate`'s own choice.
                Slot::Vacant {
                    generation,
                    next_free,
                } => (*generation | 1, *next_free),
                Slot::Occupied { .. } => unreachable!(
                    "free_head must always point at a Vacant slot -- \
                     insert/remove are the only writers of free_head \
                     and both uphold this"
                ),
            };
            *slot = Slot::Occupied { generation, value };
            self.free_head = next_free;
            self.live_count += 1;
            ArenaKey {
                index: free_head,
                generation,
            }
        } else {
            self.insert_slow_path(value)
        }
    }

    /// The "grow" branch of [`insert`](Self::insert), pulled out and
    /// marked `#[inline(never)]` on purpose -- see this module's doc
    /// comment on growth for why this is now the *rare* case rather
    /// than, as in the first attempt at this split, the *only* case.
    /// Doubles by [`reserve`](Self::reserve)-ing `slots.len()` more
    /// slots (or 1, from empty), matching
    /// `generational-arena::Arena::insert_slow_path`'s real shape
    /// exactly, then retries -- the retry always takes the fast path
    /// above, since `reserve` just populated the free list.
    #[inline(never)]
    fn insert_slow_path(&mut self, value: T) -> ArenaKey {
        let additional = if self.slots.is_empty() {
            1
        } else {
            self.slots.len()
        };
        self.reserve(additional);
        self.insert(value)
    }

    /// Removes and returns the value at `key`, if it's still alive.
    /// Removing an already-dead or never-issued handle is a safe
    /// no-op returning `None`, matching
    /// `GenerationalIndexAllocator::deallocate`'s own no-panic
    /// contract -- a caller holding a possibly-stale handle doesn't
    /// need to check first.
    pub fn remove(&mut self, key: ArenaKey) -> Option<T> {
        let slot = self.slots.get_mut(key.index as usize)?;

        let is_match = match slot {
            Slot::Occupied { generation, .. } => *generation == key.generation,
            Slot::Vacant { .. } => false,
        };
        if !is_match {
            return None;
        }

        let next_free = self.free_head;
        let new_generation = match slot {
            // wrapping_add, not a checked add: matches
            // GenerationalIndexAllocator's own considered choice, see
            // this module's doc comment.
            Slot::Occupied { generation, .. } => generation.wrapping_add(1),
            Slot::Vacant { .. } => unreachable!("just checked is_match above"),
        };
        let old = replace(
            slot,
            Slot::Vacant {
                generation: new_generation,
                next_free,
            },
        );
        self.free_head = key.index;
        self.live_count -= 1;

        match old {
            Slot::Occupied { value, .. } => Some(value),
            Slot::Vacant { .. } => unreachable!("just replaced an Occupied slot"),
        }
    }

    /// Whether `key` still points at a live value.
    #[inline]
    pub fn contains(&self, key: ArenaKey) -> bool {
        self.get(key).is_some()
    }

    /// Immutable access to the value at `key`, or `None` if it's dead
    /// or was never a real handle from this arena.
    #[inline]
    pub fn get(&self, key: ArenaKey) -> Option<&T> {
        match self.slots.get(key.index as usize)? {
            Slot::Occupied { generation, value } if *generation == key.generation => Some(value),
            _ => None,
        }
    }

    /// Mutable access to the value at `key`, or `None` if it's dead or
    /// was never a real handle from this arena.
    #[inline]
    pub fn get_mut(&mut self, key: ArenaKey) -> Option<&mut T> {
        match self.slots.get_mut(key.index as usize)? {
            Slot::Occupied { generation, value } if *generation == key.generation => Some(value),
            _ => None,
        }
    }

    /// Iterates over every live `(key, &value)` pair. Not necessarily
    /// insertion order once anything has been removed and its slot
    /// reused (reuse is LIFO, see this module's doc comment) -- a
    /// straight index-order scan over the slot array, skipping vacant
    /// ones.
    pub fn iter(&self) -> impl Iterator<Item = (ArenaKey, &T)> {
        self.slots.iter().enumerate().filter_map(|(i, slot)| match slot {
            Slot::Occupied { generation, value } => Some((
                ArenaKey {
                    index: i as u32,
                    generation: *generation,
                },
                value,
            )),
            Slot::Vacant { .. } => None,
        })
    }

    /// Mutable counterpart to [`iter`](Self::iter).
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (ArenaKey, &mut T)> {
        self.slots
            .iter_mut()
            .enumerate()
            .filter_map(|(i, slot)| match slot {
                Slot::Occupied { generation, value } => Some((
                    ArenaKey {
                        index: i as u32,
                        generation: *generation,
                    },
                    value,
                )),
                Slot::Vacant { .. } => None,
            })
    }

    /// Drops every live value and resets to empty. Existing handles all
    /// read as dead afterward -- `slots` itself is cleared, so `get` on
    /// any old index misses outright, not just generation-mismatches.
    pub fn clear(&mut self) {
        self.slots.clear();
        self.free_head = 0;
        self.live_count = 0;
    }
}

impl<T> Default for SlotArena<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn starts_empty() {
        let a: SlotArena<u32> = SlotArena::new();
        assert_eq!(a.len(), 0);
        assert!(a.is_empty());
        assert_eq!(
            a.slot_count(),
            4,
            "new() eagerly pre-fills DEFAULT_CAPACITY slots now -- see this module's doc comment on growth"
        );
    }

    #[test]
    fn insert_get_roundtrip() {
        let mut a = SlotArena::new();
        let k = a.insert(42u32);
        assert_eq!(a.get(k), Some(&42));
        assert_eq!(a.len(), 1);
    }

    #[test]
    fn get_mut_writes_through() {
        let mut a = SlotArena::new();
        let k = a.insert(1u32);
        *a.get_mut(k).unwrap() = 2;
        assert_eq!(a.get(k), Some(&2));
    }

    #[test]
    fn remove_returns_value_and_frees_slot() {
        let mut a = SlotArena::new();
        let k = a.insert(7u32);
        assert_eq!(a.remove(k), Some(7));
        assert_eq!(a.get(k), None);
        assert!(a.is_empty());
    }

    #[test]
    fn remove_on_dead_or_unknown_handle_is_a_safe_no_op() {
        let mut a: SlotArena<u32> = SlotArena::new();
        let k = a.insert(1);
        a.remove(k);
        assert_eq!(
            a.remove(k),
            None,
            "removing the already-dead handle again must not panic"
        );

        let mut b: SlotArena<u32> = SlotArena::new();
        assert_eq!(
            b.remove(k),
            None,
            "a handle from an unrelated arena must not alias slot 0 here"
        );
    }

    #[test]
    fn reallocate_reuses_freed_slot_with_bumped_generation() {
        // The actual correctness promise of this whole structure -- same
        // property GenerationalIndexAllocator's own test of the same
        // name checks, extended here to prove the *value* is right too.
        let mut a = SlotArena::new();
        let first = a.insert(100u32);
        assert_eq!(a.remove(first), Some(100));

        let second = a.insert(200u32);
        assert_eq!(
            second.index(),
            first.index(),
            "the freed slot should be the one reused, not a fresh one"
        );
        assert_ne!(second.generation(), first.generation());
        assert_eq!(a.get(second), Some(&200));
        assert_eq!(
            a.get(first),
            None,
            "the stale first handle must not alias the new value"
        );
    }

    #[test]
    fn free_list_reuse_order_is_lifo() {
        let mut a = SlotArena::new();
        let k0 = a.insert('a');
        let k1 = a.insert('b');
        let k2 = a.insert('c');

        a.remove(k0);
        a.remove(k1);
        a.remove(k2);

        // Most-recently-freed (k2's slot) should come back first.
        let r1 = a.insert('x');
        let r2 = a.insert('y');
        let r3 = a.insert('z');
        assert_eq!(r1.index(), k2.index());
        assert_eq!(r2.index(), k1.index());
        assert_eq!(r3.index(), k0.index());
    }

    #[test]
    fn iterate_visits_every_live_value_and_skips_removed_ones() {
        let mut a = SlotArena::new();
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
        let mut a = SlotArena::new();
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
    fn clear_drops_values_and_invalidates_every_handle() {
        let mut a = SlotArena::new();
        let k0 = a.insert(1u32);
        let k1 = a.insert(2u32);
        a.clear();
        assert!(a.is_empty());
        assert_eq!(a.get(k0), None);
        assert_eq!(a.get(k1), None);
        // Reinserting after clear reuses index 0 fresh, at generation 1
        // again -- the slots Vec itself was cleared, not just marked
        // vacant.
        let k2 = a.insert(3u32);
        assert_eq!(k2.index(), 0);
        assert_eq!(k2.generation(), 1);
    }

    #[test]
    fn slot_count_tracks_total_slots_not_just_live() {
        // Rewritten for the eager-prefill growth strategy (see this
        // module's doc comment): slot_count is now set at construction
        // time by the capacity hint, not grown one at a time by insert.
        let mut a = SlotArena::with_capacity(3);
        assert_eq!(
            a.slot_count(),
            3,
            "with_capacity pre-fills its slots immediately now"
        );
        let k0 = a.insert(1u32);
        a.insert(2u32);
        a.insert(3u32);
        assert_eq!(
            a.slot_count(),
            3,
            "inserting within the pre-filled capacity doesn't grow it"
        );
        a.remove(k0);
        assert_eq!(a.slot_count(), 3, "freeing doesn't shrink slot_count");
        assert_eq!(a.len(), 2);
        a.insert(4u32); // reuses k0's freed slot
        assert_eq!(a.slot_count(), 3, "reuse shouldn't grow it either");
        assert_eq!(a.len(), 3);
        a.insert(5u32); // 4th live value, exceeds the initial capacity of 3
        assert!(
            a.slot_count() > 3,
            "exceeding the pre-filled capacity must grow slot_count"
        );
    }

    #[test]
    fn growth_doubles_by_prefilling_a_fresh_free_list_batch() {
        // The actual mechanism insert_slow_path now uses, matching
        // generational-arena's real reserve()/insert_slow_path (source
        // re-read directly before writing this, see this module's doc
        // comment on growth) -- checked directly rather than only
        // asserted in prose.
        let mut a: SlotArena<u32> = SlotArena::with_capacity(1);
        assert_eq!(a.slot_count(), 1);
        a.insert(1); // consumes the one pre-filled slot
        assert_eq!(a.slot_count(), 1);
        a.insert(2); // free list empty -> insert_slow_path -> reserve(1) -> doubles to 2
        assert_eq!(a.slot_count(), 2);
        a.insert(3); // free list empty again -> reserve(2) -> doubles to 4
        assert_eq!(a.slot_count(), 4);
        a.insert(4); // still within the 4 just reserved
        assert_eq!(a.slot_count(), 4);
        a.insert(5); // exceeds 4 -> reserve(4) -> doubles to 8
        assert_eq!(a.slot_count(), 8);
    }

    #[test]
    fn many_insert_remove_cycles_stay_consistent() {
        // Not a specific regression case -- real exercise across a mixed
        // insert/remove pattern, checking len()/get() stay correct
        // throughout, not just at the start/end.
        let mut a = SlotArena::new();
        let mut live: Vec<(ArenaKey, u32)> = Vec::new();

        for round in 0u32..50 {
            let k = a.insert(round);
            live.push((k, round));
            if round % 3 == 0 && !live.is_empty() {
                let (dead_key, dead_val) = live.remove(0);
                assert_eq!(a.remove(dead_key), Some(dead_val));
            }
            assert_eq!(a.len(), live.len());
            for &(k, v) in &live {
                assert_eq!(
                    a.get(k),
                    Some(&v),
                    "every handle still held live must read back its real value"
                );
            }
        }
    }

    #[test]
    fn default_matches_new() {
        let a: SlotArena<u32> = SlotArena::default();
        assert!(a.is_empty());
    }

    #[test]
    fn as_ffi_from_ffi_round_trips() {
        let mut a = SlotArena::new();
        let k = a.insert(9u32);
        let packed = k.as_ffi();
        let unpacked = ArenaKey::from_ffi(packed);
        assert_eq!(unpacked, k);
    }

    #[test]
    fn as_ffi_round_trips_after_reuse_with_a_different_generation() {
        let mut a = SlotArena::new();
        let first = a.insert(1u32);
        a.remove(first);
        let second = a.insert(2u32); // same index, different generation

        assert_eq!(ArenaKey::from_ffi(second.as_ffi()), second);
        assert_ne!(
            ArenaKey::from_ffi(first.as_ffi()),
            second,
            "the stale packed value must not round-trip into the live handle sharing its index"
        );
    }

    #[test]
    fn drop_runs_for_every_live_value_when_the_arena_itself_is_dropped() {
        // Runs Drop = a real, checked property here (unlike bumpalo's
        // mixed-type arena, which explicitly doesn't run it -- see
        // docs/mid-arena.md's comparison table) -- Vec<Slot<T>> owns
        // every T directly, so dropping the Vec drops them.
        use core::cell::Cell;

        struct DropCounter<'a>(&'a Cell<u32>);
        impl<'a> Drop for DropCounter<'a> {
            fn drop(&mut self) {
                self.0.set(self.0.get() + 1);
            }
        }

        let count = Cell::new(0u32);
        {
            let mut a = SlotArena::new();
            a.insert(DropCounter(&count));
            a.insert(DropCounter(&count));
            let k2 = a.insert(DropCounter(&count));
            a.remove(k2); // dropped immediately here
            assert_eq!(count.get(), 1);
        } // remaining two dropped here
        assert_eq!(count.get(), 3);
    }
}
