// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-arena.md, section "compact_slot_arena.rs"
// ============================================================================
//! Union-based generational slot arena: [`CompactSlotArena<T>`] is the
//! same idea as [`SlotArena`](crate::SlotArena), same [`ArenaKey`]
//! handle type, same even/odd generation LIFO-freelist algorithm --
//! ported to the `union`-based layout `slotmap` actually uses
//! internally (real source read: `slotmap` 1.0.7's `src/basic.rs`, not
//! reimplemented from a description of what a compact slot map
//! "probably" looks like).
//!
//! Behind the `compact` feature, and deliberately a separate type from
//! `SlotArena`, not a `#[cfg]`-swapped internal representation of it.
//! Cargo features are supposed to be purely additive -- if enabling
//! `compact` silently changed what `SlotArena` compiles to, any other
//! crate in the same build that enables the feature for its own reasons
//! would change this crate's behavior out from under a consumer who
//! never asked for it (Cargo's own feature unification makes this a
//! real risk, not a hypothetical one). A second, honestly-named type
//! avoids that entirely.
//!
//! # Why this exists (see docs/mid-arena.md "Feature gates" for the
//! full history)
//!
//! Originally justified in this crate's docs by an insert-time gap
//! between `SlotArena` and its peers that a real CI run later showed
//! wasn't actually there -- sandbox noise, not a real finding. The
//! honest justification now is memory footprint, not speed:
//! `SlotArena`'s `Slot<T>` enum needs a discriminant, which happens to
//! fit inside `T`'s own alignment padding for free when `T` is large
//! enough to have any (checked directly for the crate's own benchmark
//! payload, `docs/mid-arena.md`'s real CI section) but doesn't for
//! every `T`. A union has no discriminant to place at all -- `Vacant`'s
//! `next_free: u32` and `Occupied`'s `value` genuinely share the same
//! bytes, at the cost of the `unsafe` this module carries to make that
//! sound.
//!
//! # The real design, ported directly from `slotmap::basic::Slot`
//!
//! ```text
//! union SlotUnion<T> { value: ManuallyDrop<T>, next_free: u32 }
//! struct Slot<T> { u: SlotUnion<T>, generation: u32 }
//! ```
//!
//! Same even-vacant/odd-occupied convention `SlotArena` and
//! `mid_collections::GenerationalIndex` both already use, so all three
//! types in this workspace agree on what a generation number means.
//! `Drop for Slot<T>` only runs `ManuallyDrop::drop` when the slot is
//! occupied -- `slotmap`'s own real `Drop` impl does the same,
//! including its `needs_drop::<T>()` short-circuit for types that don't
//! need dropping at all, kept here for the same reason: skip the check
//! entirely rather than pay a branch that a `T: Copy` (for example)
//! payload can never actually take.

use alloc::vec::Vec;
use core::mem::ManuallyDrop;

use crate::slot_arena::ArenaKey;

union SlotUnion<T> {
    value: ManuallyDrop<T>,
    next_free: u32,
}

struct Slot<T> {
    u: SlotUnion<T>,
    generation: u32,
}

impl<T> Slot<T> {
    #[inline(always)]
    fn occupied(&self) -> bool {
        self.generation % 2 > 0
    }
}

impl<T> Drop for Slot<T> {
    fn drop(&mut self) {
        if core::mem::needs_drop::<T>() && self.occupied() {
            // SAFETY: occupied() confirms the union's live field right
            // now is `value`, not `next_free` -- matches
            // `slotmap::basic::Slot`'s own `Drop` impl exactly, not
            // reordered or altered.
            unsafe {
                ManuallyDrop::drop(&mut self.u.value);
            }
        }
    }
}

/// Union-based generational slot arena. See this module's doc comment
/// for the full design and why it exists next to `SlotArena`.
pub struct CompactSlotArena<T> {
    slots: Vec<Slot<T>>,
    free_head: u32,
    live_count: usize,
}

impl<T> CompactSlotArena<T> {
    pub fn new() -> Self {
        Self {
            slots: Vec::new(),
            free_head: 0,
            live_count: 0,
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            slots: Vec::with_capacity(capacity),
            free_head: 0,
            live_count: 0,
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.live_count
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.live_count == 0
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        self.slots.capacity()
    }

    #[inline]
    pub fn slot_count(&self) -> usize {
        self.slots.len()
    }

    pub fn insert(&mut self, value: T) -> ArenaKey {
        let free_head = self.free_head;

        if let Some(slot) = self.slots.get_mut(free_head as usize) {
            let occupied_generation = slot.generation | 1;
            // SAFETY: `free_head` only ever points at a slot whose last
            // write left it Vacant (the same invariant `SlotArena`
            // relies on, and the same one `slotmap::try_insert_with_key`
            // relies on for the identical read) -- so `next_free`, not
            // `value`, is the union's live field here. Must be read
            // before the union gets overwritten below.
            let next_free = unsafe { slot.u.next_free };
            // Writing an entirely new value to a `ManuallyDrop<T>` union
            // field needs no `unsafe` at all on this compiler -- checked
            // directly, not assumed from slotmap's own file-level
            // `#![allow(unused_unsafe)]` comment (which reads as if the
            // identical assignment needed it on some other rustc/edition
            // combination; whatever the reason there, it doesn't apply
            // here). `ManuallyDrop<T>` has no drop glue to skip in the
            // first place, so there's nothing an assignment could get
            // wrong regardless.
            slot.u.value = ManuallyDrop::new(value);
            slot.generation = occupied_generation;
            self.free_head = next_free;
            self.live_count += 1;
            ArenaKey::new(free_head, occupied_generation)
        } else {
            debug_assert_eq!(
                free_head as usize,
                self.slots.len(),
                "free_head should never point past a single new slot beyond the end"
            );
            let generation = 1;
            self.slots.push(Slot {
                u: SlotUnion {
                    value: ManuallyDrop::new(value),
                },
                generation,
            });
            self.free_head = free_head + 1;
            self.live_count += 1;
            ArenaKey::new(free_head, generation)
        }
    }

    pub fn remove(&mut self, key: ArenaKey) -> Option<T> {
        let slot = self.slots.get_mut(key.index() as usize)?;
        if slot.generation != key.generation() {
            return None;
        }
        // SAFETY: `slot.generation == key.generation()` and every real
        // `ArenaKey` this arena issues carries an odd generation (only
        // `insert` mints one, always via `| 1`), so this slot is
        // occupied and `value` is the union's live field.
        let value = unsafe { ManuallyDrop::take(&mut slot.u.value) };
        // Same reasoning as the `insert` comment above: assigning a
        // whole new value to the union's `next_free: u32` field needs
        // no `unsafe`, checked directly on this compiler.
        slot.u.next_free = self.free_head;
        slot.generation = slot.generation.wrapping_add(1);
        self.free_head = key.index();
        self.live_count -= 1;
        Some(value)
    }

    #[inline]
    pub fn contains(&self, key: ArenaKey) -> bool {
        self.get(key).is_some()
    }

    pub fn get(&self, key: ArenaKey) -> Option<&T> {
        let slot = self.slots.get(key.index() as usize)?;
        if slot.generation == key.generation() {
            // SAFETY: generation match implies odd, which implies
            // occupied (see this type's invariant, restated in
            // `remove`'s comment above) -- `value` is live.
            Some(unsafe { &*slot.u.value })
        } else {
            None
        }
    }

    pub fn get_mut(&mut self, key: ArenaKey) -> Option<&mut T> {
        let slot = self.slots.get_mut(key.index() as usize)?;
        if slot.generation == key.generation() {
            // SAFETY: same reasoning as `get` above.
            Some(unsafe { &mut *slot.u.value })
        } else {
            None
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = (ArenaKey, &T)> {
        self.slots.iter().enumerate().filter_map(|(i, slot)| {
            if slot.occupied() {
                // SAFETY: occupied() checked above.
                Some((ArenaKey::new(i as u32, slot.generation), unsafe {
                    &*slot.u.value
                }))
            } else {
                None
            }
        })
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = (ArenaKey, &mut T)> {
        self.slots.iter_mut().enumerate().filter_map(|(i, slot)| {
            if slot.occupied() {
                let generation = slot.generation;
                // SAFETY: occupied() checked above.
                Some((ArenaKey::new(i as u32, generation), unsafe {
                    &mut *slot.u.value
                }))
            } else {
                None
            }
        })
    }

    /// Drops every live value and resets to empty. Sound with no extra
    /// unsafe code here: `Vec::clear` runs each `Slot<T>`'s own `Drop`
    /// impl, which already only touches the union when `occupied()` is
    /// true (this module's doc comment).
    pub fn clear(&mut self) {
        self.slots.clear();
        self.free_head = 0;
        self.live_count = 0;
    }
}

impl<T> Default for CompactSlotArena<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn starts_empty() {
        let a: CompactSlotArena<u32> = CompactSlotArena::new();
        assert_eq!(a.len(), 0);
        assert!(a.is_empty());
        assert_eq!(a.slot_count(), 0);
    }

    #[test]
    fn insert_get_roundtrip() {
        let mut a = CompactSlotArena::new();
        let k = a.insert(42u32);
        assert_eq!(a.get(k), Some(&42));
        assert_eq!(a.len(), 1);
    }

    #[test]
    fn get_mut_writes_through() {
        let mut a = CompactSlotArena::new();
        let k = a.insert(1u32);
        *a.get_mut(k).unwrap() = 2;
        assert_eq!(a.get(k), Some(&2));
    }

    #[test]
    fn remove_returns_value_and_frees_slot() {
        let mut a = CompactSlotArena::new();
        let k = a.insert(7u32);
        assert_eq!(a.remove(k), Some(7));
        assert_eq!(a.get(k), None);
        assert!(a.is_empty());
    }

    #[test]
    fn remove_on_dead_or_unknown_handle_is_a_safe_no_op() {
        let mut a: CompactSlotArena<u32> = CompactSlotArena::new();
        let k = a.insert(1);
        a.remove(k);
        assert_eq!(a.remove(k), None);

        let mut b: CompactSlotArena<u32> = CompactSlotArena::new();
        assert_eq!(b.remove(k), None);
    }

    #[test]
    fn reallocate_reuses_freed_slot_with_bumped_generation() {
        let mut a = CompactSlotArena::new();
        let first = a.insert(100u32);
        assert_eq!(a.remove(first), Some(100));

        let second = a.insert(200u32);
        assert_eq!(second.index(), first.index());
        assert_ne!(second.generation(), first.generation());
        assert_eq!(a.get(second), Some(&200));
        assert_eq!(a.get(first), None);
    }

    #[test]
    fn free_list_reuse_order_is_lifo() {
        let mut a = CompactSlotArena::new();
        let k0 = a.insert('a');
        let k1 = a.insert('b');
        let k2 = a.insert('c');

        a.remove(k0);
        a.remove(k1);
        a.remove(k2);

        let r1 = a.insert('x');
        let r2 = a.insert('y');
        let r3 = a.insert('z');
        assert_eq!(r1.index(), k2.index());
        assert_eq!(r2.index(), k1.index());
        assert_eq!(r3.index(), k0.index());
    }

    #[test]
    fn iterate_visits_every_live_value_and_skips_removed_ones() {
        let mut a = CompactSlotArena::new();
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
        let mut a = CompactSlotArena::new();
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
        let mut a = CompactSlotArena::new();
        let k0 = a.insert(1u32);
        let k1 = a.insert(2u32);
        a.clear();
        assert!(a.is_empty());
        assert_eq!(a.get(k0), None);
        assert_eq!(a.get(k1), None);
        let k2 = a.insert(3u32);
        assert_eq!(k2.index(), 0);
        assert_eq!(k2.generation(), 1);
    }

    #[test]
    fn many_insert_remove_cycles_stay_consistent() {
        let mut a = CompactSlotArena::new();
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
                assert_eq!(a.get(k), Some(&v));
            }
        }
    }

    #[test]
    fn default_matches_new() {
        let a: CompactSlotArena<u32> = CompactSlotArena::default();
        assert!(a.is_empty());
    }

    #[test]
    fn drop_runs_for_every_live_value_when_the_arena_itself_is_dropped() {
        // The actual property this whole union layout has to get right
        // -- ManuallyDrop means the compiler will never run this for
        // free the way it does for SlotArena's plain enum. If Slot<T>'s
        // Drop impl above is wrong, this test either leaks (undercounts)
        // or double-frees (the test process aborts outright).
        use core::cell::Cell;
        struct DropCounter<'a>(&'a Cell<u32>);
        impl<'a> Drop for DropCounter<'a> {
            fn drop(&mut self) {
                self.0.set(self.0.get() + 1);
            }
        }

        let count = Cell::new(0u32);
        {
            let mut a = CompactSlotArena::new();
            a.insert(DropCounter(&count));
            a.insert(DropCounter(&count));
            let k2 = a.insert(DropCounter(&count));
            a.remove(k2);
            assert_eq!(count.get(), 1);
        }
        assert_eq!(count.get(), 3);
    }

    #[test]
    fn removed_slots_do_not_double_drop_on_arena_drop() {
        // Specifically checks Slot<T>'s occupied()-gated Drop impl:
        // a slot that was removed must NOT run T's destructor a second
        // time when the whole arena (and its backing Vec<Slot<T>>)
        // drops. free_list_reuse_order_is_lifo already exercises the
        // free-list mechanics; this test exists purely for the drop
        // count, which that one doesn't check.
        use core::cell::Cell;
        struct DropCounter<'a>(&'a Cell<u32>);
        impl<'a> Drop for DropCounter<'a> {
            fn drop(&mut self) {
                self.0.set(self.0.get() + 1);
            }
        }

        let count = Cell::new(0u32);
        {
            let mut a = CompactSlotArena::new();
            let k0 = a.insert(DropCounter(&count));
            a.insert(DropCounter(&count));
            a.remove(k0); // drops immediately, count -> 1
            assert_eq!(count.get(), 1);
            // Reuses k0's freed slot -- the vacant slot's union now
            // holds a fresh, live value again, not the old dropped one.
            a.insert(DropCounter(&count));
        } // both remaining live values drop here, count -> 3
        assert_eq!(count.get(), 3, "must not double-drop the value already removed above");
    }
}
