//! The ECS world — entity allocator + component storage.
//!
//! `World::spawn`/`despawn`/`is_alive` are a thin wrapper over
//! `mid_collections::GenerationalIndexAllocator` — see that module's own
//! doc comment for why staleness detection works the way it does
//! (verified against real `slotmap` source there, not assumed), and
//! `docs/mid-collections.md`'s "Generational-index arena" section for
//! why it's ranked directly above the rest of that doc's list for
//! `mid-ecs` specifically.
//!
//! `World::insert`/`get`/`get_mut`/`remove`/`has` are a thin wrapper over
//! `SparseShell` (`crate::component`) — the "Sparse Shell" half of the
//! Hybrid ECS Architecture (`docs/mid-ecs.md`). Any `T: 'static` can be
//! attached to any entity with no upfront declaration; see
//! `component.rs`'s own doc comment for the type-erasure mechanism
//! (grounded in Bevy ECS's real `ComponentId`-based design, not a naive
//! `TypeId`-keyed `HashMap`) and for the specific correctness property
//! `World` has to enforce that `SparseShell` alone can't (checking
//! liveness before every component access, to reject a stale handle that
//! happens to share a reused index with a live entity).
//!
//! The Archetype Core (dense/table storage for stable, always-present
//! components — the other half of the hybrid design) is a separate,
//! not-yet-built piece. Nothing here is trying to be both.

use std::fmt;

use mid_collections::{GenerationalIndex, GenerationalIndexAllocator, SparseSetIndex};

use crate::component::SparseShell;

/// A handle to an entity. Detects its own staleness after despawn — a
/// thin wrapper over `mid_collections::GenerationalIndex`, not a
/// reimplementation of its mechanism.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Entity(pub(crate) GenerationalIndex);

impl Entity {
    /// Wraps a raw `GenerationalIndex` as an `Entity`. `pub(crate)` and
    /// `#[cfg(test)]` deliberately -- outside this crate, an `Entity`
    /// should only ever come from `World::spawn`, never fabricated; the
    /// only current caller is `component.rs`'s own test module, which
    /// needs real, distinct `Entity` values without a full `World` in
    /// scope. Gated to `test` specifically so it doesn't sit as an
    /// always-there-but-only-used-by-tests function generating a
    /// dead-code warning in ordinary builds.
    #[cfg(test)]
    #[inline]
    pub(crate) fn from_generational_index(index: GenerationalIndex) -> Self {
        Self(index)
    }
}

impl Entity {
    /// The raw slot index. Not meaningful alone — two different
    /// generations of the same slot share this value; use
    /// `World::is_alive` to check validity, not this in isolation.
    /// Exposed mainly for debugging/FFI.
    #[inline]
    pub fn index(&self) -> u32 {
        self.0.index()
    }

    /// The generation this handle was issued with.
    #[inline]
    pub fn generation(&self) -> u32 {
        self.0.generation()
    }
}

impl SparseSetIndex for Entity {
    #[inline]
    fn sparse_index(&self) -> u32 {
        self.0.sparse_index()
    }
}

impl fmt::Display for Entity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Entity({}v{})", self.0.index(), self.0.generation())
    }
}

/// The ECS world. Owns entity lifecycle today; will own component
/// storage (Archetype Core + Sparse Shell) once that exists.
pub struct World {
    entities: GenerationalIndexAllocator,
    components: SparseShell,
}

impl World {
    /// Creates an empty world — no entities spawned yet.
    pub fn new() -> Self {
        Self {
            entities: GenerationalIndexAllocator::new(),
            components: SparseShell::new(),
        }
    }

    /// Creates a world pre-sized for `capacity` entities before the next
    /// spawn past that would reallocate.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            entities: GenerationalIndexAllocator::with_capacity(capacity),
            components: SparseShell::new(),
        }
    }

    /// Spawns a new, live entity.
    pub fn spawn(&mut self) -> Entity {
        Entity(self.entities.allocate())
    }

    /// Despawns `entity`, if it's still alive. Returns whether it
    /// actually was — despawning an already-dead or never-real handle
    /// is a safe no-op, not a panic, matching
    /// `GenerationalIndexAllocator::deallocate`'s own contract.
    ///
    /// Removes every component attached to `entity` from the Sparse
    /// Shell **before** freeing its generational slot — that ordering is
    /// load-bearing, not incidental. `SparseSet` looks up purely by raw
    /// index, not generation (see `component.rs`'s own doc comment), so
    /// freeing the slot first and cleaning up components after would
    /// leave a window where a freshly-reused index's new entity could
    /// read the dead entity's stale leftover data.
    pub fn despawn(&mut self, entity: Entity) -> bool {
        if !self.is_alive(entity) {
            return false;
        }
        self.components.remove_entity_from_all(entity);
        self.entities.deallocate(entity.0)
    }

    /// Whether `entity` is still alive.
    #[inline]
    pub fn is_alive(&self, entity: Entity) -> bool {
        self.entities.is_alive(entity.0)
    }

    /// Number of currently-live entities.
    #[inline]
    pub fn entity_count(&self) -> usize {
        self.entities.len()
    }

    /// True if nothing is currently spawned.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.entities.is_empty()
    }

    /// Attaches `component` to `entity`. A safe no-op (returns `None`,
    /// nothing is stored) if `entity` isn't alive — checked explicitly,
    /// not left to chance, because `SparseSet`'s own index-only lookup
    /// can't tell a stale handle from a live one sharing the same raw
    /// index (see `component.rs`'s doc comment). No panic in any build
    /// configuration, matching every other fallible operation in this
    /// codebase (`SparseSet::remove`, `GenerationalIndexAllocator::
    /// deallocate`, ...) — a `debug_assert!` was tried here first and
    /// removed: it directly contradicted that established convention,
    /// and the test written for the safe-fallback case immediately
    /// caught the contradiction by panicking in the very test meant to
    /// prove it doesn't.
    pub fn insert<T: 'static>(&mut self, entity: Entity, component: T) -> Option<T> {
        if !self.is_alive(entity) {
            return None;
        }
        self.components.insert(entity, component)
    }

    /// Looks up `entity`'s `T` component, if attached and `entity` is
    /// still alive.
    pub fn get<T: 'static>(&self, entity: Entity) -> Option<&T> {
        if !self.is_alive(entity) {
            return None;
        }
        self.components.get(entity)
    }

    /// Looks up `entity`'s `T` component mutably, if attached and
    /// `entity` is still alive.
    pub fn get_mut<T: 'static>(&mut self, entity: Entity) -> Option<&mut T> {
        if !self.is_alive(entity) {
            return None;
        }
        self.components.get_mut(entity)
    }

    /// Removes and returns `entity`'s `T` component, if attached and
    /// `entity` is still alive.
    pub fn remove<T: 'static>(&mut self, entity: Entity) -> Option<T> {
        if !self.is_alive(entity) {
            return None;
        }
        self.components.remove(entity)
    }

    /// Whether `entity` is alive and currently has a `T` component
    /// attached.
    #[inline]
    pub fn has<T: 'static>(&self, entity: Entity) -> bool {
        self.is_alive(entity) && self.components.has::<T>(entity)
    }
}

impl Default for World {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_world_is_empty() {
        let w = World::new();
        assert_eq!(w.entity_count(), 0);
        assert!(w.is_empty());
    }

    #[test]
    fn spawn_gives_a_live_entity() {
        let mut w = World::new();
        let e = w.spawn();
        assert!(w.is_alive(e));
        assert_eq!(w.entity_count(), 1);
        assert!(!w.is_empty());
    }

    #[test]
    fn spawn_returns_distinct_entities() {
        let mut w = World::new();
        let e1 = w.spawn();
        let e2 = w.spawn();
        assert_ne!(e1, e2);
        assert_eq!(w.entity_count(), 2);
    }

    #[test]
    fn despawn_then_not_alive() {
        let mut w = World::new();
        let e = w.spawn();
        assert!(w.despawn(e));
        assert!(!w.is_alive(e));
        assert_eq!(w.entity_count(), 0);
    }

    #[test]
    fn despawn_twice_is_a_safe_no_op() {
        let mut w = World::new();
        let e = w.spawn();
        assert!(w.despawn(e));
        assert!(
            !w.despawn(e),
            "second despawn of the same entity must not panic or double-free"
        );
    }

    #[test]
    fn despawn_stale_handle_after_slot_reuse_does_not_touch_the_new_occupant() {
        // The scenario that actually matters: a slot gets freed, reused
        // by a new entity, and something is still holding the OLD,
        // now-stale handle to that same raw index. Despawning it must
        // fail cleanly, not silently despawn whatever now legitimately
        // occupies that slot.
        //
        // (A test resembling "despawn a totally foreign handle from
        // another World" was tried here first and removed: two fresh
        // World::new() allocators both hand out {index: 0, generation:
        // 1} as their first spawn, since GenerationalIndex carries no
        // per-allocator identity -- so that handle isn't actually
        // distinguishable from a legitimate same-shape one, and the
        // test's own premise was wrong, not World's behavior. Caught by
        // actually running it, not assumed correct from reasoning about
        // it beforehand.)
        let mut w = World::new();
        let e1 = w.spawn();
        w.despawn(e1);
        let e2 = w.spawn(); // reuses e1's slot, different generation

        assert!(
            !w.despawn(e1),
            "despawning the stale e1 handle must fail, not succeed"
        );
        assert!(
            w.is_alive(e2),
            "e2 must be untouched by the failed despawn(e1) call"
        );
    }

    #[test]
    fn respawned_slot_gives_a_distinguishable_entity() {
        // The actual point of the generational design, exercised through
        // World's real public API, not just mid_collections' own tests.
        let mut w = World::new();
        let e1 = w.spawn();
        w.despawn(e1);
        let e2 = w.spawn();

        assert_eq!(e2.index(), e1.index(), "the freed slot should be reused");
        assert_ne!(e2.generation(), e1.generation());
        assert!(w.is_alive(e2));
        assert!(
            !w.is_alive(e1),
            "the stale handle from before the reuse must not alias e2"
        );
    }

    #[test]
    fn display_shows_index_and_generation() {
        let mut w = World::new();
        let e = w.spawn();
        let text = format!("{e}");
        assert_eq!(text, format!("Entity({}v{})", e.index(), e.generation()));
    }

    #[test]
    fn implements_sparse_set_index() {
        let mut w = World::new();
        let e = w.spawn();
        assert_eq!(e.sparse_index(), e.index());
    }

    #[test]
    fn with_capacity_does_not_change_observable_behavior() {
        let mut w = World::with_capacity(64);
        assert!(w.is_empty());
        let e = w.spawn();
        assert!(w.is_alive(e));
    }

    #[test]
    fn default_matches_new() {
        let w = World::default();
        assert!(w.is_empty());
    }

    #[test]
    fn many_spawn_despawn_cycles_keep_entity_count_consistent() {
        let mut w = World::new();
        let mut live = Vec::new();

        for round in 0..50u32 {
            live.push(w.spawn());
            if round % 3 == 0 && !live.is_empty() {
                let dead: Entity = live.remove(0);
                assert!(w.despawn(dead));
            }
            assert_eq!(w.entity_count(), live.len());
            for &e in &live {
                assert!(w.is_alive(e));
            }
        }
    }

    #[derive(Debug, PartialEq)]
    struct Health(u32);

    #[test]
    fn insert_then_get_component() {
        let mut w = World::new();
        let e = w.spawn();
        assert_eq!(w.insert(e, Health(100)), None);
        assert_eq!(w.get::<Health>(e), Some(&Health(100)));
        assert!(w.has::<Health>(e));
    }

    #[test]
    fn get_component_on_dead_entity_returns_none() {
        let mut w = World::new();
        let e = w.spawn();
        w.insert(e, Health(50));
        w.despawn(e);
        assert_eq!(
            w.get::<Health>(e),
            None,
            "a dead entity must not still report its old component"
        );
    }

    #[test]
    fn insert_on_dead_entity_is_a_safe_no_op() {
        let mut w = World::new();
        let e = w.spawn();
        w.despawn(e);
        assert_eq!(w.insert(e, Health(999)), None);
        assert_eq!(w.get::<Health>(e), None);
    }

    #[test]
    fn despawn_removes_all_attached_components() {
        #[derive(Debug, PartialEq)]
        struct Mana(u32);

        let mut w = World::new();
        let e = w.spawn();
        w.insert(e, Health(100));
        w.insert(e, Mana(50));

        assert!(w.despawn(e));

        assert_eq!(w.get::<Health>(e), None);
        assert_eq!(w.get::<Mana>(e), None);
    }

    #[test]
    fn despawn_does_not_touch_other_entities_components() {
        let mut w = World::new();
        let e1 = w.spawn();
        let e2 = w.spawn();
        w.insert(e1, Health(10));
        w.insert(e2, Health(20));

        w.despawn(e1);

        assert_eq!(
            w.get::<Health>(e2),
            Some(&Health(20)),
            "e2 must be untouched by e1's despawn"
        );
    }

    #[test]
    fn reused_slot_does_not_inherit_the_old_entitys_components() {
        // The actual point of checking `is_alive` before every component
        // access in World, not just SparseShell's own tests: a *new*
        // entity reusing a freed slot must never read the *old* dead
        // entity's leftover data, even transiently.
        let mut w = World::new();
        let e1 = w.spawn();
        w.insert(e1, Health(77));
        w.despawn(e1);

        let e2 = w.spawn(); // reuses e1's slot, different generation
        assert_eq!(
            w.get::<Health>(e2),
            None,
            "e2 must not inherit e1's old Health component"
        );

        w.insert(e2, Health(5));
        assert_eq!(w.get::<Health>(e2), Some(&Health(5)));
    }

    #[test]
    fn stale_handle_cannot_read_the_live_entity_now_sharing_its_index() {
        // The specific gap `SparseSet` alone can't close, documented in
        // component.rs's doc comment: a *stale* Entity handle (e1) and a
        // *live* Entity (e2) can share the same raw index after reuse.
        // World's own liveness check is what makes get(e1) correctly
        // fail here, rather than accidentally returning e2's real data.
        let mut w = World::new();
        let e1 = w.spawn();
        w.despawn(e1);
        let e2 = w.spawn(); // same raw index as e1, higher generation
        w.insert(e2, Health(42));

        assert_eq!(
            e1.index(),
            e2.index(),
            "this test only proves anything if the index was actually reused"
        );
        assert_eq!(
            w.get::<Health>(e1),
            None,
            "the stale e1 handle must not see e2's Health component"
        );
        assert_eq!(w.get::<Health>(e2), Some(&Health(42)));
    }
}
