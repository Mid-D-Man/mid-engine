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
//! `World::insert_static`/`get_static`/`get_static_mut`/`remove_static`/
//! `has_static` are a thin wrapper over `Archetypes`
//! (`crate::archetype`) — the "Archetype Core" half of the Hybrid ECS
//! Architecture, real dynamic table storage with migration between
//! archetypes as components are added/removed. See `archetype.rs`'s own
//! doc comment for the design (grounded in Bevy ECS's real source, read
//! directly) and for why it uses a completely separate `ComponentId`
//! numbering space from `SparseShell`'s own.
//!
//! Iterating this storage (`World::query`/`query2`) lives in `query.rs`,
//! not here — a distinct concern from owning the storage itself.

use std::fmt;

use mid_collections::{GenerationalIndex, GenerationalIndexAllocator, SparseSetIndex};

use crate::archetype::Archetypes;
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

/// The ECS world. Owns entity lifecycle, the Sparse Shell, and the
/// Archetype Core.
pub struct World {
    pub(crate) entities: GenerationalIndexAllocator,
    pub(crate) components: SparseShell,
    pub(crate) archetypes: Archetypes,
}

impl World {
    /// Creates an empty world — no entities spawned yet.
    pub fn new() -> Self {
        Self {
            entities: GenerationalIndexAllocator::new(),
            components: SparseShell::new(),
            archetypes: Archetypes::new(),
        }
    }

    /// Creates a world pre-sized for `capacity` entities before the next
    /// spawn past that would reallocate.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            entities: GenerationalIndexAllocator::with_capacity(capacity),
            components: SparseShell::new(),
            archetypes: Archetypes::new(),
        }
    }

    /// Spawns a new, live entity — into the empty archetype, with no
    /// components attached yet in either storage system.
    pub fn spawn(&mut self) -> Entity {
        let entity = Entity(self.entities.allocate());
        self.archetypes.spawn(entity);
        entity
    }

    /// Despawns `entity`, if it's still alive. Returns whether it
    /// actually was — despawning an already-dead or never-real handle
    /// is a safe no-op, not a panic, matching
    /// `GenerationalIndexAllocator::deallocate`'s own contract.
    ///
    /// Removes every component attached to `entity` from *both* storage
    /// systems (Sparse Shell and Archetype Core) **before** freeing its
    /// generational slot — that ordering is load-bearing, not
    /// incidental, for both. Neither `SparseSet`-backed storage looks up
    /// by anything but raw index, not generation (see `component.rs`'s
    /// and `archetype.rs`'s own doc comments), so freeing the slot first
    /// and cleaning up after would leave a window where a
    /// freshly-reused index's new entity could read the dead entity's
    /// stale leftover data from either system.
    pub fn despawn(&mut self, entity: Entity) -> bool {
        if !self.is_alive(entity) {
            return false;
        }
        self.components.remove_entity_from_all(entity);
        self.archetypes.despawn(entity);
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

    /// Attaches `component` to `entity` as an **archetype-tracked**
    /// component — migrating `entity`'s row into whichever archetype
    /// matches its new, larger component set (see `archetype.rs`'s doc
    /// comment for what "migrating" actually does). A safe no-op
    /// (returns `false`) if `entity` isn't alive, or already has a `T`
    /// attached this way — the same liveness-check requirement
    /// `insert`/`get`/etc. have for the Sparse Shell applies here too,
    /// for the identical reason (index-only lookup, can't tell a stale
    /// handle from a live one sharing a reused index on its own).
    ///
    /// Distinct method name from `insert` deliberately — a given `T`
    /// should live in exactly one of the two storage systems, never
    /// both, and there's no enforcement of that yet beyond the caller
    /// being consistent about which method they call for a given type.
    /// A `Component` trait fixing each type's storage strategy once
    /// (matching where Bevy eventually landed) is a real future
    /// refinement, not needed to get real, correct behavior today.
    pub fn insert_static<T: 'static>(&mut self, entity: Entity, component: T) -> bool {
        if !self.is_alive(entity) {
            return false;
        }
        let id = self.archetypes.component_id::<T>();
        self.archetypes.insert(entity, id, component)
    }

    /// Removes and returns `entity`'s archetype-tracked `T` component,
    /// if attached and `entity` is alive — migrating its row back into
    /// the archetype matching its new, smaller component set.
    pub fn remove_static<T: 'static>(&mut self, entity: Entity) -> Option<T> {
        if !self.is_alive(entity) {
            return None;
        }
        let id = self.archetypes.existing_component_id::<T>()?;
        self.archetypes.remove(entity, id)
    }

    /// Looks up `entity`'s archetype-tracked `T` component, if attached
    /// and `entity` is alive.
    pub fn get_static<T: 'static>(&self, entity: Entity) -> Option<&T> {
        if !self.is_alive(entity) {
            return None;
        }
        let id = self.archetypes.existing_component_id::<T>()?;
        self.archetypes.get(entity, id)
    }

    /// Looks up `entity`'s archetype-tracked `T` component mutably, if
    /// attached and `entity` is alive.
    pub fn get_static_mut<T: 'static>(&mut self, entity: Entity) -> Option<&mut T> {
        if !self.is_alive(entity) {
            return None;
        }
        let id = self.archetypes.existing_component_id::<T>()?;
        self.archetypes.get_mut(entity, id)
    }

    /// Whether `entity` is alive and currently has an archetype-tracked
    /// `T` component attached.
    #[inline]
    pub fn has_static<T: 'static>(&self, entity: Entity) -> bool {
        let Some(id) = self.archetypes.existing_component_id::<T>() else {
            return false;
        };
        self.is_alive(entity) && self.archetypes.has(entity, id)
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

    #[derive(Debug, PartialEq)]
    struct Mass(f32);
    #[derive(Debug, PartialEq)]
    struct Charge(f32);

    #[test]
    fn insert_static_then_get_static() {
        let mut w = World::new();
        let e = w.spawn();
        assert!(w.insert_static(e, Mass(10.0)));
        assert_eq!(w.get_static::<Mass>(e), Some(&Mass(10.0)));
        assert!(w.has_static::<Mass>(e));
    }

    #[test]
    fn insert_static_two_components_preserves_both() {
        let mut w = World::new();
        let e = w.spawn();
        w.insert_static(e, Mass(1.0));
        w.insert_static(e, Charge(2.0));
        assert_eq!(w.get_static::<Mass>(e), Some(&Mass(1.0)));
        assert_eq!(w.get_static::<Charge>(e), Some(&Charge(2.0)));
    }

    #[test]
    fn get_static_component_on_dead_entity_returns_none() {
        let mut w = World::new();
        let e = w.spawn();
        w.insert_static(e, Mass(5.0));
        w.despawn(e);
        assert_eq!(
            w.get_static::<Mass>(e),
            None,
            "a dead entity must not still report its old static component"
        );
    }

    #[test]
    fn insert_static_on_dead_entity_is_a_safe_no_op() {
        let mut w = World::new();
        let e = w.spawn();
        w.despawn(e);
        assert!(!w.insert_static(e, Mass(999.0)));
        assert_eq!(w.get_static::<Mass>(e), None);
    }

    #[test]
    fn despawn_removes_static_components_too() {
        let mut w = World::new();
        let e = w.spawn();
        w.insert_static(e, Mass(1.0));
        w.insert_static(e, Charge(2.0));

        assert!(w.despawn(e));

        assert_eq!(w.get_static::<Mass>(e), None);
        assert_eq!(w.get_static::<Charge>(e), None);
    }

    #[test]
    fn despawn_does_not_touch_other_entities_static_components() {
        let mut w = World::new();
        let e1 = w.spawn();
        let e2 = w.spawn();
        w.insert_static(e1, Mass(1.0));
        w.insert_static(e2, Mass(2.0));

        w.despawn(e1);

        assert_eq!(
            w.get_static::<Mass>(e2),
            Some(&Mass(2.0)),
            "e2 must be untouched by e1's despawn"
        );
    }

    #[test]
    fn reused_slot_does_not_inherit_the_old_entitys_static_components() {
        let mut w = World::new();
        let e1 = w.spawn();
        w.insert_static(e1, Mass(77.0));
        w.despawn(e1);

        let e2 = w.spawn(); // reuses e1's slot, different generation
        assert_eq!(
            w.get_static::<Mass>(e2),
            None,
            "e2 must not inherit e1's old static Mass component"
        );

        w.insert_static(e2, Mass(5.0));
        assert_eq!(w.get_static::<Mass>(e2), Some(&Mass(5.0)));
    }

    #[test]
    fn stale_handle_cannot_read_the_live_entitys_static_component() {
        let mut w = World::new();
        let e1 = w.spawn();
        w.despawn(e1);
        let e2 = w.spawn(); // same raw index as e1, higher generation
        w.insert_static(e2, Mass(42.0));

        assert_eq!(
            e1.index(),
            e2.index(),
            "this test only proves anything if the index was actually reused"
        );
        assert_eq!(
            w.get_static::<Mass>(e1),
            None,
            "the stale e1 handle must not see e2's static Mass component"
        );
        assert_eq!(w.get_static::<Mass>(e2), Some(&Mass(42.0)));
    }

    #[test]
    fn remove_static_returns_value_and_clears_it() {
        let mut w = World::new();
        let e = w.spawn();
        w.insert_static(e, Mass(3.0));
        assert_eq!(w.remove_static::<Mass>(e), Some(Mass(3.0)));
        assert_eq!(w.get_static::<Mass>(e), None);
        assert!(!w.has_static::<Mass>(e));
    }

    #[test]
    fn get_static_mut_actually_mutates() {
        let mut w = World::new();
        let e = w.spawn();
        w.insert_static(e, Mass(1.0));
        w.get_static_mut::<Mass>(e).unwrap().0 = 50.0;
        assert_eq!(w.get_static::<Mass>(e), Some(&Mass(50.0)));
    }

    #[test]
    fn static_and_sparse_shell_storage_are_independent() {
        // Position (sparse-shell, via `insert`) and Mass (archetype-core,
        // via `insert_static`) coexisting on the same entity, each
        // readable only through its own matching accessor.
        let mut w = World::new();
        let e = w.spawn();
        w.insert(e, Health(100));
        w.insert_static(e, Mass(2.5));

        assert_eq!(w.get::<Health>(e), Some(&Health(100)));
        assert_eq!(w.get_static::<Mass>(e), Some(&Mass(2.5)));

        w.despawn(e);
        assert_eq!(w.get::<Health>(e), None);
        assert_eq!(w.get_static::<Mass>(e), None);
    }
}
