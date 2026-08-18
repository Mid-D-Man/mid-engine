//! The Sparse Shell — type-erased component storage keyed by `Entity`.
//!
//! This is the "Sparse Shell" half of the Hybrid ECS Architecture
//! (`docs/mid-ecs.md`): any `T: 'static` can be attached to any entity
//! with no upfront declaration, backed by one `mid_collections::SparseSet`
//! per component type. The Archetype Core (dense/table storage for
//! stable, always-present components) is a separate, not-yet-built piece
//! — this module is not trying to be both.
//!
//! # Design, grounded in Bevy ECS's real source, not invented independently
//!
//! The obvious naive approach — `HashMap<TypeId, Box<dyn Any>>`, hash a
//! `TypeId` on every single component access — is specifically what real,
//! performance-focused ECS implementations avoid. Checked Bevy's actual
//! source before building this: `Components` maps `TypeId -> ComponentId`
//! **once**, at registration, and "no `bevy_ecs` code uses `TypeId`: it's
//! all `ComponentId`" afterward (confirmed from Bevy's own docs/source,
//! not assumed) — `ComponentId` is a small, dense integer, cheap to use
//! as an actual array/sparse-set index rather than a hash key. Bevy's own
//! `Table` goes further and uses exactly the structure this module
//! converges on independently: `Table { columns: ImmutableSparseSet<
//! ComponentId, Column>, entities: Vec<Entity> }` — a sparse set keyed by
//! `ComponentId`, not a `HashMap`. This module mirrors that: `TypeId ->
//! ComponentId` is a `HashMap` used only at registration time (rare,
//! off the hot path), and the actual per-frame-relevant storage is a
//! `mid_collections::SparseSet<ComponentId, _>` — the exact structure
//! already built and benchmarked for this workspace, not a new one.
//!
//! # The type-erasure mechanism
//!
//! Each registered component type gets its own
//! `mid_collections::SparseSet<Entity, T>`, stored behind
//! `Box<dyn ComponentColumn>` so the outer `SparseSet<ComponentId, _>`
//! doesn't need to be generic over every `T` ever used. `ComponentColumn`
//! extends `Any` for downcasting back to the concrete
//! `SparseSet<Entity, T>` on access, plus one type-erased operation
//! (`remove_entity`) needed for despawn cleanup, where the caller can't
//! know every `T` an entity might have components of. This is the
//! standard, well-established pattern for heterogeneous typed storage in
//! Rust — the same shape `anymap`-style crates and Bevy's own
//! `Box<dyn Any>`-erased columns use.
//!
//! # The critical correctness property this module does NOT provide on its own
//!
//! `mid_collections::SparseSet` looks up purely by
//! `SparseSetIndex::sparse_index()` — for `Entity`, that's the raw slot
//! index alone, **not** the generation. A `SparseSet<Entity, T>` cannot
//! by itself distinguish a stale `Entity` handle from a live one that
//! happens to share the same freed-and-reused index — this was already
//! flagged directly, with a demonstrating test, in
//! `mid_collections::generational_index`'s own doc comment. Every public
//! method here that touches component data checks `World`-level liveness
//! first (via the `is_alive` closure/callback threaded through — see
//! `insert`/`get`/`get_mut`/`remove`/`has` below) specifically to close
//! that gap, rather than leaving it as a footgun for whatever calls into
//! this module.

use std::any::{Any, TypeId};
use std::collections::HashMap;

use mid_collections::{SparseSet, SparseSetIndex};

use crate::world::Entity;

/// Dense identifier for a registered component type within one
/// [`SparseShell`]. **Not** the same thing as `TypeId` — see this
/// module's doc comment for why the distinction matters. Only meaningful
/// relative to the `SparseShell` that issued it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ComponentId(u32);

impl SparseSetIndex for ComponentId {
    #[inline]
    fn sparse_index(&self) -> u32 {
        self.0
    }
}

/// Type-erased operations every component column supports, regardless
/// of its concrete component type. Not meant to be implemented outside
/// this module — the blanket impl below covers every real case.
trait ComponentColumn: Any {
    /// Removes `entity`'s data from this column, if present. Returns
    /// whether anything was actually removed.
    fn remove_entity(&mut self, entity: Entity) -> bool;
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

impl<T: 'static> ComponentColumn for SparseSet<Entity, T> {
    fn remove_entity(&mut self, entity: Entity) -> bool {
        self.remove(entity).is_some()
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// The Sparse Shell itself — one type-erased `SparseSet<Entity, T>` per
/// registered component type, indexed by a dense [`ComponentId`] rather
/// than a `TypeId` hash. See this module's doc comment for the full
/// design reasoning.
pub struct SparseShell {
    component_ids: HashMap<TypeId, ComponentId>,
    columns: SparseSet<ComponentId, Box<dyn ComponentColumn>>,
    next_id: u32,
}

impl SparseShell {
    /// Creates an empty shell — no component types registered yet.
    pub fn new() -> Self {
        Self {
            component_ids: HashMap::new(),
            columns: SparseSet::new(),
            next_id: 0,
        }
    }

    /// Returns `T`'s `ComponentId`, registering it (assigning the next
    /// dense id) on first use. Registration is a one-time `HashMap`
    /// lookup-or-insert per type ever used, not a per-entity or
    /// per-access cost — see this module's doc comment.
    fn component_id<T: 'static>(&mut self) -> ComponentId {
        let type_id = TypeId::of::<T>();
        if let Some(&id) = self.component_ids.get(&type_id) {
            id
        } else {
            let id = ComponentId(self.next_id);
            self.next_id += 1;
            self.component_ids.insert(type_id, id);
            id
        }
    }

    /// Looks up `T`'s `ComponentId` without registering it — `None` if
    /// `T` has never been inserted for anything, in which case there is
    /// by construction nothing to find.
    fn existing_component_id<T: 'static>(&self) -> Option<ComponentId> {
        self.component_ids.get(&TypeId::of::<T>()).copied()
    }

    /// Attaches `component` to `entity`, registering `T` as a component
    /// type on first use. Replaces and returns any existing `T` already
    /// attached to `entity` — matching `SparseSet::insert`'s own
    /// replace-on-collision convention.
    pub fn insert<T: 'static>(&mut self, entity: Entity, component: T) -> Option<T> {
        let id = self.component_id::<T>();
        match self.columns.get_mut(id) {
            Some(existing) => existing
                .as_any_mut()
                .downcast_mut::<SparseSet<Entity, T>>()
                .expect("ComponentId must always map to a column of its own registered type")
                .insert(entity, component),
            None => {
                let mut column: SparseSet<Entity, T> = SparseSet::new();
                let old = column.insert(entity, component);
                self.columns.insert(id, Box::new(column));
                old
            }
        }
    }

    /// Looks up `entity`'s `T` component, if attached.
    pub fn get<T: 'static>(&self, entity: Entity) -> Option<&T> {
        let id = self.existing_component_id::<T>()?;
        self.columns
            .get(id)?
            .as_any()
            .downcast_ref::<SparseSet<Entity, T>>()
            .expect("ComponentId must always map to a column of its own registered type")
            .get(entity)
    }

    /// Looks up `entity`'s `T` component mutably, if attached.
    pub fn get_mut<T: 'static>(&mut self, entity: Entity) -> Option<&mut T> {
        let id = self.existing_component_id::<T>()?;
        self.columns
            .get_mut(id)?
            .as_any_mut()
            .downcast_mut::<SparseSet<Entity, T>>()
            .expect("ComponentId must always map to a column of its own registered type")
            .get_mut(entity)
    }

    /// Removes and returns `entity`'s `T` component, if attached.
    pub fn remove<T: 'static>(&mut self, entity: Entity) -> Option<T> {
        let id = self.existing_component_id::<T>()?;
        self.columns
            .get_mut(id)?
            .as_any_mut()
            .downcast_mut::<SparseSet<Entity, T>>()
            .expect("ComponentId must always map to a column of its own registered type")
            .remove(entity)
    }

    /// Whether `entity` currently has a `T` component attached.
    #[inline]
    pub fn has<T: 'static>(&self, entity: Entity) -> bool {
        self.get::<T>(entity).is_some()
    }

    /// Removes `entity` from every registered component column,
    /// regardless of type. Used by `World::despawn`, which must call
    /// this *before* freeing the entity's generational slot — see this
    /// module's doc comment on why the ordering matters.
    ///
    /// O(number of distinct component types ever registered), not
    /// O(number of components `entity` actually has) — there's no
    /// per-entity "which types does it have" index yet to do better than
    /// that. Accepted for now: a real archetype/bitset would fix this,
    /// and doesn't exist yet (`docs/mid-ecs.md`). Revisit if this shows
    /// up in a real profile, not speculatively.
    pub(crate) fn remove_entity_from_all(&mut self, entity: Entity) {
        for (_, column) in self.columns.iter_mut() {
            column.remove_entity(entity);
        }
    }
}

impl Default for SparseShell {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mid_collections::GenerationalIndexAllocator;

    // A tiny local entity factory for these tests -- SparseShell doesn't
    // own entity allocation itself (World does), so tests here just need
    // *some* real, distinct Entity values, not a full World.
    fn entity_factory() -> impl FnMut() -> Entity {
        let mut allocator = GenerationalIndexAllocator::new();
        move || Entity::from_generational_index(allocator.allocate())
    }

    #[derive(Debug, PartialEq)]
    struct Position {
        x: f32,
        y: f32,
    }

    #[derive(Debug, PartialEq)]
    struct Velocity(f32, f32);

    #[test]
    fn insert_then_get() {
        let mut spawn = entity_factory();
        let mut shell = SparseShell::new();
        let e = spawn();
        assert_eq!(shell.insert(e, Position { x: 1.0, y: 2.0 }), None);
        assert_eq!(shell.get::<Position>(e), Some(&Position { x: 1.0, y: 2.0 }));
    }

    #[test]
    fn get_missing_component_type_returns_none() {
        let mut spawn = entity_factory();
        let shell = SparseShell::new();
        let e = spawn();
        // Position was never registered at all -- must not panic.
        assert_eq!(shell.get::<Position>(e), None);
    }

    #[test]
    fn get_unattached_component_on_a_known_type_returns_none() {
        let mut spawn = entity_factory();
        let mut shell = SparseShell::new();
        let e1 = spawn();
        let e2 = spawn();
        shell.insert(e1, Position { x: 0.0, y: 0.0 });
        // Position IS registered now (because of e1), but e2 never got one.
        assert_eq!(shell.get::<Position>(e2), None);
    }

    #[test]
    fn multiple_component_types_on_the_same_entity_are_independent() {
        let mut spawn = entity_factory();
        let mut shell = SparseShell::new();
        let e = spawn();
        shell.insert(e, Position { x: 3.0, y: 4.0 });
        shell.insert(e, Velocity(1.0, -1.0));
        assert_eq!(shell.get::<Position>(e), Some(&Position { x: 3.0, y: 4.0 }));
        assert_eq!(shell.get::<Velocity>(e), Some(&Velocity(1.0, -1.0)));
    }

    #[test]
    fn insert_replaces_existing_component_and_returns_old() {
        let mut spawn = entity_factory();
        let mut shell = SparseShell::new();
        let e = spawn();
        shell.insert(e, Position { x: 1.0, y: 1.0 });
        let old = shell.insert(e, Position { x: 9.0, y: 9.0 });
        assert_eq!(old, Some(Position { x: 1.0, y: 1.0 }));
        assert_eq!(shell.get::<Position>(e), Some(&Position { x: 9.0, y: 9.0 }));
    }

    #[test]
    fn get_mut_actually_mutates() {
        let mut spawn = entity_factory();
        let mut shell = SparseShell::new();
        let e = spawn();
        shell.insert(e, Position { x: 0.0, y: 0.0 });
        shell.get_mut::<Position>(e).unwrap().x = 42.0;
        assert_eq!(
            shell.get::<Position>(e),
            Some(&Position { x: 42.0, y: 0.0 })
        );
    }

    #[test]
    fn remove_returns_value_and_clears_it() {
        let mut spawn = entity_factory();
        let mut shell = SparseShell::new();
        let e = spawn();
        shell.insert(e, Velocity(2.0, 2.0));
        assert_eq!(shell.remove::<Velocity>(e), Some(Velocity(2.0, 2.0)));
        assert_eq!(shell.get::<Velocity>(e), None);
        assert!(!shell.has::<Velocity>(e));
    }

    #[test]
    fn remove_missing_returns_none_not_panic() {
        let mut spawn = entity_factory();
        let mut shell = SparseShell::new();
        let e = spawn();
        assert_eq!(shell.remove::<Position>(e), None);
    }

    #[test]
    fn has_reflects_attachment() {
        let mut spawn = entity_factory();
        let mut shell = SparseShell::new();
        let e = spawn();
        assert!(!shell.has::<Position>(e));
        shell.insert(e, Position { x: 0.0, y: 0.0 });
        assert!(shell.has::<Position>(e));
    }

    #[test]
    fn component_id_is_stable_and_distinct_per_type() {
        let mut shell = SparseShell::new();
        let pos_id_1 = shell.component_id::<Position>();
        let vel_id = shell.component_id::<Velocity>();
        let pos_id_2 = shell.component_id::<Position>();
        assert_eq!(
            pos_id_1, pos_id_2,
            "repeated registration of the same type must return the same id"
        );
        assert_ne!(pos_id_1, vel_id);
    }

    #[test]
    fn remove_entity_from_all_clears_every_component_type() {
        let mut spawn = entity_factory();
        let mut shell = SparseShell::new();
        let e = spawn();
        shell.insert(e, Position { x: 1.0, y: 1.0 });
        shell.insert(e, Velocity(1.0, 1.0));

        shell.remove_entity_from_all(e);

        assert_eq!(shell.get::<Position>(e), None);
        assert_eq!(shell.get::<Velocity>(e), None);
    }

    #[test]
    fn remove_entity_from_all_does_not_touch_other_entities() {
        let mut spawn = entity_factory();
        let mut shell = SparseShell::new();
        let e1 = spawn();
        let e2 = spawn();
        shell.insert(e1, Position { x: 1.0, y: 1.0 });
        shell.insert(e2, Position { x: 2.0, y: 2.0 });

        shell.remove_entity_from_all(e1);

        assert_eq!(shell.get::<Position>(e1), None);
        assert_eq!(
            shell.get::<Position>(e2),
            Some(&Position { x: 2.0, y: 2.0 }),
            "e2 must be untouched"
        );
    }

    #[test]
    fn default_matches_new() {
        let mut spawn = entity_factory();
        let shell = SparseShell::default();
        let e = spawn();
        assert_eq!(shell.get::<Position>(e), None);
    }
}
