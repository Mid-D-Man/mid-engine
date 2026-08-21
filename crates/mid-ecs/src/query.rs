//! Query iteration over the Sparse Shell.
//!
//! Lives here, not on `World` itself in `world.rs` — `World` owns entity
//! lifecycle and the raw component storage it delegates to
//! (`component.rs`'s `SparseShell`); *iterating* over that storage is a
//! distinct concern with its own file, matching the crate's own
//! established one-concept-per-file convention (`world.rs` = entities,
//! `component.rs` = the Sparse Shell, this file = iterating it,
//! `archetype.rs` = the Archetype Core). These methods were first written
//! directly on `World` in `world.rs` and moved here shortly after —
//! caught as a real organizational miss, not a design change; the
//! implementations are unchanged.
//!
//! Archetype-Core query support (iterating dense table storage) is a
//! separate, later addition to this same file, once `archetype.rs`
//! exists to iterate over.

use crate::world::{Entity, World};

impl World {
    /// Iterates every `(Entity, &T)` currently alive with a `T`
    /// component attached (Sparse Shell only — see `archetype.rs` for
    /// the Archetype Core's own component storage). Doesn't separately
    /// check liveness per entity — every entity in `T`'s storage is
    /// alive by construction, since `despawn` removes an entity from
    /// every component column before its slot is ever freed (see
    /// `World::despawn`'s own doc comment).
    pub fn query<T: 'static>(&self) -> impl Iterator<Item = (Entity, &T)> + '_ {
        self.components.iter::<T>()
    }

    /// Iterates every `(Entity, &A, &B)` for entities alive with *both*
    /// an `A` and a `B` component attached (Sparse Shell only).
    ///
    /// Drives iteration off `A`'s storage and checks `B` per entity —
    /// not off whichever of the two is actually smaller. A real
    /// deliberate v1 simplification, not an oversight: picking the
    /// smaller side is a real optimization for a lopsided pair, but
    /// there's no consumer yet whose real query shapes would justify it
    /// over just shipping the correct, simpler version first. Revisit
    /// against a real workload, not speculatively.
    pub fn query2<A: 'static, B: 'static>(&self) -> impl Iterator<Item = (Entity, &A, &B)> + '_ {
        self.components
            .iter::<A>()
            .filter_map(move |(entity, a)| self.components.get::<B>(entity).map(|b| (entity, a, b)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, PartialEq, Clone, Copy)]
    struct Position {
        x: f32,
        y: f32,
    }

    #[derive(Debug, PartialEq, Clone, Copy)]
    struct Velocity {
        dx: f32,
        dy: f32,
    }

    #[test]
    fn query_iterates_every_entity_with_the_component() {
        let mut w = World::new();
        let e1 = w.spawn();
        let e2 = w.spawn();
        let e3 = w.spawn(); // no Position at all
        w.insert(e1, Position { x: 1.0, y: 1.0 });
        w.insert(e2, Position { x: 2.0, y: 2.0 });
        let _ = e3;

        let mut found: Vec<(Entity, Position)> =
            w.query::<Position>().map(|(e, p)| (e, *p)).collect();
        found.sort_by_key(|(e, _)| e.index());

        assert_eq!(
            found,
            vec![
                (e1, Position { x: 1.0, y: 1.0 }),
                (e2, Position { x: 2.0, y: 2.0 })
            ]
        );
    }

    #[test]
    fn query_on_never_inserted_type_is_empty() {
        let w = World::new();
        assert_eq!(w.query::<Position>().count(), 0);
    }

    #[test]
    fn query_excludes_despawned_entities() {
        let mut w = World::new();
        let e1 = w.spawn();
        let e2 = w.spawn();
        w.insert(e1, Position { x: 0.0, y: 0.0 });
        w.insert(e2, Position { x: 0.0, y: 0.0 });

        w.despawn(e1);

        let found: Vec<Entity> = w.query::<Position>().map(|(e, _)| e).collect();
        assert_eq!(found, vec![e2]);
    }

    #[test]
    fn query2_yields_only_entities_with_both_components() {
        let mut w = World::new();
        let both = w.spawn();
        let position_only = w.spawn();
        let velocity_only = w.spawn();

        w.insert(both, Position { x: 1.0, y: 1.0 });
        w.insert(both, Velocity { dx: 0.5, dy: 0.5 });
        w.insert(position_only, Position { x: 2.0, y: 2.0 });
        w.insert(velocity_only, Velocity { dx: 9.0, dy: 9.0 });

        let found: Vec<Entity> = w
            .query2::<Position, Velocity>()
            .map(|(e, _, _)| e)
            .collect();
        assert_eq!(found, vec![both]);
    }

    #[test]
    fn query2_returns_matching_component_references() {
        let mut w = World::new();
        let e = w.spawn();
        w.insert(e, Position { x: 3.0, y: 4.0 });
        w.insert(e, Velocity { dx: 1.0, dy: -1.0 });

        let (found_entity, pos, vel) = w.query2::<Position, Velocity>().next().unwrap();
        assert_eq!(found_entity, e);
        assert_eq!(*pos, Position { x: 3.0, y: 4.0 });
        assert_eq!(*vel, Velocity { dx: 1.0, dy: -1.0 });
    }

    #[test]
    fn query2_empty_when_one_side_was_never_registered() {
        let mut w = World::new();
        let e = w.spawn();
        w.insert(e, Position { x: 0.0, y: 0.0 });
        // Velocity has never been inserted for anything, anywhere.
        assert_eq!(w.query2::<Position, Velocity>().count(), 0);
    }

    #[test]
    fn query2_excludes_a_despawned_entity_even_if_it_had_both() {
        let mut w = World::new();
        let e = w.spawn();
        w.insert(e, Position { x: 0.0, y: 0.0 });
        w.insert(e, Velocity { dx: 0.0, dy: 0.0 });
        w.despawn(e);

        assert_eq!(w.query2::<Position, Velocity>().count(), 0);
    }
}
