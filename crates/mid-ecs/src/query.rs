//! Query iteration over both storage systems: the Sparse Shell
//! (`query`/`query2`) and the Archetype Core (`query_static`/
//! `query2_static`).
//!
//! Lives here, not on `World` itself in `world.rs` — `World` owns entity
//! lifecycle and the raw component storage it delegates to
//! (`component.rs`'s `SparseShell`, `archetype.rs`'s `Archetypes`);
//! *iterating* over that storage is a distinct concern with its own
//! file, matching the crate's own established one-concept-per-file
//! convention (`world.rs` = entities, `component.rs` = the Sparse
//! Shell, `archetype.rs` = the Archetype Core, this file = iterating
//! both). The Sparse Shell methods were first written directly on
//! `World` in `world.rs` and moved here shortly after — caught as a
//! real organizational miss, not a design change; the implementations
//! were unchanged. The Archetype Core methods (`_static` suffix,
//! matching `insert_static`/`get_static`'s own naming on `World`) are a
//! later addition, once `archetype.rs`'s own `Archetypes::iter`/`iter2`
//! existed to wrap.

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

    /// Iterates every `(Entity, &T)` currently alive with an
    /// archetype-tracked `T` attached — the Archetype Core counterpart
    /// to [`Self::query`]. Thin wrapper over `Archetypes::iter`; see
    /// that method's own doc comment for the real, unavoidable
    /// difference from the Sparse Shell side: `T`'s data is spread
    /// across every archetype whose signature includes it, not one
    /// place.
    pub fn query_static<T: 'static>(&self) -> impl Iterator<Item = (Entity, &T)> + '_ {
        self.archetypes.iter::<T>()
    }

    /// Iterates every `(Entity, &A, &B)` for entities alive with *both*
    /// an archetype-tracked `A` and `B` attached — the Archetype Core
    /// counterpart to [`Self::query2`]. Thin wrapper over
    /// `Archetypes::iter2`, which — unlike this method's Sparse Shell
    /// counterpart — does *not* look up `B` per entity; see
    /// `Archetypes::iter2`'s own doc comment for why the Archetype
    /// Core side needed a hand-written iterator to get there, not just
    /// a per-entity-lookup fix.
    pub fn query2_static<A: 'static, B: 'static>(
        &self,
    ) -> impl Iterator<Item = (Entity, &A, &B)> + '_ {
        self.archetypes.iter2::<A, B>()
    }

    // ── TEMPORARY, real-CI inlining-regression diagnostic ──
    // See src/diag_inline.rs's own doc comment for the full story.
    // Delete these two methods together with that module once the
    // investigation concludes. `pub`, not `pub(crate)`, only because
    // `benches/archetype_core.rs` is compiled as an external binary
    // and needs real public API to reach them — not meant for any
    // other use.
    #[doc(hidden)]
    pub fn query2_static_diag_never<A: 'static, B: 'static>(
        &self,
    ) -> impl Iterator<Item = (Entity, &A, &B)> + '_ {
        self.archetypes.iter2_diag_never::<A, B>()
    }

    #[doc(hidden)]
    pub fn query2_static_diag_always<A: 'static, B: 'static>(
        &self,
    ) -> impl Iterator<Item = (Entity, &A, &B)> + '_ {
        self.archetypes.iter2_diag_always::<A, B>()
    }

    // ── TEMPORARY, real-CI query2 unsafe/shape diagnostic ──
    // See src/diag_query2_unchecked.rs's own NOTICE header and
    // docs/mid-ecs.md's "diag_query2_unchecked.rs" section for the
    // full story. Delete these four methods together with that module
    // once the investigation concludes. `pub`, not `pub(crate)`, for
    // the same reason as the methods above.
    #[doc(hidden)]
    pub fn query_static_diag_unchecked<T: 'static>(
        &self,
    ) -> impl Iterator<Item = (Entity, &T)> + '_ {
        self.archetypes.iter_diag_unchecked::<T>()
    }

    #[doc(hidden)]
    pub fn query2_static_diag_unchecked<A: 'static, B: 'static>(
        &self,
    ) -> impl Iterator<Item = (Entity, &A, &B)> + '_ {
        self.archetypes.iter2_diag_unchecked::<A, B>()
    }

    #[doc(hidden)]
    pub fn query2_static_diag_two_tuple_item<A: 'static + Clone, B: 'static>(
        &self,
        combine: fn(&A, &B) -> A,
    ) -> impl Iterator<Item = (Entity, A)> + '_ {
        self.archetypes.iter2_diag_two_tuple_item::<A, B>(combine)
    }

    #[doc(hidden)]
    pub fn query2_static_diag_unused_b_col<A: 'static, B: 'static>(
        &self,
    ) -> impl Iterator<Item = (Entity, &A)> + '_ {
        self.archetypes.iter2_diag_unused_b_col::<A, B>()
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

    // ── Archetype Core (`_static`) ───────────────────────────────────────────

    #[test]
    fn query_static_iterates_every_entity_with_the_component() {
        let mut w = World::new();
        let e1 = w.spawn();
        let e2 = w.spawn();
        let e3 = w.spawn(); // no Position at all
        assert!(w.insert_static(e1, Position { x: 1.0, y: 1.0 }));
        assert!(w.insert_static(e2, Position { x: 2.0, y: 2.0 }));
        let _ = e3;

        let mut found: Vec<(Entity, Position)> =
            w.query_static::<Position>().map(|(e, p)| (e, *p)).collect();
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
    fn query_static_on_never_inserted_type_is_empty() {
        let w = World::new();
        assert_eq!(w.query_static::<Position>().count(), 0);
    }

    #[test]
    fn query_static_excludes_despawned_entities() {
        let mut w = World::new();
        let e1 = w.spawn();
        let e2 = w.spawn();
        assert!(w.insert_static(e1, Position { x: 0.0, y: 0.0 }));
        assert!(w.insert_static(e2, Position { x: 0.0, y: 0.0 }));

        w.despawn(e1);

        let found: Vec<Entity> = w.query_static::<Position>().map(|(e, _)| e).collect();
        assert_eq!(found, vec![e2]);
    }

    #[test]
    fn query_static_finds_the_component_across_multiple_distinct_archetypes() {
        // The one thing with no Sparse Shell equivalent: Position-having
        // entities here are deliberately split across two different
        // archetypes ({Position} and {Position, Velocity}) -- proving
        // query_static actually chains across archetypes_with's
        // fragmentation, not just reads one table.
        let mut w = World::new();
        let position_only = w.spawn();
        let position_and_velocity = w.spawn();
        assert!(w.insert_static(position_only, Position { x: 1.0, y: 1.0 }));
        assert!(w.insert_static(position_and_velocity, Position { x: 2.0, y: 2.0 }));
        assert!(w.insert_static(position_and_velocity, Velocity { dx: 9.0, dy: 9.0 }));

        let mut found: Vec<Entity> = w.query_static::<Position>().map(|(e, _)| e).collect();
        found.sort_by_key(|e| e.index());
        let mut expected = vec![position_only, position_and_velocity];
        expected.sort_by_key(|e| e.index());
        assert_eq!(found, expected);
    }

    #[test]
    fn query2_static_yields_only_entities_with_both_components() {
        let mut w = World::new();
        let both = w.spawn();
        let position_only = w.spawn();
        let velocity_only = w.spawn();

        assert!(w.insert_static(both, Position { x: 1.0, y: 1.0 }));
        assert!(w.insert_static(both, Velocity { dx: 0.5, dy: 0.5 }));
        assert!(w.insert_static(position_only, Position { x: 2.0, y: 2.0 }));
        assert!(w.insert_static(velocity_only, Velocity { dx: 9.0, dy: 9.0 }));

        let found: Vec<Entity> = w
            .query2_static::<Position, Velocity>()
            .map(|(e, _, _)| e)
            .collect();
        assert_eq!(found, vec![both]);
    }

    #[test]
    fn query2_static_returns_matching_component_references() {
        let mut w = World::new();
        let e = w.spawn();
        assert!(w.insert_static(e, Position { x: 3.0, y: 4.0 }));
        assert!(w.insert_static(e, Velocity { dx: 1.0, dy: -1.0 }));

        let (found_entity, pos, vel) = w.query2_static::<Position, Velocity>().next().unwrap();
        assert_eq!(found_entity, e);
        assert_eq!(*pos, Position { x: 3.0, y: 4.0 });
        assert_eq!(*vel, Velocity { dx: 1.0, dy: -1.0 });
    }

    #[test]
    fn query2_static_empty_when_one_side_was_never_registered() {
        let mut w = World::new();
        let e = w.spawn();
        assert!(w.insert_static(e, Position { x: 0.0, y: 0.0 }));
        // Velocity has never been inserted as an archetype-tracked
        // component for anything, anywhere.
        assert_eq!(w.query2_static::<Position, Velocity>().count(), 0);
    }

    #[test]
    fn query2_static_excludes_a_despawned_entity_even_if_it_had_both() {
        let mut w = World::new();
        let e = w.spawn();
        assert!(w.insert_static(e, Position { x: 0.0, y: 0.0 }));
        assert!(w.insert_static(e, Velocity { dx: 0.0, dy: 0.0 }));
        w.despawn(e);

        assert_eq!(w.query2_static::<Position, Velocity>().count(), 0);
    }

    #[test]
    fn query_and_query_static_are_genuinely_independent_storage() {
        // Two distinct types, one per storage system -- real proof
        // Sparse Shell and Archetype Core are independent storage
        // systems, matching their own already-independent ComponentId
        // namespaces at the storage level (see archetype.rs's own
        // dedicated test for that). Deliberately NOT the same type used
        // with both systems: World's own StorageClaims guard forbids
        // exactly that now (see world.rs's own doc comment on it) --
        // using one type with both is the actual footgun this test used
        // to (accidentally) demonstrate, not a supported pattern.
        let mut w = World::new();
        let sparse_entity = w.spawn();
        let static_entity = w.spawn();
        w.insert(sparse_entity, Position { x: 1.0, y: 1.0 });
        assert!(w.insert_static(static_entity, Velocity { dx: 2.0, dy: 2.0 }));

        let sparse_found: Vec<Entity> = w.query::<Position>().map(|(e, _)| e).collect();
        let static_found: Vec<Entity> = w.query_static::<Velocity>().map(|(e, _)| e).collect();
        assert_eq!(sparse_found, vec![sparse_entity]);
        assert_eq!(static_found, vec![static_entity]);
    }

    // ── TEMPORARY, for src/diag_query2_unchecked.rs — delete together.
    // Each `unsafe`-using diagnostic variant needs its own correctness
    // proof against the real, safe `query_static`/`query2_static`
    // output before it's trustworthy enough to bench on real CI at
    // all — a variant that's merely fast and wrong isn't useful data.

    fn two_archetype_world() -> (World, Entity, Entity) {
        // Same fragmentation shape as
        // query_static_finds_the_component_across_multiple_distinct_archetypes
        // above, reused here so every diagnostic variant is checked
        // against a real multi-archetype case, not just the trivial
        // single-archetype one.
        let mut w = World::new();
        let both = w.spawn();
        let position_only = w.spawn();
        assert!(w.insert_static(both, Position { x: 1.0, y: 1.0 }));
        assert!(w.insert_static(both, Velocity { dx: 9.0, dy: 9.0 }));
        assert!(w.insert_static(position_only, Position { x: 2.0, y: 2.0 }));
        (w, both, position_only)
    }

    #[test]
    fn diag_query_static_unchecked_matches_the_real_query_static() {
        let (w, both, position_only) = two_archetype_world();

        let mut expected: Vec<(Entity, Position)> =
            w.query_static::<Position>().map(|(e, p)| (e, *p)).collect();
        let mut actual: Vec<(Entity, Position)> = w
            .query_static_diag_unchecked::<Position>()
            .map(|(e, p)| (e, *p))
            .collect();
        expected.sort_by_key(|(e, _)| e.index());
        actual.sort_by_key(|(e, _)| e.index());

        assert_eq!(actual, expected);
        assert_eq!(actual.len(), 2);
        assert!(actual.iter().any(|(e, _)| *e == both));
        assert!(actual.iter().any(|(e, _)| *e == position_only));
    }

    #[test]
    fn diag_query_static_unchecked_on_never_inserted_type_is_empty() {
        let w = World::new();
        assert_eq!(w.query_static_diag_unchecked::<Position>().count(), 0);
    }

    #[test]
    fn diag_query2_static_unchecked_matches_the_real_query2_static() {
        let (w, both, _position_only) = two_archetype_world();

        let expected: Vec<(Entity, Position, Velocity)> = w
            .query2_static::<Position, Velocity>()
            .map(|(e, p, v)| (e, *p, *v))
            .collect();
        let actual: Vec<(Entity, Position, Velocity)> = w
            .query2_static_diag_unchecked::<Position, Velocity>()
            .map(|(e, p, v)| (e, *p, *v))
            .collect();

        assert_eq!(actual, expected);
        assert_eq!(
            actual,
            vec![(
                both,
                Position { x: 1.0, y: 1.0 },
                Velocity { dx: 9.0, dy: 9.0 }
            )]
        );
    }

    #[test]
    fn diag_query2_static_unchecked_empty_when_one_side_was_never_registered() {
        let mut w = World::new();
        let e = w.spawn();
        assert!(w.insert_static(e, Position { x: 0.0, y: 0.0 }));
        assert_eq!(
            w.query2_static_diag_unchecked::<Position, Velocity>()
                .count(),
            0
        );
    }

    #[test]
    fn diag_query2_static_two_tuple_item_matches_a_manual_combine_over_the_real_query2_static() {
        let (w, both, _position_only) = two_archetype_world();
        fn combine(p: &Position, v: &Velocity) -> Position {
            Position {
                x: p.x + v.dx,
                y: p.y + v.dy,
            }
        }

        let expected: Vec<(Entity, Position)> = w
            .query2_static::<Position, Velocity>()
            .map(|(e, p, v)| (e, combine(p, v)))
            .collect();
        let actual: Vec<(Entity, Position)> = w
            .query2_static_diag_two_tuple_item::<Position, Velocity>(combine)
            .collect();

        assert_eq!(actual, expected);
        assert_eq!(actual, vec![(both, Position { x: 10.0, y: 10.0 })]);
    }

    #[test]
    fn diag_query2_static_unused_b_col_matches_query2_static_first_component_only() {
        let (w, both, _position_only) = two_archetype_world();

        let expected: Vec<(Entity, Position)> = w
            .query2_static::<Position, Velocity>()
            .map(|(e, p, _v)| (e, *p))
            .collect();
        let actual: Vec<(Entity, Position)> = w
            .query2_static_diag_unused_b_col::<Position, Velocity>()
            .map(|(e, p)| (e, *p))
            .collect();

        assert_eq!(actual, expected);
        assert_eq!(actual, vec![(both, Position { x: 1.0, y: 1.0 })]);
    }
}
