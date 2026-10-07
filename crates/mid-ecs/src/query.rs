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

use crate::archetype::ArchetypeId;
use crate::filter::QueryFilter;
use crate::tick::{ChangeTracker, Mut, Tick};
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

    /// Entity-free counterpart to [`Self::query_static`]: yields `&T`
    /// alone. Same data, same order, same archetype walk — the entity is
    /// simply not in the item.
    ///
    /// Exists for symmetry with [`Self::query2_static_ref`], which is
    /// where the real reason lives. `query_static`'s own item is already
    /// under the 16-byte register-return threshold, so this one is not
    /// expected to be faster than it — if it measures faster by any
    /// meaningful margin, that is a result worth chasing, not a win to
    /// quietly bank.
    pub fn query_static_ref<T: 'static>(&self) -> impl Iterator<Item = &T> + '_ {
        self.archetypes.iter_ref::<T>()
    }

    /// Entity-free counterpart to [`Self::query2_static`]: yields
    /// `(&A, &B)` instead of `(Entity, &A, &B)`.
    ///
    /// **This is the apples-to-apples shape against `bevy_ecs`.**
    /// `bevy_ecs`'s own `Query<(&A, &B)>` yields exactly `(&A, &B)` —
    /// `Entity` there is opt-in query data, not a mandatory first tuple
    /// element — so `benches/ecs-vs-bevy-ecs`'s `dense_query_iteration`
    /// has never actually compared the same thing on both sides. It puts
    /// `query2_static`'s three-element item against bevy's two-element
    /// one and discards the entity with `_` on the mid-ecs side only.
    ///
    /// Why that matters, and why this method exists rather than a tuning
    /// flag: `Entity` is 8 bytes (a `mid-collections`
    /// `GenerationalIndex`, two `u32`s) and every reference is 8, so
    /// `Option<(Entity, &A, &B)>` is 24 bytes while
    /// `Option<(&A, &B)>` is 16. System V AMD64 §3.2.3 returns an
    /// aggregate over two eightbytes in MEMORY — a hidden pointer, a
    /// caller-allocated stack slot, a store and a reload, per item —
    /// and anything at or under 16 bytes in the RAX:RDX register pair.
    /// That threshold falls exactly between `query_static` (16 B, at
    /// parity with bevy: 9.42µs vs 9.35µs) and `query2_static` (24 B,
    /// 3.99x). It is the one structural difference that no amount of
    /// `unsafe`, `#[inline(always)]`, raw-pointer columns or cold-path
    /// splitting could ever have reached — which is exactly why all four
    /// of those came back negative (`docs/mid-ecs.md`, builds #11-#18).
    ///
    /// **Tested, and refuted as the full explanation.**
    /// `benches/query2-ref-isolated` ran this exact method, real
    /// `mid-ecs`, nothing else in the compilation unit — the same
    /// isolation that took `Iter1` to parity — and it still sits at
    /// ~4x the raw-slice floor in both `bench` and `bench-nolto`
    /// profiles (Query2-Ref Isolated builds #1/#2). Sixteen bytes was
    /// supposed to be the side of the register/memory threshold that
    /// gets inlined cleanly; it isn't, here. Whatever the real cause
    /// is, it isn't the ABI classification by itself, and it isn't a
    /// compilation-unit-size artifact either — isolation is exactly
    /// what fixed `Iter1`, and it didn't fix this. Still an open
    /// question, not a closed one; see `Iter2RefUncheckedAlways`
    /// (`archetype/iter.rs`) for the next thing actually being tried,
    /// not this comment, for the current state.
    pub fn query2_static_ref<A: 'static, B: 'static>(&self) -> impl Iterator<Item = (&A, &B)> + '_ {
        self.archetypes.iter2_ref::<A, B>()
    }

    // Filtered queries (`With` / `Without`, see `filter.rs`). One method
    // per unfiltered Archetype Core query above, with the filter `F` as
    // the last type parameter (`()` for none). Archetype Core only; see
    // docs/mid-ecs.md, section "filter.rs", for why.

    /// [`Self::query_static`] restricted to archetypes satisfying `F`.
    ///
    /// ```
    /// use mid_ecs::{With, Without, World};
    ///
    /// struct Position { x: f32 }
    /// struct Player;
    /// struct Frozen;
    ///
    /// let mut world = World::new();
    /// let walker = world.spawn_bundle((Position { x: 1.0 }, Player));
    /// let _frozen = world.spawn_bundle((Position { x: 2.0 }, Player, Frozen));
    /// let _npc = world.spawn_bundle((Position { x: 3.0 },));
    ///
    /// // Every (Entity, &Position) whose archetype holds `Player` and
    /// // does not hold `Frozen`:
    /// let found: Vec<_> = world
    ///     .query_static_filtered::<Position, (With<Player>, Without<Frozen>)>()
    ///     .map(|(e, p)| (e, p.x))
    ///     .collect();
    /// assert_eq!(found, vec![(walker, 1.0)]);
    /// ```
    ///
    /// The filter is evaluated once per archetype when the query is
    /// created, not per row, and the returned iterator is the same
    /// type [`Self::query_static`] returns — see `filter.rs`.
    pub fn query_static_filtered<T: 'static, F: QueryFilter>(
        &self,
    ) -> impl Iterator<Item = (Entity, &T)> + '_ {
        self.archetypes.iter_filtered::<T, F>()
    }

    /// [`Self::query2_static`] restricted to archetypes satisfying `F`.
    pub fn query2_static_filtered<A: 'static, B: 'static, F: QueryFilter>(
        &self,
    ) -> impl Iterator<Item = (Entity, &A, &B)> + '_ {
        self.archetypes.iter2_filtered::<A, B, F>()
    }

    /// [`Self::query_static_ref`] restricted to archetypes satisfying
    /// `F`.
    pub fn query_static_ref_filtered<T: 'static, F: QueryFilter>(
        &self,
    ) -> impl Iterator<Item = &T> + '_ {
        self.archetypes.iter_ref_filtered::<T, F>()
    }

    /// [`Self::query2_static_ref`] restricted to archetypes satisfying
    /// `F` — the shape that lines up with bevy's
    /// `Query<(&A, &B), F>`.
    pub fn query2_static_ref_filtered<A: 'static, B: 'static, F: QueryFilter>(
        &self,
    ) -> impl Iterator<Item = (&A, &B)> + '_ {
        self.archetypes.iter2_ref_filtered::<A, B, F>()
    }

    // ── Bulk mutable queries ────────────────────────────────────────
    //
    // The bulk counterpart of `get_static_mut`: every item carries a
    // [`Mut`], which stamps a row changed only when it is actually written
    // through, so `query_changed` (and the C `..._changed_rows`) see bulk
    // writes. Archetype Core only, `&mut self`, so no structural change
    // can happen while an iterator is alive (the borrow checker, not a
    // runtime check). Implemented in `archetype/iter_mut.rs`, deliberately
    // not on `Iter1`/`Iter2`; see that file and `docs/mid-ecs.md`,
    // "Mutable queries", for the design and what is still missing.

    /// Iterates every `(Entity, Mut<T>)` with an archetype-tracked `T`:
    /// the mutable counterpart of [`Self::query_static`], same archetype
    /// and row order. Writing through the `Mut` marks that row changed;
    /// merely visiting it does not.
    ///
    /// ```
    /// use mid_ecs::{ChangeTracker, World};
    ///
    /// struct Hp(u32);
    /// let mut world = World::new();
    /// let a = world.spawn();
    /// let b = world.spawn();
    /// world.insert_static(a, Hp(10));
    /// world.insert_static(b, Hp(20));
    ///
    /// let mut tracker = ChangeTracker::new();
    /// tracker.update(&world);
    /// world.increment_change_tick();
    /// for (_, mut hp) in world.query_static_mut::<Hp>() {
    ///     if hp.0 > 15 {
    ///         hp.0 -= 5; // only this row is marked changed
    ///     }
    /// }
    /// world.increment_change_tick();
    /// let changed: Vec<_> = world.query_changed::<Hp>(&tracker).map(|(e, hp)| (e, hp.0)).collect();
    /// assert_eq!(changed, vec![(b, 15)]);
    /// ```
    pub fn query_static_mut<T: 'static>(
        &mut self,
    ) -> impl Iterator<Item = (Entity, Mut<'_, T>)> + '_ {
        self.query_static_mut_filtered::<T, ()>()
    }

    /// [`Self::query_static_mut`] restricted to archetypes satisfying `F`.
    pub fn query_static_mut_filtered<T: 'static, F: QueryFilter>(
        &mut self,
    ) -> impl Iterator<Item = (Entity, Mut<'_, T>)> + '_ {
        let this_run = self.change_tick();
        self.archetypes.iter_mut_filtered::<T, F>(this_run)
    }

    /// Iterates every `(Entity, Mut<A>, Mut<B>)` for entities holding both
    /// archetype-tracked components, each independently change-tracked.
    ///
    /// # Panics
    /// If `A` and `B` are the same type (one value cannot be borrowed
    /// mutably twice). Raised when the query is created, not part-way
    /// through it.
    pub fn query2_static_mut<A: 'static, B: 'static>(
        &mut self,
    ) -> impl Iterator<Item = (Entity, Mut<'_, A>, Mut<'_, B>)> + '_ {
        self.query2_static_mut_filtered::<A, B, ()>()
    }

    /// [`Self::query2_static_mut`] restricted to archetypes satisfying
    /// `F`. Panics if `A` and `B` are the same type.
    pub fn query2_static_mut_filtered<A: 'static, B: 'static, F: QueryFilter>(
        &mut self,
    ) -> impl Iterator<Item = (Entity, Mut<'_, A>, Mut<'_, B>)> + '_ {
        let this_run = self.change_tick();
        self.archetypes.iter2_mut_filtered::<A, B, F>(this_run)
    }

    /// Iterates every `(Entity, Mut<A>, &B)`: `A` mutable and
    /// change-tracked, `B` read-only and untouched, the
    /// `(&mut Position, &Velocity)` shape that most per-frame updates
    /// are. `B`'s change ticks are not read or written.
    ///
    /// # Panics
    /// If `A` and `B` are the same type.
    pub fn query2_static_mut_ref<A: 'static, B: 'static>(
        &mut self,
    ) -> impl Iterator<Item = (Entity, Mut<'_, A>, &B)> + '_ {
        self.query2_static_mut_ref_filtered::<A, B, ()>()
    }

    /// [`Self::query2_static_mut_ref`] restricted to archetypes satisfying
    /// `F`. Panics if `A` and `B` are the same type.
    pub fn query2_static_mut_ref_filtered<A: 'static, B: 'static, F: QueryFilter>(
        &mut self,
    ) -> impl Iterator<Item = (Entity, Mut<'_, A>, &B)> + '_ {
        let this_run = self.change_tick();
        self.archetypes.iter2_mut_ref_filtered::<A, B, F>(this_run)
    }

    // ── Change-detection queries: `Added`/`Changed` ─────────────────
    //
    // Deliberately their own methods, not `With`/`Without`-style
    // `QueryFilter` types composed into `*_filtered`: those are
    // archetype-level (resolved once, building `matched`), but
    // added/changed are per-row facts, and checking them inside
    // `Iter1`/`Iter2`/`Iter1Ref`/`Iter2Ref`'s `next()` is exactly the
    // kind of change `archetype/iter.rs`'s own header already warns
    // off — those bodies are held byte-for-byte fixed for the LTO
    // inline budget. So these walk `Archetypes::rows_with_ticks`
    // directly (see `tick.rs`) instead, a separate path from every
    // other query on this `impl World`, and never touch the tuned
    // iterators. Single-component only, and not combinable with
    // `With`/`Without` yet — see `docs/mid-ecs.md`, "tick.rs", for the
    // scope this pass stopped at and why.

    /// Iterates every `(Entity, &T)` whose `T` was inserted after
    /// `tracker`'s own last-checked tick — including every one that
    /// currently exists, the first time a given [`ChangeTracker`] is
    /// used. Archetype Core only, like every other `_static` query.
    pub fn query_added<T: 'static>(
        &self,
        tracker: &ChangeTracker,
    ) -> impl Iterator<Item = (Entity, &T)> + '_ {
        self.change_rows::<T>(tracker.last_run(), true)
    }

    /// Iterates every `(Entity, &T)` whose `T` was inserted or last
    /// mutated through [`Self::get_static_mut`] after `tracker`'s own
    /// last-checked tick.
    pub fn query_changed<T: 'static>(
        &self,
        tracker: &ChangeTracker,
    ) -> impl Iterator<Item = (Entity, &T)> + '_ {
        self.change_rows::<T>(tracker.last_run(), false)
    }

    fn change_rows<T: 'static>(
        &self,
        last_run: Tick,
        added_only: bool,
    ) -> impl Iterator<Item = (Entity, &T)> + '_ {
        let this_run = self.change_tick();
        let id = self.archetypes.existing_component_id::<T>();
        let matched: Vec<ArchetypeId> = match id {
            Some(id) => self.archetypes.archetypes_with(id).collect(),
            None => Vec::new(),
        };
        matched.into_iter().flat_map(move |archetype_id| {
            let id = id.expect("matched is only ever populated when id is Some");
            let (entities, values, ticks) = self
                .archetypes
                .rows_with_ticks::<T>(archetype_id, id)
                .expect("every archetype in matched was just confirmed to hold this component");
            entities
                .iter()
                .copied()
                .zip(values.iter())
                .zip(ticks.iter())
                .filter_map(move |((entity, value), row_ticks)| {
                    let keep = if added_only {
                        row_ticks.is_added(last_run, this_run)
                    } else {
                        row_ticks.is_changed(last_run, this_run)
                    };
                    keep.then_some((entity, value))
                })
        })
    }

    // ── TEMPORARY, real-CI-only: unsafe + forced inlining, in true
    // isolation. See `archetype/iter.rs`'s own doc comment on
    // `Iter2RefUncheckedAlways` and `benches/query2-ref-isolated` for
    // the full story. `pub`, not `pub(crate)`, for the same reason as
    // every other `#[doc(hidden)]` method here: that crate is external
    // to this one and needs real public API to reach it.
    #[doc(hidden)]
    pub fn query2_static_ref_unchecked_always<A: 'static, B: 'static>(
        &self,
    ) -> impl Iterator<Item = (&A, &B)> + '_ {
        self.archetypes.iter2_ref_unchecked_always::<A, B>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filter::Or;

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

    #[test]
    fn query_static_ref_yields_the_same_values_as_query_static() {
        let mut w = World::new();
        let e1 = w.spawn();
        let e2 = w.spawn();
        assert!(w.insert_static(e1, Position { x: 1.0, y: 1.0 }));
        assert!(w.insert_static(e2, Position { x: 2.0, y: 2.0 }));

        let mut expected: Vec<Position> = w.query_static::<Position>().map(|(_, p)| *p).collect();
        let mut actual: Vec<Position> = w.query_static_ref::<Position>().copied().collect();
        expected.sort_by(|a, b| a.x.partial_cmp(&b.x).unwrap());
        actual.sort_by(|a, b| a.x.partial_cmp(&b.x).unwrap());
        assert_eq!(actual, expected);
        assert_eq!(actual.len(), 2);
    }

    #[test]
    fn query2_static_ref_returns_matching_component_references() {
        let mut w = World::new();
        let e = w.spawn();
        assert!(w.insert_static(e, Position { x: 3.0, y: 4.0 }));
        assert!(w.insert_static(e, Velocity { dx: 1.0, dy: -1.0 }));

        let (pos, vel) = w.query2_static_ref::<Position, Velocity>().next().unwrap();
        assert_eq!(*pos, Position { x: 3.0, y: 4.0 });
        assert_eq!(*vel, Velocity { dx: 1.0, dy: -1.0 });
    }

    #[test]
    fn query2_static_ref_empty_when_one_side_was_never_registered() {
        let mut w = World::new();
        let e = w.spawn();
        assert!(w.insert_static(e, Position { x: 0.0, y: 0.0 }));
        assert_eq!(w.query2_static_ref::<Position, Velocity>().count(), 0);
    }

    #[test]
    fn query2_static_ref_unchecked_always_matches_the_real_query2_static_ref() {
        let mut w = World::new();
        let e1 = w.spawn();
        let e2 = w.spawn();
        assert!(w.insert_static(e1, Position { x: 1.0, y: 1.0 }));
        assert!(w.insert_static(e1, Velocity { dx: 0.5, dy: 0.5 }));
        assert!(w.insert_static(e2, Position { x: 2.0, y: 2.0 }));
        assert!(w.insert_static(e2, Velocity { dx: 1.5, dy: 1.5 }));

        let mut expected: Vec<(Position, Velocity)> = w
            .query2_static_ref::<Position, Velocity>()
            .map(|(p, v)| (*p, *v))
            .collect();
        let mut actual: Vec<(Position, Velocity)> = w
            .query2_static_ref_unchecked_always::<Position, Velocity>()
            .map(|(p, v)| (*p, *v))
            .collect();
        expected.sort_by(|a, b| a.0.x.partial_cmp(&b.0.x).unwrap());
        actual.sort_by(|a, b| a.0.x.partial_cmp(&b.0.x).unwrap());

        assert_eq!(actual, expected);
        assert_eq!(actual.len(), 2);
    }

    #[test]
    fn query2_static_ref_unchecked_always_empty_when_one_side_was_never_registered() {
        let mut w = World::new();
        let e = w.spawn();
        assert!(w.insert_static(e, Position { x: 0.0, y: 0.0 }));
        assert_eq!(
            w.query2_static_ref_unchecked_always::<Position, Velocity>()
                .count(),
            0
        );
    }

    // ── Filtered queries (`With` / `Without`) ───────────────────────

    use crate::filter::{With, Without};

    /// Zero-sized markers, archetype-tracked via `insert_static`.
    struct Player;
    struct Frozen;
    /// Only ever inserted through the Sparse Shell's `insert`, never an
    /// archetype-tracked component.
    struct SparseOnly;
    /// Never inserted anywhere.
    struct NeverInserted;

    /// Four entities, all with `Position` and `Velocity`, spread over
    /// four distinct archetypes by which markers they carry:
    /// `[plain, player, frozen, both]`.
    fn filter_world() -> (World, [Entity; 4]) {
        let mut w = World::new();
        let mut make =
            |x: f32| w.spawn_bundle((Position { x, y: 0.0 }, Velocity { dx: x, dy: 0.0 }));
        let plain = make(1.0);
        let player = make(2.0);
        let frozen = make(3.0);
        let both = make(4.0);
        assert!(w.insert_static(player, Player));
        assert!(w.insert_static(frozen, Frozen));
        assert!(w.insert_static(both, Player));
        assert!(w.insert_static(both, Frozen));
        (w, [plain, player, frozen, both])
    }

    fn sorted(mut v: Vec<Entity>) -> Vec<Entity> {
        v.sort_by_key(|e| e.index());
        v
    }

    #[test]
    fn query_static_filtered_with_keeps_only_archetypes_containing_the_component() {
        let (w, [_plain, player, _frozen, both]) = filter_world();
        let found = sorted(
            w.query_static_filtered::<Position, With<Player>>()
                .map(|(e, _)| e)
                .collect(),
        );
        assert_eq!(found, sorted(vec![player, both]));
    }

    #[test]
    fn query_static_filtered_without_drops_archetypes_containing_the_component() {
        let (w, [plain, _player, frozen, _both]) = filter_world();
        let found = sorted(
            w.query_static_filtered::<Position, Without<Player>>()
                .map(|(e, _)| e)
                .collect(),
        );
        assert_eq!(found, sorted(vec![plain, frozen]));
    }

    #[test]
    fn query_static_filtered_tuple_combines_with_and_without() {
        let (w, [_plain, player, _frozen, _both]) = filter_world();
        let found: Vec<Entity> = w
            .query_static_filtered::<Position, (With<Player>, Without<Frozen>)>()
            .map(|(e, _)| e)
            .collect();
        assert_eq!(found, vec![player]);
    }

    #[test]
    fn query_static_filtered_yields_the_real_component_values() {
        let (w, [_plain, player, _frozen, both]) = filter_world();
        let mut found: Vec<(Entity, f32)> = w
            .query_static_filtered::<Position, With<Player>>()
            .map(|(e, p)| (e, p.x))
            .collect();
        found.sort_by_key(|(e, _)| e.index());
        let mut expected = vec![(player, 2.0), (both, 4.0)];
        expected.sort_by_key(|(e, _)| e.index());
        assert_eq!(found, expected);
    }

    #[test]
    fn query2_static_filtered_applies_the_filter_and_returns_both_components() {
        let (w, [_plain, player, _frozen, both]) = filter_world();
        let mut found: Vec<(Entity, f32, f32)> = w
            .query2_static_filtered::<Position, Velocity, With<Player>>()
            .map(|(e, p, v)| (e, p.x, v.dx))
            .collect();
        found.sort_by_key(|(e, _, _)| e.index());
        let mut expected = vec![(player, 2.0, 2.0), (both, 4.0, 4.0)];
        expected.sort_by_key(|(e, _, _)| e.index());
        assert_eq!(found, expected);
    }

    #[test]
    fn query_static_ref_filtered_matches_the_entity_carrying_variant() {
        let (w, _) = filter_world();
        let mut with_entity: Vec<f32> = w
            .query_static_filtered::<Position, (With<Frozen>, Without<Player>)>()
            .map(|(_, p)| p.x)
            .collect();
        let mut without_entity: Vec<f32> = w
            .query_static_ref_filtered::<Position, (With<Frozen>, Without<Player>)>()
            .map(|p| p.x)
            .collect();
        with_entity.sort_by(|a, b| a.partial_cmp(b).unwrap());
        without_entity.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(with_entity, vec![3.0]);
        assert_eq!(without_entity, with_entity);
    }

    #[test]
    fn query2_static_ref_filtered_matches_the_entity_carrying_variant() {
        let (w, _) = filter_world();
        let mut with_entity: Vec<(f32, f32)> = w
            .query2_static_filtered::<Position, Velocity, Without<Frozen>>()
            .map(|(_, p, v)| (p.x, v.dx))
            .collect();
        let mut without_entity: Vec<(f32, f32)> = w
            .query2_static_ref_filtered::<Position, Velocity, Without<Frozen>>()
            .map(|(p, v)| (p.x, v.dx))
            .collect();
        with_entity.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        without_entity.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        assert_eq!(with_entity, vec![(1.0, 1.0), (2.0, 2.0)]);
        assert_eq!(without_entity, with_entity);
    }

    #[test]
    fn unit_filter_is_identical_to_the_unfiltered_query_including_order() {
        // `()` must visit exactly the unfiltered constructors' archetype
        // set in the same order -- `matched_filtered` is a separate
        // code path from `iter`/`iter2`/`iter_ref`/`iter2_ref`, and
        // this is what keeps the two from silently drifting.
        let (w, _) = filter_world();

        let a: Vec<Entity> = w.query_static::<Position>().map(|(e, _)| e).collect();
        let b: Vec<Entity> = w
            .query_static_filtered::<Position, ()>()
            .map(|(e, _)| e)
            .collect();
        assert_eq!(a, b);
        assert_eq!(a.len(), 4);

        let a: Vec<Entity> = w
            .query2_static::<Position, Velocity>()
            .map(|(e, _, _)| e)
            .collect();
        let b: Vec<Entity> = w
            .query2_static_filtered::<Position, Velocity, ()>()
            .map(|(e, _, _)| e)
            .collect();
        assert_eq!(a, b);

        let a: Vec<f32> = w.query_static_ref::<Position>().map(|p| p.x).collect();
        let b: Vec<f32> = w
            .query_static_ref_filtered::<Position, ()>()
            .map(|p| p.x)
            .collect();
        assert_eq!(a, b);

        let a: Vec<f32> = w
            .query2_static_ref::<Position, Velocity>()
            .map(|(p, _)| p.x)
            .collect();
        let b: Vec<f32> = w
            .query2_static_ref_filtered::<Position, Velocity, ()>()
            .map(|(p, _)| p.x)
            .collect();
        assert_eq!(a, b);
    }

    #[test]
    fn with_and_without_partition_the_unfiltered_result() {
        // Every entity is in exactly one of the two, and together they
        // are the whole unfiltered set.
        let (w, _) = filter_world();
        let all = sorted(w.query_static::<Position>().map(|(e, _)| e).collect());
        let with = sorted(
            w.query_static_filtered::<Position, With<Player>>()
                .map(|(e, _)| e)
                .collect(),
        );
        let without = sorted(
            w.query_static_filtered::<Position, Without<Player>>()
                .map(|(e, _)| e)
                .collect(),
        );
        assert!(with.iter().all(|e| !without.contains(e)));
        assert_eq!(sorted([with, without].concat()), all);
    }

    #[test]
    fn or_filter_works_end_to_end_through_query_static_filtered() {
        // `Or<(...)>` is itself a `QueryFilter` -- no separate `_or`
        // query method was needed on `World` to wire it up.
        use crate::filter::Or;
        let (w, [plain, player, frozen, both]) = filter_world();
        let found = sorted(
            w.query_static_filtered::<Position, Or<(With<Player>, With<Frozen>)>>()
                .map(|(e, _)| e)
                .collect(),
        );
        assert_eq!(found, sorted(vec![player, frozen, both]));
        assert!(!found.contains(&plain));
    }

    #[test]
    fn with_on_a_never_registered_component_matches_nothing_without_matches_everything() {
        let (w, _) = filter_world();
        assert_eq!(
            w.query_static_filtered::<Position, With<NeverInserted>>()
                .count(),
            0
        );
        assert_eq!(
            w.query_static_filtered::<Position, Without<NeverInserted>>()
                .count(),
            4
        );
        // And asking must not have registered it: a later real insert
        // still works and is found.
        let mut w = w;
        let e = w.spawn_bundle((Position { x: 9.0, y: 0.0 },));
        assert!(w.insert_static(e, NeverInserted));
        assert_eq!(
            w.query_static_filtered::<Position, With<NeverInserted>>()
                .count(),
            1
        );
    }

    #[test]
    fn filter_on_a_sparse_shell_only_type_matches_nothing() {
        let (mut w, [plain, ..]) = filter_world();
        w.insert(plain, SparseOnly);
        // The Sparse Shell has it...
        assert_eq!(w.query::<SparseOnly>().count(), 1);
        // ...but it's not an archetype-tracked component, so an
        // Archetype Core filter has nothing to match against.
        assert_eq!(
            w.query_static_filtered::<Position, With<SparseOnly>>()
                .count(),
            0
        );
        assert_eq!(
            w.query_static_filtered::<Position, Without<SparseOnly>>()
                .count(),
            4
        );
    }

    #[test]
    fn contradictory_filter_yields_nothing() {
        let (w, _) = filter_world();
        assert_eq!(
            w.query_static_filtered::<Position, (With<Player>, Without<Player>)>()
                .count(),
            0
        );
    }

    #[test]
    fn filtering_on_a_component_the_query_already_fetches() {
        let (w, _) = filter_world();
        assert_eq!(
            w.query_static_filtered::<Position, With<Position>>()
                .count(),
            4
        );
        assert_eq!(
            w.query_static_filtered::<Position, Without<Position>>()
                .count(),
            0
        );
    }

    #[test]
    fn filtered_results_follow_structural_changes() {
        let (mut w, [plain, player, _frozen, both]) = filter_world();

        // Losing the marker moves the entity out of `With<Player>`...
        assert!(w.remove_static::<Player>(player).is_some());
        let found = sorted(
            w.query_static_filtered::<Position, With<Player>>()
                .map(|(e, _)| e)
                .collect(),
        );
        assert_eq!(found, vec![both]);

        // ...gaining it moves an entity in...
        assert!(w.insert_static(plain, Player));
        let found = sorted(
            w.query_static_filtered::<Position, With<Player>>()
                .map(|(e, _)| e)
                .collect(),
        );
        assert_eq!(found, sorted(vec![plain, both]));

        // ...and despawning removes it.
        assert!(w.despawn(both));
        let found = sorted(
            w.query_static_filtered::<Position, With<Player>>()
                .map(|(e, _)| e)
                .collect(),
        );
        assert_eq!(found, vec![plain]);
    }

    #[test]
    fn filters_tolerate_empty_intermediate_archetypes_left_by_insert_bundle() {
        // `insert_bundle` walks `edge_for_insert` once per element and
        // can leave zero-row intermediate archetypes behind (see
        // `Archetypes::iter`'s doc comment). A `Without` filter matches
        // those -- they lack the marker -- and must contribute nothing
        // and not panic.
        let mut w = World::new();
        let e = w.spawn();
        assert!(w.insert_bundle(
            e,
            (
                Position { x: 1.0, y: 0.0 },
                Velocity { dx: 1.0, dy: 0.0 },
                Player
            )
        ));

        assert_eq!(
            w.query_static_filtered::<Position, Without<Player>>()
                .count(),
            0
        );
        assert_eq!(
            w.query2_static_filtered::<Position, Velocity, Without<Player>>()
                .count(),
            0
        );
        let found: Vec<Entity> = w
            .query2_static_filtered::<Position, Velocity, With<Player>>()
            .map(|(e, _, _)| e)
            .collect();
        assert_eq!(found, vec![e]);
    }

    #[test]
    fn filtered_query_on_an_unregistered_query_component_is_empty() {
        let (w, _) = filter_world();
        assert_eq!(
            w.query_static_filtered::<NeverInserted, With<Player>>()
                .count(),
            0
        );
        assert_eq!(
            w.query2_static_filtered::<Position, NeverInserted, ()>()
                .count(),
            0
        );
    }

    // ── Bulk mutable queries ────────────────────────────────────────

    /// `n` entities with `Position { x: i }` and a fixed `Velocity`, in
    /// one archetype, and a tracker that has already seen them: writes made
    /// after this are the only things `query_changed` should report.
    fn mut_world(n: u32) -> (World, Vec<Entity>, ChangeTracker) {
        let mut w = World::new();
        let es: Vec<Entity> = (0..n)
            .map(|i| {
                w.spawn_bundle((
                    Position {
                        x: i as f32,
                        y: 0.0,
                    },
                    Velocity { dx: 1.0, dy: 2.0 },
                ))
            })
            .collect();
        w.increment_change_tick();
        let mut t = ChangeTracker::new();
        t.update(&w);
        w.increment_change_tick();
        (w, es, t)
    }

    fn changed_entities(w: &World, t: &ChangeTracker) -> Vec<Entity> {
        sorted(w.query_changed::<Position>(t).map(|(e, _)| e).collect())
    }

    #[test]
    fn query_static_mut_updates_values_in_place() {
        let (mut w, _, _) = mut_world(5);
        for (_, mut p) in w.query_static_mut::<Position>() {
            p.x += 10.0;
        }
        let xs: Vec<f32> = w.query_static::<Position>().map(|(_, p)| p.x).collect();
        assert_eq!(xs, vec![10.0, 11.0, 12.0, 13.0, 14.0]);
    }

    #[test]
    fn reading_through_mut_does_not_mark_changed() {
        let (mut w, _, t) = mut_world(5);
        let mut sum = 0.0;
        for (_, p) in w.query_static_mut::<Position>() {
            sum += p.x; // Deref only
        }
        assert_eq!(sum, 10.0);
        w.increment_change_tick();
        assert!(
            changed_entities(&w, &t).is_empty(),
            "a visit is not a write"
        );
    }

    #[test]
    fn writing_through_mut_marks_exactly_the_written_rows() {
        let (mut w, es, t) = mut_world(6);
        let written = [es[1], es[4]];
        for (e, mut p) in w.query_static_mut::<Position>() {
            if written.contains(&e) {
                p.y = 99.0;
            }
        }
        w.increment_change_tick();
        assert_eq!(
            changed_entities(&w, &t),
            sorted(written.to_vec()),
            "exactly the written rows, in the right rows (a tick column zipped \
             off by one would mark the wrong ones)"
        );
        assert_eq!(
            w.query_added::<Position>(&t).count(),
            0,
            "a mutation is not an addition"
        );
    }

    #[test]
    fn bulk_writes_stay_attached_to_their_entity_through_a_swap_remove() {
        // Despawning row 0 moves the last row into it. Write only to the
        // entity that will land in the freed slot, so a misaligned tick
        // column reports the wrong entity (or none).
        let (mut w, es, t) = mut_world(6);
        for (e, mut p) in w.query_static_mut::<Position>() {
            if e == es[5] || e == es[2] {
                p.x = -1.0;
            }
        }
        assert!(w.despawn(es[0]));
        w.increment_change_tick();
        assert_eq!(changed_entities(&w, &t), sorted(vec![es[2], es[5]]));
    }

    #[test]
    fn mut_api_reports_per_row_state_against_a_tracker() {
        let (mut w, _, t) = mut_world(3);
        let fresh = ChangeTracker::new();
        for (_, mut p) in w.query_static_mut::<Position>() {
            assert!(
                p.is_added(&fresh),
                "a fresh tracker sees everything as added"
            );
            assert!(!p.is_added(&t), "this tracker has already seen it");
            assert!(!p.is_changed(&t));
            p.bypass_change_detection().x = 7.0;
            assert!(!p.is_changed(&t), "a bypassed write is invisible");
            p.set_changed();
            assert!(p.is_changed(&t), "set_changed marks without a write");
        }
    }

    #[test]
    fn bypass_change_detection_writes_without_marking() {
        let (mut w, _, t) = mut_world(4);
        for (_, mut p) in w.query_static_mut::<Position>() {
            p.bypass_change_detection().x = 123.0;
        }
        w.increment_change_tick();
        assert!(changed_entities(&w, &t).is_empty());
        assert!(w.query_static::<Position>().all(|(_, p)| p.x == 123.0));
    }

    #[test]
    fn query_static_mut_filtered_matches_the_shared_filtered_query_exactly() {
        let (mut w, _) = filter_world();
        macro_rules! parity {
            ($f:ty) => {{
                let shared: Vec<Entity> = w
                    .query_static_filtered::<Position, $f>()
                    .map(|(e, _)| e)
                    .collect();
                let exclusive: Vec<Entity> = w
                    .query_static_mut_filtered::<Position, $f>()
                    .map(|(e, _)| e)
                    .collect();
                assert_eq!(shared, exclusive, "same entities, same order");
                assert_eq!(
                    shared,
                    w.query2_static_mut_filtered::<Position, Velocity, $f>()
                        .map(|(e, _, _)| e)
                        .collect::<Vec<_>>()
                );
                assert_eq!(
                    shared,
                    w.query2_static_mut_ref_filtered::<Position, Velocity, $f>()
                        .map(|(e, _, _)| e)
                        .collect::<Vec<_>>()
                );
                shared.len()
            }};
        }
        assert_eq!(parity!(()), 4);
        assert_eq!(parity!(With<Player>), 2);
        assert_eq!(parity!(Without<Player>), 2);
        assert_eq!(parity!((With<Player>, Without<Frozen>)), 1);
        assert_eq!(parity!(Or<(With<Player>, With<Frozen>)>), 3);
        assert_eq!(parity!(With<NeverInserted>), 0);
    }

    #[test]
    fn mut_queries_on_a_never_registered_component_are_empty() {
        let mut w = World::new();
        assert_eq!(w.query_static_mut::<Position>().count(), 0);
        let (mut w, _) = filter_world();
        assert_eq!(w.query_static_mut::<NeverInserted>().count(), 0);
        assert_eq!(w.query2_static_mut::<Position, NeverInserted>().count(), 0);
        assert_eq!(
            w.query2_static_mut_ref::<NeverInserted, Position>().count(),
            0
        );
    }

    #[test]
    fn mixed_mutability_query_advances_position_and_marks_only_position() {
        let (mut w, es, t) = mut_world(4);
        for (_, mut p, v) in w.query2_static_mut_ref::<Position, Velocity>() {
            p.x += v.dx;
            p.y += v.dy;
        }
        w.increment_change_tick();
        let xs: Vec<(f32, f32)> = w
            .query_static::<Position>()
            .map(|(_, p)| (p.x, p.y))
            .collect();
        assert_eq!(xs, vec![(1.0, 2.0), (2.0, 2.0), (3.0, 2.0), (4.0, 2.0)]);
        assert_eq!(changed_entities(&w, &t), sorted(es));
        assert_eq!(
            w.query_changed::<Velocity>(&t).count(),
            0,
            "the read-only side is never marked"
        );
    }

    #[test]
    fn two_mutable_components_are_marked_independently() {
        let (mut w, es, t) = mut_world(4);
        for (e, mut p, mut v) in w.query2_static_mut::<Position, Velocity>() {
            if e == es[1] {
                p.x = 50.0; // only Position written on this row
            }
            if e == es[3] {
                v.dx = 9.0; // only Velocity written on this row
            }
        }
        w.increment_change_tick();
        assert_eq!(changed_entities(&w, &t), vec![es[1]]);
        let vel: Vec<Entity> = w.query_changed::<Velocity>(&t).map(|(e, _)| e).collect();
        assert_eq!(vel, vec![es[3]]);
    }

    #[test]
    #[should_panic(expected = "two distinct component types")]
    fn query2_static_mut_on_the_same_type_panics_at_creation() {
        let (mut w, _, _) = mut_world(1);
        let _ = w.query2_static_mut::<Position, Position>();
    }

    #[test]
    #[should_panic(expected = "two distinct component types")]
    fn query2_static_mut_ref_on_the_same_type_panics_at_creation() {
        let (mut w, _, _) = mut_world(1);
        let _ = w.query2_static_mut_ref::<Position, Position>();
    }

    #[test]
    fn mut_queries_visit_what_the_shared_queries_visit_in_the_same_order() {
        let (mut w, _) = filter_world();
        let shared: Vec<Entity> = w.query_static::<Position>().map(|(e, _)| e).collect();
        assert_eq!(shared.len(), 4);
        let one: Vec<Entity> = w.query_static_mut::<Position>().map(|(e, _)| e).collect();
        let two: Vec<Entity> = w
            .query2_static_mut::<Position, Velocity>()
            .map(|(e, _, _)| e)
            .collect();
        let mixed: Vec<Entity> = w
            .query2_static_mut_ref::<Position, Velocity>()
            .map(|(e, _, _)| e)
            .collect();
        assert_eq!(shared, one);
        assert_eq!(shared, two);
        assert_eq!(shared, mixed);
    }

    #[test]
    fn mut_queries_tolerate_zero_row_intermediate_archetypes() {
        // `insert_bundle` walks one archetype edge per element, leaving
        // `{Position}` and `{Position, Velocity}` behind with a signature
        // but no column and no rows.
        let mut w = World::new();
        let e = w.spawn();
        w.insert_bundle(
            e,
            (
                Position { x: 1.0, y: 0.0 },
                Velocity { dx: 1.0, dy: 0.0 },
                Player,
            ),
        );
        let mut t = ChangeTracker::new();
        w.increment_change_tick();
        t.update(&w);
        w.increment_change_tick();
        let seen: Vec<Entity> = w.query_static_mut::<Position>().map(|(e, _)| e).collect();
        assert_eq!(seen, vec![e]);
        for (_, mut p, v) in w.query2_static_mut_ref::<Position, Velocity>() {
            p.x += v.dx;
        }
        for (_, mut p, _) in w.query2_static_mut::<Position, Velocity>() {
            p.y += 1.0;
        }
        w.increment_change_tick();
        assert_eq!(changed_entities(&w, &t), vec![e]);
        assert_eq!(
            w.get_static::<Position>(e),
            Some(&Position { x: 2.0, y: 1.0 })
        );
    }

    #[test]
    fn for_each_and_for_loop_agree_across_archetypes() {
        // A `for` loop drives `next()`/`advance()`; `for_each` drives the
        // `fold` override, which walks archetypes on its own. Both must
        // write the same values and mark the same rows, over a world with
        // several archetypes (including a filtered subset).
        fn run(by_for_each: bool) -> [(Vec<f32>, Vec<Entity>); 3] {
            let (mut w, _) = filter_world();
            let mut t = ChangeTracker::new();
            w.increment_change_tick();
            t.update(&w);
            w.increment_change_tick();
            if by_for_each {
                w.query_static_mut::<Position>()
                    .for_each(|(_, mut p)| p.x += 100.0);
                w.query2_static_mut_ref_filtered::<Position, Velocity, With<Player>>()
                    .for_each(|(_, mut p, v)| p.y += v.dx + 1.0);
                w.query2_static_mut::<Position, Velocity>()
                    .for_each(|(_, _, mut v)| v.dy = -5.0);
            } else {
                for (_, mut p) in w.query_static_mut::<Position>() {
                    p.x += 100.0;
                }
                for (_, mut p, v) in
                    w.query2_static_mut_ref_filtered::<Position, Velocity, With<Player>>()
                {
                    p.y += v.dx + 1.0;
                }
                for (_, _, mut v) in w.query2_static_mut::<Position, Velocity>() {
                    v.dy = -5.0;
                }
            }
            w.increment_change_tick();
            let pos = w
                .query_static::<Position>()
                .map(|(_, p)| p.x + p.y)
                .collect();
            let vel = w.query_static::<Velocity>().map(|(_, v)| v.dy).collect();
            [
                (
                    pos,
                    sorted(w.query_changed::<Position>(&t).map(|(e, _)| e).collect()),
                ),
                (
                    vel,
                    sorted(w.query_changed::<Velocity>(&t).map(|(e, _)| e).collect()),
                ),
                (
                    Vec::new(),
                    sorted(w.query_added::<Position>(&t).map(|(e, _)| e).collect()),
                ),
            ]
        }
        let by_loop = run(false);
        let by_fold = run(true);
        assert_eq!(by_loop, by_fold);
        assert_eq!(by_loop[0].1.len(), 4, "every Position row was written");
        assert_eq!(by_loop[1].1.len(), 4, "every Velocity row was written");
        assert!(by_loop[2].1.is_empty(), "nothing was added");
    }

    #[test]
    fn mut_iterators_keep_going_after_an_archetype_with_no_rows() {
        // Walk past empty archetypes in the middle of the list in both
        // `next()` (through `advance`) and `fold`: a zero-row archetype
        // that matches must not end the iteration early.
        let mut w = World::new();
        let a = w.spawn_bundle((Position { x: 1.0, y: 0.0 }, Velocity { dx: 0.0, dy: 0.0 }));
        let b = w.spawn();
        w.insert_bundle(
            b,
            (
                Position { x: 2.0, y: 0.0 },
                Velocity { dx: 0.0, dy: 0.0 },
                Player,
            ),
        );
        let c = w.spawn_bundle((
            Position { x: 3.0, y: 0.0 },
            Velocity { dx: 0.0, dy: 0.0 },
            Frozen,
        ));
        let want = vec![a, b, c];
        let by_next: Vec<Entity> = {
            let mut seen = Vec::new();
            for (e, _) in w.query_static_mut::<Position>() {
                seen.push(e);
            }
            seen
        };
        let mut by_fold = Vec::new();
        w.query_static_mut::<Position>()
            .for_each(|(e, _)| by_fold.push(e));
        assert_eq!(sorted(by_next), sorted(want.clone()));
        assert_eq!(sorted(by_fold), sorted(want));
    }
}
