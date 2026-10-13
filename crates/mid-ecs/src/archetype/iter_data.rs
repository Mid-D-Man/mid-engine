// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-ecs.md, section "Multi-term queries: `query_data.rs`"
// ============================================================================
//! The Archetype Core's multi-term iteration: the engine behind
//! `World::query_data` and `World::query_data_filtered`.
//!
//! A separate file from `iter.rs` on purpose, for the same reason
//! `iter_mut.rs` is: `Iter1`/`Iter2` and their `Ref` forms have `next()`
//! bodies held byte-for-byte fixed for the LTO inline budget (see
//! `iter.rs`'s header), so the new, general path is layered next to them,
//! never onto them.
//!
//! One generic iterator, [`IterData`], serves every query shape. It holds
//! the query's resolved state, the archetypes it matches, and, for the
//! archetype it is currently inside, the term-resolved column slices
//! (`Q::Fetch`) plus the row count they can all serve. `next()` indexes
//! those slices; it never looks anything up per row.
//!
//! Archetype selection reuses `matched_filtered`, so the visited
//! archetypes, and their order, are exactly what `iter2_ref_filtered`
//! visits for the same required components and filter (asserted by a
//! test). Zero `unsafe`: the slices come from `TableView::column`, which
//! is a safe `downcast_ref`.
//!
//! Child module of `archetype`, not a sibling: `Archetype`, `Table` and
//! the `Column` trait are private there, and a child sees them without
//! anything being widened.

use super::{Archetype, ArchetypeId, Archetypes};
use crate::component::ComponentId;
use crate::filter::QueryFilter;
use crate::query_data::QueryData;
use crate::world::Entity;

/// A read-only view of one archetype, handed to `QueryData::init_fetch`.
/// Opaque: the only way to get one is through the iterator in this
/// module. Public only because it appears in a sealed trait's signature;
/// it is not nameable from outside the crate.
pub struct TableView<'w> {
    archetype: &'w Archetype,
}

impl<'w> TableView<'w> {
    #[inline]
    pub(super) fn new(archetype: &'w Archetype) -> Self {
        Self { archetype }
    }

    /// Rows in this archetype's table.
    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.archetype.table.entities.len()
    }

    /// This archetype's entities, in row order.
    #[inline]
    pub(crate) fn entities(&self) -> &'w [Entity] {
        &self.archetype.table.entities
    }

    /// Whether this archetype's signature contains `id`.
    #[inline]
    pub(crate) fn contains(&self, id: ComponentId) -> bool {
        self.archetype.component_ids.contains(&id)
    }

    /// The column of `T` registered under `id`, if this table has created
    /// it. A signature can list a component before its column exists
    /// (columns are made lazily, on the first row), so `None` does not
    /// mean the archetype lacks `T`: use [`Self::contains`] for that.
    #[inline]
    pub(crate) fn column<T: 'static>(&self, id: ComponentId) -> Option<&'w [T]> {
        self.archetype.table.columns.get(id).map(|column| {
            column
                .as_any()
                .downcast_ref::<Vec<T>>()
                .expect("column type must match component_id's T")
                .as_slice()
        })
    }
}

/// The real `Iterator` behind `World::query_data`: walks the matched
/// archetypes in order and, inside each, every row up to the length all of
/// the query's column slices can serve.
pub(crate) struct IterData<'w, Q: QueryData> {
    archetypes: &'w Archetypes,
    state: Q::State,
    matched: std::vec::IntoIter<ArchetypeId>,
    /// The current archetype's resolved slices. `None` only before the
    /// first archetype is entered, when `len` is 0 and it is never read.
    fetch: Option<Q::Fetch<'w>>,
    row: usize,
    len: usize,
}

impl<'w, Q: QueryData> Iterator for IterData<'w, Q> {
    type Item = Q::Item<'w>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.row < self.len {
                if let Some(fetch) = self.fetch.as_ref() {
                    let item = Q::fetch(fetch, self.row);
                    self.row += 1;
                    return Some(item);
                }
            }
            let archetype_id = self.matched.next()?;
            let archetype = self.archetypes.archetypes.get(archetype_id).expect(
                "query_data's precomputed matched list only ever contains real, currently-existing archetype ids",
            );
            let view = TableView::new(archetype);
            let fetch = Q::init_fetch(&self.state, &view);
            self.len = Q::clamp_len(&fetch, view.len());
            self.fetch = Some(fetch);
            self.row = 0;
        }
    }
}

impl Archetypes {
    /// Builds the iterator for query data `Q` over the archetypes that
    /// contain every `&T` term's component *and* satisfy filter `F`. One
    /// pass over the archetypes and one small `Vec`; nothing here runs per
    /// row. `F = ()` is the unfiltered query.
    pub(crate) fn iter_data_filtered<Q: QueryData, F: QueryFilter>(&self) -> IterData<'_, Q> {
        let state = Q::get_state(&|type_id| self.component_ids.get(&type_id).copied());
        let matched = if Q::is_satisfiable(&state) {
            let mut required = Vec::new();
            Q::collect_required(&state, &mut required);
            self.matched_filtered::<F>(&required)
        } else {
            Vec::new()
        };
        IterData {
            archetypes: self,
            state,
            matched: matched.into_iter(),
            fetch: None,
            row: 0,
            len: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query_data::Has;
    use crate::world::World;

    struct A(u32);
    struct B(u32);
    struct C(u32);
    /// Never inserted anywhere.
    struct Never;

    /// How many archetypes the iterator for `Q` will visit. The items a
    /// query yields cannot show this (an archetype missing a `&T` column
    /// simply yields no rows), so the pruning is checked on the matched
    /// list itself.
    fn visited<Q: QueryData>(world: &World) -> usize {
        world.archetypes.iter_data_filtered::<Q, ()>().matched.len()
    }

    /// Four archetypes: `{}` (left empty once its entities migrate
    /// out), `{A}`, `{A, B}` and `{A, B, C}`.
    fn world_with_four_archetypes() -> World {
        let mut world = World::new();
        let e1 = world.spawn();
        world.insert_static(e1, A(1));
        let e2 = world.spawn();
        world.insert_static(e2, A(2));
        world.insert_static(e2, B(20));
        let e3 = world.spawn();
        world.insert_static(e3, A(3));
        world.insert_static(e3, B(30));
        world.insert_static(e3, C(300));
        world
    }

    #[test]
    fn only_ref_terms_narrow_which_archetypes_are_visited() {
        let world = world_with_four_archetypes();
        assert_eq!(visited::<Entity>(&world), 4);
        assert_eq!(visited::<&A>(&world), 3);
        assert_eq!(visited::<(&A, &B)>(&world), 2);
        assert_eq!(visited::<(&A, &B, &C)>(&world), 1);
        // Option and Has terms never narrow.
        assert_eq!(visited::<(&A, Option<&B>, Has<C>)>(&world), 3);
        assert_eq!(visited::<(Entity, Option<&C>, Has<B>)>(&world), 4);
    }

    #[test]
    fn a_ref_term_for_an_unseen_type_visits_no_archetype_at_all() {
        let world = world_with_four_archetypes();
        assert_eq!(visited::<&Never>(&world), 0);
        assert_eq!(visited::<(&A, &Never)>(&world), 0);
        // Option and Has for an unseen type do not make a query
        // unsatisfiable.
        assert_eq!(visited::<(Entity, Option<&Never>, Has<Never>)>(&world), 4);
    }
}
