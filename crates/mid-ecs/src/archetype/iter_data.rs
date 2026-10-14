// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-ecs.md, sections "Multi-term queries: `query_data.rs`" and
// "Mutable terms in `query_data`"
// ============================================================================
//! The Archetype Core's multi-term iteration: the engines behind
//! `World::query_data`, `World::query_data_filtered` and their `_mut`
//! forms.
//!
//! A separate file from `iter.rs` on purpose, for the same reason
//! `iter_mut.rs` is: `Iter1`/`Iter2` and their `Ref` forms have `next()`
//! bodies held byte-for-byte fixed for the LTO inline budget (see
//! `iter.rs`'s header), so the new, general path is layered next to them,
//! never onto them.
//!
//! Two generic iterators serve every query shape: [`IterData`] over a
//! shared `&Archetypes` (read-only terms) and [`IterDataMut`] over an
//! exclusive one (any terms). Each holds the query's resolved state, the
//! archetypes it matches, and, for the archetype it is currently inside,
//! the term-resolved column slices (`Q::Fetch`) plus the row count they
//! can all serve. `next()` steps those slices; it never looks anything up
//! per row.
//!
//! Archetype selection is one shared helper over `matched_filtered`, so
//! the visited archetypes, and their order, are exactly what
//! `iter2_ref_filtered` visits for the same required components and
//! filter (asserted by a test).
//!
//! **Zero `unsafe`, including the mutable one.** The exclusive iterator
//! takes each matched archetype as `&mut Archetype` (as `iter_mut.rs`
//! does), destructures its `Table` into independent field borrows, and
//! makes one pass over `columns.iter_mut()` and one over
//! `ticks.iter_mut()`. `SparseSet::iter_mut` yields every column exactly
//! once, so each column can be handed to exactly one place: kept shared
//! (and copied to every reader) if the query only reads it, or moved
//! whole into a slot a single `Mut` term takes it from if the query
//! writes it. Which is which comes from a per-query plan checked for
//! conflicts before anything is borrowed.
//!
//! Child module of `archetype`, not a sibling: `Archetype`, `Table` and
//! the `Column` trait are private there, and a child sees them without
//! anything being widened.

use super::iter_mut::select;
use super::{Archetype, ArchetypeId, Archetypes, Column, Table};
use crate::component::ComponentId;
use crate::filter::QueryFilter;
use crate::query_data::{AccessEntry, QueryData, ReadOnlyQueryData};
use crate::tick::{ComponentTicks, Tick};
use crate::world::Entity;

/// How the mutable iterator holds one column of the current archetype.
#[allow(clippy::borrowed_box)]
enum Slot<'w> {
    /// Read by one or more terms: copied out to each of them.
    Shared(&'w Box<dyn Column>),
    /// Written by exactly one `Mut` term, which takes it (once).
    Mut(Option<&'w mut Box<dyn Column>>),
}

/// One archetype, as the terms of a query see it, handed to
/// `QueryData::init_fetch`. Opaque: the only way to get one is through
/// the iterators in this module. Public only because it appears in a
/// sealed trait's signature; it is not nameable from outside the crate.
///
/// Two modes. *Shared* (read-only queries) looks columns up in the table
/// on demand. *Exclusive* (`_mut` queries) holds the columns the iterator
/// pre-assigned from its plan; asking for a column the plan did not give
/// it, or one already taken, yields `None`.
pub struct TableAccess<'w> {
    signature: &'w [ComponentId],
    entities: &'w [Entity],
    this_run: Tick,
    /// `Some` in shared mode only.
    table: Option<&'w Table>,
    /// Exclusive mode only.
    columns: Vec<(ComponentId, Slot<'w>)>,
    /// Exclusive mode only: the tick slices of the written components.
    ticks: Vec<(ComponentId, Option<&'w mut [ComponentTicks]>)>,
}

impl<'w> TableAccess<'w> {
    fn shared(archetype: &'w Archetype) -> Self {
        Self {
            signature: &archetype.component_ids,
            entities: &archetype.table.entities,
            this_run: Tick::default(),
            table: Some(&archetype.table),
            columns: Vec::new(),
            ticks: Vec::new(),
        }
    }

    fn exclusive(this_run: Tick) -> Self {
        Self {
            signature: &[],
            entities: &[],
            this_run,
            table: None,
            columns: Vec::new(),
            ticks: Vec::new(),
        }
    }

    /// Points this access at `archetype` and hands its columns out
    /// according to `plan` (`(component, written)` pairs, one per
    /// component the query touches).
    fn load(&mut self, archetype: &'w mut Archetype, plan: &[(ComponentId, bool)]) {
        let Archetype {
            component_ids,
            table,
            ..
        } = archetype;
        let Table {
            columns,
            ticks,
            entities,
        } = table;
        self.signature = component_ids.as_slice();
        self.entities = entities.as_slice();
        self.columns.clear();
        self.ticks.clear();
        for (id, column) in columns.iter_mut() {
            if let Some(&(_, written)) = plan.iter().find(|(planned, _)| *planned == id) {
                let slot = if written {
                    Slot::Mut(Some(column))
                } else {
                    Slot::Shared(&*column)
                };
                self.columns.push((id, slot));
            }
        }
        for (id, column_ticks) in ticks.iter_mut() {
            if plan
                .iter()
                .any(|&(planned, written)| planned == id && written)
            {
                self.ticks.push((id, Some(column_ticks.as_mut_slice())));
            }
        }
    }

    /// Rows in this archetype's table.
    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.entities.len()
    }

    /// This archetype's entities, in row order.
    #[inline]
    pub(crate) fn entities(&self) -> &'w [Entity] {
        self.entities
    }

    /// Whether this archetype's signature contains `id`.
    #[inline]
    pub(crate) fn contains(&self, id: ComponentId) -> bool {
        self.signature.contains(&id)
    }

    /// The tick a write through a `Mut` from this access is stamped with.
    #[inline]
    pub(crate) fn this_run(&self) -> Tick {
        self.this_run
    }

    /// The column of `T` registered under `id`, for reading, if this
    /// table has created it. A signature can list a component before its
    /// column exists (columns are made lazily, on the first row), so
    /// `None` does not mean the archetype lacks `T`: use
    /// [`Self::contains`] for that. In exclusive mode only a column the
    /// plan marked as read-only is available this way.
    #[inline]
    pub(crate) fn column<T: 'static>(&self, id: ComponentId) -> Option<&'w [T]> {
        let column: &'w dyn Column = match self.table {
            Some(table) => &**table.columns.get(id)?,
            None => match self.columns.iter().find(|(planned, _)| *planned == id) {
                Some((_, Slot::Shared(column))) => &***column,
                _ => return None,
            },
        };
        Some(
            column
                .as_any()
                .downcast_ref::<Vec<T>>()
                .expect("column type must match component_id's T")
                .as_slice(),
        )
    }

    /// The column of `T` registered under `id`, for writing: the one
    /// `Mut` term the plan assigned it to takes it, once. `None` if the
    /// table has no such column yet, or it was already taken, or this is
    /// a shared access.
    #[inline]
    pub(crate) fn column_mut<T: 'static>(&mut self, id: ComponentId) -> Option<&'w mut [T]> {
        let slot = self
            .columns
            .iter_mut()
            .find(|(planned, _)| *planned == id)?;
        match &mut slot.1 {
            Slot::Mut(column) => {
                let column = column.take()?;
                Some(
                    column
                        .as_any_mut()
                        .downcast_mut::<Vec<T>>()
                        .expect("column type must match component_id's T")
                        .as_mut_slice(),
                )
            }
            Slot::Shared(_) => None,
        }
    }

    /// The change ticks of the column under `id`, for writing, under the
    /// same rules as [`Self::column_mut`].
    #[inline]
    pub(crate) fn ticks_mut(&mut self, id: ComponentId) -> Option<&'w mut [ComponentTicks]> {
        self.ticks
            .iter_mut()
            .find(|(planned, _)| *planned == id)?
            .1
            .take()
    }
}

/// The matching archetypes for `Q` under filter `F`, plus `Q`'s resolved
/// state. One pass over the archetypes and one small `Vec`; nothing here
/// runs per row. Shared by both iterators, so pruning is decided in one
/// place.
fn plan_matches<Q: QueryData, F: QueryFilter>(
    archetypes: &Archetypes,
) -> (Q::State, Vec<ArchetypeId>) {
    let state = Q::get_state(&|type_id| archetypes.component_ids.get(&type_id).copied());
    let matched = if Q::is_satisfiable(&state) {
        let mut required = Vec::new();
        Q::collect_required(&state, &mut required);
        archetypes.matched_filtered::<F>(&required)
    } else {
        Vec::new()
    };
    (state, matched)
}

/// Panics if the query reads and writes, or writes twice, the same
/// component type. Keyed by `TypeId`, not by registered id, so the answer
/// does not depend on which types the world happens to hold.
fn validate_access(entries: &[AccessEntry]) {
    for (i, a) in entries.iter().enumerate() {
        for b in &entries[i + 1..] {
            if a.type_id == b.type_id && (a.write || b.write) {
                panic!(
                    "query_data_mut: conflicting access to `{}`: a component can be read by any number of terms, or written by exactly one `Mut` term and nothing else",
                    a.name
                );
            }
        }
    }
}

/// `(component, written)` for every component the query touches, one
/// entry per component (a write wins over a read). Types the world has
/// never seen have no id and are skipped: they match nothing anyway.
fn build_plan(entries: &[AccessEntry]) -> Vec<(ComponentId, bool)> {
    let mut plan: Vec<(ComponentId, bool)> = Vec::new();
    for entry in entries {
        if let Some(id) = entry.id {
            match plan.iter_mut().find(|(planned, _)| *planned == id) {
                Some((_, written)) => *written |= entry.write,
                None => plan.push((id, entry.write)),
            }
        }
    }
    plan
}

// ── Shared iteration ─────────────────────────────────────────────────

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
                if let Some(fetch) = self.fetch.as_mut() {
                    let item = Q::fetch(fetch, self.row);
                    self.row += 1;
                    return Some(item);
                }
            }
            let archetype_id = self.matched.next()?;
            let archetype = self.archetypes.archetypes.get(archetype_id).expect(
                "query_data's precomputed matched list only ever contains real, currently-existing archetype ids",
            );
            let mut table = TableAccess::shared(archetype);
            let fetch = Q::init_fetch(&self.state, &mut table);
            self.len = Q::clamp_len(&fetch, table.len());
            self.fetch = Some(fetch);
            self.row = 0;
        }
    }
}

impl Archetypes {
    /// Builds the read-only iterator for query data `Q` over the
    /// archetypes that contain every `&T` term's component *and* satisfy
    /// filter `F`. `F = ()` is the unfiltered query.
    pub(crate) fn iter_data_filtered<Q: ReadOnlyQueryData, F: QueryFilter>(
        &self,
    ) -> IterData<'_, Q> {
        let (state, matched) = plan_matches::<Q, F>(self);
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

// ── Exclusive iteration ──────────────────────────────────────────────

/// The real `Iterator` behind `World::query_data_mut`. `I` is the
/// matching-archetype walk from `select`; it is a type parameter only
/// because that walk's type cannot be named.
pub(crate) struct IterDataMut<'w, Q: QueryData, I> {
    archetypes: I,
    state: Q::State,
    /// `(component, written)` for every component the query touches.
    plan: Vec<(ComponentId, bool)>,
    /// Reused across archetypes, so entering one allocates nothing once
    /// the first has sized the slot lists.
    table: TableAccess<'w>,
    fetch: Option<Q::Fetch<'w>>,
    row: usize,
    len: usize,
}

impl<'w, Q: QueryData, I: Iterator<Item = &'w mut Archetype>> Iterator for IterDataMut<'w, Q, I> {
    type Item = Q::Item<'w>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.row < self.len {
                if let Some(fetch) = self.fetch.as_mut() {
                    let item = Q::fetch(fetch, self.row);
                    self.row += 1;
                    return Some(item);
                }
            }
            let archetype = self.archetypes.next()?;
            self.table.load(archetype, &self.plan);
            let fetch = Q::init_fetch(&self.state, &mut self.table);
            self.len = Q::clamp_len(&fetch, self.table.len());
            self.fetch = Some(fetch);
            self.row = 0;
        }
    }
}

impl Archetypes {
    /// Builds the exclusive iterator for query data `Q` (any terms) over
    /// the archetypes that contain every `&T`/`Mut<T>` term's component
    /// and satisfy `F`. Writes made through `Mut` terms are stamped with
    /// `this_run`.
    ///
    /// # Panics
    /// If `Q` reads and writes, or writes twice, the same component type
    /// (see `validate_access`). Checked before anything is borrowed and
    /// regardless of which types the world holds.
    pub(crate) fn iter_data_mut_filtered<'a, Q: QueryData + 'a, F: QueryFilter>(
        &'a mut self,
        this_run: Tick,
    ) -> impl Iterator<Item = Q::Item<'a>> + 'a {
        let (state, matched) = plan_matches::<Q, F>(self);
        let mut entries = Vec::new();
        Q::collect_access(&state, &mut entries);
        validate_access(&entries);
        IterDataMut::<Q, _> {
            archetypes: select(&mut self.archetypes, matched),
            state,
            plan: build_plan(&entries),
            table: TableAccess::exclusive(this_run),
            fetch: None,
            row: 0,
            len: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query_data::{Has, ReadOnlyQueryData};
    use crate::tick::Mut;
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
    fn visited<Q: ReadOnlyQueryData>(world: &World) -> usize {
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

    /// Like `visited`, for the exclusive path's terms: the matched list
    /// both iterators share, built without borrowing anything mutably.
    fn matched_for<Q: QueryData>(world: &World) -> usize {
        plan_matches::<Q, ()>(&world.archetypes).1.len()
    }

    #[test]
    fn mut_terms_narrow_like_ref_terms_and_option_mut_never_does() {
        let world = world_with_four_archetypes();
        assert_eq!(matched_for::<Mut<A>>(&world), 3);
        assert_eq!(matched_for::<(Mut<A>, &B)>(&world), 2);
        assert_eq!(matched_for::<(Mut<A>, Mut<B>, Mut<C>)>(&world), 1);
        assert_eq!(matched_for::<(&A, Option<Mut<B>>, Has<C>)>(&world), 3);
        assert_eq!(matched_for::<(Entity, Option<Mut<C>>)>(&world), 4);
    }

    #[test]
    fn a_mut_term_for_an_unseen_type_visits_no_archetype_but_option_mut_does_not_block() {
        let world = world_with_four_archetypes();
        assert_eq!(matched_for::<Mut<Never>>(&world), 0);
        assert_eq!(matched_for::<(Mut<A>, &Never)>(&world), 0);
        assert_eq!(matched_for::<(Entity, Option<Mut<Never>>)>(&world), 4);
    }
}
