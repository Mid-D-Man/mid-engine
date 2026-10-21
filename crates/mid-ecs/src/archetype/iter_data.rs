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
//! **Zero `unsafe`, including the mutable one** (for the exclusive
//! source; the `WorldCell` source below is the one exception, behind
//! `unsafe fn iter_data_cell`, see `world_cell.rs`). The exclusive iterator
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

use std::any::TypeId;
use std::ops::Range;

use super::iter_mut::select;
use super::{Archetype, ArchetypeId, Archetypes, Column, RawColumn, Table};
use crate::batching::{plan_batches, BatchingStrategy};
use crate::component::ComponentId;
use crate::data_filter::DataFilter;
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
    /// `Mut`, in a *windowed* access (parallel iteration): not a `&mut`
    /// but the raw handle the dispatcher captured; the term's
    /// `column_mut` builds a `&mut [T]` over this window's rows only.
    Window(Option<RawColumn>),
}

/// How the mutable iterator holds the change ticks of one component of
/// the current archetype.
enum TickSlot<'w> {
    /// Read by `Ref`/`Added`/`Changed` terms or by a row filter.
    Shared(&'w [ComponentTicks]),
    /// Written by exactly one `Mut` term, which takes it (once). A row
    /// filter on the same component reads it first, before the take.
    Mut(Option<&'w mut [ComponentTicks]>),
    /// `Mut`, in a windowed access: the written ticks of this window's
    /// rows, already cut to the window.
    Window(Option<RawTicks>),
}

/// The change ticks of one window of rows of a written component: a raw
/// pointer and a length, for the same reason [`RawColumn`] is raw (see
/// there). Created only in `load_window`, from the pointer the
/// dispatcher captured.
#[derive(Clone, Copy)]
struct RawTicks {
    ptr: *mut ComponentTicks,
    len: usize,
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
    /// The tracker's tick `Added`/`Changed` compare against. Meaningless
    /// (and unread) for a query with no such term or filter.
    last_run: Tick,
    /// `Some` in shared mode only.
    table: Option<&'w Table>,
    /// Exclusive mode only.
    columns: Vec<(ComponentId, Slot<'w>)>,
    /// Exclusive mode only: the tick slices of the written components,
    /// and of the read ones a `Ref`/`Added`/`Changed` term or a row
    /// filter needs.
    ticks: Vec<(ComponentId, TickSlot<'w>)>,
    /// `Some((start, end))` while this access shows one *window* of an
    /// archetype (parallel iteration): `entities`, the tick slices and
    /// every column the terms get are cut to rows `start..end`, so the
    /// terms see rows `0..end - start` and need not know. `None`
    /// otherwise.
    window: Option<(usize, usize)>,
}

impl<'w> TableAccess<'w> {
    fn shared(archetype: &'w Archetype, this_run: Tick, last_run: Tick) -> Self {
        Self {
            signature: &archetype.component_ids,
            entities: &archetype.table.entities,
            this_run,
            last_run,
            table: Some(&archetype.table),
            columns: Vec::new(),
            ticks: Vec::new(),
            window: None,
        }
    }

    fn exclusive(this_run: Tick, last_run: Tick) -> Self {
        Self {
            signature: &[],
            entities: &[],
            this_run,
            last_run,
            table: None,
            columns: Vec::new(),
            ticks: Vec::new(),
            window: None,
        }
    }

    /// Points this access at `archetype` and hands its columns out
    /// according to `plan` (`(component, written, ticks_read)` triples,
    /// one per component the query touches).
    fn load(&mut self, archetype: &'w mut Archetype, plan: &[(ComponentId, bool, bool)]) {
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
        self.window = None;
        self.columns.clear();
        self.ticks.clear();
        for (id, column) in columns.iter_mut() {
            if let Some(&(_, written, _)) = plan.iter().find(|(planned, _, _)| *planned == id) {
                let slot = if written {
                    Slot::Mut(Some(column))
                } else {
                    Slot::Shared(&*column)
                };
                self.columns.push((id, slot));
            }
        }
        for (id, column_ticks) in ticks.iter_mut() {
            if let Some(&(_, written, ticked)) = plan.iter().find(|(planned, _, _)| *planned == id)
            {
                if written {
                    self.ticks
                        .push((id, TickSlot::Mut(Some(column_ticks.as_mut_slice()))));
                } else if ticked {
                    self.ticks
                        .push((id, TickSlot::Shared(column_ticks.as_slice())));
                }
            }
        }
    }

    /// `load`'s counterpart for a `WorldCell`: the same plan-driven
    /// hand-out of columns and tick slices, but from a *shared*
    /// `&Archetype`, taking the written ones through their `Interior`.
    ///
    /// # Safety
    /// For the whole of `'w`: no reference to a column or tick slice this
    /// plan marks *written* exists except the one this call hands out, and
    /// no column or tick slice it marks *read* is written by anyone. A
    /// column's data and its tick slice are separate `Interior`s, each
    /// handed out at most once per call (the plan has one entry per
    /// component).
    unsafe fn load_cell(&mut self, archetype: &'w Archetype, plan: &[(ComponentId, bool, bool)]) {
        self.signature = archetype.component_ids.as_slice();
        self.entities = archetype.table.entities.as_slice();
        self.window = None;
        self.columns.clear();
        self.ticks.clear();
        for (id, column) in archetype.table.columns.iter() {
            if let Some(&(_, written, _)) = plan.iter().find(|(planned, _, _)| *planned == id) {
                let slot = if written {
                    // SAFETY: the caller's contract; this is the only
                    // `&mut` to this column.
                    Slot::Mut(Some(unsafe { column.get_mut_unchecked() }))
                } else {
                    Slot::Shared(&**column)
                };
                self.columns.push((id, slot));
            }
        }
        for (id, column_ticks) in archetype.table.ticks.iter() {
            if let Some(&(_, written, ticked)) = plan.iter().find(|(planned, _, _)| *planned == id)
            {
                if written {
                    // SAFETY: as above, for the tick slice.
                    let ticks = unsafe { column_ticks.get_mut_unchecked() };
                    self.ticks
                        .push((id, TickSlot::Mut(Some(ticks.as_mut_slice()))));
                } else if ticked {
                    self.ticks
                        .push((id, TickSlot::Shared(column_ticks.as_slice())));
                }
            }
        }
    }

    /// `load_cell`'s counterpart for one *window* of an archetype, the
    /// unit of parallel iteration: points this access at rows
    /// `desc.rows` of `desc.archetype`, with `entities`, the tick slices
    /// and (through `column`/`column_mut`) the columns cut to that range.
    ///
    /// Safe to call because the only way to get a [`WindowDesc`] is
    /// `Archetypes::par_batches_cell`, an `unsafe fn` whose contract
    /// covers every raw handle in it; this only moves them into slots.
    /// Columns the plan marks written are not touched here at all (no
    /// reference to them is formed): their slot keeps the raw handle.
    fn load_window(&mut self, desc: &WindowDesc<'w>, plan: &[(ComponentId, bool, bool)]) {
        let archetype: &'w Archetype = desc.archetype;
        let (start, end) = (desc.rows.start, desc.rows.end);
        self.window = Some((start, end));
        self.signature = archetype.component_ids.as_slice();
        let entities = archetype.table.entities.as_slice();
        let entity_end = end.min(entities.len());
        self.entities = &entities[start.min(entity_end)..entity_end];
        self.columns.clear();
        self.ticks.clear();
        for (id, column) in archetype.table.columns.iter() {
            if let Some(&(_, written, _)) = plan.iter().find(|(planned, _, _)| *planned == id) {
                let slot = if written {
                    match desc.written.iter().find(|(raw_id, _)| *raw_id == id) {
                        Some(&(_, raw)) => Slot::Window(Some(raw)),
                        None => continue,
                    }
                } else {
                    Slot::Shared(&**column)
                };
                self.columns.push((id, slot));
            }
        }
        for (id, column_ticks) in archetype.table.ticks.iter() {
            if let Some(&(_, written, ticked)) = plan.iter().find(|(planned, _, _)| *planned == id)
            {
                if written {
                    if let Some(&(_, raw)) =
                        desc.written_ticks.iter().find(|(raw_id, _)| *raw_id == id)
                    {
                        let end = end.min(raw.len);
                        let start = start.min(end);
                        // SAFETY: `start <= end <= raw.len`, so the
                        // offset is within (or one past) the buffer.
                        let ptr = unsafe { raw.ptr.add(start) };
                        self.ticks.push((
                            id,
                            TickSlot::Window(Some(RawTicks {
                                ptr,
                                len: end - start,
                            })),
                        ));
                    }
                } else if ticked {
                    let all = column_ticks.as_slice();
                    let end = end.min(all.len());
                    self.ticks
                        .push((id, TickSlot::Shared(&all[start.min(end)..end])));
                }
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

    /// The world's tick as of query creation: what a write through a
    /// `Mut` from this access is stamped with, and what a `Ref` or an
    /// `Added`/`Changed` check measures a row's ticks against.
    #[inline]
    pub(crate) fn this_run(&self) -> Tick {
        self.this_run
    }

    /// The tracker's last-checked tick (see the field's note).
    #[inline]
    pub(crate) fn last_run(&self) -> Tick {
        self.last_run
    }

    /// The change ticks of the column under `id`, for reading, with the
    /// full `'w` lifetime so a term can keep them. `None` if the table has
    /// no such column yet. In exclusive mode only a component the plan
    /// marked as ticks-read is available this way (never a written one:
    /// `Ref`/`Added`/`Changed` conflict with `Mut` on the same component).
    #[inline]
    pub(crate) fn ticks(&self, id: ComponentId) -> Option<&'w [ComponentTicks]> {
        match self.table {
            Some(table) => table.ticks.get(id).map(|ticks| ticks.as_slice()),
            None => match &self.ticks.iter().find(|(planned, _)| *planned == id)?.1 {
                TickSlot::Shared(ticks) => Some(*ticks),
                TickSlot::Mut(_) | TickSlot::Window(_) => None,
            },
        }
    }

    /// Like [`Self::ticks`] but borrowed from `self`, and also able to
    /// look at a slice a `Mut` term will later take. That is what lets a
    /// row filter (`Changed<T>`) run on a component the same query
    /// writes: the filter reads, finishes, and only then does
    /// `init_fetch` move the slice into the `Mut`. `None` once taken.
    #[inline]
    pub(crate) fn peek_ticks(&self, id: ComponentId) -> Option<&[ComponentTicks]> {
        match self.table {
            Some(table) => table.ticks.get(id).map(|ticks| ticks.as_slice()),
            None => match &self.ticks.iter().find(|(planned, _)| *planned == id)?.1 {
                TickSlot::Shared(ticks) => Some(*ticks),
                TickSlot::Mut(Some(ticks)) => Some(&**ticks),
                TickSlot::Mut(None) => None,
                TickSlot::Window(Some(raw)) => {
                    // SAFETY: `raw` is this window's own, valid, row
                    // range of a written tick column (see `RawTicks`),
                    // untaken (`Some`), so no `&mut` over it exists; the
                    // returned borrow is of `self` and ends before
                    // `ticks_mut` can take it.
                    Some(unsafe { std::slice::from_raw_parts(raw.ptr, raw.len) })
                }
                TickSlot::Window(None) => None,
            },
        }
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
            Some(table) => &***table.columns.get(id)?,
            None => match self.columns.iter().find(|(planned, _)| *planned == id) {
                Some((_, Slot::Shared(column))) => &***column,
                _ => return None,
            },
        };
        let slice = column
            .as_any()
            .downcast_ref::<Vec<T>>()
            .expect("column type must match component_id's T")
            .as_slice();
        Some(match self.window {
            None => slice,
            Some((start, end)) => {
                let end = end.min(slice.len());
                &slice[start.min(end)..end]
            }
        })
    }

    /// The column of `T` registered under `id`, for writing: the one
    /// `Mut` term the plan assigned it to takes it, once. `None` if the
    /// table has no such column yet, or it was already taken, or this is
    /// a shared access.
    #[inline]
    pub(crate) fn column_mut<T: 'static>(&mut self, id: ComponentId) -> Option<&'w mut [T]> {
        let window = self.window;
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
            Slot::Window(raw) => {
                let raw = raw.take()?;
                assert!(
                    raw.type_id == TypeId::of::<Vec<T>>(),
                    "column type must match component_id's T"
                );
                let (start, end) =
                    window.expect("a windowed column slot only exists in a windowed access");
                let end = end.min(raw.len);
                let start = start.min(end);
                // SAFETY: `raw` was captured from a `Vec<T>` (checked
                // just above) by `Archetypes::par_batches_cell`, whose
                // caller guarantees this query has exclusive access to
                // the column for `'w` and that the world's structure is
                // not changed meanwhile (so the buffer neither moves nor
                // frees). `start..end` is within `raw.len`, and the
                // windows of one query are pairwise disjoint (the exact
                // cover `plan_batches` guarantees), each handed out once
                // (`take` above), so this `&mut` aliases no other.
                Some(unsafe {
                    std::slice::from_raw_parts_mut(raw.ptr.cast::<T>().add(start), end - start)
                })
            }
            Slot::Shared(_) => None,
        }
    }

    /// The change ticks of the column under `id`, for writing, under the
    /// same rules as [`Self::column_mut`].
    #[inline]
    pub(crate) fn ticks_mut(&mut self, id: ComponentId) -> Option<&'w mut [ComponentTicks]> {
        match &mut self.ticks.iter_mut().find(|(planned, _)| *planned == id)?.1 {
            TickSlot::Mut(ticks) => ticks.take(),
            TickSlot::Window(raw) => {
                let raw = raw.take()?;
                // SAFETY: as for `column_mut`'s windowed branch: `raw` is
                // this window's own rows of a tick column the query has
                // exclusive access to, taken once, disjoint from every
                // other window's.
                Some(unsafe { std::slice::from_raw_parts_mut(raw.ptr, raw.len) })
            }
            TickSlot::Shared(_) => None,
        }
    }
}

/// The matching archetypes for `Q` under filter `F`, plus `Q`'s resolved
/// state. One pass over the archetypes and one small `Vec`; nothing here
/// runs per row. Shared by both iterators, so pruning is decided in one
/// place.
fn plan_matches<Q: QueryData, F: DataFilter>(
    archetypes: &Archetypes,
) -> (Q::State, F::State, Vec<ArchetypeId>) {
    let resolve = |type_id| archetypes.component_ids.get(&type_id).copied();
    let state = Q::get_state(&resolve);
    let filter = F::get_state(&resolve);
    let matched = if Q::is_satisfiable(&state) {
        let mut required = Vec::new();
        Q::collect_required(&state, &mut required);
        archetypes.matched_data_filtered::<F>(&filter, &required)
    } else {
        Vec::new()
    };
    (state, filter, matched)
}

/// Fills `rows` with the rows of the archetype `table` points at that
/// filter `F` keeps, ascending. Runs once per archetype, before any `Mut`
/// term takes a slice, so a filter may read the ticks of a component the
/// same query writes. Only called for a `F::TRACKED` filter: every other
/// filter is fully decided at archetype level.
fn select_rows<F: DataFilter>(filter: &F::State, table: &TableAccess<'_>, rows: &mut Vec<usize>) {
    rows.clear();
    let fetch = F::init_fetch(filter, table);
    for row in 0..table.len() {
        if F::matches_row(&fetch, row) {
            rows.push(row);
        }
    }
}

/// Panics if the query reads and writes, or writes twice, the same
/// component type. Keyed by `TypeId`, not by registered id, so the answer
/// does not depend on which types the world happens to hold.
pub(crate) fn validate_access(context: &str, entries: &[AccessEntry]) {
    for (i, a) in entries.iter().enumerate() {
        for b in &entries[i + 1..] {
            if a.type_id == b.type_id && (a.write || b.write) {
                panic!(
                    "{context}: conflicting access to `{}`: a component can be read by any number of terms, or written by exactly one `Mut` term and nothing else",
                    a.name
                );
            }
        }
    }
}

/// `(component, written, ticks_read)` for every component the query
/// touches, one entry per component (a write wins over a read). A
/// component is `ticks_read` if a term needs its tick slice for reading
/// (`entries`) or a row filter does (`filter_ticks`). Types the world has
/// never seen have no id and are skipped: they match nothing anyway.
fn build_plan(
    entries: &[AccessEntry],
    filter_ticks: &[ComponentId],
) -> Vec<(ComponentId, bool, bool)> {
    let mut plan: Vec<(ComponentId, bool, bool)> = Vec::new();
    for entry in entries {
        if let Some(id) = entry.id {
            match plan.iter_mut().find(|(planned, _, _)| *planned == id) {
                Some((_, written, ticked)) => {
                    *written |= entry.write;
                    *ticked |= entry.ticks;
                }
                None => plan.push((id, entry.write, entry.ticks)),
            }
        }
    }
    for &id in filter_ticks {
        match plan.iter_mut().find(|(planned, _, _)| *planned == id) {
            Some((_, _, ticked)) => *ticked = true,
            None => plan.push((id, false, true)),
        }
    }
    plan
}

// ── Shared iteration ─────────────────────────────────────────────────

/// The real `Iterator` behind `World::query_data`: walks the matched
/// archetypes in order and, inside each, every row up to the length all of
/// the query's column slices can serve. With a row-level filter
/// (`F::TRACKED`) it instead walks only the rows the filter selected for
/// the archetype, in ascending order.
pub(crate) struct IterData<'w, Q: QueryData, F: DataFilter> {
    archetypes: &'w Archetypes,
    state: Q::State,
    filter: F::State,
    this_run: Tick,
    last_run: Tick,
    matched: std::vec::IntoIter<ArchetypeId>,
    /// The current archetype's resolved slices. `None` only before the
    /// first archetype is entered, when `len` is 0 and it is never read.
    fetch: Option<Q::Fetch<'w>>,
    row: usize,
    len: usize,
    /// `F::TRACKED` only: the rows of the current archetype the filter
    /// kept (all below `len`), and how many have been handed out.
    rows: Vec<usize>,
    next_selected: usize,
}

impl<'w, Q: QueryData, F: DataFilter> Iterator for IterData<'w, Q, F> {
    type Item = Q::Item<'w>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if F::TRACKED {
                if let Some(fetch) = self.fetch.as_mut() {
                    if let Some(&row) = self.rows.get(self.next_selected) {
                        self.next_selected += 1;
                        return Some(Q::fetch(fetch, row));
                    }
                }
            } else if self.row < self.len {
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
            let mut table = TableAccess::shared(archetype, self.this_run, self.last_run);
            if F::TRACKED {
                select_rows::<F>(&self.filter, &table, &mut self.rows);
                self.next_selected = 0;
            }
            let fetch = Q::init_fetch(&self.state, &mut table);
            self.len = Q::clamp_len(&fetch, table.len());
            if F::TRACKED {
                let keep = self.rows.partition_point(|&row| row < self.len);
                self.rows.truncate(keep);
            }
            self.fetch = Some(fetch);
            self.row = 0;
        }
    }
}

impl Archetypes {
    /// The archetypes that contain every id in `required` *and* satisfy
    /// `F`'s archetype-level test, in dense order. For a row-level filter
    /// the test is the necessary condition (`Changed<T>` needs `T`
    /// present); the rows are chosen later, per archetype.
    fn matched_data_filtered<F: DataFilter>(
        &self,
        filter: &F::State,
        required: &[ComponentId],
    ) -> Vec<ArchetypeId> {
        self.archetypes
            .iter()
            .filter_map(|(id, archetype)| {
                let signature = &archetype.component_ids;
                let keep = required.iter().all(|c| signature.contains(c))
                    && F::matches_component_set(filter, &|c| signature.contains(&c));
                keep.then_some(id)
            })
            .collect()
    }

    /// Builds the read-only iterator for query data `Q` over the
    /// archetypes that contain every `&T` term's component *and* satisfy
    /// filter `F`. `F = ()` is the unfiltered query. `this_run` is the
    /// world's tick (what a `Ref` carries); `last_run` is the tracker's
    /// (read only by `Added`/`Changed`).
    pub(crate) fn iter_data_filtered<Q: ReadOnlyQueryData, F: DataFilter>(
        &self,
        this_run: Tick,
        last_run: Tick,
    ) -> IterData<'_, Q, F> {
        let (state, filter, matched) = plan_matches::<Q, F>(self);
        IterData {
            archetypes: self,
            state,
            filter,
            this_run,
            last_run,
            matched: matched.into_iter(),
            fetch: None,
            row: 0,
            len: 0,
            rows: Vec::new(),
            next_selected: 0,
        }
    }
}

// ── Exclusive iteration ──────────────────────────────────────────────

/// Where the exclusive iterator gets its next archetype from, and how it
/// turns it into a `TableAccess`: exclusively (`&mut Archetype`, no
/// `unsafe`) or through a `WorldCell` (`&Archetype` and per-column
/// interior mutability, behind the contract of `iter_data_cell`).
pub(crate) trait ArchetypeSource<'w> {
    /// Loads the next matched archetype into `table` according to `plan`.
    /// `false` when there are no more.
    fn load_next(
        &mut self,
        table: &mut TableAccess<'w>,
        plan: &[(ComponentId, bool, bool)],
    ) -> bool;
}

/// The safe source: the matching archetypes, taken mutably in dense order.
pub(crate) struct ExclusiveSource<I>(I);

impl<'w, I: Iterator<Item = &'w mut Archetype>> ArchetypeSource<'w> for ExclusiveSource<I> {
    fn load_next(
        &mut self,
        table: &mut TableAccess<'w>,
        plan: &[(ComponentId, bool, bool)],
    ) -> bool {
        match self.0.next() {
            Some(archetype) => {
                table.load(archetype, plan);
                true
            }
            None => false,
        }
    }
}

/// The cell source: the matching archetypes by shared reference. Only
/// constructible inside `Archetypes::iter_data_cell`, an `unsafe fn`
/// whose contract is what makes `load_next`'s `unsafe` call sound.
pub(crate) struct CellSource<'w> {
    archetypes: &'w Archetypes,
    matched: std::vec::IntoIter<ArchetypeId>,
}

impl<'w> ArchetypeSource<'w> for CellSource<'w> {
    fn load_next(
        &mut self,
        table: &mut TableAccess<'w>,
        plan: &[(ComponentId, bool, bool)],
    ) -> bool {
        let Some(id) = self.matched.next() else {
            return false;
        };
        let archetype = self.archetypes.archetypes.get(id).expect(
            "the cell iterator's precomputed matched list only ever contains real, currently-existing archetype ids",
        );
        // SAFETY: a `CellSource` exists only inside the iterator returned
        // by `Archetypes::iter_data_cell`, whose caller promised the
        // access contract for the iterator's whole lifetime.
        unsafe { table.load_cell(archetype, plan) };
        true
    }
}

/// The real `Iterator` behind `World::query_data_mut` and the
/// `WorldCell` queries. `S` is where the matching archetypes come from.
pub(crate) struct IterDataMut<'w, Q: QueryData, F: DataFilter, S> {
    source: S,
    state: Q::State,
    filter: F::State,
    /// `(component, written, ticks_read)` for every component the query
    /// touches.
    plan: Vec<(ComponentId, bool, bool)>,
    /// Reused across archetypes, so entering one allocates nothing once
    /// the first has sized the slot lists.
    table: TableAccess<'w>,
    fetch: Option<Q::Fetch<'w>>,
    /// The next row `Q::fetch` will be called for. `Mut` terms hand their
    /// slices out one element per call, so every row up to the last
    /// selected one is fetched, in order, even those a row filter skipped
    /// (their items are dropped unused: dropping a `Mut` stamps nothing).
    row: usize,
    len: usize,
    /// `F::TRACKED` only: see `IterData`.
    rows: Vec<usize>,
    next_selected: usize,
}

impl<'w, Q: QueryData, F: DataFilter, S: ArchetypeSource<'w>> Iterator
    for IterDataMut<'w, Q, F, S>
{
    type Item = Q::Item<'w>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if F::TRACKED {
                if let Some(fetch) = self.fetch.as_mut() {
                    if let Some(&target) = self.rows.get(self.next_selected) {
                        self.next_selected += 1;
                        while self.row < target {
                            drop(Q::fetch(fetch, self.row));
                            self.row += 1;
                        }
                        let item = Q::fetch(fetch, self.row);
                        self.row += 1;
                        return Some(item);
                    }
                }
            } else if self.row < self.len {
                if let Some(fetch) = self.fetch.as_mut() {
                    let item = Q::fetch(fetch, self.row);
                    self.row += 1;
                    return Some(item);
                }
            }
            if !self.source.load_next(&mut self.table, &self.plan) {
                return None;
            }
            if F::TRACKED {
                select_rows::<F>(&self.filter, &self.table, &mut self.rows);
                self.next_selected = 0;
            }
            let fetch = Q::init_fetch(&self.state, &mut self.table);
            self.len = Q::clamp_len(&fetch, self.table.len());
            if F::TRACKED {
                let keep = self.rows.partition_point(|&row| row < self.len);
                self.rows.truncate(keep);
            }
            self.fetch = Some(fetch);
            self.row = 0;
        }
    }
}

impl Archetypes {
    /// The `WorldCell` form of [`Self::iter_data_mut_filtered`]: the same
    /// iterator, reading the archetypes through `&self` and taking the
    /// columns it writes through their `Interior`, so several of these
    /// (and read-only ones) can be live at once over the same archetypes
    /// as long as they touch different columns.
    ///
    /// # Panics
    /// Under the same conditions as `iter_data_mut_filtered`.
    ///
    /// # Safety
    /// For as long as the returned iterator, or anything it yielded, is
    /// live: no `&mut Archetypes` or `&mut World` exists, and no other
    /// access conflicts with this query's own, as `FilteredAccess::of_query`
    /// describes it: nothing else reads or writes a component it writes
    /// (values *or* change ticks), and nothing else writes one it reads.
    pub(crate) unsafe fn iter_data_cell<'a, Q: QueryData + 'a, F: DataFilter + 'a>(
        &'a self,
        this_run: Tick,
        last_run: Tick,
    ) -> impl Iterator<Item = Q::Item<'a>> + 'a {
        let (state, filter, matched) = plan_matches::<Q, F>(self);
        let mut entries = Vec::new();
        Q::collect_access(&state, &mut entries);
        validate_access("WorldCell::query_iter_mut", &entries);
        let mut filter_ticks = Vec::new();
        F::collect_tick_reads(&filter, &mut filter_ticks);
        IterDataMut::<Q, F, _> {
            source: CellSource {
                archetypes: self,
                matched: matched.into_iter(),
            },
            state,
            filter,
            plan: build_plan(&entries, &filter_ticks),
            table: TableAccess::exclusive(this_run, last_run),
            fetch: None,
            row: 0,
            len: 0,
            rows: Vec::new(),
            next_selected: 0,
        }
    }

    /// Builds the exclusive iterator for query data `Q` (any terms) over
    /// the archetypes that contain every `&T`/`Mut<T>` term's component
    /// and satisfy `F`. Writes made through `Mut` terms are stamped with
    /// `this_run`; `last_run` is the tracker's tick, read only by
    /// `Added`/`Changed` terms and filters.
    ///
    /// # Panics
    /// If `Q` reads and writes, or writes twice, the same component type
    /// (see `validate_access`). Checked before anything is borrowed and
    /// regardless of which types the world holds. A row filter on a
    /// component `Q` also writes is *not* a conflict.
    pub(crate) fn iter_data_mut_filtered<'a, Q: QueryData + 'a, F: DataFilter + 'a>(
        &'a mut self,
        this_run: Tick,
        last_run: Tick,
    ) -> impl Iterator<Item = Q::Item<'a>> + 'a {
        let (state, filter, matched) = plan_matches::<Q, F>(self);
        let mut entries = Vec::new();
        Q::collect_access(&state, &mut entries);
        validate_access("query_data_mut", &entries);
        let mut filter_ticks = Vec::new();
        F::collect_tick_reads(&filter, &mut filter_ticks);
        IterDataMut::<Q, F, _> {
            source: ExclusiveSource(select(&mut self.archetypes, matched)),
            state,
            filter,
            plan: build_plan(&entries, &filter_ticks),
            table: TableAccess::exclusive(this_run, last_run),
            fetch: None,
            row: 0,
            len: 0,
            rows: Vec::new(),
            next_selected: 0,
        }
    }
}

// ── Parallel iteration ───────────────────────────────────────────────

/// One row window of one archetype, plus the raw handles to the columns
/// and tick columns the query writes there. Built only by
/// [`Archetypes::par_batches_cell`].
///
/// Invariant (what makes `TableAccess::load_window`, `column_mut` and
/// `ticks_mut` sound): every raw handle was captured by that `unsafe fn`
/// from the archetype this window names, under its contract: the query
/// has exclusive access to those columns for `'w`, the world's structure
/// does not change for `'w`, and no two windows of the query overlap.
struct WindowDesc<'w> {
    archetype: &'w Archetype,
    rows: Range<usize>,
    written: Vec<(ComponentId, RawColumn)>,
    written_ticks: Vec<(ComponentId, RawTicks)>,
}

/// The windowed source: the row windows of one [`QueryBatch`], in order.
pub(crate) struct WindowSource<'w> {
    windows: std::vec::IntoIter<WindowDesc<'w>>,
}

impl<'w> ArchetypeSource<'w> for WindowSource<'w> {
    fn load_next(
        &mut self,
        table: &mut TableAccess<'w>,
        plan: &[(ComponentId, bool, bool)],
    ) -> bool {
        match self.windows.next() {
            Some(desc) => {
                table.load_window(&desc, plan);
                true
            }
            None => false,
        }
    }
}

/// One independent piece of a parallel query: an iterator over the rows
/// of a few row windows, that can be moved to another thread and run
/// there to completion. Made by `WorldCell::query_batches` /
/// `Query::par_iter(_mut)`; every row the query matches is in exactly
/// one batch, and no two batches share a row.
///
/// Yields exactly what the same query yields sequentially (same terms,
/// same row filters, `Mut` writes stamped with the query's tick), for
/// its rows.
///
/// A batch is [`Send`] when the query's `Fetch` is, which is when its
/// component types are `Sync` for read terms and `Send` for `Mut`
/// terms: a query over a `Mut<Rc<_>>` cannot be sent to another thread
/// (and so cannot be run in parallel), checked by the compiler, not by
/// a rule here.
pub struct QueryBatch<'w, Q: QueryData, F: DataFilter> {
    inner: IterDataMut<'w, Q, F, WindowSource<'w>>,
}

// SAFETY: a batch holds (a) shared references into the archetypes'
// metadata (signatures, entity lists), plain data; (b) shared references
// to columns the query only reads, reachable only as typed `&[T]` through
// terms whose `Fetch` holds a `&[T]`; (c) raw handles to row windows of
// the columns the query writes, reachable only as `&mut [T]` through
// `Mut` terms' `Fetch`; (d) plain state. All typed access to component
// data goes through `Q::Fetch<'w>`, so requiring that to be `Send`
// requires `T: Sync` for every read term and `T: Send` for every written
// one, which is exactly what moving the access to another thread needs.
// The windows of different batches are disjoint (see `WindowDesc`), so
// two threads never write the same element, and a column one batch
// writes no batch of the same query reads (the access check).
unsafe impl<'w, Q: QueryData, F: DataFilter> Send for QueryBatch<'w, Q, F> where Q::Fetch<'w>: Send {}

impl<'w, Q: QueryData, F: DataFilter> Iterator for QueryBatch<'w, Q, F> {
    type Item = Q::Item<'w>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }
}

/// Captures the raw handles of the columns and tick columns `plan` writes
/// in `archetype`.
///
/// # Safety
/// The caller has exclusive access to those columns (values and ticks)
/// for as long as the handles are used.
unsafe fn capture_written(
    archetype: &Archetype,
    plan: &[(ComponentId, bool, bool)],
) -> (Vec<(ComponentId, RawColumn)>, Vec<(ComponentId, RawTicks)>) {
    let mut columns = Vec::new();
    let mut ticks = Vec::new();
    for &(id, written, _) in plan {
        if !written {
            continue;
        }
        if let Some(column) = archetype.table.columns.get(id) {
            // SAFETY: exclusive access, per this function's contract; the
            // `&mut` lives only for this call and only the raw handle
            // survives it.
            let raw = unsafe { column.get_mut_unchecked() }.raw_parts_mut();
            columns.push((id, raw));
        }
        if let Some(column_ticks) = archetype.table.ticks.get(id) {
            // SAFETY: as above, for the tick column.
            let vec = unsafe { column_ticks.get_mut_unchecked() };
            ticks.push((
                id,
                RawTicks {
                    ptr: vec.as_mut_ptr(),
                    len: vec.len(),
                },
            ));
        }
    }
    (columns, ticks)
}

impl Archetypes {
    /// Cuts the query `Q` filtered by `F` into [`QueryBatch`]es: each is
    /// an independent iterator over some row windows of the matching
    /// archetypes, that may run on any thread, concurrently with the
    /// others. Together the batches yield every row the sequential cell
    /// iterator (`iter_data_cell`) yields, once each.
    ///
    /// `threads` only sizes the batches (see [`BatchingStrategy`]); this
    /// function starts no thread.
    ///
    /// # Panics
    /// Under the conditions of `iter_data_mut_filtered`, or if
    /// `strategy` needs a thread count and `threads` is zero.
    ///
    /// # Safety
    /// For as long as any returned batch, or anything it yielded, is
    /// live: no `&mut Archetypes` or `&mut World` exists, the world's
    /// structure (entities, archetypes, columns) is not changed, and no
    /// other access conflicts with this query's own, as
    /// `FilteredAccess::of_query` describes it. The batches themselves
    /// never conflict with each other: their windows are disjoint.
    pub(crate) unsafe fn par_batches_cell<'a, Q: QueryData + 'a, F: DataFilter + 'a>(
        &'a self,
        this_run: Tick,
        last_run: Tick,
        strategy: &BatchingStrategy,
        threads: usize,
    ) -> Vec<QueryBatch<'a, Q, F>> {
        let (state, filter, matched) = plan_matches::<Q, F>(self);
        let mut entries = Vec::new();
        Q::collect_access(&state, &mut entries);
        validate_access("WorldCell::query_batches", &entries);
        let mut filter_ticks = Vec::new();
        F::collect_tick_reads(&filter, &mut filter_ticks);
        let plan = build_plan(&entries, &filter_ticks);

        let archetypes: Vec<&'a Archetype> = matched
            .iter()
            .map(|&id| {
                self.archetypes.get(id).expect(
                    "the precomputed matched list only ever contains real, currently-existing archetype ids",
                )
            })
            .collect();
        let lens: Vec<usize> = archetypes
            .iter()
            .map(|archetype| archetype.table.entities.len())
            .collect();
        // (A size of zero is fine: `plan_batches` treats it as one.)
        let batch_size =
            strategy.calc_batch_size(|| lens.iter().copied().max().unwrap_or(0), threads);

        type Captured = (Vec<(ComponentId, RawColumn)>, Vec<(ComponentId, RawTicks)>);
        let mut captured: Vec<Option<Captured>> = (0..archetypes.len()).map(|_| None).collect();
        let mut batches = Vec::new();
        for specs in plan_batches(&lens, batch_size) {
            let mut windows = Vec::with_capacity(specs.len());
            for spec in specs {
                let archetype = archetypes[spec.archetype];
                let (written, written_ticks) = captured[spec.archetype]
                    .get_or_insert_with(|| {
                        // SAFETY: this function's contract.
                        unsafe { capture_written(archetype, &plan) }
                    })
                    .clone();
                windows.push(WindowDesc {
                    archetype,
                    rows: spec.rows,
                    written,
                    written_ticks,
                });
            }
            batches.push(QueryBatch {
                inner: IterDataMut::<Q, F, _> {
                    source: WindowSource {
                        windows: windows.into_iter(),
                    },
                    state: state.clone(),
                    filter: filter.clone(),
                    plan: plan.clone(),
                    table: TableAccess::exclusive(this_run, last_run),
                    fetch: None,
                    row: 0,
                    len: 0,
                    rows: Vec::new(),
                    next_selected: 0,
                },
            });
        }
        batches
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
        world
            .archetypes
            .iter_data_filtered::<Q, ()>(Tick::ZERO, Tick::ZERO)
            .matched
            .len()
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
        plan_matches::<Q, ()>(&world.archetypes).2.len()
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

    /// How many archetypes the row filter `F` leaves for `Entity` to walk.
    /// Row filters decide rows later, but the archetype-level half (the
    /// component must be present) prunes first, and the rows a query
    /// yields cannot show that: an archetype without the component
    /// produces no surviving rows either way.
    fn matched_with<F: DataFilter>(world: &World) -> usize {
        plan_matches::<Entity, F>(&world.archetypes).2.len()
    }

    #[test]
    fn row_filters_prune_archetypes_that_lack_the_component() {
        use crate::data_filter::DataFilter as _;
        use crate::filter::{Or, With};
        use crate::query_data::{Added, Changed};
        let world = world_with_four_archetypes();
        assert_eq!(matched_with::<()>(&world), 4);
        assert_eq!(matched_with::<Changed<A>>(&world), 3);
        assert_eq!(matched_with::<Added<B>>(&world), 2);
        assert_eq!(matched_with::<Changed<C>>(&world), 1);
        assert_eq!(matched_with::<(Changed<A>, With<C>)>(&world), 1);
        assert_eq!(matched_with::<Changed<Never>>(&world), 0);
        assert_eq!(matched_with::<Or<(Changed<C>, With<B>)>>(&world), 2);
        assert_eq!(matched_with::<Or<(Changed<Never>, With<B>)>>(&world), 2);
        assert_eq!(
            matched_with::<Or<(Changed<Never>, Added<Never>)>>(&world),
            0
        );
    }

    #[test]
    fn a_mut_term_for_an_unseen_type_visits_no_archetype_but_option_mut_does_not_block() {
        let world = world_with_four_archetypes();
        assert_eq!(matched_for::<Mut<Never>>(&world), 0);
        assert_eq!(matched_for::<(Mut<A>, &Never)>(&world), 0);
        assert_eq!(matched_for::<(Entity, Option<Mut<Never>>)>(&world), 4);
    }

    // ── Windows ──────────────────────────────────────────────────────

    /// What a `TableAccess` loaded with the window `rows` of a 4-row
    /// archetype {A, B} shows: (entities, B column len, A column first
    /// value and len, A ticks len, B ticks len).
    fn window_shape(rows: Range<usize>) -> (usize, usize, Option<u32>, usize, usize, usize) {
        use std::any::TypeId;
        let mut world = World::new();
        for i in 0..4u32 {
            let e = world.spawn();
            world.insert_static(e, A(10 + i));
            world.insert_static(e, B(20 + i));
        }
        let archetypes = &world.archetypes;
        let id_of = |t: TypeId| archetypes.component_ids.get(&t).copied().unwrap();
        let (id_a, id_b) = (id_of(TypeId::of::<A>()), id_of(TypeId::of::<B>()));
        let archetype = archetypes
            .archetypes
            .iter()
            .map(|(_, a)| a)
            .find(|a| a.component_ids.contains(&id_a) && a.component_ids.contains(&id_b))
            .expect("the {A, B} archetype");
        // A written, B read (with its ticks).
        let plan = vec![(id_a, true, false), (id_b, false, true)];
        // SAFETY: `world` is not touched again until `table` is dropped.
        let (written, written_ticks) = unsafe { capture_written(archetype, &plan) };
        let desc = WindowDesc {
            archetype,
            rows,
            written,
            written_ticks,
        };
        let mut table = TableAccess::exclusive(Tick::default(), Tick::default());
        table.load_window(&desc, &plan);
        let entities = table.len();
        let b_len = table.column::<B>(id_b).map_or(usize::MAX, |c| c.len());
        let b_ticks = table.ticks(id_b).map_or(usize::MAX, |t| t.len());
        let a_col = table.column_mut::<A>(id_a).unwrap();
        let (a_first, a_len) = (a_col.first().map(|a| a.0), a_col.len());
        let a_ticks = table.ticks_mut(id_a).unwrap().len();
        (entities, b_len, a_first, a_len, a_ticks, b_ticks)
    }

    #[test]
    fn a_window_shows_exactly_its_rows_of_every_slice() {
        // Rows 1..3 of 4: two rows everywhere, starting at A(11).
        assert_eq!(window_shape(1..3), (2, 2, Some(11), 2, 2, 2));
        // The whole archetype.
        assert_eq!(window_shape(0..4), (4, 4, Some(10), 4, 4, 4));
        // A single last row.
        assert_eq!(window_shape(3..4), (1, 1, Some(13), 1, 1, 1));
    }

    #[test]
    fn a_window_reaching_past_its_archetype_is_cut_to_it() {
        // Not reachable from `plan_batches`, which never plans past the
        // end, but the clamps are what keep a bad window from reading
        // or writing out of bounds.
        assert_eq!(window_shape(2..10), (2, 2, Some(12), 2, 2, 2));
        // Entirely past the end: empty everywhere, no panic.
        assert_eq!(window_shape(8..10), (0, 0, None, 0, 0, 0));
    }
}
