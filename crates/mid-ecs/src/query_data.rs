// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-ecs.md, sections "Multi-term queries: `query_data.rs`",
// "Mutable terms in `query_data`" and "`Ref<T>`, `Added`/`Changed` terms and
// filters"
// ============================================================================
//! Multi-term query data for the Archetype Core: [`Entity`], `&T`,
//! `Option<&T>`, [`Has<T>`], [`Mut<T>`], `Option<Mut<T>>`, [`Ref<T>`],
//! `Option<Ref<T>>`, [`Added<T>`], [`Changed<T>`], and tuples of them (up
//! to twelve).
//!
//! Where `query_static` and `query2_static` fix their item shape in the
//! method name, `World::query_data::<Q>()` takes the shape as a type:
//! `Q = (Entity, &A, &B, &C)` is a three-component query,
//! `Q = (&A, Option<&B>, Has<C>)` visits every archetype holding an `A`
//! and reports whether that archetype also holds a `B` (and gives it) or a
//! `C`, and `Q = (Mut<A>, &B)` is a bulk write to `A` driven by `B`. The
//! terms:
//!
//! | term             | item             | archetype must contain `T`?     | access |
//! |------------------|------------------|---------------------------------|--------|
//! | `Entity`         | `Entity`         | no                              | none   |
//! | `&T`             | `&T`             | **yes**                         | read   |
//! | `Option<&T>`     | `Option<&T>`     | no: `None` where it is absent   | read   |
//! | `Has<T>`         | `bool`           | no: `false` where it is absent  | none   |
//! | `Mut<T>`         | `Mut<T>`         | **yes**                         | write  |
//! | `Option<Mut<T>>` | `Option<Mut<T>>` | no: `None` where it is absent   | write  |
//! | `Ref<T>`         | `Ref<T>`         | **yes**                         | read (+ticks) |
//! | `Option<Ref<T>>` | `Option<Ref<T>>` | no: `None` where it is absent   | read (+ticks) |
//! | `Added<T>`       | `bool`           | **yes**                         | read ticks |
//! | `Changed<T>`     | `bool`           | **yes**                         | read ticks |
//!
//! Only `&T`, `Mut<T>`, `Ref<T>`, `Added<T>` and `Changed<T>` terms narrow
//! which archetypes are visited.
//! `Option` and `Has` terms are decided once per archetype, not per row,
//! and every term resolves its column slice once when the iterator enters
//! an archetype, so a row costs slice indexing and nothing else.
//!
//! A query with none of those narrowing terms (`Entity`, or
//! `(Entity, Has<T>)`) visits every archetype, the empty one included, so
//! it enumerates every live entity.
//!
//! `T` must have been inserted with `insert_static`, `insert_bundle` or
//! `spawn_bundle` for the Archetype Core to know it. For an unknown type
//! `&T` and `Mut<T>` make the whole query empty, `Option<&T>` and
//! `Option<Mut<T>>` are always `None` and `Has<T>` is always `false`;
//! asking never registers anything, the same contract `filter.rs`
//! documents.
//!
//! **Access rules.** One query may read a component through any number of
//! terms, or write it through exactly one `Mut` term and nothing else
//! (`Has` and `Entity` never conflict: they read no data). A violation,
//! such as `(&A, Mut<A>)` or `(Mut<A>, Mut<A>)`, panics when the query is
//! created, whether or not any row holds an `A`: it is a fact about the
//! query's type, not its data. Checked at run time because stable Rust
//! cannot compare the types inside a tuple.
//!
//! **Change detection.** [`Ref<T>`] is `&T` plus the row's change ticks:
//! `is_added(&tracker)` / `is_changed(&tracker)` answer per row, with the
//! same arguments [`Mut`] takes, and it needs no tracker at query time.
//! The [`Added<T>`] and [`Changed<T>`] terms instead yield the answer as a
//! `bool`, so they need the tracker when the query is created: they are
//! only accepted by the `_since` query methods
//! (`World::query_data_since` and friends), which take a
//! [`crate::ChangeTracker`]; the plain methods panic when the query is
//! created if `Q` contains one. The same two types are also row-level
//! *filters* (`data_filter.rs`), which skip the rows that fail instead of
//! reporting them. A `Ref`, `Added` or `Changed` term on a component
//! counts as a read of it, so it conflicts with a `Mut` of the same
//! component in the same query; the *filters* do not (they are evaluated
//! before the `Mut` term takes the column), so `Mut<T>` filtered by
//! `Changed<T>` is allowed.
//!
//! **No raw `&mut T` term.** A raw write would skip the change ticks and
//! `Changed` queries would miss it. `Mut<T>` stamps a row changed only
//! when it is actually written through, exactly as `query_static_mut`
//! does. Mutable terms need `World::query_data_mut` (`&mut self`); the
//! shared `World::query_data` only accepts [`ReadOnlyQueryData`].
//!
//! This module is separate from `archetype/iter.rs`, whose `Iter1`/`Iter2`
//! bodies are held fixed, so nothing here touches the existing query
//! paths.

use std::any::{type_name, TypeId};
use std::marker::PhantomData;

use crate::access::FilteredAccess;
use crate::archetype::TableAccess;
use crate::component::ComponentId;
use crate::tick::{ComponentTicks, Mut, Ref, Tick};
use crate::world::Entity;

mod sealed {
    /// Empty on purpose: exists only so `QueryData` can't be implemented
    /// outside this crate.
    pub trait Sealed {}
}

/// One term of a query, or a tuple of terms: what a `World::query_data`
/// or `World::query_data_mut` call yields per row. Implemented for
/// [`Entity`], `&T`, `Option<&T>`, [`Has<T>`], [`Mut<T>`],
/// `Option<Mut<T>>` and tuples of those up to twelve members (tuples
/// nest). Sealed: it can't be implemented outside this crate.
///
/// Everything but [`Self::Item`] is machinery for the iterators in
/// `archetype/iter_data.rs` and hidden from the docs.
pub trait QueryData: sealed::Sealed {
    /// `true` if this term (or a tuple member) compares against a
    /// [`crate::ChangeTracker`] when the query is created: [`Added`] and
    /// [`Changed`]. Such a query is only valid through the `_since`
    /// methods, which are the ones that have a tracker to give.
    #[doc(hidden)]
    const TRACKED: bool = false;

    /// What one visited row yields.
    type Item<'w>;

    /// What this query resolves to once, from the type registry: one
    /// `Option<ComponentId>` per component term.
    #[doc(hidden)]
    type State;

    /// What one archetype resolves to: the column slices (or the
    /// presence bool) the term reads rows from.
    #[doc(hidden)]
    type Fetch<'w>;

    /// Resolves this query's component types against the Archetype
    /// Core's own `TypeId -> ComponentId` map. Read-only: never
    /// registers a type.
    #[doc(hidden)]
    fn get_state(resolve: &dyn Fn(TypeId) -> Option<ComponentId>) -> Self::State;

    /// `false` if some `&T` or `Mut<T>` term names a type the Archetype
    /// Core has never seen, in which case nothing can match.
    #[doc(hidden)]
    fn is_satisfiable(state: &Self::State) -> bool;

    /// Pushes the id of every `&T` and `Mut<T>` term: the components an
    /// archetype must contain to be visited. `Option` and `Has` terms
    /// push nothing.
    #[doc(hidden)]
    fn collect_required(state: &Self::State, out: &mut Vec<ComponentId>);

    /// Pushes one entry per term that touches component data (`&T`,
    /// `Option<&T>`, `Mut<T>`, `Option<Mut<T>>`), whether the type is
    /// known or not: the input to the access-conflict check and to the
    /// per-archetype column hand-out of the mutable iterator.
    #[doc(hidden)]
    fn collect_access(state: &Self::State, out: &mut Vec<AccessEntry>);

    /// Records what this term reads, writes and requires into `access`,
    /// without a world: the same facts as `collect_access` plus the
    /// archetype constraint a required term implies (`with`), keyed by
    /// type so a system can be analysed before the world holds any
    /// entity. See `access.rs`.
    #[doc(hidden)]
    fn update_access(access: &mut FilteredAccess);

    /// Resolves the slices and flags for one archetype's table.
    #[doc(hidden)]
    fn init_fetch<'w>(state: &Self::State, table: &mut TableAccess<'w>) -> Self::Fetch<'w>;

    /// The number of rows of this archetype the terms can all serve:
    /// `len` (the table's entity count) clamped to every column slice
    /// the terms hold, so stepping that many rows cannot run out.
    #[doc(hidden)]
    fn clamp_len(fetch: &Self::Fetch<'_>, len: usize) -> usize;

    /// The item for `row`. Called for rows `0, 1, 2, ...` of an
    /// archetype, in order, exactly once each, and only below
    /// [`Self::clamp_len`]: `Mut` terms hand out their slice one element
    /// at a time and rely on that order (it is what makes the hand-out
    /// safe code), while read terms index by `row`.
    #[doc(hidden)]
    fn fetch<'w>(fetch: &mut Self::Fetch<'w>, row: usize) -> Self::Item<'w>;
}

/// A [`QueryData`] that only reads, and so can be iterated through a
/// shared `&World`: everything except [`Mut<T>`] and `Option<Mut<T>>`,
/// and tuples made only of such terms. `World::query_data` requires it;
/// `World::query_data_mut` does not.
pub trait ReadOnlyQueryData: QueryData {}

/// One component access a query declares. Hidden: machinery for the
/// conflict check and the mutable iterator's column plan.
#[doc(hidden)]
pub struct AccessEntry {
    pub(crate) type_id: TypeId,
    pub(crate) name: &'static str,
    pub(crate) id: Option<ComponentId>,
    pub(crate) write: bool,
    /// The term also needs the component's change-tick slice (`Ref`,
    /// `Added`, `Changed`), so the iterator must hand it out.
    pub(crate) ticks: bool,
}

impl AccessEntry {
    fn read<T: 'static>(id: Option<ComponentId>) -> Self {
        Self {
            type_id: TypeId::of::<T>(),
            name: type_name::<T>(),
            id,
            write: false,
            ticks: false,
        }
    }

    fn read_ticked<T: 'static>(id: Option<ComponentId>) -> Self {
        Self {
            ticks: true,
            ..Self::read::<T>(id)
        }
    }

    fn write<T: 'static>(id: Option<ComponentId>) -> Self {
        Self {
            type_id: TypeId::of::<T>(),
            name: type_name::<T>(),
            id,
            write: true,
            ticks: false,
        }
    }
}

/// Yields whether the visited archetype contains `T`, as a `bool`.
/// Never constructed; a type-level marker only. Unlike `With<T>` it does
/// not narrow the query: rows without `T` are visited and report
/// `false`. It reads no component data, so it never conflicts with a
/// `Mut<T>` of the same type in the same query.
pub struct Has<T>(PhantomData<fn() -> T>);

// ── Entity ───────────────────────────────────────────────────────────

impl sealed::Sealed for Entity {}
impl ReadOnlyQueryData for Entity {}

impl QueryData for Entity {
    type Item<'w> = Entity;
    type State = ();
    type Fetch<'w> = &'w [Entity];

    #[inline]
    fn get_state(_resolve: &dyn Fn(TypeId) -> Option<ComponentId>) -> Self::State {}

    #[inline]
    fn is_satisfiable(_state: &Self::State) -> bool {
        true
    }

    #[inline]
    fn collect_required(_state: &Self::State, _out: &mut Vec<ComponentId>) {}

    #[inline]
    fn collect_access(_state: &Self::State, _out: &mut Vec<AccessEntry>) {}

    #[inline]
    fn update_access(_access: &mut FilteredAccess) {}

    #[inline]
    fn init_fetch<'w>(_state: &Self::State, table: &mut TableAccess<'w>) -> Self::Fetch<'w> {
        table.entities()
    }

    #[inline]
    fn clamp_len(fetch: &Self::Fetch<'_>, len: usize) -> usize {
        len.min(fetch.len())
    }

    #[inline]
    fn fetch<'w>(fetch: &mut Self::Fetch<'w>, row: usize) -> Self::Item<'w> {
        fetch[row]
    }
}

// ── &T ───────────────────────────────────────────────────────────────

impl<T: 'static> sealed::Sealed for &T {}
impl<T: 'static> ReadOnlyQueryData for &T {}

impl<T: 'static> QueryData for &T {
    type Item<'w> = &'w T;
    type State = Option<ComponentId>;
    type Fetch<'w> = &'w [T];

    #[inline]
    fn get_state(resolve: &dyn Fn(TypeId) -> Option<ComponentId>) -> Self::State {
        resolve(TypeId::of::<T>())
    }

    #[inline]
    fn is_satisfiable(state: &Self::State) -> bool {
        state.is_some()
    }

    #[inline]
    fn collect_required(state: &Self::State, out: &mut Vec<ComponentId>) {
        if let Some(id) = *state {
            out.push(id);
        }
    }

    #[inline]
    fn collect_access(state: &Self::State, out: &mut Vec<AccessEntry>) {
        out.push(AccessEntry::read::<T>(*state));
    }

    #[inline]
    fn update_access(access: &mut FilteredAccess) {
        access.add_component_read::<T>();
        access.add_with::<T>();
    }

    #[inline]
    fn init_fetch<'w>(state: &Self::State, table: &mut TableAccess<'w>) -> Self::Fetch<'w> {
        // A signature can list a component whose column the table has
        // not created yet (it is made lazily, on the first row). That
        // table has no rows, so an empty slice is the right answer.
        match *state {
            Some(id) => table.column::<T>(id).unwrap_or(&[]),
            None => &[],
        }
    }

    #[inline]
    fn clamp_len(fetch: &Self::Fetch<'_>, len: usize) -> usize {
        len.min(fetch.len())
    }

    #[inline]
    fn fetch<'w>(fetch: &mut Self::Fetch<'w>, row: usize) -> Self::Item<'w> {
        let column: &'w [T] = fetch;
        &column[row]
    }
}

// ── Option<&T> ───────────────────────────────────────────────────────

impl<T: 'static> sealed::Sealed for Option<&T> {}
impl<T: 'static> ReadOnlyQueryData for Option<&T> {}

impl<T: 'static> QueryData for Option<&T> {
    type Item<'w> = Option<&'w T>;
    type State = Option<ComponentId>;
    type Fetch<'w> = Option<&'w [T]>;

    #[inline]
    fn get_state(resolve: &dyn Fn(TypeId) -> Option<ComponentId>) -> Self::State {
        resolve(TypeId::of::<T>())
    }

    #[inline]
    fn is_satisfiable(_state: &Self::State) -> bool {
        true
    }

    #[inline]
    fn collect_required(_state: &Self::State, _out: &mut Vec<ComponentId>) {}

    #[inline]
    fn collect_access(state: &Self::State, out: &mut Vec<AccessEntry>) {
        out.push(AccessEntry::read::<T>(*state));
    }

    #[inline]
    fn update_access(access: &mut FilteredAccess) {
        access.add_component_read::<T>();
    }

    #[inline]
    fn init_fetch<'w>(state: &Self::State, table: &mut TableAccess<'w>) -> Self::Fetch<'w> {
        // Presence is the archetype's signature, not whether the column
        // object exists yet (see the `&T` impl).
        match *state {
            Some(id) if table.contains(id) => Some(table.column::<T>(id).unwrap_or(&[])),
            _ => None,
        }
    }

    #[inline]
    fn clamp_len(fetch: &Self::Fetch<'_>, len: usize) -> usize {
        match fetch {
            Some(column) => len.min(column.len()),
            None => len,
        }
    }

    #[inline]
    fn fetch<'w>(fetch: &mut Self::Fetch<'w>, row: usize) -> Self::Item<'w> {
        match *fetch {
            Some(column) => Some(&column[row]),
            None => None,
        }
    }
}

// ── Has<T> ───────────────────────────────────────────────────────────

impl<T: 'static> sealed::Sealed for Has<T> {}
impl<T: 'static> ReadOnlyQueryData for Has<T> {}

impl<T: 'static> QueryData for Has<T> {
    type Item<'w> = bool;
    type State = Option<ComponentId>;
    type Fetch<'w> = bool;

    #[inline]
    fn get_state(resolve: &dyn Fn(TypeId) -> Option<ComponentId>) -> Self::State {
        resolve(TypeId::of::<T>())
    }

    #[inline]
    fn is_satisfiable(_state: &Self::State) -> bool {
        true
    }

    #[inline]
    fn collect_required(_state: &Self::State, _out: &mut Vec<ComponentId>) {}

    #[inline]
    fn collect_access(_state: &Self::State, _out: &mut Vec<AccessEntry>) {}

    #[inline]
    fn update_access(_access: &mut FilteredAccess) {}

    #[inline]
    fn init_fetch<'w>(state: &Self::State, table: &mut TableAccess<'w>) -> Self::Fetch<'w> {
        matches!(*state, Some(id) if table.contains(id))
    }

    #[inline]
    fn clamp_len(_fetch: &Self::Fetch<'_>, len: usize) -> usize {
        len
    }

    #[inline]
    fn fetch<'w>(fetch: &mut Self::Fetch<'w>, _row: usize) -> Self::Item<'w> {
        *fetch
    }
}

// ── Mut<T> and Option<Mut<T>> ────────────────────────────────────────

/// One archetype's worth of a `Mut<T>` term: the value slice, the tick
/// slice that goes with it, and the world tick a write is stamped with.
/// Both slices are consumed from the front, one element per row.
#[doc(hidden)]
pub struct MutFetch<'w, T> {
    values: &'w mut [T],
    ticks: &'w mut [ComponentTicks],
    this_run: Tick,
}

fn mut_fetch<'w, T: 'static>(id: ComponentId, table: &mut TableAccess<'w>) -> MutFetch<'w, T> {
    MutFetch {
        values: table.column_mut::<T>(id).unwrap_or(&mut []),
        ticks: table.ticks_mut(id).unwrap_or(&mut []),
        this_run: table.this_run(),
    }
}

fn mut_clamp<T>(fetch: &MutFetch<'_, T>, len: usize) -> usize {
    len.min(fetch.values.len()).min(fetch.ticks.len())
}

/// Hands out the next row's `Mut` and moves both slices past it. This is
/// `slice::IterMut` in safe code: take the slice out of the struct, split
/// the first element off it, put the rest back.
fn mut_next<'w, T>(fetch: &mut MutFetch<'w, T>) -> Mut<'w, T> {
    let values = std::mem::take(&mut fetch.values);
    let ticks = std::mem::take(&mut fetch.ticks);
    let (value, rest_values) = values
        .split_first_mut()
        .expect("fetch is only called below clamp_len");
    let (tick, rest_ticks) = ticks
        .split_first_mut()
        .expect("fetch is only called below clamp_len");
    fetch.values = rest_values;
    fetch.ticks = rest_ticks;
    Mut::new(value, tick, fetch.this_run)
}

impl<T: 'static> sealed::Sealed for Mut<'_, T> {}

impl<T: 'static> QueryData for Mut<'_, T> {
    type Item<'w> = Mut<'w, T>;
    type State = Option<ComponentId>;
    type Fetch<'w> = MutFetch<'w, T>;

    #[inline]
    fn get_state(resolve: &dyn Fn(TypeId) -> Option<ComponentId>) -> Self::State {
        resolve(TypeId::of::<T>())
    }

    #[inline]
    fn is_satisfiable(state: &Self::State) -> bool {
        state.is_some()
    }

    #[inline]
    fn collect_required(state: &Self::State, out: &mut Vec<ComponentId>) {
        if let Some(id) = *state {
            out.push(id);
        }
    }

    #[inline]
    fn collect_access(state: &Self::State, out: &mut Vec<AccessEntry>) {
        out.push(AccessEntry::write::<T>(*state));
    }

    #[inline]
    fn update_access(access: &mut FilteredAccess) {
        access.add_component_write::<T>();
        access.add_with::<T>();
    }

    #[inline]
    fn init_fetch<'w>(state: &Self::State, table: &mut TableAccess<'w>) -> Self::Fetch<'w> {
        match *state {
            Some(id) => mut_fetch::<T>(id, table),
            None => MutFetch {
                values: &mut [],
                ticks: &mut [],
                this_run: table.this_run(),
            },
        }
    }

    #[inline]
    fn clamp_len(fetch: &Self::Fetch<'_>, len: usize) -> usize {
        mut_clamp(fetch, len)
    }

    #[inline]
    fn fetch<'w>(fetch: &mut Self::Fetch<'w>, _row: usize) -> Self::Item<'w> {
        mut_next(fetch)
    }
}

impl<T: 'static> sealed::Sealed for Option<Mut<'_, T>> {}

impl<T: 'static> QueryData for Option<Mut<'_, T>> {
    type Item<'w> = Option<Mut<'w, T>>;
    type State = Option<ComponentId>;
    type Fetch<'w> = Option<MutFetch<'w, T>>;

    #[inline]
    fn get_state(resolve: &dyn Fn(TypeId) -> Option<ComponentId>) -> Self::State {
        resolve(TypeId::of::<T>())
    }

    #[inline]
    fn is_satisfiable(_state: &Self::State) -> bool {
        true
    }

    #[inline]
    fn collect_required(_state: &Self::State, _out: &mut Vec<ComponentId>) {}

    #[inline]
    fn collect_access(state: &Self::State, out: &mut Vec<AccessEntry>) {
        out.push(AccessEntry::write::<T>(*state));
    }

    #[inline]
    fn update_access(access: &mut FilteredAccess) {
        access.add_component_write::<T>();
    }

    #[inline]
    fn init_fetch<'w>(state: &Self::State, table: &mut TableAccess<'w>) -> Self::Fetch<'w> {
        match *state {
            Some(id) if table.contains(id) => Some(mut_fetch::<T>(id, table)),
            _ => None,
        }
    }

    #[inline]
    fn clamp_len(fetch: &Self::Fetch<'_>, len: usize) -> usize {
        match fetch {
            Some(fetch) => mut_clamp(fetch, len),
            None => len,
        }
    }

    #[inline]
    fn fetch<'w>(fetch: &mut Self::Fetch<'w>, _row: usize) -> Self::Item<'w> {
        fetch.as_mut().map(mut_next)
    }
}

// ── Ref<T> and Option<Ref<T>> ───────────────────────────────────────

/// One archetype's worth of a `Ref<T>` term: the value slice, the tick
/// slice that goes with it, and the world tick the row's `Ref` carries.
#[doc(hidden)]
pub struct RefFetch<'w, T> {
    values: &'w [T],
    ticks: &'w [ComponentTicks],
    this_run: Tick,
}

fn ref_fetch<'w, T: 'static>(id: ComponentId, table: &TableAccess<'w>) -> RefFetch<'w, T> {
    RefFetch {
        values: table.column::<T>(id).unwrap_or(&[]),
        ticks: table.ticks(id).unwrap_or(&[]),
        this_run: table.this_run(),
    }
}

fn ref_clamp<T>(fetch: &RefFetch<'_, T>, len: usize) -> usize {
    len.min(fetch.values.len()).min(fetch.ticks.len())
}

fn ref_at<'w, T>(fetch: &RefFetch<'w, T>, row: usize) -> Ref<'w, T> {
    Ref::new(&fetch.values[row], &fetch.ticks[row], fetch.this_run)
}

impl<T: 'static> sealed::Sealed for Ref<'_, T> {}
impl<T: 'static> ReadOnlyQueryData for Ref<'_, T> {}

impl<T: 'static> QueryData for Ref<'_, T> {
    type Item<'w> = Ref<'w, T>;
    type State = Option<ComponentId>;
    type Fetch<'w> = RefFetch<'w, T>;

    #[inline]
    fn get_state(resolve: &dyn Fn(TypeId) -> Option<ComponentId>) -> Self::State {
        resolve(TypeId::of::<T>())
    }

    #[inline]
    fn is_satisfiable(state: &Self::State) -> bool {
        state.is_some()
    }

    #[inline]
    fn collect_required(state: &Self::State, out: &mut Vec<ComponentId>) {
        if let Some(id) = *state {
            out.push(id);
        }
    }

    #[inline]
    fn collect_access(state: &Self::State, out: &mut Vec<AccessEntry>) {
        out.push(AccessEntry::read_ticked::<T>(*state));
    }

    #[inline]
    fn update_access(access: &mut FilteredAccess) {
        access.add_component_read::<T>();
        access.add_with::<T>();
    }

    #[inline]
    fn init_fetch<'w>(state: &Self::State, table: &mut TableAccess<'w>) -> Self::Fetch<'w> {
        match *state {
            Some(id) => ref_fetch::<T>(id, table),
            None => RefFetch {
                values: &[],
                ticks: &[],
                this_run: table.this_run(),
            },
        }
    }

    #[inline]
    fn clamp_len(fetch: &Self::Fetch<'_>, len: usize) -> usize {
        ref_clamp(fetch, len)
    }

    #[inline]
    fn fetch<'w>(fetch: &mut Self::Fetch<'w>, row: usize) -> Self::Item<'w> {
        ref_at(fetch, row)
    }
}

impl<T: 'static> sealed::Sealed for Option<Ref<'_, T>> {}
impl<T: 'static> ReadOnlyQueryData for Option<Ref<'_, T>> {}

impl<T: 'static> QueryData for Option<Ref<'_, T>> {
    type Item<'w> = Option<Ref<'w, T>>;
    type State = Option<ComponentId>;
    type Fetch<'w> = Option<RefFetch<'w, T>>;

    #[inline]
    fn get_state(resolve: &dyn Fn(TypeId) -> Option<ComponentId>) -> Self::State {
        resolve(TypeId::of::<T>())
    }

    #[inline]
    fn is_satisfiable(_state: &Self::State) -> bool {
        true
    }

    #[inline]
    fn collect_required(_state: &Self::State, _out: &mut Vec<ComponentId>) {}

    #[inline]
    fn collect_access(state: &Self::State, out: &mut Vec<AccessEntry>) {
        out.push(AccessEntry::read_ticked::<T>(*state));
    }

    #[inline]
    fn update_access(access: &mut FilteredAccess) {
        access.add_component_read::<T>();
    }

    #[inline]
    fn init_fetch<'w>(state: &Self::State, table: &mut TableAccess<'w>) -> Self::Fetch<'w> {
        match *state {
            Some(id) if table.contains(id) => Some(ref_fetch::<T>(id, table)),
            _ => None,
        }
    }

    #[inline]
    fn clamp_len(fetch: &Self::Fetch<'_>, len: usize) -> usize {
        match fetch {
            Some(fetch) => ref_clamp(fetch, len),
            None => len,
        }
    }

    #[inline]
    fn fetch<'w>(fetch: &mut Self::Fetch<'w>, row: usize) -> Self::Item<'w> {
        fetch.as_ref().map(|fetch| ref_at(fetch, row))
    }
}

// ── Added<T> and Changed<T> as terms ─────────────────────────────────

/// Yields, per row, whether `T` was inserted after the tracker's
/// last-checked tick, as a `bool`. As a *term* it never skips a row; as a
/// filter (`data_filter.rs`) the same type skips the rows where this is
/// false. Needs the `_since` query methods, which take the
/// [`crate::ChangeTracker`]. The archetype must contain `T`. Never
/// constructed; a type-level marker only.
pub struct Added<T>(PhantomData<fn() -> T>);

/// Yields, per row, whether `T` was inserted or mutated after the
/// tracker's last-checked tick, as a `bool`. As a *term* it never skips a
/// row; as a filter (`data_filter.rs`) the same type skips the rows where
/// this is false. Needs the `_since` query methods, which take the
/// [`crate::ChangeTracker`]. The archetype must contain `T`. Never
/// constructed; a type-level marker only.
pub struct Changed<T>(PhantomData<fn() -> T>);

/// One archetype's worth of an `Added`/`Changed` term: the tick slice and
/// the two ticks a row is compared with.
#[doc(hidden)]
pub struct TickFetch<'w> {
    ticks: &'w [ComponentTicks],
    last_run: Tick,
    this_run: Tick,
}

macro_rules! impl_tick_term {
    ($name:ident, $check:ident) => {
        impl<T: 'static> sealed::Sealed for $name<T> {}
        impl<T: 'static> ReadOnlyQueryData for $name<T> {}

        impl<T: 'static> QueryData for $name<T> {
            const TRACKED: bool = true;
            type Item<'w> = bool;
            type State = Option<ComponentId>;
            type Fetch<'w> = TickFetch<'w>;

            #[inline]
            fn get_state(resolve: &dyn Fn(TypeId) -> Option<ComponentId>) -> Self::State {
                resolve(TypeId::of::<T>())
            }

            #[inline]
            fn is_satisfiable(state: &Self::State) -> bool {
                state.is_some()
            }

            #[inline]
            fn collect_required(state: &Self::State, out: &mut Vec<ComponentId>) {
                if let Some(id) = *state {
                    out.push(id);
                }
            }

            #[inline]
            fn collect_access(state: &Self::State, out: &mut Vec<AccessEntry>) {
                out.push(AccessEntry::read_ticked::<T>(*state));
            }

            #[inline]
            fn update_access(access: &mut FilteredAccess) {
                access.add_component_read::<T>();
                access.add_with::<T>();
            }

            #[inline]
            fn init_fetch<'w>(state: &Self::State, table: &mut TableAccess<'w>) -> Self::Fetch<'w> {
                TickFetch {
                    ticks: match *state {
                        Some(id) => table.ticks(id).unwrap_or(&[]),
                        None => &[],
                    },
                    last_run: table.last_run(),
                    this_run: table.this_run(),
                }
            }

            #[inline]
            fn clamp_len(fetch: &Self::Fetch<'_>, len: usize) -> usize {
                len.min(fetch.ticks.len())
            }

            #[inline]
            fn fetch<'w>(fetch: &mut Self::Fetch<'w>, row: usize) -> Self::Item<'w> {
                fetch.ticks[row].$check(fetch.last_run, fetch.this_run)
            }
        }
    };
}

impl_tick_term!(Added, is_added);
impl_tick_term!(Changed, is_changed);

// ── Tuples ───────────────────────────────────────────────────────────

macro_rules! impl_query_data_for_tuple {
    ($($T:ident),+) => {
        impl<$($T: QueryData),+> sealed::Sealed for ($($T,)+) {}

        impl<$($T: ReadOnlyQueryData),+> ReadOnlyQueryData for ($($T,)+) {}

        impl<$($T: QueryData),+> QueryData for ($($T,)+) {
            const TRACKED: bool = false $(|| $T::TRACKED)+;
            type Item<'w> = ($($T::Item<'w>,)+);
            type State = ($($T::State,)+);
            type Fetch<'w> = ($($T::Fetch<'w>,)+);

            #[inline]
            fn get_state(resolve: &dyn Fn(TypeId) -> Option<ComponentId>) -> Self::State {
                ($($T::get_state(resolve),)+)
            }

            // Same trick `filter.rs` uses: a type parameter and the local
            // that destructures its state share a name, which is fine
            // (types and values are separate namespaces).
            #[inline]
            #[allow(non_snake_case)]
            fn is_satisfiable(state: &Self::State) -> bool {
                let ($($T,)+) = state;
                true $(&& $T::is_satisfiable($T))+
            }

            #[inline]
            #[allow(non_snake_case)]
            fn collect_required(state: &Self::State, out: &mut Vec<ComponentId>) {
                let ($($T,)+) = state;
                $($T::collect_required($T, out);)+
            }

            #[inline]
            #[allow(non_snake_case)]
            fn collect_access(state: &Self::State, out: &mut Vec<AccessEntry>) {
                let ($($T,)+) = state;
                $($T::collect_access($T, out);)+
            }

            #[inline]
            fn update_access(access: &mut FilteredAccess) {
                $($T::update_access(access);)+
            }

            #[inline]
            #[allow(non_snake_case)]
            fn init_fetch<'w>(state: &Self::State, table: &mut TableAccess<'w>) -> Self::Fetch<'w> {
                let ($($T,)+) = state;
                ($($T::init_fetch($T, table),)+)
            }

            #[inline]
            #[allow(non_snake_case)]
            fn clamp_len(fetch: &Self::Fetch<'_>, len: usize) -> usize {
                let ($($T,)+) = fetch;
                let mut len = len;
                $(len = $T::clamp_len($T, len);)+
                len
            }

            #[inline]
            #[allow(non_snake_case)]
            fn fetch<'w>(fetch: &mut Self::Fetch<'w>, row: usize) -> Self::Item<'w> {
                let ($($T,)+) = fetch;
                ($($T::fetch($T, row),)+)
            }
        }
    };
}

impl_query_data_for_tuple!(A);
impl_query_data_for_tuple!(A, B);
impl_query_data_for_tuple!(A, B, C);
impl_query_data_for_tuple!(A, B, C, D);
impl_query_data_for_tuple!(A, B, C, D, E);
impl_query_data_for_tuple!(A, B, C, D, E, F);
impl_query_data_for_tuple!(A, B, C, D, E, F, G);
impl_query_data_for_tuple!(A, B, C, D, E, F, G, H);
impl_query_data_for_tuple!(A, B, C, D, E, F, G, H, I);
impl_query_data_for_tuple!(A, B, C, D, E, F, G, H, I, J);
impl_query_data_for_tuple!(A, B, C, D, E, F, G, H, I, J, K);
impl_query_data_for_tuple!(A, B, C, D, E, F, G, H, I, J, K, L);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filter::{Or, With, Without};
    use crate::tick::ChangeTracker;
    use crate::world::World;

    #[derive(Debug, PartialEq, Clone, Copy)]
    struct A(u32);
    #[derive(Debug, PartialEq, Clone, Copy)]
    struct B(u32);
    #[derive(Debug, PartialEq, Clone, Copy)]
    struct C(u32);
    #[derive(Debug, PartialEq, Clone, Copy)]
    struct D(u32);
    /// Zero-sized on purpose.
    #[derive(Debug, PartialEq, Clone, Copy)]
    struct Tag;
    /// Never inserted anywhere: the Archetype Core has never seen it.
    struct Never;

    /// Spawns an entity and attaches the chosen components, with values
    /// derived from `n` (`A(n)`, `B(n * 10)`, `C(n * 100)`) so a test can
    /// tell whether a row's columns line up with each other.
    fn make(world: &mut World, n: u32, a: bool, b: bool, c: bool) -> Entity {
        let e = world.spawn();
        if a {
            world.insert_static(e, A(n));
        }
        if b {
            world.insert_static(e, B(n * 10));
        }
        if c {
            world.insert_static(e, C(n * 100));
        }
        e
    }

    #[test]
    fn three_component_query_yields_exactly_the_entities_holding_all_three() {
        let mut world = World::new();
        let full = make(&mut world, 1, true, true, true);
        make(&mut world, 2, true, true, false);
        make(&mut world, 3, true, false, true);
        make(&mut world, 4, false, true, true);
        let full2 = make(&mut world, 5, true, true, true);

        let mut got: Vec<_> = world
            .query_data::<(Entity, &A, &B, &C)>()
            .map(|(e, a, b, c)| (e, *a, *b, *c))
            .collect();
        got.sort_by_key(|row| row.1 .0);
        assert_eq!(
            got,
            vec![(full, A(1), B(10), C(100)), (full2, A(5), B(50), C(500))]
        );
    }

    #[test]
    fn single_non_tuple_terms_work() {
        let mut world = World::new();
        let e = make(&mut world, 7, true, false, false);
        let a: Vec<_> = world.query_data::<&A>().map(|a| a.0).collect();
        assert_eq!(a, vec![7]);
        let ids: Vec<_> = world.query_data::<Entity>().collect();
        assert_eq!(ids, vec![e]);
        let one: Vec<_> = world.query_data::<(Entity,)>().collect();
        assert_eq!(one, vec![(e,)]);
    }

    #[test]
    fn option_term_is_some_where_present_none_where_absent_and_never_narrows() {
        let mut world = World::new();
        let with_b = make(&mut world, 1, true, true, false);
        let without_b = make(&mut world, 2, true, false, false);
        make(&mut world, 3, false, true, false); // no A: not visited

        let mut got: Vec<_> = world
            .query_data::<(Entity, &A, Option<&B>)>()
            .map(|(e, a, b)| (e, a.0, b.map(|b| b.0)))
            .collect();
        got.sort_by_key(|row| row.1);
        assert_eq!(got, vec![(with_b, 1, Some(10)), (without_b, 2, None)]);
        assert_eq!(
            world.query_data::<(&A, Option<&B>)>().count(),
            world.query_data::<&A>().count()
        );
    }

    #[test]
    fn has_term_reports_archetype_presence_per_row() {
        let mut world = World::new();
        let tagged = make(&mut world, 1, true, false, false);
        world.insert_static(tagged, Tag);
        let plain = make(&mut world, 2, true, false, false);

        let mut got: Vec<_> = world
            .query_data::<(Entity, &A, Has<Tag>)>()
            .map(|(e, a, tag)| (e, a.0, tag))
            .collect();
        got.sort_by_key(|row| row.1);
        assert_eq!(got, vec![(tagged, 1, true), (plain, 2, false)]);
    }

    #[test]
    fn zero_sized_components_work_as_ref_and_has_terms() {
        let mut world = World::new();
        let e = make(&mut world, 4, true, false, false);
        world.insert_static(e, Tag);
        make(&mut world, 5, true, false, false);

        let got: Vec<_> = world
            .query_data::<(Entity, &Tag, &A, Has<Tag>)>()
            .map(|(e, tag, a, has)| (e, *tag, a.0, has))
            .collect();
        assert_eq!(got, vec![(e, Tag, 4, true)]);
    }

    #[test]
    fn entity_only_query_enumerates_every_live_entity_componentless_ones_included() {
        let mut world = World::new();
        let bare = world.spawn();
        let with_a = make(&mut world, 1, true, false, false);
        let gone = make(&mut world, 2, true, true, false);
        world.despawn(gone);

        let mut got: Vec<_> = world.query_data::<Entity>().collect();
        got.sort_by_key(|e| e.0.index());
        let mut want = vec![bare, with_a];
        want.sort_by_key(|e| e.0.index());
        assert_eq!(got, want);

        // The componentless entity shows up with None / false in a
        // query that has no `&T` term.
        let seen: Vec<_> = world
            .query_data::<(Entity, Option<&A>, Has<A>)>()
            .map(|(e, a, has)| (e, a.map(|a| a.0), has))
            .collect();
        assert!(seen.contains(&(bare, None, false)));
        assert!(seen.contains(&(with_a, Some(1), true)));
        assert_eq!(seen.len(), 2);
    }

    #[test]
    fn a_ref_term_for_a_type_the_core_has_never_seen_makes_the_query_empty() {
        let mut world = World::new();
        make(&mut world, 1, true, true, false);
        assert_eq!(world.query_data::<(&A, &Never)>().count(), 0);
        assert_eq!(world.query_data::<&Never>().count(), 0);
    }

    #[test]
    fn option_and_has_for_an_unseen_type_are_none_and_false_and_register_nothing() {
        let mut world = World::new();
        make(&mut world, 1, true, true, false);
        let got: Vec<_> = world
            .query_data::<(&A, Option<&Never>, Has<Never>)>()
            .map(|(a, n, h)| (a.0, n.is_some(), h))
            .collect();
        assert_eq!(got, vec![(1, false, false)]);
        // Asking never spends a ComponentId on the unseen type.
        assert!(world.archetypes.existing_component_id::<Never>().is_none());
    }

    #[test]
    fn filtered_queries_compose_with_with_without_and_or() {
        let mut world = World::new();
        let ab = make(&mut world, 1, true, true, false);
        let a_only = make(&mut world, 2, true, false, false);
        let ac = make(&mut world, 3, true, false, true);

        let sorted = |mut v: Vec<Entity>| {
            v.sort_by_key(|e| e.0.index());
            v
        };
        let want = |mut v: Vec<Entity>| sorted(std::mem::take(&mut v));

        let no_b = sorted(
            world
                .query_data_filtered::<(Entity, &A), Without<B>>()
                .map(|(e, _)| e)
                .collect(),
        );
        assert_eq!(no_b, want(vec![a_only, ac]));

        let with_c = sorted(
            world
                .query_data_filtered::<(Entity, &A), With<C>>()
                .map(|(e, _)| e)
                .collect(),
        );
        assert_eq!(with_c, vec![ac]);

        let b_or_c = sorted(
            world
                .query_data_filtered::<(Entity, &A), Or<(With<B>, With<C>)>>()
                .map(|(e, _)| e)
                .collect(),
        );
        assert_eq!(b_or_c, want(vec![ab, ac]));
    }

    #[test]
    fn same_items_in_the_same_order_as_query2_static() {
        let mut world = World::new();
        // Several archetypes holding {A, B}: plain, plus C, plus D, and
        // inserted in both orders so the signatures are built differently.
        for n in 0..4 {
            make(&mut world, n, true, true, false);
        }
        for n in 10..14 {
            make(&mut world, n, true, true, true);
        }
        for n in 20..23 {
            let e = world.spawn();
            world.insert_static(e, B(n * 10));
            world.insert_static(e, A(n));
            world.insert_static(e, D(n));
        }
        make(&mut world, 30, true, false, false);

        let general: Vec<_> = world
            .query_data::<(Entity, &A, &B)>()
            .map(|(e, a, b)| (e, a.0, b.0))
            .collect();
        let fixed: Vec<_> = world
            .query2_static::<A, B>()
            .map(|(e, a, b)| (e, a.0, b.0))
            .collect();
        assert_eq!(general.len(), 11);
        assert_eq!(general, fixed);

        let general_ref: Vec<_> = world
            .query_data::<(&A, &B)>()
            .map(|(a, b)| (a.0, b.0))
            .collect();
        let fixed_ref: Vec<_> = world
            .query2_static_ref::<A, B>()
            .map(|(a, b)| (a.0, b.0))
            .collect();
        assert_eq!(general_ref, fixed_ref);

        let general_one: Vec<_> = world
            .query_data::<(Entity, &A)>()
            .map(|(e, a)| (e, a.0))
            .collect();
        let fixed_one: Vec<_> = world.query_static::<A>().map(|(e, a)| (e, a.0)).collect();
        assert_eq!(general_one, fixed_one);
    }

    #[test]
    fn rows_stay_aligned_after_middle_entities_are_despawned() {
        let mut world = World::new();
        let es: Vec<_> = (1..=6)
            .map(|n| make(&mut world, n, true, true, true))
            .collect();
        world.despawn(es[1]);
        world.despawn(es[3]);

        let mut count = 0;
        for (e, a, b, c) in world.query_data::<(Entity, &A, &B, &C)>() {
            let n = a.0;
            assert_eq!(b.0, n * 10);
            assert_eq!(c.0, n * 100);
            assert_eq!(e, es[(n - 1) as usize]);
            count += 1;
        }
        assert_eq!(count, 4);
    }

    #[test]
    fn rows_stay_aligned_after_a_component_migrates_an_entity_out() {
        let mut world = World::new();
        let es: Vec<_> = (1..=4)
            .map(|n| make(&mut world, n, true, true, true))
            .collect();
        world.remove_static::<C>(es[1]);

        let all_three = world.query_data::<(Entity, &A, &B, &C)>().count();
        assert_eq!(all_three, 3);

        let mut seen = 0;
        for (e, a, b, c) in world.query_data::<(Entity, &A, &B, Option<&C>)>() {
            let n = a.0;
            assert_eq!(b.0, n * 10);
            assert_eq!(e, es[(n - 1) as usize]);
            match c {
                Some(c) => assert_eq!(c.0, n * 100),
                None => assert_eq!(e, es[1]),
            }
            seen += 1;
        }
        assert_eq!(seen, 4);
    }

    #[test]
    fn duplicate_ref_terms_return_the_same_row_twice() {
        let mut world = World::new();
        make(&mut world, 1, true, false, false);
        let rows: Vec<_> = world.query_data::<(&A, &A)>().collect();
        assert_eq!(rows.len(), 1);
        assert!(std::ptr::eq(rows[0].0, rows[0].1));
    }

    #[test]
    fn twelve_term_tuples_and_nested_tuples_work() {
        let mut world = World::new();
        let e = make(&mut world, 1, true, true, true);
        world.insert_static(e, D(7));

        let rows: Vec<_> = world
            .query_data::<(
                Entity,
                &A,
                &B,
                &C,
                &D,
                Option<&A>,
                Option<&Never>,
                Has<A>,
                Has<B>,
                Has<C>,
                Has<D>,
                Has<Never>,
            )>()
            .collect();
        assert_eq!(rows.len(), 1);
        let (ent, a, b, c, d, oa, on, ha, hb, hc, hd, hn) = rows[0];
        assert_eq!(ent, e);
        assert_eq!((a.0, b.0, c.0, d.0), (1, 10, 100, 7));
        assert_eq!(oa.map(|a| a.0), Some(1));
        assert!(on.is_none());
        assert!(ha && hb && hc && hd && !hn);

        let nested: Vec<_> = world
            .query_data::<((Entity, &A), (&B, Has<C>))>()
            .map(|((e, a), (b, c))| (e, a.0, b.0, c))
            .collect();
        assert_eq!(nested, vec![(e, 1, 10, true)]);
    }

    #[test]
    fn zero_row_archetypes_left_behind_are_visited_and_yield_nothing() {
        let mut world = World::new();
        // Inserting one component at a time leaves {A} and {A, B} behind
        // with no rows once the entity reaches {A, B, C}.
        let e = world.spawn();
        world.insert_static(e, A(1));
        world.insert_static(e, B(10));
        world.insert_static(e, C(100));

        let rows: Vec<_> = world
            .query_data::<(Entity, &A, Option<&B>, Has<C>)>()
            .map(|(e, a, b, c)| (e, a.0, b.map(|b| b.0), c))
            .collect();
        assert_eq!(rows, vec![(e, 1, Some(10), true)]);
    }

    #[test]
    fn an_archetype_emptied_by_despawn_yields_nothing_and_is_reusable() {
        let mut world = World::new();
        let gone = make(&mut world, 1, true, true, true);
        world.despawn(gone);
        assert_eq!(world.query_data::<(&A, &B, &C)>().count(), 0);
        assert_eq!(world.query_data::<(&A, Option<&B>)>().count(), 0);

        let back = make(&mut world, 2, true, true, true);
        let rows: Vec<_> = world
            .query_data::<(Entity, &A)>()
            .map(|(e, a)| (e, a.0))
            .collect();
        assert_eq!(rows, vec![(back, 2)]);
    }

    #[test]
    fn an_exhausted_iterator_stays_exhausted() {
        let mut world = World::new();
        make(&mut world, 1, true, false, false);
        make(&mut world, 2, true, true, false);
        let mut it = world.query_data::<&A>();
        assert!(it.next().is_some());
        assert!(it.next().is_some());
        assert!(it.next().is_none());
        assert!(it.next().is_none());
    }

    #[test]
    fn large_query_sums_match_what_was_inserted() {
        let mut world = World::new();
        let mut want_a = 0u64;
        let mut want_c = 0u64;
        for n in 0..300u32 {
            let has_c = n % 3 == 0;
            make(&mut world, n, true, true, has_c);
            want_a += u64::from(n);
            if has_c {
                want_c += u64::from(n) * 100;
            }
        }
        let got_a: u64 = world
            .query_data::<(&A, &B)>()
            .map(|(a, _)| u64::from(a.0))
            .sum();
        assert_eq!(got_a, want_a);
        let got_c: u64 = world
            .query_data::<(&A, Option<&C>)>()
            .filter_map(|(_, c)| c.map(|c| u64::from(c.0)))
            .sum();
        assert_eq!(got_c, want_c);
    }

    // ── Mutable terms ────────────────────────────────────────────────

    /// `n` entities holding `A(i)` and `B(i * 10)` for `i` in `1..=n`, in
    /// one archetype, and a tracker that has already seen them: writes
    /// made after this are the only things `query_changed` should report.
    fn tracked(n: u32) -> (World, Vec<Entity>, ChangeTracker) {
        let mut w = World::new();
        let es: Vec<Entity> = (1..=n)
            .map(|i| make(&mut w, i, true, true, false))
            .collect();
        w.increment_change_tick();
        let mut t = ChangeTracker::new();
        t.update(&w);
        w.increment_change_tick();
        (w, es, t)
    }

    fn changed<T: 'static>(w: &World, t: &ChangeTracker) -> Vec<Entity> {
        let mut v: Vec<Entity> = w.query_changed::<T>(t).map(|(e, _)| e).collect();
        v.sort_by_key(|e| e.0.index());
        v
    }

    #[test]
    fn a_mut_term_writes_land_in_the_world() {
        let mut world = World::new();
        let es: Vec<_> = (1..=3)
            .map(|n| make(&mut world, n, true, true, false))
            .collect();
        for (mut a, b) in world.query_data_mut::<(Mut<A>, &B)>() {
            a.0 += b.0;
        }
        let got: Vec<_> = es
            .iter()
            .map(|e| world.get_static::<A>(*e).unwrap().0)
            .collect();
        assert_eq!(got, vec![11, 22, 33]);
        // The read-only term was not touched.
        let b: Vec<_> = es
            .iter()
            .map(|e| world.get_static::<B>(*e).unwrap().0)
            .collect();
        assert_eq!(b, vec![10, 20, 30]);
    }

    #[test]
    fn mut_terms_mix_with_entity_option_and_has_terms() {
        let mut world = World::new();
        let tagged = make(&mut world, 1, true, true, false);
        world.insert_static(tagged, Tag);
        let plain = make(&mut world, 2, true, false, false);

        let mut seen = Vec::new();
        for (e, mut a, b, tag) in world.query_data_mut::<(Entity, Mut<A>, Option<&B>, Has<Tag>)>() {
            seen.push((e, a.0, b.map(|b| b.0), tag));
            a.0 += 1000;
        }
        seen.sort_by_key(|row| row.1);
        assert_eq!(
            seen,
            vec![(tagged, 1, Some(10), true), (plain, 2, None, false)]
        );
        assert_eq!(world.get_static::<A>(tagged), Some(&A(1001)));
        assert_eq!(world.get_static::<A>(plain), Some(&A(1002)));
    }

    #[test]
    fn option_mut_term_writes_where_present_is_none_elsewhere_and_never_narrows() {
        let mut world = World::new();
        let with_b = make(&mut world, 1, true, true, false);
        let without_b = make(&mut world, 2, true, false, false);

        let mut visited = 0;
        let mut none = 0;
        for (a, b) in world.query_data_mut::<(&A, Option<Mut<B>>)>() {
            visited += 1;
            match b {
                Some(mut b) => b.0 += a.0,
                None => none += 1,
            }
        }
        assert_eq!((visited, none), (2, 1));
        assert_eq!(world.get_static::<B>(with_b), Some(&B(11)));
        assert_eq!(world.get_static::<B>(without_b), None);
    }

    #[test]
    fn several_mut_terms_in_one_query_write_independently() {
        let mut world = World::new();
        let es: Vec<_> = (1..=4)
            .map(|n| make(&mut world, n, true, true, true))
            .collect();
        for (mut a, mut b, mut c) in world.query_data_mut::<(Mut<A>, Mut<B>, Mut<C>)>() {
            let n = a.0;
            a.0 = n + 1;
            b.0 = n + 2;
            c.0 = n + 3;
        }
        for (i, e) in es.iter().enumerate() {
            let n = i as u32 + 1;
            assert_eq!(world.get_static::<A>(*e), Some(&A(n + 1)));
            assert_eq!(world.get_static::<B>(*e), Some(&B(n + 2)));
            assert_eq!(world.get_static::<C>(*e), Some(&C(n + 3)));
        }
    }

    #[test]
    fn writes_are_stamped_only_on_the_rows_and_components_written() {
        let (mut w, es, t) = tracked(6);
        for (mut a, b) in w.query_data_mut::<(Mut<A>, &B)>() {
            if a.0 % 2 == 0 {
                a.0 += b.0;
            }
        }
        w.increment_change_tick();
        assert_eq!(changed::<A>(&w, &t), vec![es[1], es[3], es[5]]);
        assert!(changed::<B>(&w, &t).is_empty());
    }

    #[test]
    fn reading_through_a_mut_term_does_not_mark_changed() {
        let (mut w, _, t) = tracked(5);
        let mut sum = 0;
        for (a, _) in w.query_data_mut::<(Mut<A>, &B)>() {
            sum += a.0; // Deref only
        }
        assert_eq!(sum, 15);
        w.increment_change_tick();
        assert!(changed::<A>(&w, &t).is_empty());
    }

    #[test]
    fn mut_query_matches_query_static_mut_in_rows_values_and_stamps() {
        let (mut left, _, left_t) = tracked(7);
        let (mut right, _, right_t) = tracked(7);
        for (_, mut a) in left.query_static_mut::<A>() {
            if a.0 % 3 == 0 {
                a.0 += 100;
            }
        }
        for (_, mut a) in right.query_data_mut::<(Entity, Mut<A>)>() {
            if a.0 % 3 == 0 {
                a.0 += 100;
            }
        }
        left.increment_change_tick();
        right.increment_change_tick();
        let values =
            |w: &World| -> Vec<_> { w.query_static::<A>().map(|(e, a)| (e, a.0)).collect() };
        assert_eq!(values(&left), values(&right));
        assert_eq!(changed::<A>(&left, &left_t), changed::<A>(&right, &right_t));
        assert_eq!(changed::<A>(&right, &right_t).len(), 2);
    }

    #[test]
    fn mut_rows_stay_aligned_after_despawn_and_migration() {
        let mut world = World::new();
        let es: Vec<_> = (1..=6)
            .map(|n| make(&mut world, n, true, true, true))
            .collect();
        world.despawn(es[1]);
        world.despawn(es[4]);
        world.remove_static::<C>(es[2]);

        let mut seen = 0;
        for (e, mut a, mut b, c) in world.query_data_mut::<(Entity, Mut<A>, Mut<B>, Option<&C>)>() {
            let n = a.0;
            assert_eq!(b.0, n * 10);
            assert_eq!(e, es[(n - 1) as usize]);
            match c {
                Some(c) => assert_eq!(c.0, n * 100),
                None => assert_eq!(e, es[2]),
            }
            a.0 = n + 1000;
            b.0 = n + 2000;
            seen += 1;
        }
        assert_eq!(seen, 4);
        for (i, e) in es.iter().enumerate() {
            if i == 1 || i == 4 {
                continue;
            }
            let n = i as u32 + 1;
            assert_eq!(world.get_static::<A>(*e), Some(&A(n + 1000)));
            assert_eq!(world.get_static::<B>(*e), Some(&B(n + 2000)));
        }
    }

    #[test]
    fn mut_terms_work_across_several_archetypes_in_one_query() {
        let mut world = World::new();
        // {A, B}, {A, B, C} and {A, B, D}: each archetype is entered in
        // turn with the same two terms and must start from a clean slate.
        let ab = make(&mut world, 1, true, true, false);
        let abc = make(&mut world, 2, true, true, true);
        let abd = make(&mut world, 3, true, true, false);
        world.insert_static(abd, D(9));
        let ab2 = make(&mut world, 4, true, true, false);

        let mut rows = 0;
        for (mut a, b) in world.query_data_mut::<(Mut<A>, &B)>() {
            a.0 += b.0;
            rows += 1;
        }
        assert_eq!(rows, 4);
        for (e, n) in [(ab, 1), (abc, 2), (abd, 3), (ab2, 4)] {
            assert_eq!(world.get_static::<A>(e), Some(&A(n + n * 10)));
        }
    }

    #[test]
    fn zero_sized_components_work_as_mut_terms_and_stamp_on_write() {
        let (mut w, es, _) = tracked(3);
        w.insert_static(es[0], Tag);
        w.insert_static(es[2], Tag);
        w.increment_change_tick();
        let mut t = ChangeTracker::new();
        t.update(&w);
        w.increment_change_tick();

        let mut rows = 0;
        for (e, mut tag, a) in w.query_data_mut::<(Entity, Mut<Tag>, &A)>() {
            rows += 1;
            assert!(e == es[0] || e == es[2]);
            if a.0 == 3 {
                *tag = Tag; // a write, even of the same value, marks changed
            }
        }
        assert_eq!(rows, 2);
        w.increment_change_tick();
        assert_eq!(changed::<Tag>(&w, &t), vec![es[2]]);
    }

    #[test]
    fn mut_terms_for_a_type_the_core_has_never_seen() {
        let mut world = World::new();
        make(&mut world, 1, true, true, false);
        assert_eq!(world.query_data_mut::<(&A, Mut<Never>)>().count(), 0);
        let none: Vec<_> = world
            .query_data_mut::<(Mut<A>, Option<Mut<Never>>)>()
            .map(|(_, never)| never.is_none())
            .collect();
        assert_eq!(none, vec![true]);
        assert!(world.archetypes.existing_component_id::<Never>().is_none());
    }

    #[test]
    fn filtered_mut_queries_compose_with_filters() {
        let mut world = World::new();
        let ab = make(&mut world, 1, true, true, false);
        let a_only = make(&mut world, 2, true, false, false);
        let ac = make(&mut world, 3, true, false, true);
        for (mut a,) in world.query_data_mut_filtered::<(Mut<A>,), Without<B>>() {
            a.0 += 500;
        }
        assert_eq!(world.get_static::<A>(ab), Some(&A(1)));
        assert_eq!(world.get_static::<A>(a_only), Some(&A(502)));
        assert_eq!(world.get_static::<A>(ac), Some(&A(503)));

        for (mut a,) in world.query_data_mut_filtered::<(Mut<A>,), With<C>>() {
            a.0 += 1;
        }
        assert_eq!(world.get_static::<A>(ac), Some(&A(504)));
        assert_eq!(world.get_static::<A>(a_only), Some(&A(502)));
    }

    #[test]
    fn mut_query_visits_rows_in_the_same_order_as_the_read_only_query() {
        let mut world = World::new();
        for n in 0..4 {
            make(&mut world, n, true, true, false);
        }
        for n in 10..14 {
            make(&mut world, n, true, true, true);
        }
        let shared: Vec<_> = world
            .query_data::<(Entity, &A, &B)>()
            .map(|(e, a, b)| (e, a.0, b.0))
            .collect();
        let exclusive: Vec<_> = world
            .query_data_mut::<(Entity, Mut<A>, &B)>()
            .map(|(e, a, b)| (e, a.0, b.0))
            .collect();
        assert_eq!(shared.len(), 8);
        assert_eq!(shared, exclusive);
    }

    #[test]
    fn dropping_a_mut_iterator_early_leaves_the_remaining_rows_untouched() {
        let (mut w, es, t) = tracked(5);
        for (mut a, _) in w.query_data_mut::<(Mut<A>, &B)>().take(2) {
            a.0 += 100;
        }
        w.increment_change_tick();
        assert_eq!(changed::<A>(&w, &t), vec![es[0], es[1]]);
        let values: Vec<_> = es
            .iter()
            .map(|e| w.get_static::<A>(*e).unwrap().0)
            .collect();
        assert_eq!(values, vec![101, 102, 3, 4, 5]);
    }

    #[test]
    fn an_exhausted_mut_iterator_stays_exhausted_and_an_emptied_archetype_is_reusable() {
        let mut world = World::new();
        let gone = make(&mut world, 1, true, true, true);
        world.despawn(gone);
        assert_eq!(world.query_data_mut::<(Mut<A>, Mut<B>)>().count(), 0);
        let back = make(&mut world, 2, true, true, true);
        let mut it = world.query_data_mut::<(Entity, Mut<A>)>();
        assert!(matches!(it.next(), Some((e, _)) if e == back));
        assert!(it.next().is_none());
        assert!(it.next().is_none());
    }

    #[test]
    fn large_mut_query_over_several_archetypes_updates_every_row() {
        let mut world = World::new();
        let mut want = 0u64;
        for n in 0..300u32 {
            make(&mut world, n, true, true, n % 3 == 0);
            want += u64::from(n) + 7;
        }
        for (mut a, _) in world.query_data_mut::<(Mut<A>, &B)>() {
            a.0 += 7;
        }
        let got: u64 = world.query_data::<&A>().map(|a| u64::from(a.0)).sum();
        assert_eq!(got, want);
    }

    #[test]
    fn twelve_mixed_terms_with_four_mut_terms_work() {
        let mut world = World::new();
        let e = make(&mut world, 1, true, true, true);
        world.insert_static(e, D(4));
        world.insert_static(e, Tag);

        for (ent, mut a, mut b, mut c, mut d, ha, hb, hc, hd, ht, tag, hn) in world
            .query_data_mut::<(
                Entity,
                Mut<A>,
                Mut<B>,
                Mut<C>,
                Mut<D>,
                Has<A>,
                Has<B>,
                Has<C>,
                Has<D>,
                Has<Tag>,
                Option<&Tag>,
                Has<Never>,
            )>()
        {
            assert_eq!(ent, e);
            assert!(ha && hb && hc && hd && ht && !hn);
            assert_eq!(tag, Some(&Tag));
            a.0 += 1;
            b.0 += 1;
            c.0 += 1;
            d.0 += 1;
        }
        assert_eq!(world.get_static::<A>(e), Some(&A(2)));
        assert_eq!(world.get_static::<B>(e), Some(&B(11)));
        assert_eq!(world.get_static::<C>(e), Some(&C(101)));
        assert_eq!(world.get_static::<D>(e), Some(&D(5)));
    }

    // ── Access conflicts ─────────────────────────────────────────────

    /// Expands to one `#[should_panic]` test per conflicting shape.
    macro_rules! conflict_tests {
        ($($name:ident: $q:ty),+ $(,)?) => {$(
            #[test]
            #[should_panic(expected = "conflicting access")]
            fn $name() {
                let mut world = World::new();
                make(&mut world, 1, true, true, false);
                let _ = world.query_data_mut::<$q>();
            }
        )+};
    }

    conflict_tests! {
        read_then_write_conflicts: (&A, Mut<A>),
        write_then_read_conflicts: (Mut<A>, &A),
        write_twice_conflicts: (Mut<A>, Mut<A>),
        write_and_optional_read_conflict: (Mut<A>, Option<&A>),
        optional_write_and_read_conflict: (Option<Mut<A>>, &A),
        write_and_optional_write_conflict: (Mut<A>, Option<Mut<A>>),
        conflict_inside_nested_tuples: ((&A,), (Entity, Mut<A>)),
        conflict_for_a_type_the_core_has_never_seen: (&Never, Mut<Never>),
        conflict_with_no_row_of_that_type_anywhere: (Mut<D>, Option<&D>),
    }

    #[test]
    #[should_panic(expected = "conflicting access")]
    fn the_filtered_form_checks_conflicts_too() {
        let mut world = World::new();
        make(&mut world, 1, true, true, false);
        let _ = world.query_data_mut_filtered::<(&A, Mut<A>), With<B>>();
    }

    #[test]
    fn the_conflict_message_names_the_type() {
        let result = std::panic::catch_unwind(|| {
            let mut world = World::new();
            make(&mut world, 1, true, true, false);
            let _ = world.query_data_mut::<(&A, Mut<A>)>();
        });
        let message = result
            .err()
            .and_then(|p| p.downcast_ref::<String>().cloned())
            .unwrap_or_default();
        assert!(message.contains("conflicting access"), "{message}");
        assert!(message.contains("::A"), "{message}");
    }

    #[test]
    fn terms_that_do_not_conflict_are_accepted() {
        let mut world = World::new();
        let e = make(&mut world, 1, true, true, true);
        // Any number of readers of one component.
        assert_eq!(world.query_data_mut::<(&A, &A, Option<&A>)>().count(), 1);
        // Has and Entity read no data, so they never conflict with a write.
        assert_eq!(
            world.query_data_mut::<(Mut<A>, Has<A>, Entity)>().count(),
            1
        );
        // Distinct components in any mix of reads and writes.
        assert_eq!(
            world
                .query_data_mut::<(Mut<A>, &B, Option<Mut<C>>, Has<B>)>()
                .count(),
            1
        );
        let _ = e;
    }

    #[test]
    fn distinct_types_the_core_has_never_seen_do_not_conflict_with_each_other() {
        struct AlsoNever;
        let mut world = World::new();
        make(&mut world, 1, true, true, false);
        assert_eq!(
            world.query_data_mut::<(&Never, Mut<AlsoNever>)>().count(),
            0
        );
    }

    // ── Ref<T>, Added<T>, Changed<T> ─────────────────────────────────

    fn key(e: &Entity) -> (u32, u32) {
        (e.index(), e.generation())
    }

    /// `old` (inserted at tick 0, then written at tick 2), `new` (inserted
    /// at tick 2), and a tracker last checked at tick 1.
    fn old_and_new() -> (World, Entity, Entity, ChangeTracker) {
        let mut world = World::new();
        let old = make(&mut world, 1, true, false, false);
        world.increment_change_tick();
        let mut tracker = ChangeTracker::new();
        tracker.update(&world);
        world.increment_change_tick();
        let new = make(&mut world, 2, true, false, false);
        world.get_static_mut::<A>(old).unwrap().0 = 11;
        (world, old, new, tracker)
    }

    #[test]
    fn ref_yields_the_value_and_per_row_tick_answers() {
        let (world, old, new, tracker) = old_and_new();
        let mut rows: Vec<(Entity, u32, bool, bool)> = world
            .query_data::<(Entity, Ref<A>)>()
            .map(|(e, r)| (e, r.0, r.is_added(&tracker), r.is_changed(&tracker)))
            .collect();
        rows.sort_by_key(|r| key(&r.0));
        assert_eq!(rows, vec![(old, 11, false, true), (new, 2, true, true)]);

        // A tracker that has never looked (last-checked tick 0) sees every
        // value as added and changed (the world starts at tick 1). The
        // answer depends on the tick the `Ref` carries, not only on the two
        // stored ticks: measured against a wrong "now" the wrapping
        // comparison says no.
        let fresh = ChangeTracker::new();
        let mut rows: Vec<(Entity, bool, bool)> = world
            .query_data::<(Entity, Ref<A>)>()
            .map(|(e, r)| (e, r.is_added(&fresh), r.is_changed(&fresh)))
            .collect();
        rows.sort_by_key(|r| key(&r.0));
        assert_eq!(rows, vec![(old, true, true), (new, true, true)]);
    }

    #[test]
    fn ref_and_a_plain_read_of_the_same_component_share_one_plan_entry() {
        // `&B` first, `Ref<B>` second: the plan entry made for the plain
        // read must pick up the tick-slice request of the later term.
        let mut world = World::new();
        for n in 1..=4 {
            make(&mut world, n, true, true, false);
        }
        let mut rows: Vec<(u32, u32, u32)> = world
            .query_data_mut::<(Mut<A>, &B, Ref<B>)>()
            .map(|(mut a, b, r)| {
                a.0 += 1;
                (a.0, b.0, r.0)
            })
            .collect();
        rows.sort();
        assert_eq!(
            rows,
            vec![(2, 10, 10), (3, 20, 20), (4, 30, 30), (5, 40, 40)]
        );
    }

    #[test]
    fn ref_terms_narrow_like_ref_terms_and_option_ref_does_not() {
        let mut world = World::new();
        make(&mut world, 1, true, true, false);
        make(&mut world, 2, true, false, false);
        make(&mut world, 3, false, true, false);
        assert_eq!(world.query_data::<Ref<A>>().count(), 2);
        assert_eq!(world.query_data::<(Ref<A>, Ref<B>)>().count(), 1);
        let rows: Vec<(u32, Option<u32>)> = world
            .query_data::<(&A, Option<Ref<B>>)>()
            .map(|(a, b)| (a.0, b.map(|b| b.0)))
            .collect();
        let mut rows = rows;
        rows.sort();
        assert_eq!(rows, vec![(1, Some(10)), (2, None)]);
        assert_eq!(world.query_data::<Ref<Never>>().count(), 0);
        assert!(world
            .query_data::<(Entity, Option<Ref<Never>>)>()
            .all(|(_, r)| r.is_none()));
    }

    #[test]
    fn ref_carries_the_tick_of_the_query_it_came_from() {
        // `Ref::is_changed` measures against the world tick at query
        // creation, exactly as `Mut` does for the write it stamps.
        let (mut world, _, _, tracker) = old_and_new();
        world.increment_change_tick();
        let seen: Vec<bool> = world
            .query_data::<Ref<A>>()
            .map(|r| r.is_changed(&tracker))
            .collect();
        assert_eq!(seen.len(), 2);
        assert!(
            seen.iter().all(|&changed| changed),
            "both are newer than tick 1"
        );
        // After the tracker catches up, nothing is.
        let mut caught_up = tracker;
        caught_up.update(&world);
        world.increment_change_tick();
        assert!(world
            .query_data::<Ref<A>>()
            .all(|r| !r.is_changed(&caught_up) && !r.is_added(&caught_up)));
    }

    #[test]
    fn ref_reads_in_an_exclusive_query_while_another_component_is_written() {
        let mut world = World::new();
        for n in 0..5 {
            make(&mut world, n, true, true, false);
        }
        world.increment_change_tick();
        let mut tracker = ChangeTracker::new();
        tracker.update(&world);
        world.increment_change_tick();
        for (a, mut b) in world.query_data_mut::<(Ref<A>, Mut<B>)>() {
            b.0 += a.0;
        }
        let mut got: Vec<(u32, u32)> = world
            .query_data::<(&A, &B)>()
            .map(|(a, b)| (a.0, b.0))
            .collect();
        got.sort();
        assert_eq!(got, vec![(0, 0), (1, 11), (2, 22), (3, 33), (4, 44)]);
        world.increment_change_tick();
        assert_eq!(
            world.query_changed::<A>(&tracker).count(),
            0,
            "reading A through Ref stamps nothing"
        );
        assert_eq!(
            world.query_changed::<B>(&tracker).count(),
            5,
            "every row was written through its Mut, even `b.0 += 0`"
        );
    }

    #[test]
    #[should_panic(expected = "conflicting access")]
    fn ref_and_mut_of_the_same_component_conflict() {
        let mut world = World::new();
        make(&mut world, 1, true, false, false);
        let _ = world.query_data_mut::<(Ref<A>, Mut<A>)>();
    }

    #[test]
    #[should_panic(expected = "conflicting access")]
    fn option_ref_and_mut_of_the_same_component_conflict() {
        let mut world = World::new();
        let _ = world.query_data_mut::<(Option<Ref<A>>, Mut<A>)>();
    }

    #[test]
    fn added_and_changed_terms_yield_a_bool_per_row() {
        let (world, old, new, tracker) = old_and_new();
        let mut rows: Vec<(Entity, bool, bool)> = world
            .query_data_since::<(Entity, Added<A>, Changed<A>)>(&tracker)
            .collect();
        rows.sort_by_key(|r| key(&r.0));
        assert_eq!(rows, vec![(old, false, true), (new, true, true)]);
    }

    #[test]
    fn tick_terms_narrow_to_archetypes_holding_the_component() {
        let mut world = World::new();
        make(&mut world, 1, true, false, false);
        make(&mut world, 2, false, true, false);
        let tracker = ChangeTracker::new();
        assert_eq!(world.query_data_since::<Added<A>>(&tracker).count(), 1);
        assert_eq!(world.query_data_since::<Changed<B>>(&tracker).count(), 1);
        assert_eq!(
            world
                .query_data_since::<(Added<A>, Changed<B>)>(&tracker)
                .count(),
            0
        );
        assert_eq!(world.query_data_since::<Added<Never>>(&tracker).count(), 0);
    }

    #[test]
    fn tick_terms_work_next_to_mut_terms_of_other_components() {
        let (mut world, _, _, tracker) = old_and_new();
        let e_b = world.spawn();
        world.insert_static(e_b, B(5));
        let mut visited = 0;
        // As a term `Added<B>` would conflict with `Mut<B>`; as a filter it
        // does not.
        for (e, mut b) in
            world.query_data_mut_filtered_since::<(Entity, Mut<B>), Added<B>>(&tracker)
        {
            assert_eq!(e, e_b);
            b.0 = 6;
            visited += 1;
        }
        assert_eq!(visited, 1);
        assert_eq!(world.get_static::<B>(e_b).map(|b| b.0), Some(6));
        // A tick term on one component next to an optional write of another.
        let flags: Vec<bool> = world
            .query_data_mut_since::<(Changed<A>, Option<Mut<C>>)>(&tracker)
            .map(|(changed, c)| {
                assert!(c.is_none());
                changed
            })
            .collect();
        assert_eq!(flags.len(), 2);
        assert!(flags.iter().all(|&changed| changed));
    }

    #[test]
    #[should_panic(expected = "conflicting access")]
    fn a_tick_term_conflicts_with_a_mut_of_the_same_component() {
        let mut world = World::new();
        let tracker = ChangeTracker::new();
        let _ = world.query_data_mut_since::<(Mut<A>, Changed<A>)>(&tracker);
    }

    #[test]
    #[should_panic(expected = "ChangeTracker")]
    fn a_tick_term_needs_the_since_form() {
        let world = World::new();
        let _ = world.query_data::<(Entity, Added<A>)>();
    }
}
