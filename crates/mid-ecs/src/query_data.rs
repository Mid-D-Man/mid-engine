// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-ecs.md, section "Multi-term queries: `query_data.rs`"
// ============================================================================
//! Multi-term, read-only query data for the Archetype Core: [`Entity`],
//! `&T`, `Option<&T>`, [`Has<T>`], and tuples of them (up to twelve).
//!
//! Where `query_static` and `query2_static` fix their item shape in the
//! method name, `World::query_data::<Q>()` takes the shape as a type:
//! `Q = (Entity, &A, &B, &C)` is a three-component query,
//! `Q = (&A, Option<&B>, Has<C>)` visits every archetype holding an `A`
//! and reports whether that archetype also holds a `B` (and gives it) or a
//! `C`. The terms:
//!
//! | term          | item        | archetype must contain `T`? |
//! |---------------|-------------|-----------------------------|
//! | `Entity`      | `Entity`    | no                          |
//! | `&T`          | `&T`        | **yes**                     |
//! | `Option<&T>`  | `Option<&T>`| no: `None` where it is absent |
//! | `Has<T>`      | `bool`      | no: `false` where it is absent |
//!
//! Only `&T` terms narrow which archetypes are visited. `Option<&T>` and
//! `Has<T>` are decided once per archetype, not per row, and every term
//! resolves its column slice once when the iterator enters an archetype,
//! so a row costs slice indexing and nothing else.
//!
//! A query with no `&T` term at all (`Entity`, or `(Entity, Has<T>)`)
//! visits every archetype, the empty one included, so it enumerates every
//! live entity.
//!
//! `T` must have been inserted with `insert_static`, `insert_bundle` or
//! `spawn_bundle` for the Archetype Core to know it. For an unknown type
//! `&T` makes the whole query empty, `Option<&T>` is always `None` and
//! `Has<T>` is always `false`; asking never registers anything, the same
//! contract `filter.rs` documents.
//!
//! Read-only on purpose: mutable terms need disjoint column borrows and
//! change-tick stamping and get their own pass. This module is also
//! separate from `archetype/iter.rs`, whose `Iter1`/`Iter2` bodies are
//! held fixed, so nothing here touches the existing query paths.

use std::any::TypeId;
use std::marker::PhantomData;

use crate::archetype::TableView;
use crate::component::ComponentId;
use crate::world::Entity;

mod sealed {
    /// Empty on purpose: exists only so `QueryData` can't be implemented
    /// outside this crate.
    pub trait Sealed {}
}

/// One term of a query, or a tuple of terms: what a `World::query_data`
/// call yields per row. Implemented for [`Entity`], `&T`, `Option<&T>`,
/// [`Has<T>`] and tuples of those up to twelve members (tuples nest).
/// Sealed: it can't be implemented outside this crate.
///
/// Everything but [`Self::Item`] is machinery for the iterator in
/// `archetype/iter_data.rs` and hidden from the docs.
pub trait QueryData: sealed::Sealed {
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

    /// `false` if some `&T` term names a type the Archetype Core has
    /// never seen, in which case nothing can match.
    #[doc(hidden)]
    fn is_satisfiable(state: &Self::State) -> bool;

    /// Pushes the id of every `&T` term: the components an archetype
    /// must contain to be visited. `Option<&T>` and `Has<T>` push
    /// nothing.
    #[doc(hidden)]
    fn collect_required(state: &Self::State, out: &mut Vec<ComponentId>);

    /// Resolves the slices and flags for one archetype's table.
    #[doc(hidden)]
    fn init_fetch<'w>(state: &Self::State, view: &TableView<'w>) -> Self::Fetch<'w>;

    /// The number of rows of this archetype the terms can all serve:
    /// `len` (the table's entity count) clamped to every column slice
    /// the terms hold, so indexing by a row below the result cannot go
    /// out of bounds.
    #[doc(hidden)]
    fn clamp_len(fetch: &Self::Fetch<'_>, len: usize) -> usize;

    /// The item for `row`, which the caller keeps below
    /// [`Self::clamp_len`].
    #[doc(hidden)]
    fn fetch<'w>(fetch: &Self::Fetch<'w>, row: usize) -> Self::Item<'w>;
}

/// Yields whether the visited archetype contains `T`, as a `bool`.
/// Never constructed; a type-level marker only. Unlike `With<T>` it does
/// not narrow the query: rows without `T` are visited and report
/// `false`.
pub struct Has<T>(PhantomData<fn() -> T>);

// ── Entity ───────────────────────────────────────────────────────────

impl sealed::Sealed for Entity {}

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
    fn init_fetch<'w>(_state: &Self::State, view: &TableView<'w>) -> Self::Fetch<'w> {
        view.entities()
    }

    #[inline]
    fn clamp_len(fetch: &Self::Fetch<'_>, len: usize) -> usize {
        len.min(fetch.len())
    }

    #[inline]
    fn fetch<'w>(fetch: &Self::Fetch<'w>, row: usize) -> Self::Item<'w> {
        fetch[row]
    }
}

// ── &T ───────────────────────────────────────────────────────────────

impl<T: 'static> sealed::Sealed for &T {}

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
    fn init_fetch<'w>(state: &Self::State, view: &TableView<'w>) -> Self::Fetch<'w> {
        // A signature can list a component whose column the table has
        // not created yet (it is made lazily, on the first row). That
        // table has no rows, so an empty slice is the right answer.
        match *state {
            Some(id) => view.column::<T>(id).unwrap_or(&[]),
            None => &[],
        }
    }

    #[inline]
    fn clamp_len(fetch: &Self::Fetch<'_>, len: usize) -> usize {
        len.min(fetch.len())
    }

    #[inline]
    fn fetch<'w>(fetch: &Self::Fetch<'w>, row: usize) -> Self::Item<'w> {
        &fetch[row]
    }
}

// ── Option<&T> ───────────────────────────────────────────────────────

impl<T: 'static> sealed::Sealed for Option<&T> {}

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
    fn init_fetch<'w>(state: &Self::State, view: &TableView<'w>) -> Self::Fetch<'w> {
        // Presence is the archetype's signature, not whether the column
        // object exists yet (see the `&T` impl).
        match *state {
            Some(id) if view.contains(id) => Some(view.column::<T>(id).unwrap_or(&[])),
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
    fn fetch<'w>(fetch: &Self::Fetch<'w>, row: usize) -> Self::Item<'w> {
        match *fetch {
            Some(column) => Some(&column[row]),
            None => None,
        }
    }
}

// ── Has<T> ───────────────────────────────────────────────────────────

impl<T: 'static> sealed::Sealed for Has<T> {}

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
    fn init_fetch<'w>(state: &Self::State, view: &TableView<'w>) -> Self::Fetch<'w> {
        matches!(*state, Some(id) if view.contains(id))
    }

    #[inline]
    fn clamp_len(_fetch: &Self::Fetch<'_>, len: usize) -> usize {
        len
    }

    #[inline]
    fn fetch<'w>(fetch: &Self::Fetch<'w>, _row: usize) -> Self::Item<'w> {
        *fetch
    }
}

// ── Tuples ───────────────────────────────────────────────────────────

macro_rules! impl_query_data_for_tuple {
    ($($T:ident),+) => {
        impl<$($T: QueryData),+> sealed::Sealed for ($($T,)+) {}

        impl<$($T: QueryData),+> QueryData for ($($T,)+) {
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
            fn init_fetch<'w>(state: &Self::State, view: &TableView<'w>) -> Self::Fetch<'w> {
                let ($($T,)+) = state;
                ($($T::init_fetch($T, view),)+)
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
            fn fetch<'w>(fetch: &Self::Fetch<'w>, row: usize) -> Self::Item<'w> {
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
}
