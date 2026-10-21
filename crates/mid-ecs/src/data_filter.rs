// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-ecs.md, section "`Ref<T>`, `Added`/`Changed` terms and
// filters"
// ============================================================================
//! Row-level query filters: [`Added`] and [`Changed`], composing with
//! [`With`], [`Without`] and [`Or`] in one filter type.
//!
//! [`QueryFilter`](crate::filter::QueryFilter) decides per *archetype*,
//! once, while a query's matched list is built; the iterators in
//! `archetype/iter.rs` never see it, which is exactly why they pay
//! nothing for it. `Added<T>` and `Changed<T>` cannot work that way: a
//! row's change ticks differ from the next row's in the same archetype.
//! They need a per-row decision, and the frozen `Iter1`/`Iter2` bodies
//! have no place for one. So row-level filtering is a second, separate
//! trait, [`DataFilter`], accepted only by the `query_data_filtered`
//! family (`World::query_data_filtered`, `World::query_data_mut_filtered`
//! and their `_since` forms), whose iterators were written for it.
//! Everything `QueryFilter` accepts, `DataFilter` accepts too, with
//! identical archetype-level behaviour; the reverse is the new part.
//!
//! | filter           | keeps a row if                                          |
//! |------------------|---------------------------------------------------------|
//! | `With<T>`        | the archetype has `T`                                   |
//! | `Without<T>`     | the archetype lacks `T`                                 |
//! | `Added<T>`       | the archetype has `T` **and** this row's `T` was inserted after the tracker's last-checked tick |
//! | `Changed<T>`     | the archetype has `T` **and** this row's `T` was inserted or written after it |
//! | `(F1, F2, ..)`   | every member keeps it                                   |
//! | `Or<(F1, F2, ..)>` | any member keeps it, each member judged on its own    |
//!
//! `Or<(Added<A>, Changed<B>)>` therefore keeps a row of an `{A, B}`
//! archetype if its `A` is new *or* its `B` changed, and a row of an
//! `{A}` archetype if its `A` is new; `Or<(With<C>, Changed<A>)>` keeps
//! every row of an archetype holding a `C` and only the changed `A`s of
//! the others.
//!
//! **How a row filter runs.** For an archetype the query enters, the
//! filter is evaluated once over all of its rows and the survivors are
//! recorded (a reused `Vec<usize>`, so entering an archetype allocates
//! nothing once the first has sized it); the iterator then walks only
//! those rows. Evaluating before the terms resolve their slices is what
//! lets a filter read the ticks of a component the same query writes:
//! `World::query_data_mut_filtered_since::<Mut<Hp>, Changed<Hp>>` is
//! valid, and is the usual way to write it. A filter that is not
//! row-level (`With`, `Without`, `()`, and tuples or `Or`s of only those)
//! costs nothing per row: that path is the code that existed before this
//! file.
//!
//! **The tracker.** `Added` and `Changed` compare against a
//! [`crate::ChangeTracker`], so a query using one is only accepted by the
//! `_since` methods; the plain ones panic when the query is created, since
//! a missing tracker would silently mean "everything is new".
//!
//! A filter names archetype-tracked components, like every filter:
//! `Added<T>`/`Changed<T>` of a type the Archetype Core has never seen
//! keeps nothing, and `Without` of it keeps everything.

use std::any::TypeId;

use crate::access::FilteredAccess;
use crate::archetype::TableAccess;
use crate::component::ComponentId;
use crate::filter::{Or, QueryFilter, With, Without};
use crate::query_data::{Added, Changed};
use crate::tick::{ComponentTicks, Tick};

mod sealed {
    /// Empty on purpose: exists only so `DataFilter` can't be implemented
    /// outside this crate.
    pub trait Sealed {}

    /// Same, for `DataOrGroup`.
    pub trait OrSealed {}
}

/// Something that narrows which archetypes *and rows* a filtered
/// `query_data` call visits. Implemented for [`With`], [`Without`],
/// [`Added`], [`Changed`], `()`, tuples of filters up to eight members
/// (every member must keep the row) and [`Or`] of such a tuple (any
/// member). Sealed: it can't be implemented outside this crate.
///
/// Everything but [`Self::TRACKED`] is machinery for the iterators in
/// `archetype/iter_data.rs` and hidden from the docs.
pub trait DataFilter: sealed::Sealed {
    /// `true` if some member decides per row, which is exactly when some
    /// member compares against a [`crate::ChangeTracker`] (`Added`,
    /// `Changed`). Such a filter needs the `_since` query methods, and
    /// the iterators run their row-selection pass only for it.
    #[doc(hidden)]
    const TRACKED: bool;

    /// What this filter resolves to, once per query, from the type
    /// registry.
    #[doc(hidden)]
    type State: Clone + Send + Sync;

    /// What one archetype resolves to: the tick slices (and per-member
    /// archetype verdicts) `matches_row` reads. Borrowed from the table
    /// access for the duration of the selection pass only.
    #[doc(hidden)]
    type Fetch<'a>;

    /// Resolves this filter's component types. Read-only: never
    /// registers a type.
    #[doc(hidden)]
    fn get_state(resolve: &dyn Fn(TypeId) -> Option<ComponentId>) -> Self::State;

    /// Whether an archetype whose signature answers `contains` can have
    /// any row this filter keeps. For a row-level filter this is the
    /// necessary condition (the component must be present), not the whole
    /// answer.
    #[doc(hidden)]
    fn matches_component_set(state: &Self::State, contains: &impl Fn(ComponentId) -> bool) -> bool;

    /// Pushes the id of every component whose change ticks `matches_row`
    /// reads, so the mutable iterator hands that slice out.
    #[doc(hidden)]
    fn collect_tick_reads(state: &Self::State, out: &mut Vec<ComponentId>);

    /// Records this filter's archetype constraints (`with`, `without`)
    /// and the reads it makes (`Added`/`Changed` read a component's
    /// ticks) into `access`, keyed by type, without a world. See
    /// `access.rs`.
    #[doc(hidden)]
    fn update_access(access: &mut FilteredAccess);

    /// Resolves what `matches_row` needs for the archetype `table`
    /// points at.
    #[doc(hidden)]
    fn init_fetch<'a>(state: &Self::State, table: &'a TableAccess<'_>) -> Self::Fetch<'a>;

    /// Whether `row` of the archetype passes. Only meaningful for an
    /// archetype `matches_component_set` accepted.
    #[doc(hidden)]
    fn matches_row(fetch: &Self::Fetch<'_>, row: usize) -> bool;
}

// ── Archetype-level filters: delegate to `QueryFilter` ───────────────

/// `With`, `Without` and `()` already say everything there is to say at
/// archetype level; this reuses their `QueryFilter` impl so the two
/// traits cannot drift apart (a test pins that they select the same
/// archetypes).
macro_rules! impl_data_filter_via_query_filter {
    ($(impl[$($gen:tt)*] $ty:ty, |$access:ident| $update:block;)+) => {
        $(
            impl<$($gen)*> sealed::Sealed for $ty {}

            impl<$($gen)*> DataFilter for $ty {
                const TRACKED: bool = false;
                type State = <$ty as QueryFilter>::State;
                type Fetch<'a> = ();

                #[inline]
                fn get_state(resolve: &dyn Fn(TypeId) -> Option<ComponentId>) -> Self::State {
                    <$ty as QueryFilter>::get_state(resolve)
                }

                #[inline]
                fn matches_component_set(
                    state: &Self::State,
                    contains: &impl Fn(ComponentId) -> bool,
                ) -> bool {
                    <$ty as QueryFilter>::matches_component_set(state, contains)
                }

                #[inline]
                fn collect_tick_reads(_state: &Self::State, _out: &mut Vec<ComponentId>) {}

                #[inline]
                fn update_access($access: &mut FilteredAccess) $update

                #[inline]
                fn init_fetch<'a>(_state: &Self::State, _table: &'a TableAccess<'_>) -> Self::Fetch<'a> {}

                #[inline]
                fn matches_row(_fetch: &Self::Fetch<'_>, _row: usize) -> bool {
                    true
                }
            }
        )+
    };
}

impl_data_filter_via_query_filter! {
    impl[T: 'static] With<T>, |access| { access.add_with::<T>(); };
    impl[T: 'static] Without<T>, |access| { access.add_without::<T>(); };
    impl[] (), |_access| {};
}

// ── Added<T> and Changed<T> ──────────────────────────────────────────

/// One archetype's worth of an `Added`/`Changed` filter.
#[doc(hidden)]
pub struct TickFilterFetch<'a> {
    ticks: &'a [ComponentTicks],
    last_run: Tick,
    this_run: Tick,
}

macro_rules! impl_tick_filter {
    ($name:ident, $check:ident) => {
        impl<T: 'static> sealed::Sealed for $name<T> {}

        impl<T: 'static> DataFilter for $name<T> {
            const TRACKED: bool = true;
            type State = Option<ComponentId>;
            type Fetch<'a> = TickFilterFetch<'a>;

            #[inline]
            fn get_state(resolve: &dyn Fn(TypeId) -> Option<ComponentId>) -> Self::State {
                resolve(TypeId::of::<T>())
            }

            #[inline]
            fn matches_component_set(
                state: &Self::State,
                contains: &impl Fn(ComponentId) -> bool,
            ) -> bool {
                match *state {
                    Some(id) => contains(id),
                    // `T` has never been archetype-tracked, so no row can
                    // have a tick for it.
                    None => false,
                }
            }

            #[inline]
            fn collect_tick_reads(state: &Self::State, out: &mut Vec<ComponentId>) {
                if let Some(id) = *state {
                    out.push(id);
                }
            }

            #[inline]
            fn update_access(access: &mut FilteredAccess) {
                access.add_component_read::<T>();
                access.add_with::<T>();
            }

            #[inline]
            fn init_fetch<'a>(state: &Self::State, table: &'a TableAccess<'_>) -> Self::Fetch<'a> {
                TickFilterFetch {
                    ticks: match *state {
                        Some(id) if table.contains(id) => table.peek_ticks(id).unwrap_or(&[]),
                        _ => &[],
                    },
                    last_run: table.last_run(),
                    this_run: table.this_run(),
                }
            }

            #[inline]
            fn matches_row(fetch: &Self::Fetch<'_>, row: usize) -> bool {
                match fetch.ticks.get(row) {
                    Some(ticks) => ticks.$check(fetch.last_run, fetch.this_run),
                    None => false,
                }
            }
        }
    };
}

impl_tick_filter!(Added, is_added);
impl_tick_filter!(Changed, is_changed);

// ── Tuples ───────────────────────────────────────────────────────────

macro_rules! impl_data_filter_for_tuple {
    ($($F:ident),+) => {
        impl<$($F: DataFilter),+> sealed::Sealed for ($($F,)+) {}

        impl<$($F: DataFilter),+> DataFilter for ($($F,)+) {
            const TRACKED: bool = false $(|| $F::TRACKED)+;
            type State = ($($F::State,)+);
            type Fetch<'a> = ($($F::Fetch<'a>,)+);

            #[inline]
            fn get_state(resolve: &dyn Fn(TypeId) -> Option<ComponentId>) -> Self::State {
                ($($F::get_state(resolve),)+)
            }

            // Same trick `filter.rs` uses: a type parameter and the local
            // that destructures its state share a name, which is fine
            // (types and values are separate namespaces).
            #[inline]
            #[allow(non_snake_case)]
            fn matches_component_set(
                state: &Self::State,
                contains: &impl Fn(ComponentId) -> bool,
            ) -> bool {
                let ($($F,)+) = state;
                true $(&& $F::matches_component_set($F, contains))+
            }

            #[inline]
            #[allow(non_snake_case)]
            fn collect_tick_reads(state: &Self::State, out: &mut Vec<ComponentId>) {
                let ($($F,)+) = state;
                $($F::collect_tick_reads($F, out);)+
            }

            #[inline]
            fn update_access(access: &mut FilteredAccess) {
                $($F::update_access(access);)+
            }

            #[inline]
            #[allow(non_snake_case)]
            fn init_fetch<'a>(state: &Self::State, table: &'a TableAccess<'_>) -> Self::Fetch<'a> {
                let ($($F,)+) = state;
                ($($F::init_fetch($F, table),)+)
            }

            #[inline]
            #[allow(non_snake_case)]
            fn matches_row(fetch: &Self::Fetch<'_>, row: usize) -> bool {
                let ($($F,)+) = fetch;
                true $(&& $F::matches_row($F, row))+
            }
        }
    };
}

impl_data_filter_for_tuple!(A);
impl_data_filter_for_tuple!(A, B);
impl_data_filter_for_tuple!(A, B, C);
impl_data_filter_for_tuple!(A, B, C, D);
impl_data_filter_for_tuple!(A, B, C, D, E);
impl_data_filter_for_tuple!(A, B, C, D, E, F);
impl_data_filter_for_tuple!(A, B, C, D, E, F, G);
impl_data_filter_for_tuple!(A, B, C, D, E, F, G, H);

// ── Or ───────────────────────────────────────────────────────────────

/// The inside of `Or<(...)>` for [`DataFilter`]: a tuple of filters
/// evaluated with OR semantics. Each member is judged on its own: its
/// archetype-level verdict is taken once per archetype (stored beside its
/// fetch), and a row is kept if some member whose archetype verdict is
/// "yes" also keeps that row. Sealed, implemented only for tuples of
/// `DataFilter` up to eight members; callers only ever name `Or<(...)>`.
pub trait DataOrGroup: sealed::OrSealed {
    #[doc(hidden)]
    const TRACKED: bool;
    #[doc(hidden)]
    type State: Clone + Send + Sync;
    #[doc(hidden)]
    type Fetch<'a>;
    #[doc(hidden)]
    fn get_state(resolve: &dyn Fn(TypeId) -> Option<ComponentId>) -> Self::State;
    #[doc(hidden)]
    fn any_matches_component_set(
        state: &Self::State,
        contains: &impl Fn(ComponentId) -> bool,
    ) -> bool;
    #[doc(hidden)]
    fn collect_tick_reads(state: &Self::State, out: &mut Vec<ComponentId>);
    #[doc(hidden)]
    fn update_access(access: &mut FilteredAccess);
    #[doc(hidden)]
    fn init_fetch<'a>(state: &Self::State, table: &'a TableAccess<'_>) -> Self::Fetch<'a>;
    #[doc(hidden)]
    fn any_matches_row(fetch: &Self::Fetch<'_>, row: usize) -> bool;
}

macro_rules! impl_data_or_group_for_tuple {
    ($($F:ident),+) => {
        impl<$($F: DataFilter),+> sealed::OrSealed for ($($F,)+) {}

        impl<$($F: DataFilter),+> DataOrGroup for ($($F,)+) {
            const TRACKED: bool = false $(|| $F::TRACKED)+;
            type State = ($($F::State,)+);
            type Fetch<'a> = ($((bool, $F::Fetch<'a>),)+);

            #[inline]
            fn get_state(resolve: &dyn Fn(TypeId) -> Option<ComponentId>) -> Self::State {
                ($($F::get_state(resolve),)+)
            }

            #[inline]
            #[allow(non_snake_case)]
            fn any_matches_component_set(
                state: &Self::State,
                contains: &impl Fn(ComponentId) -> bool,
            ) -> bool {
                let ($($F,)+) = state;
                false $(|| $F::matches_component_set($F, contains))+
            }

            #[inline]
            #[allow(non_snake_case)]
            fn collect_tick_reads(state: &Self::State, out: &mut Vec<ComponentId>) {
                let ($($F,)+) = state;
                $($F::collect_tick_reads($F, out);)+
            }

            // Every arm's reads count; none of its constraints do (only
            // one arm needs to hold, and dropping a constraint can only
            // make two accesses look more likely to overlap).
            #[inline]
            fn update_access(access: &mut FilteredAccess) {
                $(
                    let mut arm = FilteredAccess::new();
                    $F::update_access(&mut arm);
                    access.extend_access(&arm);
                )+
            }

            #[inline]
            #[allow(non_snake_case)]
            fn init_fetch<'a>(state: &Self::State, table: &'a TableAccess<'_>) -> Self::Fetch<'a> {
                let ($($F,)+) = state;
                ($(
                    (
                        $F::matches_component_set($F, &|id| table.contains(id)),
                        $F::init_fetch($F, table),
                    ),
                )+)
            }

            #[inline]
            #[allow(non_snake_case)]
            fn any_matches_row(fetch: &Self::Fetch<'_>, row: usize) -> bool {
                let ($($F,)+) = fetch;
                false $(|| ($F.0 && <$F as DataFilter>::matches_row(&$F.1, row)))+
            }
        }
    };
}

impl_data_or_group_for_tuple!(A);
impl_data_or_group_for_tuple!(A, B);
impl_data_or_group_for_tuple!(A, B, C);
impl_data_or_group_for_tuple!(A, B, C, D);
impl_data_or_group_for_tuple!(A, B, C, D, E);
impl_data_or_group_for_tuple!(A, B, C, D, E, F);
impl_data_or_group_for_tuple!(A, B, C, D, E, F, G);
impl_data_or_group_for_tuple!(A, B, C, D, E, F, G, H);

impl<T: DataOrGroup> sealed::Sealed for Or<T> {}

impl<T: DataOrGroup> DataFilter for Or<T> {
    const TRACKED: bool = T::TRACKED;
    type State = T::State;
    type Fetch<'a> = T::Fetch<'a>;

    #[inline]
    fn get_state(resolve: &dyn Fn(TypeId) -> Option<ComponentId>) -> Self::State {
        T::get_state(resolve)
    }

    #[inline]
    fn matches_component_set(state: &Self::State, contains: &impl Fn(ComponentId) -> bool) -> bool {
        T::any_matches_component_set(state, contains)
    }

    #[inline]
    fn collect_tick_reads(state: &Self::State, out: &mut Vec<ComponentId>) {
        T::collect_tick_reads(state, out);
    }

    #[inline]
    fn update_access(access: &mut FilteredAccess) {
        T::update_access(access);
    }

    #[inline]
    fn init_fetch<'a>(state: &Self::State, table: &'a TableAccess<'_>) -> Self::Fetch<'a> {
        T::init_fetch(state, table)
    }

    #[inline]
    fn matches_row(fetch: &Self::Fetch<'_>, row: usize) -> bool {
        T::any_matches_row(fetch, row)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;
    use crate::tick::{ChangeTracker, Mut};
    use crate::world::{Entity, World};

    #[derive(Debug, Clone, Copy, PartialEq)]
    struct A(u32);
    #[derive(Debug, Clone, Copy, PartialEq)]
    struct B(u32);
    #[derive(Debug, Clone, Copy, PartialEq)]
    struct C(u32);
    /// Never inserted anywhere.
    struct Never;

    fn key(e: &Entity) -> (u32, u32) {
        (e.index(), e.generation())
    }

    fn sorted(mut v: Vec<Entity>) -> Vec<Entity> {
        v.sort_by_key(key);
        v
    }

    /// Everything inserted so far becomes "seen"; the world then moves on
    /// two ticks, so a write or insert made after this call is stamped
    /// strictly after the tracker's last-checked tick.
    fn settle(world: &mut World, tracker: &mut ChangeTracker) {
        world.increment_change_tick();
        tracker.update(world);
        world.increment_change_tick();
    }

    /// Six entities, `A(i)` on all of them and `B(i * 10)` on the even
    /// ones: archetypes `{A}` and `{A, B}`.
    fn ab_world() -> (World, Vec<Entity>, ChangeTracker) {
        let mut world = World::new();
        let es: Vec<Entity> = (0..6)
            .map(|i| {
                let e = world.spawn();
                world.insert_static(e, A(i));
                if i % 2 == 0 {
                    world.insert_static(e, B(i * 10));
                }
                e
            })
            .collect();
        let mut tracker = ChangeTracker::new();
        settle(&mut world, &mut tracker);
        (world, es, tracker)
    }

    fn write_a(world: &mut World, e: Entity, value: u32) {
        world.get_static_mut::<A>(e).expect("has A").0 = value;
    }

    fn write_b(world: &mut World, e: Entity, value: u32) {
        world.get_static_mut::<B>(e).expect("has B").0 = value;
    }

    fn entities<F: DataFilter + 'static>(world: &World, tracker: &ChangeTracker) -> Vec<Entity> {
        sorted(
            world
                .query_data_filtered_since::<Entity, F>(tracker)
                .collect(),
        )
    }

    #[test]
    fn changed_keeps_only_the_written_rows_across_archetypes() {
        let (mut world, es, tracker) = ab_world();
        // es[1], es[5] live in {A}; es[2] in {A, B}.
        for &i in &[1usize, 2, 5] {
            write_a(&mut world, es[i], 100 + i as u32);
        }
        let got: Vec<(Entity, u32)> = world
            .query_data_filtered_since::<(Entity, &A), Changed<A>>(&tracker)
            .map(|(e, a)| (e, a.0))
            .collect();
        let mut got_sorted = got.clone();
        got_sorted.sort_by_key(|(e, _)| key(e));
        assert_eq!(
            got_sorted,
            vec![(es[1], 101), (es[2], 102), (es[5], 105)],
            "exactly the written rows, each with its own value"
        );
        assert_eq!(got.len(), 3, "and each exactly once");
        assert!(entities::<Added<A>>(&world, &tracker).is_empty());
    }

    #[test]
    fn added_keeps_only_values_inserted_after_the_tracker() {
        let (mut world, es, tracker) = ab_world();
        let fresh = world.spawn();
        world.insert_static(fresh, A(77));
        // A migration: es[1] gains a B, keeping its old A.
        world.insert_static(es[1], B(5));
        assert_eq!(entities::<Added<A>>(&world, &tracker), vec![fresh]);
        assert_eq!(
            entities::<Added<B>>(&world, &tracker),
            vec![es[1]],
            "only the new B, not the even entities' older ones"
        );
        // An insert is also a change.
        assert_eq!(entities::<Changed<A>>(&world, &tracker), vec![fresh]);
        assert_eq!(
            entities::<Changed<B>>(&world, &tracker),
            vec![es[1]],
            "a migrated entity keeps its A's old ticks"
        );
    }

    #[test]
    fn row_filters_compose_with_with_and_without_in_one_call() {
        let (mut world, es, tracker) = ab_world();
        for &i in &[0usize, 1, 2, 3] {
            write_a(&mut world, es[i], 9);
        }
        // 0 and 2 have a B, 1 and 3 do not.
        assert_eq!(
            entities::<(Changed<A>, With<B>)>(&world, &tracker),
            vec![es[0], es[2]]
        );
        assert_eq!(
            entities::<(Changed<A>, Without<B>)>(&world, &tracker),
            vec![es[1], es[3]]
        );
        assert_eq!(
            entities::<(With<B>, Changed<A>, Without<C>)>(&world, &tracker),
            vec![es[0], es[2]]
        );
        assert!(entities::<(Changed<A>, With<C>)>(&world, &tracker).is_empty());
    }

    #[test]
    fn or_judges_each_member_on_its_own_per_row() {
        let (mut world, es, tracker) = ab_world();
        // {A, B} entities: 0, 2, 4. {A}-only: 1, 3, 5.
        write_a(&mut world, es[0], 1); // A changed, B not
        write_b(&mut world, es[2], 1); // B changed, A not
                                       // es[4]: neither. {A}-only: es[1] A changed, es[3], es[5] untouched.
        write_a(&mut world, es[1], 1);

        assert_eq!(
            entities::<Or<(Changed<A>, Changed<B>)>>(&world, &tracker),
            vec![es[0], es[1], es[2]],
            "es[4] shares an archetype with es[0] and es[2] but neither of its values changed"
        );
        assert_eq!(
            entities::<Or<(Changed<B>, Without<B>)>>(&world, &tracker),
            sorted(vec![es[2], es[1], es[3], es[5]]),
            "Without<B> keeps every {{A}}-only row whole; Changed<B> only the written B"
        );
        assert_eq!(
            entities::<Or<(Changed<A>, With<B>)>>(&world, &tracker),
            sorted(vec![es[0], es[2], es[4], es[1]]),
            "With<B> keeps every {{A, B}} row whole; Changed<A> adds the written {{A}}-only one"
        );
        // An Or nested in a tuple with a plain filter.
        assert_eq!(
            entities::<(Or<(Changed<A>, Changed<B>)>, Without<B>)>(&world, &tracker),
            vec![es[1]]
        );
    }

    #[test]
    fn a_plain_or_is_not_row_level_and_still_works_here() {
        let (world, es, tracker) = ab_world();
        assert!(!<Or<(With<A>, With<B>)> as DataFilter>::TRACKED);
        assert!(<Or<(With<A>, Changed<B>)> as DataFilter>::TRACKED);
        assert_eq!(
            entities::<Or<(With<B>, With<C>)>>(&world, &tracker),
            vec![es[0], es[2], es[4]]
        );
    }

    #[test]
    fn mut_and_changed_on_the_same_component_write_only_the_selected_rows() {
        let (mut world, es, tracker) = ab_world();
        let written = [es[1], es[2], es[5]];
        for &e in &written {
            write_a(&mut world, e, 50);
        }
        // A second marker, so the `Mut` loop's own stamping is observable.
        let mut after = ChangeTracker::new();
        settle(&mut world, &mut after);
        for mut a in world.query_data_mut_filtered_since::<Mut<A>, Changed<A>>(&tracker) {
            a.0 += 1000;
        }
        for (i, &e) in es.iter().enumerate() {
            let expected = if written.contains(&e) { 1050 } else { i as u32 };
            assert_eq!(
                world.get_static::<A>(e).map(|a| a.0),
                Some(expected),
                "row {i}: a skipped row must neither be written nor shift the writes of later rows"
            );
        }
        world.increment_change_tick();
        assert_eq!(
            sorted(world.query_changed::<A>(&after).map(|(e, _)| e).collect()),
            sorted(written.to_vec()),
            "only the rows actually written were stamped"
        );
    }

    #[test]
    fn mut_filtered_by_another_components_changes() {
        let (mut world, es, tracker) = ab_world();
        write_b(&mut world, es[2], 1);
        write_b(&mut world, es[4], 1);
        for (_, mut a) in
            world.query_data_mut_filtered_since::<(Entity, Mut<A>), Changed<B>>(&tracker)
        {
            a.0 = 999;
        }
        for (i, &e) in es.iter().enumerate() {
            let expected = if i == 2 || i == 4 { 999 } else { i as u32 };
            assert_eq!(world.get_static::<A>(e).map(|a| a.0), Some(expected));
        }
    }

    /// `Mut<A>` over `{A, B}` and `{A, C}` rows, filtered by changes to
    /// components the query has no term for. The filter's tick slices must
    /// reach the iterator through every composite form (tuple, `Or`, and
    /// a tuple holding an `Or`), not just a bare `Changed<B>`.
    fn a_with_b_or_c() -> (World, Vec<Entity>, Vec<Entity>, ChangeTracker) {
        let mut world = World::new();
        let with_b: Vec<Entity> = (0..4)
            .map(|i| {
                let e = world.spawn();
                world.insert_static(e, A(i));
                world.insert_static(e, B(i));
                e
            })
            .collect();
        let with_c: Vec<Entity> = (10..14)
            .map(|i| {
                let e = world.spawn();
                world.insert_static(e, A(i));
                world.insert_static(e, C(i));
                e
            })
            .collect();
        let mut tracker = ChangeTracker::new();
        settle(&mut world, &mut tracker);
        (world, with_b, with_c, tracker)
    }

    fn a_values(world: &World, es: &[Entity]) -> Vec<u32> {
        es.iter()
            .map(|&e| world.get_static::<A>(e).expect("has A").0)
            .collect()
    }

    #[test]
    fn composite_filters_on_components_the_query_does_not_name() {
        let (mut world, with_b, with_c, tracker) = a_with_b_or_c();
        write_b(&mut world, with_b[1], 99);
        world.get_static_mut::<C>(with_c[2]).unwrap().0 = 99;

        // Tuple.
        for (_, mut a) in
            world.query_data_mut_filtered_since::<(Entity, Mut<A>), (Changed<B>, With<A>)>(&tracker)
        {
            a.0 += 100;
        }
        assert_eq!(a_values(&world, &with_b), vec![0, 101, 2, 3]);
        assert_eq!(a_values(&world, &with_c), vec![10, 11, 12, 13]);

        // Or.
        for (_, mut a) in world
            .query_data_mut_filtered_since::<(Entity, Mut<A>), Or<(Changed<B>, Changed<C>)>>(
                &tracker,
            )
        {
            a.0 += 1000;
        }
        assert_eq!(a_values(&world, &with_b), vec![0, 1101, 2, 3]);
        assert_eq!(a_values(&world, &with_c), vec![10, 11, 1012, 13]);

        // A tuple holding an Or.
        for (_, mut a) in world.query_data_mut_filtered_since::<(Entity, Mut<A>), (
            Or<(Changed<B>, Changed<C>)>,
            Without<B>,
        )>(&tracker)
        {
            a.0 += 10_000;
        }
        assert_eq!(a_values(&world, &with_b), vec![0, 1101, 2, 3]);
        assert_eq!(a_values(&world, &with_c), vec![10, 11, 11_012, 13]);
    }

    #[test]
    fn a_filter_may_read_the_ticks_of_a_component_the_query_also_reads_as_a_term() {
        // The plan already holds `B` (from the `&B` term, ticks not
        // needed) when the filter asks for its ticks: the two must merge.
        let (mut world, es, tracker) = ab_world();
        write_b(&mut world, es[2], 5);
        let mut seen = Vec::new();
        for (e, mut a, b) in
            world.query_data_mut_filtered_since::<(Entity, Mut<A>, &B), Changed<B>>(&tracker)
        {
            a.0 += 100;
            seen.push((e, b.0));
        }
        assert_eq!(seen, vec![(es[2], 5)]);
        assert_eq!(world.get_static::<A>(es[2]).map(|a| a.0), Some(102));
        assert_eq!(world.get_static::<A>(es[0]).map(|a| a.0), Some(0));
    }

    #[test]
    fn filtered_rows_keep_the_unfiltered_order() {
        let (mut world, es, tracker) = ab_world();
        for &i in &[0usize, 3, 4] {
            write_a(&mut world, es[i], 7);
        }
        let all: Vec<Entity> = world.query_data::<(Entity, &A)>().map(|(e, _)| e).collect();
        let changed = sorted(vec![es[0], es[3], es[4]]);
        let expected: Vec<Entity> = all
            .iter()
            .copied()
            .filter(|e| changed.contains(e))
            .collect();
        let got: Vec<Entity> = world
            .query_data_filtered_since::<(Entity, &A), Changed<A>>(&tracker)
            .map(|(e, _)| e)
            .collect();
        assert_eq!(got, expected);
        assert_eq!(got.len(), 3);
    }

    #[test]
    fn plain_methods_refuse_a_query_that_needs_a_tracker() {
        fn refused(f: impl FnOnce() + std::panic::UnwindSafe) -> bool {
            let hook = std::panic::take_hook();
            std::panic::set_hook(Box::new(|_| {}));
            let result = std::panic::catch_unwind(f);
            std::panic::set_hook(hook);
            match result {
                Err(payload) => payload
                    .downcast_ref::<String>()
                    .is_some_and(|m| m.contains("ChangeTracker")),
                Ok(()) => false,
            }
        }
        assert!(refused(|| {
            let world = World::new();
            let _ = world.query_data_filtered::<Entity, Added<A>>();
        }));
        assert!(refused(|| {
            let world = World::new();
            let _ = world.query_data::<(Entity, Changed<A>)>();
        }));
        assert!(refused(|| {
            let mut world = World::new();
            let _ = world.query_data_mut_filtered::<Mut<A>, Changed<A>>();
        }));
        assert!(refused(|| {
            let mut world = World::new();
            let _ = world.query_data_mut::<(Mut<A>, Added<B>)>();
        }));
        // The untracked forms of the same calls are fine.
        let world = World::new();
        assert_eq!(world.query_data_filtered::<Entity, With<A>>().count(), 0);
    }

    #[test]
    fn filters_naming_unseen_types() {
        let (world, es, tracker) = ab_world();
        assert!(entities::<Changed<Never>>(&world, &tracker).is_empty());
        assert!(entities::<Added<Never>>(&world, &tracker).is_empty());
        assert!(entities::<(With<A>, Added<Never>)>(&world, &tracker).is_empty());
        assert_eq!(
            entities::<Or<(Changed<Never>, With<B>)>>(&world, &tracker),
            vec![es[0], es[2], es[4]]
        );
        assert_eq!(
            entities::<(Without<Never>, With<B>)>(&world, &tracker),
            vec![es[0], es[2], es[4]]
        );
    }

    /// Both filter traits must pick the same archetypes for everything
    /// they both accept: `DataFilter` reuses `QueryFilter`'s impls and
    /// this pins that it still does.
    #[test]
    fn archetype_level_filters_select_what_query_filter_selects() {
        fn same<F: QueryFilter + DataFilter + 'static>(world: &World) {
            let via_query: Vec<_> = world
                .query_static_filtered::<A, F>()
                .map(|(e, a)| (e, a.0))
                .collect();
            let via_data: Vec<_> = world
                .query_data_filtered::<(Entity, &A), F>()
                .map(|(e, a)| (e, a.0))
                .collect();
            assert_eq!(via_query, via_data);
        }
        let mut world = World::new();
        for i in 0..8u32 {
            let e = world.spawn();
            world.insert_static(e, A(i));
            if i % 2 == 0 {
                world.insert_static(e, B(i));
            }
            if i % 3 == 0 {
                world.insert_static(e, C(i));
            }
        }
        same::<()>(&world);
        same::<With<B>>(&world);
        same::<Without<B>>(&world);
        same::<(With<B>, Without<C>)>(&world);
        same::<Or<(With<B>, With<C>)>>(&world);
        same::<(Or<(With<B>, With<C>)>, Without<B>)>(&world);
        same::<With<Never>>(&world);
        same::<Without<Never>>(&world);
    }

    /// A small deterministic generator: the sequence of operations is
    /// reproducible and needs no dependency.
    struct Lcg(u64);

    impl Lcg {
        fn next(&mut self, bound: usize) -> usize {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 33) as usize) % bound
        }
    }

    fn set_of(v: impl Iterator<Item = Entity>) -> HashSet<Entity> {
        v.collect()
    }

    /// The row filters against the older, independent implementation of
    /// the same questions (`query_added`/`query_changed` and plain
    /// membership checks), over a long random mix of spawns, despawns
    /// (swap-removes), migrations in both directions, writes, tick steps
    /// and tracker updates. A tick column zipped off by one row, a skipped
    /// row that shifts a later `Mut`, or a wrong `Or` verdict shows up as
    /// a set mismatch on some step.
    #[test]
    fn row_filters_match_query_added_and_query_changed_over_a_random_history() {
        let mut rng = Lcg(0x9E3779B97F4A7C15);
        let mut world = World::new();
        let mut tracker = ChangeTracker::new();
        let mut alive: Vec<Entity> = Vec::new();
        let mut partial_steps = 0;
        let mut empty_steps = 0;

        for step in 0..600 {
            for _ in 0..1 + rng.next(4) {
                match rng.next(9) {
                    0 | 1 => {
                        let e = world.spawn();
                        world.insert_static(e, A(step));
                        if rng.next(2) == 0 {
                            world.insert_static(e, B(step));
                        }
                        alive.push(e);
                    }
                    2 if !alive.is_empty() => {
                        let i = rng.next(alive.len());
                        let e = alive.swap_remove(i);
                        world.despawn(e);
                    }
                    3 if !alive.is_empty() => {
                        let e = alive[rng.next(alive.len())];
                        world.insert_static(e, B(step));
                    }
                    4 if !alive.is_empty() => {
                        let e = alive[rng.next(alive.len())];
                        world.remove_static::<B>(e);
                    }
                    5 if !alive.is_empty() => {
                        let e = alive[rng.next(alive.len())];
                        world.remove_static::<A>(e);
                    }
                    6 | 7 if !alive.is_empty() => {
                        let e = alive[rng.next(alive.len())];
                        if let Some(a) = world.get_static_mut::<A>(e) {
                            a.0 += 1;
                        }
                    }
                    8 if !alive.is_empty() => {
                        let e = alive[rng.next(alive.len())];
                        if let Some(b) = world.get_static_mut::<B>(e) {
                            b.0 += 1;
                        }
                    }
                    _ => {}
                }
            }
            if rng.next(3) == 0 {
                world.increment_change_tick();
            }
            if rng.next(4) == 0 {
                tracker.update(&world);
                world.increment_change_tick();
            }

            let changed_a = set_of(world.query_changed::<A>(&tracker).map(|(e, _)| e));
            let added_a = set_of(world.query_added::<A>(&tracker).map(|(e, _)| e));
            let changed_b = set_of(world.query_changed::<B>(&tracker).map(|(e, _)| e));

            let got = |v: Vec<Entity>| -> HashSet<Entity> {
                let n = v.len();
                let s: HashSet<Entity> = v.into_iter().collect();
                assert_eq!(s.len(), n, "step {step}: an entity was yielded twice");
                s
            };
            let ent = |f: &dyn Fn(&World) -> Vec<Entity>| got(f(&world));

            assert_eq!(
                ent(&|w| w
                    .query_data_filtered_since::<Entity, Changed<A>>(&tracker)
                    .collect()),
                changed_a,
                "step {step}: Changed<A>"
            );
            assert_eq!(
                ent(&|w| w
                    .query_data_filtered_since::<Entity, Added<A>>(&tracker)
                    .collect()),
                added_a,
                "step {step}: Added<A>"
            );
            let or_changed: HashSet<Entity> = changed_a.union(&changed_b).copied().collect();
            assert_eq!(
                ent(&|w| w
                    .query_data_filtered_since::<Entity, Or<(Changed<A>, Changed<B>)>>(&tracker)
                    .collect()),
                or_changed,
                "step {step}: Or<(Changed<A>, Changed<B>)>"
            );
            let or_mixed: HashSet<Entity> = added_a
                .iter()
                .copied()
                .chain(
                    alive
                        .iter()
                        .copied()
                        .filter(|&e| world.has_static::<A>(e) && !world.has_static::<B>(e)),
                )
                .collect();
            assert_eq!(
                ent(&|w| w
                    .query_data_filtered_since::<Entity, Or<(Added<A>, (With<A>, Without<B>))>>(
                        &tracker
                    )
                    .collect()),
                or_mixed,
                "step {step}: Or<(Added<A>, (With<A>, Without<B>))>"
            );
            let and_set: HashSet<Entity> = changed_a
                .iter()
                .copied()
                .filter(|&e| world.has_static::<B>(e))
                .collect();
            assert_eq!(
                ent(&|w| w
                    .query_data_filtered_since::<Entity, (Changed<A>, With<B>)>(&tracker)
                    .collect()),
                and_set,
                "step {step}: (Changed<A>, With<B>)"
            );

            // The history must actually exercise the interesting cases:
            // some steps with a strict subset selected, some with none.
            let with_a = alive.iter().filter(|&&e| world.has_static::<A>(e)).count();
            if !changed_a.is_empty() && changed_a.len() < with_a {
                partial_steps += 1;
            }
            if changed_a.is_empty() {
                empty_steps += 1;
            }

            // The same selection through the exclusive iterator, writing
            // to the rows it yields and nothing else.
            if step % 5 == 0 {
                let before: Vec<(Entity, u32)> = world
                    .query_data::<(Entity, &A)>()
                    .map(|(e, a)| (e, a.0))
                    .collect();
                let mut touched = Vec::new();
                for (e, mut a) in
                    world.query_data_mut_filtered_since::<(Entity, Mut<A>), Changed<A>>(&tracker)
                {
                    a.0 += 1_000_000;
                    touched.push(e);
                }
                assert_eq!(got(touched), changed_a, "step {step}: exclusive Changed<A>");
                for (e, old) in before {
                    let now = world.get_static::<A>(e).map(|a| a.0);
                    let want = if changed_a.contains(&e) {
                        old + 1_000_000
                    } else {
                        old
                    };
                    assert_eq!(now, Some(want), "step {step}: value of {e:?}");
                }
            }
        }
        assert!(
            partial_steps > 100,
            "only {partial_steps} steps selected a strict subset of the rows"
        );
        assert!(
            empty_steps > 10,
            "only {empty_steps} steps selected nothing"
        );
    }
}
