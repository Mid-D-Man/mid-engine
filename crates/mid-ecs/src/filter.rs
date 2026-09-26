// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-ecs.md, section "filter.rs"
// ============================================================================
//! Archetype-level query filters: [`With`], [`Without`], [`Or`], and
//! tuples of them.
//!
//! A filter narrows which archetypes a `*_filtered` query on the
//! Archetype Core (`World::query_static_filtered` and its three
//! siblings in `query.rs`) visits. It never adds anything to the query's
//! item. `With<T>` keeps an archetype only if `T` is in its signature,
//! `Without<T>` only if it is not, a tuple `(F1, F2, ...)` only if
//! *every* member does, and `Or<(F1, F2, ...)>` only if *any* member
//! does.
//!
//! The whole filter runs once per archetype while the query's matched
//! list is built. The iterators in `archetype/iter.rs` never see it, so
//! a filtered query pays nothing per row.
//!
//! Filters name archetype-tracked components only (`insert_static`,
//! `insert_bundle`, `spawn_bundle`). A type that only ever lived in the
//! Sparse Shell has no `ComponentId` in the Archetype Core's numbering,
//! so `With` of it matches nothing and `Without` of it matches
//! everything. The same holds for any type nothing has inserted yet, and
//! asking about a type never registers it.

use std::any::TypeId;
use std::marker::PhantomData;

use crate::component::ComponentId;

mod sealed {
    /// Empty on purpose: exists only so `QueryFilter` can't be
    /// implemented outside this crate.
    pub trait Sealed {}

    /// Same, for `OrFilterGroup` -- only tuples of [`super::QueryFilter`]
    /// implement it, via `impl_or_filter_group_for_tuple!`.
    pub trait OrSealed {}
}

/// Something that narrows which archetypes a filtered query visits.
/// Implemented for [`With`], [`Without`], `()`, and tuples of filters
/// up to eight members (every member must match). Sealed: it can't be
/// implemented outside this crate.
///
/// Both methods are called by the query machinery, once per query
/// (`get_state`) and once per candidate archetype
/// (`matches_component_set`) — never per row.
pub trait QueryFilter: sealed::Sealed {
    /// What this filter resolves to, per query, from the type
    /// registry: one `Option<ComponentId>` per leaf filter (`None` =
    /// that type has never been archetype-tracked).
    type State;

    /// Resolves this filter's component types against the Archetype
    /// Core's own `TypeId -> ComponentId` map. `resolve` must be a
    /// read-only lookup (never registers).
    fn get_state(resolve: &dyn Fn(TypeId) -> Option<ComponentId>) -> Self::State;

    /// Whether an archetype whose signature answers `contains` should
    /// be visited.
    fn matches_component_set(state: &Self::State, contains: &impl Fn(ComponentId) -> bool) -> bool;
}

/// Keeps only archetypes that contain `T`. `T`'s *value* is not
/// fetched — to read it, put it in the query's own type parameters
/// instead. Never constructed; a type-level marker only.
pub struct With<T>(PhantomData<fn() -> T>);

impl<T: 'static> sealed::Sealed for With<T> {}

impl<T: 'static> QueryFilter for With<T> {
    type State = Option<ComponentId>;

    #[inline]
    fn get_state(resolve: &dyn Fn(TypeId) -> Option<ComponentId>) -> Self::State {
        resolve(TypeId::of::<T>())
    }

    #[inline]
    fn matches_component_set(state: &Self::State, contains: &impl Fn(ComponentId) -> bool) -> bool {
        match *state {
            Some(id) => contains(id),
            // `T` has never been archetype-tracked, so no archetype
            // can contain it.
            None => false,
        }
    }
}

/// Keeps only archetypes that do *not* contain `T`. Never constructed;
/// a type-level marker only.
pub struct Without<T>(PhantomData<fn() -> T>);

impl<T: 'static> sealed::Sealed for Without<T> {}

impl<T: 'static> QueryFilter for Without<T> {
    type State = Option<ComponentId>;

    #[inline]
    fn get_state(resolve: &dyn Fn(TypeId) -> Option<ComponentId>) -> Self::State {
        resolve(TypeId::of::<T>())
    }

    #[inline]
    fn matches_component_set(state: &Self::State, contains: &impl Fn(ComponentId) -> bool) -> bool {
        match *state {
            Some(id) => !contains(id),
            // `T` has never been archetype-tracked, so every archetype
            // trivially lacks it.
            None => true,
        }
    }
}

impl sealed::Sealed for () {}

/// No filter: matches every archetype. A `*_filtered` query with
/// `F = ()` is observably identical to its unfiltered counterpart
/// (tested).
impl QueryFilter for () {
    type State = ();

    #[inline]
    fn get_state(_resolve: &dyn Fn(TypeId) -> Option<ComponentId>) -> Self::State {}

    #[inline]
    fn matches_component_set(
        _state: &Self::State,
        _contains: &impl Fn(ComponentId) -> bool,
    ) -> bool {
        true
    }
}

macro_rules! impl_query_filter_for_tuple {
    ($($F:ident),+) => {
        impl<$($F: QueryFilter),+> sealed::Sealed for ($($F,)+) {}

        impl<$($F: QueryFilter),+> QueryFilter for ($($F,)+) {
            type State = ($($F::State,)+);

            #[inline]
            fn get_state(resolve: &dyn Fn(TypeId) -> Option<ComponentId>) -> Self::State {
                ($($F::get_state(resolve),)+)
            }

            #[inline]
            #[allow(non_snake_case)]
            fn matches_component_set(
                state: &Self::State,
                contains: &impl Fn(ComponentId) -> bool,
            ) -> bool {
                // Same trick bevy's own tuple impl uses: a type
                // parameter and the local that destructures its state
                // share a name, which is fine (types and values are
                // separate namespaces).
                let ($($F,)+) = state;
                true $(&& $F::matches_component_set($F, contains))+
            }
        }
    };
}

impl_query_filter_for_tuple!(A);
impl_query_filter_for_tuple!(A, B);
impl_query_filter_for_tuple!(A, B, C);
impl_query_filter_for_tuple!(A, B, C, D);
impl_query_filter_for_tuple!(A, B, C, D, E);
impl_query_filter_for_tuple!(A, B, C, D, E, F);
impl_query_filter_for_tuple!(A, B, C, D, E, F, G);
impl_query_filter_for_tuple!(A, B, C, D, E, F, G, H);

/// The inside of `Or<(...)>`: a tuple of [`QueryFilter`]s evaluated with
/// OR semantics (matches if *any* member does) instead of the plain
/// tuple `QueryFilter` impl's AND. Sealed, implemented only for tuples
/// of `QueryFilter` up to eight members, by
/// `impl_or_filter_group_for_tuple!` below. Not exported: callers only
/// ever name `Or<(...)>` itself.
pub trait OrFilterGroup: sealed::OrSealed {
    type State;
    fn get_state(resolve: &dyn Fn(TypeId) -> Option<ComponentId>) -> Self::State;
    fn any_matches(state: &Self::State, contains: &impl Fn(ComponentId) -> bool) -> bool;
}

macro_rules! impl_or_filter_group_for_tuple {
    ($($F:ident),+) => {
        impl<$($F: QueryFilter),+> sealed::OrSealed for ($($F,)+) {}

        impl<$($F: QueryFilter),+> OrFilterGroup for ($($F,)+) {
            type State = ($($F::State,)+);

            #[inline]
            fn get_state(resolve: &dyn Fn(TypeId) -> Option<ComponentId>) -> Self::State {
                ($($F::get_state(resolve),)+)
            }

            #[inline]
            #[allow(non_snake_case)]
            fn any_matches(state: &Self::State, contains: &impl Fn(ComponentId) -> bool) -> bool {
                let ($($F,)+) = state;
                false $(|| $F::matches_component_set($F, contains))+
            }
        }
    };
}

impl_or_filter_group_for_tuple!(A);
impl_or_filter_group_for_tuple!(A, B);
impl_or_filter_group_for_tuple!(A, B, C);
impl_or_filter_group_for_tuple!(A, B, C, D);
impl_or_filter_group_for_tuple!(A, B, C, D, E);
impl_or_filter_group_for_tuple!(A, B, C, D, E, F);
impl_or_filter_group_for_tuple!(A, B, C, D, E, F, G);
impl_or_filter_group_for_tuple!(A, B, C, D, E, F, G, H);

/// Keeps archetypes matching *any* member of the wrapped tuple, instead
/// of a plain tuple's *every* member. `T` is always a tuple of
/// [`QueryFilter`]s, e.g. `Or<(With<A>, With<B>)>`. `Or` is itself a
/// [`QueryFilter`], so it composes: `(Or<(With<A>, With<B>)>,
/// Without<C>)` keeps archetypes with `A` or `B`, and without `C`.
/// Never constructed; a type-level marker only.
pub struct Or<T>(PhantomData<fn() -> T>);

impl<T: OrFilterGroup> sealed::Sealed for Or<T> {}

impl<T: OrFilterGroup> QueryFilter for Or<T> {
    type State = T::State;

    #[inline]
    fn get_state(resolve: &dyn Fn(TypeId) -> Option<ComponentId>) -> Self::State {
        T::get_state(resolve)
    }

    #[inline]
    fn matches_component_set(state: &Self::State, contains: &impl Fn(ComponentId) -> bool) -> bool {
        T::any_matches(state, contains)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct X;
    struct Y;
    struct Z;
    /// Never registered by any test below.
    struct Unregistered;

    fn id_x() -> ComponentId {
        ComponentId::from_u32(10)
    }
    fn id_y() -> ComponentId {
        ComponentId::from_u32(11)
    }
    fn id_z() -> ComponentId {
        ComponentId::from_u32(12)
    }

    /// Registry where X, Y, Z are registered and everything else isn't.
    fn resolve(type_id: TypeId) -> Option<ComponentId> {
        if type_id == TypeId::of::<X>() {
            Some(id_x())
        } else if type_id == TypeId::of::<Y>() {
            Some(id_y())
        } else if type_id == TypeId::of::<Z>() {
            Some(id_z())
        } else {
            None
        }
    }

    /// Runs filter `F` against a signature made of exactly `sig`.
    fn matches<F: QueryFilter>(sig: &[ComponentId]) -> bool {
        let state = F::get_state(&resolve);
        F::matches_component_set(&state, &|c| sig.contains(&c))
    }

    #[test]
    fn with_matches_only_signatures_containing_the_component() {
        assert!(matches::<With<X>>(&[id_x()]));
        assert!(matches::<With<X>>(&[id_x(), id_y()]));
        assert!(!matches::<With<X>>(&[id_y()]));
        assert!(!matches::<With<X>>(&[]));
    }

    #[test]
    fn without_matches_only_signatures_lacking_the_component() {
        assert!(!matches::<Without<X>>(&[id_x()]));
        assert!(!matches::<Without<X>>(&[id_x(), id_y()]));
        assert!(matches::<Without<X>>(&[id_y()]));
        assert!(matches::<Without<X>>(&[]));
    }

    #[test]
    fn unregistered_component_with_matches_nothing_without_matches_everything() {
        assert!(!matches::<With<Unregistered>>(&[]));
        assert!(!matches::<With<Unregistered>>(&[id_x(), id_y()]));
        assert!(matches::<Without<Unregistered>>(&[]));
        assert!(matches::<Without<Unregistered>>(&[id_x(), id_y()]));
    }

    #[test]
    fn unit_filter_matches_everything() {
        assert!(matches::<()>(&[]));
        assert!(matches::<()>(&[id_x(), id_y(), id_z()]));
    }

    #[test]
    fn tuple_requires_every_member_to_match() {
        type F = (With<X>, Without<Y>);
        assert!(matches::<F>(&[id_x()]));
        assert!(matches::<F>(&[id_x(), id_z()]));
        assert!(!matches::<F>(&[id_x(), id_y()])); // Y present
        assert!(!matches::<F>(&[id_y()])); // X absent
        assert!(!matches::<F>(&[])); // X absent
    }

    #[test]
    fn one_element_tuple_behaves_like_its_member() {
        for sig in [&[][..], &[id_x()][..], &[id_y()][..], &[id_x(), id_y()][..]] {
            assert_eq!(matches::<(With<X>,)>(sig), matches::<With<X>>(sig));
            assert_eq!(matches::<(Without<X>,)>(sig), matches::<Without<X>>(sig));
        }
    }

    #[test]
    fn contradictory_filter_matches_nothing() {
        type F = (With<X>, Without<X>);
        assert!(!matches::<F>(&[]));
        assert!(!matches::<F>(&[id_x()]));
        assert!(!matches::<F>(&[id_x(), id_y()]));
    }

    #[test]
    fn an_unregistered_with_member_poisons_a_tuple_but_an_unregistered_without_does_not() {
        assert!(!matches::<(With<X>, With<Unregistered>)>(&[id_x()]));
        assert!(matches::<(With<X>, Without<Unregistered>)>(&[id_x()]));
    }

    #[test]
    fn nested_tuples_compose() {
        type F = ((With<X>, Without<Y>), With<Z>);
        assert!(matches::<F>(&[id_x(), id_z()]));
        assert!(!matches::<F>(&[id_x()])); // Z absent
        assert!(!matches::<F>(&[id_x(), id_y(), id_z()])); // Y present
    }

    #[test]
    fn eight_member_tuple_compiles_and_evaluates() {
        // The widest arity `impl_query_filter_for_tuple!` generates.
        type F = (
            With<X>,
            With<Y>,
            With<Z>,
            Without<Unregistered>,
            Without<Unregistered>,
            Without<Unregistered>,
            Without<Unregistered>,
            Without<Unregistered>,
        );
        assert!(matches::<F>(&[id_x(), id_y(), id_z()]));
        assert!(!matches::<F>(&[id_x(), id_y()]));
    }

    #[test]
    fn get_state_never_registers() {
        // `resolve` here is a plain read-only closure; the trait has no
        // way to hand it a registering one. This test pins that a
        // filter over an unknown type yields `None` rather than
        // anything else.
        assert_eq!(<With<Unregistered>>::get_state(&resolve), None);
        assert_eq!(<Without<Unregistered>>::get_state(&resolve), None);
        assert_eq!(<With<X>>::get_state(&resolve), Some(id_x()));
    }

    #[test]
    fn or_matches_if_any_member_matches() {
        type F = Or<(With<X>, With<Y>)>;
        assert!(matches::<F>(&[id_x()]));
        assert!(matches::<F>(&[id_y()]));
        assert!(matches::<F>(&[id_x(), id_y()]));
        assert!(matches::<F>(&[id_x(), id_z()]));
        assert!(!matches::<F>(&[id_z()]));
        assert!(!matches::<F>(&[]));
    }

    #[test]
    fn or_of_without_matches_if_either_is_absent() {
        type F = Or<(Without<X>, Without<Y>)>;
        assert!(matches::<F>(&[])); // neither present
        assert!(matches::<F>(&[id_x()])); // Y absent
        assert!(matches::<F>(&[id_y()])); // X absent
        assert!(!matches::<F>(&[id_x(), id_y()])); // both present
    }

    #[test]
    fn one_member_or_behaves_like_its_member() {
        for sig in [&[][..], &[id_x()][..], &[id_y()][..]] {
            assert_eq!(matches::<Or<(With<X>,)>>(sig), matches::<With<X>>(sig));
        }
    }

    #[test]
    fn or_composes_inside_a_tuple_with_and_semantics_at_the_outer_level() {
        // (Or<(With<X>, With<Y>)>, Without<Z>): (X or Y) and not Z.
        type F = (Or<(With<X>, With<Y>)>, Without<Z>);
        assert!(matches::<F>(&[id_x()]));
        assert!(matches::<F>(&[id_y()]));
        assert!(!matches::<F>(&[id_x(), id_z()])); // Z present
        assert!(!matches::<F>(&[])); // neither X nor Y
    }

    #[test]
    fn or_can_wrap_an_unregistered_component_and_still_match_via_the_other_member() {
        assert!(matches::<Or<(With<X>, With<Unregistered>)>>(&[id_x()]));
        assert!(!matches::<Or<(With<Unregistered>, With<Unregistered>)>>(&[
            id_x(),
            id_y()
        ]));
    }

    #[test]
    fn nested_or_inside_or() {
        // Or<(Or<(With<X>, With<Y>)>, With<Z>)>: X or Y or Z.
        type F = Or<(Or<(With<X>, With<Y>)>, With<Z>)>;
        assert!(matches::<F>(&[id_x()]));
        assert!(matches::<F>(&[id_y()]));
        assert!(matches::<F>(&[id_z()]));
        assert!(!matches::<F>(&[]));
    }

    #[test]
    fn eight_member_or_compiles_and_evaluates() {
        type F = Or<(
            With<X>,
            With<Unregistered>,
            With<Unregistered>,
            With<Unregistered>,
            With<Unregistered>,
            With<Unregistered>,
            With<Unregistered>,
            With<Unregistered>,
        )>;
        assert!(matches::<F>(&[id_x()]));
        assert!(!matches::<F>(&[id_y()]));
    }

    #[test]
    fn or_get_state_never_registers() {
        assert_eq!(<Or<(With<Unregistered>,)>>::get_state(&resolve), (None,));
    }
}
