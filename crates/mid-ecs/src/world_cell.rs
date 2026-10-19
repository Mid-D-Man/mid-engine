// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-ecs.md, section "`WorldCell`: shared access with checked
// disjointness (`world_cell.rs`)"
// ============================================================================
//! [`WorldCell`]: one `&mut World` turned into a copyable handle that can
//! serve several queries and resources at once, with the *caller* proving
//! they do not collide.
//!
//! A system with `Query<&mut A>`, `Query<&B>` and `ResMut<R>` holds three
//! borrows of one `World`. Safe Rust cannot express "these touch different
//! data" when the difference is only known from the *types* the queries
//! name, so this is the crate's one place where that proof moves from the
//! compiler to a rule: **a `WorldCell` function may be called only while
//! nothing else reads what it writes or writes what it reads.** The rule is
//! the one [`FilteredAccessSet`](crate::FilteredAccessSet) computes, which
//! is why a system checks its parameters' access sets when it is set up
//! and why the functions here are `unsafe`.
//!
//! **What makes the rule enough.** All of the cell's mutation goes through
//! [`Interior`](crate::interior): each column of component values, each
//! column of change ticks, and each resource is its own
//! `UnsafeCell`, so two `&mut` to *different* columns of one archetype
//! alias nothing (the archetype list, the tables and everything that
//! locates a column are only ever borrowed shared). The cell borrows the
//! `World` exclusively for `'w`, so nothing outside it can touch the world
//! meanwhile: any overlap has to come from two calls on the cell, and
//! those are `unsafe`.
//!
//! **What is not here.** Nothing is checked at runtime (no borrow flags):
//! violating the rule is undefined behaviour, which is what `unsafe`
//! means. There is no way to *add* an entity, insert a component or remove
//! a resource through a cell: those change structures other calls hold
//! shared references into, and belong to commands applied afterwards with
//! `&mut World`. And the cell is not `Send`/`Sync`: running two systems at
//! once is the next step (gap-list item 5), and what it needs from here is
//! a decision about those bounds, made there.

use std::cell::UnsafeCell;
use std::marker::PhantomData;

use crate::data_filter::DataFilter;
use crate::query_data::{QueryData, ReadOnlyQueryData};
use crate::tick::Tick;
use crate::world::World;

/// A copyable, `unsafe`-to-use handle on a [`World`]. See the module docs
/// for the rule every method relies on.
#[derive(Clone, Copy)]
pub struct WorldCell<'w> {
    world: *mut World,
    /// Borrows the world exclusively (it came from `&'w mut World`) and
    /// is not covariant in what it points at.
    _marker: PhantomData<(&'w UnsafeCell<World>, &'w mut World)>,
}

impl World {
    /// A [`WorldCell`] over this world, borrowing it exclusively for as
    /// long as the cell (or anything taken from it) lives.
    pub fn as_world_cell(&mut self) -> WorldCell<'_> {
        WorldCell {
            world: self as *mut World,
            _marker: PhantomData,
        }
    }
}

impl<'w> WorldCell<'w> {
    /// The world, shared, for reading *structure* (which archetypes
    /// exist, which resources are present).
    ///
    /// # Safety
    /// No `&mut` to the world or to any structure inside it may exist.
    /// True for the whole of `'w` by construction: the cell holds the
    /// world's exclusive borrow and hands out only shared references to
    /// structure, with mutation confined to `Interior`.
    #[inline]
    unsafe fn world(self) -> &'w World {
        // SAFETY: see above; the pointer came from a live `&'w mut World`.
        unsafe { &*self.world }
    }

    /// The world's change tick: what a query made from this cell stamps
    /// writes with and measures `Ref`, `Added` and `Changed` against.
    pub fn change_tick(self) -> Tick {
        // SAFETY: only a `Copy` field is read, and the world's exclusive
        // borrow is held by this cell, so nothing writes it.
        unsafe { self.world() }.change_tick()
    }

    /// Whether a resource of type `T` is present. Reads which resources
    /// exist, never a value, and the cell cannot add or remove one.
    pub fn contains_resource<T: 'static>(self) -> bool {
        // SAFETY: structure read only; see `world`.
        unsafe { self.world() }.contains_resource::<T>()
    }

    /// A read-only query `Q` filtered by `F`, over the archetypes that
    /// match, as `World::query_data_filtered_since` would yield them.
    /// `this_run` and `last_run` are the ticks a `Ref` carries and
    /// `Added`/`Changed` compare against.
    ///
    /// # Safety
    /// For as long as the iterator or anything it yielded is live: nothing
    /// may write a component this query reads, values or change ticks.
    /// (Other readers are fine.)
    pub unsafe fn query_iter<Q: ReadOnlyQueryData + 'w, F: DataFilter + 'w>(
        self,
        this_run: Tick,
        last_run: Tick,
    ) -> impl Iterator<Item = Q::Item<'w>> + 'w {
        // SAFETY: shared structure read; the data contract is the caller's.
        unsafe { self.world() }
            .archetypes
            .iter_data_filtered::<Q, F>(this_run, last_run)
    }

    /// Any query `Q` (reads and writes) filtered by `F`, as
    /// `World::query_data_mut_filtered_since` would yield it. Writes
    /// through `Mut` terms are stamped `this_run`.
    ///
    /// # Panics
    /// If `Q` reads and writes, or writes twice, the same component (the
    /// check `World::query_data_mut` makes), before anything is touched.
    ///
    /// # Safety
    /// For as long as the iterator or anything it yielded is live: nothing
    /// else may read or write a component `Q` writes (values or change
    /// ticks, so also not a `Changed<T>` filter of another query), nor
    /// write one it reads. Exactly what `FilteredAccess::of_query::<Q, F>()`
    /// says this query does, and the reason two accesses that
    /// `FilteredAccess::is_compatible` accepts may use the cell together.
    pub unsafe fn query_iter_mut<Q: QueryData + 'w, F: DataFilter + 'w>(
        self,
        this_run: Tick,
        last_run: Tick,
    ) -> impl Iterator<Item = Q::Item<'w>> + 'w {
        // SAFETY: the contract is the caller's; `iter_data_cell` spells
        // out what it relies on and this forwards it.
        unsafe {
            self.world()
                .archetypes
                .iter_data_cell::<Q, F>(this_run, last_run)
        }
    }

    /// The resource of type `T`, shared.
    ///
    /// # Safety
    /// For as long as the reference is live, nothing may write this
    /// resource.
    pub unsafe fn resource<T: 'static>(self) -> Option<&'w T> {
        // SAFETY: shared structure read; the data contract is the caller's.
        unsafe { self.world() }.get_resource::<T>()
    }

    /// The resource of type `T`, exclusive.
    ///
    /// # Safety
    /// For as long as the reference is live, nothing else may read or
    /// write this resource: not another call to this function for the
    /// same `T`, not [`Self::resource`].
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn resource_mut<T: 'static>(self) -> Option<&'w mut T> {
        // SAFETY: forwarded to the caller.
        unsafe { self.world().resources.get_mut_unchecked::<T>() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filter::{With, Without};
    use crate::tick::{ChangeTracker, Mut, Ref};
    use crate::world::Entity;

    #[derive(Debug, Clone, Copy, PartialEq)]
    struct A(u32);
    #[derive(Debug, Clone, Copy, PartialEq)]
    struct B(u32);
    #[derive(Debug, Clone, Copy, PartialEq)]
    struct C(u32);
    #[derive(Debug, PartialEq)]
    struct Score(u32);
    #[derive(Debug, PartialEq)]
    struct Other(u32);

    use crate::query_data::{Added, Changed, Has};

    fn key(e: &Entity) -> (u32, u32) {
        (e.index(), e.generation())
    }

    /// Entities in the archetypes {A, B} (6), {A, B, C} (3), {A} (2), {B} (2).
    fn world() -> (World, Vec<Entity>) {
        let mut world = World::new();
        let mut es = Vec::new();
        for i in 0..6u32 {
            let e = world.spawn();
            world.insert_static(e, A(i));
            world.insert_static(e, B(100 + i));
            es.push(e);
        }
        for i in 6..9u32 {
            let e = world.spawn();
            world.insert_static(e, A(i));
            world.insert_static(e, B(100 + i));
            world.insert_static(e, C(200 + i));
            es.push(e);
        }
        for i in 9..11u32 {
            let e = world.spawn();
            world.insert_static(e, A(i));
            es.push(e);
        }
        for i in 11..13u32 {
            let e = world.spawn();
            world.insert_static(e, B(100 + i));
            es.push(e);
        }
        (world, es)
    }

    #[test]
    fn a_writer_and_a_reader_of_different_components_run_nested_over_one_archetype() {
        let (mut world, _) = world();
        world.increment_change_tick();
        let mut tracker = ChangeTracker::new();
        tracker.update(&world);
        world.increment_change_tick();
        let this_run = world.change_tick();
        let cell = world.as_world_cell();
        let mut seen = 0;
        // SAFETY: the outer query only reads B and the inner only writes
        // A: different components, no overlap, in the same archetypes.
        unsafe {
            for (e, b) in cell.query_iter::<(Entity, &B), ()>(this_run, Tick::default()) {
                for (inner, mut a) in
                    cell.query_iter_mut::<(Entity, Mut<A>), With<B>>(this_run, Tick::default())
                {
                    if inner == e {
                        a.0 += b.0;
                        seen += 1;
                    }
                }
            }
        }
        assert_eq!(
            seen, 9,
            "every {{A, B}} and {{A, B, C}} entity met itself once"
        );
        let mut got: Vec<u32> = world
            .query_data::<(Entity, &A)>()
            .filter(|(e, _)| world.has_static::<B>(*e))
            .map(|(_, a)| a.0)
            .collect();
        got.sort();
        let want: Vec<u32> = (0..9u32).map(|i| i + 100 + i).collect();
        assert_eq!(got, want);
        // The writes were stamped, and only the written column's ticks.
        world.increment_change_tick();
        assert_eq!(world.query_changed::<A>(&tracker).count(), 9);
        assert_eq!(world.query_changed::<B>(&tracker).count(), 0);
    }

    #[test]
    fn two_writers_of_different_components_hold_their_items_at_the_same_time() {
        let (mut world, _) = world();
        let this_run = world.change_tick();
        let cell = world.as_world_cell();
        // SAFETY: one writes A, the other writes B.
        let (a_items, b_items) = unsafe {
            (
                cell.query_iter_mut::<Mut<A>, With<B>>(this_run, Tick::default())
                    .collect::<Vec<_>>(),
                cell.query_iter_mut::<Mut<B>, With<A>>(this_run, Tick::default())
                    .collect::<Vec<_>>(),
            )
        };
        assert_eq!(a_items.len(), 9);
        assert_eq!(b_items.len(), 9);
        for mut a in a_items {
            a.0 += 1000;
        }
        for mut b in b_items {
            b.0 += 2000;
        }
        assert!(world
            .query_data::<(&A, &B)>()
            .all(|(a, b)| a.0 >= 1000 && b.0 >= 2100));
    }

    /// The same world every time: `world()`, everything seen by a tracker,
    /// then every third `A` written.
    fn scenario() -> (World, ChangeTracker) {
        let (mut world, _) = world();
        world.increment_change_tick();
        let mut tracker = ChangeTracker::new();
        tracker.update(&world);
        world.increment_change_tick();
        let mut es: Vec<Entity> = world.query_data::<(Entity, &A)>().map(|(e, _)| e).collect();
        es.sort_by_key(key);
        for &e in es.iter().step_by(3) {
            world.get_static_mut::<A>(e).unwrap().0 += 7;
        }
        (world, tracker)
    }

    #[test]
    fn cell_queries_yield_what_the_safe_queries_yield() {
        let (mut world, tracker) = scenario();
        let this_run = world.change_tick();
        let last_run = tracker.last_run();
        let safe: Vec<(Entity, u32, bool)> = world
            .query_data_filtered_since::<(Entity, &A, Changed<A>), Without<C>>(&tracker)
            .map(|(e, a, c)| (e, a.0, c))
            .collect();
        let cell = world.as_world_cell();
        // SAFETY: read-only, nothing writes.
        let via_cell: Vec<(Entity, u32, bool)> =
            unsafe { cell.query_iter::<(Entity, &A, Changed<A>), Without<C>>(this_run, last_run) }
                .map(|(e, a, c)| (e, a.0, c))
                .collect();
        assert_eq!(safe, via_cell);
        assert!(safe.iter().any(|t| t.2) && safe.iter().any(|t| !t.2));
    }

    #[test]
    fn a_filtered_write_through_the_cell_matches_the_safe_exclusive_iterator() {
        let (mut left, left_tracker) = scenario();
        let (mut right, right_tracker) = scenario();
        for mut a in left.query_data_mut_filtered_since::<Mut<A>, Changed<A>>(&left_tracker) {
            a.0 += 5000;
        }
        let this_run = right.change_tick();
        let last_run = right_tracker.last_run();
        let cell = right.as_world_cell();
        // SAFETY: the only live access.
        for mut a in unsafe { cell.query_iter_mut::<Mut<A>, Changed<A>>(this_run, last_run) } {
            a.0 += 5000;
        }
        let values = |w: &World| {
            let mut v: Vec<u32> = w.query_data::<&A>().map(|a| a.0).collect();
            v.sort();
            v
        };
        assert_eq!(values(&left), values(&right));
        assert_eq!(values(&left).iter().filter(|&&v| v >= 5000).count(), 4);
        // The stamps agree too: the written rows are exactly the changed ones.
        left.increment_change_tick();
        right.increment_change_tick();
        let mut after_l = ChangeTracker::new();
        let mut after_r = ChangeTracker::new();
        after_l.update(&left);
        after_r.update(&right);
        left.increment_change_tick();
        right.increment_change_tick();
        assert_eq!(
            left.query_changed::<A>(&after_l).count(),
            right.query_changed::<A>(&after_r).count()
        );
    }

    #[test]
    fn terms_and_filters_of_every_shape_work_through_the_cell() {
        let (mut world, _) = world();
        let this_run = world.change_tick();
        let cell = world.as_world_cell();
        // SAFETY: read-only.
        unsafe {
            let with_c = cell
                .query_iter::<(Entity, Has<C>), ()>(this_run, Tick::default())
                .filter(|(_, has)| *has)
                .count();
            assert_eq!(with_c, 3);
            let opt: Vec<Option<u32>> = cell
                .query_iter::<Option<&C>, With<A>>(this_run, Tick::default())
                .map(|c| c.map(|c| c.0))
                .collect();
            assert_eq!(opt.len(), 11);
            assert_eq!(opt.iter().filter(|c| c.is_some()).count(), 3);
            assert_eq!(
                cell.query_iter::<Ref<A>, ()>(this_run, Tick::default())
                    .count(),
                11
            );
            assert_eq!(
                cell.query_iter::<Added<C>, ()>(this_run, Tick::default())
                    .count(),
                3
            );
            assert_eq!(
                cell.query_iter::<&C, ()>(this_run, Tick::default())
                    .map(|c| c.0)
                    .sum::<u32>(),
                200 + 6 + 200 + 7 + 200 + 8
            );
        }
        // An unseen type matches nothing, and an optional one yields None.
        struct Never;
        unsafe {
            assert_eq!(
                cell.query_iter::<&Never, ()>(this_run, Tick::default())
                    .count(),
                0
            );
            assert!(cell
                .query_iter_mut::<(Entity, Option<Mut<Never>>), ()>(this_run, Tick::default())
                .all(|(_, n)| n.is_none()));
        }
    }

    #[test]
    #[should_panic(expected = "WorldCell::query_iter_mut: conflicting access")]
    fn a_query_that_conflicts_with_itself_is_refused_before_anything_is_touched() {
        let (mut world, _) = world();
        let this_run = world.change_tick();
        let cell = world.as_world_cell();
        // SAFETY: refused before any access happens.
        let _ = unsafe { cell.query_iter_mut::<(&A, Mut<A>), ()>(this_run, Tick::default()) };
    }

    #[test]
    fn a_cell_writer_reads_the_ticks_of_components_it_does_not_write() {
        // `Changed<B>` (a filter) and `Ref<B>` (a term) read B's ticks;
        // the query writes A. The tick slices of B must reach the
        // iterator even though B is never written.
        let (mut world, _) = world();
        world.increment_change_tick();
        let mut tracker = ChangeTracker::new();
        tracker.update(&world);
        world.increment_change_tick();
        let b_changed: Vec<Entity> = world
            .query_data::<(Entity, &B)>()
            .map(|(e, _)| e)
            .step_by(2)
            .collect();
        for &e in &b_changed {
            world.get_static_mut::<B>(e).unwrap().0 += 1;
        }
        let this_run = world.change_tick();
        let last_run = tracker.last_run();
        // The entities the queries below can reach: they need an A too.
        let mut want: Vec<Entity> = b_changed
            .iter()
            .copied()
            .filter(|&e| world.has_static::<A>(e))
            .collect();
        want.sort_by_key(key);
        let cell = world.as_world_cell();
        let mut touched = Vec::new();
        // SAFETY: the only live access; B is only read.
        unsafe {
            for (e, mut a) in
                cell.query_iter_mut::<(Entity, Mut<A>), Changed<B>>(this_run, last_run)
            {
                a.0 += 10_000;
                touched.push(e);
            }
            let mut flagged = 0;
            for (a, b) in cell.query_iter_mut::<(Mut<A>, Ref<B>), ()>(this_run, last_run) {
                if b.is_changed(&tracker) {
                    flagged += 1;
                }
                let _ = a;
            }
            assert_eq!(flagged, want.len());
        }
        touched.sort_by_key(key);
        assert_eq!(touched, want);
        assert!(
            want.len() < b_changed.len(),
            "one B-only entity changed too"
        );
        for (e, a) in world.query_data::<(Entity, &A)>() {
            assert_eq!(a.0 >= 10_000, want.contains(&e), "{e:?}");
        }
    }

    #[test]
    fn a_resource_is_borrowed_alongside_queries_of_components() {
        let (mut world, _) = world();
        world.insert_resource(Score(0));
        world.insert_resource(Other(5));
        let this_run = world.change_tick();
        let cell = world.as_world_cell();
        assert!(cell.contains_resource::<Score>());
        assert!(
            !cell.contains_resource::<A>(),
            "components are not resources"
        );
        // SAFETY: Score is written by exactly one reference, Other read by
        // one, and the query writes only A.
        unsafe {
            let score = cell.resource_mut::<Score>().expect("present");
            let other = cell.resource::<Other>().expect("present");
            for mut a in cell.query_iter_mut::<Mut<A>, ()>(this_run, Tick::default()) {
                a.0 += other.0;
                score.0 += 1;
            }
            assert!(cell.resource::<Missing>().is_none());
            assert!(cell.resource_mut::<Missing>().is_none());
        }
        assert_eq!(world.get_resource::<Score>(), Some(&Score(11)));
        assert_eq!(world.get_resource::<Other>(), Some(&Other(5)));
    }

    struct Missing;

    #[test]
    fn the_world_is_usable_again_once_the_cell_is_dropped() {
        let (mut world, es) = world();
        {
            let this_run = world.change_tick();
            let cell = world.as_world_cell();
            // SAFETY: the only access.
            for mut a in unsafe { cell.query_iter_mut::<Mut<A>, ()>(this_run, Tick::default()) } {
                a.0 *= 2;
            }
            assert_eq!(cell.change_tick(), this_run);
        }
        assert_eq!(world.get_static::<A>(es[3]).map(|a| a.0), Some(6));
        // Structural changes are fine now, and the next cell sees them.
        let extra = world.spawn();
        world.insert_static(extra, A(1));
        world.insert_static(extra, B(1));
        let this_run = world.change_tick();
        let cell = world.as_world_cell();
        // SAFETY: read-only.
        let n = unsafe { cell.query_iter::<(&A, &B), ()>(this_run, Tick::default()) }.count();
        assert_eq!(n, 10);
    }

    #[test]
    fn iterating_a_cell_query_twice_in_a_row_sees_the_first_writes() {
        let (mut world, _) = world();
        let this_run = world.change_tick();
        let cell = world.as_world_cell();
        // SAFETY: each iterator is finished before the next one starts.
        unsafe {
            for mut b in cell.query_iter_mut::<Mut<B>, ()>(this_run, Tick::default()) {
                b.0 += 1;
            }
            let total: u32 = cell
                .query_iter::<&B, ()>(this_run, Tick::default())
                .map(|b| b.0)
                .sum();
            // Entities 9 and 10 have no B.
            let want: u32 = (0..9u32).chain(11..13).map(|i| 100 + i + 1).sum();
            assert_eq!(total, want);
        }
    }
}
