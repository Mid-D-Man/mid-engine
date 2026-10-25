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

use crate::archetype::{LookupMiss, QueryBatch};
use crate::batching::BatchingStrategy;
use crate::data_filter::DataFilter;
use crate::query_data::{QueryData, QueryEntityError, ReadOnlyQueryData};
use crate::removal::RemovalLog;
use crate::tick::Tick;
use crate::world::{Entity, World};

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

    /// The removal log, shared: what `RemovedComponents` reads.
    ///
    /// # Safety
    /// As for [`Self::world`]. The log is only ever written by structural
    /// changes, which take `&mut World` and so cannot happen while the
    /// cell exists (the exception, `world_mut`, is `unsafe` and already
    /// excludes every other access).
    #[inline]
    pub(crate) unsafe fn removals(self) -> &'w RemovalLog {
        // SAFETY: as above.
        unsafe { &self.world().removals }
    }

    /// The whole world, exclusively: for work that takes `&mut World`
    /// (an exclusive system) and therefore has to be the only thing
    /// running.
    ///
    /// # Safety
    /// For as long as the returned reference is live, no other access of
    /// any kind to the world may exist or begin: no other copy of this
    /// cell may be used, no iterator or reference taken from one may be
    /// alive. This is the access `Access::write_all` describes, and why
    /// that access conflicts with everything that touches anything.
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn world_mut(self) -> &'w mut World {
        // SAFETY: the pointer came from a live `&'w mut World`; the rest
        // is the caller's contract.
        unsafe { &mut *self.world }
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

    /// The query `Q` filtered by `F`, cut into independent
    /// [`QueryBatch`]es that may run on different threads at once: the
    /// parallel form of [`Self::query_iter_mut`] (it also serves a
    /// read-only `Q`). Together the batches yield exactly the rows
    /// `query_iter_mut` yields, each in exactly one batch, with disjoint
    /// rows between batches. `threads` only sizes the batches; no thread
    /// is started here. See [`BatchingStrategy`].
    ///
    /// A batch can be sent to another thread when `Q::Fetch<'w>` is
    /// `Send`; this function itself has no such bound, so a single
    /// thread can run the batches one after another for any query.
    ///
    /// # Panics
    /// If `Q` reads and writes, or writes twice, the same component, or
    /// if `strategy` needs a thread count and `threads` is zero.
    ///
    /// # Safety
    /// As for [`Self::query_iter_mut`], and additionally, for as long as
    /// any batch or anything it yielded is live, nothing may change the
    /// world's structure (spawn, despawn, insert or remove a component):
    /// the batches hold raw pointers into the columns. (Nothing reachable
    /// through the cell can; `world_mut` is the exception and is already
    /// `unsafe`.)
    pub unsafe fn query_batches<Q: QueryData + 'w, F: DataFilter + 'w>(
        self,
        strategy: &BatchingStrategy,
        threads: usize,
        this_run: Tick,
        last_run: Tick,
    ) -> Vec<QueryBatch<'w, Q, F>> {
        // SAFETY: shared structure read; the data contract is the
        // caller's, and `par_batches_cell` spells out what it relies on.
        unsafe {
            self.world()
                .archetypes
                .par_batches_cell::<Q, F>(this_run, last_run, strategy, threads)
        }
    }

    /// The item the read-only query `Q` filtered by `F` yields for
    /// `entity`, or why there is none. See [`Self::query_get_mut`] for a
    /// query that writes.
    ///
    /// # Safety
    /// As for [`Self::query_iter`], for as long as the returned item is
    /// live.
    pub unsafe fn query_get<Q: ReadOnlyQueryData + 'w, F: DataFilter + 'w>(
        self,
        entity: Entity,
        this_run: Tick,
        last_run: Tick,
    ) -> Result<Q::Item<'w>, QueryEntityError> {
        // SAFETY: the contract is the caller's; a read-only `Q` writes
        // nothing, so it is the one `query_iter` states.
        unsafe { self.query_get_any::<Q, F>(entity, this_run, last_run) }
    }

    /// The item any query `Q` (reads and writes) filtered by `F` yields
    /// for `entity`, or why there is none. Writes through `Mut` terms are
    /// stamped `this_run`.
    ///
    /// # Panics
    /// If `Q` reads and writes, or writes twice, the same component.
    ///
    /// # Safety
    /// As for [`Self::query_iter_mut`], for as long as the returned item
    /// is live; in particular, while it is live nothing may look the same
    /// entity up again for a query that writes (it would hand out a second
    /// `&mut` to the same row).
    pub unsafe fn query_get_mut<Q: QueryData + 'w, F: DataFilter + 'w>(
        self,
        entity: Entity,
        this_run: Tick,
        last_run: Tick,
    ) -> Result<Q::Item<'w>, QueryEntityError> {
        // SAFETY: the contract is the caller's.
        unsafe { self.query_get_any::<Q, F>(entity, this_run, last_run) }
    }

    /// # Safety
    /// The contract of whichever public method calls this.
    unsafe fn query_get_any<Q: QueryData + 'w, F: DataFilter + 'w>(
        self,
        entity: Entity,
        this_run: Tick,
        last_run: Tick,
    ) -> Result<Q::Item<'w>, QueryEntityError> {
        // SAFETY: shared structure read; the data contract is the caller's.
        let world = unsafe { self.world() };
        if !world.is_alive(entity) {
            return Err(QueryEntityError::NoSuchEntity(entity));
        }
        // SAFETY: `get_data_cell` spells out what it relies on and this
        // forwards it.
        match unsafe {
            world
                .archetypes
                .get_data_cell::<Q, F>(entity, this_run, last_run)
        } {
            Ok(item) => Ok(item),
            Err(LookupMiss::NoLocation) => Err(QueryEntityError::NoSuchEntity(entity)),
            Err(LookupMiss::DoesNotMatch) => Err(QueryEntityError::QueryDoesNotMatch(entity)),
        }
    }

    /// Whether query `Q` filtered by `F` matches `entity` (alive, in a
    /// matching archetype, and past any row-level filter). Builds no item,
    /// so it is the same for a reading and a writing `Q`.
    ///
    /// # Safety
    /// For the duration of the call, nothing may write the change ticks of
    /// a component a row-level filter of `F` reads (the access `F` declares
    /// already forbids it).
    pub unsafe fn query_contains<Q: QueryData + 'w, F: DataFilter + 'w>(
        self,
        entity: Entity,
        this_run: Tick,
        last_run: Tick,
    ) -> bool {
        // SAFETY: shared structure read; the tick contract is the caller's.
        let world = unsafe { self.world() };
        world.is_alive(entity)
            && world
                .archetypes
                .matches_entity::<Q, F>(entity, this_run, last_run)
    }

    /// How many rows query `Q` filtered by `F` matches, counted without
    /// building any item.
    ///
    /// # Safety
    /// As for [`Self::query_contains`].
    pub unsafe fn query_count<Q: QueryData + 'w, F: DataFilter + 'w>(
        self,
        this_run: Tick,
        last_run: Tick,
    ) -> usize {
        // SAFETY: shared structure read; the tick contract is the caller's.
        unsafe { self.world() }
            .archetypes
            .count_matching::<Q, F>(this_run, last_run)
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
    fn an_exclusive_reference_can_be_taken_from_the_cell_when_nothing_else_is_live() {
        let (mut world, _) = world();
        let cell = world.as_world_cell();
        // SAFETY: the only access; no query or resource reference exists.
        let w = unsafe { cell.world_mut() };
        let e = w.spawn();
        w.insert_static(e, A(99));
        assert_eq!(world.get_static::<A>(e).map(|a| a.0), Some(99));
    }

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

    // ── Parallel batches ─────────────────────────────────────────────

    use crate::batching::BatchingStrategy;

    /// A zero-sized component, for the empty-window edge.
    #[derive(Debug, PartialEq)]
    struct Tag;

    /// `ab` entities in {A, B}, `abc` in {A, B, C}, `a` in {A}, `b` in {B}.
    /// Values are `A(i)`, `B(100 + i)`, `C(200 + i)` for a running `i`.
    fn sized_world(ab: u32, abc: u32, a: u32, b: u32) -> (World, Vec<Entity>) {
        let mut world = World::new();
        let mut es = Vec::new();
        let mut i = 0u32;
        for _ in 0..ab {
            let e = world.spawn();
            world.insert_static(e, A(i));
            world.insert_static(e, B(100 + i));
            es.push(e);
            i += 1;
        }
        for _ in 0..abc {
            let e = world.spawn();
            world.insert_static(e, A(i));
            world.insert_static(e, B(100 + i));
            world.insert_static(e, C(200 + i));
            es.push(e);
            i += 1;
        }
        for _ in 0..a {
            let e = world.spawn();
            world.insert_static(e, A(i));
            es.push(e);
            i += 1;
        }
        for _ in 0..b {
            let e = world.spawn();
            world.insert_static(e, B(100 + i));
            es.push(e);
            i += 1;
        }
        (world, es)
    }

    fn strategies() -> Vec<BatchingStrategy> {
        vec![
            BatchingStrategy::new(),
            BatchingStrategy::new().batches_per_thread(3),
            BatchingStrategy::fixed(1),
            BatchingStrategy::fixed(3),
            BatchingStrategy::fixed(7),
            BatchingStrategy::fixed(1000),
            BatchingStrategy::new().min_batch_size(4).max_batch_size(9),
        ]
    }

    #[test]
    fn batches_yield_exactly_the_rows_the_sequential_cell_iterator_yields() {
        let (mut world, _) = sized_world(20, 9, 5, 4);
        world.increment_change_tick();
        let this_run = world.change_tick();
        let cell = world.as_world_cell();
        // SAFETY: only readers of this world are live.
        let mut want: Vec<((u32, u32), u32, u32)> =
            unsafe { cell.query_iter::<(Entity, &A, &B), ()>(this_run, Tick::default()) }
                .map(|(e, a, b)| (key(&e), a.0, b.0))
                .collect();
        want.sort();
        assert_eq!(want.len(), 29);
        for strategy in strategies() {
            for threads in [1usize, 2, 4, 16] {
                // SAFETY: as above; the batches are only iterated here.
                let batches = unsafe {
                    cell.query_batches::<(Entity, &A, &B), ()>(
                        &strategy,
                        threads,
                        this_run,
                        Tick::default(),
                    )
                };
                let mut got: Vec<((u32, u32), u32, u32)> = batches
                    .into_iter()
                    .flatten()
                    .map(|(e, a, b)| (key(&e), a.0, b.0))
                    .collect();
                got.sort();
                assert_eq!(got, want, "strategy {strategy:?}, {threads} threads");
            }
        }
    }

    #[test]
    fn the_batch_count_follows_the_strategy() {
        let (mut world, _) = sized_world(20, 9, 0, 0);
        let this_run = world.change_tick();
        let cell = world.as_world_cell();
        let count = |strategy: BatchingStrategy, threads: usize| {
            // SAFETY: nothing else is live; the batches are only counted.
            unsafe { cell.query_batches::<&A, ()>(&strategy, threads, this_run, Tick::default()) }
                .len()
        };
        // 20 + 9 rows in two archetypes. Fixed 5: 4 windows + 2 windows.
        assert_eq!(count(BatchingStrategy::fixed(5), 1), 6);
        // Fixed 100: both archetypes are short, queued into one batch.
        assert_eq!(count(BatchingStrategy::fixed(100), 1), 1);
        // Default over 4 threads: ceil(20 / 4) = 5 rows per batch.
        assert_eq!(count(BatchingStrategy::new(), 4), 6);
        // One thread: batch size = the largest archetype, 20: the big
        // archetype alone is one batch and the 9 queue to the end.
        assert_eq!(count(BatchingStrategy::new(), 1), 2);
    }

    #[test]
    fn small_archetypes_are_merged_into_shared_batches() {
        let mut world = World::new();
        for combo in 1u32..8 {
            let e = world.spawn();
            if combo & 1 != 0 {
                world.insert_static(e, A(0));
            }
            if combo & 2 != 0 {
                world.insert_static(e, B(0));
            }
            if combo & 4 != 0 {
                world.insert_static(e, C(0));
            }
        }
        let this_run = world.change_tick();
        let cell = world.as_world_cell();
        // SAFETY: nothing else is live.
        let batches = unsafe {
            cell.query_batches::<Entity, ()>(
                &BatchingStrategy::fixed(4),
                1,
                this_run,
                Tick::default(),
            )
        };
        // 7 one-row archetypes with batch size 4: [4 windows] [3 windows].
        assert_eq!(batches.len(), 2);
        let mut seen: Vec<Entity> = batches.into_iter().flatten().collect();
        seen.sort_by_key(key);
        assert_eq!(seen.len(), 7);
    }

    #[test]
    fn a_query_with_no_matching_row_has_no_batches() {
        let (mut world, _) = sized_world(3, 0, 0, 0);
        let this_run = world.change_tick();
        let cell = world.as_world_cell();
        // SAFETY: nothing else is live.
        unsafe {
            // No `Other` was ever inserted: nothing can match.
            assert!(cell
                .query_batches::<&Other, ()>(&BatchingStrategy::new(), 4, this_run, Tick::default())
                .is_empty());
            // `C` exists nowhere in this world but is a known type: the
            // matched list is empty too.
            assert!(cell
                .query_batches::<&C, ()>(&BatchingStrategy::new(), 4, this_run, Tick::default())
                .is_empty());
        }
        let mut empty = World::new();
        let cell = empty.as_world_cell();
        // SAFETY: nothing else is live.
        assert!(unsafe {
            cell.query_batches::<Entity, ()>(
                &BatchingStrategy::new(),
                4,
                Tick::default(),
                Tick::default(),
            )
        }
        .is_empty());
    }

    /// The tick-tracked picture of a world: per entity its values, and
    /// which entities `Changed<A>` / `Changed<B>` select for `tracker`.
    #[allow(clippy::type_complexity)]
    fn snapshot(
        world: &World,
        es: &[Entity],
        tracker: &ChangeTracker,
    ) -> (
        Vec<((u32, u32), Option<u32>, Option<u32>, Option<u32>)>,
        Vec<(u32, u32)>,
        Vec<(u32, u32)>,
    ) {
        let mut values: Vec<_> = es
            .iter()
            .filter(|e| world.is_alive(**e))
            .map(|e| {
                (
                    key(e),
                    world.get_static::<A>(*e).map(|a| a.0),
                    world.get_static::<B>(*e).map(|b| b.0),
                    world.get_static::<C>(*e).map(|c| c.0),
                )
            })
            .collect();
        values.sort();
        let mut changed_a: Vec<_> = world
            .query_data_filtered_since::<Entity, Changed<A>>(tracker)
            .map(|e| key(&e))
            .collect();
        changed_a.sort();
        let mut changed_b: Vec<_> = world
            .query_data_filtered_since::<Entity, Changed<B>>(tracker)
            .map(|e| key(&e))
            .collect();
        changed_b.sort();
        (values, changed_a, changed_b)
    }

    /// Ticks a world forward the way the tests above do and returns the
    /// tracker standing before the writes.
    fn arm(world: &mut World) -> ChangeTracker {
        world.increment_change_tick();
        let mut tracker = ChangeTracker::new();
        tracker.update(world);
        world.increment_change_tick();
        tracker
    }

    #[test]
    fn batches_written_on_separate_threads_match_a_sequential_write() {
        let (mut seq, es) = sized_world(40, 17, 6, 5);
        let (mut par, _) = sized_world(40, 17, 6, 5);
        let seq_tracker = arm(&mut seq);
        let par_tracker = arm(&mut par);

        for (e, mut a, b) in seq.query_data_mut::<(Entity, Mut<A>, &B)>() {
            a.0 += b.0 + e.index();
        }

        let this_run = par.change_tick();
        let cell = par.as_world_cell();
        // SAFETY: the only live access to the world; each batch owns its
        // rows.
        let batches = unsafe {
            cell.query_batches::<(Entity, Mut<A>, &B), ()>(
                &BatchingStrategy::fixed(5),
                4,
                this_run,
                Tick::default(),
            )
        };
        assert!(batches.len() > 4, "the test needs real concurrency");
        std::thread::scope(|scope| {
            for batch in batches {
                scope.spawn(move || {
                    for (e, mut a, b) in batch {
                        a.0 += b.0 + e.index();
                    }
                });
            }
        });

        let (seq_values, seq_a, seq_b) = snapshot(&seq, &es, &seq_tracker);
        let (par_values, par_a, par_b) = snapshot(&par, &es, &par_tracker);
        assert_eq!(par_values, seq_values);
        assert_eq!(par_a, seq_a);
        assert_eq!(par_b, seq_b);
        assert_eq!(par_a.len(), 57, "every {{A, B}} and {{A, B, C}} row");
        assert!(par_b.is_empty(), "B was only read");
    }

    #[test]
    fn every_row_is_written_exactly_once_across_threads() {
        // `a.0 += 1` once per row: any row visited twice shows 2, any
        // skipped row 0.
        let (mut world, es) = sized_world(30, 11, 7, 0);
        let this_run = world.change_tick();
        let before: Vec<u32> = es
            .iter()
            .map(|e| world.get_static::<A>(*e).unwrap().0)
            .collect();
        let cell = world.as_world_cell();
        // SAFETY: the only live access.
        let batches = unsafe {
            cell.query_batches::<Mut<A>, ()>(
                &BatchingStrategy::fixed(4),
                4,
                this_run,
                Tick::default(),
            )
        };
        std::thread::scope(|scope| {
            for batch in batches {
                scope.spawn(move || {
                    for mut a in batch {
                        a.0 += 1;
                    }
                });
            }
        });
        for (e, old) in es.iter().zip(before) {
            assert_eq!(world.get_static::<A>(*e).unwrap().0, old + 1);
        }
    }

    #[test]
    fn a_batch_can_be_paused_and_finished_on_another_thread() {
        let (mut world, es) = sized_world(12, 0, 0, 0);
        let this_run = world.change_tick();
        let cell = world.as_world_cell();
        // SAFETY: the only live access.
        let mut batches = unsafe {
            cell.query_batches::<(Entity, Mut<A>), ()>(
                &BatchingStrategy::fixed(12),
                1,
                this_run,
                Tick::default(),
            )
        };
        assert_eq!(batches.len(), 1);
        let mut batch = batches.pop().unwrap();
        let mut first = Vec::new();
        for _ in 0..5 {
            let (e, mut a) = batch.next().unwrap();
            a.0 += 1000;
            first.push(key(&e));
        }
        let rest: Vec<(u32, u32)> = std::thread::scope(|scope| {
            scope
                .spawn(move || {
                    let mut seen = Vec::new();
                    for (e, mut a) in batch {
                        a.0 += 1000;
                        seen.push(key(&e));
                    }
                    seen
                })
                .join()
                .unwrap()
        });
        assert_eq!(first.len() + rest.len(), 12);
        let mut all: Vec<_> = first.into_iter().chain(rest).collect();
        all.sort();
        all.dedup();
        assert_eq!(all.len(), 12, "no row twice");
        for (i, e) in es.iter().enumerate() {
            assert_eq!(world.get_static::<A>(*e).unwrap().0, i as u32 + 1000);
        }
    }

    #[test]
    fn row_filters_work_inside_windows() {
        // A row filter on a different component than the one written,
        // with windows that cut the filtered rows at arbitrary places.
        let build = || {
            let (mut world, es) = sized_world(23, 10, 0, 0);
            let tracker = arm(&mut world);
            // Touch B on every third entity, after the tracker.
            for (e, mut b) in world.query_data_mut::<(Entity, Mut<B>)>() {
                if e.index() % 3 == 0 {
                    b.0 += 1;
                }
            }
            world.increment_change_tick();
            (world, es, tracker)
        };
        let (mut seq, es, seq_tracker) = build();
        let (mut par, _, par_tracker) = build();
        for mut a in seq.query_data_mut_filtered_since::<Mut<A>, Changed<B>>(&seq_tracker) {
            a.0 += 5000;
        }
        let this_run = par.change_tick();
        let last_run = par_tracker.last_run();
        for strategy in strategies() {
            // Each strategy on a fresh copy.
            let (mut par, _, par_tracker) = build();
            let this_run = par.change_tick();
            let last_run = par_tracker.last_run();
            let cell = par.as_world_cell();
            // SAFETY: the only live access.
            let batches = unsafe {
                cell.query_batches::<Mut<A>, Changed<B>>(&strategy, 3, this_run, last_run)
            };
            std::thread::scope(|scope| {
                for batch in batches {
                    scope.spawn(move || {
                        for mut a in batch {
                            a.0 += 5000;
                        }
                    });
                }
            });
            let (par_values, par_a, par_b) = snapshot(&par, &es, &par_tracker);
            let (seq_values, seq_a, seq_b) = snapshot(&seq, &es, &seq_tracker);
            assert_eq!(par_values, seq_values, "strategy {strategy:?}");
            assert_eq!(par_a, seq_a);
            assert_eq!(par_b, seq_b);
            assert_eq!(par_a.len(), 11, "entities 0, 3, 6, ... of 33");
        }
        let _ = (this_run, last_run, &mut par);
    }

    #[test]
    fn a_written_component_can_also_be_a_row_filter_in_a_window() {
        // `Mut<A>` filtered by `Changed<A>`: the filter reads the very
        // tick slice the term then takes.
        let build = || {
            let (mut world, es) = sized_world(19, 6, 0, 0);
            let tracker = arm(&mut world);
            for (e, mut a) in world.query_data_mut::<(Entity, Mut<A>)>() {
                if e.index() % 2 == 1 {
                    a.0 += 1;
                }
            }
            world.increment_change_tick();
            (world, es, tracker)
        };
        let (mut seq, es, seq_tracker) = build();
        for mut a in seq.query_data_mut_filtered_since::<Mut<A>, Changed<A>>(&seq_tracker) {
            a.0 += 7000;
        }
        let (mut par, _, par_tracker) = build();
        let this_run = par.change_tick();
        let last_run = par_tracker.last_run();
        let cell = par.as_world_cell();
        // SAFETY: the only live access.
        let batches = unsafe {
            cell.query_batches::<Mut<A>, Changed<A>>(
                &BatchingStrategy::fixed(4),
                2,
                this_run,
                last_run,
            )
        };
        std::thread::scope(|scope| {
            for batch in batches {
                scope.spawn(move || {
                    for mut a in batch {
                        a.0 += 7000;
                    }
                });
            }
        });
        let (par_values, par_a, _) = snapshot(&par, &es, &par_tracker);
        let (seq_values, seq_a, _) = snapshot(&seq, &es, &seq_tracker);
        assert_eq!(par_values, seq_values);
        assert_eq!(par_a, seq_a);
        let bumped = par_values
            .iter()
            .filter(|(_, a, _, _)| a.map_or(false, |a| a >= 7000))
            .count();
        assert_eq!(bumped, 12, "the 12 odd-indexed of 25 entities");
    }

    #[test]
    fn tick_terms_and_optional_terms_agree_with_the_sequential_iterator() {
        let (mut world, _) = sized_world(14, 9, 5, 3);
        let tracker = arm(&mut world);
        for (e, mut a) in world.query_data_mut::<(Entity, Mut<A>)>() {
            if e.index() % 4 == 0 {
                a.0 += 1;
            }
        }
        world.increment_change_tick();
        let this_run = world.change_tick();
        let last_run = tracker.last_run();
        let cell = world.as_world_cell();
        type Q<'a> = (
            Entity,
            Ref<'a, A>,
            Added<A>,
            Changed<A>,
            Option<&'a C>,
            Has<B>,
        );
        // SAFETY: only readers.
        let mut want: Vec<_> = unsafe { cell.query_iter::<Q, ()>(this_run, last_run) }
            .map(|(e, a, added, changed, c, has_b)| {
                (key(&e), a.0, added, changed, c.map(|c| c.0), has_b)
            })
            .collect();
        want.sort();
        assert!(want.iter().any(|row| row.3), "some rows changed");
        assert!(want.iter().any(|row| !row.3), "some rows did not");
        for strategy in strategies() {
            // SAFETY: only readers.
            let batches = unsafe { cell.query_batches::<Q, ()>(&strategy, 4, this_run, last_run) };
            let mut got: Vec<_> = batches
                .into_iter()
                .flatten()
                .map(|(e, a, added, changed, c, has_b)| {
                    (key(&e), a.0, added, changed, c.map(|c| c.0), has_b)
                })
                .collect();
            got.sort();
            assert_eq!(got, want, "strategy {strategy:?}");
        }
    }

    #[test]
    fn a_zero_sized_component_gets_empty_windows_and_real_stamps() {
        let mut world = World::new();
        let mut es = Vec::new();
        for i in 0..9u32 {
            let e = world.spawn();
            world.insert_static(e, Tag);
            world.insert_static(e, A(i));
            es.push(e);
        }
        let tracker = arm(&mut world);
        let this_run = world.change_tick();
        let cell = world.as_world_cell();
        // SAFETY: the only live access.
        let batches = unsafe {
            cell.query_batches::<(Mut<Tag>, Mut<A>), ()>(
                &BatchingStrategy::fixed(2),
                2,
                this_run,
                Tick::default(),
            )
        };
        let mut rows = 0;
        std::thread::scope(|scope| {
            let handles: Vec<_> = batches
                .into_iter()
                .map(|batch| {
                    scope.spawn(move || {
                        let mut n = 0;
                        for (mut tag, mut a) in batch {
                            tag.set_changed();
                            a.0 += 1;
                            n += 1;
                        }
                        n
                    })
                })
                .collect();
            for handle in handles {
                rows += handle.join().unwrap();
            }
        });
        assert_eq!(rows, 9);
        assert_eq!(
            world
                .query_data_filtered_since::<Entity, Changed<Tag>>(&tracker)
                .count(),
            9
        );
        assert_eq!(
            world
                .query_data_filtered_since::<Entity, Changed<A>>(&tracker)
                .count(),
            9
        );
        let _ = es;
    }

    #[test]
    #[should_panic(expected = "conflicting access")]
    fn a_self_conflicting_query_is_refused_before_any_batch_is_made() {
        let (mut world, _) = sized_world(3, 0, 0, 0);
        let cell = world.as_world_cell();
        // SAFETY: refused before anything is touched.
        let _ = unsafe {
            cell.query_batches::<(Mut<A>, &A), ()>(
                &BatchingStrategy::new(),
                2,
                Tick::default(),
                Tick::default(),
            )
        };
    }

    #[test]
    #[should_panic(expected = "zero threads")]
    fn zero_threads_is_refused_for_a_sized_strategy() {
        let (mut world, _) = sized_world(3, 0, 0, 0);
        let cell = world.as_world_cell();
        // SAFETY: refused before anything is touched.
        let _ = unsafe {
            cell.query_batches::<&A, ()>(
                &BatchingStrategy::new(),
                0,
                Tick::default(),
                Tick::default(),
            )
        };
    }

    /// A tiny deterministic generator, so a failure names its seed.
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0 >> 33
        }
        fn below(&mut self, n: u64) -> u64 {
            self.next() % n
        }
    }

    #[test]
    fn a_random_history_agrees_with_the_sequential_iterator_round_by_round() {
        for seed in 0..6u64 {
            let mut rng = Lcg(seed + 1);
            let (mut seq, mut es) = sized_world(17, 9, 6, 5);
            let (mut par, _) = sized_world(17, 9, 6, 5);
            let mut shapes_with_rows = [0usize; 4];
            for round in 0..8 {
                // Structural churn, identical on both worlds.
                for _ in 0..rng.below(4) {
                    let victim = es[rng.below(es.len() as u64) as usize];
                    match rng.below(3) {
                        0 => {
                            seq.despawn(victim);
                            par.despawn(victim);
                        }
                        1 => {
                            if seq.is_alive(victim) {
                                seq.insert_static(victim, C(round));
                                par.insert_static(victim, C(round));
                            }
                        }
                        _ => {
                            if seq.is_alive(victim) {
                                seq.remove_static::<B>(victim);
                                par.remove_static::<B>(victim);
                            }
                        }
                    }
                }
                let seq_tracker = arm(&mut seq);
                let par_tracker = arm(&mut par);
                // Touch some B on both, after the trackers.
                let modulus = rng.below(3) + 2;
                for world in [&mut seq, &mut par] {
                    for (e, mut b) in world.query_data_mut::<(Entity, Mut<B>)>() {
                        if u64::from(e.index()) % modulus == 0 {
                            b.0 += 1;
                        }
                    }
                    world.increment_change_tick();
                }
                let strategy =
                    strategies().swap_remove(rng.below(strategies().len() as u64) as usize);
                let threads = [1usize, 2, 4][rng.below(3) as usize];
                let shape = rng.below(4);

                let this_run = par.change_tick();
                let last_run = par_tracker.last_run();
                let cell = par.as_world_cell();
                macro_rules! write_shape {
                    ($Q:ty, $F:ty, $seq:expr, $par:expr) => {{
                        for item in seq.query_data_mut_filtered_since::<$Q, $F>(&seq_tracker) {
                            ($seq)(item);
                        }
                        // SAFETY: the only live access to `par`.
                        let batches = unsafe {
                            cell.query_batches::<$Q, $F>(&strategy, threads, this_run, last_run)
                        };
                        if batches.is_empty() {
                            // nothing matched
                        } else {
                            shapes_with_rows[shape as usize] += 1;
                        }
                        std::thread::scope(|scope| {
                            for batch in batches {
                                scope.spawn(move || {
                                    for item in batch {
                                        ($par)(item);
                                    }
                                });
                            }
                        });
                    }};
                }
                match shape {
                    0 => write_shape!(
                        (Mut<A>, &B),
                        (),
                        |(mut a, b): (Mut<A>, &B)| a.0 += b.0,
                        |(mut a, b): (Mut<A>, &B)| a.0 += b.0
                    ),
                    1 => write_shape!(
                        Mut<A>,
                        Changed<B>,
                        |mut a: Mut<A>| a.0 += 5000,
                        |mut a: Mut<A>| a.0 += 5000
                    ),
                    2 => write_shape!(
                        (Entity, Mut<B>),
                        (With<A>, Changed<B>),
                        |(e, mut b): (Entity, Mut<B>)| b.0 += e.index(),
                        |(e, mut b): (Entity, Mut<B>)| b.0 += e.index()
                    ),
                    _ => write_shape!(
                        Mut<A>,
                        crate::filter::Or<(Changed<B>, Without<B>)>,
                        |mut a: Mut<A>| a.0 += 9000,
                        |mut a: Mut<A>| a.0 += 9000
                    ),
                }
                es.retain(|e| seq.is_alive(*e));
                let (seq_values, seq_a, seq_b) = snapshot(&seq, &es, &seq_tracker);
                let (par_values, par_a, par_b) = snapshot(&par, &es, &par_tracker);
                assert_eq!(
                    par_values, seq_values,
                    "seed {seed} round {round} shape {shape}"
                );
                assert_eq!(
                    par_a, seq_a,
                    "seed {seed} round {round} shape {shape}: Changed<A>"
                );
                assert_eq!(
                    par_b, seq_b,
                    "seed {seed} round {round} shape {shape}: Changed<B>"
                );
            }
            assert!(
                shapes_with_rows.iter().all(|&n| n > 0) || seed > 0,
                "seed 0 should exercise every shape"
            );
        }
    }
}
