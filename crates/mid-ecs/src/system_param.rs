// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-ecs.md, section "System parameters: `system_param.rs`,
// `system.rs`"
// ============================================================================
//! System parameters: the things a system asks the world for by naming
//! them in its signature, and the machinery that checks they do not
//! collide.
//!
//! A [`SystemParam`] says three things about itself: what it will touch
//! (registered once, when the system is set up, into a [`SystemMeta`]), what
//! it keeps between runs ([`SystemParam::State`]), and how to produce this
//! run's value from a [`WorldCell`] ([`SystemParam::get_param`]).
//! Registration is where safety comes from: every parameter's
//! [`FilteredAccess`] is checked against the ones already registered, and a
//! system whose parameters could alias (`Query<Mut<A>>` beside `Query<&A>`,
//! `ResMut<R>` beside `Res<R>`) is refused with a panic naming the
//! conflict, before it can run. That check is what lets
//! [`SystemParam::get_param`] hand a system several live borrows of one
//! world, and what the [`WorldCell`] functions' safety contracts are
//! phrased in.
//!
//! Provided parameters: [`Query`], [`Res`] / [`ResMut`] (and their
//! `Option` forms), [`Local`], [`Commands`], and tuples of up to twelve
//! parameters (nest them for more). `SystemState` runs a parameter set
//! outside any system.

use std::any::{type_name, TypeId};
use std::borrow::Cow;
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};

use crate::access::{FilteredAccess, FilteredAccessSet};
use crate::archetype::QueryBatch;
use crate::batching::BatchingStrategy;
use crate::commands::{CommandQueue, Commands};
use crate::data_filter::DataFilter;
use crate::query_data::{QueryData, QueryEntityError, QuerySingleError, ReadOnlyQueryData};
use crate::removal::RemovalLog;
use crate::tick::{ComponentTicks, Tick};
use crate::world::{Entity, World};
use crate::world_cell::WorldCell;

// ── SystemMeta ───────────────────────────────────────────────────────

/// What the world needs to know about a system that its parameters do not
/// carry themselves: its name (for messages), everything its parameters
/// touch, and the tick it last ran at (what `Added`/`Changed` measure
/// "since" against).
#[derive(Debug)]
pub struct SystemMeta {
    name: Cow<'static, str>,
    access: FilteredAccessSet,
    last_run: Tick,
}

impl SystemMeta {
    /// An empty meta for a system called `name`. Its `last_run` is tick
    /// zero, so on its first run every component counts as added and
    /// changed.
    pub fn new(name: impl Into<Cow<'static, str>>) -> Self {
        Self {
            name: name.into(),
            access: FilteredAccessSet::new(),
            last_run: Tick::default(),
        }
    }

    /// The system's name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Everything the registered parameters touch, one part per
    /// parameter.
    pub fn access(&self) -> &FilteredAccessSet {
        &self.access
    }

    /// The tick of this system's previous run (zero before the first).
    pub fn last_run(&self) -> Tick {
        self.last_run
    }

    pub(crate) fn set_last_run(&mut self, tick: Tick) {
        self.last_run = tick;
    }

    /// Keeps the system's last-run tick within [`crate::MAX_CHANGE_AGE`] of
    /// `now` (see [`Tick::check_tick`]); whether it was clamped.
    pub fn check_change_tick(&mut self, now: Tick) -> bool {
        self.last_run.check_tick(now)
    }

    /// Registers one parameter's access.
    ///
    /// # Panics
    /// If it conflicts with a parameter already registered: the system
    /// would alias data. The message names the system and the conflicting
    /// types.
    pub fn add_param_access(&mut self, part: FilteredAccess) {
        let conflicts = self.access.get_conflicts_single(&part);
        assert!(
            conflicts.is_empty(),
            "system `{}` has parameters that conflict on {}: a component or resource can be read by any number of parameters, or written by exactly one and read by none",
            self.name,
            conflicts
        );
        self.access.add(part);
    }
}

// ── The trait ────────────────────────────────────────────────────────

/// Something a system can take as a parameter.
///
/// # Safety
/// An implementation must register, in [`Self::init_state`], *every*
/// access that [`Self::get_param`]'s result can make to the world
/// (through [`SystemMeta::add_param_access`]), and the result must not make
/// any other. The unsafe functions of [`WorldCell`] and the guarantees of
/// every other parameter rest on that being true.
pub unsafe trait SystemParam: Sized {
    /// What the parameter keeps between runs of its system.
    type State: 'static;

    /// The parameter as the system's function receives it: `Self` with
    /// its lifetimes filled in (`'w` for what it borrows from the world,
    /// `'s` for what it borrows from its own state).
    type Item<'w, 's>: SystemParam<State = Self::State>;

    /// Builds the state, once, and registers the parameter's access in
    /// `meta`.
    fn init_state(world: &mut World, meta: &mut SystemMeta) -> Self::State;

    /// This run's value.
    ///
    /// # Safety
    /// For as long as the returned item (or anything taken from it) is
    /// live, nothing may access the world in a way that conflicts with the
    /// access this parameter registered in `meta`, and `meta` must be the
    /// one `init_state` registered into. In practice: the system was
    /// initialized, and no other system that conflicts with it runs at
    /// the same time.
    unsafe fn get_param<'w, 's>(
        state: &'s mut Self::State,
        meta: &SystemMeta,
        world: WorldCell<'w>,
        this_run: Tick,
    ) -> Self::Item<'w, 's>;

    /// Runs after the system, with the whole world: where deferred work
    /// ([`Commands`]) happens. Does nothing by default.
    fn apply(_state: &mut Self::State, _meta: &SystemMeta, _world: &mut World) {}
}

/// `P`'s value for one run.
pub type SystemParamItem<'w, 's, P> = <P as SystemParam>::Item<'w, 's>;

// ── RemovedComponents ────────────────────────────────────────────────

/// The entities that lost a component of type `T` since this system last
/// ran (or, on its first run, since removals of `T` began to be recorded,
/// which is when the system was set up).
///
/// "Lost" means removed (`remove`, `remove_static`, a bundle, the FFI byte
/// paths, through `Commands` or directly) or taken with a despawned entity,
/// in either storage. Replacing a value by inserting over it is not a
/// removal. An entity removed twice appears twice; it may since have been
/// despawned or its slot reused, so check `World::is_alive` through a
/// query if that matters.
///
/// The window is the system's own previous run, as for `Added`/`Changed`:
/// removals queued by this system's own `Commands` are applied after its
/// run and so are seen on its next one. Nothing trims the log: see
/// [`World::trim_removals`].
///
/// ```
/// use mid_ecs::{IntoSystem, RemovedComponents, System, World};
///
/// struct Health(u32);
///
/// fn reaper(removed: RemovedComponents<Health>) {
///     for entity in removed.iter() {
///         let _ = entity; // clean up whatever pointed at it
///     }
/// }
///
/// let mut world = World::new();
/// let e = world.spawn();
/// world.insert_static(e, Health(3));
/// let mut system = reaper.into_system();
/// system.initialize(&mut world);
/// world.remove_static::<Health>(e);
/// system.run(&mut world);
/// ```
pub struct RemovedComponents<'w, T: 'static> {
    log: &'w RemovalLog,
    last_run: Tick,
    this_run: Tick,
    _marker: PhantomData<fn() -> T>,
}

impl<T: 'static> RemovedComponents<'_, T> {
    /// The entities, oldest removal first.
    pub fn iter(&self) -> impl Iterator<Item = Entity> + '_ {
        self.log
            .since(TypeId::of::<T>(), self.last_run, self.this_run)
    }

    /// How many removals there are to read (not how many distinct
    /// entities).
    pub fn len(&self) -> usize {
        self.iter().count()
    }

    /// Whether there is nothing to read.
    pub fn is_empty(&self) -> bool {
        self.iter().next().is_none()
    }
}

// SAFETY: reads only the removal log, which is written only by structural
// changes (`&mut World`: Commands applied between runs, or an exclusive
// system, which conflicts with every parameter). There is no component or
// resource access to register, so nothing can conflict with it.
unsafe impl<T: 'static> SystemParam for RemovedComponents<'_, T> {
    type State = ();
    type Item<'w, 's> = RemovedComponents<'w, T>;

    fn init_state(world: &mut World, _meta: &mut SystemMeta) -> Self::State {
        world.track_removals::<T>();
    }

    unsafe fn get_param<'w, 's>(
        _state: &'s mut Self::State,
        meta: &SystemMeta,
        world: WorldCell<'w>,
        this_run: Tick,
    ) -> Self::Item<'w, 's> {
        RemovedComponents {
            // SAFETY: as for the impl.
            log: unsafe { world.removals() },
            last_run: meta.last_run,
            this_run,
            _marker: PhantomData,
        }
    }
}

// ── Query ────────────────────────────────────────────────────────────

/// Query data `Q` (terms: `&T`, `Mut<T>`, `Ref<T>`, `Option<..>`, `Has<T>`,
/// `Entity`, `Added<T>`, `Changed<T>`, tuples) filtered by `F`
/// ([`crate::With`], [`crate::Without`], [`crate::Or`], `Added`,
/// `Changed`, tuples), as a system parameter. `Added`/`Changed` measure
/// against the system's own previous run.
///
/// Other parameters of the same system that touch the same components are
/// refused when the system is set up, so a `Query` can borrow while they
/// do:
///
/// ```
/// use mid_ecs::{Mut, Query, SystemState, World};
///
/// struct Position(f32);
/// struct Velocity(f32);
///
/// let mut world = World::new();
/// let e = world.spawn();
/// world.insert_static(e, Position(0.0));
/// world.insert_static(e, Velocity(2.0));
///
/// let mut state = SystemState::<(Query<Mut<Position>>, Query<&Velocity>)>::new(&mut world);
/// let (mut positions, velocities) = state.get(&mut world);
/// for v in velocities.iter() {
///     for mut p in positions.iter_mut() {
///         p.0 += v.0;
///     }
/// }
/// drop((positions, velocities));
/// assert_eq!(world.get_static::<Position>(e).map(|p| p.0), Some(2.0));
/// ```
///
/// One `Query` cannot be iterated twice at once (the second iterator
/// could hand out the same `&mut`):
///
/// ```compile_fail
/// use mid_ecs::{Mut, Query, SystemState, World};
/// struct Position(f32);
/// let mut world = World::new();
/// let mut state = SystemState::<Query<Mut<Position>>>::new(&mut world);
/// let mut q = state.get(&mut world);
/// let first = q.iter_mut();
/// let second = q.iter_mut(); // error: `q` is already borrowed mutably
/// drop((first, second));
/// ```
pub struct Query<'w, 's, Q: QueryData + 'static, F: DataFilter + 'static = ()> {
    world: WorldCell<'w>,
    this_run: Tick,
    last_run: Tick,
    _marker: PhantomData<(&'s (), fn() -> (Q, F))>,
}

impl<'w, 's, Q: QueryData + 'static, F: DataFilter + 'static> Query<'w, 's, Q, F> {
    /// Iterates the rows `Q` matches, reading. Any number of read-only
    /// iterations (and other read-only parameters) may be live together.
    pub fn iter<'a>(&'a self) -> impl Iterator<Item = Q::Item<'a>> + 'a
    where
        Q: ReadOnlyQueryData,
    {
        let cell: WorldCell<'a> = self.world;
        // SAFETY: this parameter's access, registered when the system was
        // set up, was checked against every other parameter's, so nothing
        // that conflicts with a read of `Q` is live; `&self` stops a
        // writing iteration of this same query from starting meanwhile.
        unsafe { cell.query_iter::<Q, F>(self.this_run, self.last_run) }
    }

    /// Iterates the rows `Q` matches, with write access to the `Mut`
    /// terms. The iterator borrows the query exclusively, so it cannot
    /// be started twice at once.
    pub fn iter_mut<'a>(&'a mut self) -> impl Iterator<Item = Q::Item<'a>> + 'a {
        let cell: WorldCell<'a> = self.world;
        // SAFETY: as for `iter`, for the access `Q` makes (writes
        // included); `&mut self` makes this the only iteration of this
        // query.
        unsafe { cell.query_iter_mut::<Q, F>(self.this_run, self.last_run) }
    }

    /// Iterates the rows `Q` matches on several threads at once, reading.
    /// Needs a read-only `Q` whose component types are `Sync` (checked
    /// where the iteration is run: see [`QueryParIter::for_each`]).
    pub fn par_iter(&self) -> QueryParIter<'_, Q, F>
    where
        Q: ReadOnlyQueryData,
    {
        QueryParIter {
            world: self.world,
            this_run: self.this_run,
            last_run: self.last_run,
            strategy: BatchingStrategy::new(),
            _marker: PhantomData,
        }
    }

    /// Iterates the rows `Q` matches on several threads at once, with
    /// write access to the `Mut` terms. Borrows the query exclusively,
    /// like [`Self::iter_mut`]. Component types written through `Mut`
    /// must be `Send`, read ones `Sync`.
    pub fn par_iter_mut(&mut self) -> QueryParIter<'_, Q, F> {
        QueryParIter {
            world: self.world,
            this_run: self.this_run,
            last_run: self.last_run,
            strategy: BatchingStrategy::new(),
            _marker: PhantomData,
        }
    }

    /// The item the query yields for `entity`, reading: the by-entity form
    /// of [`Self::iter`]. Every rule of an iteration holds: the entity's
    /// archetype must hold what the terms need and pass `With` / `Without` /
    /// `Or`, and a row-level `Added` / `Changed` filter must keep its row.
    ///
    /// Errors: [`QueryEntityError::NoSuchEntity`] if the entity is not alive
    /// (a stale handle whose slot was reused is not alive, and never finds
    /// the new occupant), [`QueryEntityError::QueryDoesNotMatch`] if it is
    /// alive but the query does not match it.
    ///
    /// Each call resolves the query's component ids and builds a one-row
    /// window, so it costs more than a step of an iteration; it is not
    /// benchmarked (optimization is paused). Cached query state (gap-list
    /// item 8) would take the setup out of the call.
    pub fn get<'a>(&'a self, entity: Entity) -> Result<Q::Item<'a>, QueryEntityError>
    where
        Q: ReadOnlyQueryData,
    {
        let cell: WorldCell<'a> = self.world;
        // SAFETY: as for `iter`: this parameter's registered access was
        // checked against every other parameter's, and `&self` stops a
        // writing lookup of this same query from starting meanwhile.
        unsafe { cell.query_get::<Q, F>(entity, self.this_run, self.last_run) }
    }

    /// The item the query yields for `entity`, with write access to the
    /// `Mut` terms (stamped with the system's tick when written through).
    /// Borrows the query exclusively, like [`Self::iter_mut`], so a second
    /// `get_mut` cannot start while the first item is live; for several
    /// entities at once use [`Self::get_many_mut`]. Errors as for
    /// [`Self::get`].
    ///
    /// Two lookups of one entity at once would hand out two `&mut` to one
    /// row, and the borrow checker refuses them:
    ///
    /// ```compile_fail
    /// use mid_ecs::{Mut, Query, SystemState, World};
    /// struct Hp(u32);
    /// let mut world = World::new();
    /// let e = world.spawn();
    /// world.insert_static(e, Hp(1));
    /// let mut state = SystemState::<Query<Mut<Hp>>>::new(&mut world);
    /// let mut q = state.get(&mut world);
    /// let first = q.get_mut(e);
    /// let second = q.get_mut(e); // error: `q` is already borrowed mutably
    /// drop((first, second));
    /// ```
    pub fn get_mut<'a>(&'a mut self, entity: Entity) -> Result<Q::Item<'a>, QueryEntityError> {
        let cell: WorldCell<'a> = self.world;
        // SAFETY: as for `iter_mut`, for the access `Q` makes; `&mut self`
        // makes this the only lookup or iteration of this query while the
        // item is live.
        unsafe { cell.query_get_mut::<Q, F>(entity, self.this_run, self.last_run) }
    }

    /// The items for several entities at once, reading. A repeated entity
    /// is fine here (readers can share). The first entity that cannot be
    /// found is the error.
    pub fn get_many<'a, const N: usize>(
        &'a self,
        entities: [Entity; N],
    ) -> Result<[Q::Item<'a>; N], QueryEntityError>
    where
        Q: ReadOnlyQueryData,
    {
        let mut items = Vec::with_capacity(N);
        for entity in entities {
            items.push(self.get(entity)?);
        }
        match <[Q::Item<'a>; N]>::try_from(items) {
            Ok(array) => Ok(array),
            Err(_) => unreachable!("exactly N items were pushed"),
        }
    }

    /// The items for several entities at once, with write access to the
    /// `Mut` terms. The entities must be pairwise distinct (a repeat is
    /// [`QueryEntityError::AliasedMutability`], checked before anything is
    /// looked up), so no row is handed out twice. The first entity that
    /// cannot be found is the error.
    pub fn get_many_mut<'a, const N: usize>(
        &'a mut self,
        entities: [Entity; N],
    ) -> Result<[Q::Item<'a>; N], QueryEntityError> {
        for i in 0..N {
            for j in 0..i {
                if entities[i] == entities[j] {
                    return Err(QueryEntityError::AliasedMutability(entities[i]));
                }
            }
        }
        let cell: WorldCell<'a> = self.world;
        let mut items = Vec::with_capacity(N);
        for entity in entities {
            // SAFETY: as for `get_mut`; the entities are pairwise distinct,
            // so no two items share a row, and `&mut self` makes this the
            // only lookup or iteration of this query.
            items.push(unsafe { cell.query_get_mut::<Q, F>(entity, self.this_run, self.last_run) }?);
        }
        match <[Q::Item<'a>; N]>::try_from(items) {
            Ok(array) => Ok(array),
            Err(_) => unreachable!("exactly N items were pushed"),
        }
    }

    /// The items for the entities `entities` lists, in that order,
    /// reading. An entity that is not alive or that the query does not
    /// match is skipped, not an error (use [`Self::get`] to learn which).
    pub fn iter_many<'a, I>(&'a self, entities: I) -> impl Iterator<Item = Q::Item<'a>> + 'a
    where
        Q: ReadOnlyQueryData,
        I: IntoIterator<Item = Entity>,
        I::IntoIter: 'a,
    {
        let cell: WorldCell<'a> = self.world;
        let (this_run, last_run) = (self.this_run, self.last_run);
        entities.into_iter().filter_map(move |entity| {
            // SAFETY: as for `get`.
            unsafe { cell.query_get::<Q, F>(entity, this_run, last_run) }.ok()
        })
    }

    /// The one item the query yields, reading, or
    /// [`QuerySingleError::NoEntities`] / [`QuerySingleError::MultipleEntities`].
    pub fn single<'a>(&'a self) -> Result<Q::Item<'a>, QuerySingleError>
    where
        Q: ReadOnlyQueryData,
    {
        let mut rows = self.iter();
        let first = rows.next().ok_or(QuerySingleError::NoEntities)?;
        match rows.next() {
            None => Ok(first),
            Some(_) => Err(QuerySingleError::MultipleEntities),
        }
    }

    /// The one item the query yields, with write access to the `Mut`
    /// terms; errors as for [`Self::single`]. Looking for the second item
    /// to rule out `MultipleEntities` writes nothing and stamps nothing.
    pub fn single_mut<'a>(&'a mut self) -> Result<Q::Item<'a>, QuerySingleError> {
        let mut rows = self.iter_mut();
        let first = rows.next().ok_or(QuerySingleError::NoEntities)?;
        match rows.next() {
            None => Ok(first),
            Some(_) => Err(QuerySingleError::MultipleEntities),
        }
    }

    /// Whether the query matches `entity` (alive, in a matching archetype,
    /// past any row-level filter), without fetching anything: the same
    /// answer for a reading and a writing query, and it takes no borrow of
    /// the data.
    pub fn contains(&self, entity: Entity) -> bool {
        let cell: WorldCell<'_> = self.world;
        // SAFETY: reads change ticks only, and only for a row-level
        // filter; those reads are part of this query's registered access.
        unsafe { cell.query_contains::<Q, F>(entity, self.this_run, self.last_run) }
    }

    /// How many rows the query matches, counted without fetching anything
    /// (the same number `iter().count()` gives, for a writing query too).
    pub fn count(&self) -> usize {
        let cell: WorldCell<'_> = self.world;
        // SAFETY: as for `contains`.
        unsafe { cell.query_count::<Q, F>(self.this_run, self.last_run) }
    }

    /// Whether the query matches no row.
    pub fn is_empty(&self) -> bool {
        self.count() == 0
    }
}

/// A parallel iteration over a [`Query`], from [`Query::par_iter`] or
/// [`Query::par_iter_mut`]. Nothing runs until [`Self::for_each`] (or
/// `for_each_init`, or `into_batches`).
///
/// The matched archetypes are cut into batches of row windows (see
/// [`BatchingStrategy`]); each batch runs on one thread from start to
/// finish, so rows within a batch are visited in order and no two threads
/// ever touch the same row. The order *between* batches is not defined.
/// Writes through `Mut` are stamped with the system's tick exactly as in
/// the sequential [`Query::iter_mut`], and `Added`/`Changed` terms and
/// filters work the same way.
///
/// On `wasm32` (no threads) `for_each` runs the batches one after another
/// on the calling thread.
///
/// A component type that is not `Send` (for `Mut`) or not `Sync` (for a
/// read term) cannot be iterated in parallel; the compiler refuses:
///
/// ```compile_fail
/// use mid_ecs::{Mut, Query, SystemState, World};
/// use std::rc::Rc;
///
/// struct NotSend(Rc<u32>);
///
/// let mut world = World::new();
/// let mut state = SystemState::<Query<Mut<NotSend>>>::new(&mut world);
/// let mut q = state.get(&mut world);
/// q.par_iter_mut().for_each(|_item| {});
/// ```
///
/// ```compile_fail
/// use mid_ecs::{Query, SystemState, World};
/// use std::cell::Cell;
///
/// struct NotSync(Cell<u32>);
///
/// let mut world = World::new();
/// let mut state = SystemState::<Query<&NotSync>>::new(&mut world);
/// let q = state.get(&mut world);
/// q.par_iter().for_each(|_item| {});
/// ```
pub struct QueryParIter<'a, Q: QueryData + 'static, F: DataFilter + 'static> {
    world: WorldCell<'a>,
    this_run: Tick,
    last_run: Tick,
    strategy: BatchingStrategy,
    _marker: PhantomData<fn() -> (Q, F)>,
}

impl<'a, Q: QueryData + 'static, F: DataFilter + 'static> QueryParIter<'a, Q, F> {
    /// Replaces the default batching (about one batch per thread for the
    /// largest archetype). See [`BatchingStrategy`].
    pub fn batching_strategy(mut self, strategy: BatchingStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// The query cut into independent [`QueryBatch`]es, for running on a
    /// thread pool of your own instead of the built-in one. Each batch is
    /// an iterator of this query's items for its rows; batches are
    /// pairwise disjoint and together cover every matched row once.
    /// Batches are `Send` when the component types allow it.
    pub fn into_batches(self) -> Vec<QueryBatch<'a, Q, F>> {
        // SAFETY: the `Query` this came from borrowed itself (`&` for a
        // read-only `Q`, `&mut` otherwise) for `'a`, and its access was
        // checked against the system's other parameters when the system
        // was set up, exactly as for `iter`/`iter_mut`; the world's
        // structure cannot change meanwhile (a system's parameters only
        // queue structural changes, in `Commands`).
        unsafe {
            self.world.query_batches::<Q, F>(
                &self.strategy,
                pool_threads(),
                self.this_run,
                self.last_run,
            )
        }
    }

    /// Calls `func` on every row, on the rayon thread pool.
    pub fn for_each<FN>(self, func: FN)
    where
        FN: Fn(Q::Item<'a>) + Send + Sync,
        Q::Fetch<'a>: Send,
    {
        self.for_each_init(|| (), |(), item| func(item));
    }

    /// Like [`Self::for_each`], with a per-batch scratch value: `init`
    /// runs once at the start of each batch and the result is passed to
    /// `func` for every row of that batch.
    pub fn for_each_init<T, INIT, FN>(self, init: INIT, func: FN)
    where
        INIT: Fn() -> T + Send + Sync,
        FN: Fn(&mut T, Q::Item<'a>) + Send + Sync,
        Q::Fetch<'a>: Send,
    {
        let batches = self.into_batches();
        run_batches::<Q, F, T, INIT, FN>(batches, &init, &func);
    }
}

/// How many threads a parallel query should size its batches for.
#[cfg(not(target_arch = "wasm32"))]
fn pool_threads() -> usize {
    rayon::current_num_threads()
}

#[cfg(target_arch = "wasm32")]
fn pool_threads() -> usize {
    1
}

/// Runs every batch, one per rayon task. A single batch (or none) runs on
/// the calling thread: no task to start.
#[cfg(not(target_arch = "wasm32"))]
fn run_batches<'a, Q, F, T, INIT, FN>(batches: Vec<QueryBatch<'a, Q, F>>, init: &INIT, func: &FN)
where
    Q: QueryData + 'static,
    F: DataFilter + 'static,
    Q::Fetch<'a>: Send,
    INIT: Fn() -> T + Sync,
    FN: Fn(&mut T, Q::Item<'a>) + Sync,
{
    use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

    let run = |batch: QueryBatch<'a, Q, F>| {
        let mut scratch = init();
        for item in batch {
            func(&mut scratch, item);
        }
    };
    if batches.len() <= 1 {
        batches.into_iter().for_each(run);
    } else {
        // `with_max_len(1)`: every batch is its own unit of stealing, so a
        // slow batch cannot hold back others queued behind it.
        batches.into_par_iter().with_max_len(1).for_each(run);
    }
}

#[cfg(target_arch = "wasm32")]
fn run_batches<'a, Q, F, T, INIT, FN>(batches: Vec<QueryBatch<'a, Q, F>>, init: &INIT, func: &FN)
where
    Q: QueryData + 'static,
    F: DataFilter + 'static,
    INIT: Fn() -> T,
    FN: Fn(&mut T, Q::Item<'a>),
{
    for batch in batches {
        let mut scratch = init();
        for item in batch {
            func(&mut scratch, item);
        }
    }
}

// SAFETY: registers exactly `FilteredAccess::of_query::<Q, F>()`, the
// access `query_iter` / `query_iter_mut` make.
unsafe impl<Q: QueryData + 'static, F: DataFilter + 'static> SystemParam for Query<'_, '_, Q, F> {
    type State = ();
    type Item<'w, 's> = Query<'w, 's, Q, F>;

    fn init_state(_world: &mut World, meta: &mut SystemMeta) -> Self::State {
        meta.add_param_access(FilteredAccess::of_query::<Q, F>());
    }

    unsafe fn get_param<'w, 's>(
        _state: &'s mut Self::State,
        meta: &SystemMeta,
        world: WorldCell<'w>,
        this_run: Tick,
    ) -> Self::Item<'w, 's> {
        Query {
            world,
            this_run,
            last_run: meta.last_run(),
            _marker: PhantomData,
        }
    }
}

// ── Res / ResMut ─────────────────────────────────────────────────────

/// A resource, shared. Panics when the system runs if the resource does
/// not exist; use `Option<Res<T>>` to tolerate that.
///
/// Besides the value it carries the resource's change ticks and the
/// system's last-run tick, so [`Self::is_added`] and [`Self::is_changed`]
/// answer "since this system last ran" with no argument, as in `bevy_ecs`.
/// (A component's `Ref` / `Mut` take a [`crate::ChangeTracker`] instead,
/// because they can also come from a world query that has no system.)
pub struct Res<'w, T: 'static> {
    value: &'w T,
    ticks: &'w ComponentTicks,
    last_run: Tick,
    this_run: Tick,
}

impl<T: 'static> Res<'_, T> {
    /// Whether the resource was inserted since this system last ran (on its
    /// first run, whether it exists at all).
    pub fn is_added(&self) -> bool {
        self.ticks.is_added(self.last_run, self.this_run)
    }

    /// Whether the resource was inserted, replaced, or written through a
    /// `ResMut` (or `World::get_resource_mut`) since this system last ran.
    pub fn is_changed(&self) -> bool {
        self.ticks.is_changed(self.last_run, self.this_run)
    }

    /// The tick the resource was last changed at.
    pub fn last_changed(&self) -> Tick {
        self.ticks.changed
    }

    /// The tick the resource was added at.
    pub fn added(&self) -> Tick {
        self.ticks.added
    }
}

impl<T: 'static> Deref for Res<'_, T> {
    type Target = T;

    fn deref(&self) -> &T {
        self.value
    }
}

/// A resource, exclusive. Panics when the system runs if the resource
/// does not exist; use `Option<ResMut<T>>` to tolerate that.
///
/// Reading through it ([`Deref`]) leaves the resource's change tick alone;
/// *mutably* dereferencing it ([`DerefMut`]: any `*r = ..`, `r.field =
/// ..`, `&mut *r`) stamps the resource changed at this run's tick, which
/// is what other systems' `Res::is_changed` read. A system that takes a
/// `ResMut` and only reads costs nobody a change.
pub struct ResMut<'w, T: 'static> {
    value: &'w mut T,
    ticks: &'w mut ComponentTicks,
    last_run: Tick,
    this_run: Tick,
}

impl<T: 'static> ResMut<'_, T> {
    /// Whether the resource was inserted since this system last ran.
    pub fn is_added(&self) -> bool {
        self.ticks.is_added(self.last_run, self.this_run)
    }

    /// Whether the resource was changed since this system last ran. A
    /// write this system made earlier in this same run is not "since the
    /// last run"; it counts from the next run of anything that compares
    /// against this run's tick.
    pub fn is_changed(&self) -> bool {
        self.ticks.is_changed(self.last_run, self.this_run)
    }

    /// The tick the resource was last changed at.
    pub fn last_changed(&self) -> Tick {
        self.ticks.changed
    }

    /// The tick the resource was added at.
    pub fn added(&self) -> Tick {
        self.ticks.added
    }

    /// Marks the resource changed without writing to it, for a mutation
    /// made some other way (interior mutability, say).
    pub fn set_changed(&mut self) {
        self.ticks.changed = self.this_run;
    }

    /// Mutable access that does *not* mark the resource changed: for a
    /// write no change-detecting reader needs to see. The caller is
    /// asserting exactly that.
    pub fn bypass_change_detection(&mut self) -> &mut T {
        self.value
    }
}

impl<T: 'static> Deref for ResMut<'_, T> {
    type Target = T;

    fn deref(&self) -> &T {
        self.value
    }
}

impl<T: 'static> DerefMut for ResMut<'_, T> {
    fn deref_mut(&mut self) -> &mut T {
        self.ticks.changed = self.this_run;
        self.value
    }
}

fn register_resource<T: 'static>(meta: &mut SystemMeta, write: bool) {
    let mut part = FilteredAccess::new();
    if write {
        part.add_resource_write::<T>();
    } else {
        part.add_resource_read::<T>();
    }
    meta.add_param_access(part);
}

fn missing_resource<T: 'static>(meta: &SystemMeta) -> ! {
    panic!(
        "system `{}`: resource `{}` does not exist (insert it first, or ask for `Option<..>` instead)",
        meta.name(),
        type_name::<T>()
    )
}

// SAFETY: registers a read of resource `T`, the only thing it touches.
unsafe impl<T: 'static> SystemParam for Res<'_, T> {
    type State = ();
    type Item<'w, 's> = Res<'w, T>;

    fn init_state(_world: &mut World, meta: &mut SystemMeta) -> Self::State {
        register_resource::<T>(meta, false);
    }

    unsafe fn get_param<'w, 's>(
        _state: &'s mut Self::State,
        meta: &SystemMeta,
        world: WorldCell<'w>,
        this_run: Tick,
    ) -> Self::Item<'w, 's> {
        // SAFETY: the read was registered and checked against every other
        // parameter, so nothing writes `T` meanwhile.
        match unsafe { world.resource_with_ticks::<T>() } {
            Some((value, ticks)) => Res {
                value,
                ticks,
                last_run: meta.last_run(),
                this_run,
            },
            None => missing_resource::<T>(meta),
        }
    }
}

// SAFETY: registers a read of resource `T`, the only thing it touches.
unsafe impl<T: 'static> SystemParam for Option<Res<'_, T>> {
    type State = ();
    type Item<'w, 's> = Option<Res<'w, T>>;

    fn init_state(_world: &mut World, meta: &mut SystemMeta) -> Self::State {
        register_resource::<T>(meta, false);
    }

    unsafe fn get_param<'w, 's>(
        _state: &'s mut Self::State,
        meta: &SystemMeta,
        world: WorldCell<'w>,
        this_run: Tick,
    ) -> Self::Item<'w, 's> {
        // SAFETY: as for `Res`.
        unsafe { world.resource_with_ticks::<T>() }.map(|(value, ticks)| Res {
            value,
            ticks,
            last_run: meta.last_run(),
            this_run,
        })
    }
}

// SAFETY: registers a write of resource `T`, the only thing it touches.
unsafe impl<T: 'static> SystemParam for ResMut<'_, T> {
    type State = ();
    type Item<'w, 's> = ResMut<'w, T>;

    fn init_state(_world: &mut World, meta: &mut SystemMeta) -> Self::State {
        register_resource::<T>(meta, true);
    }

    unsafe fn get_param<'w, 's>(
        _state: &'s mut Self::State,
        meta: &SystemMeta,
        world: WorldCell<'w>,
        this_run: Tick,
    ) -> Self::Item<'w, 's> {
        // SAFETY: the write was registered and checked against every
        // other parameter, so nothing else reads or writes `T` meanwhile.
        match unsafe { world.resource_mut_with_ticks::<T>() } {
            Some((value, ticks)) => ResMut {
                value,
                ticks,
                last_run: meta.last_run(),
                this_run,
            },
            None => missing_resource::<T>(meta),
        }
    }
}

// SAFETY: registers a write of resource `T`, the only thing it touches.
unsafe impl<T: 'static> SystemParam for Option<ResMut<'_, T>> {
    type State = ();
    type Item<'w, 's> = Option<ResMut<'w, T>>;

    fn init_state(_world: &mut World, meta: &mut SystemMeta) -> Self::State {
        register_resource::<T>(meta, true);
    }

    unsafe fn get_param<'w, 's>(
        _state: &'s mut Self::State,
        meta: &SystemMeta,
        world: WorldCell<'w>,
        this_run: Tick,
    ) -> Self::Item<'w, 's> {
        // SAFETY: as for `ResMut`.
        unsafe { world.resource_mut_with_ticks::<T>() }.map(|(value, ticks)| ResMut {
            value,
            ticks,
            last_run: meta.last_run(),
            this_run,
        })
    }
}

// ── Local ────────────────────────────────────────────────────────────

/// A value that belongs to one system and lives across its runs,
/// starting as `T::default()`. Two systems (or two `Local<T>` in one
/// system) each get their own.
pub struct Local<'s, T: 'static> {
    value: &'s mut T,
}

impl<T: 'static> Deref for Local<'_, T> {
    type Target = T;

    fn deref(&self) -> &T {
        self.value
    }
}

impl<T: 'static> DerefMut for Local<'_, T> {
    fn deref_mut(&mut self) -> &mut T {
        self.value
    }
}

// SAFETY: touches nothing in the world, so registers nothing.
unsafe impl<T: Default + 'static> SystemParam for Local<'_, T> {
    type State = T;
    type Item<'w, 's> = Local<'s, T>;

    fn init_state(_world: &mut World, _meta: &mut SystemMeta) -> Self::State {
        T::default()
    }

    unsafe fn get_param<'w, 's>(
        state: &'s mut Self::State,
        _meta: &SystemMeta,
        _world: WorldCell<'w>,
        _this_run: Tick,
    ) -> Self::Item<'w, 's> {
        Local { value: state }
    }
}

// ── Commands ─────────────────────────────────────────────────────────

// SAFETY: records into the parameter's own queue and touches the world
// not at all until `apply`, which has `&mut World`.
unsafe impl SystemParam for Commands<'_> {
    type State = CommandQueue;
    type Item<'w, 's> = Commands<'s>;

    fn init_state(_world: &mut World, _meta: &mut SystemMeta) -> Self::State {
        CommandQueue::new()
    }

    unsafe fn get_param<'w, 's>(
        state: &'s mut Self::State,
        _meta: &SystemMeta,
        _world: WorldCell<'w>,
        _this_run: Tick,
    ) -> Self::Item<'w, 's> {
        state.commands()
    }

    fn apply(state: &mut Self::State, _meta: &SystemMeta, world: &mut World) {
        state.apply(world);
    }
}

// ── Tuples ───────────────────────────────────────────────────────────

// SAFETY: touches nothing.
unsafe impl SystemParam for () {
    type State = ();
    type Item<'w, 's> = ();

    fn init_state(_world: &mut World, _meta: &mut SystemMeta) -> Self::State {}

    unsafe fn get_param<'w, 's>(
        _state: &'s mut Self::State,
        _meta: &SystemMeta,
        _world: WorldCell<'w>,
        _this_run: Tick,
    ) -> Self::Item<'w, 's> {
    }
}

macro_rules! impl_system_param_for_tuple {
    ($($P:ident),+) => {
        // SAFETY: registers, in order, what each member registers, and
        // each member's value makes only the access it registered.
        unsafe impl<$($P: SystemParam),+> SystemParam for ($($P,)+) {
            type State = ($($P::State,)+);
            type Item<'w, 's> = ($($P::Item<'w, 's>,)+);

            fn init_state(world: &mut World, meta: &mut SystemMeta) -> Self::State {
                ($($P::init_state(world, meta),)+)
            }

            #[allow(non_snake_case)]
            unsafe fn get_param<'w, 's>(
                state: &'s mut Self::State,
                meta: &SystemMeta,
                world: WorldCell<'w>,
                this_run: Tick,
            ) -> Self::Item<'w, 's> {
                let ($($P,)+) = state;
                // SAFETY: forwarded; each member's access was checked
                // against every other's when the tuple was initialized.
                ($(unsafe { $P::get_param($P, meta, world, this_run) },)+)
            }

            #[allow(non_snake_case)]
            fn apply(state: &mut Self::State, meta: &SystemMeta, world: &mut World) {
                let ($($P,)+) = state;
                $($P::apply($P, meta, world);)+
            }
        }
    };
}

impl_system_param_for_tuple!(P0);
impl_system_param_for_tuple!(P0, P1);
impl_system_param_for_tuple!(P0, P1, P2);
impl_system_param_for_tuple!(P0, P1, P2, P3);
impl_system_param_for_tuple!(P0, P1, P2, P3, P4);
impl_system_param_for_tuple!(P0, P1, P2, P3, P4, P5);
impl_system_param_for_tuple!(P0, P1, P2, P3, P4, P5, P6);
impl_system_param_for_tuple!(P0, P1, P2, P3, P4, P5, P6, P7);
impl_system_param_for_tuple!(P0, P1, P2, P3, P4, P5, P6, P7, P8);
impl_system_param_for_tuple!(P0, P1, P2, P3, P4, P5, P6, P7, P8, P9);
impl_system_param_for_tuple!(P0, P1, P2, P3, P4, P5, P6, P7, P8, P9, P10);
impl_system_param_for_tuple!(P0, P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11);

// ── SystemState ──────────────────────────────────────────────────────

/// A parameter set `P` initialized against a world, usable outside any
/// system: build it once with [`Self::new`], then [`Self::get`] the
/// parameters whenever needed.
///
/// `get` borrows the world exclusively for as long as its result is
/// held, so the parameters cannot be used beside other access to the
/// world; what they share is checked, once, in `new`. Changes recorded
/// through `Commands` wait until [`Self::apply`].
pub struct SystemState<P: SystemParam + 'static> {
    meta: SystemMeta,
    state: P::State,
}

impl<P: SystemParam + 'static> SystemState<P> {
    /// Initializes `P` against `world`.
    ///
    /// # Panics
    /// If two of `P`'s parameters conflict (see
    /// [`SystemMeta::add_param_access`]).
    pub fn new(world: &mut World) -> Self {
        let mut meta = SystemMeta::new(type_name::<P>());
        let state = P::init_state(world, &mut meta);
        Self { meta, state }
    }

    /// The meta `new` built: the name and the registered access.
    pub fn meta(&self) -> &SystemMeta {
        &self.meta
    }

    /// Keeps the last-call tick within [`crate::MAX_CHANGE_AGE`] of `now`;
    /// whether it was clamped. Call with the world's tick at least once per
    /// [`crate::CHECK_TICK_THRESHOLD`] ticks if this state outlives that
    /// many.
    pub fn check_change_tick(&mut self, now: Tick) -> bool {
        self.meta.check_change_tick(now)
    }

    /// The parameters for now. `Added`/`Changed` measure against the
    /// world's change tick at the previous call (all components count as
    /// new on the first). This does not advance the tick: call
    /// `World::increment_change_tick` per step of your own loop, as for
    /// `ChangeTracker`.
    pub fn get<'w, 's>(&'s mut self, world: &'w mut World) -> P::Item<'w, 's> {
        let this_run = world.change_tick();
        let cell = world.as_world_cell();
        // SAFETY: `P`'s parameters were checked against each other in
        // `new`, and `world` is borrowed exclusively for `'w`, so nothing
        // else can touch it.
        let item = unsafe { P::get_param(&mut self.state, &self.meta, cell, this_run) };
        self.meta.set_last_run(this_run);
        item
    }

    /// Applies what the parameters deferred (queued `Commands`) to
    /// `world`.
    pub fn apply(&mut self, world: &mut World) {
        P::apply(&mut self.state, &self.meta, world);
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet};

    use super::*;
    use crate::filter::{Or, With, Without};
    use crate::query_data::{Added, Changed, Has, QueryEntityError, QuerySingleError};
    use crate::tick::Mut;

    #[derive(Debug, Clone, Copy, PartialEq)]
    struct A(u32);
    #[derive(Debug, Clone, Copy, PartialEq)]
    struct B(u32);
    #[derive(Debug, Clone, Copy, PartialEq)]
    struct C(u32);

    /// Entities: `e[0]` {A}, `e[1]` {A, B}, `e[2]` {B}, `e[3]` {A, B, C},
    /// `e[4]` {A}, with `A.0 == 10 + index`.
    fn world5() -> (World, Vec<Entity>) {
        let mut world = World::new();
        let mut es = Vec::new();
        for i in 0..5u32 {
            let e = world.spawn();
            if i != 2 {
                world.insert_static(e, A(10 + i));
            }
            if i == 1 || i == 2 || i == 3 {
                world.insert_static(e, B(20 + i));
            }
            if i == 3 {
                world.insert_static(e, C(30 + i));
            }
            es.push(e);
        }
        world.increment_change_tick();
        (world, es)
    }

    #[test]
    fn get_returns_the_entitys_item_and_applies_archetype_level_rules() {
        let (mut world, e) = world5();
        let mut s = SystemState::<Query<(Entity, &A)>>::new(&mut world);
        let q = s.get(&mut world);
        assert_eq!(q.get(e[0]), Ok((e[0], &A(10))));
        assert_eq!(q.get(e[3]), Ok((e[3], &A(13))));
        assert_eq!(
            q.get(e[2]),
            Err(QueryEntityError::QueryDoesNotMatch(e[2])),
            "{{B}} has no A"
        );
        drop(q);

        let mut s = SystemState::<Query<&A, With<B>>>::new(&mut world);
        let q = s.get(&mut world);
        assert_eq!(q.get(e[1]), Ok(&A(11)));
        assert_eq!(q.get(e[0]), Err(QueryEntityError::QueryDoesNotMatch(e[0])));
        drop(q);

        let mut s = SystemState::<Query<&A, Without<B>>>::new(&mut world);
        let q = s.get(&mut world);
        assert_eq!(q.get(e[0]), Ok(&A(10)));
        assert_eq!(q.get(e[1]), Err(QueryEntityError::QueryDoesNotMatch(e[1])));
        drop(q);

        // Option and Has terms never exclude an entity.
        let mut s = SystemState::<Query<(Option<&B>, Has<C>)>>::new(&mut world);
        let q = s.get(&mut world);
        assert_eq!(q.get(e[0]), Ok((None, false)));
        assert_eq!(q.get(e[3]), Ok((Some(&B(23)), true)));
    }

    #[test]
    fn get_reports_dead_entities_and_never_reaches_a_reused_slot() {
        let (mut world, e) = world5();
        world.despawn(e[1]);
        let stale = e[4];
        world.despawn(stale);
        let new = world.spawn();
        world.insert_static(new, A(999));
        let mut s = SystemState::<Query<&A>>::new(&mut world);
        let q = s.get(&mut world);
        assert_eq!(q.get(e[1]), Err(QueryEntityError::NoSuchEntity(e[1])));
        assert_eq!(q.get(stale), Err(QueryEntityError::NoSuchEntity(stale)));
        assert_eq!(q.get(new), Ok(&A(999)));
        assert!(!q.contains(stale));
        assert!(q.contains(new));
    }

    #[test]
    fn get_applies_row_level_filters_and_agrees_with_contains_and_count() {
        let (mut world, e) = world5();
        let mut s = SystemState::<Query<&A, Changed<A>>>::new(&mut world);
        // First call: everything counts as new.
        let q = s.get(&mut world);
        assert_eq!(q.count(), 4);
        drop(q);
        world.increment_change_tick();
        world.get_static_mut::<A>(e[1]).unwrap().0 = 77;
        world.get_static_mut::<A>(e[4]).unwrap().0 = 78;
        let q = s.get(&mut world);
        assert_eq!(q.get(e[1]), Ok(&A(77)));
        assert_eq!(q.get(e[4]), Ok(&A(78)));
        assert_eq!(q.get(e[0]), Err(QueryEntityError::QueryDoesNotMatch(e[0])));
        assert_eq!(q.get(e[2]), Err(QueryEntityError::QueryDoesNotMatch(e[2])));
        assert!(q.contains(e[1]) && !q.contains(e[0]) && !q.contains(e[2]));
        assert_eq!(q.count(), 2);
        assert!(!q.is_empty());
        drop(q);
        // Nothing changed since the last call.
        let q = s.get(&mut world);
        assert!(q.is_empty());
        assert_eq!(q.get(e[1]), Err(QueryEntityError::QueryDoesNotMatch(e[1])));
    }

    #[test]
    fn get_mut_stamps_only_when_written_through_and_only_that_row() {
        let (mut world, e) = world5();
        let mut watch = SystemState::<Query<(Entity, &A), Changed<A>>>::new(&mut world);
        drop(watch.get(&mut world));
        world.increment_change_tick();

        let mut s = SystemState::<Query<Mut<A>>>::new(&mut world);
        let mut q = s.get(&mut world);
        {
            // Reading through a `Mut` does not stamp.
            let item = q.get_mut(e[3]).unwrap();
            assert_eq!(item.0, 13);
        }
        {
            let mut item = q.get_mut(e[4]).unwrap();
            item.0 = 55;
        }
        assert_eq!(
            q.get_mut(e[2]).err(),
            Some(QueryEntityError::QueryDoesNotMatch(e[2]))
        );
        drop(q);
        let q = watch.get(&mut world);
        let changed: Vec<Entity> = q.iter().map(|(en, _)| en).collect();
        assert_eq!(changed, vec![e[4]], "only the row that was written");
        assert_eq!(world.get_static::<A>(e[4]), Some(&A(55)));
    }

    #[test]
    fn get_many_reads_and_get_many_mut_refuses_a_repeated_entity() {
        let (mut world, e) = world5();
        let mut s = SystemState::<Query<(Entity, &A)>>::new(&mut world);
        let q = s.get(&mut world);
        assert_eq!(
            q.get_many([e[0], e[3], e[0]]),
            Ok([(e[0], &A(10)), (e[3], &A(13)), (e[0], &A(10))]),
            "readers may repeat an entity"
        );
        assert_eq!(
            q.get_many([e[3], e[4], e[0]]),
            Ok([(e[3], &A(13)), (e[4], &A(14)), (e[0], &A(10))]),
            "in the order asked for"
        );
        assert_eq!(
            q.get_many([e[0], e[2], e[3]]),
            Err(QueryEntityError::QueryDoesNotMatch(e[2]))
        );
        drop(q);

        let mut s = SystemState::<Query<Mut<A>>>::new(&mut world);
        let mut q = s.get(&mut world);
        assert_eq!(
            q.get_many_mut([e[0], e[1], e[0]]).err(),
            Some(QueryEntityError::AliasedMutability(e[0]))
        );
        let [mut a, mut b] = q.get_many_mut([e[0], e[4]]).unwrap();
        std::mem::swap(&mut a.0, &mut b.0);
        drop((a, b));
        assert_eq!(
            q.get_many_mut([e[0], e[2]]).err(),
            Some(QueryEntityError::QueryDoesNotMatch(e[2]))
        );
        drop(q);
        assert_eq!(world.get_static::<A>(e[0]), Some(&A(14)));
        assert_eq!(world.get_static::<A>(e[4]), Some(&A(10)));
    }

    #[test]
    fn iter_many_follows_the_list_and_skips_what_does_not_match() {
        let (mut world, e) = world5();
        world.despawn(e[4]);
        let mut s = SystemState::<Query<(Entity, &A)>>::new(&mut world);
        let q = s.get(&mut world);
        let got: Vec<Entity> = q
            .iter_many([e[3], e[2], e[0], e[4], e[3]])
            .map(|(en, _)| en)
            .collect();
        assert_eq!(got, vec![e[3], e[0], e[3]], "list order, skips {{B}} and the dead");
    }

    #[test]
    fn single_and_single_mut_say_none_one_or_many_and_a_refusal_writes_nothing() {
        let mut world = World::new();
        let mut watch = SystemState::<Query<Entity, Changed<A>>>::new(&mut world);
        let mut s = SystemState::<Query<Mut<A>>>::new(&mut world);
        assert_eq!(s.get(&mut world).single_mut().err(), Some(QuerySingleError::NoEntities));
        let e1 = world.spawn();
        world.insert_static(e1, A(1));
        let e2 = world.spawn();
        world.insert_static(e2, A(2));
        world.increment_change_tick();
        drop(watch.get(&mut world));
        world.increment_change_tick();
        {
            let mut q = s.get(&mut world);
            assert_eq!(q.single_mut().err(), Some(QuerySingleError::MultipleEntities));
        }
        assert!(
            watch.get(&mut world).is_empty(),
            "looking for a second item wrote and stamped nothing"
        );
        world.despawn(e2);
        {
            let mut q = s.get(&mut world);
            q.single_mut().unwrap().0 = 5;
        }
        assert_eq!(world.get_static::<A>(e1), Some(&A(5)));
        let mut r = SystemState::<Query<&A>>::new(&mut world);
        assert_eq!(r.get(&mut world).single(), Ok(&A(5)));
        world.insert_static(e2, A(1));
    }

    #[test]
    fn a_query_over_a_type_the_world_never_saw_matches_nothing() {
        #[derive(Debug, PartialEq)]
        struct Never(u32);
        let (mut world, e) = world5();
        let mut s = SystemState::<Query<&Never>>::new(&mut world);
        let q = s.get(&mut world);
        assert_eq!(
            q.get(e[0]),
            Err(QueryEntityError::QueryDoesNotMatch(e[0]))
        );
        assert!(!q.contains(e[0]));
        assert_eq!(q.count(), 0);
        assert!(q.is_empty());
        drop(q);
        let mut s = SystemState::<Query<(&A, &Never)>>::new(&mut world);
        let q = s.get(&mut world);
        assert_eq!(
            q.get(e[0]),
            Err(QueryEntityError::QueryDoesNotMatch(e[0])),
            "one satisfiable term does not rescue an unsatisfiable one"
        );
    }

    #[test]
    #[should_panic(expected = "WorldCell::query_get: conflicting access")]
    fn a_query_that_reads_and_writes_one_component_panics_before_any_lookup() {
        let (mut world, e) = world5();
        let tick = world.change_tick();
        let cell = world.as_world_cell();
        // Looked up on an entity that does not even match: the check comes
        // first, as it does for an iteration.
        // SAFETY: nothing else touches the world; the call panics before
        // it reads anything.
        let _ = unsafe { cell.query_get_mut::<(&A, Mut<A>), ()>(e[2], tick, tick) };
    }

    #[test]
    fn a_query_that_requires_two_components_does_not_match_an_entity_with_one() {
        let (mut world, e) = world5();
        let mut s = SystemState::<Query<(&A, &B)>>::new(&mut world);
        let q = s.get(&mut world);
        assert_eq!(q.get(e[1]), Ok((&A(11), &B(21))));
        assert_eq!(q.get(e[0]), Err(QueryEntityError::QueryDoesNotMatch(e[0])));
        assert_eq!(q.get(e[2]), Err(QueryEntityError::QueryDoesNotMatch(e[2])));
    }

    #[test]
    fn single_reads_one_item_and_says_none_or_many() {
        let mut world = World::new();
        let mut s = SystemState::<Query<&A>>::new(&mut world);
        assert_eq!(s.get(&mut world).single(), Err(QuerySingleError::NoEntities));
        let e1 = world.spawn();
        world.insert_static(e1, A(1));
        assert_eq!(s.get(&mut world).single(), Ok(&A(1)));
        let e2 = world.spawn();
        world.insert_static(e2, A(2));
        assert_eq!(
            s.get(&mut world).single(),
            Err(QuerySingleError::MultipleEntities)
        );
    }

    #[test]
    fn get_many_mut_stamps_only_the_rows_written_through() {
        let (mut world, e) = world5();
        let mut watch = SystemState::<Query<Entity, Changed<A>>>::new(&mut world);
        drop(watch.get(&mut world));
        world.increment_change_tick();
        let mut s = SystemState::<Query<Mut<A>>>::new(&mut world);
        {
            let mut q = s.get(&mut world);
            let [mut a, b] = q.get_many_mut([e[0], e[4]]).unwrap();
            a.0 = 1;
            assert_eq!(b.0, 14, "read through, not written");
        }
        let changed: Vec<Entity> = watch.get(&mut world).iter().collect();
        assert_eq!(changed, vec![e[0]]);
        assert_eq!(world.get_static::<A>(e[0]), Some(&A(1)));
    }

    /// xorshift64: enough randomness for a history generator.
    struct Xs(u64);

    impl Xs {
        fn next(&mut self, n: usize) -> usize {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 % n as u64) as usize
        }
    }

    /// For one read-only query shape, at one moment: every entity ever
    /// created (alive or not) must be found by `get` exactly when an
    /// iteration yields it, with the same item, and `contains`, `count`,
    /// `is_empty` and `iter_many` must agree. Returns the match count.
    macro_rules! check_shape {
        ($world:expr, $state:expr, $all:expr, $alive:expr, $label:expr) => {{
            let q = $state.get(&mut $world);
            let expected: HashMap<Entity, _> = q.iter().map(|item| (item.0, item)).collect();
            for &e in $all.iter() {
                match q.get(e) {
                    Ok(item) => assert_eq!(Some(&item), expected.get(&e), "{}: {e:?}", $label),
                    Err(QueryEntityError::NoSuchEntity(x)) => {
                        assert_eq!(x, e);
                        assert!(!$alive.contains(&e), "{}: {e:?} is alive", $label);
                        assert!(!expected.contains_key(&e));
                    }
                    Err(QueryEntityError::QueryDoesNotMatch(x)) => {
                        assert_eq!(x, e);
                        assert!($alive.contains(&e), "{}: {e:?} is dead", $label);
                        assert!(!expected.contains_key(&e));
                    }
                    Err(other) => panic!("{}: unexpected {other:?}", $label),
                }
                assert_eq!(q.contains(e), expected.contains_key(&e), "{}: {e:?}", $label);
            }
            assert_eq!(q.count(), expected.len(), "{}", $label);
            assert_eq!(q.is_empty(), expected.is_empty(), "{}", $label);
            let in_order: Vec<_> = $all.iter().filter_map(|e| expected.get(e).copied()).collect();
            let many: Vec<_> = q.iter_many($all.iter().copied()).collect();
            assert_eq!(many, in_order, "{}: iter_many", $label);
            expected.len()
        }};
    }

    #[test]
    fn get_contains_count_and_iter_many_agree_with_iteration_over_a_random_history() {
        let mut world = World::new();
        let mut all: Vec<Entity> = Vec::new();
        let seed = world.spawn();
        world.insert_static(seed, A(0));
        world.insert_static(seed, B(0));
        world.insert_static(seed, C(0));
        all.push(seed);
        let mut s1 = SystemState::<Query<(Entity, &A)>>::new(&mut world);
        let mut s2 = SystemState::<Query<(Entity, &A, &B)>>::new(&mut world);
        let mut s3 = SystemState::<Query<(Entity, &A), Without<B>>>::new(&mut world);
        let mut s4 = SystemState::<Query<(Entity, Option<&C>), With<A>>>::new(&mut world);
        let mut s5 = SystemState::<Query<(Entity, &A), Changed<A>>>::new(&mut world);
        let mut s6 =
            SystemState::<Query<(Entity, &A, &B), Or<(Added<A>, Changed<B>)>>>::new(&mut world);
        let mut rng = Xs(0xA11C_E5EE_D5EE_D123);
        let (mut non_empty, mut empty) = ([0usize; 6], [0usize; 6]);

        for step in 0..400u32 {
            // One tick per step, before its changes, so what a step changes
            // is newer than the previous step's `get` and the tracked
            // shapes see it.
            world.increment_change_tick();
            for _ in 0..1 + rng.next(4) {
                let alive: Vec<Entity> = all.iter().copied().filter(|e| world.is_alive(*e)).collect();
                let pick = if alive.is_empty() { 0 } else { rng.next(alive.len()) };
                match rng.next(11) {
                    0 | 1 => {
                        let e = world.spawn();
                        if rng.next(4) != 0 {
                            world.insert_static(e, A(step));
                        }
                        if rng.next(2) == 0 {
                            world.insert_static(e, B(step));
                        }
                        if rng.next(4) == 0 {
                            world.insert_static(e, C(step));
                        }
                        all.push(e);
                    }
                    2 if !alive.is_empty() => {
                        world.despawn(alive[pick]);
                    }
                    3 if !alive.is_empty() => {
                        world.insert_static(alive[pick], B(step));
                    }
                    4 if !alive.is_empty() => {
                        world.remove_static::<B>(alive[pick]);
                    }
                    5 if !alive.is_empty() => {
                        world.insert_static(alive[pick], A(step));
                    }
                    6 if !alive.is_empty() => {
                        world.remove_static::<A>(alive[pick]);
                    }
                    7 | 8 if !alive.is_empty() => {
                        if let Some(x) = world.get_static_mut::<A>(alive[pick]) {
                            x.0 += 1;
                        }
                    }
                    9 if !alive.is_empty() => {
                        if let Some(x) = world.get_static_mut::<B>(alive[pick]) {
                            x.0 += 1;
                        }
                    }
                    10 if !alive.is_empty() => {
                        world.insert_static(alive[pick], C(step));
                    }
                    _ => {}
                }
            }
            let alive: HashSet<Entity> = all.iter().copied().filter(|e| world.is_alive(*e)).collect();
            let counts = [
                check_shape!(world, s1, all, alive, "(Entity, &A)"),
                check_shape!(world, s2, all, alive, "(Entity, &A, &B)"),
                check_shape!(world, s3, all, alive, "(Entity, &A) Without<B>"),
                check_shape!(world, s4, all, alive, "(Entity, Option<&C>) With<A>"),
                check_shape!(world, s5, all, alive, "(Entity, &A) Changed<A>"),
                check_shape!(world, s6, all, alive, "(Entity, &A, &B) Or<Added<A>, Changed<B>>"),
            ];
            for (k, n) in counts.into_iter().enumerate() {
                if n == 0 {
                    empty[k] += 1;
                } else {
                    non_empty[k] += 1;
                }
            }
        }
        // The history must have made every shape match something, and the
        // two row-level-filtered shapes (4 and 5) also match nothing often
        // enough, or the comparison proves little.
        for k in 0..6 {
            assert!(non_empty[k] >= 10, "shape {k}: {} non-empty steps", non_empty[k]);
        }
        for k in [4, 5] {
            assert!(empty[k] >= 10, "shape {k}: {} empty steps", empty[k]);
        }
    }

    #[test]
    fn count_and_contains_work_for_a_writing_query_and_match_its_read_only_twin() {
        let (mut world, e) = world5();
        let mut writer = SystemState::<Query<Mut<A>, Without<C>>>::new(&mut world);
        let mut reader = SystemState::<Query<&A, Without<C>>>::new(&mut world);
        let w = writer.get(&mut world);
        assert_eq!(w.count(), 4 - 1, "e[3] has C");
        assert!(w.contains(e[0]) && !w.contains(e[3]) && !w.contains(e[2]));
        drop(w);
        assert_eq!(reader.get(&mut world).count(), 3);
    }
}
