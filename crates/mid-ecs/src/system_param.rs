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

use std::any::type_name;
use std::borrow::Cow;
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};

use crate::access::{FilteredAccess, FilteredAccessSet};
use crate::archetype::QueryBatch;
use crate::batching::BatchingStrategy;
use crate::commands::{CommandQueue, Commands};
use crate::data_filter::DataFilter;
use crate::query_data::{QueryData, ReadOnlyQueryData};
use crate::tick::Tick;
use crate::world::World;
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
pub struct Res<'w, T: 'static> {
    value: &'w T,
}

impl<T: 'static> Deref for Res<'_, T> {
    type Target = T;

    fn deref(&self) -> &T {
        self.value
    }
}

/// A resource, exclusive. Panics when the system runs if the resource
/// does not exist; use `Option<ResMut<T>>` to tolerate that.
pub struct ResMut<'w, T: 'static> {
    value: &'w mut T,
}

impl<T: 'static> Deref for ResMut<'_, T> {
    type Target = T;

    fn deref(&self) -> &T {
        self.value
    }
}

impl<T: 'static> DerefMut for ResMut<'_, T> {
    fn deref_mut(&mut self) -> &mut T {
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
        _this_run: Tick,
    ) -> Self::Item<'w, 's> {
        // SAFETY: the read was registered and checked against every other
        // parameter, so nothing writes `T` meanwhile.
        match unsafe { world.resource::<T>() } {
            Some(value) => Res { value },
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
        _meta: &SystemMeta,
        world: WorldCell<'w>,
        _this_run: Tick,
    ) -> Self::Item<'w, 's> {
        // SAFETY: as for `Res`.
        unsafe { world.resource::<T>() }.map(|value| Res { value })
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
        _this_run: Tick,
    ) -> Self::Item<'w, 's> {
        // SAFETY: the write was registered and checked against every
        // other parameter, so nothing else reads or writes `T` meanwhile.
        match unsafe { world.resource_mut::<T>() } {
            Some(value) => ResMut { value },
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
        _meta: &SystemMeta,
        world: WorldCell<'w>,
        _this_run: Tick,
    ) -> Self::Item<'w, 's> {
        // SAFETY: as for `ResMut`.
        unsafe { world.resource_mut::<T>() }.map(|value| ResMut { value })
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
