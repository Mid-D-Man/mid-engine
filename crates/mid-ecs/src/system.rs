// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-ecs.md, section "System parameters: `system_param.rs`,
// `system.rs`"
// ============================================================================
//! Systems: plain functions whose parameters are [`SystemParam`]s, turned
//! into something a runner can set up and call.
//!
//! ```
//! use mid_ecs::{Commands, Mut, Query, Res, ResMut, World};
//!
//! struct Position(f32);
//! struct Velocity(f32);
//! struct Elapsed(f32);
//! struct Ticks(u32);
//!
//! fn integrate(
//!     dt: Res<Elapsed>,
//!     mut ticks: ResMut<Ticks>,
//!     mut movers: Query<(Mut<Position>, &Velocity)>,
//!     mut commands: Commands,
//! ) {
//!     ticks.0 += 1;
//!     for (mut position, velocity) in movers.iter_mut() {
//!         position.0 += velocity.0 * dt.0;
//!     }
//!     commands.spawn(); // applied after the system returns
//! }
//!
//! let mut world = World::new();
//! world.insert_resource(Elapsed(0.5));
//! world.insert_resource(Ticks(0));
//! let e = world.spawn();
//! world.insert_static(e, Position(1.0));
//! world.insert_static(e, Velocity(4.0));
//!
//! world.run_system_once(integrate);
//! assert_eq!(world.get_static::<Position>(e).map(|p| p.0), Some(3.0));
//! assert_eq!(world.get_resource::<Ticks>().map(|t| t.0), Some(1));
//! ```
//!
//! What is here is one system at a time: build it, initialize it (which
//! checks its parameters against each other), run it. Ordering several,
//! run conditions, and running them in parallel are a runner's job, and the
//! pieces such a runner needs are exactly [`System::access`],
//! [`System::run_unchecked`] and [`System::apply_deferred`].
//!
//! A system returns nothing. A function taking `&mut World` is an
//! *exclusive* system: it conflicts with everything and runs alone.

use std::any::type_name;
use std::marker::PhantomData;

use crate::access::{FilteredAccess, FilteredAccessSet};
use crate::system_param::{SystemMeta, SystemParam, SystemParamItem};
use crate::tick::Tick;
use crate::world::World;
use crate::world_cell::WorldCell;

// ── The trait ────────────────────────────────────────────────────────

/// A unit of work over a [`World`] with a known access footprint.
pub trait System: 'static {
    /// A name for messages.
    fn name(&self) -> &str;

    /// Sets the system up against `world`: builds its parameters' state
    /// and checks them against each other. Does nothing if already done.
    ///
    /// # Panics
    /// If two of its parameters conflict.
    fn initialize(&mut self, world: &mut World);

    /// Whether [`Self::initialize`] has run.
    fn is_initialized(&self) -> bool;

    /// What the system touches, one part per parameter. Empty until
    /// initialized.
    fn access(&self) -> &FilteredAccessSet;

    /// `true` if the system takes the whole world (an exclusive system).
    fn is_exclusive(&self) -> bool {
        self.access().is_exclusive()
    }

    /// Runs the system against a world cell, stamping its writes
    /// `this_run`. Deferred work (`Commands`) is *not* applied: call
    /// [`Self::apply_deferred`] afterwards, when the world is available
    /// exclusively.
    ///
    /// # Panics
    /// If the system has not been initialized, or a resource it requires
    /// does not exist.
    ///
    /// # Safety
    /// For the duration of the call, nothing may access the world in a way
    /// that conflicts with [`Self::access`]. (For an exclusive system that
    /// means nothing at all.)
    unsafe fn run_unchecked(&mut self, world: WorldCell<'_>, this_run: Tick);

    /// Applies what the last run deferred.
    fn apply_deferred(&mut self, world: &mut World);

    /// Initializes if needed, runs at the world's current change tick and
    /// then advances it, and applies deferred work.
    ///
    /// Running *at* the current tick and advancing *afterwards* is what
    /// keeps change detection honest: the system's own writes are stamped
    /// with its `this_run`, so its next run (whose `last_run` is that tick)
    /// does not see them, while anything else that writes after this run
    /// is stamped with the new, later tick and is seen.
    fn run(&mut self, world: &mut World) {
        self.initialize(world);
        let this_run = world.change_tick();
        world.increment_change_tick();
        // SAFETY: `world` is borrowed exclusively, and the cell is gone
        // before it is used again.
        unsafe { self.run_unchecked(world.as_world_cell(), this_run) };
        self.apply_deferred(world);
    }
}

// ── IntoSystem ───────────────────────────────────────────────────────

/// Something that becomes a [`System`]: a function of system parameters,
/// a function of `&mut World`, or a system.
///
/// `Marker` only keeps the impls apart; it is inferred.
pub trait IntoSystem<Marker>: Sized {
    /// The system this becomes.
    type System: System;

    /// Converts.
    fn into_system(self) -> Self::System;
}

/// Marker for functions of system parameters.
pub struct IsFunctionSystem;
/// Marker for functions of `&mut World`.
pub struct IsExclusiveFunctionSystem;
/// Marker for something that already is a [`System`].
pub struct IsSystem;

impl<S: System> IntoSystem<IsSystem> for S {
    type System = S;

    fn into_system(self) -> S {
        self
    }
}

// ── Function systems ─────────────────────────────────────────────────

/// A function whose arguments are all [`SystemParam`]s. Implemented for
/// functions and closures of up to twelve such arguments.
pub trait SystemParamFunction<Marker>: 'static {
    /// The arguments, as one (tuple) parameter.
    type Param: SystemParam + 'static;

    /// Calls the function.
    fn run(&mut self, param: SystemParamItem<'_, '_, Self::Param>);
}

macro_rules! impl_system_param_function {
    ($($P:ident),*) => {
        #[allow(non_snake_case)]
        impl<Func, $($P: SystemParam + 'static),*> SystemParamFunction<fn($($P,)*)> for Func
        where
            Func: 'static,
            // The function must accept the parameters both with the
            // placeholder lifetimes `SystemParam` is written with and
            // with any real ones: the second bound is what the call
            // below needs, the first is what a function item satisfies.
            for<'a> &'a mut Func:
                FnMut($($P),*) + FnMut($(SystemParamItem<'_, '_, $P>),*),
        {
            type Param = ($($P,)*);

            fn run(&mut self, param: SystemParamItem<'_, '_, ($($P,)*)>) {
                fn call_inner<$($P,)*>(mut f: impl FnMut($($P,)*), $($P: $P,)*) {
                    f($($P,)*)
                }
                let ($($P,)*) = param;
                call_inner(self, $($P),*)
            }
        }
    };
}

impl_system_param_function!();
impl_system_param_function!(P0);
impl_system_param_function!(P0, P1);
impl_system_param_function!(P0, P1, P2);
impl_system_param_function!(P0, P1, P2, P3);
impl_system_param_function!(P0, P1, P2, P3, P4);
impl_system_param_function!(P0, P1, P2, P3, P4, P5);
impl_system_param_function!(P0, P1, P2, P3, P4, P5, P6);
impl_system_param_function!(P0, P1, P2, P3, P4, P5, P6, P7);
impl_system_param_function!(P0, P1, P2, P3, P4, P5, P6, P7, P8);
impl_system_param_function!(P0, P1, P2, P3, P4, P5, P6, P7, P8, P9);
impl_system_param_function!(P0, P1, P2, P3, P4, P5, P6, P7, P8, P9, P10);
impl_system_param_function!(P0, P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11);

/// A [`SystemParamFunction`] together with its parameters' state.
pub struct FunctionSystem<Marker, F: SystemParamFunction<Marker>> {
    func: F,
    state: Option<<F::Param as SystemParam>::State>,
    meta: SystemMeta,
    _marker: PhantomData<fn() -> Marker>,
}

impl<Marker, F: SystemParamFunction<Marker>> FunctionSystem<Marker, F> {
    /// The system's meta: its name, registered access and last run.
    pub fn meta(&self) -> &SystemMeta {
        &self.meta
    }
}

impl<Marker: 'static, F: SystemParamFunction<Marker>> IntoSystem<(IsFunctionSystem, Marker)> for F {
    type System = FunctionSystem<Marker, F>;

    fn into_system(self) -> Self::System {
        FunctionSystem {
            func: self,
            state: None,
            meta: SystemMeta::new(type_name::<F>()),
            _marker: PhantomData,
        }
    }
}

impl<Marker: 'static, F: SystemParamFunction<Marker>> System for FunctionSystem<Marker, F> {
    fn name(&self) -> &str {
        self.meta.name()
    }

    fn initialize(&mut self, world: &mut World) {
        if self.state.is_none() {
            self.state = Some(<F::Param as SystemParam>::init_state(world, &mut self.meta));
        }
    }

    fn is_initialized(&self) -> bool {
        self.state.is_some()
    }

    fn access(&self) -> &FilteredAccessSet {
        self.meta.access()
    }

    unsafe fn run_unchecked(&mut self, world: WorldCell<'_>, this_run: Tick) {
        let state = self
            .state
            .as_mut()
            .expect("a system must be initialized before it is run");
        // SAFETY: the caller's contract is exactly `get_param`'s.
        let param =
            unsafe { <F::Param as SystemParam>::get_param(state, &self.meta, world, this_run) };
        self.func.run(param);
        self.meta.set_last_run(this_run);
    }

    fn apply_deferred(&mut self, world: &mut World) {
        if let Some(state) = self.state.as_mut() {
            <F::Param as SystemParam>::apply(state, &self.meta, world);
        }
    }
}

// ── Exclusive systems ────────────────────────────────────────────────

/// A function of `&mut World`: runs alone, conflicts with everything.
pub struct ExclusiveFunctionSystem<F> {
    func: F,
    meta: SystemMeta,
    initialized: bool,
}

impl<F> ExclusiveFunctionSystem<F> {
    /// The system's meta: its name, registered access and last run.
    pub fn meta(&self) -> &SystemMeta {
        &self.meta
    }
}

impl<F: FnMut(&mut World) + 'static> IntoSystem<IsExclusiveFunctionSystem> for F {
    type System = ExclusiveFunctionSystem<F>;

    fn into_system(self) -> Self::System {
        ExclusiveFunctionSystem {
            func: self,
            meta: SystemMeta::new(type_name::<F>()),
            initialized: false,
        }
    }
}

impl<F: FnMut(&mut World) + 'static> System for ExclusiveFunctionSystem<F> {
    fn name(&self) -> &str {
        self.meta.name()
    }

    fn initialize(&mut self, _world: &mut World) {
        if !self.initialized {
            let mut everything = FilteredAccess::new();
            everything.write_all();
            self.meta.add_param_access(everything);
            self.initialized = true;
        }
    }

    fn is_initialized(&self) -> bool {
        self.initialized
    }

    fn access(&self) -> &FilteredAccessSet {
        self.meta.access()
    }

    unsafe fn run_unchecked(&mut self, world: WorldCell<'_>, this_run: Tick) {
        assert!(
            self.initialized,
            "a system must be initialized before it is run"
        );
        // SAFETY: the caller's contract for an exclusive system is that
        // nothing else touches the world at all, which is what
        // `world_mut` needs.
        (self.func)(unsafe { world.world_mut() });
        self.meta.set_last_run(this_run);
    }

    fn apply_deferred(&mut self, _world: &mut World) {}
}

// ── World ────────────────────────────────────────────────────────────

impl World {
    /// Builds `system`, initializes it, runs it once and applies its
    /// deferred work. A fresh system has not run before, so `Added` and
    /// `Changed` see every component as new; to keep that history, build
    /// the system once with [`IntoSystem::into_system`] and call
    /// [`System::run`] on it repeatedly.
    ///
    /// # Panics
    /// If the system's parameters conflict, or a resource it needs is
    /// missing.
    pub fn run_system_once<Marker, S: IntoSystem<Marker>>(&mut self, system: S) {
        system.into_system().run(self);
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;
    use crate::commands::Commands;
    use crate::filter::{Or, With, Without};
    use crate::query_data::{Added, Changed, Has};
    use crate::system_param::{Local, Query, Res, ResMut, SystemState};
    use crate::tick::{ChangeTracker, Mut, Ref};
    use crate::world::Entity;

    #[derive(Debug, Clone, Copy, PartialEq)]
    struct A(u32);
    #[derive(Debug, Clone, Copy, PartialEq)]
    struct B(u32);
    #[derive(Debug, Clone, Copy, PartialEq)]
    struct C(u32);
    #[derive(Default, Debug, PartialEq)]
    struct Counts {
        seen: Vec<usize>,
    }
    #[derive(Debug, PartialEq)]
    struct Score(u32);
    #[derive(Debug, PartialEq)]
    struct Other(u32);

    fn world_with_a(n: u32) -> (World, Vec<Entity>) {
        let mut world = World::new();
        world.insert_resource(Counts::default());
        let es = (0..n)
            .map(|i| {
                let e = world.spawn();
                world.insert_static(e, A(i));
                e
            })
            .collect();
        (world, es)
    }

    fn counts(world: &World) -> Vec<usize> {
        world.get_resource::<Counts>().expect("Counts").seen.clone()
    }

    // ── Several live borrows of one world ────────────────────────────

    fn nested(mut writers: Query<(Entity, Mut<A>)>, readers: Query<(Entity, &B)>) {
        for (e, b) in readers.iter() {
            for (e2, mut a) in writers.iter_mut() {
                if e == e2 {
                    a.0 += b.0;
                }
            }
        }
    }

    #[test]
    fn a_writer_and_a_reader_of_different_components_are_live_together() {
        let (mut world, es) = world_with_a(5);
        for (i, &e) in es.iter().enumerate() {
            world.insert_static(e, B(100 + i as u32));
        }
        world.run_system_once(nested);
        for (i, &e) in es.iter().enumerate() {
            let i = i as u32;
            assert_eq!(world.get_static::<A>(e).map(|a| a.0), Some(i + 100 + i));
        }
    }

    // ── Conflicts are refused when the system is set up ──────────────

    fn conflicting_queries(_w: Query<Mut<A>>, _r: Query<&A>) {}
    fn conflicting_resources(_a: Res<Score>, _b: ResMut<Score>) {}
    fn conflicting_through_ticks(_w: Query<Mut<B>>, _r: Query<&A, Changed<B>>) {}
    fn conflicting_with_itself(_q: Query<(&A, Mut<A>)>) {}
    fn disjoint_filters(mut a: Query<Mut<A>, With<B>>, mut b: Query<Mut<A>, Without<B>>) {
        for mut x in a.iter_mut() {
            x.0 += 1;
        }
        for mut x in b.iter_mut() {
            x.0 += 10;
        }
    }
    fn compatible_readers(a: Query<&A>, b: Query<&A>, r: Res<Score>, s: Res<Score>) {
        assert_eq!(a.iter().count(), b.iter().count());
        assert_eq!(r.0, s.0);
    }
    fn different_writers(mut a: Query<Mut<A>>, mut b: Query<Mut<B>>, mut s: ResMut<Score>) {
        for mut x in a.iter_mut() {
            x.0 += 1;
        }
        for mut y in b.iter_mut() {
            y.0 += 2;
        }
        s.0 += 1;
    }

    #[test]
    #[should_panic(expected = "has parameters that conflict")]
    fn a_writer_and_a_reader_of_one_component_are_refused() {
        World::new().run_system_once(conflicting_queries);
    }

    #[test]
    #[should_panic(expected = "has parameters that conflict")]
    fn a_resource_read_and_written_by_one_system_is_refused() {
        World::new().run_system_once(conflicting_resources);
    }

    #[test]
    #[should_panic(expected = "has parameters that conflict")]
    fn a_writer_conflicts_with_a_changed_filter_on_the_same_component() {
        World::new().run_system_once(conflicting_through_ticks);
    }

    #[test]
    #[should_panic(expected = "conflicting access")]
    fn a_query_that_conflicts_with_itself_is_refused() {
        World::new().run_system_once(conflicting_with_itself);
    }

    #[test]
    #[should_panic(expected = "has parameters that conflict")]
    fn a_system_state_refuses_conflicting_parameters_too() {
        let mut world = World::new();
        let _ = SystemState::<(Query<Mut<A>>, Query<&A>)>::new(&mut world);
    }

    #[test]
    fn the_conflict_message_names_the_system_and_the_type() {
        let result = std::panic::catch_unwind(|| {
            World::new().run_system_once(conflicting_queries);
        });
        let payload = result.expect_err("must panic");
        let message = payload
            .downcast_ref::<String>()
            .expect("string panic")
            .clone();
        assert!(message.contains("conflicting_queries"), "{message}");
        assert!(message.contains("::A`"), "{message}");
    }

    #[test]
    fn parameters_that_cannot_alias_are_accepted_and_work() {
        let (mut world, es) = world_with_a(4);
        for (i, &e) in es.iter().enumerate() {
            if i % 2 == 0 {
                world.insert_static(e, B(50));
            }
        }
        world.insert_resource(Score(7));
        world.run_system_once(disjoint_filters);
        let a: Vec<u32> = es
            .iter()
            .map(|&e| world.get_static::<A>(e).unwrap().0)
            .collect();
        assert_eq!(a, vec![1, 11, 3, 13], "With<B> got +1, Without<B> got +10");
        world.run_system_once(compatible_readers);
        world.run_system_once(different_writers);
        assert_eq!(world.get_resource::<Score>(), Some(&Score(8)));
        assert_eq!(world.get_static::<B>(es[0]).map(|b| b.0), Some(52));
    }

    // ── Change detection across runs ─────────────────────────────────

    fn count_changed(q: Query<Entity, Changed<A>>, mut counts: ResMut<Counts>) {
        counts.seen.push(q.iter().count());
    }

    #[test]
    fn a_system_sees_what_changed_since_its_own_previous_run() {
        let (mut world, es) = world_with_a(4);
        let mut sys = count_changed.into_system();
        sys.run(&mut world); // everything is new to a system that never ran
        sys.run(&mut world); // nothing since
        world.get_static_mut::<A>(es[1]).unwrap().0 += 1;
        sys.run(&mut world); // one write
        let e = world.spawn();
        world.insert_static(e, A(9));
        sys.run(&mut world); // an insert is a change
        sys.run(&mut world);
        assert_eq!(counts(&world), vec![4, 0, 1, 1, 0]);
    }

    fn bump_changed(mut q: Query<Mut<A>, Changed<A>>, mut total: ResMut<Score>) {
        for mut a in q.iter_mut() {
            a.0 += 1;
            total.0 += 1;
        }
    }

    #[test]
    fn a_systems_own_writes_are_invisible_to_its_next_run_but_others_are_seen() {
        let (mut world, es) = world_with_a(4);
        world.insert_resource(Score(0));
        let mut sys = bump_changed.into_system();
        sys.run(&mut world);
        assert_eq!(world.get_resource::<Score>(), Some(&Score(4)));
        sys.run(&mut world);
        assert_eq!(
            world.get_resource::<Score>(),
            Some(&Score(4)),
            "its own writes from the first run are not changes to it"
        );
        // Another writer, outside any system, right after the run.
        world.get_static_mut::<A>(es[2]).unwrap().0 = 500;
        sys.run(&mut world);
        assert_eq!(world.get_resource::<Score>(), Some(&Score(5)));
        assert_eq!(world.get_static::<A>(es[2]).map(|a| a.0), Some(501));
        assert_eq!(
            sys.meta().last_run(),
            Tick::new(world.change_tick().get() - 1)
        );
    }

    #[test]
    fn a_fresh_system_each_time_sees_everything_as_new() {
        let (mut world, _) = world_with_a(3);
        world.run_system_once(count_changed);
        world.run_system_once(count_changed);
        assert_eq!(counts(&world), vec![3, 3]);
    }

    // ── The rest of the parameter family ─────────────────────────────

    fn spawner(mut commands: Commands, q: Query<Entity>, mut counts: ResMut<Counts>) {
        counts.seen.push(q.iter().count());
        commands.spawn().insert_static(A(7));
        counts.seen.push(q.iter().count());
    }

    #[test]
    fn commands_wait_until_the_system_has_returned() {
        let (mut world, _) = world_with_a(3);
        let mut sys = spawner.into_system();
        sys.run(&mut world);
        sys.run(&mut world);
        // Each run saw 3 then (still) 3 before its own spawn applied; the
        // second run saw the first run's entity, and the queue was empty
        // again (one new entity per run, not accumulating).
        assert_eq!(counts(&world), vec![3, 3, 4, 4]);
        assert_eq!(world.query_data::<Entity>().count(), 5);
    }

    fn counter(mut a: Local<u32>, mut b: Local<u32>, mut counts: ResMut<Counts>) {
        *a += 1;
        *b += 10;
        counts.seen.push((*a + *b) as usize);
    }

    #[test]
    fn a_local_lives_across_runs_and_belongs_to_its_system() {
        let (mut world, _) = world_with_a(0);
        let mut first = counter.into_system();
        first.run(&mut world);
        first.run(&mut world);
        first.run(&mut world);
        let mut second = counter.into_system();
        second.run(&mut world);
        assert_eq!(counts(&world), vec![11, 22, 33, 11]);
    }

    fn needs_score(_s: Res<Score>) {}

    fn optional_read_beside_read(_a: Option<Res<Score>>, _b: Res<Score>) {}
    fn optional_read_beside_write(_a: Option<Res<Score>>, _b: ResMut<Score>) {}
    fn optional_write_beside_read(_a: Option<ResMut<Score>>, _b: Res<Score>) {}
    fn optional_write_beside_write(_a: Option<ResMut<Score>>, _b: Option<ResMut<Score>>) {}

    #[test]
    fn optional_resource_parameters_register_the_same_access_as_the_plain_ones() {
        let (mut world, _) = world_with_a(0);
        world.insert_resource(Score(1));
        world.run_system_once(optional_read_beside_read);
        for system_name in ["read_write", "write_read", "write_write"] {
            let result = std::panic::catch_unwind(|| {
                let mut world = World::new();
                world.insert_resource(Score(1));
                match system_name {
                    "read_write" => world.run_system_once(optional_read_beside_write),
                    "write_read" => world.run_system_once(optional_write_beside_read),
                    _ => world.run_system_once(optional_write_beside_write),
                }
            });
            let payload = result.expect_err(system_name);
            let message = payload
                .downcast_ref::<String>()
                .cloned()
                .unwrap_or_default();
            assert!(
                message.contains("has parameters that conflict"),
                "{system_name}: {message}"
            );
        }
    }

    fn optional(
        score: Option<Res<Score>>,
        other: Option<ResMut<Other>>,
        mut counts: ResMut<Counts>,
    ) {
        counts.seen.push(score.map(|s| s.0 as usize).unwrap_or(999));
        if let Some(mut o) = other {
            o.0 += 1;
        }
    }

    #[test]
    #[should_panic(expected = "does not exist")]
    fn a_missing_required_resource_panics_when_the_system_runs() {
        World::new().run_system_once(needs_score);
    }

    #[test]
    fn optional_resources_tolerate_absence_and_write_when_present() {
        let (mut world, _) = world_with_a(0);
        world.run_system_once(optional);
        world.insert_resource(Score(4));
        world.insert_resource(Other(0));
        world.run_system_once(optional);
        assert_eq!(counts(&world), vec![999, 4]);
        assert_eq!(world.get_resource::<Other>(), Some(&Other(1)));
    }

    fn terms(
        q: Query<(Entity, Has<B>, Option<&C>, Ref<A>)>,
        unseen: Query<&Never>,
        mut counts: ResMut<Counts>,
    ) {
        let with_b = q.iter().filter(|(_, has_b, _, _)| *has_b).count();
        let with_c = q.iter().filter(|(_, _, c, _)| c.is_some()).count();
        let sum: u32 = q.iter().map(|(_, _, _, a)| a.0).sum();
        counts
            .seen
            .extend([q.iter().count(), with_b, with_c, sum as usize]);
        counts.seen.push(unseen.iter().count());
    }

    struct Never;

    #[test]
    fn every_kind_of_term_works_inside_a_system() {
        let (mut world, es) = world_with_a(4);
        world.insert_static(es[0], B(1));
        world.insert_static(es[1], B(1));
        world.insert_static(es[1], C(1));
        world.run_system_once(terms);
        assert_eq!(counts(&world), vec![4, 2, 1, 6, 0]);
    }

    fn double_iteration(q: Query<&A>, mut counts: ResMut<Counts>) {
        let first = q.iter();
        let second = q.iter();
        counts.seen.push(first.count() + second.count());
    }

    #[test]
    fn a_read_only_query_may_be_iterated_twice_at_once() {
        let (mut world, _) = world_with_a(3);
        world.run_system_once(double_iteration);
        assert_eq!(counts(&world), vec![6]);
    }

    // ── Exclusive systems ────────────────────────────────────────────

    fn spawn_two(world: &mut World) {
        for i in 0..2 {
            let e = world.spawn();
            world.insert_static(e, A(100 + i));
        }
    }

    #[test]
    fn a_function_of_the_world_is_an_exclusive_system() {
        let (mut world, _) = world_with_a(1);
        let mut sys = spawn_two.into_system();
        assert!(!sys.is_initialized());
        sys.initialize(&mut world);
        assert!(sys.is_exclusive());
        assert!(sys.access().combined_access().is_exclusive());
        let reader = double_iteration.into_system();
        let mut reader = reader;
        reader.initialize(&mut world);
        assert!(
            !sys.access().is_compatible(reader.access()),
            "an exclusive system conflicts with everything"
        );
        assert!(!reader.is_exclusive());
        world.run_system_once(spawn_two);
        assert_eq!(world.query_data::<&A>().count(), 3);
    }

    #[test]
    fn an_exclusive_closure_keeps_its_captured_state_between_runs() {
        let (mut world, _) = world_with_a(0);
        let mut runs = 0u32;
        let mut sys = (move |w: &mut World| {
            runs += 1;
            let e = w.spawn();
            w.insert_static(e, A(runs));
        })
        .into_system();
        sys.run(&mut world);
        sys.run(&mut world);
        let mut values: Vec<u32> = world.query_data::<&A>().map(|a| a.0).collect();
        values.sort();
        assert_eq!(values, vec![1, 2]);
        assert_eq!(
            sys.meta().last_run(),
            Tick::new(world.change_tick().get() - 1)
        );
    }

    // ── Shapes of function ─────────────────────────────────────────────

    #[allow(clippy::too_many_arguments)]
    fn twelve_params(
        mut a: Local<u32>,
        mut b: Local<u32>,
        mut c: Local<u32>,
        mut d: Local<u32>,
        mut e: Local<u32>,
        mut f: Local<u32>,
        mut g: Local<u32>,
        mut h: Local<u32>,
        mut i: Local<u32>,
        mut j: Local<u32>,
        mut k: Local<u32>,
        mut l: Local<u32>,
    ) {
        *a += 1;
        *b += 1;
        *c += 1;
        *d += 1;
        *e += 1;
        *f += 1;
        *g += 1;
        *h += 1;
        *i += 1;
        *j += 1;
        *k += 1;
        *l += 1;
    }

    fn nested_tuple((q, score): (Query<&A>, Res<Score>), mut counts: ResMut<Counts>) {
        counts.seen.push(q.iter().count() + score.0 as usize);
    }

    fn nothing() {}

    #[test]
    fn functions_of_zero_twelve_and_tuple_parameters_all_run() {
        let (mut world, _) = world_with_a(2);
        world.insert_resource(Score(10));
        world.run_system_once(nothing);
        world.run_system_once(twelve_params);
        world.run_system_once(nested_tuple);
        assert_eq!(counts(&world), vec![12]);
        let mut sys = twelve_params.into_system();
        sys.initialize(&mut world);
        assert_eq!(sys.access().parts().len(), 0, "Locals touch no world data");
    }

    #[test]
    fn closures_whose_parameters_are_annotated_are_systems() {
        let (mut world, _) = world_with_a(3);
        let mut calls = 0u32;
        let mut sys = (move |q: Query<&A>, mut counts: ResMut<Counts>| {
            calls += 1;
            counts.seen.push(q.iter().count() * 10 + calls as usize);
        })
        .into_system();
        sys.run(&mut world);
        sys.run(&mut world);
        assert_eq!(counts(&world), vec![31, 32]);
    }

    #[test]
    fn something_that_already_is_a_system_converts_to_itself() {
        let (mut world, _) = world_with_a(2);
        let built = count_changed.into_system();
        world.run_system_once(built);
        assert_eq!(counts(&world), vec![2]);
    }

    #[test]
    fn a_system_reports_its_access_once_initialized() {
        let (mut world, _) = world_with_a(1);
        let mut sys = nested.into_system();
        assert!(!sys.is_initialized());
        assert!(sys.access().parts().is_empty());
        assert!(sys.name().ends_with("nested"), "{}", sys.name());
        sys.initialize(&mut world);
        sys.initialize(&mut world); // idempotent
        assert!(sys.is_initialized());
        assert_eq!(sys.access().parts().len(), 2);
        let combined = sys.access().combined_access();
        assert!(combined.writes_component::<A>());
        assert!(combined.reads_component::<B>() && !combined.writes_component::<B>());
        assert!(!sys.is_exclusive());
    }

    // ── Running a system by hand, the way a runner will ──────────────

    #[test]
    fn run_unchecked_and_apply_deferred_are_the_two_halves_of_run() {
        let (mut world, _) = world_with_a(3);
        let mut sys = spawner.into_system();
        sys.initialize(&mut world);
        let this_run = world.change_tick();
        {
            let cell = world.as_world_cell();
            // SAFETY: the only access to the world.
            unsafe { sys.run_unchecked(cell, this_run) };
        }
        assert_eq!(
            world.query_data::<Entity>().count(),
            3,
            "nothing applied yet"
        );
        sys.apply_deferred(&mut world);
        assert_eq!(world.query_data::<Entity>().count(), 4);
        assert_eq!(sys.meta().last_run(), this_run);
    }

    #[test]
    #[should_panic(expected = "must be initialized")]
    fn running_an_uninitialized_system_unchecked_is_refused() {
        let (mut world, _) = world_with_a(1);
        let mut sys = count_changed.into_system();
        let cell = world.as_world_cell();
        // SAFETY: refused before anything is touched.
        unsafe { sys.run_unchecked(cell, Tick::new(1)) };
    }

    // ── SystemState ──────────────────────────────────────────────────

    #[test]
    fn system_state_hands_out_parameters_and_applies_commands() {
        let (mut world, _) = world_with_a(3);
        let mut state = SystemState::<(Query<&A, Changed<A>>, Commands)>::new(&mut world);
        {
            let (q, mut cmds) = state.get(&mut world);
            assert_eq!(q.iter().count(), 3, "first call: everything is new");
            cmds.spawn().insert_static(A(50));
        }
        {
            let (q, _) = state.get(&mut world);
            assert_eq!(q.iter().count(), 0, "nothing changed since the last call");
        }
        assert_eq!(world.query_data::<&A>().count(), 3);
        state.apply(&mut world);
        assert_eq!(world.query_data::<&A>().count(), 4);
        assert!(state.meta().name().contains("Query"));
        assert_eq!(state.meta().access().parts().len(), 1);
    }

    // ── Against the world-level change detection ─────────────────────

    #[derive(Default)]
    struct Seen {
        changed: HashSet<Entity>,
        added: HashSet<Entity>,
        either: HashSet<Entity>,
    }

    fn observe(
        changed: Query<Entity, Changed<A>>,
        added: Query<Entity, Added<A>>,
        either: Query<Entity, Or<(Changed<A>, Changed<B>)>>,
        mut seen: ResMut<Seen>,
    ) {
        seen.changed = changed.iter().collect();
        seen.added = added.iter().collect();
        seen.either = either.iter().collect();
    }

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

    /// The system's change detection against `World::query_data_*_since`
    /// with a `ChangeTracker`, over a long random history of spawns,
    /// despawns, migrations, writes and tick steps. The tracker is updated
    /// at the tick the next run will use as its `this_run`, so the two
    /// measure "since the previous run" identically; any slip in which tick
    /// a system runs at, stamps with, or measures from shows up as a set
    /// mismatch on some step.
    #[test]
    fn a_systems_change_detection_matches_a_tracker_over_a_random_history() {
        let mut rng = Lcg(0xD1B54A32D192ED03);
        let mut world = World::new();
        world.insert_resource(Seen::default());
        let mut sys = observe.into_system();
        let mut tracker = ChangeTracker::new();
        let mut alive: Vec<Entity> = Vec::new();
        let (mut nonempty, mut empty) = (0, 0);

        for step in 0..500u32 {
            for _ in 0..rng.next(4) {
                match rng.next(8) {
                    0 | 1 => {
                        let e = world.spawn();
                        world.insert_static(e, A(step));
                        if rng.next(2) == 0 {
                            world.insert_static(e, B(step));
                        }
                        alive.push(e);
                    }
                    2 if !alive.is_empty() => {
                        let e = alive.swap_remove(rng.next(alive.len()));
                        world.despawn(e);
                    }
                    3 if !alive.is_empty() => {
                        let e = alive[rng.next(alive.len())];
                        world.insert_static(e, B(step));
                    }
                    4 if !alive.is_empty() => {
                        let e = alive[rng.next(alive.len())];
                        world.remove_static::<A>(e);
                    }
                    5 | 6 if !alive.is_empty() => {
                        let e = alive[rng.next(alive.len())];
                        if let Some(a) = world.get_static_mut::<A>(e) {
                            a.0 += 1;
                        }
                    }
                    7 if !alive.is_empty() => {
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

            // Evaluated at the tick the run below will use as `this_run`.
            let want_changed: HashSet<Entity> = world
                .query_data_filtered_since::<Entity, Changed<A>>(&tracker)
                .collect();
            let want_added: HashSet<Entity> = world
                .query_data_filtered_since::<Entity, Added<A>>(&tracker)
                .collect();
            let want_either: HashSet<Entity> = world
                .query_data_filtered_since::<Entity, Or<(Changed<A>, Changed<B>)>>(&tracker)
                .collect();
            tracker.update(&world);

            sys.run(&mut world);

            let seen = world.get_resource::<Seen>().expect("Seen");
            assert_eq!(seen.changed, want_changed, "step {step}: Changed<A>");
            assert_eq!(seen.added, want_added, "step {step}: Added<A>");
            assert_eq!(
                seen.either, want_either,
                "step {step}: Or<Changed A, Changed B>"
            );
            if want_changed.is_empty() {
                empty += 1;
            } else {
                nonempty += 1;
            }
        }
        assert!(
            nonempty > 100 && empty > 50,
            "{nonempty} nonempty, {empty} empty"
        );
    }

    // ── Parallel iteration ───────────────────────────────────────────

    use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
    use std::sync::Mutex;
    use std::thread::ThreadId;

    use crate::batching::BatchingStrategy;

    #[derive(Default)]
    struct Totals {
        sum: AtomicU64,
        rows: AtomicUsize,
        inits: AtomicUsize,
    }

    #[derive(Default)]
    struct Threads(Mutex<HashSet<ThreadId>>);

    /// A world of `n` entities with `A(i)`, every third also with `B(i)`,
    /// so there are two archetypes of different sizes.
    fn mixed_world(n: u32) -> (World, Vec<Entity>) {
        let (mut world, es) = world_with_a(n);
        for (i, &e) in es.iter().enumerate() {
            if i % 3 == 0 {
                world.insert_static(e, B(i as u32));
            }
        }
        world.insert_resource(Totals::default());
        (world, es)
    }

    fn a_values(world: &World, es: &[Entity]) -> Vec<Option<u32>> {
        es.iter()
            .map(|&e| world.get_static::<A>(e).map(|a| a.0))
            .collect()
    }

    /// Runs `f` on a rayon pool of `threads` threads, so a parallel query
    /// inside it sizes its batches for that pool. `World` is deliberately
    /// not `Send`, so `f` is handed to the pool through a wrapper that
    /// says it is.
    fn in_pool(threads: usize, f: impl FnOnce()) {
        struct AssertSend<T>(T);
        // SAFETY: the calling thread blocks in `install` until `f` has
        // finished, so whatever `f` borrows (the world, the system) is
        // used by exactly one thread at a time, and never outlives the
        // call.
        unsafe impl<T> Send for AssertSend<T> {}
        let f = AssertSend(f);
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("a rayon pool")
            .install(move || {
                let f = f;
                (f.0)()
            });
    }

    fn par_sum(q: Query<&A>, totals: Res<Totals>) {
        q.par_iter().for_each(|a| {
            totals.sum.fetch_add(u64::from(a.0), Ordering::Relaxed);
            totals.rows.fetch_add(1, Ordering::Relaxed);
        });
    }

    #[test]
    #[cfg(not(miri))]
    fn par_iter_visits_every_row_once_with_the_right_values() {
        for threads in [1usize, 2, 4, 7] {
            let (mut world, _) = mixed_world(300);
            in_pool(threads, || world.run_system_once(par_sum));
            let totals = world.get_resource::<Totals>().unwrap();
            assert_eq!(
                totals.rows.load(Ordering::Relaxed),
                300,
                "{threads} threads"
            );
            assert_eq!(
                totals.sum.load(Ordering::Relaxed),
                (0..300u64).sum::<u64>(),
                "{threads} threads"
            );
        }
    }

    fn seq_write(mut q: Query<(Entity, Mut<A>)>) {
        for (e, mut a) in q.iter_mut() {
            a.0 = a.0 * 3 + e.index();
        }
    }

    fn par_write(mut q: Query<(Entity, Mut<A>)>) {
        q.par_iter_mut().for_each(|(e, mut a)| {
            a.0 = a.0 * 3 + e.index();
        });
    }

    #[test]
    #[cfg(not(miri))]
    fn par_iter_mut_leaves_the_world_as_iter_mut_does() {
        let (mut seq, es) = mixed_world(500);
        let (mut par, _) = mixed_world(500);
        seq.run_system_once(seq_write);
        in_pool(4, || par.run_system_once(par_write));
        assert_eq!(a_values(&par, &es), a_values(&seq, &es));
        assert_eq!(a_values(&par, &es)[7], Some(7 * 3 + es[7].index()));
    }

    fn init_per_batch(q: Query<&A>, totals: Res<Totals>) {
        q.par_iter()
            .batching_strategy(BatchingStrategy::fixed(10))
            .for_each_init(
                || {
                    totals.inits.fetch_add(1, Ordering::Relaxed);
                    0usize
                },
                |seen_in_batch, _a| {
                    *seen_in_batch += 1;
                    assert!(
                        *seen_in_batch <= 10,
                        "a scratch value was shared by batches"
                    );
                    totals.rows.fetch_add(1, Ordering::Relaxed);
                },
            );
    }

    #[test]
    #[cfg(not(miri))]
    fn for_each_init_runs_init_once_per_batch_and_not_per_row() {
        let (mut world, _) = world_with_a(95);
        world.insert_resource(Totals::default());
        in_pool(4, || world.run_system_once(init_per_batch));
        let totals = world.get_resource::<Totals>().unwrap();
        assert_eq!(totals.rows.load(Ordering::Relaxed), 95);
        assert_eq!(totals.inits.load(Ordering::Relaxed), 10, "ceil(95 / 10)");
    }

    fn par_bump_changed(mut q: Query<Mut<A>, Changed<A>>, totals: Res<Totals>) {
        q.par_iter_mut().for_each(|mut a| {
            a.0 += 1;
            totals.rows.fetch_add(1, Ordering::Relaxed);
        });
    }

    #[test]
    #[cfg(not(miri))]
    fn a_parallel_systems_own_writes_are_invisible_to_its_next_run_but_others_are_seen() {
        let (mut world, es) = mixed_world(40);
        let mut sys = par_bump_changed.into_system();
        let rows = |world: &World| {
            world
                .get_resource::<Totals>()
                .unwrap()
                .rows
                .load(Ordering::Relaxed)
        };
        in_pool(4, || sys.run(&mut world));
        assert_eq!(rows(&world), 40);
        in_pool(4, || sys.run(&mut world));
        assert_eq!(rows(&world), 40, "its own writes are not changes to it");
        world.get_static_mut::<A>(es[21]).unwrap().0 = 900;
        in_pool(4, || sys.run(&mut world));
        assert_eq!(rows(&world), 41);
        assert_eq!(world.get_static::<A>(es[21]).map(|a| a.0), Some(901));
    }

    fn record_threads(q: Query<&A>, threads: Res<Threads>) {
        q.par_iter()
            .batching_strategy(BatchingStrategy::fixed(1))
            .for_each(|_a| {
                threads
                    .0
                    .lock()
                    .unwrap()
                    .insert(std::thread::current().id());
                std::thread::sleep(std::time::Duration::from_millis(30));
            });
    }

    #[test]
    #[cfg(not(miri))]
    fn par_iter_really_runs_batches_on_several_threads() {
        let (mut world, _) = world_with_a(8);
        world.insert_resource(Threads::default());
        in_pool(4, || world.run_system_once(record_threads));
        let used = world
            .get_resource::<Threads>()
            .unwrap()
            .0
            .lock()
            .unwrap()
            .len();
        assert!(used >= 2, "all 8 batches ran on {used} thread(s)");
    }

    fn two_writers(mut a: Query<Mut<A>>, mut b: Query<Mut<B>>) {
        a.par_iter_mut().for_each(|mut a| a.0 += 1);
        b.par_iter_mut().for_each(|mut b| b.0 += 1000);
    }

    #[test]
    #[cfg(not(miri))]
    fn two_parallel_queries_of_different_components_work_in_one_system() {
        let (mut world, es) = mixed_world(30);
        in_pool(3, || world.run_system_once(two_writers));
        assert_eq!(world.get_static::<A>(es[4]).map(|a| a.0), Some(5));
        assert_eq!(world.get_static::<B>(es[3]).map(|b| b.0), Some(3 + 1000));
    }

    fn par_filtered_read(
        q: Query<(Entity, Ref<A>, Option<&B>), (With<A>, Without<C>)>,
        totals: Res<Totals>,
    ) {
        q.par_iter().for_each(|(e, a, b)| {
            totals.sum.fetch_add(
                u64::from(e.index()) + u64::from(a.0) + b.map_or(0, |b| u64::from(b.0)),
                Ordering::Relaxed,
            );
        });
    }

    fn seq_filtered_read(
        q: Query<(Entity, Ref<A>, Option<&B>), (With<A>, Without<C>)>,
        totals: Res<Totals>,
    ) {
        for (e, a, b) in q.iter() {
            totals.sum.fetch_add(
                u64::from(e.index()) + u64::from(a.0) + b.map_or(0, |b| u64::from(b.0)),
                Ordering::Relaxed,
            );
        }
    }

    #[test]
    #[cfg(not(miri))]
    fn a_filtered_read_with_optional_and_ref_terms_matches_the_sequential_one() {
        let (mut seq, es) = mixed_world(120);
        let (mut par, _) = mixed_world(120);
        for &e in es.iter().step_by(5) {
            seq.insert_static(e, C(1));
            par.insert_static(e, C(1));
        }
        seq.run_system_once(seq_filtered_read);
        in_pool(4, || par.run_system_once(par_filtered_read));
        let total = |w: &World| {
            w.get_resource::<Totals>()
                .unwrap()
                .sum
                .load(Ordering::Relaxed)
        };
        assert!(total(&seq) > 0);
        assert_eq!(total(&par), total(&seq));
    }

    fn batches_on_scoped_threads(mut q: Query<(Entity, Mut<A>)>) {
        let batches = q
            .par_iter_mut()
            .batching_strategy(BatchingStrategy::fixed(3))
            .into_batches();
        assert!(batches.len() >= 3);
        std::thread::scope(|scope| {
            for batch in batches {
                scope.spawn(move || {
                    for (e, mut a) in batch {
                        a.0 = a.0 * 3 + e.index();
                    }
                });
            }
        });
    }

    #[test]
    fn into_batches_hands_out_batches_for_a_pool_of_your_own() {
        // No rayon here: plain scoped threads, so Miri can run it.
        let (mut seq, es) = mixed_world(12);
        let (mut par, _) = mixed_world(12);
        seq.run_system_once(seq_write);
        par.run_system_once(batches_on_scoped_threads);
        assert_eq!(a_values(&par, &es), a_values(&seq, &es));
        assert_eq!(a_values(&par, &es)[5], Some(5 * 3 + es[5].index()));
    }

    fn count_default_batches(q: Query<&A>, totals: Res<Totals>) {
        totals
            .rows
            .store(q.par_iter().into_batches().len(), Ordering::Relaxed);
    }

    #[test]
    #[cfg(not(miri))]
    fn the_default_batching_sizes_batches_for_the_pool_it_runs_in() {
        // One archetype of 400 rows: one batch per thread by default.
        for threads in [1usize, 2, 4] {
            let (mut world, _) = world_with_a(400);
            world.insert_resource(Totals::default());
            in_pool(threads, || world.run_system_once(count_default_batches));
            let batches = world
                .get_resource::<Totals>()
                .unwrap()
                .rows
                .load(Ordering::Relaxed);
            assert_eq!(batches, threads, "{threads}-thread pool");
        }
    }

    fn par_count_changed(q: Query<Entity, Changed<A>>, totals: Res<Totals>) {
        q.par_iter().for_each(|_e| {
            totals.rows.fetch_add(1, Ordering::Relaxed);
        });
    }

    #[test]
    #[cfg(not(miri))]
    fn a_read_only_parallel_query_measures_changes_since_the_systems_last_run() {
        let (mut world, es) = mixed_world(40);
        let mut sys = par_count_changed.into_system();
        let rows = |world: &World| {
            world
                .get_resource::<Totals>()
                .unwrap()
                .rows
                .load(Ordering::Relaxed)
        };
        in_pool(4, || sys.run(&mut world));
        assert_eq!(rows(&world), 40, "everything is new to a first run");
        in_pool(4, || sys.run(&mut world));
        assert_eq!(rows(&world), 40, "nothing changed since the first run");
        world.get_static_mut::<A>(es[9]).unwrap().0 = 1;
        world.get_static_mut::<A>(es[30]).unwrap().0 = 2;
        in_pool(4, || sys.run(&mut world));
        assert_eq!(rows(&world), 42, "exactly the two outside writes");
    }
}
