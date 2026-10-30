// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-ecs-schedule.md, section "condition.rs"
// ============================================================================
//! Run conditions: systems that return `bool`, decide whether another
//! system (or a whole set) runs this time, and may only read.
//!
//! ```
//! use mid_ecs::schedule::{common_conditions::resource_changed, Condition, IntoCondition};
//! use mid_ecs::World;
//!
//! struct Gauge(u32);
//!
//! fn gauge_is_high(gauge: mid_ecs::Res<Gauge>) -> bool {
//!     gauge.0 > 10
//! }
//!
//! let mut world = World::new();
//! world.insert_resource(Gauge(3));
//! let mut high = gauge_is_high.into_condition();
//! assert!(!high.run(&mut world));
//! world.get_resource_mut::<Gauge>().unwrap().0 = 30;
//! assert!(high.run(&mut world));
//!
//! let mut changed = resource_changed::<Gauge>.into_condition();
//! assert!(changed.run(&mut world)); // first run: everything is new
//! assert!(!changed.run(&mut world));
//! ```

use crate::access::FilteredAccessSet;
use crate::system::{FunctionSystem, SystemParamFunction};
use crate::tick::Tick;
use crate::world::World;
use crate::world_cell::WorldCell;

// ── The trait ────────────────────────────────────────────────────────

/// A read-only system that returns `bool`.
///
/// It is set up and run like a [`crate::System`] and shares the machinery
/// for it (its parameters, its last run, its tick), but its parameters may
/// only read, and nothing it defers (`Commands`) is ever applied.
pub trait Condition: 'static {
    /// A name for messages.
    fn name(&self) -> &str;

    /// Builds the condition's parameters' state. Does nothing if already
    /// done.
    ///
    /// # Panics
    /// If two of its parameters conflict, or any of them writes.
    fn initialize(&mut self, world: &mut World);

    /// Whether [`Self::initialize`] has run.
    fn is_initialized(&self) -> bool;

    /// What the condition touches. Empty until initialized.
    fn access(&self) -> &FilteredAccessSet;

    /// Evaluates the condition at `this_run`.
    ///
    /// # Panics
    /// If the condition has not been initialized, or a resource it
    /// requires does not exist.
    ///
    /// # Safety
    /// For the duration of the call nothing may write anything in
    /// [`Self::access`].
    unsafe fn run_unchecked(&mut self, world: WorldCell<'_>, this_run: Tick) -> bool;

    /// The previous run's tick, or for a condition made of several, the one
    /// furthest back from `now`.
    fn oldest_last_run(&self, now: Tick) -> Tick;

    /// Keeps the last-run ticks within [`crate::MAX_CHANGE_AGE`] of `now`;
    /// whether any was clamped. The owner calls this as it does for a
    /// system's (see [`crate::System::check_change_tick`]).
    fn check_change_tick(&mut self, now: Tick) -> bool;

    /// Initializes if needed, evaluates at the world's current tick and
    /// advances it, as [`crate::System::run`] does.
    fn run(&mut self, world: &mut World) -> bool {
        self.initialize(world);
        let this_run = world.change_tick();
        world.increment_change_tick();
        // SAFETY: `world` is borrowed exclusively and the cell is gone
        // before it is used again.
        unsafe { self.run_unchecked(world.as_world_cell(), this_run) }
    }
}

/// A condition behind a pointer, as a schedule stores it.
pub type BoxedCondition = Box<dyn Condition>;

impl Condition for Box<dyn Condition> {
    fn name(&self) -> &str {
        (**self).name()
    }

    fn initialize(&mut self, world: &mut World) {
        (**self).initialize(world);
    }

    fn is_initialized(&self) -> bool {
        (**self).is_initialized()
    }

    fn access(&self) -> &FilteredAccessSet {
        (**self).access()
    }

    unsafe fn run_unchecked(&mut self, world: WorldCell<'_>, this_run: Tick) -> bool {
        // SAFETY: forwarded.
        unsafe { (**self).run_unchecked(world, this_run) }
    }

    fn oldest_last_run(&self, now: Tick) -> Tick {
        (**self).oldest_last_run(now)
    }

    fn check_change_tick(&mut self, now: Tick) -> bool {
        (**self).check_change_tick(now)
    }

    fn run(&mut self, world: &mut World) -> bool {
        (**self).run(world)
    }
}

// ── IntoCondition ────────────────────────────────────────────────────

/// Something that becomes a [`Condition`]: a function of read-only system
/// parameters that returns `bool`, or a condition.
///
/// `Marker` only keeps the impls apart; it is inferred.
pub trait IntoCondition<Marker>: Sized {
    /// The condition this becomes.
    type Condition: Condition;

    /// Converts.
    fn into_condition(self) -> Self::Condition;
}

/// Marker for something that already is a [`Condition`].
pub struct IsCondition;
/// Marker for a function of system parameters returning `bool`.
pub struct IsFunctionCondition;

impl<C: Condition> IntoCondition<IsCondition> for C {
    type Condition = C;

    fn into_condition(self) -> C {
        self
    }
}

impl<Marker: 'static, F: SystemParamFunction<Marker, Out = bool>>
    IntoCondition<(IsFunctionCondition, Marker)> for F
{
    type Condition = FunctionSystem<Marker, F>;

    fn into_condition(self) -> Self::Condition {
        FunctionSystem::new(self)
    }
}

impl<Marker: 'static, F: SystemParamFunction<Marker, Out = bool>> Condition
    for FunctionSystem<Marker, F>
{
    fn name(&self) -> &str {
        self.meta().name()
    }

    fn initialize(&mut self, world: &mut World) {
        if self.params_ready() {
            return;
        }
        self.init_params(world);
        assert!(
            self.meta().access().combined_access().is_read_only(),
            "run condition `{}` has a parameter that writes: a condition may only read",
            self.meta().name()
        );
    }

    fn is_initialized(&self) -> bool {
        self.params_ready()
    }

    fn access(&self) -> &FilteredAccessSet {
        self.meta().access()
    }

    unsafe fn run_unchecked(&mut self, world: WorldCell<'_>, this_run: Tick) -> bool {
        // SAFETY: the caller's contract is `call`'s.
        unsafe { self.call(world, this_run) }
    }

    fn oldest_last_run(&self, _now: Tick) -> Tick {
        self.meta().last_run()
    }

    fn check_change_tick(&mut self, now: Tick) -> bool {
        self.check_tick(now)
    }
}

// ── Combinators ──────────────────────────────────────────────────────

fn older_of(now: Tick, a: Tick, b: Tick) -> Tick {
    if now.relative_to(a).get() >= now.relative_to(b).get() {
        a
    } else {
        b
    }
}

fn union_access(a: &FilteredAccessSet, b: &FilteredAccessSet) -> FilteredAccessSet {
    let mut out = FilteredAccessSet::new();
    for part in a.parts().iter().chain(b.parts()) {
        out.add(part.clone());
    }
    out
}

/// The opposite of a condition. See [`not`].
pub struct NotCondition<C> {
    inner: C,
    name: String,
}

/// `true` where `condition` is `false`.
pub fn not<M, C: IntoCondition<M>>(condition: C) -> NotCondition<C::Condition> {
    let inner = condition.into_condition();
    NotCondition {
        name: format!("!{}", inner.name()),
        inner,
    }
}

impl<C: Condition> Condition for NotCondition<C> {
    fn name(&self) -> &str {
        &self.name
    }

    fn initialize(&mut self, world: &mut World) {
        self.inner.initialize(world);
    }

    fn is_initialized(&self) -> bool {
        self.inner.is_initialized()
    }

    fn access(&self) -> &FilteredAccessSet {
        self.inner.access()
    }

    unsafe fn run_unchecked(&mut self, world: WorldCell<'_>, this_run: Tick) -> bool {
        // SAFETY: forwarded.
        !unsafe { self.inner.run_unchecked(world, this_run) }
    }

    fn oldest_last_run(&self, now: Tick) -> Tick {
        self.inner.oldest_last_run(now)
    }

    fn check_change_tick(&mut self, now: Tick) -> bool {
        self.inner.check_change_tick(now)
    }
}

/// Two conditions that must both hold. See [`and`].
pub struct AndCondition<A, B> {
    a: A,
    b: B,
    name: String,
    access: FilteredAccessSet,
    initialized: bool,
}

/// `true` where `a` and `b` both are. `b` is not evaluated when `a` is
/// `false`, so its last run does not advance and what it missed
/// accumulates, as for a skipped system.
pub fn and<MA, MB, A: IntoCondition<MA>, B: IntoCondition<MB>>(
    a: A,
    b: B,
) -> AndCondition<A::Condition, B::Condition> {
    let (a, b) = (a.into_condition(), b.into_condition());
    AndCondition {
        name: format!("{} && {}", a.name(), b.name()),
        a,
        b,
        access: FilteredAccessSet::new(),
        initialized: false,
    }
}

impl<A: Condition, B: Condition> Condition for AndCondition<A, B> {
    fn name(&self) -> &str {
        &self.name
    }

    fn initialize(&mut self, world: &mut World) {
        if self.initialized {
            return;
        }
        self.a.initialize(world);
        self.b.initialize(world);
        self.access = union_access(self.a.access(), self.b.access());
        self.initialized = true;
    }

    fn is_initialized(&self) -> bool {
        self.initialized
    }

    fn access(&self) -> &FilteredAccessSet {
        &self.access
    }

    unsafe fn run_unchecked(&mut self, world: WorldCell<'_>, this_run: Tick) -> bool {
        // SAFETY: both only read, and the caller's contract covers the
        // union of what they read.
        unsafe { self.a.run_unchecked(world, this_run) && self.b.run_unchecked(world, this_run) }
    }

    fn oldest_last_run(&self, now: Tick) -> Tick {
        older_of(
            now,
            self.a.oldest_last_run(now),
            self.b.oldest_last_run(now),
        )
    }

    fn check_change_tick(&mut self, now: Tick) -> bool {
        let a = self.a.check_change_tick(now);
        let b = self.b.check_change_tick(now);
        a | b
    }
}

/// Two conditions of which one must hold. See [`or`].
pub struct OrCondition<A, B> {
    a: A,
    b: B,
    name: String,
    access: FilteredAccessSet,
    initialized: bool,
}

/// `true` where `a` or `b` is. `b` is not evaluated when `a` is `true`
/// (see [`and`] for what that means for its last run).
pub fn or<MA, MB, A: IntoCondition<MA>, B: IntoCondition<MB>>(
    a: A,
    b: B,
) -> OrCondition<A::Condition, B::Condition> {
    let (a, b) = (a.into_condition(), b.into_condition());
    OrCondition {
        name: format!("{} || {}", a.name(), b.name()),
        a,
        b,
        access: FilteredAccessSet::new(),
        initialized: false,
    }
}

impl<A: Condition, B: Condition> Condition for OrCondition<A, B> {
    fn name(&self) -> &str {
        &self.name
    }

    fn initialize(&mut self, world: &mut World) {
        if self.initialized {
            return;
        }
        self.a.initialize(world);
        self.b.initialize(world);
        self.access = union_access(self.a.access(), self.b.access());
        self.initialized = true;
    }

    fn is_initialized(&self) -> bool {
        self.initialized
    }

    fn access(&self) -> &FilteredAccessSet {
        &self.access
    }

    unsafe fn run_unchecked(&mut self, world: WorldCell<'_>, this_run: Tick) -> bool {
        // SAFETY: as for `AndCondition`.
        unsafe { self.a.run_unchecked(world, this_run) || self.b.run_unchecked(world, this_run) }
    }

    fn oldest_last_run(&self, now: Tick) -> Tick {
        older_of(
            now,
            self.a.oldest_last_run(now),
            self.b.oldest_last_run(now),
        )
    }

    fn check_change_tick(&mut self, now: Tick) -> bool {
        let a = self.a.check_change_tick(now);
        let b = self.b.check_change_tick(now);
        a | b
    }
}

// ── Ready-made conditions ────────────────────────────────────────────

/// Conditions for the cases that come up in every game. Each is a
/// function, so `run_if(resource_changed::<Gauge>)` names it directly.
pub mod common_conditions {
    use crate::system_param::Res;

    pub use super::{and, not, or};

    /// `true` while a resource of type `T` exists.
    pub fn resource_exists<T: 'static>(resource: Option<Res<T>>) -> bool {
        resource.is_some()
    }

    /// `true` if `T` was inserted since this condition last ran (on its
    /// first run, if it exists at all).
    ///
    /// # Panics
    /// If `T` does not exist. See [`resource_exists_and_added`].
    pub fn resource_added<T: 'static>(resource: Res<T>) -> bool {
        resource.is_added()
    }

    /// `true` if `T` was inserted, replaced or written since this
    /// condition last ran.
    ///
    /// # Panics
    /// If `T` does not exist. See [`resource_exists_and_changed`].
    pub fn resource_changed<T: 'static>(resource: Res<T>) -> bool {
        resource.is_changed()
    }

    /// [`resource_added`] that is `false` instead of panicking when `T`
    /// does not exist.
    pub fn resource_exists_and_added<T: 'static>(resource: Option<Res<T>>) -> bool {
        resource.is_some_and(|r| r.is_added())
    }

    /// [`resource_changed`] that is `false` instead of panicking when `T`
    /// does not exist.
    pub fn resource_exists_and_changed<T: 'static>(resource: Option<Res<T>>) -> bool {
        resource.is_some_and(|r| r.is_changed())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU32, Ordering};

    use super::common_conditions::*;
    use super::*;
    use crate::system_param::{Local, Query, Res, ResMut};
    use crate::tick::{Mut, MAX_CHANGE_AGE};

    struct Flag(bool);
    struct Gauge(u32);
    struct Evals(AtomicU32);
    struct Other(u32);
    struct Comp(u32);

    fn flag(f: Res<Flag>) -> bool {
        f.0
    }

    fn gauge_high(g: Res<Gauge>) -> bool {
        g.0 > 10
    }

    fn always_true(e: Res<Evals>) -> bool {
        e.0.fetch_add(1, Ordering::SeqCst);
        true
    }

    fn always_false(e: Res<Evals>) -> bool {
        e.0.fetch_add(1, Ordering::SeqCst);
        false
    }

    fn evals(world: &World) -> u32 {
        world
            .get_resource::<Evals>()
            .unwrap()
            .0
            .load(Ordering::SeqCst)
    }

    fn reset(world: &World) {
        world
            .get_resource::<Evals>()
            .unwrap()
            .0
            .store(0, Ordering::SeqCst);
    }

    fn world() -> World {
        let mut w = World::new();
        w.insert_resource(Flag(true));
        w.insert_resource(Gauge(3));
        w.insert_resource(Evals(AtomicU32::new(0)));
        w
    }

    #[test]
    fn a_function_condition_returns_what_the_function_returns() {
        let mut w = world();
        let mut c = flag.into_condition();
        assert!(c.run(&mut w));
        w.get_resource_mut::<Flag>().unwrap().0 = false;
        assert!(!c.run(&mut w));
    }

    #[test]
    fn a_condition_reads_the_world_it_runs_against() {
        let mut w = world();
        let mut high = gauge_high.into_condition();
        assert!(!high.run(&mut w));
        w.get_resource_mut::<Gauge>().unwrap().0 = 11;
        assert!(high.run(&mut w));
        w.get_resource_mut::<Gauge>().unwrap().0 = 10;
        assert!(!high.run(&mut w), "10 is not above 10");
    }

    #[test]
    fn running_a_condition_advances_the_world_tick_by_one() {
        let mut w = world();
        let mut c = flag.into_condition();
        let before = w.change_tick();
        c.run(&mut w);
        assert_eq!(w.change_tick().get(), before.get() + 1);
        c.run(&mut w);
        assert_eq!(w.change_tick().get(), before.get() + 2);
    }

    #[test]
    fn a_condition_is_initialized_by_its_first_run() {
        let mut w = world();
        let mut c = flag.into_condition();
        assert!(!c.is_initialized());
        assert!(c.access().parts().is_empty());
        c.run(&mut w);
        assert!(c.is_initialized());
        assert!(c.access().combined_access().reads_resource::<Flag>());
        assert!(!c.access().combined_access().writes_resource::<Flag>());
    }

    #[test]
    fn initializing_twice_changes_nothing() {
        let mut w = world();
        let mut c = flag.into_condition();
        c.initialize(&mut w);
        let parts = c.access().parts().len();
        c.initialize(&mut w);
        assert_eq!(c.access().parts().len(), parts);
    }

    #[test]
    fn a_condition_has_a_name() {
        let c = flag.into_condition();
        assert!(c.name().contains("flag"), "{}", c.name());
    }

    // ── read-only ────────────────────────────────────────────────────

    fn writes_resource(mut g: ResMut<Gauge>) -> bool {
        g.0 += 1;
        true
    }

    fn writes_component(mut q: Query<Mut<Comp>>) -> bool {
        q.iter_mut().count() > 0
    }

    fn reads_everything(_g: Res<Gauge>, _q: Query<&Comp>, _l: Local<u32>) -> bool {
        true
    }

    #[test]
    #[should_panic(expected = "may only read")]
    fn a_condition_that_writes_a_resource_is_refused() {
        let mut w = world();
        writes_resource.into_condition().initialize(&mut w);
    }

    #[test]
    #[should_panic(expected = "may only read")]
    fn a_condition_that_writes_a_component_is_refused() {
        let mut w = world();
        writes_component.into_condition().initialize(&mut w);
    }

    #[test]
    fn reading_parameters_and_locals_are_accepted() {
        let mut w = world();
        assert!(reads_everything.into_condition().run(&mut w));
    }

    // ── resource conditions ──────────────────────────────────────────

    #[test]
    fn resource_exists_follows_the_resource() {
        let mut w = World::new();
        let mut c = resource_exists::<Gauge>.into_condition();
        assert!(!c.run(&mut w));
        w.insert_resource(Gauge(1));
        assert!(c.run(&mut w));
        w.remove_resource::<Gauge>();
        assert!(!c.run(&mut w));
    }

    #[test]
    fn resource_added_is_true_once_per_insertion() {
        let mut w = world();
        let mut c = resource_added::<Gauge>.into_condition();
        assert!(
            c.run(&mut w),
            "first run: it exists, so it is new to this condition"
        );
        assert!(!c.run(&mut w));
        w.get_resource_mut::<Gauge>().unwrap().0 = 9;
        assert!(!c.run(&mut w), "a change is not an addition");
        w.remove_resource::<Gauge>();
        w.insert_resource(Gauge(1));
        assert!(c.run(&mut w), "a fresh insertion is");
        assert!(!c.run(&mut w));
    }

    #[test]
    fn resource_changed_is_true_after_a_write_and_accumulates_while_unread() {
        let mut w = world();
        let mut c = resource_changed::<Gauge>.into_condition();
        assert!(c.run(&mut w));
        assert!(!c.run(&mut w));
        w.get_resource_mut::<Gauge>().unwrap().0 = 5;
        assert!(c.run(&mut w));
        assert!(!c.run(&mut w));
        // two writes between runs are still one `true`
        w.get_resource_mut::<Gauge>().unwrap().0 = 6;
        w.get_resource_mut::<Gauge>().unwrap().0 = 7;
        assert!(c.run(&mut w));
        assert!(!c.run(&mut w));
        w.insert_resource(Gauge(0));
        assert!(c.run(&mut w), "a replacement is a change");
    }

    #[test]
    fn two_conditions_measure_change_independently() {
        let mut w = world();
        let mut a = resource_changed::<Gauge>.into_condition();
        let mut b = resource_changed::<Gauge>.into_condition();
        assert!(a.run(&mut w));
        assert!(!a.run(&mut w));
        assert!(b.run(&mut w), "b has not run, so it has not seen anything");
        assert!(!b.run(&mut w));
    }

    #[test]
    #[should_panic(expected = "Gauge")]
    fn resource_changed_panics_on_a_missing_resource() {
        let mut w = World::new();
        resource_changed::<Gauge>.into_condition().run(&mut w);
    }

    #[test]
    fn the_optional_forms_are_false_instead_of_panicking() {
        let mut w = World::new();
        let mut added = resource_exists_and_added::<Gauge>.into_condition();
        let mut changed = resource_exists_and_changed::<Gauge>.into_condition();
        assert!(!added.run(&mut w));
        assert!(!changed.run(&mut w));
        w.insert_resource(Gauge(1));
        assert!(added.run(&mut w));
        assert!(changed.run(&mut w));
        assert!(!added.run(&mut w));
        assert!(!changed.run(&mut w));
        w.get_resource_mut::<Gauge>().unwrap().0 = 2;
        assert!(!added.run(&mut w));
        assert!(changed.run(&mut w));
    }

    // ── combinators ──────────────────────────────────────────────────

    #[test]
    fn not_inverts() {
        let mut w = world();
        let mut c = not(flag);
        assert!(!c.run(&mut w));
        w.get_resource_mut::<Flag>().unwrap().0 = false;
        assert!(c.run(&mut w));
        assert!(c.name().starts_with('!'), "{}", c.name());
    }

    #[test]
    fn and_is_a_truth_table() {
        for (a, b) in [(false, false), (false, true), (true, false), (true, true)] {
            let mut w = world();
            w.get_resource_mut::<Flag>().unwrap().0 = a;
            w.get_resource_mut::<Gauge>().unwrap().0 = if b { 50 } else { 0 };
            assert_eq!(and(flag, gauge_high).run(&mut w), a && b, "{a} && {b}");
        }
    }

    #[test]
    fn or_is_a_truth_table() {
        for (a, b) in [(false, false), (false, true), (true, false), (true, true)] {
            let mut w = world();
            w.get_resource_mut::<Flag>().unwrap().0 = a;
            w.get_resource_mut::<Gauge>().unwrap().0 = if b { 50 } else { 0 };
            assert_eq!(or(flag, gauge_high).run(&mut w), a || b, "{a} || {b}");
        }
    }

    #[test]
    fn and_does_not_evaluate_the_second_after_a_false_first() {
        let mut w = world();
        assert!(!and(always_false, always_true).run(&mut w));
        assert_eq!(evals(&w), 1);
        reset(&w);
        assert!(and(always_true, always_true).run(&mut w));
        assert_eq!(evals(&w), 2);
    }

    #[test]
    fn or_does_not_evaluate_the_second_after_a_true_first() {
        let mut w = world();
        assert!(or(always_true, always_false).run(&mut w));
        assert_eq!(evals(&w), 1);
        reset(&w);
        assert!(or(always_false, always_true).run(&mut w));
        assert_eq!(evals(&w), 2);
    }

    #[test]
    fn a_skipped_half_keeps_what_it_missed() {
        let mut w = world();
        let mut c = and(flag, resource_changed::<Gauge>);
        w.get_resource_mut::<Flag>().unwrap().0 = false;
        assert!(!c.run(&mut w), "flag is false: the change is not looked at");
        w.get_resource_mut::<Flag>().unwrap().0 = true;
        assert!(c.run(&mut w), "and now the change it missed is seen");
        assert!(!c.run(&mut w));
    }

    #[test]
    fn combinators_nest_and_take_boxed_conditions() {
        let mut w = world();
        let boxed: BoxedCondition = Box::new(gauge_high.into_condition());
        let mut c = not(and(flag, boxed));
        assert!(
            c.run(&mut w),
            "flag is true, gauge is low: and is false, not is true"
        );
        w.get_resource_mut::<Gauge>().unwrap().0 = 40;
        assert!(!c.run(&mut w));
        let mut d = or(not(flag), not(gauge_high));
        assert!(!d.run(&mut w));
    }

    #[test]
    fn a_combined_condition_reports_both_halves_access() {
        let mut w = world();
        w.insert_resource(Other(1));
        let mut c = and(flag, resource_exists::<Other>);
        assert!(c.access().parts().is_empty());
        c.initialize(&mut w);
        assert!(c.is_initialized());
        let acc = c.access().combined_access();
        assert!(acc.reads_resource::<Flag>());
        assert!(acc.reads_resource::<Other>());
        assert!(!acc.reads_resource::<Gauge>());
        let mut o = or(flag, gauge_high);
        o.initialize(&mut w);
        assert!(o.access().combined_access().reads_resource::<Gauge>());
        assert!(c.name().contains(" && "), "{}", c.name());
        assert!(o.name().contains(" || "), "{}", o.name());
    }

    #[test]
    #[should_panic(expected = "may only read")]
    fn a_combined_condition_that_writes_is_refused_through_its_half() {
        let mut w = world();
        and(flag, writes_resource).initialize(&mut w);
    }

    // ── ticks ────────────────────────────────────────────────────────

    #[test]
    fn the_oldest_last_run_of_one_condition_is_its_last_run() {
        let mut w = world();
        let mut a = flag.into_condition();
        assert_eq!(a.oldest_last_run(w.change_tick()), Tick::ZERO, "never ran");
        let t0 = w.change_tick();
        a.run(&mut w);
        w.increment_change_tick();
        w.increment_change_tick();
        let mut b = gauge_high.into_condition();
        let t1 = w.change_tick();
        b.run(&mut w);
        let now = w.change_tick();
        assert_eq!(a.oldest_last_run(now), t0);
        assert_eq!(b.oldest_last_run(now), t1);
        assert_ne!(t0, t1);
    }

    #[test]
    fn the_oldest_last_run_of_a_pair_is_the_half_that_ran_longest_ago() {
        let mut w = world();
        w.get_resource_mut::<Gauge>().unwrap().0 = 50;
        let mut c = and(flag, gauge_high);
        let t0 = w.change_tick();
        assert!(c.run(&mut w), "both halves run at t0");
        w.get_resource_mut::<Flag>().unwrap().0 = false;
        assert!(!c.run(&mut w), "only the first half runs at t0 + 1");
        let now = w.change_tick();
        assert_eq!(c.oldest_last_run(now), t0);
        // flag is still false, so the second half of a fresh pair never runs
        let mut n = not(and(flag, gauge_high));
        n.run(&mut w);
        n.run(&mut w);
        assert_eq!(n.oldest_last_run(w.change_tick()), Tick::ZERO);
    }

    #[test]
    fn check_change_tick_clamps_each_half() {
        let mut w = world();
        let mut a = flag.into_condition();
        a.run(&mut w);
        let last = a.oldest_last_run(w.change_tick());
        let now = Tick::new(last.get().wrapping_add(MAX_CHANGE_AGE + 40));
        assert!(a.check_change_tick(now));
        assert_eq!(
            now.relative_to(a.oldest_last_run(now)).get(),
            MAX_CHANGE_AGE
        );
        assert!(!a.check_change_tick(now), "already within range");

        let mut c = and(flag, gauge_high);
        c.run(&mut w);
        let last = c.oldest_last_run(w.change_tick());
        let now = Tick::new(last.get().wrapping_add(MAX_CHANGE_AGE + 7));
        assert!(c.check_change_tick(now));
        assert!(!c.check_change_tick(now));
        let mut o = or(flag, gauge_high);
        o.run(&mut w);
        let last = o.oldest_last_run(w.change_tick());
        let now = Tick::new(last.get().wrapping_add(MAX_CHANGE_AGE + 7));
        assert!(o.check_change_tick(now));
        let mut n = not(flag);
        n.run(&mut w);
        let last = n.oldest_last_run(w.change_tick());
        let now = Tick::new(last.get().wrapping_add(MAX_CHANGE_AGE + 7));
        assert!(n.check_change_tick(now));
    }

    #[test]
    fn a_skipped_half_is_clamped_even_when_the_other_is_not() {
        // `and` skips its second half after a false first, `or` after a
        // true one. The half that never ran is the one out of range.
        let mut w = world();
        w.get_resource_mut::<Flag>().unwrap().0 = false;
        let mut c = and(flag, gauge_high);
        assert!(!c.run(&mut w));
        let now = Tick::new(MAX_CHANGE_AGE + 1);
        assert!(c.check_change_tick(now), "gauge_high never ran");
        assert!(!c.check_change_tick(now));

        let mut w = world();
        let mut o = or(flag, gauge_high);
        assert!(o.run(&mut w));
        assert!(o.check_change_tick(Tick::new(MAX_CHANGE_AGE + 1)));
    }

    #[test]
    fn both_halves_are_clamped_even_when_only_one_is_old() {
        let mut w = world();
        let mut c = and(flag, gauge_high);
        w.get_resource_mut::<Gauge>().unwrap().0 = 50;
        c.run(&mut w);
        let last = c.oldest_last_run(w.change_tick());
        let now = Tick::new(last.get().wrapping_add(MAX_CHANGE_AGE + 2));
        assert!(c.check_change_tick(now));
        // both were old; after clamping neither is still out of range
        assert!(!c.check_change_tick(now));
    }
}
