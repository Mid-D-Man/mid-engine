// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-ecs-schedule.md, section "tests.rs"
// ============================================================================
//! Schedules end to end: order, sets, conditions, ticks, deferred work,
//! errors, ambiguity, and running by label.

use std::sync::atomic::{AtomicU32, Ordering};

use super::common_conditions::{resource_changed, resource_exists};
use super::*;
use crate::commands::Commands;
use crate::system_param::{Local, Query, Res, ResMut};
use crate::tick::{Mut, Tick, MAX_CHANGE_AGE};
use crate::world::Entity;

// ── fixtures ─────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct Update;
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct Other;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum Phase {
    Input,
    Simulate,
    Render,
}
impl SystemSet for Phase {}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct Group(u32);
impl SystemSet for Group {}

#[derive(Default)]
struct Log(Vec<&'static str>);

struct Flag(bool);
struct Gauge(u32);
struct Counter(AtomicU32);
struct Counter2(AtomicU32);

macro_rules! loggers {
    ($($name:ident),*) => {
        $(fn $name(mut log: ResMut<Log>) { log.0.push(stringify!($name)); })*
    };
}
loggers!(a, b, c, d, e, f);

fn world() -> World {
    let mut w = World::new();
    w.insert_resource(Log::default());
    w
}

fn log(w: &World) -> Vec<&'static str> {
    w.get_resource::<Log>().unwrap().0.clone()
}

fn clear(w: &mut World) {
    w.get_resource_mut::<Log>().unwrap().0.clear();
}

/// Builds, runs once, returns what ran in what order.
fn order_of<M>(systems: impl IntoScheduleConfigs<ScheduleSystem, M>) -> Vec<&'static str> {
    let mut w = world();
    let mut s = Schedule::new(Update);
    s.add_systems(systems);
    s.run(&mut w);
    log(&w)
}

fn flag(f: Res<Flag>) -> bool {
    f.0
}

fn count_one(c: Res<Counter>) -> bool {
    c.0.fetch_add(1, Ordering::SeqCst);
    true
}

fn count_two(c: Res<Counter2>) -> bool {
    c.0.fetch_add(1, Ordering::SeqCst);
    true
}

fn count_false(c: Res<Counter>) -> bool {
    c.0.fetch_add(1, Ordering::SeqCst);
    false
}

fn counter(w: &World) -> u32 {
    w.get_resource::<Counter>()
        .unwrap()
        .0
        .load(Ordering::SeqCst)
}

fn counter2(w: &World) -> u32 {
    w.get_resource::<Counter2>()
        .unwrap()
        .0
        .load(Ordering::SeqCst)
}

fn with_counters() -> World {
    let mut w = world();
    w.insert_resource(Counter(AtomicU32::new(0)));
    w.insert_resource(Counter2(AtomicU32::new(0)));
    w.insert_resource(Flag(true));
    w
}

// ── order ────────────────────────────────────────────────────────────

#[test]
fn systems_run_in_the_order_they_were_added() {
    assert_eq!(order_of((a, b, c)), ["a", "b", "c"]);
    assert_eq!(order_of((c, a, b)), ["c", "a", "b"]);
}

#[test]
fn an_empty_schedule_runs() {
    let mut w = world();
    let mut s = Schedule::new(Update);
    s.run(&mut w);
    assert!(s.is_empty());
    assert_eq!(s.len(), 0);
    assert_eq!(s.run_order(), Some(vec![]));
}

#[test]
fn chain_orders_a_tuple() {
    assert_eq!(order_of((c, b, a).chain()), ["c", "b", "a"]);
}

#[test]
fn after_a_function_runs_behind_it_wherever_it_was_added() {
    assert_eq!(order_of((a.after(b), b)), ["b", "a"]);
    assert_eq!(order_of((a.after(c), b, c)), ["b", "c", "a"]);
}

#[test]
fn before_a_function_runs_ahead_of_it() {
    assert_eq!(order_of((a, b, c.before(a))), ["b", "c", "a"]);
}

#[test]
fn a_function_added_twice_is_ordered_as_both() {
    // `b` waits for every system made from `a`; of what is ready the
    // earliest added goes first, so the first `a`, then `c`, then the
    // second `a`
    assert_eq!(order_of((b.after(a), a, c, a)), ["a", "c", "a", "b"]);
}

#[test]
fn ordering_against_a_function_that_was_never_added_changes_nothing() {
    assert_eq!(order_of((a.after(f), b)), ["a", "b"]);
}

#[test]
fn nested_chains_order_every_part_of_a_group() {
    // (b, c) both run after a and before d, in either order
    assert_eq!(order_of((d, (c, b), a).chain()), ["d", "c", "b", "a"]);
    assert_eq!(
        order_of((a, (c, b).chain(), d).chain()),
        ["a", "c", "b", "d"]
    );
}

#[test]
fn the_order_is_the_same_on_every_run_and_every_build() {
    let make = || {
        let mut s = Schedule::new(Update);
        s.add_systems((e.after(a), d.before(b), a, b, c, f.after(c).before(e)));
        s
    };
    let mut w = world();
    let mut one = make();
    one.run(&mut w);
    let first = log(&w);
    for _ in 0..3 {
        clear(&mut w);
        one.run(&mut w);
        assert_eq!(log(&w), first);
    }
    let mut w2 = world();
    let mut two = make();
    two.run(&mut w2);
    assert_eq!(log(&w2), first);
    assert_eq!(one.run_order(), two.run_order());
}

#[test]
fn run_order_names_the_systems_in_order_and_needs_a_build() {
    let mut s = Schedule::new(Update);
    s.add_systems((b.after(a), a));
    assert_eq!(s.run_order(), None);
    s.build(&mut world()).unwrap();
    let names = s.run_order().unwrap();
    assert_eq!(names.len(), 2);
    assert!(
        names[0].ends_with("::a") && names[1].ends_with("::b"),
        "{names:?}"
    );
}

// ── sets ─────────────────────────────────────────────────────────────

#[test]
fn a_set_ordered_before_another_orders_all_its_members() {
    let mut w = world();
    let mut s = Schedule::new(Update);
    s.configure_sets(Phase::Input.before(Phase::Simulate))
        .add_systems((
            c.in_set(Phase::Simulate),
            b.in_set(Phase::Simulate),
            a.in_set(Phase::Input),
            d.in_set(Phase::Input),
        ));
    s.run(&mut w);
    let l = log(&w);
    let pos = |n| l.iter().position(|&x| x == n).unwrap();
    assert!(pos("a") < pos("c") && pos("a") < pos("b"));
    assert!(pos("d") < pos("c") && pos("d") < pos("b"));
    assert_eq!(l.len(), 4);
}

#[test]
fn a_chain_of_sets_is_a_chain_of_phases() {
    let mut w = world();
    let mut s = Schedule::new(Update);
    s.configure_sets((Phase::Input, Phase::Simulate, Phase::Render).chain())
        .add_systems((
            c.in_set(Phase::Render),
            a.in_set(Phase::Input),
            b.in_set(Phase::Simulate),
        ));
    s.run(&mut w);
    assert_eq!(log(&w), ["a", "b", "c"]);
}

#[test]
fn a_tuple_in_a_set_puts_each_member_there() {
    let mut w = world();
    let mut s = Schedule::new(Update);
    s.configure_sets(Phase::Input.before(Phase::Simulate))
        .add_systems((a.in_set(Phase::Simulate), (b, c).in_set(Phase::Input)));
    s.run(&mut w);
    assert_eq!(log(&w), ["b", "c", "a"]);
}

#[test]
fn a_system_can_be_in_several_sets() {
    let mut w = world();
    let mut s = Schedule::new(Update);
    s.configure_sets((Phase::Input, Phase::Simulate).chain())
        .configure_sets((Group(1), Phase::Render).chain())
        .add_systems((
            a.in_set(Phase::Simulate).in_set(Group(1)),
            b.in_set(Phase::Input),
            c.in_set(Phase::Render),
        ));
    s.run(&mut w);
    assert_eq!(log(&w), ["b", "a", "c"]);
}

#[test]
fn a_set_inside_a_set_orders_with_the_outer_one() {
    let mut w = world();
    let mut s = Schedule::new(Update);
    s.configure_sets(Group(1).in_set(Phase::Simulate))
        .configure_sets(Phase::Input.before(Phase::Simulate))
        .add_systems((a.in_set(Group(1)), b.in_set(Phase::Input)));
    s.run(&mut w);
    assert_eq!(log(&w), ["b", "a"]);
}

#[test]
fn a_system_can_be_ordered_against_a_set() {
    let mut w = world();
    let mut s = Schedule::new(Update);
    s.add_systems((
        a.after(Phase::Input),
        b.in_set(Phase::Input),
        c.in_set(Phase::Input),
        d.before(Phase::Input),
    ));
    s.run(&mut w);
    let l = log(&w);
    assert_eq!(l[0], "d");
    assert_eq!(l[3], "a");
}

#[test]
fn a_set_can_be_ordered_against_a_function() {
    let mut w = world();
    let mut s = Schedule::new(Update);
    s.configure_sets(Phase::Input.after(a))
        .add_systems((b.in_set(Phase::Input), a));
    s.run(&mut w);
    assert_eq!(log(&w), ["a", "b"]);
}

#[test]
fn settings_on_the_set_apply_to_systems_added_later() {
    let mut w = world();
    let mut s = Schedule::new(Update);
    s.configure_sets(Phase::Input.before(Phase::Simulate));
    s.add_systems(a.in_set(Phase::Simulate));
    s.add_systems(b.in_set(Phase::Input));
    s.run(&mut w);
    assert_eq!(log(&w), ["b", "a"]);
}

// ── errors ───────────────────────────────────────────────────────────

#[test]
fn a_cycle_is_an_error_naming_the_systems_and_the_orderings() {
    let mut w = world();
    let mut s = Schedule::new(Update);
    s.add_systems((a.after(b), b.after(a), c));
    let Err(ScheduleBuildError::DependencyCycle(cycles)) = s.try_run(&mut w) else {
        panic!("expected a cycle");
    };
    assert_eq!(cycles.len(), 1);
    assert_eq!(cycles[0].systems.len(), 2);
    assert!(cycles[0].systems[0].ends_with("::a") && cycles[0].systems[1].ends_with("::b"));
    assert_eq!(cycles[0].orderings.len(), 2);
    assert!(log(&w).is_empty(), "nothing ran");
}

#[test]
fn a_cycle_through_sets_names_the_sets() {
    let mut w = world();
    let mut s = Schedule::new(Update);
    s.configure_sets((
        Phase::Input.before(Phase::Simulate),
        Phase::Simulate.before(Phase::Input),
    ))
    .add_systems((a.in_set(Phase::Input), b.in_set(Phase::Simulate)));
    let err = s.try_run(&mut w).unwrap_err();
    let text = err.to_string();
    assert!(
        text.contains("Input") && text.contains("Simulate"),
        "{text}"
    );
}

#[test]
#[should_panic(expected = "could not be built")]
fn run_panics_on_a_cycle_and_says_why() {
    let mut s = Schedule::new(Update);
    s.add_systems((a.after(b), b.after(a)));
    s.run(&mut world());
}

#[test]
fn a_set_ordered_against_its_own_member_is_an_error() {
    let mut s = Schedule::new(Update);
    s.configure_sets(Phase::Input.before(a))
        .add_systems(a.in_set(Phase::Input));
    let Err(ScheduleBuildError::OverlappingOrder(list)) = s.try_run(&mut world()) else {
        panic!("expected an overlapping order");
    };
    assert_eq!(list.len(), 1);
    assert_eq!(list[0].before, "Input");
}

#[test]
fn a_set_that_contains_itself_is_an_error() {
    let mut s = Schedule::new(Update);
    s.configure_sets(Phase::Input.in_set(Phase::Input));
    assert!(matches!(
        s.try_run(&mut world()),
        Err(ScheduleBuildError::HierarchyCycle(_))
    ));
}

#[test]
fn a_failed_build_can_be_fixed_by_adding_nothing_it_just_fails_again() {
    let mut w = world();
    let mut s = Schedule::new(Update);
    s.add_systems((a.after(b), b.after(a)));
    assert!(s.try_run(&mut w).is_err());
    assert!(s.try_run(&mut w).is_err());
    assert!(s.run_order().is_none());
}

// ── conditions ───────────────────────────────────────────────────────

#[test]
fn a_false_condition_skips_the_system() {
    let mut w = with_counters();
    let mut s = Schedule::new(Update);
    s.add_systems((a.run_if(flag), b));
    s.run(&mut w);
    assert_eq!(log(&w), ["a", "b"]);
    w.get_resource_mut::<Flag>().unwrap().0 = false;
    clear(&mut w);
    s.run(&mut w);
    assert_eq!(log(&w), ["b"]);
    w.get_resource_mut::<Flag>().unwrap().0 = true;
    clear(&mut w);
    s.run(&mut w);
    assert_eq!(log(&w), ["a", "b"]);
}

struct Seen(Vec<bool>);

fn watch_gauge(g: Res<Gauge>, mut seen: ResMut<Seen>) {
    seen.0.push(g.is_changed());
}

#[test]
fn a_skipped_system_still_sees_the_changes_it_missed() {
    let mut w = with_counters();
    w.insert_resource(Gauge(0));
    w.insert_resource(Seen(Vec::new()));
    let mut s = Schedule::new(Update);
    s.add_systems(watch_gauge.run_if(flag));
    s.run(&mut w);
    s.run(&mut w);
    w.get_resource_mut::<Flag>().unwrap().0 = false;
    w.get_resource_mut::<Gauge>().unwrap().0 = 1;
    s.run(&mut w);
    s.run(&mut w);
    w.get_resource_mut::<Flag>().unwrap().0 = true;
    s.run(&mut w);
    s.run(&mut w);
    // new, nothing, (skipped, skipped), the change it missed, nothing
    assert_eq!(
        w.get_resource::<Seen>().unwrap().0,
        [true, false, true, false]
    );
}

#[test]
fn run_if_takes_a_common_condition_by_name() {
    let mut w = world();
    let mut s = Schedule::new(Update);
    s.add_systems((a.run_if(resource_exists::<Gauge>), b));
    s.run(&mut w);
    assert_eq!(log(&w), ["b"]);
    w.insert_resource(Gauge(1));
    clear(&mut w);
    s.run(&mut w);
    assert_eq!(log(&w), ["a", "b"]);
}

#[test]
fn several_conditions_on_a_system_must_all_hold_and_all_are_asked() {
    let mut w = with_counters();
    let mut s = Schedule::new(Update);
    s.add_systems(a.run_if(count_false).run_if(count_one).run_if(count_two));
    s.run(&mut w);
    assert!(log(&w).is_empty());
    assert_eq!(
        (counter(&w), counter2(&w)),
        (2, 1),
        "count_false, count_one, count_two: the first answered no and the rest were still asked"
    );
}

#[test]
fn a_condition_on_a_set_gates_every_member_and_is_asked_once() {
    let mut w = with_counters();
    let mut s = Schedule::new(Update);
    s.configure_sets(Phase::Input.run_if(count_one))
        .add_systems((
            a.in_set(Phase::Input),
            b.in_set(Phase::Input),
            c.in_set(Phase::Input),
            d,
        ));
    s.run(&mut w);
    assert_eq!(log(&w), ["a", "b", "c", "d"]);
    assert_eq!(counter(&w), 1, "one answer for three members");
    s.run(&mut w);
    assert_eq!(counter(&w), 2, "and a fresh one per run");
}

#[test]
fn a_false_set_condition_skips_all_members_and_not_the_others() {
    let mut w = with_counters();
    let mut s = Schedule::new(Update);
    s.configure_sets(Phase::Input.run_if(flag)).add_systems((
        a.in_set(Phase::Input),
        b,
        c.in_set(Phase::Input),
    ));
    w.get_resource_mut::<Flag>().unwrap().0 = false;
    s.run(&mut w);
    assert_eq!(log(&w), ["b"]);
}

#[test]
fn a_system_turned_away_by_its_set_does_not_ask_its_own_conditions() {
    let mut w = with_counters();
    let mut s = Schedule::new(Update);
    s.configure_sets(Phase::Input.run_if(flag))
        .add_systems(a.in_set(Phase::Input).run_if(count_one));
    w.get_resource_mut::<Flag>().unwrap().0 = false;
    s.run(&mut w);
    assert_eq!(counter(&w), 0);
    w.get_resource_mut::<Flag>().unwrap().0 = true;
    s.run(&mut w);
    assert_eq!(counter(&w), 1);
    assert_eq!(log(&w), ["a"]);
}

#[test]
fn every_set_above_a_system_answers_even_when_one_says_no() {
    let mut w = with_counters();
    let mut s = Schedule::new(Update);
    s.configure_sets(Phase::Input.run_if(count_false))
        .configure_sets(Group(1).run_if(count_two))
        .add_systems(a.in_set(Phase::Input).in_set(Group(1)));
    s.run(&mut w);
    assert!(log(&w).is_empty());
    assert_eq!(counter(&w), 1);
    assert_eq!(
        counter2(&w),
        1,
        "asked although the other set had already said no"
    );
}

#[test]
fn a_condition_on_the_outer_set_gates_the_inner_sets_members() {
    let mut w = with_counters();
    let mut s = Schedule::new(Update);
    s.configure_sets(Group(1).in_set(Phase::Input))
        .configure_sets(Phase::Input.run_if(flag))
        .add_systems((a.in_set(Group(1)), b));
    w.get_resource_mut::<Flag>().unwrap().0 = false;
    s.run(&mut w);
    assert_eq!(log(&w), ["b"]);
}

#[test]
fn a_condition_on_a_group_is_asked_once_for_the_whole_group() {
    let mut w = with_counters();
    let mut s = Schedule::new(Update);
    s.add_systems(((a, b, c).run_if(count_one), d));
    s.run(&mut w);
    assert_eq!(log(&w), ["a", "b", "c", "d"]);
    assert_eq!(counter(&w), 1);

    let mut w = with_counters();
    let mut s = Schedule::new(Update);
    s.add_systems(((a, b, c).run_if(count_false), d));
    s.run(&mut w);
    assert_eq!(log(&w), ["d"]);
    assert_eq!(counter(&w), 1);
}

#[test]
fn a_chained_group_with_a_condition_is_still_chained() {
    let mut w = with_counters();
    let mut s = Schedule::new(Update);
    s.add_systems((c, b, a).chain().run_if(flag));
    s.run(&mut w);
    assert_eq!(log(&w), ["c", "b", "a"]);
}

#[test]
fn conditions_can_be_combined_and_boxed() {
    let mut w = with_counters();
    w.insert_resource(Gauge(0));
    let mut s = Schedule::new(Update);
    let boxed: BoxedCondition = Box::new(IntoCondition::into_condition(flag));
    s.add_systems((
        a.run_if(and(flag, resource_exists::<Gauge>)),
        b.run_if(not(flag)),
        c.run_if(or(not(flag), resource_changed::<Gauge>)),
        d.run_if(boxed),
    ));
    s.run(&mut w);
    assert_eq!(log(&w), ["a", "c", "d"]);
}

#[test]
fn a_set_condition_added_by_configure_sets_after_the_systems_still_applies() {
    let mut w = with_counters();
    let mut s = Schedule::new(Update);
    s.add_systems(a.in_set(Phase::Input));
    s.run(&mut w);
    assert_eq!(log(&w), ["a"]);
    s.configure_sets(Phase::Input.run_if(count_false));
    clear(&mut w);
    s.run(&mut w);
    assert!(log(&w).is_empty(), "the rebuild picked the condition up");
}

#[test]
#[should_panic(expected = "may only read")]
fn a_condition_that_writes_is_refused_when_the_schedule_is_built() {
    fn writer(mut g: ResMut<Gauge>) -> bool {
        g.0 += 1;
        true
    }
    let mut w = world();
    w.insert_resource(Gauge(0));
    let mut s = Schedule::new(Update);
    s.add_systems(a.run_if(writer));
    s.run(&mut w);
}

// ── ticks ────────────────────────────────────────────────────────────

#[test]
fn each_system_that_runs_advances_the_world_tick_by_one() {
    let mut w = world();
    let mut s = Schedule::new(Update);
    s.add_systems((a, b, c));
    let before = w.change_tick().get();
    s.run(&mut w);
    assert_eq!(w.change_tick().get(), before + 3);
}

#[test]
fn a_skipped_system_does_not_advance_it_and_a_condition_does() {
    let mut w = with_counters();
    let mut s = Schedule::new(Update);
    s.add_systems((a.run_if(flag), b));
    let t0 = w.change_tick().get();
    s.run(&mut w);
    assert_eq!(w.change_tick().get(), t0 + 3, "the condition, a, b");
    w.get_resource_mut::<Flag>().unwrap().0 = false;
    let t1 = w.change_tick().get();
    s.run(&mut w);
    assert_eq!(w.change_tick().get(), t1 + 2, "the condition, b");
}

#[test]
fn a_set_condition_costs_one_tick_per_run_not_one_per_member() {
    let mut w = with_counters();
    let mut s = Schedule::new(Update);
    s.configure_sets(Phase::Input.run_if(flag)).add_systems((
        a.in_set(Phase::Input),
        b.in_set(Phase::Input),
        c.in_set(Phase::Input),
    ));
    let t0 = w.change_tick().get();
    s.run(&mut w);
    assert_eq!(
        w.change_tick().get(),
        t0 + 4,
        "the set's condition once, then three systems"
    );
}

fn bump(mut g: ResMut<Gauge>) {
    g.0 += 1;
}

fn see_change(g: Res<Gauge>, mut seen: ResMut<Seen>) {
    seen.0.push(g.is_changed());
}

#[test]
fn a_system_sees_what_an_earlier_system_in_the_same_run_wrote() {
    let mut w = world();
    w.insert_resource(Gauge(0));
    w.insert_resource(Seen(Vec::new()));
    let mut s = Schedule::new(Update);
    s.add_systems((see_change.after(bump), bump));
    s.run(&mut w);
    s.run(&mut w);
    assert_eq!(w.get_resource::<Seen>().unwrap().0, [true, true]);
}

#[test]
fn a_system_that_reads_before_the_writer_sees_the_write_next_run() {
    let mut w = world();
    w.insert_resource(Gauge(0));
    w.insert_resource(Seen(Vec::new()));
    let mut s = Schedule::new(Update);
    s.add_systems((see_change.before(bump), bump));
    s.run(&mut w);
    s.run(&mut w);
    s.run(&mut w);
    // first: new to it; then the previous run's write each time
    assert_eq!(w.get_resource::<Seen>().unwrap().0, [true, true, true]);
}

#[test]
fn stored_ticks_of_systems_that_did_not_run_are_kept_within_range() {
    let mut w = with_counters();
    let mut s = Schedule::new(Update);
    s.add_systems(a.run_if(flag));
    s.run(&mut w); // a ran at tick 1
    w.get_resource_mut::<Flag>().unwrap().0 = false;
    let last = s.oldest_last_run(w.change_tick()).unwrap();
    // the condition's and the system's are both near zero now
    w.set_change_tick(Tick::new(last.get().wrapping_add(MAX_CHANGE_AGE + 500)));
    s.run(&mut w); // skipped; the sweep afterwards clamps
    let now = w.change_tick();
    let oldest = s.oldest_last_run(now).unwrap();
    assert!(
        now.relative_to(oldest).get() <= MAX_CHANGE_AGE,
        "{} > {MAX_CHANGE_AGE}",
        now.relative_to(oldest).get()
    );
}

#[test]
fn the_sweep_reaches_conditions_that_have_not_been_asked() {
    // A system turned away by its set never consults its own conditions, so
    // they sit unevaluated while the world's tick runs on.
    let mut w = with_counters();
    let mut s = Schedule::new(Update);
    s.configure_sets(Phase::Input.run_if(flag))
        .add_systems(a.in_set(Phase::Input).run_if(count_one));
    w.get_resource_mut::<Flag>().unwrap().0 = false;
    s.run(&mut w);
    w.set_change_tick(Tick::new(MAX_CHANGE_AGE + 900));
    s.run(&mut w);
    assert_eq!(counter(&w), 0, "it was never asked");
    let now = w.change_tick();
    let age = now.relative_to(s.oldest_last_run(now).unwrap()).get();
    assert!(age <= MAX_CHANGE_AGE, "{age} > {MAX_CHANGE_AGE}");
}

#[test]
fn the_sweep_reaches_a_set_condition_nobody_reached() {
    // A set without members is never evaluated.
    let mut w = with_counters();
    let mut s = Schedule::new(Update);
    s.configure_sets(Phase::Input.run_if(count_false))
        .add_systems(b);
    s.run(&mut w);
    w.set_change_tick(Tick::new(MAX_CHANGE_AGE + 900));
    s.run(&mut w);
    assert_eq!(counter(&w), 0, "it was never asked");
    let now = w.change_tick();
    let age = now.relative_to(s.oldest_last_run(now).unwrap()).get();
    assert!(age <= MAX_CHANGE_AGE, "{age} > {MAX_CHANGE_AGE}");
}

#[test]
fn the_sweep_reaches_a_system_that_has_not_run() {
    let mut w = with_counters();
    let mut s = Schedule::new(Update);
    s.add_systems(a.run_if(flag));
    s.run(&mut w);
    w.get_resource_mut::<Flag>().unwrap().0 = false;
    w.set_change_tick(Tick::new(MAX_CHANGE_AGE + 900));
    s.run(&mut w);
    let now = w.change_tick();
    let age = now.relative_to(s.oldest_last_run(now).unwrap()).get();
    assert!(age <= MAX_CHANGE_AGE, "{age} > {MAX_CHANGE_AGE}");
}

#[test]
fn oldest_last_run_is_none_without_systems_and_the_furthest_back_otherwise() {
    let mut w = with_counters();
    let s = Schedule::new(Update);
    assert_eq!(s.oldest_last_run(w.change_tick()), None);

    let mut s = Schedule::new(Update);
    s.add_systems((a, b.run_if(flag)));
    assert_eq!(s.oldest_last_run(w.change_tick()), Some(Tick::ZERO));
    let first = w.change_tick();
    s.run(&mut w);
    // `a` ran first, at the tick the world was at; its condition and `b`
    // came after
    let oldest = s.oldest_last_run(w.change_tick()).unwrap();
    assert_eq!(oldest, first, "a ran first");
    w.get_resource_mut::<Flag>().unwrap().0 = false;
    s.run(&mut w);
    // `a` and `b`'s condition ran again; `b` did not: it last ran at the
    // third tick of the first run
    let oldest = s.oldest_last_run(w.change_tick()).unwrap();
    assert_eq!(oldest.get(), first.get() + 2, "b's last run did not move");
}

// ── deferred work ────────────────────────────────────────────────────

fn spawn_one(mut commands: Commands) {
    commands.spawn();
}

struct Counts(Vec<usize>);

fn count_entities(q: Query<Entity>, mut counts: ResMut<Counts>) {
    counts.0.push(q.iter().count());
}

#[test]
fn deferred_work_is_applied_before_the_next_system_even_when_unordered() {
    let mut w = world();
    w.insert_resource(Counts(Vec::new()));
    let mut s = Schedule::new(Update);
    s.add_systems((spawn_one, count_entities, spawn_one, count_entities));
    s.run(&mut w);
    assert_eq!(w.get_resource::<Counts>().unwrap().0, [1, 2]);
    assert_eq!(w.entity_count(), 2);
}

#[test]
fn deferred_work_of_the_last_system_is_applied_too() {
    let mut w = world();
    let mut s = Schedule::new(Update);
    s.add_systems(spawn_one);
    s.run(&mut w);
    assert_eq!(w.entity_count(), 1);
}

#[test]
fn a_skipped_system_defers_nothing() {
    let mut w = with_counters();
    w.get_resource_mut::<Flag>().unwrap().0 = false;
    let mut s = Schedule::new(Update);
    s.add_systems(spawn_one.run_if(flag));
    s.run(&mut w);
    assert_eq!(w.entity_count(), 0);
}

// ── exclusive systems ────────────────────────────────────────────────

fn exclusive_push(world: &mut World) {
    world.get_resource_mut::<Log>().unwrap().0.push("exclusive");
}

#[test]
fn an_exclusive_system_runs_in_its_place() {
    assert_eq!(order_of((a, exclusive_push, b)), ["a", "exclusive", "b"]);
    assert_eq!(
        order_of((exclusive_push.after(b), a, b)),
        ["a", "b", "exclusive"]
    );
}

#[test]
fn an_exclusive_system_can_run_a_schedule_by_label() {
    fn run_other(world: &mut World) {
        world.run_schedule(Other);
    }
    let mut w = world();
    let mut inner = Schedule::new(Other);
    inner.add_systems((b, c));
    w.add_schedule(inner);
    let mut outer = Schedule::new(Update);
    outer.add_systems((a, run_other, d).chain());
    w.add_schedule(outer);
    w.run_schedule(Update);
    assert_eq!(log(&w), ["a", "b", "c", "d"]);
    assert!(w.contains_schedule(Update) && w.contains_schedule(Other));
}

// ── state kept across builds ─────────────────────────────────────────

fn local_counter(mut n: Local<u32>, mut log: ResMut<Log>) {
    *n += 1;
    log.0.push(if *n == 1 { "first" } else { "again" });
}

#[test]
fn adding_a_system_after_a_run_rebuilds_without_losing_state() {
    let mut w = world();
    let mut s = Schedule::new(Update);
    s.add_systems(local_counter);
    s.run(&mut w);
    s.add_systems(a);
    clear(&mut w);
    s.run(&mut w);
    assert_eq!(log(&w), ["again", "a"], "the local survived the rebuild");
    assert_eq!(s.len(), 2);
}

#[test]
fn adding_a_system_that_must_run_first_puts_it_first_on_the_next_run() {
    let mut w = world();
    let mut s = Schedule::new(Update);
    s.add_systems(a);
    s.run(&mut w);
    s.add_systems(b.before(a));
    clear(&mut w);
    s.run(&mut w);
    assert_eq!(log(&w), ["b", "a"]);
}

// ── labels and the world ─────────────────────────────────────────────

#[test]
fn schedules_are_run_by_label_and_are_independent() {
    let mut w = world();
    w.init_schedule(Update).add_systems(a);
    w.init_schedule(Other).add_systems(b);
    w.run_schedule(Other);
    w.run_schedule(Update);
    w.run_schedule(Other);
    assert_eq!(log(&w), ["b", "a", "b"]);
}

#[test]
fn init_schedule_makes_one_and_then_finds_it() {
    let mut w = world();
    assert!(!w.contains_schedule(Update));
    w.init_schedule(Update).add_systems(a);
    w.init_schedule(Update).add_systems(b);
    assert_eq!(w.init_schedule(Update).len(), 2);
    assert!(w.contains_schedule(Update));
    assert!(!w.contains_schedule(Other));
    w.run_schedule(Update);
    assert_eq!(log(&w), ["a", "b"]);
}

#[test]
fn labels_of_different_types_with_equal_text_are_different() {
    #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    struct Update2;
    let mut w = world();
    w.init_schedule(Update).add_systems(a);
    w.init_schedule(Update2).add_systems(b);
    w.init_schedule("update").add_systems(c);
    w.run_schedule(Update2);
    w.run_schedule("update");
    assert_eq!(log(&w), ["b", "c"]);
}

#[test]
fn add_schedule_returns_the_one_it_replaces() {
    let mut w = world();
    let mut first = Schedule::new(Update);
    first.add_systems(a);
    assert!(w.add_schedule(first).is_none());
    let mut second = Schedule::new(Update);
    second.add_systems((b, c));
    let old = w.add_schedule(second).unwrap();
    assert_eq!(old.len(), 1);
    w.run_schedule(Update);
    assert_eq!(log(&w), ["b", "c"]);
}

#[test]
fn running_a_missing_schedule_is_an_error_or_a_panic() {
    let mut w = world();
    assert_eq!(
        w.try_run_schedule(Update),
        Err(ScheduleRunError::NotFound("Update".to_string()))
    );
    let err = w.try_run_schedule(Update).unwrap_err();
    assert!(err.to_string().contains("no schedule"), "{err}");
}

#[test]
#[should_panic(expected = "no schedule `Update`")]
fn run_schedule_panics_when_there_is_none() {
    world().run_schedule(Update);
}

#[test]
fn a_schedule_that_cannot_be_built_reports_it_through_the_world() {
    let mut w = world();
    w.init_schedule(Update)
        .add_systems((a.after(b), b.after(a)));
    let err = w.try_run_schedule(Update).unwrap_err();
    assert!(matches!(
        err,
        ScheduleRunError::Build(ScheduleBuildError::DependencyCycle(_))
    ));
    assert!(w.contains_schedule(Update), "and it is still there");
}

#[test]
fn a_schedule_cannot_be_run_or_changed_from_inside_itself() {
    fn reenter(world: &mut World) {
        let r = world.try_run_schedule(Update);
        world.get_resource_mut::<Log>().unwrap().0.push(match r {
            Err(ScheduleRunError::Running(_)) => "refused",
            _ => "allowed",
        });
    }
    let mut w = world();
    w.init_schedule(Update).add_systems(reenter);
    w.run_schedule(Update);
    assert_eq!(log(&w), ["refused"]);
    assert!(w.contains_schedule(Update), "put back afterwards");
}

#[test]
fn a_running_schedule_counts_as_existing() {
    fn probe(world: &mut World) {
        let s = world.get_resource::<Schedules>().unwrap();
        let text = format!(
            "{}{}{}{}",
            s.contains(Update),
            s.is_running(Update),
            s.get(Update).is_some(),
            s.len()
        );
        world
            .get_resource_mut::<Log>()
            .unwrap()
            .0
            .push(match text.as_str() {
                "truetruefalse1" => "as expected",
                _ => "wrong",
            });
    }
    let mut w = world();
    w.init_schedule(Update).add_systems(probe);
    w.run_schedule(Update);
    assert_eq!(log(&w), ["as expected"]);
    let s = w.get_resource::<Schedules>().unwrap();
    assert!(!s.is_running(Update) && s.get(Update).is_some());
    assert_eq!(s.len(), 1);
    assert!(!s.is_empty());
}

#[test]
#[should_panic(expected = "while it is being run")]
fn replacing_a_running_schedule_panics() {
    fn replace(world: &mut World) {
        world.add_schedule(Schedule::new(Update));
    }
    let mut w = world();
    w.init_schedule(Update).add_systems(replace);
    w.run_schedule(Update);
}

#[test]
#[should_panic(expected = "while it is being run")]
fn changing_a_running_schedule_panics() {
    fn change(world: &mut World) {
        world.init_schedule(Update);
    }
    let mut w = world();
    w.init_schedule(Update).add_systems(change);
    w.run_schedule(Update);
}

#[test]
fn a_schedule_goes_back_when_the_scope_panics() {
    let mut w = world();
    w.init_schedule(Update).add_systems(a);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        w.schedule_scope(Update, |_, _| panic!("boom"));
    }));
    assert!(result.is_err());
    assert!(w.contains_schedule(Update));
    assert!(!w.get_resource::<Schedules>().unwrap().is_running(Update));
    w.run_schedule(Update);
    assert_eq!(log(&w), ["a"]);
}

#[test]
fn a_schedule_goes_back_when_a_system_in_it_panics() {
    fn boom() {
        panic!("a system failed");
    }
    let mut w = world();
    w.init_schedule(Update).add_systems((a, boom));
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        w.run_schedule(Update);
    }));
    assert!(result.is_err());
    assert!(w.contains_schedule(Update));
    assert!(!w.get_resource::<Schedules>().unwrap().is_running(Update));
}

#[test]
fn schedule_scope_lends_the_world_and_the_schedule() {
    let mut w = world();
    w.init_schedule(Update).add_systems(a);
    let n = w.schedule_scope(Update, |world, schedule| {
        schedule.run(world);
        schedule.run(world);
        schedule.len()
    });
    assert_eq!(n, 1);
    assert_eq!(log(&w), ["a", "a"]);
}

#[test]
fn schedules_can_be_looked_up_and_changed_through_the_resource() {
    let mut w = world();
    w.init_schedule(Update).add_systems(a);
    let schedules = w.get_resource_mut::<Schedules>().unwrap();
    assert!(schedules.get(Update).is_some());
    assert!(schedules.get(Other).is_none());
    assert!(schedules.get_mut(Update).is_some());
    assert!(schedules.get_mut(Other).is_none());
    schedules.get_mut(Update).unwrap().add_systems(b);
    assert_eq!(schedules.get(Update).unwrap().len(), 2);
    assert!(schedules.contains(Update) && !schedules.contains(Other));
    assert!(!schedules.is_running(Update));
}

#[test]
fn run_errors_say_what_happened() {
    let running = ScheduleRunError::Running("Update".to_string()).to_string();
    assert!(
        running.contains("Update") && running.contains("being run"),
        "{running}"
    );
    let missing = ScheduleRunError::NotFound("Update".to_string()).to_string();
    assert!(
        missing.contains("Update") && missing.contains("no schedule"),
        "{missing}"
    );
    let build: ScheduleRunError =
        ScheduleBuildError::HierarchyCycle(vec![vec!["S".to_string()]]).into();
    assert!(build.to_string().contains("contain themselves"), "{build}");
}

#[test]
fn the_schedule_label_is_kept() {
    let s = Schedule::new(Update);
    assert_eq!(s.label(), &Update.label_key());
    assert!(format!("{s:?}").contains("Update"));
}

// ── ambiguity ────────────────────────────────────────────────────────

struct Pos(u32);

fn write_gauge_1(mut g: ResMut<Gauge>) {
    g.0 += 1;
}

fn write_gauge_2(mut g: ResMut<Gauge>) {
    g.0 += 2;
}

fn read_gauge(_g: Res<Gauge>) {}

fn read_gauge_2(_g: Res<Gauge>) {}

fn write_pos(mut q: Query<Mut<Pos>>) {
    for mut p in q.iter_mut() {
        p.0 += 1;
    }
}

fn gauge_world() -> World {
    let mut w = world();
    w.insert_resource(Gauge(0));
    w
}

#[test]
fn two_writers_with_nothing_between_them_are_ambiguous() {
    let mut w = gauge_world();
    let mut s = Schedule::new(Update);
    s.add_systems((write_gauge_1, write_gauge_2));
    let found = s.ambiguities(&mut w).unwrap();
    assert_eq!(found.len(), 1);
    assert!(found[0].first.ends_with("write_gauge_1"));
    assert!(found[0].second.ends_with("write_gauge_2"));
    assert!(
        found[0].conflicts.iter().any(|n| n.contains("Gauge")),
        "{:?}",
        found[0].conflicts
    );
    assert!(!found[0].whole_world);
}

#[test]
fn ordering_them_ends_the_ambiguity_directly_or_through_others() {
    let mut w = gauge_world();
    let mut s = Schedule::new(Update);
    s.add_systems((write_gauge_1, write_gauge_2).chain());
    assert!(s.ambiguities(&mut w).unwrap().is_empty());

    let mut s = Schedule::new(Update);
    s.add_systems((write_gauge_1, a, write_gauge_2).chain());
    assert!(
        s.ambiguities(&mut w).unwrap().is_empty(),
        "ordered through the system between them"
    );
}

#[test]
fn it_does_not_matter_which_was_added_first() {
    let mut w = gauge_world();
    let mut s = Schedule::new(Update);
    s.add_systems((write_gauge_1.after(write_gauge_2), write_gauge_2));
    assert!(s.ambiguities(&mut w).unwrap().is_empty());
    let mut s = Schedule::new(Update);
    s.add_systems((write_gauge_1.before(write_gauge_2), write_gauge_2));
    assert!(s.ambiguities(&mut w).unwrap().is_empty());
}

#[test]
fn ordering_through_sets_ends_it_too() {
    let mut w = gauge_world();
    let mut s = Schedule::new(Update);
    s.configure_sets(Phase::Input.before(Phase::Simulate))
        .add_systems((
            write_gauge_1.in_set(Phase::Input),
            write_gauge_2.in_set(Phase::Simulate),
        ));
    assert!(s.ambiguities(&mut w).unwrap().is_empty());
}

#[test]
fn reads_do_not_conflict_with_reads_but_do_with_writes() {
    let mut w = gauge_world();
    let mut s = Schedule::new(Update);
    s.add_systems((read_gauge, read_gauge_2));
    assert!(s.ambiguities(&mut w).unwrap().is_empty());
    let mut s = Schedule::new(Update);
    s.add_systems((read_gauge, write_gauge_1));
    assert_eq!(s.ambiguities(&mut w).unwrap().len(), 1);
}

#[test]
fn unrelated_data_does_not_conflict() {
    let mut w = gauge_world();
    let mut s = Schedule::new(Update);
    s.add_systems((write_gauge_1, write_pos));
    assert!(s.ambiguities(&mut w).unwrap().is_empty());
}

#[test]
fn ambiguous_with_says_it_is_intended() {
    let mut w = gauge_world();
    let mut s = Schedule::new(Update);
    s.add_systems((write_gauge_1.ambiguous_with(write_gauge_2), write_gauge_2));
    assert!(s.ambiguities(&mut w).unwrap().is_empty());

    let mut s = Schedule::new(Update);
    s.add_systems((
        write_gauge_1.in_set(Group(1)),
        write_gauge_2.ambiguous_with(Group(1)),
    ));
    assert!(
        s.ambiguities(&mut w).unwrap().is_empty(),
        "through a set, from either side"
    );

    let mut s = Schedule::new(Update);
    s.add_systems((write_gauge_1.ambiguous_with(read_gauge), write_gauge_2));
    assert_eq!(
        s.ambiguities(&mut w).unwrap().len(),
        1,
        "silencing a different system does not help"
    );
}

#[test]
fn an_exclusive_system_conflicts_with_anything_that_touches_something() {
    let mut w = gauge_world();
    let mut s = Schedule::new(Update);
    s.add_systems((exclusive_push, read_gauge));
    let found = s.ambiguities(&mut w).unwrap();
    assert_eq!(found.len(), 1);
    assert!(found[0].whole_world);
    let mut s = Schedule::new(Update);
    s.add_systems((exclusive_push, a.after(exclusive_push)));
    assert!(s.ambiguities(&mut w).unwrap().is_empty());
}

#[test]
fn ignore_builds_without_looking_and_error_refuses() {
    let mut w = gauge_world();
    let mut s = Schedule::new(Update);
    s.add_systems((write_gauge_1, write_gauge_2));
    assert_eq!(s.ambiguity_detection(), AmbiguityDetection::Ignore);
    s.build(&mut w).unwrap();
    assert!(s.run_order().is_some());

    s.set_ambiguity_detection(AmbiguityDetection::Error);
    let Err(ScheduleBuildError::Ambiguities(list)) = s.try_run(&mut w) else {
        panic!("expected ambiguities");
    };
    assert_eq!(list.len(), 1);
    assert!(s.run_order().is_none(), "it did not build");
    assert_eq!(w.get_resource::<Gauge>().unwrap().0, 0, "and did not run");

    s.set_ambiguity_detection(AmbiguityDetection::Ignore);
    s.run(&mut w);
    assert_eq!(w.get_resource::<Gauge>().unwrap().0, 3);
}

#[test]
fn error_level_accepts_a_schedule_with_none() {
    let mut w = gauge_world();
    let mut s = Schedule::new(Update);
    s.set_ambiguity_detection(AmbiguityDetection::Error);
    s.add_systems((write_gauge_1, write_gauge_2).chain());
    s.run(&mut w);
    assert_eq!(w.get_resource::<Gauge>().unwrap().0, 3);
}

#[test]
fn warn_level_builds_and_runs() {
    let mut w = gauge_world();
    let mut s = Schedule::new(Update);
    s.set_ambiguity_detection(AmbiguityDetection::Warn);
    s.add_systems((write_gauge_1, write_gauge_2));
    s.run(&mut w);
    assert_eq!(w.get_resource::<Gauge>().unwrap().0, 3);
}

#[test]
fn changing_the_level_forces_a_rebuild() {
    let mut w = gauge_world();
    let mut s = Schedule::new(Update);
    s.add_systems((write_gauge_1, write_gauge_2));
    s.build(&mut w).unwrap();
    s.set_ambiguity_detection(AmbiguityDetection::Error);
    assert!(s.build(&mut w).is_err());
    s.set_ambiguity_detection(AmbiguityDetection::Error);
    assert!(s.build(&mut w).is_err(), "still refused");
}

#[test]
fn ambiguities_are_reported_for_every_pair() {
    let mut w = gauge_world();
    let mut s = Schedule::new(Update);
    s.add_systems((write_gauge_1, write_gauge_2, read_gauge));
    assert_eq!(s.ambiguities(&mut w).unwrap().len(), 3);
}

// ── scale ────────────────────────────────────────────────────────────

fn noop() {}

#[test]
fn a_few_hundred_systems_in_phases_build_and_run() {
    let mut w = world();
    let mut s = Schedule::new(Update);
    s.configure_sets((Phase::Input, Phase::Simulate, Phase::Render).chain());
    for i in 0..300 {
        match i % 3 {
            0 => s.add_systems(noop.in_set(Phase::Render)),
            1 => s.add_systems(noop.in_set(Phase::Input)),
            _ => s.add_systems(noop.in_set(Phase::Simulate)),
        };
    }
    s.run(&mut w);
    assert_eq!(s.len(), 300);
    let before = w.change_tick().get();
    s.run(&mut w);
    assert_eq!(w.change_tick().get(), before + 300);
}
