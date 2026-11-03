// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-ecs-schedule.md, section "tests_removals.rs"
// ============================================================================
//! Which systems and conditions read the removal log, what a schedule says
//! about the oldest of them, and trimming the log by it.

use super::*;
use crate::system::IntoSystem;
use crate::system_param::{RemovedComponents, ResMut};
use crate::world::Entity;

// ── fixtures ─────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct Update;
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct Other;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct Group;
impl SystemSet for Group {}

struct Hp;
struct Flag(bool);
#[derive(Default)]
struct SeenA(Vec<Entity>);
#[derive(Default)]
struct SeenB(Vec<Entity>);
struct Trimmed(bool);

fn reader_a(removed: RemovedComponents<Hp>, mut seen: ResMut<SeenA>) {
    seen.0.extend(removed.iter());
}

fn reader_b(removed: RemovedComponents<Hp>, mut seen: ResMut<SeenB>) {
    seen.0.extend(removed.iter());
}

fn plain(_flag: crate::system_param::Res<Flag>) {}

fn plain_too() {}

fn sees_removals(removed: RemovedComponents<Hp>) -> bool {
    removed.iter().next().is_some()
}

fn always(_flag: crate::system_param::Res<Flag>) -> bool {
    true
}

fn never(flag: crate::system_param::Res<Flag>) -> bool {
    flag.0
}

fn try_trim(world: &mut World) {
    let trimmed = world.trim_removals_for_schedules();
    world.insert_resource(Trimmed(trimmed));
}

fn world() -> World {
    let mut w = World::new();
    w.insert_resource(Flag(false));
    w.insert_resource(SeenA::default());
    w.insert_resource(SeenB::default());
    w
}

fn spawn_with_hp(w: &mut World, n: usize) -> Vec<Entity> {
    (0..n)
        .map(|_| {
            let e = w.spawn();
            w.insert_static(e, Hp);
            e
        })
        .collect()
}

fn seen_a(w: &World) -> Vec<Entity> {
    w.get_resource::<SeenA>().unwrap().0.clone()
}

fn seen_b(w: &World) -> Vec<Entity> {
    w.get_resource::<SeenB>().unwrap().0.clone()
}

// ── the flag ─────────────────────────────────────────────────────────

#[test]
fn a_system_reads_removals_once_it_is_set_up_with_the_parameter() {
    let mut w = world();
    let mut reader = IntoSystem::into_system(reader_a);
    assert!(!reader.is_initialized());
    reader.initialize(&mut w);
    assert!(reader.reads_removals());

    let mut other = IntoSystem::into_system(plain);
    other.initialize(&mut w);
    assert!(!other.reads_removals());
}

#[test]
fn an_exclusive_system_does_not_read_removals() {
    let mut w = world();
    let mut ex = IntoSystem::into_system(try_trim);
    ex.initialize(&mut w);
    assert!(!ex.reads_removals());
}

#[test]
fn conditions_report_the_parameter_through_every_combinator() {
    let mut w = world();
    let mut reads = sees_removals.into_condition();
    let mut not_reads = never.into_condition();
    reads.initialize(&mut w);
    not_reads.initialize(&mut w);
    assert!(reads.reads_removals());
    assert!(!not_reads.reads_removals());

    let mut n = not(sees_removals);
    n.initialize(&mut w);
    assert!(n.reads_removals(), "not forwards to the inner one");
    let mut n2 = not(never);
    n2.initialize(&mut w);
    assert!(!n2.reads_removals());

    // either half is enough, in either position
    let mut a1 = and(never, sees_removals);
    let mut a2 = and(sees_removals, never);
    let mut a3 = and(never, always);
    let mut o1 = or(never, sees_removals);
    let mut o2 = or(sees_removals, never);
    let mut o3 = or(never, always);
    for c in [
        &mut a1 as &mut dyn Condition,
        &mut a2,
        &mut a3,
        &mut o1,
        &mut o2,
        &mut o3,
    ] {
        c.initialize(&mut w);
    }
    assert!(a1.reads_removals() && a2.reads_removals());
    assert!(!a3.reads_removals());
    assert!(o1.reads_removals() && o2.reads_removals());
    assert!(!o3.reads_removals());

    let boxed: BoxedCondition = Box::new(sees_removals.into_condition());
    let mut boxed = boxed;
    boxed.initialize(&mut w);
    assert!(boxed.reads_removals(), "a boxed one forwards too");
    let plain_boxed: BoxedCondition = Box::new(never.into_condition());
    let mut plain_boxed = plain_boxed;
    plain_boxed.initialize(&mut w);
    assert!(!plain_boxed.reads_removals());
}

/// A system that says nothing about removals: it takes the trait's answer.
struct Silent(crate::FilteredAccessSet);

impl Silent {
    fn new() -> Self {
        Self(crate::FilteredAccessSet::new())
    }
}

impl System for Silent {
    fn name(&self) -> &str {
        "silent"
    }
    fn initialize(&mut self, _world: &mut World) {}
    fn is_initialized(&self) -> bool {
        true
    }
    fn access(&self) -> &crate::FilteredAccessSet {
        &self.0
    }
    unsafe fn run_unchecked(&mut self, _world: crate::WorldCell<'_>, _this_run: Tick) {}
    fn has_deferred(&self) -> bool {
        false
    }
    fn last_run(&self) -> Tick {
        Tick::ZERO
    }
    fn check_change_tick(&mut self, _now: Tick) -> bool {
        false
    }
    fn apply_deferred(&mut self, _world: &mut World) {}
}

/// The same for a condition.
struct SilentCondition(crate::FilteredAccessSet);

impl Condition for SilentCondition {
    fn name(&self) -> &str {
        "silent"
    }
    fn initialize(&mut self, _world: &mut World) {}
    fn is_initialized(&self) -> bool {
        true
    }
    fn access(&self) -> &crate::FilteredAccessSet {
        &self.0
    }
    unsafe fn run_unchecked(&mut self, _world: crate::WorldCell<'_>, _this_run: Tick) -> bool {
        true
    }
    fn oldest_last_run(&self, _now: Tick) -> Tick {
        Tick::ZERO
    }
    fn check_change_tick(&mut self, _now: Tick) -> bool {
        false
    }
}

#[test]
fn a_system_or_condition_that_does_not_say_is_taken_to_read_removals() {
    let sys = Silent::new();
    assert!(sys.is_initialized());
    assert!(
        sys.reads_removals(),
        "the safe answer: it only keeps the log longer"
    );
    let cond = SilentCondition(crate::FilteredAccessSet::new());
    assert!(cond.reads_removals());
    assert!(and(never, SilentCondition(crate::FilteredAccessSet::new())).reads_removals());

    let w = world();
    let mut s = Schedule::new(Update);
    s.add_systems(Silent::new());
    assert_eq!(
        s.oldest_removal_reader(w.change_tick()),
        Some(Tick::ZERO),
        "so a schedule counts it even though it is set up"
    );
}

// ── what a schedule says ─────────────────────────────────────────────

#[test]
fn a_schedule_without_readers_has_no_oldest_reader_once_it_is_set_up() {
    let mut w = world();
    let s = Schedule::new(Update);
    assert_eq!(s.oldest_removal_reader(w.change_tick()), None, "empty");

    let mut s = Schedule::new(Update);
    s.add_systems((plain, plain_too));
    assert_eq!(
        s.oldest_removal_reader(w.change_tick()),
        Some(Tick::ZERO),
        "not set up yet, so it cannot say and counts"
    );
    s.run(&mut w);
    assert_eq!(s.oldest_removal_reader(w.change_tick()), None);
    assert!(s.oldest_last_run(w.change_tick()).is_some());
}

#[test]
fn a_system_that_does_not_read_removals_does_not_hold_the_answer_back() {
    let mut w = world();
    let mut s = Schedule::new(Update);
    s.add_systems((plain.run_if(never), reader_a));
    s.run(&mut w);
    // `plain` never ran: the plain oldest is tick zero. The reader ran.
    let now = w.change_tick();
    assert_eq!(s.oldest_last_run(now), Some(Tick::ZERO));
    let reader = s.oldest_removal_reader(now).unwrap();
    assert!(
        reader.get() > 0,
        "the reader ran, so its last run is not zero"
    );
}

#[test]
fn a_reader_skipped_by_its_condition_holds_the_answer_back() {
    let mut w = world();
    let mut s = Schedule::new(Update);
    s.add_systems((reader_a.run_if(never), reader_b));
    s.run(&mut w);
    assert_eq!(
        s.oldest_removal_reader(w.change_tick()),
        Some(Tick::ZERO),
        "reader_a never ran"
    );
    w.get_resource_mut::<Flag>().unwrap().0 = true;
    s.run(&mut w);
    assert!(
        s.oldest_removal_reader(w.change_tick()).unwrap().get() > 0,
        "now it ran"
    );
}

#[test]
fn a_condition_that_reads_removals_holds_the_answer_back() {
    let mut w = world();
    // The system has no removal parameter; its condition does, and runs.
    let mut on_system = Schedule::new(Update);
    on_system.add_systems(plain.run_if(sees_removals));
    on_system.run(&mut w);
    let t = on_system.oldest_removal_reader(w.change_tick()).unwrap();
    assert!(t.get() > 0, "the condition ran");

    // A condition on a set counts, and so does one nested in `and`.
    let mut on_set = Schedule::new(Update);
    on_set
        .configure_sets(Group.run_if(sees_removals))
        .add_systems(plain.in_set(Group));
    on_set.run(&mut w);
    assert!(on_set.oldest_removal_reader(w.change_tick()).is_some());

    let mut nested = Schedule::new(Update);
    nested.add_systems(plain.run_if(and(always, sees_removals)));
    nested.run(&mut w);
    assert!(nested.oldest_removal_reader(w.change_tick()).is_some());

    // The same shapes with a condition that does not read: nothing.
    let mut none = Schedule::new(Update);
    none.configure_sets(Group.run_if(always))
        .add_systems((plain.run_if(always), plain_too.in_set(Group)));
    none.run(&mut w);
    assert_eq!(none.oldest_removal_reader(w.change_tick()), None);
}

#[test]
fn schedules_answer_with_the_oldest_over_all_that_are_not_running() {
    let mut w = world();
    w.init_schedule(Update).add_systems(reader_a);
    w.init_schedule(Other).add_systems(reader_b);
    w.run_schedule(Other);
    let other_ran = w.change_tick();
    w.run_schedule(Update);
    let now = w.change_tick();
    let schedules = w.get_resource::<Schedules>().unwrap();
    let oldest = schedules.oldest_removal_reader(now).unwrap();
    assert!(
        oldest.get() < other_ran.get() + 2,
        "Other's reader ran first, and the answer is the older of the two"
    );
    assert_eq!(
        oldest,
        schedules
            .get(Other)
            .unwrap()
            .oldest_removal_reader(now)
            .unwrap()
    );
    assert!(!schedules.any_running());
}

// ── trimming the world's log by them ─────────────────────────────────

#[test]
fn nothing_is_trimmed_without_schedules_or_without_readers() {
    let mut w = world();
    assert!(!w.trim_removals_for_schedules(), "no Schedules resource");

    w.track_removals::<Hp>();
    let es = spawn_with_hp(&mut w, 3);
    for &e in &es {
        w.remove_static::<Hp>(e);
    }
    w.init_schedule(Update).add_systems(plain);
    w.run_schedule(Update);
    assert!(!w.trim_removals_for_schedules(), "no system reads removals");
    assert_eq!(w.removal_count::<Hp>(), 3, "so nothing was dropped");
}

#[test]
fn a_reader_that_ran_lets_the_log_be_trimmed_after_it_saw_everything() {
    let mut w = world();
    w.init_schedule(Update).add_systems(reader_a);
    w.run_schedule(Update); // sets it up and starts tracking
    let es = spawn_with_hp(&mut w, 4);
    for &e in &es {
        w.remove_static::<Hp>(e);
    }
    assert_eq!(w.removal_count::<Hp>(), 4);
    assert!(
        w.trim_removals_for_schedules(),
        "a reader exists, though it has not seen these yet"
    );
    assert_eq!(
        w.removal_count::<Hp>(),
        4,
        "the reader has not run since: it still needs all four"
    );
    w.run_schedule(Update);
    assert_eq!(seen_a(&w), es);
    assert!(w.trim_removals_for_schedules());
    assert_eq!(w.removal_count::<Hp>(), 0, "everything was seen");
}

#[test]
fn a_reader_that_has_not_run_for_a_while_keeps_what_it_has_yet_to_see() {
    let mut w = world();
    w.init_schedule(Update).add_systems(reader_a);
    w.init_schedule(Other).add_systems(reader_b);
    w.run_schedule(Update);
    w.run_schedule(Other);
    let es = spawn_with_hp(&mut w, 3);
    w.remove_static::<Hp>(es[0]);
    w.run_schedule(Update); // A sees es[0]; B does not run
    w.remove_static::<Hp>(es[1]);
    w.run_schedule(Update); // A sees es[1]
    assert!(w.trim_removals_for_schedules());
    assert_eq!(
        w.removal_count::<Hp>(),
        2,
        "B's last run is before both, so both stay"
    );
    w.run_schedule(Other);
    assert_eq!(seen_b(&w), vec![es[0], es[1]]);
    assert!(w.trim_removals_for_schedules());
    assert_eq!(w.removal_count::<Hp>(), 0);
    assert_eq!(seen_a(&w), vec![es[0], es[1]]);
}

#[test]
fn a_schedule_that_never_runs_and_reads_nothing_does_not_stop_the_trim() {
    let mut w = world();
    w.init_schedule(Update).add_systems(reader_a);
    w.init_schedule(Other).add_systems(plain);
    w.run_schedule(Other); // set up, once, long ago
    w.run_schedule(Update);
    let es = spawn_with_hp(&mut w, 2);
    for &e in &es {
        w.remove_static::<Hp>(e);
    }
    w.run_schedule(Update);
    assert!(w.trim_removals_for_schedules());
    assert_eq!(
        w.removal_count::<Hp>(),
        0,
        "Other's last run is older, but it reads nothing"
    );
}

#[test]
fn nothing_is_trimmed_from_inside_a_running_schedule() {
    let mut w = world();
    w.init_schedule(Other).add_systems(reader_b);
    w.run_schedule(Other);
    w.init_schedule(Update).add_systems(try_trim);
    let es = spawn_with_hp(&mut w, 2);
    for &e in &es {
        w.remove_static::<Hp>(e);
    }
    w.run_schedule(Other);
    assert_eq!(seen_b(&w), es);
    w.run_schedule(Update);
    assert!(
        !w.get_resource::<Trimmed>().unwrap().0,
        "Update was being run, so its systems were out of reach"
    );
    assert_eq!(w.removal_count::<Hp>(), 2, "and the log was left alone");
    assert!(w.trim_removals_for_schedules(), "from outside it works");
    assert_eq!(w.removal_count::<Hp>(), 0);
}

// ── random: everything is seen exactly once, however trimming falls ──

struct Xorshift(u64);
impl Xorshift {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

#[test]
fn random_removals_reads_and_trims_lose_and_repeat_nothing() {
    for seed in 1..=60u64 {
        let mut rng = Xorshift(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
        let mut w = world();
        w.init_schedule(Update).add_systems(reader_a);
        w.init_schedule(Other).add_systems(reader_b);
        w.run_schedule(Update);
        w.run_schedule(Other);
        let es = spawn_with_hp(&mut w, 6);
        let mut has = vec![true; es.len()];
        let mut truth: Vec<Entity> = Vec::new();
        let mut longest = 0;
        for _ in 0..120 {
            match rng.below(6) {
                0 | 1 => {
                    let i = rng.below(es.len());
                    if has[i] {
                        w.remove_static::<Hp>(es[i]);
                        has[i] = false;
                        truth.push(es[i]);
                    }
                }
                2 => {
                    let i = rng.below(es.len());
                    if !has[i] {
                        w.insert_static(es[i], Hp);
                        has[i] = true;
                    }
                }
                3 => w.run_schedule(Update),
                4 => w.run_schedule(Other),
                _ => {
                    w.trim_removals_for_schedules();
                }
            }
            longest = longest.max(w.removal_count::<Hp>());
        }
        w.run_schedule(Update);
        w.run_schedule(Other);
        assert_eq!(seen_a(&w), truth, "seed {seed}: reader A");
        assert_eq!(seen_b(&w), truth, "seed {seed}: reader B");
        assert!(w.trim_removals_for_schedules());
        assert_eq!(
            w.removal_count::<Hp>(),
            0,
            "seed {seed}: all seen, all dropped"
        );
        assert!(longest <= truth.len(), "seed {seed}");
    }
}

#[test]
fn trimming_every_frame_keeps_the_log_short() {
    let mut w = world();
    w.init_schedule(Update).add_systems(reader_a);
    w.run_schedule(Update);
    let es = spawn_with_hp(&mut w, 2);
    let mut longest = 0;
    for frame in 0..200 {
        let e = es[frame % 2];
        w.remove_static::<Hp>(e);
        w.insert_static(e, Hp);
        w.run_schedule(Update);
        w.trim_removals_for_schedules();
        longest = longest.max(w.removal_count::<Hp>());
    }
    assert_eq!(seen_a(&w).len(), 200, "every removal, once");
    assert_eq!(longest, 0, "trimmed after every frame");
}
