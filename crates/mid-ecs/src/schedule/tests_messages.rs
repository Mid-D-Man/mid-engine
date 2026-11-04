// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-ecs-events.md, section "tests_messages.rs"
// ============================================================================
//! Messages through systems and schedules: who sees what and when, the
//! readers that trimming has to wait for, and the trim itself.

use std::any::TypeId;

use super::common_conditions::on_message;
use super::tests_removals::{Silent, SilentCondition};
use super::*;
use crate::commands::Commands;
use crate::message::{MessageReader, MessageWriter, Messages};
use crate::system::IntoSystem;
use crate::system_param::{Res, ResMut};
use crate::tick::ChangeTracker;

// ── fixtures ─────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct Update;
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct Other;

struct Hit(u32);
struct Note(u32);

struct Counter(u32);
struct Gate(bool);
struct Runs(u32);
#[derive(Default)]
struct SeenA(Vec<u32>);
#[derive(Default)]
struct SeenB(Vec<u32>);

fn hit() -> TypeId {
    TypeId::of::<Hit>()
}

fn note() -> TypeId {
    TypeId::of::<Note>()
}

/// Writes 1, 2, 3, ... one per run.
fn emit(mut hits: MessageWriter<Hit>, mut n: ResMut<Counter>) {
    n.0 += 1;
    hits.write(Hit(n.0));
}

fn collect_a(hits: MessageReader<Hit>, mut seen: ResMut<SeenA>) {
    seen.0.extend(hits.read().map(|h| h.0));
}

fn collect_b(hits: MessageReader<Hit>, mut seen: ResMut<SeenB>) {
    seen.0.extend(hits.read().map(|h| h.0));
}

fn read_twice(hits: MessageReader<Hit>, mut seen: ResMut<SeenA>) {
    seen.0.extend(hits.read().map(|h| h.0));
    seen.0.extend(hits.read().map(|h| h.0));
}

fn count_run(mut runs: ResMut<Runs>) {
    runs.0 += 1;
}

fn open(gate: Res<Gate>) -> bool {
    gate.0
}

fn plain() {}

fn try_trim(world: &mut World) {
    let trimmed = world.trim_messages_for_schedules();
    world.insert_resource(Runs(trimmed as u32));
}

fn world() -> World {
    let mut w = World::new();
    w.insert_resource(Counter(0));
    w.insert_resource(Gate(false));
    w.insert_resource(Runs(0));
    w.insert_resource(SeenA::default());
    w.insert_resource(SeenB::default());
    w
}

fn seen_a(w: &World) -> Vec<u32> {
    w.get_resource::<SeenA>().unwrap().0.clone()
}

fn seen_b(w: &World) -> Vec<u32> {
    w.get_resource::<SeenB>().unwrap().0.clone()
}

fn runs(w: &World) -> u32 {
    w.get_resource::<Runs>().unwrap().0
}

// ── who sees what, and when ──────────────────────────────────────────

#[test]
fn a_reader_after_its_writer_sees_the_message_in_the_same_frame() {
    let mut w = world();
    let mut s = Schedule::new(Update);
    s.add_systems((emit, collect_a).chain());
    for _ in 0..3 {
        s.run(&mut w);
    }
    assert_eq!(seen_a(&w), [1, 2, 3]);
}

#[test]
fn a_reader_before_its_writer_sees_the_message_a_frame_later() {
    let mut w = world();
    let mut s = Schedule::new(Update);
    s.add_systems((collect_a, emit).chain());
    for _ in 0..3 {
        s.run(&mut w);
    }
    assert_eq!(seen_a(&w), [1, 2], "the third is not read yet");
    s.run(&mut w);
    assert_eq!(seen_a(&w), [1, 2, 3]);
}

#[test]
fn every_reader_sees_every_message() {
    let mut w = world();
    let mut s = Schedule::new(Update);
    s.add_systems((emit, collect_a, collect_b).chain());
    for _ in 0..4 {
        s.run(&mut w);
    }
    assert_eq!(seen_a(&w), [1, 2, 3, 4]);
    assert_eq!(seen_b(&w), [1, 2, 3, 4]);
}

#[test]
fn reading_twice_in_one_run_gives_the_same_messages_twice() {
    let mut w = world();
    let mut s = Schedule::new(Update);
    s.add_systems((emit, read_twice).chain());
    s.run(&mut w);
    s.run(&mut w);
    assert_eq!(seen_a(&w), [1, 1, 2, 2]);
}

#[test]
fn a_reader_that_was_skipped_misses_nothing() {
    let mut w = world();
    let mut s = Schedule::new(Update);
    s.add_systems((emit, collect_a.run_if(open)).chain());
    for _ in 0..3 {
        s.run(&mut w);
    }
    assert_eq!(seen_a(&w), Vec::<u32>::new());
    w.get_resource_mut::<Gate>().unwrap().0 = true;
    s.run(&mut w);
    assert_eq!(seen_a(&w), [1, 2, 3, 4], "all four, in the order written");
    s.run(&mut w);
    assert_eq!(seen_a(&w), [1, 2, 3, 4, 5], "and only the new one after");
}

#[test]
fn the_length_and_emptiness_of_a_read_count_what_read_yields() {
    #[derive(Default)]
    struct Lens(Vec<(usize, bool)>);
    fn measure(hits: MessageReader<Hit>, mut lens: ResMut<Lens>) {
        lens.0.push((hits.len(), hits.is_empty()));
    }
    let mut w = world();
    w.insert_resource(Lens::default());
    let mut s = Schedule::new(Update);
    s.add_systems((measure, emit, emit, measure).chain());
    s.run(&mut w);
    s.run(&mut w);
    let lens = &w.get_resource::<Lens>().unwrap().0;
    assert_eq!(lens, &[(0, true), (2, false), (2, false), (2, false)]);
}

#[test]
fn a_system_cannot_read_and_write_the_same_type() {
    fn both(_r: MessageReader<Hit>, _w: MessageWriter<Hit>) {}
    let mut w = world();
    let mut s = Schedule::new(Update);
    s.add_systems(both);
    let err = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| s.run(&mut w)))
        .expect_err("two parameters that conflict");
    let text = err
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| err.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_default();
    assert!(text.contains("conflict"), "{text}");
}

#[test]
fn a_system_can_read_one_type_and_write_another() {
    fn forward(hits: MessageReader<Hit>, mut notes: MessageWriter<Note>) {
        for h in hits.read() {
            notes.write(Note(h.0 * 10));
        }
    }
    fn collect_notes(notes: MessageReader<Note>, mut seen: ResMut<SeenA>) {
        seen.0.extend(notes.read().map(|n| n.0));
    }
    let mut w = world();
    let mut s = Schedule::new(Update);
    s.add_systems((emit, forward, collect_notes).chain());
    s.run(&mut w);
    s.run(&mut w);
    assert_eq!(seen_a(&w), [10, 20]);
}

#[test]
fn a_system_can_have_two_readers_of_one_type() {
    fn both(a: MessageReader<Hit>, b: MessageReader<Hit>, mut seen: ResMut<SeenA>) {
        seen.0.push(a.len() as u32);
        seen.0.push(b.len() as u32);
    }
    let mut w = world();
    let mut s = Schedule::new(Update);
    s.add_systems((emit, both).chain());
    s.run(&mut w);
    assert_eq!(seen_a(&w), [1, 1]);
}

#[test]
fn writing_marks_the_log_changed_and_not_writing_does_not() {
    #[derive(Default)]
    struct Marks(Vec<bool>);
    fn mark(log: Res<Messages<Hit>>, mut marks: ResMut<Marks>) {
        marks.0.push(log.is_changed());
    }
    fn write_when_open(mut hits: MessageWriter<Hit>, gate: Res<Gate>) {
        if gate.0 {
            hits.write(Hit(1));
        }
    }
    let mut w = world();
    w.insert_resource(Marks::default());
    let mut s = Schedule::new(Update);
    s.add_systems((write_when_open, mark).chain());
    s.run(&mut w); // the log was inserted: changed
    s.run(&mut w); // nobody wrote
    w.get_resource_mut::<Gate>().unwrap().0 = true;
    s.run(&mut w); // written
    w.get_resource_mut::<Gate>().unwrap().0 = false;
    s.run(&mut w); // not
    let marks = &w.get_resource::<Marks>().unwrap().0;
    assert_eq!(marks, &[true, false, true, false]);
}

#[test]
fn write_batch_writes_in_order() {
    fn batch(mut hits: MessageWriter<Hit>) {
        hits.write_batch((1..=4).map(Hit));
    }
    let mut w = world();
    let mut s = Schedule::new(Update);
    s.add_systems((batch, collect_a).chain());
    s.run(&mut w);
    assert_eq!(seen_a(&w), [1, 2, 3, 4]);
}

#[test]
fn commands_write_messages_that_later_systems_read() {
    fn via_commands(mut commands: Commands) {
        commands.write_message(Hit(7));
        commands.write_message(Hit(8));
    }
    let mut w = world();
    let mut s = Schedule::new(Update);
    s.add_systems((via_commands, collect_a).chain());
    s.run(&mut w);
    assert_eq!(seen_a(&w), [7, 8]);
}

#[test]
fn writers_of_one_type_do_not_overlap_when_multi_threaded() {
    fn burst<const N: u32>(mut hits: MessageWriter<Hit>) {
        for i in 0..50 {
            hits.write(Hit(N * 100 + i));
        }
    }
    let mut w = world();
    let mut s = Schedule::new(Update);
    s.set_executor(ExecutorKind::MultiThreaded);
    s.add_systems((burst::<1>, burst::<2>, burst::<3>, burst::<4>, plain, plain));
    for _ in 0..20 {
        w.clear_messages();
        s.run(&mut w);
        let mut all: Vec<u32> = w
            .messages_since::<Hit>(&ChangeTracker::new())
            .map(|h| h.0)
            .collect();
        assert_eq!(all.len(), 200);
        all.sort_unstable();
        all.dedup();
        assert_eq!(all.len(), 200, "every message, once");
    }
}

// ── run conditions ───────────────────────────────────────────────────

#[test]
fn on_message_runs_its_system_only_when_something_was_written() {
    let mut w = world();
    let mut s = Schedule::new(Update);
    s.add_systems(count_run.run_if(on_message::<Hit>));
    s.run(&mut w);
    assert_eq!(runs(&w), 0, "nothing written");
    w.write_message(Hit(1));
    s.run(&mut w);
    assert_eq!(runs(&w), 1);
    s.run(&mut w);
    assert_eq!(runs(&w), 1, "nothing new since");
    w.write_message(Hit(2));
    w.write_message(Hit(3));
    s.run(&mut w);
    assert_eq!(runs(&w), 2);
}

#[test]
fn the_gated_system_reads_all_since_its_own_last_run() {
    let mut w = world();
    let mut s = Schedule::new(Update);
    s.add_systems(collect_a.run_if(on_message::<Hit>));
    w.write_message(Hit(1));
    s.run(&mut w);
    w.write_message(Hit(2));
    w.write_message(Hit(3));
    s.run(&mut w);
    s.run(&mut w);
    assert_eq!(seen_a(&w), [1, 2, 3]);
}

// ── what reads what ──────────────────────────────────────────────────

fn initialized<S: System>(mut s: S, w: &mut World) -> S {
    s.initialize(w);
    s
}

#[test]
fn a_system_reads_the_types_its_readers_name_and_no_other() {
    let mut w = world();
    let r = initialized(IntoSystem::into_system(collect_a), &mut w);
    assert!(r.reads_message(hit()));
    assert!(!r.reads_message(note()));
    let w_only = initialized(IntoSystem::into_system(emit), &mut w);
    assert!(!w_only.reads_message(hit()), "writing is not reading");
    let ex = initialized(IntoSystem::into_system(try_trim), &mut w);
    assert!(!ex.reads_message(hit()));
    let none = initialized(IntoSystem::into_system(plain), &mut w);
    assert!(!none.reads_message(hit()));

    fn two(_h: MessageReader<Hit>, _n: MessageReader<Note>) {}
    let both = initialized(IntoSystem::into_system(two), &mut w);
    assert!(both.reads_message(hit()) && both.reads_message(note()));
}

#[test]
fn conditions_report_the_types_through_every_combinator() {
    let mut w = world();
    let mut reads = on_message::<Hit>.into_condition();
    reads.initialize(&mut w);
    assert!(reads.reads_message(hit()));
    assert!(!reads.reads_message(note()));

    let mut n = not(on_message::<Hit>);
    n.initialize(&mut w);
    assert!(n.reads_message(hit()) && !n.reads_message(note()));

    let mut a1 = and(open, on_message::<Hit>);
    let mut a2 = and(on_message::<Hit>, open);
    let mut a3 = and(open, open);
    let mut o1 = or(open, on_message::<Hit>);
    let mut o2 = or(on_message::<Hit>, open);
    let mut o3 = or(open, open);
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
    assert!(a1.reads_message(hit()) && a2.reads_message(hit()));
    assert!(!a3.reads_message(hit()));
    assert!(o1.reads_message(hit()) && o2.reads_message(hit()));
    assert!(!o3.reads_message(hit()));
    assert!(!a1.reads_message(note()) && !o2.reads_message(note()));

    let mut boxed: BoxedCondition = Box::new(on_message::<Hit>.into_condition());
    boxed.initialize(&mut w);
    assert!(boxed.reads_message(hit()) && !boxed.reads_message(note()));
}

#[test]
fn a_system_or_condition_that_does_not_say_is_taken_to_read_any_type() {
    let sys = Silent::new();
    assert!(sys.reads_message(hit()) && sys.reads_message(note()));
    let cond = SilentCondition(crate::FilteredAccessSet::new());
    assert!(cond.reads_message(hit()) && cond.reads_message(note()));
    assert!(and(open, SilentCondition(crate::FilteredAccessSet::new())).reads_message(hit()));
}

// ── what a schedule says ─────────────────────────────────────────────

#[test]
fn a_system_or_condition_not_set_up_yet_counts_for_every_kind_of_reader() {
    let mut w = world();
    let kinds = || {
        [
            Reader::Removals,
            Reader::Message(hit()),
            Reader::Message(note()),
        ]
    };

    let mut sys = IntoSystem::into_system(plain);
    for r in kinds() {
        assert!(r.counts_system(&sys), "a system not set up yet cannot say");
    }
    sys.initialize(&mut w);
    assert!(Reader::Any.counts_system(&sys));
    for r in kinds() {
        assert!(!r.counts_system(&sys), "set up, and it reads nothing");
    }

    let mut cond = open.into_condition();
    for r in kinds() {
        assert!(
            r.counts_condition(&cond),
            "a condition not set up yet cannot say"
        );
    }
    cond.initialize(&mut w);
    assert!(Reader::Any.counts_condition(&cond));
    for r in kinds() {
        assert!(!r.counts_condition(&cond), "set up, and it reads nothing");
    }
}

#[test]
fn a_schedule_with_no_reader_of_a_type_has_no_oldest_reader_of_it() {
    let mut w = world();
    let mut s = Schedule::new(Update);
    assert_eq!(s.oldest_message_reader(hit(), w.change_tick()), None);
    s.add_systems((emit, collect_a));
    s.run(&mut w);
    let now = w.change_tick();
    assert!(s.oldest_message_reader(hit(), now).is_some());
    assert_eq!(s.oldest_message_reader(note(), now), None);
}

#[test]
fn a_system_not_set_up_yet_counts_for_every_type() {
    let w = world();
    let mut s = Schedule::new(Update);
    s.add_systems(plain);
    assert_eq!(
        s.oldest_message_reader(hit(), w.change_tick()),
        Some(Tick::ZERO)
    );
    assert_eq!(
        s.oldest_message_reader(note(), w.change_tick()),
        Some(Tick::ZERO)
    );
}

#[test]
fn a_skipped_reader_holds_back_its_own_type_only() {
    fn read_notes(notes: MessageReader<Note>) {
        let _ = notes.len();
    }
    let mut w = world();
    let mut s = Schedule::new(Update);
    s.add_systems((collect_a.run_if(open), read_notes));
    s.run(&mut w);
    let now = w.change_tick();
    assert_eq!(
        s.oldest_message_reader(hit(), now),
        Some(Tick::ZERO),
        "collect_a never ran"
    );
    assert!(
        s.oldest_message_reader(note(), now).unwrap().get() > 0,
        "read_notes did, and collect_a does not read notes"
    );
}

#[test]
fn a_condition_that_reads_a_type_holds_back_that_type() {
    let mut w = world();
    let mut on_system = Schedule::new(Update);
    on_system.add_systems(plain.run_if(on_message::<Hit>));
    on_system.run(&mut w);
    let now = w.change_tick();
    assert!(on_system.oldest_message_reader(hit(), now).unwrap().get() > 0);
    assert_eq!(on_system.oldest_message_reader(note(), now), None);

    #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    struct Group;
    impl SystemSet for Group {}
    let mut on_set = Schedule::new(Update);
    on_set
        .configure_sets(Group.run_if(on_message::<Hit>))
        .add_systems(plain.in_set(Group));
    on_set.run(&mut w);
    let now = w.change_tick();
    assert!(on_set.oldest_message_reader(hit(), now).is_some());
    assert_eq!(on_set.oldest_message_reader(note(), now), None);

    let mut nested = Schedule::new(Update);
    nested.add_systems(plain.run_if(and(open, on_message::<Hit>)));
    nested.run(&mut w);
    assert!(nested
        .oldest_message_reader(hit(), w.change_tick())
        .is_some());
}

#[test]
fn schedules_answer_with_the_oldest_reader_of_the_type_over_all_not_running() {
    let mut w = world();
    w.init_schedule(Update).add_systems(collect_a);
    w.init_schedule(Other).add_systems(collect_b);
    w.run_schedule(Other);
    let other_ran = w.change_tick();
    w.run_schedule(Update);
    let now = w.change_tick();
    let schedules = w.get_resource::<Schedules>().unwrap();
    let oldest = schedules.oldest_message_reader(hit(), now).unwrap();
    assert!(oldest.get() < other_ran.get() + 2);
    assert_eq!(
        oldest,
        schedules
            .get(Other)
            .unwrap()
            .oldest_message_reader(hit(), now)
            .unwrap()
    );
    assert_eq!(schedules.oldest_message_reader(note(), now), None);
}

// ── the world and its logs ───────────────────────────────────────────

#[test]
fn the_world_writes_and_reads_messages_through_a_tracker() {
    let mut w = World::new();
    let mut tracker = ChangeTracker::new();
    assert_eq!(w.messages_since::<Hit>(&tracker).count(), 0, "never held");
    assert_eq!(w.message_count::<Hit>(), 0);
    w.write_message(Hit(1));
    w.write_message(Hit(2));
    assert_eq!(w.message_count::<Hit>(), 2);
    let got: Vec<u32> = w.messages_since::<Hit>(&tracker).map(|h| h.0).collect();
    assert_eq!(got, [1, 2]);
    tracker.update(&w);
    w.increment_change_tick();
    w.write_message(Hit(3));
    let got: Vec<u32> = w.messages_since::<Hit>(&tracker).map(|h| h.0).collect();
    assert_eq!(got, [3]);
}

#[test]
fn init_message_is_idempotent_and_keeps_what_is_there() {
    let mut w = World::new();
    w.init_message::<Hit>();
    w.write_message(Hit(1));
    w.init_message::<Hit>();
    assert_eq!(w.message_count::<Hit>(), 1);
    assert!(w.contains_resource::<Messages<Hit>>());
}

#[test]
fn trim_messages_and_clear_messages_reach_every_type() {
    let mut w = World::new();
    w.write_message(Hit(1));
    w.write_message(Note(1));
    let upto = w.change_tick();
    w.increment_change_tick();
    w.write_message(Hit(2));
    w.write_message(Note(2));
    w.trim_messages(upto);
    assert_eq!(w.message_count::<Hit>(), 1, "stamped after the trim tick");
    assert_eq!(w.message_count::<Note>(), 1);
    w.clear_messages();
    assert_eq!(w.message_count::<Hit>(), 0);
    assert_eq!(w.message_count::<Note>(), 0);
    w.write_message(Hit(3));
    assert_eq!(w.message_count::<Hit>(), 1, "still held after a clear");
}

// ── trimming by the schedules ────────────────────────────────────────

#[test]
fn a_type_nobody_reads_is_kept_for_the_interval_after_it_was_written() {
    let mut w = World::new();
    w.write_message(Hit(1));
    w.write_message(Hit(2));
    assert!(w.trim_messages_for_schedules());
    assert_eq!(
        w.message_count::<Hit>(),
        2,
        "the first trim keeps everything"
    );
    w.write_message(Hit(3));
    assert!(w.trim_messages_for_schedules());
    assert_eq!(
        w.message_count::<Hit>(),
        1,
        "two written before the previous trim are gone, one since is kept"
    );
    assert!(w.trim_messages_for_schedules());
    assert_eq!(w.message_count::<Hit>(), 0);
}

#[test]
fn a_message_written_just_after_a_trim_is_not_trimmed_by_the_next() {
    let mut w = World::new();
    w.write_message(Hit(1));
    w.trim_messages_for_schedules();
    w.write_message(Hit(2)); // right after, with no tick passing
    w.trim_messages_for_schedules();
    assert_eq!(w.message_count::<Hit>(), 1, "it has had its interval");
    w.trim_messages_for_schedules();
    assert_eq!(w.message_count::<Hit>(), 0);
}

#[test]
fn a_tracker_that_reads_once_per_interval_loses_nothing_to_the_trim() {
    let mut w = World::new();
    let mut tracker = ChangeTracker::new();
    let mut got = Vec::new();
    let mut next = 0;
    for _ in 0..30 {
        for _ in 0..3 {
            next += 1;
            w.write_message(Hit(next));
        }
        got.extend(w.messages_since::<Hit>(&tracker).map(|h| h.0));
        tracker.update(&w);
        w.trim_messages_for_schedules();
        for _ in 0..2 {
            next += 1;
            w.write_message(Hit(next)); // after the trim, before the read
        }
    }
    got.extend(w.messages_since::<Hit>(&tracker).map(|h| h.0));
    assert_eq!(got, (1..=next).collect::<Vec<_>>());
}

#[test]
fn a_reader_in_a_schedule_holds_the_log_until_it_has_run() {
    let mut w = world();
    w.init_schedule(Update).add_systems(collect_a);
    w.run_schedule(Update); // set up
    for i in 1..=4 {
        w.write_message(Hit(i));
    }
    for _ in 0..4 {
        assert!(w.trim_messages_for_schedules());
    }
    assert_eq!(
        w.message_count::<Hit>(),
        4,
        "four trims later, the reader still has not run"
    );
    w.run_schedule(Update);
    assert_eq!(seen_a(&w), [1, 2, 3, 4]);
    w.trim_messages_for_schedules();
    w.trim_messages_for_schedules();
    assert_eq!(w.message_count::<Hit>(), 0);
}

#[test]
fn a_skipped_reader_of_one_type_does_not_hold_back_another() {
    fn read_notes(notes: MessageReader<Note>, mut seen: ResMut<SeenB>) {
        seen.0.extend(notes.read().map(|n| n.0));
    }
    let mut w = world();
    w.init_schedule(Update)
        .add_systems((collect_a.run_if(open), read_notes));
    w.run_schedule(Update);
    w.write_message(Hit(1));
    w.write_message(Note(1));
    w.run_schedule(Update);
    for _ in 0..3 {
        w.trim_messages_for_schedules();
    }
    assert_eq!(w.message_count::<Hit>(), 1, "collect_a is still to read it");
    assert_eq!(w.message_count::<Note>(), 0, "read_notes has read it");
    assert_eq!(seen_b(&w), [1]);
}

#[test]
fn nothing_is_trimmed_from_inside_a_running_schedule() {
    let mut w = world();
    w.write_message(Hit(1));
    w.trim_messages_for_schedules();
    w.write_message(Hit(2));
    w.init_schedule(Update).add_systems(try_trim);
    w.run_schedule(Update);
    assert_eq!(runs(&w), 0, "Update was being run");
    assert_eq!(w.message_count::<Hit>(), 2, "and the log was left alone");
    assert!(w.trim_messages_for_schedules(), "from outside it works");
}

#[test]
fn a_world_with_no_schedules_trims_by_the_previous_call() {
    let mut w = World::new();
    assert!(!w.contains_resource::<Schedules>());
    w.write_message(Hit(1));
    w.trim_messages_for_schedules();
    w.trim_messages_for_schedules();
    assert_eq!(w.message_count::<Hit>(), 0);
}

// ── random: every reader sees every message exactly once ─────────────

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
fn random_writes_reads_and_trims_lose_and_repeat_nothing() {
    for seed in 1..=60u64 {
        let mut rng = Xorshift(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
        let mut w = world();
        w.init_schedule(Update).add_systems(collect_a);
        w.init_schedule(Other).add_systems(collect_b);
        w.run_schedule(Update);
        w.run_schedule(Other);
        let mut written = 0u32;
        let mut longest = 0;
        for _ in 0..150 {
            match rng.below(6) {
                0 | 1 => {
                    written += 1;
                    w.write_message(Hit(written));
                }
                2 => w.run_schedule(Update),
                3 => w.run_schedule(Other),
                _ => {
                    w.trim_messages_for_schedules();
                }
            }
            longest = longest.max(w.message_count::<Hit>());
        }
        w.run_schedule(Update);
        w.run_schedule(Other);
        let truth: Vec<u32> = (1..=written).collect();
        assert_eq!(seen_a(&w), truth, "seed {seed}: reader A");
        assert_eq!(seen_b(&w), truth, "seed {seed}: reader B");
        w.trim_messages_for_schedules();
        w.trim_messages_for_schedules();
        assert_eq!(w.message_count::<Hit>(), 0, "seed {seed}: seen and dropped");
        assert!(longest as u32 <= written, "seed {seed}");
    }
}

#[test]
fn trimming_every_frame_keeps_the_log_short_and_misses_nothing() {
    let mut w = world();
    w.init_schedule(Update)
        .add_systems((emit, collect_a).chain());
    w.run_schedule(Update);
    let mut longest = 0;
    for _ in 0..200 {
        w.run_schedule(Update);
        w.trim_messages_for_schedules();
        longest = longest.max(w.message_count::<Hit>());
    }
    assert_eq!(seen_a(&w), (1..=201).collect::<Vec<_>>());
    assert!(longest <= 2, "at most the last interval's, was {longest}");
}
