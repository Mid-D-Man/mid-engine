// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-app.md, section "tests.rs"
// ============================================================================
//! The app end to end: stage order, the fixed timestep, plugins, the time
//! resources, and trimming the removal log.

use std::time::Duration;

use mid_ecs::schedule::{ExecutorKind, IntoScheduleConfigs};
use mid_ecs::{Commands, RemovedComponents, Res, ResMut, World};
use mid_time::{Clock, FixedTimestep};

use crate::*;

// ── fixtures ─────────────────────────────────────────────────────────

#[derive(Default)]
struct Log(Vec<&'static str>);

#[derive(Default)]
struct Count(u32);

fn ms(n: u64) -> Duration {
    Duration::from_millis(n)
}

/// One step of the default timestep, as a delta that makes exactly one.
fn one_step() -> Duration {
    Duration::from_secs_f32(1.0 / DEFAULT_FIXED_HZ)
}

fn app_with_log() -> App {
    let mut app = App::new();
    app.insert_resource(Log::default());
    app
}

fn log(app: &App) -> Vec<&'static str> {
    app.world.get_resource::<Log>().unwrap().0.clone()
}

macro_rules! logger {
    ($name:ident) => {
        fn $name(mut log: ResMut<Log>) {
            log.0.push(stringify!($name));
        }
    };
}
logger!(pre);
logger!(update);
logger!(net);
logger!(physics);
logger!(post);
logger!(first);
logger!(second);
logger!(third);

fn bump(mut count: ResMut<Count>) {
    count.0 += 1;
}

// ── stages ───────────────────────────────────────────────────────────

#[test]
fn a_new_app_has_a_schedule_for_every_stage_and_for_main() {
    let app = App::new();
    assert!(app.world.contains_schedule(Main));
    assert!(app.world.contains_schedule(PreUpdate));
    assert!(app.world.contains_schedule(StateTransition));
    assert!(app.world.contains_schedule(Update));
    assert!(app.world.contains_schedule(Net));
    assert!(app.world.contains_schedule(Physics));
    assert!(app.world.contains_schedule(PostUpdate));
}

#[test]
fn stages_run_in_their_fixed_order_whatever_order_they_were_added_in() {
    let mut app = app_with_log();
    app.add_systems(PostUpdate, post)
        .add_systems(Physics, physics)
        .add_systems(Net, net)
        .add_systems(Update, update)
        .add_systems(PreUpdate, pre);
    app.update_with(one_step());
    assert_eq!(log(&app), ["pre", "update", "net", "physics", "post"]);
}

#[test]
fn exclusive_closures_still_work_as_systems() {
    let mut app = App::new();
    let e = app.world.spawn();
    app.world.insert(e, Vec::<&'static str>::new());
    app.add_systems(PostUpdate, move |world: &mut World| {
        world.get_mut::<Vec<&'static str>>(e).unwrap().push("post");
    });
    app.add_systems(PreUpdate, move |world: &mut World| {
        world.get_mut::<Vec<&'static str>>(e).unwrap().push("pre");
    });
    app.add_systems(Update, move |world: &mut World| {
        world
            .get_mut::<Vec<&'static str>>(e)
            .unwrap()
            .push("update");
    });
    app.update_with(ms(16));
    let seen = app.world.get::<Vec<&'static str>>(e).unwrap();
    assert_eq!(seen.as_slice(), &["pre", "update", "post"]);
}

#[test]
fn an_app_with_no_systems_runs_a_frame() {
    let mut app = App::new();
    app.update_with(ms(16));
    app.update();
}

#[test]
fn systems_in_one_stage_are_ordered_by_the_schedule_not_by_when_they_were_added() {
    let mut app = app_with_log();
    app.add_systems(Update, (third.after(second), first.before(second), second));
    app.update_with(ms(1));
    assert_eq!(log(&app), ["first", "second", "third"]);
}

#[test]
fn conditions_and_chains_work_through_the_app() {
    let mut app = app_with_log();
    app.insert_resource(Count(0));
    app.add_systems(Update, (first, second).chain());
    app.add_systems(Update, third.run_if(|count: Res<Count>| count.0 >= 2));
    app.add_systems(PostUpdate, bump);
    for _ in 0..3 {
        app.update_with(ms(1));
    }
    // the condition is true from the third frame, when the count is 2
    assert_eq!(
        log(&app),
        ["first", "second", "first", "second", "first", "second", "third"]
    );
}

#[test]
fn a_system_can_be_added_to_main_and_runs_each_frame() {
    let mut app = App::new();
    app.insert_resource(Count(0));
    app.add_systems(Main, bump);
    app.update_with(ms(1));
    app.update_with(ms(1));
    assert_eq!(app.world.get_resource::<Count>().unwrap().0, 2);
}

#[test]
fn a_stage_can_be_set_to_run_multi_threaded() {
    let mut app = App::new();
    app.insert_resource(Count(0));
    app.insert_resource(Log::default());
    app.schedule_mut(Update)
        .set_executor(ExecutorKind::MultiThreaded);
    app.add_systems(Update, (bump, first, second));
    app.update_with(ms(1));
    assert_eq!(app.world.get_resource::<Count>().unwrap().0, 1);
    assert_eq!(log(&app).len(), 2);
}

#[test]
fn configure_sets_reaches_the_schedule() {
    use mid_ecs::schedule::SystemSet;

    #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    enum Phase {
        Early,
        Late,
    }
    impl SystemSet for Phase {}

    let mut app = app_with_log();
    app.configure_sets(Update, (Phase::Early, Phase::Late).chain());
    app.add_systems(
        Update,
        (second.in_set(Phase::Late), first.in_set(Phase::Early)),
    );
    app.update_with(ms(1));
    assert_eq!(log(&app), ["first", "second"]);
}

// ── the fixed timestep ───────────────────────────────────────────────

fn app_counting_physics() -> App {
    let mut app = App::new();
    app.insert_resource(Count(0));
    app.add_systems(Physics, bump);
    app
}

fn physics_runs(app: &App) -> u32 {
    app.world.get_resource::<Count>().unwrap().0
}

#[test]
fn physics_runs_once_for_each_step_the_frames_time_makes() {
    let mut app = app_counting_physics();
    app.update_with(one_step());
    assert_eq!(physics_runs(&app), 1);
    app.update_with(one_step() * 3);
    assert_eq!(physics_runs(&app), 4, "three more");
}

#[test]
fn a_frame_shorter_than_a_step_runs_physics_not_at_all_and_the_time_carries() {
    let mut app = app_counting_physics();
    let third = one_step() / 3;
    app.update_with(third);
    app.update_with(third);
    assert_eq!(physics_runs(&app), 0);
    // Nanosecond rounding in the division could leave it a hair short.
    app.update_with(third + Duration::from_micros(10));
    assert_eq!(physics_runs(&app), 1, "the three thirds added up to a step");
}

#[test]
fn the_other_stages_run_once_a_frame_however_many_steps_it_made() {
    let mut app = app_with_log();
    app.add_systems(Update, update)
        .add_systems(Physics, physics);
    app.update_with(one_step() * 3);
    assert_eq!(log(&app), ["update", "physics", "physics", "physics"]);
    app.update_with(Duration::ZERO);
    assert_eq!(
        log(&app),
        ["update", "physics", "physics", "physics", "update"],
        "no time, no steps, still a frame"
    );
}

#[test]
fn a_long_stall_is_capped_and_the_rest_dropped() {
    let mut app = app_counting_physics();
    app.set_fixed_timestep(60.0, 3);
    app.update_with(Duration::from_secs(1));
    assert_eq!(physics_runs(&app), 3, "the cap");
    app.update_with(Duration::ZERO);
    assert_eq!(physics_runs(&app), 3, "nothing is owed after the stall");
}

#[test]
fn the_default_timestep_is_sixty_hertz_with_a_cap_of_eight() {
    let mut app = app_counting_physics();
    assert_eq!(DEFAULT_FIXED_HZ, 60.0);
    assert_eq!(DEFAULT_MAX_FIXED_STEPS, 8);
    app.update_with(Duration::from_secs(1));
    assert_eq!(physics_runs(&app), 8);
}

#[test]
fn a_different_rate_changes_how_many_steps_a_frame_makes() {
    let mut app = app_counting_physics();
    app.set_fixed_timestep(10.0, 100);
    app.update_with(Duration::from_millis(350));
    assert_eq!(physics_runs(&app), 3, "a step is 100 ms");
}

// ── time as a resource ───────────────────────────────────────────────

#[test]
fn systems_read_the_frames_delta_and_the_steps_length() {
    #[derive(Default)]
    struct Seen {
        frame: Duration,
        step: f32,
        in_physics: f32,
    }
    fn read_frame(clock: Res<Clock>, step: Res<FixedTimestep>, mut seen: ResMut<Seen>) {
        seen.frame = clock.delta();
        seen.step = step.step_secs();
    }
    fn read_step(step: Res<FixedTimestep>, mut seen: ResMut<Seen>) {
        seen.in_physics = step.step_secs();
    }

    let mut app = App::new();
    app.insert_resource(Seen::default());
    app.add_systems(Update, read_frame)
        .add_systems(Physics, read_step);
    app.update_with(ms(40));
    let seen = app.world.get_resource::<Seen>().unwrap();
    assert_eq!(seen.frame, ms(40));
    assert!((seen.step - 1.0 / 60.0).abs() < 1e-6);
    assert!((seen.in_physics - 1.0 / 60.0).abs() < 1e-6);
    let clock = app.world.get_resource::<Clock>().unwrap();
    assert_eq!(clock.elapsed(), ms(40));
}

#[test]
fn update_reads_the_platform_clock() {
    let mut app = App::new();
    app.update();
    std::thread::sleep(ms(5));
    app.update();
    let clock = app.world.get_resource::<Clock>().unwrap();
    assert!(clock.delta() >= ms(5), "{:?}", clock.delta());
    assert!(clock.elapsed() >= ms(5));
}

// ── plugins ──────────────────────────────────────────────────────────

#[test]
fn a_plugin_builds_at_once_and_its_systems_run() {
    struct CountPlugin;
    impl Plugin for CountPlugin {
        fn build(&self, app: &mut App) {
            app.insert_resource(Count(0))
                .add_systems(Update, bump)
                .add_systems(Update, bump);
        }
    }

    let mut app = App::new();
    app.add_plugin(CountPlugin);
    assert!(
        app.world.get_resource::<Count>().is_some(),
        "built before add_plugin returned"
    );
    app.update_with(ms(1));
    assert_eq!(app.world.get_resource::<Count>().unwrap().0, 2);
}

#[test]
fn a_plugin_can_add_another() {
    struct Inner;
    impl Plugin for Inner {
        fn build(&self, app: &mut App) {
            app.insert_resource(Count(0)).add_systems(Update, bump);
        }
    }
    struct Outer;
    impl Plugin for Outer {
        fn build(&self, app: &mut App) {
            app.add_plugin(Inner);
        }
    }
    let mut app = App::new();
    app.add_plugin(Outer);
    app.update_with(ms(1));
    assert_eq!(app.world.get_resource::<Count>().unwrap().0, 1);
}

// ── run_until ────────────────────────────────────────────────────────

#[test]
fn run_until_runs_frames_until_done_says_so() {
    let mut app = App::new();
    app.insert_resource(Count(0));
    app.add_systems(Update, bump);
    let mut asked = 0;
    app.run_until(|app| {
        asked += 1;
        app.world.get_resource::<Count>().unwrap().0 >= 4
    });
    assert_eq!(app.world.get_resource::<Count>().unwrap().0, 4);
    assert_eq!(asked, 4, "asked after each frame");
}

#[test]
fn run_until_runs_at_least_one_frame() {
    let mut app = App::new();
    app.insert_resource(Count(0));
    app.add_systems(Update, bump);
    app.run_until(|_| true);
    assert_eq!(app.world.get_resource::<Count>().unwrap().0, 1);
}

// ── the removal log ──────────────────────────────────────────────────

struct Hp;

#[derive(Default)]
struct Reaped(u32);

fn reap(removed: RemovedComponents<Hp>, mut reaped: ResMut<Reaped>) {
    reaped.0 += removed.iter().count() as u32;
}

fn app_with_reader() -> App {
    let mut app = App::new();
    app.insert_resource(Reaped::default());
    app.add_systems(Update, reap);
    app.update_with(ms(1)); // set up, so removals are recorded
    app
}

fn kill(app: &mut App, n: usize) {
    for _ in 0..n {
        let e = app.world.spawn();
        app.world.insert_static(e, Hp);
        app.world.remove_static::<Hp>(e);
    }
}

#[test]
fn the_removal_log_is_trimmed_after_each_frame_and_nothing_is_missed() {
    let mut app = app_with_reader();
    let mut longest = 0;
    for _ in 0..50 {
        kill(&mut app, 3);
        assert_eq!(app.world.removal_count::<Hp>(), 3);
        app.update_with(ms(1));
        longest = longest.max(app.world.removal_count::<Hp>());
    }
    assert_eq!(app.world.get_resource::<Reaped>().unwrap().0, 150);
    assert_eq!(longest, 0, "every frame ends with the log empty");
}

#[test]
fn trimming_can_be_turned_off() {
    let mut app = app_with_reader();
    assert!(app.trims_removals());
    app.set_trim_removals(false);
    assert!(!app.trims_removals());
    kill(&mut app, 5);
    app.update_with(ms(1));
    assert_eq!(app.world.removal_count::<Hp>(), 5, "left for the owner");
    app.set_trim_removals(true);
    app.update_with(ms(1));
    assert_eq!(app.world.removal_count::<Hp>(), 0);
}

#[test]
fn nothing_is_trimmed_when_no_system_reads_removals() {
    let mut app = App::new();
    app.world.track_removals::<Hp>();
    app.add_systems(Update, |_world: &mut World| {});
    kill(&mut app, 4);
    app.update_with(ms(1));
    assert_eq!(app.world.removal_count::<Hp>(), 4);
}

#[test]
fn a_reader_in_a_stage_that_is_skipped_keeps_the_log_for_its_next_run() {
    #[derive(Default)]
    struct Gate(bool);
    let mut app = App::new();
    app.insert_resource(Reaped::default());
    app.insert_resource(Gate(false));
    app.add_systems(Physics, reap.run_if(|gate: Res<Gate>| gate.0));
    app.update_with(one_step());
    kill(&mut app, 2);
    app.update_with(one_step());
    app.update_with(one_step());
    assert_eq!(
        app.world.removal_count::<Hp>(),
        2,
        "the reader was skipped, so it still has them to see"
    );
    app.world.get_resource_mut::<Gate>().unwrap().0 = true;
    app.update_with(one_step());
    assert_eq!(app.world.get_resource::<Reaped>().unwrap().0, 2);
    assert_eq!(app.world.removal_count::<Hp>(), 0);
}

#[test]
fn a_reader_in_physics_sees_each_removal_once_however_many_steps_run() {
    let mut app = App::new();
    app.insert_resource(Reaped::default());
    app.add_systems(Physics, reap);
    app.update_with(one_step()); // set up
    for frame in 0..40u32 {
        kill(&mut app, 2);
        // frames of zero, one and three steps
        let delta = match frame % 3 {
            0 => Duration::ZERO,
            1 => one_step(),
            _ => one_step() * 3,
        };
        app.update_with(delta);
    }
    app.update_with(one_step());
    assert_eq!(app.world.get_resource::<Reaped>().unwrap().0, 80);
}

// ── messages ─────────────────────────────────────────────────────────

use mid_ecs::{MessageReader, MessageWriter};

struct Ping(u32);

#[derive(Default)]
struct Pings(Vec<u32>);

/// Writes 1, 2, 3, ... one per run.
fn send_ping(mut pings: MessageWriter<Ping>, mut n: ResMut<Count>) {
    n.0 += 1;
    pings.write(Ping(n.0));
}

fn take_pings(pings: MessageReader<Ping>, mut got: ResMut<Pings>) {
    got.0.extend(pings.read().map(|p| p.0));
}

fn got(app: &App) -> Vec<u32> {
    app.world.get_resource::<Pings>().unwrap().0.clone()
}

fn app_with_pings() -> App {
    let mut app = App::new();
    app.insert_resource(Count(0))
        .insert_resource(Pings::default());
    app
}

#[test]
fn a_message_written_in_one_stage_is_read_in_a_later_one_the_same_frame() {
    let mut app = app_with_pings();
    app.add_systems(PreUpdate, send_ping)
        .add_systems(Update, take_pings);
    app.update_with(ms(1));
    assert_eq!(got(&app), [1]);
    app.update_with(ms(1));
    assert_eq!(got(&app), [1, 2]);
}

#[test]
fn a_message_written_in_a_later_stage_is_read_by_an_earlier_one_next_frame() {
    let mut app = app_with_pings();
    app.add_systems(PostUpdate, send_ping)
        .add_systems(PreUpdate, take_pings);
    app.update_with(ms(1));
    assert_eq!(got(&app), Vec::<u32>::new());
    app.update_with(ms(1));
    assert_eq!(got(&app), [1]);
}

#[test]
fn a_reader_in_physics_sees_every_message_however_many_steps_the_frames_make() {
    let mut app = app_with_pings();
    app.add_systems(Update, send_ping)
        .add_systems(Physics, take_pings);
    app.update_with(one_step()); // set up
    let mut written = 1;
    for frame in 0..40u32 {
        let delta = match frame % 3 {
            0 => Duration::ZERO, // a frame that makes no step
            1 => one_step(),
            _ => one_step() * 3,
        };
        app.update_with(delta);
        written += 1;
    }
    app.update_with(one_step());
    written += 1;
    // Physics reads at most once a step, and a frame of no step reads
    // nothing: the messages of such frames arrive with the next one.
    assert_eq!(got(&app), (1..=written).collect::<Vec<u32>>());
}

#[test]
fn a_writer_in_physics_runs_once_per_step_and_update_reads_them_all_next_frame() {
    let mut app = app_with_pings();
    app.add_systems(Physics, send_ping)
        .add_systems(Update, take_pings);
    app.update_with(one_step() * 3);
    assert_eq!(got(&app), Vec::<u32>::new(), "Update ran before Physics");
    app.update_with(Duration::ZERO);
    assert_eq!(got(&app), [1, 2, 3], "all three steps' messages");
}

#[test]
fn the_message_log_is_trimmed_after_each_frame_and_nothing_is_missed() {
    let mut app = app_with_pings();
    app.add_systems(Update, (send_ping, take_pings).chain());
    app.update_with(ms(1));
    let mut longest = 0;
    for _ in 0..100 {
        app.update_with(ms(1));
        longest = longest.max(app.world.message_count::<Ping>());
    }
    assert_eq!(got(&app), (1..=101).collect::<Vec<u32>>());
    assert!(longest <= 1, "only the last frame's, was {longest}");
}

#[test]
fn a_type_with_no_reader_is_kept_a_frame_and_then_dropped() {
    let mut app = App::new();
    app.insert_resource(Count(0));
    app.add_systems(Update, send_ping);
    for _ in 0..50 {
        app.update_with(ms(1));
        assert!(app.world.message_count::<Ping>() <= 1);
    }
}

#[test]
fn message_trimming_can_be_turned_off() {
    let mut app = app_with_pings();
    app.add_systems(Update, (send_ping, take_pings).chain());
    assert!(app.trims_messages());
    app.set_trim_messages(false);
    assert!(!app.trims_messages());
    for _ in 0..10 {
        app.update_with(ms(1));
    }
    assert_eq!(app.world.message_count::<Ping>(), 10, "left for the owner");
    app.set_trim_messages(true);
    app.update_with(ms(1));
    app.update_with(ms(1));
    assert!(app.world.message_count::<Ping>() <= 1);
}

#[test]
fn a_reader_that_is_skipped_keeps_its_messages_for_when_it_runs() {
    #[derive(Default)]
    struct Gate(bool);
    let mut app = app_with_pings();
    app.insert_resource(Gate(false));
    app.add_systems(Update, send_ping);
    app.add_systems(Physics, take_pings.run_if(|gate: Res<Gate>| gate.0));
    for _ in 0..5 {
        app.update_with(one_step());
    }
    assert_eq!(
        app.world.message_count::<Ping>(),
        5,
        "all held for the reader"
    );
    app.world.get_resource_mut::<Gate>().unwrap().0 = true;
    app.update_with(one_step());
    assert_eq!(got(&app), [1, 2, 3, 4, 5, 6]);
}

#[test]
fn on_message_gates_a_stage_system_through_the_app() {
    use mid_ecs::schedule::common_conditions::on_message;
    let mut app = app_with_pings();
    app.add_systems(Update, count_runs_of_pings.run_if(on_message::<Ping>));
    app.update_with(ms(1));
    assert_eq!(app.world.get_resource::<Count>().unwrap().0, 0);
    app.world.write_message(Ping(1));
    app.update_with(ms(1));
    assert_eq!(app.world.get_resource::<Count>().unwrap().0, 1);
    app.update_with(ms(1));
    assert_eq!(
        app.world.get_resource::<Count>().unwrap().0,
        1,
        "nothing new"
    );
}

fn count_runs_of_pings(mut count: ResMut<Count>) {
    count.0 += 1;
}

// ── states ───────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
enum Mode {
    #[default]
    Menu,
    Play,
}

logger!(enter_menu);
logger!(enter_play);
logger!(exit_menu);

fn go_play(mut next: ResMut<NextState<Mode>>) {
    next.set(Mode::Play);
}

#[test]
fn the_transition_runs_between_pre_update_and_update() {
    let mut app = app_with_log();
    app.init_state::<Mode>();
    app.add_systems(PreUpdate, pre)
        .add_systems(Update, update)
        .add_systems(OnEnter(Mode::Menu), enter_menu);
    app.update_with(ms(1));
    assert_eq!(log(&app), ["pre", "enter_menu", "update"]);
    app.update_with(ms(1));
    assert_eq!(
        log(&app),
        ["pre", "enter_menu", "update", "pre", "update"],
        "OnEnter of the initial state runs once"
    );
}

#[test]
fn a_change_asked_for_in_pre_update_is_in_force_the_same_frame() {
    let mut app = app_with_log();
    app.init_state::<Mode>();
    app.add_systems(PreUpdate, go_play.run_if(in_state(Mode::Menu)))
        .add_systems(OnExit(Mode::Menu), exit_menu)
        .add_systems(OnEnter(Mode::Play), enter_play)
        .add_systems(Update, update.run_if(in_state(Mode::Play)));
    app.update_with(ms(1));
    assert!(log(&app).is_empty(), "frame 1 only makes the state: Menu");
    app.update_with(ms(1));
    assert_eq!(
        log(&app),
        ["exit_menu", "enter_play", "update"],
        "exit and enter, then Update already sees Play"
    );
    app.update_with(ms(1));
    assert_eq!(
        log(&app),
        ["exit_menu", "enter_play", "update", "update"],
        "and no second exit or enter"
    );
}

#[test]
fn a_change_asked_for_in_update_is_in_force_next_frame() {
    let mut app = app_with_log();
    app.init_state::<Mode>();
    app.add_systems(Update, go_play)
        .add_systems(OnEnter(Mode::Play), enter_play);
    app.update_with(ms(1));
    assert!(log(&app).is_empty());
    app.update_with(ms(1));
    assert_eq!(log(&app), ["enter_play"]);
}

#[test]
fn the_state_exists_from_the_first_frame_and_is_readable() {
    let mut app = App::new();
    app.insert_state(Mode::Play);
    assert!(app.world.get_resource::<State<Mode>>().is_none());
    app.update_with(ms(1));
    assert_eq!(
        app.world.get_resource::<State<Mode>>().unwrap().get(),
        &Mode::Play
    );
}

#[test]
fn init_state_twice_is_one_state_and_insert_state_asks_for_a_change() {
    let mut app = app_with_log();
    app.init_state::<Mode>().init_state::<Mode>();
    app.add_systems(OnEnter(Mode::Menu), enter_menu)
        .add_systems(OnEnter(Mode::Play), enter_play);
    app.update_with(ms(1));
    app.insert_state(Mode::Play);
    app.update_with(ms(1));
    assert_eq!(log(&app), ["enter_menu", "enter_play"], "each once");
}

#[test]
fn a_state_gates_physics_and_reads_state_changes_as_messages() {
    #[derive(Default)]
    struct Seen(Vec<bool>);
    fn watch(m: MessageReader<StateTransitionMessage<Mode>>, mut seen: ResMut<Seen>) {
        seen.0.extend(m.read().map(|m| m.is_identity()));
    }
    let mut app = app_with_log();
    app.insert_resource(Seen::default());
    app.init_state::<Mode>();
    app.add_systems(Physics, physics.run_if(in_state(Mode::Play)))
        .add_systems(PostUpdate, watch)
        .add_systems(PreUpdate, go_play.run_if(in_state(Mode::Menu)));
    app.update_with(one_step() * 2);
    assert!(log(&app).is_empty(), "frame 1 only makes the state: Menu");
    app.update_with(one_step() * 2);
    assert_eq!(log(&app), ["physics", "physics"], "Play from this frame on");
    assert_eq!(
        app.world.get_resource::<Seen>().unwrap().0,
        [false, false],
        "the state's making and the change, neither an identity"
    );
}

// ── observers ────────────────────────────────────────────────────────

struct Boom(u32);
impl Event for Boom {}

struct Marker;

fn log_boom(on: On<Boom>, mut log: ResMut<Log>) {
    log.0.push(if on.0 == 1 { "boom 1" } else { "boom" });
}

fn fire(mut commands: Commands) {
    commands.trigger(Boom(1));
}

#[test]
fn a_system_triggers_an_event_and_its_observer_runs_at_once() {
    let mut app = app_with_log();
    app.add_observer(log_boom)
        .add_systems(Update, (fire, update).chain());
    app.update_with(ms(1));
    assert_eq!(
        log(&app),
        ["boom 1", "update"],
        "the observer ran when fire's commands applied, before update"
    );
}

#[test]
fn a_lifecycle_observer_added_to_the_app_sees_spawns_made_by_systems() {
    fn made(_on: On<Add<Marker>>, mut log: ResMut<Log>) {
        log.0.push("made");
    }
    fn spawn_marker(mut commands: Commands) {
        commands.spawn().queue(|w, e| {
            w.insert_static(e, Marker);
        });
    }
    let mut app = app_with_log();
    app.add_observer(made).add_systems(Update, spawn_marker);
    app.update_with(ms(1));
    app.update_with(ms(1));
    assert_eq!(log(&app), ["made", "made"]);
}
