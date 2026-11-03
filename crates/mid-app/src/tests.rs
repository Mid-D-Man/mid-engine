// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-app.md, section "tests.rs"
// ============================================================================
//! The app end to end: stage order, the fixed timestep, plugins, the time
//! resources, and trimming the removal log.

use std::time::Duration;

use mid_ecs::schedule::{ExecutorKind, IntoScheduleConfigs};
use mid_ecs::{RemovedComponents, Res, ResMut, World};
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
