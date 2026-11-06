// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-app.md, section "app.rs"
// ============================================================================
//! [`App`], [`Plugin`] and the frame driver.

use std::time::Duration;

use mid_ecs::schedule::{IntoScheduleConfigs, Schedule, ScheduleLabel, ScheduleSystem, SetKey};
use mid_ecs::state::{StateTransition, States};
use mid_ecs::system::IntoSystem;
use mid_ecs::World;
use mid_time::{Clock, FixedTimestep};

use crate::labels::{Main, Net, Physics, PostUpdate, PreUpdate, Update};

/// The fixed timestep's rate until [`App::set_fixed_timestep`] says
/// otherwise: the physics budget in `docs/architecture.md`.
pub const DEFAULT_FIXED_HZ: f32 = 60.0;

/// The most [`Physics`] runs one frame can make until
/// [`App::set_fixed_timestep`] says otherwise. After a stall longer than
/// this many steps, the rest of the time is dropped rather than caught up
/// (`FixedTimestep`'s spiral-of-death guard).
pub const DEFAULT_MAX_FIXED_STEPS: u32 = 8;

/// Something that adds systems, sets and resources to an [`App`]. Every
/// engine feature is expected to ship one, so its users do not have to
/// know what it consists of.
pub trait Plugin {
    /// Called by [`App::add_plugin`], at once.
    fn build(&self, app: &mut App);
}

/// A world and the schedules that run it.
///
/// A new `App` has a schedule for each of the stages ([`PreUpdate`],
/// [`StateTransition`], [`Update`], [`Net`], [`Physics`], [`PostUpdate`]) and for [`Main`], the
/// one that runs them, and holds a `mid_time::Clock` and a
/// `mid_time::FixedTimestep` as resources, so a system can read the
/// frame's time as `Res<Clock>` and the step's length as
/// `Res<FixedTimestep>`.
pub struct App {
    /// The world the systems run on.
    pub world: World,
    trim_removals: bool,
    trim_messages: bool,
}

impl App {
    /// An app with its schedules and its time resources, and nothing else.
    pub fn new() -> Self {
        let mut world = World::new();
        world.insert_resource(Clock::new());
        world.insert_resource(FixedTimestep::new(
            DEFAULT_FIXED_HZ,
            DEFAULT_MAX_FIXED_STEPS,
        ));
        world.init_schedule(PreUpdate);
        world.init_schedule(StateTransition);
        world.init_schedule(Update);
        world.init_schedule(Net);
        world.init_schedule(Physics);
        world.init_schedule(PostUpdate);
        world.init_schedule(Main).add_systems(run_stages);
        Self {
            world,
            trim_removals: true,
            trim_messages: true,
        }
    }

    /// Adds systems to the schedule called `label`, making it if there is
    /// none: a system or a tuple of them, with ordering and conditions
    /// attached (see `mid_ecs::schedule::IntoScheduleConfigs`).
    ///
    /// # Panics
    /// If that schedule is being run right now.
    pub fn add_systems<M>(
        &mut self,
        label: impl ScheduleLabel,
        systems: impl IntoScheduleConfigs<ScheduleSystem, M>,
    ) -> &mut Self {
        self.world.init_schedule(label).add_systems(systems);
        self
    }

    /// Declares sets in the schedule called `label`, making it if there is
    /// none (see `Schedule::configure_sets`).
    ///
    /// # Panics
    /// If that schedule is being run right now.
    pub fn configure_sets<M>(
        &mut self,
        label: impl ScheduleLabel,
        sets: impl IntoScheduleConfigs<SetKey, M>,
    ) -> &mut Self {
        self.world.init_schedule(label).configure_sets(sets);
        self
    }

    /// The schedule called `label`, made empty first if there is none, for
    /// the settings that belong to a schedule (`set_executor`,
    /// `set_ambiguity_detection`).
    ///
    /// # Panics
    /// If that schedule is being run right now.
    pub fn schedule_mut(&mut self, label: impl ScheduleLabel) -> &mut Schedule {
        self.world.init_schedule(label)
    }

    /// Sets a resource, replacing the one of that type if there is one.
    pub fn insert_resource<T: 'static>(&mut self, value: T) -> &mut Self {
        self.world.insert_resource(value);
        self
    }

    /// Adds an observer: a system with an `On<E>` parameter, run each time an
    /// `E` is triggered (`World::add_observer`; `docs/mid-ecs-events.md`).
    pub fn add_observer<M, S: IntoSystem<M>>(&mut self, system: S) -> &mut Self {
        self.world.add_observer(system);
        self
    }

    /// Gives the app a state of type `S`, `S::default()` to begin with: the
    /// world has `State<S>` from the first frame (the transition runs
    /// between `PreUpdate` and `Update`, so `OnEnter` of the initial value
    /// and `in_state` are right in that frame). Does nothing if the app
    /// has that state type already.
    pub fn init_state<S: States + Default>(&mut self) -> &mut Self {
        self.world.init_state::<S>();
        self
    }

    /// Gives the app a state of type `S`, `initial` to begin with; if it has
    /// the type already, asks for `initial` as `NextState::set` would.
    pub fn insert_state<S: States>(&mut self, initial: S) -> &mut Self {
        self.world.insert_state(initial);
        self
    }

    /// Calls `plugin.build(self)` now, so what the plugin adds is there as
    /// soon as this returns.
    pub fn add_plugin(&mut self, plugin: impl Plugin) -> &mut Self {
        plugin.build(self);
        self
    }

    /// Replaces the fixed timestep: `hz` steps a second, and at most
    /// `max_steps_per_frame` of them in one frame. Time already
    /// accumulated is dropped.
    pub fn set_fixed_timestep(&mut self, hz: f32, max_steps_per_frame: u32) -> &mut Self {
        self.world
            .insert_resource(FixedTimestep::new(hz, max_steps_per_frame));
        self
    }

    /// Whether the removal log is trimmed after every frame (the default).
    /// See [`Self::set_trim_removals`].
    pub fn trims_removals(&self) -> bool {
        self.trim_removals
    }

    /// Turns the trimming of the removal log after each frame on or off.
    ///
    /// On, the log is cut back after each frame to what some system that
    /// reads removals still has to see (`World::trim_removals_for_schedules`),
    /// so a type that is removed often does not grow it without bound.
    /// Turn it off to trim yourself, which you need to when something
    /// outside the schedules (a `ChangeTracker`) also reads the log.
    pub fn set_trim_removals(&mut self, on: bool) -> &mut Self {
        self.trim_removals = on;
        self
    }

    /// Whether the message logs are trimmed after every frame (the
    /// default). See [`Self::set_trim_messages`].
    pub fn trims_messages(&self) -> bool {
        self.trim_messages
    }

    /// Turns the trimming of the message logs after each frame on or off.
    ///
    /// On, each log is cut back after each frame to what the systems that
    /// read that type still have to see, and never to less than the last
    /// frame's worth (`World::trim_messages_for_schedules`), so a type
    /// written often does not grow without bound and a message is kept for
    /// at least a whole frame whoever reads it. Turn it off to trim
    /// yourself.
    pub fn set_trim_messages(&mut self, on: bool) -> &mut Self {
        self.trim_messages = on;
        self
    }

    /// Runs one frame, with the time the platform clock says has passed
    /// since the last one (the first frame's is nearly zero).
    pub fn update(&mut self) {
        self.clock_mut().tick();
        self.run_frame();
    }

    /// Runs one frame as if `delta` had passed, without reading the
    /// platform clock: for tests, replay and lockstep.
    pub fn update_with(&mut self, delta: Duration) {
        self.clock_mut().tick_with(delta);
        self.run_frame();
    }

    /// Runs frames, one [`Self::update`] after another, until `done` says
    /// to stop (it is asked after each frame). It does not sleep between
    /// frames, so a loop with nothing to wait for uses a whole core:
    /// sleep in `done`, or drive [`Self::update`] yourself.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn run_until(&mut self, mut done: impl FnMut(&mut App) -> bool) {
        loop {
            self.update();
            if done(self) {
                break;
            }
        }
    }

    fn clock_mut(&mut self) -> &mut Clock {
        self.world
            .get_resource_mut::<Clock>()
            .expect("the `Clock` resource was removed from the app's world")
    }

    fn run_frame(&mut self) {
        self.world.run_schedule(Main);
        if self.trim_removals {
            self.world.trim_removals_for_schedules();
        }
        if self.trim_messages {
            self.world.trim_messages_for_schedules();
        }
    }
}

impl Default for App {
    fn default() -> Self {
        Self::new()
    }
}

/// The one system of [`Main`]: runs the stages in order (with
/// [`StateTransition`] between [`PreUpdate`] and [`Update`]), [`Physics`] as
/// many times as the frame's time has made steps ready.
fn run_stages(world: &mut World) {
    world.run_schedule(PreUpdate);
    world.run_schedule(StateTransition);
    world.run_schedule(Update);
    world.run_schedule(Net);
    let delta = world
        .get_resource::<Clock>()
        .expect("the `Clock` resource was removed from the app's world")
        .delta();
    let steps = world
        .get_resource_mut::<FixedTimestep>()
        .expect("the `FixedTimestep` resource was removed from the app's world")
        .accumulate(delta);
    for _ in 0..steps {
        world.run_schedule(Physics);
    }
    world.run_schedule(PostUpdate);
}
