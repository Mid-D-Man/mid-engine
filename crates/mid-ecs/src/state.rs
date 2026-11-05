// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-ecs-events.md, section "state.rs"
// ============================================================================
//! States: a value the whole world is "in", with schedules that run when it
//! changes.
//!
//! Any type that is `Debug + Clone + Eq + Hash + Send + Sync + 'static` is
//! a [`States`], usually an enum. A world holds one [`State<S>`] per state
//! type; systems read it, and ask for a change through [`NextState<S>`].
//! The change is applied by [`StateTransition`], a schedule the frame owner
//! runs once a frame (`mid-app` runs it between `PreUpdate` and `Update`):
//! it runs [`OnExit`] of the state left, [`OnTransition`], and [`OnEnter`]
//! of the state entered, each a schedule of its own that only exists if
//! something was added to it.
//!
//! ```
//! use mid_ecs::schedule::{IntoScheduleConfigs, Schedule};
//! use mid_ecs::state::{in_state, OnEnter, StateTransition};
//! use mid_ecs::{NextState, ResMut, World};
//!
//! #[derive(Debug, Clone, PartialEq, Eq, Hash)]
//! enum Mode {
//!     Menu,
//!     Play,
//! }
//!
//! struct Score(u32);
//!
//! fn start(mut score: ResMut<Score>) {
//!     score.0 = 0;
//! }
//!
//! fn tick(mut score: ResMut<Score>) {
//!     score.0 += 1;
//! }
//!
//! let mut world = World::new();
//! world.insert_resource(Score(99));
//! world.insert_state(Mode::Menu);
//! world.init_schedule(OnEnter(Mode::Play)).add_systems(start);
//! let mut update = Schedule::new("update");
//! update.add_systems(tick.run_if(in_state(Mode::Play)));
//!
//! world.run_schedule(StateTransition); // the state comes into being: Menu
//! update.run(&mut world);
//! assert_eq!(world.get_resource::<Score>().unwrap().0, 99, "not in Play");
//!
//! world.get_resource_mut::<NextState<Mode>>().unwrap().set(Mode::Play);
//! world.run_schedule(StateTransition); // OnEnter(Play) runs `start`
//! update.run(&mut world);
//! assert_eq!(world.get_resource::<Score>().unwrap().0, 1);
//! ```
//!
//! # What is the same as `bevy_state`, and what is not
//!
//! The same: the state does not exist until the first [`StateTransition`]
//! (so `in_state` is false before it and `OnEnter` of the initial value
//! runs in that first frame); exit, then transition, then enter; setting
//! the state it is already in runs exit and enter again ([`NextState::set`]),
//! unless asked not to ([`NextState::set_if_neq`]); the change is also a
//! message.
//!
//! Not built: `SubStates`, `ComputedStates`, `StateScoped` entities, a
//! `PreviousState` resource (the message has the state that was left), and
//! a state that goes away. Only one transition is applied per run of
//! [`StateTransition`]: a change asked for from inside an `OnEnter` waits
//! for the next run.

use std::fmt::Debug;
use std::hash::Hash;
use std::mem;
use std::ops::Deref;

use crate::schedule::{ScheduleLabel, ScheduleRunError};
use crate::system_param::Res;
use crate::world::World;

/// A value the world can be "in". Implemented for every type that is
/// `Debug + Clone + Eq + Hash + Send + Sync + 'static`, so an enum needs
/// only its derives.
pub trait States: Debug + Clone + PartialEq + Eq + Hash + Send + Sync + 'static {}

impl<T: Debug + Clone + PartialEq + Eq + Hash + Send + Sync + 'static> States for T {}

/// The state the world is in: a resource. It does not exist until the
/// first [`StateTransition`] after [`World::insert_state`], and it is
/// changed only by that schedule.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct State<S: States>(S);

impl<S: States> State<S> {
    /// The state.
    pub fn get(&self) -> &S {
        &self.0
    }
}

impl<S: States> Deref for State<S> {
    type Target = S;

    fn deref(&self) -> &S {
        &self.0
    }
}

/// What the state is to become at the next [`StateTransition`]: a resource
/// that systems set. Applying it resets it to `Unchanged`.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum NextState<S: States> {
    /// No change asked for.
    #[default]
    Unchanged,
    /// Become this state, running exit and enter even if it is the one the
    /// world is in.
    Pending(S),
    /// Become this state unless the world is in it already.
    PendingIfNeq(S),
}

impl<S: States> NextState<S> {
    /// Asks to become `state`. If the world is in it already, its exit and
    /// enter schedules run anyway.
    pub fn set(&mut self, state: S) {
        *self = NextState::Pending(state);
    }

    /// Asks to become `state`, and does nothing if the world is in it
    /// already when the change is applied. A `set` of the same state
    /// already asked for is kept as it is.
    pub fn set_if_neq(&mut self, state: S) {
        if !matches!(self, NextState::Pending(pending) if *pending == state) {
            *self = NextState::PendingIfNeq(state);
        }
    }

    /// Withdraws a change that was asked for and not applied yet.
    pub fn reset(&mut self) {
        *self = NextState::Unchanged;
    }
}

/// Runs the transition of every state type: put into [`World::run_schedule`]
/// once a frame, before the systems that depend on the state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct StateTransition;

/// The schedule run when the world enters the state: after the state has
/// changed, so `Res<State<S>>` is the new one.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct OnEnter<S: States>(pub S);

/// The schedule run when the world leaves the state: after the state has
/// changed, so `Res<State<S>>` is already the new one.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct OnExit<S: States>(pub S);

/// The schedule run for one particular change, between the exit of the
/// state left and the enter of the one entered. Not run for the first
/// transition, which leaves nothing.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct OnTransition<S: States> {
    /// The state left.
    pub exited: S,
    /// The state entered.
    pub entered: S,
}

/// Written (as a message) at each transition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StateTransitionMessage<S: States> {
    /// The state left: `None` for the first transition, which makes the
    /// state.
    pub exited: Option<S>,
    /// The state entered.
    pub entered: S,
}

impl<S: States> StateTransitionMessage<S> {
    /// Whether the world was in `entered` already.
    pub fn is_identity(&self) -> bool {
        self.exited.as_ref() == Some(&self.entered)
    }
}

impl World {
    /// Gives the world a state of type `S`, to become `initial` at the next
    /// [`StateTransition`]: inserts [`NextState`] asking for it, makes the
    /// transition's message type, and adds the system that applies changes
    /// of `S` to [`StateTransition`] (once per type).
    ///
    /// If the world has the state type already, this only asks for
    /// `initial` as [`NextState::set`] would.
    ///
    /// # Panics
    /// If it adds the system and [`StateTransition`] is being run.
    pub fn insert_state<S: States>(&mut self, initial: S) {
        if let Some(next) = self.get_resource_mut::<NextState<S>>() {
            next.set(initial);
            return;
        }
        self.insert_resource(NextState::Pending(initial));
        self.init_message::<StateTransitionMessage<S>>();
        self.init_schedule(StateTransition)
            .add_systems(apply_state_transition::<S>);
    }

    /// [`Self::insert_state`] with `S::default()`, unless the world has the
    /// state type already, in which case it does nothing.
    pub fn init_state<S: States + Default>(&mut self) {
        if !self.contains_resource::<NextState<S>>() {
            self.insert_state(S::default());
        }
    }
}

/// Applies the change of state `S` that [`NextState<S>`] asks for, if any:
/// the system [`World::insert_state`] adds to [`StateTransition`].
///
/// Changes [`State<S>`] (making it the first time), writes the
/// [`StateTransitionMessage`], then runs [`OnExit`] of the state left,
/// [`OnTransition`] and [`OnEnter`] of the state entered, skipping the
/// schedules nothing was added to.
pub fn apply_state_transition<S: States>(world: &mut World) {
    let Some(next) = world.get_resource_mut::<NextState<S>>() else {
        return;
    };
    let entered = match mem::take(next) {
        NextState::Unchanged => return,
        NextState::Pending(state) => state,
        NextState::PendingIfNeq(state) => {
            if world
                .get_resource::<State<S>>()
                .is_some_and(|current| current.0 == state)
            {
                return;
            }
            state
        }
    };
    let exited = match world.get_resource_mut::<State<S>>() {
        Some(current) => Some(mem::replace(&mut current.0, entered.clone())),
        None => {
            world.insert_resource(State(entered.clone()));
            None
        }
    };
    world.write_message(StateTransitionMessage {
        exited: exited.clone(),
        entered: entered.clone(),
    });
    if let Some(exited) = exited {
        run_if_there(world, OnExit(exited.clone()));
        run_if_there(
            world,
            OnTransition {
                exited,
                entered: entered.clone(),
            },
        );
    }
    run_if_there(world, OnEnter(entered));
}

/// Runs the schedule if there is one; a state nothing reacts to is fine.
fn run_if_there(world: &mut World, label: impl ScheduleLabel) {
    match world.try_run_schedule(label) {
        Ok(()) | Err(ScheduleRunError::NotFound(_)) => {}
        Err(error) => panic!("{error}"),
    }
}

// ── run conditions ───────────────────────────────────────────────────

/// A run condition: the world is in `state`. False before the state exists.
pub fn in_state<S: States>(state: S) -> impl FnMut(Option<Res<State<S>>>) -> bool + Clone {
    move |current| current.is_some_and(|current| current.0 == state)
}

/// A run condition: the state changed since the condition last ran, which
/// includes coming into being, and a transition to the state it was in.
pub fn state_changed<S: States>(current: Option<Res<State<S>>>) -> bool {
    current.is_some_and(|current| current.is_changed())
}

/// A run condition: the world has a state of type `S`.
pub fn state_exists<S: States>(current: Option<Res<State<S>>>) -> bool {
    current.is_some()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::MessageReader;
    use crate::schedule::common_conditions::{and, not};
    use crate::schedule::{IntoScheduleConfigs, Schedule};
    use crate::system_param::ResMut;

    #[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
    enum Mode {
        #[default]
        Menu,
        Play,
        Over,
    }

    #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    enum Light {
        Red,
        Green,
    }

    #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    struct Level(u32);

    #[derive(Default)]
    struct Log(Vec<String>);

    fn logged(w: &World) -> Vec<String> {
        w.get_resource::<Log>().unwrap().0.clone()
    }

    macro_rules! log_system {
        ($name:ident, $text:expr) => {
            fn $name(mut log: ResMut<Log>) {
                log.0.push($text.to_string());
            }
        };
    }
    log_system!(enter_menu, "enter menu");
    log_system!(exit_menu, "exit menu");
    log_system!(enter_play, "enter play");
    log_system!(exit_play, "exit play");
    log_system!(enter_over, "enter over");
    log_system!(menu_to_play, "menu to play");
    log_system!(enter_red, "enter red");
    log_system!(enter_green, "enter green");
    log_system!(update_in_play, "update in play");

    fn world() -> World {
        let mut w = World::new();
        w.insert_resource(Log::default());
        w
    }

    fn mode(w: &World) -> Option<Mode> {
        w.get_resource::<State<Mode>>().map(|s| s.get().clone())
    }

    fn set(w: &mut World, mode: Mode) {
        w.get_resource_mut::<NextState<Mode>>().unwrap().set(mode);
    }

    fn transition(w: &mut World) {
        w.run_schedule(StateTransition);
    }

    fn messages(w: &World) -> Vec<StateTransitionMessage<Mode>> {
        w.messages_since::<StateTransitionMessage<Mode>>(&crate::tick::ChangeTracker::new())
            .cloned()
            .collect()
    }

    // ── NextState ────────────────────────────────────────────────────

    #[test]
    fn next_state_set_set_if_neq_and_reset() {
        let mut n = NextState::<Mode>::default();
        assert_eq!(n, NextState::Unchanged);
        n.set(Mode::Play);
        assert_eq!(n, NextState::Pending(Mode::Play));
        n.set_if_neq(Mode::Over);
        assert_eq!(n, NextState::PendingIfNeq(Mode::Over));
        n.reset();
        assert_eq!(n, NextState::Unchanged);
    }

    #[test]
    fn set_if_neq_keeps_a_set_of_the_same_state() {
        let mut n = NextState::Pending(Mode::Play);
        n.set_if_neq(Mode::Play);
        assert_eq!(n, NextState::Pending(Mode::Play), "stays a set");
        n.set_if_neq(Mode::Over);
        assert_eq!(n, NextState::PendingIfNeq(Mode::Over));
    }

    #[test]
    fn every_hashable_type_is_a_state() {
        fn is_state<S: States>() {}
        is_state::<Mode>();
        is_state::<u32>();
        is_state::<(bool, Light)>();
    }

    #[test]
    fn state_derefs_to_its_value() {
        let s = State(Mode::Play);
        assert_eq!(*s, Mode::Play);
        assert_eq!(s.get(), &Mode::Play);
    }

    // ── the first transition ─────────────────────────────────────────

    #[test]
    fn the_state_does_not_exist_until_the_first_transition_runs() {
        let mut w = world();
        w.insert_state(Mode::Menu);
        assert_eq!(mode(&w), None);
        assert_eq!(
            w.get_resource::<NextState<Mode>>(),
            Some(&NextState::Pending(Mode::Menu))
        );
        transition(&mut w);
        assert_eq!(mode(&w), Some(Mode::Menu));
    }

    #[test]
    fn the_first_transition_enters_the_initial_state_and_leaves_nothing() {
        let mut w = world();
        w.init_schedule(OnEnter(Mode::Menu)).add_systems(enter_menu);
        w.init_schedule(OnExit(Mode::Menu)).add_systems(exit_menu);
        w.insert_state(Mode::Menu);
        transition(&mut w);
        assert_eq!(logged(&w), ["enter menu"]);
        assert_eq!(
            messages(&w),
            [StateTransitionMessage {
                exited: None,
                entered: Mode::Menu
            }]
        );
        transition(&mut w);
        assert_eq!(logged(&w), ["enter menu"], "and only once");
    }

    #[test]
    fn init_state_takes_the_default_and_does_nothing_twice() {
        let mut w = world();
        w.init_schedule(OnEnter(Mode::Menu)).add_systems(enter_menu);
        w.init_state::<Mode>();
        transition(&mut w);
        assert_eq!(mode(&w), Some(Mode::Menu));
        set(&mut w, Mode::Play);
        w.init_state::<Mode>(); // already there: no request of its own
        transition(&mut w);
        assert_eq!(mode(&w), Some(Mode::Play), "the earlier set stood");
        assert_eq!(logged(&w), ["enter menu"]);
    }

    // ── changes ──────────────────────────────────────────────────────

    #[test]
    fn a_change_runs_exit_then_transition_then_enter() {
        let mut w = world();
        w.init_schedule(OnExit(Mode::Menu)).add_systems(exit_menu);
        w.init_schedule(OnTransition {
            exited: Mode::Menu,
            entered: Mode::Play,
        })
        .add_systems(menu_to_play);
        w.init_schedule(OnEnter(Mode::Play)).add_systems(enter_play);
        w.insert_state(Mode::Menu);
        transition(&mut w);
        set(&mut w, Mode::Play);
        transition(&mut w);
        assert_eq!(logged(&w), ["exit menu", "menu to play", "enter play"]);
        assert_eq!(mode(&w), Some(Mode::Play));
        assert_eq!(
            messages(&w).last().unwrap(),
            &StateTransitionMessage {
                exited: Some(Mode::Menu),
                entered: Mode::Play
            }
        );
        assert!(!messages(&w).last().unwrap().is_identity());
    }

    #[test]
    fn enter_and_exit_schedules_see_the_new_state() {
        fn read(current: Res<State<Mode>>, mut log: ResMut<Log>) {
            log.0.push(format!("{:?}", current.get()));
        }
        let mut w = world();
        w.init_schedule(OnExit(Mode::Menu)).add_systems(read);
        w.init_schedule(OnEnter(Mode::Play)).add_systems(read);
        w.insert_state(Mode::Menu);
        transition(&mut w);
        set(&mut w, Mode::Play);
        transition(&mut w);
        assert_eq!(logged(&w), ["Play", "Play"]);
    }

    #[test]
    fn a_change_is_applied_once() {
        let mut w = world();
        w.init_schedule(OnEnter(Mode::Play)).add_systems(enter_play);
        w.insert_state(Mode::Menu);
        transition(&mut w);
        set(&mut w, Mode::Play);
        transition(&mut w);
        transition(&mut w);
        transition(&mut w);
        assert_eq!(logged(&w), ["enter play"]);
        assert_eq!(
            w.get_resource::<NextState<Mode>>(),
            Some(&NextState::Unchanged)
        );
    }

    #[test]
    fn setting_the_state_it_is_in_runs_exit_and_enter_again() {
        let mut w = world();
        w.init_schedule(OnEnter(Mode::Menu)).add_systems(enter_menu);
        w.init_schedule(OnExit(Mode::Menu)).add_systems(exit_menu);
        w.insert_state(Mode::Menu);
        transition(&mut w);
        set(&mut w, Mode::Menu);
        transition(&mut w);
        assert_eq!(logged(&w), ["enter menu", "exit menu", "enter menu"]);
        assert!(messages(&w).last().unwrap().is_identity());
        assert!(!messages(&w)[0].is_identity());
    }

    #[test]
    fn set_if_neq_ignores_the_state_it_is_in_and_applies_another() {
        let mut w = world();
        w.init_schedule(OnEnter(Mode::Menu)).add_systems(enter_menu);
        w.init_schedule(OnEnter(Mode::Play)).add_systems(enter_play);
        w.insert_state(Mode::Menu);
        transition(&mut w);
        w.get_resource_mut::<NextState<Mode>>()
            .unwrap()
            .set_if_neq(Mode::Menu);
        transition(&mut w);
        assert_eq!(logged(&w), ["enter menu"], "nothing ran");
        assert_eq!(messages(&w).len(), 1, "and no message");
        w.get_resource_mut::<NextState<Mode>>()
            .unwrap()
            .set_if_neq(Mode::Play);
        transition(&mut w);
        assert_eq!(logged(&w), ["enter menu", "enter play"]);
    }

    #[test]
    fn a_withdrawn_change_is_not_applied() {
        let mut w = world();
        w.init_schedule(OnEnter(Mode::Play)).add_systems(enter_play);
        w.insert_state(Mode::Menu);
        transition(&mut w);
        set(&mut w, Mode::Play);
        w.get_resource_mut::<NextState<Mode>>().unwrap().reset();
        transition(&mut w);
        assert_eq!(mode(&w), Some(Mode::Menu));
        assert!(logged(&w).is_empty());
    }

    #[test]
    fn a_change_asked_for_during_enter_waits_for_the_next_run() {
        fn chain(mut next: ResMut<NextState<Mode>>) {
            next.set(Mode::Over);
        }
        let mut w = world();
        w.init_schedule(OnEnter(Mode::Play)).add_systems(chain);
        w.init_schedule(OnEnter(Mode::Over)).add_systems(enter_over);
        w.insert_state(Mode::Menu);
        transition(&mut w);
        set(&mut w, Mode::Play);
        transition(&mut w);
        assert_eq!(mode(&w), Some(Mode::Play), "one change per run");
        assert!(logged(&w).is_empty());
        transition(&mut w);
        assert_eq!(mode(&w), Some(Mode::Over));
        assert_eq!(logged(&w), ["enter over"]);
    }

    #[test]
    fn a_state_nothing_reacts_to_changes_without_schedules() {
        let mut w = world();
        w.insert_state(Mode::Menu);
        transition(&mut w);
        set(&mut w, Mode::Over);
        transition(&mut w);
        assert_eq!(mode(&w), Some(Mode::Over));
    }

    #[test]
    fn a_state_can_carry_data() {
        let mut w = world();
        fn log_level(current: Res<State<Level>>, mut log: ResMut<Log>) {
            log.0.push(format!("level {}", current.0 .0));
        }
        w.init_schedule(OnEnter(Level(2))).add_systems(log_level);
        w.insert_state(Level(1));
        w.run_schedule(StateTransition);
        w.get_resource_mut::<NextState<Level>>()
            .unwrap()
            .set(Level(2));
        w.run_schedule(StateTransition);
        w.get_resource_mut::<NextState<Level>>()
            .unwrap()
            .set(Level(3));
        w.run_schedule(StateTransition);
        assert_eq!(logged(&w), ["level 2"], "only Level(2) has a schedule");
    }

    #[test]
    fn two_state_types_change_independently() {
        let mut w = world();
        w.init_schedule(OnEnter(Mode::Play)).add_systems(enter_play);
        w.init_schedule(OnEnter(Light::Green))
            .add_systems(enter_green);
        w.init_schedule(OnEnter(Light::Red)).add_systems(enter_red);
        w.insert_state(Mode::Menu);
        w.insert_state(Light::Red);
        transition(&mut w);
        w.get_resource_mut::<NextState<Light>>()
            .unwrap()
            .set(Light::Green);
        transition(&mut w);
        set(&mut w, Mode::Play);
        transition(&mut w);
        assert_eq!(logged(&w), ["enter red", "enter green", "enter play"]);
        assert_eq!(
            w.get_resource::<State<Light>>().unwrap().get(),
            &Light::Green
        );
        assert_eq!(mode(&w), Some(Mode::Play));
    }

    #[test]
    fn inserting_a_state_again_asks_for_it_without_a_second_system() {
        let mut w = world();
        w.init_schedule(OnEnter(Mode::Menu)).add_systems(enter_menu);
        w.init_schedule(OnEnter(Mode::Play)).add_systems(enter_play);
        w.insert_state(Mode::Menu);
        w.insert_state(Mode::Play); // before the first transition: replaces the ask
        transition(&mut w);
        assert_eq!(logged(&w), ["enter play"], "once, and Play, not Menu");
        w.insert_state(Mode::Menu);
        transition(&mut w);
        assert_eq!(logged(&w), ["enter play", "enter menu"], "applied once");
    }

    // ── messages and conditions ──────────────────────────────────────

    #[test]
    fn a_system_reads_the_transition_as_a_message() {
        fn watch(reader: MessageReader<StateTransitionMessage<Mode>>, mut log: ResMut<Log>) {
            for m in reader.read() {
                log.0.push(format!("{:?} -> {:?}", m.exited, m.entered));
            }
        }
        let mut w = world();
        w.insert_state(Mode::Menu);
        let mut s = Schedule::new("watch");
        s.add_systems(watch);
        transition(&mut w);
        s.run(&mut w);
        set(&mut w, Mode::Play);
        transition(&mut w);
        s.run(&mut w);
        s.run(&mut w);
        assert_eq!(logged(&w), ["None -> Menu", "Some(Menu) -> Play"]);
    }

    #[test]
    fn in_state_runs_a_system_only_in_that_state() {
        let mut w = world();
        w.insert_state(Mode::Menu);
        let mut s = Schedule::new("update");
        s.add_systems(update_in_play.run_if(in_state(Mode::Play)));
        s.run(&mut w); // the state does not exist yet
        transition(&mut w);
        s.run(&mut w);
        set(&mut w, Mode::Play);
        transition(&mut w);
        s.run(&mut w);
        s.run(&mut w);
        set(&mut w, Mode::Over);
        transition(&mut w);
        s.run(&mut w);
        assert_eq!(logged(&w), ["update in play", "update in play"]);
    }

    #[test]
    fn in_state_combines_with_not_and_and() {
        let mut w = world();
        w.insert_state(Mode::Play);
        w.insert_state(Light::Green);
        let mut s = Schedule::new("update");
        s.add_systems((
            enter_menu.run_if(not(in_state(Mode::Play))),
            enter_play.run_if(and(in_state(Mode::Play), in_state(Light::Green))),
            enter_over.run_if(and(in_state(Mode::Play), in_state(Light::Red))),
        ));
        transition(&mut w);
        s.run(&mut w);
        assert_eq!(logged(&w), ["enter play"]);
    }

    #[test]
    fn state_changed_is_true_on_the_frame_of_a_change_only() {
        let mut w = world();
        w.insert_state(Mode::Menu);
        let mut s = Schedule::new("update");
        s.add_systems(update_in_play.run_if(state_changed::<Mode>));
        s.run(&mut w);
        transition(&mut w);
        s.run(&mut w); // came into being
        s.run(&mut w);
        set(&mut w, Mode::Play);
        transition(&mut w);
        s.run(&mut w);
        s.run(&mut w);
        set(&mut w, Mode::Play); // the same state: still a change
        transition(&mut w);
        s.run(&mut w);
        assert_eq!(logged(&w).len(), 3);
    }

    #[test]
    fn state_exists_is_true_from_the_first_transition() {
        let mut w = world();
        w.insert_state(Mode::Menu);
        let mut s = Schedule::new("update");
        s.add_systems(update_in_play.run_if(state_exists::<Mode>));
        s.run(&mut w);
        assert!(logged(&w).is_empty());
        transition(&mut w);
        s.run(&mut w);
        assert_eq!(logged(&w).len(), 1);
    }

    #[test]
    fn an_exit_schedule_of_a_state_never_left_does_not_run() {
        let mut w = world();
        w.init_schedule(OnExit(Mode::Over)).add_systems(exit_play);
        w.insert_state(Mode::Menu);
        transition(&mut w);
        set(&mut w, Mode::Play);
        transition(&mut w);
        assert!(logged(&w).is_empty());
    }
}
