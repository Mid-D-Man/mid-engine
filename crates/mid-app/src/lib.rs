// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-app.md, section "lib.rs"
// ============================================================================
//! mid-app: `App`, `Plugin` and the `Main` schedule for Mid Engine.
//!
//! The layer that turns "systems that exist" into "a program that runs":
//! the role `bevy_app` has in Bevy. [`App`] owns a [`mid_ecs::World`]; the
//! systems live in schedules in that world, one per stage, and
//! [`Main`] runs the stages in a fixed order once per frame:
//!
//! ```text
//! PreUpdate → StateTransition → Update → Net → Physics (0..n times) → PostUpdate
//! ```
//!
//! [`Physics`] is the one stage that runs on a fixed timestep (60 Hz by
//! default, `mid_time::FixedTimestep`): each frame it runs as many times as
//! the frame's time has made ready, which may be none.
//!
//! ```
//! use mid_app::{App, Plugin, Update};
//! use mid_ecs::ResMut;
//! use std::time::Duration;
//!
//! struct Frames(u32);
//!
//! fn count(mut frames: ResMut<Frames>) {
//!     frames.0 += 1;
//! }
//!
//! struct CountPlugin;
//! impl Plugin for CountPlugin {
//!     fn build(&self, app: &mut App) {
//!         app.insert_resource(Frames(0)).add_systems(Update, count);
//!     }
//! }
//!
//! let mut app = App::new();
//! app.add_plugin(CountPlugin);
//! app.update_with(Duration::from_millis(16));
//! app.update_with(Duration::from_millis(16));
//! assert_eq!(app.world.get_resource::<Frames>().unwrap().0, 2);
//! ```
//!
//! Messages (`mid_ecs::MessageWriter` and `MessageReader`) are trimmed after
//! each frame. What is not here, on purpose: observers and states (the next
//! pieces of `mid-ecs`), a second timestep for the network stage (one is
//! built; the 128 Hz network tick is a decision for when `mid-net` is
//! driven from here), and a startup stage (add one when something needs it).

mod app;
mod labels;
#[cfg(test)]
mod tests;

pub use app::{App, Plugin, DEFAULT_FIXED_HZ, DEFAULT_MAX_FIXED_STEPS};
pub use labels::{Main, Net, Physics, PostUpdate, PreUpdate, Update};
pub use mid_ecs::observer::{Add, Despawn, Discard, Event, Insert, ObserverId, On, Remove};
pub use mid_ecs::state::{
    in_state, state_changed, state_exists, NextState, OnEnter, OnExit, OnTransition, State,
    StateTransition, StateTransitionMessage, States,
};

// The crates a plugin is written against, so one `mid-app` dependency
// brings matching versions of them.
pub use mid_ecs;
pub use mid_time;
