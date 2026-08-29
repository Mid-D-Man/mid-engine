// crates/mid-app/src/lib.rs
//! mid-app — App / Plugin / schedule-runner for Mid Engine.
//!
//! New this pass (`docs/roadmap.md`, Decision 1): the layer that turns
//! "systems that exist" into "a program that runs" — `bevy_app`'s role
//! in Bevy (`docs/bevy-comparison.md` §2), missing entirely from this
//! workspace until now. `examples/headless-server` currently hand-rolls
//! its own bootstrap in `main()`; this crate exists so that stops being
//! necessary for whatever gets built next (`mid-physics`, `mid-anim`,
//! ...).
//!
//! ## What's real in this pass
//!
//! [`App`] owns a `mid_ecs::World` and a fixed 5-stage system list
//! (`docs/roadmap.md`'s Decision 1 proposed exactly this shape:
//! `PreUpdate → Update → Net → Physics → PostUpdate`). [`App::run_once`]
//! runs every registered system once, in that fixed order. [`Plugin`] is
//! the trait a feature crate implements to register its systems without
//! its consumer having to know what those systems are.
//!
//! ## What's not built yet, on purpose
//!
//! - **Systems are `Box<dyn FnMut(&mut World)>`**, not generic over
//!   query parameters the way `bevy_ecs::system::SystemParam` is.
//!   `mid-ecs`'s own `query`/`query2` modules aren't wired into a
//!   system-parameter-extraction convention yet — inventing one here,
//!   in `mid-app`, ahead of `mid-ecs` growing its own, would mean
//!   designing it twice. A system reaches into `world` itself
//!   (`world.query::<...>()`, `world.get::<...>()`, etc.) using
//!   whatever `mid-ecs` already exposes; that's the smallest thing
//!   that's still real and useful today.
//! - **No real timed loop.** [`App::run_once`] runs one tick; there is
//!   deliberately no `App::run()` wrapping it in a real loop yet.
//!   Wiring `mid-time`'s `Clock`/`FixedTimestep` into a real timed
//!   `run()` is the natural next step once both crates exist side by
//!   side (`docs/roadmap.md`) — not duplicated here ahead of that.
//! - **Only 5 fixed stages, no dynamic ordering/labels.** Matches this
//!   project's own established rule: add stages when something real
//!   needs one, not speculatively.

use mid_ecs::World;

/// One system: a plain function (or closure) that mutates the `World`.
/// See the module doc's "not built yet" section for why this isn't
/// generic over query parameters.
pub type System = Box<dyn FnMut(&mut World)>;

/// Fixed execution order every [`App`] runs its systems in. Matches the
/// shape proposed in `docs/roadmap.md`'s Decision 1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(usize)]
pub enum Stage {
    PreUpdate = 0,
    Update = 1,
    Net = 2,
    Physics = 3,
    PostUpdate = 4,
}

/// Number of [`Stage`] variants — kept in sync with the enum by hand
/// (small, fixed, changes rarely); used to size `App`'s stage array.
const STAGE_COUNT: usize = 5;

/// Something that adds systems (and, later, resources) to an [`App`]
/// when added. Every engine feature (net, physics, ...) is expected to
/// ship one of these rather than every consumer wiring its systems up
/// by hand — the problem `examples/headless-server` currently solves by
/// improvising its own bootstrap, per `docs/roadmap.md`.
pub trait Plugin {
    fn build(&self, app: &mut App);
}

/// Owns a `World` and the fixed-stage system list.
pub struct App {
    pub world: World,
    stages: [Vec<System>; STAGE_COUNT],
}

impl App {
    pub fn new() -> Self {
        Self {
            world: World::new(),
            stages: Default::default(),
        }
    }

    /// Registers `system` to run in `stage`, in the order it was added
    /// relative to other systems in the *same* stage. Returns `&mut
    /// Self` for chaining (`app.add_system(...).add_system(...)`).
    pub fn add_system(&mut self, stage: Stage, system: impl FnMut(&mut World) + 'static) -> &mut Self {
        self.stages[stage as usize].push(Box::new(system));
        self
    }

    /// Calls `plugin.build(self)` immediately — matches Bevy's own
    /// eager-build-on-add convention, so a plugin's systems are live as
    /// soon as `add_plugin` returns, not deferred to some later
    /// "finalize" step.
    pub fn add_plugin(&mut self, plugin: impl Plugin) -> &mut Self {
        plugin.build(self);
        self
    }

    /// Runs every registered system once, in fixed stage order
    /// (`PreUpdate → Update → Net → Physics → PostUpdate`). Call this
    /// yourself in a loop — with your own timing — until `mid-time` is
    /// wired in; see the module doc.
    pub fn run_once(&mut self) {
        for stage_systems in &mut self.stages {
            for system in stage_systems.iter_mut() {
                system(&mut self.world);
            }
        }
    }
}

impl Default for App {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn app_runs_systems_in_fixed_stage_order() {
        let mut app = App::new();
        let e = app.world.spawn();
        app.world.insert(e, Vec::<&'static str>::new());

        app.add_system(Stage::PostUpdate, move |world| {
            world.get_mut::<Vec<&'static str>>(e).unwrap().push("post");
        });
        app.add_system(Stage::PreUpdate, move |world| {
            world.get_mut::<Vec<&'static str>>(e).unwrap().push("pre");
        });
        app.add_system(Stage::Update, move |world| {
            world.get_mut::<Vec<&'static str>>(e).unwrap().push("update");
        });

        app.run_once();

        let log = app.world.get::<Vec<&'static str>>(e).unwrap();
        assert_eq!(log.as_slice(), &["pre", "update", "post"]);
    }

    #[test]
    fn plugin_build_adds_its_systems() {
        struct CountPlugin;
        impl Plugin for CountPlugin {
            fn build(&self, app: &mut App) {
                app.add_system(Stage::Update, |_world| {});
                app.add_system(Stage::Update, |_world| {});
            }
        }

        let mut app = App::new();
        app.add_plugin(CountPlugin);
        assert_eq!(app.stages[Stage::Update as usize].len(), 2);
    }

    #[test]
    fn run_once_with_no_systems_does_not_panic() {
        let mut app = App::new();
        app.run_once();
    }
}
