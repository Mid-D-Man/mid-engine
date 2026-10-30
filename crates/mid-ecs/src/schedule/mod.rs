// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-ecs-schedule.md, section "mod.rs"
// ============================================================================
//! Ordering systems and running them: labels, sets, run conditions, the
//! schedule graph and a single-threaded executor.
//!
//! ```
//! use mid_ecs::schedule::{IntoScheduleConfigs, Schedule, SystemSet};
//! use mid_ecs::{ResMut, World};
//!
//! #[derive(Debug, Clone, PartialEq, Eq, Hash)]
//! struct Update;
//!
//! #[derive(Debug, Clone, PartialEq, Eq, Hash)]
//! enum Phase { Input, Simulate }
//! impl SystemSet for Phase {}
//!
//! struct Log(Vec<&'static str>);
//!
//! fn read_input(mut log: ResMut<Log>) { log.0.push("input"); }
//! fn move_units(mut log: ResMut<Log>) { log.0.push("move"); }
//! fn apply_damage(mut log: ResMut<Log>) { log.0.push("damage"); }
//!
//! let mut world = World::new();
//! world.insert_resource(Log(Vec::new()));
//!
//! let mut schedule = Schedule::new(Update);
//! schedule
//!     .configure_sets((Phase::Input, Phase::Simulate).chain())
//!     .add_systems((
//!         // added first, but its set runs last
//!         (move_units, apply_damage).chain().in_set(Phase::Simulate),
//!         read_input.in_set(Phase::Input),
//!     ));
//! world.add_schedule(schedule);
//!
//! world.run_schedule(Update);
//! assert_eq!(world.get_resource::<Log>().unwrap().0, ["input", "move", "damage"]);
//! ```
//!
//! What runs when, in short: a schedule is built on its first run (and
//! again after anything is added), into one fixed order that does not change
//! from run to run. Each system runs on the calling thread, one at a time,
//! and whatever it deferred through `Commands` is applied before the next
//! one starts.

mod condition;
mod config;
mod executor;
mod graph;
mod label;

#[cfg(test)]
mod tests;

use std::collections::{HashMap, HashSet};
use std::fmt;

use crate::system::System;
use crate::tick::Tick;
use crate::world::World;

pub use condition::{
    and, common_conditions, not, or, AndCondition, BoxedCondition, Condition, IntoCondition,
    IsCondition, IsFunctionCondition, NotCondition, OrCondition,
};
pub use config::{IntoScheduleConfigs, IsConfigs, IsTuple, ScheduleConfigs, ScheduleSystem};
pub use graph::{Ambiguity, DependencyCycle, OverlappingOrder, ScheduleBuildError};
pub use label::{
    IntoSystemSet, IsSystemTypeSet, IsUserSet, LabelKey, ScheduleLabel, SetKey, SystemSet,
};

use config::FlatLeaf;
use graph::{systems_of, GraphInput, NodeId};

// ── Settings ─────────────────────────────────────────────────────────

/// What a schedule does about two systems that conflict with nothing
/// ordering them. See [`Schedule::ambiguities`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AmbiguityDetection {
    /// Does not look. The default: a single-threaded schedule runs the
    /// same way every time either way.
    #[default]
    Ignore,
    /// Looks when the schedule is built and prints each one to standard
    /// error.
    Warn,
    /// Looks when the schedule is built and refuses to build if there are
    /// any ([`ScheduleBuildError::Ambiguities`]).
    Error,
}

// ── The schedule ─────────────────────────────────────────────────────

struct SystemNode {
    system: Box<dyn System>,
    conditions: Vec<BoxedCondition>,
}

struct SetNode {
    key: SetKey,
    conditions: Vec<BoxedCondition>,
}

/// What a build leaves behind.
struct BuiltSchedule {
    graph: graph::Built,
    /// For each system, the sets above it that carry conditions.
    condition_sets: Vec<Vec<usize>>,
}

/// Systems, how they are ordered, and when they run.
///
/// Build one with [`Schedule::new`], add systems with
/// [`Self::add_systems`] and configure sets with [`Self::configure_sets`],
/// then run it with [`Self::run`] or hand it to a world
/// ([`World::add_schedule`]) and run it by label.
pub struct Schedule {
    label: LabelKey,
    systems: Vec<SystemNode>,
    sets: Vec<SetNode>,
    set_index: HashMap<SetKey, usize>,
    /// `(set, member)`.
    members: Vec<(usize, NodeId)>,
    /// `(first, second)`.
    deps: Vec<(NodeId, NodeId)>,
    /// Pairs whose conflict is intended.
    silenced: Vec<(NodeId, NodeId)>,
    anon_counter: u32,
    ambiguity: AmbiguityDetection,
    built: Option<BuiltSchedule>,
    /// Each set's condition result in the current run.
    set_results: Vec<Option<bool>>,
}

impl fmt::Debug for Schedule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Schedule")
            .field("label", &self.label)
            .field("systems", &self.systems.len())
            .field("sets", &self.sets.len())
            .field("built", &self.built.is_some())
            .finish()
    }
}

impl Schedule {
    /// An empty schedule called `label`.
    pub fn new(label: impl ScheduleLabel) -> Self {
        Self {
            label: label.label_key(),
            systems: Vec::new(),
            sets: Vec::new(),
            set_index: HashMap::new(),
            members: Vec::new(),
            deps: Vec::new(),
            silenced: Vec::new(),
            anon_counter: 0,
            ambiguity: AmbiguityDetection::default(),
            built: None,
            set_results: Vec::new(),
        }
    }

    /// The label this schedule goes by.
    pub fn label(&self) -> &LabelKey {
        &self.label
    }

    /// How many systems have been added.
    pub fn len(&self) -> usize {
        self.systems.len()
    }

    /// `true` if no system has been added.
    pub fn is_empty(&self) -> bool {
        self.systems.is_empty()
    }

    /// Sets what happens to conflicting, unordered systems at build time.
    pub fn set_ambiguity_detection(&mut self, level: AmbiguityDetection) -> &mut Self {
        if level != self.ambiguity {
            self.ambiguity = level;
            self.built = None;
        }
        self
    }

    /// The current setting.
    pub fn ambiguity_detection(&self) -> AmbiguityDetection {
        self.ambiguity
    }

    fn set_node(&mut self, key: &SetKey) -> usize {
        if let Some(&i) = self.set_index.get(key) {
            return i;
        }
        let i = self.sets.len();
        self.sets.push(SetNode {
            key: key.clone(),
            conditions: Vec::new(),
        });
        self.set_index.insert(key.clone(), i);
        i
    }

    /// Records what a leaf was configured with.
    fn link<T>(&mut self, node: NodeId, leaf: &FlatLeaf<T>) {
        for key in &leaf.in_sets {
            let set = self.set_node(key);
            self.members.push((set, node));
        }
        for key in &leaf.before {
            let set = self.set_node(key);
            self.deps.push((node, NodeId::Set(set)));
        }
        for key in &leaf.after {
            let set = self.set_node(key);
            self.deps.push((NodeId::Set(set), node));
        }
        for key in &leaf.ambiguous_with {
            let set = self.set_node(key);
            self.silenced.push((node, NodeId::Set(set)));
        }
    }

    fn add_chains(&mut self, chains: &[(Vec<usize>, Vec<usize>)], node_of: &[NodeId]) {
        for (before, after) in chains {
            for &a in before {
                for &b in after {
                    self.deps.push((node_of[a], node_of[b]));
                }
            }
        }
    }

    fn add_anonymous_sets(&mut self, anon: Vec<config::FlatAnonSet>) {
        for set in anon {
            let index = self.set_node(&set.key);
            self.sets[index].conditions.extend(set.conditions);
        }
    }

    /// Adds systems: a system, or a tuple of them, with ordering and
    /// conditions attached (see [`IntoScheduleConfigs`]). The schedule is
    /// built again the next time it runs.
    pub fn add_systems<M>(
        &mut self,
        systems: impl IntoScheduleConfigs<ScheduleSystem, M>,
    ) -> &mut Self {
        let flat = systems.into_configs().flatten(&mut self.anon_counter);
        self.built = None;
        self.add_anonymous_sets(flat.anon_sets);
        let mut nodes = Vec::with_capacity(flat.leaves.len());
        for leaf in flat.leaves {
            let id = self.systems.len();
            let node = NodeId::System(id);
            nodes.push(node);
            let type_set = self.set_node(&leaf.item.type_set);
            self.members.push((type_set, node));
            self.link(node, &leaf);
            self.systems.push(SystemNode {
                system: leaf.item.system,
                conditions: leaf.conditions,
            });
        }
        self.add_chains(&flat.chains, &nodes);
        self
    }

    /// Declares sets: their order against each other, their place inside
    /// other sets, and conditions that gate every system in them. A set
    /// exists as soon as anything refers to it, so this is only needed to
    /// say something *about* a set.
    pub fn configure_sets<M>(&mut self, sets: impl IntoScheduleConfigs<SetKey, M>) -> &mut Self {
        let flat = sets.into_configs().flatten(&mut self.anon_counter);
        self.built = None;
        self.add_anonymous_sets(flat.anon_sets);
        let mut nodes = Vec::with_capacity(flat.leaves.len());
        for leaf in flat.leaves {
            let index = self.set_node(&leaf.item);
            let node = NodeId::Set(index);
            nodes.push(node);
            self.link(node, &leaf);
            self.sets[index].conditions.extend(leaf.conditions);
        }
        self.add_chains(&flat.chains, &nodes);
        self
    }

    /// Initializes every system and condition, and works out the run
    /// order if anything changed since the last time. Called by
    /// [`Self::run`].
    ///
    /// # Errors
    /// See [`ScheduleBuildError`].
    ///
    /// # Panics
    /// If two parameters of one system conflict, or a condition's
    /// parameters write.
    pub fn build(&mut self, world: &mut World) -> Result<(), ScheduleBuildError> {
        self.prepare(world)?;
        if self.ambiguity != AmbiguityDetection::Ignore {
            let found = self.find_ambiguities();
            match self.ambiguity {
                AmbiguityDetection::Ignore => {}
                AmbiguityDetection::Warn => {
                    for a in &found {
                        eprintln!("schedule `{:?}`: {a}", self.label);
                    }
                }
                AmbiguityDetection::Error if !found.is_empty() => {
                    self.built = None;
                    return Err(ScheduleBuildError::Ambiguities(found));
                }
                AmbiguityDetection::Error => {}
            }
        }
        Ok(())
    }

    fn prepare(&mut self, world: &mut World) -> Result<(), ScheduleBuildError> {
        for node in &mut self.systems {
            node.system.initialize(world);
            for condition in &mut node.conditions {
                condition.initialize(world);
            }
        }
        for set in &mut self.sets {
            for condition in &mut set.conditions {
                condition.initialize(world);
            }
        }
        if self.built.is_some() {
            return Ok(());
        }
        let system_names: Vec<String> = self
            .systems
            .iter()
            .map(|n| n.system.name().to_owned())
            .collect();
        let set_names: Vec<String> = self.sets.iter().map(|s| format!("{:?}", s.key)).collect();
        let graph = graph::build(&GraphInput {
            system_names: &system_names,
            set_names: &set_names,
            members: &self.members,
            deps: &self.deps,
        })?;
        let condition_sets = graph
            .sets_of_system
            .iter()
            .map(|sets| {
                sets.iter()
                    .copied()
                    .filter(|&s| !self.sets[s].conditions.is_empty())
                    .collect()
            })
            .collect();
        self.built = Some(BuiltSchedule {
            graph,
            condition_sets,
        });
        Ok(())
    }

    /// Builds if needed, then runs every system once, in order, skipping
    /// those whose conditions say so.
    ///
    /// # Errors
    /// If the schedule cannot be built.
    pub fn try_run(&mut self, world: &mut World) -> Result<(), ScheduleBuildError> {
        self.build(world)?;
        self.execute(world);
        Ok(())
    }

    /// [`Self::try_run`], panicking with the build error's message.
    pub fn run(&mut self, world: &mut World) {
        if let Err(error) = self.try_run(world) {
            panic!("schedule `{:?}` could not be built: {error}", self.label);
        }
    }

    /// The names of the systems in the order they run, or `None` before
    /// the first build.
    pub fn run_order(&self) -> Option<Vec<&str>> {
        let built = self.built.as_ref()?;
        Some(
            built
                .graph
                .order
                .iter()
                .map(|&i| self.systems[i].system.name())
                .collect(),
        )
    }

    /// Every pair of systems that conflict (see
    /// [`crate::FilteredAccessSet::get_conflicts`]) with nothing ordering
    /// them, directly or through others, and no `ambiguous_with` to say it
    /// is intended. Builds the schedule if it is not built.
    ///
    /// An exclusive system conflicts with every system that touches
    /// anything.
    ///
    /// # Errors
    /// If the schedule cannot be built (an ambiguity is not one, whatever
    /// the setting).
    pub fn ambiguities(&mut self, world: &mut World) -> Result<Vec<Ambiguity>, ScheduleBuildError> {
        self.prepare(world)?;
        Ok(self.find_ambiguities())
    }

    fn find_ambiguities(&self) -> Vec<Ambiguity> {
        let built = self.built.as_ref().expect("built before it is searched");
        let graph = &built.graph;
        let reach = graph.reachability();
        let mut silenced: HashSet<(usize, usize)> = HashSet::new();
        for (a, b) in &self.silenced {
            for &x in systems_of(&graph.systems_of_set, a) {
                for &y in systems_of(&graph.systems_of_set, b) {
                    if x != y {
                        silenced.insert((x.min(y), x.max(y)));
                    }
                }
            }
        }
        let mut found = Vec::new();
        for i in 0..self.systems.len() {
            for j in i + 1..self.systems.len() {
                if reach.contains(i, j) || reach.contains(j, i) || silenced.contains(&(i, j)) {
                    continue;
                }
                let conflicts = self.systems[i]
                    .system
                    .access()
                    .get_conflicts(self.systems[j].system.access());
                if conflicts.is_empty() {
                    continue;
                }
                found.push(Ambiguity {
                    first: self.systems[i].system.name().to_owned(),
                    second: self.systems[j].system.name().to_owned(),
                    conflicts: conflicts.names().to_vec(),
                    whole_world: conflicts.is_all(),
                });
            }
        }
        found
    }

    /// The last-run tick of whichever system or condition last ran
    /// furthest back from `now` (never having run counts as tick zero), or
    /// `None` if the schedule has none. What to pass to
    /// [`World::trim_removals`] so nothing a system here has yet to see is
    /// dropped (include any `ChangeTracker` held outside the schedule).
    pub fn oldest_last_run(&self, now: Tick) -> Option<Tick> {
        let mut oldest: Option<Tick> = None;
        let mut consider = |tick: Tick| {
            oldest = Some(match oldest {
                Some(o) if now.relative_to(o).get() >= now.relative_to(tick).get() => o,
                _ => tick,
            });
        };
        for node in &self.systems {
            consider(node.system.last_run());
            for c in &node.conditions {
                consider(c.oldest_last_run(now));
            }
        }
        for set in &self.sets {
            for c in &set.conditions {
                consider(c.oldest_last_run(now));
            }
        }
        oldest
    }
}

// ── Schedules ────────────────────────────────────────────────────────

/// Every schedule a world holds, by label. A world keeps one as a
/// resource; use the `World` methods ([`World::add_schedule`],
/// [`World::run_schedule`], ...) rather than this directly.
#[derive(Default)]
pub struct Schedules {
    schedules: HashMap<LabelKey, Schedule>,
    /// Labels whose schedule is taken out for a run.
    running: HashSet<LabelKey>,
}

impl Schedules {
    /// How many schedules there are, including any being run.
    pub fn len(&self) -> usize {
        self.schedules.len() + self.running.len()
    }

    /// `true` if there are none.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The schedule called `label`, unless it is being run right now.
    pub fn get(&self, label: impl ScheduleLabel) -> Option<&Schedule> {
        self.schedules.get(&label.label_key())
    }

    /// The schedule called `label`, to change, unless it is being run.
    pub fn get_mut(&mut self, label: impl ScheduleLabel) -> Option<&mut Schedule> {
        self.schedules.get_mut(&label.label_key())
    }

    /// Whether a schedule called `label` exists, running or not.
    pub fn contains(&self, label: impl ScheduleLabel) -> bool {
        let key = label.label_key();
        self.schedules.contains_key(&key) || self.running.contains(&key)
    }

    /// Whether the schedule called `label` is being run right now.
    pub fn is_running(&self, label: impl ScheduleLabel) -> bool {
        self.running.contains(&label.label_key())
    }
}

/// Why a schedule could not be run by label.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScheduleRunError {
    /// No schedule has this label (its debug name is given).
    NotFound(String),
    /// The schedule is being run right now, so it cannot be run or
    /// changed from inside itself.
    Running(String),
    /// The schedule could not be built.
    Build(ScheduleBuildError),
}

impl fmt::Display for ScheduleRunError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotFound(name) => write!(f, "there is no schedule `{name}`"),
            Self::Running(name) => write!(
                f,
                "the schedule `{name}` is being run right now, so it cannot be used from inside itself"
            ),
            Self::Build(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for ScheduleRunError {}

impl From<ScheduleBuildError> for ScheduleRunError {
    fn from(error: ScheduleBuildError) -> Self {
        Self::Build(error)
    }
}

/// Puts a schedule back in the world it was taken from, however the
/// closure that had it ends, panic included.
struct ScopeGuard<'w> {
    world: &'w mut World,
    key: LabelKey,
    schedule: Option<Schedule>,
}

impl Drop for ScopeGuard<'_> {
    fn drop(&mut self) {
        if let Some(schedule) = self.schedule.take() {
            let schedules = self.world.schedules_mut();
            schedules.running.remove(&self.key);
            schedules.schedules.insert(self.key.clone(), schedule);
        }
    }
}

impl World {
    fn schedules_mut(&mut self) -> &mut Schedules {
        if !self.contains_resource::<Schedules>() {
            self.insert_resource(Schedules::default());
        }
        self.get_resource_mut::<Schedules>()
            .expect("inserted just above")
    }

    /// Adds a schedule, returning the one it replaces if there was one.
    ///
    /// # Panics
    /// If a schedule with this label is being run right now.
    pub fn add_schedule(&mut self, schedule: Schedule) -> Option<Schedule> {
        let schedules = self.schedules_mut();
        assert!(
            !schedules.running.contains(&schedule.label),
            "cannot replace the schedule `{:?}` while it is being run",
            schedule.label
        );
        schedules.schedules.insert(schedule.label.clone(), schedule)
    }

    /// The schedule called `label`, made empty first if there is none, so
    /// systems can be added to it without owning it:
    /// `world.init_schedule(Update).add_systems(f)`.
    ///
    /// # Panics
    /// If the schedule is being run right now.
    pub fn init_schedule(&mut self, label: impl ScheduleLabel) -> &mut Schedule {
        let key = label.label_key();
        let schedules = self.schedules_mut();
        assert!(
            !schedules.running.contains(&key),
            "cannot change the schedule `{key:?}` while it is being run"
        );
        schedules
            .schedules
            .entry(key)
            .or_insert_with(|| Schedule::new(label))
    }

    /// Whether a schedule called `label` exists.
    pub fn contains_schedule(&self, label: impl ScheduleLabel) -> bool {
        self.get_resource::<Schedules>()
            .is_some_and(|s| s.contains(label))
    }

    /// Takes the schedule out of the world, calls `f` with the world and
    /// the schedule, and puts it back. The schedule goes back if `f`
    /// panics. This is what [`Self::run_schedule`] is made of, and what
    /// lets code that runs inside one schedule (a `Main` schedule, say)
    /// run another.
    ///
    /// # Errors
    /// If there is no schedule called `label`, or it is the one being run.
    pub fn try_schedule_scope<R>(
        &mut self,
        label: impl ScheduleLabel,
        f: impl FnOnce(&mut World, &mut Schedule) -> R,
    ) -> Result<R, ScheduleRunError> {
        let key = label.label_key();
        let schedules = self.schedules_mut();
        let Some(schedule) = schedules.schedules.remove(&key) else {
            let name = format!("{key:?}");
            return Err(if schedules.running.contains(&key) {
                ScheduleRunError::Running(name)
            } else {
                ScheduleRunError::NotFound(name)
            });
        };
        schedules.running.insert(key.clone());
        let mut guard = ScopeGuard {
            world: self,
            key,
            schedule: Some(schedule),
        };
        let ScopeGuard {
            world, schedule, ..
        } = &mut guard;
        let schedule = schedule.as_mut().expect("put back only by the guard");
        Ok(f(&mut **world, schedule))
    }

    /// [`Self::try_schedule_scope`], panicking if there is no such
    /// schedule.
    pub fn schedule_scope<R>(
        &mut self,
        label: impl ScheduleLabel,
        f: impl FnOnce(&mut World, &mut Schedule) -> R,
    ) -> R {
        match self.try_schedule_scope(label, f) {
            Ok(result) => result,
            Err(error) => panic!("{error}"),
        }
    }

    /// Runs the schedule called `label` once.
    ///
    /// # Errors
    /// If there is no such schedule, it is already being run, or it cannot
    /// be built.
    pub fn try_run_schedule(&mut self, label: impl ScheduleLabel) -> Result<(), ScheduleRunError> {
        self.try_schedule_scope(label, |world, schedule| schedule.try_run(world))?
            .map_err(ScheduleRunError::Build)
    }

    /// Runs the schedule called `label` once.
    ///
    /// # Panics
    /// If there is no such schedule, it is already being run, or it cannot
    /// be built (the message names the systems involved).
    pub fn run_schedule(&mut self, label: impl ScheduleLabel) {
        if let Err(error) = self.try_run_schedule(label) {
            panic!("{error}");
        }
    }
}
