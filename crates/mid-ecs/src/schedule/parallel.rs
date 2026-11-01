// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-ecs-schedule-parallel.md, section "parallel.rs"
// ============================================================================
//! Running a built schedule on the rayon pool: every system that can run does,
//! as many at a time as the pool has threads, and none runs beside one it
//! conflicts with.
//!
//! There is no coordinator thread. The state of the run (which systems wait
//! for what, which run, which are done) is one value behind a mutex, and
//! whoever finishes a task, or starts the run, takes the lock, updates it,
//! and starts whatever has become able to run. Nothing ever blocks waiting
//! for another task, so a pool of one thread works, and so does running a
//! schedule from inside a task of the same pool.
//!
//! What may start, for a system that has been reached:
//!
//! 1. every system ordered before it has finished (or was skipped);
//! 2. none of those left deferred work that has not been applied yet: if one
//!    did, nothing new starts until everything running has finished, then all
//!    pending deferred work is applied, in the schedule's order (this is the
//!    sync point; it is found when needed rather than placed in advance);
//! 3. its conditions (and those of the sets above it), read-only, are
//!    evaluated, on whichever thread is advancing the run, once nothing
//!    running conflicts with what they read; any false one skips it;
//! 4. nothing running conflicts with what it touches; an exclusive system
//!    needs nothing running at all, and runs on the advancing thread.
//!
//! Ticks are counted here, not by the world: each condition evaluation and
//! each system run takes the next number, in the order they start, and the
//! world's tick is set to the count whenever this code holds the whole world
//! (before an exclusive system, before deferred work, at the end).

use std::any::Any;
use std::collections::BTreeSet;
use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};
use std::sync::{Mutex, MutexGuard, PoisonError};

use crate::access::FilteredAccessSet;
use crate::schedule::graph::{Built, Reach};
use crate::schedule::{BoxedCondition, Schedule, SetNode, SystemNode};
use crate::system::System;
use crate::tick::Tick;
use crate::world_cell::WorldCell;

// ── The plan ─────────────────────────────────────────────────────────

/// What the parallel executor needs to know about a built schedule, worked
/// out once per build.
pub(super) struct ParallelPlan {
    /// For each system, the systems that must wait for it.
    successors: Vec<Vec<usize>>,
    /// For each system, how many systems it waits for.
    predecessors: Vec<usize>,
    /// Which system must run before which, transitively.
    reach: Reach,
    /// The position of each system in the run order: the order in which
    /// ready systems are considered.
    rank: Vec<usize>,
    /// What each system touches.
    system_access: Vec<FilteredAccessSet>,
    /// What each system's conditions and its sets' conditions read.
    gate_access: Vec<FilteredAccessSet>,
    exclusive: Vec<bool>,
    has_deferred: Vec<bool>,
}

impl ParallelPlan {
    pub(super) fn new(
        graph: &Built,
        systems: &[SystemNode],
        sets: &[SetNode],
        condition_sets: &[Vec<usize>],
    ) -> Self {
        let n = systems.len();
        let mut successors = vec![Vec::new(); n];
        let mut predecessors = vec![0; n];
        for &(a, b) in &graph.edges {
            successors[a].push(b);
            predecessors[b] += 1;
        }
        let mut rank = vec![0; n];
        for (position, &system) in graph.order.iter().enumerate() {
            rank[system] = position;
        }
        let union = |into: &mut FilteredAccessSet, from: &FilteredAccessSet| {
            for part in from.parts() {
                into.add(part.clone());
            }
        };
        let mut gate_access = Vec::with_capacity(n);
        for (i, node) in systems.iter().enumerate() {
            let mut gate = FilteredAccessSet::new();
            for condition in &node.conditions {
                union(&mut gate, condition.access());
            }
            for &set in &condition_sets[i] {
                for condition in &sets[set].conditions {
                    union(&mut gate, condition.access());
                }
            }
            gate_access.push(gate);
        }
        Self {
            successors,
            predecessors,
            reach: graph.reachability(),
            rank,
            system_access: systems.iter().map(|n| n.system.access().clone()).collect(),
            gate_access,
            exclusive: systems.iter().map(|n| n.system.is_exclusive()).collect(),
            has_deferred: systems.iter().map(|n| n.system.has_deferred()).collect(),
        }
    }
}

// ── The run ──────────────────────────────────────────────────────────

/// A world handle that may be sent to a task. Sound because the scheduler
/// only gives a task a handle for a system whose access nothing running
/// conflicts with, and takes the whole world back only when nothing runs.
#[derive(Clone, Copy)]
struct SendCell<'w>(WorldCell<'w>);

// SAFETY: see the type's docs; the rule is the schedule's, kept by `Run`.
unsafe impl Send for SendCell<'_> {}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Phase {
    /// Waiting for systems ordered before it.
    Waiting,
    /// Everything before it is done; its conditions are not evaluated yet.
    Reached,
    /// Its conditions held; waiting for room to run.
    Cleared,
    Running,
    Finished,
}

struct Run<'a> {
    plan: &'a ParallelPlan,
    cell: SendCell<'a>,
    systems: Vec<Option<&'a mut Box<dyn System>>>,
    conditions: Vec<&'a mut Vec<BoxedCondition>>,
    sets: &'a mut [SetNode],
    condition_sets: &'a [Vec<usize>],
    set_results: Vec<Option<bool>>,
    waiting_for: Vec<usize>,
    phase: Vec<Phase>,
    /// Reached or cleared systems, by `(rank, system)`.
    reached: BTreeSet<(usize, usize)>,
    running: Vec<usize>,
    /// Finished systems whose deferred work has not been applied.
    pending: Vec<usize>,
    next_tick: Tick,
    /// A system is held back by pending deferred work: nothing starts until
    /// everything running has finished and the work is applied.
    sync_wanted: bool,
    finished: usize,
    panic: Option<Box<dyn Any + Send>>,
}

fn lock<'m, T>(mutex: &'m Mutex<T>) -> MutexGuard<'m, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

impl Schedule {
    /// Runs every system once on the rayon pool. The schedule must be built
    /// with a parallel plan.
    pub(super) fn execute_parallel(&mut self, world: &mut crate::world::World) {
        let n = self.systems.len();
        let first_tick = world.change_tick();
        let (panic, next_tick);
        {
            let Schedule {
                systems,
                sets,
                built,
                thread_pool,
                ..
            } = self;
            let built = built.as_ref().expect("built before it is run");
            let plan = built
                .plan
                .as_ref()
                .expect("a parallel schedule is built with its plan");

            let mut system_refs = Vec::with_capacity(n);
            let mut condition_refs = Vec::with_capacity(n);
            for node in systems.iter_mut() {
                let SystemNode { system, conditions } = node;
                system_refs.push(Some(system));
                condition_refs.push(conditions);
            }
            let mut run = Run {
                plan,
                cell: SendCell(world.as_world_cell()),
                systems: system_refs,
                conditions: condition_refs,
                set_results: vec![None; sets.len()],
                sets,
                condition_sets: &built.condition_sets,
                waiting_for: plan.predecessors.clone(),
                phase: vec![Phase::Waiting; n],
                reached: BTreeSet::new(),
                running: Vec::new(),
                pending: Vec::new(),
                next_tick: first_tick,
                sync_wanted: false,
                finished: 0,
                panic: None,
            };
            for i in 0..n {
                if run.waiting_for[i] == 0 {
                    run.phase[i] = Phase::Reached;
                    run.reached.insert((plan.rank[i], i));
                }
            }
            let shared = Mutex::new(run);
            {
                let shared = &shared;
                match thread_pool {
                    Some(pool) => pool.in_place_scope(|scope| start(scope, shared)),
                    None => rayon::in_place_scope(|scope| start(scope, shared)),
                }
            }
            let run = shared.into_inner().unwrap_or_else(PoisonError::into_inner);
            panic = run.panic;
            next_tick = run.next_tick;
        }
        // Everything borrowed from the world is gone. A panic comes after
        // the tick bookkeeping, so the schedule is usable again.
        world.advance_change_tick_to(next_tick);
        self.finish_run(world);
        if let Some(payload) = panic {
            resume_unwind(payload);
        }
    }
}

/// The first advance of a run, on the thread that called it.
fn start<'a, 's>(scope: &rayon::Scope<'s>, shared: &'s Mutex<Run<'a>>)
where
    'a: 's,
{
    let mut run = lock(shared);
    run.advance(scope, shared);
}

impl<'a> Run<'a> {
    fn take_tick(&mut self) -> Tick {
        let tick = self.next_tick;
        self.next_tick = Tick::new(tick.get().wrapping_add(1));
        tick
    }

    fn compatible_with_running(&self, access: &FilteredAccessSet) -> bool {
        self.running
            .iter()
            .all(|&r| self.plan.system_access[r].is_compatible(access))
    }

    fn blocked_by_pending(&self, system: usize) -> bool {
        self.pending
            .iter()
            .any(|&p| self.plan.reach.contains(p, system))
    }

    /// Starts everything that can start now. Called with the lock held, by
    /// whoever just changed the state.
    fn advance<'s>(&mut self, scope: &rayon::Scope<'s>, shared: &'s Mutex<Run<'a>>)
    where
        'a: 's,
    {
        loop {
            if self.panic.is_some() {
                return;
            }
            if self.sync_wanted {
                if !self.running.is_empty() {
                    return;
                }
                self.apply_pending();
                self.sync_wanted = false;
                if self.panic.is_some() {
                    return;
                }
            }
            let mut progressed = false;
            let candidates: Vec<(usize, usize)> = self.reached.iter().copied().collect();
            for (rank, i) in candidates {
                if self.blocked_by_pending(i) {
                    self.sync_wanted = true;
                    progressed = true;
                    break;
                }
                if self.phase[i] == Phase::Reached {
                    if !self.compatible_with_running(&self.plan.gate_access[i]) {
                        continue;
                    }
                    if self.evaluate(i) {
                        self.phase[i] = Phase::Cleared;
                    } else {
                        self.reached.remove(&(rank, i));
                        self.finish(i, false);
                        progressed = true;
                        continue;
                    }
                    if self.panic.is_some() {
                        return;
                    }
                }
                if self.plan.exclusive[i] {
                    if !self.running.is_empty() {
                        // let what runs finish; nothing later starts first
                        break;
                    }
                    self.reached.remove(&(rank, i));
                    self.run_exclusive(i);
                    progressed = true;
                    if self.panic.is_some() {
                        return;
                    }
                    continue;
                }
                if !self.compatible_with_running(&self.plan.system_access[i]) {
                    continue;
                }
                self.reached.remove(&(rank, i));
                self.dispatch(scope, shared, i);
                progressed = true;
            }
            if !progressed {
                break;
            }
        }
        if self.finished == self.systems.len() && self.running.is_empty() {
            // the last of the work: whatever it deferred
            self.apply_pending();
        }
    }

    /// Evaluates the conditions of the sets above `system` (each set once
    /// per run) and then, if they all held, its own. `true` if it may run.
    fn evaluate(&mut self, system: usize) -> bool {
        let result = catch_unwind(AssertUnwindSafe(|| {
            let cell = self.cell.0;
            let mut allowed = true;
            for k in 0..self.condition_sets[system].len() {
                let set = self.condition_sets[system][k];
                let answer = match self.set_results[set] {
                    Some(answer) => answer,
                    None => {
                        let mut all = true;
                        for c in 0..self.sets[set].conditions.len() {
                            let tick = self.take_tick();
                            // SAFETY: nothing running conflicts with what the
                            // condition reads: the caller checked the gate
                            // access, which includes every set condition.
                            all &=
                                unsafe { self.sets[set].conditions[c].run_unchecked(cell, tick) };
                        }
                        self.set_results[set] = Some(all);
                        all
                    }
                };
                allowed &= answer;
            }
            if allowed {
                for c in 0..self.conditions[system].len() {
                    let tick = self.take_tick();
                    // SAFETY: as above.
                    allowed &= unsafe { self.conditions[system][c].run_unchecked(cell, tick) };
                }
            }
            allowed
        }));
        match result {
            Ok(allowed) => allowed,
            Err(payload) => {
                self.panic = Some(payload);
                false
            }
        }
    }

    fn dispatch<'s>(&mut self, scope: &rayon::Scope<'s>, shared: &'s Mutex<Run<'a>>, i: usize)
    where
        'a: 's,
    {
        let system = self.systems[i].take().expect("a system runs once");
        let this_run = self.take_tick();
        self.phase[i] = Phase::Running;
        self.running.push(i);
        let cell = self.cell;
        scope.spawn(move |scope| run_task(scope, shared, i, system, cell, this_run));
    }

    fn run_exclusive(&mut self, i: usize) {
        let system = self.systems[i].take().expect("a system runs once");
        let this_run = self.take_tick();
        let cell = self.cell.0;
        // Nothing else runs, so this code holds the whole world.
        // SAFETY: the caller checked that nothing is running.
        unsafe { cell.world_mut() }.advance_change_tick_to(self.next_tick);
        let result = catch_unwind(AssertUnwindSafe(|| {
            // SAFETY: as above; an exclusive system takes the whole world.
            unsafe { system.run_unchecked(cell, this_run) }
        }));
        self.systems[i] = Some(system);
        match result {
            Ok(()) => self.finish(i, false),
            Err(payload) => {
                self.phase[i] = Phase::Finished;
                self.panic = Some(payload);
            }
        }
    }

    /// Records that `system` is over: it ran (`ran`) or was skipped.
    fn finish(&mut self, system: usize, ran: bool) {
        self.running.retain(|&r| r != system);
        self.phase[system] = Phase::Finished;
        self.finished += 1;
        if ran && self.plan.has_deferred[system] {
            self.pending.push(system);
        }
        for k in 0..self.plan.successors[system].len() {
            let next = self.plan.successors[system][k];
            self.waiting_for[next] -= 1;
            if self.waiting_for[next] == 0 {
                self.phase[next] = Phase::Reached;
                self.reached.insert((self.plan.rank[next], next));
            }
        }
    }

    /// Applies every system's deferred work, in the schedule's order.
    /// Nothing may be running.
    fn apply_pending(&mut self) {
        if self.pending.is_empty() {
            return;
        }
        let rank = &self.plan.rank;
        self.pending.sort_by_key(|&i| rank[i]);
        let pending = std::mem::take(&mut self.pending);
        let cell = self.cell.0;
        // SAFETY: nothing is running (the caller's contract), so this code
        // holds the whole world.
        let world = unsafe { cell.world_mut() };
        world.advance_change_tick_to(self.next_tick);
        let result = catch_unwind(AssertUnwindSafe(|| {
            for &i in &pending {
                self.systems[i]
                    .as_mut()
                    .expect("a finished system is back")
                    .apply_deferred(world);
            }
        }));
        if let Err(payload) = result {
            self.panic = Some(payload);
        }
    }
}

/// One system's run, on a pool thread.
fn run_task<'a, 's>(
    scope: &rayon::Scope<'s>,
    shared: &'s Mutex<Run<'a>>,
    i: usize,
    system: &'a mut Box<dyn System>,
    cell: SendCell<'a>,
    this_run: Tick,
) where
    'a: 's,
{
    let result = catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: the scheduler started this system only after checking that
        // nothing running conflicts with its access, and nothing that does
        // starts before it is done.
        unsafe { system.run_unchecked(cell.0, this_run) }
    }));
    let mut run = lock(shared);
    run.systems[i] = Some(system);
    match result {
        Ok(()) => run.finish(i, true),
        Err(payload) => {
            run.running.retain(|&r| r != i);
            run.phase[i] = Phase::Finished;
            if run.panic.is_none() {
                run.panic = Some(payload);
            }
        }
    }
    run.advance(scope, shared);
}

impl Schedule {
    /// What follows the parallel part of a run: the world's tick, then the
    /// sweep of stored ticks, as for the single-threaded executor.
    fn finish_run(&mut self, world: &mut crate::world::World) {
        let now = world.change_tick();
        for node in self.systems.iter_mut() {
            node.system.check_change_tick(now);
            for condition in &mut node.conditions {
                condition.check_change_tick(now);
            }
        }
        for set in self.sets.iter_mut() {
            for condition in &mut set.conditions {
                condition.check_change_tick(now);
            }
        }
    }
}
