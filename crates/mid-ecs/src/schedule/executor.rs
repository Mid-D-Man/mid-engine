// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-ecs-schedule.md, section "executor.rs"
// ============================================================================
//! Running a built schedule on the calling thread.
//!
//! For each system in order: decide whether it runs (the conditions of the
//! sets above it, each set's evaluated at most once per run, then the
//! system's own), run it with the same tick bookkeeping as
//! [`crate::System::run`] (a fresh tick, then whatever it deferred applied),
//! and move on. After the last, every stored tick is kept within range of
//! the world's.

use crate::schedule::condition::BoxedCondition;
use crate::schedule::Schedule;
use crate::world::World;

/// Runs every condition (none is skipped because another said no, so each
/// keeps its own record of what it has seen) and joins the answers.
fn evaluate(conditions: &mut [BoxedCondition], world: &mut World) -> bool {
    let mut all = true;
    for condition in conditions {
        all &= condition.run(world);
    }
    all
}

impl Schedule {
    /// Runs every system once. The schedule must be built.
    pub(super) fn execute(&mut self, world: &mut World) {
        let Schedule {
            systems,
            sets,
            built,
            set_results,
            ..
        } = self;
        let built = built.as_ref().expect("built before it is run");
        set_results.clear();
        set_results.resize(sets.len(), None);

        for &i in &built.graph.order {
            let mut run = true;
            // Every set above the system answers, even after one has said
            // no: a set's condition is evaluated the first time any of its
            // members is reached, whatever the other sets say.
            for &s in &built.condition_sets[i] {
                let answer =
                    *set_results[s].get_or_insert_with(|| evaluate(&mut sets[s].conditions, world));
                run &= answer;
            }
            let node = &mut systems[i];
            // A system that a set has turned away does not consult its own
            // conditions, so what they would have seen is still there when
            // it is let back in.
            if run {
                run = evaluate(&mut node.conditions, world);
            }
            if run {
                node.system.run(world);
            }
        }

        let now = world.change_tick();
        for node in systems.iter_mut() {
            node.system.check_change_tick(now);
            for condition in &mut node.conditions {
                condition.check_change_tick(now);
            }
        }
        for set in sets.iter_mut() {
            for condition in &mut set.conditions {
                condition.check_change_tick(now);
            }
        }
    }
}
