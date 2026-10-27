# mid-ecs: schedule (design part)

This is a separate part of the `mid-ecs` documentation (see "Large crates:
splitting by part" in `docs/DOCUMENTATION_AND_COMMENTING_GUIDELINES.md`). The
main file is `docs/mid-ecs.md`. **Status: design only, nothing is built.** When
phase 1 lands, the sections below that describe code become the per-file
sections of the implementation, and the decisions stay.

## Overview

Systems exist (`System`, `IntoSystem`, `SystemParam`, `Commands`, access sets)
and nothing runs them in an order. `mid-app` runs a fixed list of
`Box<dyn FnMut(&mut World)>` in five hard-wired stages, and says in its own header
that it is waiting for the ECS system-parameter convention, which now exists. The
gap-list item is *the schedule*: declare systems, say what must run before what,
say when a system should be skipped, and have something run them correctly.
`bevy_ecs`'s version of this is about 16,000 lines including tests and features
this engine does not need; this part describes the smaller thing that covers what
a game or a server loop actually uses.

It is also the keystone for three other open items, which is why it comes next:

- nothing yet calls `System::check_change_tick` for you (the tick-wraparound work
  made that the owner's job);
- the removal log is never trimmed (it needs a frame owner that knows the oldest
  `last_run` of every reader);
- events and observers need a per-frame update to run somewhere.

## What already exists (verified against the code)

| piece | where | what a scheduler can build on |
|---|---|---|
| `System` | `system.rs` | `name`, `initialize`, `access() -> &FilteredAccessSet`, `is_exclusive`, `unsafe run_unchecked(WorldCell, this_run)`, `apply_deferred`, `check_change_tick`, and `run` (initialize, run at the world's tick, advance it, apply deferred) |
| `IntoSystem` | `system.rs` | for a `System`, for a function of system parameters, and for `FnMut(&mut World)` (so `mid-app`'s existing closures become exclusive systems unchanged) |
| conflict test | `access.rs` | `FilteredAccessSet` / `Access::get_conflicts` and `is_compatible`; the conflicts carry component and resource *names*, enough for an ambiguity report |
| deferred work | `system_param.rs`, `commands.rs` | `Commands` buffer per system, applied by `apply_deferred` |
| ticks | `world.rs`, `tick.rs` | `System::run` advances the tick once per system run; `World::check_change_ticks` runs itself; a system's own last run is the owner's to clamp |
| resources | `resource.rs` | a `Schedules` collection can live as a resource |

What does **not** exist: any notion of order between systems, labels or sets, a
way to say "skip this system", a `Send` bound on `System` (`System: 'static`
only), and a return value from a system (`SystemParamFunction::run` returns
`()`), which run conditions need.

## Where it lives

**The graph and the executor go in `mid-ecs`; the app loop goes in `mid-app`.**
This is `bevy`'s split (`bevy_ecs::schedule`, `bevy_app::{App, Plugin}`), and it
follows from what the code needs: building a schedule reads each system's access
set and runs systems through `run_unchecked` on a `WorldCell`, both internal to
`mid-ecs`, and it must be testable without a loop. The roadmap's Decision 1
describes `mid-app` as "the schedule-runner"; read that as the *loop driver and
plugin assembly*: `mid-app` keeps the fixed stages (`PreUpdate`, `Update`, `Net`,
`Physics`, `PostUpdate`) and the `Plugin` trait, and its stages become labels of
real schedules instead of `Vec<Box<dyn FnMut>>`.

## The shape

```rust
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct Update;                       // a schedule label: any such type is one
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum Phase { Input, Simulate }       // a system set: any such type is one

let mut schedule = Schedule::new(Update);
schedule
    .configure_sets((Phase::Input, Phase::Simulate).chain())
    .add_systems((
        read_input.in_set(Phase::Input),
        (move_units, apply_damage).chain().in_set(Phase::Simulate),
        trim_log.after(Phase::Simulate).run_if(resource_changed::<Log>),
    ));
world.add_schedule(schedule);
world.run_schedule(Update);          // build on first run, then execute
```

- **Labels and sets without a derive macro.** `mid-ecs` has no proc-macro crate.
  `ScheduleLabel` and `SystemSet` are blanket-implemented for every
  `Debug + Clone + Eq + Hash + 'static` type, compared through a small
  dyn-equality shim (a boxed value plus a `TypeId`), so a unit struct or an enum
  with the usual `#[derive]` is a label. A system function can be named as a set
  (`after(move_units)`), identified by the function item's type: the set of "the
  system made from that function".
- **Configuration is a builder over a tuple.** `IntoScheduleConfigs` is
  implemented for systems, sets, and tuples of them (by macro, up to a fixed
  arity, as the parameter tuples are), producing a `ScheduleConfigs<T>` with
  `.before(x)`, `.after(x)`, `.in_set(s)`, `.chain()`, `.run_if(cond)`,
  `.ambiguous_with(x)`. A config applies to everything in its tuple, as in bevy.
- **A schedule is built lazily, once, and rebuilt when changed.** Two graphs, as
  in bevy: the *hierarchy* (system in set, set in set; must be acyclic) and the
  *dependency* graph (`before`/`after`, `chain`). Ordering a set before another
  expands to every member of one before every member of the other.
- **Build produces one flat order.** A topological sort of the systems with ties
  broken by insertion order, so a schedule runs the same way every time. The result
  is the system list in that order, plus the set-membership and condition tables
  the executor needs.

## Run conditions (needs one change to `System`)

A condition is a system that returns `bool` and takes only read-only parameters:
`run_if(resource_changed::<Gauge>)` passes a function `fn(Res<Gauge>) -> bool`.
That needs `SystemParamFunction` to carry an output type
(`fn run(...) -> Self::Out`; the existing impls become `Out = ()`), which touches
the macro-generated impls in `system.rs` and nothing else, and a `Condition`
trait (`IntoCondition`, a `ConditionSystem`) alongside `System`. In exchange the
whole `common_conditions` library comes for free and reads like bevy's:
`resource_exists`, `resource_changed`, `resource_added`, `not`, `and`, `or`
(and `on_message` once events exist).

Semantics: a condition runs just before its system, on the same thread, through the
same `this_run` / last-run machinery as a system (so `resource_changed` inside a
condition measures "since this condition last ran"); a system whose condition is
false is skipped and its last run does **not** advance, so what it missed
accumulates and it sees all of it next time it runs. A condition on a *set* is
evaluated once per run and gates every member.

## Execution, phase 1: single-threaded

One executor runs the flat order on the calling thread:

```text
for each system in order:
    evaluate set conditions (once per set per run) and the system's conditions
    if any is false: skip
    else: this_run = world.change_tick(); world.increment_change_tick();
          run (exclusive or not); apply its deferred work immediately
after the loop: system.check_change_tick(world.change_tick()) for every system
```

- **Deferred work is applied right after each system**, not at bevy's sync points.
  This is the simplest correct policy, and it is *stronger* than bevy's: a later
  system sees the earlier one's commands whether or not they were ordered, where
  bevy only promises it after a sync point. Nothing that is right under bevy's rule
  is wrong under this one. Phase 2 (parallel) needs real sync points; see below.
- **Tick wraparound closes here.** Every run sweeps `check_change_tick` over every
  system and condition: one comparison each, once per schedule run.
- **Removal-log trimming** needs the oldest last run of all readers, which only the
  owner of the systems can compute: the schedule exposes
  `Schedule::oldest_last_run()` and `mid-app`'s main loop passes it to
  `World::trim_removals`. A `ChangeTracker` held outside the schedule is that
  holder's to include.

## Ambiguity and errors

- **Build errors** (a `ScheduleBuildError`, returned by `try_run` / `build`, and a
  panic with names from `run`): a dependency cycle (every cycle, with system and
  set names), a hierarchy cycle (a set containing itself), and ordering a set
  against one of its own members (which can never be satisfied).
- **Ambiguity detection**: two systems *conflict* if their access sets are
  incompatible (an exclusive system conflicts with everything) and neither
  reaches the other in the dependency graph, directly or transitively. A
  schedule has a setting, `Ignore` (the default), `Warn` or `Error`, and an
  `ambiguities()` query that returns the pairs and the component and resource names
  that clash (`AccessConflicts::names()` already provides them). `ambiguous_with`
  silences a pair. This matters for phase 1 less than for phase 2, because a
  single-threaded schedule is deterministic regardless; it is what makes parallel
  execution safe to reason about, so the query exists from the start.
- A panic inside a system propagates (there is no fallible-system error handler;
  that is a separate gap). The schedule being run is taken out of `Schedules` for the
  duration, so the run holds a guard that puts it back during unwinding rather than
  losing it.

## World integration

`Schedules` is a resource (a map from label to `Schedule`). `World` gets
`add_schedule`, `run_schedule(label)`, `schedule_scope(label, |world, schedule| ..)`
(the take-out and put-back that `run_schedule` is built on, exposed because
`mid-app`'s `Main` schedule needs to run others from inside itself), and `init_schedule`
for adding systems without owning the `Schedule`. A `Main` schedule that runs
labelled schedules in order is `mid-app`'s, not `mid-ecs`'s.

## Phases

1. **Phase 1, single-threaded schedule** (build now): labels and sets, the config
   builder, run conditions (with the `SystemParamFunction` change), graph build with
   cycle and ordering errors, ambiguity query, the single-threaded executor,
   `Schedules`, the `World` methods, the tick sweep. New module `schedule/`:
   `label.rs`, `config.rs`, `condition.rs`, `graph.rs`, `executor.rs`, `mod.rs`.
   Estimate: 3,000 lines with tests, two working sessions. No FFI surface yet.
2. **Phase 2, parallel executor.** Needs: `Send` on `System` (see decision 5),
   sync points inserted automatically between a system with deferred work and
   anything ordered after it, a per-system tick assigned by the executor before
   the parallel section (the world's tick is a plain `u32`, so systems cannot
   advance it from inside), and a conflict-aware scheduler over `rayon` (already a
   dependency off wasm32, used by `par_iter`; the wasm32 path would run the
   schedule on one thread, as `par_iter` runs its batches serially there). The
   ambiguity query is the correctness argument for it.
3. **Phase 3, `mid-app` on top.** `App`, `Plugin`, `Main`, the fixed stages as
   labels, the loop with `mid-time`'s fixed timestep, the trim call. Separate crate,
   small.
4. **Phase 4, events/messages and observers** (the rest of gap-list item 6) and
   states. Both need a schedule to run their update.

## Decisions, with the recommendation taken

1. **Where:** graph and executor in `mid-ecs`, loop and plugins in `mid-app`
   (above). Alternative: everything in `mid-app`. Rejected: it would have to reach
   `mid-ecs` internals, and could not be tested without a loop.
2. **Labels and sets via blanket impls, no derive macro.** Alternative: a proc-macro
   crate. Rejected: a new crate and a build-time cost for something a blanket impl
   does.
3. **Conditions are bool-returning systems with read-only parameters**, which needs
   `SystemParamFunction` to have an output type. Alternative: conditions as closures
   over `&World`. Rejected: they cannot take `Res<T>` or a `Query`, so
   `resource_changed` and every condition like it would need a bespoke form.
4. **Phase 1 applies deferred work after every system.** Alternative: bevy's sync
   points from the start. Deferred: they matter only once systems run in parallel,
   and doing them first adds a graph pass with nothing to test it against.
5. **`System: Send` arrives with phase 2**, not before. Today a system may hold
   something non-`Send`, and phase 1 runs on one thread. Adding the bound is a
   breaking change for any such system, so it is made once, with the executor that
   needs it, and with a compile-fail test.
6. **Ambiguity detection defaults to `Ignore`**, with the query always available.
   Alternative: warn by default. Rejected: a single-threaded schedule is
   deterministic, and the engine has no logging dependency in `mid-ecs` to warn
   through.
7. **Out of scope, named so it is not forgotten:** states (`bevy_state`'s role),
   stepping and debug tooling, schedule build passes (plugin hooks into the build),
   removing systems from a built schedule, fallible systems and an error handler,
   a C surface (C cannot register a Rust system; a later decision is whether C can
   run a *named* schedule).

## Risks

- The `SystemParamFunction` change is mechanical but touches every arity of the
  macro-generated impls; mutation testing of the existing system tests is the check.
- Set ordering expands to a product of edges, quadratic in the worst case
  (two large sets ordered against each other); at engine scale (hundreds of
  systems) this is fine, and bevy does the same expansion before reducing.
- Condition ticks: evaluating a condition advances the world tick like a system
  run, which makes the wraparound clock tick faster (already true per system; see
  "Tick wraparound" in `docs/mid-ecs.md`).

## Test plan for phase 1

Property tests against a brute-force oracle: generate random acyclic dependency
sets over random systems, build, and check every declared `before`/`after` holds in
the resulting order, the order is deterministic, and every cycle is reported (cycles
generated by adding one back edge); set expansion checked against hand-expanded
edges; conditions checked for skip, accumulation of missed changes, and set-level
evaluation once per run; the ambiguity query checked against a brute-force pairwise
reachability and conflict check; the tick sweep checked by clamping a system's last
run; mutation testing of the graph build and the executor, as for the other parts.

## Fixes and Problems

None yet: nothing is built.
