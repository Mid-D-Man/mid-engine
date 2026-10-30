# mid-ecs: schedule (design part)

This is a separate part of the `mid-ecs` documentation (see "Large crates:
splitting by part" in `docs/DOCUMENTATION_AND_COMMENTING_GUIDELINES.md`). The
main file is `docs/mid-ecs.md`. **Status: phase 1 is built** (labels, sets, the
config builder, run conditions, the graph, a single-threaded executor, `Schedules`
and the `World` methods). Phases 2 to 4 are still design. The sections down to
"Test plan for phase 1" are the design as agreed; where the code differs, the
text says "As built". The per-file sections after the test plan describe what is
actually in `crates/mid-ecs/src/schedule/`.

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
  `ScheduleLabel` is blanket-implemented for every
  `Debug + Clone + Eq + Hash + 'static` type, compared through a small
  dyn-equality shim (a boxed value plus a `TypeId`), so a unit struct or an enum
  with the usual `#[derive]` is a label. **As built, `SystemSet` is not
  blanket-implemented**: it is the same bounds with an empty opt-in impl
  (`impl SystemSet for Phase {}`), because a blanket impl makes a tuple of two
  enums a set as well as a tuple of sets, and `(Phase::Input, Phase::Simulate).chain()`
  becomes ambiguous (see "Fixes and Problems"). A system function can be named as a
  set (`after(move_units)`), identified by the function item's type: the set of "the
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

As built, there is no `ConditionSystem` type. `Condition` is its own trait, and
`FunctionSystem` implements it when the function returns `bool` (and `System` when
it returns `()`), so a function condition is the same struct a system is. Reading
only is enforced when the condition is initialized, by checking that the
combined access writes nothing (`Access::is_read_only`); a `Commands` parameter
passes that check, and what it queues is never applied.

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

As built, the condition rules in that sketch are: every set above a system answers
(each set once per run, the first time any of its members is reached, and none is
skipped because another said no); then, only if all of them said yes, the
system's own conditions answer, all of them (none is skipped because another said
no). A system a set has turned away does not consult its own conditions, so what
they would have seen is still there when the set lets it back in. bevy evaluates a
system's own conditions even then; this is deliberate and the one place the
executor reads differently.

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
  against one of its own members (which can never be satisfied). As built the
  last is the general case "both sides of an ordering contain the same system"
  (`OverlappingOrder`), which also covers a thing ordered against itself and two
  sets that overlap.
- **Ambiguity detection**: two systems *conflict* if their access sets are
  incompatible (an exclusive system conflicts with everything) and neither
  reaches the other in the dependency graph, directly or transitively. A
  schedule has a setting, `Ignore` (the default), `Warn` or `Error`, and an
  `ambiguities()` query that returns the pairs and the component and resource names
  that clash (`AccessConflicts::names()` already provides them). `ambiguous_with`
  silences a pair. This matters for phase 1 less than for phase 2, because a
  single-threaded schedule is deterministic regardless; it is what makes parallel
  execution safe to reason about, so the query exists from the start. As built,
  `Warn` prints each pair to standard error with `eprintln!` (there is no logging
  dependency), `Error` makes the build fail with `ScheduleBuildError::Ambiguities`,
  and an exclusive system conflicts with every system that touches *anything*
  (a system with no parameters at all conflicts with nothing, since there is
  nothing for the order to matter to).
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
2. **Labels and sets without a derive macro.** Alternative: a proc-macro
   crate. Rejected: a new crate and a build-time cost for something a trait with
   provided methods does. **As built:** `ScheduleLabel` is a blanket impl as
   agreed; `SystemSet` is an opt-in empty impl instead, because the blanket impl
   cannot coexist with tuples of sets in the config builder (E0283, found while
   building; see "Fixes and Problems"). The cost is one line per set type.
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

## Test plan for phase 1, and what it found

Done as planned. Property tests against a brute-force oracle (`graph.rs`):
random acyclic orderings checked for every declared `before`/`after`, for the
exact order (the lexicographically smallest topological order, computed the slow
way), for determinism and for reachability against a Floyd-Warshall closure; one
random back edge injected per case and the reported cycles compared with the
oracle's strongly connected components; random sets with random orderings,
checked for the expected overlap errors, cycles, expanded edges, each set's
systems and each system's sets; random hierarchies checked for cycles; graphs
wider than a machine word for the reachability matrix; and a chain, a hierarchy
and a cycle of twenty thousand nodes to show nothing recurses. End-to-end tests
cover the rest (see "tests.rs" below).

Results, rustc 1.91 in the sandbox:

| | before | after |
|---|---|---|
| `mid-ecs` lib tests | 742 | 924 (181 in `schedule`, 1 in `access`) |
| with `--features scratch-arena` | 747 | 929 |
| doctests | 19 | 26 (four examples, three `compile_fail`) |

The three `compile_fail` doctests (a set passed to `add_systems`, a system passed
to `configure_sets`, a non-set passed to `in_set`) were each checked in a scratch
example to fail with E0277 for that reason, with a passing control.

Mutation testing, about a hundred mutants over `graph.rs`, `mod.rs`,
`executor.rs`, `config.rs`, `condition.rs`, `label.rs`, `Access::is_read_only` and
`FunctionSystem`: every mutant is caught except four that change nothing anyone
can observe. Two in `graph.rs` (removing the "advance the second index" step in
`intersect`, which sorted unique input makes redundant, and removing the dedup of
child and parent lists in the hierarchy, which the later per-set dedup makes
redundant) and two in `mod.rs` (skipping the "already built" early return in
`prepare`, and the filter that keeps only sets that carry conditions): the last
two change how much work a run does, not what it does. They were left in. What
the first pass missed, and tests were added for:

- the reachability matrix only ever had bits below 32 of a word checked (`graph.rs`);
- the tick sweep after a run reaching system conditions and set conditions that
  were never evaluated that run, which needs a system turned away by its set and
  a set with no members;
- `AndCondition`/`OrCondition::check_change_tick` reporting a clamp when only the
  half that was skipped was out of range;
- `Access::is_read_only` for each whole-world flag on its own.

Not run here: the C smoke test and valgrind on the schedule tests are part of the
verification of the delivery, not of this section; real CI on the newest stable
rustc, Miri, and wasm32 are not run.

## Implementation, phase 1

Source: `crates/mid-ecs/src/schedule/`. The module is public as
`mid_ecs::schedule`. Each file below starts with a notice pointing at its section
here.

### label.rs

`ScheduleLabel` (blanket, see above) and `SystemSet` (opt-in) both reduce to a
boxed value that can be cloned, compared and hashed without its type: an
`Interned` over a private `DynValue` trait, where equality needs the same
`TypeId` on both sides and the hash includes it. `LabelKey` and `SetKey` wrap
that. A `LabelKey` is itself a `ScheduleLabel` and `label_key()` on it returns
a clone, so passing a stored key where a label is expected does not nest one key
in another.

A `SetKey` is a user set, "the system made from this function" (the `TypeId` and
name of `S::System`), or an anonymous set the builder made (a number unique within
one schedule). `IntoSystemSet<Marker>` is what `before`, `after` and
`ambiguous_with` take: a `SystemSet` value, or anything that is `IntoSystem`.
A function item does not implement `SystemSet`, so it matches only the second
impl and the call is not ambiguous.

The two key methods are named `label_key` and `set_key`, not both `intern`, because
a type that is both a label (everything is) and a set would otherwise have two
methods of one name in scope and every call would be ambiguous.

### config.rs

`ScheduleConfigs<T>` is a tree: a leaf (`ScheduleSystem` for systems, `SetKey`
for sets) or a group, each with the settings given to it (`in_set`, `before`,
`after`, `ambiguous_with`, conditions) and, for a group, whether it is chained.
`IntoScheduleConfigs<T, Marker>` converts a `ScheduleConfigs`, a set, a system,
or a tuple of up to twelve of them, and carries the same methods so they can be
called on anything that converts. The markers are distinct types (`IsConfigs`,
`IsUserSet`, `(IsSystemTypeSet, M)`, `(IsTuple, (M0, ...))`) so no two impls
overlap.

`flatten` reduces the tree to leaves with everything their groups said about them
(`in_sets`, `before`, `after` and `ambiguous_with` are inherited outside in), the
chain orderings as pairs of leaf-index lists (every leaf of the first before every
leaf of the second; a member with no leaves is skipped so it cannot break a
chain), and the anonymous sets. A condition on a leaf stays on the leaf. A
condition on a group makes one anonymous set that carries it, and every leaf of
the group lists that set in `in_sets`, which is what makes the condition run once
per run for the whole group. The counter that numbers anonymous sets belongs to
the schedule.

### condition.rs

See "Run conditions" above for the rules. `Condition` mirrors `System`
(`initialize`, `access`, `run_unchecked`, `check_change_tick`, `run`) with one
difference: `oldest_last_run(now)` instead of a single last run, because a
combined condition holds two. `run` initializes if needed, evaluates at the
world's tick, and advances it by one, as `System::run` does.

`not`, `and` and `or` build `NotCondition`, `AndCondition` and `OrCondition`.
`and` and `or` short-circuit, so the second half's last run does not advance when
it is skipped and what it missed accumulates, as for a skipped system; both
halves' access is merged into one set at initialization, and
`check_change_tick` clamps both halves without short-circuiting.
`common_conditions` has `resource_exists`, `resource_added`, `resource_changed`
and the two `resource_exists_and_*` forms that are `false` instead of panicking
when the resource is missing, plus re-exports of `not`, `and` and `or`.

### graph.rs

Pure algorithms over indices and names, so the whole file is tested against brute
force without a single system. `build` takes the set membership list and the
ordering list and returns the run order, each set's systems, each system's
sets, and the system-to-system edges, or the first group of errors that applies:

1. the hierarchy is checked for cycles (Tarjan, iterative, so a chain of twenty
   thousand sets does not overflow the stack) and each set's systems are
   collected children first;
2. every ordering is expanded to the product of its two sides' systems; if the
   two sides share a system the ordering is `OverlappingOrder` and none of the
   orderings is expanded;
3. Kahn's algorithm with a min-heap on the system number, so the lowest-numbered
   ready system always goes first (this is what makes the order identical on every
   build, and it equals the lexicographically smallest topological order, which
   is what the oracle test checks);
4. if systems are left over, every strongly connected component is reported with
   the orderings, as asked for, that join its members.

`Built::reachability` (which system must run before which, transitively) is a
bit matrix computed on request, because it costs a bit per pair of systems and
only the ambiguity query uses it.

### executor.rs

One function, `Schedule::execute`, as in "Execution, phase 1" above. The per-run
set results are a vector reused between runs.

### mod.rs

`Schedule` holds the declared graph (systems with their conditions, sets with
theirs, membership, orderings, silenced pairs) and, once built, the graph result
and for each system the sets above it that carry conditions. Anything that adds
to the schedule discards the build; `prepare` initializes every system and
condition (idempotent, so a rebuild keeps their state: a system's `Local` and
last run survive) and runs the graph build; `build` adds the ambiguity setting on
top. `ambiguities` is a pairwise check over the order's reachability and
`get_conflicts`, with `ambiguous_with` pairs expanded to system pairs and
normalized so the order of the pair does not matter.

`Schedules` is the resource. A schedule being run is taken out of it and its
label recorded as running, so running or replacing it from inside itself is a
`ScheduleRunError::Running` (or a panic from `add_schedule` and
`init_schedule`) rather than a use of a schedule that is not there. `ScopeGuard`
owns the world borrow and the schedule for the closure's duration and puts the
schedule back in `Drop`, so it is back after a panic too.

### tests.rs

End to end: order (by insertion, `chain`, `before`/`after` against functions and
sets, nesting, determinism), sets, errors, conditions (skip, accumulation of
missed changes, once per run, all asked, a set turning a system away), ticks (one
per system run, none for a skip, one per condition, the sweep), deferred work,
exclusive systems and nested schedules, state kept across rebuilds, labels and the
world methods, panic safety, ambiguity (including `ambiguous_with` and the
levels), and a few hundred systems in phases.

### Changes outside `schedule/`

- `system.rs`: `SystemParamFunction` gains `type Out` and `fn run` returns it;
  the macro-generated impls take the output type as part of the marker
  (`fn(P0, ...) -> Out`), so a function returning `()` and one returning `bool`
  have different markers and never overlap. `IntoSystem` and `System` for
  `FunctionSystem` require `Out = ()`. `FunctionSystem` gets crate-private
  `new`, `init_params`, `call`, `check_tick` and friends that both `System`
  and `Condition` share. **`System` gains a required method, `last_run`** (a
  breaking change for any outside implementor; the two in the repo are updated).
- `access.rs`: `Access::is_read_only`.
- `lib.rs`: `pub mod schedule`.

## Fixes and Problems

- **The blanket `SystemSet` could not stay.** With `impl<T: Debug + Clone + Eq +
  Hash + 'static> SystemSet for T`, the tuple `(Phase::Input, Phase::Simulate)` is
  a set (it meets the bounds) and also a tuple of sets, so
  `configure_sets((Phase::Input, Phase::Simulate).chain())` has two applicable
  impls and fails with E0283. It showed up when the tuple impls were written. The
  fix is the opt-in `impl SystemSet for Phase {}`. `ScheduleLabel` keeps the blanket
  impl, since nothing takes a tuple of labels.
- **Two methods named `intern`** (one on each trait) made every call on a type
  that is both a label and a set ambiguous (E0034). Renamed `label_key` and
  `set_key`.
- **The reachability matrix is not part of the build.** The first version built it
  in `build`, a bit per pair of systems, which for twenty thousand systems is
  about fifty megabytes the default `Ignore` setting never reads. It is computed in
  `Built::reachability` when the ambiguity query asks.
- **The world's change tick starts at 1, not 0.** Zero is "never ran". Tests that
  expect "the first run is at tick zero" are wrong; compare against
  `world.change_tick()` taken before the run.
- **Open, not done (phase 2 and later):** `System: Send`, automatic sync points, the
  parallel executor, and the `mid-app` side (`App`, `Plugin`, `Main`, the fixed
  stages as labels, the call to `World::trim_removals` with
  `Schedule::oldest_last_run`). `mid-app`'s closures have not been migrated to
  exclusive systems yet.
