# mid-ecs schedule, phase 2: threads

Part of the schedule design (`docs/mid-ecs-schedule.md`); the main file is
`docs/mid-ecs.md`. **Status: phase 2 is built**: `System` and everything it is
made of is `Send`, systems say whether they defer work, and a schedule can run on
the rayon pool. Phases 3 and 4 (the `mid-app` side; events, observers, states) are
unchanged and still design.

## Overview

A schedule has two executors, chosen per schedule with `Schedule::set_executor`:

- `ExecutorKind::SingleThreaded` (the default, and what phase 1 was): one system at
  a time, in order, on the calling thread, whatever a system defers applied before
  the next starts.
- `ExecutorKind::MultiThreaded`: on the rayon pool, as many systems at a time as the
  pool has threads and their access sets allow. On `wasm32`, which has no pool, it
  runs as `SingleThreaded`.

`Schedule::set_thread_pool(Some(pool))` chooses the pool; without it the global
rayon pool is used. Nothing about how a system is written changes between the two,
which is the point: a schedule can be switched, and a system that is correct one way
is correct the other, **if it is ordered against the systems it conflicts with**
(`Schedule::ambiguities` finds the ones that are not).

The default stays single-threaded because it is the stronger, simpler guarantee
(deferred work visible to everything after it, not only to what is ordered after),
and flipping it is a one-line change once the schedules in use have been checked
with `AmbiguityDetection::Error`.

## What had to become `Send`

A schedule that may move a system to another thread needs the system, its state,
and everything it hands the function to be fit for that. The changes, all breaking:

- `System: Send + 'static`. `SystemParamFunction<Marker>: Send + 'static`;
  `FnMut(&mut World)` closures used as exclusive systems must be `Send`.
- `SystemParam::State: Send + 'static`, so `Local<T>` needs `T: Send`.
- **`SystemParam::Item<'w, 's>: Send`.** This is the soundness bound. The item is
  what touches data, on the thread that runs the system, while other systems run
  on other threads. `Res<T>` is `Send` only if `T: Sync` (several systems may
  hold `&T` at once), `ResMut<T>` only if `T: Send`, a query only if its fetch is
  `Send` (the bound `par_iter` already used). So `Res`, `Option<Res>` need
  `T: Sync`, `ResMut`, `Option<ResMut>` need `T: Send`, and `Query<..>` needs
  `for<'a> Q::Fetch<'a>: Send`; a system over a type that fails is not a system
  (`IntoSystem` is not implemented), checked by six `compile_fail` doctests on
  `System`. A first attempt put the bound on the function impls as a higher-ranked
  `for<'w, 's> SystemParamItem<'w, 's, P>: Send`; rustc cannot prove that through
  the associated type, so it lives on the associated type itself.
- `Query` gets `unsafe impl Send` where `Q::Fetch<'a>: Send`: it holds a world
  handle, which is not `Send` by default; the bound says the component values it
  can reach may be touched from another thread.
- `CommandQueue` is `Send`: queued closures are `FnOnce(&mut World) + Send`, and
  `spawn_bundle`, `insert_bundle`, `insert_resource` and `insert_static` need `Send`
  values.
- `Condition: Send + 'static`; schedule labels and sets are `Send + Sync`
  (`ScheduleLabel` is blanket-implemented only for such types, `SystemSet` requires
  them), so `Schedule` and `Schedules` are `Send`.
- `System::has_deferred()` is a new required method (breaking for outside
  implementors; the two in the repo are updated). `SystemMeta::set_has_deferred` is
  called by `Commands::init_state`; it is meaningful once the system is
  initialized, which a schedule always does first.

The existing `compile_fail` doctests on `par_iter` for non-`Send` components now
fail one step earlier (the query is not a parameter at all); their text says so.

## The executor

There is no coordinator thread. The state of the run is one value (`Run`) behind a
mutex. The caller makes the first call to `advance`, which starts everything that
can start; each system runs as a task on the pool, and when it finishes it takes
the lock, records itself done, calls `advance` again and starts whatever it freed.
Nothing waits for another task, so a pool of one thread works, and so does running
a schedule from inside a task of the same pool (both tested).

For a system whose predecessors are all done, `advance` applies four rules in
turn:

1. **Pending deferred work.** If any finished system that deferred work is ordered
   before it (checked against the reachability matrix), the work must be applied
   first. Applying needs the whole world, so nothing new starts until everything
   running has finished; then all pending work is applied, in run order, and the
   run continues. This is the sync point, found when needed rather than placed in
   the graph in advance (the design described inserted nodes; the effect on
   systems ordered after a deferring system is the same, and no node exists to
   schedule).
2. **Conditions.** The conditions of the sets above it, then its own, are
   evaluated on whichever thread is advancing the run, once nothing running
   conflicts with what they read (the union of all of them, `gate_access`, is
   checked first). Each set's conditions are evaluated once per run, as in
   phase 1; a system turned away by a set does not consult its own conditions; a
   false one skips the system, which counts as done for what waits for it.
3. **Room.** Nothing running conflicts with the system's access
   (`FilteredAccessSet::is_compatible`).
4. **Exclusive systems** need nothing running at all, run on the advancing thread,
   and nothing later in the run order starts until they have.

Ready systems are considered in run order, so the schedule is as deterministic as
the timing allows: which of two unordered, non-conflicting systems starts first is
up to the pool; which of two systems that conflict starts first is the lower run
order position unless a gate or a sync point holds it.

## Ticks

The world's tick is a plain `u32` that systems cannot advance from inside. The
executor counts instead: every condition evaluation and every system run takes the
next number, in the order they start, and the number is the `this_run` it runs
with. The world's tick is set to the count (by `World::advance_change_tick_to`,
which also runs the wraparound check when due) whenever the executor holds the
whole world: before an exclusive system, before deferred work is applied, and at the
end. So work done through `&mut World` is stamped with a tick no later than what
starts after it and no earlier than what finished before it, as in phase 1.

For a schedule whose conflicting systems are all ordered, the ticks handed out, the
final world tick and everything change detection reports are the same as the
single-threaded executor's (tested over random schedules, and for `Added` and
`Changed` across deferred work and exclusive systems). The world's own tick is
constant while systems run in parallel; code that reads it from inside a system
(rather than the `this_run` it was given) sees the value from the last sync.

## Differences from the single-threaded executor

- A system sees what a system **ordered before it** deferred. It may or may not see
  what an unordered one deferred (single-threaded: it always does, once that one
  has run).
- Unordered systems that do not conflict run at the same time; their relative start
  order and tick numbers vary run to run.
- Unordered systems that conflict run in some order, one at a time; use
  `ambiguities` to find them.
- Conditions are evaluated when the system is first able to run (all its
  predecessors done and no running system in the way of what they read), not at the
  position in a total order; a condition that reads what an unordered system
  writes can see either value.
- A panic in a system, a condition, an exclusive system or an applied command stops
  new work from starting, lets the running systems finish, and then comes out of
  `run` with the original payload. The schedule is usable again (systems are
  returned to it before the panic is re-raised; ticks handed out are kept).

## The unsafe code, and why it holds

Three kinds of `unsafe` carry the executor:

- `SendCell`, a world handle that may cross to another thread. It is only given to
  a task for a system whose access nothing running conflicts with, and the whole
  world is used (for `world_mut`) only when nothing is running.
- `run_unchecked` on a system or condition, whose contract is that nothing
  conflicting with its access runs meanwhile. Rule 3 (and the gate for
  conditions) is that check. A system's `running` entry is added before its task
  is spawned and removed, under the lock, after it has finished and handed its
  state back.
- `world_mut`, in `apply_pending` and for exclusive systems, only reached when
  `running` is empty, under the lock, so no task can start meanwhile.

Everything that decides what may start happens under the one lock, so the order
of completions and starts is a total order with the lock's happens-before between
them. `Send` on parameter items and system state (above) is what makes handing the
data to another thread sound at all.

What checked it: the random schedule tests (below) look for ordering and overlap
violations and compare with the single-threaded executor; the parallel tests ran
40 times in a loop without a failure (all 42 of them); the two-systems-that-need-each-other test
fails if the systems do not really run at the same time. A data race detector was
not available: valgrind's DRD reports every access guarded by `std`'s futex-based
mutex as a conflict (it cannot see those locks), so its output on this code is
noise (hundreds of reports inside `std` and in code under the mutex); Miri and
ThreadSanitizer need a nightly compiler the sandbox does not have. Run them in CI
if a nightly is available.

## Implementation, phase 2

### parallel.rs

`ParallelPlan` is built with the schedule (only when the executor is
multi-threaded): successors and predecessor counts from the graph's edges, the
reachability matrix, each system's rank in the run order, its access, the access
its conditions and its sets' conditions need, whether it is exclusive and whether it
defers. `Run` is the state described above; `Phase` is `Waiting`, `Reached`,
`Cleared`, `Running`, `Finished`. `advance`, `evaluate`, `dispatch`,
`run_exclusive`, `finish` and `apply_pending` are its steps; `run_task` is what a
pool thread does. `Schedule::execute_parallel` borrows the systems and conditions
out of the schedule as separate `&mut`s (so a task can own its system), builds the
`Run`, and uses `in_place_scope` on the chosen pool so the caller's thread starts
the run and the closure need not be `Send`.

### tests_parallel.rs

Random schedules (access patterns over four resources, random orderings) checked
for every ordering, no overlap of conflicting systems, each system once, and, when
every conflict is ordered, the same final resource sequences as the single-threaded
executor; concurrency that cannot be faked (two systems that can only finish by
running at once); deferred work (sync before a dependent, the diamond, applied in
run order not completion order, last system's work, exclusive after deferral,
`Added` across runs); exclusive systems that overlap nothing; conditions (skip,
once per run for racing members, not reading what a running system writes, for a
system's own and for a set's); ticks (same as single-threaded, one per condition
and system, swept after the run, exclusive writes counted as changed); panics in a
system and in a condition; a pool of one thread and running inside the pool; the
global pool; switching executors; a few hundred systems in phases.

### Other files

- `system.rs`: the bounds above; `System::has_deferred`; the `Send` doctests.
- `system_param.rs`: `SystemMeta::has_deferred`/`set_has_deferred`, the `Item`
  bound and the impl bounds, `Query: Send`, `Commands` sets the flag.
- `commands.rs`: `Send` on queued closures and on the values that go in them.
- `world.rs`: `advance_change_tick_to`.
- `schedule/mod.rs`: `ExecutorKind`, `set_executor`, `set_thread_pool`, the plan in
  the build; `executor.rs` chooses the executor.
- `schedule/label.rs`, `condition.rs`: `Send + Sync` labels and sets, `Send`
  conditions.

## Test results

Rustc 1.91 in the sandbox, on one core (the pool's threads are real OS threads,
so the concurrency tests do run side by side):

| | before phase 2 | after |
|---|---|---|
| `mid-ecs` lib tests | 924 | 971 (224 in `schedule`, of which 42 in `tests_parallel`) |
| with `--features scratch-arena` | 929 | 976 |
| doctests | 26 | 33 (six `compile_fail` for the `Send` rules and one passing example, all on `System`) |
| `ffi::` | 103 | 103 |

The six `compile_fail` doctests on `System` were each checked in a scratch example
to fail for the intended reason (E0277, or E0599 when written as a method call)
with a compiling control.

All 42 parallel tests were run 40 times in a loop without a failure.

Mutation testing, 45 mutants over `parallel.rs` and the pieces it relies on
(`has_deferred`, `advance_change_tick_to`, the executor switch): every mutant is
caught except two that change nothing anyone can observe: a neutral control, and
removing the update of the running list after a panic (nothing reads that list once
a run has panicked). A few of the mutants deadlock or spin and are "caught" by a timeout.
The first pass missed, and tests were added for:

- the tick sync before an exclusive system, and before deferred work, which needs
  a reader that ran earlier in the same run (so that its last run is later than a
  stale stamp);
- a system in two conditioned sets needing both answers;
- applying deferred work while another system was still running, which needs a
  command that checks nothing else is active;
- apply order by run order when completion order differs, and an exclusive system
  against a system that touches nothing (whose access conflicts with nothing);
- no new system starting after a panic, for a system that was waiting for room and
  for one that was waiting for its predecessor;
- the sweep of stored ticks reaching conditions that were never evaluated in a
  parallel run;
- a panic in deferred work and in an exclusive system coming out of `run`.

Not run: Miri, ThreadSanitizer, any other data race detector (see "The unsafe code,
and why it holds"), the wasm32 build (the `cfg(not(target_arch = "wasm32"))` code
is written but not compiled for that target), real CI on the newest stable rustc.

## Fixes and Problems

- **The higher-ranked `Send` bound did not work** (see "What had to become
  `Send`"): the bound is on the associated type.
- **The first draft of the `Send` unit tests found labels and sets were not
  `Send`** (a boxed `dyn` value inside the `Schedule`), so `Schedule` was not
  `Send` and could not be moved to a thread. Labels and sets now require
  `Send + Sync`.
- **`cargo fmt` reformats code that was not touched.** The repository is not clean
  under the sandbox's rustfmt (1.91); only the new files are formatted, and
  unrelated reformatting was reverted by hand. Check the formatting of the new
  files with the CI toolchain.
- **A data race detector was not available** (see above).
- **Open:** the default executor is single-threaded by choice, not by limitation;
  a system that calls `par_iter` inside a multi-threaded schedule shares the same
  pool with the other systems and has not been measured or tuned; `mid-app`
  (phase 3) does not use any of this yet; `NonSend` resources (a resource that must stay on one thread) do not exist,
  so types that are not `Send`/`Sync` cannot be used by systems at all.
