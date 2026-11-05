# mid-app

## Overview

`mid-app` turns a `mid_ecs::World` into a program that runs. An `App` owns a
world; the systems live in schedules in that world, one per stage; `Main` runs the
stages in a fixed order once per frame; a `Plugin` adds systems, sets and
resources to an app so its users do not have to know what a feature is made of.
It is `bevy_app`'s role in Bevy (`docs/bevy-comparison.md` section 2) and phase 3
of the schedule work (`docs/mid-ecs-schedule.md`).

A frame:

```text
PreUpdate -> StateTransition -> Update -> Net -> Physics (0..n times) -> PostUpdate
```

`Physics` is the one stage on a fixed timestep (`mid_time::FixedTimestep`, 60 Hz
by default, at most 8 steps in a frame). Every other stage runs once a frame. After
the frame, the removal log and the message logs are trimmed to what some system still
has to see (messages: `docs/mid-ecs-events.md`).

Before this phase, `mid-app` held `Vec<Box<dyn FnMut(&mut World)>>` in five hard
wired stages and said in its header that it was waiting for the ECS
system-parameter convention. That convention and the schedule now exist, so the
stages became labels of real schedules and the closures became exclusive systems.

## What the crate depends on

`mid-ecs` (the world, schedules and systems) and `mid-time` (`Clock`,
`FixedTimestep`). Both are re-exported (`mid_app::mid_ecs`, `mid_app::mid_time`)
so a plugin crate that depends on `mid-app` gets matching versions.

## Modules

### `lib.rs`

**What it does:** Crate documentation with a runnable example, the module list,
and the re-exports.

**Decisions:**
- No `prelude` yet. The public surface is small enough to name.
- Events, observers and states are not here. They need a schedule to run their
  update in, which exists now, and are phase 4 of the `mid-ecs` work.

**Tests:** the doctest on the crate (a plugin, two frames, a counter).

### `labels.rs`

**What it does:** The six schedule labels: `Main`, `PreUpdate`, `Update`, `Net`,
`Physics`, `PostUpdate`. (`StateTransition`, the seventh schedule of a frame, is defined
in `mid-ecs` with the state types and re-exported by `mid-app`.) Each is a unit struct; `ScheduleLabel` is a blanket
impl in `mid-ecs`, so nothing more is needed.

**Decisions:**
- **The five stages are the ones `mid-app` already had**, in the order it already
  ran them, which is the shape `docs/roadmap.md` (Decision 1) proposed. No
  `First`, `Last` or `Startup` stage. The project rule is to add a stage when
  something real needs one.
- **`Physics` is the fixed stage**, because the performance targets in
  `docs/architecture.md` name physics as the 60 Hz system and `docs/roadmap.md`
  names it as the first real consumer of a fixed timestep. The alternative was a
  separate `FixedUpdate` label next to a per-frame `Physics`, which would leave
  two stages for the same thing. A frame of variable length cannot step
  physics at a variable rate and stay deterministic, which is the point of the
  fixed stage.
- **`Net` runs once a frame**, not at 128 Hz. The 128 Hz network tick needs its
  own accumulator, and what drives `Connection::poll` from here is a decision for
  when `mid-net` is wired in. Nothing here blocks it: a second `FixedTimestep`
  resource and a second loop in `run_stages` is the shape.

### `app.rs`

**What it does:** `App`, `Plugin`, the constants `DEFAULT_FIXED_HZ` (60.0) and
`DEFAULT_MAX_FIXED_STEPS` (8), and `run_stages`, the one system of `Main`.

`App::new` makes the world, inserts a `Clock` and a `FixedTimestep` as resources
(so `Res<Clock>` gives a system the frame's time and `Res<FixedTimestep>` the
step's length), makes an empty schedule for each stage, and gives `Main` its
driver. Then:

| method | does |
|---|---|
| `add_systems(label, systems)` | `Schedule::add_systems` on that schedule, made if missing. Ordering, sets and conditions as in `mid-ecs`. |
| `configure_sets(label, sets)` | `Schedule::configure_sets` likewise. |
| `schedule_mut(label)` | The schedule itself, for `set_executor` and `set_ambiguity_detection`. |
| `insert_resource(value)` | `World::insert_resource`, chainable. |
| `init_state::<S>()` / `insert_state(s)` | Gives the app a state of type `S` (`docs/mid-ecs-events.md`): `Default` or the value given. `State<S>` exists from the first frame, in time for `OnEnter` of the initial value. |
| `add_plugin(plugin)` | Calls `plugin.build(self)` at once. |
| `set_fixed_timestep(hz, max)` | Replaces the timestep; time already accumulated is dropped. |
| `set_trim_removals(on)` / `trims_removals()` | Whether the removal log is trimmed after each frame (on by default). |
| `set_trim_messages(on)` / `trims_messages()` | Whether the message logs are trimmed after each frame (on by default). |
| `update()` | One frame, with the platform clock's delta. |
| `update_with(delta)` | One frame as if `delta` had passed. For tests, replay and lockstep. |
| `run_until(done)` | `update` in a loop until `done` says stop. Not on `wasm32`. |

`run_stages` runs `PreUpdate`, `StateTransition`, `Update`, `Net`, then asks `FixedTimestep` how many
steps the frame's `Clock::delta` made ready and runs `Physics` that many times,
then `PostUpdate`.

**Decisions:**
- **`Main` is a real schedule with one exclusive system**, as `docs/mid-ecs-schedule.md`
  planned ("a `Main` schedule that runs labelled schedules in order is
  `mid-app`'s"), not a loop in `App::update`. The cost is one schedule run per
  frame; the gain is that `world.run_schedule(Main)` is the one entry point,
  anything with a world can run a frame, and a system can be added to `Main`.
  Where a system added to `Main` falls relative to the driver is not defined (the
  driver is exclusive, so they never overlap, and the single-threaded executor runs
  them in the order added); use a stage for anything that must be ordered.
- **Time is resources, not fields.** `Clock` and `FixedTimestep` are in the world
  so systems read them as parameters. The cost is that `App::update` has to
  find them (it panics with a message if a caller removed one).
- **`update` and `update_with`, not `run_once`.** The old `run_once` ran every
  system exactly once. A frame is not that any more (`Physics` runs zero or more
  times), so the old name would have promised something false. `update_with` with
  one step's worth of time is the "each stage once" case.
- **`run_until` does not sleep.** Sleeping is policy (a server wants a frame cap,
  a window gets one from vsync). The method says so, and `done` is the place to
  sleep. On `wasm32` the host drives `update`, so the method is not there.
- **Plugins build at once**, as before and as Bevy does, so what a plugin adds is
  there when `add_plugin` returns. A plugin can add another plugin.
- **The trim runs after `Main` returns, not inside it.** `Main` is out of the
  schedule map while it runs, so from inside it the world cannot see its systems,
  and one of them could be the reader that still needs the log.
  `World::trim_removals_for_schedules` refuses to trim while any schedule is
  running for this reason.
- **`StateTransition` runs between `PreUpdate` and `Update`,** where `bevy_app` runs its
  own. A system in `PreUpdate` can ask for a change and `Update` sees it the same frame; a
  system in `Update` or later asks for the next frame. The first frame makes the state, so
  a change asked for in `PreUpdate` of the very first frame finds no state yet and
  `in_state` is false (the state exists from the first transition on, as in `bevy_state`).
- **Messages are trimmed the same way,** by `World::trim_messages_for_schedules`, per
  type and never past the previous frame's trim (so a message lives at least a whole
  frame whoever reads it, including code outside the schedules). A reader in `Physics`
  sees every message however many steps each frame makes, which the double-buffered
  messages of `bevy_ecs` cannot promise; that is the reason for the tick-stamped design
  (`docs/mid-ecs-events.md`).
- **What trimming counts.** Only schedule systems that read the removal log
  (`docs/mid-ecs-schedule.md`, "Phase 3 additions in `mid-ecs`"). A `ChangeTracker`
  held outside the schedules is not counted, so `set_trim_removals(false)` is the
  way to take over. A reader that is skipped by its conditions holds the log back
  until it runs again, which is correct and costs memory while it waits. If no
  system in a schedule reads removals, nothing is trimmed.
- **The closures migrated.** The old `System = Box<dyn FnMut(&mut World)>` and
  the `Stage` enum are gone. A closure `|world: &mut World| {..}` is an exclusive
  system through `IntoSystem`, and must be `Send` (`System: Send` since phase 2).
  Nothing else in the workspace used `mid-app`, so nothing outside the crate
  changed.

### `tests.rs`

41 tests and the doctest. Stage order whatever the order added; exclusive
closures as systems; an empty app; order inside a stage (`before`, `after`,
`chain`, sets, conditions) reaching the schedule through the app; a system on
`Main`; a multi-threaded stage; the fixed timestep (one step per step's worth,
carry between frames shorter than a step, other stages once a frame, a stall
capped and the rest dropped, the default 60 Hz and cap of 8, another rate);
systems reading `Clock` and `FixedTimestep`; `update` against the platform clock;
plugins (built at once, nested); `run_until`; and the removal log (trimmed to empty
each frame with nothing missed, trimming switched off and on, no trim without a
reader, a skipped reader keeping the log until it runs, a reader in `Physics`
seeing each removal once across frames of zero, one and three steps), and messages
(written and read across stages in a frame and a frame apart; a `Physics` reader seeing
every message across frames of zero, one and three steps; a `Physics` writer's messages
all read next frame; the log kept at one entry by trimming; a type with no reader kept a
frame; trimming switched off and on; a skipped reader keeping its messages;
`on_message` through the app), and states (the transition between `PreUpdate` and
`Update` and `OnEnter` of the initial state once; a change from `PreUpdate` in force the
same frame and one from `Update` the next; the state present from the first frame;
`init_state` twice; a state gating `Physics` and its changes read as messages).

## What is not built, on purpose

- Observers (phase 4c; messages and states are built).
- The 128 Hz network tick (see `labels.rs`).
- A startup stage, a `Last` stage, schedule order as data. Add when something
  needs one.
- Stepping, debug tooling, removing a system from a built schedule (named as out of
  scope in `docs/mid-ecs-schedule.md`).
- The `wasm32` build of any of this: `run_until` is `cfg`-ed out and `mid-time`'s
  wasm clock is the existing, unverified one (`docs/roadmap.md`, Decision 2).
- Optimization. It is paused; nothing here was measured. One cost is known from
  the code: a frame runs `Main` and five schedules, each one a build check and a
  run, and the trim walks every system of every schedule once.

## CI and Workflows

None yet. No workflow runs `cargo test -p mid-app` or the schedule tests in
`mid-ecs`; `deploy-site.yml` runs only `mid-ecs`'s `ffi::` tests. Adding one is a
`.github` change, delivered separately.

## Fixes and Problems

### `app.rs`

- **The trim could not be `Schedule::oldest_last_run`.** The plan was to call
  `World::trim_removals` with it. It counts every system, so any system that runs
  rarely holds the log back for good. `mid-ecs` got a `reads_removals` flag on
  systems and conditions and `World::trim_removals_for_schedules` (full account in
  `docs/mid-ecs-schedule.md`).
- **Breaking change to the crate's API** (version 0.1.0, no dependents in the
  workspace): `Stage`, the `System` alias, `add_system(Stage, ..)` and `run_once`
  are gone; `add_systems(label, ..)`, `update` and `update_with` replace them.

### `tests.rs`

- **A first pass proves little.** All 26 tests passed on the first run (and the nine
  message tests on theirs). Mutation
  testing of the new code (below) is what shows the tests can fail.

## Verification

Rustc 1.91 in the sandbox, one core, static reading plus the tests. Nothing was
benchmarked.

| | before phase 3 | after |
|---|---|---|
| `mid-ecs` lib tests | 971 | 987 (240 in `schedule`, 16 in `tests_removals`) |
| with `--features scratch-arena` | 976 | 992 |
| `mid-ecs` doctests | 33 | 33 |
| `ffi::` | 103 | 103 |
| `mid-app` tests | 3 | 26, plus one doctest |

Phase 4a (messages) took these to 1033 / 1038 / 34 / 103 and 35 `mid-app` tests; its
table and mutation results are in `docs/mid-ecs-events.md`.

Mutation testing of everything new, 49 mutants in a copy of the tree and a neutral
control (which survived, as it should): the flag and its setter and getter, the
three `System` answers and the default, every `Condition` answer and `and`/`or`
in each position, the schedule's counting rule (each part of it, systems and both
kinds of condition, and the comparison that picks the oldest), `any_running`, the
trim's guard, tick and return value, and in `app.rs` the stage order, the step
loop, the clock calls, the timestep arguments and constants, the `run_until`
order, and the trim switch and its default. All 49 were killed, and two were
re-run by hand to check the failing tests are the ones that should fail. The
first pass of the tests had nothing to say about this; the default answer of the
two traits was the one thing no test could reach until a hand-written system and
condition were added to `tests_removals.rs`.
