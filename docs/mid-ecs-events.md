# mid-ecs events: messages, states and observers

## Overview

Phase 4 of the schedule work (`docs/mid-ecs-schedule.md`) is the three ways a system
reacts to something it did not run for: **messages** (values one system writes and
others read later), **states** (an enum the whole app is "in", with schedules that run
on entering and leaving a value) and **observers** (callbacks that run at once when
something is triggered or a component is added or removed). All three needed a frame
owner to run their update in, which `mid-app` (phase 3) now is.

Built so far: **messages** (phase 4a) and **states** (phase 4b), both in this document.
Not built yet: observers (4c). Bevy 0.20-dev splits "message" (buffered, read by a system later) from
"event" (triggered, run by observers at once); this crate follows that split, so
"event" is the word for what observers will take.

## Messages: the design

`bevy_ecs` keeps two buffers per message type and swaps them once a frame
(`Messages::update` in `crates/bevy_ecs/src/message/messages.rs`, called from
`message_update_system`, and `RemovedComponents` is the same machinery:
`World::clear_trackers` calls `removed_components.update()`, which `bevy_app` calls once
per `App::update`). A message therefore lives one or two updates. A system that does not
run in that time never sees it. That is bounded memory bought with silent loss, and it is
why `bevy_ecs` needs `signal_message_update_system` to hold the swap back until a fixed
update has run (`message/update.rs`): a reader in `FixedUpdate` would otherwise miss
messages on any frame that ran no fixed step.

`mid-ecs` already had the other answer for removals: stamp each entry with the world tick
and let each reader see what is newer than its own last run, then trim by the oldest
reader (`docs/mid-ecs-schedule.md`, "Phase 3 additions in `mid-ecs`"). Messages use it
too:

- A `Messages<M>` resource holds `(Tick, M)` entries, oldest first.
- `MessageWriter<M>` stamps each with the tick of the run that wrote it.
- `MessageReader<M>` shows the entries newer than its system's last run and not newer
  than this run.
- `World::trim_messages_for_schedules` cuts each type's log at the last run of the system
  that reads that type and ran longest ago, and never past the previous trim.

What this buys: a reader in `Physics`, which runs zero, one or three times a frame, loses
nothing; a reader skipped by its conditions sees everything written meanwhile when it
next runs; two readers each see every message. What it costs: reading does not consume
(calling `read` twice in one run gives the same messages twice, where `bevy_ecs`'s
`MessageReader::read` advances a cursor), a reader that is skipped for good holds back
its type's log for good, and a message written by a system later in the frame than its
reader arrives a frame later (as in `bevy_ecs`).

### Where a message becomes visible

A message written by a system at tick `T` is seen by every reader run that starts after
the writer's run started, from the first one on. A reader that started earlier (in a
multi-threaded schedule, with no ordering) sees it with its next run, because the
message is stamped after that reader's run began. A write from code that has a world
(`World::write_message`, `Commands::write_message` when the commands apply) is stamped
with the world's tick at that moment.

### Trimming, and the grace interval

`World::trim_messages_for_schedules` is the message counterpart of
`trim_removals_for_schedules`, with one difference. Removals are trimmed exactly at the
oldest reader. Messages are trimmed at the **older of** that tick and the world tick at
the previous call. Two reasons:

- A type no schedule system reads has no oldest reader. Trimming it at "now" would lose
  messages to a `ChangeTracker` that reads once a frame from outside the schedules (an
  exclusive system, a C caller). Keeping everything would grow without bound. So it is
  kept for between one and two intervals, which is what `bevy_ecs` does with every type.
- The same reader outside the schedules is not counted for a type that does have
  schedule readers. The grace interval covers it, provided it reads at least once per
  interval.

The call advances the world tick by one after recording it, so a message written right
afterwards is stamped later than the trim and the next trim does not take it for older
than it is. It does nothing while any schedule is being run (their systems are out of
reach), and `mid-app` calls it after `Main` returns.

## States: the design

A state is a value the whole world is "in", usually an enum. `bevy_state` is a separate
crate on top of `bevy_app`; here the state types live in `mid-ecs` (`state.rs`) because
they need only schedules and messages, and `mid-app` supplies the frame that runs them.

- `State<S>` (resource): the current value. `NextState<S>` (resource): `Unchanged`,
  `Pending(S)` or `PendingIfNeq(S)`, set by systems with `set`, `set_if_neq`, `reset`.
- `StateTransition` (schedule label): the schedule that applies changes. Every state type
  adds one system to it (`apply_state_transition::<S>`, an exclusive system) when it is
  inserted. `mid-app` runs it once a frame between `PreUpdate` and `Update`, where
  `bevy_app` runs its `StateTransition` too (after `PreUpdate`, before the fixed loop and
  `Update`).
- `OnExit(s)`, `OnTransition { exited, entered }`, `OnEnter(s)`: labels of schedules that
  exist only if something was added to them, and are skipped silently otherwise. Run in
  that order, after `State<S>` has changed, so a system in them reads the new state.
- `StateTransitionMessage<S> { exited: Option<S>, entered: S }`: written at each
  transition, for any system that wants to react without owning a schedule.
- Run conditions `in_state(s)`, `state_changed::<S>`, `state_exists::<S>`.

**The same as `bevy_state`** (read from `crates/bevy_state/src/state/transitions.rs`): the
state does not exist until the first `StateTransition` (so `in_state` is false before it
and the `OnEnter` of the initial value runs in that first frame); exit, then transition,
then enter; `set` of the state the world is already in is an identity transition that
runs exit and enter again, and `set_if_neq` is the one that does nothing; every
transition is also a message.

**Different, on purpose:**
- **`States` is a blanket trait** for every `Debug + Clone + Eq + Hash + Send + Sync +
  'static` type, where `bevy_state` asks for a derive or an impl. There is no derive
  macro here, and the one-line impl per enum bought nothing. (`SystemSet` is opt-in for a
  different reason: a blanket one made tuples of sets ambiguous. Nothing like that
  happens for states.)
- **One transition per run.** A change asked for from inside an `OnEnter` waits for the
  next run of `StateTransition`, so a frame is never an unbounded loop of transitions.
- **Not built:** `SubStates`, `ComputedStates`, `StateScoped` entities, a `PreviousState`
  resource (the message has the state that was left), a state that goes away
  (`bevy_state`'s `Option<S>` machinery), and `OnEnter`/`OnExit` for a whole set of
  states at once.
- **`apply_state_transition` is one exclusive system per state type,** in the order the
  types were inserted. Bevy sorts exit, transition and enter schedules of all types into
  three sets so that every type exits before any enters. Here a type's exit, transition and
  enter run together, then the next type's. With two state types that depend on each
  other, put the dependent one second, or use one state type. Nothing in the repo has two
  yet.

## Modules

### `message.rs`

**What it does:** `Messages<M>` (the log, a resource), `MessageWriter` and
`MessageReader` (system parameters), the registry of message types the world holds, and
the unit tests of the log.

**Decisions:**
- **Parameters over `Res`/`ResMut` access to `Messages<M>`.** A writer is a write of the
  resource and a reader a read, so the existing conflict rules do the work: a system
  cannot read and write one type, two writers of a type never overlap, readers overlap
  freely. Nothing new in the executor.
- **`M: Send + Sync + 'static` for both,** as `bevy_ecs`'s `Message` is.
- **A reader or writer initializes its type** (`World::init_message`), so there is no
  `add_message` step to forget. `init_message` is idempotent and public for code that
  wants the type held before any system names it.
- **Per-type reader tracking.** `SystemMeta` records which message types a system reads
  (`add_message_read`), `System::reads_message(TypeId)` and `Condition::reads_message`
  report it (provided methods, default `true`, same safe default as `reads_removals`;
  exclusive systems answer `false`). A system skipped for good holds back only the types
  it reads. One flag for all message types would have held every type back.
- **`Messages::since` scans the whole log,** for the reason `RemovalLog::since` does: an
  entry stamped after the reader's run began is "not newer" and must not hide the older
  ones before it.
- **The registry holds function pointers** (`trim`, `check_ticks`, `clear`) per type, so
  the world can reach logs of types it does not name, for trimming, the tick-wraparound
  check and clearing.
- **Wraparound.** `World::check_change_ticks` now also clamps every message stamp and the
  last-trim tick, as it does for the removal log.

**World API:** `init_message`, `write_message`, `messages_since(&ChangeTracker)`,
`message_count`, `trim_messages(tick)`, `clear_messages`, `trim_messages_for_schedules`.
`Commands::write_message`. Run condition `common_conditions::on_message::<M>`: true if a
message was written since the condition last ran.

**Tests:** 10 unit tests of the log, the registry and the world's tick check inside the file.

### `state.rs`

**What it does:** `States`, `State<S>`, `NextState<S>`, the labels, the message,
`World::insert_state` and `World::init_state`, `apply_state_transition`, and the three
run conditions.

**Decisions:**
- **`insert_state` on a type the world already has only asks for the value** (as
  `NextState::set` would), so calling it twice cannot add the system twice or reset a
  state that is in force. `init_state` (the `Default` one) does nothing at all in that
  case.
- **`NextState` is taken with `mem::take` before anything runs,** so a transition that
  panics or a system that asks again from inside an `OnEnter` cannot apply the same
  request twice.
- **A missing schedule is not an error; any other failure is.** `try_run_schedule`'s
  `NotFound` is skipped, and a schedule that is already running or does not build panics
  with its message.
- **`in_state` returns a closure over `Option<Res<State<S>>>`**, so it is false, not a
  panic, before the state exists.

**Tests:** 24 inside the file and a doctest: `NextState`'s three methods and the blanket
trait; the state not existing before the first run, and the first run entering the initial
value with no exit; `init_state`; exit, transition, enter in that order; schedules seeing
the new state; a change applied once; identity transitions and `set_if_neq`; a withdrawn
change; one change per run; no schedules at all; a state with data; two state types; a
second `insert_state`; the transition read as a message; `in_state` (alone, with `not` and
`and`), `state_changed`, `state_exists`; and an exit schedule of a state never left.

### `schedule/tests_messages.rs`

Who sees what and when (reader after, before and in the same frame as its writer; every
reader sees every message; reading twice; a skipped reader misses nothing; a system that
reads one type and writes another; two readers of one type in one system; read and write
of one type in one system fails with a conflict; `write_batch`; `Commands`; four writers
of one type in a multi-threaded schedule, 200 messages each time, none repeated or lost,
twenty times); `on_message` (gates a system; the gated system reads everything since its
own last run); the per-type flag on a system, an exclusive system, a writer-only system
and every condition combinator, plus a hand-written system and condition that say
nothing; the schedule's `oldest_message_reader` (per type, not set up yet, a skipped
reader of one type not holding back another, conditions on a system, on a set and
nested); the world's calls (tracker reads, `init_message`, `trim_messages`,
`clear_messages`); the trim (a type nobody reads kept for an interval, a message written
just after a trim, a tracker reading once an interval, a reader that has not run, a
skipped reader of one type, refusing inside a running schedule, no `Schedules`
resource); sixty seeds of random writes, runs of two readers and trims, checking that
both see every message exactly once; and two hundred frames of trimming that keep the log
at one entry.

### Changes outside `message.rs`

- `system_param.rs`: `SystemMeta::add_message_read` / `reads_message`;
  `register_resource` and `missing_resource` are `pub(crate)`.
- `system.rs`, `schedule/condition.rs`: `reads_message` on `System` and `Condition`, and
  `on_message`.
- `schedule/mod.rs`: the removal-reader question became a three-way one (`Reader::Any`,
  `Removals`, `Message(TypeId)`) so the three share the walk; `oldest_message_reader` on
  `Schedule` and `Schedules`; `World::trim_messages_for_schedules`.
- `world.rs`: the registry field, the methods above, the tick check.
- `commands.rs`: `Commands::write_message`. `lib.rs`: `pub mod message` and the exports.
- `mid-app`: `App` trims the message logs after each frame (`set_trim_messages`,
  `trims_messages`), runs `StateTransition` between `PreUpdate` and `Update`, and has
  `init_state` and `insert_state`; see `docs/mid-app.md`.
- `lib.rs` also exports `state` and `State`, `NextState`, `States`.

## Not built, on purpose

- **A consuming read** (`bevy_ecs`'s `MessageMutator`, `drain`). Nothing needs one, and
  with a shared log a draining reader would take messages from the others.
- **Messages from C.** A C caller would need a byte payload and a registered size, the way
  component data has; decide it with the C surface for schedules, which does not exist.
- **Deferred-world writes from hooks** (`DeferredWorld::write_message`). Hooks queue
  commands, and `Commands::write_message` is one.
- **Observers**, the rest of phase 4 (4c), and the rest of `bevy_state` listed above.

## CI and Workflows

None yet; see `docs/mid-app.md`.

## Fixes and Problems

### `message.rs`

- **The stamp of a message written right after a trim.** The first version recorded the
  trim's tick and left the world tick where it was, so a message written next got a stamp
  equal to the trim's, and the next trim (which drops stamps at or before the previous
  trim's tick) dropped it after a fraction of its interval. Found by working through the
  tick arithmetic for the grace interval before running anything; fixed by advancing the
  world tick after recording it, and covered by `a_message_written_just_after_a_trim_is_
  not_trimmed_by_the_next`.
- **A test of an impossible state removed.** A unit test asserted that `trim` keeps an
  entry stamped after `this_run`. It does not (such an entry is "not newer" and is
  dropped, as in the removal log), and no real trim has one: it runs outside any schedule,
  where every stamp is at or before the world's tick. The test was deleted rather than
  made to pass.

### `state.rs`

- **Two of the first `mid-app` state tests were wrong, not the code.** They set a state
  from `PreUpdate` on the first frame and expected it to take effect. The state does not
  exist until the first `StateTransition`, which runs after `PreUpdate`, so on frame 1
  `in_state` is false and the first change can only be asked for on frame 2. That is the
  `bevy_state` behavior this module copies, so the tests were changed and the first-frame
  rule is now written down in the tests and in the module docs.
- **`is_identity` survived mutation** because no test asserted it false for a real change.
  One assertion added.

## Verification

Rustc 1.91 in the sandbox, one core, static reading plus the tests. Nothing was
benchmarked.

| | before 4a | after 4a | after 4b |
|---|---|---|---|
| `mid-ecs` lib tests | 987 | 1033 (276 in `schedule`, 36 of them in `tests_messages`; 10 in `message`) | 1057 (24 in `state`) |
| with `--features scratch-arena` | 992 | 1038 | 1062 |
| `mid-ecs` doctests | 33 | 34 | 35 |
| `ffi::` | 103 | 103 | 103 |
| `mid-app` tests | 26 | 35, plus one doctest | 41, plus one doctest |

Mutation testing of everything new, 58 mutants in a copy of the tree and a neutral
control (which survived, as it should): the log (`since`, `trim`, `check_ticks`, `clear`,
`len`, the arguments of each), the parameters (what each registers, the changed stamp,
`write_batch`, `is_empty`, `len`), the registry (once per type, each of the three
function pointers), the world's methods and the tick check, the schedule's reader walk
(an uninitialized system and condition, which question is asked, the comparison that
picks the oldest), the trim (the running guard, the previous tick, which of the two ticks
wins, the tick bump, ignoring readers), every `reads_message` answer and combinator, the
meta list, `Commands::write_message`, `on_message`, and the app's trim switch and its
default. 57 were killed by the first pass of the tests. One survived: a condition that is
not set up yet was not counted as a reader and nothing noticed (the system path was
covered, the condition path was not). `a_system_or_condition_not_set_up_yet_counts_for_
every_kind_of_reader` covers both for all three kinds of question and kills it. The
first-pass tests had nothing to say about the removal path of that same line either.

Mutation testing of 4b, 28 mutants (and the control): `NextState`'s three methods, `is_identity`,
the second `insert_state`, the system `insert_state` adds, `init_state`'s guard, the
`PendingIfNeq` comparison, replacing the state and making it the first time, the message's
`exited`, each of the exit, transition and enter schedules and the order between them,
taking the pending change, the three run conditions, and in `mid-app` the position of
`StateTransition` in the frame, `init_state`, `insert_state` and the schedule's existence.
27 were killed by the first pass. `is_identity` survived: no test asserted it false for a
real change. One assertion added, killed.
