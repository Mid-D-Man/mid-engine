# mid-ecs events: messages, states and observers

## Overview

Phase 4 of the schedule work (`docs/mid-ecs-schedule.md`) is the three ways a system
reacts to something it did not run for: **messages** (values one system writes and
others read later), **states** (an enum the whole app is "in", with schedules that run
on entering and leaving a value) and **observers** (callbacks that run at once when
something is triggered or a component is added or removed). All three needed a frame
owner to run their update in, which `mid-app` (phase 3) now is.

Built: **messages** (phase 4a), **states** (phase 4b), **observers** (phase 4c) and
**observer propagation** through `ChildOf` (an event bubbling up the hierarchy, see
`docs/mid-ecs-hierarchy.md`), all in this document. Not built: the rest of what each section
below lists. Bevy 0.20-dev splits "message" (buffered, read by a system later) from
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

## Observers: the design

An observer is a system with one `On<E>` parameter, added with `World::add_observer`
(`App::add_observer`, `Commands::add_observer`). `World::trigger(event)` runs every
observer of `E`, in the order they were added; `World::trigger_targets(event, entity)`
also runs the ones added for that entity with `World::observe`. `On<E>` gives the event
and the target. The observer is an ordinary system otherwise: queries, resources,
`Commands`, message writers, `Local` state that lasts across triggers, and `Added` /
`Changed` filters that mean "since this observer last ran".

**Different from `bevy_ecs`** (read from `crates/bevy_ecs/src/observer/`):
- **An observer is not an entity.** `bevy_ecs` spawns an entity with an `Observer`
  component, which is how it is despawned, how it is scoped to an entity, and how it
  watches a set of components. Here the world has a table of boxed systems and an
  `ObserverId`. Scoping to an entity is a field on the entry, and the observer of a
  despawned entity is dropped the next time something is triggered at that entity. If
  observers need to be entities later (inspecting them, despawning them with a
  scene), the table is the one thing to change.
- **`Event` is an explicit trait** (`impl Event for Hit {}`), not a blanket one, because
  the lifecycle events need behavior of their own per type. There is no derive.
- **A trigger made from inside an observer waits** until the observers of the trigger in
  progress have all run, then runs; it does not run inside the observer that made it. An
  observer reaches the world only through `Commands`, which `System::run` applies after
  the observer returns, so a nested trigger arrives as a queued one. The result is a queue
  and not a recursion: a chain of five thousand triggers (an explosion setting off its
  neighbors, the same observer each time) is a loop, not a stack. `bevy_ecs` runs a nested
  trigger at once, inside the observer. The cost is order: with observers `[a, b]` of `E1`
  and `a` triggering `E2`, the order is `a`, `b`, then the observers of `E2`, where Bevy
  runs `a`, the `E2` observers, then `b`.
- **Lifecycle events are `Add<T>`, `Insert<T>`, `Discard<T>`, `Remove<T>` and
  `Despawn<T>`** (event types, where Bevy writes `On<Add, T>`), triggered at the entity.
  They ride on the five component-hook points, so they cover Archetype and Sparse Shell
  components and the FFI byte paths alike. **They are delivered when the outermost
  structural operation has finished**, as the commands a hook queues are; Bevy delivers
  them inside the operation. So an observer of `Add<T>` or `Insert<T>` finds the
  component, and one of `Discard<T>`, `Remove<T>` or `Despawn<T>` does not (the entity is
  gone for `Despawn<T>`). To see the value being removed, use a component hook, which
  runs before the removal. A world pays nothing for this until an observer of a lifecycle
  event is added (`HookRegistry::is_active`).
- **On an archetype component, `Discard`/`Insert` on overwrite never happen:**
  `insert_static` on a component the entity has does nothing and returns `false`. Only
  Sparse Shell `insert` overwrites.

### Propagation: the design

An event type opts in with `const PROPAGATE: bool = true` in its `Event` impl. Triggered at
an entity (`World::trigger_targets`, `Commands::trigger_targets`, `EntityCommands::trigger`),
it is delivered there first, then at that entity's `ChildOf` parent, then the grandparent,
and so on up to a root. At every step the global observers of the event run and the ones
added for that entity with `World::observe`, in the order they were added, so an observer
scoped to a panel hears the clicks that happen on the buttons inside it. `On::target` is
the entity of the step, `On::original_target` is the one the event was first triggered at,
and the event value is the same one at every step. An observer ends the climb with
`on.propagate(false)`.

**The same as `bevy_ecs`** (read from `crates/bevy_ecs/src/event/trigger.rs`,
`PropagateEntityTrigger`, and `observer/system_param.rs`): the event is delivered at the
target before anything is looked up; each step runs every observer of that entity, global
and scoped, whatever the flag says (the flag is read after they have all run); the climb
ends at an entity with no parent, or when the entity is gone; `propagate` is one flag that
the last writer in a step decides; the target an observer sees changes from step to step
and the original one does not.

**Different, on purpose:**
- **Propagation is a property of the event type, and always follows `ChildOf`.** `bevy_ecs`
  lets each event pick a `Traversal` (any query that yields the next entity) and a
  non-automatic mode where an observer has to ask for each step. Here there is one
  traversal, the hierarchy, and the flag starts `true`. Nothing in the repo needs another
  route or the opt-in mode; both are additions to the trait if something does.
- **The event is not mutated.** `bevy_ecs` writes the current entity into the event
  (`set_event_target`). Here the event is shared and read only, and the world keeps the
  current and original entities next to it, which is what `On` reads.
- **The next entity is chosen after the observers of a step have run and their commands
  have been applied.** An observer reaches the world through `Commands`, which `System::run`
  applies as soon as the observer returns, so an observer that moves the current entity to
  another parent sends the climb to that parent, and one that despawns the current entity
  ends it (the entity has no parent any more). This is the same order difference as for
  nested triggers, above. `bevy_ecs` walks the whole route before its deferred commands
  run.
- **The observers of a trigger are fixed when it starts,** as they already were for a
  single delivery: an observer added by an observer during the climb does not hear the
  rest of it, and one removed during it is skipped at the later steps.
- **A trigger without a target does not climb,** and neither do the lifecycle events
  (`Add<T>` and the others), which stay at the entity they happened to.
- **A loop of `ChildOf` ends the climb.** `set_parent` and `add_children` refuse a loop,
  and a raw `ChildOf` that makes one is removed once the operation finishes, but only in a
  world that enabled the hierarchy. In a world that did not, raw inserts can make one, and a
  climb around it would never end. The climb remembers an earlier entity of its path and
  compares each next one with it (Brent's cycle detection: no allocation, nothing kept per
  entity), moving the remembered entity forward at doubling intervals. Once a loop is
  closed the climb ends within a lap or two, so each entity of a loop is delivered to once
  or twice. A bound on the depth, as `World::ancestors` has, was the alternative, and
  would have run a million observers around a two-entity loop first.
- **The cost for an event that does not propagate** is a loop that runs once: the constant
  is known when the delivery is compiled, so the compiler can drop the extra code (not
  checked in the assembly). The slot an observer reads from is a little larger (the
  original entity and the flag). Nothing was measured.

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

### `observer.rs`

**What it does:** `Event` (with `PROPAGATE`), the five lifecycle events, `On<E>` (with
`target`, `original_target` and `propagate`), `ObserverId`, the observer table and the
trigger queue, and `World::add_observer`, `observe`, `remove_observer`, `observer_count`,
`trigger`, `trigger_targets`. Propagation is the loop in `World::deliver`, which calls
`run_observers` once per step.

**Decisions:**
- **`On<E>` is a read of a resource, `TriggerSlot<E>`,** that the world fills for the
  trigger in progress and empties after. It needs no new kind of system input: the
  observer is a system like any other. Using `On<E>` in a scheduled system (not added
  with `add_observer`) panics with a message that says so.
- **Registration reads the system's `On<E>` by probing its initialization.** `On<E>`'s
  `init_state` notes its event type in a probe the world sets up while `add_observer`
  initializes the system, so `add_observer::<E, ..>` needs no type annotation. Zero `On`
  parameters or two panic at registration, as do conflicting parameters.
- **The system is taken out of the table while it runs** and put back after, so it can
  `remove_observer` itself through `Commands` (the system is then dropped, not put back).
- **A panic in an observer** resets the delivery flag and clears the queue before it
  propagates, so the world can deliver again. That observer is left disabled: its system
  was taken out and is not put back.
- **The list of observers of an event is cloned per trigger.** One allocation per trigger
  with observers, none without, and one for the whole climb of a propagating event, not one
  per step. Performance work is paused; this is where to start.
- **The delivery state is one struct in the slot** (`Delivery`: event, current entity,
  original entity, the flag), replacing the pair the slot held. The flag is an `AtomicBool`
  so that `On::propagate` can set it through the shared read of the slot resource that
  `On` has, which keeps `On` a plain read parameter (no new access kind, nothing for the
  executor to learn). Only one thread ever touches it; the atomic is for the type system,
  not for sharing, and uses relaxed ordering.
- **`run_observers` is the old loop body, unchanged,** split out so a propagating delivery
  can call it for each entity. A scoped observer is matched against the entity of the step.

**Tests:** 57 inside the file (35 before propagation, 22 for it) and two doctests (one for
propagation). The 35 are: delivery order, one event only, no observers,
queries and resources, commands applied after the observer, state across triggers, messages
written by an observer; registration (exactly one `On`, conflicts, outside a trigger,
removal, self-removal, no id left behind); entities (global and scoped observers, the
target, a despawned target); order (a trigger from an observer waits, a chain of five
thousand, a panic leaves the world usable and drops the triggers that were waiting);
`Commands` and `EntityCommands` triggers and
observers; lifecycle events (add, overwrite, remove, despawn, the entity's own observers,
Sparse Shell, beside a component hook, nothing paid before the first, other component
types, an observer reacting with commands, an `insert_static` of what is there).

The 22 for propagation: an event that does not propagate stays at its target; a climb
reaches every ancestor nearest first; global observers run at every step and see the
original entity; within a step global and scoped observers run in the order added;
`propagate(false)` ends the climb after its own step (the rest of that step still runs);
the last observer of a step to set the flag wins, both ways round; the flag starts set
for each trigger; `propagate` does nothing for an event that does not propagate; no target
means no climb; a root has one step; a dead target is delivered to once; an observer that
moves the current entity changes the route; one that despawns it ends the climb; a trigger
made by an observer waits for the whole climb; an observer added during a climb misses the
rest of it; lifecycle events do not climb; `Commands::trigger_targets` and
`EntityCommands::trigger` climb; propagation needs only a `ChildOf`, not the hierarchy
hooks; a 5,000 deep chain is a loop and not a recursion; loops of raw `ChildOf` end the
climb (one, two and three entities, and a tail leading into a loop, with the exact visit
order pinned); a panic in the middle of a climb leaves the world able to climb again; and
eight seeds of random forests with random stop points and scoped observers, each climb
compared with an independent model, which fails if it did not reach enough long climbs,
early stops and scoped deliveries (the run reaches 112 long climbs, 105 early stops and 145
scoped deliveries in 320 climbs; the minimums are 60, 50 and 80).

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
- `lib.rs` also exports `state` and `State`, `NextState`, `States`, and `observer` and
  `Event`, `On`, `ObserverId`.
- Observers: `hooks.rs` (`HookRegistry` gains the lifecycle table, `is_active` counts it),
  `world.rs` (`run_hook` queues the lifecycle trigger; the `observers` field),
  `commands.rs` (`Commands::trigger`, `trigger_targets`, `add_observer`;
  `EntityCommands::observe`, `trigger`), and `mid-app`'s `App::add_observer` and re-exports.

## Not built, on purpose

- **A consuming read** (`bevy_ecs`'s `MessageMutator`, `drain`). Nothing needs one, and
  with a shared log a draining reader would take messages from the others.
- **Messages from C.** A C caller would need a byte payload and a registered size, the way
  component data has; decide it with the C surface for schedules, which does not exist.
- **Deferred-world writes from hooks** (`DeferredWorld::write_message`). Hooks queue
  commands, and `Commands::write_message` is one.
- **Other propagation routes and the opt-in mode** (see the propagation design above):
  a traversal that is not `ChildOf`, a per-event choice of route, an event whose observers
  must each ask for the next step, and propagation of the lifecycle events.
- **Propagation from C.** Observers are not on the C surface (decided with hooks, in
  `docs/mid-ecs.md`), so neither is their climb.
- **Observers before the change** for `Discard`, `Remove` and `Despawn` (Bevy's
  `On<Remove, T>` runs with the value still there). Needs the observer to run inside the
  operation with a restricted world, as hooks do; use a hook for the value.
- **An observer of several events**, and `On<Add, (A, B)>` style component sets. One event
  type per observer; add one observer per event.
- **Observers as entities**, and run conditions on observers.
- The rest of `bevy_state` listed above.

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

### `observer.rs`

- **Two assumptions about the existing code were wrong,** both found by the first test
  run: a mutable query is `Query<Mut<T>>` here, not `Query<&mut T>`, and `insert_static`
  of a component the entity already has does nothing, so "overwrite" lifecycle events
  exist only for Sparse Shell `insert`. The test and the docs now say so.
- **Nested triggers were going to be depth-first, like Bevy's.** Writing the re-entrancy
  case down showed that an observer is taken out of the table while it runs, so a nested
  trigger of the same event would find it missing and skip it silently, and chain reactions
  (the same observer triggered by its own consequences) are exactly what games do. The
  queue came from that: it needs no re-entrancy and no recursion depth. The order
  difference from Bevy is in the design section.

- **Propagation: a bound on the depth was the first plan for loops, and it was wrong.**
  Reusing the hierarchy's walk limit (2^20) would have made a loop of two raw `ChildOf`
  deliver a million events before it stopped, which is a hang for any observer that does
  real work. Brent's check replaced it before anything was written; the loop tests pin the
  exact visit order.
- **Propagation: one test expectation was wrong, not the code.** The first version of the
  three-entity loop test listed the visits in the direction the entities were made in
  (`a, b, c`); the loop runs the other way (`a, c, b`), because each entity was made a child
  of the one before it. The hand-worked Brent sequence for the other cases (`a, b, a`; `x,
  a, b`) matched on the first run.

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

| | before 4a | after 4a | after 4b | after 4c | after propagation |
|---|---|---|---|---|---|
| `mid-ecs` lib tests | 987 | 1033 (276 in `schedule`, 36 of them in `tests_messages`; 10 in `message`) | 1057 (24 in `state`) | 1092 (35 in `observer`) | 1165 (57 in `observer`; 1143 on main before, the rest came with the hierarchy) |
| with `--features scratch-arena` | 992 | 1038 | 1062 | 1097 | 1170 |
| `mid-ecs` doctests | 33 | 34 | 35 | 36 | 39 (38 on main before, one new) |
| `ffi::` | 103 | 103 | 103 | 103 | unchanged |
| `mid-app` tests | 26 | 35, plus one doctest | 41, plus one doctest | 43, plus one doctest | 45, plus one doctest (unchanged by this work) |

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

Mutation testing of 4c, 35 mutants (and the control): which hook kind each lifecycle event
stands for (all five) and its component type, `On::target`, the registration probe, the
lifecycle registration, id allocation, observer order, removal (the id list, the return
value), `observer_count`, the target passed by `trigger_targets`, the queue guard, the
delivery flag, the slot (set and cleared), the scope check, the alive check and dropping a
dead scope's observer, putting the system back, `HookRegistry::is_active`, the lifecycle
table's kind index at both ends, the queued lifecycle trigger, and each `Commands` and
`EntityCommands` method and `App::add_observer`. 34 were killed by the first pass. One
survived: nothing checked that a panic drops the triggers that were waiting, so a stale
trigger would have run on the next delivery. `a_panic_drops_the_triggers_that_were_waiting`
covers it and kills it.

Propagation was checked in a sandbox with rustc 1.97.1 on one core, applied to a fresh
clone of main (`22df79d`). `cargo check` passes for `mid-ecs` and `mid-app` on
`wasm32-unknown-unknown` and for `headless-server`, `rustfmt --check` is clean on
`observer.rs`, and `cargo clippy --no-deps --all-targets` on `mid-ecs` reports 3,147
warning lines both on main and on the changed tree (the five in `observer.rs` are the
lifecycle macro's, as before).

Mutation testing of propagation, 18 mutants of `observer.rs` and a clean baseline checked
before every run (the harness restores the file after each mutant): the default of
`PROPAGATE`, the check that stops an event that does not propagate, the initial flag, the
flag being ignored or inverted, skipping ancestors, the current entity not being updated
for `On`, the original entity not being recorded, the step not advancing, the loop check
removed, the remembered entity never or wrongly moved (two ways), the counter, `propagate`
ignoring its argument, `original_target` returning the current one, the slot not being
cleared, the scope comparison inverted, and a dead scoped observer not being dropped. All
18 were killed; two of them (the loop check removed, the window not doubled) by timing out,
because the climb then never ends.

**Not verified:** real CI has not run this; the cost of a propagating trigger (or of the
larger slot on triggers that do not propagate) was not measured; Miri does not cover
`observer::` (its workflow runs `world_cell::`, `system::` and `interior::`), and the
change adds no `unsafe`; the wasm32 tests were type-checked, not run; the three bench
crates were not rebuilt (nothing they use changed).
