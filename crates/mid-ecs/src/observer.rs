// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-ecs-events.md, section "observer.rs"
// ============================================================================
//! Events and observers: systems that run at once when something is
//! triggered.
//!
//! An [`Event`] is a value you [`trigger`](World::trigger). An *observer* is
//! a system with an [`On<E>`] parameter, added with
//! [`World::add_observer`]; each time an `E` is triggered, every observer of
//! `E` runs, in the order they were added, and reads the event through
//! `On<E>`. An event can also be triggered *at an entity*
//! ([`World::trigger_targets`]): observers added for that one entity
//! ([`World::observe`]) run as well as the global ones.
//!
//! Unlike a [message](crate::message), an event is not stored: it exists while
//! its observers run.
//!
//! ```
//! use mid_ecs::observer::{Event, On};
//! use mid_ecs::{ResMut, World};
//!
//! struct Damage(u32);
//! impl Event for Damage {}
//!
//! struct Total(u32);
//!
//! fn tally(on: On<Damage>, mut total: ResMut<Total>) {
//!     total.0 += on.event().0;
//! }
//!
//! let mut world = World::new();
//! world.insert_resource(Total(0));
//! world.add_observer(tally);
//! world.trigger(Damage(3));
//! world.trigger(Damage(4));
//! assert_eq!(world.get_resource::<Total>().unwrap().0, 7);
//! ```
//!
//! # Lifecycle events
//!
//! [`Add<T>`], [`Insert<T>`], [`Discard<T>`], [`Remove<T>`] and
//! [`Despawn<T>`] are triggered at the entity whenever the component `T` is
//! added, inserted, discarded, removed or despawned with its entity: the
//! same five moments as the component hooks
//! ([`HookContext`](crate::HookContext)), and any number of observers can
//! react to each, where a hook is one function per component type. They are
//! delivered when the structural operation that caused them has finished (the
//! way a hook's queued commands are), so an observer of [`Remove<T>`] no
//! longer finds the component, and an observer of [`Despawn<T>`] is told
//! about an entity that is gone. For the component's last value, use a hook.
//!
//! # Order
//!
//! A trigger made from inside an observer (by `Commands`, the way an
//! observer reaches the world) does not run at once: it waits until every
//! observer of the trigger that is running has finished, then runs, so a chain
//! of triggers is a queue and not a recursion, and the same observer can be
//! triggered again by its own consequences. (`bevy_ecs` runs a nested trigger
//! at once, inside the observer.)

use std::any::{type_name, TypeId};
use std::collections::{HashMap, VecDeque};
use std::marker::PhantomData;
use std::ops::Deref;
use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};

use crate::hooks::HookKind;
use crate::system::{IntoSystem, System};
use crate::system_param::{missing_resource, register_resource, SystemMeta, SystemParam};
use crate::tick::Tick;
use crate::world::{Entity, World};
use crate::world_cell::WorldCell;

// ── events ───────────────────────────────────────────────────────────

/// Which component moment a lifecycle event stands for (not for use outside
/// this crate).
#[doc(hidden)]
#[derive(Clone, Copy)]
pub struct Lifecycle {
    pub(crate) kind: HookKind,
    pub(crate) component: TypeId,
    pub(crate) fire: fn(&mut World, Entity),
}

/// A value that can be triggered. Implement it with an empty impl:
/// `impl Event for Hit {}`.
pub trait Event: Send + Sync + 'static {
    /// `Some` for the lifecycle events of this module; `None` for every
    /// event a user writes.
    #[doc(hidden)]
    fn lifecycle() -> Option<Lifecycle> {
        None
    }
}

macro_rules! lifecycle_event {
    ($name:ident, $kind:ident, $doc:literal) => {
        #[doc = $doc]
        pub struct $name<T: 'static>(PhantomData<fn() -> T>);

        impl<T: 'static> $name<T> {
            fn new() -> Self {
                Self(PhantomData)
            }
        }

        impl<T: 'static> Default for $name<T> {
            fn default() -> Self {
                Self::new()
            }
        }

        impl<T: 'static> Clone for $name<T> {
            fn clone(&self) -> Self {
                Self::new()
            }
        }

        impl<T: 'static> Copy for $name<T> {}

        impl<T: 'static> std::fmt::Debug for $name<T> {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "{}<{}>", stringify!($name), type_name::<T>())
            }
        }

        impl<T: 'static> Event for $name<T> {
            fn lifecycle() -> Option<Lifecycle> {
                Some(Lifecycle {
                    kind: HookKind::$kind,
                    component: TypeId::of::<T>(),
                    fire: |world, entity| world.trigger_targets($name::<T>::new(), entity),
                })
            }
        }
    };
}

lifecycle_event!(
    Add,
    Add,
    "Triggered at an entity when it gets the component `T` and had none before."
);
lifecycle_event!(
    Insert,
    Insert,
    "Triggered at an entity when the component `T` is added or overwritten."
);
lifecycle_event!(
    Discard,
    Discard,
    "Triggered at an entity when the component `T` is removed or overwritten."
);
lifecycle_event!(
    Remove,
    Remove,
    "Triggered at an entity when the component `T` is removed and not replaced."
);
lifecycle_event!(
    Despawn,
    Despawn,
    "Triggered, once for each component `T` it had, when an entity is despawned."
);

// ── the parameter ────────────────────────────────────────────────────

/// What an observer is told: the event being triggered and the entity it
/// was triggered at, if any. A system parameter that only means something
/// inside an observer.
pub struct On<'w, E: Event> {
    event: &'w E,
    target: Option<Entity>,
}

impl<E: Event> On<'_, E> {
    /// The event.
    pub fn event(&self) -> &E {
        self.event
    }

    /// The entity the event was triggered at ([`World::trigger_targets`]),
    /// `None` for [`World::trigger`].
    pub fn target(&self) -> Option<Entity> {
        self.target
    }
}

impl<E: Event> Deref for On<'_, E> {
    type Target = E;

    fn deref(&self) -> &E {
        self.event
    }
}

/// The event being delivered, as a resource: how `On<E>` reaches it.
struct TriggerSlot<E: Event>(Option<(E, Option<Entity>)>);

// SAFETY: registers a read of resource `TriggerSlot<E>`, the only thing it
// touches.
unsafe impl<E: Event> SystemParam for On<'_, E> {
    type State = ();
    type Item<'w, 's> = On<'w, E>;

    fn init_state(world: &mut World, meta: &mut SystemMeta) -> Self::State {
        if !world.contains_resource::<TriggerSlot<E>>() {
            world.insert_resource(TriggerSlot::<E>(None));
        }
        register_resource::<TriggerSlot<E>>(meta, false);
        world.observers.note::<E>();
    }

    unsafe fn get_param<'w, 's>(
        _state: &'s mut Self::State,
        meta: &SystemMeta,
        world: WorldCell<'w>,
        _this_run: Tick,
    ) -> Self::Item<'w, 's> {
        // SAFETY: the read was registered and checked against every other
        // parameter; the world only writes the slot between observers.
        match unsafe { world.resource_with_ticks::<TriggerSlot<E>>() } {
            Some((TriggerSlot(Some((event, target))), _)) => On {
                event,
                target: *target,
            },
            Some((TriggerSlot(None), _)) => panic!(
                "`On<{}>` used outside a trigger (system `{}`): an observer must be added with \\
                 `add_observer`, not scheduled",
                type_name::<E>(),
                meta.name()
            ),
            None => missing_resource::<TriggerSlot<E>>(meta),
        }
    }
}

// ── the registry ─────────────────────────────────────────────────────

/// Names an observer, to [`World::remove_observer`] it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ObserverId(u64);

struct ObserverEntry {
    event: TypeId,
    /// `Some`: runs only for events triggered at this entity.
    scope: Option<Entity>,
    /// `None` while the observer is running.
    system: Option<Box<dyn System>>,
}

type Job = Box<dyn FnOnce(&mut World) + Send>;

/// The world's observers and the triggers waiting to be delivered.
#[derive(Default)]
pub(crate) struct ObserverRegistry {
    next: u64,
    entries: HashMap<u64, ObserverEntry>,
    /// The observers of each event type, in the order they were added.
    by_event: HashMap<TypeId, Vec<u64>>,
    /// While an observer is being added: the `On<E>` parameters its system
    /// has, as `initialize` finds them.
    probe: Option<Vec<(TypeId, Option<Lifecycle>)>>,
    queue: VecDeque<Job>,
    dispatching: bool,
}

impl ObserverRegistry {
    /// Called by `On<E>`'s `init_state`.
    fn note<E: Event>(&mut self) {
        if let Some(probe) = &mut self.probe {
            probe.push((TypeId::of::<E>(), E::lifecycle()));
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }
}

impl World {
    /// Adds an observer: a system with exactly one `On<E>` parameter, run
    /// for every `E` triggered from now on. Observers of one event run in
    /// the order they were added. The system is initialized now, so
    /// conflicting parameters panic here.
    ///
    /// # Panics
    /// If the system has no `On<E>` parameter or more than one, or its
    /// parameters conflict.
    pub fn add_observer<M, S: IntoSystem<M>>(&mut self, system: S) -> ObserverId {
        self.register_observer(None, system)
    }

    /// Like [`Self::add_observer`], for events triggered at `entity`
    /// ([`Self::trigger_targets`]) and no others. When `entity` is despawned
    /// the observer is dropped the next time something is triggered at it.
    pub fn observe<M, S: IntoSystem<M>>(&mut self, entity: Entity, system: S) -> ObserverId {
        self.register_observer(Some(entity), system)
    }

    fn register_observer<M, S: IntoSystem<M>>(
        &mut self,
        scope: Option<Entity>,
        system: S,
    ) -> ObserverId {
        let mut system: Box<dyn System> = Box::new(system.into_system());
        self.observers.probe = Some(Vec::new());
        system.initialize(self);
        let notes = self.observers.probe.take().unwrap_or_default();
        let (event, lifecycle) = match notes.as_slice() {
            [one] => *one,
            [] => panic!("observer `{}` has no `On<E>` parameter", system.name()),
            _ => panic!(
                "observer `{}` has more than one `On<E>` parameter",
                system.name()
            ),
        };
        if let Some(lifecycle) = lifecycle {
            self.hooks.register_lifecycle(lifecycle);
        }
        let id = self.observers.next;
        self.observers.next += 1;
        self.observers.entries.insert(
            id,
            ObserverEntry {
                event,
                scope,
                system: Some(system),
            },
        );
        self.observers.by_event.entry(event).or_default().push(id);
        ObserverId(id)
    }

    /// Drops an observer. Returns whether there was one. An observer may
    /// remove itself.
    pub fn remove_observer(&mut self, id: ObserverId) -> bool {
        let Some(entry) = self.observers.entries.remove(&id.0) else {
            return false;
        };
        if let Some(ids) = self.observers.by_event.get_mut(&entry.event) {
            ids.retain(|other| *other != id.0);
        }
        true
    }

    /// How many observers the world has.
    pub fn observer_count(&self) -> usize {
        self.observers.len()
    }

    /// Triggers `event`: every observer of it that is not scoped to an
    /// entity runs now (see the module docs for a trigger made from inside
    /// an observer).
    pub fn trigger<E: Event>(&mut self, event: E) {
        self.queue_trigger(event, None);
    }

    /// Triggers `event` at `target`: the global observers of it run, and the
    /// ones added for `target` with [`Self::observe`].
    pub fn trigger_targets<E: Event>(&mut self, event: E, target: Entity) {
        self.queue_trigger(event, Some(target));
    }

    fn queue_trigger<E: Event>(&mut self, event: E, target: Option<Entity>) {
        self.observers
            .queue
            .push_back(Box::new(move |world| world.deliver(event, target)));
        self.deliver_queued();
    }

    /// Runs the waiting triggers one after another, unless that is already
    /// being done further up the stack.
    fn deliver_queued(&mut self) {
        if self.observers.dispatching {
            return;
        }
        self.observers.dispatching = true;
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            while let Some(job) = self.observers.queue.pop_front() {
                job(self);
            }
        }));
        self.observers.dispatching = false;
        if let Err(panic) = outcome {
            self.observers.queue.clear();
            resume_unwind(panic);
        }
    }

    /// Runs the observers of `E` for one trigger.
    fn deliver<E: Event>(&mut self, event: E, target: Option<Entity>) {
        let ids = match self.observers.by_event.get(&TypeId::of::<E>()) {
            Some(ids) if !ids.is_empty() => ids.clone(),
            _ => return,
        };
        if let Some(slot) = self.get_resource_mut::<TriggerSlot<E>>() {
            slot.0 = Some((event, target));
        }
        for id in ids {
            let Some(scope) = self.observers.entries.get(&id).map(|entry| entry.scope) else {
                continue; // removed since
            };
            if let Some(scope) = scope {
                if target != Some(scope) {
                    continue;
                }
                if !self.is_alive(scope) {
                    self.remove_observer(ObserverId(id));
                    continue;
                }
            }
            let Some(mut system) = self
                .observers
                .entries
                .get_mut(&id)
                .and_then(|entry| entry.system.take())
            else {
                continue;
            };
            system.run(self);
            if let Some(entry) = self.observers.entries.get_mut(&id) {
                entry.system = Some(system);
            }
        }
        if let Some(slot) = self.get_resource_mut::<TriggerSlot<E>>() {
            slot.0 = None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commands::Commands;
    use crate::message::{MessageReader, MessageWriter};
    use crate::schedule::{IntoScheduleConfigs, Schedule};
    use crate::system_param::{Local, Query, Res, ResMut};
    use crate::tick::Mut;

    struct Hit(u32);
    impl Event for Hit {}

    struct Miss(u32);
    impl Event for Miss {}

    struct Hp(u32);
    struct Tag(u32);

    #[derive(Default)]
    struct Log(Vec<String>);

    fn logged(w: &World) -> Vec<String> {
        w.get_resource::<Log>().unwrap().0.clone()
    }

    fn world() -> World {
        let mut w = World::new();
        w.insert_resource(Log::default());
        w
    }

    fn on_hit_a(on: On<Hit>, mut log: ResMut<Log>) {
        log.0.push(format!("a hit {}", on.event().0));
    }

    fn on_hit_b(on: On<Hit>, mut log: ResMut<Log>) {
        log.0.push(format!("b hit {}", on.0));
    }

    fn on_miss(on: On<Miss>, mut log: ResMut<Log>) {
        log.0.push(format!("miss {}", on.event().0));
    }

    // ── delivery ─────────────────────────────────────────────────────

    #[test]
    fn observers_of_an_event_run_in_the_order_they_were_added() {
        let mut w = world();
        w.add_observer(on_hit_b);
        w.add_observer(on_hit_a);
        w.trigger(Hit(1));
        w.trigger(Hit(2));
        assert_eq!(logged(&w), ["b hit 1", "a hit 1", "b hit 2", "a hit 2"]);
    }

    #[test]
    fn observers_only_hear_their_own_event() {
        let mut w = world();
        w.add_observer(on_hit_a);
        w.add_observer(on_miss);
        w.trigger(Miss(5));
        assert_eq!(logged(&w), ["miss 5"]);
    }

    #[test]
    fn triggering_an_event_nobody_observes_does_nothing() {
        let mut w = world();
        w.trigger(Hit(1));
        let e = w.spawn();
        w.trigger_targets(Hit(1), e);
        assert!(logged(&w).is_empty());
        assert_eq!(w.observer_count(), 0);
    }

    #[test]
    fn an_observer_can_use_queries_and_resources() {
        fn heal(on: On<Hit>, mut hp: Query<Mut<Hp>>, log: Res<Log>) {
            for mut h in hp.iter_mut() {
                h.0 += on.0;
            }
            let _ = log.0.len();
        }
        let mut w = world();
        let e = w.spawn();
        w.insert_static(e, Hp(10));
        w.add_observer(heal);
        w.trigger(Hit(5));
        assert_eq!(w.get_static::<Hp>(e).unwrap().0, 15);
    }

    #[test]
    fn an_observers_commands_are_applied_when_it_has_run() {
        #[derive(Default)]
        struct Spawned(Vec<Entity>);
        fn spawn_tag(on: On<Hit>, mut commands: Commands) {
            let n = on.0;
            commands.spawn().queue(move |w, e| {
                w.insert_static(e, Tag(n));
                w.get_resource_mut::<Spawned>().unwrap().0.push(e);
            });
        }
        let mut w = world();
        w.insert_resource(Spawned::default());
        w.add_observer(spawn_tag);
        w.trigger(Hit(7));
        w.trigger(Hit(8));
        let spawned = w.get_resource::<Spawned>().unwrap().0.clone();
        let tags: Vec<u32> = spawned
            .iter()
            .map(|e| w.get_static::<Tag>(*e).unwrap().0)
            .collect();
        assert_eq!(tags, [7, 8], "applied when each observer had run");
    }

    #[test]
    fn an_observers_own_state_lasts_across_triggers() {
        fn count(_on: On<Hit>, mut n: Local<u32>, mut log: ResMut<Log>) {
            *n += 1;
            log.0.push(format!("{}", *n));
        }
        let mut w = world();
        w.add_observer(count);
        w.trigger(Hit(0));
        w.trigger(Hit(0));
        w.trigger(Hit(0));
        assert_eq!(logged(&w), ["1", "2", "3"]);
    }

    #[test]
    fn an_observer_can_write_messages() {
        fn forward(on: On<Hit>, mut out: MessageWriter<Miss>) {
            out.write(Miss(on.0 * 2));
        }
        fn collect(reader: MessageReader<Miss>, mut log: ResMut<Log>) {
            for m in reader.read() {
                log.0.push(format!("got {}", m.0));
            }
        }
        let mut w = world();
        w.add_observer(forward);
        let mut s = Schedule::new("collect");
        s.add_systems(collect);
        w.trigger(Hit(2));
        w.trigger(Hit(3));
        s.run(&mut w);
        assert_eq!(logged(&w), ["got 4", "got 6"]);
    }

    // ── adding observers ─────────────────────────────────────────────

    #[test]
    fn an_observer_needs_exactly_one_event_parameter() {
        fn none(_log: ResMut<Log>) {}
        fn two(_a: On<Hit>, _b: On<Miss>) {}
        for add in [
            (|w: &mut World| {
                w.add_observer(none);
            }) as fn(&mut World),
            |w| {
                w.add_observer(two);
            },
        ] {
            let mut w = world();
            assert!(std::panic::catch_unwind(AssertUnwindSafe(|| add(&mut w))).is_err());
            assert_eq!(w.observer_count(), 0, "and nothing was added");
        }
    }

    #[test]
    fn an_observer_whose_parameters_conflict_is_refused_at_once() {
        fn bad(_on: On<Hit>, _a: ResMut<Log>, _b: ResMut<Log>) {}
        let mut w = world();
        assert!(std::panic::catch_unwind(AssertUnwindSafe(|| {
            w.add_observer(bad);
        }))
        .is_err());
    }

    #[test]
    fn on_outside_a_trigger_panics() {
        let mut w = world();
        w.add_observer(on_hit_a); // makes the slot
        let mut s = Schedule::new("wrong");
        s.add_systems(on_hit_a);
        let err = std::panic::catch_unwind(AssertUnwindSafe(|| s.run(&mut w)))
            .expect_err("scheduled observer");
        let text = err
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| err.downcast_ref::<&str>().map(|s| s.to_string()))
            .unwrap_or_default();
        assert!(text.contains("outside a trigger"), "{text}");
    }

    #[test]
    fn remove_observer_stops_it_and_says_whether_it_was_there() {
        let mut w = world();
        let a = w.add_observer(on_hit_a);
        let b = w.add_observer(on_hit_b);
        assert_eq!(w.observer_count(), 2);
        assert!(w.remove_observer(a));
        assert!(!w.remove_observer(a));
        assert_eq!(w.observer_count(), 1);
        w.trigger(Hit(1));
        assert_eq!(logged(&w), ["b hit 1"]);
        assert!(w.remove_observer(b));
        w.trigger(Hit(2));
        assert_eq!(logged(&w), ["b hit 1"]);
    }

    #[test]
    fn a_removed_observer_leaves_no_id_behind() {
        let mut w = world();
        let a = w.add_observer(on_hit_a);
        let _b = w.add_observer(on_hit_b);
        w.remove_observer(a);
        assert_eq!(w.observers.by_event[&TypeId::of::<Hit>()].len(), 1);
        assert_eq!(w.observers.entries.len(), 1);
    }

    #[test]
    fn on_panics_when_a_trigger_is_over_too() {
        let mut w = world();
        w.add_observer(on_hit_a);
        w.trigger(Hit(1));
        let mut s = Schedule::new("late");
        s.add_systems(on_hit_a);
        assert!(std::panic::catch_unwind(AssertUnwindSafe(|| s.run(&mut w))).is_err());
    }

    #[test]
    fn commands_trigger_at_an_entity_in_both_ways() {
        fn scoped(on: On<Hit>, mut log: ResMut<Log>) {
            log.0.push(format!("scoped {}", on.0));
        }
        fn aim(mut commands: Commands, target: Res<Target>) {
            commands.trigger_targets(Hit(1), target.0);
            commands.entity(target.0).trigger(Hit(2));
        }
        struct Target(Entity);
        let mut w = world();
        let e = w.spawn();
        w.insert_resource(Target(e));
        w.observe(e, scoped);
        let mut s = Schedule::new("aim");
        s.add_systems(aim);
        s.run(&mut w);
        assert_eq!(logged(&w), ["scoped 1", "scoped 2"]);
    }

    #[test]
    fn an_observer_can_remove_itself_and_the_others_still_run() {
        #[derive(Default)]
        struct Mine(Option<ObserverId>);
        fn once(_on: On<Hit>, mut commands: Commands, mine: Res<Mine>, mut log: ResMut<Log>) {
            log.0.push("once".into());
            let id = mine.0.unwrap();
            commands.queue(move |w| {
                w.remove_observer(id);
            });
        }
        let mut w = world();
        w.insert_resource(Mine::default());
        let id = w.add_observer(once);
        w.get_resource_mut::<Mine>().unwrap().0 = Some(id);
        w.add_observer(on_hit_a);
        w.trigger(Hit(1));
        w.trigger(Hit(2));
        assert_eq!(logged(&w), ["once", "a hit 1", "a hit 2"]);
        assert_eq!(w.observer_count(), 1);
    }

    // ── entities ─────────────────────────────────────────────────────

    #[test]
    fn a_trigger_at_an_entity_runs_global_observers_and_that_entitys() {
        fn at(on: On<Hit>, mut log: ResMut<Log>) {
            log.0
                .push(format!("global target {:?}", on.target().is_some()));
        }
        fn scoped(on: On<Hit>, mut log: ResMut<Log>) {
            log.0.push(format!("scoped {}", on.event().0));
        }
        let mut w = world();
        let e = w.spawn();
        let other = w.spawn();
        w.add_observer(at);
        w.observe(e, scoped);
        w.trigger_targets(Hit(1), e);
        w.trigger_targets(Hit(2), other);
        w.trigger(Hit(3));
        assert_eq!(
            logged(&w),
            [
                "global target true",
                "scoped 1",
                "global target true",
                "global target false"
            ]
        );
    }

    #[test]
    fn the_target_is_the_entity_the_event_was_triggered_at() {
        fn note(on: On<Hit>, mut seen: ResMut<Seen>) {
            seen.0.push(on.target());
        }
        #[derive(Default)]
        struct Seen(Vec<Option<Entity>>);
        let mut w = world();
        w.insert_resource(Seen::default());
        let e = w.spawn();
        w.add_observer(note);
        w.trigger_targets(Hit(0), e);
        w.trigger(Hit(0));
        assert_eq!(w.get_resource::<Seen>().unwrap().0, [Some(e), None]);
    }

    #[test]
    fn an_observer_of_a_despawned_entity_is_dropped_when_it_is_next_triggered_at() {
        let mut w = world();
        let e = w.spawn();
        w.observe(e, on_hit_a);
        assert_eq!(w.observer_count(), 1);
        w.despawn(e);
        w.trigger_targets(Hit(1), e);
        assert!(logged(&w).is_empty());
        assert_eq!(w.observer_count(), 0);
    }

    // ── order ────────────────────────────────────────────────────────

    #[test]
    fn a_trigger_from_inside_an_observer_waits_for_the_triggers_other_observers() {
        fn first(_on: On<Hit>, mut commands: Commands, mut log: ResMut<Log>) {
            log.0.push("first".into());
            commands.queue(|w| w.trigger(Miss(1)));
        }
        fn second(_on: On<Hit>, mut log: ResMut<Log>) {
            log.0.push("second".into());
        }
        let mut w = world();
        w.add_observer(first);
        w.add_observer(second);
        w.add_observer(on_miss);
        w.trigger(Hit(0));
        assert_eq!(logged(&w), ["first", "second", "miss 1"]);
    }

    #[test]
    fn a_chain_of_triggers_is_a_queue_not_a_recursion() {
        fn again(on: On<Hit>, mut commands: Commands, mut log: ResMut<Log>) {
            log.0.push(format!("{}", on.0));
            let next = on.0 + 1;
            if next < 5_000 {
                commands.queue(move |w| w.trigger(Hit(next)));
            }
        }
        let mut w = world();
        w.add_observer(again);
        w.trigger(Hit(0));
        assert_eq!(
            logged(&w).len(),
            5_000,
            "the same observer, 5000 times, no stack"
        );
        assert_eq!(logged(&w)[4_999], "4999");
    }

    #[test]
    fn a_panicking_observer_leaves_the_world_able_to_deliver_again() {
        fn boom(_on: On<Miss>) {
            panic!("boom");
        }
        let mut w = world();
        w.add_observer(boom);
        w.add_observer(on_hit_a);
        assert!(std::panic::catch_unwind(AssertUnwindSafe(|| w.trigger(Miss(0)))).is_err());
        w.trigger(Hit(1));
        assert_eq!(logged(&w), ["a hit 1"]);
    }

    #[test]
    fn a_panic_drops_the_triggers_that_were_waiting() {
        #[derive(Default)]
        struct Armed(bool);
        fn queue_a_miss(_on: On<Hit>, mut commands: Commands, mut armed: ResMut<Armed>) {
            if !armed.0 {
                armed.0 = true;
                commands.queue(|w| w.trigger(Miss(1)));
            }
        }
        fn boom(_on: On<Hit>) {
            panic!("boom");
        }
        let mut w = world();
        w.insert_resource(Armed::default());
        let first = w.add_observer(queue_a_miss);
        w.add_observer(boom);
        w.add_observer(on_miss);
        assert!(std::panic::catch_unwind(AssertUnwindSafe(|| w.trigger(Hit(0)))).is_err());
        w.remove_observer(first);
        w.trigger(Miss(5));
        assert_eq!(
            logged(&w),
            ["miss 5"],
            "the Miss(1) that was waiting when it panicked is gone"
        );
    }

    #[test]
    fn commands_can_trigger_and_add_observers() {
        fn arm(mut commands: Commands) {
            commands.add_observer(on_hit_a);
            commands.trigger(Hit(9));
        }
        let mut w = world();
        let mut s = Schedule::new("arm");
        s.add_systems(arm);
        s.run(&mut w);
        assert_eq!(w.observer_count(), 1);
        assert_eq!(
            logged(&w),
            ["a hit 9"],
            "queued in order: add, then trigger"
        );
        s.run(&mut w);
        assert_eq!(w.observer_count(), 2);
        assert_eq!(logged(&w).len(), 3, "the second trigger met both observers");
    }

    #[test]
    fn entity_commands_observe_and_trigger_at() {
        fn scoped(on: On<Hit>, mut log: ResMut<Log>) {
            log.0.push(format!("scoped {}", on.0));
        }
        let mut w = world();
        #[derive(Default)]
        struct Spawned(Vec<Entity>);
        w.insert_resource(Spawned::default());
        let mut queue = crate::commands::CommandQueue::default();
        {
            let mut commands = queue.commands();
            commands
                .spawn()
                .observe(scoped)
                .queue(|w, e| w.get_resource_mut::<Spawned>().unwrap().0.push(e));
        }
        queue.apply(&mut w);
        assert_eq!(w.observer_count(), 1);
        let e = w.get_resource::<Spawned>().unwrap().0[0];
        w.trigger_targets(Hit(4), e);
        assert_eq!(logged(&w), ["scoped 4"]);
    }

    // ── lifecycle ────────────────────────────────────────────────────

    struct Mark(u32);

    macro_rules! record {
        ($fn_name:ident, $event:ty, $label:expr) => {
            fn $fn_name(on: On<$event>, mut log: ResMut<Log>, marks: Query<&Mark>) {
                let value = on
                    .target()
                    .and_then(|e| marks.get(e).ok())
                    .map(|m| m.0.to_string())
                    .unwrap_or_else(|| "none".to_string());
                log.0.push(format!("{} {}", $label, value));
            }
        };
    }
    record!(saw_add, Add<Mark>, "add");
    record!(saw_insert, Insert<Mark>, "insert");
    record!(saw_discard, Discard<Mark>, "discard");
    record!(saw_remove, Remove<Mark>, "remove");
    record!(saw_despawn, Despawn<Mark>, "despawn");

    fn lifecycle_world() -> World {
        let mut w = world();
        w.add_observer(saw_add);
        w.add_observer(saw_insert);
        w.add_observer(saw_discard);
        w.add_observer(saw_remove);
        w.add_observer(saw_despawn);
        w
    }

    #[test]
    fn adding_a_component_triggers_add_and_insert_after_it_is_there() {
        let mut w = lifecycle_world();
        let e = w.spawn();
        w.insert_static(e, Mark(1));
        assert_eq!(logged(&w), ["add 1", "insert 1"]);
    }

    #[test]
    fn overwriting_a_component_triggers_discard_and_insert_not_add() {
        // Only Sparse Shell data can be overwritten by `insert`; on an
        // archetype component `insert_static` does nothing if it is there.
        // (A `Query` does not see Shell data, hence "none".)
        let mut w = lifecycle_world();
        let e = w.spawn();
        w.insert(e, Mark(1));
        w.get_resource_mut::<Log>().unwrap().0.clear();
        w.insert(e, Mark(2));
        assert_eq!(logged(&w), ["discard none", "insert none"]);
    }

    #[test]
    fn inserting_an_archetype_component_that_is_there_triggers_nothing() {
        let mut w = lifecycle_world();
        let e = w.spawn();
        w.insert_static(e, Mark(1));
        w.get_resource_mut::<Log>().unwrap().0.clear();
        assert!(!w.insert_static(e, Mark(2)));
        assert!(logged(&w).is_empty());
    }

    #[test]
    fn removing_a_component_triggers_discard_and_remove_and_it_is_gone() {
        let mut w = lifecycle_world();
        let e = w.spawn();
        w.insert_static(e, Mark(1));
        w.get_resource_mut::<Log>().unwrap().0.clear();
        w.remove_static::<Mark>(e);
        assert_eq!(logged(&w), ["discard none", "remove none"]);
    }

    #[test]
    fn despawning_triggers_despawn_discard_and_remove_for_each_component() {
        let mut w = lifecycle_world();
        let e = w.spawn();
        w.insert_static(e, Mark(1));
        w.get_resource_mut::<Log>().unwrap().0.clear();
        w.despawn(e);
        let mut got = logged(&w);
        got.sort();
        assert_eq!(got, ["despawn none", "discard none", "remove none"]);
    }

    #[test]
    fn lifecycle_events_reach_the_entitys_own_observers() {
        fn mine(on: On<Add<Mark>>, mut log: ResMut<Log>) {
            log.0.push(format!("mine {:?}", on.target().is_some()));
        }
        let mut w = world();
        let e = w.spawn();
        let other = w.spawn();
        w.observe(e, mine);
        w.insert_static(other, Mark(1));
        w.insert_static(e, Mark(2));
        assert_eq!(logged(&w), ["mine true"], "only for e");
    }

    #[test]
    fn lifecycle_events_cover_sparse_shell_components_too() {
        let mut w = lifecycle_world();
        let e = w.spawn();
        w.insert(e, Mark(3));
        assert_eq!(
            logged(&w),
            ["add none", "insert none"],
            "(a Query does not see Shell data)"
        );
    }

    #[test]
    fn lifecycle_events_work_beside_a_component_hook() {
        fn hook(world: &mut crate::DeferredWorld<'_>, _ctx: crate::HookContext) {
            world
                .get_resource_mut::<Log>()
                .unwrap()
                .0
                .push("hook".into());
        }
        let mut w = world();
        w.register_component_hooks::<Mark>().on_add(hook);
        w.add_observer(saw_add);
        let e = w.spawn();
        w.insert_static(e, Mark(1));
        assert_eq!(
            logged(&w),
            ["hook", "add 1"],
            "the hook at once, the observer after"
        );
    }

    #[test]
    fn a_world_with_no_observers_has_no_hooks_to_pay_for() {
        let mut w = world();
        assert!(!w.hooks.is_active());
        w.add_observer(on_hit_a);
        assert!(!w.hooks.is_active(), "a plain event adds none");
        w.add_observer(saw_add);
        assert!(w.hooks.is_active());
    }

    #[test]
    fn lifecycle_events_of_other_component_types_do_not_fire() {
        let mut w = lifecycle_world();
        let e = w.spawn();
        w.insert_static(e, Tag(1));
        assert!(logged(&w).is_empty());
    }

    #[test]
    fn a_lifecycle_observer_can_react_with_commands() {
        fn mark_tag(on: On<Add<Mark>>, mut commands: Commands) {
            let target = on.target().unwrap();
            commands.entity(target).queue(|w, e| {
                w.insert_static(e, Tag(99));
            });
        }
        let mut w = world();
        w.add_observer(mark_tag);
        let e = w.spawn();
        w.insert_static(e, Mark(1));
        assert_eq!(w.get_static::<Tag>(e).unwrap().0, 99);
    }
}
