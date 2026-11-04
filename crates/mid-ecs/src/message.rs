// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-ecs-events.md, section "message.rs"
// ============================================================================
//! Messages: values that one system writes and others read.
//!
//! A [`Messages<M>`] is a resource holding every message of type `M`
//! written and not yet trimmed, each stamped with the tick it was written
//! at. A [`MessageReader`] shows its system the messages written since
//! that system last ran, the same "since the last run" that `Added` and
//! `Changed` measure, and a [`MessageWriter`] adds to the log.
//!
//! ```
//! use mid_ecs::schedule::{IntoScheduleConfigs, Schedule};
//! use mid_ecs::{MessageReader, MessageWriter, ResMut, World};
//!
//! struct Hit(u32);
//! struct Total(u32);
//!
//! fn strike(mut hits: MessageWriter<Hit>) {
//!     hits.write(Hit(3));
//!     hits.write(Hit(4));
//! }
//!
//! fn tally(hits: MessageReader<Hit>, mut total: ResMut<Total>) {
//!     total.0 += hits.read().map(|h| h.0).sum::<u32>();
//! }
//!
//! let mut world = World::new();
//! world.insert_resource(Total(0));
//! let mut schedule = Schedule::new("frame");
//! schedule.add_systems((strike, tally).chain());
//! schedule.run(&mut world);
//! schedule.run(&mut world);
//! assert_eq!(world.get_resource::<Total>().unwrap().0, 14);
//! ```
//!
//! # How this differs from `bevy_ecs`
//!
//! `bevy_ecs` keeps two buffers and swaps them once a frame, so a message
//! lives for one or two frames and a reader that does not run in that time
//! never sees it. Here a message lives until it is trimmed, and trimming
//! (`World::trim_messages_for_schedules`, which `mid-app` calls after each
//! frame) waits for every system that reads that type to have run past it.
//! A reader that runs rarely, or once per fixed step of a frame that made
//! none or three, misses nothing. The cost is that a reader which is
//! skipped for good holds its type's log back for good; see
//! [`World::trim_messages_for_schedules`].
//!
//! Reading does not consume: two readers each see every message, and
//! calling [`MessageReader::read`] twice in one run gives the same
//! messages twice.

use std::any::TypeId;

use crate::system_param::{missing_resource, register_resource, SystemMeta, SystemParam};
use crate::tick::{ComponentTicks, Tick};
use crate::world::World;
use crate::world_cell::WorldCell;

// ── the log ──────────────────────────────────────────────────────────

/// Every message of type `M` that has been written and not trimmed, oldest
/// first, as a resource. Systems reach it through [`MessageReader`] and
/// [`MessageWriter`]; code with a world uses [`World::write_message`] and
/// [`World::messages_since`].
pub struct Messages<M: 'static> {
    log: Vec<(Tick, M)>,
}

impl<M: 'static> Default for Messages<M> {
    fn default() -> Self {
        Self { log: Vec::new() }
    }
}

impl<M: 'static> Messages<M> {
    /// How many messages the log holds.
    pub fn len(&self) -> usize {
        self.log.len()
    }

    /// Whether the log holds none.
    pub fn is_empty(&self) -> bool {
        self.log.is_empty()
    }

    /// Every message the log holds, oldest first.
    pub fn iter(&self) -> impl Iterator<Item = &M> + '_ {
        self.log.iter().map(|(_, m)| m)
    }

    /// Drops every message.
    pub fn clear(&mut self) {
        self.log.clear();
    }

    #[inline]
    pub(crate) fn push(&mut self, tick: Tick, message: M) {
        self.log.push((tick, message));
    }

    /// The messages written after `last_run`, as seen at `this_run`,
    /// oldest first.
    ///
    /// Scans the whole log rather than stopping at the first old entry: an
    /// entry stamped later than `this_run` (a message written after the
    /// reader's run began) is "not newer" by the wrapping comparison and
    /// must not hide the older ones before it. Trim the log and the scan
    /// stays short.
    pub(crate) fn since(&self, last_run: Tick, this_run: Tick) -> impl Iterator<Item = &M> + '_ {
        self.log
            .iter()
            .filter(move |(tick, _)| tick.is_newer_than(last_run, this_run))
            .map(|(_, m)| m)
    }

    /// Drops every message no reader whose last run was `at_or_before`
    /// could still see, as judged at `this_run`.
    pub(crate) fn trim(&mut self, at_or_before: Tick, this_run: Tick) {
        self.log
            .retain(|(tick, _)| tick.is_newer_than(at_or_before, this_run));
    }

    /// Keeps every stamp within [`crate::MAX_CHANGE_AGE`] of `now`.
    pub(crate) fn check_ticks(&mut self, now: Tick) {
        for (tick, _) in self.log.iter_mut() {
            tick.check_tick(now);
        }
    }
}

// ── the parameters ───────────────────────────────────────────────────

/// Writes messages of type `M`: a system parameter. Each message is
/// stamped with the tick of the run that wrote it, so a reader sees it
/// from the first of its runs that starts after this one.
///
/// It is exclusive access to `Messages<M>`, so two systems that write the
/// same type are never run at the same time, and a system cannot also read
/// that type (it would see messages written during its own run).
pub struct MessageWriter<'w, M: 'static> {
    messages: &'w mut Messages<M>,
    ticks: &'w mut ComponentTicks,
    this_run: Tick,
}

impl<M: 'static> MessageWriter<'_, M> {
    /// Writes one message.
    pub fn write(&mut self, message: M) {
        self.ticks.changed = self.this_run;
        self.messages.push(self.this_run, message);
    }

    /// Writes every message of `messages`, in order.
    pub fn write_batch(&mut self, messages: impl IntoIterator<Item = M>) {
        for message in messages {
            self.write(message);
        }
    }
}

/// Reads the messages of type `M` written since its system last ran (on
/// its first run, all of them that are still in the log): a system
/// parameter.
pub struct MessageReader<'w, M: 'static> {
    messages: &'w Messages<M>,
    last_run: Tick,
    this_run: Tick,
}

impl<M: 'static> MessageReader<'_, M> {
    /// The messages written since this system last ran, oldest first. A
    /// message written during this run (by a system that started before
    /// this one ended) is not among them: it comes with the next run.
    pub fn read(&self) -> impl Iterator<Item = &M> + '_ {
        self.messages.since(self.last_run, self.this_run)
    }

    /// How many messages [`Self::read`] yields.
    pub fn len(&self) -> usize {
        self.read().count()
    }

    /// Whether [`Self::read`] yields none.
    pub fn is_empty(&self) -> bool {
        self.read().next().is_none()
    }
}

// SAFETY: registers a write of resource `Messages<M>`, the only thing it
// touches.
unsafe impl<M: Send + Sync + 'static> SystemParam for MessageWriter<'_, M> {
    type State = ();
    type Item<'w, 's> = MessageWriter<'w, M>;

    fn init_state(world: &mut World, meta: &mut SystemMeta) -> Self::State {
        world.init_message::<M>();
        register_resource::<Messages<M>>(meta, true);
    }

    unsafe fn get_param<'w, 's>(
        _state: &'s mut Self::State,
        meta: &SystemMeta,
        world: WorldCell<'w>,
        this_run: Tick,
    ) -> Self::Item<'w, 's> {
        // SAFETY: the write was registered and checked against every
        // other parameter, so nothing else reads or writes the log
        // meanwhile.
        match unsafe { world.resource_mut_with_ticks::<Messages<M>>() } {
            Some((messages, ticks)) => MessageWriter {
                messages,
                ticks,
                this_run,
            },
            None => missing_resource::<Messages<M>>(meta),
        }
    }
}

// SAFETY: registers a read of resource `Messages<M>`, the only thing it
// touches.
unsafe impl<M: Send + Sync + 'static> SystemParam for MessageReader<'_, M> {
    type State = ();
    type Item<'w, 's> = MessageReader<'w, M>;

    fn init_state(world: &mut World, meta: &mut SystemMeta) -> Self::State {
        world.init_message::<M>();
        register_resource::<Messages<M>>(meta, false);
        meta.add_message_read(TypeId::of::<M>());
    }

    unsafe fn get_param<'w, 's>(
        _state: &'s mut Self::State,
        meta: &SystemMeta,
        world: WorldCell<'w>,
        this_run: Tick,
    ) -> Self::Item<'w, 's> {
        // SAFETY: the read was registered and checked against every other
        // parameter, so nothing writes the log meanwhile.
        match unsafe { world.resource_with_ticks::<Messages<M>>() } {
            Some((messages, _)) => MessageReader {
                messages,
                last_run: meta.last_run(),
                this_run,
            },
            None => missing_resource::<Messages<M>>(meta),
        }
    }
}

// ── the registry ─────────────────────────────────────────────────────

/// What the world does to a message log of a type it does not name: one
/// per type that was initialized ([`World::init_message`]).
pub(crate) struct MessageType {
    pub(crate) type_id: TypeId,
    pub(crate) trim: fn(&mut World, Tick, Tick),
    pub(crate) check_ticks: fn(&mut World, Tick),
    pub(crate) clear: fn(&mut World),
}

impl MessageType {
    fn of<M: Send + Sync + 'static>() -> Self {
        Self {
            type_id: TypeId::of::<M>(),
            trim: |world, at_or_before, this_run| {
                if let Some(messages) = world.get_resource_mut::<Messages<M>>() {
                    messages.trim(at_or_before, this_run);
                }
            },
            check_ticks: |world, now| {
                if let Some(messages) = world.get_resource_mut::<Messages<M>>() {
                    messages.check_ticks(now);
                }
            },
            clear: |world| {
                if let Some(messages) = world.get_resource_mut::<Messages<M>>() {
                    messages.clear();
                }
            },
        }
    }
}

/// Every message type the world has, and when its logs were last trimmed
/// for the schedules ([`World::trim_messages_for_schedules`]).
#[derive(Default)]
pub(crate) struct MessageRegistry {
    pub(crate) types: Vec<MessageType>,
    pub(crate) last_trim: Tick,
}

impl MessageRegistry {
    pub(crate) fn contains(&self, type_id: TypeId) -> bool {
        self.types.iter().any(|t| t.type_id == type_id)
    }

    pub(crate) fn register<M: Send + Sync + 'static>(&mut self) {
        self.types.push(MessageType::of::<M>());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tick::MAX_CHANGE_AGE;

    fn t(n: u32) -> Tick {
        Tick::new(n)
    }

    fn seen(m: &Messages<&'static str>, last: u32, this: u32) -> Vec<&'static str> {
        m.since(t(last), t(this)).copied().collect()
    }

    fn log() -> Messages<&'static str> {
        let mut m = Messages::default();
        m.push(t(2), "a");
        m.push(t(4), "b");
        m.push(t(4), "c");
        m.push(t(7), "d");
        m
    }

    #[test]
    fn a_new_log_is_empty() {
        let m = Messages::<u8>::default();
        assert!(m.is_empty());
        assert_eq!(m.len(), 0);
        assert_eq!(m.iter().count(), 0);
    }

    #[test]
    fn since_gives_what_was_stamped_after_the_last_run_up_to_this_one() {
        let m = log();
        assert_eq!(seen(&m, 0, 10), ["a", "b", "c", "d"], "first run: all");
        assert_eq!(seen(&m, 2, 10), ["b", "c", "d"], "stamp 2 is not after 2");
        assert_eq!(seen(&m, 3, 10), ["b", "c", "d"]);
        assert_eq!(seen(&m, 4, 10), ["d"], "both stamped 4 were seen");
        assert_eq!(seen(&m, 7, 10), Vec::<&str>::new());
    }

    #[test]
    fn a_message_stamped_after_this_run_waits_for_the_next() {
        let m = log();
        assert_eq!(seen(&m, 0, 4), ["a", "b", "c"], "d is stamped 7, after 4");
        assert_eq!(seen(&m, 4, 4), Vec::<&str>::new());
        assert_eq!(seen(&m, 4, 9), ["d"], "and it is there for the run after");
    }

    #[test]
    fn a_late_entry_in_the_middle_does_not_hide_the_ones_before_it() {
        // A writer that started earlier can finish after one stamped later.
        let mut m = Messages::default();
        m.push(t(3), "early");
        m.push(t(9), "late");
        m.push(t(4), "middle");
        assert_eq!(seen(&m, 0, 5), ["early", "middle"]);
    }

    #[test]
    fn trim_drops_what_a_reader_at_that_tick_has_seen() {
        let mut m = log();
        m.trim(t(4), t(10));
        assert_eq!(m.iter().copied().collect::<Vec<_>>(), ["d"]);
        m.trim(t(7), t(10));
        assert!(m.is_empty());
    }

    #[test]
    fn trim_at_zero_drops_nothing() {
        let mut m = log();
        m.trim(Tick::ZERO, t(10));
        assert_eq!(m.len(), 4);
    }

    #[test]
    fn clear_empties_the_log() {
        let mut m = log();
        m.clear();
        assert!(m.is_empty());
    }

    #[test]
    fn init_message_registers_a_type_once() {
        let mut world = World::new();
        world.init_message::<u8>();
        world.init_message::<u8>();
        world.write_message(1u8);
        world.init_message::<u16>();
        assert_eq!(world.messages.types.len(), 2);
    }

    #[test]
    fn the_worlds_tick_check_reaches_every_log_and_the_last_trim() {
        let mut world = World::new();
        world.write_message(1u8);
        world.write_message("text");
        world.messages.last_trim = Tick::new(1);
        world.set_change_tick(Tick::new(MAX_CHANGE_AGE + 500));
        world.check_change_ticks();
        let stamp = |w: &World| {
            (
                w.get_resource::<Messages<u8>>().unwrap().log[0].0.get(),
                w.get_resource::<Messages<&str>>().unwrap().log[0].0.get(),
            )
        };
        assert_eq!(stamp(&world), (500, 500), "both logs, to now minus the age");
        assert_eq!(world.messages.last_trim.get(), 500);
    }

    #[test]
    fn check_ticks_clamps_stamps_that_have_aged_past_the_limit() {
        let mut m = Messages::default();
        m.push(t(1), "old");
        m.push(t(MAX_CHANGE_AGE + 100), "recent");
        let now = t(MAX_CHANGE_AGE + 200);
        m.check_ticks(now);
        let stamps: Vec<u32> = m.log.iter().map(|(tick, _)| tick.get()).collect();
        assert_eq!(stamps[0], 200 + 0, "now minus the maximum age");
        assert_eq!(stamps[1], MAX_CHANGE_AGE + 100, "left alone");
    }
}
