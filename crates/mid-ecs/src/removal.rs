// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-ecs.md, section "Removal tracking (`removal.rs`,
// `RemovedComponents`)"
// ============================================================================
//! The log behind [`RemovedComponents`](crate::RemovedComponents) and
//! [`World::removed_since`](crate::World::removed_since).
//!
//! A component that is removed (or goes with its entity's despawn) leaves
//! no trace in storage, so "which entities lost a `T` since I last
//! looked" needs a record. This is it: per tracked component type, a
//! list of `(entity, tick)` pairs, the tick being the world's change tick
//! when the removal happened.
//!
//! Three choices, each the opposite of bevy's, each for a reason the
//! rest of this crate already settled:
//!
//! * **Ticks, not message cursors.** Change detection here is "newer than
//!   the reader's last run". A removal is stamped the same way, so every
//!   reader (a system, or code holding a [`ChangeTracker`]) has its own
//!   window for free, with no per-reader state and no double buffer.
//! * **Opt-in per type.** Removal sits in the benchmarked migration paths.
//!   A world that tracks nothing pays one `is_empty` check per removal;
//!   only a type somebody asked about ([`World::track_removals`], which
//!   `RemovedComponents<T>` calls when its system is set up) pays for a
//!   lookup and a push.
//! * **Keyed by `TypeId`.** The archetype storage and the Sparse Shell have
//!   separate `ComponentId` spaces, and a type uses exactly one storage per
//!   world, so the type is the one key both agree on and a reader can name
//!   before anything of that type has ever been inserted.
//!
//! Nothing here trims the log: only the owner of *all* the readers knows
//! the oldest `last_run` any of them still holds. See
//! [`World::trim_removals`](crate::World::trim_removals).

use std::any::TypeId;

use crate::hash::TypeIdMap;
use crate::tick::Tick;
use crate::world::Entity;

/// Per tracked component type, the removals so far, oldest first.
#[derive(Default)]
pub(crate) struct RemovalLog {
    tracked: TypeIdMap<Vec<(Entity, Tick)>>,
}

impl RemovalLog {
    /// Starts recording removals of `type_id` from now on. Idempotent;
    /// removals before the first call are not recorded.
    pub(crate) fn track(&mut self, type_id: TypeId) {
        self.tracked.entry(type_id).or_default();
    }

    /// Whether any type is tracked at all: the one check an untracked
    /// world pays per removal.
    #[inline]
    pub(crate) fn is_active(&self) -> bool {
        !self.tracked.is_empty()
    }

    pub(crate) fn is_tracked(&self, type_id: TypeId) -> bool {
        self.tracked.contains_key(&type_id)
    }

    /// Records that `entity` lost its `type_id` component at `tick`. A
    /// type nobody tracks is ignored.
    #[inline]
    pub(crate) fn record(&mut self, type_id: TypeId, entity: Entity, tick: Tick) {
        if let Some(log) = self.tracked.get_mut(&type_id) {
            log.push((entity, tick));
        }
    }

    /// The entities that lost a `type_id` component after `last_run`, as
    /// seen at `this_run`, oldest first. An entity removed, re-added and
    /// removed again appears once per removal.
    ///
    /// Scans the whole retained log rather than stopping at the first old
    /// entry: an entry stamped later than `this_run` (a removal made
    /// after the reader's run began) is "not newer" by the wrapping
    /// comparison and must not hide the older ones before it. Trim the
    /// log and the scan stays short.
    pub(crate) fn since(
        &self,
        type_id: TypeId,
        last_run: Tick,
        this_run: Tick,
    ) -> impl Iterator<Item = Entity> + '_ {
        self.tracked
            .get(&type_id)
            .into_iter()
            .flatten()
            .filter(move |(_, tick)| tick.is_newer_than(last_run, this_run))
            .map(|&(entity, _)| entity)
    }

    /// How many removals of `type_id` the log holds (0 if untracked).
    pub(crate) fn len(&self, type_id: TypeId) -> usize {
        self.tracked.get(&type_id).map_or(0, Vec::len)
    }

    /// Drops every entry no reader whose last run was `at_or_before` could
    /// still see, as judged at `this_run`.
    pub(crate) fn trim(&mut self, at_or_before: Tick, this_run: Tick) {
        for log in self.tracked.values_mut() {
            log.retain(|(_, tick)| tick.is_newer_than(at_or_before, this_run));
        }
    }

    /// Drops every entry. Types stay tracked.
    pub(crate) fn clear(&mut self) {
        for log in self.tracked.values_mut() {
            log.clear();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commands::Commands;
    use crate::system::{IntoSystem, System};
    use crate::system_param::RemovedComponents;
    use crate::tick::ChangeTracker;
    use crate::world::World;
    use crate::{Res, ResMut};
    use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

    /// Archetype-storage components.
    #[derive(Debug, PartialEq)]
    struct Hp(u32);
    #[derive(Debug, PartialEq)]
    struct Mp(u32);
    /// A Sparse Shell component (inserted with `insert`, not `insert_static`).
    #[derive(Debug, PartialEq)]
    struct Cn(u32);
    /// Never tracked in any test.
    #[derive(Debug, PartialEq)]
    struct Other(u32);

    #[derive(Debug, Clone, Copy, PartialEq, FromBytes, IntoBytes, Immutable, KnownLayout)]
    #[repr(C)]
    struct Fa(u32);
    #[derive(Debug, Clone, Copy, PartialEq, FromBytes, IntoBytes, Immutable, KnownLayout)]
    #[repr(C)]
    struct Fb(u32);

    fn ent(world: &mut World) -> Entity {
        world.spawn()
    }

    /// The entities `T`'s removals show for `tracker`.
    fn removed<T: 'static>(world: &World, tracker: &ChangeTracker) -> Vec<Entity> {
        world.removed_since::<T>(tracker).collect()
    }

    /// Ticks forward the way the other tracker tests do and returns a
    /// tracker that sees only what happens after this call.
    fn arm(world: &mut World) -> ChangeTracker {
        world.increment_change_tick();
        let mut tracker = ChangeTracker::new();
        tracker.update(world);
        world.increment_change_tick();
        tracker
    }

    // ── The log ──────────────────────────────────────────────────────

    fn e(n: u32) -> Entity {
        let mut world = World::new();
        let mut last = world.spawn();
        for _ in 0..n {
            last = world.spawn();
        }
        last
    }

    #[test]
    fn an_untracked_type_is_not_recorded_and_a_tracked_one_is() {
        let mut log = RemovalLog::default();
        let (a, b) = (TypeId::of::<Hp>(), TypeId::of::<Mp>());
        assert!(!log.is_active());
        log.record(a, e(1), Tick::new(5));
        assert_eq!(log.len(a), 0, "nothing tracked yet");
        log.track(a);
        assert!(log.is_active());
        assert!(log.is_tracked(a) && !log.is_tracked(b));
        log.record(a, e(1), Tick::new(5));
        log.record(b, e(2), Tick::new(5));
        assert_eq!((log.len(a), log.len(b)), (1, 0));
    }

    #[test]
    fn tracking_twice_keeps_what_was_recorded() {
        let mut log = RemovalLog::default();
        let a = TypeId::of::<Hp>();
        log.track(a);
        log.record(a, e(1), Tick::new(3));
        log.track(a);
        assert_eq!(log.len(a), 1);
    }

    #[test]
    fn since_is_exactly_the_entries_newer_than_last_run_oldest_first() {
        let mut log = RemovalLog::default();
        let a = TypeId::of::<Hp>();
        log.track(a);
        for (n, tick) in [(1u32, 2u32), (2, 3), (3, 3), (4, 4), (5, 6)] {
            log.record(a, e(n), Tick::new(tick));
        }
        let at = |last: u32, this: u32| -> Vec<Entity> {
            log.since(a, Tick::new(last), Tick::new(this)).collect()
        };
        assert_eq!(at(0, 6), vec![e(1), e(2), e(3), e(4), e(5)]);
        assert_eq!(
            at(2, 6),
            vec![e(2), e(3), e(4), e(5)],
            "tick 2 is not newer than 2"
        );
        assert_eq!(at(3, 6), vec![e(4), e(5)]);
        assert_eq!(at(4, 6), vec![e(5)]);
        assert_eq!(at(6, 6), vec![]);
        assert_eq!(
            log.since(TypeId::of::<Mp>(), Tick::new(0), Tick::new(6))
                .count(),
            0,
            "an unknown type has nothing"
        );
    }

    #[test]
    fn an_entry_from_after_this_run_does_not_hide_older_ones_before_it() {
        // Stamped 9 while the reader runs at 5: not newer by the wrapping
        // comparison (it is "from the future"), and it sits at the END,
        // where a backward scan that stopped at the first old entry would
        // have stopped.
        let mut log = RemovalLog::default();
        let a = TypeId::of::<Hp>();
        log.track(a);
        log.record(a, e(1), Tick::new(4));
        log.record(a, e(2), Tick::new(9));
        let seen: Vec<_> = log.since(a, Tick::new(2), Tick::new(5)).collect();
        assert_eq!(seen, vec![e(1)]);
    }

    #[test]
    fn trim_drops_what_a_reader_at_that_tick_could_not_see_and_keeps_the_rest() {
        let mut log = RemovalLog::default();
        let a = TypeId::of::<Hp>();
        log.track(a);
        for (n, tick) in [(1u32, 2u32), (2, 3), (3, 4), (4, 5)] {
            log.record(a, e(n), Tick::new(tick));
        }
        log.trim(Tick::new(3), Tick::new(5));
        // Ticks 2 and 3 go (not newer than 3); 4 and 5 stay.
        let left: Vec<_> = log.since(a, Tick::new(0), Tick::new(5)).collect();
        assert_eq!(left, vec![e(3), e(4)]);
        assert!(log.is_tracked(a), "trimming does not untrack");
    }

    #[test]
    fn clear_empties_every_log_and_keeps_them_tracked() {
        let mut log = RemovalLog::default();
        let (a, b) = (TypeId::of::<Hp>(), TypeId::of::<Mp>());
        log.track(a);
        log.track(b);
        log.record(a, e(1), Tick::new(2));
        log.record(b, e(2), Tick::new(2));
        log.clear();
        assert_eq!((log.len(a), log.len(b)), (0, 0));
        assert!(log.is_tracked(a) && log.is_tracked(b) && log.is_active());
    }

    // ── World: what counts as a removal ──────────────────────────────

    #[test]
    fn a_removal_before_tracking_began_is_not_recorded() {
        let mut world = World::new();
        let early = ent(&mut world);
        world.insert_static(early, Hp(1));
        world.remove_static::<Hp>(early);
        world.track_removals::<Hp>();
        let tracker = ChangeTracker::new();
        assert!(removed::<Hp>(&world, &tracker).is_empty());
        assert!(world.tracks_removals::<Hp>());
        assert!(!world.tracks_removals::<Mp>());
    }

    #[test]
    #[should_panic(expected = "are not tracked")]
    fn asking_for_removals_of_an_untracked_type_is_a_loud_error_not_an_empty_answer() {
        let world = World::new();
        let _ = world.removed_since::<Hp>(&ChangeTracker::new()).count();
    }

    #[test]
    fn remove_static_is_recorded_with_the_world_tick() {
        let mut world = World::new();
        world.track_removals::<Hp>();
        let (a, b) = (ent(&mut world), ent(&mut world));
        world.insert_static(a, Hp(1));
        world.insert_static(b, Hp(2));
        let tracker = arm(&mut world);
        world.remove_static::<Hp>(a);
        assert_eq!(removed::<Hp>(&world, &tracker), vec![a]);
        world.increment_change_tick();
        let later = arm(&mut world);
        world.remove_static::<Hp>(b);
        assert_eq!(removed::<Hp>(&world, &tracker), vec![a, b]);
        assert_eq!(
            removed::<Hp>(&world, &later),
            vec![b],
            "each tracker has its own window"
        );
    }

    #[test]
    fn a_sparse_shell_component_is_recorded_too() {
        let mut world = World::new();
        world.track_removals::<Cn>();
        let e = ent(&mut world);
        world.insert(e, Cn(1));
        let tracker = arm(&mut world);
        assert_eq!(world.remove::<Cn>(e), Some(Cn(1)));
        assert_eq!(removed::<Cn>(&world, &tracker), vec![e]);
    }

    #[test]
    fn only_tracked_types_are_recorded() {
        let mut world = World::new();
        world.track_removals::<Hp>();
        let e = ent(&mut world);
        world.insert_static(e, Hp(1));
        world.insert_static(e, Other(2));
        world.remove_static::<Other>(e);
        world.remove_static::<Hp>(e);
        assert_eq!(world.removal_count::<Hp>(), 1);
        assert_eq!(world.removal_count::<Other>(), 0);
        assert!(!world.tracks_removals::<Other>());
    }

    #[test]
    fn nothing_that_removes_nothing_is_recorded() {
        let mut world = World::new();
        world.track_removals::<Hp>();
        world.track_removals::<Cn>();
        let (alive, dead) = (ent(&mut world), ent(&mut world));
        world.insert_static(dead, Hp(1));
        world.insert(dead, Cn(1));
        world.despawn(dead);
        let before = (world.removal_count::<Hp>(), world.removal_count::<Cn>());
        assert_eq!(before, (1, 1), "the despawn itself was recorded");
        // A component the entity does not have; a dead entity; a type
        // never registered.
        assert_eq!(world.remove_static::<Hp>(alive), None);
        assert_eq!(world.remove::<Cn>(alive), None);
        assert_eq!(world.remove_static::<Hp>(dead), None);
        assert_eq!(world.remove::<Cn>(dead), None);
        assert_eq!(world.remove_static::<Other>(alive), None);
        assert!(
            !world.despawn(dead),
            "a second despawn is not a second removal"
        );
        assert_eq!(
            (world.removal_count::<Hp>(), world.removal_count::<Cn>()),
            before
        );
    }

    #[test]
    fn replacing_a_value_by_inserting_over_it_is_not_a_removal() {
        let mut world = World::new();
        world.track_removals::<Hp>();
        world.track_removals::<Cn>();
        let e = ent(&mut world);
        world.insert_static(e, Hp(1));
        world.insert_static(e, Hp(2));
        world.insert(e, Cn(1));
        world.insert(e, Cn(2));
        assert_eq!(world.removal_count::<Hp>() + world.removal_count::<Cn>(), 0);
    }

    #[test]
    fn remove_bundle_records_every_member_once_and_only_when_it_removed() {
        let mut world = World::new();
        world.track_removals::<Hp>();
        world.track_removals::<Mp>();
        let (full, partial) = (ent(&mut world), ent(&mut world));
        world.insert_static(full, Hp(1));
        world.insert_static(full, Mp(2));
        world.insert_static(partial, Hp(3));
        let tracker = arm(&mut world);

        // The bundle is not whole on `partial`: nothing is removed,
        // nothing is recorded.
        assert!(world.remove_bundle::<(Hp, Mp)>(partial).is_none());
        assert!(removed::<Hp>(&world, &tracker).is_empty());

        assert_eq!(world.remove_bundle::<(Hp, Mp)>(full), Some((Hp(1), Mp(2))));
        assert_eq!(removed::<Hp>(&world, &tracker), vec![full]);
        assert_eq!(removed::<Mp>(&world, &tracker), vec![full]);
    }

    #[test]
    fn despawn_records_every_tracked_component_in_both_storages() {
        let mut world = World::new();
        world.track_removals::<Hp>();
        world.track_removals::<Mp>();
        world.track_removals::<Cn>();
        let (victim, bystander) = (ent(&mut world), ent(&mut world));
        for e in [victim, bystander] {
            world.insert_static(e, Hp(1));
            world.insert(e, Cn(1));
        }
        world.insert_static(victim, Mp(2));
        world.insert_static(victim, Other(9));
        let tracker = arm(&mut world);

        assert!(world.despawn(victim));
        assert_eq!(removed::<Hp>(&world, &tracker), vec![victim]);
        assert_eq!(removed::<Mp>(&world, &tracker), vec![victim]);
        assert_eq!(removed::<Cn>(&world, &tracker), vec![victim]);
        assert_eq!(world.removal_count::<Other>(), 0, "untracked");
        // The bystander kept everything.
        assert!(
            world.get_static::<Hp>(bystander).is_some() && world.get::<Cn>(bystander).is_some()
        );
    }

    #[test]
    fn despawning_an_entity_with_no_components_records_nothing() {
        let mut world = World::new();
        world.track_removals::<Hp>();
        let e = ent(&mut world);
        assert!(world.despawn(e));
        assert_eq!(world.removal_count::<Hp>(), 0);
    }

    #[test]
    fn despawn_still_removes_sparse_components_when_nothing_is_tracked() {
        // The untracked despawn is the old loop, unchanged.
        let mut world = World::new();
        let e = ent(&mut world);
        world.insert(e, Cn(1));
        world.insert_static(e, Hp(1));
        assert!(world.despawn(e));
        assert!(world.get::<Cn>(e).is_none() && world.get_static::<Hp>(e).is_none());
        assert!(!world.removals.is_active());
    }

    #[test]
    fn ffi_byte_removals_by_id_and_by_ids_are_recorded() {
        let mut world = World::new();
        let fa = world.register_ffi_static_component_writable::<Fa>("Fa");
        let fb = world.register_ffi_static_component_writable::<Fb>("Fb");
        world.track_removals::<Fa>();
        world.track_removals::<Fb>();
        let (x, y) = (ent(&mut world), ent(&mut world));
        for e in [x, y] {
            world.insert_static(e, Fa(1));
            world.insert_static(e, Fb(2));
        }
        let tracker = arm(&mut world);

        assert_eq!(world.remove_static_by_id(x, fa), Ok(()));
        assert_eq!(removed::<Fa>(&world, &tracker), vec![x]);
        assert!(removed::<Fb>(&world, &tracker).is_empty());

        assert_eq!(world.remove_static_by_ids(y, &[fa, fb]), Ok(()));
        assert_eq!(removed::<Fa>(&world, &tracker), vec![x, y]);
        assert_eq!(removed::<Fb>(&world, &tracker), vec![y]);

        // A refused removal (x no longer has Fa) records nothing.
        let n = world.removal_count::<Fa>();
        assert!(world.remove_static_by_id(x, fa).is_err());
        assert_eq!(world.removal_count::<Fa>(), n);
    }

    #[test]
    fn a_refused_many_removal_records_nothing_for_any_listed_component() {
        // `remove_static_by_ids` is all-or-nothing: `x` lacks `Fb`, so
        // nothing is removed, and nothing may be logged for `Fa` either.
        let mut world = World::new();
        let fa = world.register_ffi_static_component_writable::<Fa>("Fa");
        let fb = world.register_ffi_static_component_writable::<Fb>("Fb");
        world.track_removals::<Fa>();
        world.track_removals::<Fb>();
        let x = ent(&mut world);
        world.insert_static(x, Fa(1));
        assert!(world.remove_static_by_ids(x, &[fa, fb]).is_err());
        assert_eq!(world.removal_count::<Fa>(), 0);
        assert_eq!(world.removal_count::<Fb>(), 0);
        assert!(world.has_static::<Fa>(x), "and nothing was removed");
    }

    #[test]
    fn trim_and_clear_through_the_world() {
        let mut world = World::new();
        world.track_removals::<Hp>();
        let es: Vec<_> = (0..4).map(|_| ent(&mut world)).collect();
        for &e in &es {
            world.insert_static(e, Hp(1));
        }
        let mut trackers = Vec::new();
        for &e in &es {
            world.increment_change_tick();
            world.remove_static::<Hp>(e);
            let mut t = ChangeTracker::new();
            t.update(&world);
            trackers.push(t);
        }
        // trackers[i] was taken just after removal i: it sees the later ones.
        assert_eq!(removed::<Hp>(&world, &trackers[1]), vec![es[2], es[3]]);
        world.trim_removals(trackers[1].last_run());
        assert_eq!(
            world.removal_count::<Hp>(),
            2,
            "the two a tracker at that tick cannot see"
        );
        assert_eq!(removed::<Hp>(&world, &trackers[1]), vec![es[2], es[3]]);
        assert_eq!(
            removed::<Hp>(&world, &trackers[0]),
            vec![es[2], es[3]],
            "an older reader lost what was trimmed"
        );
        world.clear_removals();
        assert_eq!(world.removal_count::<Hp>(), 0);
        assert!(world.tracks_removals::<Hp>());
    }

    // ── As a system parameter ────────────────────────────────────────

    #[derive(Default)]
    struct Seen(Vec<Entity>);
    #[derive(Default)]
    struct SeenB(Vec<Entity>);
    struct Target(Entity);

    fn collect_hp(removed: RemovedComponents<Hp>, mut seen: ResMut<Seen>) {
        seen.0.extend(removed.iter());
    }

    #[test]
    fn a_system_sees_removals_since_its_own_previous_run() {
        let mut world = World::new();
        world.insert_resource(Seen::default());
        let es: Vec<_> = (0..4).map(|_| ent(&mut world)).collect();
        for &e in &es {
            world.insert_static(e, Hp(1));
        }
        let mut sys = collect_hp.into_system();
        sys.initialize(&mut world);
        let seen = |w: &World| w.get_resource::<Seen>().unwrap().0.clone();

        world.remove_static::<Hp>(es[0]);
        sys.run(&mut world);
        assert_eq!(seen(&world), vec![es[0]]);
        sys.run(&mut world);
        assert_eq!(seen(&world), vec![es[0]], "nothing new since the last run");
        world.remove_static::<Hp>(es[1]);
        world.despawn(es[2]);
        sys.run(&mut world);
        assert_eq!(seen(&world), vec![es[0], es[1], es[2]]);
    }

    #[test]
    fn a_removal_before_the_system_was_set_up_is_not_seen() {
        let mut world = World::new();
        world.insert_resource(Seen::default());
        let e = ent(&mut world);
        world.insert_static(e, Hp(1));
        world.remove_static::<Hp>(e);
        let mut sys = collect_hp.into_system();
        sys.run(&mut world);
        assert!(world.get_resource::<Seen>().unwrap().0.is_empty());
    }

    fn remove_via_commands(
        mut commands: Commands,
        removed: RemovedComponents<Hp>,
        target: Res<Target>,
        mut seen: ResMut<Seen>,
    ) {
        seen.0.extend(removed.iter());
        commands.entity(target.0).remove_static::<Hp>();
    }

    #[test]
    fn removals_a_system_queues_itself_are_seen_on_its_next_run() {
        // `Commands` apply after the run, at the world's later tick, so
        // the window of the system's next run contains them.
        let mut world = World::new();
        world.insert_resource(Seen::default());
        let e = ent(&mut world);
        world.insert_static(e, Hp(1));
        world.insert_resource(Target(e));
        let mut sys = remove_via_commands.into_system();
        sys.run(&mut world);
        assert!(
            world.get_resource::<Seen>().unwrap().0.is_empty(),
            "not yet"
        );
        assert!(world.get_static::<Hp>(e).is_none(), "the command ran");
        sys.run(&mut world);
        assert_eq!(world.get_resource::<Seen>().unwrap().0, vec![e]);
    }

    fn collect_cn(removed: RemovedComponents<Cn>, mut seen: ResMut<SeenB>) {
        seen.0.extend(removed.iter());
    }

    struct Count(usize, bool);

    fn measure(hp: RemovedComponents<Hp>, mut out: ResMut<Count>) {
        assert_eq!(hp.len(), hp.iter().count());
        out.0 = hp.len();
        out.1 = hp.is_empty();
    }

    #[test]
    fn len_and_is_empty_count_removals_and_an_entity_removed_twice_counts_twice() {
        let mut world = World::new();
        world.insert_resource(Count(99, false));
        let e = ent(&mut world);
        world.insert_static(e, Hp(1));
        let mut sys = measure.into_system();
        sys.initialize(&mut world);
        let read = |w: &World| {
            let c = w.get_resource::<Count>().unwrap();
            (c.0, c.1)
        };
        sys.run(&mut world);
        assert_eq!(read(&world), (0, true));
        world.remove_static::<Hp>(e);
        world.insert_static(e, Hp(2));
        world.remove_static::<Hp>(e);
        sys.run(&mut world);
        assert_eq!(read(&world), (2, false));
        sys.run(&mut world);
        assert_eq!(read(&world), (0, true));
    }

    #[test]
    fn two_systems_and_two_types_keep_independent_windows() {
        let mut world = World::new();
        world.insert_resource(Seen::default());
        world.insert_resource(SeenB::default());
        let (a, b) = (ent(&mut world), ent(&mut world));
        world.insert_static(a, Hp(1));
        world.insert(a, Cn(1));
        world.insert_static(b, Hp(1));
        world.insert(b, Cn(1));
        let mut hp_sys = collect_hp.into_system();
        let mut cn_sys = collect_cn.into_system();
        hp_sys.initialize(&mut world);
        cn_sys.initialize(&mut world);

        world.remove_static::<Hp>(a);
        hp_sys.run(&mut world);
        world.remove::<Cn>(b);
        world.despawn(a);
        hp_sys.run(&mut world);
        cn_sys.run(&mut world);
        // hp_sys: a at the first run; nothing at the second (a's Hp was
        // already gone when it was despawned).
        assert_eq!(world.get_resource::<Seen>().unwrap().0, vec![a]);
        // cn_sys ran once, late: b's removal and a's despawn, oldest first.
        assert_eq!(world.get_resource::<SeenB>().unwrap().0, vec![b, a]);
    }

    // ── Against a model ──────────────────────────────────────────────

    /// A tiny deterministic generator, so a failure names its seed.
    struct Lcg(u64);
    impl Lcg {
        fn below(&mut self, n: usize) -> usize {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 33) as usize) % n
        }
    }

    /// What the world should hold and what it should have logged, kept
    /// the dull way: a list of live entities, who holds what, and a list
    /// of `(entity, op)` per type for the removals that were being
    /// tracked when they happened. Shares nothing with the log.
    #[derive(Default)]
    struct Model {
        alive: Vec<Entity>,
        held: std::collections::HashMap<Entity, [bool; 3]>,
        removed: [Vec<(Entity, usize)>; 3],
        tracked: [bool; 3],
    }

    impl Model {
        fn lose(&mut self, entity: Entity, ty: usize, op: usize) {
            let held = self.held.get_mut(&entity).unwrap();
            if held[ty] {
                held[ty] = false;
                if self.tracked[ty] {
                    self.removed[ty].push((entity, op));
                }
            }
        }
    }

    /// Hp = 0, Mp = 1 (archetype storage), Cn = 2 (Sparse Shell).
    fn expect(model: &Model, ty: usize, after_op: usize) -> Vec<Entity> {
        model.removed[ty]
            .iter()
            .filter(|(_, op)| *op > after_op)
            .map(|(entity, _)| *entity)
            .collect()
    }

    fn actual(world: &World, ty: usize, tracker: &ChangeTracker) -> Vec<Entity> {
        match ty {
            0 => removed::<Hp>(world, tracker),
            1 => removed::<Mp>(world, tracker),
            _ => removed::<Cn>(world, tracker),
        }
    }

    #[test]
    fn a_random_history_of_both_storages_matches_a_model_for_every_tracker() {
        const OPS: usize = 300;
        for seed in 0..8u64 {
            let mut rng = Lcg(seed + 1);
            let mut world = World::new();
            let mut model = Model::default();
            world.track_removals::<Hp>();
            world.track_removals::<Cn>();
            model.tracked = [true, false, true];
            // (op the tracker was taken after, tracker)
            let mut trackers: Vec<(usize, ChangeTracker)> = vec![(0, ChangeTracker::new())];
            let mut removals_seen = [0usize; 3];

            for op in 1..=OPS {
                world.increment_change_tick();
                if op == 120 {
                    // Mp starts being tracked mid-history: removals before
                    // this are not recorded, by the model either.
                    world.track_removals::<Mp>();
                    model.tracked[1] = true;
                }
                let pick = |rng: &mut Lcg, model: &Model| -> Option<Entity> {
                    (!model.alive.is_empty()).then(|| model.alive[rng.below(model.alive.len())])
                };
                match rng.below(11) {
                    0 | 1 => {
                        let e = world.spawn();
                        model.alive.push(e);
                        model.held.insert(e, [false; 3]);
                    }
                    2 => {
                        if let Some(e) = pick(&mut rng, &model) {
                            world.insert_static(e, Hp(op as u32));
                            model.held.get_mut(&e).unwrap()[0] = true;
                        }
                    }
                    3 => {
                        if let Some(e) = pick(&mut rng, &model) {
                            world.insert_static(e, Mp(op as u32));
                            model.held.get_mut(&e).unwrap()[1] = true;
                        }
                    }
                    4 => {
                        if let Some(e) = pick(&mut rng, &model) {
                            world.insert(e, Cn(op as u32));
                            model.held.get_mut(&e).unwrap()[2] = true;
                        }
                    }
                    5 => {
                        if let Some(e) = pick(&mut rng, &model) {
                            world.remove_static::<Hp>(e);
                            model.lose(e, 0, op);
                        }
                    }
                    6 => {
                        if let Some(e) = pick(&mut rng, &model) {
                            world.remove_static::<Mp>(e);
                            model.lose(e, 1, op);
                        }
                    }
                    7 => {
                        if let Some(e) = pick(&mut rng, &model) {
                            world.remove::<Cn>(e);
                            model.lose(e, 2, op);
                        }
                    }
                    8 => {
                        if let Some(e) = pick(&mut rng, &model) {
                            let got = world.remove_bundle::<(Hp, Mp)>(e).is_some();
                            let held = model.held[&e];
                            assert_eq!(got, held[0] && held[1], "seed {seed} op {op}");
                            if got {
                                model.lose(e, 0, op);
                                model.lose(e, 1, op);
                            }
                        }
                    }
                    9 => {
                        if !model.alive.is_empty() {
                            let i = rng.below(model.alive.len());
                            let e = model.alive.swap_remove(i);
                            assert!(world.despawn(e));
                            for ty in 0..3 {
                                model.lose(e, ty, op);
                            }
                            model.held.remove(&e);
                        }
                    }
                    _ => {
                        let mut t = ChangeTracker::new();
                        t.update(&world);
                        trackers.push((op, t));
                    }
                }
                for ty in 0..3 {
                    removals_seen[ty] = model.removed[ty].len();
                }
            }

            for (after, tracker) in &trackers {
                for ty in 0..3 {
                    assert_eq!(
                        actual(&world, ty, tracker),
                        expect(&model, ty, *after),
                        "seed {seed}: type {ty}, tracker taken after op {after}"
                    );
                }
            }
            assert!(
                removals_seen.iter().all(|&n| n > 5),
                "seed {seed} must exercise every type: {removals_seen:?}"
            );
            assert!(trackers.len() > 3, "seed {seed} took too few trackers");

            // Trim at a middle tracker: every tracker at or after it must
            // see exactly what it saw before.
            let mid = trackers.len() / 2;
            let (mid_op, mid_tracker) = (trackers[mid].0, &trackers[mid].1);
            world.trim_removals(mid_tracker.last_run());
            for (after, tracker) in &trackers[mid..] {
                for ty in 0..3 {
                    assert_eq!(
                        actual(&world, ty, tracker),
                        expect(&model, ty, *after),
                        "seed {seed}: after trimming at op {mid_op}, type {ty}, tracker after op {after}"
                    );
                }
            }
            for ty in 0..3 {
                let log_len = match ty {
                    0 => world.removal_count::<Hp>(),
                    1 => world.removal_count::<Mp>(),
                    _ => world.removal_count::<Cn>(),
                };
                assert_eq!(
                    log_len,
                    expect(&model, ty, mid_op).len(),
                    "seed {seed}: trimmed log length, type {ty}"
                );
            }
        }
    }
}
