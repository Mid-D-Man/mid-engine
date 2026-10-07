// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-ecs.md, section "tick.rs"
// ============================================================================
//! Change ticks: the counter and comparison logic `Added`/`Changed` query
//! filters (`filter.rs`) are built on.
//!
//! `World` owns one monotonically increasing [`Tick`], advanced by
//! [`crate::World::increment_change_tick`]. Every archetype-tracked
//! component value has a paired `ComponentTicks` recording the tick it
//! was last inserted and the tick it was last mutated through
//! [`crate::World::get_static_mut`]. A [`ChangeTracker`] is a caller-held
//! "since when" marker: create one, pass it to a query, call
//! [`ChangeTracker::update`] after — the same shape as a `bevy_ecs`
//! system's own last-run tick, made explicit because this crate has no
//! `Schedule` to hold it implicitly.
//!
//! **Simplified relative to `bevy_ecs`.** The comparison
//! (`Tick::is_newer_than`) is the same wrapping-counter technique, read
//! directly from `Mid-D-Man/bevy`'s `change_detection/tick.rs`. What's
//! deliberately not here: bevy periodically scans every stored tick
//! (`check_tick`) to clamp its age below `MAX_CHANGE_AGE`, because a
//! `Schedule` running indefinitely will otherwise let `this_run - tick`
//! overflow `u32::MAX` and invert the comparison. Nothing here runs that
//! scan. The comparison stays correct as long as `this_run - last_run`
//! and `this_run - added`/`this_run - changed` never exceed
//! `u32::MAX / 2` in practice — at one `increment_change_tick` per
//! rendered frame, that's on the order of a billion frames between a
//! value's insertion and a query checking it, so this is a real limit,
//! not a rounding error, but not one worth a background scanner for yet.

use std::ops::{Deref, DerefMut};

/// A point on the world's change-tick counter. Comparisons
/// (`is_newer_than`) are all that ever matters about a `Tick` — the raw
/// value has no meaning on its own.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Tick(u32);

impl Tick {
    /// The tick before any `World::increment_change_tick` call. Also
    /// [`ChangeTracker::new`]'s `last_run`, so a fresh tracker sees every
    /// component inserted so far as added/changed.
    pub const ZERO: Self = Self(0);

    pub const fn new(value: u32) -> Self {
        Self(value)
    }

    pub const fn get(self) -> u32 {
        self.0
    }

    /// Whether `self` (an insertion or mutation tick) counts as having
    /// happened after `last_run` — i.e. is at least as recent as
    /// anything from `last_run` up to and including `this_run`. See this
    /// module's doc comment for the wraparound caveat.
    #[inline]
    pub fn is_newer_than(self, last_run: Tick, this_run: Tick) -> bool {
        let ticks_since_insert = this_run.0.wrapping_sub(self.0);
        let ticks_since_last_run = this_run.0.wrapping_sub(last_run.0);
        ticks_since_last_run > ticks_since_insert
    }
}

/// The two ticks stored per archetype-tracked component value: when it
/// was inserted, and when it was last mutated through
/// [`crate::World::get_static_mut`]. A freshly inserted value has
/// `added == changed`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ComponentTicks {
    pub(crate) added: Tick,
    pub(crate) changed: Tick,
}

impl ComponentTicks {
    pub(crate) fn new(tick: Tick) -> Self {
        Self {
            added: tick,
            changed: tick,
        }
    }

    pub(crate) fn is_added(&self, last_run: Tick, this_run: Tick) -> bool {
        self.added.is_newer_than(last_run, this_run)
    }

    pub(crate) fn is_changed(&self, last_run: Tick, this_run: Tick) -> bool {
        self.changed.is_newer_than(last_run, this_run)
    }
}

/// A caller-held "since when" marker for [`crate::World::query_added`] and
/// [`crate::World::query_changed`] — the manual stand-in for what a
/// `bevy_ecs` system's own last-run tick tracks automatically. Typical
/// use: keep one per place in the caller's own game loop that needs
/// change detection, and call [`Self::update`] once that place has
/// finished using the query's results for this step.
#[derive(Debug, Clone, Copy)]
pub struct ChangeTracker {
    last_run: Tick,
}

impl ChangeTracker {
    /// Starts at [`Tick::ZERO`], so the first query against this tracker
    /// sees every component inserted so far as added/changed — the same
    /// "first run sees everything" behavior a brand-new `bevy_ecs` system
    /// has.
    pub const fn new() -> Self {
        Self {
            last_run: Tick::ZERO,
        }
    }

    pub(crate) fn last_run(&self) -> Tick {
        self.last_run
    }

    /// Advances this tracker to `world`'s current tick. Call once per
    /// step, after using the query results this tracker gated — mirrors
    /// a system's last-run tick being set to the tick it just ran at.
    pub fn update(&mut self, world: &crate::World) {
        self.last_run = world.change_tick();
    }
}

impl Default for ChangeTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Mutable access to one archetype-tracked component value that records
/// the write: obtained from the bulk mutable queries
/// ([`crate::World::query_static_mut`] and friends), the bulk counterpart
/// of [`crate::World::get_static_mut`].
///
/// Reading through it ([`Deref`]) leaves the value's change tick alone;
/// *mutably* dereferencing it ([`DerefMut`], so any `*m = ..`, `m.field =
/// ..` or `&mut *m`) stamps the value as changed at the world's tick as of
/// query creation, which is what [`crate::World::query_changed`] and the
/// C `..._changed_rows` function read. That makes a row a query merely
/// visits free of charge: only rows actually written to are reported
/// changed, unlike [`crate::World::get_static_mut`], which marks on the
/// call whether or not the caller then writes. Marking is a single `u32`
/// store through a pointer the iterator already holds, no lookup.
///
/// What it deliberately is not: bevy's `Mut` also carries the system's
/// last-run tick, so `is_changed()` takes no argument there. There is no
/// system here, so [`Self::is_added`]/[`Self::is_changed`] take the
/// [`ChangeTracker`] the caller is already holding. See
/// `docs/mid-ecs.md`, "tick.rs" and "Mutable queries".
pub struct Mut<'a, T> {
    value: &'a mut T,
    ticks: &'a mut ComponentTicks,
    this_run: Tick,
}

impl<'a, T> Mut<'a, T> {
    pub(crate) fn new(value: &'a mut T, ticks: &'a mut ComponentTicks, this_run: Tick) -> Self {
        Self {
            value,
            ticks,
            this_run,
        }
    }

    /// Whether this value was inserted after `tracker`'s last-checked
    /// tick, the per-row form of [`crate::World::query_added`].
    pub fn is_added(&self, tracker: &ChangeTracker) -> bool {
        self.ticks.is_added(tracker.last_run(), self.this_run)
    }

    /// Whether this value was inserted or mutated after `tracker`'s
    /// last-checked tick, the per-row form of
    /// [`crate::World::query_changed`]. Includes a write made earlier in
    /// this same iteration through this very `Mut` only if the world's
    /// tick has moved past `tracker`'s since: see `tick.rs`'s note on
    /// calling [`crate::World::increment_change_tick`] between a step's
    /// mutations and the check that should see them.
    pub fn is_changed(&self, tracker: &ChangeTracker) -> bool {
        self.ticks.is_changed(tracker.last_run(), self.this_run)
    }

    /// Marks the value changed without writing to it, for a mutation made
    /// some other way (through interior mutability, say).
    pub fn set_changed(&mut self) {
        self.ticks.changed = self.this_run;
    }

    /// Mutable access that does *not* mark the value changed: for a write
    /// that should be invisible to `Changed`, such as re-applying a value
    /// derived from state the reader already has. The caller is asserting
    /// that no change-detecting reader needs to see this write.
    pub fn bypass_change_detection(&mut self) -> &mut T {
        self.value
    }
}

impl<T> Deref for Mut<'_, T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &T {
        self.value
    }
}

impl<T> DerefMut for Mut<'_, T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut T {
        self.ticks.changed = self.this_run;
        self.value
    }
}

impl<T: std::fmt::Debug> std::fmt::Debug for Mut<'_, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("Mut").field(&*self.value).finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_newer_than_basic_ordering() {
        let t0 = Tick::new(0);
        let t5 = Tick::new(5);
        let t10 = Tick::new(10);
        // Inserted at 5, tracker last ran at 0, world now at 10: newer.
        assert!(t5.is_newer_than(t0, t10));
        // Inserted at 5, tracker last ran at 5, world now at 10: not
        // newer -- it was already there last time this tracker checked.
        assert!(!t5.is_newer_than(t5, t10));
        // Inserted at 5, tracker last ran at 6: not newer.
        assert!(!t5.is_newer_than(Tick::new(6), t10));
        // Inserted exactly at this_run (just happened): newer.
        assert!(t10.is_newer_than(t0, t10));
    }

    #[test]
    fn fresh_tracker_at_tick_zero_sees_nothing_before_any_increment() {
        // last_run == this_run == 0: nothing can be "newer".
        assert!(!Tick::ZERO.is_newer_than(Tick::ZERO, Tick::ZERO));
    }

    #[test]
    fn a_component_older_than_the_trackers_last_run_is_not_newer() {
        let inserted_at = Tick::new(3);
        let last_run = Tick::new(3);
        let this_run = Tick::new(3);
        assert!(!inserted_at.is_newer_than(last_run, this_run));
    }

    #[test]
    fn is_newer_than_handles_u32_wraparound() {
        // this_run has wrapped past u32::MAX back to a small value;
        // last_run and the insertion tick are both "before" the wrap, in
        // wrapping-subtraction terms. `wrapping_sub` makes the small
        // `this_run` value still act as strictly after them.
        let last_run = Tick::new(u32::MAX - 5);
        let this_run = Tick::new(4); // wrapped: 10 ticks after last_run
        let inserted_after_last_run = Tick::new(u32::MAX - 2); // 3 ticks after last_run
        let inserted_before_last_run = Tick::new(u32::MAX - 8); // 3 ticks before last_run
        assert!(inserted_after_last_run.is_newer_than(last_run, this_run));
        assert!(!inserted_before_last_run.is_newer_than(last_run, this_run));
    }

    #[test]
    fn component_ticks_is_added_only_true_right_after_insertion() {
        let ct = ComponentTicks::new(Tick::new(5));
        assert!(ct.is_added(Tick::new(0), Tick::new(6)));
        assert!(!ct.is_added(Tick::new(5), Tick::new(6)));
        assert!(ct.is_changed(Tick::new(0), Tick::new(6)));
    }

    #[test]
    fn component_ticks_changed_can_move_independently_of_added() {
        let mut ct = ComponentTicks::new(Tick::new(1));
        ct.changed = Tick::new(9);
        assert!(
            !ct.is_added(Tick::new(5), Tick::new(10)),
            "added stays at 1"
        );
        assert!(
            ct.is_changed(Tick::new(5), Tick::new(10)),
            "changed moved to 9"
        );
    }

    #[test]
    fn change_tracker_starts_at_zero_and_update_advances_it() {
        let tracker = ChangeTracker::new();
        assert_eq!(tracker.last_run(), Tick::ZERO);
        assert_eq!(ChangeTracker::default().last_run(), Tick::ZERO);
    }

    #[test]
    fn mut_reads_do_not_mark_and_writes_do() {
        let mut value = 5u32;
        let mut ticks = ComponentTicks::new(Tick::new(1));
        {
            let m = Mut::new(&mut value, &mut ticks, Tick::new(7));
            assert_eq!(*m, 5);
        }
        assert_eq!(ticks.changed, Tick::new(1), "a read leaves the tick alone");
        {
            let mut m = Mut::new(&mut value, &mut ticks, Tick::new(7));
            *m = 6;
        }
        assert_eq!(
            ticks.changed,
            Tick::new(7),
            "a write stamps this run's tick"
        );
        assert_eq!(ticks.added, Tick::new(1), "and never touches `added`");
        assert_eq!(value, 6);
    }

    #[test]
    fn mut_set_changed_and_bypass() {
        let mut value = 5u32;
        let mut ticks = ComponentTicks::new(Tick::new(1));
        {
            let mut m = Mut::new(&mut value, &mut ticks, Tick::new(9));
            *m.bypass_change_detection() = 8;
        }
        assert_eq!(ticks.changed, Tick::new(1));
        assert_eq!(value, 8);
        Mut::new(&mut value, &mut ticks, Tick::new(9)).set_changed();
        assert_eq!(ticks.changed, Tick::new(9));
    }

    #[test]
    fn mut_debug_shows_the_value_not_the_ticks() {
        let mut value = 5u32;
        let mut ticks = ComponentTicks::new(Tick::new(1));
        let m = Mut::new(&mut value, &mut ticks, Tick::new(2));
        assert_eq!(format!("{m:?}"), "Mut(5)");
    }
}
