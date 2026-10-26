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
//! **Wraparound.** A tick is a `u32` that wraps, and the comparison
//! (`Tick::is_newer_than`) is the same wrapping-counter technique as
//! `bevy_ecs`'s, read directly from `Mid-D-Man/bevy`'s
//! `change_detection/tick.rs`: it measures two *ages* (`this_run - tick`) as
//! unsigned numbers and compares them, so it is right as long as no age
//! reaches `u32::MAX`; a value that is more than 2^32 ticks old would look
//! brand new. [`World::check_change_ticks`](crate::World::check_change_ticks)
//! is what prevents that: it walks every tick the world stores (component
//! rows, resources, the removal log) and clamps any whose age exceeds
//! [`MAX_CHANGE_AGE`] to exactly that age (an old-enough value is "old", and
//! the exact age no longer matters). [`crate::World::increment_change_tick`]
//! runs the scan itself every [`CHECK_TICK_THRESHOLD`] ticks, so a world's
//! own ticks stay in range with no caller effort. Ticks the world does not
//! store are the caller's to keep in range: a system's last run
//! ([`crate::System::check_change_tick`], [`crate::SystemState::check_change_tick`])
//! and a [`ChangeTracker`] ([`ChangeTracker::check_tick`]); call them with
//! the world's tick at least once per [`CHECK_TICK_THRESHOLD`] ticks (once
//! per step is fine, each is one compare).

use std::ops::{Deref, DerefMut};

/// How many ticks may pass between two scans of every stored tick (see the
/// module docs). `bevy_ecs`'s value: at one tick per rendered frame, a
/// scan every 518 million frames.
pub const CHECK_TICK_THRESHOLD: u32 = 518_400_000;

/// The oldest age a stored tick is allowed to have: a scan clamps anything
/// older to exactly this. Chosen (as in `bevy_ecs`) so that even a tick
/// that has gone `CHECK_TICK_THRESHOLD` ticks unchecked since the last scan
/// is still younger than `u32::MAX` ticks, the one thing
/// [`Tick::is_newer_than`] needs.
pub const MAX_CHANGE_AGE: u32 = u32::MAX - (2 * CHECK_TICK_THRESHOLD - 1);

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
    /// The number of ticks from `other` forward to `self`, wrapping: how
    /// old `other` is at the moment `self`. `now.relative_to(stamp)` is
    /// the age of `stamp`.
    pub const fn relative_to(self, other: Tick) -> Tick {
        Tick(self.0.wrapping_sub(other.0))
    }

    /// Keeps a stored tick's age at `now` within [`MAX_CHANGE_AGE`]: if it
    /// is older, sets it to exactly that age and returns `true`. This is
    /// what [`crate::World::check_change_ticks`] does to every tick the
    /// world stores; call it on ticks you store yourself.
    pub fn check_tick(&mut self, now: Tick) -> bool {
        if now.relative_to(*self).get() > MAX_CHANGE_AGE {
            *self = now.relative_to(Tick::new(MAX_CHANGE_AGE));
            true
        } else {
            false
        }
    }

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

    /// Keeps both ticks within [`MAX_CHANGE_AGE`] of `now`; whether
    /// either was clamped.
    pub(crate) fn check_tick(&mut self, now: Tick) -> bool {
        let added = self.added.check_tick(now);
        let changed = self.changed.check_tick(now);
        added || changed
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

    /// A tracker whose last run is `tick`, for tests that need a
    /// specific age.
    #[cfg(test)]
    pub(crate) const fn at(tick: Tick) -> Self {
        Self { last_run: tick }
    }

    /// Keeps the tracker's last-run tick within [`MAX_CHANGE_AGE`] of
    /// `now` (see [`Tick::check_tick`]); whether it was clamped. A tracker
    /// the caller holds across more than [`CHECK_TICK_THRESHOLD`] ticks
    /// needs this called with the world's tick.
    pub fn check_tick(&mut self, now: Tick) -> bool {
        self.last_run.check_tick(now)
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

/// Shared access to one archetype-tracked component value together with
/// its change ticks: the read-side counterpart of [`Mut`], obtained as a
/// term of [`crate::World::query_data`] and friends (`Ref<T>`,
/// `Option<Ref<T>>`).
///
/// Dereferences to `&T` and costs nothing beyond what `&T` costs plus two
/// extra pointers. What it adds is the per-row question "was this value
/// added or changed since my tracker last looked?":
/// [`Self::is_added`] and [`Self::is_changed`], the same arguments
/// [`Mut`] takes, for the same reason (there is no system here to hold a
/// last-run tick). Use it when the answer should be an `if` inside the
/// loop; use the [`crate::Added`] / [`crate::Changed`] filters when rows
/// that fail should not be visited at all.
///
/// `Ref` is `Copy`: it is two shared references and a tick.
pub struct Ref<'a, T> {
    value: &'a T,
    ticks: &'a ComponentTicks,
    this_run: Tick,
}

impl<'a, T> Ref<'a, T> {
    pub(crate) fn new(value: &'a T, ticks: &'a ComponentTicks, this_run: Tick) -> Self {
        Self {
            value,
            ticks,
            this_run,
        }
    }

    /// Whether this value was inserted after `tracker`'s last-checked
    /// tick: the per-row form of [`crate::World::query_added`].
    pub fn is_added(&self, tracker: &ChangeTracker) -> bool {
        self.ticks.is_added(tracker.last_run(), self.this_run)
    }

    /// Whether this value was inserted or mutated after `tracker`'s
    /// last-checked tick: the per-row form of
    /// [`crate::World::query_changed`].
    pub fn is_changed(&self, tracker: &ChangeTracker) -> bool {
        self.ticks.is_changed(tracker.last_run(), self.this_run)
    }

    /// The plain reference, with the full lifetime of the query borrow
    /// rather than this wrapper's.
    pub fn into_inner(self) -> &'a T {
        self.value
    }
}

impl<T> Clone for Ref<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for Ref<'_, T> {}

impl<T> Deref for Ref<'_, T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &T {
        self.value
    }
}

impl<T: std::fmt::Debug> std::fmt::Debug for Ref<'_, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("Ref").field(self.value).finish()
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

    #[test]
    fn ref_reads_through_deref_and_reports_ticks_against_a_tracker() {
        let value = 5u32;
        let ticks = ComponentTicks::new(Tick::new(4));
        let r = Ref::new(&value, &ticks, Tick::new(6));
        assert_eq!(*r, 5);
        let fresh = ChangeTracker::new();
        assert!(r.is_added(&fresh));
        assert!(r.is_changed(&fresh));
        let mut caught_up = ChangeTracker::new();
        caught_up.last_run = Tick::new(4);
        assert!(!r.is_added(&caught_up), "inserted at the tracker's tick");
        assert!(!r.is_changed(&caught_up));
        caught_up.last_run = Tick::new(3);
        assert!(r.is_added(&caught_up));
    }

    #[test]
    fn ref_is_copy_and_into_inner_returns_the_plain_reference() {
        let value = 7u32;
        let ticks = ComponentTicks::new(Tick::new(1));
        let r = Ref::new(&value, &ticks, Tick::new(2));
        let copy = r;
        assert_eq!(*r, *copy);
        let plain: &u32 = r.into_inner();
        assert_eq!(*plain, 7);
        assert_eq!(format!("{copy:?}"), "Ref(7)");
    }

    // ── wraparound: check_tick ───────────────────────────────────────

    #[test]
    fn relative_to_is_the_wrapping_age() {
        assert_eq!(Tick::new(10).relative_to(Tick::new(4)), Tick::new(6));
        assert_eq!(Tick::new(4).relative_to(Tick::new(10)), Tick::new(u32::MAX - 5));
        assert_eq!(Tick::new(3).relative_to(Tick::new(u32::MAX)), Tick::new(4));
    }

    #[test]
    fn check_tick_clamps_exactly_what_is_older_than_max_change_age() {
        let now = Tick::new(4_000_000_000);
        // Age 0, and exactly MAX_CHANGE_AGE: untouched.
        for age in [0, 1, MAX_CHANGE_AGE - 1, MAX_CHANGE_AGE] {
            let mut t = Tick::new(now.get().wrapping_sub(age));
            let before = t;
            assert!(!t.check_tick(now), "age {age}");
            assert_eq!(t, before);
        }
        // One older, and far older: set to exactly MAX_CHANGE_AGE.
        for age in [MAX_CHANGE_AGE + 1, MAX_CHANGE_AGE + 12_345, u32::MAX] {
            let mut t = Tick::new(now.get().wrapping_sub(age));
            assert!(t.check_tick(now), "age {age}");
            assert_eq!(now.relative_to(t).get(), MAX_CHANGE_AGE, "age {age}");
        }
    }

    #[test]
    fn check_tick_works_across_the_counter_wrapping() {
        // `now` just after a wrap: a stamp from before it is young, a
        // stamp "ahead" of it (huge wrapping age) is old.
        let now = Tick::new(5);
        let mut young = Tick::new(u32::MAX - 100);
        assert!(!young.check_tick(now));
        assert_eq!(now.relative_to(young).get(), 106);
        let mut old = Tick::new(10);
        assert!(old.check_tick(now));
        assert_eq!(now.relative_to(old).get(), MAX_CHANGE_AGE);
    }

    #[test]
    fn component_ticks_check_clamps_each_tick_on_its_own() {
        let now = Tick::new(3_000_000_000);
        let mut t = ComponentTicks {
            added: Tick::new(now.get().wrapping_sub(MAX_CHANGE_AGE + 77)),
            changed: Tick::new(now.get() - 5),
        };
        assert!(t.check_tick(now));
        assert_eq!(now.relative_to(t.added).get(), MAX_CHANGE_AGE);
        assert_eq!(now.relative_to(t.changed).get(), 5, "the young one is left alone");
        assert!(!t.check_tick(now), "a second check changes nothing");
    }

    #[test]
    fn a_tracker_is_clamped_like_any_other_tick() {
        let mut tracker = ChangeTracker::new(); // last run: tick 0
        let now = Tick::new(MAX_CHANGE_AGE + 50);
        assert!(tracker.check_tick(now));
        assert_eq!(now.relative_to(tracker.last_run()).get(), MAX_CHANGE_AGE);
        assert!(!tracker.check_tick(now));
    }

    /// The failure `check_tick` exists to prevent, shown with bare ticks:
    /// a stamp that is 2^32 + 5_000_000 ticks old reads as 5 million ticks
    /// old, so it looks newer than a tracker that last ran 10 million ticks
    /// ago.
    #[test]
    fn a_stamp_that_wrapped_looks_new_unless_it_was_checked_on_the_way() {
        let stamp = Tick::new(100);
        let first_check = Tick::new(stamp.get().wrapping_add(3_400_000_000));
        let much_later = Tick::new(stamp.get().wrapping_add(5_000_000)); // 2^32 + 5M ticks on
        let last_run = Tick::new(much_later.get().wrapping_sub(10_000_000));

        // Unchecked: the age wrapped to 5M, below the tracker's 10M.
        assert!(stamp.is_newer_than(last_run, much_later), "the bug");

        // Checked at `first_check`, when it was 3.4e9 old (over the limit).
        let mut checked = stamp;
        assert!(checked.check_tick(first_check));
        assert!(
            !checked.is_newer_than(last_run, much_later),
            "clamped on the way, it is still old when `much_later` comes"
        );
    }
}
