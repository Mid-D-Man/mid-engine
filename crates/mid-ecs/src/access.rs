// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-ecs.md, section "Access sets: `access.rs`"
// ============================================================================
//! Access sets: what a piece of work reads and writes, and whether two
//! pieces of work can run at the same time.
//!
//! This is the primitive layer a scheduler and a parallel iterator stand
//! on (gap-list items 4 and 5). Nothing here runs anything. It answers
//! one question, exactly: *could these two accesses touch the same data
//! with at least one of them writing?*
//!
//! Three types, from flat to structured:
//!
//! - [`Access`]: component and resource reads and writes, as sets, plus
//!   "everything" flags for exclusive work (a system taking `&mut World`).
//! - [`FilteredAccess`]: an [`Access`] plus the archetype constraints of
//!   the query that produced it (`with` / `without` component sets). Two
//!   filtered accesses whose constraints can never hold for the same
//!   entity cannot conflict, whatever they write: `Query<&mut A,
//!   With<B>>` and `Query<&mut A, Without<B>>` run side by side.
//! - [`FilteredAccessSet`]: the filtered accesses of one system, one per
//!   parameter, plus their union for a fast "no conflict at all" answer.
//!
//! **Keyed by `TypeId`, not by registered id.** A system is analysed
//! before the world holds any entity, and a component type the world has
//! never seen has no `ComponentId` yet; keying by `TypeId` makes the
//! answer a fact about the types alone, the same choice `query_data_mut`'s
//! own conflict check makes. The `&'static str` kept beside each id is
//! the type's name, for messages only; it takes no part in comparisons.
//!
//! **Components and resources are separate namespaces.** A component `A`
//! and a resource `A` never conflict.
//!
//! **What is not modelled.** Row-level filters (`Added`, `Changed`) are
//! treated as what they are at archetype level (the component must be
//! present) and as a *read* of that component's change ticks, so a system
//! that writes `T` conflicts with one that filters on `Changed<T>`. An
//! `Or` filter adds none of its arms' `with`/`without` constraints (only
//! one arm needs to hold, and dropping a constraint can only make two
//! accesses look more likely to overlap, never less), but it does add
//! their reads. Both choices are conservative: they can report a conflict
//! that a finer analysis would not, never miss a real one.
//!
//! Exactness: for queries whose filters use no `Or`, [`FilteredAccess`]
//! conflicts are *exact* (a test enumerates every archetype of a small
//! component universe and compares); with `Or` they are a superset.

use std::any::{type_name, TypeId};
use std::collections::BTreeMap;
use std::fmt;

use crate::data_filter::DataFilter;
use crate::query_data::QueryData;

/// A `TypeId` and the type's name (names are for messages only).
type Named = BTreeMap<TypeId, &'static str>;

// ── AccessConflicts ──────────────────────────────────────────────────

/// What two accesses disagree about, as returned by the `get_conflicts`
/// methods. Empty means they can run together.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct AccessConflicts {
    all: bool,
    names: Vec<&'static str>,
}

impl AccessConflicts {
    fn none() -> Self {
        Self::default()
    }

    /// `true` if there is no conflict.
    pub fn is_empty(&self) -> bool {
        !self.all && self.names.is_empty()
    }

    /// `true` if one side takes a whole class of data (every component,
    /// or every resource) and the other touches any of it: exclusive
    /// work against anything.
    pub fn is_all(&self) -> bool {
        self.all
    }

    /// The names of the specific component and resource types in
    /// conflict, sorted and without repeats. Empty when the conflict is
    /// only [`Self::is_all`].
    pub fn names(&self) -> &[&'static str] {
        &self.names
    }

    fn merge(&mut self, other: AccessConflicts) {
        self.all |= other.all;
        self.names.extend(other.names);
        self.names.sort_unstable();
        self.names.dedup();
    }
}

impl fmt::Display for AccessConflicts {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_empty() {
            return f.write_str("no conflict");
        }
        let mut first = true;
        if self.all {
            f.write_str("whole-world access")?;
            first = false;
        }
        for name in &self.names {
            if !first {
                f.write_str(", ")?;
            }
            write!(f, "`{name}`")?;
            first = false;
        }
        Ok(())
    }
}

// ── Access ───────────────────────────────────────────────────────────

/// Component and resource reads and writes, with no regard to which
/// entities are involved (see [`FilteredAccess`] for that). A type that is
/// written is *not* also listed as read: a write already conflicts with
/// everything a read does.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Access {
    component_reads: Named,
    component_writes: Named,
    resource_reads: Named,
    resource_writes: Named,
    read_all_components: bool,
    write_all_components: bool,
    read_all_resources: bool,
    write_all_resources: bool,
}

/// One class of data (components, or resources): its read set, write set
/// and the two "everything" flags, borrowed from an [`Access`].
struct Class<'a> {
    reads: &'a Named,
    writes: &'a Named,
    read_all: bool,
    write_all: bool,
}

impl Class<'_> {
    fn touches_any(&self) -> bool {
        self.read_all || self.write_all || !self.reads.is_empty() || !self.writes.is_empty()
    }

    fn writes_any(&self) -> bool {
        self.write_all || !self.writes.is_empty()
    }

    /// The conflicts between `self` and `other` in this class. `ignore`
    /// removes specific types from the *named* result (types the caller
    /// knows cannot be present on a shared entity); the "all" verdicts
    /// are not affected by it.
    fn conflicts(&self, other: &Class<'_>, ignore: &dyn Fn(TypeId) -> bool) -> AccessConflicts {
        let mut out = AccessConflicts::none();
        if (self.write_all && other.touches_any())
            || (other.write_all && self.touches_any())
            || (self.read_all && other.writes_any())
            || (other.read_all && self.writes_any())
        {
            out.all = true;
        }
        let mut add = |(id, name): (&TypeId, &&'static str)| {
            if !ignore(*id) {
                out.names.push(*name);
            }
        };
        // A write against anything the other touches...
        for entry in self.writes.iter() {
            let (id, _) = entry;
            if other.writes.contains_key(id) || other.reads.contains_key(id) {
                add(entry);
            }
        }
        // ...and a read against the other's writes.
        for entry in self.reads.iter() {
            let (id, _) = entry;
            if other.writes.contains_key(id) {
                add(entry);
            }
        }
        out.names.sort_unstable();
        out.names.dedup();
        out
    }
}

impl Access {
    /// No access at all.
    pub fn new() -> Self {
        Self::default()
    }

    fn components(&self) -> Class<'_> {
        Class {
            reads: &self.component_reads,
            writes: &self.component_writes,
            read_all: self.read_all_components,
            write_all: self.write_all_components,
        }
    }

    fn resources(&self) -> Class<'_> {
        Class {
            reads: &self.resource_reads,
            writes: &self.resource_writes,
            read_all: self.read_all_resources,
            write_all: self.write_all_resources,
        }
    }

    /// Records a read of component `T`. A no-op if `T` is already written
    /// (the write covers it).
    pub fn add_component_read<T: 'static>(&mut self) {
        let id = TypeId::of::<T>();
        if !self.component_writes.contains_key(&id) {
            self.component_reads.insert(id, type_name::<T>());
        }
    }

    /// Records a write of component `T` (which also covers reading it).
    pub fn add_component_write<T: 'static>(&mut self) {
        let id = TypeId::of::<T>();
        self.component_reads.remove(&id);
        self.component_writes.insert(id, type_name::<T>());
    }

    /// Records a read of resource `T`.
    pub fn add_resource_read<T: 'static>(&mut self) {
        let id = TypeId::of::<T>();
        if !self.resource_writes.contains_key(&id) {
            self.resource_reads.insert(id, type_name::<T>());
        }
    }

    /// Records a write of resource `T` (which also covers reading it).
    pub fn add_resource_write<T: 'static>(&mut self) {
        let id = TypeId::of::<T>();
        self.resource_reads.remove(&id);
        self.resource_writes.insert(id, type_name::<T>());
    }

    /// Reads every component and every resource, present and future.
    pub fn read_all(&mut self) {
        self.read_all_components = true;
        self.read_all_resources = true;
    }

    /// Reads and writes everything: the access of work that takes
    /// `&mut World`. Conflicts with any other access that touches
    /// anything, and with nothing that is empty.
    pub fn write_all(&mut self) {
        self.write_all_components = true;
        self.write_all_resources = true;
        self.read_all_components = true;
        self.read_all_resources = true;
    }

    /// `true` if this access reads or writes anything at all.
    pub fn is_empty(&self) -> bool {
        !self.components().touches_any() && !self.resources().touches_any()
    }

    /// `true` if this access writes everything ([`Self::write_all`]).
    pub fn is_exclusive(&self) -> bool {
        self.write_all_components && self.write_all_resources
    }

    /// Whether component `T` is read (or written, or everything is read).
    pub fn reads_component<T: 'static>(&self) -> bool {
        let id = TypeId::of::<T>();
        self.read_all_components
            || self.component_reads.contains_key(&id)
            || self.component_writes.contains_key(&id)
    }

    /// Whether component `T` is written (or everything is).
    pub fn writes_component<T: 'static>(&self) -> bool {
        self.write_all_components || self.component_writes.contains_key(&TypeId::of::<T>())
    }

    /// Whether resource `T` is read (or written, or everything is read).
    pub fn reads_resource<T: 'static>(&self) -> bool {
        let id = TypeId::of::<T>();
        self.read_all_resources
            || self.resource_reads.contains_key(&id)
            || self.resource_writes.contains_key(&id)
    }

    /// Whether resource `T` is written (or everything is).
    pub fn writes_resource<T: 'static>(&self) -> bool {
        self.write_all_resources || self.resource_writes.contains_key(&TypeId::of::<T>())
    }

    /// Adds everything `other` reads and writes to `self`.
    pub fn extend(&mut self, other: &Access) {
        for (id, name) in &other.component_writes {
            self.component_reads.remove(id);
            self.component_writes.insert(*id, name);
        }
        for (id, name) in &other.component_reads {
            if !self.component_writes.contains_key(id) {
                self.component_reads.insert(*id, name);
            }
        }
        for (id, name) in &other.resource_writes {
            self.resource_reads.remove(id);
            self.resource_writes.insert(*id, name);
        }
        for (id, name) in &other.resource_reads {
            if !self.resource_writes.contains_key(id) {
                self.resource_reads.insert(*id, name);
            }
        }
        self.read_all_components |= other.read_all_components;
        self.write_all_components |= other.write_all_components;
        self.read_all_resources |= other.read_all_resources;
        self.write_all_resources |= other.write_all_resources;
    }

    /// What `self` and `other` conflict on, as if every entity could have
    /// every component (no archetype constraints: see
    /// [`FilteredAccess::get_conflicts`] for those).
    pub fn get_conflicts(&self, other: &Access) -> AccessConflicts {
        self.conflicts_ignoring(other, &|_| false)
    }

    /// `true` if `self` and `other` can run at the same time.
    pub fn is_compatible(&self, other: &Access) -> bool {
        self.get_conflicts(other).is_empty()
    }

    /// Like [`Self::get_conflicts`] but leaving the component types for
    /// which `ignore` answers `true` out of the named result.
    fn conflicts_ignoring(
        &self,
        other: &Access,
        ignore: &dyn Fn(TypeId) -> bool,
    ) -> AccessConflicts {
        let mut out = self.components().conflicts(&other.components(), ignore);
        out.merge(self.resources().conflicts(&other.resources(), &|_| false));
        out
    }
}

// ── FilteredAccess ───────────────────────────────────────────────────

/// An [`Access`] together with the archetype constraints of the query it
/// came from: component types every matched entity has (`with`) and has
/// not (`without`).
///
/// The constraints are what let two accesses to the same component
/// coexist: if one only matches entities with a `B` and the other only
/// entities without, no entity is in both and nothing is shared.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FilteredAccess {
    access: Access,
    with: Named,
    without: Named,
}

impl FilteredAccess {
    /// No access, no constraints.
    pub fn new() -> Self {
        Self::default()
    }

    /// The access of the query `Q` filtered by `F`, without a world: the
    /// terms' reads and writes, the components those terms require, and
    /// `F`'s constraints and tick reads. This is the same description the
    /// real iterators act on.
    ///
    /// ```
    /// use mid_ecs::{FilteredAccess, Mut, With, Without};
    ///
    /// struct Position;
    /// struct Frozen;
    ///
    /// // Writers of `Position` over disjoint sets of entities can run
    /// // side by side...
    /// let frozen = FilteredAccess::of_query::<Mut<Position>, With<Frozen>>();
    /// let moving = FilteredAccess::of_query::<Mut<Position>, Without<Frozen>>();
    /// assert!(frozen.is_compatible(&moving));
    ///
    /// // ...but an unconstrained writer overlaps both.
    /// let everyone = FilteredAccess::of_query::<Mut<Position>, ()>();
    /// assert!(!everyone.is_compatible(&moving));
    /// assert!(!everyone.is_compatible(&frozen));
    /// ```
    ///
    /// # Panics
    /// If `Q` reads and writes, or writes twice, the same component (the
    /// rule `World::query_data_mut` enforces when a query is created). An
    /// `Added`/`Changed` *filter* on a component `Q` writes is allowed.
    pub fn of_query<Q: QueryData, F: DataFilter>() -> Self {
        let state = Q::get_state(&|_| None);
        let mut entries = Vec::new();
        Q::collect_access(&state, &mut entries);
        crate::archetype::validate_access("FilteredAccess::of_query", &entries);
        let mut out = Self::new();
        Q::update_access(&mut out);
        F::update_access(&mut out);
        out
    }

    /// Records a read of component `T`.
    pub fn add_component_read<T: 'static>(&mut self) {
        self.access.add_component_read::<T>();
    }

    /// Records a write of component `T`.
    pub fn add_component_write<T: 'static>(&mut self) {
        self.access.add_component_write::<T>();
    }

    /// Records a read of resource `T`.
    pub fn add_resource_read<T: 'static>(&mut self) {
        self.access.add_resource_read::<T>();
    }

    /// Records a write of resource `T`.
    pub fn add_resource_write<T: 'static>(&mut self) {
        self.access.add_resource_write::<T>();
    }

    /// Reads everything.
    pub fn read_all(&mut self) {
        self.access.read_all();
    }

    /// Reads and writes everything (exclusive work).
    pub fn write_all(&mut self) {
        self.access.write_all();
    }

    /// Requires matched entities to have component `T`.
    pub fn add_with<T: 'static>(&mut self) {
        self.with.insert(TypeId::of::<T>(), type_name::<T>());
    }

    /// Requires matched entities to lack component `T`.
    pub fn add_without<T: 'static>(&mut self) {
        self.without.insert(TypeId::of::<T>(), type_name::<T>());
    }

    /// The reads and writes, without the constraints.
    pub fn access(&self) -> &Access {
        &self.access
    }

    /// Whether the constraints require component `T`.
    pub fn requires<T: 'static>(&self) -> bool {
        self.with.contains_key(&TypeId::of::<T>())
    }

    /// Whether the constraints exclude component `T`.
    pub fn excludes<T: 'static>(&self) -> bool {
        self.without.contains_key(&TypeId::of::<T>())
    }

    /// `true` if the constraints contradict each other (a component both
    /// required and excluded): nothing can match, so this access can
    /// never touch data and conflicts with nothing.
    pub fn is_unsatisfiable(&self) -> bool {
        self.with.keys().any(|id| self.without.contains_key(id))
    }

    /// Adds what `other` records into `self`'s *access* only, leaving
    /// `self`'s constraints alone. `Or` uses this: every arm's reads
    /// count, none of its constraints do.
    pub(crate) fn extend_access(&mut self, other: &FilteredAccess) {
        self.access.extend(&other.access);
    }

    /// `true` if no entity can satisfy both `self`'s constraints and
    /// `other`'s: one requires a component the other excludes.
    fn filters_disjoint(&self, other: &FilteredAccess) -> bool {
        self.with.keys().any(|id| other.without.contains_key(id))
            || other.with.keys().any(|id| self.without.contains_key(id))
    }

    /// What `self` and `other` conflict on, taking the constraints into
    /// account: component conflicts vanish when no entity can be matched
    /// by both, and a component one side excludes cannot be the shared
    /// one. Resource conflicts do not depend on entities and always
    /// stand.
    pub fn get_conflicts(&self, other: &FilteredAccess) -> AccessConflicts {
        if self.is_unsatisfiable() || other.is_unsatisfiable() {
            return AccessConflicts::none();
        }
        let both_without =
            |id: TypeId| self.without.contains_key(&id) || other.without.contains_key(&id);
        let mut out = self
            .access
            .resources()
            .conflicts(&other.access.resources(), &|_| false);
        if !self.filters_disjoint(other) {
            out.merge(
                self.access
                    .components()
                    .conflicts(&other.access.components(), &both_without),
            );
        }
        out
    }

    /// `true` if `self` and `other` can run at the same time.
    pub fn is_compatible(&self, other: &FilteredAccess) -> bool {
        self.get_conflicts(other).is_empty()
    }
}

// ── FilteredAccessSet ────────────────────────────────────────────────

/// The accesses of one piece of work made of several parts (a system and
/// its parameters), one [`FilteredAccess`] per part, plus their union.
///
/// Two uses. *Inside* one system: before adding a parameter, ask
/// [`Self::get_conflicts_single`] whether it overlaps one already there
/// (two queries both writing `A` would alias). *Between* systems: ask
/// [`Self::get_conflicts`], which compares part against part, so two
/// systems whose queries have disjoint constraints can still run
/// together.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FilteredAccessSet {
    combined: Access,
    parts: Vec<FilteredAccess>,
}

impl FilteredAccessSet {
    /// An empty set: accesses nothing.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds one part.
    pub fn add(&mut self, part: FilteredAccess) {
        self.combined.extend(&part.access);
        self.parts.push(part);
    }

    /// The parts, in the order they were added.
    pub fn parts(&self) -> &[FilteredAccess] {
        &self.parts
    }

    /// The union of every part's access, constraints dropped: what the
    /// whole piece of work may touch, for a coarse check.
    pub fn combined_access(&self) -> &Access {
        &self.combined
    }

    /// `true` if some part is exclusive ([`Access::write_all`]).
    pub fn is_exclusive(&self) -> bool {
        self.combined.is_exclusive()
    }

    /// What `part` conflicts with among the parts already here. The check
    /// a system makes for each new parameter.
    pub fn get_conflicts_single(&self, part: &FilteredAccess) -> AccessConflicts {
        let mut out = AccessConflicts::none();
        for existing in &self.parts {
            out.merge(existing.get_conflicts(part));
        }
        out
    }

    /// What this set and `other` conflict on, part against part. Empty
    /// means the two pieces of work can run at the same time.
    pub fn get_conflicts(&self, other: &FilteredAccessSet) -> AccessConflicts {
        // The union conflicts at least wherever any pair does, so a clean
        // union is a clean answer.
        if self.combined.is_compatible(&other.combined) {
            return AccessConflicts::none();
        }
        let mut out = AccessConflicts::none();
        for mine in &self.parts {
            for theirs in &other.parts {
                out.merge(mine.get_conflicts(theirs));
            }
        }
        out
    }

    /// `true` if this set and `other` can run at the same time.
    pub fn is_compatible(&self, other: &FilteredAccessSet) -> bool {
        self.get_conflicts(other).is_empty()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;
    use crate::component::ComponentId;
    use crate::filter::{Or, With, Without};
    use crate::query_data::{Added, Changed, Has};
    use crate::tick::{Mut, Ref};
    use crate::world::Entity;

    struct A;
    struct B;
    struct C;
    struct D;

    fn access(f: impl FnOnce(&mut Access)) -> Access {
        let mut a = Access::new();
        f(&mut a);
        a
    }

    // ── Access ───────────────────────────────────────────────────────

    #[test]
    fn reads_never_conflict_with_reads() {
        let a = access(|a| a.add_component_read::<A>());
        let b = access(|a| a.add_component_read::<A>());
        assert!(a.is_compatible(&b));
        assert!(access(|a| a.add_resource_read::<A>())
            .is_compatible(&access(|a| a.add_resource_read::<A>())));
    }

    #[test]
    fn a_write_conflicts_with_a_read_or_a_write_of_the_same_type_only() {
        let w = access(|a| a.add_component_write::<A>());
        let r = access(|a| a.add_component_read::<A>());
        let other = access(|a| a.add_component_write::<B>());
        assert!(!w.is_compatible(&r));
        assert!(!r.is_compatible(&w), "symmetric");
        assert!(!w.is_compatible(&w));
        assert!(w.is_compatible(&other));
        assert!(r.is_compatible(&other));
        let c = w.get_conflicts(&r);
        assert!(!c.is_all());
        assert_eq!(c.names().len(), 1);
        assert!(c.names()[0].ends_with("::A"));
    }

    #[test]
    fn conflict_names_are_sorted_and_have_no_repeats() {
        let both = access(|a| {
            a.add_component_write::<B>();
            a.add_component_write::<A>();
            a.add_resource_write::<C>();
        });
        let c = both.get_conflicts(&both);
        let names = c.names();
        assert_eq!(
            names.len(),
            3,
            "A and B as components, C as a resource: {names:?}"
        );
        let mut sorted = names.to_vec();
        sorted.sort_unstable();
        assert_eq!(names, sorted.as_slice());
        assert!(c.to_string().contains("::B`"));
    }

    #[test]
    fn a_type_in_conflict_through_several_pairs_is_named_once() {
        // Two parts on each side all writing A: four conflicting pairs,
        // one name.
        let left = set(vec![fa::<Mut<A>, ()>(), fa::<Mut<A>, With<B>>()]);
        let right = set(vec![fa::<Mut<A>, ()>(), fa::<&A, ()>()]);
        let c = left.get_conflicts(&right);
        assert_eq!(c.names().len(), 1, "{:?}", c.names());
        let single = left.get_conflicts_single(&fa::<Mut<A>, ()>());
        assert_eq!(single.names().len(), 1, "{:?}", single.names());
    }

    #[test]
    fn conflicts_display_lists_every_name_in_order() {
        let both = access(|a| {
            a.add_component_write::<B>();
            a.add_component_write::<A>();
        });
        let c = both.get_conflicts(&both);
        assert_eq!(
            c.to_string(),
            "`mid_ecs::access::tests::A`, `mid_ecs::access::tests::B`"
        );
        // A whole-class claim and specific names together.
        let sweeping = access(|a| {
            a.add_component_write::<A>();
            a.read_all();
        });
        let mixed = sweeping.get_conflicts(&access(|a| a.add_component_write::<A>()));
        assert!(mixed.is_all());
        assert_eq!(
            mixed.to_string(),
            "whole-world access, `mid_ecs::access::tests::A`"
        );
    }

    #[test]
    fn components_and_resources_are_separate_namespaces() {
        let comp = access(|a| a.add_component_write::<A>());
        let res = access(|a| a.add_resource_write::<A>());
        assert!(comp.is_compatible(&res));
        assert!(!res.is_compatible(&access(|a| a.add_resource_read::<A>())));
        assert!(!comp.is_compatible(&access(|a| a.add_component_read::<A>())));
    }

    #[test]
    fn a_write_covers_a_read_of_the_same_type_in_either_order() {
        let mut a = Access::new();
        a.add_component_read::<A>();
        a.add_component_write::<A>();
        assert!(a.reads_component::<A>() && a.writes_component::<A>());
        let mut b = Access::new();
        b.add_component_write::<A>();
        b.add_component_read::<A>();
        assert_eq!(a, b, "same set whichever order, no separate read entry");
        let mut r = Access::new();
        r.add_resource_read::<A>();
        r.add_resource_write::<A>();
        let mut w = Access::new();
        w.add_resource_write::<A>();
        w.add_resource_read::<A>();
        assert_eq!(r, w);
        assert!(!r.writes_component::<A>() && r.writes_resource::<A>());
    }

    #[test]
    fn extend_unions_and_keeps_writes() {
        let mut base = access(|a| {
            a.add_component_read::<A>();
            a.add_resource_read::<B>();
        });
        let more = access(|a| {
            a.add_component_write::<A>();
            a.add_component_read::<C>();
            a.add_resource_write::<B>();
        });
        base.extend(&more);
        assert!(base.writes_component::<A>() && base.reads_component::<C>());
        assert!(!base.writes_component::<C>());
        assert!(base.writes_resource::<B>());
        // Extending with a read of something already written keeps the write,
        // and leaves no separate read entry behind: the result is the same
        // set as building the write directly, in every combination.
        let mut w = access(|a| a.add_component_write::<A>());
        w.extend(&access(|a| a.add_component_read::<A>()));
        assert_eq!(w, access(|a| a.add_component_write::<A>()));
        let mut wr = access(|a| a.add_resource_write::<A>());
        wr.extend(&access(|a| a.add_resource_read::<A>()));
        assert_eq!(wr, access(|a| a.add_resource_write::<A>()));
        // And a read that a later write replaces.
        let mut r = access(|a| a.add_component_read::<A>());
        r.extend(&access(|a| a.add_component_write::<A>()));
        assert_eq!(r, access(|a| a.add_component_write::<A>()));
        let mut rr = access(|a| a.add_resource_read::<A>());
        rr.extend(&access(|a| a.add_resource_write::<A>()));
        assert_eq!(rr, access(|a| a.add_resource_write::<A>()));
        // Flags carry over.
        let mut x = Access::new();
        x.extend(&access(|a| a.read_all()));
        assert!(x.reads_component::<D>() && x.reads_resource::<D>());
        assert!(!x.writes_component::<D>());
    }

    #[test]
    fn exclusive_access_conflicts_with_anything_that_touches_anything() {
        let ex = access(|a| a.write_all());
        assert!(ex.is_exclusive());
        assert!(ex.writes_component::<A>() && ex.reads_resource::<B>());
        assert!(ex.writes_resource::<B>() && ex.reads_component::<A>());
        assert!(ex.is_compatible(&Access::new()), "nothing to collide with");
        for other in [
            access(|a| a.add_component_read::<A>()),
            access(|a| a.add_component_write::<A>()),
            access(|a| a.add_resource_read::<A>()),
            access(|a| a.add_resource_write::<A>()),
            access(|a| a.read_all()),
            access(|a| a.write_all()),
        ] {
            let c = ex.get_conflicts(&other);
            assert!(c.is_all(), "{other:?}");
            assert!(!other.is_compatible(&ex));
        }
    }

    #[test]
    fn read_all_conflicts_with_writes_only() {
        let ra = access(|a| a.read_all());
        assert!(!ra.is_exclusive());
        assert!(ra.is_compatible(&ra));
        assert!(ra.is_compatible(&access(|a| a.add_component_read::<A>())));
        assert!(ra.is_compatible(&access(|a| a.add_resource_read::<A>())));
        assert!(!ra.is_compatible(&access(|a| a.add_component_write::<A>())));
        assert!(!ra.is_compatible(&access(|a| a.add_resource_write::<A>())));
        // And from the other side: the writer asking about the reader.
        assert!(!access(|a| a.add_component_write::<A>()).is_compatible(&ra));
        assert!(!access(|a| a.add_resource_write::<A>()).is_compatible(&ra));
        assert!(access(|a| a.add_component_write::<A>())
            .get_conflicts(&ra)
            .is_all());
        assert!(access(|a| a.add_component_read::<A>()).is_compatible(&ra));
        assert!(ra
            .get_conflicts(&access(|a| a.add_component_write::<A>()))
            .is_all());
    }

    #[test]
    fn is_empty_tracks_every_kind_of_access() {
        assert!(Access::new().is_empty());
        assert!(!access(|a| a.add_component_read::<A>()).is_empty());
        assert!(!access(|a| a.add_component_write::<A>()).is_empty());
        assert!(!access(|a| a.add_resource_read::<A>()).is_empty());
        assert!(!access(|a| a.add_resource_write::<A>()).is_empty());
        assert!(!access(|a| a.read_all()).is_empty());
        assert!(!access(|a| a.write_all()).is_empty());
        assert!(!Access::new().is_exclusive());
    }

    #[test]
    fn conflicts_display_and_accessors() {
        assert!(AccessConflicts::none().is_empty());
        assert_eq!(AccessConflicts::none().to_string(), "no conflict");
        let ex = access(|a| a.write_all());
        let c = ex.get_conflicts(&access(|a| a.add_component_read::<A>()));
        assert!(c.is_all());
        assert!(c.names().is_empty());
        assert!(c.to_string().contains("whole-world access"));
        let mixed = access(|a| a.write_all()).get_conflicts(&access(|a| {
            a.add_component_write::<A>();
        }));
        assert!(!mixed.is_empty());
    }

    // ── FilteredAccess ───────────────────────────────────────────────

    fn fa<Q: QueryData, F: DataFilter>() -> FilteredAccess {
        FilteredAccess::of_query::<Q, F>()
    }

    #[test]
    fn disjoint_filters_let_two_writers_coexist() {
        let with_b = fa::<Mut<A>, With<B>>();
        let without_b = fa::<Mut<A>, Without<B>>();
        assert!(with_b.is_compatible(&without_b));
        assert!(without_b.is_compatible(&with_b));
        assert!(
            !with_b.is_compatible(&fa::<Mut<A>, ()>()),
            "an unconstrained writer overlaps both"
        );
        assert!(
            !with_b.is_compatible(&fa::<&A, With<B>>()),
            "same side of the constraint"
        );
        assert!(!with_b.is_compatible(&with_b));
        // A required *term* constrains like With does.
        assert!(fa::<(Mut<A>, &B), ()>().is_compatible(&fa::<Mut<A>, Without<B>>()));
        assert!(
            !fa::<(Mut<A>, Option<&B>), ()>().is_compatible(&fa::<Mut<A>, Without<B>>()),
            "an optional term requires nothing"
        );
    }

    #[test]
    fn a_constraint_on_an_unrelated_type_does_not_separate_two_writers() {
        assert!(!fa::<Mut<A>, With<C>>().is_compatible(&fa::<Mut<A>, Without<D>>()));
        assert!(!fa::<Mut<A>, With<C>>().is_compatible(&fa::<Mut<A>, With<D>>()));
        assert!(!fa::<Mut<A>, Without<C>>().is_compatible(&fa::<Mut<A>, Without<D>>()));
    }

    #[test]
    fn an_unsatisfiable_query_conflicts_with_nothing() {
        let never = fa::<Mut<A>, (With<B>, Without<B>)>();
        assert!(never.is_unsatisfiable());
        assert!(never.is_compatible(&fa::<Mut<A>, ()>()));
        assert!(never.is_compatible(&never));
        assert!(!fa::<Mut<A>, ()>().is_unsatisfiable());
        // A required term plus Without of the same type is contradictory too.
        assert!(fa::<&A, Without<A>>().is_unsatisfiable());
    }

    #[test]
    fn a_component_one_side_excludes_cannot_be_the_shared_one() {
        // The second query reads A only on entities that have it, and
        // requires them not to.
        let optional_writer = fa::<Option<Mut<A>>, ()>();
        let excluded_reader = fa::<Option<&A>, Without<A>>();
        assert!(optional_writer.is_compatible(&excluded_reader));
        // But another component in the same pair still conflicts.
        let mut both = fa::<(Option<Mut<A>>, Option<Mut<B>>), ()>();
        both.add_component_write::<B>();
        assert!(!both.is_compatible(&fa::<Option<&B>, Without<A>>()));
    }

    #[test]
    fn or_keeps_its_reads_but_none_of_its_constraints() {
        let or = fa::<Mut<A>, Or<(With<B>, With<C>)>>();
        assert!(!or.requires::<B>() && !or.requires::<C>());
        assert!(
            !or.is_compatible(&fa::<Mut<A>, Without<B>>()),
            "an entity with a C is in both"
        );
        // An Or arm that reads ticks still reads them.
        let or_changed = fa::<&A, Or<(Changed<B>, With<C>)>>();
        assert!(or_changed.access().reads_component::<B>());
        assert!(!or_changed.is_compatible(&fa::<Mut<B>, ()>()));
        // A nested Or inside a tuple behaves the same.
        let nested = fa::<Mut<A>, (Or<(With<B>, With<C>)>, Without<D>)>();
        assert!(nested.excludes::<D>());
        assert!(!nested.requires::<B>());
    }

    #[test]
    fn row_filters_read_the_ticks_they_test() {
        let writer = fa::<Mut<B>, ()>();
        for reader in [fa::<&A, Changed<B>>(), fa::<&A, Added<B>>()] {
            assert!(reader.access().reads_component::<B>());
            assert!(reader.requires::<B>());
            assert!(!reader.is_compatible(&writer));
        }
        assert!(
            fa::<&A, With<B>>().is_compatible(&writer),
            "With<B> reads no data of B"
        );
        assert!(fa::<&A, Without<B>>().is_compatible(&writer));
    }

    #[test]
    fn a_term_and_a_filter_on_the_same_component_are_one_access() {
        let q = fa::<Mut<A>, Changed<A>>();
        assert!(q.access().writes_component::<A>());
        assert!(q.requires::<A>());
        let q2 = fa::<(Mut<A>, &B), (Changed<A>, With<C>)>();
        assert!(q2.access().writes_component::<A>() && q2.access().reads_component::<B>());
        assert!(q2.requires::<A>() && q2.requires::<B>() && q2.requires::<C>());
    }

    #[test]
    #[should_panic(expected = "FilteredAccess::of_query: conflicting access")]
    fn of_query_refuses_a_query_that_conflicts_with_itself() {
        let _ = fa::<(&A, Mut<A>), ()>();
    }

    #[test]
    fn terms_describe_their_access_and_requirements() {
        let q = fa::<(Entity, Has<A>, Option<&B>, Option<Mut<C>>), ()>();
        assert!(q.access().reads_component::<B>() && q.access().writes_component::<C>());
        assert!(!q.access().reads_component::<A>(), "Has reads no data");
        assert!(!q.requires::<A>() && !q.requires::<B>() && !q.requires::<C>());
        let q = fa::<(&A, Mut<B>, Ref<C>, Added<D>), ()>();
        assert!(q.requires::<A>() && q.requires::<B>() && q.requires::<C>() && q.requires::<D>());
        assert!(q.access().reads_component::<A>() && q.access().writes_component::<B>());
        assert!(q.access().reads_component::<C>() && q.access().reads_component::<D>());
        assert!(!q.access().writes_component::<A>());
        let q = fa::<Option<Ref<A>>, Without<B>>();
        assert!(q.access().reads_component::<A>() && !q.requires::<A>() && q.excludes::<B>());
    }

    #[test]
    fn resources_conflict_whatever_the_entity_filters_say() {
        let mut a = fa::<Mut<A>, With<B>>();
        a.add_resource_write::<C>();
        let mut b = fa::<Mut<A>, Without<B>>();
        b.add_resource_read::<C>();
        let c = a.get_conflicts(&b);
        assert_eq!(
            c.names().len(),
            1,
            "the component part is disjoint, the resource is not"
        );
        assert!(c.names()[0].ends_with("::C"));
        let mut d = fa::<Mut<A>, Without<B>>();
        d.add_resource_read::<D>();
        assert!(a.is_compatible(&d));
    }

    #[test]
    fn filtered_access_mutators_record_what_they_say() {
        let mut f = FilteredAccess::new();
        f.add_component_read::<A>();
        f.add_component_write::<B>();
        f.add_resource_read::<C>();
        f.add_resource_write::<D>();
        f.add_with::<A>();
        f.add_without::<D>();
        assert!(f.access().reads_component::<A>() && f.access().writes_component::<B>());
        assert!(f.access().reads_resource::<C>() && f.access().writes_resource::<D>());
        assert!(f.requires::<A>() && f.excludes::<D>());
        assert!(!f.requires::<D>() && !f.excludes::<A>());
        let mut ex = FilteredAccess::new();
        ex.write_all();
        assert!(ex.access().is_exclusive());
        let mut ra = FilteredAccess::new();
        ra.read_all();
        assert!(ra.access().reads_component::<A>() && !ra.access().writes_component::<A>());
    }

    // ── FilteredAccessSet ────────────────────────────────────────────

    fn set(parts: Vec<FilteredAccess>) -> FilteredAccessSet {
        let mut s = FilteredAccessSet::new();
        for p in parts {
            s.add(p);
        }
        s
    }

    #[test]
    fn a_set_keeps_its_parts_and_their_union() {
        let s = set(vec![fa::<Mut<A>, With<B>>(), fa::<&C, ()>()]);
        assert_eq!(s.parts().len(), 2);
        assert!(s.combined_access().writes_component::<A>());
        assert!(s.combined_access().reads_component::<C>());
        assert!(!s.is_exclusive());
        let mut ex = FilteredAccess::new();
        ex.write_all();
        assert!(set(vec![ex]).is_exclusive());
        assert!(FilteredAccessSet::new().combined_access().is_empty());
    }

    #[test]
    fn a_new_part_is_checked_against_the_parts_already_in_the_system() {
        let s = set(vec![fa::<Mut<A>, With<B>>(), fa::<&C, ()>()]);
        assert!(s
            .get_conflicts_single(&fa::<Mut<A>, Without<B>>())
            .is_empty());
        let c = s.get_conflicts_single(&fa::<&A, ()>());
        assert_eq!(c.names().len(), 1);
        assert!(!s.get_conflicts_single(&fa::<Mut<C>, ()>()).is_empty());
        // Two readers are fine.
        assert!(s.get_conflicts_single(&fa::<&C, ()>()).is_empty());
    }

    #[test]
    fn systems_are_compared_part_against_part_not_by_their_unions() {
        let left = set(vec![fa::<Mut<A>, With<B>>(), fa::<&C, ()>()]);
        let right = set(vec![fa::<Mut<A>, Without<B>>(), fa::<&C, ()>()]);
        assert!(
            !left
                .combined_access()
                .is_compatible(right.combined_access()),
            "the unions both write A"
        );
        assert!(
            left.is_compatible(&right),
            "but no entity is in both writers"
        );
        // One more part on the right that touches what the left writes.
        let right2 = set(vec![fa::<Mut<A>, Without<B>>(), fa::<&A, With<B>>()]);
        let c = left.get_conflicts(&right2);
        assert!(!c.is_empty());
        assert_eq!(c.names().len(), 1);
    }

    #[test]
    fn an_empty_or_read_only_set_never_conflicts_and_exclusive_always_does() {
        let reader = set(vec![fa::<&A, ()>(), fa::<&B, ()>()]);
        assert!(reader.is_compatible(&reader));
        assert!(reader.is_compatible(&FilteredAccessSet::new()));
        let mut ex = FilteredAccess::new();
        ex.write_all();
        let exclusive = set(vec![ex]);
        assert!(!exclusive.is_compatible(&reader));
        assert!(exclusive.get_conflicts(&reader).is_all());
        assert!(exclusive.is_compatible(&FilteredAccessSet::new()));
    }

    // ── The exhaustive oracle ────────────────────────────────────────

    /// Components A, B, C are the universe: ids 0, 1, 2; an archetype is
    /// a subset of them, 0..8.
    fn universe(type_id: TypeId) -> Option<ComponentId> {
        if type_id == TypeId::of::<A>() {
            Some(ComponentId::from_u32(0))
        } else if type_id == TypeId::of::<B>() {
            Some(ComponentId::from_u32(1))
        } else if type_id == TypeId::of::<C>() {
            Some(ComponentId::from_u32(2))
        } else {
            None
        }
    }

    /// What a query does, computed from the *real* archetype-level
    /// semantics of its terms and filter, with no reference to
    /// `FilteredAccess`: for each of the 8 archetypes, whether the query
    /// matches it, and for each component how the query touches it there
    /// (0 none, 1 read, 2 write).
    struct Oracle {
        matches: [bool; 8],
        level: [u8; 3],
    }

    fn oracle<Q: QueryData, F: DataFilter>() -> Oracle {
        let qstate = Q::get_state(&universe);
        let fstate = F::get_state(&universe);
        let mut required = Vec::new();
        Q::collect_required(&qstate, &mut required);
        let mut entries = Vec::new();
        Q::collect_access(&qstate, &mut entries);
        let mut ticks = Vec::new();
        F::collect_tick_reads(&fstate, &mut ticks);
        let mut level = [0u8; 3];
        for e in &entries {
            if let Some(id) = e.id {
                let slot = &mut level[id.as_u32() as usize];
                *slot = (*slot).max(if e.write { 2 } else { 1 });
            }
        }
        for id in ticks {
            let slot = &mut level[id.as_u32() as usize];
            *slot = (*slot).max(1);
        }
        let mut matches = [false; 8];
        for (subset, m) in matches.iter_mut().enumerate() {
            let has = |id: ComponentId| subset & (1 << id.as_u32()) != 0;
            *m = Q::is_satisfiable(&qstate)
                && required.iter().all(|&id| has(id))
                && F::matches_component_set(&fstate, &has);
        }
        Oracle { matches, level }
    }

    /// Whether some entity could be reached by both while one writes a
    /// component of it the other touches.
    fn oracle_conflict(a: &Oracle, b: &Oracle) -> bool {
        (0..8usize).any(|subset| {
            a.matches[subset]
                && b.matches[subset]
                && (0..3usize).any(|c| {
                    subset & (1 << c) != 0
                        && ((a.level[c] == 2 && b.level[c] >= 1)
                            || (b.level[c] == 2 && a.level[c] >= 1))
                })
        })
    }

    struct Spec {
        name: &'static str,
        access: FilteredAccess,
        oracle: Oracle,
        /// No `Or` anywhere: the verdict must match the oracle exactly,
        /// not merely cover it.
        exact: bool,
    }

    fn spec<Q: QueryData, F: DataFilter>(name: &'static str, exact: bool) -> Spec {
        Spec {
            name,
            access: FilteredAccess::of_query::<Q, F>(),
            oracle: oracle::<Q, F>(),
            exact,
        }
    }

    fn specs() -> Vec<Spec> {
        vec![
            spec::<&A, ()>("&A", true),
            spec::<Mut<A>, ()>("Mut<A>", true),
            spec::<Mut<A>, With<B>>("Mut<A> With<B>", true),
            spec::<Mut<A>, Without<B>>("Mut<A> Without<B>", true),
            spec::<Option<Mut<A>>, ()>("Option<Mut<A>>", true),
            spec::<Option<&A>, Without<A>>("Option<&A> Without<A>", true),
            spec::<(Mut<A>, &B), With<C>>("(Mut<A>, &B) With<C>", true),
            spec::<(&A, Option<Mut<B>>), Without<C>>("(&A, Option<Mut<B>>) Without<C>", true),
            spec::<Mut<B>, (With<A>, Without<C>)>("Mut<B> (With<A>, Without<C>)", true),
            spec::<Mut<A>, Or<(With<B>, With<C>)>>("Mut<A> Or<With<B>, With<C>>", false),
            spec::<Mut<A>, Or<(Without<B>, Without<C>)>>(
                "Mut<A> Or<Without<B>, Without<C>>",
                false,
            ),
            spec::<&A, Changed<B>>("&A Changed<B>", true),
            spec::<Mut<B>, Added<A>>("Mut<B> Added<A>", true),
            spec::<(Mut<A>, Option<&C>), Changed<A>>("(Mut<A>, Option<&C>) Changed<A>", true),
            spec::<Ref<A>, Without<B>>("Ref<A> Without<B>", true),
            spec::<Option<Ref<B>>, With<A>>("Option<Ref<B>> With<A>", true),
            spec::<(Entity, Has<A>), ()>("(Entity, Has<A>)", true),
            spec::<Mut<C>, (Without<A>, Without<B>)>("Mut<C> (Without<A>, Without<B>)", true),
            spec::<Added<A>, ()>("Added<A> term", true),
            spec::<Mut<C>, Or<(Changed<A>, With<B>)>>("Mut<C> Or<Changed<A>, With<B>>", false),
            spec::<Mut<A>, (With<A>, Without<A>)>("Mut<A> unsatisfiable", true),
            spec::<(Option<Mut<A>>, Option<Mut<B>>), Without<C>>(
                "(Option<Mut<A>>, Option<Mut<B>>) Without<C>",
                true,
            ),
            spec::<&C, (Or<(With<A>, Without<B>)>, Without<C>)>("&C (Or, Without<C>)", false),
        ]
    }

    #[test]
    fn filtered_conflicts_match_a_brute_force_search_over_every_archetype() {
        let all = specs();
        let mut checked = 0;
        let (mut conflicts, mut compatible, mut or_only_extra) = (0, 0, 0);
        for a in &all {
            for b in &all {
                let want = oracle_conflict(&a.oracle, &b.oracle);
                let got = !a.access.is_compatible(&b.access);
                assert!(
                    !want || got,
                    "unsound: `{}` vs `{}` can reach one entity but were called compatible",
                    a.name,
                    b.name
                );
                if a.exact && b.exact {
                    assert_eq!(got, want, "`{}` vs `{}`", a.name, b.name);
                } else if got && !want {
                    or_only_extra += 1;
                }
                // Symmetry.
                assert_eq!(
                    got,
                    !b.access.is_compatible(&a.access),
                    "`{}` vs `{}`",
                    a.name,
                    b.name
                );
                checked += 1;
                if got {
                    conflicts += 1;
                } else {
                    compatible += 1;
                }
            }
        }
        // The table must exercise both verdicts, and the conservative `Or`
        // case must actually occur, or the test proves nothing.
        assert_eq!(checked, all.len() * all.len());
        assert!(
            conflicts > 100 && compatible > 100,
            "{conflicts} conflicts, {compatible} compatible"
        );
        assert!(
            or_only_extra > 0,
            "no `Or` pair was ever more conservative than the oracle"
        );
    }

    #[test]
    fn update_access_agrees_with_collect_access_and_collect_required() {
        fn check<Q: QueryData>() {
            let state = Q::get_state(&universe);
            let mut entries = Vec::new();
            Q::collect_access(&state, &mut entries);
            let mut fa = FilteredAccess::new();
            Q::update_access(&mut fa);
            let from_entries: BTreeSet<(TypeId, bool)> =
                entries.iter().map(|e| (e.type_id, e.write)).collect();
            let from_access: BTreeSet<(TypeId, bool)> = fa
                .access
                .component_writes
                .keys()
                .map(|&k| (k, true))
                .chain(fa.access.component_reads.keys().map(|&k| (k, false)))
                .collect();
            assert_eq!(from_entries, from_access, "{}", std::any::type_name::<Q>());
            let mut required = Vec::new();
            Q::collect_required(&state, &mut required);
            let required: BTreeSet<TypeId> = required
                .into_iter()
                .map(|id| {
                    [TypeId::of::<A>(), TypeId::of::<B>(), TypeId::of::<C>()][id.as_u32() as usize]
                })
                .collect();
            let with: BTreeSet<TypeId> = fa.with.keys().copied().collect();
            assert_eq!(required, with, "{}", std::any::type_name::<Q>());
        }
        check::<Entity>();
        check::<&A>();
        check::<Option<&A>>();
        check::<Has<A>>();
        check::<Mut<A>>();
        check::<Option<Mut<A>>>();
        check::<Ref<A>>();
        check::<Option<Ref<A>>>();
        check::<Added<A>>();
        check::<Changed<A>>();
        check::<(&A, Mut<B>, Option<&C>)>();
        check::<(Entity, (Ref<A>, Option<Mut<B>>), Changed<C>)>();
    }
}
