// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-ecs-hierarchy.md
// ============================================================================
//! Parent/child hierarchy: [`ChildOf`] and [`Children`].
//!
//! An entity's parent is its [`ChildOf`]; a parent's [`Children`] lists its
//! children in the order they were added. The two are kept in step by
//! component hooks, so inserting or removing a `ChildOf` is all it takes: the
//! parent's `Children` follows, and an empty one is removed. Both components
//! live in the Archetype Core, so they can be queried like any other.
//!
//! The hooks are not registered until [`World::enable_hierarchy`] is called
//! (a world without hooks pays nothing on structural changes). The
//! hierarchy methods on `World` call it themselves; `mid_app::HierarchyPlugin`
//! calls it for an `App`.
//!
//! ```
//! use mid_ecs::World;
//!
//! let mut world = World::new();
//! let root = world.spawn();
//! let a = world.spawn_child(root).unwrap();
//! let b = world.spawn_child(root).unwrap();
//! assert_eq!(world.children(root), &[a, b]);
//!
//! // Despawning a parent despawns everything below it.
//! world.despawn(root);
//! assert!(!world.is_alive(a) && !world.is_alive(b));
//! ```

use std::fmt;
use std::ops::Deref;

use crate::hooks::{DeferredWorld, HookContext};
use crate::world::{Entity, World};

/// Hierarchies deeper than this are treated as cycles when walking up.
/// A real hierarchy is nowhere near it; the bound only guarantees that a
/// corrupted one cannot make a walk spin forever.
const MAX_DEPTH: usize = 1 << 20;

/// The parent of an entity.
///
/// Insert it with [`World::insert_static`] or the methods on `World` and
/// `Commands`; it is Archetype Core only (see the module docs). An entity
/// has at most one. A `ChildOf` that cannot be honoured (the parent is dead,
/// is the entity itself, or is one of its descendants) is removed again as
/// soon as the operation that inserted it finishes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ChildOf(pub Entity);

impl ChildOf {
    /// The parent entity.
    #[inline]
    pub fn parent(&self) -> Entity {
        self.0
    }
}

/// The children of an entity, in the order they were added.
///
/// Maintained entirely by the hierarchy hooks: there is no public
/// constructor, so it cannot be inserted by hand. An entity with no
/// children has no `Children` at all.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Children(Vec<Entity>);

impl Children {
    fn single(child: Entity) -> Self {
        Self(vec![child])
    }

    /// The children as a slice, in order.
    #[inline]
    pub fn as_slice(&self) -> &[Entity] {
        &self.0
    }
}

impl Deref for Children {
    type Target = [Entity];

    #[inline]
    fn deref(&self) -> &[Entity] {
        &self.0
    }
}

impl<'a> IntoIterator for &'a Children {
    type Item = &'a Entity;
    type IntoIter = std::slice::Iter<'a, Entity>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

/// Why a hierarchy change was refused. Nothing changed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HierarchyError {
    /// The entity is not alive.
    NotAlive(Entity),
    /// An entity cannot be its own parent.
    SelfParent(Entity),
    /// The new parent is the child or one of its descendants, so the link
    /// would make the child its own ancestor.
    Cycle {
        /// The entity being given a parent.
        child: Entity,
        /// The parent that would have closed the loop.
        parent: Entity,
    },
}

impl fmt::Display for HierarchyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HierarchyError::NotAlive(e) => write!(f, "{e:?} is not alive"),
            HierarchyError::SelfParent(e) => write!(f, "{e:?} cannot be its own parent"),
            HierarchyError::Cycle { child, parent } => {
                write!(f, "making {parent:?} the parent of {child:?} would create a cycle")
            }
        }
    }
}

impl std::error::Error for HierarchyError {}

// ── hooks ───────────────────────────────────────────────────────────────────

// Whether walking up from `at` reaches `target` (including `at == target`).
fn reaches(mut at: Entity, target: Entity, parent_of: impl Fn(Entity) -> Option<Entity>) -> bool {
    for _ in 0..MAX_DEPTH {
        if at == target {
            return true;
        }
        match parent_of(at) {
            Some(parent) => at = parent,
            None => return false,
        }
    }
    true
}

// Appends `child` to `parent`'s `Children`, creating it if there is none.
// Runs as a hook-queued command, so it re-checks that the link it was queued
// for still stands: a later change in the same operation may have undone it.
fn link_child(world: &mut World, parent: Entity, child: Entity) {
    if !world.is_alive(parent)
        || !world.is_alive(child)
        || world.get_static::<ChildOf>(child) != Some(&ChildOf(parent))
    {
        return;
    }
    match world.get_static_mut::<Children>(parent) {
        Some(children) => {
            debug_assert!(!children.0.contains(&child), "child linked twice");
            children.0.push(child);
        }
        None => {
            world.insert_static(parent, Children::single(child));
        }
    }
}

fn child_of_on_insert(world: &mut DeferredWorld<'_>, ctx: HookContext) {
    let child = ctx.entity;
    let Some(&ChildOf(parent)) = world.get_static::<ChildOf>(child) else {
        // Inserted through the Sparse Shell, which the hierarchy does not
        // read. Take it out again rather than leave a link nothing follows.
        if world.get::<ChildOf>(child).is_some() {
            world.commands().queue(move |w| {
                w.remove::<ChildOf>(child);
            });
        }
        return;
    };
    // `reaches` counts `parent == child` as a loop, so a self parent is
    // refused here too.
    let valid = world.is_alive(parent)
        && !reaches(parent, child, |e| world.get_static::<ChildOf>(e).map(|c| c.0));
    if valid {
        world.commands().queue(move |w| link_child(w, parent, child));
    } else {
        world.commands().queue(move |w| {
            if w.get_static::<ChildOf>(child) == Some(&ChildOf(parent)) {
                w.remove_static::<ChildOf>(child);
            }
        });
    }
}

fn child_of_on_discard(world: &mut DeferredWorld<'_>, ctx: HookContext) {
    let child = ctx.entity;
    let Some(&ChildOf(parent)) = world.get_static::<ChildOf>(child) else { return };
    let Some(children) = world.get_static_mut::<Children>(parent) else { return };
    // Searched from the back: removing the newest child first (the order a
    // despawn walks them in) is then constant time.
    if let Some(index) = children.0.iter().rposition(|&e| e == child) {
        children.0.remove(index);
    }
    if children.0.is_empty() {
        world.commands().queue(move |w| {
            if w.get_static::<Children>(parent).is_some_and(|c| c.0.is_empty()) {
                w.remove_static::<Children>(parent);
            }
        });
    }
}

fn children_on_discard(world: &mut DeferredWorld<'_>, ctx: HookContext) {
    let parent = ctx.entity;
    let Some(children) = world.get_static::<Children>(parent) else { return };
    if children.0.is_empty() {
        return;
    }
    // Losing `Children` while still alive means the children are detached.
    // When the parent is being despawned they are despawned instead (see
    // `children_on_despawn`), and the parent is gone by the time this runs.
    let kids = children.0.clone();
    world.commands().queue(move |w| {
        if !w.is_alive(parent) {
            return;
        }
        for kid in kids {
            if w.get_static::<ChildOf>(kid) == Some(&ChildOf(parent)) {
                w.remove_static::<ChildOf>(kid);
            }
        }
    });
}

fn children_on_despawn(world: &mut DeferredWorld<'_>, ctx: HookContext) {
    let parent = ctx.entity;
    let Some(children) = world.get_static::<Children>(parent) else { return };
    if children.0.is_empty() {
        return;
    }
    let kids = children.0.clone();
    // One command for the whole subtree, not one per level: levels would each
    // cost a round of the hook-command loop, which is bounded.
    world.commands().queue(move |w| {
        despawn_subtrees(w, &kids, Some(parent));
    });
}

// Despawns `roots` and everything below them, deepest and last first. When
// `expected_parent` is given, a root that no longer points at it has been
// moved by something else in the meantime and is left alone. Returns how many
// entities it despawned.
fn despawn_subtrees(world: &mut World, roots: &[Entity], expected_parent: Option<Entity>) -> usize {
    let mut order = Vec::new();
    let mut stack = Vec::new();
    for &root in roots {
        if let Some(parent) = expected_parent {
            if world.get_static::<ChildOf>(root) != Some(&ChildOf(parent)) {
                continue;
            }
        }
        stack.push(root);
        while let Some(entity) = stack.pop() {
            order.push(entity);
            stack.extend(world.children(entity).iter().rev().copied());
        }
    }
    let mut despawned = 0;
    // Reverse pre-order: every child goes before its parent, and the last
    // sibling before the first, which keeps each `Children` removal at the
    // end of its list.
    for &entity in order.iter().rev() {
        if world.despawn(entity) {
            despawned += 1;
        }
    }
    despawned
}

// ── World API ───────────────────────────────────────────────────────────────

impl World {
    /// Registers the hooks that keep [`ChildOf`] and [`Children`] in step.
    /// Idempotent. Takes effect for changes made after the call: a `ChildOf`
    /// inserted before it has no matching `Children`.
    ///
    /// The methods below call this themselves, so it only needs calling
    /// directly before inserting `ChildOf` by hand with
    /// [`World::insert_static`] (or in a bundle).
    ///
    /// Once enabled, every structural change in this world checks the hook
    /// table, which a world without any hooks does not. That is the cost of a
    /// hierarchy.
    ///
    /// # Panics
    /// If a hook of the same kind was already registered for `ChildOf` or
    /// `Children` (one per kind: react from an observer instead).
    pub fn enable_hierarchy(&mut self) {
        if self.hierarchy_enabled {
            return;
        }
        self.hierarchy_enabled = true;
        self.register_component_hooks::<ChildOf>()
            .on_insert(child_of_on_insert)
            .on_discard(child_of_on_discard);
        self.register_component_hooks::<Children>()
            .on_discard(children_on_discard)
            .on_despawn(children_on_despawn);
    }

    /// The parent of `entity`, if it has one.
    #[inline]
    pub fn parent(&self, entity: Entity) -> Option<Entity> {
        self.get_static::<ChildOf>(entity).map(|c| c.0)
    }

    /// The children of `entity`, in the order they were added. Empty if it
    /// has none (or is not alive).
    #[inline]
    pub fn children(&self, entity: Entity) -> &[Entity] {
        self.get_static::<Children>(entity).map_or(&[], |c| c.as_slice())
    }

    /// The topmost ancestor of `entity`, or `entity` itself if it has no
    /// parent.
    pub fn root_of(&self, entity: Entity) -> Entity {
        self.ancestors(entity).last().unwrap_or(entity)
    }

    /// Whether `ancestor` is somewhere above `entity`. An entity is not its
    /// own ancestor.
    pub fn is_ancestor_of(&self, ancestor: Entity, entity: Entity) -> bool {
        self.ancestors(entity).any(|a| a == ancestor)
    }

    /// The parent, grandparent and so on of `entity`, nearest first.
    pub fn ancestors(&self, entity: Entity) -> Ancestors<'_> {
        Ancestors {
            world: self,
            next: self.parent(entity),
            left: MAX_DEPTH,
        }
    }

    /// Everything below `entity`, depth first, each entity's children in
    /// order. Does not include `entity` itself.
    pub fn descendants(&self, entity: Entity) -> Descendants<'_> {
        Descendants {
            world: self,
            stack: self.children(entity).iter().rev().copied().collect(),
        }
    }

    // Everything `set_parent` checks, with no changes made.
    fn check_parent(&self, child: Entity, parent: Entity) -> Result<(), HierarchyError> {
        if !self.is_alive(child) {
            return Err(HierarchyError::NotAlive(child));
        }
        if !self.is_alive(parent) {
            return Err(HierarchyError::NotAlive(parent));
        }
        if child == parent {
            return Err(HierarchyError::SelfParent(child));
        }
        if reaches(parent, child, |e| self.parent(e)) {
            return Err(HierarchyError::Cycle { child, parent });
        }
        Ok(())
    }

    /// Makes `parent` the parent of `child`, last in its `Children`. If
    /// `child` already has a different parent it is moved; if it already has
    /// this one nothing changes (it keeps its place). Enables the hierarchy.
    ///
    /// # Errors
    /// [`HierarchyError`] if either entity is dead, they are the same, or
    /// `parent` is a descendant of `child`. Nothing changes then.
    pub fn set_parent(&mut self, child: Entity, parent: Entity) -> Result<(), HierarchyError> {
        self.enable_hierarchy();
        self.check_parent(child, parent)?;
        if self.parent(child) == Some(parent) {
            return Ok(());
        }
        // The Archetype Core never overwrites, so a move is a removal (which
        // takes `child` out of the old parent's list) and an insert.
        self.remove_static::<ChildOf>(child);
        self.insert_static(child, ChildOf(parent));
        Ok(())
    }

    /// [`Self::set_parent`] with the arguments the other way round.
    pub fn add_child(&mut self, parent: Entity, child: Entity) -> Result<(), HierarchyError> {
        self.set_parent(child, parent)
    }

    /// Makes `parent` the parent of each of `children`, in order. All are
    /// checked first: if any is refused nothing changes.
    pub fn add_children(
        &mut self,
        parent: Entity,
        children: &[Entity],
    ) -> Result<(), HierarchyError> {
        self.enable_hierarchy();
        // Adding siblings under one parent cannot create a cycle between
        // them, so checking each against the current hierarchy is enough.
        for &child in children {
            self.check_parent(child, parent)?;
        }
        for &child in children {
            self.set_parent(child, parent)?;
        }
        Ok(())
    }

    /// Spawns a new entity as the last child of `parent`.
    pub fn spawn_child(&mut self, parent: Entity) -> Result<Entity, HierarchyError> {
        if !self.is_alive(parent) {
            return Err(HierarchyError::NotAlive(parent));
        }
        let child = self.spawn();
        self.set_parent(child, parent)?;
        Ok(child)
    }

    /// Detaches `child` from its parent, leaving it alive. `false` if it had
    /// none.
    pub fn remove_parent(&mut self, child: Entity) -> bool {
        self.remove_static::<ChildOf>(child).is_some()
    }

    /// Detaches those of `children` that are children of `parent`, leaving
    /// them alive. Returns how many it detached.
    pub fn remove_children(&mut self, parent: Entity, children: &[Entity]) -> usize {
        let mut detached = 0;
        for &child in children {
            if self.parent(child) == Some(parent) && self.remove_parent(child) {
                detached += 1;
            }
        }
        detached
    }

    /// Detaches every child of `parent`, leaving them alive. Returns how
    /// many there were.
    pub fn clear_children(&mut self, parent: Entity) -> usize {
        let count = self.children(parent).len();
        if count > 0 {
            // Losing `Children` detaches what it listed (`children_on_discard`).
            self.remove_static::<Children>(parent);
        }
        count
    }

    /// Despawns every descendant of `parent`, keeping `parent`. Returns how
    /// many entities it despawned.
    pub fn despawn_children(&mut self, parent: Entity) -> usize {
        let kids = self.children(parent).to_vec();
        if kids.is_empty() {
            return 0;
        }
        despawn_subtrees(self, &kids, Some(parent))
    }
}

/// The ancestors of an entity, nearest first. See [`World::ancestors`].
pub struct Ancestors<'w> {
    world: &'w World,
    next: Option<Entity>,
    left: usize,
}

impl Iterator for Ancestors<'_> {
    type Item = Entity;

    fn next(&mut self) -> Option<Entity> {
        let current = self.next?;
        if self.left == 0 {
            return None;
        }
        self.left -= 1;
        self.next = self.world.parent(current);
        Some(current)
    }
}

/// The descendants of an entity, depth first. See [`World::descendants`].
pub struct Descendants<'w> {
    world: &'w World,
    stack: Vec<Entity>,
}

impl Iterator for Descendants<'_> {
    type Item = Entity;

    fn next(&mut self) -> Option<Entity> {
        let entity = self.stack.pop()?;
        self.stack.extend(self.world.children(entity).iter().rev().copied());
        Some(entity)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commands::CommandQueue;
    use crate::observer::{Add, Despawn, On, Remove};
    use crate::system_param::ResMut;
    use std::collections::{HashMap, HashSet};

    #[derive(Debug, PartialEq)]
    struct Tag(u32);

    fn world() -> World {
        let mut w = World::new();
        w.enable_hierarchy();
        w
    }

    // Every structural claim the hierarchy makes, checked from the outside:
    // `ChildOf` and `Children` agree, order is kept, an empty `Children`
    // does not exist, nothing points at a dead entity, and there is no loop.
    fn assert_consistent(w: &World, entities: &[Entity]) {
        for &e in entities {
            if !w.is_alive(e) {
                assert!(w.get_static::<ChildOf>(e).is_none(), "{e:?} dead but has ChildOf");
                assert!(w.get_static::<Children>(e).is_none(), "{e:?} dead but has Children");
                continue;
            }
            let kids = w.children(e);
            assert_eq!(
                w.get_static::<Children>(e).is_some(),
                !kids.is_empty(),
                "{e:?}: Children exists exactly when it is non-empty"
            );
            let mut seen = HashSet::new();
            for &k in kids {
                assert!(w.is_alive(k), "{e:?} lists dead child {k:?}");
                assert!(seen.insert(k), "{e:?} lists {k:?} twice");
                assert_eq!(w.parent(k), Some(e), "{k:?} listed under {e:?} but points elsewhere");
            }
            if let Some(p) = w.parent(e) {
                assert!(w.is_alive(p), "{e:?} points at dead parent {p:?}");
                assert!(w.children(p).contains(&e), "{e:?} not listed under its parent {p:?}");
            }
            assert!(w.ancestors(e).count() < 100_000, "{e:?}: ancestors never end");
            assert!(w.ancestors(e).all(|a| a != e), "{e:?} is its own ancestor");
        }
    }

    // ── building a hierarchy ─────────────────────────────────────────────

    #[test]
    fn spawn_child_links_both_ways_and_keeps_order() {
        let mut w = world();
        let root = w.spawn();
        let a = w.spawn_child(root).unwrap();
        let b = w.spawn_child(root).unwrap();
        let c = w.spawn_child(root).unwrap();
        assert_eq!(w.children(root), &[a, b, c]);
        assert_eq!((w.parent(a), w.parent(b), w.parent(c)), (Some(root), Some(root), Some(root)));
        assert_eq!(w.parent(root), None);
        assert!(w.children(a).is_empty());
        assert_consistent(&w, &[root, a, b, c]);
    }

    #[test]
    fn the_world_methods_enable_the_hierarchy_themselves() {
        let mut w = World::new();
        let p = w.spawn();
        let c = w.spawn();
        w.set_parent(c, p).unwrap();
        assert_eq!(w.children(p), &[c]);
    }

    #[test]
    fn enabling_twice_is_harmless() {
        let mut w = world();
        w.enable_hierarchy();
        w.enable_hierarchy();
        let (p, c) = (w.spawn(), w.spawn());
        w.set_parent(c, p).unwrap();
        assert_eq!(w.children(p), &[c]);
    }

    #[test]
    fn a_raw_insert_is_followed_once_the_hierarchy_is_enabled() {
        let mut w = world();
        let (p, c) = (w.spawn(), w.spawn());
        assert!(w.insert_static(c, ChildOf(p)));
        assert_eq!(w.children(p), &[c]);
        assert_consistent(&w, &[p, c]);
    }

    #[test]
    fn a_raw_insert_before_enabling_is_not_followed() {
        let mut w = World::new();
        let (p, c) = (w.spawn(), w.spawn());
        w.insert_static(c, ChildOf(p));
        assert_eq!(w.parent(c), Some(p));
        assert!(w.children(p).is_empty(), "no hooks were registered yet");
    }

    #[test]
    fn a_bundle_spawn_with_a_parent_is_linked() {
        let mut w = world();
        let p = w.spawn();
        let c = w.spawn_bundle((ChildOf(p), Tag(9)));
        assert_eq!(w.children(p), &[c]);
        assert_eq!(w.get_static::<Tag>(c), Some(&Tag(9)));
    }

    #[test]
    fn add_children_appends_in_order_and_is_all_or_nothing() {
        let mut w = world();
        let p = w.spawn();
        let first = w.spawn_child(p).unwrap();
        let (x, y) = (w.spawn(), w.spawn());
        w.add_children(p, &[x, y]).unwrap();
        assert_eq!(w.children(p), &[first, x, y]);

        let dead = w.spawn();
        w.despawn(dead);
        let z = w.spawn();
        let err = w.add_children(p, &[z, dead]).unwrap_err();
        assert_eq!(err, HierarchyError::NotAlive(dead));
        assert!(w.parent(z).is_none(), "nothing was applied");
        assert_eq!(w.children(p), &[first, x, y]);
    }

    // ── moving and removing ──────────────────────────────────────────────

    #[test]
    fn reparenting_moves_the_child_and_drops_an_emptied_children() {
        let mut w = world();
        let (p1, p2) = (w.spawn(), w.spawn());
        let c = w.spawn_child(p1).unwrap();
        w.set_parent(c, p2).unwrap();
        assert_eq!(w.parent(c), Some(p2));
        assert_eq!(w.children(p2), &[c]);
        assert!(w.children(p1).is_empty());
        assert!(w.get_static::<Children>(p1).is_none(), "an empty Children is removed");
        assert_consistent(&w, &[p1, p2, c]);
    }

    #[test]
    fn setting_the_same_parent_again_keeps_the_childs_place() {
        let mut w = world();
        let p = w.spawn();
        let a = w.spawn_child(p).unwrap();
        let b = w.spawn_child(p).unwrap();
        w.set_parent(a, p).unwrap();
        assert_eq!(w.children(p), &[a, b], "a stays first");
    }

    #[test]
    fn moving_a_child_puts_it_last_under_the_new_parent() {
        let mut w = world();
        let (p1, p2) = (w.spawn(), w.spawn());
        let existing = w.spawn_child(p2).unwrap();
        let c = w.spawn_child(p1).unwrap();
        w.set_parent(c, p2).unwrap();
        assert_eq!(w.children(p2), &[existing, c]);
    }

    #[test]
    fn remove_parent_detaches_and_leaves_the_child_alive() {
        let mut w = world();
        let p = w.spawn();
        let (a, b) = (w.spawn_child(p).unwrap(), w.spawn_child(p).unwrap());
        assert!(w.remove_parent(a));
        assert!(!w.remove_parent(a), "already detached");
        assert!(w.is_alive(a) && w.parent(a).is_none());
        assert_eq!(w.children(p), &[b]);
        assert!(w.remove_parent(b));
        assert!(w.get_static::<Children>(p).is_none());
    }

    #[test]
    fn removing_a_middle_child_keeps_the_order_of_the_rest() {
        let mut w = world();
        let p = w.spawn();
        let kids: Vec<Entity> = (0..5).map(|_| w.spawn_child(p).unwrap()).collect();
        w.remove_parent(kids[2]);
        assert_eq!(w.children(p), &[kids[0], kids[1], kids[3], kids[4]]);
    }

    #[test]
    fn remove_children_detaches_only_the_named_children_of_that_parent() {
        let mut w = world();
        let (p, other) = (w.spawn(), w.spawn());
        let (a, b, c) = (w.spawn_child(p).unwrap(), w.spawn_child(p).unwrap(), w.spawn_child(p).unwrap());
        let stranger = w.spawn_child(other).unwrap();
        assert_eq!(w.remove_children(p, &[a, c, stranger]), 2, "stranger is not p's child");
        assert_eq!(w.children(p), &[b]);
        assert_eq!(w.parent(stranger), Some(other));
    }

    #[test]
    fn clear_children_detaches_everything_and_keeps_them_alive() {
        let mut w = world();
        let p = w.spawn();
        let kids: Vec<Entity> = (0..4).map(|_| w.spawn_child(p).unwrap()).collect();
        assert_eq!(w.clear_children(p), 4);
        assert!(w.get_static::<Children>(p).is_none());
        for &k in &kids {
            assert!(w.is_alive(k) && w.parent(k).is_none());
        }
        assert_eq!(w.clear_children(p), 0);
    }

    #[test]
    fn removing_the_children_component_by_hand_detaches_the_children() {
        let mut w = world();
        let p = w.spawn();
        let (a, b) = (w.spawn_child(p).unwrap(), w.spawn_child(p).unwrap());
        w.remove_static::<Children>(p);
        assert!(w.parent(a).is_none() && w.parent(b).is_none());
        assert!(w.is_alive(a) && w.is_alive(b));
    }

    // ── refusals ─────────────────────────────────────────────────────────

    #[test]
    fn refused_links_change_nothing() {
        let mut w = world();
        let (p, c) = (w.spawn(), w.spawn());
        w.set_parent(c, p).unwrap();
        let dead = w.spawn();
        w.despawn(dead);

        assert_eq!(w.set_parent(c, c), Err(HierarchyError::SelfParent(c)));
        assert_eq!(w.set_parent(c, dead), Err(HierarchyError::NotAlive(dead)));
        assert_eq!(w.set_parent(dead, p), Err(HierarchyError::NotAlive(dead)));
        assert_eq!(w.spawn_child(dead).unwrap_err(), HierarchyError::NotAlive(dead));
        assert_eq!(w.parent(c), Some(p));
        assert_eq!(w.children(p), &[c]);
    }

    #[test]
    fn a_link_that_would_close_a_loop_is_refused_at_any_depth() {
        let mut w = world();
        let root = w.spawn();
        let mut chain = vec![root];
        for _ in 0..6 {
            let next = w.spawn_child(*chain.last().unwrap()).unwrap();
            chain.push(next);
        }
        let (top, bottom) = (chain[0], chain[6]);
        assert_eq!(w.set_parent(top, bottom), Err(HierarchyError::Cycle { child: top, parent: bottom }));
        assert_eq!(w.set_parent(chain[2], chain[5]), Err(HierarchyError::Cycle { child: chain[2], parent: chain[5] }));
        assert!(w.parent(top).is_none());
        assert_consistent(&w, &chain);
        // A sibling move that does not loop is fine.
        let side = w.spawn_child(root).unwrap();
        w.set_parent(chain[3], side).unwrap();
        assert_consistent(&w, &chain);
    }

    #[test]
    fn an_invalid_raw_insert_is_taken_out_again() {
        let mut w = world();
        let (a, b) = (w.spawn(), w.spawn());
        let dead = w.spawn();
        w.despawn(dead);

        w.insert_static(a, ChildOf(a));
        assert!(w.parent(a).is_none(), "self parent");
        w.insert_static(a, ChildOf(dead));
        assert!(w.parent(a).is_none(), "dead parent");

        w.set_parent(b, a).unwrap();
        w.insert_static(a, ChildOf(b));
        assert!(w.parent(a).is_none(), "cycle");
        assert_eq!(w.parent(b), Some(a), "the valid link survived");
        assert_consistent(&w, &[a, b]);
    }

    #[test]
    fn a_sparse_shell_child_of_is_rejected() {
        let mut w = world();
        let (p, c) = (w.spawn(), w.spawn());
        w.insert(c, ChildOf(p));
        assert!(w.get::<ChildOf>(c).is_none(), "the hierarchy reads only the Archetype Core");
        assert!(w.children(p).is_empty());
    }

    // ── despawning ───────────────────────────────────────────────────────

    #[test]
    fn despawning_a_child_updates_the_parent() {
        let mut w = world();
        let p = w.spawn();
        let (a, b) = (w.spawn_child(p).unwrap(), w.spawn_child(p).unwrap());
        w.despawn(a);
        assert_eq!(w.children(p), &[b]);
        w.despawn(b);
        assert!(w.get_static::<Children>(p).is_none());
        assert_consistent(&w, &[p, a, b]);
    }

    #[test]
    fn despawning_a_parent_despawns_its_whole_subtree_and_nothing_else() {
        let mut w = world();
        let root = w.spawn();
        let other_root = w.spawn();
        let a = w.spawn_child(root).unwrap();
        let a1 = w.spawn_child(a).unwrap();
        let a2 = w.spawn_child(a).unwrap();
        let a1x = w.spawn_child(a1).unwrap();
        let b = w.spawn_child(root).unwrap();
        let keep = w.spawn_child(other_root).unwrap();
        let all = [root, other_root, a, a1, a2, a1x, b, keep];

        assert!(w.despawn(root));
        for dead in [root, a, a1, a2, a1x, b] {
            assert!(!w.is_alive(dead), "{dead:?} should be gone");
        }
        assert!(w.is_alive(other_root) && w.is_alive(keep));
        assert_eq!(w.children(other_root), &[keep]);
        assert_consistent(&w, &all);
    }

    #[test]
    fn despawning_a_middle_node_takes_its_subtree_and_leaves_the_rest_of_the_tree() {
        let mut w = world();
        let root = w.spawn();
        let (a, b) = (w.spawn_child(root).unwrap(), w.spawn_child(root).unwrap());
        let a1 = w.spawn_child(a).unwrap();
        w.despawn(a);
        assert!(!w.is_alive(a) && !w.is_alive(a1));
        assert_eq!(w.children(root), &[b]);
    }

    #[test]
    fn despawn_children_keeps_the_parent_and_drops_its_children_component() {
        let mut w = world();
        let p = w.spawn();
        let a = w.spawn_child(p).unwrap();
        let a1 = w.spawn_child(a).unwrap();
        let b = w.spawn_child(p).unwrap();
        assert_eq!(w.despawn_children(p), 3);
        assert!(w.is_alive(p));
        assert!(!w.is_alive(a) && !w.is_alive(a1) && !w.is_alive(b));
        assert!(w.get_static::<Children>(p).is_none());
        assert_eq!(w.despawn_children(p), 0);
    }

    #[test]
    fn a_chain_far_deeper_than_the_hook_round_limit_despawns() {
        // The hook-command loop gives up after 1,000 rounds, so a despawn
        // that took one round per level would panic here.
        let mut w = world();
        let root = w.spawn();
        let mut last = root;
        let mut all = vec![root];
        for _ in 0..2_500 {
            last = w.spawn_child(last).unwrap();
            all.push(last);
        }
        assert_eq!(w.ancestors(last).count(), 2_500);
        assert_eq!(w.root_of(last), root);
        assert!(w.despawn(root));
        assert!(all.iter().all(|&e| !w.is_alive(e)));
    }

    #[test]
    fn a_very_wide_parent_clears_and_despawns() {
        let mut w = world();
        let p = w.spawn();
        let kids: Vec<Entity> = (0..20_000).map(|_| w.spawn_child(p).unwrap()).collect();
        assert_eq!(w.children(p).len(), 20_000);
        assert_eq!(w.clear_children(p), 20_000);
        assert!(kids.iter().all(|&k| w.is_alive(k) && w.parent(k).is_none()));

        let q = w.spawn();
        let kids2: Vec<Entity> = (0..20_000).map(|_| w.spawn_child(q).unwrap()).collect();
        assert!(w.despawn(q));
        assert!(kids2.iter().all(|&k| !w.is_alive(k)));
    }

    #[test]
    fn despawning_an_entity_that_was_moved_out_in_the_same_flush_is_not_the_parents_business() {
        // `p` is despawned while `c` is being re-parented by a command that
        // runs in the same flush: `c` must survive, `p`'s own child must not.
        let mut w = world();
        let (p, safe) = (w.spawn(), w.spawn());
        let (c, doomed) = (w.spawn_child(p).unwrap(), w.spawn_child(p).unwrap());
        let mut q = CommandQueue::new();
        q.commands().entity(c).set_parent(safe);
        q.commands().entity(p).despawn();
        q.apply(&mut w);
        assert!(w.is_alive(c) && !w.is_alive(doomed) && !w.is_alive(p));
        assert_eq!(w.parent(c), Some(safe));
    }

    #[test]
    fn a_link_queued_for_a_child_that_loses_its_parent_in_the_same_flush_is_dropped() {
        // `Undo`'s hook runs before `ChildOf`'s, so its command to remove the
        // parent link is queued first. The link command that follows must
        // notice the link is gone instead of listing the child anyway.
        struct Undo;
        fn undo_on_insert(world: &mut DeferredWorld<'_>, ctx: HookContext) {
            let entity = ctx.entity;
            world.commands().queue(move |w| {
                w.remove_static::<ChildOf>(entity);
            });
        }
        let mut w = world();
        w.register_component_hooks::<Undo>().on_insert(undo_on_insert);
        let p = w.spawn();
        let c = w.spawn_bundle((Undo, ChildOf(p)));
        assert!(w.parent(c).is_none());
        assert!(w.children(p).is_empty(), "the child was never listed");
        assert!(w.get_static::<Children>(p).is_none());
        assert_consistent(&w, &[p, c]);
    }

    #[test]
    fn a_child_moved_away_by_another_hook_before_the_subtree_despawn_is_spared() {
        // `Mover`'s despawn hook queues "move the first child to `Safe`"
        // before `Children`'s hook queues the subtree despawn. By the time
        // the subtree despawn runs that child no longer belongs to the
        // despawned parent, and must not be taken with it.
        struct Mover;
        struct Safe(Entity);
        fn mover_on_despawn(world: &mut DeferredWorld<'_>, ctx: HookContext) {
            let Some(&Safe(safe)) = world.get_resource::<Safe>() else { return };
            let Some(kids) = world.get_static::<Children>(ctx.entity).map(|c| c.as_slice().to_vec()) else {
                return;
            };
            world.commands().queue(move |w| {
                if let Some(&first) = kids.first() {
                    let _ = w.set_parent(first, safe);
                }
            });
        }
        let mut w = world();
        w.register_component_hooks::<Mover>().on_despawn(mover_on_despawn);
        let safe = w.spawn();
        w.insert_resource(Safe(safe));
        let p = w.spawn_bundle((Mover,));
        let (moved, doomed) = (w.spawn_child(p).unwrap(), w.spawn_child(p).unwrap());
        w.despawn(p);
        assert!(!w.is_alive(p) && !w.is_alive(doomed));
        assert!(w.is_alive(moved), "it had already been given to `safe`");
        assert_eq!(w.parent(moved), Some(safe));
        assert_eq!(w.children(safe), &[moved]);
    }

    // ── traversal ────────────────────────────────────────────────────────

    // root ── a ── a1
    //      │    └─ a2 ── a2x
    //      └─ b
    fn tree() -> (World, [Entity; 6]) {
        let mut w = world();
        let root = w.spawn();
        let a = w.spawn_child(root).unwrap();
        let a1 = w.spawn_child(a).unwrap();
        let a2 = w.spawn_child(a).unwrap();
        let a2x = w.spawn_child(a2).unwrap();
        let b = w.spawn_child(root).unwrap();
        (w, [root, a, a1, a2, a2x, b])
    }

    #[test]
    fn descendants_are_depth_first_in_child_order() {
        let (w, [root, a, a1, a2, a2x, b]) = tree();
        assert_eq!(w.descendants(root).collect::<Vec<_>>(), [a, a1, a2, a2x, b]);
        assert_eq!(w.descendants(a).collect::<Vec<_>>(), [a1, a2, a2x]);
        assert_eq!(w.descendants(b).count(), 0);
    }

    #[test]
    fn ancestors_run_nearest_first_and_exclude_the_start() {
        let (w, [root, a, _, a2, a2x, _]) = tree();
        assert_eq!(w.ancestors(a2x).collect::<Vec<_>>(), [a2, a, root]);
        assert_eq!(w.ancestors(root).count(), 0);
    }

    #[test]
    fn root_of_and_is_ancestor_of() {
        let (w, [root, a, _, a2, a2x, b]) = tree();
        assert_eq!(w.root_of(a2x), root);
        assert_eq!(w.root_of(root), root);
        assert!(w.is_ancestor_of(a, a2x) && w.is_ancestor_of(root, a2x));
        assert!(!w.is_ancestor_of(a2x, a), "not the other way round");
        assert!(!w.is_ancestor_of(a, a), "not its own ancestor");
        assert!(!w.is_ancestor_of(b, a2x));
    }

    #[test]
    fn dead_entities_have_no_family() {
        let (mut w, [_, _, _, _, _, b]) = tree();
        w.despawn(b);
        assert_eq!(w.parent(b), None);
        assert!(w.children(b).is_empty());
        assert_eq!(w.ancestors(b).count(), 0);
        assert_eq!(w.descendants(b).count(), 0);
    }

    #[test]
    fn children_can_be_queried_like_any_archetype_component() {
        let (w, [root, a, ..]) = tree();
        let mut with_children = Vec::new();
        for (e, kids) in w.query_data::<(Entity, &Children)>() {
            with_children.push((e, kids.len()));
        }
        with_children.sort_by_key(|(e, _)| e.as_ffi());
        assert_eq!(with_children[0], (root, 2));
        assert_eq!(with_children[1], (a, 2));
        assert_eq!(with_children.len(), 3, "root, a and a2 have children");
    }

    // ── Commands ─────────────────────────────────────────────────────────

    #[test]
    fn commands_set_and_remove_a_parent() {
        let mut w = world();
        let (p, c) = (w.spawn(), w.spawn());
        let mut q = CommandQueue::new();
        q.commands().entity(c).set_parent(p);
        q.apply(&mut w);
        assert_eq!(w.children(p), &[c]);
        let mut q = CommandQueue::new();
        q.commands().entity(c).remove_parent();
        q.apply(&mut w);
        assert!(w.children(p).is_empty());
    }

    #[test]
    fn commands_add_children_clear_and_despawn_children() {
        let mut w = world();
        let p = w.spawn();
        let (a, b, c) = (w.spawn(), w.spawn(), w.spawn());
        let mut q = CommandQueue::new();
        q.commands().entity(p).add_child(a).add_children(&[b, c]);
        q.apply(&mut w);
        assert_eq!(w.children(p), &[a, b, c]);

        let mut q = CommandQueue::new();
        q.commands().entity(p).clear_children();
        q.apply(&mut w);
        assert!(w.children(p).is_empty() && w.is_alive(a));

        w.add_children(p, &[a, b]).unwrap();
        let mut q = CommandQueue::new();
        q.commands().entity(p).despawn_children();
        q.apply(&mut w);
        assert!(!w.is_alive(a) && !w.is_alive(b) && w.is_alive(p));
    }

    #[test]
    fn a_refused_command_does_nothing_and_does_not_panic() {
        let mut w = world();
        let p = w.spawn();
        let c = w.spawn_child(p).unwrap();
        let mut q = CommandQueue::new();
        q.commands().entity(p).set_parent(c); // would be a cycle
        q.commands().entity(c).set_parent(c); // itself
        q.apply(&mut w);
        assert!(w.parent(p).is_none());
        assert_eq!(w.parent(c), Some(p));
    }

    #[test]
    fn with_children_spawns_in_order_under_a_queued_parent() {
        let mut w = world();
        let mut q = CommandQueue::new();
        q.commands().spawn().with_children(|kids| {
            kids.spawn_bundle((Tag(1),));
            kids.spawn().insert_static(Tag(2));
            kids.spawn();
        });
        let spawned = q.apply(&mut w);
        assert_eq!(spawned.len(), 4);
        let parent = spawned[0];
        assert_eq!(w.children(parent), &spawned[1..]);
        assert_eq!(w.get_static::<Tag>(spawned[1]), Some(&Tag(1)));
        assert_eq!(w.get_static::<Tag>(spawned[2]), Some(&Tag(2)));
        assert_consistent(&w, &spawned);
    }

    #[test]
    fn with_children_works_under_an_existing_entity_and_can_nest() {
        let mut w = world();
        let root = w.spawn();
        let mut q = CommandQueue::new();
        q.commands().entity(root).with_children(|kids| {
            kids.spawn().with_children(|grand| {
                grand.spawn();
            });
            kids.spawn();
        });
        let spawned = q.apply(&mut w);
        // spawned: child1, grandchild, child2
        assert_eq!(spawned.len(), 3);
        assert_eq!(w.children(root), &[spawned[0], spawned[2]]);
        assert_eq!(w.children(spawned[0]), &[spawned[1]]);
    }

    // ── observers see the lifecycle ──────────────────────────────────────

    #[derive(Default)]
    struct Log(Vec<String>);

    fn saw_child_of(on: On<Add<ChildOf>>, mut log: ResMut<Log>) {
        log.0.push(format!("add ChildOf {}", on.target().is_some()));
    }
    fn saw_children_added(on: On<Add<Children>>, mut log: ResMut<Log>) {
        log.0.push(format!("add Children {}", on.target().is_some()));
    }
    fn saw_children_removed(on: On<Remove<Children>>, mut log: ResMut<Log>) {
        log.0.push(format!("remove Children {}", on.target().is_some()));
    }
    fn saw_child_despawn(_on: On<Despawn<ChildOf>>, mut log: ResMut<Log>) {
        log.0.push("despawn ChildOf".to_string());
    }

    #[test]
    fn observers_see_the_hierarchy_components_come_and_go() {
        let mut w = world();
        w.insert_resource(Log::default());
        w.add_observer(saw_child_of);
        w.add_observer(saw_children_added);
        w.add_observer(saw_children_removed);
        let p = w.spawn();
        let (a, b) = (w.spawn(), w.spawn());
        w.set_parent(a, p).unwrap();
        w.set_parent(b, p).unwrap();
        w.remove_parent(a);
        w.remove_parent(b);
        let log = &w.get_resource::<Log>().unwrap().0;
        assert_eq!(
            log.iter().map(String::as_str).collect::<Vec<_>>(),
            [
                "add ChildOf true",
                "add Children true", // only for the first child
                "add ChildOf true",
                "remove Children true", // only when the last one left
            ]
        );
    }

    #[test]
    fn despawning_a_subtree_notifies_each_child() {
        let mut w = world();
        w.insert_resource(Log::default());
        w.add_observer(saw_child_despawn);
        let p = w.spawn();
        let a = w.spawn_child(p).unwrap();
        w.spawn_child(a).unwrap();
        w.spawn_child(p).unwrap();
        w.despawn(p);
        let n = w.get_resource::<Log>().unwrap().0.len();
        assert_eq!(n, 3, "all three descendants had a ChildOf");
    }

    // ── against a model ──────────────────────────────────────────────────

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

    #[derive(Default)]
    struct Model {
        parent: HashMap<Entity, Entity>,
        kids: HashMap<Entity, Vec<Entity>>,
        alive: Vec<Entity>,
    }

    impl Model {
        fn is_alive(&self, e: Entity) -> bool {
            self.alive.contains(&e)
        }
        fn kids_of(&self, e: Entity) -> Vec<Entity> {
            self.kids.get(&e).cloned().unwrap_or_default()
        }
        fn would_cycle(&self, child: Entity, parent: Entity) -> bool {
            let mut at = parent;
            loop {
                if at == child {
                    return true;
                }
                match self.parent.get(&at) {
                    Some(&p) => at = p,
                    None => return false,
                }
            }
        }
        fn detach(&mut self, child: Entity) -> bool {
            let Some(p) = self.parent.remove(&child) else { return false };
            // The parent's list may already be gone when a whole subtree is
            // being removed; that is not a mismatch.
            if let Some(list) = self.kids.get_mut(&p) {
                list.retain(|&e| e != child);
                if list.is_empty() {
                    self.kids.remove(&p);
                }
            }
            true
        }
        fn attach(&mut self, child: Entity, parent: Entity) {
            self.parent.insert(child, parent);
            self.kids.entry(parent).or_default().push(child);
        }
        fn check(&self, child: Entity, parent: Entity) -> Result<(), HierarchyError> {
            if !self.is_alive(child) {
                Err(HierarchyError::NotAlive(child))
            } else if !self.is_alive(parent) {
                Err(HierarchyError::NotAlive(parent))
            } else if child == parent {
                Err(HierarchyError::SelfParent(child))
            } else if self.would_cycle(child, parent) {
                Err(HierarchyError::Cycle { child, parent })
            } else {
                Ok(())
            }
        }
        fn set_parent(&mut self, child: Entity, parent: Entity) -> Result<(), HierarchyError> {
            self.check(child, parent)?;
            if self.parent.get(&child) != Some(&parent) {
                self.detach(child);
                self.attach(child, parent);
            }
            Ok(())
        }
        fn subtree(&self, root: Entity) -> Vec<Entity> {
            let mut out = vec![root];
            let mut i = 0;
            while i < out.len() {
                out.extend(self.kids_of(out[i]));
                i += 1;
            }
            out
        }
        fn despawn_all(&mut self, set: &[Entity]) {
            for &e in set {
                self.detach(e);
                self.kids.remove(&e);
                self.alive.retain(|&a| a != e);
            }
            // Anything that pointed at a removed entity is gone too.
            self.parent.retain(|c, p| !set.contains(c) && !set.contains(p));
        }
    }

    #[test]
    fn a_random_history_matches_a_model_after_every_step() {
        // Counted so the run cannot pass by never reaching the interesting
        // cases.
        let (mut linked, mut cycles, mut subtree_despawns, mut refused_raw, mut moves) = (0, 0, 0, 0, 0);
        for seed in 0..8u64 {
            let mut rng = Lcg(seed * 7919 + 1);
            let mut w = world();
            let mut m = Model::default();
            let mut everyone: Vec<Entity> = Vec::new();
            for _ in 0..6 {
                let e = w.spawn();
                m.alive.push(e);
                everyone.push(e);
            }
            for step in 0..600 {
                // Mostly live entities, sometimes a dead one, so refusals of
                // dead targets are exercised too.
                let pick = |rng: &mut Lcg, pool: &[Entity], live: &[Entity]| {
                    if rng.below(10) == 0 { pool[rng.below(pool.len())] } else { live[rng.below(live.len())] }
                };
                // Keep enough entities alive for trees to form.
                while m.alive.len() < 10 {
                    let e = w.spawn();
                    m.alive.push(e);
                    everyone.push(e);
                }
                let live = m.alive.clone();
                let a = pick(&mut rng, &everyone, &live);
                let b = pick(&mut rng, &everyone, &live);
                // Links are the common case; despawns are rare because each
                // one removes a whole subtree.
                let op = match rng.below(16) {
                    0 | 1 => 0,
                    2..=6 => 2,
                    7 => 4,
                    8 => 5,
                    9 => 6,
                    10 => 7,
                    11 | 12 => 8,
                    13 => 9,
                    14 => 10,
                    _ => 11,
                };
                let ctx = format!("seed {seed} step {step} op {op}");
                match op {
                    0 | 1 => {
                        let e = w.spawn();
                        m.alive.push(e);
                        everyone.push(e);
                    }
                    2 | 3 => {
                        let had_parent = m.parent.contains_key(&a);
                        let expect = m.set_parent(a, b);
                        assert_eq!(w.set_parent(a, b), expect, "{ctx}");
                        match expect {
                            Ok(()) => {
                                linked += 1;
                                moves += usize::from(had_parent);
                            }
                            Err(HierarchyError::Cycle { .. }) => cycles += 1,
                            Err(_) => {}
                        }
                    }
                    4 => {
                        assert_eq!(w.remove_parent(a), m.detach(a), "{ctx}");
                    }
                    5 => {
                        let set = if m.is_alive(a) { m.subtree(a) } else { vec![] };
                        subtree_despawns += usize::from(set.len() > 2);
                        assert_eq!(w.despawn(a), m.is_alive(a), "{ctx}");
                        m.despawn_all(&set);
                    }
                    6 => {
                        let mut set = m.subtree(a);
                        set.remove(0);
                        let expect = if m.is_alive(a) { set.len() } else { 0 };
                        assert_eq!(w.despawn_children(a), expect, "{ctx}");
                        if m.is_alive(a) {
                            m.despawn_all(&set);
                        }
                    }
                    7 => {
                        let n = m.kids_of(a).len();
                        assert_eq!(w.clear_children(a), n, "{ctx}");
                        for k in m.kids_of(a) {
                            m.detach(k);
                        }
                    }
                    8 => {
                        // A raw insert: ignored if `a` already has a parent,
                        // refused afterwards if the link is invalid.
                        let had = m.parent.contains_key(&a);
                        let inserted = if m.is_alive(a) { !had } else { false };
                        assert_eq!(w.insert_static(a, ChildOf(b)), inserted, "{ctx}");
                        if inserted && m.check(a, b).is_ok() {
                            m.attach(a, b);
                        } else if inserted {
                            refused_raw += 1;
                        }
                    }
                    9 => {
                        let list = [b, pick(&mut rng, &everyone, &live)];
                        let mut expect = 0;
                        for &k in &list {
                            if m.parent.get(&k) == Some(&a) && m.detach(k) {
                                expect += 1;
                            }
                        }
                        assert_eq!(w.remove_children(a, &list), expect, "{ctx}");
                    }
                    10 => {
                        let list = [b, pick(&mut rng, &everyone, &live)];
                        let all_ok = list.iter().all(|&k| m.check(k, a).is_ok());
                        let got = w.add_children(a, &list);
                        assert_eq!(got.is_ok(), all_ok, "{ctx}: {got:?}");
                        if all_ok {
                            for &k in &list {
                                m.set_parent(k, a).unwrap();
                            }
                        }
                    }
                    _ => {
                        // Removing Children by hand detaches the children.
                        let had = !m.kids_of(a).is_empty();
                        assert_eq!(w.remove_static::<Children>(a).is_some(), had, "{ctx}");
                        for k in m.kids_of(a) {
                            m.detach(k);
                        }
                    }
                }
                for &e in &everyone {
                    assert_eq!(w.is_alive(e), m.is_alive(e), "{ctx}: liveness of {e:?}");
                    assert_eq!(w.parent(e), m.parent.get(&e).copied(), "{ctx}: parent of {e:?}");
                    assert_eq!(w.children(e), m.kids_of(e).as_slice(), "{ctx}: children of {e:?}");
                }
                assert_consistent(&w, &everyone);
            }
        }
        assert!(linked > 600, "only {linked} successful links");
        assert!(moves > 200, "only {moves} re-parentings");
        assert!(cycles > 40, "only {cycles} refused cycles");
        assert!(subtree_despawns > 20, "only {subtree_despawns} despawns of a real subtree");
        assert!(refused_raw > 30, "only {refused_raw} invalid raw inserts");
    }
}
