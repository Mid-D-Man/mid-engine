// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-ecs.md, section "Component hooks (`hooks.rs`,
// `ComponentHooks`, `DeferredWorld`)"
// ============================================================================
//! Component hooks: functions that run when a component is added to,
//! overwritten on, or removed from an entity.
//!
//! A hook is a plain `fn` (no captures; keep state in a resource), set once
//! per component type with [`World::register_component_hooks`]. It runs
//! synchronously inside the `World` operation that caused it, with a
//! [`DeferredWorld`]: a handle that can read and write *values* but cannot
//! change the world's *structure*, so a hook can never invalidate the
//! operation it runs inside. Structural changes a hook wants (spawn,
//! insert, remove, despawn) go through [`DeferredWorld::commands`] and are
//! applied when the outermost operation has finished.
//!
//! | hook | fires | when relative to the change |
//! |---|---|---|
//! | `on_add` | a component is added to an entity that did not have it | after |
//! | `on_insert` | a component is added or overwritten | after |
//! | `on_discard` | a component is removed, or overwritten | before (the old value is still readable) |
//! | `on_remove` | a component is removed and not replaced | before |
//! | `on_despawn` | each component of an entity being despawned | before |
//!
//! Orders (bevy's): `on_add` before `on_insert`; for a removal `on_discard`
//! before `on_remove`; for a despawn `on_despawn` for every component, then
//! `on_discard` for every component, then `on_remove` for every component,
//! all before anything is removed. Between components of one kind the order
//! is not defined.
//!
//! The registry is keyed by `TypeId`, so a hook works for a component in
//! either storage, and for the FFI byte paths.

use std::any::TypeId;

use crate::commands::{CommandQueue, Commands};
use crate::hash::TypeIdMap;
use crate::tick::Tick;
use crate::world::{Entity, World};

/// A component hook. Plain `fn`, not a closure: keep state in a resource
/// and reach it through the [`DeferredWorld`].
pub type ComponentHook = fn(&mut DeferredWorld<'_>, HookContext);

/// What a hook is told about the event that fired it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HookContext {
    /// The entity the component was added to / removed from.
    pub entity: Entity,
    /// The component's type.
    pub type_id: TypeId,
}

/// Which hook of a [`ComponentHooks`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum HookKind {
    Add,
    Insert,
    Discard,
    Remove,
    Despawn,
}

/// The hooks of one component type: at most one of each kind.
///
/// Setting a kind twice panics: a hook is the component type's own
/// constructor/destructor, and two of them would silently fight over who
/// runs. (For several independent reactions to one event, react from a
/// single hook, or use `RemovedComponents` / `Added` from a system.)
#[derive(Clone, Copy, Default)]
pub struct ComponentHooks {
    on_add: Option<ComponentHook>,
    on_insert: Option<ComponentHook>,
    on_discard: Option<ComponentHook>,
    on_remove: Option<ComponentHook>,
    on_despawn: Option<ComponentHook>,
}

impl ComponentHooks {
    pub(crate) fn of(&self, kind: HookKind) -> Option<ComponentHook> {
        match kind {
            HookKind::Add => self.on_add,
            HookKind::Insert => self.on_insert,
            HookKind::Discard => self.on_discard,
            HookKind::Remove => self.on_remove,
            HookKind::Despawn => self.on_despawn,
        }
    }

    /// Runs after a component is added to an entity that did not have it.
    ///
    /// # Panics
    /// If an `on_add` hook is already set.
    pub fn on_add(&mut self, hook: ComponentHook) -> &mut Self {
        assert!(
            self.on_add.is_none(),
            "this component already has an on_add hook"
        );
        self.on_add = Some(hook);
        self
    }

    /// Runs after a component is added or overwritten (after `on_add`).
    ///
    /// # Panics
    /// If an `on_insert` hook is already set.
    pub fn on_insert(&mut self, hook: ComponentHook) -> &mut Self {
        assert!(
            self.on_insert.is_none(),
            "this component already has an on_insert hook"
        );
        self.on_insert = Some(hook);
        self
    }

    /// Runs before a component is removed or overwritten, with the old
    /// value still in place (before `on_remove`).
    ///
    /// # Panics
    /// If an `on_discard` hook is already set.
    pub fn on_discard(&mut self, hook: ComponentHook) -> &mut Self {
        assert!(
            self.on_discard.is_none(),
            "this component already has an on_discard hook"
        );
        self.on_discard = Some(hook);
        self
    }

    /// Runs before a component is removed and not replaced, with the value
    /// still in place (after `on_discard`).
    ///
    /// # Panics
    /// If an `on_remove` hook is already set.
    pub fn on_remove(&mut self, hook: ComponentHook) -> &mut Self {
        assert!(
            self.on_remove.is_none(),
            "this component already has an on_remove hook"
        );
        self.on_remove = Some(hook);
        self
    }

    /// Runs, for each of its components, when an entity is despawned,
    /// before anything is removed (before `on_discard` and `on_remove`).
    ///
    /// # Panics
    /// If an `on_despawn` hook is already set.
    pub fn on_despawn(&mut self, hook: ComponentHook) -> &mut Self {
        assert!(
            self.on_despawn.is_none(),
            "this component already has an on_despawn hook"
        );
        self.on_despawn = Some(hook);
        self
    }
}

/// How many times the commands a hook queued are applied, one round after
/// another, before the world gives up: a hook that always queues another
/// structural change whose hook queues another would otherwise never end.
pub(crate) const MAX_HOOK_ROUNDS: usize = 1_000;

/// The world's registered hooks and the commands they have queued.
#[derive(Default)]
pub(crate) struct HookRegistry {
    hooks: TypeIdMap<ComponentHooks>,
    /// Structural changes hooks asked for, applied when the outermost
    /// operation finishes (see `World::flush_hook_commands`).
    pub(crate) pending: CommandQueue,
    /// Set while those are being applied, so a nested operation's own
    /// flush leaves them to the loop that is already running.
    pub(crate) flushing: bool,
}

impl HookRegistry {
    /// Whether any component type has any hook: the one check a world
    /// without hooks pays per structural operation.
    #[inline]
    pub(crate) fn is_active(&self) -> bool {
        !self.hooks.is_empty()
    }

    pub(crate) fn entry(&mut self, type_id: TypeId) -> &mut ComponentHooks {
        self.hooks.entry(type_id).or_default()
    }

    /// The hooks of `type_id` (copied out, so a hook can run with the
    /// world borrowed mutably).
    #[inline]
    pub(crate) fn get(&self, type_id: TypeId) -> Option<ComponentHooks> {
        self.hooks.get(&type_id).copied()
    }
}

/// What a hook gets instead of `&mut World`: everything that reads or
/// writes *values*, nothing that changes the world's *structure*.
///
/// It has no `spawn`, `despawn`, `insert*` or `remove*`: a hook runs inside
/// a structural operation, and one that changed structure under it could
/// invalidate what the operation is doing. Queue such changes with
/// [`Self::commands`]; they are applied, in order, when the outermost
/// operation has finished (and their own hooks run then).
///
/// ```compile_fail
/// use mid_ecs::{DeferredWorld, HookContext};
///
/// fn bad(world: &mut DeferredWorld<'_>, ctx: HookContext) {
///     world.despawn(ctx.entity);
/// }
/// ```
pub struct DeferredWorld<'w> {
    world: &'w mut World,
}

impl<'w> DeferredWorld<'w> {
    pub(crate) fn new(world: &'w mut World) -> Self {
        Self { world }
    }

    /// Whether `entity` is alive.
    pub fn is_alive(&self, entity: Entity) -> bool {
        self.world.is_alive(entity)
    }

    /// The world's current change tick.
    pub fn change_tick(&self) -> Tick {
        self.world.change_tick()
    }

    /// `entity`'s archetype-storage `T`, if it has one.
    pub fn get_static<T: 'static>(&self, entity: Entity) -> Option<&T> {
        self.world.get_static(entity)
    }

    /// `entity`'s archetype-storage `T`, mutably (stamped changed).
    pub fn get_static_mut<T: 'static>(&mut self, entity: Entity) -> Option<&mut T> {
        self.world.get_static_mut(entity)
    }

    /// Whether `entity` has an archetype-storage `T`.
    pub fn has_static<T: 'static>(&self, entity: Entity) -> bool {
        self.world.has_static::<T>(entity)
    }

    /// `entity`'s Sparse Shell `T`, if it has one.
    pub fn get<T: 'static>(&self, entity: Entity) -> Option<&T> {
        self.world.get(entity)
    }

    /// `entity`'s Sparse Shell `T`, mutably.
    pub fn get_mut<T: 'static>(&mut self, entity: Entity) -> Option<&mut T> {
        self.world.get_mut(entity)
    }

    /// The resource of type `T`, if there is one.
    pub fn get_resource<T: 'static>(&self) -> Option<&T> {
        self.world.get_resource()
    }

    /// The resource of type `T`, mutably.
    pub fn get_resource_mut<T: 'static>(&mut self) -> Option<&mut T> {
        self.world.get_resource_mut()
    }

    /// Queues structural changes, applied when the outermost world
    /// operation finishes.
    pub fn commands(&mut self) -> Commands<'_> {
        self.world.hooks.pending.commands()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::component::ComponentId;
    use crate::system::{IntoSystem, System};
    use crate::system_param::RemovedComponents;
    use crate::tick::ChangeTracker;
    use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

    /// Archetype-storage components with hooks.
    #[derive(Debug, PartialEq)]
    struct Hp(u32);
    #[derive(Debug, PartialEq)]
    struct Mp(u32);
    /// A Sparse Shell component with hooks.
    #[derive(Debug, PartialEq)]
    struct Cn(u32);
    /// No hooks, ever.
    #[derive(Debug, PartialEq)]
    struct Other(u32);

    #[derive(Debug, Clone, Copy, PartialEq, FromBytes, IntoBytes, Immutable, KnownLayout)]
    #[repr(C)]
    struct Fa(u32);
    #[derive(Debug, Clone, Copy, PartialEq, FromBytes, IntoBytes, Immutable, KnownLayout)]
    #[repr(C)]
    struct Fb(u32);
    /// Registered read-only: C may not insert or remove it.
    #[derive(Debug, Clone, Copy, PartialEq, IntoBytes, Immutable, KnownLayout)]
    #[repr(C)]
    struct Ro(u32);

    /// One recorded hook call, with what the hook saw of the world.
    #[derive(Debug, Clone, PartialEq)]
    struct Ev {
        op: usize,
        kind: HookKind,
        entity: Entity,
        ty: &'static str,
        /// The component's value as the hook saw it (`None`: absent).
        value: Option<u32>,
        alive: bool,
    }

    #[derive(Default)]
    struct Log {
        op: usize,
        events: Vec<Ev>,
    }

    fn kind_of(k: u8) -> HookKind {
        [
            HookKind::Add,
            HookKind::Insert,
            HookKind::Discard,
            HookKind::Remove,
            HookKind::Despawn,
        ][k as usize]
    }

    const ADD: u8 = 0;
    const INSERT: u8 = 1;
    const DISCARD: u8 = 2;
    const REMOVE: u8 = 3;
    const DESPAWN: u8 = 4;

    /// A hook that logs the event and the component's value as seen.
    macro_rules! probe {
        ($name:ident, $ty:ident, $get:ident, $field:tt) => {
            fn $name<const K: u8>(w: &mut DeferredWorld<'_>, c: HookContext) {
                let value = w.$get::<$ty>(c.entity).map(|v| v.$field);
                let alive = w.is_alive(c.entity);
                let log = w.get_resource_mut::<Log>().unwrap();
                let op = log.op;
                log.events.push(Ev {
                    op,
                    kind: kind_of(K),
                    entity: c.entity,
                    ty: stringify!($ty),
                    value,
                    alive,
                });
            }
        };
    }
    probe!(probe_hp, Hp, get_static, 0);
    probe!(probe_mp, Mp, get_static, 0);
    probe!(probe_cn, Cn, get, 0);
    probe!(probe_fa, Fa, get_static, 0);
    probe!(probe_fb, Fb, get_static, 0);
    probe!(probe_ro, Ro, get_static, 0);

    fn hook_all<T: 'static>(world: &mut World, hooks: [ComponentHook; 5]) {
        world
            .register_component_hooks::<T>()
            .on_add(hooks[0])
            .on_insert(hooks[1])
            .on_discard(hooks[2])
            .on_remove(hooks[3])
            .on_despawn(hooks[4]);
    }

    macro_rules! all_kinds {
        ($f:ident) => {
            [
                $f::<ADD>,
                $f::<INSERT>,
                $f::<DISCARD>,
                $f::<REMOVE>,
                $f::<DESPAWN>,
            ]
        };
    }

    /// A world with every kind of hook on Hp, Mp (archetype) and Cn
    /// (Sparse Shell), logging into a `Log`.
    fn hooked() -> World {
        let mut world = World::new();
        world.insert_resource(Log::default());
        hook_all::<Hp>(&mut world, all_kinds!(probe_hp));
        hook_all::<Mp>(&mut world, all_kinds!(probe_mp));
        hook_all::<Cn>(&mut world, all_kinds!(probe_cn));
        world
    }

    fn take(world: &mut World) -> Vec<Ev> {
        std::mem::take(&mut world.get_resource_mut::<Log>().unwrap().events)
    }

    /// `(kind, type, value)` of every event, dropping the rest.
    fn short(events: &[Ev]) -> Vec<(HookKind, &'static str, Option<u32>)> {
        events.iter().map(|e| (e.kind, e.ty, e.value)).collect()
    }

    use HookKind::{Add, Despawn, Discard, Insert, Remove};

    // ── What fires, in what order, seeing what ───────────────────────

    #[test]
    fn a_new_archetype_component_fires_add_then_insert_with_the_value_in_place() {
        let mut world = hooked();
        let e = world.spawn();
        assert!(world.insert_static(e, Hp(7)));
        let events = take(&mut world);
        assert_eq!(
            short(&events),
            vec![(Add, "Hp", Some(7)), (Insert, "Hp", Some(7))]
        );
        assert!(events.iter().all(|ev| ev.entity == e && ev.alive));
    }

    #[test]
    fn an_insert_that_changes_nothing_fires_nothing() {
        let mut world = hooked();
        let (e, dead) = (world.spawn(), world.spawn());
        world.insert_static(e, Hp(1));
        world.despawn(dead);
        take(&mut world);
        assert!(!world.insert_static(e, Hp(2)), "already has it");
        assert!(!world.insert_static(dead, Hp(3)), "not alive");
        assert!(!world.insert_bundle(dead, (Hp(1), Mp(1))));
        assert_eq!(world.insert(dead, Cn(1)), None);
        assert!(take(&mut world).is_empty());
    }

    #[test]
    fn remove_static_fires_discard_then_remove_before_the_value_goes() {
        let mut world = hooked();
        let e = world.spawn();
        world.insert_static(e, Hp(7));
        take(&mut world);
        assert_eq!(world.remove_static::<Hp>(e), Some(Hp(7)));
        let events = take(&mut world);
        assert_eq!(
            short(&events),
            vec![(Discard, "Hp", Some(7)), (Remove, "Hp", Some(7))],
            "both hooks still see the value"
        );
        assert!(world.get_static::<Hp>(e).is_none());
    }

    #[test]
    fn a_removal_that_removes_nothing_fires_nothing() {
        let mut world = hooked();
        let (e, dead) = (world.spawn(), world.spawn());
        world.insert_static(dead, Hp(1));
        world.despawn(dead);
        take(&mut world);
        assert_eq!(world.remove_static::<Hp>(e), None, "never had it");
        assert_eq!(world.remove_static::<Hp>(dead), None);
        assert_eq!(world.remove::<Cn>(e), None);
        assert_eq!(world.remove_static::<Other>(e), None, "unregistered type");
        assert!(world.remove_bundle::<(Hp, Mp)>(e).is_none());
        assert!(!world.despawn(dead), "already gone");
        assert!(take(&mut world).is_empty());
    }

    #[test]
    fn the_sparse_shell_fires_add_insert_then_discard_insert_on_overwrite_then_discard_remove() {
        let mut world = hooked();
        let e = world.spawn();
        assert_eq!(world.insert(e, Cn(1)), None);
        assert_eq!(
            short(&take(&mut world)),
            vec![(Add, "Cn", Some(1)), (Insert, "Cn", Some(1))]
        );
        // Overwrite: discard sees the OLD value, insert the new; no add.
        assert_eq!(world.insert(e, Cn(2)), Some(Cn(1)));
        assert_eq!(
            short(&take(&mut world)),
            vec![(Discard, "Cn", Some(1)), (Insert, "Cn", Some(2))]
        );
        assert_eq!(world.remove::<Cn>(e), Some(Cn(2)));
        assert_eq!(
            short(&take(&mut world)),
            vec![(Discard, "Cn", Some(2)), (Remove, "Cn", Some(2))]
        );
    }

    #[test]
    fn despawn_fires_every_despawn_then_every_discard_then_every_remove_all_before_removal() {
        let mut world = hooked();
        let e = world.spawn();
        world.insert_static(e, Hp(1));
        world.insert_static(e, Mp(2));
        world.insert(e, Cn(3));
        world.insert_static(e, Other(9));
        take(&mut world);
        assert!(world.despawn(e));
        let events = take(&mut world);
        assert_eq!(events.len(), 9, "three components, three kinds");
        let kinds: Vec<_> = events.iter().map(|ev| ev.kind).collect();
        assert_eq!(
            kinds,
            [Despawn, Despawn, Despawn, Discard, Discard, Discard, Remove, Remove, Remove]
        );
        for group in events.chunks(3) {
            let mut types: Vec<_> = group.iter().map(|ev| ev.ty).collect();
            types.sort();
            assert_eq!(types, ["Cn", "Hp", "Mp"]);
        }
        assert!(
            events.iter().all(|ev| ev.value.is_some() && ev.alive),
            "every hook sees the entity alive and the value in place"
        );
        assert!(!world.is_alive(e));
    }

    #[test]
    fn despawning_an_entity_with_no_hooked_components_fires_nothing() {
        let mut world = hooked();
        let e = world.spawn();
        world.insert_static(e, Other(1));
        let bare = world.spawn();
        take(&mut world);
        assert!(world.despawn(e));
        assert!(world.despawn(bare));
        assert!(take(&mut world).is_empty());
    }

    // ── Bundles ──────────────────────────────────────────────────────

    fn kinds_and_types(events: &[Ev]) -> (Vec<HookKind>, Vec<&'static str>) {
        (
            events.iter().map(|e| e.kind).collect(),
            events.iter().map(|e| e.ty).collect(),
        )
    }

    #[test]
    fn a_bundle_fires_every_add_then_every_insert() {
        let mut world = hooked();
        let spawned = world.spawn_bundle((Hp(1), Mp(2)));
        let events = take(&mut world);
        assert_eq!(kinds_and_types(&events).0, [Add, Add, Insert, Insert]);
        assert!(events
            .iter()
            .all(|e| e.entity == spawned && e.value.is_some()));

        let e = world.spawn();
        assert!(world.insert_bundle(e, (Hp(1), Mp(2))));
        let events = take(&mut world);
        assert_eq!(kinds_and_types(&events).0, [Add, Add, Insert, Insert]);

        // All or nothing: a member already present refuses the whole bundle.
        let half = world.spawn();
        world.insert_static(half, Hp(5));
        take(&mut world);
        assert!(!world.insert_bundle(half, (Hp(6), Mp(6))));
        assert!(take(&mut world).is_empty());
    }

    #[test]
    fn remove_bundle_fires_every_discard_then_every_remove_and_a_partial_bundle_fires_nothing() {
        let mut world = hooked();
        let (full, partial) = (world.spawn(), world.spawn());
        world.insert_bundle(full, (Hp(1), Mp(2)));
        world.insert_static(partial, Hp(3));
        take(&mut world);
        assert!(world.remove_bundle::<(Hp, Mp)>(partial).is_none());
        assert!(take(&mut world).is_empty());
        assert!(world.remove_bundle::<(Hp, Mp)>(full).is_some());
        let events = take(&mut world);
        assert_eq!(
            kinds_and_types(&events).0,
            [Discard, Discard, Remove, Remove]
        );
        assert!(events.iter().all(|e| e.value.is_some()), "still in place");
    }

    // ── The FFI byte paths ───────────────────────────────────────────

    fn ffi_world() -> (World, [ComponentId; 3]) {
        let mut world = World::new();
        world.insert_resource(Log::default());
        let fa = world.register_ffi_static_component_writable::<Fa>("Fa");
        let fb = world.register_ffi_static_component_writable::<Fb>("Fb");
        let ro = world.register_ffi_static_component::<Ro>("Ro");
        hook_all::<Fa>(&mut world, all_kinds!(probe_fa));
        hook_all::<Fb>(&mut world, all_kinds!(probe_fb));
        hook_all::<Ro>(&mut world, all_kinds!(probe_ro));
        (world, [fa, fb, ro])
    }

    #[test]
    fn byte_inserts_fire_add_and_insert_and_refused_ones_fire_nothing() {
        let (mut world, [fa, fb, ro]) = ffi_world();
        let e = world.spawn();
        assert_eq!(world.insert_static_bytes(e, fa, Fa(5).as_bytes()), Ok(()));
        assert_eq!(
            short(&take(&mut world)),
            vec![(Add, "Fa", Some(5)), (Insert, "Fa", Some(5))]
        );
        // Refused: already present, wrong size, read-only, dead.
        assert!(world.insert_static_bytes(e, fa, Fa(6).as_bytes()).is_err());
        assert!(world.insert_static_bytes(e, fb, &[1, 2]).is_err());
        assert!(world.insert_static_bytes(e, ro, Ro(1).as_bytes()).is_err());
        let dead = world.spawn();
        world.despawn(dead);
        assert!(world
            .insert_static_bytes(dead, fb, Fb(1).as_bytes())
            .is_err());
        assert!(take(&mut world).is_empty());

        // Several at once: every add, then every insert; an empty list
        // changes nothing and fires nothing.
        let f = world.spawn();
        let (a, b) = (Fa(1), Fb(2));
        assert_eq!(
            world.insert_static_bytes_many(f, &[(fa, a.as_bytes()), (fb, b.as_bytes())]),
            Ok(())
        );
        let events = take(&mut world);
        assert_eq!(kinds_and_types(&events).0, [Add, Add, Insert, Insert]);
        assert_eq!(world.insert_static_bytes_many(f, &[]), Ok(()));
        assert!(take(&mut world).is_empty());
        // A refused many (one already present) fires nothing.
        assert!(world
            .insert_static_bytes_many(f, &[(fa, Fa(9).as_bytes())])
            .is_err());
        assert!(take(&mut world).is_empty());
    }

    #[test]
    fn byte_removals_fire_discard_and_remove_and_refused_ones_fire_nothing() {
        let (mut world, [fa, fb, ro]) = ffi_world();
        let e = world.spawn();
        world.insert_static_bytes(e, fa, Fa(5).as_bytes()).unwrap();
        world.insert_static_bytes(e, fb, Fb(6).as_bytes()).unwrap();
        take(&mut world);

        assert_eq!(world.remove_static_by_id(e, fa), Ok(()));
        assert_eq!(
            short(&take(&mut world)),
            vec![(Discard, "Fa", Some(5)), (Remove, "Fa", Some(5))],
            "the value is still there for the hooks"
        );
        // Refused: absent, unknown/read-only, duplicate in the list, dead.
        assert!(world.remove_static_by_id(e, fa).is_err(), "absent now");
        assert!(world.remove_static_by_id(e, ro).is_err(), "read-only");
        assert!(
            world.remove_static_by_ids(e, &[fb, fb]).is_err(),
            "duplicate"
        );
        assert!(
            world.remove_static_by_ids(e, &[fa, fb]).is_err(),
            "fa is gone"
        );
        assert!(
            world.remove_static_by_ids(e, &[]).is_ok(),
            "empty is a no-op"
        );
        assert!(take(&mut world).is_empty());
        assert!(world.get_static::<Fb>(e).is_some(), "nothing was removed");

        assert_eq!(world.remove_static_by_ids(e, &[fb]), Ok(()));
        assert_eq!(
            short(&take(&mut world)),
            vec![(Discard, "Fb", Some(6)), (Remove, "Fb", Some(6))]
        );
    }

    // ── What a hook can do ───────────────────────────────────────────

    #[test]
    fn a_hook_can_change_values_and_the_change_is_stamped() {
        fn bump(w: &mut DeferredWorld<'_>, c: HookContext) {
            w.get_static_mut::<Hp>(c.entity).unwrap().0 += 100;
        }
        let mut world = World::new();
        world.register_component_hooks::<Hp>().on_add(bump);
        let e = world.spawn();
        world.increment_change_tick();
        let mut tracker = ChangeTracker::new();
        tracker.update(&world);
        world.increment_change_tick();
        world.insert_static(e, Hp(7));
        assert_eq!(world.get_static::<Hp>(e), Some(&Hp(107)));
        assert_eq!(world.query_changed::<Hp>(&tracker).count(), 1);
    }

    #[test]
    fn a_hook_can_reach_the_sparse_shell_and_resources_and_the_tick() {
        fn touch(w: &mut DeferredWorld<'_>, c: HookContext) {
            w.get_mut::<Cn>(c.entity).unwrap().0 += 1;
            let tick = w.change_tick().get();
            w.get_resource_mut::<Log>().unwrap().op = tick as usize;
        }
        let mut world = World::new();
        world.insert_resource(Log::default());
        world.register_component_hooks::<Hp>().on_add(touch);
        let e = world.spawn();
        world.insert(e, Cn(1));
        world.insert_static(e, Hp(0));
        assert_eq!(world.get::<Cn>(e), Some(&Cn(2)));
        assert_eq!(
            world.get_resource::<Log>().unwrap().op,
            world.change_tick().get() as usize
        );
    }

    #[test]
    fn structural_changes_a_hook_queues_apply_after_the_operation_with_their_own_hooks() {
        fn queue_mp(w: &mut DeferredWorld<'_>, c: HookContext) {
            // Not visible yet from inside the hook.
            assert!(!w.has_static::<Mp>(c.entity));
            w.commands().entity(c.entity).insert_static(Mp(50));
            probe_hp::<ADD>(w, c);
        }
        let mut world = World::new();
        world.insert_resource(Log::default());
        world.register_component_hooks::<Hp>().on_add(queue_mp);
        hook_all::<Mp>(&mut world, all_kinds!(probe_mp));
        let e = world.spawn();
        assert!(world.insert_static(e, Hp(1)));
        // Applied by the time `insert_static` returned.
        assert_eq!(world.get_static::<Mp>(e), Some(&Mp(50)));
        // Hp's hook ran first, then Mp's own hooks for the queued insert.
        let events = take(&mut world);
        assert_eq!(
            short(&events),
            vec![
                (Add, "Hp", Some(1)),
                (Add, "Mp", Some(50)),
                (Insert, "Mp", Some(50))
            ]
        );
    }

    #[test]
    fn a_hook_can_despawn_through_commands() {
        fn reap(w: &mut DeferredWorld<'_>, c: HookContext) {
            w.commands().despawn(c.entity);
        }
        let mut world = World::new();
        world.register_component_hooks::<Hp>().on_add(reap);
        let e = world.spawn();
        world.insert_static(e, Hp(1));
        assert!(!world.is_alive(e));
    }

    #[test]
    #[should_panic(expected = "did not settle")]
    fn hooks_that_keep_queueing_each_other_are_stopped() {
        fn add_queues_remove(w: &mut DeferredWorld<'_>, c: HookContext) {
            w.commands().entity(c.entity).remove_static::<Hp>();
        }
        fn remove_queues_add(w: &mut DeferredWorld<'_>, c: HookContext) {
            w.commands().entity(c.entity).insert_static(Hp(1));
        }
        let mut world = World::new();
        world
            .register_component_hooks::<Hp>()
            .on_add(add_queues_remove)
            .on_remove(remove_queues_add);
        let e = world.spawn();
        world.insert_static(e, Hp(1));
    }

    #[test]
    fn the_world_is_usable_after_a_runaway_chain_was_stopped() {
        fn add_queues_remove(w: &mut DeferredWorld<'_>, c: HookContext) {
            w.commands().entity(c.entity).remove_static::<Hp>();
        }
        fn remove_queues_add(w: &mut DeferredWorld<'_>, c: HookContext) {
            w.commands().entity(c.entity).insert_static(Hp(1));
        }
        let mut world = World::new();
        world
            .register_component_hooks::<Hp>()
            .on_add(add_queues_remove)
            .on_remove(remove_queues_add);
        let e = world.spawn();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            world.insert_static(e, Hp(1));
        }));
        assert!(result.is_err());
        assert!(!world.hooks.flushing, "the flush flag was reset");
        assert!(world.hooks.pending.is_empty(), "and the queue dropped");
    }

    #[test]
    fn every_structural_operation_applies_the_commands_its_hooks_queued() {
        struct Marker;
        struct Target(Entity);
        fn queue_marker(w: &mut DeferredWorld<'_>, _c: HookContext) {
            let target = w.get_resource::<Target>().unwrap().0;
            w.commands().entity(target).insert_static(Marker);
        }
        type Ids = [ComponentId; 2];
        type Step = Box<dyn Fn(&mut World, Entity, Ids)>;

        let mut world = World::new();
        let fa = world.register_ffi_static_component_writable::<Fa>("Fa");
        let fb = world.register_ffi_static_component_writable::<Fb>("Fb");
        for hooks in [[queue_marker as ComponentHook; 5]] {
            hook_all::<Hp>(&mut world, hooks);
            hook_all::<Mp>(&mut world, hooks);
            hook_all::<Cn>(&mut world, hooks);
            hook_all::<Fa>(&mut world, hooks);
            hook_all::<Fb>(&mut world, hooks);
        }
        let scratch = world.spawn();
        world.insert_resource(Target(scratch));

        let nothing: Step = Box::new(|_, _, _| {});
        let ops: Vec<(&str, Step, Step)> = vec![
            (
                "insert_static",
                nothing_step(),
                Box::new(|w, a, _| {
                    w.insert_static(a, Hp(1));
                }),
            ),
            (
                "insert (sparse, new)",
                nothing_step(),
                Box::new(|w, a, _| {
                    w.insert(a, Cn(1));
                }),
            ),
            (
                "insert (sparse, overwrite)",
                Box::new(|w, a, _| {
                    w.insert(a, Cn(1));
                }),
                Box::new(|w, a, _| {
                    w.insert(a, Cn(2));
                }),
            ),
            (
                "spawn_bundle",
                nothing_step(),
                Box::new(|w, _, _| {
                    w.spawn_bundle((Hp(1), Mp(1)));
                }),
            ),
            (
                "insert_bundle",
                nothing_step(),
                Box::new(|w, a, _| {
                    w.insert_bundle(a, (Hp(1), Mp(1)));
                }),
            ),
            (
                "remove_static",
                Box::new(|w, a, _| {
                    w.insert_static(a, Hp(1));
                }),
                Box::new(|w, a, _| {
                    w.remove_static::<Hp>(a);
                }),
            ),
            (
                "remove (sparse)",
                Box::new(|w, a, _| {
                    w.insert(a, Cn(1));
                }),
                Box::new(|w, a, _| {
                    w.remove::<Cn>(a);
                }),
            ),
            (
                "remove_bundle",
                Box::new(|w, a, _| {
                    w.insert_bundle(a, (Hp(1), Mp(1)));
                }),
                Box::new(|w, a, _| {
                    w.remove_bundle::<(Hp, Mp)>(a);
                }),
            ),
            (
                "despawn",
                Box::new(|w, a, _| {
                    w.insert_static(a, Hp(1));
                }),
                Box::new(|w, a, _| {
                    w.despawn(a);
                }),
            ),
            (
                "insert_static_bytes",
                nothing_step(),
                Box::new(|w, a, ids| {
                    w.insert_static_bytes(a, ids[0], Fa(1).as_bytes()).unwrap();
                }),
            ),
            (
                "insert_static_bytes_many",
                nothing_step(),
                Box::new(|w, a, ids| {
                    let (x, y) = (Fa(1), Fb(2));
                    w.insert_static_bytes_many(
                        a,
                        &[(ids[0], x.as_bytes()), (ids[1], y.as_bytes())],
                    )
                    .unwrap();
                }),
            ),
            (
                "remove_static_by_id",
                Box::new(|w, a, ids| {
                    w.insert_static_bytes(a, ids[0], Fa(1).as_bytes()).unwrap();
                }),
                Box::new(|w, a, ids| {
                    w.remove_static_by_id(a, ids[0]).unwrap();
                }),
            ),
            (
                "remove_static_by_ids",
                Box::new(|w, a, ids| {
                    let (x, y) = (Fa(1), Fb(2));
                    w.insert_static_bytes_many(
                        a,
                        &[(ids[0], x.as_bytes()), (ids[1], y.as_bytes())],
                    )
                    .unwrap();
                }),
                Box::new(|w, a, ids| {
                    w.remove_static_by_ids(a, &[ids[0], ids[1]]).unwrap();
                }),
            ),
        ];
        let _ = nothing;
        fn nothing_step() -> Step {
            Box::new(|_, _, _| {})
        }

        // One world for all of them, in order: a flush that forgot to
        // clear its flag would silently skip every later operation.
        for (name, prep, op) in &ops {
            let actor = world.spawn();
            prep(&mut world, actor, [fa, fb]);
            let target = world.spawn();
            world.insert_resource(Target(target));
            op(&mut world, actor, [fa, fb]);
            assert!(
                world.has_static::<Marker>(target),
                "{name}: the commands its hooks queued were not applied"
            );
        }
    }

    #[test]
    fn a_chain_of_hook_commands_that_settles_is_applied_in_full() {
        struct C1;
        struct C2;
        struct C3;
        struct C4;
        fn queue_c2(w: &mut DeferredWorld<'_>, c: HookContext) {
            w.commands().entity(c.entity).insert_static(C2);
        }
        fn queue_c3(w: &mut DeferredWorld<'_>, c: HookContext) {
            w.commands().entity(c.entity).insert_static(C3);
        }
        fn queue_c4(w: &mut DeferredWorld<'_>, c: HookContext) {
            w.commands().entity(c.entity).insert_static(C4);
        }
        let mut world = World::new();
        world.register_component_hooks::<C1>().on_add(queue_c2);
        world.register_component_hooks::<C2>().on_add(queue_c3);
        world.register_component_hooks::<C3>().on_add(queue_c4);
        let e = world.spawn();
        world.insert_static(e, C1);
        // Three rounds, each one's hook queueing the next.
        assert!(
            world.has_static::<C2>(e) && world.has_static::<C3>(e) && world.has_static::<C4>(e)
        );
    }

    #[test]
    fn a_hook_can_ask_about_other_entities_and_other_components() {
        #[derive(Default)]
        struct Probe {
            dead_alive: Option<bool>,
            self_has_hp: Option<bool>,
            self_has_mp: Option<bool>,
        }
        struct Dead(Entity);
        fn peek(w: &mut DeferredWorld<'_>, c: HookContext) {
            let dead = w.get_resource::<Dead>().unwrap().0;
            let (dead_alive, hp, mp) = (
                w.is_alive(dead),
                w.has_static::<Hp>(c.entity),
                w.has_static::<Mp>(c.entity),
            );
            let probe = w.get_resource_mut::<Probe>().unwrap();
            probe.dead_alive = Some(dead_alive);
            probe.self_has_hp = Some(hp);
            probe.self_has_mp = Some(mp);
        }
        let mut world = World::new();
        world.insert_resource(Probe::default());
        let dead = world.spawn();
        world.despawn(dead);
        world.insert_resource(Dead(dead));
        world.register_component_hooks::<Hp>().on_add(peek);
        let e = world.spawn();
        world.insert_static(e, Hp(1));
        let probe = world.get_resource::<Probe>().unwrap();
        assert_eq!(probe.dead_alive, Some(false));
        assert_eq!(probe.self_has_hp, Some(true), "the added value is in place");
        assert_eq!(probe.self_has_mp, Some(false));
    }

    // ── Registration ─────────────────────────────────────────────────

    #[test]
    #[should_panic(expected = "already has an on_add hook")]
    fn a_second_hook_of_the_same_kind_is_refused() {
        let mut world = World::new();
        world
            .register_component_hooks::<Hp>()
            .on_add(probe_hp::<ADD>)
            .on_add(probe_hp::<ADD>);
    }

    #[test]
    fn each_kind_refuses_its_own_duplicate_but_not_another_kind() {
        for kind in 0..5 {
            let result = std::panic::catch_unwind(|| {
                let mut hooks = ComponentHooks::default();
                let f: ComponentHook = probe_hp::<ADD>;
                // Every other kind is fine alongside this one.
                hooks
                    .on_add(f)
                    .on_insert(f)
                    .on_discard(f)
                    .on_remove(f)
                    .on_despawn(f);
                match kind {
                    0 => hooks.on_add(f),
                    1 => hooks.on_insert(f),
                    2 => hooks.on_discard(f),
                    3 => hooks.on_remove(f),
                    _ => hooks.on_despawn(f),
                };
            });
            assert!(result.is_err(), "kind {kind} accepted a duplicate");
        }
    }

    #[test]
    fn hooks_are_per_type_and_a_world_without_any_is_unaffected() {
        let mut world = hooked();
        let e = world.spawn();
        world.insert_static(e, Other(1));
        world.remove_static::<Other>(e);
        assert!(take(&mut world).is_empty(), "Other has no hooks");

        let mut plain = World::new();
        let e = plain.spawn();
        plain.insert_static(e, Hp(1));
        plain.insert(e, Cn(1));
        assert!(plain.remove_static::<Hp>(e).is_some());
        assert!(plain.despawn(e));
        assert!(!plain.hooks.is_active());
    }

    #[test]
    fn a_hook_registered_later_only_sees_later_changes() {
        let mut world = World::new();
        world.insert_resource(Log::default());
        let e = world.spawn();
        world.insert_static(e, Hp(1));
        hook_all::<Hp>(&mut world, all_kinds!(probe_hp));
        assert!(take(&mut world).is_empty(), "nothing fires retroactively");
        world.remove_static::<Hp>(e);
        assert_eq!(take(&mut world).len(), 2);
    }

    // ── From a system, and next to removal tracking ──────────────────

    fn insert_hp_via_commands(mut commands: crate::commands::Commands, target: crate::Res<Target>) {
        commands.entity(target.0).insert_static(Hp(42));
    }
    struct Target(Entity);

    #[test]
    fn commands_from_a_system_fire_hooks_when_they_apply() {
        let mut world = hooked();
        let e = world.spawn();
        world.insert_resource(Target(e));
        world.run_system_once(insert_hp_via_commands);
        assert_eq!(
            short(&take(&mut world)),
            vec![(Add, "Hp", Some(42)), (Insert, "Hp", Some(42))]
        );
    }

    #[derive(Default)]
    struct Seen(Vec<Entity>);
    fn reaper(removed: RemovedComponents<Hp>, mut seen: crate::ResMut<Seen>) {
        seen.0.extend(removed.iter());
    }

    #[test]
    fn hooks_and_removal_tracking_both_see_a_removal() {
        let mut world = hooked();
        world.insert_resource(Seen::default());
        let mut sys = reaper.into_system();
        sys.initialize(&mut world);
        let e = world.spawn();
        world.insert_static(e, Hp(1));
        take(&mut world);
        world.remove_static::<Hp>(e);
        assert_eq!(take(&mut world).len(), 2, "discard and remove");
        sys.run(&mut world);
        assert_eq!(world.get_resource::<Seen>().unwrap().0, vec![e]);
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

    /// The order hooks of one operation must come in. A removal runs
    /// despawn, discard, remove; an addition add, insert; an overwrite
    /// discard then insert: all consistent with this single ranking.
    fn rank(kind: HookKind) -> u8 {
        match kind {
            Despawn => 0,
            Discard => 1,
            Remove => 2,
            Add => 3,
            Insert => 4,
        }
    }

    const NAMES: [&str; 3] = ["Hp", "Mp", "Cn"];

    #[test]
    fn a_random_history_fires_exactly_the_hooks_a_model_predicts() {
        use std::collections::HashMap;
        for seed in 0..8u64 {
            let mut rng = Lcg(seed + 1);
            let mut world = hooked();
            let mut alive: Vec<Entity> = Vec::new();
            // Per entity: holds Hp, Mp, Cn.
            let mut held: HashMap<Entity, [bool; 3]> = HashMap::new();
            let mut exercised = [0usize; 5];

            for op in 1..=300usize {
                world.get_resource_mut::<Log>().unwrap().op = op;
                let mut expected: Vec<(HookKind, &'static str)> = Vec::new();
                let mut target: Option<Entity> = None;
                let picked = if alive.is_empty() {
                    None
                } else {
                    Some(alive[rng.below(alive.len())])
                };
                match (rng.below(12), picked) {
                    (0 | 1, _) => {
                        let e = world.spawn();
                        alive.push(e);
                        held.insert(e, [false; 3]);
                    }
                    (2, Some(e)) => {
                        target = Some(e);
                        world.insert_static(e, Hp(op as u32));
                        let h = held.get_mut(&e).unwrap();
                        if !h[0] {
                            h[0] = true;
                            expected.extend([(Add, "Hp"), (Insert, "Hp")]);
                        }
                    }
                    (3, Some(e)) => {
                        target = Some(e);
                        world.insert_static(e, Mp(op as u32));
                        let h = held.get_mut(&e).unwrap();
                        if !h[1] {
                            h[1] = true;
                            expected.extend([(Add, "Mp"), (Insert, "Mp")]);
                        }
                    }
                    (4, Some(e)) => {
                        target = Some(e);
                        world.insert(e, Cn(op as u32));
                        let h = held.get_mut(&e).unwrap();
                        if h[2] {
                            expected.extend([(Discard, "Cn"), (Insert, "Cn")]);
                        } else {
                            expected.extend([(Add, "Cn"), (Insert, "Cn")]);
                        }
                        h[2] = true;
                    }
                    (5, Some(e)) => {
                        target = Some(e);
                        world.remove_static::<Hp>(e);
                        let h = held.get_mut(&e).unwrap();
                        if h[0] {
                            h[0] = false;
                            expected.extend([(Discard, "Hp"), (Remove, "Hp")]);
                        }
                    }
                    (6, Some(e)) => {
                        target = Some(e);
                        world.remove_static::<Mp>(e);
                        let h = held.get_mut(&e).unwrap();
                        if h[1] {
                            h[1] = false;
                            expected.extend([(Discard, "Mp"), (Remove, "Mp")]);
                        }
                    }
                    (7, Some(e)) => {
                        target = Some(e);
                        world.remove::<Cn>(e);
                        let h = held.get_mut(&e).unwrap();
                        if h[2] {
                            h[2] = false;
                            expected.extend([(Discard, "Cn"), (Remove, "Cn")]);
                        }
                    }
                    (8, Some(e)) => {
                        target = Some(e);
                        let got = world.remove_bundle::<(Hp, Mp)>(e).is_some();
                        let h = held.get_mut(&e).unwrap();
                        assert_eq!(got, h[0] && h[1], "seed {seed} op {op}");
                        if got {
                            h[0] = false;
                            h[1] = false;
                            expected.extend([
                                (Discard, "Hp"),
                                (Discard, "Mp"),
                                (Remove, "Hp"),
                                (Remove, "Mp"),
                            ]);
                        }
                    }
                    (9, Some(e)) => {
                        target = Some(e);
                        let got = world.insert_bundle(e, (Hp(op as u32), Mp(op as u32)));
                        let h = held.get_mut(&e).unwrap();
                        assert_eq!(got, !h[0] && !h[1], "seed {seed} op {op}");
                        if got {
                            h[0] = true;
                            h[1] = true;
                            expected.extend([
                                (Add, "Hp"),
                                (Add, "Mp"),
                                (Insert, "Hp"),
                                (Insert, "Mp"),
                            ]);
                        }
                    }
                    (10, Some(e)) => {
                        target = Some(e);
                        let h = held.remove(&e).unwrap();
                        assert!(world.despawn(e));
                        alive.retain(|x| *x != e);
                        for kind in [Despawn, Discard, Remove] {
                            for (i, name) in NAMES.iter().enumerate() {
                                if h[i] {
                                    expected.push((kind, name));
                                }
                            }
                        }
                    }
                    (_, _) => {
                        let e = world.spawn_bundle((Hp(op as u32), Mp(op as u32)));
                        target = Some(e);
                        alive.push(e);
                        held.insert(e, [true, true, false]);
                        expected.extend([(Add, "Hp"), (Add, "Mp"), (Insert, "Hp"), (Insert, "Mp")]);
                    }
                }

                let events = take(&mut world);
                let mut got: Vec<(u8, &str)> =
                    events.iter().map(|e| (rank(e.kind), e.ty)).collect();
                assert!(
                    got.windows(2).all(|w| w[0].0 <= w[1].0),
                    "seed {seed} op {op}: hooks fired out of order: {events:?}"
                );
                got.sort();
                let mut want: Vec<(u8, &str)> =
                    expected.iter().map(|(k, t)| (rank(*k), *t)).collect();
                want.sort();
                assert_eq!(got, want, "seed {seed} op {op}");
                for ev in &events {
                    exercised[rank(ev.kind) as usize] += 1;
                    assert!(
                        ev.alive && ev.value.is_some(),
                        "seed {seed} op {op}: a hook saw a dead entity or a missing value: {ev:?}"
                    );
                    if let Some(t) = target {
                        assert_eq!(ev.entity, t, "seed {seed} op {op}");
                    }
                }
            }
            assert!(
                exercised.iter().all(|&n| n > 5),
                "seed {seed} must exercise every hook kind: {exercised:?}"
            );
        }
    }
}
