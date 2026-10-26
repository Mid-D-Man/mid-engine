//! The ECS world — entity allocator + component storage.
//!
//! `World::spawn`/`despawn`/`is_alive` are a thin wrapper over
//! `mid_collections::GenerationalIndexAllocator` — see that module's own
//! doc comment for why staleness detection works the way it does
//! (verified against real `slotmap` source there, not assumed), and
//! `docs/mid-collections.md`'s "Generational-index arena" section for
//! why it's ranked directly above the rest of that doc's list for
//! `mid-ecs` specifically.
//!
//! `World::insert`/`get`/`get_mut`/`remove`/`has` are a thin wrapper over
//! `SparseShell` (`crate::component`) — the "Sparse Shell" half of the
//! Hybrid ECS Architecture (`docs/mid-ecs.md`). Any `T: 'static` can be
//! attached to any entity with no upfront declaration; see
//! `component.rs`'s own doc comment for the type-erasure mechanism
//! (grounded in Bevy ECS's real `ComponentId`-based design, not a naive
//! `TypeId`-keyed `HashMap`) and for the specific correctness property
//! `World` has to enforce that `SparseShell` alone can't (checking
//! liveness before every component access, to reject a stale handle that
//! happens to share a reused index with a live entity).
//!
//! `World::insert_static`/`get_static`/`get_static_mut`/`remove_static`/
//! `has_static` are a thin wrapper over `Archetypes`
//! (`crate::archetype`) — the "Archetype Core" half of the Hybrid ECS
//! Architecture, real dynamic table storage with migration between
//! archetypes as components are added/removed. See `archetype.rs`'s own
//! doc comment for the design (grounded in Bevy ECS's real source, read
//! directly) and for why it uses a completely separate `ComponentId`
//! numbering space from `SparseShell`'s own.
//!
//! Iterating this storage (`World::query`/`query2`) lives in `query.rs`,
//! not here — a distinct concern from owning the storage itself.

use std::any::TypeId;
use std::fmt;

use mid_collections::{FfiSpan, GenerationalIndex, GenerationalIndexAllocator, SparseSetIndex};
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

use crate::archetype::{ArchetypeId, Archetypes, Bundle, ComponentFfiError, FfiSpanMut};
use crate::component::{ComponentId, SparseShell};
use crate::hash::TypeIdMap;
use crate::hooks::{
    ComponentHooks, DeferredWorld, HookContext, HookKind, HookRegistry, MAX_HOOK_ROUNDS,
};
use crate::removal::RemovalLog;
use crate::resource::{ResourceFfiError, ResourceId, Resources};
use crate::tick::{ChangeTracker, Tick, CHECK_TICK_THRESHOLD};

/// A handle to an entity. Detects its own staleness after despawn — a
/// thin wrapper over `mid_collections::GenerationalIndex`, not a
/// reimplementation of its mechanism.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Entity(pub(crate) GenerationalIndex);

impl Entity {
    /// Wraps a raw `GenerationalIndex` as an `Entity`. `pub(crate)` and
    /// `#[cfg(test)]` deliberately -- outside this crate, an `Entity`
    /// should only ever come from `World::spawn`, never fabricated; the
    /// only current caller is `component.rs`'s own test module, which
    /// needs real, distinct `Entity` values without a full `World` in
    /// scope. Gated to `test` specifically so it doesn't sit as an
    /// always-there-but-only-used-by-tests function generating a
    /// dead-code warning in ordinary builds.
    #[cfg(test)]
    #[inline]
    pub(crate) fn from_generational_index(index: GenerationalIndex) -> Self {
        Self(index)
    }
}

impl Entity {
    /// The raw slot index. Not meaningful alone — two different
    /// generations of the same slot share this value; use
    /// `World::is_alive` to check validity, not this in isolation.
    /// Exposed mainly for debugging/FFI.
    #[inline]
    pub fn index(&self) -> u32 {
        self.0.index()
    }

    /// The generation this handle was issued with.
    #[inline]
    pub fn generation(&self) -> u32 {
        self.0.generation()
    }

    /// Packs this entity into a single `u64` for handing to non-Rust
    /// code as one opaque, easy-to-pass-by-value handle — see
    /// `mid_collections::GenerationalIndex::as_ffi`'s own doc comment
    /// for the exact bit layout and the reasoning (grounded in
    /// `slotmap::KeyData::as_ffi`'s real design). This is `ffi.rs`'s
    /// whole reason for existing as a *thin* wrapper here: the real
    /// packing logic lives in `mid_collections`, not duplicated.
    #[inline]
    pub fn as_ffi(self) -> u64 {
        self.0.as_ffi()
    }

    /// Reconstructs an `Entity` from a `u64` produced by
    /// [`as_ffi`](Self::as_ffi). Safe even for a bogus `value` that
    /// never actually came from a real `as_ffi()` call — see
    /// `GenerationalIndex::from_ffi`'s own doc comment for exactly why:
    /// every real `World` operation re-validates the handle's
    /// generation against the slot's current one regardless of where it
    /// came from, so a bogus reconstructed `Entity` just reads back as
    /// not alive, never as some other real, live entity.
    #[inline]
    pub fn from_ffi(value: u64) -> Self {
        Self(GenerationalIndex::from_ffi(value))
    }
}

impl SparseSetIndex for Entity {
    #[inline]
    fn sparse_index(&self) -> u32 {
        self.0.sparse_index()
    }
}

impl fmt::Display for Entity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Entity({}v{})", self.0.index(), self.0.generation())
    }
}

/// Which of the two storage systems a component type has been used
/// with — see [`StorageClaims`] for what this actually guards against.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StorageKind {
    Sparse,
    Archetype,
}

impl fmt::Display for StorageKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Sparse => write!(f, "Sparse Shell (insert/register_ffi_component)"),
            Self::Archetype => write!(
                f,
                "Archetype Core (insert_static/register_ffi_static_component)"
            ),
        }
    }
}

/// Tracks which storage system each component type has actually been
/// used with, and panics on the first call that would use a type with
/// *both*.
///
/// **A real, deliberately scoped-down stand-in for a full `Component`
/// trait** (the kind Bevy eventually landed on: `trait Component {
/// const STORAGE: StorageKind; }`, enforced at the type level, one
/// storage strategy fixed per type for good). That real version is a
/// genuine, large undertaking — it would mean unifying `insert`/
/// `insert_static` (and `get`/`get_static`, `remove`/`remove_static`,
/// `has`/`has_static`, `register_ffi_component`/
/// `register_ffi_static_component`) into one set of methods, a real
/// breaking change to every one of them, and retrofitting every
/// existing component type in this whole workspace's tests with a
/// trait impl. Correctly flagged as deferred, not attempted here.
///
/// What *is* small enough to build now: the actual footgun a missing
/// `Component` trait creates isn't really "the API has two names" —
/// it's that nothing stops the *same* `T` from silently ending up in
/// *both* systems for different entities, which fragments every query
/// against it (`World::query::<T>()` only ever sees the Sparse Shell
/// half, `query_static::<T>()` only the Archetype Core half) with no
/// error, just entities that quietly don't show up where a caller
/// would reasonably expect them to. This closes exactly that gap, at
/// runtime, with no change to any existing method's signature: the
/// first system a `T` is ever used with is remembered, and any call
/// through the *other* system for that same `T` panics immediately,
/// with a clear message, rather than silently fragmenting.
///
/// Real, known limitation, not silently swept under: `World::
/// insert_bundle`/`remove_bundle` don't go through this check yet —
/// `Bundle` would need its own `type_ids()`-style method to claim each
/// element, which is a real, contained follow-up, not attempted in
/// this pass to keep this specific change bounded.
#[derive(Debug, Default)]
struct StorageClaims {
    claimed: TypeIdMap<StorageKind>,
}

impl StorageClaims {
    fn claim<T: 'static>(&mut self, kind: StorageKind) {
        let type_id = TypeId::of::<T>();
        match self.claimed.get(&type_id) {
            Some(&existing) if existing != kind => panic!(
                "component type `{}` was already used with {existing} — a component type must use \
                 exactly one storage system for its entire lifetime within one World, never both \
                 (this check exists specifically because using both silently fragments any query \
                 against this type, with no error, rather than failing loudly like this instead)",
                std::any::type_name::<T>()
            ),
            Some(_) => {}
            None => {
                self.claimed.insert(type_id, kind);
            }
        }
    }
}

/// The ECS world. Owns entity lifecycle, the Sparse Shell, and the
/// Archetype Core.
pub struct World {
    pub(crate) entities: GenerationalIndexAllocator,
    pub(crate) components: SparseShell,
    pub(crate) archetypes: Archetypes,
    pub(crate) sync: crate::sync::SyncRegistry,
    pub(crate) resources: Resources,
    /// Opt-in log of component removals (`removal.rs`).
    pub(crate) removals: RemovalLog,
    /// Component hooks and the commands they have queued (`hooks.rs`).
    pub(crate) hooks: HookRegistry,
    storage_claims: StorageClaims,
    change_tick: Tick,
    /// The change tick at which every stored tick was last clamped
    /// ([`Self::check_change_ticks`]).
    last_check_tick: Tick,
}

impl World {
    /// Creates an empty world — no entities spawned yet.
    pub fn new() -> Self {
        Self {
            entities: GenerationalIndexAllocator::new(),
            components: SparseShell::new(),
            archetypes: Archetypes::new(),
            sync: crate::sync::SyncRegistry::new(),
            resources: Resources::new(),
            removals: RemovalLog::default(),
            hooks: HookRegistry::default(),
            storage_claims: StorageClaims::default(),
            change_tick: Tick::new(1),
            last_check_tick: Tick::ZERO,
        }
    }

    /// Creates a world pre-sized for `capacity` entities before the next
    /// spawn past that would reallocate.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            entities: GenerationalIndexAllocator::with_capacity(capacity),
            components: SparseShell::new(),
            archetypes: Archetypes::new(),
            sync: crate::sync::SyncRegistry::new(),
            resources: Resources::new(),
            removals: RemovalLog::default(),
            hooks: HookRegistry::default(),
            storage_claims: StorageClaims::default(),
            change_tick: Tick::new(1),
            last_check_tick: Tick::ZERO,
        }
    }

    /// Spawns a new, live entity — into the empty archetype, with no
    /// components attached yet in either storage system.
    pub fn spawn(&mut self) -> Entity {
        let entity = Entity(self.entities.allocate());
        self.archetypes.spawn(entity);
        entity
    }

    /// Despawns `entity`, if it's still alive. Returns whether it
    /// actually was — despawning an already-dead or never-real handle
    /// is a safe no-op, not a panic, matching
    /// `GenerationalIndexAllocator::deallocate`'s own contract.
    ///
    /// Removes every component attached to `entity` from *both* storage
    /// systems (Sparse Shell and Archetype Core) **before** freeing its
    /// generational slot — that ordering is load-bearing, not
    /// incidental, for both. Neither `SparseSet`-backed storage looks up
    /// by anything but raw index, not generation (see `component.rs`'s
    /// and `archetype.rs`'s own doc comments), so freeing the slot first
    /// and cleaning up after would leave a window where a
    /// freshly-reused index's new entity could read the dead entity's
    /// stale leftover data from either system.
    pub fn despawn(&mut self, entity: Entity) -> bool {
        if !self.is_alive(entity) {
            return false;
        }
        let hooked = self.hooks.is_active();
        if hooked {
            let types = self.entity_type_ids(entity);
            if !types.is_empty() {
                self.fire_removed(entity, &types, true);
            }
        }
        if self.removals.is_active() {
            self.record_despawn_removals(entity);
        } else {
            self.components.remove_entity_from_all(entity);
        }
        self.archetypes.despawn(entity);
        let despawned = self.entities.deallocate(entity.0);
        if hooked {
            self.flush_hook_commands();
        }
        despawned
    }

    /// The despawn of an entity, when removal tracking is on: removes its
    /// Sparse Shell components, and records every component it held in
    /// either storage as removed (the archetype ones are read off its
    /// signature before the archetype despawn takes them).
    fn record_despawn_removals(&mut self, entity: Entity) {
        let tick = self.change_tick;
        for &id in self.archetypes.components_of(entity) {
            if let Some(type_id) = self.archetypes.type_id_of(id) {
                self.removals.record(type_id, entity, tick);
            }
        }
        let mut sparse = Vec::new();
        self.components
            .remove_entity_from_all_collecting(entity, &mut sparse);
        for type_id in sparse {
            self.removals.record(type_id, entity, tick);
        }
    }

    /// Whether `entity` is still alive.
    #[inline]
    pub fn is_alive(&self, entity: Entity) -> bool {
        self.entities.is_alive(entity.0)
    }

    /// Number of currently-live entities.
    #[inline]
    pub fn entity_count(&self) -> usize {
        self.entities.len()
    }

    /// True if nothing is currently spawned.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.entities.is_empty()
    }

    /// Attaches `component` to `entity`. A safe no-op (returns `None`,
    /// nothing is stored) if `entity` isn't alive — checked explicitly,
    /// not left to chance, because `SparseSet`'s own index-only lookup
    /// can't tell a stale handle from a live one sharing the same raw
    /// index (see `component.rs`'s doc comment). No panic in any build
    /// configuration, matching every other fallible operation in this
    /// codebase (`SparseSet::remove`, `GenerationalIndexAllocator::
    /// deallocate`, ...) — a `debug_assert!` was tried here first and
    /// removed: it directly contradicted that established convention,
    /// and the test written for the safe-fallback case immediately
    /// caught the contradiction by panicking in the very test meant to
    /// prove it doesn't.
    pub fn insert<T: 'static>(&mut self, entity: Entity, component: T) -> Option<T> {
        if !self.is_alive(entity) {
            return None;
        }
        self.storage_claims.claim::<T>(StorageKind::Sparse);
        let hooked = self.hooks.is_active();
        let type_id = TypeId::of::<T>();
        if hooked && self.components.get::<T>(entity).is_some() {
            self.run_hook(HookKind::Discard, type_id, entity);
        }
        let old = self.components.insert(entity, component);
        if hooked {
            if old.is_none() {
                self.run_hook(HookKind::Add, type_id, entity);
            }
            self.run_hook(HookKind::Insert, type_id, entity);
            self.flush_hook_commands();
        }
        old
    }

    /// Looks up `entity`'s `T` component, if attached and `entity` is
    /// still alive.
    pub fn get<T: 'static>(&self, entity: Entity) -> Option<&T> {
        if !self.is_alive(entity) {
            return None;
        }
        self.components.get(entity)
    }

    /// Looks up `entity`'s `T` component mutably, if attached and
    /// `entity` is still alive.
    pub fn get_mut<T: 'static>(&mut self, entity: Entity) -> Option<&mut T> {
        if !self.is_alive(entity) {
            return None;
        }
        self.components.get_mut(entity)
    }

    /// Removes and returns `entity`'s `T` component, if attached and
    /// `entity` is still alive.
    pub fn remove<T: 'static>(&mut self, entity: Entity) -> Option<T> {
        if !self.is_alive(entity) {
            return None;
        }
        let hooked = self.hooks.is_active();
        if hooked && self.components.get::<T>(entity).is_some() {
            self.fire_removed(entity, &[TypeId::of::<T>()], false);
        }
        let removed = self.components.remove(entity);
        if removed.is_some() && self.removals.is_active() {
            self.removals
                .record(TypeId::of::<T>(), entity, self.change_tick);
        }
        if hooked {
            self.flush_hook_commands();
        }
        removed
    }

    /// Whether `entity` is alive and currently has a `T` component
    /// attached.
    #[inline]
    pub fn has<T: 'static>(&self, entity: Entity) -> bool {
        self.is_alive(entity) && self.components.has::<T>(entity)
    }

    /// Opts `T` into FFI span exposure under `name`, for the Sparse
    /// Shell — thin wrapper over `SparseShell::register_ffi`, which
    /// carries the full design reasoning (why this is generic and has
    /// to be called from Rust, why the extra `IntoBytes`/`Immutable`/
    /// `KnownLayout` bounds don't apply to every component type in the
    /// shell, just this one).
    pub fn register_ffi_component<T>(&mut self, name: &'static str) -> ComponentId
    where
        T: 'static + IntoBytes + Immutable + KnownLayout,
    {
        self.storage_claims.claim::<T>(StorageKind::Sparse);
        self.components.register_ffi::<T>(name)
    }

    /// Non-generic, `ComponentId`-keyed raw span over every currently-
    /// attached instance of a Sparse-Shell component type — thin
    /// wrapper over `SparseShell::raw_span`. `None` if `id` doesn't
    /// exist or was never opted in via
    /// [`Self::register_ffi_component`]. Pair with
    /// [`Self::component_entity_ids`] for which `Entity` owns each
    /// element of the span this returns. See
    /// `mid_collections::FfiSpan`'s own doc comment for the
    /// invalidation contract the caller has to honor.
    pub fn component_raw_span(&self, id: ComponentId) -> Option<FfiSpan> {
        self.components.raw_span(id)
    }

    /// Entity-correlation counterpart to [`Self::component_raw_span`] —
    /// thin wrapper over `SparseShell::entity_ids`. `component_entity_ids(id)[i]`
    /// is the `Entity` that owns `component_raw_span(id)`'s element
    /// `i`, for every valid `i`.
    pub fn component_entity_ids(&self, id: ComponentId) -> Option<Vec<u64>> {
        self.components.entity_ids(id)
    }

    /// Looks up the `ComponentId` a Sparse-Shell type was registered
    /// under via [`Self::register_ffi_component`], by name — thin
    /// wrapper over `SparseShell::lookup_ffi_id`. This is how a C
    /// caller, which has no way to name a Rust type directly, actually
    /// obtains a `ComponentId` at all.
    pub fn lookup_ffi_component_id(&self, name: &str) -> Option<ComponentId> {
        self.components.lookup_ffi_id(name)
    }

    /// Attaches `component` to `entity` as an **archetype-tracked**
    /// component — migrating `entity`'s row into whichever archetype
    /// matches its new, larger component set (see `archetype.rs`'s doc
    /// comment for what "migrating" actually does). A safe no-op
    /// (returns `false`) if `entity` isn't alive, or already has a `T`
    /// attached this way — the same liveness-check requirement
    /// `insert`/`get`/etc. have for the Sparse Shell applies here too,
    /// for the identical reason (index-only lookup, can't tell a stale
    /// handle from a live one sharing a reused index on its own).
    ///
    /// Distinct method name from `insert` deliberately — a given `T`
    /// should live in exactly one of the two storage systems, never
    /// both, and there's no enforcement of that yet beyond the caller
    /// being consistent about which method they call for a given type.
    /// A `Component` trait fixing each type's storage strategy once
    /// (matching where Bevy eventually landed) is a real future
    /// refinement, not needed to get real, correct behavior today.
    pub fn insert_static<T: 'static>(&mut self, entity: Entity, component: T) -> bool {
        if !self.is_alive(entity) {
            return false;
        }
        self.storage_claims.claim::<T>(StorageKind::Archetype);
        let id = self.archetypes.component_id::<T>();
        let inserted = self
            .archetypes
            .insert(entity, id, component, self.change_tick);
        if inserted && self.hooks.is_active() {
            self.fire_added(entity, &[TypeId::of::<T>()]);
            self.flush_hook_commands();
        }
        inserted
    }

    /// Inserts every element of `bundle` as one atomic structural
    /// change — one migration total, not one per element. `B` is a
    /// tuple of archetype-tracked component types, e.g.
    /// `world.insert_bundle(e, (Position { .. }, Velocity { .. }))`.
    /// Returns `false` (no-op) if `entity` isn't alive, or if it
    /// already has *any* of `B`'s component types — see
    /// `Archetypes::insert_bundle`'s own doc comment for the full
    /// "no partial application" reasoning. The real point of this over
    /// calling `insert_static` once per element: spawning many
    /// entities with the same known-at-creation-time component set
    /// (the common real case) does one migration instead of `N`,
    /// `N - 1` of which would otherwise be through intermediate
    /// archetypes nothing ever actually queries against.
    ///
    /// `B: Bundle` is a real, deliberate `private_bounds` case, not an
    /// oversight — see `Bundle`'s own doc comment in `archetype.rs` for
    /// why: keeping `Table`/`Archetypes` `pub(crate)` (architecturally
    /// important) is worth more than letting downstream code write its
    /// own function generic over "any bundle" (a narrow capability;
    /// calling this method directly with a concrete tuple works fine
    /// either way).
    /// Spawns a brand-new entity already holding every component in
    /// `bundle`, in one step -- no separate `insert_bundle` call, and
    /// no stop in the empty archetype in between. The direct
    /// counterpart to bevy's own `World::spawn(bundle)`; prefer this
    /// over `spawn()` + `insert_bundle()` whenever the bundle is known
    /// up front (see `docs/mid-ecs.md`, "Direct bundle spawn", for what
    /// the two-call path costs that this avoids).
    #[allow(private_bounds)]
    pub fn spawn_bundle<B: Bundle>(&mut self, bundle: B) -> Entity {
        let entity = Entity(self.entities.allocate());
        self.archetypes
            .spawn_bundle(entity, bundle, self.change_tick);
        if self.hooks.is_active() {
            let types = self.bundle_type_ids::<B>();
            self.fire_added(entity, &types);
            self.flush_hook_commands();
        }
        entity
    }

    #[allow(private_bounds)]
    pub fn insert_bundle<B: Bundle>(&mut self, entity: Entity, bundle: B) -> bool {
        if !self.is_alive(entity) {
            return false;
        }
        let inserted = self
            .archetypes
            .insert_bundle(entity, bundle, self.change_tick);
        if inserted && self.hooks.is_active() {
            let types = self.bundle_type_ids::<B>();
            self.fire_added(entity, &types);
            self.flush_hook_commands();
        }
        inserted
    }

    /// Removes and returns `entity`'s archetype-tracked `T` component,
    /// if attached and `entity` is alive — migrating its row back into
    /// the archetype matching its new, smaller component set.
    pub fn remove_static<T: 'static>(&mut self, entity: Entity) -> Option<T> {
        if !self.is_alive(entity) {
            return None;
        }
        let id = self.archetypes.existing_component_id::<T>()?;
        let hooked = self.hooks.is_active();
        if hooked && self.archetypes.components_of(entity).contains(&id) {
            self.fire_removed(entity, &[TypeId::of::<T>()], false);
        }
        let removed = self.archetypes.remove(entity, id);
        if removed.is_some() && self.removals.is_active() {
            self.removals
                .record(TypeId::of::<T>(), entity, self.change_tick);
        }
        if hooked {
            self.flush_hook_commands();
        }
        removed
    }

    /// Removes every element of `B` as one atomic structural change,
    /// returning them reassembled as `B` — one migration total, not
    /// one per element. Returns `None` if `entity` isn't alive, if
    /// *any* element of `B` was never registered as an
    /// archetype-tracked component for anything, or if `entity` is
    /// missing *any* of `B`'s component types — see
    /// `Archetypes::remove_bundle`'s own doc comment for the full
    /// "no partial application" reasoning.
    ///
    /// Same deliberate `private_bounds` tradeoff as `insert_bundle` —
    /// see that method's own doc comment.
    #[allow(private_bounds)]
    pub fn remove_bundle<B: Bundle>(&mut self, entity: Entity) -> Option<B> {
        if !self.is_alive(entity) {
            return None;
        }
        let hooked = self.hooks.is_active();
        let ids = if hooked || self.removals.is_active() {
            B::existing_component_ids(&self.archetypes)
        } else {
            None
        };
        if hooked {
            if let Some(ids) = &ids {
                // The same condition `Archetypes::remove_bundle` removes
                // under: every member registered and present.
                let have = self.archetypes.components_of(entity);
                if ids.iter().all(|id| have.contains(id)) {
                    let types: Vec<TypeId> = ids
                        .iter()
                        .filter_map(|&id| self.archetypes.type_id_of(id))
                        .collect();
                    self.fire_removed(entity, &types, false);
                }
            }
        }
        let removed = self.archetypes.remove_bundle(entity);
        if removed.is_some() && self.removals.is_active() {
            if let Some(ids) = ids {
                for &id in ids.iter() {
                    if let Some(type_id) = self.archetypes.type_id_of(id) {
                        self.removals.record(type_id, entity, self.change_tick);
                    }
                }
            }
        }
        if hooked {
            self.flush_hook_commands();
        }
        removed
    }

    /// Looks up `entity`'s archetype-tracked `T` component, if attached
    /// and `entity` is alive.
    pub fn get_static<T: 'static>(&self, entity: Entity) -> Option<&T> {
        if !self.is_alive(entity) {
            return None;
        }
        let id = self.archetypes.existing_component_id::<T>()?;
        self.archetypes.get(entity, id)
    }

    /// Looks up `entity`'s archetype-tracked `T` component mutably, if
    /// attached and `entity` is alive.
    pub fn get_static_mut<T: 'static>(&mut self, entity: Entity) -> Option<&mut T> {
        if !self.is_alive(entity) {
            return None;
        }
        let id = self.archetypes.existing_component_id::<T>()?;
        self.archetypes.get_mut(entity, id, self.change_tick)
    }

    /// The world's current point on its change-tick counter. Every
    /// archetype-tracked component records this at insertion, and
    /// [`Self::get_static_mut`] records it again on every mutable
    /// access. The bulk mutable queries ([`Self::query_static_mut`] and
    /// friends) record it per row actually written through — see
    /// `tick.rs`.
    pub fn change_tick(&self) -> Tick {
        self.change_tick
    }

    /// Advances the world's change-tick counter by one and returns the
    /// new value. Call once per step of the caller's own game loop —
    /// there is no `Schedule` here to do this automatically. Until this
    /// is called at least once, every [`crate::ChangeTracker`] reads
    /// every component currently present as added/changed (see
    /// [`crate::ChangeTracker::new`]).
    ///
    /// Every [`CHECK_TICK_THRESHOLD`] ticks this also runs
    /// [`Self::check_change_ticks`], so the world's own stored ticks never
    /// get old enough to wrap.
    pub fn increment_change_tick(&mut self) -> Tick {
        self.change_tick = Tick::new(self.change_tick.get().wrapping_add(1));
        if self.change_tick.relative_to(self.last_check_tick).get() >= CHECK_TICK_THRESHOLD {
            self.check_change_ticks();
        }
        self.change_tick
    }

    /// Keeps every tick the world stores (the inserted and changed tick of
    /// each component value, each resource's, each entry of the removal
    /// log) within [`crate::MAX_CHANGE_AGE`] of the current tick, clamping
    /// anything older to exactly that age. A tick is a wrapping `u32`;
    /// without this a value that went untouched for 2^32 ticks would look
    /// brand new (see `tick.rs`). [`Self::increment_change_tick`] calls it
    /// every [`CHECK_TICK_THRESHOLD`] ticks, so there is rarely a reason to
    /// call it directly. It does not reach ticks held outside the world:
    /// see [`crate::System::check_change_tick`] and
    /// [`ChangeTracker::check_tick`].
    pub fn check_change_ticks(&mut self) {
        let now = self.change_tick;
        self.archetypes.check_ticks(now);
        self.resources.check_ticks(now);
        self.removals.check_ticks(now);
        self.last_check_tick = now;
    }

    /// The change tick at which [`Self::check_change_ticks`] last ran
    /// (zero before the first). Whoever holds ticks outside the world can
    /// re-check them when this changes, or on their own cadence.
    pub fn last_check_tick(&self) -> Tick {
        self.last_check_tick
    }

    /// Jumps the change tick to `tick`, for tests of wraparound that
    /// cannot afford 2^32 increments.
    #[cfg(test)]
    pub(crate) fn set_change_tick(&mut self, tick: Tick) {
        self.change_tick = tick;
    }

    // ── Component hooks (`hooks.rs`) ─────────────────────────────────

    /// The hooks of component type `T`, to set: `world
    /// .register_component_hooks::<T>().on_add(f).on_remove(g)`. See
    /// [`ComponentHooks`] for what each fires on and in what order. Takes
    /// effect for changes made after the call; nothing fires
    /// retroactively. Works for a type in either storage.
    pub fn register_component_hooks<T: 'static>(&mut self) -> &mut ComponentHooks {
        self.hooks.entry(TypeId::of::<T>())
    }

    /// Runs `type_id`'s `kind` hook for `entity`, if it has one.
    fn run_hook(&mut self, kind: HookKind, type_id: TypeId, entity: Entity) {
        if let Some(hook) = self.hooks.get(type_id).and_then(|h| h.of(kind)) {
            hook(
                &mut DeferredWorld::new(self),
                HookContext { entity, type_id },
            );
        }
    }

    /// The hooks for components just added: every `on_add`, then every
    /// `on_insert`.
    fn fire_added(&mut self, entity: Entity, types: &[TypeId]) {
        for &type_id in types {
            self.run_hook(HookKind::Add, type_id, entity);
        }
        for &type_id in types {
            self.run_hook(HookKind::Insert, type_id, entity);
        }
    }

    /// The hooks for components about to be removed (still in place):
    /// for a despawn every `on_despawn`, then every `on_discard`, then
    /// every `on_remove`.
    fn fire_removed(&mut self, entity: Entity, types: &[TypeId], despawn: bool) {
        if despawn {
            for &type_id in types {
                self.run_hook(HookKind::Despawn, type_id, entity);
            }
        }
        for &type_id in types {
            self.run_hook(HookKind::Discard, type_id, entity);
        }
        for &type_id in types {
            self.run_hook(HookKind::Remove, type_id, entity);
        }
    }

    /// The types behind `ids` (archetype `ComponentId`s).
    fn types_of_ids(&self, ids: &[ComponentId]) -> Vec<TypeId> {
        ids.iter()
            .filter_map(|&id| self.archetypes.type_id_of(id))
            .collect()
    }

    /// The types of the members of `B`, once they are registered.
    #[allow(private_bounds)]
    fn bundle_type_ids<B: Bundle>(&self) -> Vec<TypeId> {
        B::existing_component_ids(&self.archetypes)
            .map_or_else(Vec::new, |ids| self.types_of_ids(&ids))
    }

    /// Every component type `entity` has, in either storage.
    fn entity_type_ids(&self, entity: Entity) -> Vec<TypeId> {
        let mut types = self.types_of_ids(self.archetypes.components_of(entity));
        self.components.type_ids_of(entity, &mut types);
        types
    }

    /// Applies the commands hooks have queued, round after round (their
    /// own hooks may queue more), once the operation that fired the hooks
    /// is done. A nested operation inside that loop leaves it to the loop.
    ///
    /// # Panics
    /// After `MAX_HOOK_ROUNDS` rounds without the queue emptying: a hook
    /// that always queues a structural change whose own hook queues
    /// another. The remaining commands are dropped first.
    pub(crate) fn flush_hook_commands(&mut self) {
        if self.hooks.flushing || self.hooks.pending.is_empty() {
            return;
        }
        self.hooks.flushing = true;
        let mut rounds = 0usize;
        while !self.hooks.pending.is_empty() {
            rounds += 1;
            if rounds > MAX_HOOK_ROUNDS {
                self.hooks.flushing = false;
                self.hooks.pending = Default::default();
                panic!(
                    "commands queued by component hooks did not settle after {MAX_HOOK_ROUNDS} rounds: \
                     a hook that always queues a structural change whose own hook queues another?"
                );
            }
            let mut queue = std::mem::take(&mut self.hooks.pending);
            queue.apply(self);
        }
        self.hooks.flushing = false;
    }

    // ── Removal tracking (`removal.rs`) ──────────────────────────────

    /// Starts recording removals of component type `T`: from now on,
    /// every time a `T` is removed from an entity (by `remove` /
    /// `remove_static` / a bundle / the FFI byte paths) or goes with its
    /// entity's despawn, the entity and the world's change tick are
    /// logged. Idempotent. Removals before the first call are not
    /// recorded; a `RemovedComponents<T>` system parameter calls this
    /// when its system is set up.
    ///
    /// Works for a type in either storage. Replacing a value by inserting
    /// over it is not a removal.
    pub fn track_removals<T: 'static>(&mut self) {
        self.removals.track(TypeId::of::<T>());
    }

    /// Whether removals of `T` are being recorded.
    pub fn tracks_removals<T: 'static>(&self) -> bool {
        self.removals.is_tracked(TypeId::of::<T>())
    }

    /// The entities that lost a `T` after `tracker`'s last update, oldest
    /// first: an entity removed twice appears twice, and an entity may
    /// since have been despawned or its slot reused, so check
    /// [`Self::is_alive`] if it matters. The counterpart of
    /// `query_changed` for removals.
    ///
    /// # Panics
    /// If `T` is not tracked ([`Self::track_removals`]): nothing would
    /// ever have been recorded, and an empty answer would hide that.
    pub fn removed_since<T: 'static>(
        &self,
        tracker: &ChangeTracker,
    ) -> impl Iterator<Item = Entity> + '_ {
        let type_id = TypeId::of::<T>();
        assert!(
            self.removals.is_tracked(type_id),
            "removals of `{}` are not tracked: call `World::track_removals::<{}>()` first",
            std::any::type_name::<T>(),
            std::any::type_name::<T>(),
        );
        self.removals
            .since(type_id, tracker.last_run(), self.change_tick)
    }

    /// How many removals of `T` the log currently holds (0 if untracked).
    pub fn removal_count<T: 'static>(&self) -> usize {
        self.removals.len(TypeId::of::<T>())
    }

    /// Drops every logged removal that a reader whose last run was
    /// `at_or_before` (or later) could no longer see anyway. The log is
    /// never trimmed on its own, because only whoever owns *all* the
    /// readers knows the oldest `last_run` among them: pass that. Without
    /// trimming, the log of a type removed often grows without bound.
    pub fn trim_removals(&mut self, at_or_before: Tick) {
        self.removals.trim(at_or_before, self.change_tick);
    }

    /// Drops every logged removal. Tracked types stay tracked.
    pub fn clear_removals(&mut self) {
        self.removals.clear();
    }

    /// Whether `entity` is alive and currently has an archetype-tracked
    /// `T` component attached.
    #[inline]
    pub fn has_static<T: 'static>(&self, entity: Entity) -> bool {
        let Some(id) = self.archetypes.existing_component_id::<T>() else {
            return false;
        };
        self.is_alive(entity) && self.archetypes.has(entity, id)
    }

    /// Opts `T` into FFI span exposure under `name`, for the Archetype
    /// Core — thin wrapper over `Archetypes::register_ffi`. A *separate*
    /// name space from [`Self::register_ffi_component`]'s own (Sparse
    /// Shell), matching `Archetypes`' already-separate `ComponentId`
    /// numbering space — the same `name` can resolve to a different
    /// `ComponentId` in each system, by design.
    pub fn register_ffi_static_component<T>(&mut self, name: &'static str) -> ComponentId
    where
        T: 'static + IntoBytes + Immutable + KnownLayout,
    {
        self.storage_claims.claim::<T>(StorageKind::Archetype);
        self.archetypes.register_ffi::<T>(name)
    }

    /// Non-generic, per-archetype raw span over `component_id`'s column
    /// within `archetype_id`'s table — thin wrapper over
    /// `Archetypes::raw_span`. See that method's own doc comment for
    /// the real fragmentation this has that
    /// [`Self::component_raw_span`] (Sparse Shell) doesn't: one
    /// component type's data isn't one stable thing here, it's spread
    /// across every archetype currently containing it. Pair with
    /// [`Self::archetypes_with_static_component`] to enumerate them,
    /// and with [`Self::static_component_entity_ids`] for which
    /// `Entity` owns each element of the span this returns.
    pub fn static_component_raw_span(
        &self,
        archetype_id: ArchetypeId,
        component_id: ComponentId,
    ) -> Option<FfiSpan> {
        self.archetypes.raw_span(archetype_id, component_id)
    }

    /// [`Self::register_ffi_static_component`], and additionally lets C
    /// *write* the type: [`Self::static_component_raw_span_mut`] and
    /// [`Self::write_static_component_bytes`] work on it. `T: FromBytes`
    /// is the extra bound: C writes raw bytes, and `FromBytes` is the
    /// proof that every bit pattern is a valid `T`, so no C write can
    /// create an invalid Rust value. Idempotent. Registering read-only
    /// first and writable later upgrades the type; registering it
    /// read-only after it was writable does not downgrade it.
    ///
    /// # Panics
    /// If `name` was already registered for a different component type.
    pub fn register_ffi_static_component_writable<T>(&mut self, name: &'static str) -> ComponentId
    where
        T: 'static + FromBytes + IntoBytes + Immutable + KnownLayout,
    {
        self.storage_claims.claim::<T>(StorageKind::Archetype);
        self.archetypes.register_ffi_writable::<T>(name)
    }

    /// A *writable* span over `component_id`'s column in `archetype_id`'s
    /// table: [`Self::static_component_raw_span`]'s mutable counterpart,
    /// for components registered with
    /// [`Self::register_ffi_static_component_writable`].
    ///
    /// Every row of that column is stamped changed at the world's current
    /// tick *when the span is handed out* (not added: the values already
    /// existed), the same moment [`Self::get_static_mut`] stamps. C cannot
    /// say which rows it wrote through a raw pointer, so this can
    /// over-report, and it can never miss a write. For a few rows, use
    /// [`Self::write_static_component_bytes`], which stamps exactly the
    /// row it writes.
    ///
    /// Errors: [`ComponentFfiError::UnknownId`] under the conditions
    /// `static_component_raw_span` answers `None`;
    /// [`ComponentFfiError::NotWritable`] for a component registered
    /// read-only. Nothing is stamped on an error.
    pub fn static_component_raw_span_mut(
        &mut self,
        archetype_id: ArchetypeId,
        component_id: ComponentId,
    ) -> Result<FfiSpanMut, ComponentFfiError> {
        self.archetypes
            .raw_span_mut(archetype_id, component_id, self.change_tick)
    }

    /// The byte size every write or insert of `component_id` must have:
    /// the registered type's size, the `stride` of its spans. For a
    /// component registered with
    /// [`Self::register_ffi_static_component_writable`] only:
    /// [`ComponentFfiError::UnknownId`] if it was never registered for
    /// FFI, [`ComponentFfiError::NotWritable`] if it was registered
    /// read-only.
    pub fn static_component_write_size(
        &self,
        component_id: ComponentId,
    ) -> Result<usize, ComponentFfiError> {
        self.archetypes.write_size(component_id)
    }

    /// Attaches a `component_id` component, built from `bytes`, to
    /// `entity`: the byte-driven counterpart of [`Self::insert_static`],
    /// for components registered with
    /// [`Self::register_ffi_static_component_writable`] (the `FromBytes`
    /// bound is what makes building a value from arbitrary bytes sound).
    /// The entity migrates into the archetype for its larger component
    /// set, and the new value is stamped added and changed at the current
    /// tick. `bytes` is copied, needs no alignment, and its length must be
    /// exactly [`Self::static_component_write_size`].
    ///
    /// Unlike `insert_static`, which is a silent no-op when the entity
    /// already has the component, this reports it as
    /// [`ComponentFfiError::AlreadyPresent`] (nothing is written; use
    /// [`Self::write_static_component_bytes`] to overwrite). Errors,
    /// checked in this order: [`ComponentFfiError::NotAlive`],
    /// [`ComponentFfiError::UnknownId`], [`ComponentFfiError::NotWritable`],
    /// [`ComponentFfiError::SizeMismatch`], `AlreadyPresent`. A structural
    /// change: it invalidates every span previously handed out for the
    /// entity's old and new archetype.
    pub fn insert_static_bytes(
        &mut self,
        entity: Entity,
        component_id: ComponentId,
        bytes: &[u8],
    ) -> Result<(), ComponentFfiError> {
        if !self.is_alive(entity) {
            return Err(ComponentFfiError::NotAlive);
        }
        let result =
            self.archetypes
                .insert_row_bytes(entity, component_id, bytes, self.change_tick);
        if result.is_ok() && self.hooks.is_active() {
            if let Some(type_id) = self.archetypes.type_id_of(component_id) {
                self.fire_added(entity, &[type_id]);
            }
            self.flush_hook_commands();
        }
        result
    }

    /// Detaches `component_id` from `entity` and drops its value: the
    /// id-driven counterpart of [`Self::remove_static`], which needs the
    /// type, for components registered with
    /// [`Self::register_ffi_static_component_writable`] (so "writable"
    /// means C may write, insert and remove it). The entity migrates into
    /// the archetype for its smaller component set; its other values and
    /// their ticks come across untouched. Nothing is stamped: there is no
    /// change notification for removals.
    ///
    /// Errors, checked in this order: [`ComponentFfiError::NotAlive`],
    /// [`ComponentFfiError::UnknownId`], [`ComponentFfiError::NotWritable`],
    /// [`ComponentFfiError::Absent`] (alive, but without this component).
    /// Nothing changes on an error. A structural change: it invalidates
    /// every span previously handed out for the entity's old and new
    /// archetype.
    pub fn remove_static_by_id(
        &mut self,
        entity: Entity,
        component_id: ComponentId,
    ) -> Result<(), ComponentFfiError> {
        if !self.is_alive(entity) {
            return Err(ComponentFfiError::NotAlive);
        }
        let hooked = self.hooks.is_active();
        if hooked && self.archetypes.would_remove_rows(entity, &[component_id]) {
            let types = self.types_of_ids(&[component_id]);
            self.fire_removed(entity, &types, false);
        }
        let result = self.archetypes.remove_row(entity, component_id);
        if result.is_ok() && self.removals.is_active() {
            if let Some(type_id) = self.archetypes.type_id_of(component_id) {
                self.removals.record(type_id, entity, self.change_tick);
            }
        }
        if hooked {
            self.flush_hook_commands();
        }
        result
    }

    /// Checks, changing nothing, a list of components an id-driven bundle
    /// call would name: each `(component, len)` must be registered
    /// writable and, when `len` is `Some`, `len` must be its size; they are
    /// checked in list order, and then no component may repeat. The
    /// checks [`Self::insert_static_bytes_many`] and
    /// [`Self::remove_static_by_ids`] make before they look at an entity,
    /// exposed so a caller with no entity yet (a command queue) can raise
    /// the same errors up front. [`ComponentFfiError::UnknownId`],
    /// [`ComponentFfiError::NotWritable`], [`ComponentFfiError::SizeMismatch`],
    /// [`ComponentFfiError::Duplicate`], in that order of precedence.
    pub fn check_static_component_list(
        &self,
        entries: &[(ComponentId, Option<usize>)],
    ) -> Result<(), ComponentFfiError> {
        self.archetypes.check_list(entries.iter().copied())
    }

    /// Attaches several components, each built from its bytes, to `entity`
    /// in *one* structural change: the byte-driven counterpart of
    /// [`Self::insert_bundle`], for components registered with
    /// [`Self::register_ffi_static_component_writable`]. Every new value
    /// is stamped added and changed at the current tick; the entity's
    /// existing values come across untouched.
    ///
    /// All or nothing. Errors, checked in this order:
    /// [`ComponentFfiError::NotAlive`]; then each listed component in
    /// order ([`ComponentFfiError::UnknownId`],
    /// [`ComponentFfiError::NotWritable`], [`ComponentFfiError::SizeMismatch`]);
    /// then [`ComponentFfiError::Duplicate`] if a component is listed
    /// twice; then [`ComponentFfiError::AlreadyPresent`] if the entity
    /// already has *any* of them. Nothing is inserted on an error. An
    /// empty list is `Ok` and changes nothing. A structural change: it
    /// invalidates every span previously handed out for the entity's old
    /// and new archetype.
    pub fn insert_static_bytes_many(
        &mut self,
        entity: Entity,
        entries: &[(ComponentId, &[u8])],
    ) -> Result<(), ComponentFfiError> {
        if !self.is_alive(entity) {
            return Err(ComponentFfiError::NotAlive);
        }
        let result = self
            .archetypes
            .insert_many_bytes(entity, entries, self.change_tick);
        if result.is_ok() && !entries.is_empty() && self.hooks.is_active() {
            let ids: Vec<ComponentId> = entries.iter().map(|(id, _)| *id).collect();
            let types = self.types_of_ids(&ids);
            self.fire_added(entity, &types);
            self.flush_hook_commands();
        }
        result
    }

    /// Detaches several components from `entity` in *one* structural change
    /// and drops their values: the id-driven counterpart of
    /// [`Self::remove_bundle`], for components registered with
    /// [`Self::register_ffi_static_component_writable`]. The entity's other
    /// values and their ticks come across untouched; nothing is stamped.
    ///
    /// All or nothing. Errors, in order: [`ComponentFfiError::NotAlive`];
    /// each listed component in order ([`ComponentFfiError::UnknownId`],
    /// [`ComponentFfiError::NotWritable`]); [`ComponentFfiError::Duplicate`];
    /// then [`ComponentFfiError::Absent`] if the entity lacks *any* of them.
    /// Nothing is removed on an error. An empty list is `Ok` and changes
    /// nothing.
    pub fn remove_static_by_ids(
        &mut self,
        entity: Entity,
        ids: &[ComponentId],
    ) -> Result<(), ComponentFfiError> {
        if !self.is_alive(entity) {
            return Err(ComponentFfiError::NotAlive);
        }
        let hooked = self.hooks.is_active();
        if hooked && self.archetypes.would_remove_rows(entity, ids) {
            let types = self.types_of_ids(ids);
            self.fire_removed(entity, &types, false);
        }
        let result = self.archetypes.remove_many_rows(entity, ids);
        if result.is_ok() && self.removals.is_active() {
            for &id in ids {
                if let Some(type_id) = self.archetypes.type_id_of(id) {
                    self.removals.record(type_id, entity, self.change_tick);
                }
            }
        }
        if hooked {
            self.flush_hook_commands();
        }
        result
    }

    /// Copies `bytes` in as `entity`'s `component_id` value and stamps
    /// exactly that row changed (not added) at the current tick: the
    /// precise counterpart of [`Self::static_component_raw_span_mut`].
    /// `bytes` is copied, so it needs no alignment and is not retained;
    /// its length must be exactly the registered type's size.
    ///
    /// Errors, checked in this order: [`ComponentFfiError::NotAlive`],
    /// [`ComponentFfiError::UnknownId`] (never registered),
    /// [`ComponentFfiError::NotWritable`], [`ComponentFfiError::SizeMismatch`],
    /// [`ComponentFfiError::Absent`] (alive, but without this component).
    /// Nothing is written or stamped on an error.
    pub fn write_static_component_bytes(
        &mut self,
        entity: Entity,
        component_id: ComponentId,
        bytes: &[u8],
    ) -> Result<(), ComponentFfiError> {
        if !self.is_alive(entity) {
            return Err(ComponentFfiError::NotAlive);
        }
        self.archetypes
            .write_row_bytes(entity, component_id, bytes, self.change_tick)
    }

    /// A one-element writable span over `entity`'s `component_id` value,
    /// with exactly that row stamped changed at the current tick: the
    /// in-place form of [`Self::write_static_component_bytes`], and the
    /// FFI counterpart of [`Self::get_static_mut`] (stamped when the span
    /// is handed out, not when it is written, and `changed`, not `added`).
    /// Together with the row list [`Self::static_component_changed_rows`]
    /// and friends produce, it is the precise way to mutate a filtered
    /// subset in place; [`Self::static_component_raw_span_mut`] is the bulk
    /// form that stamps every row.
    ///
    /// The span has `count == 1`, `stride` the component's size, and is
    /// valid until the next call that changes that archetype's storage.
    /// Errors, checked in this order: [`ComponentFfiError::NotAlive`],
    /// [`ComponentFfiError::UnknownId`] (never registered),
    /// [`ComponentFfiError::NotWritable`], [`ComponentFfiError::Absent`]
    /// (alive, but without this component). Nothing is stamped on an error.
    /// No component hook fires: this is a mutation, not an insert.
    pub fn static_component_row_mut(
        &mut self,
        entity: Entity,
        component_id: ComponentId,
    ) -> Result<FfiSpanMut, ComponentFfiError> {
        if !self.is_alive(entity) {
            return Err(ComponentFfiError::NotAlive);
        }
        self.archetypes
            .row_span_mut(entity, component_id, self.change_tick)
    }

    /// Entity-correlation counterpart to
    /// [`Self::static_component_raw_span`] — thin wrapper over
    /// `Archetypes::entity_ids`, for the same `(archetype_id,
    /// component_id)` pair.
    pub fn static_component_entity_ids(
        &self,
        archetype_id: ArchetypeId,
        component_id: ComponentId,
    ) -> Option<Vec<u64>> {
        self.archetypes.entity_ids(archetype_id, component_id)
    }

    /// The untyped, id-based counterpart of [`Self::query_added`] for
    /// one `(archetype_id, component_id)` pair: the row indices, in
    /// ascending order, whose value was inserted after `last_run`.
    /// Row `i` is element `i` of [`Self::static_component_raw_span`]
    /// and [`Self::static_component_entity_ids`] for the same pair.
    /// Takes a plain [`Tick`] instead of a [`crate::ChangeTracker`]
    /// because its caller (the FFI surface) holds the marker as a bare
    /// `u32`. `None` under exactly the conditions
    /// `static_component_raw_span` returns `None`; a zero-row archetype
    /// is `Some` of an empty list. Measured against [`Self::change_tick`]
    /// as it stands now, so call [`Self::increment_change_tick`] between
    /// a step's mutations and the check that should see them (see
    /// `tick.rs`).
    pub fn static_component_added_rows(
        &self,
        archetype_id: ArchetypeId,
        component_id: ComponentId,
        last_run: Tick,
    ) -> Option<Vec<u32>> {
        self.static_component_change_rows(archetype_id, component_id, last_run, true)
    }

    /// As [`Self::static_component_added_rows`], for values inserted
    /// *or* last mutated through [`Self::get_static_mut`] after
    /// `last_run` — the id-based counterpart of [`Self::query_changed`].
    pub fn static_component_changed_rows(
        &self,
        archetype_id: ArchetypeId,
        component_id: ComponentId,
        last_run: Tick,
    ) -> Option<Vec<u32>> {
        self.static_component_change_rows(archetype_id, component_id, last_run, false)
    }

    fn static_component_change_rows(
        &self,
        archetype_id: ArchetypeId,
        component_id: ComponentId,
        last_run: Tick,
        added_only: bool,
    ) -> Option<Vec<u32>> {
        let this_run = self.change_tick;
        let ticks = self.archetypes.ticks_span(archetype_id, component_id)?;
        Some(
            ticks
                .iter()
                .enumerate()
                .filter(|(_, t)| {
                    if added_only {
                        t.is_added(last_run, this_run)
                    } else {
                        t.is_changed(last_run, this_run)
                    }
                })
                .map(|(row, _)| row as u32)
                .collect(),
        )
    }

    /// Enumerates every currently-existing archetype whose signature
    /// includes `component_id` — thin wrapper over
    /// `Archetypes::archetypes_with`.
    pub fn archetypes_with_static_component(
        &self,
        component_id: ComponentId,
    ) -> impl Iterator<Item = ArchetypeId> + '_ {
        self.archetypes.archetypes_with(component_id)
    }

    /// Enumerates every currently-existing archetype whose signature
    /// contains all of `with`, none of `without`, and — if `any_of` is
    /// non-empty — at least one of `any_of` — thin wrapper over
    /// `Archetypes::archetypes_matching`, the id-based counterpart to
    /// the typed `*_static_filtered` queries in `query.rs`
    /// (`With`/`Without`/`Or`). Include the component you intend to
    /// read in `with`; pair the ids this yields with
    /// [`Self::static_component_raw_span`] and
    /// [`Self::static_component_entity_ids`]. Pass `&[]` for `any_of`
    /// when there's no `Or` constraint — that's "no constraint", not
    /// "match nothing".
    pub fn archetypes_matching_static<'a>(
        &'a self,
        with: &'a [ComponentId],
        without: &'a [ComponentId],
        any_of: &'a [ComponentId],
    ) -> impl Iterator<Item = ArchetypeId> + 'a {
        self.archetypes.archetypes_matching(with, without, any_of)
    }

    /// Looks up the `ComponentId` an Archetype-Core type was registered
    /// under via [`Self::register_ffi_static_component`], by name —
    /// thin wrapper over `Archetypes::lookup_ffi_id`.
    pub fn lookup_ffi_static_component_id(&self, name: &str) -> Option<ComponentId> {
        self.archetypes.lookup_ffi_id(name)
    }
}

impl Default for World {
    fn default() -> Self {
        Self::new()
    }
}

/// Resources: at most one value per Rust type, owned by the world but
/// attached to no entity (see `resource.rs`).
impl World {
    /// Stores `value` as the world's resource of type `T`, returning the
    /// previous one if there was one. A type can be a resource and a
    /// component at the same time; the two are separate.
    ///
    /// A new resource is stamped added and changed at the current tick;
    /// replacing one keeps its added tick and stamps it changed (see
    /// [`Self::resource_added`], [`Self::resource_changed`]).
    pub fn insert_resource<T: 'static>(&mut self, value: T) -> Option<T> {
        self.resources.insert(value, self.change_tick)
    }

    /// The world's resource of type `T`, if one has been inserted.
    pub fn get_resource<T: 'static>(&self) -> Option<&T> {
        self.resources.get::<T>()
    }

    /// Mutable access to the world's resource of type `T`, if any. The
    /// resource is stamped *changed* at the current tick when this is
    /// called, whether or not the caller then writes (the same rule as
    /// [`Self::get_static_mut`]; a system's `ResMut` stamps only when it
    /// is written through).
    pub fn get_resource_mut<T: 'static>(&mut self) -> Option<&mut T> {
        self.resources.get_mut::<T>(self.change_tick)
    }

    /// Whether the resource `T` was inserted after `tracker`'s last-checked
    /// tick (`false` if there is none): the resource form of
    /// [`Self::query_added`].
    pub fn resource_added<T: 'static>(&self, tracker: &ChangeTracker) -> bool {
        self.resources
            .ticks::<T>()
            .is_some_and(|t| t.is_added(tracker.last_run(), self.change_tick))
    }

    /// Whether the resource `T` was inserted, replaced, handed out by
    /// [`Self::get_resource_mut`] or written from C after `tracker`'s
    /// last-checked tick (`false` if there is none): the resource form of
    /// [`Self::query_changed`].
    pub fn resource_changed<T: 'static>(&self, tracker: &ChangeTracker) -> bool {
        self.resources
            .ticks::<T>()
            .is_some_and(|t| t.is_changed(tracker.last_run(), self.change_tick))
    }

    /// Removes and returns the world's resource of type `T`, if any.
    pub fn remove_resource<T: 'static>(&mut self) -> Option<T> {
        self.resources.remove::<T>()
    }

    /// Whether a resource of type `T` is currently inserted.
    pub fn contains_resource<T: 'static>(&self) -> bool {
        self.resources.contains::<T>()
    }

    /// Opts resource type `T` into FFI access under `name`, returning the
    /// id C callers will use. `T` must accept any bit pattern
    /// (`FromBytes`), because C writes it as raw bytes. Must be called
    /// from Rust; idempotent for the same `T`.
    ///
    /// # Panics
    /// If `name` was already registered for a different resource type.
    pub fn register_ffi_resource<T>(&mut self, name: &'static str) -> ResourceId
    where
        T: 'static + FromBytes + IntoBytes + Immutable + KnownLayout,
    {
        self.resources.register_ffi::<T>(name)
    }

    /// Looks up the id a resource type was registered under, by name.
    pub fn lookup_ffi_resource_id(&self, name: &str) -> Option<ResourceId> {
        self.resources.lookup_ffi_id(name)
    }

    /// The registered resource's value as a one-element span, or an empty
    /// span if it is registered but not inserted. `None` for an id that
    /// was never issued. Valid until the resource is removed or replaced;
    /// [`Self::write_resource_bytes`] on an existing resource updates it
    /// in place and does not invalidate the span.
    pub fn resource_raw_span(&self, id: ResourceId) -> Option<FfiSpan> {
        self.resources.ffi_span(id)
    }

    /// Copies `bytes` in as the registered resource's value, inserting it
    /// if absent. `bytes` must be exactly the registered type's size. An
    /// existing resource is stamped changed, a new one added and changed.
    pub fn write_resource_bytes(
        &mut self,
        id: ResourceId,
        bytes: &[u8],
    ) -> Result<(), ResourceFfiError> {
        self.resources.ffi_write(id, bytes, self.change_tick)
    }

    /// Whether the registered resource was inserted after `last_run`
    /// (`false` if it is registered but not inserted): the by-id form of
    /// [`Self::resource_added`] for a caller that holds a last-run tick
    /// rather than a [`ChangeTracker`]. `UnknownId` for an id never issued.
    pub fn resource_added_since(
        &self,
        id: ResourceId,
        last_run: Tick,
    ) -> Result<bool, ResourceFfiError> {
        Ok(self
            .resources
            .ffi_ticks(id)?
            .is_some_and(|t| t.is_added(last_run, self.change_tick)))
    }

    /// Whether the registered resource was inserted, replaced, handed out
    /// by [`Self::get_resource_mut`] or written from C after `last_run`
    /// (`false` if it is registered but not inserted); the by-id form of
    /// [`Self::resource_changed`].
    pub fn resource_changed_since(
        &self,
        id: ResourceId,
        last_run: Tick,
    ) -> Result<bool, ResourceFfiError> {
        Ok(self
            .resources
            .ffi_ticks(id)?
            .is_some_and(|t| t.is_changed(last_run, self.change_tick)))
    }

    /// Removes the registered resource's value. `Absent` if the type is
    /// registered but nothing is inserted.
    pub fn remove_ffi_resource(&mut self, id: ResourceId) -> Result<(), ResourceFfiError> {
        self.resources.ffi_remove(id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_world_is_empty() {
        let w = World::new();
        assert_eq!(w.entity_count(), 0);
        assert!(w.is_empty());
    }

    #[test]
    fn spawn_gives_a_live_entity() {
        let mut w = World::new();
        let e = w.spawn();
        assert!(w.is_alive(e));
        assert_eq!(w.entity_count(), 1);
        assert!(!w.is_empty());
    }

    #[test]
    fn spawn_returns_distinct_entities() {
        let mut w = World::new();
        let e1 = w.spawn();
        let e2 = w.spawn();
        assert_ne!(e1, e2);
        assert_eq!(w.entity_count(), 2);
    }

    #[test]
    fn despawn_then_not_alive() {
        let mut w = World::new();
        let e = w.spawn();
        assert!(w.despawn(e));
        assert!(!w.is_alive(e));
        assert_eq!(w.entity_count(), 0);
    }

    #[test]
    fn despawn_twice_is_a_safe_no_op() {
        let mut w = World::new();
        let e = w.spawn();
        assert!(w.despawn(e));
        assert!(
            !w.despawn(e),
            "second despawn of the same entity must not panic or double-free"
        );
    }

    #[test]
    fn despawn_stale_handle_after_slot_reuse_does_not_touch_the_new_occupant() {
        // The scenario that actually matters: a slot gets freed, reused
        // by a new entity, and something is still holding the OLD,
        // now-stale handle to that same raw index. Despawning it must
        // fail cleanly, not silently despawn whatever now legitimately
        // occupies that slot.
        //
        // (A test resembling "despawn a totally foreign handle from
        // another World" was tried here first and removed: two fresh
        // World::new() allocators both hand out {index: 0, generation:
        // 1} as their first spawn, since GenerationalIndex carries no
        // per-allocator identity -- so that handle isn't actually
        // distinguishable from a legitimate same-shape one, and the
        // test's own premise was wrong, not World's behavior. Caught by
        // actually running it, not assumed correct from reasoning about
        // it beforehand.)
        let mut w = World::new();
        let e1 = w.spawn();
        w.despawn(e1);
        let e2 = w.spawn(); // reuses e1's slot, different generation

        assert!(
            !w.despawn(e1),
            "despawning the stale e1 handle must fail, not succeed"
        );
        assert!(
            w.is_alive(e2),
            "e2 must be untouched by the failed despawn(e1) call"
        );
    }

    #[test]
    fn respawned_slot_gives_a_distinguishable_entity() {
        // The actual point of the generational design, exercised through
        // World's real public API, not just mid_collections' own tests.
        let mut w = World::new();
        let e1 = w.spawn();
        w.despawn(e1);
        let e2 = w.spawn();

        assert_eq!(e2.index(), e1.index(), "the freed slot should be reused");
        assert_ne!(e2.generation(), e1.generation());
        assert!(w.is_alive(e2));
        assert!(
            !w.is_alive(e1),
            "the stale handle from before the reuse must not alias e2"
        );
    }

    #[test]
    fn display_shows_index_and_generation() {
        let mut w = World::new();
        let e = w.spawn();
        let text = format!("{e}");
        assert_eq!(text, format!("Entity({}v{})", e.index(), e.generation()));
    }

    #[test]
    fn implements_sparse_set_index() {
        let mut w = World::new();
        let e = w.spawn();
        assert_eq!(e.sparse_index(), e.index());
    }

    #[test]
    fn with_capacity_does_not_change_observable_behavior() {
        let mut w = World::with_capacity(64);
        assert!(w.is_empty());
        let e = w.spawn();
        assert!(w.is_alive(e));
    }

    #[test]
    fn default_matches_new() {
        let w = World::default();
        assert!(w.is_empty());
    }

    #[test]
    fn many_spawn_despawn_cycles_keep_entity_count_consistent() {
        let mut w = World::new();
        let mut live = Vec::new();

        for round in 0..50u32 {
            live.push(w.spawn());
            if round % 3 == 0 && !live.is_empty() {
                let dead: Entity = live.remove(0);
                assert!(w.despawn(dead));
            }
            assert_eq!(w.entity_count(), live.len());
            for &e in &live {
                assert!(w.is_alive(e));
            }
        }
    }

    #[derive(Debug, PartialEq)]
    struct Health(u32);

    #[test]
    fn insert_then_get_component() {
        let mut w = World::new();
        let e = w.spawn();
        assert_eq!(w.insert(e, Health(100)), None);
        assert_eq!(w.get::<Health>(e), Some(&Health(100)));
        assert!(w.has::<Health>(e));
    }

    #[test]
    fn get_component_on_dead_entity_returns_none() {
        let mut w = World::new();
        let e = w.spawn();
        w.insert(e, Health(50));
        w.despawn(e);
        assert_eq!(
            w.get::<Health>(e),
            None,
            "a dead entity must not still report its old component"
        );
    }

    #[test]
    fn insert_on_dead_entity_is_a_safe_no_op() {
        let mut w = World::new();
        let e = w.spawn();
        w.despawn(e);
        assert_eq!(w.insert(e, Health(999)), None);
        assert_eq!(w.get::<Health>(e), None);
    }

    #[test]
    fn despawn_removes_all_attached_components() {
        #[derive(Debug, PartialEq)]
        struct Mana(u32);

        let mut w = World::new();
        let e = w.spawn();
        w.insert(e, Health(100));
        w.insert(e, Mana(50));

        assert!(w.despawn(e));

        assert_eq!(w.get::<Health>(e), None);
        assert_eq!(w.get::<Mana>(e), None);
    }

    #[test]
    fn despawn_does_not_touch_other_entities_components() {
        let mut w = World::new();
        let e1 = w.spawn();
        let e2 = w.spawn();
        w.insert(e1, Health(10));
        w.insert(e2, Health(20));

        w.despawn(e1);

        assert_eq!(
            w.get::<Health>(e2),
            Some(&Health(20)),
            "e2 must be untouched by e1's despawn"
        );
    }

    #[test]
    fn reused_slot_does_not_inherit_the_old_entitys_components() {
        // The actual point of checking `is_alive` before every component
        // access in World, not just SparseShell's own tests: a *new*
        // entity reusing a freed slot must never read the *old* dead
        // entity's leftover data, even transiently.
        let mut w = World::new();
        let e1 = w.spawn();
        w.insert(e1, Health(77));
        w.despawn(e1);

        let e2 = w.spawn(); // reuses e1's slot, different generation
        assert_eq!(
            w.get::<Health>(e2),
            None,
            "e2 must not inherit e1's old Health component"
        );

        w.insert(e2, Health(5));
        assert_eq!(w.get::<Health>(e2), Some(&Health(5)));
    }

    #[test]
    fn stale_handle_cannot_read_the_live_entity_now_sharing_its_index() {
        // The specific gap `SparseSet` alone can't close, documented in
        // component.rs's doc comment: a *stale* Entity handle (e1) and a
        // *live* Entity (e2) can share the same raw index after reuse.
        // World's own liveness check is what makes get(e1) correctly
        // fail here, rather than accidentally returning e2's real data.
        let mut w = World::new();
        let e1 = w.spawn();
        w.despawn(e1);
        let e2 = w.spawn(); // same raw index as e1, higher generation
        w.insert(e2, Health(42));

        assert_eq!(
            e1.index(),
            e2.index(),
            "this test only proves anything if the index was actually reused"
        );
        assert_eq!(
            w.get::<Health>(e1),
            None,
            "the stale e1 handle must not see e2's Health component"
        );
        assert_eq!(w.get::<Health>(e2), Some(&Health(42)));
    }

    #[derive(Debug, PartialEq)]
    struct Mass(f32);
    #[derive(Debug, PartialEq)]
    struct Charge(f32);

    #[test]
    fn insert_static_then_get_static() {
        let mut w = World::new();
        let e = w.spawn();
        assert!(w.insert_static(e, Mass(10.0)));
        assert_eq!(w.get_static::<Mass>(e), Some(&Mass(10.0)));
        assert!(w.has_static::<Mass>(e));
    }

    #[test]
    fn insert_static_two_components_preserves_both() {
        let mut w = World::new();
        let e = w.spawn();
        w.insert_static(e, Mass(1.0));
        w.insert_static(e, Charge(2.0));
        assert_eq!(w.get_static::<Mass>(e), Some(&Mass(1.0)));
        assert_eq!(w.get_static::<Charge>(e), Some(&Charge(2.0)));
    }

    #[test]
    fn get_static_component_on_dead_entity_returns_none() {
        let mut w = World::new();
        let e = w.spawn();
        w.insert_static(e, Mass(5.0));
        w.despawn(e);
        assert_eq!(
            w.get_static::<Mass>(e),
            None,
            "a dead entity must not still report its old static component"
        );
    }

    #[test]
    fn insert_static_on_dead_entity_is_a_safe_no_op() {
        let mut w = World::new();
        let e = w.spawn();
        w.despawn(e);
        assert!(!w.insert_static(e, Mass(999.0)));
        assert_eq!(w.get_static::<Mass>(e), None);
    }

    #[test]
    fn despawn_removes_static_components_too() {
        let mut w = World::new();
        let e = w.spawn();
        w.insert_static(e, Mass(1.0));
        w.insert_static(e, Charge(2.0));

        assert!(w.despawn(e));

        assert_eq!(w.get_static::<Mass>(e), None);
        assert_eq!(w.get_static::<Charge>(e), None);
    }

    #[test]
    fn despawn_does_not_touch_other_entities_static_components() {
        let mut w = World::new();
        let e1 = w.spawn();
        let e2 = w.spawn();
        w.insert_static(e1, Mass(1.0));
        w.insert_static(e2, Mass(2.0));

        w.despawn(e1);

        assert_eq!(
            w.get_static::<Mass>(e2),
            Some(&Mass(2.0)),
            "e2 must be untouched by e1's despawn"
        );
    }

    #[test]
    fn reused_slot_does_not_inherit_the_old_entitys_static_components() {
        let mut w = World::new();
        let e1 = w.spawn();
        w.insert_static(e1, Mass(77.0));
        w.despawn(e1);

        let e2 = w.spawn(); // reuses e1's slot, different generation
        assert_eq!(
            w.get_static::<Mass>(e2),
            None,
            "e2 must not inherit e1's old static Mass component"
        );

        w.insert_static(e2, Mass(5.0));
        assert_eq!(w.get_static::<Mass>(e2), Some(&Mass(5.0)));
    }

    #[test]
    fn stale_handle_cannot_read_the_live_entitys_static_component() {
        let mut w = World::new();
        let e1 = w.spawn();
        w.despawn(e1);
        let e2 = w.spawn(); // same raw index as e1, higher generation
        w.insert_static(e2, Mass(42.0));

        assert_eq!(
            e1.index(),
            e2.index(),
            "this test only proves anything if the index was actually reused"
        );
        assert_eq!(
            w.get_static::<Mass>(e1),
            None,
            "the stale e1 handle must not see e2's static Mass component"
        );
        assert_eq!(w.get_static::<Mass>(e2), Some(&Mass(42.0)));
    }

    #[test]
    fn remove_static_returns_value_and_clears_it() {
        let mut w = World::new();
        let e = w.spawn();
        w.insert_static(e, Mass(3.0));
        assert_eq!(w.remove_static::<Mass>(e), Some(Mass(3.0)));
        assert_eq!(w.get_static::<Mass>(e), None);
        assert!(!w.has_static::<Mass>(e));
    }

    #[test]
    fn insert_bundle_attaches_every_element() {
        let mut w = World::new();
        let e = w.spawn();
        assert!(w.insert_bundle(e, (Mass(1.0), Charge(2.0))));
        assert_eq!(w.get_static::<Mass>(e), Some(&Mass(1.0)));
        assert_eq!(w.get_static::<Charge>(e), Some(&Charge(2.0)));
    }

    #[test]
    fn insert_bundle_reaches_the_same_archetype_as_sequential_single_inserts() {
        // The actual point of Bundle: one migration instead of two, but
        // landing in the exact same place either way -- proven by
        // checking both entities end up able to be found by the same
        // archetypes_with_static_component query.
        let mut w = World::new();
        let via_bundle = w.spawn();
        let via_sequential = w.spawn();

        assert!(w.insert_bundle(via_bundle, (Mass(1.0), Charge(2.0))));
        assert!(w.insert_static(via_sequential, Mass(1.0)));
        assert!(w.insert_static(via_sequential, Charge(2.0)));

        let ids: Vec<Entity> = w.query_static::<Mass>().map(|(e, _)| e).collect();
        let mut sorted = ids.clone();
        sorted.sort_by_key(|e| e.index());
        let mut expected = vec![via_bundle, via_sequential];
        expected.sort_by_key(|e| e.index());
        assert_eq!(
            sorted, expected,
            "both entities must be found by the same query regardless of insertion path"
        );
    }

    #[test]
    fn insert_bundle_on_dead_entity_is_a_safe_no_op() {
        let mut w = World::new();
        let e = w.spawn();
        w.despawn(e);
        assert!(!w.insert_bundle(e, (Mass(1.0), Charge(2.0))));
    }

    #[test]
    fn insert_bundle_no_ops_if_entity_already_has_any_element() {
        let mut w = World::new();
        let e = w.spawn();
        w.insert_static(e, Mass(1.0));

        // Already has Mass -- the whole bundle insert must be a no-op,
        // not a partial application that adds Charge alone.
        assert!(!w.insert_bundle(e, (Mass(2.0), Charge(3.0))));
        assert_eq!(
            w.get_static::<Mass>(e),
            Some(&Mass(1.0)),
            "original Mass must be untouched"
        );
        assert_eq!(
            w.get_static::<Charge>(e),
            None,
            "Charge must not have been partially applied"
        );
    }

    #[test]
    fn remove_bundle_detaches_every_element_and_returns_them() {
        let mut w = World::new();
        let e = w.spawn();
        w.insert_bundle(e, (Mass(1.0), Charge(2.0)));

        assert_eq!(
            w.remove_bundle::<(Mass, Charge)>(e),
            Some((Mass(1.0), Charge(2.0)))
        );
        assert_eq!(w.get_static::<Mass>(e), None);
        assert_eq!(w.get_static::<Charge>(e), None);
    }

    #[test]
    fn remove_bundle_on_dead_entity_is_none() {
        let mut w = World::new();
        let e = w.spawn();
        w.insert_bundle(e, (Mass(1.0), Charge(2.0)));
        w.despawn(e);
        assert_eq!(w.remove_bundle::<(Mass, Charge)>(e), None);
    }

    #[test]
    fn remove_bundle_is_none_if_any_element_was_never_registered() {
        let mut w = World::new();
        let e = w.spawn();
        w.insert_static(e, Mass(1.0));
        // Charge has never been registered as an archetype-tracked
        // component for anything, anywhere in this World.
        assert_eq!(w.remove_bundle::<(Mass, Charge)>(e), None);
        // And Mass must be untouched -- no partial application.
        assert_eq!(w.get_static::<Mass>(e), Some(&Mass(1.0)));
    }

    #[test]
    fn remove_bundle_no_ops_if_entity_is_missing_any_element() {
        let mut w = World::new();
        let e = w.spawn();
        w.insert_static(e, Mass(1.0));
        // Charge is registered elsewhere but e itself doesn't have it.
        let other = w.spawn();
        w.insert_static(other, Charge(9.0));

        assert_eq!(w.remove_bundle::<(Mass, Charge)>(e), None);
        assert_eq!(
            w.get_static::<Mass>(e),
            Some(&Mass(1.0)),
            "Mass must not have been partially removed"
        );
    }

    #[test]
    fn remove_bundle_migrates_a_surviving_component_to_the_new_archetype() {
        // Exercises remove_bundle's *general* path (`Bundle::take_from`,
        // the `Box<dyn Any>` migration loop) rather than its fast path
        // (`take_direct`) -- every other existing remove_bundle test
        // removes an entity's *entire* archetype-tracked signature, so
        // none of them actually reach this branch. `Health` here is a
        // third archetype-tracked component that must survive the
        // removal and migrate to the new archetype, which is exactly
        // the condition (`ids.len() != columns.len()`) that selects the
        // general path over the fast one.
        let mut w = World::new();
        let e = w.spawn();
        w.insert(e, Health(100));
        w.insert_bundle(e, (Mass(1.0), Charge(2.0)));

        assert_eq!(
            w.remove_bundle::<(Mass, Charge)>(e),
            Some((Mass(1.0), Charge(2.0)))
        );
        assert_eq!(w.get::<Mass>(e), None);
        assert_eq!(w.get::<Charge>(e), None);
        assert_eq!(
            w.get::<Health>(e),
            Some(&Health(100)),
            "a surviving component must migrate to the new archetype intact, not just avoid being removed"
        );
    }

    #[test]
    fn insert_bundle_then_remove_bundle_round_trips() {
        let mut w = World::new();
        let e = w.spawn();
        assert!(w.insert_bundle(e, (Mass(7.0), Charge(8.0))));
        let removed = w.remove_bundle::<(Mass, Charge)>(e).unwrap();
        assert_eq!(removed, (Mass(7.0), Charge(8.0)));
        assert!(!w.has_static::<Mass>(e));
        assert!(!w.has_static::<Charge>(e));
    }

    #[test]
    fn query_static_after_bulk_insert_bundle_does_not_panic() {
        // Regression test for a real panic hit setting up the ECS bench
        // harness: "archetypes_with guarantees component_id is in this
        // archetype's own signature", inside Archetypes::iter.
        //
        // Root cause: World::insert_bundle chains edge_for_insert once
        // per bundle element (see its own doc comment) to compute the
        // final target archetype. Each intermediate step really does
        // call Archetypes::get_or_create, which really does register a
        // correct `component_ids` signature for that intermediate
        // archetype -- but insert_bundle only ever moves data into the
        // *final* archetype in the chain, never the intermediate ones.
        // So an intermediate archetype can exist with a signature that
        // claims a component, while never having had a column created
        // for it (no entity was ever actually placed there). None of
        // the other insert_bundle tests above call query_static, so a
        // single insert_bundle call alone was never enough to surface
        // this -- it takes a real iteration over Archetypes::iter to
        // reach the archetype and its absent column.
        let mut w = World::new();
        let entities: Vec<Entity> = (0..64)
            .map(|i| {
                let e = w.spawn();
                assert!(w.insert_bundle(e, (Mass(i as f32), Charge(i as f32 * 2.0))));
                e
            })
            .collect();

        let found: Vec<Entity> = w.query_static::<Mass>().map(|(e, _)| e).collect();
        assert_eq!(found.len(), entities.len());
        for e in &entities {
            assert!(found.contains(e));
        }

        let found2: Vec<Entity> = w
            .query2_static::<Mass, Charge>()
            .map(|(e, _, _)| e)
            .collect();
        assert_eq!(found2.len(), entities.len());
    }

    #[test]
    #[should_panic(expected = "was already used with Sparse Shell")]
    fn using_a_sparse_type_with_insert_static_panics() {
        let mut w = World::new();
        let e1 = w.spawn();
        let e2 = w.spawn();
        w.insert(e1, Mass(1.0)); // claims Mass for Sparse Shell
        w.insert_static(e2, Mass(2.0)); // must panic: Mass already claimed
    }

    #[test]
    #[should_panic(expected = "was already used with Archetype Core")]
    fn using_a_static_type_with_insert_panics() {
        let mut w = World::new();
        let e1 = w.spawn();
        let e2 = w.spawn();
        w.insert_static(e1, Mass(1.0)); // claims Mass for Archetype Core
        w.insert(e2, Mass(2.0)); // must panic: Mass already claimed
    }

    #[test]
    #[should_panic(expected = "was already used with Sparse Shell")]
    fn register_ffi_static_component_panics_if_the_type_was_already_used_sparse() {
        let mut w = World::new();
        let e = w.spawn();
        w.insert(e, FfiHealth { hp: 1 }); // claims FfiHealth for Sparse Shell
        w.register_ffi_static_component::<FfiHealth>("FfiHealth"); // must panic
    }

    #[test]
    fn repeated_use_with_the_same_system_never_panics() {
        // The guard remembers *what* a type was claimed for, not just
        // *that* it was claimed -- repeated ordinary use with the same
        // system it was already claimed for must stay completely silent.
        let mut w = World::new();
        let e1 = w.spawn();
        let e2 = w.spawn();
        w.insert(e1, Mass(1.0));
        w.insert(e2, Mass(2.0)); // same type, same system again -- fine
        w.insert_static(e1, Charge(3.0));
        w.insert_static(e2, Charge(4.0)); // same type, same system again -- fine
    }

    #[test]
    fn different_types_in_different_systems_never_panics() {
        // The actual supported, intended pattern -- distinct types each
        // committed to one system, matching static_and_sparse_shell_
        // storage_are_independent above, just isolated here specifically
        // as the guard's own "this must never false-positive" case.
        let mut w = World::new();
        let e = w.spawn();
        w.insert(e, Health(1)); // Sparse Shell
        w.insert_static(e, Mass(1.0)); // Archetype Core, different type
    }

    #[test]
    fn get_static_mut_actually_mutates() {
        let mut w = World::new();
        let e = w.spawn();
        w.insert_static(e, Mass(1.0));
        w.get_static_mut::<Mass>(e).unwrap().0 = 50.0;
        assert_eq!(w.get_static::<Mass>(e), Some(&Mass(50.0)));
    }

    #[test]
    fn static_and_sparse_shell_storage_are_independent() {
        // Position (sparse-shell, via `insert`) and Mass (archetype-core,
        // via `insert_static`) coexisting on the same entity, each
        // readable only through its own matching accessor.
        let mut w = World::new();
        let e = w.spawn();
        w.insert(e, Health(100));
        w.insert_static(e, Mass(2.5));

        assert_eq!(w.get::<Health>(e), Some(&Health(100)));
        assert_eq!(w.get_static::<Mass>(e), Some(&Mass(2.5)));

        w.despawn(e);
        assert_eq!(w.get::<Health>(e), None);
        assert_eq!(w.get_static::<Mass>(e), None);
    }

    #[test]
    fn entity_as_ffi_from_ffi_round_trips() {
        let mut w = World::new();
        let e = w.spawn();
        let packed = e.as_ffi();
        let unpacked = Entity::from_ffi(packed);
        assert_eq!(unpacked, e);
        assert!(w.is_alive(unpacked));
    }

    #[test]
    fn entity_from_ffi_after_reuse_correctly_rejects_the_stale_packed_value() {
        let mut w = World::new();
        let e1 = w.spawn();
        let packed_e1 = e1.as_ffi();
        w.despawn(e1);
        let e2 = w.spawn(); // reuses e1's slot

        let reconstructed_stale = Entity::from_ffi(packed_e1);
        assert!(
            !w.is_alive(reconstructed_stale),
            "a stale packed handle must not read as alive after its slot was reused"
        );
        assert!(w.is_alive(e2));
    }

    #[derive(Debug, Clone, Copy, PartialEq, IntoBytes, KnownLayout, Immutable)]
    #[repr(C)]
    struct FfiHealth {
        hp: u32,
    }

    #[derive(Debug, Clone, Copy, PartialEq, IntoBytes, KnownLayout, Immutable)]
    #[repr(C)]
    struct FfiStamina {
        stamina: u32,
    }

    #[test]
    fn register_ffi_component_then_span_reflects_attached_values() {
        let mut w = World::new();
        let id = w.register_ffi_component::<FfiHealth>("FfiHealth");
        let e1 = w.spawn();
        let e2 = w.spawn();
        w.insert(e1, FfiHealth { hp: 10 });
        w.insert(e2, FfiHealth { hp: 20 });

        let span = w.component_raw_span(id).expect("registered and populated");
        assert_eq!(span.count, 2);
        // SAFETY: span points at World's own live storage, unmutated
        // since the calls above.
        let values =
            unsafe { core::slice::from_raw_parts(span.ptr.cast::<FfiHealth>(), span.count) };
        assert_eq!(values[0], FfiHealth { hp: 10 });
        assert_eq!(values[1], FfiHealth { hp: 20 });
    }

    #[test]
    fn component_raw_span_on_never_registered_id_is_none() {
        let w = World::new();
        assert_eq!(w.component_raw_span(ComponentId(0)), None);
    }

    #[test]
    fn component_entity_ids_correlate_with_raw_span_through_world() {
        let mut w = World::new();
        let id = w.register_ffi_component::<FfiHealth>("FfiHealth");
        let e1 = w.spawn();
        let e2 = w.spawn();
        w.insert(e1, FfiHealth { hp: 10 });
        w.insert(e2, FfiHealth { hp: 20 });

        let ids = w
            .component_entity_ids(id)
            .expect("registered and populated");
        let span = w.component_raw_span(id).expect("registered and populated");
        assert_eq!(ids.len(), span.count);
        // SAFETY: span points at World's own live storage, unmutated
        // since the calls above.
        let values =
            unsafe { core::slice::from_raw_parts(span.ptr.cast::<FfiHealth>(), span.count) };
        // Real round trip through the same packing every other Entity
        // FFI path already uses (World::spawn/despawn's own
        // as_ffi/from_ffi), not just "some u64 at this index".
        assert_eq!(Entity::from_ffi(ids[0]), e1);
        assert_eq!(Entity::from_ffi(ids[1]), e2);
        assert_eq!(values[0], FfiHealth { hp: 10 });
        assert_eq!(values[1], FfiHealth { hp: 20 });
    }

    #[test]
    fn component_entity_ids_on_never_registered_id_is_none() {
        let w = World::new();
        assert_eq!(w.component_entity_ids(ComponentId(0)), None);
    }

    #[test]
    fn lookup_ffi_component_id_round_trips_through_world() {
        let mut w = World::new();
        let id = w.register_ffi_component::<FfiHealth>("FfiHealth");
        assert_eq!(w.lookup_ffi_component_id("FfiHealth"), Some(id));
        assert_eq!(w.lookup_ffi_component_id("NeverRegistered"), None);
    }

    #[test]
    fn register_ffi_static_component_then_span_reflects_attached_values() {
        let mut w = World::new();
        let id = w.register_ffi_static_component::<FfiHealth>("FfiHealthStatic");
        let e1 = w.spawn();
        let e2 = w.spawn();
        assert!(w.insert_static(e1, FfiHealth { hp: 100 }));
        assert!(w.insert_static(e2, FfiHealth { hp: 200 }));

        let archetype_id = w
            .archetypes_with_static_component(id)
            .next()
            .expect("both entities must share one archetype");
        let span = w
            .static_component_raw_span(archetype_id, id)
            .expect("registered and populated");
        assert_eq!(span.count, 2);
        // SAFETY: span points at World's own live storage, unmutated
        // since the calls above.
        let values =
            unsafe { core::slice::from_raw_parts(span.ptr.cast::<FfiHealth>(), span.count) };
        assert_eq!(values[0], FfiHealth { hp: 100 });
        assert_eq!(values[1], FfiHealth { hp: 200 });
    }

    #[test]
    fn static_component_entity_ids_correlate_with_raw_span_through_world() {
        let mut w = World::new();
        let id = w.register_ffi_static_component::<FfiHealth>("FfiHealthStatic");
        let e1 = w.spawn();
        let e2 = w.spawn();
        assert!(w.insert_static(e1, FfiHealth { hp: 100 }));
        assert!(w.insert_static(e2, FfiHealth { hp: 200 }));

        let archetype_id = w
            .archetypes_with_static_component(id)
            .next()
            .expect("both entities must share one archetype");
        let ids = w
            .static_component_entity_ids(archetype_id, id)
            .expect("registered and populated");
        let span = w
            .static_component_raw_span(archetype_id, id)
            .expect("registered and populated");
        assert_eq!(ids.len(), span.count);
        // SAFETY: span points at World's own live storage, unmutated
        // since the calls above.
        let values =
            unsafe { core::slice::from_raw_parts(span.ptr.cast::<FfiHealth>(), span.count) };
        assert_eq!(Entity::from_ffi(ids[0]), e1);
        assert_eq!(Entity::from_ffi(ids[1]), e2);
        assert_eq!(values[0], FfiHealth { hp: 100 });
        assert_eq!(values[1], FfiHealth { hp: 200 });
    }

    #[test]
    fn static_component_entity_ids_on_never_registered_id_is_none() {
        let mut w = World::new();
        let health_id = w.register_ffi_static_component::<FfiHealth>("FfiHealthStatic");
        let e = w.spawn();
        assert!(w.insert_static(e, FfiHealth { hp: 1 }));
        let archetype_id = w
            .archetypes_with_static_component(health_id)
            .next()
            .expect("just inserted, must exist");

        assert_eq!(
            w.static_component_entity_ids(archetype_id, ComponentId(9999)),
            None,
            "9999 was never registered, unlike `health_id` itself"
        );
    }

    #[test]
    fn static_component_raw_span_on_never_registered_id_is_none() {
        let mut w = World::new();
        // A real, legitimately-obtained ArchetypeId -- ArchetypeId can't
        // be constructed directly from outside archetype.rs, so this
        // test exercises the "component never registered" branch
        // (checked first in `raw_span`) against a real archetype rather
        // than a value that can't exist in the first place.
        let health_id = w.register_ffi_static_component::<FfiHealth>("FfiHealthStatic");
        let e = w.spawn();
        assert!(w.insert_static(e, FfiHealth { hp: 1 }));
        let archetype_id = w
            .archetypes_with_static_component(health_id)
            .next()
            .expect("just inserted, must exist");

        assert_eq!(
            w.static_component_raw_span(archetype_id, ComponentId(9999)),
            None,
            "9999 was never registered, unlike `health_id` itself"
        );
    }

    #[test]
    fn lookup_ffi_static_component_id_round_trips_through_world() {
        let mut w = World::new();
        let id = w.register_ffi_static_component::<FfiHealth>("FfiHealthStatic");
        assert_eq!(
            w.lookup_ffi_static_component_id("FfiHealthStatic"),
            Some(id)
        );
        assert_eq!(w.lookup_ffi_static_component_id("NeverRegistered"), None);
    }

    #[test]
    fn sparse_and_static_ffi_registrations_use_independent_name_spaces() {
        // Same name, two different storage systems -- must not collide,
        // matching Archetypes' already-independent ComponentId numbering
        // space from SparseShell's own (component.rs/archetype.rs's own
        // doc comments). Two distinct types, deliberately, not the same
        // type reused across both systems -- StorageClaims now guards
        // against exactly that (see this file's own doc comment), so
        // proving name-namespace independence needs two real component
        // types, one per system, the way an actual caller would use this.
        let mut w = World::new();
        let sparse_id = w.register_ffi_component::<FfiHealth>("Health");
        let static_id = w.register_ffi_static_component::<FfiStamina>("Health");
        assert_eq!(w.lookup_ffi_component_id("Health"), Some(sparse_id));
        assert_eq!(w.lookup_ffi_static_component_id("Health"), Some(static_id));
    }

    // Migration tests for the archetype-tracked (`insert_static`) path.
    // Every survivor here is a real, non-zero-sized, archetype-tracked
    // column, which is what `Column::move_row_to` actually has to move;
    // the older `remove_bundle` test above uses sparse-shell `insert`
    // for its survivor and so never exercised that.

    #[test]
    fn insert_static_migration_preserves_every_entitys_values() {
        let mut w = World::new();
        let es: Vec<_> = (0..4u32)
            .map(|i| {
                let e = w.spawn();
                w.insert_static(e, Health(i));
                e
            })
            .collect();
        // Migrating the first entity swap-removes it from a table whose
        // last row belongs to a different entity.
        assert!(w.insert_static(es[0], Mass(9.0)));
        for (i, &e) in es.iter().enumerate() {
            assert_eq!(w.get_static::<Health>(e), Some(&Health(i as u32)));
        }
        assert_eq!(w.get_static::<Mass>(es[0]), Some(&Mass(9.0)));
        assert_eq!(w.get_static::<Mass>(es[1]), None);
    }

    #[test]
    fn remove_static_returns_the_value_and_migrates_survivors() {
        let mut w = World::new();
        let es: Vec<_> = (0..3u32)
            .map(|i| {
                let e = w.spawn();
                w.insert_static(e, Health(i));
                w.insert_static(e, Mass(i as f32));
                e
            })
            .collect();
        assert_eq!(w.remove_static::<Mass>(es[0]), Some(Mass(0.0)));
        assert_eq!(w.get_static::<Mass>(es[0]), None);
        assert_eq!(w.get_static::<Health>(es[0]), Some(&Health(0)));
        for (i, &e) in es.iter().enumerate().skip(1) {
            assert_eq!(w.get_static::<Health>(e), Some(&Health(i as u32)));
            assert_eq!(w.get_static::<Mass>(e), Some(&Mass(i as f32)));
        }
        assert_eq!(w.remove_static::<Mass>(es[0]), None);
    }

    #[test]
    fn migration_moves_values_and_drops_each_exactly_once() {
        use std::cell::Cell;
        use std::rc::Rc;

        struct Tracked(Rc<Cell<u32>>);
        impl Drop for Tracked {
            fn drop(&mut self) {
                self.0.set(self.0.get() + 1);
            }
        }

        let drops = Rc::new(Cell::new(0));
        let mut w = World::new();
        let e = w.spawn();
        w.insert_static(e, Tracked(drops.clone()));
        w.insert_static(e, Mass(1.0));
        w.insert_static(e, Charge(2.0));
        assert_eq!(
            drops.get(),
            0,
            "migration must move values, never drop them"
        );
        assert_eq!(w.remove_static::<Mass>(e), Some(Mass(1.0)));
        assert_eq!(
            drops.get(),
            0,
            "removing a sibling must not drop the survivor"
        );
        assert!(w.despawn(e));
        assert_eq!(
            drops.get(),
            1,
            "dropped exactly once, when the entity is despawned"
        );
    }

    #[test]
    fn remove_static_hands_back_the_value_undropped() {
        use std::cell::Cell;
        use std::rc::Rc;

        struct Tracked(Rc<Cell<u32>>);
        impl Drop for Tracked {
            fn drop(&mut self) {
                self.0.set(self.0.get() + 1);
            }
        }

        let drops = Rc::new(Cell::new(0));
        let mut w = World::new();
        let e = w.spawn();
        w.insert_static(e, Tracked(drops.clone()));
        w.insert_static(e, Mass(1.0));
        let value = w.remove_static::<Tracked>(e);
        assert!(value.is_some());
        assert_eq!(
            drops.get(),
            0,
            "the removed value belongs to the caller now"
        );
        drop(value);
        assert_eq!(drops.get(), 1);
    }

    #[test]
    fn insert_bundle_migrates_existing_archetype_components() {
        let mut w = World::new();
        let e = w.spawn();
        w.insert_static(e, Health(5));
        assert!(w.insert_bundle(e, (Mass(1.0), Charge(2.0))));
        assert_eq!(w.get_static::<Health>(e), Some(&Health(5)));
        assert_eq!(w.get_static::<Mass>(e), Some(&Mass(1.0)));
        assert_eq!(w.get_static::<Charge>(e), Some(&Charge(2.0)));
    }

    #[test]
    fn remove_bundle_migrates_a_surviving_archetype_component() {
        let mut w = World::new();
        let e = w.spawn();
        w.insert_static(e, Health(7));
        w.insert_bundle(e, (Mass(1.0), Charge(2.0)));
        assert_eq!(
            w.remove_bundle::<(Mass, Charge)>(e),
            Some((Mass(1.0), Charge(2.0)))
        );
        assert_eq!(w.get_static::<Health>(e), Some(&Health(7)));
        assert_eq!(w.get_static::<Mass>(e), None);
    }

    // `World::spawn_bundle` -- direct one-step spawn.

    #[test]
    fn spawn_bundle_is_alive_and_holds_every_value() {
        let mut w = World::new();
        let e = w.spawn_bundle((Mass(1.0), Charge(2.0)));
        assert!(w.is_alive(e));
        assert_eq!(w.get_static::<Mass>(e), Some(&Mass(1.0)));
        assert_eq!(w.get_static::<Charge>(e), Some(&Charge(2.0)));
    }

    #[test]
    fn spawn_bundle_gives_distinct_entities_and_correct_count() {
        let mut w = World::new();
        let e1 = w.spawn_bundle((Mass(1.0), Charge(2.0)));
        let e2 = w.spawn_bundle((Mass(3.0), Charge(4.0)));
        assert_ne!(e1, e2);
        assert_eq!(w.entity_count(), 2);
        assert_eq!(w.get_static::<Mass>(e1), Some(&Mass(1.0)));
        assert_eq!(w.get_static::<Mass>(e2), Some(&Mass(3.0)));
    }

    #[test]
    fn spawn_bundle_repeated_calls_share_one_archetype_and_dont_corrupt_rows() {
        // Every entity after the first lands in the *same*, already-
        // created archetype -- this is what exercises the `to_id`
        // lookup finding a real, existing archetype rather than only
        // ever creating a fresh one.
        let mut w = World::new();
        let es: Vec<_> = (0..5u32)
            .map(|i| w.spawn_bundle((Mass(i as f32), Charge(i as f32 * 10.0))))
            .collect();
        for (i, &e) in es.iter().enumerate() {
            assert_eq!(w.get_static::<Mass>(e), Some(&Mass(i as f32)));
            assert_eq!(w.get_static::<Charge>(e), Some(&Charge(i as f32 * 10.0)));
        }
    }

    #[test]
    fn spawn_bundle_then_remove_bundle_round_trips() {
        let mut w = World::new();
        let e = w.spawn_bundle((Mass(7.0), Charge(8.0)));
        let removed = w.remove_bundle::<(Mass, Charge)>(e).unwrap();
        assert_eq!(removed, (Mass(7.0), Charge(8.0)));
        assert!(!w.has_static::<Mass>(e));
        assert!(!w.has_static::<Charge>(e));
    }

    #[test]
    fn spawn_bundle_and_spawn_then_insert_bundle_reach_the_same_archetype() {
        // Mixing the two call paths for the same bundle type must land
        // in one shared archetype, not two accidentally-distinct ones
        // -- `spawn_bundle`'s own `edge_for_insert` chain has to agree
        // with `insert_bundle`'s.
        let mut w = World::new();
        let via_direct = w.spawn_bundle((Mass(1.0), Charge(2.0)));
        let via_two_step = w.spawn();
        w.insert_bundle(via_two_step, (Mass(3.0), Charge(4.0)));
        assert_eq!(w.get_static::<Mass>(via_direct), Some(&Mass(1.0)));
        assert_eq!(w.get_static::<Mass>(via_two_step), Some(&Mass(3.0)));
        // Query over the archetype-tracked storage must see both --
        // only true if they share one archetype's table.
        let seen: std::collections::HashSet<_> = w
            .query2_static::<Mass, Charge>()
            .map(|(e, _, _)| e)
            .collect();
        assert!(seen.contains(&via_direct));
        assert!(seen.contains(&via_two_step));
        assert_eq!(seen.len(), 2);
    }

    #[test]
    fn spawn_bundle_drops_values_exactly_once() {
        use std::cell::Cell;
        use std::rc::Rc;

        struct Tracked(Rc<Cell<u32>>);
        impl Drop for Tracked {
            fn drop(&mut self) {
                self.0.set(self.0.get() + 1);
            }
        }

        let drops = Rc::new(Cell::new(0));
        let mut w = World::new();
        let e = w.spawn_bundle((Tracked(drops.clone()), Mass(1.0)));
        assert_eq!(
            drops.get(),
            0,
            "spawning must move the value, never drop it"
        );
        assert!(w.despawn(e));
        assert_eq!(
            drops.get(),
            1,
            "dropped exactly once, when the entity is despawned"
        );
    }

    // ── Resources ───────────────────────────────────────────────────

    #[derive(Debug, Clone, Copy, PartialEq)]
    struct Clock {
        secs: f64,
    }

    #[test]
    fn resources_insert_get_mutate_remove_through_the_world() {
        let mut w = World::new();
        assert!(!w.contains_resource::<Clock>());
        assert_eq!(w.get_resource::<Clock>(), None);

        assert_eq!(w.insert_resource(Clock { secs: 1.0 }), None);
        assert!(w.contains_resource::<Clock>());
        w.get_resource_mut::<Clock>().unwrap().secs += 0.5;
        assert_eq!(w.get_resource::<Clock>(), Some(&Clock { secs: 1.5 }));

        assert_eq!(
            w.insert_resource(Clock { secs: 9.0 }),
            Some(Clock { secs: 1.5 })
        );
        assert_eq!(w.remove_resource::<Clock>(), Some(Clock { secs: 9.0 }));
        assert!(!w.contains_resource::<Clock>());
    }

    #[test]
    fn a_type_can_be_a_resource_and_a_component_at_once() {
        let mut w = World::new();
        let e = w.spawn();
        assert!(w.insert_static(e, Clock { secs: 1.0 }));
        w.insert_resource(Clock { secs: 2.0 });
        assert_eq!(w.get_static::<Clock>(e), Some(&Clock { secs: 1.0 }));
        assert_eq!(w.get_resource::<Clock>(), Some(&Clock { secs: 2.0 }));
    }

    #[test]
    fn resources_are_untouched_by_entity_lifecycle() {
        let mut w = World::new();
        w.insert_resource(Clock { secs: 3.0 });
        let e = w.spawn();
        assert!(w.despawn(e));
        assert_eq!(w.get_resource::<Clock>(), Some(&Clock { secs: 3.0 }));
        assert_eq!(w.entity_count(), 0);
    }

    #[test]
    fn ffi_resource_write_and_read_round_trip_through_the_world() {
        #[derive(Debug, Clone, Copy, PartialEq, FromBytes, IntoBytes, KnownLayout, Immutable)]
        #[repr(C)]
        struct Wind {
            x: f32,
            y: f32,
        }
        let mut w = World::new();
        let id = w.register_ffi_resource::<Wind>("Wind");
        assert_eq!(w.lookup_ffi_resource_id("Wind"), Some(id));
        assert_eq!(w.resource_raw_span(id).unwrap().count, 0);

        let bytes = Wind { x: 1.0, y: -2.0 }.as_bytes().to_vec();
        w.write_resource_bytes(id, &bytes).unwrap();
        assert_eq!(w.get_resource::<Wind>(), Some(&Wind { x: 1.0, y: -2.0 }));
        assert_eq!(w.resource_raw_span(id).unwrap().count, 1);

        assert_eq!(
            w.write_resource_bytes(id, &bytes[..4]),
            Err(ResourceFfiError::SizeMismatch)
        );
        assert_eq!(w.remove_ffi_resource(id), Ok(()));
        assert_eq!(w.remove_ffi_resource(id), Err(ResourceFfiError::Absent));
        assert_eq!(w.get_resource::<Wind>(), None);
    }

    // ── Change detection: Tick / ChangeTracker / Added / Changed ────

    use crate::{ChangeTracker, Tick};

    #[derive(Debug, Clone, Copy, PartialEq)]
    struct Vigor {
        hp: u32,
    }
    #[derive(Debug, Clone, Copy, PartialEq)]
    struct Marker2;

    #[test]
    fn new_world_starts_at_tick_one_and_increment_advances_it() {
        let mut w = World::new();
        assert_eq!(w.change_tick(), Tick::new(1));
        assert_eq!(w.increment_change_tick(), Tick::new(2));
        assert_eq!(w.change_tick(), Tick::new(2));
        assert_eq!(w.increment_change_tick(), Tick::new(3));
    }

    #[test]
    fn a_fresh_tracker_sees_every_component_already_present_as_added_and_changed() {
        let mut w = World::new();
        let e1 = w.spawn_bundle((Vigor { hp: 1 },));
        let e2 = w.spawn_bundle((Vigor { hp: 2 },));
        w.increment_change_tick();

        let tracker = ChangeTracker::new();
        let mut added: Vec<Entity> = w.query_added::<Vigor>(&tracker).map(|(e, _)| e).collect();
        added.sort_by_key(|e| e.index());
        let mut expected = vec![e1, e2];
        expected.sort_by_key(|e| e.index());
        assert_eq!(added, expected);

        let mut changed: Vec<Entity> = w.query_changed::<Vigor>(&tracker).map(|(e, _)| e).collect();
        changed.sort_by_key(|e| e.index());
        assert_eq!(changed, expected);
    }

    #[test]
    fn after_update_a_tracker_no_longer_sees_old_insertions_only_new_ones() {
        let mut w = World::new();
        let old = w.spawn_bundle((Vigor { hp: 1 },));
        w.increment_change_tick();

        let mut tracker = ChangeTracker::new();
        assert_eq!(w.query_added::<Vigor>(&tracker).count(), 1);
        tracker.update(&w);
        assert_eq!(
            w.query_added::<Vigor>(&tracker).count(),
            0,
            "already seen, and nothing new since"
        );

        w.increment_change_tick();
        let new = w.spawn_bundle((Vigor { hp: 2 },));
        let found: Vec<Entity> = w.query_added::<Vigor>(&tracker).map(|(e, _)| e).collect();
        assert_eq!(found, vec![new]);
        assert!(!found.contains(&old));
    }

    #[test]
    fn get_static_mut_marks_changed_but_not_added() {
        let mut w = World::new();
        let e = w.spawn_bundle((Vigor { hp: 1 },));
        w.increment_change_tick();

        let mut tracker = ChangeTracker::new();
        tracker.update(&w); // now last_run == the tick Vigor was inserted at

        assert_eq!(w.query_added::<Vigor>(&tracker).count(), 0);
        assert_eq!(w.query_changed::<Vigor>(&tracker).count(), 0);

        w.increment_change_tick();
        w.get_static_mut::<Vigor>(e).unwrap().hp = 99;

        assert_eq!(
            w.query_added::<Vigor>(&tracker).count(),
            0,
            "mutation is not a re-add"
        );
        let changed: Vec<Entity> = w.query_changed::<Vigor>(&tracker).map(|(e, _)| e).collect();
        assert_eq!(changed, vec![e]);
        assert_eq!(w.get_static::<Vigor>(e), Some(&Vigor { hp: 99 }));
    }

    #[test]
    fn a_read_only_get_static_never_marks_changed() {
        let mut w = World::new();
        let e = w.spawn_bundle((Vigor { hp: 1 },));
        w.increment_change_tick();
        let mut tracker = ChangeTracker::new();
        tracker.update(&w);

        w.increment_change_tick();
        let _ = w.get_static::<Vigor>(e);
        assert_eq!(w.query_changed::<Vigor>(&tracker).count(), 0);
    }

    #[test]
    fn structural_migration_preserves_ticks_of_surviving_components() {
        // Adding Marker2 to `e` moves it to a new archetype. Vigor's own
        // ticks must migrate with it unchanged -- gaining an unrelated
        // component must not look like Vigor was re-added or changed.
        let mut w = World::new();
        let e = w.spawn_bundle((Vigor { hp: 1 },));
        w.increment_change_tick();

        let mut tracker = ChangeTracker::new();
        tracker.update(&w);

        w.increment_change_tick();
        assert!(w.insert_static(e, Marker2));

        assert_eq!(
            w.query_added::<Vigor>(&tracker).count(),
            0,
            "Vigor wasn't re-added by an unrelated structural change"
        );
        assert_eq!(w.query_changed::<Vigor>(&tracker).count(), 0);
        // Marker2 itself, inserted after tracker's last_run, does show up
        // as newly added on its own type.
        let added_marker: Vec<Entity> =
            w.query_added::<Marker2>(&tracker).map(|(e, _)| e).collect();
        assert_eq!(added_marker, vec![e]);
    }

    #[test]
    fn removing_an_unrelated_component_also_preserves_survivor_ticks() {
        let mut w = World::new();
        let e = w.spawn_bundle((Vigor { hp: 1 }, Marker2));
        w.increment_change_tick();
        let mut tracker = ChangeTracker::new();
        tracker.update(&w);

        w.increment_change_tick();
        assert_eq!(w.remove_static::<Marker2>(e), Some(Marker2));

        assert_eq!(
            w.query_changed::<Vigor>(&tracker).count(),
            0,
            "Vigor untouched by removing Marker2"
        );
        assert_eq!(w.get_static::<Vigor>(e), Some(&Vigor { hp: 1 }));
    }

    #[test]
    fn query_added_and_changed_are_empty_for_a_never_registered_type() {
        let w = World::new();
        let tracker = ChangeTracker::new();
        assert_eq!(w.query_added::<Vigor>(&tracker).count(), 0);
        assert_eq!(w.query_changed::<Vigor>(&tracker).count(), 0);
    }

    #[test]
    fn ticks_stay_row_aligned_when_a_despawn_swaps_a_mutated_entity_into_the_freed_slot() {
        // Adversarial regression coverage for the row-alignment risk in
        // threading `Table::ticks` through `Table::swap_remove_row`:
        // `Vec::swap_remove(0)` moves the table's LAST row into row 0, so
        // despawning entity 0 while entity 5 (the last row, and the one
        // we mutated) is the one that lands in the freed slot is the
        // case that actually exposes a tick vector that didn't shrink or
        // shrank at the wrong index -- a merely-unshrunk-but-never-read
        // tick vector, or one shrunk at the wrong row, would make the
        // *swapped-in* entity read back with the *despawned* entity's
        // old, never-changed ticks instead of its own.
        let mut w = World::new();
        let entities: Vec<Entity> = (0..6).map(|i| w.spawn_bundle((Vigor { hp: i },))).collect();
        w.increment_change_tick();
        let mut tracker = ChangeTracker::new();
        tracker.update(&w);

        w.increment_change_tick();
        w.get_static_mut::<Vigor>(entities[1]).unwrap().hp = 100;
        w.get_static_mut::<Vigor>(entities[5]).unwrap().hp = 500;

        // Despawn entity 0: `entities`/`columns`/`ticks` must each swap
        // the table's last row (entity 5's) into row 0.
        assert!(w.despawn(entities[0]));

        let mut changed: Vec<Entity> = w.query_changed::<Vigor>(&tracker).map(|(e, _)| e).collect();
        changed.sort_by_key(|e| e.index());
        let mut expected = vec![entities[1], entities[5]];
        expected.sort_by_key(|e| e.index());
        assert_eq!(
            changed, expected,
            "entity 5's own changed tick must follow it into row 0, not entity 0's stale one"
        );

        // And the surviving values are still each entity's own, not
        // shuffled by the swap-remove.
        assert_eq!(w.get_static::<Vigor>(entities[1]), Some(&Vigor { hp: 100 }));
        assert_eq!(w.get_static::<Vigor>(entities[5]), Some(&Vigor { hp: 500 }));
        assert_eq!(w.get_static::<Vigor>(entities[2]), Some(&Vigor { hp: 2 }));
        assert_eq!(w.get_static::<Vigor>(entities[3]), Some(&Vigor { hp: 3 }));
        assert_eq!(w.get_static::<Vigor>(entities[4]), Some(&Vigor { hp: 4 }));
        assert_eq!(w.get_static::<Vigor>(entities[0]), None);
    }

    #[test]
    fn ticks_stay_row_aligned_through_a_structural_migration_swap_remove() {
        // Same adversarial shape as the despawn test above, but for the
        // `insert<T>`/`insert_bundle` survivor-migration path instead of
        // `Table::swap_remove_row`: giving entity 0 an unrelated Marker2
        // migrates it to a new archetype, which swap-removes entity 0's
        // OLD row the same way a despawn would -- entity 5 (mutated,
        // last row) lands in the freed slot and must keep its own tick,
        // not inherit entity 0's.
        let mut w = World::new();
        let entities: Vec<Entity> = (0..6).map(|i| w.spawn_bundle((Vigor { hp: i },))).collect();
        w.increment_change_tick();
        let mut tracker = ChangeTracker::new();
        tracker.update(&w);

        w.increment_change_tick();
        w.get_static_mut::<Vigor>(entities[5]).unwrap().hp = 500;
        assert!(w.insert_static(entities[0], Marker2));

        let changed: Vec<Entity> = w.query_changed::<Vigor>(&tracker).map(|(e, _)| e).collect();
        assert_eq!(changed, vec![entities[5]]);
        assert_eq!(w.get_static::<Vigor>(entities[5]), Some(&Vigor { hp: 500 }));
        assert_eq!(w.get_static::<Vigor>(entities[0]), Some(&Vigor { hp: 0 }));
    }

    use crate::tick::MAX_CHANGE_AGE;

    // ── resource change detection and the tick scan ──────────────────

    #[derive(Debug, PartialEq)]
    struct Cfg(u32);
    #[derive(Debug, PartialEq)]
    struct W1(u32);
    #[derive(Debug, PartialEq)]
    struct W2(u32);

    #[derive(
        Debug,
        Clone,
        Copy,
        PartialEq,
        zerocopy::FromBytes,
        zerocopy::IntoBytes,
        zerocopy::KnownLayout,
        zerocopy::Immutable,
    )]
    #[repr(C)]
    struct FfiCfg {
        v: u32,
    }

    #[test]
    fn resource_added_and_changed_follow_a_tracker() {
        let mut w = World::new();
        let mut t = ChangeTracker::new();
        assert!(!w.resource_added::<Cfg>(&t) && !w.resource_changed::<Cfg>(&t), "none yet");

        w.insert_resource(Cfg(1));
        assert!(w.resource_added::<Cfg>(&t) && w.resource_changed::<Cfg>(&t));

        t.update(&w);
        w.increment_change_tick();
        assert!(!w.resource_added::<Cfg>(&t) && !w.resource_changed::<Cfg>(&t));

        // Replacing is a change, not an addition.
        w.insert_resource(Cfg(2));
        assert!(!w.resource_added::<Cfg>(&t) && w.resource_changed::<Cfg>(&t));

        t.update(&w);
        w.increment_change_tick();
        let _ = w.get_resource::<Cfg>();
        assert!(!w.resource_changed::<Cfg>(&t), "reading is not a change");
        let _ = w.get_resource_mut::<Cfg>();
        assert!(!w.resource_added::<Cfg>(&t) && w.resource_changed::<Cfg>(&t));

        // Removed and inserted again: a new addition.
        t.update(&w);
        w.increment_change_tick();
        w.remove_resource::<Cfg>();
        assert!(!w.resource_changed::<Cfg>(&t), "gone");
        w.insert_resource(Cfg(3));
        assert!(w.resource_added::<Cfg>(&t) && w.resource_changed::<Cfg>(&t));
    }

    #[test]
    fn a_resource_written_from_c_is_added_when_new_and_changed_when_replaced() {
        let mut w = World::new();
        let id = w.register_ffi_resource::<FfiCfg>("FfiCfg");
        let mut t = ChangeTracker::new();
        w.write_resource_bytes(id, &7u32.to_ne_bytes()).unwrap();
        assert!(w.resource_added::<FfiCfg>(&t));
        t.update(&w);
        w.increment_change_tick();
        w.write_resource_bytes(id, &8u32.to_ne_bytes()).unwrap();
        assert!(!w.resource_added::<FfiCfg>(&t) && w.resource_changed::<FfiCfg>(&t));
        t.update(&w);
        w.increment_change_tick();
        assert!(w.write_resource_bytes(id, &[1]).is_err());
        assert!(!w.resource_changed::<FfiCfg>(&t), "a refused write stamps nothing");
    }

    #[test]
    fn check_change_ticks_clamps_components_resources_and_the_removal_log() {
        let mut w = World::new();
        w.set_change_tick(Tick::new(100));
        let e = w.spawn();
        w.insert_static(e, W1(1));
        w.insert_resource(Cfg(1));
        w.track_removals::<W2>();
        let e2 = w.spawn();
        w.insert_static(e2, W2(1));
        w.remove_static::<W2>(e2);

        // 3.5e9 ticks later: everything stamped at tick 100 is older than
        // MAX_CHANGE_AGE. A tracker that last ran MAX_CHANGE_AGE + 1 ticks
        // ago is "newer" than a clamped stamp but not than the real one.
        let now = Tick::new(100u32.wrapping_add(3_500_000_000));
        w.set_change_tick(now);
        let tracker = ChangeTracker::at(Tick::new(now.get().wrapping_sub(MAX_CHANGE_AGE + 1)));
        let seen = |w: &mut World| {
            (
                w.query_data_filtered_since::<Entity, crate::Changed<W1>>(&tracker)
                    .count(),
                w.resource_changed::<Cfg>(&tracker),
                w.removed_since::<W2>(&tracker).count(),
            )
        };
        assert_eq!(seen(&mut w), (0, false, 0), "unchecked, they read as very old");

        w.check_change_ticks();
        assert_eq!(seen(&mut w), (1, true, 1), "clamped to exactly MAX_CHANGE_AGE");
        assert_eq!(w.last_check_tick(), now);

        // A tracker exactly MAX_CHANGE_AGE old is *not* older than the clamp.
        let at_limit = ChangeTracker::at(Tick::new(now.get().wrapping_sub(MAX_CHANGE_AGE)));
        assert_eq!(
            w.query_data_filtered_since::<Entity, crate::Changed<W1>>(&at_limit)
                .count(),
            0
        );
        // And a second scan changes nothing.
        w.check_change_ticks();
        assert_eq!(seen(&mut w), (1, true, 1));
    }

    #[test]
    fn increment_change_tick_runs_the_scan_when_due_and_not_before() {
        let mut w = World::new();
        assert_eq!(w.last_check_tick(), Tick::ZERO);
        w.set_change_tick(Tick::new(CHECK_TICK_THRESHOLD - 2));
        w.increment_change_tick();
        assert_eq!(w.last_check_tick(), Tick::ZERO, "one tick short of due");
        let now = w.increment_change_tick();
        assert_eq!(now, Tick::new(CHECK_TICK_THRESHOLD));
        assert_eq!(w.last_check_tick(), now, "due: scanned");
        w.increment_change_tick();
        assert_eq!(w.last_check_tick(), now, "not due again for a whole threshold");
    }

    /// Two worlds see the same history: a component stamped once, then a
    /// long wait of 2^32 + 5 million ticks in 520-million-tick jumps (each
    /// followed by the `increment_change_tick` a game loop makes). One
    /// world's loop increments, so it scans on the way; the other's does
    /// not. The stamp must read as old in the first and, without a scan,
    /// wrongly new in the second.
    #[test]
    fn a_world_that_scans_on_the_way_keeps_old_values_old_across_a_full_wrap() {
        fn world_after_the_wait(increment_each_jump: bool) -> World {
            let mut w = World::new();
            w.set_change_tick(Tick::new(100));
            let e = w.spawn();
            w.insert_static(e, W1(1));
            for k in 1..=8u32 {
                w.set_change_tick(Tick::new(100 + k * 520_000_000));
                if increment_each_jump {
                    w.increment_change_tick();
                }
            }
            w.set_change_tick(Tick::new(100u32.wrapping_add(5_000_000)));
            w
        }
        let changed = |w: &mut World| {
            let now = w.change_tick();
            let tracker = ChangeTracker::at(Tick::new(now.get().wrapping_sub(10_000_000)));
            w.query_data_filtered_since::<Entity, crate::Changed<W1>>(&tracker)
                .count()
        };
        assert_eq!(changed(&mut world_after_the_wait(true)), 0, "scanned: still old");
        assert_eq!(
            changed(&mut world_after_the_wait(false)),
            1,
            "without a scan the wrapped age looks new: the bug"
        );
    }
}
