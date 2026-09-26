//! The Archetype Core — dense, SoA table storage with real dynamic
//! migration between archetypes as an entity's component set changes.
//!
//! This is the "Static Core" half of the Hybrid ECS Architecture
//! (`docs/mid-ecs.md`), the counterpart to the Sparse Shell
//! (`component.rs`). An entity's *archetype* is the exact set of
//! archetype-tracked component types it currently has; every entity with
//! the same set lives in the same `Table` — one contiguous `Vec<T>`
//! per component type, all in lockstep by row — so iterating a table is
//! a straight cache-friendly scan, no per-entity type dispatch. Adding or
//! removing a component moves the entity's whole row from its old table
//! to a new one, real migration, not a simplification of it — see the
//! note on scope below for what "real" means here specifically.
//!
//! # Grounded in Bevy ECS's real source, read directly, not from memory
//!
//! Cloned `bevyengine/bevy` and read `archetype.rs` (1002 lines) and
//! `storage/table/{mod,column}.rs` (1428 lines) directly before writing
//! any of this. Confirmed, not assumed: `Table` really is
//! `{ columns: ImmutableSparseSet<ComponentId, Column>, entities:
//! Vec<Entity> }` — the same `SparseSet<ComponentId, _>`-keyed shape
//! `component.rs`'s `SparseShell` converged on independently for its own
//! columns, now confirmed twice over as the right structure for this
//! class of problem, not a coincidence. `Edges` (`archetype.rs`) really
//! does memoize "if I insert this bundle while in archetype A, go to
//! archetype B" per source archetype, specifically so a repeated
//! structural change doesn't re-sort a signature and re-hash a lookup
//! every single time — adopted here as a plain
//! `HashMap<ComponentId, ArchetypeId>` per archetype (`Archetype::
//! add_edges`/`remove_edges`), the same idea, simpler storage since this
//! project has no `SparseArray`-without-dense-iteration primitive built
//! and building one solely for this would be over-engineering a small,
//! low-frequency-access need.
//!
//! # Where this deliberately diverges from Bevy, and why
//!
//! Bevy's real row-migration (`Tables::move_row`) is a sorted
//! merge-join over two tables' columns using raw pointers and `unsafe`
//! non-overlapping copies, plus change-detection ticks this project
//! doesn't have yet. Copying that technique wholesale would mean
//! importing a large amount of `unsafe` for a perf technique with no
//! profiled need here — directly against this project's own established
//! precedent (`SparseSet`, `GenerationalIndexAllocator`: zero `unsafe`,
//! by choice, revisit only against a real profile). This module gets the
//! same real capability — full dynamic migration, any entity, any
//! component, at any time — through safe Rust instead: each migrated
//! value moves through `Column::move_row_to`, which swap-removes it from
//! the source column and pushes it onto the destination, both typed
//! inside the one generic impl, rather than being copied via a raw
//! pointer. No `unsafe` and no heap allocation (an earlier version boxed
//! every migrated value as `Box<dyn Any>`; `docs/mid-ecs.md` has why
//! that was replaced and what it measured).
//!
//! Bevy also unifies sparse-set-stored and table-stored components under
//! one `Archetype` concept, because component storage strategy is a
//! per-component, developer-chosen attribute in Bevy's design. This
//! project keeps that split hard and architectural instead — Sparse
//! Shell and Archetype Core are two independent systems, not one
//! archetype concept covering both — so `Archetype` here maps 1:1 onto
//! exactly one `Table`, with no separate archetype-row-vs-table-row
//! indirection the way Bevy needs for its own unified design.
//!
//! Single-component structural changes (`World::insert_static`/
//! `remove_static`, one `T` at a time) plus atomic multi-component
//! `Bundle`-style structural changes (`World::insert_bundle`/
//! `remove_bundle`, several `T`s as one migration). Both rely on the
//! same real property: a single-component add (or a `Bundle` insert of
//! several genuinely new components) always makes the destination
//! signature a strict superset of the source, and a single-component
//! remove (or a `Bundle` remove of several genuinely present
//! components) always makes it a strict subset — which is what lets
//! the migration code below move every *other* column unconditionally
//! without a merge-join to figure out which columns are actually
//! shared, whether one component is changing or several are at once.
//! `Bundle` was deliberately deferred past the initial single-component
//! pass ("not needed yet, a real extension once something in this
//! workspace actually needs it") — this is that extension, once
//! spawning entities with a known, fixed component set at creation
//! time (the actual common case, per Bevy's own real `Bundle` design)
//! became a real, present need rather than a speculative one.

use std::any::{Any, TypeId};
use std::collections::HashMap;

use mid_collections::{FfiSpan, SparseSet, SparseSetIndex};
use zerocopy::{Immutable, IntoBytes, KnownLayout};

use crate::component::ComponentId;
use crate::hash::{DenseIdMap, TypeIdMap};
use crate::tick::{ComponentTicks, Tick};
use crate::world::Entity;

/// Type-erased accessor for reading one archetype's column as a raw
/// [`FfiSpan`], without the caller needing to know the concrete
/// component type. Identical mechanism and identical reasoning to
/// `component.rs`'s own `FfiSpanAccessor` — see that module's doc
/// comment for the full explanation of why this exists as a plain `fn`
/// pointer rather than a required trait method. Kept as a distinct type
/// (not reused from `component.rs`) because it closes over `Vec<T>`
/// directly here (`Column`'s own concrete backing type), not
/// `SparseSet<Entity, T>` — a real, if small, difference in what the
/// two systems actually store.
type FfiSpanAccessor = fn(&dyn Any) -> FfiSpan;

// Query iterators live in the child module `iter` (crates/mid-ecs/src/
// archetype/iter.rs). Child, not sibling: `Table::columns`, `Table::
// entities`, the `Column` trait and `Archetype` itself are all private to
// this module, and a child can see its parent's private items while a
// sibling cannot. Nothing had to be made more visible to move them there.
mod iter;
pub(crate) use iter::{Iter1, Iter1Ref, Iter2, Iter2Ref};

/// Dense identifier for one archetype (one exact component-type set).
/// `ArchetypeId(0)` is always the empty archetype — every entity starts
/// there at spawn, before any archetype-tracked component is ever
/// attached to it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ArchetypeId(u32);

impl ArchetypeId {
    /// Unpacks this id as a plain `u32`, for crossing the FFI boundary.
    /// Same shape as `ComponentId::as_u32`, for the same reason —
    /// `ffi.rs` needs a real accessor since this field is private even
    /// within this crate (more so than `ComponentId`'s own
    /// `pub(crate)` field), by design: nothing outside this module
    /// should construct an `ArchetypeId` except through a real
    /// `Archetypes` operation, and this accessor doesn't weaken that —
    /// it only ever hands out the `u32` of an `ArchetypeId` that
    /// already exists.
    #[inline]
    pub fn as_u32(self) -> u32 {
        self.0
    }

    /// Reconstructs an `ArchetypeId` from a `u32` produced by
    /// [`Self::as_u32`]. Safe even for a bogus value — every real
    /// `Archetypes` method looks it up in `self.archetypes` before
    /// touching anything, so a bogus id just reads back as "doesn't
    /// exist" everywhere it's used, never as some other real archetype.
    #[inline]
    pub fn from_u32(value: u32) -> Self {
        Self(value)
    }
}

impl SparseSetIndex for ArchetypeId {
    #[inline]
    fn sparse_index(&self) -> u32 {
        self.0
    }
}

/// Type-erased operations one dense column (one component type, within
/// one [`Table`]) supports. Not meant to be implemented outside this
/// module — the blanket impl below covers every real case.
trait Column: Any {
    /// Removes the value at `row` (swap-remove: the last element takes
    /// its place) and drops it.
    fn swap_remove_and_drop(&mut self, row: usize);
    /// Removes the value at `row` (swap-remove) and appends it to
    /// `dest`, which must be a column of the same concrete type. No
    /// boxing: both ends are typed inside the one generic impl. Panics
    /// (via `expect`, an internal invariant violation, not user-facing
    /// misuse — every caller obtains `dest` from `new_same_type` on this
    /// very column, or from a column already registered under the same
    /// `ComponentId`) if `dest` isn't actually a `Vec<T>` of this `T`.
    fn move_row_to(&mut self, row: usize, dest: &mut dyn Column);
    /// Creates a new, empty column of the same concrete type as this
    /// one — this is what lets a brand-new archetype's table acquire
    /// correctly-typed columns lazily, purely by copying the type of
    /// whatever's being migrated into it, with no separate
    /// `ComponentId -> Type` registry needed anywhere.
    fn new_same_type(&self) -> Box<dyn Column>;
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

impl<T: 'static> Column for Vec<T> {
    fn swap_remove_and_drop(&mut self, row: usize) {
        Vec::swap_remove(self, row);
    }
    fn move_row_to(&mut self, row: usize, dest: &mut dyn Column) {
        let value = Vec::swap_remove(self, row);
        dest.as_any_mut()
            .downcast_mut::<Vec<T>>()
            .expect("Column<T>::move_row_to called with a destination of the wrong concrete type")
            .push(value);
    }
    fn new_same_type(&self) -> Box<dyn Column> {
        Box::<Vec<T>>::default()
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// Dense SoA storage for every entity sharing one exact archetype: one
/// [`Column`] per component type, one [`Entity`] per row, all in
/// lockstep. See this module's doc comment for the full design.
pub(crate) struct Table {
    columns: SparseSet<ComponentId, Box<dyn Column>>,
    /// Per-component, row-aligned with `columns`' own `Vec<T>` at the
    /// same `ComponentId` key -- always the same key set and the same
    /// length as that column, maintained by `ensure_ticks` alongside
    /// every `ensure_column` call and updated everywhere a row is
    /// pushed, migrated, or swap-removed. See `tick.rs`.
    ticks: SparseSet<ComponentId, Vec<ComponentTicks>>,
    entities: Vec<Entity>,
}

impl Table {
    fn new() -> Self {
        Self {
            columns: SparseSet::new(),
            ticks: SparseSet::new(),
            entities: Vec::new(),
        }
    }

    fn len(&self) -> usize {
        self.entities.len()
    }

    /// Removes the row for `row`, dropping every column's value there.
    /// Returns the entity that got swapped into `row`'s old position, if
    /// any (i.e. if `row` wasn't already the last row) — the caller must
    /// update *that* entity's own tracked row to match, or its location
    /// goes stale.
    fn swap_remove_row(&mut self, row: usize) -> Option<Entity> {
        let last = self.entities.len() - 1;
        for (_, column) in self.columns.iter_mut() {
            column.swap_remove_and_drop(row);
        }
        for (_, ticks) in self.ticks.iter_mut() {
            ticks.swap_remove(row);
        }
        self.entities.swap_remove(row);
        if row == last {
            None
        } else {
            Some(self.entities[row])
        }
    }
}

/// One archetype: an exact, sorted set of archetype-tracked component
/// types, the [`Table`] holding every entity with exactly that set, and
/// a memoized cache of where a single-component add/remove from here
/// leads — see this module's doc comment on `Edges`/Bevy for why the
/// cache exists.
struct Archetype {
    component_ids: Vec<ComponentId>,
    table: Table,
    add_edges: DenseIdMap<ComponentId, ArchetypeId>,
    remove_edges: DenseIdMap<ComponentId, ArchetypeId>,
}

impl Archetype {
    fn empty() -> Self {
        Self {
            component_ids: Vec::new(),
            table: Table::new(),
            add_edges: DenseIdMap::default(),
            remove_edges: DenseIdMap::default(),
        }
    }

    fn with_signature(component_ids: Vec<ComponentId>) -> Self {
        Self {
            component_ids,
            table: Table::new(),
            add_edges: DenseIdMap::default(),
            remove_edges: DenseIdMap::default(),
        }
    }
}

/// Where one entity's archetype-tracked data currently lives: which
/// archetype, and which row of that archetype's table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct EntityLocation {
    archetype_id: ArchetypeId,
    row: usize,
}

/// Several component types inserted or removed as one atomic
/// structural change — see `insert_bundle`/`remove_bundle` on
/// [`Archetypes`], and the module doc comment above for why this
/// doesn't need a merge-join even though it moves several columns at
/// once. Implemented for tuples `(A,)` through `(A, B, C, D, E, F, G,
/// H)` below, via a macro — not hand-written per arity, and not
/// implemented for the unit type `()`: a zero-component bundle isn't a
/// meaningful structural change to atomically apply, and every real
/// caller already has at least one component in mind.
///
/// `pub(crate)`, deliberately, even though `World::insert_bundle`/
/// `remove_bundle` (see `world.rs`) are real public API and reference
/// it in their own `B: Bundle` bound. Tried making this genuinely
/// `pub` first, including the standard sealed-supertrait pattern (an
/// empty `pub trait Bundle: sealed::SealedBundle {}` with the real
/// methods on a `SealedBundle` inside a private module) — it doesn't
/// actually avoid the warning here, because the supertrait bound
/// itself makes `SealedBundle` transitively reachable from `Bundle`'s
/// own public bound, which is what the lint is actually checking, not
/// the enclosing module's privacy. The real fix would be widening
/// `Table`/`Archetypes` to `pub` so every `Bundle` method's signature
/// only mentions already-public types — the wrong tradeoff: those two
/// staying internal is far more architecturally important than
/// silencing this one lint. Net effect, accepted deliberately: a
/// downstream crate can still call `world.insert_bundle(e, (a, b))`
/// directly with concrete tuple types (that works fine, `Bundle`'s own
/// visibility doesn't block the call), it just can't write its own
/// function generic over `B: Bundle` — a real, narrow limitation,
/// flagged here rather than worked around at real architectural cost.
/// Fixed-capacity, stack-allocated replacement for `Vec<ComponentId>` in
/// `Bundle::component_ids`/`existing_component_ids`. `impl_bundle_for_tuple!`
/// below only ever generates arities 1..=8, so 8 inline slots plus a
/// real length cover every case this crate actually produces, with no
/// heap allocation at all. Not a guess: a real allocation-counting
/// diagnostic (`examples/diag_alloc_count.rs`, see
/// `docs/mid-ecs.md`) confirmed this `Vec` was a real, unconditional
/// allocation on *every* `insert_bundle`/`remove_bundle` call, every
/// arity — the single most universal allocation source these two
/// methods had, since it fires on every structural change regardless
/// of what else that change needs to migrate.
///
/// `ComponentId::from_u32(0)` as the unused-slot filler is deliberate,
/// not arbitrary — its own doc comment already establishes that a
/// bogus, never-registered value is safe to hold and simply reads back
/// as "not registered" anywhere it's (incorrectly) read; `Deref`/
/// `DerefMut` below bound every real access to `..len`, so those slots
/// are never actually read regardless.
#[derive(Clone, Copy)]
pub(crate) struct ComponentIdList {
    buf: [ComponentId; Self::CAP],
    len: u8,
}

impl ComponentIdList {
    const CAP: usize = 8;

    fn new() -> Self {
        Self {
            buf: [ComponentId::from_u32(0); Self::CAP],
            len: 0,
        }
    }

    /// Panics past 8 elements — a real bug, not an input to handle
    /// gracefully, since `impl_bundle_for_tuple!` never generates more
    /// than 8 and every caller of `push` is that macro's own generated
    /// code.
    fn push(&mut self, id: ComponentId) {
        self.buf[self.len as usize] = id;
        self.len += 1;
    }
}

impl std::ops::Deref for ComponentIdList {
    type Target = [ComponentId];
    fn deref(&self) -> &[ComponentId] {
        &self.buf[..self.len as usize]
    }
}

impl std::ops::DerefMut for ComponentIdList {
    fn deref_mut(&mut self) -> &mut [ComponentId] {
        &mut self.buf[..self.len as usize]
    }
}

pub(crate) trait Bundle: Sized + 'static {
    /// Component ids for every element, in the same tuple-position
    /// order every other method here expects them back in —
    /// registering each type in `archetypes`'s own numbering space on
    /// first use. Matches plain `insert::<T>`'s own registering
    /// `Archetypes::component_id::<T>()` convention (see
    /// `World::insert_static`), generalized to every element.
    fn component_ids(archetypes: &mut Archetypes) -> ComponentIdList;

    /// Same ids, without registering — `None` if *any* element was
    /// never registered as an archetype-tracked component for
    /// anything. Matches plain `remove::<T>`'s own non-registering
    /// `Archetypes::existing_component_id::<T>()` convention (see
    /// `World::remove_static`), generalized: one missing registration
    /// anywhere in the bundle means the whole bundle can't possibly be
    /// present on any entity, by construction.
    fn existing_component_ids(archetypes: &Archetypes) -> Option<ComponentIdList>;

    /// Pushes every element into its matching column of `table`, in
    /// the exact order `ids` gives them back in. `table` must already
    /// have a same-typed column ensured for every id in `ids` — by the
    /// time `insert_bundle` calls this, the target archetype (and
    /// therefore its columns) already exists, so this only ever
    /// appends, never creates a table.
    fn push_into(self, table: &mut Table, ids: &[ComponentId], tick: Tick);

    /// `remove_bundle`'s allocation-free extraction for `B`'s own
    /// elements: pulls `Self` straight out of `columns` via typed
    /// `swap_remove`, no `Box<dyn Any>` round trip. Unconditionally
    /// correct regardless of what else survives on the source
    /// archetype — every type involved is already known statically
    /// through `B`, unlike the *other*, non-`B` columns `remove_bundle`
    /// still has to migrate through the type-erased
    /// `Column::move_row_to`, since which components those are is only
    /// knowable at runtime. `remove_bundle` calls this only
    /// after every non-`B` column has already been migrated out of
    /// `columns` — never touches those, only the ids in `ids`.
    ///
    /// This replaced an earlier, narrower version of this idea (a
    /// fast path gated on "every column is being removed") after real
    /// measurement (`examples/diag_alloc_count.rs`) showed that gate
    /// never actually fires for the real `remove_bundle_two_components`
    /// benchmark — `insert_static` (confirmed directly from this
    /// crate's own tests, not assumed: `using_a_sparse_type_with_
    /// insert_static_panics`) claims a type for the *Archetype Core*,
    /// not the Sparse Shell, so `Marker` there is a real, surviving,
    /// archetype-tracked column, not a sparse one. `B`'s own elements
    /// never needing boxing at all, independent of what survives, is
    /// the version of this idea that's actually true unconditionally.
    fn take_direct(table: &mut Table, ids: &[ComponentId], row: usize) -> Self;
}

macro_rules! impl_bundle_for_tuple {
    ($($t:ident : $idx:tt),+) => {
        impl<$($t: 'static),+> Bundle for ($($t,)+) {
            fn component_ids(archetypes: &mut Archetypes) -> ComponentIdList {
                let mut list = ComponentIdList::new();
                $( list.push(archetypes.component_id::<$t>()); )+
                list
            }

            fn existing_component_ids(archetypes: &Archetypes) -> Option<ComponentIdList> {
                let mut list = ComponentIdList::new();
                $( list.push(archetypes.existing_component_id::<$t>()?); )+
                Some(list)
            }

            fn push_into(self, table: &mut Table, ids: &[ComponentId], tick: Tick) {
                $(
                    ensure_column(&mut table.columns, ids[$idx], || Box::<Vec<$t>>::default());
                    table
                        .columns
                        .get_mut(ids[$idx])
                        .expect("just ensured present")
                        .as_any_mut()
                        .downcast_mut::<Vec<$t>>()
                        .expect("column type must match component_id's T — component_ids and push_into share one fixed tuple-position order")
                        .push(self.$idx);
                    ensure_ticks(&mut table.ticks, ids[$idx]);
                    table
                        .ticks
                        .get_mut(ids[$idx])
                        .expect("just ensured present")
                        .push(ComponentTicks::new(tick));
                )+
            }

            fn take_direct(table: &mut Table, ids: &[ComponentId], row: usize) -> Self {
                (
                    $(
                        {
                            // Ticks first: both are swap-removed at the
                            // same `row`, from the row count each had
                            // before either removal, so the order
                            // between the two doesn't change which
                            // element is removed from either -- they're
                            // disjoint `Table` fields.
                            table
                                .ticks
                                .get_mut(ids[$idx])
                                .expect("remove_bundle's fast path only runs when every id in `ids` has a real column, and ensure_ticks always keeps ticks alongside it")
                                .swap_remove(row);
                            table
                                .columns
                                .get_mut(ids[$idx])
                                .expect("remove_bundle's fast path only runs when every id in `ids` has a real column")
                                .as_any_mut()
                                .downcast_mut::<Vec<$t>>()
                                .expect("column type must match component_id's T — existing_component_ids and take_direct share one fixed tuple-position order")
                                .swap_remove(row)
                        },
                    )+
                )
            }
        }
    };
}

impl_bundle_for_tuple!(A: 0);
impl_bundle_for_tuple!(A: 0, B: 1);
impl_bundle_for_tuple!(A: 0, B: 1, C: 2);
impl_bundle_for_tuple!(A: 0, B: 1, C: 2, D: 3);
impl_bundle_for_tuple!(A: 0, B: 1, C: 2, D: 3, E: 4);
impl_bundle_for_tuple!(A: 0, B: 1, C: 2, D: 3, E: 4, F: 5);
impl_bundle_for_tuple!(A: 0, B: 1, C: 2, D: 3, E: 4, F: 5, G: 6);
impl_bundle_for_tuple!(A: 0, B: 1, C: 2, D: 3, E: 4, F: 5, G: 6, H: 7);

/// The Archetype Core itself: every [`Archetype`] that exists, looked up
/// by exact component-type-set signature, plus where every
/// archetype-tracked entity currently lives.
pub(crate) struct Archetypes {
    archetypes: SparseSet<ArchetypeId, Archetype>,
    signature_to_id: HashMap<Vec<ComponentId>, ArchetypeId>,
    locations: SparseSet<Entity, EntityLocation>,
    next_id: u32,
    /// Deliberately a *separate* `TypeId -> ComponentId` numbering space
    /// from `SparseShell`'s own — the same `ComponentId` value can mean
    /// a completely different type in each system. Never a problem in
    /// practice: a `ComponentId` produced here is only ever compared
    /// against other `ComponentId`s produced here, never against
    /// `SparseShell`'s. Reusing `component.rs`'s `ComponentId` *type*
    /// rather than defining a near-identical second type is just to
    /// avoid the duplication; the numbering spaces themselves are kept
    /// hard-separated on purpose, matching this whole module's
    /// Sparse-Shell-vs-Archetype-Core architectural split.
    component_ids: TypeIdMap<ComponentId>,
    next_component_id: u32,
    /// Populated only for component types opted into FFI exposure via
    /// [`Self::register_ffi`] — mirrors `SparseShell`'s own field of the
    /// same name exactly (`component.rs`), same reasoning: most types
    /// will never have an entry here, which is the point.
    ffi_accessors: HashMap<ComponentId, FfiSpanAccessor>,
    /// The FFI-facing counterpart to Rust's own `TypeId`-based lookup —
    /// mirrors `SparseShell::ffi_names` exactly. A *separate* name
    /// space from `SparseShell`'s own, matching this whole struct's
    /// already-established separate `ComponentId` numbering space: the
    /// same name could resolve to a different `ComponentId` in each
    /// system, by design.
    ffi_names: HashMap<&'static str, ComponentId>,
}

const EMPTY_ARCHETYPE: ArchetypeId = ArchetypeId(0);

impl Archetypes {
    pub(crate) fn new() -> Self {
        let mut archetypes = SparseSet::new();
        archetypes.insert(EMPTY_ARCHETYPE, Archetype::empty());
        let mut signature_to_id = HashMap::new();
        signature_to_id.insert(Vec::new(), EMPTY_ARCHETYPE);
        Self {
            archetypes,
            signature_to_id,
            locations: SparseSet::new(),
            next_id: 1,
            component_ids: TypeIdMap::default(),
            next_component_id: 0,
            ffi_accessors: HashMap::new(),
            ffi_names: HashMap::new(),
        }
    }

    /// Returns `T`'s `ComponentId` in this `Archetypes`' own numbering
    /// space, registering it on first use.
    pub(crate) fn component_id<T: 'static>(&mut self) -> ComponentId {
        let type_id = TypeId::of::<T>();
        if let Some(&id) = self.component_ids.get(&type_id) {
            id
        } else {
            let id = ComponentId(self.next_component_id);
            self.next_component_id += 1;
            self.component_ids.insert(type_id, id);
            id
        }
    }

    /// Looks up `T`'s `ComponentId` without registering it — `None` if
    /// `T` has never been inserted as an archetype-tracked component for
    /// anything, in which case there is by construction nothing to
    /// find. Mirrors `SparseShell::existing_component_id`'s exact
    /// reasoning: read paths (`get`/`get_mut`/`has`) shouldn't spend a
    /// `ComponentId` slot on a type that's never actually been used.
    pub(crate) fn existing_component_id<T: 'static>(&self) -> Option<ComponentId> {
        self.component_ids.get(&TypeId::of::<T>()).copied()
    }

    /// Iterates every `(Entity, &T)` currently alive with an
    /// archetype-tracked `T` attached — the Archetype Core's own
    /// counterpart to `query.rs`'s `World::query::<T>()` (Sparse
    /// Shell). The one real difference from that method, not present
    /// on the Sparse Shell side at all: `T`'s data isn't in one place
    /// here — it's spread across every archetype whose signature
    /// includes it (see [`Self::archetypes_with`]/[`Self::raw_span`]'s
    /// own doc comments for the same fragmentation), so this chains
    /// across all of them rather than reading one column. Doesn't
    /// separately check liveness per entity, for the same reason
    /// `query.rs`'s own doc comment gives: every entity in an
    /// archetype's table is alive by construction (`despawn` removes
    /// the entity's row from its table before the slot is ever freed —
    /// see `Archetypes::despawn`). Empty (not a panic or a special
    /// case) if `T` was never inserted as an archetype-tracked
    /// component for anything.
    ///
    /// Returns [`Iter1`], a hand-written `Iterator` state machine, for
    /// the exact same reason [`Self::iter2`] does — see [`Iter2`]'s
    /// doc comment for the full real-numbers writeup. Found *because*
    /// of that writeup, not independently: fixing `iter2` alone left a
    /// two-component query (9.16µs, N=10,000) ~16x *faster* than this
    /// method's own single-component query (145.20µs, same N, same
    /// bench run) — a two-component query outrunning a one-component
    /// one on the same data is a real correctness-of-reasoning signal
    /// on its own, not just a number worth double-checking. Same root
    /// cause, same fix, same result: this method's old
    /// `Option::into_iter().flat_map(|_| archetypes_with(...).flat_map(|_|
    /// entities.zip(column)))` shape measured 145.20µs at N=10,000;
    /// the flat hand-written loop below (bounds-checked, no `unsafe`)
    /// measured 7.2131µs — a ~95% reduction. Confirms [`Iter2`]'s own
    /// finding wasn't specific to the two-column case: the generic
    /// adaptor-chain overhead, not anything about *this* query's
    /// shape, was the dominant cost both times.
    ///
    /// A real, non-obvious case this handles: `component_ids` listing
    /// a component doesn't guarantee a column for it has actually been
    /// created yet. [`Self::insert_bundle`]'s chained
    /// [`Self::edge_for_insert`] calls (one per bundle element, see its
    /// own doc comment) can create — and correctly register in
    /// `component_ids` — an *intermediate* archetype on the way to a
    /// bundle's final target, without ever moving an entity into it
    /// (this was a real panic, `archetypes_with guarantees
    /// component_id is in this archetype's own signature`, hit during
    /// bench-harness setup: bulk `insert_bundle` calls create exactly
    /// these intermediate archetypes). Such an archetype always has
    /// zero rows too, so "no column" and "a column that exists but is
    /// empty" are observationally identical from here — both
    /// contribute zero items. Handled below without a panic, not
    /// treated as a broken invariant.
    pub(crate) fn iter<T: 'static>(&self) -> Iter1<'_, T> {
        let id = self.existing_component_id::<T>();
        let matched: Vec<ArchetypeId> = match id {
            Some(id) => self.archetypes_with(id).collect(),
            None => Vec::new(),
        };
        Iter1::new(self, id, matched)
    }

    /// Iterates every `(Entity, &A, &B)` for entities alive with *both*
    /// an archetype-tracked `A` and `B` attached.
    ///
    /// Returns [`Iter2`], a hand-written `Iterator` state machine, not
    /// an `impl Iterator` composed from `flat_map`/`filter`/`zip`
    /// adaptors — that combinator-chain version is what shipped first,
    /// see [`Iter2`]'s own doc comment for the full real-numbers
    /// writeup of why it changed again. Same "no column implies no
    /// rows" edge case as [`Self::iter`] applies independently to both
    /// `a_col` and `b_col`. Empty if either `A` or `B` was never
    /// inserted as an archetype-tracked component for anything.
    pub(crate) fn iter2<A: 'static, B: 'static>(&self) -> Iter2<'_, A, B> {
        let ids = self
            .existing_component_id::<A>()
            .zip(self.existing_component_id::<B>());
        let matched: Vec<ArchetypeId> = match ids {
            Some((a_id, b_id)) => self
                .archetypes_with(a_id)
                .filter(|&archetype_id| {
                    self.archetypes
                        .get(archetype_id)
                        .is_some_and(|archetype| archetype.component_ids.contains(&b_id))
                })
                .collect(),
            None => Vec::new(),
        };
        Iter2::new(self, ids, matched)
    }

    /// Opts `T` into FFI span exposure under `name` — the Archetype
    /// Core's own counterpart to `SparseShell::register_ffi`, identical
    /// reasoning throughout (see that method's doc comment). Must be
    /// called from Rust; idempotent for the same `T`.
    ///
    /// # Panics
    /// If `name` was already registered for a *different* component
    /// type — see `SparseShell::register_ffi`'s identical panic
    /// condition and rationale.
    pub(crate) fn register_ffi<T>(&mut self, name: &'static str) -> ComponentId
    where
        T: 'static + IntoBytes + Immutable + KnownLayout,
    {
        let id = self.component_id::<T>();
        self.ffi_accessors.entry(id).or_insert_with(|| {
            (|column: &dyn Any| -> FfiSpan {
                let vec = column
                    .downcast_ref::<Vec<T>>()
                    .expect("column type must match component_id's T");
                FfiSpan::from_slice(vec)
            }) as FfiSpanAccessor
        });
        match self.ffi_names.get(name) {
            Some(&existing) if existing != id => panic!(
                "Archetypes::register_ffi: name {name:?} is already registered for a different component type"
            ),
            _ => {
                self.ffi_names.insert(name, id);
            }
        }
        id
    }

    /// Non-generic, per-archetype raw span over `component_id`'s column
    /// within `archetype_id`'s table. `None` if `component_id` was
    /// never opted into FFI exposure, `archetype_id` doesn't exist, or
    /// — a real, permanent structural fact, not a transient "not
    /// populated yet" — `archetype_id`'s exact signature doesn't
    /// include `component_id` at all. That last case deliberately does
    /// *not* fall back to an empty span the way `SparseShell::raw_span`
    /// does for its own "registered but nothing inserted yet" case:
    /// unlike a Sparse Shell type (one global column, genuinely absent
    /// vs. genuinely empty are different points in the *same* column's
    /// life), a component simply not being part of one specific
    /// archetype's signature is permanent for that archetype — matches
    /// `has`'s own established `false`-not-panic convention for exactly
    /// this. A real, *empty-but-present* column (the component is in
    /// the signature, but every entity that ever had it has since
    /// migrated away) is a genuinely different case and does return
    /// `Some` with `count == 0` — proven by a real test, not assumed,
    /// since `ensure_column` only ever adds columns, never removes them
    /// once an archetype has been created with that signature. A
    /// signature that includes the component but has no column yet
    /// (a zero-row intermediate left behind by `insert_bundle` or
    /// `spawn_bundle`) answers the same way, `Some` with `count == 0`.
    ///
    /// This is the real, unavoidable difference from `SparseShell`'s
    /// side of the FFI-span mechanism: a component type here isn't one
    /// stable thing to read — it's fragmented across every archetype
    /// that currently contains it. Pair with [`Self::archetypes_with`]
    /// to enumerate all of them, and with [`Self::entity_ids`] for
    /// which `Entity` owns each element of the span this returns.
    pub(crate) fn raw_span(
        &self,
        archetype_id: ArchetypeId,
        component_id: ComponentId,
    ) -> Option<FfiSpan> {
        let accessor = self.ffi_accessors.get(&component_id)?;
        let archetype = self.archetypes.get(archetype_id)?;
        match archetype.table.columns.get(component_id) {
            Some(column) => Some(accessor(column.as_any())),
            // The signature includes the component but no entity has
            // ever migrated in, so no column exists yet (the zero-row
            // intermediates `insert_bundle`/`spawn_bundle` leave
            // behind). Same answer as an emptied column.
            None if archetype.component_ids.contains(&component_id) => Some(FfiSpan::empty()),
            None => None,
        }
    }

    /// Entity-correlation counterpart to [`Self::raw_span`], for the
    /// same `(archetype_id, component_id)` pair: `entity_ids(a, c)[i]`
    /// is the `Entity` that owns `raw_span(a, c)`'s element `i`, for
    /// every valid `i`.
    ///
    /// Deliberately takes the same two-key signature as `raw_span` even
    /// though a `Table`'s `entities` list doesn't actually vary by
    /// `component_id` — every column in one archetype's table shares
    /// the same row index space, `component_id` here only *gates*
    /// which pairs are valid, exactly the way `raw_span`'s own
    /// `component_id` does. Requiring both keys keeps the two calls a
    /// matched pair: a caller can't get a real entity list back for a
    /// `component_id` it could never have gotten a real data span for
    /// in the first place. Same `None`/permanent-vs-empty distinction
    /// as `raw_span`'s own doc comment describes, for the same reason
    /// (an archetype either has this component in its signature or it
    /// permanently doesn't; whether any entities currently populate
    /// that archetype is a separate, transient fact).
    pub(crate) fn entity_ids(
        &self,
        archetype_id: ArchetypeId,
        component_id: ComponentId,
    ) -> Option<Vec<u64>> {
        self.ffi_accessors.get(&component_id)?;
        let archetype = self.archetypes.get(archetype_id)?;
        if !archetype.component_ids.contains(&component_id) {
            return None;
        }
        Some(
            archetype
                .table
                .entities
                .iter()
                .map(|entity| entity.as_ffi())
                .collect(),
        )
    }

    /// Per-archetype, row-aligned view for change-detection queries:
    /// entity ids, `T` values, and their [`ComponentTicks`], all indexed
    /// the same way. `None` if `archetype_id` doesn't exist or its
    /// signature doesn't include `component_id` — same permanent-fact
    /// `None` as [`Self::raw_span`], including the same "signature has
    /// it, no column yet" case reading back as `Some` with empty
    /// slices. `T`-generic and typed (unlike `raw_span`), so only
    /// reachable from Rust, not FFI. Used only by
    /// `World::query_added`/`World::query_changed` — a dedicated path,
    /// not layered onto `Iter1`/`Iter2` (see `tick.rs`'s doc comment).
    pub(crate) fn rows_with_ticks<T: 'static>(
        &self,
        archetype_id: ArchetypeId,
        component_id: ComponentId,
    ) -> Option<(&[Entity], &[T], &[ComponentTicks])> {
        let archetype = self.archetypes.get(archetype_id)?;
        if !archetype.component_ids.contains(&component_id) {
            return None;
        }
        let values: &[T] = match archetype.table.columns.get(component_id) {
            Some(column) => column
                .as_any()
                .downcast_ref::<Vec<T>>()
                .expect("column type must match component_id's T")
                .as_slice(),
            None => &[],
        };
        let ticks: &[ComponentTicks] = match archetype.table.ticks.get(component_id) {
            Some(ticks) => ticks.as_slice(),
            None => &[],
        };
        Some((&archetype.table.entities, values, ticks))
    }

    /// Enumerates every currently-existing archetype whose signature
    /// includes `component_id` — the real fragmentation
    /// [`Self::raw_span`]'s own doc comment describes. Not gated by
    /// FFI registration itself (a pure structural query, matching
    /// `has`'s own spirit) — `raw_span` is what actually enforces that,
    /// per archetype, when the caller gets there.
    pub(crate) fn archetypes_with(
        &self,
        component_id: ComponentId,
    ) -> impl Iterator<Item = ArchetypeId> + '_ {
        self.archetypes.iter().filter_map(move |(id, archetype)| {
            archetype
                .component_ids
                .contains(&component_id)
                .then_some(id)
        })
    }

    /// Enumerates every currently-existing archetype whose signature
    /// contains *all* of `with`, *none* of `without`, and — if `any_of`
    /// is non-empty — *at least one* of `any_of`. The runtime, id-based
    /// counterpart to the typed `With`/`Without`/`Or` filters
    /// (`filter.rs`), for callers that only have `ComponentId`s (the FFI
    /// surface). Same structural semantics as [`Self::archetypes_with`]:
    /// zero-row archetypes are included, an id that names no
    /// archetype-tracked component simply matches nothing in `with` or
    /// `any_of` and is ignored in `without`, and `with`/`without` both
    /// empty with `any_of` also empty means every archetype. An empty
    /// `any_of` is "no `Or` constraint", not "match nothing" — matching
    /// `Or`'s own empty-tuple case has no typed spelling (`Or<()>`
    /// doesn't implement `QueryFilter`), so there is nothing for this to
    /// stay consistent with either way; this is the more useful of the
    /// two readings for a caller who simply isn't using `any_of`.
    /// Duplicate ids are harmless, and an id in both `with` and
    /// `without` matches nothing regardless of `any_of`.
    pub(crate) fn archetypes_matching<'a>(
        &'a self,
        with: &'a [ComponentId],
        without: &'a [ComponentId],
        any_of: &'a [ComponentId],
    ) -> impl Iterator<Item = ArchetypeId> + 'a {
        self.archetypes.iter().filter_map(move |(id, archetype)| {
            let signature = &archetype.component_ids;
            (with.iter().all(|c| signature.contains(c))
                && !without.iter().any(|c| signature.contains(c))
                && (any_of.is_empty() || any_of.iter().any(|c| signature.contains(c))))
            .then_some(id)
        })
    }

    /// Looks up the [`ComponentId`] a type was registered under via
    /// [`Self::register_ffi`], by name. The FFI-facing counterpart to
    /// `register_ffi`'s Rust-facing generic registration.
    pub(crate) fn lookup_ffi_id(&self, name: &str) -> Option<ComponentId> {
        self.ffi_names.get(name).copied()
    }

    fn get_or_create(&mut self, signature: Vec<ComponentId>) -> ArchetypeId {
        if let Some(&id) = self.signature_to_id.get(&signature) {
            return id;
        }
        let id = ArchetypeId(self.next_id);
        self.next_id += 1;
        self.archetypes
            .insert(id, Archetype::with_signature(signature.clone()));
        self.signature_to_id.insert(signature, id);
        id
    }

    /// Registers a freshly spawned entity into the empty archetype.
    /// Every entity has a location from the moment it's spawned, even if
    /// it never gets an archetype-tracked component — this is what lets
    /// `insert`/`remove`/`despawn` below all assume a location always
    /// exists rather than treating "never touched the Archetype Core" as
    /// a separate case to check for everywhere.
    pub(crate) fn spawn(&mut self, entity: Entity) {
        let table = &mut self
            .archetypes
            .get_mut(EMPTY_ARCHETYPE)
            .expect("the empty archetype always exists")
            .table;
        let row = table.len();
        table.entities.push(entity);
        self.locations.insert(
            entity,
            EntityLocation {
                archetype_id: EMPTY_ARCHETYPE,
                row,
            },
        );
    }

    /// Removes `entity` from whichever archetype it's currently in.
    /// Called by `World::despawn`, before the entity's generational slot
    /// is freed — same load-bearing ordering requirement
    /// `SparseShell::remove_entity_from_all` has, for the same
    /// underlying reason (see `component.rs`'s doc comment): storage
    /// here is also keyed purely by `Entity::sparse_index()`, not
    /// generation.
    pub(crate) fn despawn(&mut self, entity: Entity) {
        let Some(location) = self.locations.remove(entity) else {
            return;
        };
        let archetype = self
            .archetypes
            .get_mut(location.archetype_id)
            .expect("a tracked location must always point at a real archetype");
        if let Some(swapped) = archetype.table.swap_remove_row(location.row) {
            self.fix_up_row_after_swap(swapped, location.row);
        }
    }

    fn fix_up_row_after_swap(&mut self, swapped_entity: Entity, new_row: usize) {
        self.locations
            .get_mut(swapped_entity)
            .expect("a table row's entity must always have a tracked location")
            .row = new_row;
    }

    /// Moves `entity` from its current archetype into the one for
    /// `current signature ∪ {component_id}`, migrating every existing
    /// column's value across and appending `value` as the new column.
    /// Returns `false` (no-op) if `entity` has no tracked location at
    /// all (never spawned through this `Archetypes`) or already has
    /// this exact `component_id` — matching `SparseSet::insert`'s
    /// replace-in-place convention would require knowing `T` to replace
    /// the old value with the new one; simpler and equally safe for a
    /// v1 to just treat "already has it" as a no-op and let the caller
    /// call `get_mut` instead. Revisit if that's ever a real friction
    /// point.
    pub(crate) fn insert<T: 'static>(
        &mut self,
        entity: Entity,
        component_id: ComponentId,
        value: T,
        tick: Tick,
    ) -> bool {
        let Some(&from_location) = self.locations.get(entity) else {
            return false;
        };
        let from_id = from_location.archetype_id;

        if self
            .archetypes
            .get(from_id)
            .is_some_and(|a| a.component_ids.contains(&component_id))
        {
            return false;
        }

        let to_id = self.edge_for_insert(from_id, component_id);

        let (from_archetype, to_archetype) = self.get_two_mut(from_id, to_id);

        for (comp, column) in from_archetype.table.columns.iter_mut() {
            ensure_column(&mut to_archetype.table.columns, comp, || {
                column.new_same_type()
            });
            ensure_ticks(&mut to_archetype.table.ticks, comp);
            let dest = to_archetype
                .table
                .columns
                .get_mut(comp)
                .expect("just ensured present");
            column.move_row_to(from_location.row, &mut **dest);
            let moved_ticks = from_archetype
                .table
                .ticks
                .get_mut(comp)
                .expect("ensure_ticks always keeps ticks alongside its column")
                .swap_remove(from_location.row);
            to_archetype
                .table
                .ticks
                .get_mut(comp)
                .expect("just ensured present")
                .push(moved_ticks);
        }
        let old_last = from_archetype.table.entities.len() - 1;
        from_archetype.table.entities.swap_remove(from_location.row);
        let swapped_entity = (from_location.row != old_last)
            .then(|| from_archetype.table.entities[from_location.row]);

        let new_row = to_archetype.table.entities.len();
        to_archetype.table.entities.push(entity);
        ensure_column(&mut to_archetype.table.columns, component_id, || {
            Box::<Vec<T>>::default()
        });
        to_archetype
            .table
            .columns
            .get_mut(component_id)
            .expect("just ensured present")
            .as_any_mut()
            .downcast_mut::<Vec<T>>()
            .expect(
                "column for component_id must hold Vec<T> for this T — component_id is T's own id",
            )
            .push(value);
        ensure_ticks(&mut to_archetype.table.ticks, component_id);
        to_archetype
            .table
            .ticks
            .get_mut(component_id)
            .expect("just ensured present")
            .push(ComponentTicks::new(tick));

        self.locations.insert(
            entity,
            EntityLocation {
                archetype_id: to_id,
                row: new_row,
            },
        );
        if let Some(swapped) = swapped_entity {
            self.fix_up_row_after_swap(swapped, from_location.row);
        }
        true
    }

    fn edge_for_insert(&mut self, from_id: ArchetypeId, component_id: ComponentId) -> ArchetypeId {
        if let Some(&cached) = self
            .archetypes
            .get(from_id)
            .and_then(|a| a.add_edges.get(&component_id))
        {
            return cached;
        }
        let mut signature = self
            .archetypes
            .get(from_id)
            .expect("location must point at a real archetype")
            .component_ids
            .clone();
        let insert_at =
            signature.partition_point(|&id| id.sparse_index() < component_id.sparse_index());
        signature.insert(insert_at, component_id);
        let to_id = self.get_or_create(signature);
        self.archetypes
            .get_mut(from_id)
            .expect("checked above")
            .add_edges
            .insert(component_id, to_id);
        to_id
    }

    /// Moves `entity` from its current archetype into the one for
    /// `current signature \ {component_id}`, migrating every other
    /// column's value across and returning the removed component's
    /// value. `None` if `entity` has no tracked location, or doesn't
    /// actually have this component.
    pub(crate) fn remove<T: 'static>(
        &mut self,
        entity: Entity,
        component_id: ComponentId,
    ) -> Option<T> {
        let from_location = *self.locations.get(entity)?;
        let from_id = from_location.archetype_id;

        if !self
            .archetypes
            .get(from_id)?
            .component_ids
            .contains(&component_id)
        {
            return None;
        }

        let to_id = self.edge_for_remove(from_id, component_id);

        let (from_archetype, to_archetype) = self.get_two_mut(from_id, to_id);

        let mut removed_value: Option<T> = None;
        for (comp, column) in from_archetype.table.columns.iter_mut() {
            if comp == component_id {
                removed_value = Some(
                    column
                        .as_any_mut()
                        .downcast_mut::<Vec<T>>()
                        .expect(
                            "column for component_id must hold Vec<T> for this T — component_id is T's own id",
                        )
                        .swap_remove(from_location.row),
                );
                from_archetype
                    .table
                    .ticks
                    .get_mut(comp)
                    .expect("ensure_ticks always keeps ticks alongside its column")
                    .swap_remove(from_location.row);
                continue;
            }
            ensure_column(&mut to_archetype.table.columns, comp, || {
                column.new_same_type()
            });
            ensure_ticks(&mut to_archetype.table.ticks, comp);
            let dest = to_archetype
                .table
                .columns
                .get_mut(comp)
                .expect("just ensured present");
            column.move_row_to(from_location.row, &mut **dest);
            let moved_ticks = from_archetype
                .table
                .ticks
                .get_mut(comp)
                .expect("ensure_ticks always keeps ticks alongside its column")
                .swap_remove(from_location.row);
            to_archetype
                .table
                .ticks
                .get_mut(comp)
                .expect("just ensured present")
                .push(moved_ticks);
        }
        let old_last = from_archetype.table.entities.len() - 1;
        from_archetype.table.entities.swap_remove(from_location.row);
        let swapped_entity = (from_location.row != old_last)
            .then(|| from_archetype.table.entities[from_location.row]);

        let new_row = to_archetype.table.entities.len();
        to_archetype.table.entities.push(entity);

        self.locations.insert(
            entity,
            EntityLocation {
                archetype_id: to_id,
                row: new_row,
            },
        );
        if let Some(swapped) = swapped_entity {
            self.fix_up_row_after_swap(swapped, from_location.row);
        }
        removed_value
    }

    fn edge_for_remove(&mut self, from_id: ArchetypeId, component_id: ComponentId) -> ArchetypeId {
        if let Some(&cached) = self
            .archetypes
            .get(from_id)
            .and_then(|a| a.remove_edges.get(&component_id))
        {
            return cached;
        }
        let mut signature = self
            .archetypes
            .get(from_id)
            .expect("checked by caller")
            .component_ids
            .clone();
        signature.retain(|&id| id != component_id);
        let to_id = self.get_or_create(signature);
        self.archetypes
            .get_mut(from_id)
            .expect("checked by caller")
            .remove_edges
            .insert(component_id, to_id);
        to_id
    }

    /// Moves `entity` from its current archetype into the one for
    /// `current signature ∪ B`, migrating every existing column across
    /// and pushing every element of `bundle` into its own new column —
    /// one migration total, not one per element of `B`. Returns
    /// `false` (no-op) if `entity` has no tracked location, or if it
    /// already has *any* of `B`'s component types — the bundle
    /// equivalent of [`Self::insert`]'s own "already has it" no-op
    /// convention, generalized: a partial bundle insert would leave
    /// the entity in an archetype matching neither its old signature
    /// nor a signature the caller actually asked for.
    ///
    /// The final target archetype is computed by chaining
    /// [`Self::edge_for_insert`] once per element of `B`, reusing the
    /// exact same per-source-archetype cache single-component inserts
    /// already populate and benefit from — a repeated
    /// `insert_bundle::<SameBundle>` call (spawning many entities with
    /// the same component set, the common real case a `Bundle` API
    /// exists for) hits a fully warm cache after the first call, same
    /// as repeated single-component inserts already do.
    /// Spawns a brand-new entity directly into the archetype for `B`,
    /// with no stop in the empty archetype. `World::spawn` followed by
    /// `World::insert_bundle` pays for that stop twice -- once to write
    /// the entity into the empty archetype's table and `locations`,
    /// once to immediately migrate it back out through `get_two_mut`
    /// and a (trivially empty, but not free) column-migration loop --
    /// even though nothing ever reads that intermediate state for a
    /// fresh entity. This walks the same `edge_for_insert` chain
    /// `insert_bundle` does, starting from `EMPTY_ARCHETYPE`, and
    /// writes the entity's row and location exactly once. Infallible
    /// (unlike `insert_bundle`): a brand-new entity can never already
    /// hold one of `B`'s components, so there is no failure case to
    /// report.
    pub(crate) fn spawn_bundle<B: Bundle>(&mut self, entity: Entity, bundle: B, tick: Tick) {
        let ids = B::component_ids(self);

        debug_assert!(
            {
                let mut sorted = ids.clone();
                sorted.sort_by_key(|id| id.as_u32());
                sorted.windows(2).all(|pair| pair[0] != pair[1])
            },
            "Bundle must not repeat the same component type twice — every element needs its own column"
        );

        let mut to_id = EMPTY_ARCHETYPE;
        for &id in ids.iter() {
            to_id = self.edge_for_insert(to_id, id);
        }

        let to_archetype = self
            .archetypes
            .get_mut(to_id)
            .expect("edge_for_insert/get_or_create must always yield a real archetype");

        let row = to_archetype.table.entities.len();
        to_archetype.table.entities.push(entity);
        bundle.push_into(&mut to_archetype.table, &ids, tick);

        self.locations.insert(
            entity,
            EntityLocation {
                archetype_id: to_id,
                row,
            },
        );
    }

    pub(crate) fn insert_bundle<B: Bundle>(
        &mut self,
        entity: Entity,
        bundle: B,
        tick: Tick,
    ) -> bool {
        let Some(&from_location) = self.locations.get(entity) else {
            return false;
        };
        let from_id = from_location.archetype_id;
        let ids = B::component_ids(self);

        debug_assert!(
            {
                let mut sorted = ids.clone();
                sorted.sort_by_key(|id| id.as_u32());
                sorted.windows(2).all(|pair| pair[0] != pair[1])
            },
            "Bundle must not repeat the same component type twice — every element needs its own column"
        );

        if self
            .archetypes
            .get(from_id)
            .is_some_and(|a| ids.iter().any(|id| a.component_ids.contains(id)))
        {
            return false;
        }

        let mut to_id = from_id;
        for &id in ids.iter() {
            to_id = self.edge_for_insert(to_id, id);
        }

        let (from_archetype, to_archetype) = self.get_two_mut(from_id, to_id);

        for (comp, column) in from_archetype.table.columns.iter_mut() {
            ensure_column(&mut to_archetype.table.columns, comp, || {
                column.new_same_type()
            });
            ensure_ticks(&mut to_archetype.table.ticks, comp);
            let dest = to_archetype
                .table
                .columns
                .get_mut(comp)
                .expect("just ensured present");
            column.move_row_to(from_location.row, &mut **dest);
            let moved_ticks = from_archetype
                .table
                .ticks
                .get_mut(comp)
                .expect("ensure_ticks always keeps ticks alongside its column")
                .swap_remove(from_location.row);
            to_archetype
                .table
                .ticks
                .get_mut(comp)
                .expect("just ensured present")
                .push(moved_ticks);
        }
        let old_last = from_archetype.table.entities.len() - 1;
        from_archetype.table.entities.swap_remove(from_location.row);
        let swapped_entity = (from_location.row != old_last)
            .then(|| from_archetype.table.entities[from_location.row]);

        let new_row = to_archetype.table.entities.len();
        to_archetype.table.entities.push(entity);
        bundle.push_into(&mut to_archetype.table, &ids, tick);

        self.locations.insert(
            entity,
            EntityLocation {
                archetype_id: to_id,
                row: new_row,
            },
        );
        if let Some(swapped) = swapped_entity {
            self.fix_up_row_after_swap(swapped, from_location.row);
        }
        true
    }

    /// Moves `entity` from its current archetype into the one for
    /// `current signature ∖ B`, migrating every other column across and
    /// handing back every element of `B` reassembled from the columns
    /// it's removed from — one migration total, not one per element of
    /// `B`. Returns `None` if `entity` has no tracked location, if
    /// *any* element of `B` was never registered as an
    /// archetype-tracked component for anything (mirrors
    /// [`Self::remove`]'s own non-registering `existing_component_id`
    /// lookup, generalized — see `World::remove_static`), or if the
    /// entity is missing *any* of `B`'s component types — the bundle
    /// equivalent of `remove`'s own "doesn't have it" `None` case, for
    /// the same "no partial application" reasoning
    /// [`Self::insert_bundle`]'s own doc comment gives.
    pub(crate) fn remove_bundle<B: Bundle>(&mut self, entity: Entity) -> Option<B> {
        let from_location = *self.locations.get(entity)?;
        let from_id = from_location.archetype_id;
        let ids = B::existing_component_ids(self)?;

        debug_assert!(
            {
                let mut sorted = ids.clone();
                sorted.sort_by_key(|id| id.as_u32());
                sorted.windows(2).all(|pair| pair[0] != pair[1])
            },
            "Bundle must not repeat the same component type twice — every element needs its own column"
        );

        let has_all = self
            .archetypes
            .get(from_id)?
            .component_ids
            .iter()
            .filter(|id| ids.contains(id))
            .count()
            == ids.len();
        if !has_all {
            return None;
        }

        let to_id = self.edge_for_remove_bundle(from_id, &ids);

        let (from_archetype, to_archetype) = self.get_two_mut(from_id, to_id);

        // Every non-B column on this archetype survives the removal
        // and must migrate to `to_archetype` — type erasure is
        // unavoidable for those specifically, since which components
        // they are is only known at runtime. B's own columns are
        // skipped here entirely (`continue`) and extracted afterward by
        // `take_direct` instead, which needs no boxing at all, since
        // every type involved is already known statically through B —
        // true regardless of how many other columns also survive.
        for (comp, column) in from_archetype.table.columns.iter_mut() {
            if ids.contains(&comp) {
                continue;
            }
            ensure_column(&mut to_archetype.table.columns, comp, || {
                column.new_same_type()
            });
            ensure_ticks(&mut to_archetype.table.ticks, comp);
            let dest = to_archetype
                .table
                .columns
                .get_mut(comp)
                .expect("just ensured present");
            column.move_row_to(from_location.row, &mut **dest);
            let moved_ticks = from_archetype
                .table
                .ticks
                .get_mut(comp)
                .expect("ensure_ticks always keeps ticks alongside its column")
                .swap_remove(from_location.row);
            to_archetype
                .table
                .ticks
                .get_mut(comp)
                .expect("just ensured present")
                .push(moved_ticks);
        }
        let result = B::take_direct(&mut from_archetype.table, &ids, from_location.row);

        let old_last = from_archetype.table.entities.len() - 1;
        from_archetype.table.entities.swap_remove(from_location.row);
        let swapped_entity = (from_location.row != old_last)
            .then(|| from_archetype.table.entities[from_location.row]);

        let new_row = to_archetype.table.entities.len();
        to_archetype.table.entities.push(entity);

        self.locations.insert(
            entity,
            EntityLocation {
                archetype_id: to_id,
                row: new_row,
            },
        );
        if let Some(swapped) = swapped_entity {
            self.fix_up_row_after_swap(swapped, from_location.row);
        }
        Some(result)
    }

    /// Chains [`Self::edge_for_remove`] once per id in `ids` — the
    /// `remove_bundle` counterpart to `insert_bundle`'s own
    /// `edge_for_insert` chain, same reasoning: reuses the existing
    /// per-source-archetype single-component cache rather than a new
    /// multi-key one.
    fn edge_for_remove_bundle(&mut self, from_id: ArchetypeId, ids: &[ComponentId]) -> ArchetypeId {
        let mut to_id = from_id;
        for &id in ids {
            to_id = self.edge_for_remove(to_id, id);
        }
        to_id
    }

    pub(crate) fn get<T: 'static>(&self, entity: Entity, component_id: ComponentId) -> Option<&T> {
        let location = self.locations.get(entity)?;
        let archetype = self.archetypes.get(location.archetype_id)?;
        let column = archetype.table.columns.get(component_id)?;
        column
            .as_any()
            .downcast_ref::<Vec<T>>()
            .expect("column type must match component_id's T")
            .get(location.row)
    }

    pub(crate) fn get_mut<T: 'static>(
        &mut self,
        entity: Entity,
        component_id: ComponentId,
        tick: Tick,
    ) -> Option<&mut T> {
        let location = *self.locations.get(entity)?;
        let archetype = self.archetypes.get_mut(location.archetype_id)?;
        if let Some(ticks) = archetype.table.ticks.get_mut(component_id) {
            if let Some(row_ticks) = ticks.get_mut(location.row) {
                row_ticks.changed = tick;
            }
        }
        let column = archetype.table.columns.get_mut(component_id)?;
        column
            .as_any_mut()
            .downcast_mut::<Vec<T>>()
            .expect("column type must match component_id's T")
            .get_mut(location.row)
    }

    pub(crate) fn has(&self, entity: Entity, component_id: ComponentId) -> bool {
        self.locations
            .get(entity)
            .and_then(|loc| self.archetypes.get(loc.archetype_id))
            .is_some_and(|a| a.component_ids.contains(&component_id))
    }

    /// Fetches `&mut Archetype` for both `a` and `b` at once (always
    /// distinct, for a real structural change), in place — no removal,
    /// no reinsertion.
    ///
    /// This replaces a prior `take_two`/`give_back_two` pair that
    /// physically `remove`d both archetypes from `self.archetypes` as
    /// owned values (to sidestep needing two live mutable borrows into
    /// one collection) and reinserted them afterward. That round trip
    /// was real, measured cost, not a rounding error: real CI's
    /// `structural_churn_insert_remove` runs at ~271.8 ns per
    /// insert-or-remove call, and `mid-collections/benches/
    /// sparse_set.rs`'s own `two_at_once_archetype_sized` group (value
    /// type sized to `Archetype`'s real 216 bytes, measured directly
    /// via `size_of` rather than assumed) showed the old remove+
    /// reinsert pattern costing 88-114 ns per call regardless of
    /// archetype count — 32-42% of the entire structural-change cost,
    /// just from shuffling two structs in and out of a `SparseSet` that
    /// never needed to move them.
    ///
    /// Grounded directly in `bevy_ecs`'s own real source
    /// (`Mid-D-Man/bevy`, `archetype.rs`'s `Archetypes::
    /// get_maybe_disjoint_mut`, read directly): bevy stores archetypes
    /// in a plain `Vec` and reaches into it at two indices via `unsafe`
    /// disjoint indexing, specifically to avoid this exact move. This
    /// gets the same result through `mid_collections::SparseSet::
    /// get_disjoint_mut` — `split_at_mut`-based, **zero new `unsafe`**,
    /// matching this crate's own zero-unsafe-by-default precedent (see
    /// this module's top doc comment) rather than importing bevy's
    /// technique wholesale the way that comment originally declined to.
    /// The "costs two extra `SparseSet` operations... deliberately"
    /// trade-off this comment used to describe is exactly what got
    /// measured and replaced here.
    fn get_two_mut(&mut self, a: ArchetypeId, b: ArchetypeId) -> (&mut Archetype, &mut Archetype) {
        debug_assert_ne!(
            a, b,
            "get_two_mut must never be called with the same archetype twice"
        );
        let (archetype_a, archetype_b) = self.archetypes.get_disjoint_mut(a, b);
        (
            archetype_a.expect("archetype must exist"),
            archetype_b.expect("archetype must exist"),
        )
    }
}

/// Small helper: insert a column built from `make` iff one isn't already
/// present for `id`. Kept free-standing rather than a `SparseShell`-style
/// method, since it's only ever called from inside a `column_mut`-style
/// borrow where `self` isn't conveniently available as `&mut Archetypes`.
fn ensure_column(
    columns: &mut SparseSet<ComponentId, Box<dyn Column>>,
    id: ComponentId,
    make: impl FnOnce() -> Box<dyn Column>,
) {
    if !columns.contains(id) {
        columns.insert(id, make());
    }
}

/// `ensure_column`'s counterpart for `Table::ticks` -- called alongside
/// every `ensure_column` so the two stay at the same key set.
fn ensure_ticks(ticks: &mut SparseSet<ComponentId, Vec<ComponentTicks>>, id: ComponentId) {
    if !ticks.contains(id) {
        ticks.insert(id, Vec::new());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mid_collections::GenerationalIndexAllocator;

    fn entity_factory() -> impl FnMut() -> Entity {
        let mut allocator = GenerationalIndexAllocator::new();
        move || Entity::from_generational_index(allocator.allocate())
    }

    #[derive(Debug, PartialEq, Clone, Copy)]
    struct A(u32);
    #[derive(Debug, PartialEq, Clone, Copy)]
    struct B(u32);
    #[derive(Debug, PartialEq, Clone, Copy)]
    struct C(u32);

    #[test]
    fn spawn_gives_a_location_in_the_empty_archetype() {
        let mut spawn = entity_factory();
        let mut ar = Archetypes::new();
        let e = spawn();
        ar.spawn(e);
        assert!(
            !ar.has(e, ComponentId(0)),
            "not registered at all yet, must not panic"
        );
    }

    #[test]
    fn insert_then_get() {
        let mut spawn = entity_factory();
        let mut ar = Archetypes::new();
        let e = spawn();
        ar.spawn(e);

        let id_a = ar.component_id::<A>();
        assert!(ar.insert(e, id_a, A(42), Tick::new(1)));
        assert_eq!(ar.get::<A>(e, id_a), Some(&A(42)));
        assert!(ar.has(e, id_a));
    }

    #[test]
    fn insert_same_component_twice_is_a_no_op() {
        let mut spawn = entity_factory();
        let mut ar = Archetypes::new();
        let e = spawn();
        ar.spawn(e);
        let id_a = ar.component_id::<A>();

        assert!(ar.insert(e, id_a, A(1), Tick::new(1)));
        assert!(
            !ar.insert(e, id_a, A(999), Tick::new(1)),
            "second insert of the same component must be a no-op"
        );
        assert_eq!(
            ar.get::<A>(e, id_a),
            Some(&A(1)),
            "the original value must be untouched"
        );
    }

    #[test]
    fn insert_two_components_sequentially_preserves_the_first() {
        // The actual point of migration: adding B must not lose A.
        let mut spawn = entity_factory();
        let mut ar = Archetypes::new();
        let e = spawn();
        ar.spawn(e);
        let id_a = ar.component_id::<A>();
        let id_b = ar.component_id::<B>();

        assert!(ar.insert(e, id_a, A(10), Tick::new(1)));
        assert!(ar.insert(e, id_b, B(20), Tick::new(1)));

        assert_eq!(
            ar.get::<A>(e, id_a),
            Some(&A(10)),
            "A must survive the migration triggered by adding B"
        );
        assert_eq!(ar.get::<B>(e, id_b), Some(&B(20)));
    }

    #[test]
    fn insert_three_components_preserves_all_previous() {
        let mut spawn = entity_factory();
        let mut ar = Archetypes::new();
        let e = spawn();
        ar.spawn(e);
        let id_a = ar.component_id::<A>();
        let id_b = ar.component_id::<B>();
        let id_c = ar.component_id::<C>();

        ar.insert(e, id_a, A(1), Tick::new(1));
        ar.insert(e, id_b, B(2), Tick::new(1));
        ar.insert(e, id_c, C(3), Tick::new(1));

        assert_eq!(ar.get::<A>(e, id_a), Some(&A(1)));
        assert_eq!(ar.get::<B>(e, id_b), Some(&B(2)));
        assert_eq!(ar.get::<C>(e, id_c), Some(&C(3)));
    }

    #[test]
    fn remove_returns_the_value_and_migrates_correctly() {
        let mut spawn = entity_factory();
        let mut ar = Archetypes::new();
        let e = spawn();
        ar.spawn(e);
        let id_a = ar.component_id::<A>();
        let id_b = ar.component_id::<B>();
        ar.insert(e, id_a, A(10), Tick::new(1));
        ar.insert(e, id_b, B(20), Tick::new(1));

        let removed = ar.remove::<B>(e, id_b);
        assert_eq!(removed, Some(B(20)));
        assert_eq!(ar.get::<B>(e, id_b), None);
        assert_eq!(
            ar.get::<A>(e, id_a),
            Some(&A(10)),
            "A must survive removing B"
        );
    }

    #[test]
    fn remove_missing_component_returns_none() {
        let mut spawn = entity_factory();
        let mut ar = Archetypes::new();
        let e = spawn();
        ar.spawn(e);
        let id_a = ar.component_id::<A>();
        assert_eq!(ar.remove::<A>(e, id_a), None);
    }

    #[test]
    fn remove_never_registered_component_returns_none_not_panic() {
        let mut spawn = entity_factory();
        let mut ar = Archetypes::new();
        let e = spawn();
        ar.spawn(e);
        assert_eq!(ar.remove::<A>(e, ComponentId(0)), None);
    }

    #[test]
    fn remove_all_components_returns_entity_to_the_empty_archetype() {
        let mut spawn = entity_factory();
        let mut ar = Archetypes::new();
        let e = spawn();
        ar.spawn(e);
        let id_a = ar.component_id::<A>();
        ar.insert(e, id_a, A(5), Tick::new(1));
        ar.remove::<A>(e, id_a);

        // Back in the empty archetype -- inserting A again must work
        // exactly like the very first time.
        assert!(ar.insert(e, id_a, A(99), Tick::new(1)));
        assert_eq!(ar.get::<A>(e, id_a), Some(&A(99)));
    }

    #[test]
    fn swap_remove_during_migration_fixes_up_the_swapped_entitys_row() {
        // The single most important correctness property of this whole
        // module: when a middle entity migrates out of a shared
        // archetype, the entity that gets swapped into its old row must
        // keep working correctly afterward -- not read stale/wrong data,
        // not have ITS OWN future migrations touch the wrong row.
        let mut spawn = entity_factory();
        let mut ar = Archetypes::new();
        let id_a = ar.component_id::<A>();
        let id_b = ar.component_id::<B>();

        let e1 = spawn();
        let e2 = spawn();
        let e3 = spawn();
        ar.spawn(e1);
        ar.spawn(e2);
        ar.spawn(e3);

        // All three land in the same {A} archetype, e1 row 0, e2 row 1,
        // e3 row 2 (insertion order).
        ar.insert(e1, id_a, A(1), Tick::new(1));
        ar.insert(e2, id_a, A(2), Tick::new(1));
        ar.insert(e3, id_a, A(3), Tick::new(1));

        // e2 (the middle one, not the last) migrates out by gaining B.
        // e3 (currently last) should get swapped into e2's old row.
        assert!(ar.insert(e2, id_b, B(200), Tick::new(1)));

        // e1 must be completely unaffected.
        assert_eq!(ar.get::<A>(e1, id_a), Some(&A(1)));

        // e3 must still read correctly -- this only works if its
        // tracked row got fixed up after the swap.
        assert_eq!(ar.get::<A>(e3, id_a), Some(&A(3)));
        assert!(!ar.has(e3, id_b));

        // e2 must have both, correctly, in its new archetype.
        assert_eq!(ar.get::<A>(e2, id_a), Some(&A(2)));
        assert_eq!(ar.get::<B>(e2, id_b), Some(&B(200)));

        // And e3's own row must actually be usable for a FURTHER
        // migration afterward -- proves the fix-up wasn't just
        // "readable" but structurally correct for future writes too.
        assert!(ar.insert(e3, id_b, B(300), Tick::new(1)));
        assert_eq!(
            ar.get::<A>(e3, id_a),
            Some(&A(3)),
            "A must survive e3's own later migration too"
        );
        assert_eq!(ar.get::<B>(e3, id_b), Some(&B(300)));
        assert_eq!(
            ar.get::<A>(e1, id_a),
            Some(&A(1)),
            "e1 still must be untouched by any of this"
        );
    }

    #[test]
    fn despawn_removes_entity_and_fixes_up_the_swapped_entity() {
        let mut spawn = entity_factory();
        let mut ar = Archetypes::new();
        let id_a = ar.component_id::<A>();

        let e1 = spawn();
        let e2 = spawn();
        let e3 = spawn();
        ar.spawn(e1);
        ar.spawn(e2);
        ar.spawn(e3);
        ar.insert(e1, id_a, A(1), Tick::new(1));
        ar.insert(e2, id_a, A(2), Tick::new(1));
        ar.insert(e3, id_a, A(3), Tick::new(1));

        // Despawn the middle entity -- e3 should get swapped into its row.
        ar.despawn(e2);

        assert_eq!(ar.get::<A>(e1, id_a), Some(&A(1)));
        assert_eq!(
            ar.get::<A>(e3, id_a),
            Some(&A(3)),
            "e3 must still read correctly after being swapped"
        );

        // And e3's row must still be structurally usable afterward.
        let id_b = ar.component_id::<B>();
        assert!(ar.insert(e3, id_b, B(30), Tick::new(1)));
        assert_eq!(ar.get::<A>(e3, id_a), Some(&A(3)));
    }

    #[test]
    fn despawn_never_spawned_entity_is_a_safe_no_op() {
        let mut spawn = entity_factory();
        let mut ar = Archetypes::new();
        let e = spawn();
        // Never called ar.spawn(e) -- no tracked location at all.
        ar.despawn(e); // must not panic
        assert!(!ar.has(e, ComponentId(0)));
    }

    #[test]
    fn get_mut_actually_mutates() {
        let mut spawn = entity_factory();
        let mut ar = Archetypes::new();
        let e = spawn();
        ar.spawn(e);
        let id_a = ar.component_id::<A>();
        ar.insert(e, id_a, A(1), Tick::new(1));

        ar.get_mut::<A>(e, id_a, Tick::new(2)).unwrap().0 = 999;
        assert_eq!(ar.get::<A>(e, id_a), Some(&A(999)));
    }

    #[test]
    fn edge_cache_routes_different_entities_from_the_same_source_to_the_same_target() {
        // Two entities, same starting archetype ({A}), same component
        // added (B) -- must land in the exact same resulting archetype,
        // proven indirectly: after both migrations, removing A from
        // *both* must independently succeed and leave *both* still
        // holding B, which only works if they're both real, consistent,
        // correctly-tracked archetypes (not, say, one of them silently
        // corrupted by an edge-cache bug returning the wrong target).
        let mut spawn = entity_factory();
        let mut ar = Archetypes::new();
        let id_a = ar.component_id::<A>();
        let id_b = ar.component_id::<B>();

        let e1 = spawn();
        let e2 = spawn();
        ar.spawn(e1);
        ar.spawn(e2);
        ar.insert(e1, id_a, A(1), Tick::new(1));
        ar.insert(e2, id_a, A(2), Tick::new(1));

        ar.insert(e1, id_b, B(10), Tick::new(1));
        ar.insert(e2, id_b, B(20), Tick::new(1));

        assert_eq!(ar.remove::<A>(e1, id_a), Some(A(1)));
        assert_eq!(ar.remove::<A>(e2, id_a), Some(A(2)));
        assert_eq!(ar.get::<B>(e1, id_b), Some(&B(10)));
        assert_eq!(ar.get::<B>(e2, id_b), Some(&B(20)));
    }

    #[test]
    fn many_entities_many_migrations_stay_consistent() {
        // Broader, less targeted exercise across a mixed pattern of
        // spawns/inserts/removes -- not a specific regression case,
        // checking every still-tracked entity reads correctly
        // throughout, not just at the start/end.
        let mut spawn = entity_factory();
        let mut ar = Archetypes::new();
        let id_a = ar.component_id::<A>();
        let id_b = ar.component_id::<B>();
        let id_c = ar.component_id::<C>();

        let mut entities = Vec::new();
        for i in 0..30u32 {
            let e = spawn();
            ar.spawn(e);
            ar.insert(e, id_a, A(i), Tick::new(1));
            if i % 2 == 0 {
                ar.insert(e, id_b, B(i * 10), Tick::new(1));
            }
            if i % 3 == 0 {
                ar.insert(e, id_c, C(i * 100), Tick::new(1));
            }
            entities.push((e, i));
        }

        // Remove B from every third entity that had it.
        for &(e, i) in &entities {
            if i % 6 == 0 {
                ar.remove::<B>(e, id_b);
            }
        }

        for &(e, i) in &entities {
            assert_eq!(
                ar.get::<A>(e, id_a),
                Some(&A(i)),
                "A must always still be correct for entity {i}"
            );
            if i % 2 == 0 && i % 6 != 0 {
                assert_eq!(ar.get::<B>(e, id_b), Some(&B(i * 10)));
            } else if i % 6 == 0 {
                assert_eq!(
                    ar.get::<B>(e, id_b),
                    None,
                    "B was explicitly removed for entity {i}"
                );
            }
            if i % 3 == 0 {
                assert_eq!(ar.get::<C>(e, id_c), Some(&C(i * 100)));
            }
        }
    }

    // Deliberately distinct from A/B/C above, which are NOT
    // `#[repr(C)]` and must stay that way to prove `register_ffi`'s
    // extra bounds don't leak onto every component type here, only
    // ones that opt in -- mirrors `component.rs::tests::FfiPosition`'s
    // exact reasoning.
    #[derive(Debug, Clone, Copy, PartialEq, IntoBytes, KnownLayout, Immutable)]
    #[repr(C)]
    struct FfiHealth {
        hp: u32,
    }

    #[test]
    fn raw_span_on_a_never_registered_component_id_is_none() {
        let ar = Archetypes::new();
        assert_eq!(ar.raw_span(EMPTY_ARCHETYPE, ComponentId(0)), None);
    }

    #[test]
    fn raw_span_on_an_archetype_that_does_not_have_this_component_is_none() {
        let mut spawn = entity_factory();
        let mut ar = Archetypes::new();
        let id = ar.register_ffi::<FfiHealth>("FfiHealth");
        let e = spawn();
        ar.spawn(e);
        // e lives in the empty archetype -- FfiHealth is registered for
        // FFI, but this specific archetype's signature doesn't include
        // it, which is a real, permanent fact about this archetype, not
        // "not populated yet".
        assert_eq!(ar.raw_span(EMPTY_ARCHETYPE, id), None);
    }

    #[test]
    fn register_ffi_then_raw_span_reflects_attached_values() {
        let mut spawn = entity_factory();
        let mut ar = Archetypes::new();
        let id = ar.register_ffi::<FfiHealth>("FfiHealth");
        let e1 = spawn();
        let e2 = spawn();
        ar.spawn(e1);
        ar.spawn(e2);
        assert!(ar.insert(e1, id, FfiHealth { hp: 10 }, Tick::new(1)));
        assert!(ar.insert(e2, id, FfiHealth { hp: 20 }, Tick::new(1)));

        let archetype_id = ar
            .archetypes_with(id)
            .next()
            .expect("both entities must share one archetype");
        let span = ar
            .raw_span(archetype_id, id)
            .expect("registered and populated");
        assert_eq!(span.count, 2);
        // SAFETY: span points at ar's own live storage, unmutated since
        // the inserts above.
        let values =
            unsafe { core::slice::from_raw_parts(span.ptr.cast::<FfiHealth>(), span.count) };
        assert_eq!(values[0], FfiHealth { hp: 10 });
        assert_eq!(values[1], FfiHealth { hp: 20 });
    }

    #[test]
    fn raw_span_is_some_and_empty_after_every_entity_migrates_away() {
        let mut spawn = entity_factory();
        let mut ar = Archetypes::new();
        let id = ar.register_ffi::<FfiHealth>("FfiHealth");
        let e = spawn();
        ar.spawn(e);
        assert!(ar.insert(e, id, FfiHealth { hp: 5 }, Tick::new(1)));
        let archetype_id = ar
            .archetypes_with(id)
            .next()
            .expect("must exist right after insert");

        // Remove the component -- e migrates back to the empty
        // archetype, leaving `archetype_id`'s table with zero rows but
        // its FfiHealth column still present (columns are never removed
        // once an archetype has been created with that signature).
        assert_eq!(ar.remove::<FfiHealth>(e, id), Some(FfiHealth { hp: 5 }));

        let span = ar
            .raw_span(archetype_id, id)
            .expect("the column itself is still genuinely present, just empty");
        assert_eq!(
            span.count, 0,
            "empty-but-present must be Some(count=0), not None"
        );
    }

    #[derive(Debug, Clone, Copy, PartialEq, IntoBytes, KnownLayout, Immutable)]
    #[repr(C)]
    struct FfiMana {
        mp: u32,
    }

    #[test]
    fn raw_span_and_entity_ids_are_empty_not_none_for_a_column_less_intermediate_archetype() {
        // A bundle walks `edge_for_insert` once per element, so
        // `(FfiHealth, FfiMana)` creates a `{FfiHealth}` archetype on the
        // way to `{FfiHealth, FfiMana}` that no entity is ever moved
        // into, and so never gets a column.
        let mut spawn = entity_factory();
        let mut ar = Archetypes::new();
        let health = ar.register_ffi::<FfiHealth>("FfiHealth");
        ar.register_ffi::<FfiMana>("FfiMana");
        let e = spawn();
        ar.spawn(e);
        assert!(ar.insert_bundle(e, (FfiHealth { hp: 1 }, FfiMana { mp: 2 }), Tick::new(1)));

        let mut counts = Vec::new();
        for archetype_id in ar.archetypes_with(health) {
            let span = ar
                .raw_span(archetype_id, health)
                .expect("every archetype archetypes_with hands out must resolve, column or not");
            let ids = ar
                .entity_ids(archetype_id, health)
                .expect("entity_ids agrees with raw_span");
            assert_eq!(span.count, ids.len());
            counts.push(span.count);
        }
        counts.sort_unstable();
        assert_eq!(counts, vec![0, 1], "one empty intermediate, one real row");
    }

    #[test]
    fn raw_span_is_still_none_for_a_signature_without_the_component() {
        let mut spawn = entity_factory();
        let mut ar = Archetypes::new();
        let health = ar.register_ffi::<FfiHealth>("FfiHealth");
        let mana = ar.register_ffi::<FfiMana>("FfiMana");
        let e = spawn();
        ar.spawn(e);
        assert!(ar.insert(e, health, FfiHealth { hp: 1 }, Tick::new(1)));
        let health_only = ar.archetypes_with(health).next().unwrap();
        assert_eq!(ar.raw_span(health_only, mana), None);
    }

    /// `[e1: A] [e2: A+B] [e3: B] [e4: A+B+C]`, one archetype each.
    fn matching_fixture() -> (Archetypes, [ComponentId; 3], [ArchetypeId; 4]) {
        let mut spawn = entity_factory();
        let mut ar = Archetypes::new();
        let a = ar.component_id::<A>();
        let b = ar.component_id::<B>();
        let c = ar.component_id::<C>();
        let e1 = spawn();
        let e2 = spawn();
        let e3 = spawn();
        let e4 = spawn();
        for e in [e1, e2, e3, e4] {
            ar.spawn(e);
        }
        assert!(ar.insert(e1, a, A(1), Tick::new(1)));
        assert!(ar.insert(e2, a, A(2), Tick::new(1)));
        assert!(ar.insert(e2, b, B(2), Tick::new(1)));
        assert!(ar.insert(e3, b, B(3), Tick::new(1)));
        assert!(ar.insert(e4, a, A(4), Tick::new(1)));
        assert!(ar.insert(e4, b, B(4), Tick::new(1)));
        assert!(ar.insert(e4, c, C(4), Tick::new(1)));
        let at = |e| ar.locations.get(e).unwrap().archetype_id;
        let archetypes = [at(e1), at(e2), at(e3), at(e4)];
        (ar, [a, b, c], archetypes)
    }

    fn matching(
        ar: &Archetypes,
        with: &[ComponentId],
        without: &[ComponentId],
    ) -> Vec<ArchetypeId> {
        matching_any(ar, with, without, &[])
    }

    fn matching_any(
        ar: &Archetypes,
        with: &[ComponentId],
        without: &[ComponentId],
        any_of: &[ComponentId],
    ) -> Vec<ArchetypeId> {
        let mut v: Vec<ArchetypeId> = ar.archetypes_matching(with, without, any_of).collect();
        v.sort_by_key(|id| id.as_u32());
        v
    }

    fn sorted_ids(mut v: Vec<ArchetypeId>) -> Vec<ArchetypeId> {
        v.sort_by_key(|id| id.as_u32());
        v
    }

    #[test]
    fn archetypes_matching_with_only_equals_archetypes_with() {
        let (ar, [a, ..], _) = matching_fixture();
        let expected: Vec<ArchetypeId> = ar.archetypes_with(a).collect();
        let actual: Vec<ArchetypeId> = ar.archetypes_matching(&[a], &[], &[]).collect();
        assert_eq!(actual, expected, "same set, same dense order");
    }

    #[test]
    fn archetypes_matching_requires_all_of_with_and_none_of_without() {
        let (ar, [a, b, c], [only_a, a_b, only_b, a_b_c]) = matching_fixture();
        assert_eq!(matching(&ar, &[a], &[b]), sorted_ids(vec![only_a]));
        assert_eq!(matching(&ar, &[a, b], &[]), sorted_ids(vec![a_b, a_b_c]));
        assert_eq!(matching(&ar, &[a, b], &[c]), sorted_ids(vec![a_b]));
        assert_eq!(matching(&ar, &[b], &[a]), sorted_ids(vec![only_b]));
        assert_eq!(matching(&ar, &[], &[a]), {
            // Everything without A, which includes the empty archetype
            // (id 0) and the never-populated intermediates, not just
            // e3's.
            let mut all_without_a: Vec<ArchetypeId> = ar
                .archetypes
                .iter()
                .filter(|(_, arch)| !arch.component_ids.contains(&a))
                .map(|(id, _)| id)
                .collect();
            all_without_a.sort_by_key(|id| id.as_u32());
            all_without_a
        });
    }

    #[test]
    fn archetypes_matching_with_both_lists_empty_is_every_archetype() {
        let (ar, _, _) = matching_fixture();
        let all: Vec<ArchetypeId> = ar.archetypes.iter().map(|(id, _)| id).collect();
        let matched: Vec<ArchetypeId> = ar.archetypes_matching(&[], &[], &[]).collect();
        assert_eq!(matched, all);
        assert!(matched.contains(&EMPTY_ARCHETYPE));
    }

    #[test]
    fn archetypes_matching_any_of_requires_at_least_one() {
        let (ar, [a, b, c], [only_a, a_b, only_b, a_b_c]) = matching_fixture();
        // any_of {A, C}: matches only_a (A), a_b_c (A and C), not only_b
        // (neither), not a_b (A but that's covered; b_only lacks both).
        assert_eq!(
            matching_any(&ar, &[], &[], &[a, c]),
            sorted_ids(vec![only_a, a_b, a_b_c]),
            "a_b has A, so it's included too — any_of is 'at least one', not 'only'"
        );
        // Combined with `with`: (has B) and (A or C).
        assert_eq!(
            matching_any(&ar, &[b], &[], &[a, c]),
            sorted_ids(vec![a_b, a_b_c])
        );
        // Combined with `without`: (A or C) and not B.
        assert_eq!(
            matching_any(&ar, &[], &[b], &[a, c]),
            sorted_ids(vec![only_a])
        );
        // any_of naming only ids nothing has: matches nothing.
        let unknown = ComponentId(9999);
        assert!(matching_any(&ar, &[], &[], &[unknown]).is_empty());
        // Contradiction between with and any_of is still just AND-then-OR:
        // with={B}, any_of={A} -> needs B AND (A) -> a_b, a_b_c.
        assert_eq!(
            matching_any(&ar, &[b], &[], &[a]),
            sorted_ids(vec![a_b, a_b_c])
        );
    }

    #[test]
    fn archetypes_matching_empty_any_of_is_no_constraint_not_match_nothing() {
        let (ar, [a, ..], _) = matching_fixture();
        assert_eq!(
            matching_any(&ar, &[a], &[], &[]),
            matching(&ar, &[a], &[]),
            "an empty any_of behaves exactly like archetypes_matching before any_of existed"
        );
    }

    #[test]
    fn archetypes_matching_edge_cases() {
        let (ar, [a, b, ..], _) = matching_fixture();
        let unknown = ComponentId(9999);

        // Contradiction matches nothing.
        assert!(matching(&ar, &[a], &[a]).is_empty());
        // An id no archetype contains: `with` matches nothing,
        // `without` changes nothing.
        assert!(matching(&ar, &[unknown], &[]).is_empty());
        assert_eq!(matching(&ar, &[a], &[unknown]), matching(&ar, &[a], &[]));
        // Duplicates are harmless.
        assert_eq!(matching(&ar, &[a, a], &[b, b]), matching(&ar, &[a], &[b]));
    }

    #[test]
    fn archetypes_with_is_empty_for_an_unregistered_component() {
        let ar = Archetypes::new();
        assert_eq!(ar.archetypes_with(ComponentId(0)).count(), 0);
    }

    #[test]
    fn archetypes_with_finds_every_archetype_containing_the_component() {
        let mut spawn = entity_factory();
        let mut ar = Archetypes::new();
        let health_id = ar.register_ffi::<FfiHealth>("FfiHealth");
        let a_id = ar.component_id::<A>();

        // e1: just FfiHealth. e2: FfiHealth + A (a different archetype).
        let e1 = spawn();
        let e2 = spawn();
        ar.spawn(e1);
        ar.spawn(e2);
        assert!(ar.insert(e1, health_id, FfiHealth { hp: 1 }, Tick::new(1)));
        assert!(ar.insert(e2, health_id, FfiHealth { hp: 2 }, Tick::new(1)));
        assert!(ar.insert(e2, a_id, A(99), Tick::new(1)));

        let archetypes_with_health: Vec<_> = ar.archetypes_with(health_id).collect();
        assert_eq!(
            archetypes_with_health.len(),
            2,
            "e1 and e2 ended up in two distinct archetypes, both containing FfiHealth"
        );
    }

    #[test]
    fn entity_ids_on_a_never_registered_component_id_is_none() {
        let ar = Archetypes::new();
        assert_eq!(ar.entity_ids(EMPTY_ARCHETYPE, ComponentId(0)), None);
    }

    #[test]
    fn archetype_id_as_u32_from_u32_round_trips() {
        assert_eq!(
            ArchetypeId::from_u32(EMPTY_ARCHETYPE.as_u32()),
            EMPTY_ARCHETYPE
        );
        let mut spawn = entity_factory();
        let mut ar = Archetypes::new();
        let id = ar.register_ffi::<FfiHealth>("FfiHealth");
        let e = spawn();
        ar.spawn(e);
        assert!(ar.insert(e, id, FfiHealth { hp: 1 }, Tick::new(1)));
        let archetype_id = ar.archetypes_with(id).next().expect("just inserted");
        assert_eq!(ArchetypeId::from_u32(archetype_id.as_u32()), archetype_id);
    }

    #[test]
    fn entity_ids_on_an_archetype_that_does_not_have_this_component_is_none() {
        let mut spawn = entity_factory();
        let mut ar = Archetypes::new();
        let id = ar.register_ffi::<FfiHealth>("FfiHealth");
        let e = spawn();
        ar.spawn(e);
        // Same permanent-not-transient distinction as raw_span: e lives
        // in the empty archetype, whose signature will never include
        // FfiHealth.
        assert_eq!(ar.entity_ids(EMPTY_ARCHETYPE, id), None);
    }

    #[test]
    fn entity_ids_is_some_and_empty_after_every_entity_migrates_away() {
        let mut spawn = entity_factory();
        let mut ar = Archetypes::new();
        let id = ar.register_ffi::<FfiHealth>("FfiHealth");
        let e = spawn();
        ar.spawn(e);
        assert!(ar.insert(e, id, FfiHealth { hp: 5 }, Tick::new(1)));
        let archetype_id = ar
            .archetypes_with(id)
            .next()
            .expect("must exist right after insert");

        assert_eq!(ar.remove::<FfiHealth>(e, id), Some(FfiHealth { hp: 5 }));

        assert_eq!(
            ar.entity_ids(archetype_id, id),
            Some(Vec::new()),
            "empty-but-present must be Some(empty), not None, matching raw_span's own convention"
        );
    }

    #[test]
    fn entity_ids_correlate_with_raw_span_in_dense_order() {
        let mut spawn = entity_factory();
        let mut ar = Archetypes::new();
        let id = ar.register_ffi::<FfiHealth>("FfiHealth");
        let e1 = spawn();
        let e2 = spawn();
        ar.spawn(e1);
        ar.spawn(e2);
        assert!(ar.insert(e1, id, FfiHealth { hp: 10 }, Tick::new(1)));
        assert!(ar.insert(e2, id, FfiHealth { hp: 20 }, Tick::new(1)));

        let archetype_id = ar
            .archetypes_with(id)
            .next()
            .expect("both entities must share one archetype");
        let ids = ar
            .entity_ids(archetype_id, id)
            .expect("registered and present");
        let span = ar
            .raw_span(archetype_id, id)
            .expect("registered and populated");
        assert_eq!(ids.len(), span.count);
        // SAFETY: span points at ar's own live storage, unmutated since
        // the inserts above.
        let values =
            unsafe { core::slice::from_raw_parts(span.ptr.cast::<FfiHealth>(), span.count) };
        assert_eq!(Entity::from_ffi(ids[0]), e1);
        assert_eq!(Entity::from_ffi(ids[1]), e2);
        assert_eq!(values[0], FfiHealth { hp: 10 });
        assert_eq!(values[1], FfiHealth { hp: 20 });
    }

    #[test]
    fn entity_ids_still_correlate_after_a_swap_remove_during_migration() {
        // Same setup as swap_remove_during_migration_fixes_up_the_swapped_entitys_row
        // above, but checking entity_ids/raw_span correlation survives
        // the fixup rather than checking get() directly.
        let mut spawn = entity_factory();
        let mut ar = Archetypes::new();
        let health_id = ar.register_ffi::<FfiHealth>("FfiHealth");
        let a_id = ar.component_id::<A>();

        let e1 = spawn();
        let e2 = spawn();
        let e3 = spawn();
        ar.spawn(e1);
        ar.spawn(e2);
        ar.spawn(e3);

        // All three land in the same {FfiHealth} archetype, e1 row 0,
        // e2 row 1, e3 row 2 (insertion order). The archetype itself
        // doesn't exist until the first insert creates it via
        // migration, so it's captured after, not before.
        assert!(ar.insert(e1, health_id, FfiHealth { hp: 1 }, Tick::new(1)));
        assert!(ar.insert(e2, health_id, FfiHealth { hp: 2 }, Tick::new(1)));
        assert!(ar.insert(e3, health_id, FfiHealth { hp: 3 }, Tick::new(1)));
        let archetype_id = ar
            .archetypes_with(health_id)
            .next()
            .expect("must exist right after the inserts above");

        // e2 (the middle one, not the last) migrates out of
        // archetype_id by gaining A. e3 (currently last in
        // archetype_id's table) gets swapped into e2's old row.
        assert!(ar.insert(e2, a_id, A(200), Tick::new(1)));

        let ids = ar
            .entity_ids(archetype_id, health_id)
            .expect("still registered and present (e1, and now e3, remain)");
        let span = ar
            .raw_span(archetype_id, health_id)
            .expect("still registered and present");
        assert_eq!(ids.len(), 2);
        assert_eq!(span.count, 2);
        // SAFETY: span points at ar's own live storage, unmutated since
        // the insert above.
        let values =
            unsafe { core::slice::from_raw_parts(span.ptr.cast::<FfiHealth>(), span.count) };
        for i in 0..2 {
            let owner = Entity::from_ffi(ids[i]);
            let expected = if owner == e1 {
                FfiHealth { hp: 1 }
            } else if owner == e3 {
                FfiHealth { hp: 3 }
            } else {
                panic!(
                    "entity_ids[{i}] names an entity that was never in this archetype: {owner:?}"
                );
            };
            assert_eq!(values[i], expected);
        }
    }

    #[test]
    fn lookup_ffi_id_resolves_a_registered_name() {
        let mut ar = Archetypes::new();
        let id = ar.register_ffi::<FfiHealth>("FfiHealth");
        assert_eq!(ar.lookup_ffi_id("FfiHealth"), Some(id));
        assert_eq!(ar.lookup_ffi_id("NeverRegistered"), None);
    }

    #[test]
    fn register_ffi_is_idempotent_for_the_same_type() {
        let mut ar = Archetypes::new();
        let id1 = ar.register_ffi::<FfiHealth>("FfiHealth");
        let id2 = ar.register_ffi::<FfiHealth>("FfiHealth");
        assert_eq!(id1, id2);
    }

    #[test]
    #[should_panic(expected = "already registered for a different component type")]
    fn register_ffi_panics_on_a_name_collision_between_distinct_types() {
        #[derive(Debug, Clone, Copy, PartialEq, IntoBytes, KnownLayout, Immutable)]
        #[repr(C)]
        struct OtherFfiType {
            v: u32,
        }
        let mut ar = Archetypes::new();
        ar.register_ffi::<FfiHealth>("SameName");
        ar.register_ffi::<OtherFfiType>("SameName");
    }
}
