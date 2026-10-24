// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/mid-ecs.md, section "Writing component data from C"
// ============================================================================
//! Writing Archetype Core component data through the FFI: the pieces
//! `mid_ecs_world_static_component_raw_span_mut` and
//! `mid_ecs_world_static_component_write` are built on, in safe Rust.
//!
//! The read-only spans (`FfiSpan`, `const` pointer) stay as they were. A
//! component type becomes *writable* from C only if it is registered with
//! [`Archetypes::register_ffi_writable`], which requires
//! `zerocopy::FromBytes` on top of the read-only bounds: C writes raw
//! bytes, and `FromBytes` is the proof that every bit pattern is a valid
//! value of the type, so no C write can ever create an invalid Rust value
//! (a `bool` holding 2, an enum with no such variant). Resources already
//! work this way (`register_ffi_resource`).
//!
//! Two ways to write, because they answer two different needs:
//! - [`Archetypes::raw_span_mut`], bulk: a writable span over one column
//!   of one archetype. C cannot tell Rust which rows it wrote through a
//!   raw pointer, so the whole column is stamped changed at the moment the
//!   span is handed out, the same moment `get_static_mut` stamps. That
//!   can over-report (a `Changed` consumer sees rows C left alone) but can
//!   never miss a write.
//! - [`Archetypes::write_row_bytes`], precise: copy one value into one
//!   entity's component and stamp exactly that row. There is no
//!   "write now, mark later" mode to forget.
//!
//! No `unsafe` here: the byte views are zerocopy's `as_mut_bytes` and
//! `read_from_bytes`. The raw pointers exist only in the returned
//! [`FfiSpanMut`]; dereferencing them is the C caller's contract, and the
//! `extern "C"` wrappers in `ffi.rs` are the only code that touches them.
//!
//! Child module of `archetype`, not a sibling: `Archetype`, `Table` and
//! the `Column` trait are private there.

use std::any::Any;
use std::mem::size_of;

use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

use super::{
    ensure_column, ensure_ticks, Archetype, ArchetypeId, Archetypes, EntityLocation, Table,
};
use crate::component::ComponentId;
use crate::tick::{ComponentTicks, Tick};
use crate::world::Entity;

/// A plain, C-representable *writable* view into one archetype column:
/// pointer to the first element, size of one element in bytes, element
/// count. The mutable sibling of `mid_collections::FfiSpan`, with the
/// same shape and the same empty sentinel (`ptr` null, `count` 0).
///
/// Valid until the next call that changes that archetype's storage (a
/// spawn, despawn, insert, remove, or applying a command queue). Writes
/// belong in the tick the span was obtained in: the rows were stamped
/// changed at the tick current when it was handed out.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FfiSpanMut {
    /// First element, or null for an empty span.
    pub ptr: *mut u8,
    /// Size of one element in bytes.
    pub stride: usize,
    /// Number of elements.
    pub count: usize,
}

impl FfiSpanMut {
    /// The empty span: `ptr` null, `count` 0. The answer for a real but
    /// row-less column.
    pub const fn empty() -> Self {
        Self {
            ptr: std::ptr::null_mut(),
            stride: 0,
            count: 0,
        }
    }

    /// Whether this is the empty sentinel.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }
}

/// Why an id-based (FFI-facing) component write or insert did not
/// happen. Checked in this order for
/// [`crate::World::write_static_component_bytes`]: `NotAlive`,
/// `UnknownId`, `NotWritable`, `SizeMismatch`, `Absent`; and for
/// [`crate::World::insert_static_bytes`]: `NotAlive`, `UnknownId`,
/// `NotWritable`, `SizeMismatch`, `AlreadyPresent`; and for
/// [`crate::World::remove_static_by_id`]: `NotAlive`, `UnknownId`,
/// `NotWritable`, `Absent`. The list forms
/// ([`crate::World::insert_static_bytes_many`],
/// [`crate::World::remove_static_by_ids`]) are all-or-nothing: `NotAlive`,
/// then each listed component in order (`UnknownId`, `NotWritable`, and
/// for an insert `SizeMismatch`), then `Duplicate`, then `AlreadyPresent`
/// (insert) or `Absent` (remove) if the entity has any / lacks any of
/// them. Nothing changes on any error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComponentFfiError {
    /// The component was never registered with
    /// `register_ffi_static_component[_writable]`, or the archetype does
    /// not exist, or its signature does not include the component: the
    /// same conditions `static_component_raw_span` answers `None` for.
    UnknownId,
    /// The component is registered, but only read-only
    /// (`register_ffi_static_component`, not `..._writable`).
    NotWritable,
    /// The entity is not alive: never spawned here, already despawned, or
    /// a stale handle whose slot was reused.
    NotAlive,
    /// The entity is alive but does not currently have this component
    /// (a write or a removal had nothing to act on).
    Absent,
    /// A write's byte length is not exactly the registered type's size.
    /// Nothing was written or stamped.
    SizeMismatch,
    /// An insert onto an entity that already has the component. Nothing
    /// was written: `World::insert_static` is a no-op there too (the first
    /// value stays), and this says so out loud, because a caller who
    /// handed over bytes would otherwise assume they were stored. Use the
    /// write path to overwrite. For a bundle insert: the entity has *any*
    /// of the listed components, and none of them was inserted.
    AlreadyPresent,
    /// A list of components (a bundle insert or removal) names the same
    /// component twice. The list is rejected as a whole, before the
    /// entity is looked at: every component of a bundle needs its own
    /// column.
    Duplicate,
}

/// Type-erased write operations for one writable component type, built
/// from `T` by [`Archetypes::register_ffi_writable`].
#[derive(Clone, Copy)]
pub(super) struct FfiWriter {
    size: usize,
    span_mut: fn(&mut dyn Any) -> FfiSpanMut,
    write_row: fn(&mut dyn Any, usize, &[u8]) -> bool,
    insert: fn(&mut Archetypes, Entity, ComponentId, &[u8], Tick) -> bool,
    push: fn(&mut Table, ComponentId, &[u8], Tick) -> bool,
}

fn span_mut_of<T>(column: &mut dyn Any) -> FfiSpanMut
where
    T: 'static + FromBytes + IntoBytes + Immutable + KnownLayout,
{
    let values = column
        .downcast_mut::<Vec<T>>()
        .expect("column type must match component_id's T");
    let count = values.len();
    if count == 0 {
        return FfiSpanMut::empty();
    }
    let bytes = values.as_mut_slice().as_mut_bytes();
    FfiSpanMut {
        ptr: bytes.as_mut_ptr(),
        stride: size_of::<T>(),
        count,
    }
}

/// `false` when `bytes` is not exactly `size_of::<T>()` long or `row` is
/// out of range; nothing is written then.
fn write_row_of<T>(column: &mut dyn Any, row: usize, bytes: &[u8]) -> bool
where
    T: 'static + FromBytes + IntoBytes + Immutable + KnownLayout,
{
    let Ok(value) = T::read_from_bytes(bytes) else {
        return false;
    };
    let values = column
        .downcast_mut::<Vec<T>>()
        .expect("column type must match component_id's T");
    match values.get_mut(row) {
        Some(slot) => {
            *slot = value;
            true
        }
        None => false,
    }
}

/// Builds a `T` from `bytes` and inserts it as `entity`'s new component:
/// the type-erased `World::insert_static`. `false` when `bytes` is not
/// exactly `size_of::<T>()` long, or `Archetypes::insert` declines (the
/// entity is untracked or already has the component).
fn insert_of<T>(
    archetypes: &mut Archetypes,
    entity: Entity,
    component_id: ComponentId,
    bytes: &[u8],
    tick: Tick,
) -> bool
where
    T: 'static + FromBytes + IntoBytes + Immutable + KnownLayout,
{
    let Ok(value) = T::read_from_bytes(bytes) else {
        return false;
    };
    archetypes.insert(entity, component_id, value, tick)
}

/// Appends a `T` built from `bytes` to `table`'s column for
/// `component_id` (creating the column if the table has none yet), with
/// fresh ticks stamped added and changed at `tick`: the type-erased push
/// `Bundle::push_into` does per element. `false` when `bytes` is not
/// exactly `size_of::<T>()` long; nothing is pushed then.
fn push_of<T>(table: &mut Table, component_id: ComponentId, bytes: &[u8], tick: Tick) -> bool
where
    T: 'static + FromBytes + IntoBytes + Immutable + KnownLayout,
{
    let Ok(value) = T::read_from_bytes(bytes) else {
        return false;
    };
    ensure_column(&mut table.columns, component_id, || {
        Box::<Vec<T>>::default()
    });
    table
        .columns
        .get_mut(component_id)
        .expect("just ensured present")
        .as_any_mut()
        .downcast_mut::<Vec<T>>()
        .expect("column type must match component_id's T")
        .push(value);
    ensure_ticks(&mut table.ticks, component_id);
    table
        .ticks
        .get_mut(component_id)
        .expect("just ensured present")
        .push(ComponentTicks::new(tick));
    true
}

impl Archetypes {
    /// Whether `id` was registered for the FFI (`register_ffi_static_component`
    /// and its writable form), i.e. whether the untyped id-based accessors
    /// can see it at all. An id issued only through the typed API is not.
    pub(crate) fn is_ffi_registered(&self, id: ComponentId) -> bool {
        self.ffi_accessors.contains_key(&id)
    }

    /// Opts `T` into FFI span exposure under `name` *and* into writes from
    /// C: the read-only registration ([`Self::register_ffi`]) plus a
    /// writer. `T: FromBytes` is the extra bound, see the module docs.
    /// Idempotent. Registering a type read-only first and writable later
    /// upgrades it; registering it read-only after it was writable does
    /// not downgrade it.
    ///
    /// # Panics
    /// Under `register_ffi`'s panic condition: `name` already registered
    /// for a different component type.
    pub(crate) fn register_ffi_writable<T>(&mut self, name: &'static str) -> ComponentId
    where
        T: 'static + FromBytes + IntoBytes + Immutable + KnownLayout,
    {
        let id = self.register_ffi::<T>(name);
        self.ffi_writers.entry(id).or_insert(FfiWriter {
            size: size_of::<T>(),
            span_mut: span_mut_of::<T>,
            write_row: write_row_of::<T>,
            insert: insert_of::<T>,
            push: push_of::<T>,
        });
        id
    }

    /// A writable span over `component_id`'s column in `archetype_id`'s
    /// table, with every row of that column stamped changed at
    /// `this_run` (not added: the values already existed). The mutable
    /// counterpart of [`Self::raw_span`], under the same `UnknownId`
    /// conditions, plus `NotWritable` for a component registered
    /// read-only. A signature that includes the component but has no
    /// rows (an emptied column, or a zero-row intermediate with no
    /// column yet) answers `Ok` with the empty span.
    pub(crate) fn raw_span_mut(
        &mut self,
        archetype_id: ArchetypeId,
        component_id: ComponentId,
        this_run: Tick,
    ) -> Result<FfiSpanMut, ComponentFfiError> {
        if !self.ffi_accessors.contains_key(&component_id) {
            return Err(ComponentFfiError::UnknownId);
        }
        let writer = *self
            .ffi_writers
            .get(&component_id)
            .ok_or(ComponentFfiError::NotWritable)?;
        let archetype = self
            .archetypes
            .get_mut(archetype_id)
            .ok_or(ComponentFfiError::UnknownId)?;
        let Archetype {
            component_ids,
            table,
            ..
        } = archetype;
        let Table { columns, ticks, .. } = table;
        match columns.get_mut(component_id) {
            Some(column) => {
                if let Some(row_ticks) = ticks.get_mut(component_id) {
                    for stamp in row_ticks.iter_mut() {
                        stamp.changed = this_run;
                    }
                }
                Ok((writer.span_mut)(column.as_any_mut()))
            }
            None if component_ids.contains(&component_id) => Ok(FfiSpanMut::empty()),
            None => Err(ComponentFfiError::UnknownId),
        }
    }

    /// Copies `bytes` in as `entity`'s `component_id` value and stamps
    /// exactly that row changed at `this_run`. `entity` must already be
    /// known alive: the caller (`World`) checks, because a stale handle
    /// whose slot was reused must not reach the new occupant's row.
    /// Nothing is written or stamped on any error.
    pub(crate) fn write_row_bytes(
        &mut self,
        entity: Entity,
        component_id: ComponentId,
        bytes: &[u8],
        this_run: Tick,
    ) -> Result<(), ComponentFfiError> {
        if !self.ffi_accessors.contains_key(&component_id) {
            return Err(ComponentFfiError::UnknownId);
        }
        let writer = *self
            .ffi_writers
            .get(&component_id)
            .ok_or(ComponentFfiError::NotWritable)?;
        if bytes.len() != writer.size {
            return Err(ComponentFfiError::SizeMismatch);
        }
        let location = *self
            .locations
            .get(entity)
            .ok_or(ComponentFfiError::Absent)?;
        let archetype = self
            .archetypes
            .get_mut(location.archetype_id)
            .expect("a located entity's archetype exists");
        let Table { columns, ticks, .. } = &mut archetype.table;
        let column = columns
            .get_mut(component_id)
            .ok_or(ComponentFfiError::Absent)?;
        assert!(
            (writer.write_row)(column.as_any_mut(), location.row, bytes),
            "a located entity's row exists in each of its archetype's columns"
        );
        if let Some(stamp) = ticks
            .get_mut(component_id)
            .and_then(|row_ticks| row_ticks.as_mut_slice().get_mut(location.row))
        {
            stamp.changed = this_run;
        }
        Ok(())
    }

    /// The byte size a write or insert of `component_id` must have, if it
    /// is registered writable. Lets a caller (the FFI command queue) check
    /// registration and size without an entity.
    pub(crate) fn write_size(&self, component_id: ComponentId) -> Result<usize, ComponentFfiError> {
        if !self.ffi_accessors.contains_key(&component_id) {
            return Err(ComponentFfiError::UnknownId);
        }
        self.ffi_writers
            .get(&component_id)
            .map(|writer| writer.size)
            .ok_or(ComponentFfiError::NotWritable)
    }

    /// Builds the component from `bytes` and attaches it to `entity`,
    /// migrating the entity into the archetype for its larger component
    /// set, stamped added *and* changed at `this_run`: the byte-driven
    /// counterpart of `World::insert_static`. `entity` must already be
    /// known alive (the caller, `World`, checks, for the same stale-handle
    /// reason as [`Self::write_row_bytes`]). Nothing happens on any error.
    ///
    /// Any span previously handed out for the entity's old or new
    /// archetype is invalidated: this is a structural change.
    pub(crate) fn insert_row_bytes(
        &mut self,
        entity: Entity,
        component_id: ComponentId,
        bytes: &[u8],
        this_run: Tick,
    ) -> Result<(), ComponentFfiError> {
        let size = self.write_size(component_id)?;
        if bytes.len() != size {
            return Err(ComponentFfiError::SizeMismatch);
        }
        if self.has(entity, component_id) {
            return Err(ComponentFfiError::AlreadyPresent);
        }
        let insert = self
            .ffi_writers
            .get(&component_id)
            .map(|writer| writer.insert)
            .expect("write_size just found the writer");
        if insert(self, entity, component_id, bytes, this_run) {
            Ok(())
        } else {
            // The entity has no tracked location: not alive as far as the
            // Archetype Core is concerned.
            Err(ComponentFfiError::NotAlive)
        }
    }

    /// Moves `entity` into the archetype for its signature without
    /// `component_id`, carrying every other column's value (and ticks)
    /// across, and *drops* the removed value. `false` if `entity` has no
    /// tracked location or does not have the component. The one-component
    /// case of [`Self::remove_many_dropping`].
    pub(crate) fn remove_dropping(&mut self, entity: Entity, component_id: ComponentId) -> bool {
        self.remove_many_dropping(entity, &[component_id])
    }

    /// Moves `entity` into the archetype for its signature without
    /// `ids`, in one migration: every other column's value (and ticks)
    /// comes across, and the values in `ids` are *dropped*. `false`, with
    /// nothing changed, if `entity` has no tracked location or lacks *any*
    /// of `ids` (no partial application). `ids` must not repeat a
    /// component.
    ///
    /// This is the migration of [`Self::remove`] and [`Self::remove_bundle`]
    /// with their type-specific steps replaced by the type-erased
    /// `Column::swap_remove_and_drop`, so it needs no `T`. It is a
    /// deliberate sibling rather than a refactor of those, which are
    /// benchmarked and left byte-for-byte as they were; the differential
    /// tests `remove_by_id_matches_the_typed_remove_step_for_step` and
    /// `bundle_ops_match_the_typed_bundles_step_for_step` keep it in step
    /// with both.
    pub(crate) fn remove_many_dropping(&mut self, entity: Entity, ids: &[ComponentId]) -> bool {
        let Some(&from_location) = self.locations.get(entity) else {
            return false;
        };
        let from_id = from_location.archetype_id;
        let Some(from) = self.archetypes.get(from_id) else {
            return false;
        };
        if !ids.iter().all(|id| from.component_ids.contains(id)) {
            return false;
        }
        if ids.is_empty() {
            return true;
        }
        debug_assert!(
            ids.iter().enumerate().all(|(i, id)| !ids[..i].contains(id)),
            "remove_many_dropping must not be given a repeated component"
        );

        let to_id = self.edge_for_remove_bundle(from_id, ids);
        let (from_archetype, to_archetype) = self.get_two_mut(from_id, to_id);

        for (comp, column) in from_archetype.table.columns.iter_mut() {
            if ids.contains(&comp) {
                column.swap_remove_and_drop(from_location.row);
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
            column.move_row_to(from_location.row, &mut ***dest);
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
        true
    }

    /// Checks, changing nothing, a list of components an FFI bundle call
    /// names: each must be registered writable (and, when a byte length is
    /// given, that length must be its size), checked in list order, and
    /// then no component may repeat. The checks every bytes- or id-driven
    /// bundle operation makes before it looks at an entity, shared so the
    /// queue-time check and the real one cannot disagree.
    pub(crate) fn check_list(
        &self,
        entries: impl IntoIterator<Item = (ComponentId, Option<usize>)>,
    ) -> Result<(), ComponentFfiError> {
        let mut seen: Vec<ComponentId> = Vec::new();
        for (id, len) in entries {
            let size = self.write_size(id)?;
            if len.is_some_and(|len| len != size) {
                return Err(ComponentFfiError::SizeMismatch);
            }
            seen.push(id);
        }
        for (i, id) in seen.iter().enumerate() {
            if seen[..i].contains(id) {
                return Err(ComponentFfiError::Duplicate);
            }
        }
        Ok(())
    }

    /// Attaches every `(component, bytes)` in `entries` to `entity` in
    /// *one* migration, each value built from its bytes and stamped added
    /// and changed at `this_run`: the byte-driven counterpart of
    /// `insert_bundle`, whose migration this repeats (see
    /// `bundle_ops_match_the_typed_bundles_step_for_step`). All or nothing:
    /// the list is validated as a unit ([`Self::check_list`]) and against
    /// the entity before anything moves, and `AlreadyPresent` means the
    /// entity has *any* of the components. An empty list is `Ok` and
    /// changes nothing. `entity` must already be known alive.
    ///
    /// A structural change: every span previously handed out for the
    /// entity's old and new archetype is invalidated.
    pub(crate) fn insert_many_bytes(
        &mut self,
        entity: Entity,
        entries: &[(ComponentId, &[u8])],
        this_run: Tick,
    ) -> Result<(), ComponentFfiError> {
        self.check_list(entries.iter().map(|(id, bytes)| (*id, Some(bytes.len()))))?;
        let Some(&from_location) = self.locations.get(entity) else {
            return Err(ComponentFfiError::NotAlive);
        };
        let from_id = from_location.archetype_id;
        let from = self
            .archetypes
            .get(from_id)
            .ok_or(ComponentFfiError::NotAlive)?;
        if entries
            .iter()
            .any(|(id, _)| from.component_ids.contains(id))
        {
            return Err(ComponentFfiError::AlreadyPresent);
        }
        if entries.is_empty() {
            return Ok(());
        }
        let writers: Vec<FfiWriter> = entries
            .iter()
            .map(|(id, _)| {
                *self
                    .ffi_writers
                    .get(id)
                    .expect("check_list found each component's writer")
            })
            .collect();

        let mut to_id = from_id;
        for (id, _) in entries {
            to_id = self.edge_for_insert(to_id, *id);
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
            column.move_row_to(from_location.row, &mut ***dest);
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
        for ((id, bytes), writer) in entries.iter().zip(&writers) {
            assert!(
                (writer.push)(&mut to_archetype.table, *id, bytes, this_run),
                "check_list already verified every value's length"
            );
        }

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
        Ok(())
    }

    /// Detaches every component in `ids` from `entity` in one migration and
    /// drops their values: the id-driven counterpart of `remove_bundle`.
    /// All or nothing: validated as a unit ([`Self::check_list`], without
    /// sizes) before anything moves, and `Absent` means the entity lacks
    /// *any* of them, with nothing removed. An empty list is `Ok` and
    /// changes nothing. Needs no bytes and stamps nothing. `entity` must
    /// already be known alive, for the same stale-handle reason as
    /// [`Self::write_row_bytes`]. A structural change.
    pub(crate) fn remove_many_rows(
        &mut self,
        entity: Entity,
        ids: &[ComponentId],
    ) -> Result<(), ComponentFfiError> {
        self.check_list(ids.iter().map(|id| (*id, None)))?;
        if ids.is_empty() {
            return Ok(());
        }
        if self.remove_many_dropping(entity, ids) {
            Ok(())
        } else {
            Err(ComponentFfiError::Absent)
        }
    }

    /// Detaches `component_id` from `entity` for the FFI: the id-driven
    /// counterpart of `World::remove_static`, for components registered
    /// writable. The one-component case of [`Self::remove_many_rows`].
    /// Whether [`Self::remove_many_rows`] of `ids` from `entity` would
    /// remove something: the list passes the same checks (known, writable,
    /// no duplicates), is not empty, and the entity has all of them. Reads
    /// only: what the before-removal hooks need to decide whether to fire,
    /// so a refused removal never runs a hook.
    pub(crate) fn would_remove_rows(&self, entity: Entity, ids: &[ComponentId]) -> bool {
        if ids.is_empty() || self.check_list(ids.iter().map(|id| (*id, None))).is_err() {
            return false;
        }
        let have = self.components_of(entity);
        ids.iter().all(|id| have.contains(id))
    }

    pub(crate) fn remove_row(
        &mut self,
        entity: Entity,
        component_id: ComponentId,
    ) -> Result<(), ComponentFfiError> {
        self.remove_many_rows(entity, &[component_id])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tick::ChangeTracker;
    use crate::world::World;

    #[derive(Debug, Clone, Copy, PartialEq, FromBytes, IntoBytes, Immutable, KnownLayout)]
    #[repr(C)]
    struct Hp(u32);

    #[derive(Debug, Clone, Copy, PartialEq, FromBytes, IntoBytes, Immutable, KnownLayout)]
    #[repr(C)]
    struct Pair {
        a: u16,
        b: u16,
    }

    /// Registered read-only only.
    #[derive(Debug, Clone, Copy, PartialEq, IntoBytes, Immutable, KnownLayout)]
    #[repr(C)]
    struct Ro(u32);

    /// Zero-sized on purpose.
    #[derive(Debug, Clone, Copy, PartialEq, FromBytes, IntoBytes, Immutable, KnownLayout)]
    #[repr(C)]
    struct Unit;

    /// `n` entities holding `Hp(1..=n)`, in one archetype, `Hp` registered
    /// writable, and a tracker that has already seen them: writes made
    /// after this are the only things `query_changed` should report.
    fn tracked_hp(n: u32) -> (World, Vec<Entity>, ComponentId, ChangeTracker) {
        let mut w = World::new();
        let id = w.register_ffi_static_component_writable::<Hp>("Hp");
        let es: Vec<Entity> = (1..=n)
            .map(|i| {
                let e = w.spawn();
                w.insert_static(e, Hp(i));
                e
            })
            .collect();
        w.increment_change_tick();
        let mut t = ChangeTracker::new();
        t.update(&w);
        w.increment_change_tick();
        (w, es, id, t)
    }

    /// The archetype that currently holds rows of `id`.
    fn archetype_with_rows(w: &World, id: ComponentId) -> ArchetypeId {
        w.archetypes_with_static_component(id)
            .find(|&a| {
                w.static_component_raw_span(a, id)
                    .is_some_and(|s| s.count > 0)
            })
            .expect("some archetype holds rows")
    }

    fn changed_hp(w: &World, t: &ChangeTracker) -> Vec<u32> {
        let mut v: Vec<u32> = w.query_changed::<Hp>(t).map(|(_, h)| h.0).collect();
        v.sort_unstable();
        v
    }

    fn added_hp(w: &World, t: &ChangeTracker) -> Vec<u32> {
        let mut v: Vec<u32> = w.query_added::<Hp>(t).map(|(_, h)| h.0).collect();
        v.sort_unstable();
        v
    }

    #[test]
    fn span_mut_points_at_the_live_column_with_the_registered_stride_and_count() {
        let (mut w, _, id, _) = tracked_hp(5);
        let arch = archetype_with_rows(&w, id);
        let shared = w.static_component_raw_span(arch, id).unwrap();
        let span = w.static_component_raw_span_mut(arch, id).unwrap();
        assert_eq!(span.ptr as *const u8, shared.ptr);
        assert_eq!(span.stride, 4);
        assert_eq!(span.count, 5);
        assert!(!span.is_empty());
    }

    #[test]
    fn span_mut_stamps_every_row_of_the_column_changed_but_not_added() {
        let (mut w, _, id, t) = tracked_hp(4);
        let arch = archetype_with_rows(&w, id);
        w.static_component_raw_span_mut(arch, id).unwrap();
        w.increment_change_tick();
        assert_eq!(changed_hp(&w, &t), vec![1, 2, 3, 4]);
        // The values already existed: nothing was added.
        assert!(added_hp(&w, &t).is_empty());
    }

    #[test]
    fn span_mut_stamps_only_the_archetype_and_component_it_was_asked_for() {
        let (mut w, es, id, t) = tracked_hp(4);
        // Move two entities to a second archetype {Hp, Pair}, and give the
        // first archetype's entities a second component of their own.
        w.register_ffi_static_component_writable::<Pair>("Pair");
        w.insert_static(es[0], Pair { a: 1, b: 1 });
        w.insert_static(es[1], Pair { a: 2, b: 2 });
        w.increment_change_tick();
        let mut t = t;
        t.update(&w);
        w.increment_change_tick();

        let pair_id = w.lookup_ffi_static_component_id("Pair").unwrap();
        // The one archetype holding both Hp and Pair (two rows).
        let two = w.archetypes_with_static_component(pair_id).next().unwrap();
        assert_eq!(w.static_component_raw_span(two, id).unwrap().count, 2);
        // Stamp the two-row {Hp, Pair} archetype's *Pair* column only.
        w.static_component_raw_span_mut(two, pair_id).unwrap();
        w.increment_change_tick();
        assert!(changed_hp(&w, &t).is_empty());
        assert_eq!(w.query_changed::<Pair>(&t).count(), 2);
        // Now its Hp column: only the two rows in that archetype.
        w.increment_change_tick();
        t.update(&w);
        w.increment_change_tick();
        w.static_component_raw_span_mut(two, id).unwrap();
        w.increment_change_tick();
        assert_eq!(changed_hp(&w, &t), vec![1, 2]);
    }

    #[test]
    fn a_read_only_registration_is_not_writable_and_stamps_nothing() {
        let mut w = World::new();
        let id = w.register_ffi_static_component::<Ro>("Ro");
        let e = w.spawn();
        w.insert_static(e, Ro(9));
        w.increment_change_tick();
        let arch = archetype_with_rows(&w, id);
        assert_eq!(
            w.static_component_raw_span_mut(arch, id),
            Err(ComponentFfiError::NotWritable)
        );
        assert_eq!(
            w.write_static_component_bytes(e, id, &1u32.to_ne_bytes()),
            Err(ComponentFfiError::NotWritable)
        );
        assert_eq!(w.get_static::<Ro>(e), Some(&Ro(9)));
    }

    #[test]
    fn span_mut_unknown_ids_are_unknown_id() {
        let (mut w, _, id, _) = tracked_hp(2);
        let arch = archetype_with_rows(&w, id);
        // Never registered for FFI at all.
        struct Plain;
        let e = w.spawn();
        w.insert_static(e, Plain);
        let plain_id = ComponentId::from_u32(9999);
        assert_eq!(
            w.static_component_raw_span_mut(arch, plain_id),
            Err(ComponentFfiError::UnknownId)
        );
        // An archetype that does not exist.
        assert_eq!(
            w.static_component_raw_span_mut(ArchetypeId::from_u32(9999), id),
            Err(ComponentFfiError::UnknownId)
        );
        // The empty archetype exists but its signature lacks Hp.
        assert_eq!(
            w.static_component_raw_span_mut(ArchetypeId::from_u32(0), id),
            Err(ComponentFfiError::UnknownId)
        );
    }

    #[test]
    fn span_mut_on_an_emptied_or_columnless_archetype_is_an_empty_span() {
        let (mut w, es, id, _) = tracked_hp(1);
        let arch = archetype_with_rows(&w, id);
        w.despawn(es[0]);
        // The column still exists, with no rows.
        let span = w.static_component_raw_span_mut(arch, id).unwrap();
        assert!(span.is_empty());
        assert!(span.ptr.is_null());

        // A zero-row intermediate: a bundle spawn leaves archetypes behind
        // whose signature includes Hp but which never held a row, so they
        // have no column yet.
        let mut w = World::new();
        let id = w.register_ffi_static_component_writable::<Hp>("Hp");
        w.register_ffi_static_component_writable::<Pair>("Pair");
        w.spawn_bundle((Hp(1), Pair { a: 0, b: 0 }));
        let columnless = w
            .archetypes_with_static_component(id)
            .find(|&a| {
                w.static_component_raw_span(a, id)
                    .is_some_and(|s| s.count == 0)
            })
            .expect("the bundle spawn leaves a zero-row archetype containing Hp");
        let span = w.static_component_raw_span_mut(columnless, id).unwrap();
        assert!(span.is_empty() && span.ptr.is_null());
        // The populated archetype is a different, non-empty answer.
        let populated = archetype_with_rows(&w, id);
        assert!(!w
            .static_component_raw_span_mut(populated, id)
            .unwrap()
            .is_empty());
    }

    #[test]
    fn registering_writable_after_read_only_upgrades_and_the_reverse_does_not_downgrade() {
        let mut w = World::new();
        let ro = w.register_ffi_static_component::<Hp>("Hp");
        let e = w.spawn();
        w.insert_static(e, Hp(1));
        assert_eq!(
            w.write_static_component_bytes(e, ro, &5u32.to_ne_bytes()),
            Err(ComponentFfiError::NotWritable)
        );
        let rw = w.register_ffi_static_component_writable::<Hp>("Hp");
        assert_eq!(ro, rw);
        assert_eq!(
            w.write_static_component_bytes(e, rw, &5u32.to_ne_bytes()),
            Ok(())
        );
        // Registering it read-only again leaves it writable.
        w.register_ffi_static_component::<Hp>("Hp");
        assert_eq!(
            w.write_static_component_bytes(e, rw, &6u32.to_ne_bytes()),
            Ok(())
        );
        assert_eq!(w.get_static::<Hp>(e), Some(&Hp(6)));
    }

    #[test]
    fn write_bytes_changes_only_that_entitys_value_and_stamps_only_its_row() {
        let (mut w, es, id, t) = tracked_hp(4);
        assert_eq!(
            w.write_static_component_bytes(es[2], id, &300u32.to_ne_bytes()),
            Ok(())
        );
        w.increment_change_tick();
        let values: Vec<u32> = es
            .iter()
            .map(|e| w.get_static::<Hp>(*e).unwrap().0)
            .collect();
        assert_eq!(values, vec![1, 2, 300, 4]);
        assert_eq!(changed_hp(&w, &t), vec![300]);
        assert!(added_hp(&w, &t).is_empty());
    }

    #[test]
    fn write_bytes_round_trips_a_multi_field_struct() {
        let mut w = World::new();
        let id = w.register_ffi_static_component_writable::<Pair>("Pair");
        let e = w.spawn();
        w.insert_static(e, Pair { a: 1, b: 2 });
        let new = Pair {
            a: 0xABCD,
            b: 0x1234,
        };
        assert_eq!(
            w.write_static_component_bytes(e, id, new.as_bytes()),
            Ok(())
        );
        assert_eq!(w.get_static::<Pair>(e), Some(&new));
    }

    #[test]
    fn write_bytes_errors_leave_everything_untouched() {
        let (mut w, es, id, t) = tracked_hp(3);

        // Wrong sizes, each side of the real one and empty.
        for bad in [&[][..], &[1u8, 2, 3][..], &[1u8, 2, 3, 4, 5][..]] {
            assert_eq!(
                w.write_static_component_bytes(es[0], id, bad),
                Err(ComponentFfiError::SizeMismatch)
            );
        }
        // Never-registered component id.
        assert_eq!(
            w.write_static_component_bytes(es[0], ComponentId::from_u32(9999), &1u32.to_ne_bytes()),
            Err(ComponentFfiError::UnknownId)
        );
        // Alive but without the component.
        let bare = w.spawn();
        assert_eq!(
            w.write_static_component_bytes(bare, id, &1u32.to_ne_bytes()),
            Err(ComponentFfiError::Absent)
        );
        w.increment_change_tick();
        let values: Vec<u32> = es
            .iter()
            .map(|e| w.get_static::<Hp>(*e).unwrap().0)
            .collect();
        assert_eq!(values, vec![1, 2, 3]);
        assert!(changed_hp(&w, &t).is_empty());
    }

    #[test]
    fn write_bytes_rejects_dead_entities_and_never_reaches_a_reused_slot() {
        let (mut w, es, id, t) = tracked_hp(2);
        let gone = es[0];
        w.despawn(gone);
        assert_eq!(
            w.write_static_component_bytes(gone, id, &9u32.to_ne_bytes()),
            Err(ComponentFfiError::NotAlive)
        );
        // The slot is reused by a new entity holding Hp(50); the stale
        // handle must not write to it.
        let reused = w.spawn();
        w.insert_static(reused, Hp(50));
        assert_ne!(gone, reused);
        assert_eq!(
            w.write_static_component_bytes(gone, id, &9u32.to_ne_bytes()),
            Err(ComponentFfiError::NotAlive)
        );
        assert_eq!(w.get_static::<Hp>(reused), Some(&Hp(50)));
        w.increment_change_tick();
        assert_eq!(changed_hp(&w, &t), vec![50]);
    }

    #[test]
    fn zero_sized_writable_components_work_with_both_paths() {
        let mut w = World::new();
        let id = w.register_ffi_static_component_writable::<Unit>("Unit");
        let a = w.spawn();
        let b = w.spawn();
        w.insert_static(a, Unit);
        w.insert_static(b, Unit);
        w.increment_change_tick();
        let mut t = ChangeTracker::new();
        t.update(&w);
        w.increment_change_tick();

        let arch = archetype_with_rows(&w, id);
        let span = w.static_component_raw_span_mut(arch, id).unwrap();
        assert_eq!((span.stride, span.count), (0, 2));
        assert_eq!(w.write_static_component_bytes(a, id, &[]), Ok(()));
        assert_eq!(
            w.write_static_component_bytes(a, id, &[0]),
            Err(ComponentFfiError::SizeMismatch)
        );
        w.increment_change_tick();
        assert_eq!(w.query_changed::<Unit>(&t).count(), 2);
    }

    #[test]
    fn rows_stay_aligned_when_writing_after_a_swap_remove() {
        let (mut w, es, id, t) = tracked_hp(5);
        w.despawn(es[1]); // the last row moves into row 1
        assert_eq!(
            w.write_static_component_bytes(es[4], id, &500u32.to_ne_bytes()),
            Ok(())
        );
        assert_eq!(
            w.write_static_component_bytes(es[2], id, &300u32.to_ne_bytes()),
            Ok(())
        );
        w.increment_change_tick();
        for (e, want) in [(es[0], 1), (es[2], 300), (es[3], 4), (es[4], 500)] {
            assert_eq!(w.get_static::<Hp>(e), Some(&Hp(want)));
        }
        assert_eq!(changed_hp(&w, &t), vec![300, 500]);
    }

    #[test]
    fn write_errors_are_checked_in_the_documented_order() {
        let mut w = World::new();
        let rw = w.register_ffi_static_component_writable::<Hp>("Hp");
        let ro = w.register_ffi_static_component::<Ro>("Ro");
        let e = w.spawn();
        w.insert_static(e, Hp(1));
        let wrong = [1u8, 2, 3];
        let right = 1u32.to_ne_bytes();

        // NotAlive beats everything, even an unknown id and a wrong size.
        let dead = w.spawn();
        w.despawn(dead);
        assert_eq!(
            w.write_static_component_bytes(dead, ComponentId::from_u32(9999), &wrong),
            Err(ComponentFfiError::NotAlive)
        );
        // UnknownId beats NotWritable/SizeMismatch.
        assert_eq!(
            w.write_static_component_bytes(e, ComponentId::from_u32(9999), &wrong),
            Err(ComponentFfiError::UnknownId)
        );
        // NotWritable beats SizeMismatch (and Absent: `e` has no Ro).
        assert_eq!(
            w.write_static_component_bytes(e, ro, &wrong),
            Err(ComponentFfiError::NotWritable)
        );
        // SizeMismatch beats Absent.
        let bare = w.spawn();
        assert_eq!(
            w.write_static_component_bytes(bare, rw, &wrong),
            Err(ComponentFfiError::SizeMismatch)
        );
        assert_eq!(
            w.write_static_component_bytes(bare, rw, &right),
            Err(ComponentFfiError::Absent)
        );
    }

    // ── Inserting a missing component ────────────────────────────────

    #[test]
    fn insert_bytes_attaches_a_missing_component_stamped_added_and_changed() {
        let mut w = World::new();
        let hp = w.register_ffi_static_component_writable::<Hp>("Hp");
        w.register_ffi_static_component_writable::<Pair>("Pair");
        let e = w.spawn();
        w.insert_static(e, Pair { a: 1, b: 2 });
        w.increment_change_tick();
        let mut t = ChangeTracker::new();
        t.update(&w);
        w.increment_change_tick();

        assert_eq!(w.insert_static_bytes(e, hp, &7u32.to_ne_bytes()), Ok(()));
        w.increment_change_tick();
        assert_eq!(w.get_static::<Hp>(e), Some(&Hp(7)));
        // The value it already had came across the migration intact...
        assert_eq!(w.get_static::<Pair>(e), Some(&Pair { a: 1, b: 2 }));
        // ...the new one is both added and changed...
        assert_eq!(added_hp(&w, &t), vec![7]);
        assert_eq!(changed_hp(&w, &t), vec![7]);
        // ...and the old one was moved, not re-stamped.
        assert!(w.query_added::<Pair>(&t).next().is_none());
        assert!(w.query_changed::<Pair>(&t).next().is_none());
    }

    #[test]
    fn insert_bytes_works_on_an_entity_with_no_components_at_all() {
        let mut w = World::new();
        let hp = w.register_ffi_static_component_writable::<Hp>("Hp");
        let e = w.spawn();
        assert_eq!(w.insert_static_bytes(e, hp, &3u32.to_ne_bytes()), Ok(()));
        assert_eq!(w.get_static::<Hp>(e), Some(&Hp(3)));
        // The component is now a normal tracked one: writable and spannable.
        assert_eq!(
            w.write_static_component_bytes(e, hp, &4u32.to_ne_bytes()),
            Ok(())
        );
        let arch = archetype_with_rows(&w, hp);
        assert_eq!(w.static_component_raw_span_mut(arch, hp).unwrap().count, 1);
        assert_eq!(w.get_static::<Hp>(e), Some(&Hp(4)));
    }

    #[test]
    fn insert_bytes_on_a_present_component_reports_it_and_keeps_the_first_value() {
        let (mut w, es, id, t) = tracked_hp(2);
        assert_eq!(
            w.insert_static_bytes(es[0], id, &99u32.to_ne_bytes()),
            Err(ComponentFfiError::AlreadyPresent)
        );
        w.increment_change_tick();
        assert_eq!(w.get_static::<Hp>(es[0]), Some(&Hp(1)));
        assert!(changed_hp(&w, &t).is_empty());
        assert!(added_hp(&w, &t).is_empty());
    }

    #[test]
    fn insert_errors_are_checked_in_the_documented_order() {
        let mut w = World::new();
        let rw = w.register_ffi_static_component_writable::<Hp>("Hp");
        let ro = w.register_ffi_static_component::<Ro>("Ro");
        let e = w.spawn();
        w.insert_static(e, Hp(1));
        let wrong = [1u8, 2, 3];
        let right = 1u32.to_ne_bytes();
        let nope = ComponentId::from_u32(9999);

        let dead = w.spawn();
        w.despawn(dead);
        // NotAlive beats everything.
        assert_eq!(
            w.insert_static_bytes(dead, nope, &wrong),
            Err(ComponentFfiError::NotAlive)
        );
        // UnknownId beats NotWritable / SizeMismatch / AlreadyPresent.
        assert_eq!(
            w.insert_static_bytes(e, nope, &wrong),
            Err(ComponentFfiError::UnknownId)
        );
        // NotWritable beats SizeMismatch.
        assert_eq!(
            w.insert_static_bytes(e, ro, &wrong),
            Err(ComponentFfiError::NotWritable)
        );
        // SizeMismatch beats AlreadyPresent.
        assert_eq!(
            w.insert_static_bytes(e, rw, &wrong),
            Err(ComponentFfiError::SizeMismatch)
        );
        assert_eq!(
            w.insert_static_bytes(e, rw, &right),
            Err(ComponentFfiError::AlreadyPresent)
        );
        // None of that attached anything.
        assert!(!w.has_static::<Ro>(e));
    }

    #[test]
    fn insert_bytes_rejects_stale_handles_and_never_reaches_a_reused_slot() {
        let mut w = World::new();
        let hp = w.register_ffi_static_component_writable::<Hp>("Hp");
        let gone = w.spawn();
        w.despawn(gone);
        let reused = w.spawn();
        assert_ne!(gone, reused);
        assert_eq!(
            w.insert_static_bytes(gone, hp, &9u32.to_ne_bytes()),
            Err(ComponentFfiError::NotAlive)
        );
        assert!(!w.has_static::<Hp>(reused));
    }

    #[test]
    fn insert_bytes_into_a_middle_row_leaves_every_other_row_aligned() {
        let (mut w, es, _, _) = tracked_hp(5);
        let pair = w.register_ffi_static_component_writable::<Pair>("Pair");
        let new = Pair {
            a: 0xAAAA,
            b: 0xBBBB,
        };
        assert_eq!(w.insert_static_bytes(es[2], pair, new.as_bytes()), Ok(()));
        for (i, e) in es.iter().enumerate() {
            assert_eq!(w.get_static::<Hp>(*e), Some(&Hp(i as u32 + 1)));
            assert_eq!(w.get_static::<Pair>(*e), (i == 2).then_some(&new));
        }
    }

    #[test]
    fn insert_bytes_accepts_a_zero_sized_component() {
        let mut w = World::new();
        let unit = w.register_ffi_static_component_writable::<Unit>("Unit");
        let e = w.spawn();
        assert_eq!(w.insert_static_bytes(e, unit, &[]), Ok(()));
        assert!(w.has_static::<Unit>(e));
        assert_eq!(
            w.insert_static_bytes(e, unit, &[]),
            Err(ComponentFfiError::AlreadyPresent)
        );
    }

    #[test]
    fn write_size_reports_the_registered_size_or_why_not() {
        let mut w = World::new();
        let hp = w.register_ffi_static_component_writable::<Hp>("Hp");
        let pair = w.register_ffi_static_component_writable::<Pair>("Pair");
        let unit = w.register_ffi_static_component_writable::<Unit>("Unit");
        let ro = w.register_ffi_static_component::<Ro>("Ro");
        assert_eq!(w.static_component_write_size(hp), Ok(4));
        assert_eq!(w.static_component_write_size(pair), Ok(4));
        assert_eq!(w.static_component_write_size(unit), Ok(0));
        assert_eq!(
            w.static_component_write_size(ro),
            Err(ComponentFfiError::NotWritable)
        );
        assert_eq!(
            w.static_component_write_size(ComponentId::from_u32(9999)),
            Err(ComponentFfiError::UnknownId)
        );
    }

    // ── Removing a component ─────────────────────────────────────────

    #[test]
    fn remove_by_id_detaches_one_component_and_keeps_every_other_value_and_tick() {
        let mut w = World::new();
        let hp = w.register_ffi_static_component_writable::<Hp>("Hp");
        w.register_ffi_static_component_writable::<Pair>("Pair");
        let e = w.spawn();
        w.insert_static(e, Hp(1));
        w.insert_static(e, Pair { a: 3, b: 4 });
        w.increment_change_tick();
        let mut t = ChangeTracker::new();
        t.update(&w);
        w.increment_change_tick();

        assert_eq!(w.remove_static_by_id(e, hp), Ok(()));
        w.increment_change_tick();
        assert!(!w.has_static::<Hp>(e));
        assert_eq!(w.get_static::<Pair>(e), Some(&Pair { a: 3, b: 4 }));
        assert!(w.is_alive(e));
        // The survivor was moved, not re-stamped.
        assert!(w.query_changed::<Pair>(&t).next().is_none());
        assert!(w.query_added::<Pair>(&t).next().is_none());
    }

    #[test]
    fn removing_the_last_component_leaves_a_live_entity_that_can_be_given_one_again() {
        let mut w = World::new();
        let hp = w.register_ffi_static_component_writable::<Hp>("Hp");
        let e = w.spawn();
        w.insert_static(e, Hp(5));
        assert_eq!(w.remove_static_by_id(e, hp), Ok(()));
        assert!(w.is_alive(e) && !w.has_static::<Hp>(e));
        assert_eq!(w.remove_static_by_id(e, hp), Err(ComponentFfiError::Absent));
        assert_eq!(w.insert_static_bytes(e, hp, &6u32.to_ne_bytes()), Ok(()));
        assert_eq!(w.get_static::<Hp>(e), Some(&Hp(6)));
    }

    #[test]
    fn remove_by_id_errors_are_checked_in_the_documented_order_and_change_nothing() {
        let mut w = World::new();
        let rw = w.register_ffi_static_component_writable::<Hp>("Hp");
        let ro = w.register_ffi_static_component::<Ro>("Ro");
        let e = w.spawn();
        w.insert_static(e, Hp(1));
        let nope = ComponentId::from_u32(9999);

        let dead = w.spawn();
        w.despawn(dead);
        // NotAlive beats everything.
        assert_eq!(
            w.remove_static_by_id(dead, nope),
            Err(ComponentFfiError::NotAlive)
        );
        // UnknownId beats NotWritable / Absent.
        assert_eq!(
            w.remove_static_by_id(e, nope),
            Err(ComponentFfiError::UnknownId)
        );
        // NotWritable beats Absent (`e` has no Ro).
        assert_eq!(
            w.remove_static_by_id(e, ro),
            Err(ComponentFfiError::NotWritable)
        );
        // A read-only component the entity *has* is still not removable.
        w.insert_static(e, Ro(9));
        assert_eq!(
            w.remove_static_by_id(e, ro),
            Err(ComponentFfiError::NotWritable)
        );
        assert_eq!(w.get_static::<Ro>(e), Some(&Ro(9)));
        // Absent last.
        let bare = w.spawn();
        assert_eq!(
            w.remove_static_by_id(bare, rw),
            Err(ComponentFfiError::Absent)
        );
        assert_eq!(w.get_static::<Hp>(e), Some(&Hp(1)));
    }

    #[test]
    fn remove_by_id_rejects_stale_handles_and_never_reaches_a_reused_slot() {
        let mut w = World::new();
        let hp = w.register_ffi_static_component_writable::<Hp>("Hp");
        let gone = w.spawn();
        w.insert_static(gone, Hp(1));
        w.despawn(gone);
        let reused = w.spawn();
        w.insert_static(reused, Hp(50));
        assert_ne!(gone, reused);
        assert_eq!(
            w.remove_static_by_id(gone, hp),
            Err(ComponentFfiError::NotAlive)
        );
        assert_eq!(w.get_static::<Hp>(reused), Some(&Hp(50)));
    }

    #[test]
    fn remove_by_id_from_a_middle_row_keeps_every_other_row_aligned() {
        let (mut w, es, id, _) = tracked_hp(5);
        let pair = w.register_ffi_static_component_writable::<Pair>("Pair");
        for (i, e) in es.iter().enumerate() {
            w.insert_static_bytes(*e, pair, Pair { a: i as u16, b: 0 }.as_bytes())
                .unwrap();
        }
        assert_eq!(w.remove_static_by_id(es[2], id), Ok(()));
        for (i, e) in es.iter().enumerate() {
            assert_eq!(
                w.get_static::<Hp>(*e),
                (i != 2).then_some(&Hp(i as u32 + 1))
            );
            assert_eq!(w.get_static::<Pair>(*e), Some(&Pair { a: i as u16, b: 0 }));
        }
    }

    #[test]
    fn remove_by_id_handles_a_zero_sized_component() {
        let mut w = World::new();
        let unit = w.register_ffi_static_component_writable::<Unit>("Unit");
        let e = w.spawn();
        w.insert_static(e, Unit);
        assert_eq!(w.remove_static_by_id(e, unit), Ok(()));
        assert!(!w.has_static::<Unit>(e));
        assert_eq!(
            w.remove_static_by_id(e, unit),
            Err(ComponentFfiError::Absent)
        );
    }

    #[test]
    fn remove_dropping_drops_the_removed_value_exactly_once_and_nothing_else() {
        use std::cell::Cell;
        use std::rc::Rc;

        struct Guard(Rc<Cell<u32>>);
        impl Drop for Guard {
            fn drop(&mut self) {
                self.0.set(self.0.get() + 1);
            }
        }
        struct Other(Rc<Cell<u32>>);
        impl Drop for Other {
            fn drop(&mut self) {
                self.0.set(self.0.get() + 100);
            }
        }

        let drops = Rc::new(Cell::new(0));
        let mut w = World::new();
        let e = w.spawn();
        w.insert_static(e, Guard(drops.clone()));
        w.insert_static(e, Other(drops.clone()));
        let guard_id = w.archetypes.existing_component_id::<Guard>().unwrap();
        assert_eq!(drops.get(), 0);
        assert!(w.archetypes.remove_dropping(e, guard_id));
        // Guard dropped once (+1); Other moved with the entity, not dropped.
        assert_eq!(drops.get(), 1);
        assert!(w.has_static::<Other>(e));
        // Nothing left to remove.
        assert!(!w.archetypes.remove_dropping(e, guard_id));
        assert_eq!(drops.get(), 1);
        drop(w);
        // Dropping the world drops the survivor.
        assert_eq!(drops.get(), 101);
    }

    /// The type-erased removal must do exactly what the typed
    /// `World::remove_static` does, because it repeats that migration by
    /// hand. Twin worlds, the same pseudo-random sequence of removals,
    /// re-insertions and tick advances applied through the typed path to
    /// one and the id-driven path to the other: every step must agree on
    /// whether it did anything, and afterwards both worlds must hold the
    /// same entities, components, values, iteration order and change
    /// stamps.
    #[test]
    fn remove_by_id_matches_the_typed_remove_step_for_step() {
        fn build() -> (World, Vec<Entity>) {
            let mut w = World::new();
            w.register_ffi_static_component_writable::<Hp>("Hp");
            w.register_ffi_static_component_writable::<Pair>("Pair");
            w.register_ffi_static_component_writable::<Unit>("Unit");
            let mut es = Vec::new();
            for i in 0..24u32 {
                let e = w.spawn();
                if i % 2 == 0 {
                    w.insert_static(e, Hp(i));
                }
                if i % 3 == 0 {
                    w.insert_static(
                        e,
                        Pair {
                            a: i as u16,
                            b: (i * 2) as u16,
                        },
                    );
                }
                if i % 5 == 0 {
                    w.insert_static(e, Unit);
                }
                es.push(e);
            }
            w.increment_change_tick();
            (w, es)
        }
        type Snapshot = Vec<(Entity, Option<Hp>, Option<Pair>, bool)>;
        fn snapshot(w: &World) -> Snapshot {
            w.query_data::<(
                Entity,
                Option<&Hp>,
                Option<&Pair>,
                crate::query_data::Has<Unit>,
            )>()
            .map(|(e, hp, pair, unit)| (e, hp.copied(), pair.copied(), unit))
            .collect()
        }

        let (mut typed, es) = build();
        let (mut erased, es_b) = build();
        assert_eq!(es, es_b);
        let mut t_typed = ChangeTracker::new();
        t_typed.update(&typed);
        let mut t_erased = ChangeTracker::new();
        t_erased.update(&erased);
        typed.increment_change_tick();
        erased.increment_change_tick();
        let ids = [
            erased.lookup_ffi_static_component_id("Hp").unwrap(),
            erased.lookup_ffi_static_component_id("Pair").unwrap(),
            erased.lookup_ffi_static_component_id("Unit").unwrap(),
        ];
        assert_eq!(snapshot(&typed), snapshot(&erased));

        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let (mut removed, mut reinserted) = (0, 0);
        for _ in 0..400 {
            let e = es[(next() % es.len() as u64) as usize];
            let which = (next() % 3) as usize;
            match next() % 5 {
                // Remove, through each path.
                0..=2 => {
                    let did_typed = match which {
                        0 => typed.remove_static::<Hp>(e).is_some(),
                        1 => typed.remove_static::<Pair>(e).is_some(),
                        _ => typed.remove_static::<Unit>(e).is_some(),
                    };
                    let did_erased = erased.remove_static_by_id(e, ids[which]).is_ok();
                    assert_eq!(did_typed, did_erased);
                    removed += usize::from(did_typed);
                }
                // Put one back, the same typed way in both worlds.
                3 => {
                    for w in [&mut typed, &mut erased] {
                        match which {
                            0 => w.insert_static(e, Hp(7)),
                            1 => w.insert_static(e, Pair { a: 7, b: 7 }),
                            _ => w.insert_static(e, Unit),
                        };
                    }
                    reinserted += 1;
                }
                // Advance the tick, so stamps differ across steps.
                _ => {
                    typed.increment_change_tick();
                    erased.increment_change_tick();
                }
            }
            assert_eq!(snapshot(&typed), snapshot(&erased));
        }
        // The run really exercised both paths.
        assert!(removed > 40 && reinserted > 20, "{removed} {reinserted}");

        let stamps = |w: &World, t: &ChangeTracker| {
            (
                w.query_changed::<Hp>(t)
                    .map(|(e, h)| (e, *h))
                    .collect::<Vec<_>>(),
                w.query_added::<Hp>(t)
                    .map(|(e, h)| (e, *h))
                    .collect::<Vec<_>>(),
                w.query_changed::<Pair>(t)
                    .map(|(e, p)| (e, *p))
                    .collect::<Vec<_>>(),
                w.query_added::<Pair>(t)
                    .map(|(e, p)| (e, *p))
                    .collect::<Vec<_>>(),
            )
        };
        assert_eq!(stamps(&typed, &t_typed), stamps(&erased, &t_erased));
    }

    // ── Bundles: several components in one structural change ─────────

    /// `(id, bytes)` for the subset of `[Hp, Pair, Unit]` a bitmask names
    /// (bit 0 Hp, bit 1 Pair, bit 2 Unit), in that order, with values
    /// derived from `v`.
    fn bundle_entries(ids: [ComponentId; 3], mask: u8, v: u32) -> Vec<(ComponentId, Vec<u8>)> {
        let mut out = Vec::new();
        if mask & 1 != 0 {
            out.push((ids[0], Hp(v).as_bytes().to_vec()));
        }
        if mask & 2 != 0 {
            out.push((
                ids[1],
                Pair {
                    a: v as u16,
                    b: (v * 3) as u16,
                }
                .as_bytes()
                .to_vec(),
            ));
        }
        if mask & 4 != 0 {
            out.push((ids[2], Vec::new()));
        }
        out
    }

    fn borrowed(entries: &[(ComponentId, Vec<u8>)]) -> Vec<(ComponentId, &[u8])> {
        entries.iter().map(|(id, b)| (*id, b.as_slice())).collect()
    }

    fn three_writable() -> (World, [ComponentId; 3]) {
        let mut w = World::new();
        let ids = [
            w.register_ffi_static_component_writable::<Hp>("Hp"),
            w.register_ffi_static_component_writable::<Pair>("Pair"),
            w.register_ffi_static_component_writable::<Unit>("Unit"),
        ];
        (w, ids)
    }

    #[test]
    fn insert_many_attaches_every_component_stamped_added_and_changed() {
        let (mut w, ids) = three_writable();
        let e = w.spawn();
        w.insert_static(e, Ro(1));
        w.increment_change_tick();
        let mut t = ChangeTracker::new();
        t.update(&w);
        w.increment_change_tick();

        let entries = bundle_entries(ids, 0b111, 7);
        assert_eq!(w.insert_static_bytes_many(e, &borrowed(&entries)), Ok(()));
        w.increment_change_tick();
        assert_eq!(w.get_static::<Hp>(e), Some(&Hp(7)));
        assert_eq!(w.get_static::<Pair>(e), Some(&Pair { a: 7, b: 21 }));
        assert!(w.has_static::<Unit>(e));
        // The value it already had came across intact and was not re-stamped.
        assert_eq!(w.get_static::<Ro>(e), Some(&Ro(1)));
        assert!(w.query_changed::<Ro>(&t).next().is_none());
        // Every new one is both added and changed.
        assert_eq!(added_hp(&w, &t), vec![7]);
        assert_eq!(changed_hp(&w, &t), vec![7]);
        assert_eq!(w.query_added::<Pair>(&t).count(), 1);
        assert_eq!(w.query_added::<Unit>(&t).count(), 1);
    }

    #[test]
    fn insert_many_is_all_or_nothing() {
        let (mut w, ids) = three_writable();
        let ro = w.register_ffi_static_component::<Ro>("Ro");
        let e = w.spawn();
        w.insert_static(e, Pair { a: 1, b: 1 });
        let hp = Hp(5);
        let ok = hp.as_bytes();
        let none_attached = |w: &World| !w.has_static::<Hp>(e) && !w.has_static::<Unit>(e);

        // One of them already present.
        let entries = [(ids[0], ok), (ids[1], &[0u8; 4][..]), (ids[2], &[][..])];
        assert_eq!(
            w.insert_static_bytes_many(e, &entries),
            Err(ComponentFfiError::AlreadyPresent)
        );
        assert!(none_attached(&w));
        // One of them the wrong size.
        let entries = [(ids[0], ok), (ids[2], &[1u8][..])];
        assert_eq!(
            w.insert_static_bytes_many(e, &entries),
            Err(ComponentFfiError::SizeMismatch)
        );
        assert!(none_attached(&w));
        // One of them unknown, one read-only.
        let entries = [(ids[0], ok), (ComponentId::from_u32(9999), ok)];
        assert_eq!(
            w.insert_static_bytes_many(e, &entries),
            Err(ComponentFfiError::UnknownId)
        );
        let entries = [(ids[0], ok), (ro, ok)];
        assert_eq!(
            w.insert_static_bytes_many(e, &entries),
            Err(ComponentFfiError::NotWritable)
        );
        // A repeat.
        let entries = [(ids[0], ok), (ids[2], &[][..]), (ids[0], ok)];
        assert_eq!(
            w.insert_static_bytes_many(e, &entries),
            Err(ComponentFfiError::Duplicate)
        );
        assert!(none_attached(&w));
        assert_eq!(w.get_static::<Pair>(e), Some(&Pair { a: 1, b: 1 }));
    }

    #[test]
    fn insert_many_errors_are_checked_in_the_documented_order() {
        let (mut w, ids) = three_writable();
        let ro = w.register_ffi_static_component::<Ro>("Ro");
        let e = w.spawn();
        w.insert_static(e, Hp(1));
        let dead = w.spawn();
        w.despawn(dead);
        let four = [0u8; 4];
        let nope = ComponentId::from_u32(9999);

        // NotAlive beats everything.
        assert_eq!(
            w.insert_static_bytes_many(dead, &[(nope, &four[..])]),
            Err(ComponentFfiError::NotAlive)
        );
        // Entries are checked in list order: the first bad one wins.
        assert_eq!(
            w.insert_static_bytes_many(e, &[(nope, &four[..]), (ro, &four[..])]),
            Err(ComponentFfiError::UnknownId)
        );
        assert_eq!(
            w.insert_static_bytes_many(e, &[(ro, &four[..]), (nope, &four[..])]),
            Err(ComponentFfiError::NotWritable)
        );
        // Per-entry errors beat Duplicate, Duplicate beats AlreadyPresent.
        assert_eq!(
            w.insert_static_bytes_many(e, &[(ids[1], &four[..]), (ids[1], &[1u8][..])]),
            Err(ComponentFfiError::SizeMismatch)
        );
        assert_eq!(
            w.insert_static_bytes_many(e, &[(ids[0], &four[..]), (ids[0], &four[..])]),
            Err(ComponentFfiError::Duplicate)
        );
        assert_eq!(
            w.insert_static_bytes_many(e, &[(ids[0], &four[..])]),
            Err(ComponentFfiError::AlreadyPresent)
        );
    }

    #[test]
    fn an_empty_list_is_ok_and_changes_nothing_for_both_directions() {
        let (mut w, _) = three_writable();
        let e = w.spawn();
        w.insert_static(e, Hp(1));
        assert_eq!(w.insert_static_bytes_many(e, &[]), Ok(()));
        assert_eq!(w.remove_static_by_ids(e, &[]), Ok(()));
        assert_eq!(w.get_static::<Hp>(e), Some(&Hp(1)));
        // Still checked for liveness first.
        let dead = w.spawn();
        w.despawn(dead);
        assert_eq!(
            w.insert_static_bytes_many(dead, &[]),
            Err(ComponentFfiError::NotAlive)
        );
        assert_eq!(
            w.remove_static_by_ids(dead, &[]),
            Err(ComponentFfiError::NotAlive)
        );
    }

    #[test]
    fn insert_many_into_a_middle_row_leaves_every_other_row_aligned() {
        let (mut w, ids) = three_writable();
        let es: Vec<Entity> = (1..=5)
            .map(|i| {
                let e = w.spawn();
                w.insert_static(e, Ro(i));
                e
            })
            .collect();
        let entries = bundle_entries(ids, 0b011, 9);
        assert_eq!(
            w.insert_static_bytes_many(es[2], &borrowed(&entries)),
            Ok(())
        );
        for (i, e) in es.iter().enumerate() {
            assert_eq!(w.get_static::<Ro>(*e), Some(&Ro(i as u32 + 1)));
            assert_eq!(w.get_static::<Hp>(*e), (i == 2).then_some(&Hp(9)));
        }
    }

    #[test]
    fn insert_many_rejects_stale_handles_and_never_reaches_a_reused_slot() {
        let (mut w, ids) = three_writable();
        let gone = w.spawn();
        w.despawn(gone);
        let reused = w.spawn();
        let entries = bundle_entries(ids, 0b011, 1);
        assert_eq!(
            w.insert_static_bytes_many(gone, &borrowed(&entries)),
            Err(ComponentFfiError::NotAlive)
        );
        assert!(!w.has_static::<Hp>(reused) && !w.has_static::<Pair>(reused));
    }

    #[test]
    fn remove_many_detaches_every_listed_component_and_keeps_the_rest() {
        let (mut w, ids) = three_writable();
        let e = w.spawn();
        w.insert_static(e, Ro(4));
        let entries = bundle_entries(ids, 0b111, 2);
        w.insert_static_bytes_many(e, &borrowed(&entries)).unwrap();
        w.increment_change_tick();
        let mut t = ChangeTracker::new();
        t.update(&w);
        w.increment_change_tick();

        assert_eq!(w.remove_static_by_ids(e, &[ids[0], ids[2]]), Ok(()));
        w.increment_change_tick();
        assert!(!w.has_static::<Hp>(e) && !w.has_static::<Unit>(e));
        assert_eq!(w.get_static::<Pair>(e), Some(&Pair { a: 2, b: 6 }));
        assert_eq!(w.get_static::<Ro>(e), Some(&Ro(4)));
        // Survivors were moved, not re-stamped.
        assert_eq!(w.query_changed::<Pair>(&t).count(), 0);
        assert_eq!(w.query_changed::<Ro>(&t).count(), 0);
    }

    #[test]
    fn remove_many_is_all_or_nothing() {
        let (mut w, ids) = three_writable();
        let ro = w.register_ffi_static_component::<Ro>("Ro");
        let e = w.spawn();
        w.insert_static(e, Hp(1));
        w.insert_static(e, Ro(2));
        // Lacks one of them.
        assert_eq!(
            w.remove_static_by_ids(e, &[ids[0], ids[1]]),
            Err(ComponentFfiError::Absent)
        );
        // Read-only in the list, even though the entity has it.
        assert_eq!(
            w.remove_static_by_ids(e, &[ids[0], ro]),
            Err(ComponentFfiError::NotWritable)
        );
        // Unknown, and a repeat.
        assert_eq!(
            w.remove_static_by_ids(e, &[ids[0], ComponentId::from_u32(9999)]),
            Err(ComponentFfiError::UnknownId)
        );
        assert_eq!(
            w.remove_static_by_ids(e, &[ids[0], ids[0]]),
            Err(ComponentFfiError::Duplicate)
        );
        assert_eq!(w.get_static::<Hp>(e), Some(&Hp(1)));
        assert_eq!(w.get_static::<Ro>(e), Some(&Ro(2)));
    }

    #[test]
    fn remove_many_errors_in_order_rejects_stale_handles_and_keeps_rows_aligned() {
        let (mut w, ids) = three_writable();
        let es: Vec<Entity> = (1..=5)
            .map(|i| {
                let e = w.spawn();
                let entries = bundle_entries(ids, 0b011, i);
                w.insert_static_bytes_many(e, &borrowed(&entries)).unwrap();
                e
            })
            .collect();
        let dead = w.spawn();
        w.despawn(dead);
        // NotAlive beats everything; per-id errors beat Duplicate beat Absent.
        assert_eq!(
            w.remove_static_by_ids(dead, &[ComponentId::from_u32(9999)]),
            Err(ComponentFfiError::NotAlive)
        );
        assert_eq!(
            w.remove_static_by_ids(es[0], &[ids[2], ids[2]]),
            Err(ComponentFfiError::Duplicate)
        );
        assert_eq!(
            w.remove_static_by_ids(es[0], &[ids[2]]),
            Err(ComponentFfiError::Absent)
        );
        // A stale handle never reaches the slot's new occupant.
        let reused = w.spawn();
        let entries = bundle_entries(ids, 0b011, 99);
        w.insert_static_bytes_many(reused, &borrowed(&entries))
            .unwrap();
        assert_eq!(
            w.remove_static_by_ids(dead, &[ids[0], ids[1]]),
            Err(ComponentFfiError::NotAlive)
        );
        assert!(w.has_static::<Hp>(reused));
        // Remove from a middle row: every other row stays aligned.
        assert_eq!(w.remove_static_by_ids(es[2], &[ids[0], ids[1]]), Ok(()));
        for (i, e) in es.iter().enumerate() {
            assert_eq!(
                w.get_static::<Hp>(*e),
                (i != 2).then_some(&Hp(i as u32 + 1))
            );
        }
    }

    #[test]
    fn check_static_component_list_runs_the_per_entry_checks_then_duplicates() {
        let (w, ids) = {
            let (mut w, ids) = three_writable();
            w.register_ffi_static_component::<Ro>("Ro");
            (w, ids)
        };
        let ro = w.lookup_ffi_static_component_id("Ro").unwrap();
        let nope = ComponentId::from_u32(9999);
        assert_eq!(
            w.check_static_component_list(&[(ids[0], Some(4)), (ids[2], Some(0)), (ids[1], None)]),
            Ok(())
        );
        assert_eq!(w.check_static_component_list(&[]), Ok(()));
        assert_eq!(
            w.check_static_component_list(&[(ids[0], Some(3))]),
            Err(ComponentFfiError::SizeMismatch)
        );
        // `None` skips the size check (removals have no bytes).
        assert_eq!(w.check_static_component_list(&[(ids[0], None)]), Ok(()));
        assert_eq!(
            w.check_static_component_list(&[(ro, None)]),
            Err(ComponentFfiError::NotWritable)
        );
        assert_eq!(
            w.check_static_component_list(&[(nope, None)]),
            Err(ComponentFfiError::UnknownId)
        );
        // A per-entry error anywhere beats a repeat earlier in the list.
        assert_eq!(
            w.check_static_component_list(&[(ids[0], None), (ids[0], None), (nope, None)]),
            Err(ComponentFfiError::UnknownId)
        );
        assert_eq!(
            w.check_static_component_list(&[(ids[0], None), (ids[1], None), (ids[0], None)]),
            Err(ComponentFfiError::Duplicate)
        );
    }

    /// The list forms repeat the typed bundle migrations by hand, so they
    /// are held to them the way the single removal is: twin worlds, the
    /// same pseudo-random sequence of bundle inserts and removals over
    /// every non-empty subset of three component types (one zero-sized),
    /// applied through the typed `insert_bundle` / `remove_bundle` to one
    /// world and the id-driven list forms to the other. Every step must
    /// agree on whether it did anything (the all-or-nothing rule included),
    /// and afterwards both worlds must hold the same entities, components,
    /// values, iteration order and change stamps.
    #[test]
    fn bundle_ops_match_the_typed_bundles_step_for_step() {
        fn build() -> (World, [ComponentId; 3], Vec<Entity>) {
            let (mut w, ids) = three_writable();
            let mut es = Vec::new();
            for i in 0..20u32 {
                let e = w.spawn();
                if i % 2 == 0 {
                    w.insert_static(e, Hp(i));
                }
                if i % 3 == 0 {
                    w.insert_static(e, Pair { a: i as u16, b: 0 });
                }
                if i % 5 == 0 {
                    w.insert_static(e, Unit);
                }
                es.push(e);
            }
            w.increment_change_tick();
            (w, ids, es)
        }
        type Snapshot = Vec<(Entity, Option<Hp>, Option<Pair>, bool)>;
        fn snapshot(w: &World) -> Snapshot {
            w.query_data::<(
                Entity,
                Option<&Hp>,
                Option<&Pair>,
                crate::query_data::Has<Unit>,
            )>()
            .map(|(e, hp, pair, unit)| (e, hp.copied(), pair.copied(), unit))
            .collect()
        }
        fn typed_insert(w: &mut World, e: Entity, mask: u8, v: u32) -> bool {
            let hp = Hp(v);
            let pair = Pair {
                a: v as u16,
                b: (v * 3) as u16,
            };
            match mask {
                1 => w.insert_bundle(e, (hp,)),
                2 => w.insert_bundle(e, (pair,)),
                3 => w.insert_bundle(e, (hp, pair)),
                4 => w.insert_bundle(e, (Unit,)),
                5 => w.insert_bundle(e, (hp, Unit)),
                6 => w.insert_bundle(e, (pair, Unit)),
                _ => w.insert_bundle(e, (hp, pair, Unit)),
            }
        }
        fn typed_remove(w: &mut World, e: Entity, mask: u8) -> bool {
            match mask {
                1 => w.remove_bundle::<(Hp,)>(e).is_some(),
                2 => w.remove_bundle::<(Pair,)>(e).is_some(),
                3 => w.remove_bundle::<(Hp, Pair)>(e).is_some(),
                4 => w.remove_bundle::<(Unit,)>(e).is_some(),
                5 => w.remove_bundle::<(Hp, Unit)>(e).is_some(),
                6 => w.remove_bundle::<(Pair, Unit)>(e).is_some(),
                _ => w.remove_bundle::<(Hp, Pair, Unit)>(e).is_some(),
            }
        }

        let (mut typed, _, es) = build();
        let (mut erased, ids, es_b) = build();
        assert_eq!(es, es_b);
        let mut t_typed = ChangeTracker::new();
        t_typed.update(&typed);
        let mut t_erased = ChangeTracker::new();
        t_erased.update(&erased);
        typed.increment_change_tick();
        erased.increment_change_tick();
        assert_eq!(snapshot(&typed), snapshot(&erased));

        let mut state = 0xD1B5_4A32_D192_ED03u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let (mut inserted, mut removed, mut refused) = (0, 0, 0);
        for step in 0..600u32 {
            let e = es[(next() % es.len() as u64) as usize];
            let mask = (next() % 7 + 1) as u8;
            match next() % 5 {
                0..=1 => {
                    let did_typed = typed_insert(&mut typed, e, mask, step);
                    let entries = bundle_entries(ids, mask, step);
                    let did_erased = erased
                        .insert_static_bytes_many(e, &borrowed(&entries))
                        .is_ok();
                    assert_eq!(did_typed, did_erased, "insert step {step} mask {mask}");
                    if did_typed {
                        inserted += 1;
                    } else {
                        refused += 1;
                    }
                }
                2..=3 => {
                    let did_typed = typed_remove(&mut typed, e, mask);
                    let list: Vec<ComponentId> = bundle_entries(ids, mask, 0)
                        .iter()
                        .map(|(id, _)| *id)
                        .collect();
                    let did_erased = erased.remove_static_by_ids(e, &list).is_ok();
                    assert_eq!(did_typed, did_erased, "remove step {step} mask {mask}");
                    if did_typed {
                        removed += 1;
                    } else {
                        refused += 1;
                    }
                }
                _ => {
                    typed.increment_change_tick();
                    erased.increment_change_tick();
                }
            }
            assert_eq!(snapshot(&typed), snapshot(&erased), "after step {step}");
        }
        // The run exercised both directions and the all-or-nothing refusals.
        assert!(
            inserted > 30 && removed > 30 && refused > 30,
            "{inserted} {removed} {refused}"
        );

        let stamps = |w: &World, t: &ChangeTracker| {
            (
                w.query_changed::<Hp>(t)
                    .map(|(e, h)| (e, *h))
                    .collect::<Vec<_>>(),
                w.query_added::<Hp>(t)
                    .map(|(e, h)| (e, *h))
                    .collect::<Vec<_>>(),
                w.query_changed::<Pair>(t)
                    .map(|(e, p)| (e, *p))
                    .collect::<Vec<_>>(),
                w.query_added::<Pair>(t)
                    .map(|(e, p)| (e, *p))
                    .collect::<Vec<_>>(),
                w.query_changed::<Unit>(t)
                    .map(|(e, _)| e)
                    .collect::<Vec<_>>(),
                w.query_added::<Unit>(t).map(|(e, _)| e).collect::<Vec<_>>(),
            )
        };
        assert_eq!(stamps(&typed, &t_typed), stamps(&erased, &t_erased));
    }
}
