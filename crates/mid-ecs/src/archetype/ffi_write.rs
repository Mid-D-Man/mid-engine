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

use super::{Archetype, ArchetypeId, Archetypes, Table};
use crate::component::ComponentId;
use crate::tick::Tick;
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

/// Why an id-based (FFI-facing) component write did not happen. Checked
/// in this order for [`crate::World::write_static_component_bytes`]:
/// `NotAlive`, `UnknownId`, `NotWritable`, `SizeMismatch`, `Absent`.
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
    /// The entity is alive but does not currently have this component.
    Absent,
    /// A write's byte length is not exactly the registered type's size.
    /// Nothing was written or stamped.
    SizeMismatch,
}

/// Type-erased write operations for one writable component type, built
/// from `T` by [`Archetypes::register_ffi_writable`].
#[derive(Clone, Copy)]
pub(super) struct FfiWriter {
    size: usize,
    span_mut: fn(&mut dyn Any) -> FfiSpanMut,
    write_row: fn(&mut dyn Any, usize, &[u8]) -> bool,
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

impl Archetypes {
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
            .and_then(|row_ticks| row_ticks.get_mut(location.row))
        {
            stamp.changed = this_run;
        }
        Ok(())
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
}
