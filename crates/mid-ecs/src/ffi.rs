//! C-compatible FFI exports for mid-ecs.
//!
//! Scope for this pass: `World` lifecycle only — `new`/`free`/`spawn`/
//! `despawn`/`is_alive`/`entity_count`. Genuinely useful right now on its
//! own: a non-Rust caller can already create a world and manage entity
//! lifetimes, with real generational safety, before any component data
//! can cross the boundary at all. Component data access (reading a
//! `Position` column from C, for instance) needs the FFI-span design
//! this crate's `docs/mid-ecs.md` calls out as the next real FFI piece —
//! deliberately not attempted here in the same pass, since it has a
//! genuinely harder safety story (a raw pointer into a `Vec<T>` that a
//! later `insert`/`remove`/migration call can reallocate or move out
//! from under, unlike anything in this file, where every value crosses
//! by-value or through an opaque handle with no live pointers into
//! mutable interior storage). Built incrementally, tested as it's
//! built — the same conclusion `mid-net-transport-quinn`/`-wasm` reached
//! for network code applies here too: get one real, working, tested
//! slice of FFI surface out before reaching for the harder piece.
//!
//! ## Conventions — copied directly from `mid-net`'s real `ffi.rs`, not
//! reinvented
//! - Every function checks its pointer arguments for null before
//!   dereferencing and returns a defined error code (or a safe default:
//!   `false` for `is_alive`, `0` for `entity_count`) instead of
//!   dereferencing a null pointer.
//! - Every function's body runs inside [`std::panic::catch_unwind`] via
//!   `ffi_guard` — unwinding across an `extern "C"` boundary is
//!   undefined behavior, so a panic here becomes
//!   `MidEcsStatus::InternalPanic` instead.
//! - Every function taking a raw pointer is `unsafe fn` with a `# Safety`
//!   doc comment, matching `clippy::not_unsafe_ptr_arg_deref`'s
//!   requirement (deny-by-default on real CI, same MSRV-gap pattern
//!   already documented in `mid-net`'s own `ffi.rs`).
//! - `World` is not `repr(C)` (it owns `SparseSet`s, `HashMap`s, `Vec`s —
//!   nothing about it is C-representable), so it crosses the boundary as
//!   an opaque heap-allocated handle (`Box::into_raw`/`Box::from_raw`),
//!   matching `mid-net`'s own `MidNetPlayerEvent` handle pattern exactly.
//!
//! ## The one genuinely new piece: packing `Entity` as a plain `u64`
//! `Entity` itself can't cross the boundary as a Rust value (its inner
//! `GenerationalIndex` fields are private, deliberately — an `Entity`
//! should only ever come from a real `World::spawn`), and a two-field
//! `repr(C)` struct would make every FFI caller's language agree on a
//! struct layout for no real benefit. `Entity::as_ffi`/`from_ffi` (thin
//! wrappers over `mid_collections::GenerationalIndex::as_ffi`/
//! `from_ffi`, which do the real packing) give one plain `u64` instead —
//! directly grounded in `slotmap::KeyData::as_ffi`/`from_ffi`'s real,
//! shipped design (checked directly, not assumed), including its
//! critical safety property: a `u64` that never actually came from a
//! real `as_ffi()` call is still safe to pass to `from_ffi` and every
//! `World` method — it can only ever produce *some* valid-shaped
//! `Entity`, and every real operation re-validates its generation
//! against the slot's current one regardless of where the value came
//! from. A bogus handle just reads back as not alive; it can never
//! alias a real, live entity it wasn't actually issued for.

use std::ffi::CStr;
use std::os::raw::c_char;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr;
use std::slice;

use crate::archetype::ArchetypeId;
use crate::component::ComponentId;
use crate::world::{Entity, World};
use mid_collections::FfiSpan;

/// Status codes. `is_alive` returns a plain `bool` (a pure query, not an
/// operation with distinct failure reasons) and `entity_count` returns
/// `usize` directly — everything else that can fail returns one of
/// these (`MidEcsStatus::Ok` is always `0`), matching `mid-net`'s own
/// `MidNetStatus` convention.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MidEcsStatus {
    Ok = 0,
    NullPointer = -1,
    /// The entity wasn't alive — either never spawned through this
    /// `World`, already despawned, or a stale handle whose slot has
    /// since been reused by a different entity.
    NotAlive = -2,
    /// Something inside this crate panicked. Should never happen for
    /// well-formed input per each function's documented contract —
    /// exists so a caller gets a defined code instead of UB from an
    /// unwind crossing the FFI boundary.
    InternalPanic = -3,
    /// A `component_id`/`archetype_id` (or the `(archetype_id,
    /// component_id)` pair) doesn't resolve to real, currently-valid
    /// data — never registered via the matching Rust-side
    /// `register_ffi_*` call, or (for the Archetype Core pair) a
    /// structurally real archetype whose signature just doesn't
    /// include that component. Mirrors `World::component_raw_span`'s
    /// own `None` case at the Rust level — see that method's doc
    /// comment for the full reasoning.
    NotFound = -4,
    /// A non-null output buffer was too small for the real element
    /// count. Matches `mid-net`'s own `MidNetStatus::BufferTooSmall`
    /// convention exactly: pass a NULL buffer first to query the real
    /// count via this function's own return value, then call again
    /// with a buffer sized to hold at least that many elements.
    BufferTooSmall = -5,
}

fn ffi_guard(f: impl FnOnce() -> i32) -> i32 {
    catch_unwind(AssertUnwindSafe(f)).unwrap_or(MidEcsStatus::InternalPanic as i32)
}

/// Opaque handle to a `World`. Always heap-allocated by this crate;
/// every handle returned by `mid_ecs_world_new` must be freed with
/// `mid_ecs_world_free` exactly once.
pub struct MidEcsWorld(World);

/// Creates a new, empty `World`. Never returns NULL — allocation failure
/// aborts the process the same way any other Rust `Box` allocation
/// failure would, not a documented error path here.
#[no_mangle]
pub extern "C" fn mid_ecs_world_new() -> *mut MidEcsWorld {
    Box::into_raw(Box::new(MidEcsWorld(World::new())))
}

/// Frees a handle returned by `mid_ecs_world_new`. NULL is a safe
/// no-op.
///
/// # Safety
/// `world` must either be NULL or a handle previously returned by
/// `mid_ecs_world_new` that hasn't been freed yet.
#[no_mangle]
pub unsafe extern "C" fn mid_ecs_world_free(world: *mut MidEcsWorld) {
    if world.is_null() {
        return;
    }
    drop(unsafe { Box::from_raw(world) });
}

/// Spawns a new, live entity, packed as a `u64` — see this module's doc
/// comment for the packing scheme. Returns `0` (index `0`, generation
/// `0`) on a null `world` or an internal panic — **not** a valid "empty"
/// sentinel to check against, since index/generation `0` could
/// theoretically also be a real packed value in a future allocator
/// state; check `world` for null yourself before calling if that
/// distinction matters to the caller.
///
/// # Safety
/// `world` must be a valid, non-null handle from `mid_ecs_world_new`.
#[no_mangle]
pub unsafe extern "C" fn mid_ecs_world_spawn(world: *mut MidEcsWorld) -> u64 {
    if world.is_null() {
        return 0;
    }
    let result = catch_unwind(AssertUnwindSafe(|| {
        let world = unsafe { &mut *world };
        world.0.spawn().as_ffi()
    }));
    result.unwrap_or(0)
}

/// Despawns the entity packed in `entity`. Returns `MidEcsStatus::Ok` if
/// it was actually alive, `MidEcsStatus::NotAlive` otherwise (never
/// spawned, already despawned, or a stale handle) — both are safe,
/// defined outcomes, never a panic.
///
/// # Safety
/// `world` must be a valid, non-null handle from `mid_ecs_world_new`.
#[no_mangle]
pub unsafe extern "C" fn mid_ecs_world_despawn(world: *mut MidEcsWorld, entity: u64) -> i32 {
    ffi_guard(|| {
        if world.is_null() {
            return MidEcsStatus::NullPointer as i32;
        }
        let world = unsafe { &mut *world };
        if world.0.despawn(Entity::from_ffi(entity)) {
            MidEcsStatus::Ok as i32
        } else {
            MidEcsStatus::NotAlive as i32
        }
    })
}

/// Whether the entity packed in `entity` is currently alive. `false` on
/// a null `world` — matching `mid-net`'s own "null handle getters return
/// safe defaults, not crash" convention.
///
/// # Safety
/// `world` must either be NULL or a valid handle from
/// `mid_ecs_world_new`.
#[no_mangle]
pub unsafe extern "C" fn mid_ecs_world_is_alive(world: *const MidEcsWorld, entity: u64) -> bool {
    if world.is_null() {
        return false;
    }
    let result = catch_unwind(AssertUnwindSafe(|| {
        let world = unsafe { &*world };
        world.0.is_alive(Entity::from_ffi(entity))
    }));
    result.unwrap_or(false)
}

/// Number of currently-live entities. `0` on a null `world`.
///
/// # Safety
/// `world` must either be NULL or a valid handle from
/// `mid_ecs_world_new`.
#[no_mangle]
pub unsafe extern "C" fn mid_ecs_world_entity_count(world: *const MidEcsWorld) -> usize {
    if world.is_null() {
        return 0;
    }
    let result = catch_unwind(AssertUnwindSafe(|| {
        let world = unsafe { &*world };
        world.0.entity_count()
    }));
    result.unwrap_or(0)
}

/// A fixed `#[repr(C)]` component type used only by
/// [`mid_ecs_test_fixture_world_new`] below — real code that's part of
/// the compiled library like everything else in this file (needed for
/// `ffi-smoke-test/test.c` to link against it), not `#[cfg(test)]`
/// Rust-only scaffolding. Kept distinct from this module's own
/// `#[cfg(test)]` `FfiHealth` (used by this file's *Rust*-level tests)
/// deliberately — that one only exists in `cargo test` builds, and
/// can't be what a separately-compiled C program links against.
///
/// Sparse Shell's own fixture type — see [`MidEcsTestHealthStatic`]
/// for why this isn't the *same* type shared across both storage
/// systems, the way an earlier version of this fixture actually did.
#[derive(zerocopy::IntoBytes, zerocopy::Immutable, zerocopy::KnownLayout)]
#[repr(C)]
pub struct MidEcsTestHealth {
    pub hp: u32,
}

/// The Archetype Core's own fixture type — deliberately a *distinct*
/// Rust type from [`MidEcsTestHealth`], even though both are
/// `{ hp: u32 }` and identical from C's side of the FFI boundary.
/// An earlier version of this fixture used the same Rust type for
/// both storage systems; `World`'s own `StorageClaims` guard (see
/// `world.rs`) correctly panicked on that — using one component type
/// with both storage systems is a real footgun it exists specifically
/// to catch, and the test fixture itself wasn't exempt from that just
/// because it's test-only code.
#[derive(zerocopy::IntoBytes, zerocopy::Immutable, zerocopy::KnownLayout)]
#[repr(C)]
pub struct MidEcsTestHealthStatic {
    pub hp: u32,
}

/// **Test-fixture only — not a real part of this library's intended
/// public API.** Exists for one specific, narrow reason: `register_ffi_component`/
/// `register_ffi_static_component`/`insert`/`insert_static` are all
/// generic over the Rust component type, so none of them can ever be
/// `extern "C"` — a real, permanent, unavoidable fact about what a C
/// ABI can express, not an oversight (see this module's own top-level
/// doc comment). That means a *pure* C program has no way to populate
/// any component data at all, which would otherwise leave
/// `ffi-smoke-test/test.c` unable to exercise the actual data
/// round-trip through the component-data functions below — only their
/// null-pointer and not-found paths, never real data through real
/// compiled C memory. This function does the necessary Rust-side setup
/// once, in Rust, and hands back an already-populated `World`:
/// registers [`MidEcsTestHealth`] with Sparse Shell (as `"FfiHealth"`)
/// and [`MidEcsTestHealthStatic`] with Archetype Core (as
/// `"FfiHealthStatic"`) — two distinct Rust types, not one shared
/// across both (see [`MidEcsTestHealthStatic`]'s own doc comment for
/// why), matching this file's own Rust-level test names, deliberately,
/// so the same fixture shape is verified from both directions. Spawns
/// two entities and inserts `{hp: 10}`/`{hp: 20}` (Sparse Shell) and
/// `{hp: 100}`/`{hp: 200}` (Archetype Core) on them respectively —
/// exactly the fixed values `test.c` asserts against. Never returns
/// NULL, matching `mid_ecs_world_new`'s own convention.
#[no_mangle]
pub extern "C" fn mid_ecs_test_fixture_world_new() -> *mut MidEcsWorld {
    let mut world = World::new();
    world.register_ffi_component::<MidEcsTestHealth>("FfiHealth");
    world.register_ffi_static_component::<MidEcsTestHealthStatic>("FfiHealthStatic");
    let e1 = world.spawn();
    let e2 = world.spawn();
    world.insert(e1, MidEcsTestHealth { hp: 10 });
    world.insert(e2, MidEcsTestHealth { hp: 20 });
    world.insert_static(e1, MidEcsTestHealthStatic { hp: 100 });
    world.insert_static(e2, MidEcsTestHealthStatic { hp: 200 });
    Box::into_raw(Box::new(MidEcsWorld(world)))
}

/// A sentinel `component_id`/`archetype_id` value meaning "not found" —
/// returned by the `lookup_ffi_*` functions below on a null pointer,
/// invalid UTF-8, or a name that was never registered. Not `0`: `0` is
/// a real, valid id for whichever type happened to be registered
/// first, so it can't double as a not-found signal. Matches
/// `mid_collections::sparse_set`'s own `u32::MAX`-as-sentinel
/// precedent (real code in this workspace, not invented for this
/// function) — reaching the actual 4-billionth distinct registered
/// component type is not a real scenario this needs to guard against.
pub const MID_ECS_INVALID_ID: u32 = u32::MAX;

/// Looks up the Sparse-Shell `component_id` a Rust type was registered
/// under via `World::register_ffi_component`, by the name given then.
/// Returns [`MID_ECS_INVALID_ID`] on a null `world`/`name`, invalid
/// UTF-8, an internal panic, or a name that was never registered — a C
/// caller can't distinguish those four cases from the return value
/// alone, matching `mid_net_player_event_new`'s own "collapse distinct
/// failure reasons to one sentinel for a getter-shaped function"
/// precedent (a status-code return doesn't fit a function whose whole
/// job is returning one value).
///
/// # Safety
/// `world` must either be NULL or a valid handle from
/// `mid_ecs_world_new`. If non-null, `name` must be a valid,
/// null-terminated C string.
#[no_mangle]
pub unsafe extern "C" fn mid_ecs_world_lookup_ffi_component_id(
    world: *const MidEcsWorld,
    name: *const c_char,
) -> u32 {
    if world.is_null() || name.is_null() {
        return MID_ECS_INVALID_ID;
    }
    let result = catch_unwind(AssertUnwindSafe(|| {
        let world = unsafe { &*world };
        let name = unsafe { CStr::from_ptr(name) }.to_str().ok()?;
        world.0.lookup_ffi_component_id(name)
    }));
    match result {
        Ok(Some(id)) => id.as_u32(),
        _ => MID_ECS_INVALID_ID,
    }
}

/// Non-generic, `component_id`-keyed raw span over every currently-
/// attached instance of a Sparse-Shell component type, written into
/// `*out_span` — thin FFI wrapper over `World::component_raw_span`.
/// Returns `MidEcsStatus::Ok` (`*out_span` written) or
/// `MidEcsStatus::NotFound` (`component_id` doesn't exist or was never
/// opted into FFI exposure) — never `MidEcsStatus::NotAlive`, this
/// isn't an entity-liveness operation.
///
/// `out_span`'s `ptr` is a live, zero-copy view into `world`'s own
/// storage — see `mid_collections::FfiSpan`'s own doc comment for the
/// exact invalidation contract: valid only until the next
/// `mid_ecs_world_*` call that mutates *this exact* component type on
/// this `world`.
///
/// # Safety
/// `world` and `out_span` must both be non-null; `world` a valid
/// handle from `mid_ecs_world_new`, `out_span` valid for one
/// `FfiSpan` write.
#[no_mangle]
pub unsafe extern "C" fn mid_ecs_world_component_raw_span(
    world: *const MidEcsWorld,
    component_id: u32,
    out_span: *mut FfiSpan,
) -> i32 {
    ffi_guard(|| {
        if world.is_null() || out_span.is_null() {
            return MidEcsStatus::NullPointer as i32;
        }
        let world = unsafe { &*world };
        match world
            .0
            .component_raw_span(ComponentId::from_u32(component_id))
        {
            Some(span) => {
                unsafe { ptr::write(out_span, span) };
                MidEcsStatus::Ok as i32
            }
            None => MidEcsStatus::NotFound as i32,
        }
    })
}

/// Entity-correlation counterpart to
/// [`mid_ecs_world_component_raw_span`]: `entity_ids[i]` (the packed
/// `u64` written to `out_buf[i]`, unpack with `Entity::from_ffi` on the
/// Rust side) is the entity that owns that same raw-span call's element
/// `i`, for every valid `i`.
///
/// Same "NULL buffer queries the required count" idiom as
/// `mid_net_player_state_encode`: pass `out_buf = NULL` to learn the
/// real element count from the return value alone, without writing
/// anything, then call again with a buffer sized to hold at least that
/// many `uint64_t`s. `out_buf_capacity` counts **elements, not
/// bytes** — unlike `mid-net`'s own byte-oriented buffers, since this
/// one is always exactly `uint64_t`-strided.
///
/// Returns the real element count (`>= 0`) on success (whether querying
/// or actually filling `out_buf`), or a negative `MidEcsStatus`:
/// `NullPointer` (null `world`), `NotFound` (`component_id` never
/// registered), or `BufferTooSmall` (non-null `out_buf` too small for
/// the real count — call again after querying, don't try to use a
/// partial fill).
///
/// # Safety
/// `world` must be a valid, non-null handle from `mid_ecs_world_new`.
/// If `out_buf` is non-null, it must be valid for `out_buf_capacity`
/// `uint64_t` elements.
#[no_mangle]
pub unsafe extern "C" fn mid_ecs_world_component_entity_ids(
    world: *const MidEcsWorld,
    component_id: u32,
    out_buf: *mut u64,
    out_buf_capacity: usize,
) -> i32 {
    ffi_guard(|| {
        if world.is_null() {
            return MidEcsStatus::NullPointer as i32;
        }
        let world = unsafe { &*world };
        let Some(ids) = world
            .0
            .component_entity_ids(ComponentId::from_u32(component_id))
        else {
            return MidEcsStatus::NotFound as i32;
        };
        if out_buf.is_null() {
            return ids.len() as i32;
        }
        if ids.len() > out_buf_capacity {
            return MidEcsStatus::BufferTooSmall as i32;
        }
        let out = unsafe { slice::from_raw_parts_mut(out_buf, out_buf_capacity) };
        out[..ids.len()].copy_from_slice(&ids);
        ids.len() as i32
    })
}

/// Looks up the Archetype-Core `component_id` a Rust type was
/// registered under via `World::register_ffi_static_component`, by
/// name. Same [`MID_ECS_INVALID_ID`] sentinel and collapsed-failure
/// shape as [`mid_ecs_world_lookup_ffi_component_id`] — see that
/// function's own doc comment. A **separate id namespace** from the
/// Sparse-Shell lookup above: the same `name` can resolve to a
/// different numeric id in each system, matching
/// `Archetypes`/`SparseShell`'s own already-independent `ComponentId`
/// registries at the Rust level.
///
/// # Safety
/// `world` must either be NULL or a valid handle from
/// `mid_ecs_world_new`. If non-null, `name` must be a valid,
/// null-terminated C string.
#[no_mangle]
pub unsafe extern "C" fn mid_ecs_world_lookup_ffi_static_component_id(
    world: *const MidEcsWorld,
    name: *const c_char,
) -> u32 {
    if world.is_null() || name.is_null() {
        return MID_ECS_INVALID_ID;
    }
    let result = catch_unwind(AssertUnwindSafe(|| {
        let world = unsafe { &*world };
        let name = unsafe { CStr::from_ptr(name) }.to_str().ok()?;
        world.0.lookup_ffi_static_component_id(name)
    }));
    match result {
        Ok(Some(id)) => id.as_u32(),
        _ => MID_ECS_INVALID_ID,
    }
}

/// Non-generic, per-archetype raw span over `component_id`'s column
/// within `archetype_id`'s table, written into `*out_span` — thin FFI
/// wrapper over `World::static_component_raw_span`. Returns
/// `MidEcsStatus::NotFound` if `component_id` was never registered,
/// `archetype_id` doesn't currently exist, or (a real, permanent
/// structural fact, not a transient one) `archetype_id`'s signature
/// simply doesn't include `component_id` — see that Rust method's own
/// doc comment for the full reasoning, including why an
/// empty-but-present column is `Ok` with `count == 0`, not
/// `NotFound`. Pair with
/// [`mid_ecs_world_archetypes_with_static_component`] to enumerate
/// every archetype currently containing a given component.
///
/// Same live, zero-copy `out_span` invalidation contract as
/// [`mid_ecs_world_component_raw_span`].
///
/// # Safety
/// `world` and `out_span` must both be non-null; `world` a valid
/// handle from `mid_ecs_world_new`, `out_span` valid for one
/// `FfiSpan` write.
#[no_mangle]
pub unsafe extern "C" fn mid_ecs_world_static_component_raw_span(
    world: *const MidEcsWorld,
    archetype_id: u32,
    component_id: u32,
    out_span: *mut FfiSpan,
) -> i32 {
    ffi_guard(|| {
        if world.is_null() || out_span.is_null() {
            return MidEcsStatus::NullPointer as i32;
        }
        let world = unsafe { &*world };
        match world.0.static_component_raw_span(
            ArchetypeId::from_u32(archetype_id),
            ComponentId::from_u32(component_id),
        ) {
            Some(span) => {
                unsafe { ptr::write(out_span, span) };
                MidEcsStatus::Ok as i32
            }
            None => MidEcsStatus::NotFound as i32,
        }
    })
}

/// Entity-correlation counterpart to
/// [`mid_ecs_world_static_component_raw_span`], for the same
/// `(archetype_id, component_id)` pair. Same NULL-buffer-queries-count
/// idiom, same element-count (not byte-count) `out_buf_capacity`, same
/// return-value shape as [`mid_ecs_world_component_entity_ids`] — see
/// that function's own doc comment.
///
/// # Safety
/// `world` must be a valid, non-null handle from `mid_ecs_world_new`.
/// If `out_buf` is non-null, it must be valid for `out_buf_capacity`
/// `uint64_t` elements.
#[no_mangle]
pub unsafe extern "C" fn mid_ecs_world_static_component_entity_ids(
    world: *const MidEcsWorld,
    archetype_id: u32,
    component_id: u32,
    out_buf: *mut u64,
    out_buf_capacity: usize,
) -> i32 {
    ffi_guard(|| {
        if world.is_null() {
            return MidEcsStatus::NullPointer as i32;
        }
        let world = unsafe { &*world };
        let Some(ids) = world.0.static_component_entity_ids(
            ArchetypeId::from_u32(archetype_id),
            ComponentId::from_u32(component_id),
        ) else {
            return MidEcsStatus::NotFound as i32;
        };
        if out_buf.is_null() {
            return ids.len() as i32;
        }
        if ids.len() > out_buf_capacity {
            return MidEcsStatus::BufferTooSmall as i32;
        }
        let out = unsafe { slice::from_raw_parts_mut(out_buf, out_buf_capacity) };
        out[..ids.len()].copy_from_slice(&ids);
        ids.len() as i32
    })
}

/// Enumerates every currently-existing archetype whose signature
/// includes `component_id`, writing each as a plain `u32` (unpack with
/// `mid_ecs_archetype_id_from_u32`-shaped logic on the Rust side, or
/// just pass it straight back into
/// [`mid_ecs_world_static_component_raw_span`]/
/// [`mid_ecs_world_static_component_entity_ids`] as-is — that's the
/// whole point of this function). Same NULL-buffer-queries-count idiom
/// as [`mid_ecs_world_component_entity_ids`]. **Not gated by FFI
/// registration** — matches `Archetypes::archetypes_with`'s own
/// Rust-level behavior exactly (a pure structural query; an
/// unregistered `component_id` just naturally matches zero archetypes,
/// not an error), so this never returns `MidEcsStatus::NotFound`,
/// only `NullPointer`/`BufferTooSmall`/`InternalPanic`.
///
/// # Safety
/// `world` must be a valid, non-null handle from `mid_ecs_world_new`.
/// If `out_buf` is non-null, it must be valid for `out_buf_capacity`
/// `uint32_t` elements.
#[no_mangle]
pub unsafe extern "C" fn mid_ecs_world_archetypes_with_static_component(
    world: *const MidEcsWorld,
    component_id: u32,
    out_buf: *mut u32,
    out_buf_capacity: usize,
) -> i32 {
    ffi_guard(|| {
        if world.is_null() {
            return MidEcsStatus::NullPointer as i32;
        }
        let world = unsafe { &*world };
        let ids: Vec<u32> = world
            .0
            .archetypes_with_static_component(ComponentId::from_u32(component_id))
            .map(|id| id.as_u32())
            .collect();
        if out_buf.is_null() {
            return ids.len() as i32;
        }
        if ids.len() > out_buf_capacity {
            return MidEcsStatus::BufferTooSmall as i32;
        }
        let out = unsafe { slice::from_raw_parts_mut(out_buf, out_buf_capacity) };
        out[..ids.len()].copy_from_slice(&ids);
        ids.len() as i32
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use zerocopy::{Immutable, IntoBytes, KnownLayout};

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
    fn world_new_free_round_trips() {
        let world = mid_ecs_world_new();
        assert!(!world.is_null());
        // SAFETY: `world` is non-null, just created, not yet freed.
        unsafe { mid_ecs_world_free(world) };
    }

    #[test]
    fn free_null_is_a_safe_no_op() {
        // SAFETY: NULL is the documented safe-no-op case.
        unsafe { mid_ecs_world_free(std::ptr::null_mut()) };
    }

    #[test]
    fn spawn_despawn_is_alive_round_trip_through_ffi() {
        let world = mid_ecs_world_new();
        // SAFETY: `world` non-null and not yet freed for this whole block.
        unsafe {
            let entity = mid_ecs_world_spawn(world);
            assert!(mid_ecs_world_is_alive(world, entity));
            assert_eq!(mid_ecs_world_entity_count(world), 1);

            let status = mid_ecs_world_despawn(world, entity);
            assert_eq!(status, MidEcsStatus::Ok as i32);
            assert!(!mid_ecs_world_is_alive(world, entity));
            assert_eq!(mid_ecs_world_entity_count(world), 0);

            mid_ecs_world_free(world);
        }
    }

    #[test]
    fn despawn_already_dead_returns_not_alive_not_panic() {
        let world = mid_ecs_world_new();
        // SAFETY: `world` non-null and not yet freed for this whole block.
        unsafe {
            let entity = mid_ecs_world_spawn(world);
            mid_ecs_world_despawn(world, entity);
            let second = mid_ecs_world_despawn(world, entity);
            assert_eq!(second, MidEcsStatus::NotAlive as i32);
            mid_ecs_world_free(world);
        }
    }

    #[test]
    fn stale_packed_entity_after_reuse_correctly_reads_as_not_alive() {
        let world = mid_ecs_world_new();
        // SAFETY: `world` non-null and not yet freed for this whole block.
        unsafe {
            let e1 = mid_ecs_world_spawn(world);
            mid_ecs_world_despawn(world, e1);
            let e2 = mid_ecs_world_spawn(world); // reuses e1's slot

            assert!(
                !mid_ecs_world_is_alive(world, e1),
                "the stale packed handle must not read as alive"
            );
            assert!(mid_ecs_world_is_alive(world, e2));

            mid_ecs_world_free(world);
        }
    }

    #[test]
    fn bogus_packed_entity_is_safe_and_reads_as_not_alive() {
        let world = mid_ecs_world_new();
        // SAFETY: `world` non-null and not yet freed for this whole block.
        unsafe {
            mid_ecs_world_spawn(world);
            let bogus: u64 = 0xFFFF_FFFF_FFFF_FFFF;
            assert!(!mid_ecs_world_is_alive(world, bogus));
            let status = mid_ecs_world_despawn(world, bogus);
            assert_eq!(status, MidEcsStatus::NotAlive as i32);
            mid_ecs_world_free(world);
        }
    }

    #[test]
    fn null_world_functions_return_safe_defaults_not_crash() {
        let null_world: *mut MidEcsWorld = std::ptr::null_mut();
        // SAFETY: every one of these has a documented, tested NULL-handle path.
        unsafe {
            assert_eq!(mid_ecs_world_spawn(null_world), 0);
            assert!(!mid_ecs_world_is_alive(null_world, 0));
            assert_eq!(mid_ecs_world_entity_count(null_world), 0);
            assert_eq!(
                mid_ecs_world_despawn(null_world, 0),
                MidEcsStatus::NullPointer as i32
            );
        }
    }

    // --- Sparse Shell component-data C surface ---

    #[test]
    fn lookup_ffi_component_id_resolves_a_registered_name() {
        let world_ptr = mid_ecs_world_new();
        let world = unsafe { &mut *world_ptr };
        let id = world.0.register_ffi_component::<FfiHealth>("FfiHealth");

        let name = std::ffi::CString::new("FfiHealth").unwrap();
        // SAFETY: world_ptr non-null and not yet freed; name is a real,
        // null-terminated C string.
        let looked_up = unsafe { mid_ecs_world_lookup_ffi_component_id(world_ptr, name.as_ptr()) };
        assert_eq!(looked_up, id.as_u32());

        unsafe { mid_ecs_world_free(world_ptr) };
    }

    #[test]
    fn lookup_ffi_component_id_on_unregistered_name_is_invalid_sentinel() {
        let world_ptr = mid_ecs_world_new();
        let name = std::ffi::CString::new("NeverRegistered").unwrap();
        // SAFETY: world_ptr non-null and not yet freed.
        let looked_up = unsafe { mid_ecs_world_lookup_ffi_component_id(world_ptr, name.as_ptr()) };
        assert_eq!(looked_up, MID_ECS_INVALID_ID);
        unsafe { mid_ecs_world_free(world_ptr) };
    }

    #[test]
    fn lookup_ffi_component_id_on_null_world_or_name_is_invalid_sentinel() {
        let world_ptr = mid_ecs_world_new();
        let name = std::ffi::CString::new("FfiHealth").unwrap();
        // SAFETY: NULL world and NULL name are both documented safe cases.
        unsafe {
            assert_eq!(
                mid_ecs_world_lookup_ffi_component_id(std::ptr::null(), name.as_ptr()),
                MID_ECS_INVALID_ID
            );
            assert_eq!(
                mid_ecs_world_lookup_ffi_component_id(world_ptr, std::ptr::null()),
                MID_ECS_INVALID_ID
            );
            mid_ecs_world_free(world_ptr);
        }
    }

    #[test]
    fn component_raw_span_and_entity_ids_correlate_through_the_c_surface() {
        let world_ptr = mid_ecs_world_new();
        let world = unsafe { &mut *world_ptr };
        let id = world.0.register_ffi_component::<FfiHealth>("FfiHealth");
        let e1 = world.0.spawn();
        let e2 = world.0.spawn();
        world.0.insert(e1, FfiHealth { hp: 10 });
        world.0.insert(e2, FfiHealth { hp: 20 });

        let mut span = FfiSpan {
            ptr: std::ptr::null(),
            stride: 0,
            count: 0,
        };
        // SAFETY: world_ptr and &mut span both non-null and valid.
        let status = unsafe { mid_ecs_world_component_raw_span(world_ptr, id.as_u32(), &mut span) };
        assert_eq!(status, MidEcsStatus::Ok as i32);
        assert_eq!(span.count, 2);

        // Query mode: NULL buffer returns the real count, writes nothing.
        let queried = unsafe {
            mid_ecs_world_component_entity_ids(world_ptr, id.as_u32(), std::ptr::null_mut(), 0)
        };
        assert_eq!(queried, 2);

        let mut ids = [0u64; 2];
        // SAFETY: world_ptr valid; ids valid for 2 elements.
        let written = unsafe {
            mid_ecs_world_component_entity_ids(world_ptr, id.as_u32(), ids.as_mut_ptr(), ids.len())
        };
        assert_eq!(written, 2);

        // SAFETY: span.ptr points at world's own live storage, unmutated
        // since the calls above.
        let values =
            unsafe { std::slice::from_raw_parts(span.ptr.cast::<FfiHealth>(), span.count) };
        assert_eq!(Entity::from_ffi(ids[0]), e1);
        assert_eq!(Entity::from_ffi(ids[1]), e2);
        assert_eq!(values[0], FfiHealth { hp: 10 });
        assert_eq!(values[1], FfiHealth { hp: 20 });

        unsafe { mid_ecs_world_free(world_ptr) };
    }

    #[test]
    fn component_entity_ids_buffer_too_small_is_a_real_error_not_a_partial_fill() {
        let world_ptr = mid_ecs_world_new();
        let world = unsafe { &mut *world_ptr };
        let id = world.0.register_ffi_component::<FfiHealth>("FfiHealth");
        let e1 = world.0.spawn();
        let e2 = world.0.spawn();
        world.0.insert(e1, FfiHealth { hp: 1 });
        world.0.insert(e2, FfiHealth { hp: 2 });

        let mut too_small = [0u64; 1];
        // SAFETY: world_ptr valid; too_small valid for its own length (1).
        let status = unsafe {
            mid_ecs_world_component_entity_ids(
                world_ptr,
                id.as_u32(),
                too_small.as_mut_ptr(),
                too_small.len(),
            )
        };
        assert_eq!(status, MidEcsStatus::BufferTooSmall as i32);

        unsafe { mid_ecs_world_free(world_ptr) };
    }

    #[test]
    fn component_raw_span_and_entity_ids_on_never_registered_id_is_not_found() {
        let world_ptr = mid_ecs_world_new();
        let mut span = FfiSpan {
            ptr: std::ptr::null(),
            stride: 0,
            count: 0,
        };
        // SAFETY: world_ptr non-null; &mut span valid.
        unsafe {
            assert_eq!(
                mid_ecs_world_component_raw_span(world_ptr, MID_ECS_INVALID_ID, &mut span),
                MidEcsStatus::NotFound as i32
            );
            assert_eq!(
                mid_ecs_world_component_entity_ids(
                    world_ptr,
                    MID_ECS_INVALID_ID,
                    std::ptr::null_mut(),
                    0
                ),
                MidEcsStatus::NotFound as i32
            );
            mid_ecs_world_free(world_ptr);
        }
    }

    #[test]
    fn component_raw_span_null_pointer_cases() {
        let world_ptr = mid_ecs_world_new();
        let mut span = FfiSpan {
            ptr: std::ptr::null(),
            stride: 0,
            count: 0,
        };
        // SAFETY: exercising the documented NULL-pointer error paths.
        unsafe {
            assert_eq!(
                mid_ecs_world_component_raw_span(std::ptr::null(), 0, &mut span),
                MidEcsStatus::NullPointer as i32
            );
            assert_eq!(
                mid_ecs_world_component_raw_span(world_ptr, 0, std::ptr::null_mut()),
                MidEcsStatus::NullPointer as i32
            );
            mid_ecs_world_free(world_ptr);
        }
    }

    // --- Archetype Core component-data C surface ---

    #[test]
    fn static_component_raw_span_and_entity_ids_correlate_through_the_c_surface() {
        let world_ptr = mid_ecs_world_new();
        let world = unsafe { &mut *world_ptr };
        let id = world
            .0
            .register_ffi_static_component::<FfiHealth>("FfiHealthStatic");
        let e1 = world.0.spawn();
        let e2 = world.0.spawn();
        world.0.insert_static(e1, FfiHealth { hp: 100 });
        world.0.insert_static(e2, FfiHealth { hp: 200 });

        let archetype_id = world
            .0
            .archetypes_with_static_component(id)
            .next()
            .expect("both entities share one archetype")
            .as_u32();

        let mut span = FfiSpan {
            ptr: std::ptr::null(),
            stride: 0,
            count: 0,
        };
        // SAFETY: world_ptr and &mut span both valid.
        let status = unsafe {
            mid_ecs_world_static_component_raw_span(world_ptr, archetype_id, id.as_u32(), &mut span)
        };
        assert_eq!(status, MidEcsStatus::Ok as i32);
        assert_eq!(span.count, 2);

        let mut ids = [0u64; 2];
        // SAFETY: world_ptr valid; ids valid for 2 elements.
        let written = unsafe {
            mid_ecs_world_static_component_entity_ids(
                world_ptr,
                archetype_id,
                id.as_u32(),
                ids.as_mut_ptr(),
                ids.len(),
            )
        };
        assert_eq!(written, 2);

        // SAFETY: span.ptr points at world's own live storage, unmutated
        // since the calls above.
        let values =
            unsafe { std::slice::from_raw_parts(span.ptr.cast::<FfiHealth>(), span.count) };
        assert_eq!(Entity::from_ffi(ids[0]), e1);
        assert_eq!(Entity::from_ffi(ids[1]), e2);
        assert_eq!(values[0], FfiHealth { hp: 100 });
        assert_eq!(values[1], FfiHealth { hp: 200 });

        unsafe { mid_ecs_world_free(world_ptr) };
    }

    #[test]
    fn lookup_ffi_static_component_id_uses_an_independent_namespace_from_sparse_shell() {
        let world_ptr = mid_ecs_world_new();
        let world = unsafe { &mut *world_ptr };
        // Same name, two different types, one per storage system -- real
        // proof the two id namespaces are independent, matching
        // Archetypes/SparseShell's own separate registries. Two distinct
        // types deliberately, not the same type reused across both
        // systems -- World's own StorageClaims guard now forbids exactly
        // that (see world.rs's own doc comment on it).
        let sparse_id = world.0.register_ffi_component::<FfiHealth>("FfiHealth");
        let static_id = world
            .0
            .register_ffi_static_component::<FfiStamina>("FfiHealth");

        let name = std::ffi::CString::new("FfiHealth").unwrap();
        // SAFETY: world_ptr non-null; name is a real C string.
        unsafe {
            assert_eq!(
                mid_ecs_world_lookup_ffi_component_id(world_ptr, name.as_ptr()),
                sparse_id.as_u32()
            );
            assert_eq!(
                mid_ecs_world_lookup_ffi_static_component_id(world_ptr, name.as_ptr()),
                static_id.as_u32()
            );
            mid_ecs_world_free(world_ptr);
        }
    }

    #[test]
    fn archetypes_with_static_component_query_then_fill_through_the_c_surface() {
        let world_ptr = mid_ecs_world_new();
        let world = unsafe { &mut *world_ptr };
        let health_id = world
            .0
            .register_ffi_static_component::<FfiHealth>("FfiHealthStatic");
        let e1 = world.0.spawn();
        let e2 = world.0.spawn();
        world.0.insert_static(e1, FfiHealth { hp: 1 });
        world.0.insert_static(e2, FfiHealth { hp: 2 });

        // SAFETY: world_ptr valid; NULL buffer is the documented query mode.
        let queried = unsafe {
            mid_ecs_world_archetypes_with_static_component(
                world_ptr,
                health_id.as_u32(),
                std::ptr::null_mut(),
                0,
            )
        };
        assert_eq!(queried, 1, "e1 and e2 share one archetype");

        let mut buf = [0u32; 1];
        // SAFETY: world_ptr valid; buf valid for 1 element.
        let written = unsafe {
            mid_ecs_world_archetypes_with_static_component(
                world_ptr,
                health_id.as_u32(),
                buf.as_mut_ptr(),
                buf.len(),
            )
        };
        assert_eq!(written, 1);

        // SAFETY: world_ptr and &mut span both valid.
        let mut span = FfiSpan {
            ptr: std::ptr::null(),
            stride: 0,
            count: 0,
        };
        let status = unsafe {
            mid_ecs_world_static_component_raw_span(
                world_ptr,
                buf[0],
                health_id.as_u32(),
                &mut span,
            )
        };
        assert_eq!(
            status,
            MidEcsStatus::Ok as i32,
            "the archetype id this function handed back must be immediately usable"
        );
        assert_eq!(span.count, 2);

        unsafe { mid_ecs_world_free(world_ptr) };
    }

    #[test]
    fn archetypes_with_static_component_on_unregistered_id_is_zero_not_an_error() {
        let world_ptr = mid_ecs_world_new();
        // SAFETY: world_ptr non-null; NULL buffer is the documented query mode.
        let count = unsafe {
            mid_ecs_world_archetypes_with_static_component(
                world_ptr,
                MID_ECS_INVALID_ID,
                std::ptr::null_mut(),
                0,
            )
        };
        assert_eq!(
            count, 0,
            "matches Archetypes::archetypes_with's own not-gated-by-registration behavior"
        );
        unsafe { mid_ecs_world_free(world_ptr) };
    }

    #[test]
    fn static_component_raw_span_and_entity_ids_on_never_registered_id_is_not_found() {
        let world_ptr = mid_ecs_world_new();
        let mut span = FfiSpan {
            ptr: std::ptr::null(),
            stride: 0,
            count: 0,
        };
        // SAFETY: world_ptr non-null; &mut span valid.
        unsafe {
            assert_eq!(
                mid_ecs_world_static_component_raw_span(
                    world_ptr,
                    0,
                    MID_ECS_INVALID_ID,
                    &mut span
                ),
                MidEcsStatus::NotFound as i32
            );
            assert_eq!(
                mid_ecs_world_static_component_entity_ids(
                    world_ptr,
                    0,
                    MID_ECS_INVALID_ID,
                    std::ptr::null_mut(),
                    0
                ),
                MidEcsStatus::NotFound as i32
            );
            mid_ecs_world_free(world_ptr);
        }
    }
}
