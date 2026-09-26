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
use crate::resource::{ResourceFfiError, ResourceId};
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
    /// A write's byte length isn't exactly the size of the registered
    /// resource type. Nothing was written.
    SizeMismatch = -6,
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

/// Extra Archetype Core fixture types, used only by
/// [`mid_ecs_test_filter_fixture_world_new`].
#[derive(zerocopy::IntoBytes, zerocopy::Immutable, zerocopy::KnownLayout)]
#[repr(C)]
pub struct MidEcsTestFlagA {
    pub v: u32,
}

/// See [`MidEcsTestFlagA`].
#[derive(zerocopy::IntoBytes, zerocopy::Immutable, zerocopy::KnownLayout)]
#[repr(C)]
pub struct MidEcsTestFlagB {
    pub v: u32,
}

/// **Test-fixture only, like [`mid_ecs_test_fixture_world_new`].**
/// Builds a world for exercising
/// [`mid_ecs_world_archetypes_matching_static`] from a pure C program:
/// `"FfiHealthStatic"`, `"FfiFlagA"` and `"FfiFlagB"` registered with the
/// Archetype Core, and three entities spread over distinct archetypes
/// (`hp` in parentheses):
///
/// - `e1`: `{Health}` (1)
/// - `e2`: `{Health, FlagA}` (2), built with `insert_bundle`
/// - `e3`: `{FlagB, Health, FlagA}` (3), built with `insert_bundle`
///
/// Building `e3` in that order deliberately leaves zero-row intermediate
/// archetypes `{FlagB}` and `{FlagB, Health}` behind, so every
/// enumeration a C caller runs also has to cope with archetypes that hold
/// the component in their signature but no rows. Never returns NULL.
#[no_mangle]
pub extern "C" fn mid_ecs_test_filter_fixture_world_new() -> *mut MidEcsWorld {
    let mut world = World::new();
    world.register_ffi_static_component::<MidEcsTestHealthStatic>("FfiHealthStatic");
    world.register_ffi_static_component::<MidEcsTestFlagA>("FfiFlagA");
    world.register_ffi_static_component::<MidEcsTestFlagB>("FfiFlagB");
    let e1 = world.spawn();
    let e2 = world.spawn();
    let e3 = world.spawn();
    world.insert_static(e1, MidEcsTestHealthStatic { hp: 1 });
    world.insert_bundle(
        e2,
        (MidEcsTestHealthStatic { hp: 2 }, MidEcsTestFlagA { v: 20 }),
    );
    world.insert_bundle(
        e3,
        (
            MidEcsTestFlagB { v: 300 },
            MidEcsTestHealthStatic { hp: 3 },
            MidEcsTestFlagA { v: 30 },
        ),
    );
    Box::into_raw(Box::new(MidEcsWorld(world)))
}

/// Resource fixture types, used only by
/// [`mid_ecs_test_resource_fixture_world_new`]. Resources must accept any
/// bit pattern (`FromBytes`), since C writes them as raw bytes.
#[derive(zerocopy::FromBytes, zerocopy::IntoBytes, zerocopy::Immutable, zerocopy::KnownLayout)]
#[repr(C)]
pub struct MidEcsTestTime {
    pub delta: f32,
    pub frame: u32,
}

/// See [`MidEcsTestTime`].
#[derive(zerocopy::FromBytes, zerocopy::IntoBytes, zerocopy::Immutable, zerocopy::KnownLayout)]
#[repr(C)]
pub struct MidEcsTestGravity {
    pub g: f32,
}

/// **Test-fixture only, like [`mid_ecs_test_fixture_world_new`].** A world
/// for exercising the resource functions from a pure C program (C cannot
/// call the generic `register_ffi_resource`/`insert_resource`):
/// `"FfiTime"` registered and inserted as `{ delta: 0.016, frame: 7 }`,
/// and `"FfiGravity"` registered but not inserted. Never returns NULL.
#[no_mangle]
pub extern "C" fn mid_ecs_test_resource_fixture_world_new() -> *mut MidEcsWorld {
    let mut world = World::new();
    world.register_ffi_resource::<MidEcsTestTime>("FfiTime");
    world.register_ffi_resource::<MidEcsTestGravity>("FfiGravity");
    world.insert_resource(MidEcsTestTime {
        delta: 0.016,
        frame: 7,
    });
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

/// Enumerates every currently-existing archetype whose signature
/// contains *all* of `with_ids`, *none* of `without_ids`, and — if
/// `any_of_ids` is non-empty — *at least one* of `any_of_ids`, writing
/// each as a plain `u32` — the runtime counterpart to the typed
/// `With`/`Without`/`Or` query filters on the Rust side. Pass the ids
/// back into [`mid_ecs_world_static_component_raw_span`]/
/// [`mid_ecs_world_static_component_entity_ids`] as-is. Include the
/// component you intend to read in `with_ids`: an archetype that doesn't
/// hold it answers those two calls with `NotFound`.
///
/// Structural, like [`mid_ecs_world_archetypes_with_static_component`]:
/// archetypes with zero rows are included (their spans come back `Ok`
/// with `count == 0`), an id that names no registered component matches
/// nothing in `with_ids`/`any_of_ids` and is ignored in `without_ids`, an
/// id in both `with_ids` and `without_ids` matches nothing, and
/// `with_ids`/`without_ids` both empty with `any_of_ids` also empty
/// matches every archetype. An empty `any_of_ids` is "no `Or`
/// constraint", not "match nothing" — pass a zero length (the pointer
/// may then be NULL or dangling) when there's no `any_of` list. Never
/// returns `NotFound`. Same NULL-buffer-queries-count idiom as the other
/// enumerations.
///
/// **Signature change:** this function gained `any_of_ids`/`any_of_len`
/// (inserted before `out_buf`) when `Or` was added on the Rust side —
/// every existing call site needs those two arguments now, `NULL, 0` if
/// unused. See `docs/mid-ecs.md`, "filter.rs" (the `Or` section), for
/// why this signature was changed rather than adding a second function.
///
/// # Safety
/// `world` must be a valid, non-null handle from `mid_ecs_world_new`.
/// `with_ids` must be NULL only if `with_len` is 0, otherwise valid for
/// `with_len` `uint32_t` elements; likewise `without_ids`/`without_len`
/// and `any_of_ids`/`any_of_len`. If `out_buf` is non-null, it must be
/// valid for `out_buf_capacity` `uint32_t` elements.
#[no_mangle]
pub unsafe extern "C" fn mid_ecs_world_archetypes_matching_static(
    world: *const MidEcsWorld,
    with_ids: *const u32,
    with_len: usize,
    without_ids: *const u32,
    without_len: usize,
    any_of_ids: *const u32,
    any_of_len: usize,
    out_buf: *mut u32,
    out_buf_capacity: usize,
) -> i32 {
    ffi_guard(|| {
        if world.is_null()
            || (with_ids.is_null() && with_len > 0)
            || (without_ids.is_null() && without_len > 0)
            || (any_of_ids.is_null() && any_of_len > 0)
        {
            return MidEcsStatus::NullPointer as i32;
        }
        let world = unsafe { &*world };
        let read_ids = |ptr: *const u32, len: usize| -> Vec<ComponentId> {
            if len == 0 {
                return Vec::new();
            }
            unsafe { slice::from_raw_parts(ptr, len) }
                .iter()
                .map(|&id| ComponentId::from_u32(id))
                .collect()
        };
        let with = read_ids(with_ids, with_len);
        let without = read_ids(without_ids, without_len);
        let any_of = read_ids(any_of_ids, any_of_len);
        let ids: Vec<u32> = world
            .0
            .archetypes_matching_static(&with, &without, &any_of)
            .map(|id| id.as_u32())
            .collect();
        if out_buf.is_null() {
            return ids.len() as i32;
        }
        if ids.len() > out_buf_capacity {
            return MidEcsStatus::BufferTooSmall as i32;
        }
        let out = unsafe { slice::from_raw_parts_mut(out_buf, ids.len()) };
        out.copy_from_slice(&ids);
        ids.len() as i32
    })
}

fn resource_status(error: ResourceFfiError) -> i32 {
    match error {
        ResourceFfiError::UnknownId | ResourceFfiError::Absent => MidEcsStatus::NotFound as i32,
        ResourceFfiError::SizeMismatch => MidEcsStatus::SizeMismatch as i32,
    }
}

/// Looks up the `resource_id` a resource type was registered under via
/// `World::register_ffi_resource`, by name. Returns [`MID_ECS_INVALID_ID`]
/// on a null `world`/`name`, invalid UTF-8, an internal panic, or a name
/// that was never registered, the same collapse to one sentinel as
/// [`mid_ecs_world_lookup_ffi_component_id`]. Resource ids are their own
/// namespace, separate from component ids.
///
/// # Safety
/// `world` must either be NULL or a valid handle from `mid_ecs_world_new`.
/// If non-null, `name` must be a valid, null-terminated C string.
#[no_mangle]
pub unsafe extern "C" fn mid_ecs_world_lookup_ffi_resource_id(
    world: *const MidEcsWorld,
    name: *const c_char,
) -> u32 {
    if world.is_null() || name.is_null() {
        return MID_ECS_INVALID_ID;
    }
    let result = catch_unwind(AssertUnwindSafe(|| {
        let world = unsafe { &*world };
        let name = unsafe { CStr::from_ptr(name) }.to_str().ok()?;
        world.0.lookup_ffi_resource_id(name)
    }));
    match result {
        Ok(Some(id)) => id.as_u32(),
        _ => MID_ECS_INVALID_ID,
    }
}

/// A view of the registered resource's current value, written into
/// `*out_span`: one element (`count == 1`, `stride` the type's size), or
/// the empty span (`count == 0`) if the resource is registered but not
/// currently inserted. Returns `MidEcsStatus::NotFound` if `resource_id`
/// was never issued.
///
/// The span points into the live value. It stays valid across
/// [`mid_ecs_world_resource_write`] (which updates an existing value in
/// place) and is invalidated by [`mid_ecs_world_resource_remove`], by a
/// Rust-side `insert_resource`/`remove_resource` of the same type, and by
/// freeing the world.
///
/// # Safety
/// `world` must be a valid, non-null handle from `mid_ecs_world_new`.
/// `out_span` must be valid for writing one `MidEcsFfiSpan`.
#[no_mangle]
pub unsafe extern "C" fn mid_ecs_world_resource_raw_span(
    world: *const MidEcsWorld,
    resource_id: u32,
    out_span: *mut FfiSpan,
) -> i32 {
    ffi_guard(|| {
        if world.is_null() || out_span.is_null() {
            return MidEcsStatus::NullPointer as i32;
        }
        let world = unsafe { &*world };
        match world.0.resource_raw_span(ResourceId::from_u32(resource_id)) {
            Some(span) => {
                unsafe { ptr::write(out_span, span) };
                MidEcsStatus::Ok as i32
            }
            None => MidEcsStatus::NotFound as i32,
        }
    })
}

/// Copies `len` bytes from `bytes` in as the registered resource's new
/// value, inserting it if it isn't currently inserted. `len` must be
/// exactly the registered type's size (`stride` in the span), otherwise
/// nothing is written and the result is `MidEcsStatus::SizeMismatch`.
/// `MidEcsStatus::NotFound` if `resource_id` was never issued. Bytes are
/// copied, so `bytes` needs no particular alignment and is not retained;
/// registration requires the type to accept any bit pattern.
///
/// # Safety
/// `world` must be a valid, non-null handle from `mid_ecs_world_new`.
/// `bytes` must be NULL only if `len` is 0, otherwise valid for reading
/// `len` bytes.
#[no_mangle]
pub unsafe extern "C" fn mid_ecs_world_resource_write(
    world: *mut MidEcsWorld,
    resource_id: u32,
    bytes: *const u8,
    len: usize,
) -> i32 {
    ffi_guard(|| {
        if world.is_null() || (bytes.is_null() && len > 0) {
            return MidEcsStatus::NullPointer as i32;
        }
        let world = unsafe { &mut *world };
        let bytes: &[u8] = if len == 0 {
            &[]
        } else {
            unsafe { slice::from_raw_parts(bytes, len) }
        };
        match world
            .0
            .write_resource_bytes(ResourceId::from_u32(resource_id), bytes)
        {
            Ok(()) => MidEcsStatus::Ok as i32,
            Err(error) => resource_status(error),
        }
    })
}

/// Removes the registered resource's value. `MidEcsStatus::NotFound` if
/// `resource_id` was never issued or the resource isn't currently
/// inserted (nothing was removed either way). Invalidates any span
/// previously obtained for it.
///
/// # Safety
/// `world` must be a valid, non-null handle from `mid_ecs_world_new`.
#[no_mangle]
pub unsafe extern "C" fn mid_ecs_world_resource_remove(
    world: *mut MidEcsWorld,
    resource_id: u32,
) -> i32 {
    ffi_guard(|| {
        if world.is_null() {
            return MidEcsStatus::NullPointer as i32;
        }
        let world = unsafe { &mut *world };
        match world
            .0
            .remove_ffi_resource(ResourceId::from_u32(resource_id))
        {
            Ok(()) => MidEcsStatus::Ok as i32,
            Err(error) => resource_status(error),
        }
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

    // ── archetypes_matching_static ──────────────────────────────────

    use crate::filter::{Or, QueryFilter, With, Without};

    fn empty_span() -> FfiSpan {
        FfiSpan {
            ptr: std::ptr::null(),
            stride: 0,
            count: 0,
        }
    }

    fn static_id(world: *const MidEcsWorld, name: &str) -> u32 {
        let name = std::ffi::CString::new(name).unwrap();
        // SAFETY: `world` is a live handle and `name` a valid C string.
        let id = unsafe { mid_ecs_world_lookup_ffi_static_component_id(world, name.as_ptr()) };
        assert_ne!(id, MID_ECS_INVALID_ID);
        id
    }

    /// Runs the count-then-fill idiom the C header documents, with no
    /// `any_of` constraint. See [`matching_ids_any`] for the `Or` form.
    fn matching_ids(world: *const MidEcsWorld, with: &[u32], without: &[u32]) -> Vec<u32> {
        matching_ids_any(world, with, without, &[])
    }

    /// [`matching_ids`], with an explicit `any_of` (`Or`) list.
    fn matching_ids_any(
        world: *const MidEcsWorld,
        with: &[u32],
        without: &[u32],
        any_of: &[u32],
    ) -> Vec<u32> {
        // SAFETY: `world` is a live handle; the id slices outlive the call.
        let count = unsafe {
            mid_ecs_world_archetypes_matching_static(
                world,
                with.as_ptr(),
                with.len(),
                without.as_ptr(),
                without.len(),
                any_of.as_ptr(),
                any_of.len(),
                std::ptr::null_mut(),
                0,
            )
        };
        assert!(count >= 0, "count query failed with status {count}");
        let mut buf = vec![0u32; count as usize];
        // SAFETY: as above; `buf` is valid for `buf.len()` elements.
        let written = unsafe {
            mid_ecs_world_archetypes_matching_static(
                world,
                with.as_ptr(),
                with.len(),
                without.as_ptr(),
                without.len(),
                any_of.as_ptr(),
                any_of.len(),
                buf.as_mut_ptr(),
                buf.len(),
            )
        };
        assert_eq!(written, count);
        buf
    }

    /// Every entity found by walking the matching archetypes the way a C
    /// caller would. Asserts that each archetype the enumeration hands
    /// out resolves through both per-archetype calls.
    fn entities_via_ffi(
        world: *const MidEcsWorld,
        read: u32,
        with: &[u32],
        without: &[u32],
        any_of: &[u32],
    ) -> Vec<u64> {
        let mut out = Vec::new();
        for archetype in matching_ids_any(world, with, without, any_of) {
            let mut span = empty_span();
            // SAFETY: `world` is a live handle, `span` is valid.
            let status = unsafe {
                mid_ecs_world_static_component_raw_span(world, archetype, read, &mut span)
            };
            assert_eq!(
                status,
                MidEcsStatus::Ok as i32,
                "archetype {archetype} was enumerated, so raw_span must resolve it"
            );
            // SAFETY: as above; NULL buffer asks for the count.
            let n = unsafe {
                mid_ecs_world_static_component_entity_ids(
                    world,
                    archetype,
                    read,
                    std::ptr::null_mut(),
                    0,
                )
            };
            assert_eq!(n as usize, span.count);
            let mut ids = vec![0u64; n as usize];
            // SAFETY: `ids` is valid for `ids.len()` elements.
            let written = unsafe {
                mid_ecs_world_static_component_entity_ids(
                    world,
                    archetype,
                    read,
                    ids.as_mut_ptr(),
                    ids.len(),
                )
            };
            assert_eq!(written, n);
            out.extend(ids);
        }
        out.sort_unstable();
        out
    }

    fn typed_entities<F: QueryFilter>(world: &World) -> Vec<u64> {
        let mut v: Vec<u64> = world
            .query_static_filtered::<MidEcsTestHealthStatic, F>()
            .map(|(e, _)| e.as_ffi())
            .collect();
        v.sort_unstable();
        v
    }

    #[test]
    fn matching_static_enumerates_the_expected_archetypes() {
        let world = mid_ecs_test_filter_fixture_world_new();
        let h = static_id(world, "FfiHealthStatic");
        let a = static_id(world, "FfiFlagA");
        let b = static_id(world, "FfiFlagB");

        // Health: {H}, {H,A}, the zero-row {B,H}, and {B,H,A}.
        assert_eq!(matching_ids(world, &[h], &[]).len(), 4);
        assert_eq!(matching_ids(world, &[h], &[a]).len(), 2); // {H}, {B,H}
        assert_eq!(matching_ids(world, &[h, a], &[]).len(), 2);
        assert_eq!(matching_ids(world, &[h], &[a, b]).len(), 1);
        assert_eq!(matching_ids(world, &[h, a, b], &[]).len(), 1);
        // Two empty lists: every archetype, the empty one and the
        // zero-row intermediates included.
        assert_eq!(matching_ids(world, &[], &[]).len(), 6);

        // SAFETY: `world` is a live handle, freed exactly once.
        unsafe { mid_ecs_world_free(world) };
    }

    #[test]
    fn every_enumerated_archetype_resolves_including_zero_row_intermediates() {
        let world = mid_ecs_test_filter_fixture_world_new();
        let h = static_id(world, "FfiHealthStatic");

        // `entities_via_ffi` asserts `Ok` from raw_span on all four
        // archetypes, one of which never held a row.
        let found = entities_via_ffi(world, h, &[h], &[], &[]);
        assert_eq!(found.len(), 3);

        // SAFETY: as above.
        unsafe { mid_ecs_world_free(world) };
    }

    #[test]
    fn ffi_enumeration_matches_the_typed_filtered_queries() {
        let world_ptr = mid_ecs_test_filter_fixture_world_new();
        let h = static_id(world_ptr, "FfiHealthStatic");
        let a = static_id(world_ptr, "FfiFlagA");
        let b = static_id(world_ptr, "FfiFlagB");
        // SAFETY: `world_ptr` is a live handle for this whole test.
        let world = unsafe { &(*world_ptr).0 };

        let check = |typed: Vec<u64>, with: &[u32], without: &[u32], expected_len: usize| {
            let via_ffi = entities_via_ffi(world_ptr, h, with, without, &[]);
            assert_eq!(via_ffi, typed, "with {with:?} without {without:?}");
            assert_eq!(
                typed.len(),
                expected_len,
                "with {with:?} without {without:?}"
            );
        };

        check(typed_entities::<()>(world), &[h], &[], 3);
        check(
            typed_entities::<With<MidEcsTestFlagA>>(world),
            &[h, a],
            &[],
            2,
        );
        check(
            typed_entities::<Without<MidEcsTestFlagA>>(world),
            &[h],
            &[a],
            1,
        );
        check(
            typed_entities::<With<MidEcsTestFlagB>>(world),
            &[h, b],
            &[],
            1,
        );
        check(
            typed_entities::<Without<MidEcsTestFlagB>>(world),
            &[h],
            &[b],
            2,
        );
        check(
            typed_entities::<(With<MidEcsTestFlagA>, Without<MidEcsTestFlagB>)>(world),
            &[h, a],
            &[b],
            1,
        );
        check(
            typed_entities::<(With<MidEcsTestFlagA>, With<MidEcsTestFlagB>)>(world),
            &[h, a, b],
            &[],
            1,
        );
        check(
            typed_entities::<(Without<MidEcsTestFlagA>, Without<MidEcsTestFlagB>)>(world),
            &[h],
            &[a, b],
            1,
        );

        // SAFETY: freed exactly once.
        unsafe { mid_ecs_world_free(world_ptr) };
    }

    #[test]
    fn matching_static_buffer_idiom() {
        let world = mid_ecs_test_filter_fixture_world_new();
        let h = static_id(world, "FfiHealthStatic");
        let with = [h];
        let call = |buf: *mut u32, cap: usize| unsafe {
            // SAFETY: `world` is live; `with` outlives the call; callers
            // pass a buffer valid for `cap` elements (or NULL).
            mid_ecs_world_archetypes_matching_static(
                world,
                with.as_ptr(),
                with.len(),
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                buf,
                cap,
            )
        };

        assert_eq!(
            call(std::ptr::null_mut(), 0),
            4,
            "NULL buffer asks for the count"
        );

        let mut small = [0u32; 3];
        assert_eq!(
            call(small.as_mut_ptr(), small.len()),
            MidEcsStatus::BufferTooSmall as i32,
            "too small is an error, not a partial fill"
        );
        assert_eq!(small, [0, 0, 0], "and nothing was written");

        let mut exact = [0u32; 4];
        assert_eq!(call(exact.as_mut_ptr(), exact.len()), 4);
        let mut roomy = [u32::MAX; 8];
        assert_eq!(call(roomy.as_mut_ptr(), roomy.len()), 4);
        assert_eq!(&roomy[..4], &exact[..]);
        assert!(
            roomy[4..].iter().all(|&x| x == u32::MAX),
            "spare capacity untouched"
        );

        // SAFETY: freed exactly once.
        unsafe { mid_ecs_world_free(world) };
    }

    #[test]
    fn matching_static_null_pointer_cases() {
        let world = mid_ecs_test_filter_fixture_world_new();
        let h = static_id(world, "FfiHealthStatic");
        let ids = [h];
        let np = MidEcsStatus::NullPointer as i32;

        // SAFETY: every pointer passed is either NULL (the case under
        // test) or valid for the stated length.
        unsafe {
            assert_eq!(
                mid_ecs_world_archetypes_matching_static(
                    std::ptr::null(),
                    ids.as_ptr(),
                    1,
                    std::ptr::null(),
                    0,
                    std::ptr::null(),
                    0,
                    std::ptr::null_mut(),
                    0
                ),
                np,
                "NULL world"
            );
            assert_eq!(
                mid_ecs_world_archetypes_matching_static(
                    world,
                    std::ptr::null(),
                    1,
                    std::ptr::null(),
                    0,
                    std::ptr::null(),
                    0,
                    std::ptr::null_mut(),
                    0
                ),
                np,
                "NULL with_ids with a non-zero length"
            );
            assert_eq!(
                mid_ecs_world_archetypes_matching_static(
                    world,
                    ids.as_ptr(),
                    1,
                    std::ptr::null(),
                    2,
                    std::ptr::null(),
                    0,
                    std::ptr::null_mut(),
                    0
                ),
                np,
                "NULL without_ids with a non-zero length"
            );
            assert_eq!(
                mid_ecs_world_archetypes_matching_static(
                    world,
                    std::ptr::null(),
                    0,
                    std::ptr::null(),
                    0,
                    std::ptr::null(),
                    0,
                    std::ptr::null_mut(),
                    0
                ),
                6,
                "(NULL, 0) is a valid empty list: every archetype matches"
            );
            assert_eq!(
                mid_ecs_world_archetypes_matching_static(
                    world,
                    ids.as_ptr(),
                    1,
                    std::ptr::null(),
                    0,
                    std::ptr::null(),
                    1,
                    std::ptr::null_mut(),
                    0
                ),
                np,
                "NULL any_of_ids with a non-zero length"
            );
            mid_ecs_world_free(world);
        }
    }

    #[test]
    fn matching_static_any_of_matches_the_typed_or_filter() {
        let world_ptr = mid_ecs_test_filter_fixture_world_new();
        let h = static_id(world_ptr, "FfiHealthStatic");
        let a = static_id(world_ptr, "FfiFlagA");
        let b = static_id(world_ptr, "FfiFlagB");
        // SAFETY: `world_ptr` is a live handle for this whole test.
        let world = unsafe { &(*world_ptr).0 };

        // any_of {A, B}: e2 (A) and e3 (A and B), not e1 (neither).
        let via_ffi = entities_via_ffi(world_ptr, h, &[h], &[], &[a, b]);
        let via_typed = typed_entities::<Or<(With<MidEcsTestFlagA>, With<MidEcsTestFlagB>)>>(world);
        assert_eq!(via_ffi, via_typed);
        assert_eq!(via_ffi.len(), 2);

        // Empty any_of is "no constraint", matching the plain with/without
        // case exactly.
        assert_eq!(
            matching_ids_any(world_ptr, &[h], &[], &[]),
            matching_ids(world_ptr, &[h], &[])
        );

        // any_of naming only an id nothing has matches nothing, even
        // though with_ids alone would have matched.
        assert!(matching_ids_any(world_ptr, &[h], &[], &[MID_ECS_INVALID_ID]).is_empty());

        // SAFETY: freed exactly once.
        unsafe { mid_ecs_world_free(world_ptr) };
    }

    #[test]
    fn matching_static_bogus_and_contradictory_ids_are_empty_not_errors() {
        let world = mid_ecs_test_filter_fixture_world_new();
        let h = static_id(world, "FfiHealthStatic");

        assert!(matching_ids(world, &[MID_ECS_INVALID_ID], &[]).is_empty());
        assert!(matching_ids(world, &[h], &[h]).is_empty());
        assert_eq!(
            matching_ids(world, &[h], &[MID_ECS_INVALID_ID]),
            matching_ids(world, &[h], &[]),
            "an id nothing has registered excludes nothing"
        );

        // SAFETY: freed exactly once.
        unsafe { mid_ecs_world_free(world) };
    }

    // ── Resources through the C surface ─────────────────────────────

    fn resource_id(world: *const MidEcsWorld, name: &str) -> u32 {
        let name = std::ffi::CString::new(name).unwrap();
        // SAFETY: `world` is a live handle and `name` a valid C string.
        unsafe { mid_ecs_world_lookup_ffi_resource_id(world, name.as_ptr()) }
    }

    fn resource_span(world: *const MidEcsWorld, id: u32) -> (i32, FfiSpan) {
        let mut span = empty_span();
        // SAFETY: `world` is a live handle and `span` is valid.
        let status = unsafe { mid_ecs_world_resource_raw_span(world, id, &mut span) };
        (status, span)
    }

    fn write_time(world: *mut MidEcsWorld, id: u32, delta: f32, frame: u32) -> i32 {
        let value = MidEcsTestTime { delta, frame };
        let bytes = zerocopy::IntoBytes::as_bytes(&value);
        // SAFETY: `world` is a live handle; `bytes` is valid for its length.
        unsafe { mid_ecs_world_resource_write(world, id, bytes.as_ptr(), bytes.len()) }
    }

    #[test]
    fn resource_lookup_resolves_registered_names_and_rejects_the_rest() {
        let world = mid_ecs_test_resource_fixture_world_new();
        assert_ne!(resource_id(world, "FfiTime"), MID_ECS_INVALID_ID);
        assert_ne!(resource_id(world, "FfiGravity"), MID_ECS_INVALID_ID);
        assert_ne!(
            resource_id(world, "FfiTime"),
            resource_id(world, "FfiGravity")
        );
        assert_eq!(resource_id(world, "Nope"), MID_ECS_INVALID_ID);
        assert_eq!(resource_id(std::ptr::null(), "FfiTime"), MID_ECS_INVALID_ID);
        // SAFETY: freed exactly once; NULL name is the case under test.
        unsafe {
            assert_eq!(
                mid_ecs_world_lookup_ffi_resource_id(world, std::ptr::null()),
                MID_ECS_INVALID_ID
            );
            mid_ecs_world_free(world);
        }
    }

    #[test]
    fn resource_span_reads_the_inserted_value_and_is_empty_for_an_absent_one() {
        let world = mid_ecs_test_resource_fixture_world_new();
        let time = resource_id(world, "FfiTime");
        let gravity = resource_id(world, "FfiGravity");

        let (status, span) = resource_span(world, time);
        assert_eq!(status, MidEcsStatus::Ok as i32);
        assert_eq!(
            (span.count, span.stride),
            (1, std::mem::size_of::<MidEcsTestTime>())
        );
        // SAFETY: the span points at the live value; nothing has changed it.
        let seen = unsafe { &*(span.ptr as *const MidEcsTestTime) };
        assert_eq!((seen.delta, seen.frame), (0.016, 7));

        let (status, span) = resource_span(world, gravity);
        assert_eq!(status, MidEcsStatus::Ok as i32, "registered, not inserted");
        assert_eq!(span.count, 0);

        assert_eq!(
            resource_span(world, 999).0,
            MidEcsStatus::NotFound as i32,
            "an id that was never issued"
        );
        // SAFETY: freed exactly once.
        unsafe { mid_ecs_world_free(world) };
    }

    #[test]
    fn resource_write_updates_in_place_and_the_old_span_sees_it() {
        let world = mid_ecs_test_resource_fixture_world_new();
        let time = resource_id(world, "FfiTime");
        let (_, before) = resource_span(world, time);

        assert_eq!(write_time(world, time, 0.033, 8), MidEcsStatus::Ok as i32);

        let (_, after) = resource_span(world, time);
        assert_eq!(before.ptr, after.ptr, "written in place, same address");
        // SAFETY: `before.ptr` is the live value's address.
        let seen = unsafe { &*(before.ptr as *const MidEcsTestTime) };
        assert_eq!((seen.delta, seen.frame), (0.033, 8));
        // SAFETY: `world` is live; the typed API agrees.
        let typed = unsafe { (*world).0.get_resource::<MidEcsTestTime>().unwrap() };
        assert_eq!((typed.delta, typed.frame), (0.033, 8));

        // SAFETY: freed exactly once.
        unsafe { mid_ecs_world_free(world) };
    }

    #[test]
    fn resource_write_inserts_an_absent_resource() {
        let world = mid_ecs_test_resource_fixture_world_new();
        let gravity = resource_id(world, "FfiGravity");
        let bytes = 9.8f32.to_ne_bytes();
        // SAFETY: `world` is live; `bytes` is valid for 4 bytes.
        let status =
            unsafe { mid_ecs_world_resource_write(world, gravity, bytes.as_ptr(), bytes.len()) };
        assert_eq!(status, MidEcsStatus::Ok as i32);

        let (_, span) = resource_span(world, gravity);
        assert_eq!(span.count, 1);
        // SAFETY: `world` is live.
        let typed = unsafe { (*world).0.get_resource::<MidEcsTestGravity>().unwrap() };
        assert_eq!(typed.g, 9.8);
        // SAFETY: freed exactly once.
        unsafe { mid_ecs_world_free(world) };
    }

    #[test]
    fn resource_write_size_mismatch_changes_nothing() {
        let world = mid_ecs_test_resource_fixture_world_new();
        let time = resource_id(world, "FfiTime");
        let bytes = [0u8; 12];
        // SAFETY: `world` is live; `bytes` is valid for the lengths passed.
        unsafe {
            for len in [0usize, 4, 7, 9, 12] {
                assert_eq!(
                    mid_ecs_world_resource_write(world, time, bytes.as_ptr(), len),
                    MidEcsStatus::SizeMismatch as i32,
                    "len {len}"
                );
            }
            assert_eq!(MidEcsStatus::SizeMismatch as i32, -6);
            let typed = (*world).0.get_resource::<MidEcsTestTime>().unwrap();
            assert_eq!((typed.delta, typed.frame), (0.016, 7));
            mid_ecs_world_free(world);
        }
    }

    #[test]
    fn resource_remove_then_span_and_second_remove() {
        let world = mid_ecs_test_resource_fixture_world_new();
        let time = resource_id(world, "FfiTime");

        // SAFETY: `world` is live.
        unsafe {
            assert_eq!(
                mid_ecs_world_resource_remove(world, time),
                MidEcsStatus::Ok as i32
            );
            assert_eq!(resource_span(world, time).1.count, 0);
            assert!((*world).0.get_resource::<MidEcsTestTime>().is_none());
            assert_eq!(
                mid_ecs_world_resource_remove(world, time),
                MidEcsStatus::NotFound as i32,
                "already absent"
            );
            assert_eq!(
                mid_ecs_world_resource_remove(world, 999),
                MidEcsStatus::NotFound as i32,
                "never issued"
            );
            // Writing again brings it back.
            assert_eq!(write_time(world, time, 1.0, 1), MidEcsStatus::Ok as i32);
            assert_eq!(resource_span(world, time).1.count, 1);
            mid_ecs_world_free(world);
        }
    }

    #[test]
    fn resource_functions_null_and_unknown_id_cases() {
        let world = mid_ecs_test_resource_fixture_world_new();
        let time = resource_id(world, "FfiTime");
        let np = MidEcsStatus::NullPointer as i32;
        let byte = [0u8; 8];

        // SAFETY: every pointer is valid for what the callee may do with
        // it, or NULL where that's the case under test.
        unsafe {
            assert_eq!(resource_span(std::ptr::null(), time).0, np);
            assert_eq!(
                mid_ecs_world_resource_raw_span(world, time, std::ptr::null_mut()),
                np
            );
            assert_eq!(
                mid_ecs_world_resource_write(std::ptr::null_mut(), time, byte.as_ptr(), 8),
                np
            );
            assert_eq!(
                mid_ecs_world_resource_write(world, time, std::ptr::null(), 8),
                np,
                "NULL bytes with a non-zero length"
            );
            assert_eq!(
                mid_ecs_world_resource_remove(std::ptr::null_mut(), time),
                np
            );
            assert_eq!(
                mid_ecs_world_resource_write(world, 999, byte.as_ptr(), 8),
                MidEcsStatus::NotFound as i32
            );
            mid_ecs_world_free(world);
        }
    }
}
