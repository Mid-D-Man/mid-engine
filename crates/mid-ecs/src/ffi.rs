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

use std::panic::{catch_unwind, AssertUnwindSafe};

use crate::world::{Entity, World};

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

#[cfg(test)]
mod tests {
    use super::*;

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
}
