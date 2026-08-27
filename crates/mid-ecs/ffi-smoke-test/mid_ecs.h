// crates/mid-ecs/ffi-smoke-test/mid_ecs.h
//
// Hand-written to match crates/mid-ecs/src/ffi.rs exactly -- not
// auto-generated (no cbindgen dependency added for this pass, matching
// mid-net's own ffi-smoke-test/mid_net.h convention). If ffi.rs's
// signatures change, this needs updating by hand alongside it. Verified
// against the real compiled library, not just written to match the Rust
// source by eye: see test.c, run against both libmid_ecs.so and
// libmid_ecs.a with real gcc.
//
// Scope: World lifecycle (spawn/despawn/is_alive/entity_count) plus
// component-data access -- raw spans and entity correlation for both
// storage systems (Sparse Shell and Archetype Core). `register_ffi_*`
// itself is NOT exposed here and never will be: it's generic over the
// Rust component type, which a C ABI fundamentally can't express. A
// real Rust-side setup step (registering each component type once at
// startup) has to happen before any of the lookup/span/entity-id
// functions below have anything to find -- see ffi.rs's own doc
// comment for the full reasoning.
#ifndef MID_ECS_H
#define MID_ECS_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// --- Status codes returned by functions below (see ffi.rs's MidEcsStatus) ---
#define MID_ECS_OK               0
#define MID_ECS_NULL_POINTER    -1
#define MID_ECS_NOT_ALIVE       -2
#define MID_ECS_INTERNAL_PANIC  -3
#define MID_ECS_NOT_FOUND       -4
#define MID_ECS_BUFFER_TOO_SMALL -5

// Sentinel component_id/archetype_id meaning "not found" -- returned by
// the lookup_ffi_* functions below. Not 0: 0 is a real, valid id for
// whichever type was registered first.
#define MID_ECS_INVALID_ID ((uint32_t)0xFFFFFFFFu)

// Opaque handle -- always heap-allocated by this library. Every handle
// returned by mid_ecs_world_new must be freed with mid_ecs_world_free
// exactly once.
typedef struct MidEcsWorld MidEcsWorld;

// A zero-copy view into this library's own live component storage --
// see mid_collections::FfiSpan's own doc comment (crates/mid-collections/
// src/ffi_span.rs) for the exact invalidation contract: valid only
// until the next mid_ecs_world_* call that mutates *this exact*
// component type on the same world. `stride` is sizeof one element;
// cast `ptr` to your own matching C struct type and index it as an
// array of `count` elements, `stride` bytes apart.
typedef struct MidEcsFfiSpan {
    const uint8_t *ptr;
    size_t stride;
    size_t count;
} MidEcsFfiSpan;

// Creates a new, empty World. Never returns NULL.
MidEcsWorld *mid_ecs_world_new(void);

// Frees a handle returned by mid_ecs_world_new. NULL is a safe no-op.
void mid_ecs_world_free(MidEcsWorld *world);

// Spawns a new, live entity, packed as a single uint64_t -- see ffi.rs's
// doc comment for the packing scheme (index in the low 32 bits,
// generation in the high 32 bits). Treat the return value as opaque:
// pass it back to despawn/is_alive, don't parse the two halves yourself.
// Returns 0 on a null world -- not a valid "empty" sentinel to check
// against on its own, check world for NULL yourself if that matters.
uint64_t mid_ecs_world_spawn(MidEcsWorld *world);

// Despawns the entity packed in `entity`. Returns MID_ECS_OK if it was
// actually alive, MID_ECS_NOT_ALIVE otherwise (never spawned, already
// despawned, or a stale handle whose slot was reused) -- never a crash
// either way, even for a bogus, never-issued entity value.
int32_t mid_ecs_world_despawn(MidEcsWorld *world, uint64_t entity);

// Whether the entity packed in `entity` is currently alive. false on a
// null world or a bogus/stale entity value -- never a crash.
bool mid_ecs_world_is_alive(const MidEcsWorld *world, uint64_t entity);

// Number of currently-live entities. 0 on a null world.
size_t mid_ecs_world_entity_count(const MidEcsWorld *world);

// --- Sparse Shell component-data surface (volatile/toggle components) ---

// Looks up the component_id a Rust type was registered under via the
// Rust-side World::register_ffi_component, by the name given then.
// Returns MID_ECS_INVALID_ID on a null world/name, invalid UTF-8, or a
// name that was never registered -- these four cases aren't
// distinguishable from the return value alone.
uint32_t mid_ecs_world_lookup_ffi_component_id(const MidEcsWorld *world, const char *name);

// Raw span over every currently-attached instance of component_id,
// written into *out_span. Returns MID_ECS_OK (*out_span written) or
// MID_ECS_NOT_FOUND (component_id doesn't exist, or was never
// registered for FFI exposure).
int32_t mid_ecs_world_component_raw_span(const MidEcsWorld *world, uint32_t component_id, MidEcsFfiSpan *out_span);

// Entity-correlation counterpart to mid_ecs_world_component_raw_span:
// entity_ids[i] (unpack with the same generation-aware scheme as
// mid_ecs_world_spawn's return value) is the entity that owns that same
// raw-span call's element i, for every valid i.
//
// Pass out_buf = NULL to query the real element count from the return
// value alone, without writing anything, then call again with a buffer
// sized to hold at least that many uint64_t elements. out_buf_capacity
// counts elements, not bytes. Returns the real count (>= 0) on success,
// or a negative status: MID_ECS_NULL_POINTER, MID_ECS_NOT_FOUND, or
// MID_ECS_BUFFER_TOO_SMALL (a non-null buffer too small for the real
// count -- query first, don't try to use a partial fill).
int32_t mid_ecs_world_component_entity_ids(const MidEcsWorld *world, uint32_t component_id, uint64_t *out_buf, size_t out_buf_capacity);

// --- Archetype Core component-data surface (stable/structural components) ---

// Same shape as mid_ecs_world_lookup_ffi_component_id, but for the
// Archetype Core system -- a SEPARATE id namespace from the Sparse
// Shell lookup above; the same name can resolve to a different numeric
// id in each system.
uint32_t mid_ecs_world_lookup_ffi_static_component_id(const MidEcsWorld *world, const char *name);

// Raw span over component_id's column within archetype_id's table,
// written into *out_span. Returns MID_ECS_NOT_FOUND if component_id was
// never registered, archetype_id doesn't exist, or archetype_id's
// signature simply doesn't include component_id (a real, permanent
// structural fact -- an archetype whose signature DOES include it but
// currently has zero rows returns MID_ECS_OK with out_span->count == 0,
// not MID_ECS_NOT_FOUND). Pair with
// mid_ecs_world_archetypes_with_static_component to enumerate every
// archetype currently containing a given component.
int32_t mid_ecs_world_static_component_raw_span(const MidEcsWorld *world, uint32_t archetype_id, uint32_t component_id, MidEcsFfiSpan *out_span);

// Entity-correlation counterpart to mid_ecs_world_static_component_raw_span,
// for the same (archetype_id, component_id) pair. Same NULL-buffer-
// queries-count idiom as mid_ecs_world_component_entity_ids.
int32_t mid_ecs_world_static_component_entity_ids(const MidEcsWorld *world, uint32_t archetype_id, uint32_t component_id, uint64_t *out_buf, size_t out_buf_capacity);

// Enumerates every currently-existing archetype whose signature
// includes component_id, writing each as a plain uint32_t -- pass one
// straight into mid_ecs_world_static_component_raw_span/
// _entity_ids as archetype_id. Same NULL-buffer-queries-count idiom.
// NOT gated by FFI registration: an unregistered component_id just
// naturally matches zero archetypes (a real, valid answer, not an
// error) -- this never returns MID_ECS_NOT_FOUND.
int32_t mid_ecs_world_archetypes_with_static_component(const MidEcsWorld *world, uint32_t component_id, uint32_t *out_buf, size_t out_buf_capacity);

// --- Test fixture (see ffi.rs's own doc comment on this function) ---

// NOT a real part of this library's intended public API -- exists only
// so a pure C program (which has no way to call the generic
// register_ffi_*/insert*/insert_static Rust functions itself) can
// exercise the real component-data round trip below. Returns a fresh
// World with a component type registered in each storage system
// ("FfiHealth" in Sparse Shell, "FfiHealthStatic" in Archetype Core --
// two distinct Rust types on that side, since one type can't be used
// with both storage systems in the same World, but both are
// `{ uint32_t hp; }` and indistinguishable from here), two entities
// spawned, and known values already inserted (hp 10/20 in Sparse
// Shell, hp 100/200 in Archetype Core) -- exactly the fixed values
// test.c checks against. Never returns NULL.
MidEcsWorld *mid_ecs_test_fixture_world_new(void);

#ifdef __cplusplus
}
#endif

#endif // MID_ECS_H
