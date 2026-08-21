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
// Scope for this pass: World lifecycle only -- spawn/despawn/is_alive/
// entity_count. Component data access isn't exposed yet; see ffi.rs's
// own doc comment for why that's a separate, harder piece.
#ifndef MID_ECS_H
#define MID_ECS_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// --- Status codes returned by functions below (see ffi.rs's MidEcsStatus) ---
#define MID_ECS_OK             0
#define MID_ECS_NULL_POINTER  -1
#define MID_ECS_NOT_ALIVE     -2
#define MID_ECS_INTERNAL_PANIC -3

// Opaque handle -- always heap-allocated by this library. Every handle
// returned by mid_ecs_world_new must be freed with mid_ecs_world_free
// exactly once.
typedef struct MidEcsWorld MidEcsWorld;

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

#ifdef __cplusplus
}
#endif

#endif // MID_ECS_H
