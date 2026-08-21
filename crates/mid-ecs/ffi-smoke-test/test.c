// Real end-to-end FFI verification -- not a Rust unit test, an actual C
// program compiled with gcc and linked against libmid_ecs.so.
#include <stdio.h>
#include <stdlib.h>
#include "mid_ecs.h"

static int failures = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { printf("FAIL: %s (line %d)\n", msg, __LINE__); failures++; } \
    else { printf("ok:   %s\n", msg); } \
} while (0)

int main(void) {
    printf("=== mid-ecs FFI: real C program, real gcc, real link against libmid_ecs.so ===\n\n");

    // --- World lifecycle ---
    MidEcsWorld *world = mid_ecs_world_new();
    CHECK(world != NULL, "world_new returns non-null handle");
    CHECK(mid_ecs_world_entity_count(world) == 0, "fresh world has zero entities");

    // --- spawn / is_alive / entity_count ---
    uint64_t e1 = mid_ecs_world_spawn(world);
    CHECK(mid_ecs_world_is_alive(world, e1), "freshly spawned entity is alive");
    CHECK(mid_ecs_world_entity_count(world) == 1, "entity_count is 1 after one spawn");

    uint64_t e2 = mid_ecs_world_spawn(world);
    CHECK(e1 != e2, "two spawns return distinct packed entity values");
    CHECK(mid_ecs_world_entity_count(world) == 2, "entity_count is 2 after two spawns");

    // --- despawn ---
    int32_t status = mid_ecs_world_despawn(world, e1);
    CHECK(status == MID_ECS_OK, "despawn of a live entity returns MID_ECS_OK");
    CHECK(!mid_ecs_world_is_alive(world, e1), "despawned entity is no longer alive");
    CHECK(mid_ecs_world_entity_count(world) == 1, "entity_count drops to 1 after despawn");
    CHECK(mid_ecs_world_is_alive(world, e2), "e2 is untouched by e1's despawn");

    // --- despawn again: safe no-op, not a crash ---
    int32_t second = mid_ecs_world_despawn(world, e1);
    CHECK(second == MID_ECS_NOT_ALIVE, "despawning an already-dead entity returns MID_ECS_NOT_ALIVE, not OK");

    // --- generational safety through the real packed u64, through real C memory ---
    uint64_t e3 = mid_ecs_world_spawn(world); // reuses e1's freed slot, real allocator state
    CHECK(!mid_ecs_world_is_alive(world, e1), "the stale packed e1 value must not read as alive after slot reuse");
    CHECK(mid_ecs_world_is_alive(world, e3), "e3 (the entity that reused the slot) is alive");

    // --- bogus packed value: safe, never a crash ---
    uint64_t bogus = 0xFFFFFFFFFFFFFFFFULL;
    CHECK(!mid_ecs_world_is_alive(world, bogus), "a bogus packed u64 reads as not alive, not a crash");
    int32_t bogus_despawn = mid_ecs_world_despawn(world, bogus);
    CHECK(bogus_despawn == MID_ECS_NOT_ALIVE, "despawning a bogus packed u64 returns MID_ECS_NOT_ALIVE, not a crash");

    // --- NULL world: every function has a documented safe path ---
    CHECK(mid_ecs_world_spawn(NULL) == 0, "spawn on NULL world returns 0, not a crash");
    CHECK(!mid_ecs_world_is_alive(NULL, e2), "is_alive on NULL world returns false, not a crash");
    CHECK(mid_ecs_world_entity_count(NULL) == 0, "entity_count on NULL world returns 0, not a crash");
    CHECK(mid_ecs_world_despawn(NULL, e2) == MID_ECS_NULL_POINTER, "despawn on NULL world returns MID_ECS_NULL_POINTER");

    mid_ecs_world_free(world);
    mid_ecs_world_free(NULL); // documented safe no-op

    printf("\n=== %d check(s) failed ===\n", failures);
    return failures == 0 ? 0 : 1;
}
