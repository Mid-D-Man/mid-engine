// Real end-to-end FFI verification -- not a Rust unit test, an actual C
// program compiled with gcc and linked against libmid_ecs.so.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mid_ecs.h"

// Mirrors FfiHealth (mid-ecs/src/ffi.rs's own test type) exactly --
// #[repr(C)] { hp: u32 } on the Rust side. This is what a real C
// caller has to do: hand-write a matching struct for each component
// type it wants to read, the same way this whole file hand-writes
// mid_ecs.h to match ffi.rs.
typedef struct {
    uint32_t hp;
} FfiHealthC;

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

    // --- lookup_ffi_component_id: real component_id has to come from
    // this C program's own name lookup, since register_ffi_* itself
    // isn't exposed to C at all (it's generic) ---
    uint32_t never_registered = mid_ecs_world_lookup_ffi_component_id(world, "NeverRegistered");
    CHECK(never_registered == MID_ECS_INVALID_ID, "looking up an unregistered name returns MID_ECS_INVALID_ID");
    CHECK(mid_ecs_world_lookup_ffi_component_id(NULL, "Health") == MID_ECS_INVALID_ID, "lookup on NULL world returns MID_ECS_INVALID_ID");
    CHECK(mid_ecs_world_lookup_ffi_component_id(world, NULL) == MID_ECS_INVALID_ID, "lookup with NULL name returns MID_ECS_INVALID_ID");

    mid_ecs_world_free(world);
    mid_ecs_world_free(NULL); // documented safe no-op

    // === Component-data round trip, through a real, separately-compiled
    // === C program reading real memory Rust laid out -- not a Rust unit
    // === test calling the same extern "C" fn from the same binary.
    MidEcsWorld *fixture = mid_ecs_test_fixture_world_new();
    CHECK(fixture != NULL, "test_fixture_world_new returns non-null handle");

    // --- Sparse Shell: lookup -> raw_span -> entity_ids ---
    uint32_t health_id = mid_ecs_world_lookup_ffi_component_id(fixture, "FfiHealth");
    CHECK(health_id != MID_ECS_INVALID_ID, "FfiHealth resolves to a real component_id");

    MidEcsFfiSpan span;
    int32_t span_status = mid_ecs_world_component_raw_span(fixture, health_id, &span);
    CHECK(span_status == MID_ECS_OK, "component_raw_span on a real registered id returns MID_ECS_OK");
    CHECK(span.count == 2, "component_raw_span sees both fixture entities");
    CHECK(span.stride == sizeof(FfiHealthC), "span.stride matches the real C struct size");
    const FfiHealthC *health = (const FfiHealthC *)span.ptr;
    CHECK(health[0].hp == 10 && health[1].hp == 20, "raw component bytes match the fixture's real Rust-side values, read through a real C struct");

    // Query mode first (NULL buffer), matching the documented idiom.
    int32_t queried_count = mid_ecs_world_component_entity_ids(fixture, health_id, NULL, 0);
    CHECK(queried_count == 2, "component_entity_ids query mode reports the real count");

    uint64_t entity_ids[2];
    int32_t written = mid_ecs_world_component_entity_ids(fixture, health_id, entity_ids, 2);
    CHECK(written == 2, "component_entity_ids fills exactly 2 real entity ids");
    CHECK(mid_ecs_world_is_alive(fixture, entity_ids[0]) && mid_ecs_world_is_alive(fixture, entity_ids[1]),
          "the packed entity ids read back from C are real, live entities in this world");
    CHECK(entity_ids[0] != entity_ids[1], "the two correlated entity ids are distinct");

    uint64_t too_small_buf[1];
    int32_t too_small = mid_ecs_world_component_entity_ids(fixture, health_id, too_small_buf, 1);
    CHECK(too_small == MID_ECS_BUFFER_TOO_SMALL, "a too-small non-null buffer returns MID_ECS_BUFFER_TOO_SMALL, not a partial fill");

    // --- Archetype Core: lookup -> archetypes_with -> raw_span -> entity_ids ---
    uint32_t static_health_id = mid_ecs_world_lookup_ffi_static_component_id(fixture, "FfiHealthStatic");
    CHECK(static_health_id != MID_ECS_INVALID_ID, "FfiHealthStatic resolves to a real component_id");

    int32_t archetype_count = mid_ecs_world_archetypes_with_static_component(fixture, static_health_id, NULL, 0);
    CHECK(archetype_count == 1, "both fixture entities share exactly one archetype");

    uint32_t archetype_ids[1];
    int32_t archetypes_written = mid_ecs_world_archetypes_with_static_component(fixture, static_health_id, archetype_ids, 1);
    CHECK(archetypes_written == 1, "archetypes_with_static_component fills exactly 1 real archetype id");

    MidEcsFfiSpan static_span;
    int32_t static_span_status = mid_ecs_world_static_component_raw_span(fixture, archetype_ids[0], static_health_id, &static_span);
    CHECK(static_span_status == MID_ECS_OK, "static_component_raw_span on a real (archetype_id, component_id) pair returns MID_ECS_OK");
    CHECK(static_span.count == 2, "static_component_raw_span sees both fixture entities");
    const FfiHealthC *static_health = (const FfiHealthC *)static_span.ptr;
    CHECK(static_health[0].hp == 100 && static_health[1].hp == 200, "Archetype Core raw bytes match the fixture's real Rust-side values");

    uint64_t static_entity_ids[2];
    int32_t static_written = mid_ecs_world_static_component_entity_ids(fixture, archetype_ids[0], static_health_id, static_entity_ids, 2);
    CHECK(static_written == 2, "static_component_entity_ids fills exactly 2 real entity ids");
    CHECK(mid_ecs_world_is_alive(fixture, static_entity_ids[0]) && mid_ecs_world_is_alive(fixture, static_entity_ids[1]),
          "the Archetype Core correlated entity ids are real, live entities in this world");

    // --- NotFound paths, on the real populated fixture, not just an empty world ---
    MidEcsFfiSpan bogus_span;
    CHECK(mid_ecs_world_component_raw_span(fixture, MID_ECS_INVALID_ID, &bogus_span) == MID_ECS_NOT_FOUND,
          "raw_span on a never-registered id is MID_ECS_NOT_FOUND, even on a populated world");
    CHECK(mid_ecs_world_static_component_raw_span(fixture, archetype_ids[0], MID_ECS_INVALID_ID, &bogus_span) == MID_ECS_NOT_FOUND,
          "static raw_span with a never-registered component_id is MID_ECS_NOT_FOUND, even for a real archetype_id");

    mid_ecs_world_free(fixture);

    printf("\n=== %d check(s) failed ===\n", failures);
    return failures == 0 ? 0 : 1;
}
