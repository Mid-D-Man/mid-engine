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

    // --- Filtered archetype enumeration (with/without id lists) ---
    MidEcsWorld *fw = mid_ecs_test_filter_fixture_world_new();
    uint32_t h = mid_ecs_world_lookup_ffi_static_component_id(fw, "FfiHealthStatic");
    uint32_t fa = mid_ecs_world_lookup_ffi_static_component_id(fw, "FfiFlagA");
    uint32_t fb = mid_ecs_world_lookup_ffi_static_component_id(fw, "FfiFlagB");
    CHECK(h != MID_ECS_INVALID_ID && fa != MID_ECS_INVALID_ID && fb != MID_ECS_INVALID_ID,
          "the filter fixture's three component names all resolve");

    // Sum Health.hp and count rows across every archetype matching the
    // filter, exactly as a C caller would. Every enumerated archetype must
    // resolve through raw_span with MID_ECS_OK, including the zero-row
    // intermediates.
    uint32_t sum_hp = 0, rows = 0, archetypes_seen = 0;
    bool all_ok = true;
#define WALK(with_arr, nwith, without_arr, nwithout, anyof_arr, nanyof) do { \
        sum_hp = 0; rows = 0; archetypes_seen = 0; all_ok = true; \
        uint32_t ids[16]; \
        int32_t n = mid_ecs_world_archetypes_matching_static(fw, (with_arr), (nwith), (without_arr), (nwithout), (anyof_arr), (nanyof), ids, 16); \
        if (n < 0) { all_ok = false; n = 0; } \
        archetypes_seen = (uint32_t)n; \
        for (int32_t i = 0; i < n; i++) { \
            MidEcsFfiSpan sp; \
            if (mid_ecs_world_static_component_raw_span(fw, ids[i], h, &sp) != MID_ECS_OK) { all_ok = false; continue; } \
            const FfiHealthC *hp = (const FfiHealthC *)sp.ptr; \
            for (size_t r = 0; r < sp.count; r++) { sum_hp += hp[r].hp; rows++; } \
        } \
    } while (0)

    uint32_t w_h[] = { h };
    uint32_t w_ha[] = { h, fa };
    uint32_t wo_a[] = { fa };
    uint32_t wo_b[] = { fb };
    uint32_t wo_ab[] = { fa, fb };
    uint32_t ao_ab[] = { fa, fb };

    WALK(w_h, 1, NULL, 0, NULL, 0);
    CHECK(all_ok && archetypes_seen == 4 && rows == 3 && sum_hp == 6,
          "with {Health}: 4 archetypes (one zero-row), 3 rows, hp 1+2+3, every raw_span OK");
    WALK(w_h, 1, wo_a, 1, NULL, 0);
    CHECK(all_ok && archetypes_seen == 2 && rows == 1 && sum_hp == 1,
          "with {Health} without {FlagA}: only e1's row, plus the zero-row {FlagB, Health}");
    WALK(w_ha, 2, NULL, 0, NULL, 0);
    CHECK(all_ok && archetypes_seen == 2 && rows == 2 && sum_hp == 5,
          "with {Health, FlagA}: e2 and e3, hp 2+3");
    WALK(w_ha, 2, wo_b, 1, NULL, 0);
    CHECK(all_ok && archetypes_seen == 1 && rows == 1 && sum_hp == 2,
          "with {Health, FlagA} without {FlagB}: e2 only");
    WALK(w_h, 1, wo_ab, 2, NULL, 0);
    CHECK(all_ok && archetypes_seen == 1 && rows == 1 && sum_hp == 1,
          "with {Health} without {FlagA, FlagB}: e1 only");
    // any_of {FlagA, FlagB}: e2 (FlagA) and e3 (FlagA and FlagB), not e1.
    WALK(w_h, 1, NULL, 0, ao_ab, 2);
    CHECK(all_ok && archetypes_seen == 3 && rows == 2 && sum_hp == 5,
          "with {Health} any_of {FlagA, FlagB}: 3 archetypes (the zero-row"
          " {FlagB, Health} has FlagB in its signature too), e2 and e3's rows, hp 2+3");
    // Empty any_of is "no constraint", identical to the plain with/without
    // case above.
    WALK(w_h, 1, NULL, 0, NULL, 0);
    int32_t no_any_of_seen = (int32_t)archetypes_seen;
    WALK(w_h, 1, NULL, 0, ao_ab, 0);
    CHECK(all_ok && (int32_t)archetypes_seen == no_any_of_seen,
          "an empty any_of list (non-NULL pointer, zero length) is still 'no constraint'");

    CHECK(mid_ecs_world_archetypes_matching_static(fw, NULL, 0, NULL, 0, NULL, 0, NULL, 0) == 6,
          "three empty lists (NULL, 0) match every archetype, the empty one included");
    CHECK(mid_ecs_world_archetypes_matching_static(fw, NULL, 1, NULL, 0, NULL, 0, NULL, 0) == MID_ECS_NULL_POINTER,
          "a NULL id list with a non-zero length is MID_ECS_NULL_POINTER");
    CHECK(mid_ecs_world_archetypes_matching_static(NULL, w_h, 1, NULL, 0, NULL, 0, NULL, 0) == MID_ECS_NULL_POINTER,
          "a NULL world is MID_ECS_NULL_POINTER");
    CHECK(mid_ecs_world_archetypes_matching_static(fw, w_h, 1, NULL, 0, NULL, 1, NULL, 0) == MID_ECS_NULL_POINTER,
          "a NULL any_of_ids with a non-zero length is MID_ECS_NULL_POINTER");
    uint32_t bogus_ids[] = { MID_ECS_INVALID_ID };
    CHECK(mid_ecs_world_archetypes_matching_static(fw, bogus_ids, 1, NULL, 0, NULL, 0, NULL, 0) == 0,
          "a never-registered id in with_ids matches nothing, and is not an error");
    CHECK(mid_ecs_world_archetypes_matching_static(fw, w_h, 1, NULL, 0, bogus_ids, 1, NULL, 0) == 0,
          "any_of naming only a never-registered id matches nothing, even though with_ids alone would");
    uint32_t small_buf[1];
    CHECK(mid_ecs_world_archetypes_matching_static(fw, w_h, 1, NULL, 0, NULL, 0, small_buf, 1) == MID_ECS_BUFFER_TOO_SMALL,
          "a too-small buffer is MID_ECS_BUFFER_TOO_SMALL, not a partial fill");
    mid_ecs_world_free(fw);

    // --- Resources: lookup / read / write / remove ---
    typedef struct { float delta; uint32_t frame; } FfiTimeC;
    MidEcsWorld *rw = mid_ecs_test_resource_fixture_world_new();
    uint32_t time_id = mid_ecs_world_lookup_ffi_resource_id(rw, "FfiTime");
    uint32_t gravity_id = mid_ecs_world_lookup_ffi_resource_id(rw, "FfiGravity");
    CHECK(time_id != MID_ECS_INVALID_ID && gravity_id != MID_ECS_INVALID_ID && time_id != gravity_id,
          "both resource names resolve to distinct resource ids");
    CHECK(mid_ecs_world_lookup_ffi_resource_id(rw, "Nope") == MID_ECS_INVALID_ID,
          "an unregistered resource name is MID_ECS_INVALID_ID");

    MidEcsFfiSpan time_span;
    CHECK(mid_ecs_world_resource_raw_span(rw, time_id, &time_span) == MID_ECS_OK, "resource_raw_span on FfiTime is MID_ECS_OK");
    CHECK(time_span.count == 1 && time_span.stride == sizeof(FfiTimeC), "the span is one element of the registered size");
    const FfiTimeC *time_view = (const FfiTimeC *)time_span.ptr;
    CHECK(time_view->delta == 0.016f && time_view->frame == 7, "the fixture's inserted value reads back through C memory");

    MidEcsFfiSpan gravity_span;
    CHECK(mid_ecs_world_resource_raw_span(rw, gravity_id, &gravity_span) == MID_ECS_OK && gravity_span.count == 0,
          "a registered but not inserted resource is MID_ECS_OK with count 0");
    CHECK(mid_ecs_world_resource_raw_span(rw, 999, &gravity_span) == MID_ECS_NOT_FOUND,
          "a resource_id that was never issued is MID_ECS_NOT_FOUND");

    FfiTimeC next_time = { 0.033f, 8 };
    CHECK(mid_ecs_world_resource_write(rw, time_id, (const uint8_t *)&next_time, sizeof next_time) == MID_ECS_OK,
          "resource_write of a correctly sized value is MID_ECS_OK");
    MidEcsFfiSpan time_span_after;
    mid_ecs_world_resource_raw_span(rw, time_id, &time_span_after);
    CHECK(time_span_after.ptr == time_span.ptr, "the write updated the value in place (same address)");
    CHECK(time_view->delta == 0.033f && time_view->frame == 8, "the span taken before the write sees the new value");

    uint8_t too_short[4] = { 0 };
    CHECK(mid_ecs_world_resource_write(rw, time_id, too_short, sizeof too_short) == MID_ECS_SIZE_MISMATCH,
          "a wrong-sized write is MID_ECS_SIZE_MISMATCH");
    CHECK(time_view->delta == 0.033f && time_view->frame == 8, "and it changed nothing");

    float g = 9.8f;
    CHECK(mid_ecs_world_resource_write(rw, gravity_id, (const uint8_t *)&g, sizeof g) == MID_ECS_OK,
          "writing a registered but not inserted resource inserts it");
    mid_ecs_world_resource_raw_span(rw, gravity_id, &gravity_span);
    CHECK(gravity_span.count == 1 && *(const float *)gravity_span.ptr == 9.8f, "the inserted resource reads back");

    CHECK(mid_ecs_world_resource_write(rw, 999, (const uint8_t *)&g, sizeof g) == MID_ECS_NOT_FOUND,
          "writing an id that was never issued is MID_ECS_NOT_FOUND");
    CHECK(mid_ecs_world_resource_write(rw, time_id, NULL, sizeof next_time) == MID_ECS_NULL_POINTER,
          "NULL bytes with a non-zero length is MID_ECS_NULL_POINTER");
    CHECK(mid_ecs_world_resource_write(NULL, time_id, (const uint8_t *)&next_time, sizeof next_time) == MID_ECS_NULL_POINTER,
          "a NULL world is MID_ECS_NULL_POINTER");

    CHECK(mid_ecs_world_resource_remove(rw, time_id) == MID_ECS_OK, "resource_remove of an inserted resource is MID_ECS_OK");
    mid_ecs_world_resource_raw_span(rw, time_id, &time_span_after);
    CHECK(time_span_after.count == 0, "after remove the span is empty");
    CHECK(mid_ecs_world_resource_remove(rw, time_id) == MID_ECS_NOT_FOUND, "removing it again is MID_ECS_NOT_FOUND");
    mid_ecs_world_free(rw);

    printf("\n=== %d check(s) failed ===\n", failures);
    return failures == 0 ? 0 : 1;
}
