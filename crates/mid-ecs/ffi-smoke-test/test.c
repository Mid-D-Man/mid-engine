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

// Mirrors the filter fixture's FfiFlagA / FfiFlagB: #[repr(C)] { v: u32 }.
typedef struct {
    uint32_t v;
} FfiFlagC;

static int failures = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { printf("FAIL: %s (line %d)\n", msg, __LINE__); failures++; } \
    else { printf("ok:   %s\n", msg); } \
} while (0)

// The first 4 bytes (a uint32_t) of `component`'s value for `entity`, found
// the way any C caller finds it: through the archetypes that hold the
// component. Returns 1 and sets *out if the entity has it, else 0.
static int u32_of(const MidEcsWorld *world, uint32_t component, uint64_t entity, uint32_t *out) {
    uint32_t archetypes[8];
    int32_t n = mid_ecs_world_archetypes_matching_static(world, &component, 1, NULL, 0, NULL, 0, archetypes, 8);
    for (int32_t i = 0; i < n; i++) {
        uint64_t ids[8];
        MidEcsFfiSpan span;
        int32_t count = mid_ecs_world_static_component_entity_ids(world, archetypes[i], component, ids, 8);
        if (count < 0 || mid_ecs_world_static_component_raw_span(world, archetypes[i], component, &span) != MID_ECS_OK) {
            continue;
        }
        for (int32_t r = 0; r < count; r++) {
            if (ids[r] == entity) {
                *out = *(const uint32_t *)((const uint8_t *)span.ptr + (size_t)r * span.stride);
                return 1;
            }
        }
    }
    return 0;
}

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

    // --- Three components plus optional terms, from the same calls ---
    // Rust's World::query_data::<(Entity, &Health, &FlagA, &FlagB)> is
    // `with = {Health, FlagA, FlagB}` here. Its Option<&FlagA> and
    // Has<FlagB> terms are a per-archetype status check: raw_span is
    // MID_ECS_OK when the archetype's signature holds the component and
    // MID_ECS_NOT_FOUND when it does not. There is no dedicated C function
    // for this: an item tuple cannot cross a C ABI, and these calls already
    // carry everything it needs.
    {
        uint32_t w_hab[] = { h, fa, fb };
        uint32_t ids3[16];
        int32_t n3 = mid_ecs_world_archetypes_matching_static(fw, w_hab, 3, NULL, 0, NULL, 0, ids3, 16);
        uint32_t rows3 = 0, sum_h3 = 0, sum_fa3 = 0, sum_fb3 = 0;
        bool ok3 = (n3 == 1);
        for (int32_t i = 0; i < n3; i++) {
            MidEcsFfiSpan hs, as, bs;
            if (mid_ecs_world_static_component_raw_span(fw, ids3[i], h, &hs) != MID_ECS_OK ||
                mid_ecs_world_static_component_raw_span(fw, ids3[i], fa, &as) != MID_ECS_OK ||
                mid_ecs_world_static_component_raw_span(fw, ids3[i], fb, &bs) != MID_ECS_OK) { ok3 = false; continue; }
            if (hs.count != as.count || hs.count != bs.count) ok3 = false;
            for (size_t r = 0; r < hs.count; r++) {
                sum_h3 += ((const FfiHealthC *)hs.ptr)[r].hp;
                sum_fa3 += ((const FfiFlagC *)as.ptr)[r].v;
                sum_fb3 += ((const FfiFlagC *)bs.ptr)[r].v;
                rows3++;
            }
        }
        CHECK(ok3 && rows3 == 1 && sum_h3 == 3 && sum_fa3 == 30 && sum_fb3 == 300,
              "three-component query, with {Health, FlagA, FlagB}: one archetype, e3's row, hp 3 / FlagA 30 / FlagB 300 in step");

        uint32_t ids_o[16];
        int32_t no = mid_ecs_world_archetypes_matching_static(fw, w_h, 1, NULL, 0, NULL, 0, ids_o, 16);
        uint32_t total_o = 0, some_a = 0, none_a = 0, sum_a = 0, has_b = 0;
        bool ok_o = (no == 4);
        for (int32_t i = 0; i < no; i++) {
            MidEcsFfiSpan hs, as, bs;
            if (mid_ecs_world_static_component_raw_span(fw, ids_o[i], h, &hs) != MID_ECS_OK) { ok_o = false; continue; }
            int32_t a_status = mid_ecs_world_static_component_raw_span(fw, ids_o[i], fa, &as);
            int32_t b_status = mid_ecs_world_static_component_raw_span(fw, ids_o[i], fb, &bs);
            if ((a_status != MID_ECS_OK && a_status != MID_ECS_NOT_FOUND) ||
                (b_status != MID_ECS_OK && b_status != MID_ECS_NOT_FOUND)) { ok_o = false; continue; }
            for (size_t r = 0; r < hs.count; r++) {
                total_o++;
                if (a_status == MID_ECS_OK) { some_a++; sum_a += ((const FfiFlagC *)as.ptr)[r].v; }
                else { none_a++; }
                if (b_status == MID_ECS_OK) { has_b++; }
            }
        }
        CHECK(ok_o && total_o == 3 && some_a == 2 && none_a == 1 && sum_a == 50 && has_b == 1,
              "Option<FlagA> / Has<FlagB> over {Health}: raw_span OK = present, NOT_FOUND = absent; 2 rows with FlagA (20+30), 1 without, 1 with FlagB");
    }

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

    // --- Change detection: added / changed rows ---
    MidEcsWorld *cw = mid_ecs_test_change_fixture_world_new();
    uint32_t ch_id = mid_ecs_world_lookup_ffi_static_component_id(cw, "FfiHealthStatic");
    CHECK(ch_id != MID_ECS_INVALID_ID, "change fixture: FfiHealthStatic resolves");
    uint32_t ch_arch[8];
    int32_t ch_arch_n = mid_ecs_world_archetypes_matching_static(cw, &ch_id, 1, NULL, 0, NULL, 0, ch_arch, 8);
    // The fixture's entities were spawned into the empty archetype and
    // migrated, so exactly one archetype holds Health and it is populated.
    CHECK(ch_arch_n == 1, "change fixture: exactly one archetype holds Health");
    uint32_t ch_a = ch_arch[0];

    CHECK(mid_ecs_world_change_tick(cw) == 2, "fixture leaves the change tick at 2");
    CHECK(mid_ecs_world_change_tick(NULL) == 0, "change_tick on a NULL world is 0");
    CHECK(mid_ecs_world_increment_change_tick(NULL) == 0, "increment_change_tick on a NULL world is 0");

    uint32_t ch_rows[8];
    // A fresh tracker (last_run 0) sees all three rows as added and changed.
    CHECK(mid_ecs_world_static_component_added_rows(cw, ch_a, ch_id, 0, NULL, 0) == 3,
          "NULL buffer queries the added-row count: 3 for a fresh tracker");
    CHECK(mid_ecs_world_static_component_added_rows(cw, ch_a, ch_id, 0, ch_rows, 8) == 3
              && ch_rows[0] == 0 && ch_rows[1] == 1 && ch_rows[2] == 2,
          "fresh tracker: added rows are 0, 1, 2 in ascending order");
    CHECK(mid_ecs_world_static_component_changed_rows(cw, ch_a, ch_id, 0, ch_rows, 8) == 3,
          "fresh tracker: all three rows are also changed");
    CHECK(mid_ecs_world_static_component_added_rows(cw, ch_a, ch_id, 0, ch_rows, 2) == MID_ECS_BUFFER_TOO_SMALL,
          "a too-small buffer is MID_ECS_BUFFER_TOO_SMALL, not a partial fill");

    // Having read the tick, the tracker sees nothing.
    uint32_t last_run = mid_ecs_world_change_tick(cw);
    CHECK(mid_ecs_world_static_component_added_rows(cw, ch_a, ch_id, last_run, NULL, 0) == 0
              && mid_ecs_world_static_component_changed_rows(cw, ch_a, ch_id, last_run, NULL, 0) == 0,
          "after last_run = change_tick, nothing is added or changed");

    // One frame of Rust-side game code: hp 2 -> 22, plus a new hp 4 entity.
    CHECK(mid_ecs_world_increment_change_tick(cw) == 3, "increment_change_tick returns the new tick, 3");
    CHECK(mid_ecs_test_change_fixture_mutate(cw) == MID_ECS_OK, "the fixture's scripted mutation runs");
    CHECK(mid_ecs_world_static_component_added_rows(cw, ch_a, ch_id, last_run, ch_rows, 8) == 1 && ch_rows[0] == 3,
          "added since last_run: only the new entity, row 3 (a mutation is not an addition)");
    CHECK(mid_ecs_world_static_component_changed_rows(cw, ch_a, ch_id, last_run, ch_rows, 8) == 2
              && ch_rows[0] == 1 && ch_rows[1] == 3,
          "changed since last_run: the mutated row 1 and the new row 3");

    // The rows index the same span and entity list a C caller already reads.
    MidEcsFfiSpan ch_span;
    CHECK(mid_ecs_world_static_component_raw_span(cw, ch_a, ch_id, &ch_span) == MID_ECS_OK && ch_span.count == 4,
          "the span now has 4 elements");
    const FfiHealthC *ch_view = (const FfiHealthC *)ch_span.ptr;
    CHECK(ch_view[1].hp == 22 && ch_view[3].hp == 4,
          "the changed rows point at hp 22 and hp 4 in the span");
    uint64_t ch_ids[8];
    CHECK(mid_ecs_world_static_component_entity_ids(cw, ch_a, ch_id, ch_ids, 8) == 4
              && mid_ecs_world_is_alive(cw, ch_ids[1]) && mid_ecs_world_is_alive(cw, ch_ids[3]),
          "the same rows resolve to live entities through entity_ids");

    CHECK(mid_ecs_test_change_fixture_mutate(cw) == MID_ECS_NOT_FOUND, "the scripted mutation only runs once");
    CHECK(mid_ecs_world_static_component_added_rows(cw, ch_a, 9999, 0, NULL, 0) == MID_ECS_NOT_FOUND,
          "an unregistered component id is MID_ECS_NOT_FOUND");
    CHECK(mid_ecs_world_static_component_changed_rows(cw, 9999, ch_id, 0, NULL, 0) == MID_ECS_NOT_FOUND,
          "an unknown archetype id is MID_ECS_NOT_FOUND");
    CHECK(mid_ecs_world_static_component_added_rows(NULL, ch_a, ch_id, 0, NULL, 0) == MID_ECS_NULL_POINTER
              && mid_ecs_world_static_component_changed_rows(NULL, ch_a, ch_id, 0, NULL, 0) == MID_ECS_NULL_POINTER,
          "a NULL world is MID_ECS_NULL_POINTER");
    CHECK(mid_ecs_test_change_fixture_mutate(NULL) == MID_ECS_NULL_POINTER,
          "the fixture mutation on a NULL world is MID_ECS_NULL_POINTER");
    mid_ecs_world_free(cw);

    // --- Writing component data from C: raw_span_mut (bulk) and write (one row) ---
    MidEcsWorld *ww = mid_ecs_test_write_fixture_world_new();
    uint32_t w_health = mid_ecs_world_lookup_ffi_static_component_id(ww, "FfiHealthStatic");
    uint32_t w_flag_a = mid_ecs_world_lookup_ffi_static_component_id(ww, "FfiFlagA");
    uint32_t w_flag_b = mid_ecs_world_lookup_ffi_static_component_id(ww, "FfiFlagB");
    CHECK(w_health != MID_ECS_INVALID_ID && w_flag_a != MID_ECS_INVALID_ID && w_flag_b != MID_ECS_INVALID_ID,
          "write fixture: FfiHealthStatic / FfiFlagA / FfiFlagB resolve");
    uint32_t w_arch[8];
    int32_t w_arch_n = mid_ecs_world_archetypes_matching_static(ww, &w_health, 1, NULL, 0, NULL, 0, w_arch, 8);
    CHECK(w_arch_n == 2, "write fixture: two archetypes hold Health ({Health} and {Health, FlagA})");
    uint32_t w_plain = MID_ECS_INVALID_ID, w_flagged = MID_ECS_INVALID_ID;
    for (int32_t i = 0; i < w_arch_n; i++) {
        MidEcsFfiSpan probe;
        if (mid_ecs_world_static_component_raw_span(ww, w_arch[i], w_health, &probe) == MID_ECS_OK) {
            if (probe.count == 2) w_plain = w_arch[i];
            if (probe.count == 1) w_flagged = w_arch[i];
        }
    }
    CHECK(w_plain != MID_ECS_INVALID_ID && w_flagged != MID_ECS_INVALID_ID, "the two archetypes told apart by row count (2 and 1)");

    uint32_t w_last_run = mid_ecs_world_change_tick(ww);
    mid_ecs_world_increment_change_tick(ww);
    uint32_t w_rows[8];
    CHECK(mid_ecs_world_static_component_changed_rows(ww, w_plain, w_health, w_last_run, NULL, 0) == 0
              && mid_ecs_world_static_component_changed_rows(ww, w_flagged, w_health, w_last_run, NULL, 0) == 0,
          "before any write, nothing is changed since last_run");

    // Bulk: a writable span over the {Health} column.
    MidEcsFfiSpanMut w_span = { NULL, 0, 0 };
    CHECK(mid_ecs_world_static_component_raw_span_mut(ww, w_plain, w_health, &w_span) == MID_ECS_OK
              && w_span.count == 2 && w_span.stride == sizeof(FfiHealthC) && w_span.ptr != NULL,
          "raw_span_mut on {Health}: a writable span of 2 rows, stride 4");
    for (size_t r = 0; r < w_span.count; r++) {
        ((FfiHealthC *)(w_span.ptr + r * w_span.stride))->hp += 100;
    }
    MidEcsFfiSpan w_read;
    CHECK(mid_ecs_world_static_component_raw_span(ww, w_plain, w_health, &w_read) == MID_ECS_OK
              && (const uint8_t *)w_read.ptr == (const uint8_t *)w_span.ptr
              && ((const FfiHealthC *)w_read.ptr)[0].hp == 101 && ((const FfiHealthC *)w_read.ptr)[1].hp == 103,
          "writes through the writable span show in the read-only span (same memory): hp 101 and 103");
    CHECK(mid_ecs_world_static_component_changed_rows(ww, w_plain, w_health, w_last_run, w_rows, 8) == 2
              && w_rows[0] == 0 && w_rows[1] == 1
              && mid_ecs_world_static_component_changed_rows(ww, w_flagged, w_health, w_last_run, NULL, 0) == 0,
          "raw_span_mut stamps the whole acquired column changed (rows 0, 1) and no other archetype");
    CHECK(mid_ecs_world_static_component_added_rows(ww, w_plain, w_health, w_last_run, NULL, 0) == 0,
          "...as changed, not added");

    // Precise: one entity's value, one row stamped.
    uint64_t w_flagged_ids[4], w_plain_ids[4];
    CHECK(mid_ecs_world_static_component_entity_ids(ww, w_flagged, w_health, w_flagged_ids, 4) == 1
              && mid_ecs_world_static_component_entity_ids(ww, w_plain, w_health, w_plain_ids, 4) == 2,
          "entity ids for the two archetypes: one entity and two entities");
    uint32_t hp_222 = 222;
    CHECK(mid_ecs_world_static_component_write(ww, w_flagged_ids[0], w_health, (const uint8_t *)&hp_222, sizeof hp_222) == MID_ECS_OK,
          "write: copy hp 222 into the {Health, FlagA} entity");
    MidEcsFfiSpan w_read2;
    CHECK(mid_ecs_world_static_component_raw_span(ww, w_flagged, w_health, &w_read2) == MID_ECS_OK
              && ((const FfiHealthC *)w_read2.ptr)[0].hp == 222
              && ((const FfiHealthC *)w_read.ptr)[0].hp == 101,
          "the written value reads back as 222 and the other archetype is untouched");
    CHECK(mid_ecs_world_static_component_changed_rows(ww, w_flagged, w_health, w_last_run, w_rows, 8) == 1 && w_rows[0] == 0,
          "write stamps exactly that one row changed");

    // Errors: nothing written, nothing stamped.
    MidEcsFfiSpanMut w_none = { NULL, 0, 0 };
    uint32_t hp_77 = 77;
    CHECK(mid_ecs_world_static_component_raw_span_mut(ww, w_flagged, w_flag_a, &w_none) == MID_ECS_NOT_WRITABLE
              && mid_ecs_world_static_component_write(ww, w_flagged_ids[0], w_flag_a, (const uint8_t *)&hp_77, 4) == MID_ECS_NOT_WRITABLE,
          "a component registered read-only is MID_ECS_NOT_WRITABLE, for both raw_span_mut and write");
    CHECK(mid_ecs_world_static_component_raw_span_mut(ww, w_plain, 9999, &w_none) == MID_ECS_NOT_FOUND
              && mid_ecs_world_static_component_raw_span_mut(ww, 9999, w_health, &w_none) == MID_ECS_NOT_FOUND
              && mid_ecs_world_static_component_raw_span_mut(ww, 0, w_health, &w_none) == MID_ECS_NOT_FOUND
              && mid_ecs_world_static_component_raw_span_mut(ww, w_plain, w_flag_b, &w_none) == MID_ECS_NOT_FOUND,
          "raw_span_mut: unregistered id, unknown archetype, signature lacking the component: MID_ECS_NOT_FOUND");
    CHECK(mid_ecs_world_static_component_write(ww, w_plain_ids[0], w_flag_b, (const uint8_t *)&hp_77, 4) == MID_ECS_NOT_FOUND
              && mid_ecs_world_static_component_write(ww, w_plain_ids[0], 9999, (const uint8_t *)&hp_77, 4) == MID_ECS_NOT_FOUND,
          "write: an alive entity without the component, or an unregistered id, is MID_ECS_NOT_FOUND");
    CHECK(mid_ecs_world_static_component_write(ww, w_plain_ids[0], w_health, (const uint8_t *)&hp_77, 3) == MID_ECS_SIZE_MISMATCH
              && mid_ecs_world_static_component_write(ww, w_plain_ids[0], w_health, (const uint8_t *)&hp_77, 5) == MID_ECS_SIZE_MISMATCH
              && mid_ecs_world_static_component_write(ww, w_plain_ids[0], w_health, NULL, 0) == MID_ECS_SIZE_MISMATCH,
          "write: a length other than the type's size (3, 5, or NULL with 0) is MID_ECS_SIZE_MISMATCH");
    CHECK(mid_ecs_world_static_component_write(ww, 0xFFFFFFFFFFFFFFFFull, w_health, (const uint8_t *)&hp_77, 4) == MID_ECS_NOT_ALIVE,
          "write: a bogus entity value is MID_ECS_NOT_ALIVE");
    CHECK(mid_ecs_world_static_component_raw_span_mut(NULL, w_plain, w_health, &w_none) == MID_ECS_NULL_POINTER
              && mid_ecs_world_static_component_raw_span_mut(ww, w_plain, w_health, NULL) == MID_ECS_NULL_POINTER
              && mid_ecs_world_static_component_write(NULL, w_plain_ids[0], w_health, (const uint8_t *)&hp_77, 4) == MID_ECS_NULL_POINTER
              && mid_ecs_world_static_component_write(ww, w_plain_ids[0], w_health, NULL, 4) == MID_ECS_NULL_POINTER,
          "NULL world, NULL out-span, or NULL bytes with a length is MID_ECS_NULL_POINTER");
    CHECK(((const FfiHealthC *)w_read.ptr)[0].hp == 101 && ((const FfiHealthC *)w_read.ptr)[1].hp == 103
              && ((const FfiHealthC *)w_read2.ptr)[0].hp == 222
              && mid_ecs_world_static_component_changed_rows(ww, w_flagged, w_health, w_last_run, NULL, 0) == 1,
          "none of the failed calls changed a value or stamped a row");

    // A write to a despawned entity is NOT_ALIVE and never reaches the slot's next occupant.
    CHECK(mid_ecs_world_despawn(ww, w_flagged_ids[0]) == MID_ECS_OK
              && mid_ecs_world_static_component_write(ww, w_flagged_ids[0], w_health, (const uint8_t *)&hp_77, 4) == MID_ECS_NOT_ALIVE,
          "write to a despawned entity is MID_ECS_NOT_ALIVE");
    mid_ecs_world_free(ww);

    // --- Inserting a component from C: insert (direct) and the queued forms ---
    {
        MidEcsWorld *iw = mid_ecs_test_write_fixture_world_new();
        uint32_t i_health = mid_ecs_world_lookup_ffi_static_component_id(iw, "FfiHealthStatic");
        uint32_t i_flag_a = mid_ecs_world_lookup_ffi_static_component_id(iw, "FfiFlagA");
        uint32_t i_flag_b = mid_ecs_world_lookup_ffi_static_component_id(iw, "FfiFlagB");
        uint32_t i_arch[8];
        int32_t i_n = mid_ecs_world_archetypes_matching_static(iw, &i_health, 1, NULL, 0, NULL, 0, i_arch, 8);
        uint64_t e1 = 0, e2 = 0, e3 = 0;
        {
            uint64_t all[8];
            int32_t found = 0;
            for (int32_t i = 0; i < i_n; i++) {
                uint64_t ids[8];
                int32_t c = mid_ecs_world_static_component_entity_ids(iw, i_arch[i], i_health, ids, 8);
                for (int32_t r = 0; r < c && found < 8; r++) all[found++] = ids[r];
            }
            CHECK(found == 3, "insert fixture: three entities hold Health");
            e1 = all[0]; e2 = all[1]; e3 = all[2];
        }
        uint32_t i_last_run = mid_ecs_world_change_tick(iw);
        mid_ecs_world_increment_change_tick(iw);
        uint32_t seen = 0, v41 = 41, v77 = 77;
        uint32_t e2_hp = 0;
        CHECK(u32_of(iw, i_health, e2, &e2_hp), "insert fixture: the second Health entity's hp reads back");
        CHECK(mid_ecs_world_archetypes_matching_static(iw, &i_flag_b, 1, NULL, 0, NULL, 0, i_arch, 8) == 0
                  && !u32_of(iw, i_flag_b, e1, &seen),
              "before any insert, no archetype holds FlagB");

        // Direct insert.
        CHECK(mid_ecs_world_static_component_insert(iw, e1, i_flag_b, (const uint8_t *)&v41, sizeof v41) == MID_ECS_OK
                  && u32_of(iw, i_flag_b, e1, &seen) && seen == 41,
              "insert: FlagB 41 is attached to an entity that had none, and reads back");
        {
            uint32_t fb_arch[8];
            int32_t fb_n = mid_ecs_world_archetypes_matching_static(iw, &i_flag_b, 1, NULL, 0, NULL, 0, fb_arch, 8);
            CHECK(fb_n == 1
                      && mid_ecs_world_static_component_changed_rows(iw, fb_arch[0], i_flag_b, i_last_run, NULL, 0) == 1
                      && mid_ecs_world_static_component_added_rows(iw, fb_arch[0], i_flag_b, i_last_run, NULL, 0) == 1,
                  "the inserted value is stamped both added and changed");
        }
        {
            uint32_t h_arch[8];
            int32_t h_n = mid_ecs_world_archetypes_matching_static(iw, &i_health, 1, NULL, 0, NULL, 0, h_arch, 8);
            int32_t restamped = 0;
            for (int32_t i = 0; i < h_n; i++) {
                restamped += mid_ecs_world_static_component_changed_rows(iw, h_arch[i], i_health, i_last_run, NULL, 0);
            }
            CHECK(h_n >= 2 && restamped == 0 && u32_of(iw, i_health, e1, &seen) && seen == 1,
                  "the entity's existing Health moved with it, intact and not re-stamped");
        }

        // Direct-insert errors: nothing inserted, nothing changed.
        CHECK(mid_ecs_world_static_component_insert(iw, e1, i_flag_b, (const uint8_t *)&v77, 4) == MID_ECS_ALREADY_PRESENT
                  && mid_ecs_world_static_component_insert(iw, e2, i_health, (const uint8_t *)&v77, 4) == MID_ECS_ALREADY_PRESENT
                  && u32_of(iw, i_flag_b, e1, &seen) && seen == 41
                  && u32_of(iw, i_health, e2, &seen) && seen == e2_hp,
              "insert of a component the entity already has is MID_ECS_ALREADY_PRESENT and the first value stays");
        CHECK(mid_ecs_world_static_component_insert(iw, e2, i_flag_a, (const uint8_t *)&v77, 4) == MID_ECS_NOT_WRITABLE
                  && mid_ecs_world_static_component_insert(iw, e2, 9999, (const uint8_t *)&v77, 4) == MID_ECS_NOT_FOUND,
              "insert: a read-only component is MID_ECS_NOT_WRITABLE, an unregistered id MID_ECS_NOT_FOUND");
        CHECK(mid_ecs_world_static_component_insert(iw, e2, i_flag_b, (const uint8_t *)&v77, 3) == MID_ECS_SIZE_MISMATCH
                  && mid_ecs_world_static_component_insert(iw, e2, i_flag_b, (const uint8_t *)&v77, 5) == MID_ECS_SIZE_MISMATCH
                  && mid_ecs_world_static_component_insert(iw, e2, i_flag_b, NULL, 0) == MID_ECS_SIZE_MISMATCH,
              "insert: a length other than the type's size (3, 5, or NULL with 0) is MID_ECS_SIZE_MISMATCH");
        CHECK(mid_ecs_world_static_component_insert(iw, 0xFFFFFFFFFFFFFFFFull, i_flag_b, (const uint8_t *)&v77, 4) == MID_ECS_NOT_ALIVE
                  && mid_ecs_world_despawn(iw, e3) == MID_ECS_OK
                  && mid_ecs_world_static_component_insert(iw, e3, i_flag_b, (const uint8_t *)&v77, 4) == MID_ECS_NOT_ALIVE,
              "insert: a bogus or despawned entity is MID_ECS_NOT_ALIVE");
        CHECK(mid_ecs_world_static_component_insert(NULL, e2, i_flag_b, (const uint8_t *)&v77, 4) == MID_ECS_NULL_POINTER
                  && mid_ecs_world_static_component_insert(iw, e2, i_flag_b, NULL, 4) == MID_ECS_NULL_POINTER
                  && !u32_of(iw, i_flag_b, e2, &seen),
              "insert: NULL world, or NULL bytes with a length, is MID_ECS_NULL_POINTER and e2 still has no FlagB");

        // Queued: spawn two entities and give them components in one apply, plus one existing entity.
        MidEcsCommands *iq = mid_ecs_commands_new();
        uint32_t v10 = 10, v20 = 20, v99 = 99;
        CHECK(mid_ecs_commands_spawn(iq) == MID_ECS_OK && mid_ecs_commands_spawn(iq) == MID_ECS_OK
                  && mid_ecs_commands_spawn_count(iq) == 2,
              "queue: two spawns");
        CHECK(mid_ecs_commands_static_component_insert_spawned(iq, iw, 1, i_flag_b, (const uint8_t *)&v20, 4) == MID_ECS_OK
                  && mid_ecs_commands_static_component_insert_spawned(iq, iw, 0, i_flag_b, (const uint8_t *)&v10, 4) == MID_ECS_OK
                  && mid_ecs_commands_static_component_insert(iq, iw, e2, i_flag_b, (const uint8_t *)&v99, 4) == MID_ECS_OK
                  && mid_ecs_commands_len(iq) == 5,
              "queue: inserts onto spawn slots 1 and 0 and onto an existing entity are queued (5 commands)");
        CHECK(!u32_of(iw, i_flag_b, e2, &seen) && mid_ecs_world_entity_count(iw) == 2,
              "nothing has touched the world before apply");
        uint64_t spawned[2] = { 0, 0 };
        CHECK(mid_ecs_commands_apply(iq, iw, spawned, 2) == 2
                  && u32_of(iw, i_flag_b, spawned[0], &seen) && seen == 10
                  && u32_of(iw, i_flag_b, spawned[1], &seen) && seen == 20
                  && u32_of(iw, i_flag_b, e2, &seen) && seen == 99
                  && mid_ecs_commands_len(iq) == 0,
              "apply: the spawned entities got FlagB 10 and 20 (by slot), the existing one 99, and the queue is empty");

        // Queue-time errors queue nothing.
        CHECK(mid_ecs_commands_static_component_insert(iq, iw, e2, i_flag_a, (const uint8_t *)&v10, 4) == MID_ECS_NOT_WRITABLE
                  && mid_ecs_commands_static_component_insert(iq, iw, e2, 9999, (const uint8_t *)&v10, 4) == MID_ECS_NOT_FOUND
                  && mid_ecs_commands_static_component_insert(iq, iw, e2, i_flag_b, (const uint8_t *)&v10, 3) == MID_ECS_SIZE_MISMATCH
                  && mid_ecs_commands_static_component_insert_spawned(iq, iw, 0, i_flag_b, (const uint8_t *)&v10, 4) == MID_ECS_NOT_FOUND
                  && mid_ecs_commands_static_component_insert_spawned(iq, iw, (size_t)-1, i_flag_b, (const uint8_t *)&v10, 4) == MID_ECS_NOT_FOUND
                  && mid_ecs_commands_len(iq) == 0,
              "queued insert errors (not writable, unknown id, wrong length, no such spawn slot) queue nothing");
        CHECK(mid_ecs_commands_static_component_insert(NULL, iw, e2, i_flag_b, (const uint8_t *)&v10, 4) == MID_ECS_NULL_POINTER
                  && mid_ecs_commands_static_component_insert(iq, NULL, e2, i_flag_b, (const uint8_t *)&v10, 4) == MID_ECS_NULL_POINTER
                  && mid_ecs_commands_static_component_insert(iq, iw, e2, i_flag_b, NULL, 4) == MID_ECS_NULL_POINTER
                  && mid_ecs_commands_static_component_insert_spawned(iq, NULL, 0, i_flag_b, (const uint8_t *)&v10, 4) == MID_ECS_NULL_POINTER
                  && mid_ecs_commands_len(iq) == 0,
              "queued insert: NULL commands, world or bytes is MID_ECS_NULL_POINTER and queues nothing");

        // A queued insert that no longer applies is a no-op at apply.
        CHECK(mid_ecs_commands_static_component_insert(iq, iw, e2, i_health, (const uint8_t *)&v77, 4) == MID_ECS_OK
                  && mid_ecs_commands_apply(iq, iw, NULL, 0) == 0
                  && u32_of(iw, i_health, e2, &seen) && seen == e2_hp,
              "a queued insert of a component the entity already has is a no-op when applied (hp unchanged)");
        CHECK(mid_ecs_commands_static_component_insert(iq, iw, spawned[0], i_flag_b, (const uint8_t *)&v77, 4) == MID_ECS_OK
                  && mid_ecs_world_despawn(iw, spawned[0]) == MID_ECS_OK
                  && mid_ecs_commands_apply(iq, iw, NULL, 0) == 0
                  && !mid_ecs_world_is_alive(iw, spawned[0]),
              "a queued insert onto an entity despawned before apply is a no-op");
        mid_ecs_commands_free(iq);
        mid_ecs_world_free(iw);
    }

    // --- Command queue: spawns and despawns recorded now, applied after ---
    {
        MidEcsWorld *qw = mid_ecs_world_new();
        MidEcsCommands *q = mid_ecs_commands_new();
        CHECK(q != NULL, "commands_new returns a non-null handle");
        CHECK(mid_ecs_commands_len(q) == 0 && mid_ecs_commands_spawn_count(q) == 0,
              "a fresh queue is empty");

        uint64_t keep = mid_ecs_world_spawn(qw);
        uint64_t doomed = mid_ecs_world_spawn(qw);
        CHECK(mid_ecs_commands_spawn(q) == MID_ECS_OK
                  && mid_ecs_commands_spawn(q) == MID_ECS_OK
                  && mid_ecs_commands_spawn(q) == MID_ECS_OK,
              "three spawns queue with MID_ECS_OK");
        CHECK(mid_ecs_commands_despawn(q, doomed) == MID_ECS_OK, "a despawn queues with MID_ECS_OK");
        CHECK(mid_ecs_commands_len(q) == 4 && mid_ecs_commands_spawn_count(q) == 3,
              "len counts every command, spawn_count only the spawns");
        CHECK(mid_ecs_world_entity_count(qw) == 2 && mid_ecs_world_is_alive(qw, doomed),
              "nothing happens to the world until apply");

        uint64_t out[8];
        for (int i = 0; i < 8; i++) out[i] = 0xDEADBEEFULL;
        CHECK(mid_ecs_commands_apply(q, qw, out, 2) == MID_ECS_BUFFER_TOO_SMALL,
              "apply with a buffer smaller than the spawn count is MID_ECS_BUFFER_TOO_SMALL");
        CHECK(mid_ecs_world_entity_count(qw) == 2 && mid_ecs_world_is_alive(qw, doomed)
                  && mid_ecs_commands_len(q) == 4 && out[0] == 0xDEADBEEFULL,
              "a too-small apply applied nothing: world, queue and buffer are untouched");
        CHECK(mid_ecs_commands_apply(q, qw, NULL, 0) == MID_ECS_BUFFER_TOO_SMALL,
              "a NULL buffer with spawns waiting is MID_ECS_BUFFER_TOO_SMALL");
        CHECK(mid_ecs_commands_apply(q, qw, NULL, 4) == MID_ECS_BUFFER_TOO_SMALL
                  && mid_ecs_commands_len(q) == 4,
              "a NULL buffer is never written through, whatever capacity is claimed for it");

        CHECK(mid_ecs_commands_apply(q, qw, out, 8) == 3, "apply returns the number of spawned entities");
        CHECK(out[3] == 0xDEADBEEFULL, "only the spawned prefix of the buffer is written");
        CHECK(mid_ecs_world_is_alive(qw, out[0]) && mid_ecs_world_is_alive(qw, out[1])
                  && mid_ecs_world_is_alive(qw, out[2]),
              "all three returned entities are alive");
        CHECK(out[0] != out[1] && out[1] != out[2] && out[0] != out[2],
              "the returned entities are distinct");
        CHECK(!mid_ecs_world_is_alive(qw, doomed) && mid_ecs_world_is_alive(qw, keep),
              "the queued despawn ran and the untouched entity survived");
        CHECK(mid_ecs_world_entity_count(qw) == 4, "entity_count is 4 after apply (keep + 3 spawned)");
        CHECK(mid_ecs_commands_len(q) == 0 && mid_ecs_commands_spawn_count(q) == 0,
              "the queue is empty after apply");
        CHECK(mid_ecs_commands_apply(q, qw, NULL, 0) == 0,
              "a second apply with nothing queued returns 0, and a NULL buffer is fine");

        // A queued despawn of a stale handle must not touch the entity that
        // reused its slot.
        uint64_t stale = out[0];
        CHECK(mid_ecs_world_despawn(qw, stale) == MID_ECS_OK, "despawn the first spawned entity directly");
        uint64_t reuser = mid_ecs_world_spawn(qw);
        CHECK(reuser != stale, "the slot-reusing entity has a different packed value");
        CHECK(mid_ecs_commands_despawn(q, stale) == MID_ECS_OK
                  && mid_ecs_commands_apply(q, qw, NULL, 0) == 0,
              "a queued despawn of the stale handle applies cleanly");
        CHECK(mid_ecs_world_is_alive(qw, reuser),
              "the stale despawn did not kill the entity that reused the slot");

        // Bogus values are queued fine and are a no-op when run.
        CHECK(mid_ecs_commands_despawn(q, 0xFFFFFFFFFFFFFFFFULL) == MID_ECS_OK
                  && mid_ecs_commands_apply(q, qw, NULL, 0) == 0,
              "a bogus packed value queues and runs as a no-op, not a crash");

        // clear drops pending commands unrun.
        mid_ecs_commands_spawn(q);
        CHECK(mid_ecs_commands_clear(q) == MID_ECS_OK && mid_ecs_commands_len(q) == 0,
              "clear empties the queue");
        size_t before = mid_ecs_world_entity_count(qw);
        CHECK(mid_ecs_commands_apply(q, qw, NULL, 0) == 0 && mid_ecs_world_entity_count(qw) == before,
              "a cleared spawn never runs");

        // NULL handles are safe everywhere.
        CHECK(mid_ecs_commands_len(NULL) == 0 && mid_ecs_commands_spawn_count(NULL) == 0,
              "len and spawn_count on a NULL queue are 0");
        CHECK(mid_ecs_commands_spawn(NULL) == MID_ECS_NULL_POINTER
                  && mid_ecs_commands_despawn(NULL, 1) == MID_ECS_NULL_POINTER
                  && mid_ecs_commands_clear(NULL) == MID_ECS_NULL_POINTER,
              "spawn, despawn and clear on a NULL queue are MID_ECS_NULL_POINTER");
        CHECK(mid_ecs_commands_apply(NULL, qw, NULL, 0) == MID_ECS_NULL_POINTER
                  && mid_ecs_commands_apply(q, NULL, NULL, 0) == MID_ECS_NULL_POINTER,
              "apply with a NULL queue or a NULL world is MID_ECS_NULL_POINTER");
        mid_ecs_commands_free(NULL);
        CHECK(true, "freeing a NULL queue is a safe no-op");

        mid_ecs_commands_free(q);
        mid_ecs_world_free(qw);
    }

    printf("\n=== %d check(s) failed ===\n", failures);
    return failures == 0 ? 0 : 1;
}
