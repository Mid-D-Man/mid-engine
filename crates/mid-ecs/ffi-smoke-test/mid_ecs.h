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
#define MID_ECS_SIZE_MISMATCH  -6
#define MID_ECS_NOT_WRITABLE   -7
#define MID_ECS_ALREADY_PRESENT -8
#define MID_ECS_DUPLICATE       -9
#define MID_ECS_NOT_TRACKED    -10
#define MID_ECS_INVALID_ARGUMENT -11

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

// The writable sibling of MidEcsFfiSpan, from the *_raw_span_mut functions:
// the same shape with a non-const ptr, so you can write the values in
// place. Same stride / count / empty-sentinel meaning (ptr NULL, count 0).
// Only components registered writable on the Rust side hand one out.
typedef struct MidEcsFfiSpanMut {
    uint8_t *ptr;
    size_t stride;
    size_t count;
} MidEcsFfiSpanMut;

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

// --- Deferred changes (command queue) ---
//
// Record spawns, despawns and component inserts, then run them against a
// world with mid_ecs_commands_apply. Nothing touches the world until apply.
// Queued inserts (mid_ecs_commands_static_component_insert[_spawned], below
// with the other insert functions) are for components registered writable;
// see docs/mid-ecs.md, "Inserting a component from C".

typedef struct MidEcsCommands MidEcsCommands;

// Creates an empty queue. Never returns NULL. Free with
// mid_ecs_commands_free.
MidEcsCommands *mid_ecs_commands_new(void);

// Frees a queue; anything still queued is dropped unapplied. Safe on NULL.
void mid_ecs_commands_free(MidEcsCommands *commands);

// Number of commands waiting. 0 on a NULL queue.
size_t mid_ecs_commands_len(const MidEcsCommands *commands);

// Number of spawn commands waiting: how many entities the next apply
// returns, so the buffer size to give it. 0 on a NULL queue.
size_t mid_ecs_commands_spawn_count(const MidEcsCommands *commands);

// Drops every waiting command unrun. MID_ECS_NULL_POINTER on NULL, else
// MID_ECS_OK.
int32_t mid_ecs_commands_clear(MidEcsCommands *commands);

// Queues a spawn; the entity exists only after apply, which returns the
// spawned entities in the order their spawns were queued.
// MID_ECS_NULL_POINTER on NULL, else MID_ECS_OK.
int32_t mid_ecs_commands_spawn(MidEcsCommands *commands);

// Queues a despawn of `entity`. Liveness is checked when the command
// runs, not now: a dead, stale or bogus value is then a safe no-op and
// never touches a different entity that reused the slot.
// MID_ECS_NULL_POINTER on NULL, else MID_ECS_OK.
int32_t mid_ecs_commands_despawn(MidEcsCommands *commands, uint64_t entity);

// Runs every queued command against `world` in the order queued and
// leaves the queue empty and reusable. Writes the entities its spawns
// produced into out_entities, in spawn order. out_capacity counts
// ELEMENTS and must be at least mid_ecs_commands_spawn_count; if it is
// not (or out_entities is NULL while spawns are waiting) the result is
// MID_ECS_BUFFER_TOO_SMALL and NOTHING is applied, so call again with a
// bigger buffer. With no spawns waiting, out_entities may be NULL.
// Returns the number of entities written (>= 0), or MID_ECS_NULL_POINTER /
// MID_ECS_BUFFER_TOO_SMALL.
int32_t mid_ecs_commands_apply(MidEcsCommands *commands, MidEcsWorld *world,
                               uint64_t *out_entities, size_t out_capacity);

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

// Filtered counterpart of mid_ecs_world_archetypes_with_static_component:
// enumerates every currently-existing archetype whose signature contains
// ALL of with_ids, NONE of without_ids, and -- if any_of_ids is non-empty
// -- AT LEAST ONE of any_of_ids (the runtime equivalent of the Rust side's
// With/Without/Or query filters). Include the component you mean to read
// in with_ids -- an archetype without it answers
// mid_ecs_world_static_component_raw_span/_entity_ids with
// MID_ECS_NOT_FOUND. Structural, like its sibling: archetypes with zero
// rows are included (their spans come back MID_ECS_OK with count == 0),
// an id that names no registered component matches nothing in
// with_ids/any_of_ids and is ignored in without_ids, an id in both
// with_ids and without_ids matches nothing, and with_ids/without_ids both
// empty with any_of_ids also empty matches every archetype. Each id list
// may be NULL only when its own length is 0 (MID_ECS_NULL_POINTER
// otherwise). Same NULL-buffer-queries-count idiom; never returns
// MID_ECS_NOT_FOUND.
// Signature changed when Or was added on the Rust side: any_of_ids/any_of_len
// were inserted before out_buf. Every existing call site needs those two
// extra arguments now -- NULL, 0 if unused (an empty any_of is "no Or
// constraint", matching every archetype on that axis, not "match nothing").
int32_t mid_ecs_world_archetypes_matching_static(const MidEcsWorld *world, const uint32_t *with_ids, size_t with_len, const uint32_t *without_ids, size_t without_len, const uint32_t *any_of_ids, size_t any_of_len, uint32_t *out_buf, size_t out_buf_capacity);

// --- Resources ---
//
// A resource is at most one value of a Rust type, owned by the world and
// attached to no entity. Rust registers each C-visible resource type once
// (register_ffi_resource, generic, so not callable from C) under a name;
// C looks the name up and then reads, writes and removes by resource_id.
// Resource ids are their own namespace, separate from component ids.

// Resolves a registered resource name to its resource_id, or
// MID_ECS_INVALID_ID on a null world/name, invalid UTF-8, or an unknown
// name.
uint32_t mid_ecs_world_lookup_ffi_resource_id(const MidEcsWorld *world, const char *name);

// Writes a view of the resource's current value into *out_span: one
// element (count == 1, stride == sizeof the type), or count == 0 if the
// resource is registered but not currently inserted. MID_ECS_NOT_FOUND if
// resource_id was never issued. The span stays valid across
// mid_ecs_world_resource_write (which updates an existing value in place)
// and is invalidated by mid_ecs_world_resource_remove, by a Rust-side
// insert/remove of the same resource, and by mid_ecs_world_free.
int32_t mid_ecs_world_resource_raw_span(const MidEcsWorld *world, uint32_t resource_id, MidEcsFfiSpan *out_span);

// Copies len bytes from bytes in as the resource's new value, inserting it
// if it isn't currently inserted. len must equal the type's size exactly
// (the span's stride), else MID_ECS_SIZE_MISMATCH and nothing is written.
// MID_ECS_NOT_FOUND if resource_id was never issued. bytes is copied, so it
// needs no particular alignment; it may be NULL only when len is 0.
int32_t mid_ecs_world_resource_write(MidEcsWorld *world, uint32_t resource_id, const uint8_t *bytes, size_t len);

// Removes the resource's value. MID_ECS_NOT_FOUND if resource_id was never
// issued or the resource isn't currently inserted. Invalidates any span
// previously obtained for it.
int32_t mid_ecs_world_resource_remove(MidEcsWorld *world, uint32_t resource_id);

// --- Change detection: added / changed rows ---
//
// The C counterpart of Rust's World::query_added / query_changed, for
// components in the Archetype Core (the only storage that tracks ticks).
// The world keeps one change tick, starting at 1. Your "tracker" is a
// plain uint32_t last_run you hold yourself, starting at 0 (a fresh
// tracker sees everything present); after using the rows below, set it
// to mid_ecs_world_change_tick(world). Advance the tick once per step of
// your own loop with mid_ecs_world_increment_change_tick, BETWEEN the
// step's mutations and the check that should see them: a value touched
// in the same tick last_run was set to is not reported.
//
// Results are row indices into one (archetype_id, component_id) pair, so
// they line up with mid_ecs_world_static_component_raw_span and
// _entity_ids for that same pair: row i is element i of both. Walk
// archetypes with mid_ecs_world_archetypes_matching_static and call the
// row functions per archetype.

// The world's current change tick. 0 on a NULL world (a real world's tick
// is always >= 1).
uint32_t mid_ecs_world_change_tick(const MidEcsWorld *world);

// Advances the change tick by one and returns the new value. 0 on a NULL
// world.
uint32_t mid_ecs_world_increment_change_tick(MidEcsWorld *world);

// Writes, in ascending order, the rows whose value was INSERTED after
// last_run into out_rows, and returns how many. NULL out_rows returns the
// count; a too-small buffer is MID_ECS_BUFFER_TOO_SMALL and nothing is
// written. MID_ECS_NOT_FOUND under exactly the conditions
// mid_ecs_world_static_component_raw_span returns it (component never
// registered for FFI, unknown archetype, or the archetype's signature
// lacks the component); an archetype with no rows is 0, not an error.
// Row indices are valid against the span only until the next call that
// changes that component's storage.
int32_t mid_ecs_world_static_component_added_rows(const MidEcsWorld *world, uint32_t archetype_id, uint32_t component_id, uint32_t last_run, uint32_t *out_rows, size_t out_capacity);

// As above, for rows whose value was inserted OR last mutated after
// last_run: through Rust's World::get_static_mut or bulk mutable queries, or
// from C through mid_ecs_world_static_component_raw_span_mut and
// mid_ecs_world_static_component_write below.
int32_t mid_ecs_world_static_component_changed_rows(const MidEcsWorld *world, uint32_t archetype_id, uint32_t component_id, uint32_t last_run, uint32_t *out_rows, size_t out_capacity);

// --- Filtered change queries: rows added/changed since last_run, across archetypes ---
//
// One call replaces the loop a caller otherwise writes: enumerate archetypes
// with mid_ecs_world_archetypes_matching_static, ask *_added_rows /
// *_changed_rows for each, then intersect or union the row lists. The C
// counterpart of Rust's Added/Changed filters combined with
// With/Without/Or. All ids are Archetype Core ids.
//
// Seven (pointer, length) lists. A length of 0 is an empty list (the pointer
// may then be NULL); a NULL pointer with a nonzero length is
// MID_ECS_NULL_POINTER. Zero-initialize the struct and set only the lists you
// need: MidEcsRowFilter f = { .all_changed_ids = ids, .all_changed_len = 1 };
//
//   Structural, decided once per archetype (as in
//   mid_ecs_world_archetypes_matching_static):
//     with     every id present        without  no id present
//     any_of   at least one present, when non-empty
//   Row-level, at least one id in total across these four lists:
//     all_added    every listed component's value was INSERTED after last_run
//     all_changed  every listed component's value was inserted or mutated
//                  after last_run
//     any_added /  at least one listed component qualifies, the two lists
//     any_changed  forming ONE group. A listed component the archetype lacks
//                  counts as not qualifying; an archetype holding none of
//                  them has no matching row.
//   A row must pass the structural lists, every all_* id, and, if either any_*
//   list is non-empty, at least one any_* id.
typedef struct {
    const uint32_t *with_ids;
    size_t with_len;
    const uint32_t *without_ids;
    size_t without_len;
    const uint32_t *any_of_ids;
    size_t any_of_len;
    const uint32_t *all_added_ids;
    size_t all_added_len;
    const uint32_t *all_changed_ids;
    size_t all_changed_len;
    const uint32_t *any_added_ids;
    size_t any_added_len;
    const uint32_t *any_changed_ids;
    size_t any_changed_len;
} MidEcsRowFilter;

// One result row: the entity, and where it sits. row is element row of
// mid_ecs_world_static_component_raw_span and _entity_ids for archetype_id and
// any component that archetype holds.
typedef struct {
    uint64_t entity;
    uint32_t archetype_id;
    uint32_t row;
} MidEcsMatchedRow;

// Writes the matching rows into out_rows and returns how many: archetypes in
// the order mid_ecs_world_archetypes_matching_static yields them, rows
// ascending within one. last_run is a change tick you kept, as for the
// *_rows functions (0 sees everything; a change in the same tick as last_run
// is not reported). NULL out_rows returns the count; a too-small buffer is
// MID_ECS_BUFFER_TOO_SMALL and nothing is written. Row indices are valid
// against the spans only until the next call that changes that archetype's
// storage.
//   MID_ECS_NULL_POINTER     NULL world or filter, or a list with a NULL pointer
//                            and a nonzero length.
//   MID_ECS_INVALID_ARGUMENT all four change lists are empty (use
//                            mid_ecs_world_archetypes_matching_static for a
//                            purely structural query).
//   MID_ECS_NOT_FOUND        an id in a change list was never registered for
//                            FFI. Checked first, so a typo'd id is an error
//                            rather than "nothing changed". An unknown id in
//                            a structural list matches nothing (with, any_of)
//                            or is ignored (without).
int32_t mid_ecs_world_static_query_since(const MidEcsWorld *world, uint32_t last_run, const MidEcsRowFilter *filter, MidEcsMatchedRow *out_rows, size_t out_capacity);

// --- Removal tracking: which entities lost a component ---
//
// A removed component leaves nothing in storage, so "which entities lost X
// since I last looked" needs a record. Opt in per component with
// *_track_removals; from then on every removal of that component (any of the
// removal functions, a queued removal, or the entity's despawn) logs the
// entity and the world's change tick, and *_removed_since reports the ones
// newer than a tick the caller kept: the same last_run convention as
// mid_ecs_world_static_component_added_rows. Call
// mid_ecs_world_increment_change_tick once per step of your loop; a removal
// in the same tick as last_run is not reported. Replacing a value
// (mid_ecs_world_static_component_write) is not a removal, and a refused
// removal records nothing.
//
// Two id spaces, two families: mid_ecs_world_static_component_* takes
// Archetype Core ids (mid_ecs_world_lookup_ffi_static_component_id),
// mid_ecs_world_component_* takes Sparse Shell ids
// (mid_ecs_world_lookup_ffi_component_id). The spaces are independent: the
// same integer can name a different component in each, and tracking one does
// not track the other.
//
// Returns of the per-component functions: MID_ECS_NULL_POINTER on a NULL
// world; MID_ECS_NOT_FOUND if this world never issued that id in that space.

// Starts recording removals of the component. Idempotent. Removals before the
// first call are not recorded. Needs only a valid id, not a writable
// component. MID_ECS_OK on success.
int32_t mid_ecs_world_static_component_track_removals(MidEcsWorld *world, uint32_t component_id);
int32_t mid_ecs_world_component_track_removals(MidEcsWorld *world, uint32_t component_id);

// 1 if removals of the component are being recorded, 0 if not.
int32_t mid_ecs_world_static_component_tracks_removals(const MidEcsWorld *world, uint32_t component_id);
int32_t mid_ecs_world_component_tracks_removals(const MidEcsWorld *world, uint32_t component_id);

// How many removals the log holds for the component; 0 if it is not tracked.
int32_t mid_ecs_world_static_component_removal_count(const MidEcsWorld *world, uint32_t component_id);
int32_t mid_ecs_world_component_removal_count(const MidEcsWorld *world, uint32_t component_id);

// Writes, oldest first, the packed entity handles that lost the component
// after last_run into out_entities, and returns how many. An entity removed
// twice appears twice. The entity may since have been despawned or its slot
// reused: check mid_ecs_world_is_alive if it matters. last_run 0 reports
// everything still in the log. NULL out_entities returns the count; a
// too-small buffer is MID_ECS_BUFFER_TOO_SMALL and nothing is written.
// MID_ECS_NOT_TRACKED if *_track_removals was never called for the component:
// nothing was ever recorded, and an empty list would hide that.
int32_t mid_ecs_world_static_component_removed_since(const MidEcsWorld *world, uint32_t component_id, uint32_t last_run, uint64_t *out_entities, size_t out_capacity);
int32_t mid_ecs_world_component_removed_since(const MidEcsWorld *world, uint32_t component_id, uint32_t last_run, uint64_t *out_entities, size_t out_capacity);

// Drops every logged removal of every tracked component that a reader whose
// last run was at_or_before (or later) could no longer see. Nothing trims the
// log automatically: only the owner of all the readers knows the oldest
// last_run any of them still holds, so pass that. Without trimming, the log
// of a component removed often grows without bound. Tracked components stay
// tracked. MID_ECS_OK, or MID_ECS_NULL_POINTER.
int32_t mid_ecs_world_trim_removals(MidEcsWorld *world, uint32_t at_or_before);

// Drops every logged removal. Tracked components stay tracked.
// MID_ECS_OK, or MID_ECS_NULL_POINTER.
int32_t mid_ecs_world_clear_removals(MidEcsWorld *world);

// --- Writing component data from C ---
//
// The read-only spans above stay read-only. A component type can be
// written, inserted or removed from C only if its Rust type was registered
// WRITABLE (World::register_ffi_static_component_writable), which requires
// the type to accept any bit pattern, so nothing C writes can be an invalid
// value.
// Three ways to write, for three needs:
//   * raw_span_mut (bulk): a writable span over one archetype's column.
//     C cannot tell Rust which rows it then writes through a raw pointer,
//     so EVERY row of that column is stamped changed (not added) at the
//     world's current tick when the call returns. That can over-report
//     (a Changed consumer sees rows you left alone) and never misses a
//     write. Do the writes in the same tick.
//   * write (precise): copies one value into one entity's component and
//     stamps exactly that row. Use it for a few rows.
//   * row_mut (precise, in place): a pointer to one entity's value with
//     exactly that row stamped. Use it to mutate a filtered row list.

// Writable counterpart of mid_ecs_world_static_component_raw_span: writes a
// span over component_id's column in archetype_id's table to *out_span.
// MID_ECS_OK; MID_ECS_NULL_POINTER; MID_ECS_NOT_FOUND under exactly the
// conditions raw_span returns it; MID_ECS_NOT_WRITABLE if the component was
// registered read-only. Nothing is stamped on an error. An archetype with
// no rows is OK with count 0 and a NULL ptr. Same invalidation contract as
// the read-only span: valid until the next call that changes that
// archetype's storage (spawn, despawn, insert/remove, applying a command
// queue). Never write past count elements of stride bytes.
int32_t mid_ecs_world_static_component_raw_span_mut(MidEcsWorld *world, uint32_t archetype_id, uint32_t component_id, MidEcsFfiSpanMut *out_span);

// One entity's component_id value, writable IN PLACE, with exactly that row
// stamped changed (not added) at the current tick: the precise form of
// raw_span_mut, and the C counterpart of Rust's get_static_mut (the stamp
// happens when the span is handed out, not when you write through it, so do
// the write in the same tick). *out_span has count 1 and stride the
// component's size. This is how to mutate a filtered subset in place: take
// the rows from mid_ecs_world_static_query_since (or the *_rows functions),
// then call this for each row you really mean to write; a Changed consumer
// then sees those rows and no others. To touch every row of an archetype,
// raw_span_mut stamps them all in one call. There is no C query *iterator*:
// a cursor would need an invalidation rule for no gain over a row list.
// MID_ECS_OK; MID_ECS_NULL_POINTER (NULL world or out_span);
// MID_ECS_NOT_ALIVE (a dead or stale entity; it never reaches the new
// occupant of a reused slot); MID_ECS_NOT_FOUND (id never registered for
// FFI, or the entity is alive but lacks the component); MID_ECS_NOT_WRITABLE
// (registered read-only), checked in that order. *out_span is untouched and
// nothing is stamped on an error. No component hook fires (a mutation, not
// an insert). Same invalidation contract as the other spans.
int32_t mid_ecs_world_static_component_row_mut(MidEcsWorld *world, uint64_t entity, uint32_t component_id, MidEcsFfiSpanMut *out_span);

// Copies len bytes from bytes in as entity's component_id value and stamps
// exactly that row changed (not added) at the current tick. len must equal
// the type's size exactly (the span's stride). bytes is copied, so it needs
// no particular alignment; it may be NULL only when len is 0. Statuses,
// checked in this order: MID_ECS_NULL_POINTER; MID_ECS_NOT_ALIVE (dead,
// bogus or stale entity: it never reaches a different entity that reused
// the slot); MID_ECS_NOT_FOUND (component never registered for FFI, or the
// entity is alive without it); MID_ECS_NOT_WRITABLE; MID_ECS_SIZE_MISMATCH.
// Nothing is written or stamped on an error.
int32_t mid_ecs_world_static_component_write(MidEcsWorld *world, uint64_t entity, uint32_t component_id, const uint8_t *bytes, size_t len);

// --- Inserting a component from C ---
//
// `write` above overwrites a value the entity already has. Giving an entity
// a component it does NOT have yet is a structural change: the entity moves
// to the archetype for its larger component set (so every span previously
// handed out for its old and new archetype is invalidated), and the new
// value is stamped added AND changed at the current tick. The same rule as
// for writes decides which components can be inserted: only those
// registered WRITABLE on the Rust side.

// Attaches a component_id component, built from len bytes at bytes, to
// entity. len must equal the type's size exactly (the span's stride);
// bytes is copied, needs no alignment, and may be NULL only when len is 0.
// Statuses, checked in this order: MID_ECS_NULL_POINTER; MID_ECS_NOT_ALIVE
// (dead, bogus or stale entity); MID_ECS_NOT_FOUND (component never
// registered for FFI); MID_ECS_NOT_WRITABLE; MID_ECS_SIZE_MISMATCH;
// MID_ECS_ALREADY_PRESENT (the entity already has it: nothing is written and
// the first value stays; use mid_ecs_world_static_component_write to
// overwrite). Nothing is written or stamped on an error.
int32_t mid_ecs_world_static_component_insert(MidEcsWorld *world, uint64_t entity, uint32_t component_id, const uint8_t *bytes, size_t len);

// The queued form: records an insert onto an existing entity, applied by
// mid_ecs_commands_apply in the order queued. `world` is only READ, now, to
// check that component_id is registered writable and len is its size (a
// queue has no world of its own, and this puts those errors on the call that
// caused them); it is not retained and nothing in it changes. bytes is
// copied now. Whether the entity is alive is checked when the command
// runs, not now: a dead or stale entity, an entity that already has the
// component (the first value stays), or a world that does not know the
// component, makes the command a no-op then. Returns MID_ECS_OK (queued),
// MID_ECS_NULL_POINTER, MID_ECS_NOT_FOUND (component never registered for
// FFI), MID_ECS_NOT_WRITABLE or MID_ECS_SIZE_MISMATCH. Nothing is queued on
// an error.
int32_t mid_ecs_commands_static_component_insert(MidEcsCommands *commands, const MidEcsWorld *world, uint64_t entity, uint32_t component_id, const uint8_t *bytes, size_t len);

// As above, onto the entity the spawn_index'th queued spawn will produce
// (counting from 0 in the order mid_ecs_commands_spawn was called since the
// last apply or clear; mid_ecs_commands_spawn_count is how many there are),
// so one apply can create an entity AND give it components. The inserts run
// after that spawn, in the order queued. Same statuses, plus
// MID_ECS_NOT_FOUND for a spawn_index not below the number of spawns queued
// so far (checked after the component checks).
int32_t mid_ecs_commands_static_component_insert_spawned(MidEcsCommands *commands, const MidEcsWorld *world, size_t spawn_index, uint32_t component_id, const uint8_t *bytes, size_t len);

// --- Removing a component from C ---
//
// The mirror of insert. Same rule: only components registered WRITABLE can be
// removed (a read-only registration is MID_ECS_NOT_WRITABLE, even on an
// entity that has the component). Needs no bytes. The value is dropped, the
// entity moves to the archetype for its smaller component set, and its
// other values and their change stamps come across untouched. A removal
// stamps nothing: there is no change notification for removals. Removing an
// entity's last component leaves it alive with no components. A structural
// change, so every span previously handed out for the entity's old and new
// archetype is invalidated.

// Detaches component_id from entity and drops its value. Statuses, checked in
// this order: MID_ECS_NULL_POINTER; MID_ECS_NOT_ALIVE (dead, bogus or stale
// entity); MID_ECS_NOT_FOUND (component never registered for FFI, or the
// entity is alive but does not have it); MID_ECS_NOT_WRITABLE. Nothing
// changes on an error.
int32_t mid_ecs_world_static_component_remove(MidEcsWorld *world, uint64_t entity, uint32_t component_id);

// The queued form: records a removal from an existing entity, applied by
// mid_ecs_commands_apply in the order queued. `world` is only READ, now, to
// check that component_id is registered writable; it is not retained.
// Whether the entity is alive and still has the component is checked when
// the command runs: a dead or stale entity (never a different entity that
// reused the slot), an entity without the component, or a world that does not
// know it, makes the command a no-op then. There is no _spawned form: a new
// entity has nothing to remove until an insert has run. Returns MID_ECS_OK
// (queued), MID_ECS_NULL_POINTER, MID_ECS_NOT_FOUND (component never
// registered for FFI) or MID_ECS_NOT_WRITABLE. Nothing is queued on an error.
int32_t mid_ecs_commands_static_component_remove(MidEcsCommands *commands, const MidEcsWorld *world, uint64_t entity, uint32_t component_id);

// --- Bundles: several components in ONE structural change ---
//
// The list forms of insert and remove (Rust's insert_bundle / remove_bundle):
// the entity migrates once, not once per component, and the call is ALL OR
// NOTHING. The whole list is validated as a unit before anything moves, and
// on any error nothing changes. Errors are reported in this order: the entity
// (MID_ECS_NOT_ALIVE); then each listed component in list order
// (MID_ECS_NOT_FOUND for an id never registered for FFI, MID_ECS_NOT_WRITABLE
// for a read-only one, and for an insert MID_ECS_SIZE_MISMATCH); then
// MID_ECS_DUPLICATE if a component is listed twice; then
// MID_ECS_ALREADY_PRESENT (insert: the entity has ANY of them) or
// MID_ECS_NOT_FOUND (remove: the entity lacks ANY of them). A count of 0 is
// MID_ECS_OK and changes nothing (the list pointer may then be NULL). Same
// writable rule as the single forms, same invalidation of spans.

// One entry of a bundle insert: the component and the bytes of its value.
// len must equal the type's size exactly (the span's stride); bytes may be
// NULL only if len is 0. Each value is copied.
typedef struct MidEcsComponentValue {
    uint32_t component_id;
    const uint8_t *bytes;
    size_t len;
} MidEcsComponentValue;

// Attaches count components to entity in one go. Every new value is stamped
// added AND changed at the current tick; the entity's existing values and
// their stamps come across untouched. Also MID_ECS_NULL_POINTER for a NULL
// world, a NULL values with a non-zero count, or an entry with NULL bytes and
// a non-zero len.
int32_t mid_ecs_world_static_component_insert_many(MidEcsWorld *world, uint64_t entity, const MidEcsComponentValue *values, size_t count);

// Detaches count components from entity in one go and drops their values.
// Stamps nothing. Also MID_ECS_NULL_POINTER for a NULL world or a NULL
// component_ids with a non-zero count.
int32_t mid_ecs_world_static_component_remove_many(MidEcsWorld *world, uint64_t entity, const uint32_t *component_ids, size_t count);

// The queued forms. `world` is only READ, now, to run the list-level checks
// (registered writable, lengths, no repeats) so those errors come back on the
// call that caused them; it is not retained. Whether the entity is alive and
// the all-or-nothing entity checks (has none / has all) happen when the
// command runs: any miss makes the WHOLE command a no-op then. Bytes are
// copied now. Nothing is queued on an error, and a count of 0 queues
// nothing.
int32_t mid_ecs_commands_static_component_insert_many(MidEcsCommands *commands, const MidEcsWorld *world, uint64_t entity, const MidEcsComponentValue *values, size_t count);
// As above, onto the entity the spawn_index'th queued spawn will produce
// (MID_ECS_NOT_FOUND for a spawn_index not below the number queued, checked
// after the list checks): one apply can create an entity and give it a whole
// bundle.
int32_t mid_ecs_commands_static_component_insert_many_spawned(MidEcsCommands *commands, const MidEcsWorld *world, size_t spawn_index, const MidEcsComponentValue *values, size_t count);
int32_t mid_ecs_commands_static_component_remove_many(MidEcsCommands *commands, const MidEcsWorld *world, uint64_t entity, const uint32_t *component_ids, size_t count);

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

// NOT a real part of the public API, like the fixture above. A world for
// exercising mid_ecs_world_archetypes_matching_static: "FfiHealthStatic",
// "FfiFlagA" and "FfiFlagB" registered in the Archetype Core (each
// `{ uint32_t <field>; }`), and three entities in distinct archetypes:
// e1 {Health hp=1}, e2 {Health hp=2, FlagA}, e3 {FlagB, Health hp=3,
// FlagA}. e3 is built as one bundle in that order, which leaves zero-row
// intermediate archetypes {FlagB} and {FlagB, Health} behind. Never
// returns NULL.
MidEcsWorld *mid_ecs_test_filter_fixture_world_new(void);

// NOT a real part of the public API, like the fixtures above. A world for
// exercising the resource functions: "FfiTime" (`{ float delta; uint32_t
// frame; }`) registered and inserted as { 0.016f, 7 }, and "FfiGravity"
// (`{ float g; }`) registered but not inserted. Never returns NULL.
MidEcsWorld *mid_ecs_test_resource_fixture_world_new(void);

// NOT a real part of the public API, like the fixtures above. A world for
// exercising the change-detection functions: "FfiHealthStatic" registered
// in the Archetype Core (`{ uint32_t hp; }`) and three entities {Health}
// with hp 1, 2, 3 (rows 0, 1, 2 of their one archetype), inserted at the
// starting tick, with the tick then advanced once (so it reads 2). Never
// returns NULL.
MidEcsWorld *mid_ecs_test_change_fixture_world_new(void);

// NOT a real part of the public API. The Rust-side "frame of game code" C
// cannot write itself: against a world from
// mid_ecs_test_change_fixture_world_new, sets the entity with hp == 2 to
// hp == 22 through get_static_mut (changed, not added) and inserts a new
// {Health hp=4} entity (row 3), both at the CURRENT tick. Call
// mid_ecs_world_increment_change_tick first. MID_ECS_NOT_FOUND if no
// entity has hp == 2 (not that fixture, or already run).
int32_t mid_ecs_test_change_fixture_mutate(MidEcsWorld *world);

// NOT a real part of the public API. A world for exercising writes from C:
// "FfiHealthStatic" registered WRITABLE (`{ uint32_t hp; }`), "FfiFlagA"
// registered READ-ONLY (`{ uint32_t v; }`), and "FfiFlagB" registered
// writable but held by no entity. Three entities: {Health 1},
// {Health 2, FlagA 20}, {Health 3}. So two archetypes hold Health:
// {Health} (entities 1 and 3, two rows) and {Health, FlagA} (entity 2, one
// row). The tick is advanced once after building. Never returns NULL.
MidEcsWorld *mid_ecs_test_write_fixture_world_new(void);

#ifdef __cplusplus
}
#endif

#endif // MID_ECS_H
