// crates/mid-platform/ffi-smoke-test/mid_platform.h
//
// Hand-written to match crates/mid-platform/src/ffi.rs exactly -- not
// auto-generated (no cbindgen dependency added for this pass, matching
// mid-net's/mid-ecs's own ffi-smoke-test header convention). If ffi.rs's
// signatures change, this needs updating by hand alongside it. Verified
// against the real compiled library, not just written to match the Rust
// source by eye: see test.c, run against both libmid_platform.so and
// libmid_platform.a with real gcc.
//
// Scope: Once and Barrier only -- Mutex/RwLock/OnceLock/LazyLock all need
// an opaque guard handle design (something a C caller explicitly unlocks,
// since C has no destructors) that hasn't been built yet. See ffi.rs's own
// doc comment for the full reasoning.
#ifndef MID_PLATFORM_H
#define MID_PLATFORM_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// --- Status codes returned by functions below (see ffi.rs's MidPlatformStatus) ---
#define MID_PLATFORM_OK              0
#define MID_PLATFORM_NULL_POINTER   -1
#define MID_PLATFORM_INTERNAL_PANIC -2

// --- Barrier ---

// Opaque handle -- always heap-allocated by this library. Every handle
// returned by mid_platform_barrier_new must be freed with
// mid_platform_barrier_free exactly once.
typedef struct MidPlatformBarrier MidPlatformBarrier;

// Creates a new barrier that blocks num_threads calls to
// mid_platform_barrier_wait before releasing all of them at once. Never
// returns NULL.
MidPlatformBarrier *mid_platform_barrier_new(size_t num_threads);

// Frees a handle returned by mid_platform_barrier_new. NULL is a safe
// no-op. Must not be called while another thread may still be blocked
// inside mid_platform_barrier_wait on this same handle.
void mid_platform_barrier_free(MidPlatformBarrier *barrier);

// Blocks until num_threads total calls to this function (across however
// many threads) have been made on this barrier, then releases all of them
// at once. Returns true for exactly one (arbitrary) call per rendezvous
// -- the "leader" -- false for every other. Reusable: calling this again
// after a full rendezvous works the same way a second time. Returns false
// on a NULL barrier, without blocking.
bool mid_platform_barrier_wait(const MidPlatformBarrier *barrier);

// --- Once ---

// Opaque handle -- always heap-allocated by this library. Every handle
// returned by mid_platform_once_new must be freed with
// mid_platform_once_free exactly once.
typedef struct MidPlatformOnce MidPlatformOnce;

// Callback type for mid_platform_once_call.
typedef void (*MidPlatformOnceFn)(void *ctx);

// Creates a new Once, not yet completed. Never returns NULL.
MidPlatformOnce *mid_platform_once_new(void);

// Frees a handle returned by mid_platform_once_new. NULL is a safe no-op.
void mid_platform_once_free(MidPlatformOnce *once);

// Calls f(ctx) exactly once across every call to this function made on
// this once, however many times (or from however many threads) it is
// called. Does not report whether *this* call was the one that ran f --
// neither does the underlying Rust method. Returns MID_PLATFORM_NULL_POINTER
// on a NULL once or a NULL f, MID_PLATFORM_INTERNAL_PANIC if f (or
// anything it calls back into on the Rust side) panics, MID_PLATFORM_OK
// otherwise.
int32_t mid_platform_once_call(MidPlatformOnce *once, MidPlatformOnceFn f, void *ctx);

// Returns true if mid_platform_once_call has completed successfully at
// least once on this once. false on a NULL once.
bool mid_platform_once_is_completed(const MidPlatformOnce *once);

#ifdef __cplusplus
}
#endif

#endif // MID_PLATFORM_H
