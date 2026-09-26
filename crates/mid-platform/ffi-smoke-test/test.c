// Real end-to-end FFI verification -- not a Rust unit test, an actual C
// program compiled with gcc and linked against libmid_platform.so.
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include "mid_platform.h"

static int failures = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { printf("FAIL: %s (line %d)\n", msg, __LINE__); failures++; } \
    else { printf("ok:   %s\n", msg); } \
} while (0)

// --- Once callback fixture ---
static int once_call_count = 0;
static void increment_once_call_count(void *ctx) {
    (void)ctx;
    once_call_count++;
}

// --- Barrier real-pthread fixture ---
#define BARRIER_THREADS 4
typedef struct {
    MidPlatformBarrier *barrier;
    bool is_leader;
} barrier_thread_arg;

static void *barrier_thread_fn(void *raw_arg) {
    barrier_thread_arg *arg = (barrier_thread_arg *)raw_arg;
    arg->is_leader = mid_platform_barrier_wait(arg->barrier);
    return NULL;
}

int main(void) {
    printf("=== mid-platform FFI: real C program, real gcc, real link against libmid_platform.so ===\n\n");

    // --- Once: lifecycle ---
    MidPlatformOnce *once = mid_platform_once_new();
    CHECK(once != NULL, "once_new returns non-null handle");
    CHECK(!mid_platform_once_is_completed(once), "a fresh Once is not completed");

    // --- Once: call runs exactly once ---
    for (int i = 0; i < 5; i++) {
        int32_t status = mid_platform_once_call(once, increment_once_call_count, NULL);
        CHECK(status == MID_PLATFORM_OK, "once_call returns MID_PLATFORM_OK");
    }
    CHECK(once_call_count == 1, "the callback ran exactly once across 5 calls");
    CHECK(mid_platform_once_is_completed(once), "Once reports completed after a real call");

    mid_platform_once_free(once);
    mid_platform_once_free(NULL); // documented safe no-op

    // --- Once: NULL handling ---
    MidPlatformOnce *once2 = mid_platform_once_new();
    CHECK(mid_platform_once_call(NULL, increment_once_call_count, NULL) == MID_PLATFORM_NULL_POINTER,
          "once_call on NULL once returns MID_PLATFORM_NULL_POINTER");
    CHECK(mid_platform_once_call(once2, NULL, NULL) == MID_PLATFORM_NULL_POINTER,
          "once_call with NULL fn returns MID_PLATFORM_NULL_POINTER");
    CHECK(!mid_platform_once_is_completed(NULL), "is_completed on NULL once returns false, not a crash");
    mid_platform_once_free(once2);

    // --- Barrier: lifecycle ---
    MidPlatformBarrier *solo = mid_platform_barrier_new(1);
    CHECK(solo != NULL, "barrier_new returns non-null handle");
    CHECK(mid_platform_barrier_wait(solo), "a barrier of 1 is immediately its own leader");
    mid_platform_barrier_free(solo);
    mid_platform_barrier_free(NULL); // documented safe no-op

    CHECK(!mid_platform_barrier_wait(NULL), "barrier_wait on NULL returns false, not a crash");

    // --- Barrier: real pthreads racing through the C surface ---
    MidPlatformBarrier *barrier = mid_platform_barrier_new(BARRIER_THREADS);
    pthread_t threads[BARRIER_THREADS];
    barrier_thread_arg args[BARRIER_THREADS];
    for (int i = 0; i < BARRIER_THREADS; i++) {
        args[i].barrier = barrier;
        args[i].is_leader = false;
        pthread_create(&threads[i], NULL, barrier_thread_fn, &args[i]);
    }
    int leader_count = 0;
    for (int i = 0; i < BARRIER_THREADS; i++) {
        pthread_join(threads[i], NULL);
        if (args[i].is_leader) leader_count++;
    }
    CHECK(leader_count == 1, "exactly one real pthread is leader across a real rendezvous");
    mid_platform_barrier_free(barrier);

    printf("\n=== %d check(s) failed ===\n", failures);
    return failures == 0 ? 0 : 1;
}
