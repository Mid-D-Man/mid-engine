/* tsoding_arena_bench.c
 * Minimal bump arena (github.com/tsoding/arena, MIT, fetched at CI time --
 * see bench-vs-c-arena-libs.yml). No per-item free; arena_reset() reuses the
 * whole arena's backing regions at once. Directly comparable in shape to
 * bumpalo/typed-arena (mid-arena's own real Rust bench, docs/mid-arena.md).
 *
 * Timing: clock_gettime(CLOCK_MONOTONIC), same as cglm_bench.c. A single
 * pass per operation, not 4-way interleaved -- these are allocator calls
 * with real side effects (region growth, pointer bump) that a single
 * dependency chain already can't dead-code-eliminate the way a pure
 * arithmetic op can, unlike cglm_bench.c's mat4 multiplies.
 *
 * Build: see bench-vs-c-arena-libs.yml
 */

#define ARENA_IMPLEMENTATION
#include "arena.h"

#include <stdio.h>
#include <stdint.h>
#include <time.h>

typedef struct {
    uint64_t a;
    uint64_t b;
} Payload;

#define N 100000

static inline long long ns_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long long)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

volatile uint64_t sink = 0;

static void report(const char *op, long long ns, int n) {
    /* Matches cglm_bench.c/handmademath_bench.c's own report() line shape
     * exactly ("  %-Ns %7.2f ns/op") so scripts/bench_vs_c_arena_libs.py
     * can reuse bench_vs_c_libs.py's parse_c() unchanged. */
    printf("  %-28s %8.2f ns/op\n", op, (double)ns / n);
    fflush(stdout);
}

int main(void) {
    printf("tsoding/arena.h -- real run, this sandbox, -O3 -march=native\n");
    printf("N = %d, single core\n\n", N);

    Arena arena = {0};
    Payload *ptrs[N];

    long long t0 = ns_now();
    for (int i = 0; i < N; i++) {
        Payload *p = (Payload *)arena_alloc(&arena, sizeof(Payload));
        p->a = (uint64_t)i;
        p->b = (uint64_t)i * 2654435761ULL;
        ptrs[i] = p;
    }
    report("insert", ns_now() - t0, N);

    t0 = ns_now();
    uint64_t sum = 0;
    for (int i = 0; i < N; i++) {
        sum += ptrs[i]->a;
    }
    sink = sum;
    report("get", ns_now() - t0, N);

    /* Whole-arena reset -- the realistic reuse pattern for this kind of
     * arena (e.g. once per frame), not per-item free (there is none). */
    t0 = ns_now();
    arena_reset(&arena);
    report("reset_whole_arena", ns_now() - t0, 1);

    t0 = ns_now();
    for (int i = 0; i < N; i++) {
        Payload *p = (Payload *)arena_alloc(&arena, sizeof(Payload));
        p->a = (uint64_t)i;
        p->b = (uint64_t)i;
        ptrs[i] = p;
    }
    report("reinsert_after_reset", ns_now() - t0, N);

    arena_free(&arena);
    printf("\n(sink=%llu, prevents dead-code elimination)\n", (unsigned long long)sink);
    return 0;
}
