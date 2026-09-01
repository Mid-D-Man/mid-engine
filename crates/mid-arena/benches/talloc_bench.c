/* talloc_bench.c
 * Samba's talloc (talloc.h) -- hierarchical reference-counted allocator,
 * a genuinely different paradigm from every arena in mid-arena's Rust
 * survey (docs/mid-arena.md): every allocation is a node in a tree, and
 * talloc_free() on a parent recursively frees every child. talloc_pool()
 * gives it arena-shaped bump allocation for children of one context, which
 * is the mode benched here for a fair comparison against the other two.
 *
 * Per talloc.h's own documented contract (quoted in this project's
 * docs/mid-arena.md, not re-derived here): freeing one child of a pool
 * does NOT give that slot's bytes back to the pool for reuse -- only
 * freeing the whole pool does. So unlike apr_pools/tsoding-arena, a
 * per-item talloc_free() is benched here too, honestly labeled as "runs
 * the free, does not reclaim space" rather than skipped.
 *
 * Build: see bench-vs-c-arena-libs.yml (links -ltalloc)
 */

#include <talloc.h>

#include <stdio.h>
#include <stdint.h>
#include <time.h>

typedef struct {
    uint64_t a;
    uint64_t b;
} Payload;

#define N 100000
#define POOL_BYTES (N * sizeof(Payload) * 2) /* headroom for talloc's per-chunk bookkeeping */

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
    printf("Samba talloc (talloc_pool) -- real run, this sandbox, -O3 -march=native\n");
    printf("N = %d, single core\n\n", N);

    void *pool = talloc_pool(NULL, POOL_BYTES);
    if (!pool) {
        fprintf(stderr, "talloc_pool failed\n");
        return 1;
    }

    static Payload *ptrs[N];

    long long t0 = ns_now();
    for (int i = 0; i < N; i++) {
        Payload *p = (Payload *)talloc_size(pool, sizeof(Payload));
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

    /* Per-item free -- runs, but per talloc_pool's documented contract
     * does not reclaim the pool's bump pointer (see this file's header
     * comment). Benched anyway rather than assumed, since the *call
     * overhead* itself is still a real, comparable number even though
     * the space isn't reused the way arena_reset()/apr_pool_clear() do. */
    t0 = ns_now();
    for (int i = 0; i < N; i += 2) {
        talloc_free(ptrs[i]);
    }
    report("free_half_no_reclaim", ns_now() - t0, N / 2);

    /* Whole-pool free -- the actual reclaim path for a talloc pool. */
    t0 = ns_now();
    talloc_free(pool);
    report("free_whole_pool", ns_now() - t0, 1);

    printf("\n(sink=%llu, prevents dead-code elimination)\n", (unsigned long long)sink);
    return 0;
}
