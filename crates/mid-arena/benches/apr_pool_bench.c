/* apr_pool_bench.c
 * Apache Portable Runtime memory pools (apr_pools.h) -- the production
 * arena allocator behind Apache HTTPD, Subversion, and much of the ASF
 * C ecosystem for decades. apr_palloc() is a bump allocation from the
 * pool's current block; apr_pool_clear() resets the whole pool for reuse
 * without giving memory back to the OS, same shape as tsoding/arena's
 * arena_reset(). No per-item free -- pools free by subtree, not by item.
 *
 * Build: see bench-vs-c-arena-libs.yml (links -lapr-1, -I/usr/include/apr-1.0
 * via `pkg-config --cflags --libs apr-1`)
 */

#include <apr_pools.h>
#include <apr_general.h>

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
    printf("Apache Portable Runtime (apr_pools.h) -- real run, this sandbox, -O3 -march=native\n");
    printf("N = %d, single core\n\n", N);

    apr_status_t rv = apr_initialize();
    if (rv != APR_SUCCESS) {
        fprintf(stderr, "apr_initialize failed\n");
        return 1;
    }

    apr_pool_t *pool = NULL;
    rv = apr_pool_create(&pool, NULL);
    if (rv != APR_SUCCESS) {
        fprintf(stderr, "apr_pool_create failed\n");
        return 1;
    }

    static Payload *ptrs[N];

    long long t0 = ns_now();
    for (int i = 0; i < N; i++) {
        Payload *p = (Payload *)apr_palloc(pool, sizeof(Payload));
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

    /* apr_pool_clear: frees every allocation and sub-pool made from this
     * pool, keeps the pool itself (and its already-grown blocks) alive for
     * reuse -- the pool-level reset pattern this allocator is built around,
     * not a per-item free. */
    t0 = ns_now();
    apr_pool_clear(pool);
    report("clear_whole_pool", ns_now() - t0, 1);

    t0 = ns_now();
    for (int i = 0; i < N; i++) {
        Payload *p = (Payload *)apr_palloc(pool, sizeof(Payload));
        p->a = (uint64_t)i;
        p->b = (uint64_t)i;
        ptrs[i] = p;
    }
    report("reinsert_after_clear", ns_now() - t0, N);

    apr_pool_destroy(pool);
    apr_terminate();

    printf("\n(sink=%llu, prevents dead-code elimination)\n", (unsigned long long)sink);
    return 0;
}
