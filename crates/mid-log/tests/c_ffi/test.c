/*
 * test.c — FFI boundary smoke test.
 *
 * Proves the Rust ring buffer is callable from C.
 * If this compiles, links, runs without crashing, and prints
 * "FFI test passed." the boundary is solid.
 *
 * Build and run: ./run_test.sh
 */

#include <stdio.h>
#include <stdlib.h>
#include "../../headers/mid_log.h"

int main(void) {
    printf("=== mid-log C FFI test ===\n");

    /* Initialise */
    if (!mid_log_init()) {
        fprintf(stderr, "FAIL: mid_log_init returned 0\n");
        return 1;
    }

    /* Exercise every level from both tiers */
    mid_log_trace_c(MID_TIER_LOW,  "[C] TRACE  — engine internals");
    mid_log_info_c (MID_TIER_HIGH, "[C] INFO   — player spawned");
    mid_log_warn_c (MID_TIER_LOW,  "[C] WARN   — buffer near capacity");
    mid_log_error_c(MID_TIER_HIGH, "[C] ERROR  — non-fatal, continuing");

    /* Give the IO thread a moment to drain before shutdown */
    volatile int i = 0;
    while (i < 1000000) { i++; }   /* crude spin — real code uses usleep() */

    mid_log_shutdown();

    printf("FFI test passed.\n");
    return 0;
}
