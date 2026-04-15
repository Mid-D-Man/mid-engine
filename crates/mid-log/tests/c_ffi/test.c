/*
 * test.c — FFI boundary smoke test for mid-log.
 *
 * Proves the Rust ring buffer is callable from plain C.
 * If this compiles, links, and exits 0 the boundary is solid.
 *
 * Build and run: ./run_test.sh
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>    /* usleep — POSIX, works on macOS and Linux */
#include "../../headers/mid_log.h"

int main(void) {
    printf("=== mid-log C FFI test ===\n");

    if (!mid_log_init()) {
        fprintf(stderr, "FAIL: mid_log_init returned 0\n");
        return 1;
    }

    mid_log_trace_c(MID_TIER_LOW,  "[C] TRACE  — engine internals");
    mid_log_info_c (MID_TIER_HIGH, "[C] INFO   — player spawned");
    mid_log_warn_c (MID_TIER_LOW,  "[C] WARN   — buffer near capacity");
    mid_log_error_c(MID_TIER_HIGH, "[C] ERROR  — non-fatal, continuing");

    /*
     * Give the IO thread time to drain the ring buffer before shutdown.
     * usleep(50000) = 50ms — more than enough for a background thread
     * to wake up and flush a handful of log entries.
     * This replaces the previous spin loop which wasted CPU for no reason.
     */
    usleep(50000);

    mid_log_shutdown();

    printf("FFI test passed.\n");
    return 0;
}
