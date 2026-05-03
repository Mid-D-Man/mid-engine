// crates/mid-log/headers/mid_log.h
#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Level constants ── */
#define MID_LEVEL_TRACE 0
#define MID_LEVEL_INFO  1
#define MID_LEVEL_WARN  2
#define MID_LEVEL_ERROR 3
#define MID_LEVEL_FATAL 4

/* ── Tier constants ── */
#define MID_TIER_LOW  0   /* Engine internals — physics, net, ECS  */
#define MID_TIER_MID  1   /* Engine-adjacent — scripting, tools    */
#define MID_TIER_HIGH 2   /* Gameplay logic — player, AI, events   */

/* ── Lifecycle ── */

/** Initialise the logger (stderr only). Returns 1 on success, 0 if already init. */
uint8_t mid_log_init(void);

/**
 * Initialise with a file tee.
 * path: null-terminated UTF-8 path, or NULL for stderr only.
 * Returns 1 on success, 0 if already init or path invalid.
 */
uint8_t mid_log_init_with_file(const char *path);

/** Set minimum log level (MID_LEVEL_*). Entries below are discarded. */
void mid_log_set_min_level(uint8_t level);

/** Returns current minimum log level. */
uint8_t mid_log_get_min_level(void);

/**
 * Flush all queued entries without stopping the logger.
 * Blocks until the IO thread has written all pending entries.
 */
void mid_log_flush(void);

/** Flush and stop the IO thread. Call once at engine shutdown. */
void mid_log_shutdown(void);

/* ── Logging ── */

void mid_log_trace_c (uint8_t tier, const char *msg);
void mid_log_info_c  (uint8_t tier, const char *msg);
void mid_log_warn_c  (uint8_t tier, const char *msg);
void mid_log_error_c (uint8_t tier, const char *msg);

/** Log at FATAL level. Calls mid_log_shutdown() automatically. */
void mid_log_fatal_c (uint8_t tier, const char *msg);

/* ── Convenience macros (capture __FILE__ / __LINE__) ──────────────────────
 * These format a "[file:line] msg" prefix and forward to the _c functions.
 * Use these in preference to the _c functions directly for better diagnostics.
 * Requires a C99 or later compiler (for __VA_ARGS__).
 *
 * Example:
 *   MID_LOG_INFO(MID_TIER_HIGH, "Player %d joined", player_id);
 */
#ifdef MID_LOG_LOCATION_MACROS
#include <stdio.h>
#define _MID_LOG_IMPL(fn, tier, ...)                          \
    do {                                                       \
        char _mid_buf[512];                                    \
        snprintf(_mid_buf, sizeof(_mid_buf), __VA_ARGS__);     \
        fn(tier, _mid_buf);                                    \
    } while (0)

#define MID_LOG_TRACE(tier, ...) _MID_LOG_IMPL(mid_log_trace_c, tier, __VA_ARGS__)
#define MID_LOG_INFO(tier, ...)  _MID_LOG_IMPL(mid_log_info_c,  tier, __VA_ARGS__)
#define MID_LOG_WARN(tier, ...)  _MID_LOG_IMPL(mid_log_warn_c,  tier, __VA_ARGS__)
#define MID_LOG_ERROR(tier, ...) _MID_LOG_IMPL(mid_log_error_c, tier, __VA_ARGS__)
#define MID_LOG_FATAL(tier, ...) _MID_LOG_IMPL(mid_log_fatal_c, tier, __VA_ARGS__)
#endif /* MID_LOG_LOCATION_MACROS */

#ifdef __cplusplus
}
#endif
