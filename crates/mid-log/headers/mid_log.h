// crates/mid-log/headers/mid_log.h
#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ════════════════════════════════════════════════════════════════════════════
   Constants
   ════════════════════════════════════════════════════════════════════════════ */

/* Level constants */
#define MID_LEVEL_TRACE  0
#define MID_LEVEL_INFO   1
#define MID_LEVEL_WARN   2
#define MID_LEVEL_ERROR  3
#define MID_LEVEL_FATAL  4

/* Tier constants */
#define MID_TIER_LOW     0   /* Engine internals — physics, net, ECS  */
#define MID_TIER_MID     1   /* Engine-adjacent — scripting, tools    */
#define MID_TIER_HIGH    2   /* Gameplay logic — player, AI, events   */

/* Color slot constants for mid_log_update_color_c() */
#define MID_COLOR_SLOT_TRACE      0
#define MID_COLOR_SLOT_INFO       1
#define MID_COLOR_SLOT_WARN       2
#define MID_COLOR_SLOT_ERROR      3
#define MID_COLOR_SLOT_FATAL      4
#define MID_COLOR_SLOT_TIER_LOW   5
#define MID_COLOR_SLOT_TIER_MID   6
#define MID_COLOR_SLOT_TIER_HIGH  7
#define MID_COLOR_SLOT_TIMESTAMP  8
#define MID_COLOR_SLOT_SOURCE     9
#define MID_COLOR_SLOT_MODULE    10
#define MID_COLOR_SLOT_THREAD    11
#define MID_COLOR_SLOT_FRAME     12
#define MID_COLOR_SLOT_MESSAGE   13

/* Colors init mode for mid_log_init_full_c() */
#define MID_COLORS_AUTO    (-1)   /* Auto-detect via TTY check (default) */
#define MID_COLORS_DISABLE   0    /* Always disable ANSI codes            */
#define MID_COLORS_FORCE     1    /* Always enable ANSI codes             */

/* ════════════════════════════════════════════════════════════════════════════
   Lifecycle
   ════════════════════════════════════════════════════════════════════════════ */

/**
 * Initialise with defaults (stderr only, auto-detect colors, INFO+ filtering).
 * Returns 1 on success, 0 if already initialised.
 */
uint8_t mid_log_init(void);

/**
 * Initialise with a file tee.
 * path: null-terminated UTF-8 path, or NULL for stderr only.
 * Returns 1 on success, 0 if already init or path is invalid.
 */
uint8_t mid_log_init_with_file(const char *path);

/**
 * Initialise with full configuration.
 *
 * log_file:        file path for tee output, or NULL for stderr only.
 * min_level:       MID_LEVEL_* constant. Entries below this are discarded.
 * show_timestamp:  1 = show HH:MM:SS.mmm prefix,  0 = hide.
 * show_source_loc: 1 = show file:line suffix,      0 = hide.
 * show_module:     1 = show Rust module path,       0 = hide.
 * show_thread:     1 = show [T:name] badge,         0 = hide.
 * show_frame:      1 = show [F:n] badge,            0 = hide.
 * colors:          MID_COLORS_AUTO / MID_COLORS_DISABLE / MID_COLORS_FORCE.
 *
 * Returns 1 on success, 0 if already init.
 *
 * Example — production config:
 *   mid_log_init_full_c(
 *       "game.log", MID_LEVEL_INFO,
 *       1, 0, 0, 0, 0,   // timestamp only, no source/module/thread/frame
 *       MID_COLORS_AUTO
 *   );
 */
uint8_t mid_log_init_full_c(
    const char *log_file,
    uint8_t     min_level,
    uint8_t     show_timestamp,
    uint8_t     show_source_loc,
    uint8_t     show_module,
    uint8_t     show_thread,
    uint8_t     show_frame,
    int8_t      colors
);

/**
 * Flush all queued entries without stopping the logger.
 * Blocks until the IO thread has written all pending entries.
 */
void mid_log_flush(void);

/**
 * Flush and stop the IO thread.
 * Call once at engine shutdown. Log calls after this are silently dropped.
 */
void mid_log_shutdown(void);

/* ════════════════════════════════════════════════════════════════════════════
   Level filter
   ════════════════════════════════════════════════════════════════════════════ */

/** Set minimum log level (MID_LEVEL_*). Entries below are discarded before formatting. */
void    mid_log_set_min_level(uint8_t level);

/** Returns the current minimum log level. */
uint8_t mid_log_get_min_level(void);

/* ════════════════════════════════════════════════════════════════════════════
   Colors
   ════════════════════════════════════════════════════════════════════════════ */

/**
 * Enable (1) or disable (0) ANSI color output.
 * Overrides the TTY auto-detection performed at init.
 */
void    mid_log_set_colors(uint8_t enabled);

/** Returns 1 if ANSI colors are enabled, 0 if disabled. */
uint8_t mid_log_get_colors(void);

/**
 * Update one color slot in the live color scheme.
 *
 * slot:     one of the MID_COLOR_SLOT_* constants.
 * r, g, b:  RGB components (0–255).
 * use_none: if non-zero, sets the slot to no-color (terminal default).
 *
 * Changes take effect on the IO thread's next log entry.
 *
 * Examples:
 *   // Make WARN bright orange:
 *   mid_log_update_color_c(MID_COLOR_SLOT_WARN, 255, 165, 0, 0);
 *
 *   // Remove color from message body:
 *   mid_log_update_color_c(MID_COLOR_SLOT_MESSAGE, 0, 0, 0, 1);
 *
 *   // Make errors bold bright red (255,50,50):
 *   mid_log_update_color_c(MID_COLOR_SLOT_ERROR, 255, 50, 50, 0);
 */
void mid_log_update_color_c(
    uint8_t slot,
    uint8_t r,
    uint8_t g,
    uint8_t b,
    uint8_t use_none
);

/* ════════════════════════════════════════════════════════════════════════════
   Format flags
   ════════════════════════════════════════════════════════════════════════════ */

/**
 * Set all format flags at once.
 * Each parameter: 0 = hide field, non-zero = show field.
 *
 * Example — show everything:
 *   mid_log_set_format_flags(1, 1, 1, 1, 1);
 *
 * Example — timestamp only:
 *   mid_log_set_format_flags(1, 0, 0, 0, 0);
 */
void mid_log_set_format_flags(
    uint8_t show_timestamp,
    uint8_t show_source_loc,
    uint8_t show_module,
    uint8_t show_thread,
    uint8_t show_frame
);

/* ════════════════════════════════════════════════════════════════════════════
   Frame counter
   ════════════════════════════════════════════════════════════════════════════ */

/**
 * Set the current game frame number.
 * Call once at the top of each game tick.
 *
 *   for (uint64_t frame = 0; running; ++frame) {
 *       mid_log_set_frame(frame);
 *       // ... tick ...
 *   }
 */
void     mid_log_set_frame(uint64_t n);

/** Returns the current game frame number. */
uint64_t mid_log_get_frame(void);

/* ════════════════════════════════════════════════════════════════════════════
   Rate limiting
   ════════════════════════════════════════════════════════════════════════════ */

/**
 * Configure log rate limiting.
 *
 * enabled:        0 = disable, 1 = enable (default: 1).
 * window_ms:      suppression window in milliseconds (default: 1000).
 * max_per_window: max identical entries per window before suppression (default: 5).
 *
 * Example — aggressive suppression for a physics-heavy game:
 *   mid_log_set_rate_limit(1, 500, 3);
 *
 * Example — disable entirely (e.g. during a crash investigation):
 *   mid_log_set_rate_limit(0, 1000, 5);
 */
void mid_log_set_rate_limit(
    uint8_t  enabled,
    uint32_t window_ms,
    uint32_t max_per_window
);

/* ════════════════════════════════════════════════════════════════════════════
   In-game console buffer
   ════════════════════════════════════════════════════════════════════════════ */

/**
 * Initialise the in-game console ring buffer.
 * Call BEFORE mid_log_init*() to capture all entries.
 * capacity: entries retained (minimum 8).
 */
void     mid_log_console_init(uint32_t capacity);

/** Returns the number of entries currently in the console buffer. */
uint32_t mid_log_console_count(void);

/* ════════════════════════════════════════════════════════════════════════════
   Logging functions
   ════════════════════════════════════════════════════════════════════════════ */

void mid_log_trace_c (uint8_t tier, const char *msg);
void mid_log_info_c  (uint8_t tier, const char *msg);
void mid_log_warn_c  (uint8_t tier, const char *msg);
void mid_log_error_c (uint8_t tier, const char *msg);

/** Log at FATAL. Calls mid_log_shutdown() automatically. */
void mid_log_fatal_c (uint8_t tier, const char *msg);

/* ════════════════════════════════════════════════════════════════════════════
   Convenience macros (require C99 or later)
   ════════════════════════════════════════════════════════════════════════════ */

#ifdef MID_LOG_LOCATION_MACROS
#include <stdio.h>

#define _MID_LOG_FMT(fn, tier, ...)                               \
    do {                                                           \
        char _mid_buf[1024];                                       \
        snprintf(_mid_buf, sizeof(_mid_buf), __VA_ARGS__);         \
        fn(tier, _mid_buf);                                        \
    } while (0)

#define MID_LOG_TRACE(tier, ...) _MID_LOG_FMT(mid_log_trace_c, tier, __VA_ARGS__)
#define MID_LOG_INFO(tier, ...)  _MID_LOG_FMT(mid_log_info_c,  tier, __VA_ARGS__)
#define MID_LOG_WARN(tier, ...)  _MID_LOG_FMT(mid_log_warn_c,  tier, __VA_ARGS__)
#define MID_LOG_ERROR(tier, ...) _MID_LOG_FMT(mid_log_error_c, tier, __VA_ARGS__)
#define MID_LOG_FATAL(tier, ...) _MID_LOG_FMT(mid_log_fatal_c, tier, __VA_ARGS__)

/* Soft assertion — logs ERROR on failure, continues. */
#define MID_SOFT_ASSERT(cond, tier, ...)                          \
    do {                                                           \
        if (!(cond)) {                                             \
            char _mid_buf[1024];                                   \
            snprintf(_mid_buf, sizeof(_mid_buf), __VA_ARGS__);     \
            char _mid_full[1200];                                  \
            snprintf(_mid_full, sizeof(_mid_full),                 \
                "SOFT_ASSERT FAILED: `" #cond "` — %s", _mid_buf);\
            mid_log_error_c(tier, _mid_full);                      \
        }                                                          \
    } while (0)

/* Hard assertion — logs FATAL then calls abort(). */
#include <stdlib.h>
#define MID_ASSERT(cond, tier, ...)                               \
    do {                                                           \
        if (!(cond)) {                                             \
            char _mid_buf[1024];                                   \
            snprintf(_mid_buf, sizeof(_mid_buf), __VA_ARGS__);     \
            char _mid_full[1200];                                  \
            snprintf(_mid_full, sizeof(_mid_full),                 \
                "ASSERT FAILED: `" #cond "` — %s", _mid_buf);     \
            mid_log_fatal_c(tier, _mid_full);                      \
            mid_log_flush();                                       \
            abort();                                               \
        }                                                          \
    } while (0)

#endif /* MID_LOG_LOCATION_MACROS */

#ifdef __cplusplus
}
#endif
