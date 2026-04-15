/*
 * mid_log.h — C header for the mid-log FFI surface.
 *
 * Usage:
 *   #include "mid_log.h"
 *   Link against libmid_log.dylib  (macOS)
 *               libmid_log.so      (Linux)
 *               mid_log.dll        (Windows)
 */

#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Tier constants — pass to every log call. */
#define MID_TIER_LOW  0   /* Engine internals */
#define MID_TIER_HIGH 1   /* Gameplay logic   */

/*
 * mid_log_init — initialise the logger.
 * Call once at engine startup.
 * Returns 1 on success, 0 if already initialised.
 */
uint8_t mid_log_init(void);

/*
 * mid_log_shutdown — flush remaining entries and stop the IO thread.
 * Call at engine shutdown.
 */
void mid_log_shutdown(void);

/* Logging functions. `tier` is MID_TIER_LOW or MID_TIER_HIGH. */
void mid_log_trace_c(uint8_t tier, const char *msg);
void mid_log_info_c (uint8_t tier, const char *msg);
void mid_log_warn_c (uint8_t tier, const char *msg);
void mid_log_error_c(uint8_t tier, const char *msg);
void mid_log_fatal_c(uint8_t tier, const char *msg);

#ifdef __cplusplus
}
#endif
