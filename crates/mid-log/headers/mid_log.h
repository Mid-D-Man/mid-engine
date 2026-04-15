#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Tier constants */
#define MID_TIER_LOW  0   /* Engine internals — physics, net, ECS  */
#define MID_TIER_MID  1   /* Engine-adjacent — scripting, tools     */
#define MID_TIER_HIGH 2   /* Gameplay logic — player, AI, events    */

uint8_t mid_log_init(void);
void    mid_log_shutdown(void);

void mid_log_trace_c(uint8_t tier, const char *msg);
void mid_log_info_c (uint8_t tier, const char *msg);
void mid_log_warn_c (uint8_t tier, const char *msg);
void mid_log_error_c(uint8_t tier, const char *msg);
void mid_log_fatal_c(uint8_t tier, const char *msg);

#ifdef __cplusplus
}
#endif
