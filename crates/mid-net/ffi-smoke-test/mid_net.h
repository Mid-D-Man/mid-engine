// crates/mid-net/ffi-smoke-test/mid_net.h
//
// Hand-written to match crates/mid-net/src/ffi.rs exactly -- not
// auto-generated (no cbindgen dependency added for this). If ffi.rs's
// signatures change, this needs updating by hand alongside it. Verified
// against the real compiled library, not just written to match the
// Rust source by eye: see test.c, run against both libmid_net.so and
// libmid_net.a with real gcc.
#ifndef MID_NET_H
#define MID_NET_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// --- Status codes returned by functions below (see ffi.rs's MidNetStatus) ---
#define MID_NET_OK               0
#define MID_NET_NULL_POINTER    -1
#define MID_NET_BUFFER_TOO_SMALL -2
#define MID_NET_UNEXPECTED_END   -3
#define MID_NET_INVALID_UTF8     -4
#define MID_NET_TRAILING_BYTES   -5
#define MID_NET_INTERNAL_PANIC   -6

// --- PlayerState: unreliable channel, 128 Hz. repr(C) on the Rust side,
//     so this struct's layout matches Rust's PlayerState field-for-field. ---
typedef struct MidNetPlayerState {
    float x, y, z;
    float rot_x, rot_y, rot_z, rot_w;
} MidNetPlayerState;

// Always 28 -- provided as a function rather than a #define so it stays
// correct even if the wire format ever changes.
size_t mid_net_player_state_wire_size(void);

// Returns bytes written (>= 0) on success, or a negative MID_NET_* code.
// Pass out_buf = NULL to query the required size without writing.
int32_t mid_net_player_state_encode(const MidNetPlayerState* state, uint8_t* out_buf, size_t out_buf_len);

// Returns MID_NET_OK (0) on success.
int32_t mid_net_player_state_decode(const uint8_t* buf, size_t buf_len, MidNetPlayerState* out_state);

// --- PlayerEvent: reliable channel. Owns strings, so it's an opaque
//     handle rather than a plain struct -- always heap-allocated by
//     this library, always free with mid_net_player_event_free. ---
typedef struct MidNetPlayerEvent MidNetPlayerEvent;

// event/payload must be null-terminated UTF-8 C strings. Returns NULL
// on a null pointer or invalid UTF-8.
MidNetPlayerEvent* mid_net_player_event_new(uint32_t player_id, const char* event, const char* payload);

// NULL is a safe no-op. Double-free or use-after-free is undefined
// behavior, same as free() -- this library can't protect against it.
void mid_net_player_event_free(MidNetPlayerEvent* event);

uint32_t mid_net_player_event_get_player_id(const MidNetPlayerEvent* event);

// NOT null-terminated -- always use the _len function alongside these.
// Valid only while the handle is alive.
const uint8_t* mid_net_player_event_get_event_ptr(const MidNetPlayerEvent* event);
size_t mid_net_player_event_get_event_len(const MidNetPlayerEvent* event);
const uint8_t* mid_net_player_event_get_payload_ptr(const MidNetPlayerEvent* event);
size_t mid_net_player_event_get_payload_len(const MidNetPlayerEvent* event);

// Same NULL-to-query-size idiom as mid_net_player_state_encode.
int32_t mid_net_player_event_encode(const MidNetPlayerEvent* event, uint8_t* out_buf, size_t out_buf_len);

// Returns NULL on a null pointer or decode failure.
MidNetPlayerEvent* mid_net_player_event_decode(const uint8_t* buf, size_t buf_len);

#ifdef __cplusplus
}
#endif

#endif // MID_NET_H
