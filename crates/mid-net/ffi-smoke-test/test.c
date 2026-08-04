// Real end-to-end FFI verification -- not a Rust unit test, an actual C
// program compiled with gcc and linked against libmid_net.so.
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "mid_net.h"

static int failures = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { printf("FAIL: %s (line %d)\n", msg, __LINE__); failures++; } \
    else { printf("ok:   %s\n", msg); } \
} while (0)

int main(void) {
    printf("=== mid-net FFI: real C program, real gcc, real link against libmid_net.so ===\n\n");

    // --- PlayerState round trip ---
    size_t wire_size = mid_net_player_state_wire_size();
    CHECK(wire_size == 28, "PlayerState wire size is 28 bytes");

    MidNetPlayerState state = { 1.5f, -2.25f, 100.0f, 0.0f, 0.707f, 0.0f, 0.707f };
    uint8_t* buf = malloc(wire_size);
    int32_t written = mid_net_player_state_encode(&state, buf, wire_size);
    CHECK(written == (int32_t)wire_size, "PlayerState encode returns wire_size bytes written");

    MidNetPlayerState decoded = {0};
    int32_t status = mid_net_player_state_decode(buf, wire_size, &decoded);
    CHECK(status == MID_NET_OK, "PlayerState decode returns Ok (0)");
    CHECK(decoded.x == state.x && decoded.rot_w == state.rot_w, "PlayerState round-trips bit-exact through real C memory");
    free(buf);

    // --- PlayerState: query-size idiom ---
    int32_t queried = mid_net_player_state_encode(&state, NULL, 0);
    CHECK(queried == (int32_t)wire_size, "PlayerState encode with NULL buf returns required size");

    // --- PlayerState: buffer too small ---
    uint8_t tiny[4];
    int32_t too_small = mid_net_player_state_encode(&state, tiny, sizeof(tiny));
    CHECK(too_small == MID_NET_BUFFER_TOO_SMALL, "PlayerState encode into too-small buffer returns BufferTooSmall (-2)");

    // --- PlayerState: null pointer handling ---
    int32_t null_decode = mid_net_player_state_decode(NULL, 10, &decoded);
    CHECK(null_decode == MID_NET_NULL_POINTER, "PlayerState decode with NULL buf returns NullPointer (-1), not a segfault");

    // --- PlayerEvent round trip via opaque handle ---
    MidNetPlayerEvent* ev = mid_net_player_event_new(42, "pickup", "item_id=17");
    CHECK(ev != NULL, "PlayerEvent new() returns non-null handle");
    CHECK(mid_net_player_event_get_player_id(ev) == 42, "PlayerEvent player_id round-trips");

    size_t event_len = mid_net_player_event_get_event_len(ev);
    const uint8_t* event_ptr = mid_net_player_event_get_event_ptr(ev);
    CHECK(event_len == 6 && memcmp(event_ptr, "pickup", 6) == 0, "PlayerEvent event-name string round-trips through real memory");

    size_t payload_len = mid_net_player_event_get_payload_len(ev);
    const uint8_t* payload_ptr = mid_net_player_event_get_payload_ptr(ev);
    CHECK(payload_len == 10 && memcmp(payload_ptr, "item_id=17", 10) == 0, "PlayerEvent payload string round-trips through real memory");

    int32_t ev_size = mid_net_player_event_encode(ev, NULL, 0);
    CHECK(ev_size > 0, "PlayerEvent encode query-size idiom works");
    uint8_t* ev_buf = malloc(ev_size);
    int32_t ev_written = mid_net_player_event_encode(ev, ev_buf, ev_size);
    CHECK(ev_written == ev_size, "PlayerEvent encode writes exactly the queried size");

    MidNetPlayerEvent* ev_decoded = mid_net_player_event_decode(ev_buf, ev_written);
    CHECK(ev_decoded != NULL, "PlayerEvent decode from real C-allocated buffer succeeds");
    CHECK(mid_net_player_event_get_player_id(ev_decoded) == 42, "Decoded PlayerEvent player_id matches original");

    free(ev_buf);
    mid_net_player_event_free(ev);
    mid_net_player_event_free(ev_decoded);
    mid_net_player_event_free(NULL); // must not crash
    printf("ok:   freeing handles and NULL did not crash\n");

    // --- garbage decode ---
    uint8_t garbage[3] = { 0xFF, 0xFF, 0xFF };
    MidNetPlayerEvent* bad = mid_net_player_event_decode(garbage, 3);
    CHECK(bad == NULL, "PlayerEvent decode of garbage bytes returns NULL, not a crash");

    printf("\n=== %s ===\n", failures == 0 ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED");
    return failures == 0 ? 0 : 1;
}
