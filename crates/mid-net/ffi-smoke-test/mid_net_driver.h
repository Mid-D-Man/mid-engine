// crates/mid-net/ffi-smoke-test/mid_net_driver.h
//
// Hand-written to match crates/mid-net/src/driver_ffi.rs exactly (no cbindgen), and verified by
// driver_test.c, a real C program run against the real library. If driver_ffi.rs changes, change this
// alongside it. The explanation of every function is in driver_ffi.rs and docs/mid-net-foundation.md.
//
// What this is: the driver (handshake, keepalive, channels, priorities, reliable streams) and a
// multi-client server, driven from C. The transport is YOURS: five callbacks. Messages are an id and bytes
// (your own packing); the library adds a length so it can find the end of a message it cannot read.
//
// All handles are single-threaded: call the functions of one handle from one thread at a time.
#ifndef MID_NET_DRIVER_H
#define MID_NET_DRIVER_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// ---- Status codes (MidNetDriverStatus). Success is 0; 1 means "nothing waiting". ----
#define MID_NET_ST_OK                 0
#define MID_NET_ST_NONE               1
#define MID_NET_ST_NULL_POINTER      -1
#define MID_NET_ST_BUFFER_TOO_SMALL  -2   // the message stays queued; *len says how big a buffer it needs
#define MID_NET_ST_INTERNAL_PANIC    -6
#define MID_NET_ST_BAD_ARGUMENT      -7
#define MID_NET_ST_BAD_CONFIG       -20
#define MID_NET_ST_BAD_SCHEMA       -21
#define MID_NET_ST_BAD_TRANSPORT    -22
#define MID_NET_ST_NOT_CONNECTED    -30
#define MID_NET_ST_ENDED            -31
#define MID_NET_ST_BAD_CHANNEL      -32
#define MID_NET_ST_UNREGISTERED     -33
#define MID_NET_ST_RESERVED         -34   // message id 0 belongs to the session
#define MID_NET_ST_QUEUE_FULL       -35   // a reliable queue overflowed: the connection has been ended
#define MID_NET_ST_TOO_LARGE        -36
#define MID_NET_ST_ENCODE           -37
#define MID_NET_ST_NO_SUCH_CONNECTION -38
#define MID_NET_ST_SERVER_FULL      -39
#define MID_NET_ST_DUPLICATE        -40
#define MID_NET_ST_SCHEMA_SEALED    -41
#define MID_NET_ST_SCHEMA_NOT_SEALED -42

// ---- The transport, by callbacks. All five must be set. ----
//   send_datagram(user, data, len)            -> 0 ok, negative error
//   recv_datagram(user, buf, cap)             -> length, 0 when none waiting, negative error
//   send_reliable(user, stream_id, data, len) -> 0 ok, negative error (a piece of an ordered stream)
//   recv_reliable(user, buf, cap, stream_id)  -> bytes (any piece of the stream), 0 when none, negative error
//   is_connected(user)                        -> non-zero once the transport can carry data
typedef struct MidNetTransportVtable {
    void *user;
    int32_t (*send_datagram)(void *user, const uint8_t *data, size_t len);
    int32_t (*recv_datagram)(void *user, uint8_t *buf, size_t cap);
    int32_t (*send_reliable)(void *user, uint32_t stream_id, const uint8_t *data, size_t len);
    int32_t (*recv_reliable)(void *user, uint8_t *buf, size_t cap, uint32_t *stream_id);
    int32_t (*is_connected)(void *user);
} MidNetTransportVtable;

// ---- The schema: what both ends must agree on. Build, seal, then make drivers. ----
#define MID_NET_CHANNEL_UNRELIABLE_SEQUENCED 0   // datagrams; a message older than one delivered is dropped
#define MID_NET_CHANNEL_UNRELIABLE_UNORDERED 1   // datagrams; every message at most once, any order
#define MID_NET_CHANNEL_RELIABLE_ORDERED     2   // its own stream: every message, once, in order

typedef struct MidNetSchema MidNetSchema;

// id_bits 1..16 (8 allows 256 message types); channel_bits 1..8. NULL on a bad width.
MidNetSchema *mid_net_schema_new(uint32_t id_bits, uint32_t channel_bits);
int32_t mid_net_schema_channel(MidNetSchema *schema, uint8_t id, int32_t kind);
// Message id 0 is the session's own: do not register it.
int32_t mid_net_schema_message(MidNetSchema *schema, uint16_t id);
// control_reliable: a RELIABLE_ORDERED channel; control_datagram: an UNRELIABLE one. Both defined above.
// On an error the schema is spent: free it and start again.
int32_t mid_net_schema_seal(MidNetSchema *schema, uint8_t control_reliable, uint8_t control_datagram);
void mid_net_schema_free(MidNetSchema *schema);   // drivers made from it keep what they need
uint64_t mid_net_fnv1a64(const uint8_t *data, size_t len);   // for the schema hash of the handshake

// ---- Configuration ----
typedef struct MidNetDriverConfig {
    uint64_t schema_hash;          // compared in the handshake; two builds that differ are refused
    uint64_t ping_interval_ms;     // client keepalive (default 500)
    uint64_t no_signal_ms;         // silence before NO SIGNAL (default 1500)
    uint64_t timeout_ms;           // silence before the connection ends (default 10000)
    uint64_t connect_timeout_ms;   // how long the handshake may take (default 5000)
    uint64_t hello_retry_ms;       // 0: never (right for a reliable transport)
    uint64_t datagram_ttl_ms;      // a datagram message that waited this long is dropped (0 never)
    uint32_t mtu;                  // largest datagram (default 1200)
    uint32_t datagram_budget;      // most datagram bytes per flush, at least mtu
    uint32_t reliable_budget;      // most stream bytes per flush
    uint32_t max_queued_datagrams;
    uint32_t max_stream_pending;   // past this on a reliable channel the connection is ended
    uint32_t max_frame;            // largest reliable message
    uint32_t max_inbox;            // received messages waiting for you; while full the transport is not read
    uint32_t max_receive_per_poll;
    uint16_t tick_rate_hz;         // what a server tells its clients
    uint16_t reserved;
} MidNetDriverConfig;

int32_t mid_net_driver_config_default(MidNetDriverConfig *out);

// ---- Events ----
#define MID_NET_EVENT_CONNECTED       1   // a = session id, b = tick rate
#define MID_NET_EVENT_NO_SIGNAL       2   // a = silent for this many ms
#define MID_NET_EVENT_RECOVERED       3   // a = the gap in ms
#define MID_NET_EVENT_ENDED           4   // a = MID_NET_END_*, b = the peer's reason code where there is one
#define MID_NET_EVENT_FAULT           5   // a = MID_NET_FAULT_*, b = the stream or channel where there is one
#define MID_NET_EVENT_DROPPED         6   // a = channel << 16 | id, b = 1 oversize, 2 encode, 3 unusable
#define MID_NET_EVENT_SEND_REJECTED   7   // a = channel << 16 | id, b = status (positive)

#define MID_NET_END_LOCAL_CLOSE       1
#define MID_NET_END_PEER_CLOSED       2   // b: 0 normal, 3 server full, 6 kicked, 7 protocol error ...
#define MID_NET_END_TIMEOUT           3
#define MID_NET_END_CONNECT_TIMEOUT   4
#define MID_NET_END_REJECTED          5   // b: 1 version mismatch, 2 schema mismatch, 3 full ...
#define MID_NET_END_PROTOCOL_ERROR    6

#define MID_NET_FAULT_TRANSPORT           1
#define MID_NET_FAULT_TRANSPORT_CLOSED    2
#define MID_NET_FAULT_UNKNOWN_STREAM      3
#define MID_NET_FAULT_STREAM_OVERFLOW     4
#define MID_NET_FAULT_MALFORMED_STREAM    5
#define MID_NET_FAULT_SEND_QUEUE_FULL     6

typedef struct MidNetEvent {
    uint32_t conn;   // the connection, for a server (0 for a single driver)
    int32_t kind;
    uint64_t a;
    uint64_t b;
} MidNetEvent;

#define MID_NET_STATE_CONNECTING 0
#define MID_NET_STATE_CONNECTED  1
#define MID_NET_STATE_NO_SIGNAL  2
#define MID_NET_STATE_ENDED      3

typedef struct MidNetDriverInfo {
    double rtt_ms;
    int64_t server_time_ms;     // -1 before the first keepalive answer
    uint64_t silent_ms;
    uint32_t session_id;
    int32_t state;
    int32_t end_kind;           // MID_NET_END_* once ended, else 0
    int32_t end_peer_reason;
    uint16_t tick_rate_hz;
    uint16_t reserved;
    uint32_t messages_waiting;
    uint32_t queued_datagrams;
} MidNetDriverInfo;

typedef struct MidNetDriverStats {
    uint64_t datagrams_sent;
    uint64_t datagram_bytes_sent;
    uint64_t datagrams_received;
    uint64_t bad_datagrams;
    uint64_t duplicate_datagrams;
    uint64_t stale_messages;
    uint64_t messages_sent;
    uint64_t messages_received;
    uint64_t reliable_bytes_sent;
    uint64_t reliable_bytes_received;
    uint64_t dropped_expired;
    uint64_t dropped_queue_full;
    uint64_t dropped_unsendable;
    uint64_t dropped_early;
} MidNetDriverStats;

// ---- One connection ----
typedef struct MidNetDriver MidNetDriver;

// The client end; it says hello on the first poll. *out is the new handle (NULL on error).
int32_t mid_net_driver_client(const MidNetSchema *schema, const MidNetDriverConfig *cfg,
                              const MidNetTransportVtable *transport, const char *name,
                              uint32_t nonce, uint64_t now_ms, MidNetDriver **out);
// The server's end of one accepted connection (for more than a client or two, use MidNetServer).
int32_t mid_net_driver_server_end(const MidNetSchema *schema, const MidNetDriverConfig *cfg,
                                  const MidNetTransportVtable *transport, uint32_t session_id,
                                  uint64_t now_ms, MidNetDriver **out);
void mid_net_driver_free(MidNetDriver *driver);   // no goodbye: close() and poll() first for that

// Call every frame with a millisecond clock that never goes backwards.
int32_t mid_net_driver_poll(MidNetDriver *driver, uint64_t now_ms);
int32_t mid_net_driver_close(MidNetDriver *driver, uint8_t reason);   // 0 normal ...; goodbye goes out on the next poll
// priority: 0 lowest .. 254 highest (255 is the session's). Reliable channels ignore it.
int32_t mid_net_driver_send(MidNetDriver *driver, uint8_t channel, uint8_t priority, uint16_t id,
                            const uint8_t *data, size_t len);
// 0 and the message in buf; 1 nothing waiting; -2 buf too small (message stays queued, *len = needed).
// buf may be NULL with cap 0 to ask only how large the next message is.
int32_t mid_net_driver_next_message(MidNetDriver *driver, uint8_t *channel, uint16_t *id,
                                    uint8_t *buf, size_t cap, size_t *len);
int32_t mid_net_driver_next_event(MidNetDriver *driver, MidNetEvent *out);   // 0, or 1 if none
int32_t mid_net_driver_info(MidNetDriver *driver, uint64_t now_ms, MidNetDriverInfo *out);
int32_t mid_net_driver_stats(MidNetDriver *driver, MidNetDriverStats *out);

// ---- Many connections ----
typedef struct MidNetServer MidNetServer;

int32_t mid_net_server_new(const MidNetSchema *schema, const MidNetDriverConfig *cfg,
                           uint32_t max_connections, MidNetServer **out);
void mid_net_server_free(MidNetServer *server);
// Takes over an accepted connection; its id (never 0) goes to *conn. When full the client is told
// (it ends with MID_NET_END_PEER_CLOSED, reason 3) and MID_NET_ST_SERVER_FULL is returned.
int32_t mid_net_server_accept(MidNetServer *server, const MidNetTransportVtable *transport,
                              uint64_t now_ms, uint32_t *conn);
int32_t mid_net_server_poll(MidNetServer *server, uint64_t now_ms);
int32_t mid_net_server_send(MidNetServer *server, uint32_t conn, uint8_t channel, uint8_t priority,
                            uint16_t id, const uint8_t *data, size_t len);
// To every connected client except except_conn (0: none); *sent = how many accepted it.
int32_t mid_net_server_broadcast(MidNetServer *server, uint32_t except_conn, uint8_t channel,
                                 uint8_t priority, uint16_t id, const uint8_t *data, size_t len,
                                 uint32_t *sent);
int32_t mid_net_server_kick(MidNetServer *server, uint32_t conn, uint8_t reason);
int32_t mid_net_server_next_message(MidNetServer *server, uint32_t *conn, uint8_t *channel,
                                    uint16_t *id, uint8_t *buf, size_t cap, size_t *len);
int32_t mid_net_server_next_event(MidNetServer *server, MidNetEvent *out);
int32_t mid_net_server_connection_count(MidNetServer *server);
int32_t mid_net_server_connection_info(MidNetServer *server, uint32_t conn, uint64_t now_ms,
                                       MidNetDriverInfo *out);
int32_t mid_net_server_connection_stats(MidNetServer *server, uint32_t conn, MidNetDriverStats *out);

#ifdef __cplusplus
}
#endif

#endif // MID_NET_DRIVER_H
