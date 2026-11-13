// crates/mid-net/transport-quinn-ffi/smoke-test/mid_net_quic.h
//
// Hand-written to match crates/mid-net/transport-quinn-ffi/src/lib.rs exactly -- not auto-generated
// (same rule as mid_net.h: no cbindgen). If lib.rs's signatures change, update this by hand. Verified
// against the real compiled library: see test.c, run against both libmid_net_quic.so and
// libmid_net_quic.a with real gcc.
//
// What this is: a native QUIC/WebTransport connection for a C program. Datagrams (may be lost) and
// reliable messages (whole, in order, once), per connection. It does not know what the bytes mean.
// Everything is non-blocking: poll it from the game loop.
#ifndef MID_NET_QUIC_H
#define MID_NET_QUIC_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// --- Status codes (see lib.rs's MidNetQuicStatus). The first three carry the same numbers as mid_net.h. ---
#define MID_NET_QUIC_OK               0
#define MID_NET_QUIC_NULL_POINTER    -1
#define MID_NET_QUIC_BUFFER_TOO_SMALL -2
#define MID_NET_QUIC_INTERNAL_PANIC  -6
#define MID_NET_QUIC_INVALID_ADDRESS -10
#define MID_NET_QUIC_BIND_FAILED     -11
#define MID_NET_QUIC_NOT_CONNECTED   -12   // still connecting
#define MID_NET_QUIC_CLOSED          -13   // closed, or it never opened (see mid_net_quic_conn_error)
#define MID_NET_QUIC_SEND_FAILED     -14   // too big for a datagram, an empty reliable message, or the link is gone
#define MID_NET_QUIC_CERT_ERROR      -15

// --- mid_net_quic_conn_state results ---
#define MID_NET_QUIC_CONNECTING 0
#define MID_NET_QUIC_CONNECTED  1
#define MID_NET_QUIC_CLOSED_STATE 2

typedef struct MidNetQuicServer MidNetQuicServer;   // opaque: a listening server
typedef struct MidNetQuicConn   MidNetQuicConn;     // opaque: one connection to one peer

// --- Server ---
// bind_host NULL = every IPv4 address. port 0 = pick a free one (then ask mid_net_quic_server_port).
// Makes a fresh self-signed certificate. Returns 0 and writes *out, or a negative status.
int      mid_net_quic_server_open(const char *bind_host, uint16_t port, MidNetQuicServer **out);
uint16_t mid_net_quic_server_port(const MidNetQuicServer *server);   // 0 for NULL
// Takes the next client that has connected: returns 1 and writes *out, 0 when nobody is waiting, or a negative status.
int      mid_net_quic_server_accept(MidNetQuicServer *server, MidNetQuicConn **out);
void     mid_net_quic_server_close(MidNetQuicServer *server);        // connections already accepted are unaffected

// --- Client ---
// Returns at once with a handle in the CONNECTING state; poll mid_net_quic_conn_state.
// host: IPv4 address, IPv6 literal, or name. The server's certificate is NOT verified (encrypted, not authenticated).
int      mid_net_quic_connect(const char *host, uint16_t port, MidNetQuicConn **out);

// --- One connection ---
int      mid_net_quic_conn_state(MidNetQuicConn *conn);              // CONNECTING / CONNECTED / CLOSED_STATE, or a negative status
// Why it failed or closed. Writes at most cap-1 bytes and a NUL; returns the full length (0: nothing to report).
int      mid_net_quic_conn_error(MidNetQuicConn *conn, char *buf, size_t cap);

// Unreliable. Returns 0, or MID_NET_QUIC_SEND_FAILED when too big for one datagram.
int      mid_net_quic_send_datagram(MidNetQuicConn *conn, const uint8_t *data, size_t len);
// Returns the length written to buf, 0 when none is waiting, MID_NET_QUIC_BUFFER_TOO_SMALL (it is dropped), or MID_NET_QUIC_CLOSED.
int      mid_net_quic_recv_datagram(MidNetQuicConn *conn, uint8_t *buf, size_t cap);

// Reliable: whole messages, in order, once, per stream id. An empty message is refused.
int      mid_net_quic_send_reliable(MidNetQuicConn *conn, uint32_t stream_id, const uint8_t *data, size_t len);
// As recv_datagram, and *stream_id (may be NULL) gets the stream the message came on.
int      mid_net_quic_recv_reliable(MidNetQuicConn *conn, uint32_t *stream_id, uint8_t *buf, size_t cap);

// Frees the handle. With linger_ms > 0 the session stays open that long in the background so queued messages are sent.
void     mid_net_quic_conn_close(MidNetQuicConn *conn, uint32_t linger_ms);

#ifdef __cplusplus
}
#endif
#endif
