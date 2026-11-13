// crates/mid-net/transport-quinn-ffi/smoke-test/test.c
//
// A real C program talking QUIC to itself over loopback through mid_net_quic.h: a server, a client, datagrams and
// reliable messages both ways, a big message, errors, close. Build and run against both library forms (see the
// bottom of this file); exit code 0 only if every check passed.
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <string.h>
#include <time.h>

#include "mid_net_quic.h"

static int g_failed = 0, g_checks = 0;
#define CHECK(cond, what) do { ++g_checks; if (!(cond)) { printf("  [FAIL] %s (line %d)\n", what, __LINE__); ++g_failed; } else printf("  [pass] %s\n", what); } while (0)

static void sleep_ms(long ms) { struct timespec ts = { 0, ms * 1000000L }; nanosleep(&ts, NULL); }
static double now_ms(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6; }

// poll until recv returns > 0 (or the time runs out); returns the length, 0 on timeout, negative on an error code
static int wait_datagram(MidNetQuicConn *c, uint8_t *buf, size_t cap, int ms)
{
    const double end = now_ms() + ms;
    while (now_ms() < end) { const int n = mid_net_quic_recv_datagram(c, buf, cap); if (n != 0) return n; sleep_ms(2); }
    return 0;
}
static int wait_reliable(MidNetQuicConn *c, uint32_t *id, uint8_t *buf, size_t cap, int ms)
{
    const double end = now_ms() + ms;
    while (now_ms() < end) { const int n = mid_net_quic_recv_reliable(c, id, buf, cap); if (n != 0) return n; sleep_ms(2); }
    return 0;
}

int main(void)
{
    printf("mid_net_quic smoke test (C, real QUIC over loopback)\n");
    MidNetQuicServer *server = NULL;
    MidNetQuicConn *client = NULL, *served = NULL;

    CHECK(mid_net_quic_server_open("127.0.0.1", 0, &server) == 0 && server, "a server opens on a free port");
    const uint16_t port = mid_net_quic_server_port(server);
    CHECK(port != 0, "and says which");
    CHECK(mid_net_quic_server_accept(server, &served) == 0 && !served, "nobody waiting: accept returns 0");

    CHECK(mid_net_quic_connect("127.0.0.1", port, &client) == 0 && client, "a client gets a handle at once");
    const double end = now_ms() + 5000;
    while (now_ms() < end && mid_net_quic_conn_state(client) != MID_NET_QUIC_CONNECTED) sleep_ms(2);
    CHECK(mid_net_quic_conn_state(client) == MID_NET_QUIC_CONNECTED, "it becomes connected");
    int acc = 0;
    while (now_ms() < end && (acc = mid_net_quic_server_accept(server, &served)) == 0) sleep_ms(2);
    CHECK(acc == 1 && served, "the server accepts it");

    uint8_t up[100], down[60], buf[4096];
    for (int i = 0; i < 100; ++i) up[i] = (uint8_t)i;
    for (int i = 0; i < 60; ++i) down[i] = (uint8_t)(200 - i);
    CHECK(mid_net_quic_send_datagram(client, up, sizeof up) == 0, "a datagram goes up");
    int n = wait_datagram(served, buf, sizeof buf, 3000);
    CHECK(n == 100 && !memcmp(buf, up, 100), "and arrives whole");
    CHECK(mid_net_quic_send_datagram(served, down, sizeof down) == 0, "a datagram goes down");
    n = wait_datagram(client, buf, sizeof buf, 3000);
    CHECK(n == 60 && !memcmp(buf, down, 60), "and arrives whole");

    static uint8_t big[150000], got[200000];
    for (size_t i = 0; i < sizeof big; ++i) big[i] = (uint8_t)(i % 251);
    CHECK(mid_net_quic_send_reliable(client, 5, big, sizeof big) == 0, "a 150 KB reliable message goes up on stream 5");
    uint32_t id = 0;
    n = wait_reliable(served, &id, got, sizeof got, 5000);
    CHECK(n == (int)sizeof big && id == 5 && !memcmp(got, big, sizeof big), "and arrives in one piece, with its stream id");
    const uint8_t hello[] = "hello from the server";
    CHECK(mid_net_quic_send_reliable(served, 0, hello, sizeof hello) == 0, "a reliable message goes down");
    n = wait_reliable(client, NULL, buf, sizeof buf, 3000);
    CHECK(n == (int)sizeof hello && !memcmp(buf, hello, sizeof hello), "and arrives (stream id may be NULL)");

    CHECK(mid_net_quic_send_reliable(client, 0, hello, 0) == MID_NET_QUIC_SEND_FAILED, "an empty reliable message is refused");
    static uint8_t huge[100000];
    CHECK(mid_net_quic_send_datagram(client, huge, sizeof huge) == MID_NET_QUIC_SEND_FAILED, "a datagram too big for one is refused");
    CHECK(mid_net_quic_send_datagram(NULL, up, 4) == MID_NET_QUIC_NULL_POINTER && mid_net_quic_conn_state(NULL) == MID_NET_QUIC_NULL_POINTER, "a NULL handle gives a code");
    CHECK(mid_net_quic_connect("", 1, &client) == MID_NET_QUIC_INVALID_ADDRESS, "an empty host is refused");
    MidNetQuicServer *bad = NULL;
    CHECK(mid_net_quic_server_open("not an address", 0, &bad) == MID_NET_QUIC_INVALID_ADDRESS && !bad, "a server on a nonsense address is refused");

    // a last message with a linger still arrives, then the other end sees the close
    const uint8_t bye[] = "bye";
    CHECK(mid_net_quic_send_reliable(client, 0, bye, sizeof bye) == 0, "a last message is sent");
    mid_net_quic_conn_close(client, 400);
    n = wait_reliable(served, NULL, buf, sizeof buf, 3000);
    CHECK(n == (int)sizeof bye && !memcmp(buf, bye, sizeof bye), "closing with a linger lets it arrive");
    const double end2 = now_ms() + 5000;
    while (now_ms() < end2 && mid_net_quic_conn_state(served) != MID_NET_QUIC_CLOSED_STATE) sleep_ms(2);
    CHECK(mid_net_quic_conn_state(served) == MID_NET_QUIC_CLOSED_STATE, "and the server's end sees the connection closed");
    CHECK(mid_net_quic_send_datagram(served, up, 4) == MID_NET_QUIC_CLOSED, "sending on it says closed");

    mid_net_quic_conn_close(served, 0);
    mid_net_quic_server_close(server);
    mid_net_quic_conn_close(NULL, 0);
    mid_net_quic_server_close(NULL);

    printf("%s (%d failed of %d checks)\n", g_failed ? "FAILED" : "ALL PASSED", g_failed, g_checks);
    return g_failed ? 1 : 0;
}

// Build (from the repo root, after `cargo build -p mid-net-transport-quinn-ffi`; target dir may differ):
//   shared: gcc -Wall -Wextra -o smoke_so  crates/mid-net/transport-quinn-ffi/smoke-test/test.c -Icrates/mid-net/transport-quinn-ffi/smoke-test -Ltarget/debug -lmid_net_quic -Wl,-rpath,target/debug
//   static: gcc -Wall -Wextra -o smoke_a   crates/mid-net/transport-quinn-ffi/smoke-test/test.c -Icrates/mid-net/transport-quinn-ffi/smoke-test target/debug/libmid_net_quic.a -lpthread -ldl -lm
