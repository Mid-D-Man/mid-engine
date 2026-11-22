// Real end-to-end verification of the driver and server C ABI: a C program, compiled with gcc and linked
// against libmid_net, with a transport implemented here in C (an in-memory link between two ends) and
// nothing from Rust but the library. Run:
//   gcc -o driver_test driver_test.c -I . -L ../../../target/debug -lmid_net -lpthread -ldl -lm
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "mid_net_driver.h"

static int failures = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { printf("FAIL: %s (line %d)\n", msg, __LINE__); failures++; } \
    else { printf("ok:   %s\n", msg); } \
} while (0)

// ---------------------------------------------------------------------------------------------------
// An in-memory link between two ends: queues of datagrams and of reliable chunks, one pair per direction.
// ---------------------------------------------------------------------------------------------------
#define QCAP 1024
typedef struct { uint8_t data[2048]; size_t len; } Packet;
typedef struct { Packet q[QCAP]; int head, count; } PacketQueue;
typedef struct { uint32_t stream; uint8_t data[4096]; size_t len; } Chunk;
typedef struct { Chunk q[QCAP]; int head, count; } ChunkQueue;
typedef struct { PacketQueue dg[2]; ChunkQueue rel[2]; int connected; int silent; } Link;   // silent: nothing arrives (an outage)
typedef struct { Link *link; int side; } End;

static int32_t c_send_datagram(void *user, const uint8_t *data, size_t len) {
    End *e = user; PacketQueue *q = &e->link->dg[e->side];
    if (len > sizeof(q->q[0].data) || q->count == QCAP) return -1;
    Packet *p = &q->q[(q->head + q->count++) % QCAP];
    memcpy(p->data, data, len); p->len = len;
    return 0;
}
static int32_t c_recv_datagram(void *user, uint8_t *buf, size_t cap) {
    End *e = user; PacketQueue *q = &e->link->dg[1 - e->side];
    if (q->count == 0 || e->link->silent) return 0;
    Packet *p = &q->q[q->head];
    if (p->len > cap) return -2;
    memcpy(buf, p->data, p->len);
    int32_t n = (int32_t)p->len;
    q->head = (q->head + 1) % QCAP; q->count--;
    return n;
}
static int32_t c_send_reliable(void *user, uint32_t stream, const uint8_t *data, size_t len) {
    End *e = user; ChunkQueue *q = &e->link->rel[e->side];
    while (len > 0) {   // the driver may hand over more than one chunk's worth
        size_t take = len < sizeof(q->q[0].data) ? len : sizeof(q->q[0].data);
        if (q->count == QCAP) return -1;
        Chunk *c = &q->q[(q->head + q->count++) % QCAP];
        memcpy(c->data, data, take); c->len = take; c->stream = stream;
        data += take; len -= take;
    }
    return 0;
}
static int32_t c_recv_reliable(void *user, uint8_t *buf, size_t cap, uint32_t *stream) {
    End *e = user; ChunkQueue *q = &e->link->rel[1 - e->side];
    if (q->count == 0 || e->link->silent) return 0;
    Chunk *c = &q->q[q->head];
    if (c->len > cap) return -2;
    memcpy(buf, c->data, c->len);
    *stream = c->stream;
    int32_t n = (int32_t)c->len;
    q->head = (q->head + 1) % QCAP; q->count--;
    return n;
}
static int32_t c_is_connected(void *user) { End *e = user; return e->link->connected; }

static MidNetTransportVtable vt_for(End *e) {
    MidNetTransportVtable v = { e, c_send_datagram, c_recv_datagram, c_send_reliable, c_recv_reliable, c_is_connected };
    return v;
}

// ---------------------------------------------------------------------------------------------------
// The game's protocol: channels and message ids.
// ---------------------------------------------------------------------------------------------------
enum { CH_CTRL_RELIABLE = 0, CH_CTRL_DATAGRAM = 1, CH_STATE = 2, CH_EVENT = 3 };
enum { MSG_STATE = 10, MSG_EVENT = 11 };

static MidNetSchema *make_schema(void) {
    MidNetSchema *s = mid_net_schema_new(8, 4);
    mid_net_schema_channel(s, CH_CTRL_RELIABLE, MID_NET_CHANNEL_RELIABLE_ORDERED);
    mid_net_schema_channel(s, CH_CTRL_DATAGRAM, MID_NET_CHANNEL_UNRELIABLE_UNORDERED);
    mid_net_schema_channel(s, CH_STATE, MID_NET_CHANNEL_UNRELIABLE_SEQUENCED);
    mid_net_schema_channel(s, CH_EVENT, MID_NET_CHANNEL_RELIABLE_ORDERED);
    mid_net_schema_message(s, MSG_STATE);
    mid_net_schema_message(s, MSG_EVENT);
    return s;
}

static uint64_t now_ms = 0;
static void tick2(MidNetDriver *a, MidNetDriver *b, uint64_t ms) {
    uint64_t end = now_ms + ms;
    while (now_ms < end) {
        now_ms += 10;
        mid_net_driver_poll(a, now_ms);
        mid_net_driver_poll(b, now_ms);
    }
}

int main(void) {
    printf("=== mid-net driver FFI: a real C program, a transport written in C ===\n\n");

    // ---- the schema ----
    {
        MidNetSchema *bad = mid_net_schema_new(0, 4);
        CHECK(bad == NULL, "a schema with 0 id bits is refused");
        bad = mid_net_schema_new(8, 9);
        CHECK(bad == NULL, "a schema with 9 channel bits is refused");

        MidNetSchema *s = mid_net_schema_new(8, 4);
        CHECK(mid_net_schema_channel(s, 0, MID_NET_CHANNEL_RELIABLE_ORDERED) == MID_NET_ST_OK, "channel defined");
        CHECK(mid_net_schema_channel(s, 0, MID_NET_CHANNEL_RELIABLE_ORDERED) == MID_NET_ST_DUPLICATE, "same channel again is a duplicate");
        CHECK(mid_net_schema_channel(s, 99, MID_NET_CHANNEL_RELIABLE_ORDERED) == MID_NET_ST_BAD_ARGUMENT, "a channel id too wide is refused");
        CHECK(mid_net_schema_channel(s, 1, 7) == MID_NET_ST_BAD_ARGUMENT, "an unknown channel kind is refused");
        CHECK(mid_net_schema_message(s, 5) == MID_NET_ST_OK, "message registered");
        CHECK(mid_net_schema_message(s, 5) == MID_NET_ST_DUPLICATE, "same message again is a duplicate");
        CHECK(mid_net_schema_message(s, 300) == MID_NET_ST_BAD_ARGUMENT, "an id wider than 8 bits is refused");
        CHECK(mid_net_schema_channel(NULL, 1, 0) == MID_NET_ST_NULL_POINTER, "NULL schema is refused");
        MidNetDriverConfig cfg; mid_net_driver_config_default(&cfg);
        Link *l0 = calloc(1, sizeof *l0); l0->connected = 1;
        End e0 = { l0, 0 }; MidNetTransportVtable vt = vt_for(&e0);
        MidNetDriver *d = NULL;
        CHECK(mid_net_driver_client(s, &cfg, &vt, "x", 1, 0, &d) == MID_NET_ST_SCHEMA_NOT_SEALED && d == NULL, "a driver needs a sealed schema");
        CHECK(mid_net_schema_seal(s, 1, 0) == MID_NET_ST_BAD_SCHEMA, "control channels of the wrong kinds are refused at seal");
        mid_net_schema_free(s);
        free(l0);

        MidNetSchema *zero = mid_net_schema_new(8, 4);
        mid_net_schema_channel(zero, 0, MID_NET_CHANNEL_RELIABLE_ORDERED);
        mid_net_schema_channel(zero, 1, MID_NET_CHANNEL_UNRELIABLE_UNORDERED);
        mid_net_schema_message(zero, 0);
        CHECK(mid_net_schema_seal(zero, 0, 1) == MID_NET_ST_BAD_SCHEMA, "message id 0 belongs to the session");
        mid_net_schema_free(zero);

        MidNetSchema *ok = make_schema();
        CHECK(mid_net_schema_seal(ok, CH_CTRL_RELIABLE, CH_CTRL_DATAGRAM) == MID_NET_ST_OK, "a good schema seals");
        CHECK(mid_net_schema_message(ok, 40) == MID_NET_ST_SCHEMA_SEALED, "nothing can be added to a sealed schema");
        mid_net_schema_free(ok);
        const uint8_t t[] = "abc";
        CHECK(mid_net_fnv1a64(t, 3) == 0xe71fa2190541574bULL, "fnv1a64 matches the known value for \"abc\"");
    }

    // ---- a client and a server end, over the C link ----
    MidNetSchema *schema = make_schema();
    CHECK(mid_net_schema_seal(schema, CH_CTRL_RELIABLE, CH_CTRL_DATAGRAM) == MID_NET_ST_OK, "schema sealed");
    MidNetDriverConfig cfg;
    CHECK(mid_net_driver_config_default(&cfg) == MID_NET_ST_OK, "default configuration");
    CHECK(cfg.mtu == 1200 && cfg.datagram_budget == 9600 && cfg.reliable_budget == 32768, "the packet defaults are the documented ones");
    CHECK(cfg.ping_interval_ms == 500 && cfg.no_signal_ms == 1500 && cfg.timeout_ms == 10000 && cfg.connect_timeout_ms == 5000 && cfg.hello_retry_ms == 500, "the session defaults are the documented ones");
    CHECK(cfg.datagram_ttl_ms == 500 && cfg.max_queued_datagrams == 1024 && cfg.max_stream_pending == 262144 && cfg.max_frame == 16384 && cfg.max_inbox == 4096 && cfg.max_receive_per_poll == 1024 && cfg.tick_rate_hz == 30, "the queue defaults are the documented ones");
    const uint8_t sch[] = "tob-test-v1";
    cfg.schema_hash = mid_net_fnv1a64(sch, sizeof sch - 1);

    Link *link = calloc(1, sizeof *link); link->connected = 1;
    End ca = { link, 0 }, cb = { link, 1 };
    MidNetTransportVtable va = vt_for(&ca), vb = vt_for(&cb);
    MidNetDriver *client = NULL, *server = NULL;
    CHECK(mid_net_driver_client(schema, &cfg, &va, "alice", 7, 0, &client) == MID_NET_ST_OK && client, "client made");
    CHECK(mid_net_driver_server_end(schema, &cfg, &vb, 77, 0, &server) == MID_NET_ST_OK && server, "server end made");
    MidNetTransportVtable incomplete = va; incomplete.recv_reliable = NULL;
    MidNetDriver *none = NULL;
    CHECK(mid_net_driver_client(schema, &cfg, &incomplete, "x", 1, 0, &none) == MID_NET_ST_BAD_TRANSPORT && none == NULL, "a transport missing a callback is refused");
    MidNetDriverConfig badcfg = cfg; badcfg.mtu = 10;
    CHECK(mid_net_driver_client(schema, &badcfg, &va, "x", 1, 0, &none) == MID_NET_ST_BAD_CONFIG, "a bad configuration is refused");

    MidNetDriverInfo info;
    mid_net_driver_info(client, now_ms, &info);
    CHECK(info.state == MID_NET_STATE_CONNECTING, "the client starts connecting");
    CHECK(mid_net_driver_send(client, CH_STATE, 128, MSG_STATE, (const uint8_t *)"x", 1) == MID_NET_ST_NOT_CONNECTED, "sending before the handshake is refused");

    tick2(client, server, 300);
    mid_net_driver_info(client, now_ms, &info);
    CHECK(info.state == MID_NET_STATE_CONNECTED && info.session_id == 77 && info.tick_rate_hz == 30, "the client connected, with the session id and tick rate");
    mid_net_driver_info(server, now_ms, &info);
    CHECK(info.state == MID_NET_STATE_CONNECTED, "the server end connected");
    MidNetEvent ev; int connected_seen = 0;
    while (mid_net_driver_next_event(client, &ev) == MID_NET_ST_OK)
        if (ev.kind == MID_NET_EVENT_CONNECTED && ev.a == 77 && ev.b == 30) connected_seen = 1;
    CHECK(connected_seen, "a Connected event with session 77 and 30 Hz");
    CHECK(mid_net_driver_next_event(client, &ev) == MID_NET_ST_NONE, "the event queue is then empty");

    // ---- messages ----
    uint8_t big[3000]; for (int i = 0; i < 3000; i++) big[i] = (uint8_t)(i * 7);
    CHECK(mid_net_driver_send(client, CH_STATE, 128, MSG_STATE, (const uint8_t *)"hello", 5) == MID_NET_ST_OK, "a state message is accepted");
    CHECK(mid_net_driver_send(client, CH_EVENT, 0, MSG_EVENT, big, sizeof big) == MID_NET_ST_OK, "a 3000-byte reliable message is accepted");
    CHECK(mid_net_driver_send(server, CH_STATE, 128, MSG_STATE, NULL, 0) == MID_NET_ST_OK, "an empty message is accepted");
    CHECK(mid_net_driver_send(client, 9, 0, MSG_STATE, (const uint8_t *)"x", 1) == MID_NET_ST_BAD_CHANNEL, "an undefined channel is refused");
    CHECK(mid_net_driver_send(client, CH_STATE, 0, 55, (const uint8_t *)"x", 1) == MID_NET_ST_UNREGISTERED, "an unregistered id is refused");
    CHECK(mid_net_driver_send(client, CH_STATE, 0, 0, (const uint8_t *)"x", 1) == MID_NET_ST_RESERVED, "id 0 is reserved for the session");
    CHECK(mid_net_driver_send(client, CH_STATE, 0, MSG_STATE, NULL, 5) == MID_NET_ST_NULL_POINTER, "NULL data with a length is refused");
    tick2(client, server, 100);

    uint8_t channel; uint16_t id; size_t len; uint8_t buf[4096];
    int got_state = 0, got_event = 0;
    for (int i = 0; i < 2; i++) {
        memset(buf, 0, sizeof buf);
        int32_t r = mid_net_driver_next_message(server, &channel, &id, buf, 4, &len);
        if (r == MID_NET_ST_BUFFER_TOO_SMALL) {
            size_t need = len;
            CHECK(need == 3000 || need == 5, "a too-small buffer is refused and says how big the message is");
            CHECK(mid_net_driver_next_message(server, &channel, &id, NULL, 0, &len) == MID_NET_ST_BUFFER_TOO_SMALL && len == need, "asking with a NULL buffer says the size and keeps the message");
            r = mid_net_driver_next_message(server, &channel, &id, buf, sizeof buf, &len);
        }
        CHECK(r == MID_NET_ST_OK, "a message is taken");
        if (id == MSG_STATE) got_state = (channel == CH_STATE && len == 5 && memcmp(buf, "hello", 5) == 0);
        if (id == MSG_EVENT) got_event = (channel == CH_EVENT && len == 3000 && memcmp(buf, big, 3000) == 0);
    }
    CHECK(got_state, "the state message arrived intact on its channel");
    CHECK(got_event, "the 3000-byte reliable message arrived intact");
    CHECK(mid_net_driver_next_message(server, &channel, &id, buf, sizeof buf, &len) == MID_NET_ST_NONE, "then nothing is waiting");
    CHECK(mid_net_driver_next_message(client, &channel, &id, NULL, 0, &len) == MID_NET_ST_OK && id == MSG_STATE && len == 0, "the empty message came the other way, and needs no buffer");

    mid_net_driver_info(client, now_ms, &info);
    CHECK(info.rtt_ms >= 5.0 && info.rtt_ms < 100.0 && info.server_time_ms >= 0 && info.silent_ms < 100, "on a healthy link the round trip is one tick and the server clock is known");

    // ---- a message that exactly fills the buffer ----
    mid_net_driver_send(client, CH_STATE, 128, MSG_STATE, (const uint8_t *)"four", 4);
    tick2(client, server, 30);
    CHECK(mid_net_driver_next_message(server, &channel, &id, buf, 4, &len) == MID_NET_ST_OK && len == 4 && memcmp(buf, "four", 4) == 0, "a message that exactly fills the buffer is taken");

    // ---- priority: three messages in one flush go in priority order ----
    mid_net_driver_send(client, CH_STATE, 0, MSG_STATE, (const uint8_t *)"a", 1);
    mid_net_driver_send(client, CH_STATE, 0, MSG_STATE, (const uint8_t *)"b", 1);
    mid_net_driver_send(client, CH_STATE, 254, MSG_STATE, (const uint8_t *)"C", 1);
    tick2(client, server, 30);
    char order[4] = {0, 0, 0, 0};
    for (int i = 0; i < 3; i++) {
        if (mid_net_driver_next_message(server, &channel, &id, buf, sizeof buf, &len) == MID_NET_ST_OK) order[i] = (char)buf[0];
    }
    CHECK(strcmp(order, "Cab") == 0, "messages leave in priority order, first come first served within one");

    MidNetDriverStats st;
    CHECK(mid_net_driver_stats(client, &st) == MID_NET_ST_OK && st.messages_sent == 6 && st.datagrams_sent > 0 && st.reliable_bytes_sent > 3000, "the counters count");

    // ---- a clean goodbye ----
    CHECK(mid_net_driver_close(client, 3) == MID_NET_ST_OK, "close, with a reason");
    tick2(client, server, 50);
    mid_net_driver_info(client, now_ms, &info);
    CHECK(info.state == MID_NET_STATE_ENDED && info.end_kind == MID_NET_END_LOCAL_CLOSE, "the client ended by its own close");
    mid_net_driver_info(server, now_ms, &info);
    CHECK(info.state == MID_NET_STATE_ENDED && info.end_kind == MID_NET_END_PEER_CLOSED && info.end_peer_reason == 3, "the server end saw the goodbye and its reason");
    CHECK(mid_net_driver_send(client, CH_STATE, 0, MSG_STATE, (const uint8_t *)"x", 1) == MID_NET_ST_ENDED, "sending after the end is refused");
    mid_net_driver_free(client); mid_net_driver_free(server); free(link);

    // ---- an outage: NO SIGNAL, then recovery, and the connection survives ----
    {
        Link *l = calloc(1, sizeof *l); l->connected = 1;
        End a = { l, 0 }, b = { l, 1 };
        MidNetTransportVtable xa = vt_for(&a), xb = vt_for(&b);
        MidNetDriver *c = NULL, *s = NULL;
        mid_net_driver_client(schema, &cfg, &xa, "flaky", 1, now_ms, &c);
        mid_net_driver_server_end(schema, &cfg, &xb, 5, now_ms, &s);
        tick2(c, s, 300);
        while (mid_net_driver_next_event(c, &ev) == MID_NET_ST_OK) {}
        l->silent = 1;
        tick2(c, s, 1000);
        mid_net_driver_info(c, now_ms, &info);
        CHECK(info.state == MID_NET_STATE_CONNECTED, "a second of silence is not yet NO SIGNAL");
        tick2(c, s, 800);
        mid_net_driver_info(c, now_ms, &info);
        CHECK(info.state == MID_NET_STATE_NO_SIGNAL && info.silent_ms >= 1500, "after 1.5 s of silence the state is NO SIGNAL");
        int no_signal = 0, recovered = 0;
        while (mid_net_driver_next_event(c, &ev) == MID_NET_ST_OK) if (ev.kind == MID_NET_EVENT_NO_SIGNAL) no_signal = 1;
        CHECK(no_signal, "a NO SIGNAL event");
        CHECK(mid_net_driver_send(c, CH_EVENT, 0, MSG_EVENT, (const uint8_t *)"late", 4) == MID_NET_ST_OK, "sending is still accepted without a signal");
        l->silent = 0;
        tick2(c, s, 800);
        mid_net_driver_info(c, now_ms, &info);
        CHECK(info.state == MID_NET_STATE_CONNECTED, "back to connected when the link returns");
        while (mid_net_driver_next_event(c, &ev) == MID_NET_ST_OK) if (ev.kind == MID_NET_EVENT_RECOVERED && ev.a >= 1500) recovered = 1;
        CHECK(recovered, "a Recovered event with the length of the gap");
        CHECK(mid_net_driver_next_message(s, &channel, &id, buf, sizeof buf, &len) == MID_NET_ST_OK && len == 4 && memcmp(buf, "late", 4) == 0, "the message sent during the outage arrived afterwards");
        mid_net_driver_info(c, now_ms, &info);
        CHECK(info.rtt_ms >= 5.0 && info.server_time_ms >= 0, "round trip and server clock are known after an outage too (the estimate is smoothed, so it remembers the gap)");
        mid_net_driver_free(c); mid_net_driver_free(s); free(l);
    }

    // ---- a transport that is not connected yet, then is ----
    {
        Link *l = calloc(1, sizeof *l);       // not connected
        End a = { l, 0 }, b = { l, 1 };
        MidNetTransportVtable xa = vt_for(&a), xb = vt_for(&b);
        MidNetDriver *c = NULL, *s = NULL;
        mid_net_driver_client(schema, &cfg, &xa, "late", 1, now_ms, &c);
        mid_net_driver_server_end(schema, &cfg, &xb, 5, now_ms, &s);
        tick2(c, s, 500);
        mid_net_driver_info(c, now_ms, &info);
        CHECK(info.state == MID_NET_STATE_CONNECTING && l->dg[0].count == 0 && l->rel[0].count == 0, "nothing is sent while the transport is not connected");
        l->connected = 1;
        tick2(c, s, 300);
        mid_net_driver_info(c, now_ms, &info);
        CHECK(info.state == MID_NET_STATE_CONNECTED, "it connects once the transport does");
        mid_net_driver_free(c); mid_net_driver_free(s); free(l);
    }

    // ---- a schema hash that differs is refused ----
    {
        Link *l = calloc(1, sizeof *l); l->connected = 1;
        End a = { l, 0 }, b = { l, 1 };
        MidNetTransportVtable xa = vt_for(&a), xb = vt_for(&b);
        MidNetDriverConfig other = cfg; other.schema_hash = 12345;
        MidNetDriver *c = NULL, *s = NULL;
        mid_net_driver_client(schema, &cfg, &xa, "old", 1, now_ms, &c);
        mid_net_driver_server_end(schema, &other, &xb, 5, now_ms, &s);
        tick2(c, s, 300);
        mid_net_driver_info(c, now_ms, &info);
        CHECK(info.state == MID_NET_STATE_ENDED && info.end_kind == MID_NET_END_REJECTED && info.end_peer_reason == 2, "a different schema hash is refused (SchemaMismatch)");
        mid_net_driver_free(c); mid_net_driver_free(s); free(l);
    }

    // ---- the server: several clients, full, broadcast, kick ----
    {
        MidNetServer *srv = NULL;
        CHECK(mid_net_server_new(schema, &cfg, 2, &srv) == MID_NET_ST_OK && srv, "a server for two clients");
        Link *ls[3]; End cl[3], sv[3]; MidNetDriver *cd[3] = {0, 0, 0}; uint32_t conn[3] = {0, 0, 0};
        for (int i = 0; i < 3; i++) {
            ls[i] = calloc(1, sizeof(Link)); ls[i]->connected = 1;
            cl[i].link = ls[i]; cl[i].side = 0; sv[i].link = ls[i]; sv[i].side = 1;
            MidNetTransportVtable vc = vt_for(&cl[i]), vs = vt_for(&sv[i]);
            char name[8]; snprintf(name, sizeof name, "c%d", i);
            mid_net_driver_client(schema, &cfg, &vc, name, 100 + i, now_ms, &cd[i]);
            int32_t r = mid_net_server_accept(srv, &vs, now_ms, &conn[i]);
            if (i < 2) CHECK(r == MID_NET_ST_OK && conn[i] != 0, "a client is accepted");
            else CHECK(r == MID_NET_ST_SERVER_FULL, "the third client is turned away: the server is full");
        }
        for (int t = 0; t < 40; t++) {
            now_ms += 10;
            mid_net_server_poll(srv, now_ms);
            for (int i = 0; i < 3; i++) mid_net_driver_poll(cd[i], now_ms);
        }
        CHECK(mid_net_server_connection_count(srv) == 2, "two connections are held");
        mid_net_driver_info(cd[0], now_ms, &info);
        CHECK(info.state == MID_NET_STATE_CONNECTED && info.session_id == conn[0], "client 0 is connected under the server's id for it");
        mid_net_driver_info(cd[2], now_ms, &info);
        CHECK(info.state == MID_NET_STATE_ENDED && info.end_kind == MID_NET_END_PEER_CLOSED && info.end_peer_reason == 3, "the refused client learns it was because the server is full");

        mid_net_driver_send(cd[0], CH_EVENT, 0, MSG_EVENT, (const uint8_t *)"zero", 4);
        mid_net_driver_send(cd[1], CH_EVENT, 0, MSG_EVENT, (const uint8_t *)"one", 3);
        for (int t = 0; t < 10; t++) { now_ms += 10; mid_net_server_poll(srv, now_ms); for (int i = 0; i < 3; i++) mid_net_driver_poll(cd[i], now_ms); }
        uint32_t from; int seen0 = 0, seen1 = 0;
        while (mid_net_server_next_message(srv, &from, &channel, &id, buf, sizeof buf, &len) == MID_NET_ST_OK) {
            if (from == conn[0] && len == 4 && memcmp(buf, "zero", 4) == 0) seen0 = 1;
            if (from == conn[1] && len == 3 && memcmp(buf, "one", 3) == 0) seen1 = 1;
        }
        CHECK(seen0 && seen1, "the server tells which client each message came from");

        uint32_t sent = 99;
        CHECK(mid_net_server_broadcast(srv, conn[0], CH_EVENT, 0, MSG_EVENT, (const uint8_t *)"all", 3, &sent) == MID_NET_ST_OK && sent == 1, "a broadcast skips the excluded client");
        CHECK(mid_net_server_send(srv, conn[0], CH_EVENT, 0, MSG_EVENT, (const uint8_t *)"you", 3) == MID_NET_ST_OK, "a message to one client");
        CHECK(mid_net_server_send(srv, 4242, CH_EVENT, 0, MSG_EVENT, (const uint8_t *)"x", 1) == MID_NET_ST_NO_SUCH_CONNECTION, "a message to nobody is refused");
        for (int t = 0; t < 10; t++) { now_ms += 10; mid_net_server_poll(srv, now_ms); for (int i = 0; i < 3; i++) mid_net_driver_poll(cd[i], now_ms); }
        CHECK(mid_net_driver_next_message(cd[1], &channel, &id, buf, sizeof buf, &len) == MID_NET_ST_OK && len == 3 && memcmp(buf, "all", 3) == 0, "client 1 got the broadcast");
        CHECK(mid_net_driver_next_message(cd[0], &channel, &id, buf, sizeof buf, &len) == MID_NET_ST_OK && len == 3 && memcmp(buf, "you", 3) == 0, "client 0 got only the message addressed to it");
        CHECK(mid_net_driver_next_message(cd[0], &channel, &id, buf, sizeof buf, &len) == MID_NET_ST_NONE, "and not the broadcast");

        MidNetDriverInfo ci; MidNetDriverStats cs;
        CHECK(mid_net_server_connection_info(srv, conn[1], now_ms, &ci) == MID_NET_ST_OK && ci.state == MID_NET_STATE_CONNECTED, "connection info");
        CHECK(mid_net_server_connection_stats(srv, conn[1], &cs) == MID_NET_ST_OK && cs.messages_received >= 1, "connection stats");
        CHECK(mid_net_server_connection_info(srv, 4242, now_ms, &ci) == MID_NET_ST_NO_SUCH_CONNECTION, "no info for a connection that is not there");

        CHECK(mid_net_server_kick(srv, conn[1], 6) == MID_NET_ST_OK, "kick");
        CHECK(mid_net_server_kick(srv, 4242, 6) == MID_NET_ST_NO_SUCH_CONNECTION, "kicking nobody is an error");
        for (int t = 0; t < 10; t++) { now_ms += 10; mid_net_server_poll(srv, now_ms); for (int i = 0; i < 3; i++) mid_net_driver_poll(cd[i], now_ms); }
        mid_net_driver_info(cd[1], now_ms, &info);
        CHECK(info.state == MID_NET_STATE_ENDED && info.end_kind == MID_NET_END_PEER_CLOSED && info.end_peer_reason == 6, "the kicked client is told (reason 6)");
        CHECK(mid_net_server_connection_count(srv) == 1, "and its connection is gone from the server");
        MidNetEvent se; int ended_for_1 = 0;
        while (mid_net_server_next_event(srv, &se) == MID_NET_ST_OK)
            if (se.conn == conn[1] && se.kind == MID_NET_EVENT_ENDED && se.a == MID_NET_END_LOCAL_CLOSE) ended_for_1 = 1;
        CHECK(ended_for_1, "the server reports that connection's end, with its id");
        for (int i = 0; i < 3; i++) { mid_net_driver_free(cd[i]); free(ls[i]); }
        mid_net_server_free(srv);
    }

    // ---- null handles never crash ----
    CHECK(mid_net_driver_poll(NULL, 0) == MID_NET_ST_NULL_POINTER, "poll(NULL) is an error, not a crash");
    CHECK(mid_net_driver_next_event(NULL, &ev) == MID_NET_ST_NULL_POINTER, "next_event(NULL) is an error");
    CHECK(mid_net_server_poll(NULL, 0) == MID_NET_ST_NULL_POINTER, "server poll(NULL) is an error");
    mid_net_driver_free(NULL); mid_net_server_free(NULL); mid_net_schema_free(NULL);
    mid_net_schema_free(schema);

    printf("\n%s: %d failure(s)\n", failures ? "FAILED" : "all passed", failures);
    return failures ? 1 : 0;
}
