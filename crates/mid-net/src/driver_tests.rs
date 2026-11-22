//! Tests of the driver, the server and the router, run over [`crate::sim`] so that every network
//! behaviour (loss, duplication, reordering, outages, hostile bytes) is exactly repeatable.

use std::sync::Arc;

use mid_net_transport::Transport;
use mid_net_wire::{
    impl_net_serialize, BitError, ChannelKind, MessageRegistry, NetMessage, NetMessageType,
    StreamWriter,
};

use crate::driver::*;
use crate::session::*;
use crate::sim::*;

// ----- the game's messages and channels ---------------------------------------------------------------

const CH_STATE: u8 = 2; // unreliable, sequenced
const CH_COSMETIC: u8 = 3; // unreliable, unordered
const CH_EVENT: u8 = 4; // reliable
const CH_CHAT: u8 = 5; // reliable
const RAW_ID: u16 = 20;

#[derive(Debug, Default, Clone, PartialEq)]
struct State {
    tick: u32,
    x: f32,
}
impl_net_serialize!(State { tick: raw(), x: quant(-1000.0, 1000.0, 16) });
impl NetMessageType for State {
    const ID: u16 = 10;
}

#[derive(Debug, Default, Clone, PartialEq)]
struct Event {
    n: u32,
}
impl_net_serialize!(Event { n: raw() });
impl NetMessageType for Event {
    const ID: u16 = 11;
}

#[derive(Debug, Default, Clone, PartialEq)]
struct Text {
    s: String,
}
impl_net_serialize!(Text { s: string(2000) });
impl NetMessageType for Text {
    const ID: u16 = 12;
}

/// A type that is in no schema.
#[derive(Debug, Default, Clone, PartialEq)]
struct Stranger {
    a: u8,
}
impl_net_serialize!(Stranger { a: raw() });
impl NetMessageType for Stranger {
    const ID: u16 = 99;
}

fn schema() -> Arc<Schema> {
    schema_with(CONTROL_DATAGRAM_CHANNEL)
}

fn schema_with(control_datagram: u8) -> Arc<Schema> {
    let mut reg = MessageRegistry::new(8);
    reg.register::<State>().unwrap();
    reg.register::<Event>().unwrap();
    reg.register::<Text>().unwrap();
    reg.register_raw(RAW_ID).unwrap();
    let mut ch = Schema::default_channels(4).unwrap();
    ch.define(CH_STATE, ChannelKind::UnreliableSequenced).unwrap();
    ch.define(CH_COSMETIC, ChannelKind::UnreliableUnordered).unwrap();
    ch.define(CH_EVENT, ChannelKind::ReliableOrdered).unwrap();
    ch.define(CH_CHAT, ChannelKind::ReliableOrdered).unwrap();
    Arc::new(Schema::new(reg, ch, CONTROL_RELIABLE_CHANNEL, control_datagram).unwrap())
}

// ----- a client and a server over a simulated link ------------------------------------------------------

struct Rig {
    net: SimNet,
    c: Driver<SimTransport>,
    s: Driver<SimTransport>,
    now: u64,
}

fn rig_with(sim: SimConfig, cfg: DriverConfig, seed: u64) -> Rig {
    let (net, a, b) = SimNet::pair(sim, seed);
    let sch = schema();
    let c = Driver::client(Arc::clone(&sch), cfg, a, "alice", 7, 0).unwrap();
    let s = Driver::server_end(sch, cfg, b, 1, 0).unwrap();
    Rig { net, c, s, now: 0 }
}

fn rig(sim: SimConfig) -> Rig {
    rig_with(sim, DriverConfig::default(), 42)
}

impl Rig {
    fn step(&mut self, dt: u64) {
        self.now += dt;
        self.net.set_time(self.now);
        self.c.poll(self.now);
        self.s.poll(self.now);
    }
    fn run(&mut self, total: u64, dt: u64) {
        let end = self.now + total;
        while self.now < end {
            self.step(dt);
        }
    }
    /// Runs until both ends are connected (within 5 s of simulated time).
    fn connect(&mut self) {
        let end = self.now + 5_000;
        while !(self.c.is_connected() && self.s.is_connected()) {
            assert!(self.now < end, "did not connect: {:?} / {:?}", self.c.state(), self.s.state());
            self.step(10);
        }
    }
}

fn events<T: Transport>(d: &mut Driver<T>) -> Vec<DriverEvent> {
    let mut v = Vec::new();
    while let Some(e) = d.next_event() {
        v.push(e);
    }
    v
}

fn messages<T: Transport>(d: &mut Driver<T>) -> Vec<Inbound> {
    let mut v = Vec::new();
    while let Some(m) = d.next_message() {
        v.push(m);
    }
    v
}

fn ticks<T: Transport>(d: &mut Driver<T>) -> Vec<u32> {
    messages(d)
        .into_iter()
        .filter_map(|m| m.downcast::<State>().ok())
        .map(|s| s.tick)
        .collect()
}

fn texts<T: Transport>(d: &mut Driver<T>) -> Vec<String> {
    messages(d)
        .into_iter()
        .filter_map(|m| m.downcast::<Text>().ok())
        .map(|t| t.s)
        .collect()
}

fn ev_numbers<T: Transport>(d: &mut Driver<T>) -> Vec<u32> {
    messages(d)
        .into_iter()
        .filter_map(|m| m.downcast::<Event>().ok())
        .map(|e| e.n)
        .collect()
}

fn has_session(evs: &[DriverEvent], f: impl Fn(&SessionEvent) -> bool) -> bool {
    evs.iter().any(|e| matches!(e, DriverEvent::Session(s) if f(s)))
}

fn ended(evs: &[DriverEvent]) -> Option<EndReason> {
    evs.iter().find_map(|e| match e {
        DriverEvent::Session(SessionEvent::Ended(r)) => Some(*r),
        _ => None,
    })
}

fn text(n: usize) -> Text {
    Text { s: "x".repeat(n) }
}

// ----- connecting --------------------------------------------------------------------------------------

#[test]
fn the_handshake_connects_both_ends() {
    let mut r = rig(SimConfig::default());
    assert_eq!(r.c.state(), SessionState::Connecting);
    r.connect();
    let ce = events(&mut r.c);
    let se = events(&mut r.s);
    assert!(has_session(&ce, |e| matches!(e, SessionEvent::Connected { session_id: 1, tick_rate_hz: 30 })));
    assert_eq!(ended(&ce), None);
    assert_eq!(ended(&se), None);
    assert!(r.c.is_connected() && r.s.is_connected());
    assert!(!r.c.is_ended());
}

#[test]
fn a_transport_that_connects_late_still_gets_its_handshake() {
    let mut r = rig(SimConfig::default());
    r.net.set_connected(false);
    r.run(1_000, 10);
    assert_eq!(r.c.state(), SessionState::Connecting);
    assert_eq!(r.c.stats().datagrams_sent, 0);
    r.net.set_connected(true);
    r.connect();
}

#[test]
fn a_handshake_that_never_finishes_ends_by_timeout() {
    let mut r = rig(SimConfig::default());
    r.net.set_connected(false);
    r.run(6_000, 10);
    assert_eq!(r.c.state(), SessionState::Ended(EndReason::ConnectTimeout));
    assert!(ended(&events(&mut r.c)).is_some());
}

#[test]
fn a_schema_hash_mismatch_is_refused() {
    let (net, a, b) = SimNet::pair(SimConfig::default(), 5);
    let mut ccfg = DriverConfig::default();
    ccfg.session.schema = SchemaHash(1);
    let mut scfg = DriverConfig::default();
    scfg.session.schema = SchemaHash(2);
    let sch = schema();
    let mut c = Driver::client(Arc::clone(&sch), ccfg, a, "a", 1, 0).unwrap();
    let mut s = Driver::server_end(sch, scfg, b, 1, 0).unwrap();
    for t in (0..500).step_by(10) {
        net.set_time(t);
        c.poll(t);
        s.poll(t);
    }
    assert_eq!(c.state(), SessionState::Ended(EndReason::Rejected(DisconnectReason::SchemaMismatch)));
    assert!(s.is_ended());
}

#[test]
fn the_clock_and_round_trip_time_are_measured_over_latency() {
    let mut r = rig(SimConfig {
        latency_ms: 50,
        ..SimConfig::default()
    });
    r.connect();
    r.run(3_000, 10);
    let rtt = r.c.rtt_ms();
    assert!((95.0..=125.0).contains(&rtt), "rtt {rtt}");
    let est = r.c.server_time_ms(r.now).unwrap();
    assert!(est.abs_diff(r.now) <= 12, "estimate {est} at {}", r.now);
}

// ----- sending and receiving -----------------------------------------------------------------------------

#[test]
fn messages_cross_both_ways_on_every_kind_of_channel() {
    let mut r = rig(SimConfig::default());
    r.connect();
    r.c.send_typed(CH_STATE, priority::NORMAL, State { tick: 5, x: 1.0 }).unwrap();
    r.c.send_typed(CH_COSMETIC, priority::NORMAL, Event { n: 6 }).unwrap();
    r.c.send_typed(CH_EVENT, priority::NORMAL, Event { n: 7 }).unwrap();
    r.s.send_typed(CH_STATE, priority::NORMAL, State { tick: 15, x: -1.0 }).unwrap();
    r.s.send_typed(CH_CHAT, priority::NORMAL, text(5)).unwrap();
    r.run(100, 10);
    assert_eq!(r.c.stats().messages_sent, 3);
    assert_eq!(r.s.stats().messages_received, 3);
    assert!(r.s.stats().datagrams_received >= 2 && r.s.stats().reliable_bytes_received >= 4);
    let at_server = messages(&mut r.s);
    assert_eq!(at_server.len(), 3);
    let mut seen: Vec<(u8, u16)> = at_server.iter().map(|m| (m.channel, m.msg.id())).collect();
    seen.sort();
    assert_eq!(seen, vec![(CH_STATE, State::ID), (CH_COSMETIC, Event::ID), (CH_EVENT, Event::ID)]);
    let at_client = messages(&mut r.c);
    assert_eq!(at_client.len(), 2);
    let state = at_client
        .into_iter()
        .find_map(|m| m.downcast::<State>().ok())
        .unwrap();
    assert_eq!(state.tick, 15);
    assert!((state.x + 1.0).abs() < 0.05);
}

#[test]
fn reliable_messages_arrive_once_and_in_order_on_a_terrible_link() {
    let mut r = rig(SimConfig {
        latency_ms: 40,
        jitter_ms: 40,
        loss_per_mille: 300,
        duplicate_per_mille: 100,
        mtu: 1200,
        reliable_piece_max: 7,
    });
    r.connect();
    let mut got_e = Vec::new();
    let mut got_c = Vec::new();
    for n in 0..1500u32 {
        r.c.send_typed(CH_EVENT, priority::NORMAL, Event { n }).unwrap();
        r.c.send_typed(CH_CHAT, priority::NORMAL, Event { n: 10_000 + n }).unwrap();
        r.c.send_typed(CH_STATE, priority::NORMAL, State { tick: n, x: 0.0 }).unwrap();
        r.step(10);
        for m in messages(&mut r.s) {
            match m.channel {
                CH_EVENT => got_e.push(m.downcast::<Event>().unwrap().n),
                CH_CHAT => got_c.push(m.downcast::<Event>().unwrap().n),
                _ => {}
            }
        }
    }
    r.run(2_000, 10);
    for m in messages(&mut r.s) {
        match m.channel {
            CH_EVENT => got_e.push(m.downcast::<Event>().unwrap().n),
            CH_CHAT => got_c.push(m.downcast::<Event>().unwrap().n),
            _ => {}
        }
    }
    assert_eq!(got_e, (0..1500).collect::<Vec<_>>());
    assert_eq!(got_c, (10_000..11_500).collect::<Vec<_>>());
    assert!(!r.c.is_ended() && !r.s.is_ended());
}

#[test]
fn a_sequenced_channel_never_goes_backwards_under_reordering() {
    let mut r = rig(SimConfig {
        latency_ms: 50,
        jitter_ms: 100,
        duplicate_per_mille: 100,
        ..SimConfig::default()
    });
    r.connect();
    let mut seen = Vec::new();
    for n in 0..1000u32 {
        r.c.send_typed(CH_STATE, priority::NORMAL, State { tick: n, x: 0.0 }).unwrap();
        r.step(5);
        seen.extend(ticks(&mut r.s));
    }
    r.run(500, 5);
    seen.extend(ticks(&mut r.s));
    assert!(seen.len() > 100, "only {} arrived", seen.len());
    assert!(seen.windows(2).all(|w| w[1] > w[0]), "went backwards: {:?}", seen.windows(2).find(|w| w[1] <= w[0]));
    assert!(r.s.stats().stale_messages > 0, "the link did reorder");
    assert!(r.s.stats().duplicate_datagrams > 0, "the link did duplicate");
}

#[test]
fn opaque_raw_messages_cross_on_datagram_and_stream_channels() {
    let mut r = rig(SimConfig::default());
    r.connect();
    r.c.send_raw(CH_COSMETIC, priority::NORMAL, RAW_ID, vec![1, 2, 3]).unwrap();
    r.c.send_raw(CH_EVENT, priority::NORMAL, RAW_ID, vec![]).unwrap();
    r.run(50, 10);
    let mut got: Vec<(u8, Vec<u8>)> = messages(&mut r.s)
        .into_iter()
        .map(|m| {
            let (ch, raw) = m.into_raw().unwrap();
            assert_eq!(raw.id, RAW_ID);
            (ch, raw.data)
        })
        .collect();
    got.sort();
    assert_eq!(got, vec![(CH_COSMETIC, vec![1, 2, 3]), (CH_EVENT, vec![])]);
}

// ----- priority, budget, mtu --------------------------------------------------------------------------------

fn small_budget() -> DriverConfig {
    DriverConfig {
        datagram_budget: 1200,
        datagram_ttl_ms: 0,
        ..DriverConfig::default()
    }
}

#[test]
fn a_crowded_flush_sends_the_higher_priority_first() {
    let mut r = rig_with(SimConfig::default(), small_budget(), 1);
    r.connect();
    messages(&mut r.s);
    for i in 0..4 {
        r.c.send_typed(CH_COSMETIC, priority::COSMETIC, Text { s: format!("low{i}{}", "x".repeat(290)) }).unwrap();
    }
    for i in 0..4 {
        r.c.send_typed(CH_COSMETIC, priority::HIGH, Text { s: format!("high{i}{}", "x".repeat(289)) }).unwrap();
    }
    r.step(10);
    let first: Vec<String> = texts(&mut r.s).into_iter().map(|s| s[..5].to_string()).collect();
    assert_eq!(first, vec!["high0", "high1", "high2", "high3"], "one packet's worth, the high ones");
    r.step(10);
    let second: Vec<String> = texts(&mut r.s).into_iter().map(|s| s[..4].to_string()).collect();
    assert_eq!(second, vec!["low0", "low1", "low2", "low3"]);
}

#[test]
fn equal_priorities_keep_their_order() {
    let mut r = rig(SimConfig::default());
    r.connect();
    for n in 0..100u32 {
        r.c.send_typed(CH_COSMETIC, priority::NORMAL, Event { n }).unwrap();
    }
    r.run(50, 10);
    assert_eq!(ev_numbers(&mut r.s), (0..100).collect::<Vec<_>>());
}

#[test]
fn a_priority_above_the_maximum_is_held_at_the_maximum() {
    // 255 is the session's; a game that asks for it must not be able to jump the pings or crowd the budget
    let mut r = rig_with(SimConfig::default(), small_budget(), 1);
    r.connect();
    for _ in 0..4 {
        r.c.send_typed(CH_COSMETIC, 255, text(295)).unwrap();
    }
    for _ in 0..4 {
        r.c.send_typed(CH_COSMETIC, priority::MAX, text(294)).unwrap();
    }
    r.step(10);
    let first = texts(&mut r.s);
    // all eight carry the same (maximum) priority, so they go in the order sent: the 295-byte ones first
    assert!(first.iter().all(|t| t.len() == 295), "{:?}", first.iter().map(|t| t.len()).collect::<Vec<_>>());
}

#[test]
fn a_flood_of_critical_messages_cannot_starve_the_keepalive() {
    let mut r = rig_with(SimConfig::default(), small_budget(), 1);
    r.connect();
    for _ in 0..3_000 {
        for _ in 0..30 {
            let _ = r.c.send_typed(CH_COSMETIC, priority::CRITICAL, text(200));
        }
        r.step(10);
        messages(&mut r.s);
        messages(&mut r.c);
    }
    assert!(r.c.is_connected() && r.s.is_connected());
    let sev = events(&mut r.s);
    assert!(!has_session(&sev, |e| matches!(e, SessionEvent::NoSignal { .. })), "{sev:?}");
    // the queue is always full, so each flush sends exactly one full packet of the flood; every packet
    // beyond those 3000 is a ping, which went out although the budget was spent
    let sent = r.c.stats().datagrams_sent;
    assert!(sent >= 3_000 + 50, "{sent} datagrams: the pings were starved");
}

#[test]
fn many_small_messages_share_datagrams_and_none_exceeds_the_mtu() {
    let mut r = rig(SimConfig {
        mtu: 1200,
        ..SimConfig::default()
    });
    r.connect();
    let before = r.c.stats().datagrams_sent;
    for n in 0..500u32 {
        r.c.send_typed(CH_COSMETIC, priority::NORMAL, Event { n }).unwrap();
    }
    r.step(10);
    let used = r.c.stats().datagrams_sent - before;
    assert!((2..=8).contains(&used), "{used} datagrams for 500 small messages");
    assert_eq!(r.net.stats().datagrams_too_big, 0);
    r.run(50, 10);
    assert_eq!(ev_numbers(&mut r.s), (0..500).collect::<Vec<_>>());
}

#[test]
fn a_message_that_cannot_fit_a_datagram_is_reported_and_the_rest_still_go() {
    let mut r = rig(SimConfig::default());
    r.connect();
    events(&mut r.c);
    r.c.send_typed(CH_COSMETIC, priority::NORMAL, Event { n: 1 }).unwrap();
    r.c.send_typed(CH_COSMETIC, priority::NORMAL, text(1990)).unwrap();
    r.c.send_typed(CH_COSMETIC, priority::NORMAL, Event { n: 2 }).unwrap();
    r.step(10);
    let evs = events(&mut r.c);
    assert!(evs.iter().any(|e| matches!(e, DriverEvent::Dropped { channel: CH_COSMETIC, id: 12, reason: DropReason::Oversize })), "{evs:?}");
    assert_eq!(r.c.stats().dropped_unsendable, 1);
    r.step(10);
    assert_eq!(ev_numbers(&mut r.s), vec![1, 2]);
}

#[test]
fn a_value_that_cannot_be_encoded_is_reported_and_dropped() {
    let mut r = rig(SimConfig::default());
    r.connect();
    events(&mut r.c);
    r.c.send_typed(CH_STATE, priority::NORMAL, State { tick: 1, x: f32::NAN }).unwrap();
    r.c.send_typed(CH_STATE, priority::NORMAL, State { tick: 2, x: 0.0 }).unwrap();
    r.step(10);
    let evs = events(&mut r.c);
    assert!(evs.iter().any(|e| matches!(e, DriverEvent::Dropped { id: 10, reason: DropReason::Encode(BitError::NotFinite), .. })), "{evs:?}");
    r.step(10);
    assert_eq!(ticks(&mut r.s), vec![2]);
}

#[test]
fn the_budget_limits_each_flush_and_the_rest_waits_for_the_next() {
    let mut r = rig_with(SimConfig::default(), small_budget(), 1);
    r.connect();
    messages(&mut r.s);
    for i in 0..40u32 {
        r.c.send_typed(CH_COSMETIC, priority::NORMAL, Text { s: format!("{i:03}{}", "x".repeat(277)) }).unwrap();
    }
    let mut per_flush = Vec::new();
    for _ in 0..30 {
        let before = r.c.stats().datagram_bytes_sent;
        r.step(10);
        per_flush.push(r.c.stats().datagram_bytes_sent - before);
    }
    // 280-byte messages: four to a packet, and the budget lets one packet go per flush (plus a ping now and then)
    assert!(per_flush.iter().all(|b| *b <= 1200 + 40), "{per_flush:?}");
    assert_eq!(per_flush.iter().filter(|b| **b > 1000).count(), 10, "{per_flush:?}");
    let got = texts(&mut r.s);
    assert_eq!(got.len(), 40);
    assert_eq!(got.iter().map(|t| t[..3].parse::<u32>().unwrap()).collect::<Vec<_>>(), (0..40).collect::<Vec<_>>());
}

#[test]
fn a_message_that_waits_too_long_expires() {
    let cfg = DriverConfig {
        datagram_budget: 1200,
        datagram_ttl_ms: 100,
        ..DriverConfig::default()
    };
    let mut r = rig_with(SimConfig::default(), cfg, 1);
    r.connect();
    messages(&mut r.s);
    for _ in 0..30 {
        r.c.send_typed(CH_COSMETIC, priority::LOW, text(400)).unwrap();
    }
    r.run(1_000, 10);
    let got = texts(&mut r.s).len();
    assert!((3..30).contains(&got), "{got} arrived");
    let s = r.c.stats();
    assert_eq!(s.dropped_expired as usize, 30 - got);
    assert_eq!(r.c.queued_datagrams(), 0);
}

#[test]
fn a_full_queue_drops_the_lowest_priority_and_oldest_first() {
    let cfg = DriverConfig {
        max_queued_datagrams: 4,
        ..DriverConfig::default()
    };
    let mut r = rig_with(SimConfig::default(), cfg, 1);
    r.connect();
    messages(&mut r.s);
    r.c.send_typed(CH_COSMETIC, 10, Event { n: 1 }).unwrap();
    r.c.send_typed(CH_COSMETIC, 10, Event { n: 2 }).unwrap();
    r.c.send_typed(CH_COSMETIC, 200, Event { n: 3 }).unwrap();
    r.c.send_typed(CH_COSMETIC, 200, Event { n: 4 }).unwrap();
    assert_eq!(r.c.queued_datagrams(), 4);
    r.c.send_typed(CH_COSMETIC, 200, Event { n: 5 }).unwrap(); // the oldest of the lowest (1) goes
    assert_eq!(r.c.queued_datagrams(), 4);
    r.c.send_typed(CH_COSMETIC, 5, Event { n: 6 }).unwrap(); // lower than everything queued: it is the one dropped
    assert_eq!(r.c.queued_datagrams(), 4);
    assert_eq!(r.c.stats().dropped_queue_full, 2);
    r.run(50, 10);
    assert_eq!(ev_numbers(&mut r.s), vec![3, 4, 5, 2]);
}

// ----- the network goes away ---------------------------------------------------------------------------------

#[test]
fn no_signal_is_shown_then_recovers_and_what_was_reliable_still_arrives() {
    let mut r = rig(SimConfig {
        latency_ms: 20,
        ..SimConfig::default()
    });
    r.connect();
    r.run(1_000, 10);
    events(&mut r.c);
    events(&mut r.s);
    let start = r.now;
    r.net.outage(start + 100, start + 3_000);
    r.run(150, 10);
    r.c.send_typed(CH_EVENT, priority::NORMAL, Event { n: 1 }).unwrap(); // during the outage
    r.c.send_typed(CH_STATE, priority::NORMAL, State { tick: 1, x: 0.0 }).unwrap(); // lost
    r.run(2_000, 10);
    for d in [&r.c, &r.s] {
        assert!(matches!(d.state(), SessionState::NoSignal { .. }), "{:?}", d.state());
        assert!(d.is_connected(), "no signal is still a connection");
    }
    assert_eq!(r.c.send_typed(CH_EVENT, priority::NORMAL, Event { n: 2 }), Ok(()), "sending is allowed without a signal");
    r.run(2_000, 10);
    let ce = events(&mut r.c);
    let se = events(&mut r.s);
    for evs in [&ce, &se] {
        assert!(has_session(evs, |e| matches!(e, SessionEvent::NoSignal { .. })), "{evs:?}");
        assert!(has_session(evs, |e| matches!(e, SessionEvent::Recovered { .. })), "{evs:?}");
        assert_eq!(ended(evs), None);
    }
    assert!(r.c.is_connected() && r.s.is_connected());
    assert_eq!(r.c.state(), SessionState::Connected);
    let got = messages(&mut r.s);
    assert_eq!(got.len(), 2, "the reliable ones arrive after the outage, the datagram never does");
    assert!(got.iter().all(|m| m.channel == CH_EVENT));
    let ns: Vec<u32> = got.into_iter().map(|m| m.downcast::<Event>().unwrap().n).collect();
    assert_eq!(ns, vec![1, 2]);
}

#[test]
fn an_outage_longer_than_the_timeout_ends_both_ends() {
    let mut r = rig(SimConfig::default());
    r.connect();
    r.run(500, 10);
    r.net.outage(r.now + 10, r.now + 60_000);
    r.run(11_000, 10);
    assert_eq!(r.c.state(), SessionState::Ended(EndReason::Timeout));
    assert_eq!(r.s.state(), SessionState::Ended(EndReason::Timeout));
    assert_eq!(r.c.send_typed(CH_STATE, 0, State::default()), Err(SendError::Ended));
}

#[test]
fn half_an_hour_on_a_bad_link_with_outages_never_drops_the_connection() {
    let mut r = rig(SimConfig {
        latency_ms: 60,
        jitter_ms: 50,
        loss_per_mille: 50,
        duplicate_per_mille: 20,
        mtu: 1200,
        reliable_piece_max: 300,
    });
    r.connect();
    let mut seed = 0x1234_5678_9ABC_DEF1u64;
    let mut next = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    let total = 1_800_000u64;
    for _ in 0..90 {
        let from = 5_000 + next() % (total - 20_000);
        r.net.outage(from, from + 200 + next() % 3_000);
    }
    let (mut sent, mut got) = (0u32, Vec::new());
    let mut tick = 0u32;
    while r.now < total {
        tick += 1;
        r.c.send_typed(CH_STATE, priority::HIGH, State { tick, x: 1.0 }).unwrap();
        r.s.send_typed(CH_STATE, priority::HIGH, State { tick, x: 2.0 }).unwrap();
        if tick % 10 == 0 {
            r.c.send_typed(CH_EVENT, priority::NORMAL, Event { n: sent }).unwrap();
            sent += 1;
        }
        r.step(33);
        got.extend(ev_numbers(&mut r.s));
        messages(&mut r.c);
        assert!(!r.c.is_ended() && !r.s.is_ended(), "ended at {} ms: {:?} / {:?}", r.now, r.c.state(), r.s.state());
    }
    r.run(5_000, 33);
    got.extend(ev_numbers(&mut r.s));
    assert_eq!(got, (0..sent).collect::<Vec<_>>(), "every reliable message, once, in order");
    assert!(sent > 5_000);
}

#[test]
fn the_same_seed_gives_the_same_run() {
    let run = |seed| {
        let mut r = rig_with(
            SimConfig {
                latency_ms: 30,
                jitter_ms: 30,
                loss_per_mille: 200,
                ..SimConfig::default()
            },
            DriverConfig::default(),
            seed,
        );
        r.connect();
        let mut got = Vec::new();
        for n in 0..300u32 {
            r.c.send_typed(CH_STATE, priority::NORMAL, State { tick: n, x: 0.0 }).unwrap();
            r.step(10);
            got.extend(ticks(&mut r.s));
        }
        (got, r.c.stats(), r.s.stats(), r.net.stats())
    };
    assert_eq!(run(5), run(5));
    assert_ne!(run(5).0, run(6).0);
}

// ----- faults ------------------------------------------------------------------------------------------------

#[test]
fn a_cut_transport_ends_the_connection_with_a_fault() {
    let mut r = rig(SimConfig::default());
    r.connect();
    events(&mut r.c);
    r.net.cut();
    r.step(10);
    let evs = events(&mut r.c);
    assert!(evs.contains(&DriverEvent::Fault(Fault::TransportClosed)), "{evs:?}");
    assert_eq!(ended(&evs), Some(EndReason::LocalClose));
    assert!(r.c.is_ended());
}

struct Broken {
    connected: bool,
}
impl Transport for Broken {
    type Error = &'static str;
    fn send_datagram(&mut self, _: &[u8]) -> Result<(), &'static str> {
        Err("send")
    }
    fn poll_datagram(&mut self) -> Result<Option<Vec<u8>>, &'static str> {
        Err("poll")
    }
    fn send_reliable(&mut self, _: u32, _: &[u8]) -> Result<(), &'static str> {
        Err("send reliable")
    }
    fn poll_reliable(&mut self) -> Result<Option<(u32, Vec<u8>)>, &'static str> {
        Err("poll reliable")
    }
    fn is_connected(&self) -> bool {
        self.connected
    }
}

#[test]
fn a_transport_that_returns_errors_ends_the_connection_with_a_fault() {
    let mut d = Driver::client(schema(), DriverConfig::default(), Broken { connected: true }, "x", 1, 0).unwrap();
    d.poll(0);
    d.poll(10);
    let evs = events(&mut d);
    let faults: Vec<_> = evs.iter().filter(|e| matches!(e, DriverEvent::Fault(_))).collect();
    assert_eq!(faults.len(), 1, "one fault, not one per failing call: {evs:?}");
    assert!(matches!(faults[0], DriverEvent::Fault(Fault::Transport(t)) if t.contains("poll reliable")));
    assert!(d.is_ended());
}

#[test]
fn a_transport_that_is_not_connected_is_left_alone() {
    let mut d = Driver::client(schema(), DriverConfig::default(), Broken { connected: false }, "x", 1, 0).unwrap();
    for t in 0..10 {
        d.poll(t * 10);
    }
    assert!(events(&mut d).is_empty());
    assert_eq!(d.state(), SessionState::Connecting);
}

fn raw_stream_bytes(r: &mut Rig, stream: u32, bytes: &[u8]) {
    r.c.transport_mut().send_reliable(stream, bytes).unwrap();
}

#[test]
fn garbage_on_a_reliable_stream_ends_the_connection_and_tells_the_peer() {
    let mut r = rig(SimConfig::default());
    r.connect();
    events(&mut r.s);
    // a frame that claims 60000 bytes, more than a frame may be
    raw_stream_bytes(&mut r, CH_EVENT as u32, &[0xE0, 0xD4, 0x03, 1, 2, 3]);
    r.run(100, 10);
    let se = events(&mut r.s);
    assert!(se.iter().any(|e| matches!(e, DriverEvent::Fault(Fault::MalformedStream { channel: CH_EVENT, .. }))), "{se:?}");
    assert_eq!(r.s.state(), SessionState::Ended(EndReason::LocalClose));
    let ce = events(&mut r.c);
    assert_eq!(ended(&ce), Some(EndReason::PeerClosed(DisconnectReason::ProtocolError)));
}

#[test]
fn an_unknown_message_in_a_valid_frame_is_a_fault_too() {
    let mut r = rig(SimConfig::default());
    r.connect();
    events(&mut r.s);
    raw_stream_bytes(&mut r, CH_EVENT as u32, &[2, 77, 0]); // length 2: id 77 is not registered
    r.run(50, 10);
    let se = events(&mut r.s);
    assert!(se.iter().any(|e| matches!(e, DriverEvent::Fault(Fault::MalformedStream { error: BitError::UnknownTag, .. }))), "{se:?}");
}

#[test]
fn bytes_on_a_stream_the_schema_does_not_have_end_the_connection() {
    for stream in [99u32, 300, CH_STATE as u32, 70_000, 256 + CH_EVENT as u32] {
        let mut r = rig(SimConfig::default());
        r.connect();
        events(&mut r.s);
        raw_stream_bytes(&mut r, stream, &[1]);
        r.run(50, 10);
        let se = events(&mut r.s);
        assert!(se.contains(&DriverEvent::Fault(Fault::UnknownStream(stream))), "stream {stream}: {se:?}");
        assert!(r.s.is_ended());
    }
}

#[test]
fn a_stream_that_sends_far_more_than_a_frame_without_finishing_one_is_cut_off() {
    let mut r = rig(SimConfig::default());
    r.connect();
    events(&mut r.s);
    let mut junk = vec![0x80u8; 1]; // a length that is never completed...
    junk.extend(std::iter::repeat(0x80).take(40_000));
    raw_stream_bytes(&mut r, CH_EVENT as u32, &junk);
    r.run(50, 10);
    let se = events(&mut r.s);
    assert!(se.contains(&DriverEvent::Fault(Fault::StreamOverflow { channel: CH_EVENT })), "{se:?}");
}

#[test]
fn garbage_datagrams_are_counted_and_do_not_end_or_keep_alive_a_connection() {
    let mut r = rig(SimConfig::default());
    r.connect();
    r.run(1_000, 10);
    events(&mut r.s);
    // the client goes silent; only garbage reaches the server from now on
    let mut seed = 99u64;
    let mut junk = |n: usize| -> Vec<u8> {
        // a packet number, then a count of 200 messages that cannot be in so few bytes
        let mut v = vec![n as u8, 0, 200];
        v.extend((0..n).map(|_| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 8) as u8
        }));
        v
    };
    let start = r.now;
    while r.now < start + 1_400 {
        r.now += 100;
        r.net.set_time(r.now);
        r.c.transport_mut().send_datagram(&junk(1 + (r.now as usize % 50))).unwrap();
        r.s.poll(r.now);
    }
    assert!(r.s.stats().bad_datagrams >= 5, "{:?}", r.s.stats());
    assert!(r.s.is_connected());
    while r.now < start + 12_000 {
        r.now += 100;
        r.net.set_time(r.now);
        r.c.transport_mut().send_datagram(&junk(1 + (r.now as usize % 50))).unwrap();
        r.s.poll(r.now);
    }
    // garbage is not a sign of life: the silent client has timed out
    assert_eq!(r.s.state(), SessionState::Ended(EndReason::Timeout));
}

#[test]
fn random_and_corrupted_bytes_never_panic_the_driver() {
    let mut r = rig(SimConfig::default());
    r.connect();
    let mut seed = 0xDEAD_BEEF_1234_5678u64;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    for round in 0..4_000u32 {
        // real packets and stream bytes, made by the client...
        r.c.send_typed(CH_STATE, priority::NORMAL, State { tick: round, x: 3.0 }).unwrap();
        r.c.send_typed(CH_COSMETIC, priority::NORMAL, text(1 + (round % 40) as usize)).unwrap();
        r.c.send_typed(CH_EVENT, priority::NORMAL, Event { n: round }).unwrap();
        r.now += 1;
        r.net.set_time(r.now);
        r.c.poll(r.now);
        // ...taken off the wire before the server sees them, corrupted, and put back
        let mut real = Vec::new();
        while let Some(p) = r.s.transport_mut().poll_datagram().unwrap() {
            real.push(p);
        }
        let mut real_stream = Vec::new();
        while let Some((id, p)) = r.s.transport_mut().poll_reliable().unwrap() {
            real_stream.push((id, p));
        }
        for mut p in real {
            for _ in 0..1 + next() % 3 {
                if p.is_empty() {
                    break;
                }
                let i = (next() as usize) % p.len();
                p[i] ^= 1 << (next() % 8);
            }
            if next() % 4 == 0 {
                p.truncate((next() as usize) % (p.len() + 1));
            }
            r.c.transport_mut().send_datagram(&p).unwrap();
        }
        for (id, mut p) in real_stream {
            if next() % 3 == 0 && !p.is_empty() {
                let i = (next() as usize) % p.len();
                p[i] ^= 1 << (next() % 8);
            }
            r.c.transport_mut().send_reliable(id, &p).unwrap();
        }
        // and pure noise
        let len = (next() % 200) as usize;
        let noise: Vec<u8> = (0..len).map(|_| next() as u8).collect();
        r.c.transport_mut().send_datagram(&noise).unwrap();
        r.s.poll(r.now);
        if r.s.is_ended() {
            // a corrupted stream is allowed to end the connection; start again from a clean one
            r = rig(SimConfig::default());
            r.connect();
        }
        messages(&mut r.s);
        events(&mut r.s);
    }
    // reaching here without a panic is the test
}

// ----- refusing a bad send ---------------------------------------------------------------------------------------

#[test]
fn sending_is_refused_before_the_handshake_and_for_bad_messages() {
    let mut r = rig(SimConfig::default());
    assert_eq!(r.c.send_typed(CH_STATE, 0, State::default()), Err(SendError::NotConnected));
    r.connect();
    assert_eq!(r.c.send_typed(CH_STATE, 0, Stranger::default()), Err(SendError::Unregistered));
    assert_eq!(r.c.send_typed(CH_STATE, 0, ControlMsg::default()), Err(SendError::Reserved));
    assert_eq!(r.c.send_typed(9, 0, State::default()), Err(SendError::BadChannel));
    assert_eq!(r.c.send_typed(200, 0, State::default()), Err(SendError::BadChannel));
    assert_eq!(r.c.send_typed(CH_EVENT, 0, State { tick: 0, x: f32::NAN }), Err(SendError::Encode(BitError::NotFinite)));
    r.c.close(DisconnectReason::Normal);
    assert_eq!(r.c.send_typed(CH_STATE, 0, State::default()), Err(SendError::Ended));
}

#[test]
fn a_message_bigger_than_a_frame_is_refused_on_a_reliable_channel() {
    let cfg = DriverConfig {
        max_frame: 1024,
        max_stream_pending: 4096,
        ..DriverConfig::default()
    };
    let mut r = rig_with(SimConfig::default(), cfg, 1);
    r.connect();
    assert_eq!(r.c.send_typed(CH_EVENT, 0, text(2_000)), Err(SendError::TooLarge));
    assert_eq!(r.c.send_typed(CH_EVENT, 0, text(1_000)), Ok(()));
}

#[test]
fn a_reliable_queue_that_fills_ends_the_connection() {
    let cfg = DriverConfig {
        max_frame: 1024,
        max_stream_pending: 4096,
        reliable_budget: 200,
        ..DriverConfig::default()
    };
    let mut r = rig_with(SimConfig::default(), cfg, 1);
    r.connect();
    events(&mut r.c);
    let mut result = Ok(());
    for _ in 0..50 {
        result = r.c.send_typed(CH_CHAT, 0, text(500));
        if result.is_err() {
            break;
        }
    }
    assert_eq!(result, Err(SendError::QueueFull));
    let evs = events(&mut r.c);
    assert!(evs.contains(&DriverEvent::Fault(Fault::SendQueueFull { channel: CH_CHAT })), "{evs:?}");
    assert!(r.c.is_ended());
    assert_eq!(r.c.send_typed(CH_CHAT, 0, text(5)), Err(SendError::Ended));
}

#[test]
fn the_reliable_budget_spreads_a_big_queue_over_several_flushes() {
    let cfg = DriverConfig {
        reliable_budget: 1000,
        max_stream_pending: 200_000,
        ..DriverConfig::default()
    };
    let mut r = rig_with(SimConfig::default(), cfg, 1);
    r.connect();
    for n in 0..100u32 {
        r.c.send_typed(CH_EVENT, 0, Text { s: format!("{n:03}{}", "y".repeat(197)) }).unwrap();
    }
    assert!(r.c.queued_stream_bytes(CH_EVENT).unwrap() > 20_000);
    let before = r.c.stats().reliable_bytes_sent;
    r.step(10);
    let one = r.c.stats().reliable_bytes_sent - before;
    assert!((900..=1100).contains(&one), "{one} bytes in the first flush");
    r.run(1_000, 10);
    let got = texts(&mut r.s);
    assert_eq!(got.len(), 100);
    assert_eq!(r.c.queued_stream_bytes(CH_EVENT), Some(0));
    assert_eq!(r.c.queued_stream_bytes(CH_STATE), None);
}

// ----- ordering across the handshake, backpressure -----------------------------------------------------------

/// A client talking to a hand-made server end, so that the order of what reaches it is ours to choose.
fn manual_server() -> (Driver<SimTransport>, SimTransport, SimNet, Arc<Schema>) {
    let (net, a, b) = SimNet::pair(SimConfig::default(), 3);
    let sch = schema();
    let c = Driver::client(Arc::clone(&sch), DriverConfig::default(), a, "m", 0xBEEF, 0).unwrap();
    (c, b, net, sch)
}

fn stream_frame(sch: &Schema, msg: &mut dyn NetMessage) -> Vec<u8> {
    let mut w = StreamWriter::new(4096, 1024);
    w.push_dyn(sch.registry(), msg).unwrap();
    let mut out = vec![0u8; 4096];
    let n = w.fill(&mut out);
    out.truncate(n);
    out
}

fn welcome(sch: &Schema) -> Vec<u8> {
    stream_frame(
        sch,
        &mut ControlMsg::Welcome {
            body: WelcomeBody {
                nonce: 0xBEEF,
                session_id: 5,
                server_time_ms: 0,
                tick_rate_hz: 30,
            },
        },
    )
}

#[test]
fn a_message_that_arrives_before_the_handshake_is_held_and_then_delivered() {
    let (mut c, mut srv, net, sch) = manual_server();
    c.poll(0);
    // the server's first words are an application message on another stream, then the welcome
    srv.send_reliable(CH_EVENT as u32, &stream_frame(&sch, &mut Event { n: 77 })).unwrap();
    net.set_time(10);
    c.poll(10);
    assert_eq!(c.state(), SessionState::Connecting);
    assert_eq!(c.messages_waiting(), 0, "not for the game before it is connected");
    srv.send_reliable(0, &welcome(&sch)).unwrap();
    net.set_time(20);
    c.poll(20);
    assert!(c.is_connected());
    assert_eq!(ev_numbers(&mut c), vec![77], "held, then released, not lost");
}

#[test]
fn a_flood_before_the_handshake_is_capped() {
    let (mut c, mut srv, net, sch) = manual_server();
    c.poll(0);
    for n in 0..1000u32 {
        srv.send_reliable(CH_EVENT as u32, &stream_frame(&sch, &mut Event { n })).unwrap();
    }
    net.set_time(10);
    c.poll(10);
    srv.send_reliable(0, &welcome(&sch)).unwrap();
    net.set_time(20);
    c.poll(20);
    let got = ev_numbers(&mut c);
    assert_eq!(got, (0..256).collect::<Vec<_>>());
    assert_eq!(c.stats().dropped_early, 744);
}

#[test]
fn a_full_inbox_stops_reading_instead_of_growing_and_nothing_is_lost() {
    let cfg = DriverConfig {
        max_inbox: 10,
        ..DriverConfig::default()
    };
    let mut r = rig_with(
        SimConfig {
            reliable_piece_max: 6, // a frame is 6 bytes: a read completes at most one message
            ..SimConfig::default()
        },
        cfg,
        1,
    );
    r.connect();
    for n in 0..200u32 {
        r.s.send_typed(CH_EVENT, 0, Event { n }).unwrap();
    }
    r.step(10);
    r.step(10);
    assert_eq!(r.c.messages_waiting(), 10, "reading stops exactly at the limit");
    let mut got = Vec::new();
    for _ in 0..500 {
        got.extend(ev_numbers(&mut r.c));
        r.step(10);
    }
    got.extend(ev_numbers(&mut r.c));
    assert_eq!(got, (0..200).collect::<Vec<_>>());
}

// ----- the schema and the configuration ------------------------------------------------------------------

#[test]
fn a_schema_needs_real_control_channels_and_leaves_id_zero_to_the_session() {
    let make = |reliable: u8, datagram: u8| {
        let reg = MessageRegistry::new(8);
        let ch = Schema::default_channels(4).unwrap();
        Schema::new(reg, ch, reliable, datagram).map(|_| ())
    };
    assert_eq!(make(0, 1), Ok(()));
    assert_eq!(make(1, 1), Err(SchemaError::ControlReliable(1)));
    assert_eq!(make(9, 1), Err(SchemaError::ControlReliable(9)));
    assert_eq!(make(0, 0), Err(SchemaError::ControlDatagram(0)));
    assert_eq!(make(0, 7), Err(SchemaError::ControlDatagram(7)));
    // id 0 already taken by the application
    let mut reg = MessageRegistry::new(8);
    reg.register_raw(0).unwrap();
    let ch = Schema::default_channels(4).unwrap();
    assert!(matches!(Schema::new(reg, ch, 0, 1), Err(SchemaError::Register(_))));
    let s = schema();
    assert_eq!((s.control_reliable(), s.control_datagram()), (0, 1));
    assert!(s.registry().is_registered(0) && s.channels().kind(CH_EVENT) == Some(ChannelKind::ReliableOrdered));
}

#[test]
fn a_bad_configuration_is_refused() {
    let bad = |f: &dyn Fn(&mut DriverConfig)| {
        let mut c = DriverConfig::default();
        f(&mut c);
        let (_n, a, _b) = SimNet::pair(SimConfig::default(), 1);
        matches!(Driver::client(schema(), c, a, "x", 1, 0), Err(DriverError::BadConfig(_)))
    };
    assert!(!bad(&|_| {}));
    assert!(bad(&|c| c.mtu = 63));
    assert!(!bad(&|c| c.mtu = 64));
    assert!(bad(&|c| c.datagram_budget = c.mtu - 1));
    assert!(!bad(&|c| c.datagram_budget = c.mtu));
    assert!(bad(&|c| c.reliable_budget = 0));
    assert!(!bad(&|c| c.reliable_budget = 1));
    assert!(!bad(&|c| c.max_queued_datagrams = 1));
    assert!(!bad(&|c| c.max_inbox = 1));
    assert!(!bad(&|c| c.max_receive_per_poll = 1));
    assert!(!bad(&|c| c.max_frame = 16));
    assert!(bad(&|c| c.max_queued_datagrams = 0));
    assert!(bad(&|c| c.max_inbox = 0));
    assert!(bad(&|c| c.max_receive_per_poll = 0));
    assert!(bad(&|c| c.max_frame = 15));
    assert!(bad(&|c| c.max_stream_pending = c.max_frame - 1));
    assert!(!bad(&|c| c.max_stream_pending = c.max_frame));
}

// ----- typed handlers -----------------------------------------------------------------------------------------------

#[derive(Default)]
struct Game {
    states: Vec<(u8, u32)>,
    events: Vec<u32>,
    raw: Vec<Vec<u8>>,
}

#[test]
fn the_router_calls_the_handler_of_each_type_and_hands_back_the_rest() {
    let mut r = rig(SimConfig::default());
    r.connect();
    r.c.send_typed(CH_STATE, 0, State { tick: 9, x: 0.0 }).unwrap();
    r.c.send_typed(CH_EVENT, 0, Event { n: 4 }).unwrap();
    r.c.send_typed(CH_EVENT, 0, text(3)).unwrap(); // nobody handles Text
    r.c.send_raw(CH_COSMETIC, 0, RAW_ID, vec![8, 9]).unwrap();
    r.run(50, 10);
    let mut router: Router<Game> = Router::new();
    router.on::<State>(|g, ch, s| g.states.push((ch, s.tick)));
    router.on::<Event>(|g, _, e| g.events.push(e.n));
    router.on_raw(RAW_ID, |g, _, m| g.raw.push(m.data));
    let mut game = Game::default();
    let (handled, unhandled) = router.drain(&mut r.s, &mut game);
    assert_eq!((handled, unhandled), (3, 1));
    assert_eq!(game.states, vec![(CH_STATE, 9)]);
    assert_eq!(game.events, vec![4]);
    assert_eq!(game.raw, vec![vec![8, 9]]);
    // a second handler for a type replaces the first
    router.on::<Event>(|g, _, e| g.events.push(e.n + 1000));
    r.c.send_typed(CH_EVENT, 0, Event { n: 1 }).unwrap();
    r.run(50, 10);
    router.drain(&mut r.s, &mut game);
    assert_eq!(game.events, vec![4, 1001]);
}

// ----- the server -----------------------------------------------------------------------------------------------------

struct Lobby {
    nets: Vec<SimNet>,
    server: Server<SimTransport>,
    clients: Vec<Driver<SimTransport>>,
    ids: Vec<Option<ConnId>>,
    now: u64,
}

fn lobby(n: usize, max: usize) -> Lobby {
    let sch = schema();
    let cfg = DriverConfig::default();
    let mut server = Server::new(Arc::clone(&sch), cfg, max).unwrap();
    let (mut nets, mut clients, mut ids) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..n {
        let (net, a, b) = SimNet::pair(SimConfig::default(), 10 + i as u64);
        clients.push(Driver::client(Arc::clone(&sch), cfg, a, &format!("c{i}"), 100 + i as u32, 0).unwrap());
        match server.accept(b, 0) {
            Ok(id) => ids.push(Some(id)),
            Err(e) => {
                assert_eq!(e, AcceptError::Full);
                ids.push(None);
            }
        }
        nets.push(net);
    }
    Lobby { nets, server, clients, ids, now: 0 }
}

impl Lobby {
    fn step(&mut self, dt: u64) {
        self.now += dt;
        for n in &self.nets {
            n.set_time(self.now);
        }
        self.server.poll(self.now);
        for c in self.clients.iter_mut() {
            c.poll(self.now);
        }
    }
    fn run(&mut self, ms: u64) {
        let end = self.now + ms;
        while self.now < end {
            self.step(10);
        }
    }
    fn server_events(&mut self) -> Vec<(ConnId, DriverEvent)> {
        let mut v = Vec::new();
        while let Some(e) = self.server.next_event() {
            v.push(e);
        }
        v
    }
}

#[test]
fn a_server_runs_many_clients_and_addresses_them_one_by_one_or_all() {
    let mut l = lobby(3, 8);
    l.run(500);
    assert_eq!(l.server.len(), 3);
    let ids: Vec<ConnId> = l.ids.iter().map(|i| i.unwrap()).collect();
    assert_eq!(l.server.connection_ids(), ids);
    for (i, c) in l.clients.iter_mut().enumerate() {
        assert!(c.is_connected(), "client {i}");
        let evs = events(c);
        assert!(has_session(&evs, |e| matches!(e, SessionEvent::Connected { session_id, .. } if *session_id == ids[i].0)));
    }
    for (i, c) in l.clients.iter_mut().enumerate() {
        c.send_typed(CH_EVENT, 0, Event { n: i as u32 }).unwrap();
    }
    l.run(100);
    let mut got = Vec::new();
    while let Some((id, m)) = l.server.next_message() {
        got.push((id, m.downcast::<Event>().unwrap().n));
    }
    got.sort();
    assert_eq!(got, vec![(ids[0], 0), (ids[1], 1), (ids[2], 2)]);

    let sent = l.server.broadcast(Some(ids[0]), CH_EVENT, 0, || Box::new(Event { n: 99 }));
    assert_eq!(sent, 2);
    l.server.send(ids[2], CH_EVENT, 0, Box::new(Event { n: 5 })).unwrap();
    assert_eq!(l.server.send(ConnId(999), CH_EVENT, 0, Box::new(Event { n: 5 })), Err(SendError::NoSuchConnection));
    l.run(100);
    assert!(ev_numbers(&mut l.clients[0]).is_empty());
    assert_eq!(ev_numbers(&mut l.clients[1]), vec![99]);
    assert_eq!(ev_numbers(&mut l.clients[2]), vec![99, 5]);
    assert!(l.server.connection(ids[1]).unwrap().is_connected());
    assert!(l.server.connection(ConnId(999)).is_none());
}

#[test]
fn a_broadcast_skips_a_client_that_is_not_connected_yet() {
    let sch = schema();
    let mut server = Server::new(Arc::clone(&sch), DriverConfig::default(), 4).unwrap();
    let (_n, _a, b) = SimNet::pair(SimConfig::default(), 1);
    server.accept(b, 0).unwrap();
    assert_eq!(server.broadcast(None, CH_EVENT, 0, || Box::new(Event { n: 1 })), 0);
}

#[test]
fn a_full_server_tells_the_client_why() {
    let mut l = lobby(3, 2);
    assert_eq!(l.ids.iter().filter(|i| i.is_some()).count(), 2);
    assert!(l.ids[2].is_none());
    l.run(300);
    assert_eq!(l.server.len(), 2);
    assert_eq!(l.clients[2].state(), SessionState::Ended(EndReason::PeerClosed(DisconnectReason::Full)));
    assert!(l.clients[0].is_connected() && l.clients[1].is_connected());
}

#[test]
fn a_connection_that_ends_is_forgotten_and_reported_with_its_id() {
    let mut l = lobby(3, 8);
    l.run(300);
    let ids: Vec<ConnId> = l.ids.iter().map(|i| i.unwrap()).collect();
    l.server_events();
    l.clients[1].close(DisconnectReason::Normal);
    l.run(100);
    let evs = l.server_events();
    assert!(
        evs.contains(&(ids[1], DriverEvent::Session(SessionEvent::Ended(EndReason::PeerClosed(DisconnectReason::Normal))))),
        "{evs:?}"
    );
    assert_eq!(l.server.len(), 2);
    assert!(l.server.connection(ids[1]).is_none());
    assert_eq!(l.server.connection_ids(), vec![ids[0], ids[2]]);
    assert!(!l.server.is_empty());
}

#[test]
fn a_kicked_client_is_told_and_removed() {
    let mut l = lobby(2, 8);
    l.run(300);
    let id0 = l.ids[0].unwrap();
    assert!(l.server.kick(id0, DisconnectReason::Kicked));
    assert!(!l.server.kick(ConnId(77), DisconnectReason::Kicked));
    l.run(100);
    assert_eq!(l.clients[0].state(), SessionState::Ended(EndReason::PeerClosed(DisconnectReason::Kicked)));
    assert_eq!(l.server.len(), 1);
    assert!(l.clients[1].is_connected());
}

#[test]
fn a_server_with_a_bad_configuration_is_refused() {
    let cfg = DriverConfig {
        mtu: 10,
        ..DriverConfig::default()
    };
    assert!(matches!(Server::<SimTransport>::new(schema(), cfg, 4), Err(DriverError::BadConfig(_))));
}

#[test]
fn a_server_hit_by_garbage_from_many_clients_stays_up_for_the_honest_ones() {
    let mut l = lobby(4, 8);
    l.run(300);
    // client 3 turns hostile: garbage on the stream
    l.clients[3].transport_mut().send_reliable(CH_EVENT as u32, &[0xE0, 0xD4, 0x03, 1]).unwrap();
    l.run(100);
    let ids: Vec<ConnId> = l.ids.iter().map(|i| i.unwrap()).collect();
    assert!(l.server.connection(ids[3]).is_none());
    for (id, client) in ids.iter().zip(&l.clients).take(3) {
        assert!(l.server.connection(*id).unwrap().is_connected());
        assert!(client.is_connected());
    }
    let evs = l.server_events();
    assert!(evs.iter().any(|(id, e)| *id == ids[3] && matches!(e, DriverEvent::Fault(Fault::MalformedStream { .. }))));
}


// ----- more of what a driver must get right -----------------------------------------------------------------------

#[test]
fn the_sessions_datagrams_may_use_a_sequenced_channel() {
    assert!(Schema::new(MessageRegistry::new(8), Schema::default_channels(4).map(|mut t| {
        t.define(CH_STATE, ChannelKind::UnreliableSequenced).unwrap();
        t
    }).unwrap(), 0, CH_STATE).is_ok());
    let sch = schema_with(CH_STATE);
    let (net, a, b) = SimNet::pair(SimConfig { latency_ms: 20, ..SimConfig::default() }, 1);
    let cfg = DriverConfig::default();
    let mut c = Driver::client(Arc::clone(&sch), cfg, a, "s", 1, 0).unwrap();
    let mut s = Driver::server_end(sch, cfg, b, 1, 0).unwrap();
    for t in (0..3_000).step_by(10) {
        net.set_time(t);
        c.poll(t);
        s.poll(t);
    }
    assert!(c.is_connected() && s.is_connected());
    assert!(c.rtt_ms() > 30.0 && c.rtt_ms() < 50.0, "pings travel on the sequenced channel: rtt {}", c.rtt_ms());
}

#[test]
fn a_goodbye_gets_through_a_reliable_stream_that_is_saturated() {
    let cfg = DriverConfig {
        reliable_budget: 200,
        ..DriverConfig::default()
    };
    let mut r = rig_with(SimConfig::default(), cfg, 1);
    r.connect();
    for n in 0..200u32 {
        r.c.send_typed(CH_EVENT, 0, Text { s: format!("{n:03}{}", "y".repeat(197)) }).unwrap();
    }
    r.c.close(DisconnectReason::Normal);
    r.step(10);
    r.step(10);
    assert_eq!(r.s.state(), SessionState::Ended(EndReason::PeerClosed(DisconnectReason::Normal)));
}

#[test]
fn an_ended_connection_sends_no_more_game_messages() {
    let mut r = rig_with(SimConfig::default(), small_budget(), 1);
    r.connect();
    for _ in 0..30 {
        r.c.send_typed(CH_COSMETIC, priority::NORMAL, text(400)).unwrap();
    }
    assert_eq!(r.c.queued_datagrams(), 30);
    let before = r.c.stats().datagrams_sent;
    r.c.close(DisconnectReason::Normal);
    r.step(10);
    assert!(r.c.is_ended());
    assert_eq!(r.c.queued_datagrams(), 0);
    assert_eq!(r.c.stats().datagrams_sent, before, "nothing but the goodbye (a stream) after a close");
    assert!(texts(&mut r.s).is_empty());
}

#[test]
fn traffic_of_the_game_alone_keeps_a_connection_alive() {
    for via_stream in [false, true] {
        let mut cfg = DriverConfig::default();
        cfg.session.ping_interval_ms = 1_000_000; // no keepalive: only the game's own messages are heard
        let mut r = rig_with(SimConfig::default(), cfg, 1);
        r.connect();
        events(&mut r.c);
        events(&mut r.s);
        for i in 0..200u32 {
            for d in [&mut r.c, &mut r.s] {
                if via_stream {
                    d.send_typed(CH_EVENT, 0, Event { n: i }).unwrap();
                } else {
                    d.send_typed(CH_STATE, 0, State { tick: i, x: 0.0 }).unwrap();
                }
            }
            r.run(100, 10);
            messages(&mut r.c);
            messages(&mut r.s);
        }
        assert!(r.c.is_connected() && r.s.is_connected(), "via_stream {via_stream}: {:?} / {:?}", r.c.state(), r.s.state());
        let (ce, se) = (events(&mut r.c), events(&mut r.s));
        assert!(!has_session(&ce, |e| matches!(e, SessionEvent::NoSignal { .. })), "{ce:?}");
        assert!(!has_session(&se, |e| matches!(e, SessionEvent::NoSignal { .. })), "{se:?}");
    }
}

#[test]
fn a_control_message_that_does_not_connect_does_not_release_held_messages() {
    let (mut c, mut srv, net, sch) = manual_server();
    c.poll(0);
    srv.send_reliable(CH_EVENT as u32, &stream_frame(&sch, &mut Event { n: 5 })).unwrap();
    // a pong, which a connecting client has no use for
    let pong = stream_frame(&sch, &mut ControlMsg::Pong { body: PongBody::default() });
    srv.send_reliable(0, &pong).unwrap();
    net.set_time(10);
    c.poll(10);
    assert_eq!(c.state(), SessionState::Connecting);
    assert_eq!(c.messages_waiting(), 0);
}

#[test]
fn a_message_of_the_wrong_type_is_handed_back_whole() {
    let inbound = Inbound {
        channel: 7,
        msg: Box::new(Event { n: 3 }),
    };
    let back = inbound.downcast::<Text>().unwrap_err();
    assert_eq!(back.channel, 7);
    let back = back.into_raw().unwrap_err();
    assert_eq!(back.channel, 7);
    assert_eq!(back.downcast::<Event>().unwrap(), Event { n: 3 });
    let raw = Inbound {
        channel: 2,
        msg: Box::new(mid_net_wire::RawMessage::new(RAW_ID, vec![1, 2])),
    };
    let (ch, m) = raw.into_raw().unwrap();
    assert_eq!((ch, m.id, m.data), (2, RAW_ID, vec![1, 2]));
}

#[test]
fn every_way_a_connection_can_end_has_words() {
    let all = [
        EndReason::LocalClose,
        EndReason::PeerClosed(DisconnectReason::Kicked),
        EndReason::Timeout,
        EndReason::ConnectTimeout,
        EndReason::Rejected(DisconnectReason::SchemaMismatch),
        EndReason::ProtocolError,
    ];
    let words: Vec<String> = all.iter().map(|r| describe_end(*r)).collect();
    for (i, w) in words.iter().enumerate() {
        assert!(!w.is_empty());
        assert_eq!(words.iter().filter(|x| *x == w).count(), 1, "{i}: {w}");
    }
    assert!(words[1].contains("Kicked") && words[4].contains("SchemaMismatch"));
}

#[test]
fn a_poll_reads_no_more_than_the_limit_and_the_rest_waits_for_the_next() {
    let cfg = DriverConfig {
        max_receive_per_poll: 3,
        ..DriverConfig::default()
    };
    let mut r = rig_with(
        SimConfig {
            reliable_piece_max: 6,
            ..SimConfig::default()
        },
        cfg,
        1,
    );
    r.connect();
    for n in 0..40u32 {
        r.s.send_typed(CH_EVENT, 0, Event { n }).unwrap();
    }
    r.step(10); // the server flushes
    assert_eq!(r.c.messages_waiting(), 0);
    r.step(10); // the client reads three pieces: at most three messages
    assert!(r.c.messages_waiting() <= 3 && r.c.messages_waiting() > 0, "{}", r.c.messages_waiting());
    let mut got = ev_numbers(&mut r.c);
    for _ in 0..300 {
        r.step(10);
        got.extend(ev_numbers(&mut r.c));
    }
    assert_eq!(got, (0..40).collect::<Vec<_>>());
}

#[test]
fn a_sim_that_counts_the_longest_datagram_is_honest() {
    let (net, mut a, _b) = SimNet::pair(SimConfig { mtu: 10, ..SimConfig::default() }, 1);
    a.send_datagram(&[0; 7]).unwrap();
    a.send_datagram(&[0; 30]).unwrap(); // dropped for its size, but it was handed over
    a.send_datagram(&[0; 3]).unwrap();
    assert_eq!(net.stats().max_datagram_len, 30);
}

#[test]
fn packets_fill_up_to_the_mtu_and_never_past_it() {
    for mtu in [200usize, 600, 1200] {
        let cfg = DriverConfig {
            mtu,
            datagram_budget: 1_000_000,
            ..DriverConfig::default()
        };
        let mut r = rig_with(SimConfig::default(), cfg, 1);
        r.connect();
        messages(&mut r.s);
        for i in 0..1000usize {
            r.c.send_typed(CH_COSMETIC, priority::NORMAL, text(1 + i % 37)).unwrap();
        }
        r.step(10);
        let longest = r.net.stats().max_datagram_len;
        assert!(longest <= mtu, "mtu {mtu}: a packet of {longest} bytes");
        assert!(longest >= mtu - 40, "mtu {mtu}: the longest packet was only {longest} bytes");
        r.run(50, 10);
        assert_eq!(messages(&mut r.s).len(), 1000);
    }
}

#[test]
fn a_clock_that_goes_backwards_does_not_age_a_message_wrongly() {
    let cfg = DriverConfig {
        datagram_ttl_ms: 500,
        ..DriverConfig::default()
    };
    let mut r = rig_with(SimConfig::default(), cfg, 1);
    r.connect();
    messages(&mut r.s);
    let t = r.now + 1_000;
    r.net.set_time(t);
    r.c.poll(t);
    r.c.poll(t - 400); // the caller's clock stepped back
    r.c.send_typed(CH_COSMETIC, priority::NORMAL, Event { n: 1 }).unwrap(); // queued at t, as far as the driver knows
    r.net.set_time(t + 450);
    r.c.poll(t + 450);
    r.s.poll(t + 450);
    assert_eq!(ev_numbers(&mut r.s), vec![1], "it has waited 450 ms, not 850");
}

#[test]
fn a_full_queue_makes_room_for_a_newer_message_of_the_same_priority() {
    let cfg = DriverConfig {
        max_queued_datagrams: 4,
        ..DriverConfig::default()
    };
    let mut r = rig_with(SimConfig::default(), cfg, 1);
    r.connect();
    messages(&mut r.s);
    for n in 1..=5u32 {
        r.c.send_typed(CH_COSMETIC, priority::NORMAL, Event { n }).unwrap();
    }
    r.run(50, 10);
    assert_eq!(ev_numbers(&mut r.s), vec![2, 3, 4, 5], "the oldest made way for the newest");
}

/// A client whose peer sent a lot while the client was not being polled.
fn backlog_rig(cfg: DriverConfig) -> Rig {
    let mut r = rig_with(SimConfig::default(), cfg, 1);
    r.connect();
    messages(&mut r.c);
    messages(&mut r.s);
    for n in 0..40u32 {
        r.s.send_typed(CH_COSMETIC, priority::NORMAL, Event { n }).unwrap();
        r.now += 10;
        r.net.set_time(r.now);
        r.s.poll(r.now); // one packet each: the client is not polled
    }
    r
}

#[test]
fn a_poll_reads_no_more_datagrams_than_the_limit() {
    let cfg = DriverConfig {
        max_receive_per_poll: 3,
        ..DriverConfig::default()
    };
    let mut r = backlog_rig(cfg);
    let before = r.c.stats().datagrams_received;
    r.c.poll(r.now);
    assert_eq!(r.c.stats().datagrams_received - before, 3);
    assert!(r.c.messages_waiting() <= 3, "a datagram may be a pong, not a message");
}

#[test]
fn a_full_inbox_stops_the_reading_of_datagrams_exactly_at_the_limit() {
    let cfg = DriverConfig {
        max_inbox: 10,
        ..DriverConfig::default()
    };
    let mut r = backlog_rig(cfg);
    r.c.poll(r.now);
    assert_eq!(r.c.messages_waiting(), 10);
    let mut got = ev_numbers(&mut r.c);
    for _ in 0..20 {
        r.c.poll(r.now);
        got.extend(ev_numbers(&mut r.c));
    }
    assert_eq!(got, (0..40).collect::<Vec<_>>());
}

#[test]
fn a_message_is_dropped_when_it_has_waited_the_ttl_and_not_before() {
    let cfg = DriverConfig {
        datagram_ttl_ms: 100,
        ..DriverConfig::default()
    };
    let mut r = rig_with(SimConfig::default(), cfg, 1);
    r.connect();
    messages(&mut r.s);
    let before = r.c.stats().dropped_expired;
    r.c.send_typed(CH_COSMETIC, priority::NORMAL, Event { n: 1 }).unwrap();
    r.step(100); // the next flush is exactly 100 ms after the message was queued
    assert_eq!(r.c.stats().dropped_expired - before, 1);
    assert!(messages(&mut r.s).is_empty());
    r.c.send_typed(CH_COSMETIC, priority::NORMAL, Event { n: 2 }).unwrap();
    r.step(99);
    assert_eq!(r.c.stats().dropped_expired - before, 1);
    assert_eq!(ev_numbers(&mut r.s), vec![2]);
}

#[test]
fn a_packet_that_exactly_spends_the_budget_still_goes_in_that_flush() {
    let cfg = DriverConfig {
        mtu: 1200,
        datagram_budget: 2400,
        ..DriverConfig::default()
    };
    let mut r = rig_with(SimConfig::default(), cfg, 1);
    r.connect();
    messages(&mut r.s);
    // calibrate: how many bytes of packet one raw message of this size costs on top of its data
    r.c.send_raw(CH_COSMETIC, priority::NORMAL, RAW_ID, vec![9; 200]).unwrap();
    r.step(10);
    let overhead = r.net.stats().max_datagram_len - 200;
    assert_eq!(messages(&mut r.s).len(), 1);
    let n = 1200 - overhead;
    r.c.send_raw(CH_COSMETIC, priority::NORMAL, RAW_ID, vec![1; n]).unwrap();
    r.c.send_raw(CH_COSMETIC, priority::NORMAL, RAW_ID, vec![2; n]).unwrap();
    r.step(10);
    assert_eq!(r.net.stats().max_datagram_len, 1200, "two packets of exactly the mtu");
    assert_eq!(messages(&mut r.s).len(), 2, "both went: the budget was spent exactly, not exceeded");
}
