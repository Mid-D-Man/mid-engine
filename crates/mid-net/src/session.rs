//! The life of a connection, independent of any socket: handshake, keepalive, time sync, and the
//! states a game can show ("connecting", "NO SIGNAL 3 s", "lost").
//!
//! This is layer 3 of `docs/mid-net-foundation.md`. A [`Session`] is a state machine driven by two
//! things the caller gives it, the clock (`now_ms`, any monotonic millisecond clock) and what arrives
//! ([`Session::on_control`] for a control message, [`Session::on_packet_received`] for *any* datagram or
//! stream bytes, which is the sign of life), and it answers with what to send ([`Session::take_outgoing`])
//! and what happened ([`Session::take_events`]). It never touches a transport, so it is tested
//! deterministically, with the clock under the test's control.
//!
//! * **Handshake.** The client sends `Hello` (protocol version, the **schema hash**, a nonce, a name) and
//!   retries it until answered; the server answers `Welcome` (session id, its clock, the tick rate) or
//!   `Reject`. A different schema hash is refused (`SchemaMismatch`), so two builds that disagree about the
//!   messages never exchange one: they are told, instead of misreading each other.
//! * **Keepalive and time.** While connected the client pings every `ping_interval_ms` and the server
//!   answers with its own clock. That is both directions of "I am still here", the round-trip time (the
//!   standard smoothed estimator), and a clock estimate: the offset to the server's clock is taken from
//!   the sample with the **lowest** round trip of the last eight, since that one was the least delayed
//!   (the error of an estimate is at most half the difference between the outward and the return delay).
//! * **No signal is not lost.** Silence for `no_signal_ms` turns `Connected` into `NoSignal` (an event, so
//!   the game can show it); anything heard turns it back (`Recovered`, with the length of the gap); only
//!   silence for `timeout_ms` ends the session. The old behaviour (everything dropped at the first 5
//!   seconds of silence) is one setting of this one, not the only one.
//! * **Ending.** `Ended` carries why: closed here, closed by the peer (with its reason), timed out,
//!   rejected, a protocol error. A session that has ended produces nothing more and ignores everything.
//!
//! Control messages are one `enum` ([`ControlMsg`], message id [`CONTROL_MESSAGE_ID`] = 0, which the
//! application must leave reserved) so they go through the same registry and framing as every other
//! message; handshake and goodbye on a reliable stream, pings on a datagram.

use std::collections::VecDeque;

use mid_net_reliable::RttEstimator;
use mid_net_wire::{impl_net_enum, impl_net_serialize, NetMessageType};

/// The version of this layer's own messages.
pub const SESSION_PROTOCOL_VERSION: u16 = 1;
/// The message id [`ControlMsg`] is registered under; applications keep it free.
pub const CONTROL_MESSAGE_ID: u16 = 0;

/// A fingerprint of the message schema both ends were built from. The code generator writes the real one; [`fnv1a64`] makes one from any bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct SchemaHash(pub u64);

/// FNV-1a, 64 bits: a small stable hash (the same on every platform) for fingerprinting a schema's bytes.
pub fn fnv1a64(bytes: &[u8]) -> SchemaHash {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    SchemaHash(h)
}

/// Why a connection was closed or refused, as it travels on the wire.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DisconnectReason {
    Normal,
    VersionMismatch,
    SchemaMismatch,
    Full,
    Timeout,
    ConnectTimeout,
    Kicked,
    ProtocolError,
    /// A reason this version does not know (a newer peer).
    Unknown(u8),
}

impl DisconnectReason {
    pub fn to_wire(self) -> u8 {
        match self {
            DisconnectReason::Normal => 0,
            DisconnectReason::VersionMismatch => 1,
            DisconnectReason::SchemaMismatch => 2,
            DisconnectReason::Full => 3,
            DisconnectReason::Timeout => 4,
            DisconnectReason::ConnectTimeout => 5,
            DisconnectReason::Kicked => 6,
            DisconnectReason::ProtocolError => 7,
            DisconnectReason::Unknown(c) => c,
        }
    }

    pub fn from_wire(code: u8) -> Self {
        match code {
            0 => DisconnectReason::Normal,
            1 => DisconnectReason::VersionMismatch,
            2 => DisconnectReason::SchemaMismatch,
            3 => DisconnectReason::Full,
            4 => DisconnectReason::Timeout,
            5 => DisconnectReason::ConnectTimeout,
            6 => DisconnectReason::Kicked,
            7 => DisconnectReason::ProtocolError,
            c => DisconnectReason::Unknown(c),
        }
    }
}

#[derive(Debug, Default, Clone, PartialEq)]
pub struct HelloBody {
    pub protocol: u16,
    pub schema: u64,
    pub nonce: u32,
    pub name: String,
}
impl_net_serialize!(HelloBody {
    protocol: raw(),
    schema: raw(),
    nonce: raw(),
    name: string(32)
});

#[derive(Debug, Default, Clone, PartialEq)]
pub struct WelcomeBody {
    pub nonce: u32,
    pub session_id: u32,
    pub server_time_ms: u64,
    pub tick_rate_hz: u16,
}
impl_net_serialize!(WelcomeBody {
    nonce: raw(),
    session_id: raw(),
    server_time_ms: varint(),
    tick_rate_hz: raw()
});

#[derive(Debug, Default, Clone, PartialEq)]
pub struct RejectBody {
    pub nonce: u32,
    pub reason: u8,
}
impl_net_serialize!(RejectBody {
    nonce: raw(),
    reason: raw()
});

#[derive(Debug, Default, Clone, PartialEq)]
pub struct PingBody {
    pub seq: u16,
    pub client_time_ms: u64,
}
impl_net_serialize!(PingBody {
    seq: raw(),
    client_time_ms: varint()
});

#[derive(Debug, Default, Clone, PartialEq)]
pub struct PongBody {
    pub seq: u16,
    pub client_time_ms: u64,
    pub server_time_ms: u64,
}
impl_net_serialize!(PongBody {
    seq: raw(),
    client_time_ms: varint(),
    server_time_ms: varint()
});

#[derive(Debug, Default, Clone, PartialEq)]
pub struct ByeBody {
    pub reason: u8,
}
impl_net_serialize!(ByeBody { reason: raw() });

/// What the two ends of a [`Session`] say to each other.
#[derive(Debug, Clone, PartialEq)]
pub enum ControlMsg {
    Hello { body: HelloBody },
    Welcome { body: WelcomeBody },
    Reject { body: RejectBody },
    Ping { body: PingBody },
    Pong { body: PongBody },
    Bye { body: ByeBody },
}
impl Default for ControlMsg {
    fn default() -> Self {
        ControlMsg::Bye {
            body: ByeBody::default(),
        }
    }
}
impl_net_enum!(ControlMsg, 3, {
    Hello { body: raw() } = 0,
    Welcome { body: raw() } = 1,
    Reject { body: raw() } = 2,
    Ping { body: raw() } = 3,
    Pong { body: raw() } = 4,
    Bye { body: raw() } = 5,
});
impl NetMessageType for ControlMsg {
    const ID: u16 = CONTROL_MESSAGE_ID;
}

/// How a control message should travel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Delivery {
    /// On the reliable stream (handshake, goodbye).
    Reliable,
    /// In a datagram (keepalive).
    Unreliable,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Outgoing {
    pub delivery: Delivery,
    pub msg: ControlMsg,
}

/// Which end of the connection a session is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    Client,
    Server,
}

/// How a session ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EndReason {
    /// Closed on purpose from this side ([`Session::close`]).
    LocalClose,
    /// The other side said goodbye, with its reason.
    PeerClosed(DisconnectReason),
    /// Nothing was heard for `timeout_ms`.
    Timeout,
    /// The handshake did not finish in `connect_timeout_ms`.
    ConnectTimeout,
    /// The handshake was refused (by the server, for the client; by this server, for a server).
    Rejected(DisconnectReason),
    /// The other side broke the protocol.
    ProtocolError,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionState {
    /// The handshake is not finished.
    Connecting,
    Connected,
    /// Connected before, and silent for at least `no_signal_ms` (since this time on our clock).
    NoSignal {
        since_ms: u64,
    },
    Ended(EndReason),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionEvent {
    Connected {
        session_id: u32,
        tick_rate_hz: u16,
    },
    NoSignal {
        silent_ms: u64,
    },
    /// Heard again after `gap_ms` of silence.
    Recovered {
        gap_ms: u64,
    },
    Ended(EndReason),
}

#[derive(Debug, Clone, Copy)]
pub struct SessionConfig {
    /// This layer's protocol version, compared in the handshake.
    pub protocol: u16,
    pub schema: SchemaHash,
    /// How often a connected client pings (0 never: the connection must then carry other traffic to stay alive).
    pub ping_interval_ms: u64,
    /// Silence before `NoSignal`.
    pub no_signal_ms: u64,
    /// Silence before the session ends.
    pub timeout_ms: u64,
    /// How long a handshake may take.
    pub connect_timeout_ms: u64,
    /// How often an unanswered `Hello` is sent again (0 never: right when the stream is reliable).
    pub hello_retry_ms: u64,
    /// What a server tells its clients.
    pub tick_rate_hz: u16,
}

impl Default for SessionConfig {
    fn default() -> Self {
        SessionConfig {
            protocol: SESSION_PROTOCOL_VERSION,
            schema: SchemaHash(0),
            ping_interval_ms: 500,
            no_signal_ms: 1_500,
            timeout_ms: 10_000,
            connect_timeout_ms: 5_000,
            hello_retry_ms: 500,
            tick_rate_hz: 30,
        }
    }
}

const CLOCK_SAMPLES: usize = 8;

#[derive(Clone, Copy)]
struct ClockSample {
    rtt_ms: u64,
    offset_ms: i64, // server clock minus ours
}

pub struct Session {
    role: Role,
    cfg: SessionConfig,
    state: SessionState,
    name: String,
    nonce: u32,
    session_id: u32,
    tick_rate_hz: u16,
    started_ms: u64,
    last_now: u64,
    last_heard_ms: u64,
    last_hello_ms: Option<u64>,
    last_ping_ms: Option<u64>,
    ping_seq: u16,
    rtt: RttEstimator,
    last_rtt_ms: Option<u64>,
    samples: VecDeque<ClockSample>,
    out: VecDeque<Outgoing>,
    events: VecDeque<SessionEvent>,
}

impl Session {
    /// The client's end, starting now. It sends its `Hello` on the first [`poll`](Self::poll).
    pub fn client(cfg: SessionConfig, name: &str, nonce: u32, now_ms: u64) -> Self {
        Self::new(
            Role::Client,
            cfg,
            name.chars().take(32).collect(),
            nonce,
            0,
            now_ms,
        )
    }

    /// The server's end for one client, starting when the connection is accepted. `session_id` is what it will tell the client.
    pub fn server(cfg: SessionConfig, session_id: u32, now_ms: u64) -> Self {
        Self::new(Role::Server, cfg, String::new(), 0, session_id, now_ms)
    }

    fn new(
        role: Role,
        cfg: SessionConfig,
        name: String,
        nonce: u32,
        session_id: u32,
        now_ms: u64,
    ) -> Self {
        Session {
            role,
            cfg,
            state: SessionState::Connecting,
            name,
            nonce,
            session_id,
            tick_rate_hz: cfg.tick_rate_hz,
            started_ms: now_ms,
            last_now: now_ms,
            last_heard_ms: now_ms,
            last_hello_ms: None,
            last_ping_ms: None,
            ping_seq: 0,
            rtt: RttEstimator::new(),
            last_rtt_ms: None,
            samples: VecDeque::new(),
            out: VecDeque::new(),
            events: VecDeque::new(),
        }
    }

    pub fn role(&self) -> Role {
        self.role
    }
    pub fn state(&self) -> SessionState {
        self.state
    }
    pub fn is_ended(&self) -> bool {
        matches!(self.state, SessionState::Ended(_))
    }
    /// 0 until the handshake has given one.
    pub fn session_id(&self) -> u32 {
        self.session_id
    }
    pub fn tick_rate_hz(&self) -> u16 {
        self.tick_rate_hz
    }
    /// The smoothed round-trip time, once a pong has come (the estimator's default before).
    pub fn rtt_ms(&self) -> f64 {
        self.rtt.smoothed_ms()
    }
    /// The last single round trip measured.
    pub fn last_rtt_ms(&self) -> Option<u64> {
        self.last_rtt_ms
    }
    /// How long nothing has been heard, at `now_ms`.
    pub fn silence_ms(&self, now_ms: u64) -> u64 {
        now_ms.saturating_sub(self.last_heard_ms)
    }

    /// The server's clock at our time `now_ms`, estimated from the pings (client only; `None` before the first pong).
    pub fn server_time_ms(&self, now_ms: u64) -> Option<u64> {
        let best = self.samples.iter().min_by_key(|s| s.rtt_ms)?;
        Some((now_ms as i64 + best.offset_ms).max(0) as u64)
    }

    pub fn take_outgoing(&mut self) -> Vec<Outgoing> {
        self.out.drain(..).collect()
    }
    pub fn take_events(&mut self) -> Vec<SessionEvent> {
        self.events.drain(..).collect()
    }

    fn end(&mut self, reason: EndReason) {
        if self.is_ended() {
            return;
        }
        self.state = SessionState::Ended(reason);
        self.events.push_back(SessionEvent::Ended(reason));
    }

    fn send(&mut self, delivery: Delivery, msg: ControlMsg) {
        self.out.push_back(Outgoing { delivery, msg });
    }

    /// Closes on purpose: queues a goodbye and ends. Does nothing if the session has already ended.
    pub fn close(&mut self, reason: DisconnectReason) {
        if self.is_ended() {
            return;
        }
        self.send(
            Delivery::Reliable,
            ControlMsg::Bye {
                body: ByeBody {
                    reason: reason.to_wire(),
                },
            },
        );
        self.end(EndReason::LocalClose);
    }

    /// Something arrived from the peer (any datagram, any stream bytes): the connection is alive. Call it for every one.
    pub fn on_packet_received(&mut self, now_ms: u64) {
        let now = now_ms.max(self.last_now);
        self.last_now = now;
        if self.is_ended() {
            return;
        }
        let gap = now.saturating_sub(self.last_heard_ms);
        self.last_heard_ms = now;
        if let SessionState::NoSignal { .. } = self.state {
            self.state = SessionState::Connected;
            self.events
                .push_back(SessionEvent::Recovered { gap_ms: gap });
        }
    }

    /// Advances the clock: timeouts, retries, pings. Call it every frame or tick.
    pub fn poll(&mut self, now_ms: u64) {
        let now = now_ms.max(self.last_now);
        self.last_now = now;
        match self.state {
            SessionState::Ended(_) => {}
            SessionState::Connecting => {
                if now.saturating_sub(self.started_ms) >= self.cfg.connect_timeout_ms {
                    self.end(EndReason::ConnectTimeout);
                } else if self.role == Role::Client {
                    let due = match self.last_hello_ms {
                        None => true,
                        Some(t) => {
                            self.cfg.hello_retry_ms > 0
                                && now.saturating_sub(t) >= self.cfg.hello_retry_ms
                        }
                    };
                    if due {
                        self.last_hello_ms = Some(now);
                        let body = HelloBody {
                            protocol: self.cfg.protocol,
                            schema: self.cfg.schema.0,
                            nonce: self.nonce,
                            name: self.name.clone(),
                        };
                        self.send(Delivery::Reliable, ControlMsg::Hello { body });
                    }
                }
            }
            SessionState::Connected | SessionState::NoSignal { .. } => {
                let silent = now.saturating_sub(self.last_heard_ms);
                if silent >= self.cfg.timeout_ms {
                    self.end(EndReason::Timeout);
                    return;
                }
                if self.state == SessionState::Connected && silent >= self.cfg.no_signal_ms {
                    self.state = SessionState::NoSignal {
                        since_ms: self.last_heard_ms,
                    };
                    self.events
                        .push_back(SessionEvent::NoSignal { silent_ms: silent });
                }
                if self.role == Role::Client && self.cfg.ping_interval_ms > 0 {
                    let due = match self.last_ping_ms {
                        None => true,
                        Some(t) => now.saturating_sub(t) >= self.cfg.ping_interval_ms,
                    };
                    if due {
                        self.last_ping_ms = Some(now);
                        self.ping_seq = self.ping_seq.wrapping_add(1);
                        let body = PingBody {
                            seq: self.ping_seq,
                            client_time_ms: now,
                        };
                        self.send(Delivery::Unreliable, ControlMsg::Ping { body });
                    }
                }
            }
        }
    }

    /// A control message arrived. (Also counts as a sign of life: no need to call [`on_packet_received`](Self::on_packet_received) for it as well.)
    pub fn on_control(&mut self, now_ms: u64, msg: ControlMsg) {
        self.on_packet_received(now_ms);
        let now = self.last_now;
        if self.is_ended() {
            return;
        }
        match (self.role, msg) {
            (Role::Server, ControlMsg::Hello { body }) => self.server_hello(body),
            (Role::Client, ControlMsg::Welcome { body }) => {
                if self.state == SessionState::Connecting && body.nonce == self.nonce {
                    self.session_id = body.session_id;
                    self.tick_rate_hz = body.tick_rate_hz;
                    self.state = SessionState::Connected;
                    self.events.push_back(SessionEvent::Connected {
                        session_id: body.session_id,
                        tick_rate_hz: body.tick_rate_hz,
                    });
                }
                // a repeated welcome (a retried hello answered twice) or one for another nonce: nothing to do
            }
            (Role::Client, ControlMsg::Reject { body }) => {
                if self.state == SessionState::Connecting && body.nonce == self.nonce {
                    self.end(EndReason::Rejected(DisconnectReason::from_wire(
                        body.reason,
                    )));
                }
            }
            (Role::Server, ControlMsg::Ping { body }) => {
                if self.state != SessionState::Connecting {
                    let pong = PongBody {
                        seq: body.seq,
                        client_time_ms: body.client_time_ms,
                        server_time_ms: now,
                    };
                    self.send(Delivery::Unreliable, ControlMsg::Pong { body: pong });
                }
            }
            (Role::Client, ControlMsg::Pong { body }) => {
                if self.state != SessionState::Connecting && body.client_time_ms <= now {
                    let rtt = now - body.client_time_ms;
                    self.rtt.on_sample(rtt as f64);
                    self.last_rtt_ms = Some(rtt);
                    let offset = body.server_time_ms as i64 + (rtt / 2) as i64 - now as i64;
                    if self.samples.len() == CLOCK_SAMPLES {
                        self.samples.pop_front();
                    }
                    self.samples.push_back(ClockSample {
                        rtt_ms: rtt,
                        offset_ms: offset,
                    });
                }
            }
            (_, ControlMsg::Bye { body }) => self.end(EndReason::PeerClosed(
                DisconnectReason::from_wire(body.reason),
            )),
            // a message that makes no sense for this end (a server receiving a Welcome, ...)
            _ => {
                self.send(
                    Delivery::Reliable,
                    ControlMsg::Bye {
                        body: ByeBody {
                            reason: DisconnectReason::ProtocolError.to_wire(),
                        },
                    },
                );
                self.end(EndReason::ProtocolError);
            }
        }
    }

    fn server_hello(&mut self, hello: HelloBody) {
        match self.state {
            SessionState::Connecting => {
                let refusal = if hello.protocol != self.cfg.protocol {
                    Some(DisconnectReason::VersionMismatch)
                } else if hello.schema != self.cfg.schema.0 {
                    Some(DisconnectReason::SchemaMismatch)
                } else {
                    None
                };
                if let Some(reason) = refusal {
                    self.send(
                        Delivery::Reliable,
                        ControlMsg::Reject {
                            body: RejectBody {
                                nonce: hello.nonce,
                                reason: reason.to_wire(),
                            },
                        },
                    );
                    self.end(EndReason::Rejected(reason));
                    return;
                }
                self.nonce = hello.nonce;
                self.name = hello.name;
                self.state = SessionState::Connected;
                self.events.push_back(SessionEvent::Connected {
                    session_id: self.session_id,
                    tick_rate_hz: self.tick_rate_hz,
                });
                self.send_welcome();
            }
            _ => {
                // A retried hello after we answered: the same nonce gets the same welcome again; another nonce is not a retry.
                if hello.nonce == self.nonce {
                    self.send_welcome();
                } else {
                    self.send(
                        Delivery::Reliable,
                        ControlMsg::Bye {
                            body: ByeBody {
                                reason: DisconnectReason::ProtocolError.to_wire(),
                            },
                        },
                    );
                    self.end(EndReason::ProtocolError);
                }
            }
        }
    }

    fn send_welcome(&mut self) {
        let body = WelcomeBody {
            nonce: self.nonce,
            session_id: self.session_id,
            server_time_ms: self.last_now,
            tick_rate_hz: self.tick_rate_hz,
        };
        self.send(Delivery::Reliable, ControlMsg::Welcome { body });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mid_net_wire::{
        net_read, net_write, BitError, ChannelKind, ChannelTable, DatagramBuilder,
        DatagramReceiver, MessageRegistry, Sequence, StreamReader, StreamWriter,
    };

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn below(&mut self, n: u64) -> u64 {
            self.next() % n
        }
    }

    const STEP: u64 = 10;
    const SERVER_CLOCK_OFFSET: u64 = 1_000_000;

    fn cfg(schema: u64) -> SessionConfig {
        SessionConfig {
            schema: SchemaHash(schema),
            ..SessionConfig::default()
        }
    }

    /// A pretend network between a client and a server with their own clocks (the server's is ahead by SERVER_CLOCK_OFFSET).
    struct Net {
        now: u64,
        c2s_ms: u64,
        s2c_ms: u64,
        jitter_ms: u64,
        jitter_to_client_only: bool,
        loss_pct: u64,
        down: Vec<(u64, u64)>, // the link carries nothing in these [from, to) intervals
        drop_reliable_before: u64, // reliable messages sent before this time are lost (a transport that does not retransmit)
        rng: Rng,
        to_server: Vec<(u64, ControlMsg)>,
        to_client: Vec<(u64, ControlMsg)>,
        sent_hellos: usize,
    }

    impl Net {
        fn new(c2s_ms: u64, s2c_ms: u64) -> Self {
            Net {
                now: 0,
                c2s_ms,
                s2c_ms,
                jitter_ms: 0,
                jitter_to_client_only: false,
                loss_pct: 0,
                down: vec![],
                drop_reliable_before: 0,
                rng: Rng(0x1234_5678_9ABC_DEF1),
                to_server: vec![],
                to_client: vec![],
                sent_hellos: 0,
            }
        }

        fn is_down(&self) -> Option<u64> {
            self.down
                .iter()
                .find(|(a, b)| self.now >= *a && self.now < *b)
                .map(|(_, b)| *b)
        }

        fn route(&mut self, outs: Vec<Outgoing>, to_server: bool) {
            for o in outs {
                if matches!(o.msg, ControlMsg::Hello { .. }) {
                    self.sent_hellos += 1;
                }
                let base = if to_server { self.c2s_ms } else { self.s2c_ms };
                let mut at = self.now
                    + base
                    + if self.jitter_ms > 0 && !(to_server && self.jitter_to_client_only) {
                        self.rng.below(self.jitter_ms + 1)
                    } else {
                        0
                    };
                match o.delivery {
                    Delivery::Unreliable => {
                        if self.is_down().is_some() || self.rng.below(100) < self.loss_pct {
                            continue;
                        }
                    }
                    Delivery::Reliable => {
                        if self.now < self.drop_reliable_before {
                            continue;
                        }
                        if let Some(up) = self.is_down() {
                            at = at.max(up + base); // held until the link is back, as a reliable stream holds
                        }
                    }
                }
                if to_server {
                    self.to_server.push((at, o.msg));
                } else {
                    self.to_client.push((at, o.msg));
                }
            }
        }

        fn step(&mut self, c: &mut Session, s: &mut Session) {
            self.now += STEP;
            c.poll(self.now);
            s.poll(self.now + SERVER_CLOCK_OFFSET);
            let (co, so) = (c.take_outgoing(), s.take_outgoing());
            self.route(co, true);
            self.route(so, false);
            self.to_server.sort_by_key(|m| m.0);
            self.to_client.sort_by_key(|m| m.0);
            while self.to_server.first().is_some_and(|m| m.0 <= self.now) {
                let (_, m) = self.to_server.remove(0);
                s.on_control(self.now + SERVER_CLOCK_OFFSET, m);
            }
            while self.to_client.first().is_some_and(|m| m.0 <= self.now) {
                let (_, m) = self.to_client.remove(0);
                c.on_control(self.now, m);
            }
        }

        fn run(&mut self, c: &mut Session, s: &mut Session, until: u64) {
            while self.now < until {
                self.step(c, s);
            }
        }
    }

    fn pair(schema_c: u64, schema_s: u64) -> (Session, Session) {
        (
            Session::client(cfg(schema_c), "brou", 42, 0),
            Session::server(cfg(schema_s), 7, SERVER_CLOCK_OFFSET),
        )
    }

    // ---- the handshake ---------------------------------------------------------------------------------------------

    #[test]
    fn a_client_and_a_server_that_agree_connect_and_both_say_so() {
        let (mut c, mut s) = pair(99, 99);
        let mut net = Net::new(20, 20);
        net.run(&mut c, &mut s, 300);
        assert_eq!(
            (c.state(), s.state()),
            (SessionState::Connected, SessionState::Connected)
        );
        assert_eq!((c.session_id(), c.tick_rate_hz()), (7, 30));
        assert_eq!(
            c.take_events(),
            vec![SessionEvent::Connected {
                session_id: 7,
                tick_rate_hz: 30
            }]
        );
        assert_eq!(
            s.take_events(),
            vec![SessionEvent::Connected {
                session_id: 7,
                tick_rate_hz: 30
            }]
        );
        assert_eq!(net.sent_hellos, 1, "answered at once: no retry");
    }

    #[test]
    fn builds_that_disagree_are_told_instead_of_misreading_each_other() {
        let (mut c, mut s) = pair(1, 2);
        let mut net = Net::new(20, 20);
        net.run(&mut c, &mut s, 300);
        assert_eq!(
            c.state(),
            SessionState::Ended(EndReason::Rejected(DisconnectReason::SchemaMismatch))
        );
        assert_eq!(
            s.state(),
            SessionState::Ended(EndReason::Rejected(DisconnectReason::SchemaMismatch))
        );
        // another protocol version of this layer
        let mut c = Session::client(
            SessionConfig {
                protocol: 9,
                ..cfg(5)
            },
            "x",
            1,
            0,
        );
        let mut s = Session::server(cfg(5), 1, SERVER_CLOCK_OFFSET);
        let mut net = Net::new(20, 20);
        net.run(&mut c, &mut s, 300);
        assert_eq!(
            c.state(),
            SessionState::Ended(EndReason::Rejected(DisconnectReason::VersionMismatch))
        );
        // and an ended session says nothing more, however long it goes on
        net.run(&mut c, &mut s, 20_000);
        assert!(c.take_outgoing().is_empty() && s.take_outgoing().is_empty());
        assert_eq!(c.take_events().len(), 1);
    }

    #[test]
    fn an_unanswered_hello_is_sent_again_until_it_is_answered() {
        let (mut c, mut s) = pair(3, 3);
        let mut net = Net::new(20, 20);
        net.drop_reliable_before = 1_200; // a transport that does not retransmit: the first hellos are lost
        net.run(&mut c, &mut s, 2_500);
        assert_eq!(c.state(), SessionState::Connected);
        assert_eq!(
            net.sent_hellos, 4,
            "at 0, 500, 1000 (lost) and 1500 (answered)"
        );
        // with retries off a lost hello is never replaced
        let mut c = Session::client(
            SessionConfig {
                hello_retry_ms: 0,
                ..cfg(3)
            },
            "x",
            1,
            0,
        );
        let mut s = Session::server(cfg(3), 1, SERVER_CLOCK_OFFSET);
        let mut net = Net::new(20, 20);
        net.drop_reliable_before = 1_200;
        net.run(&mut c, &mut s, 4_000);
        assert_eq!((net.sent_hellos, c.state()), (1, SessionState::Connecting));
        net.run(&mut c, &mut s, 6_000);
        assert_eq!(c.state(), SessionState::Ended(EndReason::ConnectTimeout));
    }

    #[test]
    fn a_handshake_that_never_finishes_ends_on_time_on_both_sides() {
        let mut c = Session::client(cfg(1), "x", 1, 1_000);
        let mut s = Session::server(cfg(1), 1, 1_000);
        c.poll(5_999);
        s.poll(5_999);
        assert_eq!(
            (c.state(), s.state()),
            (SessionState::Connecting, SessionState::Connecting)
        );
        c.poll(6_000);
        s.poll(6_000);
        assert_eq!(c.state(), SessionState::Ended(EndReason::ConnectTimeout));
        assert_eq!(s.state(), SessionState::Ended(EndReason::ConnectTimeout));
        assert_eq!(
            c.take_events(),
            vec![SessionEvent::Ended(EndReason::ConnectTimeout)]
        );
    }

    #[test]
    fn a_retried_hello_gets_the_same_welcome_and_another_nonce_is_not_a_retry() {
        let (mut c, mut s) = pair(8, 8);
        let mut net = Net::new(20, 20);
        net.run(&mut c, &mut s, 200);
        s.take_outgoing();
        let hello = HelloBody {
            protocol: SESSION_PROTOCOL_VERSION,
            schema: 8,
            nonce: 42,
            name: "brou".into(),
        };
        s.on_control(
            500,
            ControlMsg::Hello {
                body: hello.clone(),
            },
        );
        let out = s.take_outgoing();
        assert_eq!(out.len(), 1);
        assert!(
            matches!(&out[0].msg, ControlMsg::Welcome { body } if body.nonce == 42 && body.session_id == 7)
        );
        assert_eq!(s.state(), SessionState::Connected);
        s.on_control(
            600,
            ControlMsg::Hello {
                body: HelloBody { nonce: 43, ..hello },
            },
        );
        assert_eq!(s.state(), SessionState::Ended(EndReason::ProtocolError));
        assert!(matches!(s.take_outgoing()[0].msg, ControlMsg::Bye { .. }));
    }

    #[test]
    fn a_welcome_or_reject_for_another_hello_is_ignored_and_nonsense_for_a_role_is_a_protocol_error(
    ) {
        let mut c = Session::client(cfg(1), "x", 42, 0);
        c.poll(0);
        c.take_outgoing();
        c.on_control(
            10,
            ControlMsg::Welcome {
                body: WelcomeBody {
                    nonce: 41,
                    session_id: 5,
                    server_time_ms: 0,
                    tick_rate_hz: 60,
                },
            },
        );
        c.on_control(
            20,
            ControlMsg::Reject {
                body: RejectBody {
                    nonce: 41,
                    reason: 3,
                },
            },
        );
        assert_eq!((c.state(), c.session_id()), (SessionState::Connecting, 0));
        c.on_control(
            30,
            ControlMsg::Welcome {
                body: WelcomeBody {
                    nonce: 42,
                    session_id: 5,
                    server_time_ms: 0,
                    tick_rate_hz: 60,
                },
            },
        );
        assert_eq!(
            (c.state(), c.session_id(), c.tick_rate_hz()),
            (SessionState::Connected, 5, 60)
        );
        // a client receiving a Ping, a server receiving a Welcome
        c.on_control(
            40,
            ControlMsg::Ping {
                body: PingBody::default(),
            },
        );
        assert_eq!(c.state(), SessionState::Ended(EndReason::ProtocolError));
        let mut s = Session::server(cfg(1), 1, 0);
        s.on_control(
            10,
            ControlMsg::Welcome {
                body: WelcomeBody::default(),
            },
        );
        assert_eq!(s.state(), SessionState::Ended(EndReason::ProtocolError));
        // a ping before the handshake is not answered
        let mut s = Session::server(cfg(1), 1, 0);
        s.on_control(
            10,
            ControlMsg::Ping {
                body: PingBody {
                    seq: 1,
                    client_time_ms: 5,
                },
            },
        );
        assert!(s.take_outgoing().is_empty());
    }

    // ---- keepalive, round trip, the clock --------------------------------------------------------------------------

    #[test]
    fn pings_measure_the_round_trip_and_the_servers_clock_to_within_the_asymmetry() {
        let (mut c, mut s) = pair(1, 1);
        let mut net = Net::new(30, 50); // 80 ms round trip, the way back 20 ms slower than the way out
        net.run(&mut c, &mut s, 5_000);
        assert!(
            (c.rtt_ms() - 80.0).abs() <= 1.0 + STEP as f64,
            "rtt {}",
            c.rtt_ms()
        );
        assert!(c.last_rtt_ms().is_some_and(|r| (70..=100).contains(&r)));
        let truth = net.now + SERVER_CLOCK_OFFSET;
        let est = c.server_time_ms(net.now).unwrap();
        assert!(
            (est as i64 - truth as i64).abs() <= 10 + STEP as i64,
            "estimated {est}, truth {truth}"
        );
        assert!(
            Session::server(cfg(1), 1, 0).server_time_ms(100).is_none(),
            "only a client estimates"
        );
        // pings go out every 500 ms and no more
        let mut c2 = Session::client(cfg(1), "x", 1, 0);
        c2.on_control(
            0,
            ControlMsg::Welcome {
                body: WelcomeBody {
                    nonce: 1,
                    session_id: 1,
                    server_time_ms: 0,
                    tick_rate_hz: 30,
                },
            },
        );
        let mut pings = 0;
        for t in (0..=5_000).step_by(10) {
            c2.poll(t);
            c2.on_packet_received(t);
            pings += c2
                .take_outgoing()
                .iter()
                .filter(|o| {
                    matches!(o.msg, ControlMsg::Ping { .. }) && o.delivery == Delivery::Unreliable
                })
                .count();
        }
        assert_eq!(pings, 11, "at 0, 500, ... 5000");
    }

    #[test]
    fn the_least_delayed_sample_wins_so_jitter_does_not_move_the_clock_far() {
        let (mut c, mut s) = pair(1, 1);
        let mut net = Net::new(30, 30);
        net.jitter_ms = 100; // the way back is 0 to 100 ms late, the way out is steady
        net.jitter_to_client_only = true;
        net.loss_pct = 10;
        let mut worst = 0i64;
        for _ in 0..6_000 {
            net.step(&mut c, &mut s);
            if net.now > 3_000 {
                if let Some(est) = c.server_time_ms(net.now) {
                    worst = worst.max((est as i64 - (net.now + SERVER_CLOCK_OFFSET) as i64).abs());
                }
            }
        }
        assert_eq!(c.state(), SessionState::Connected);
        assert!(
            worst <= 34,
            "the least-delayed of the last eight samples is used: its extra delay was at most 68 ms, so the clock was never off by more than 34 ms (half of it); {worst}"
        );
    }

    // ---- no signal is not lost ---------------------------------------------------------------------------------------

    #[test]
    fn a_gap_in_the_link_is_shown_then_recovered_and_only_a_long_one_ends_the_session() {
        let (mut c, mut s) = pair(1, 1);
        let mut net = Net::new(20, 20);
        net.down = vec![(3_000, 7_000), (12_000, 30_000)]; // 4 s, then 18 s
        net.run(&mut c, &mut s, 2_900);
        c.take_events();
        s.take_events();
        net.run(&mut c, &mut s, 9_000);
        let ev = c.take_events();
        assert_eq!(ev.len(), 2, "{ev:?}");
        assert!(
            matches!(ev[0], SessionEvent::NoSignal { silent_ms } if (1_500..1_500 + 3 * STEP).contains(&silent_ms)),
            "{ev:?}"
        );
        assert!(
            matches!(ev[1], SessionEvent::Recovered { gap_ms } if (3_800..5_000).contains(&gap_ms)),
            "{ev:?}"
        );
        assert_eq!(c.state(), SessionState::Connected);
        let sev = s.take_events();
        assert!(
            sev.iter()
                .any(|e| matches!(e, SessionEvent::NoSignal { .. }))
                && sev
                    .iter()
                    .any(|e| matches!(e, SessionEvent::Recovered { .. }))
        );
        net.run(&mut c, &mut s, 25_000);
        assert_eq!(c.state(), SessionState::Ended(EndReason::Timeout));
        assert_eq!(s.state(), SessionState::Ended(EndReason::Timeout));
        let ev = c.take_events();
        assert_eq!(
            ev.iter()
                .filter(|e| matches!(e, SessionEvent::Ended(_)))
                .count(),
            1
        );
        net.run(&mut c, &mut s, 40_000);
        assert!(
            c.take_outgoing().is_empty() && c.take_events().is_empty(),
            "an ended session is silent"
        );
    }

    #[test]
    fn anything_heard_counts_as_a_sign_of_life_and_the_timeouts_are_exact() {
        let mut c = Session::client(cfg(1), "x", 1, 0);
        c.on_control(
            0,
            ControlMsg::Welcome {
                body: WelcomeBody {
                    nonce: 1,
                    session_id: 1,
                    server_time_ms: 0,
                    tick_rate_hz: 30,
                },
            },
        );
        for t in (0..20_000).step_by(400) {
            c.poll(t);
            c.on_packet_received(t); // game data, not a control message
        }
        assert_eq!(c.state(), SessionState::Connected);
        let last = 19_600;
        c.poll(last + 1_499);
        assert_eq!(c.state(), SessionState::Connected);
        c.poll(last + 1_500);
        assert_eq!(c.state(), SessionState::NoSignal { since_ms: last });
        c.poll(last + 9_999);
        assert!(matches!(c.state(), SessionState::NoSignal { .. }));
        c.poll(last + 10_000);
        assert_eq!(c.state(), SessionState::Ended(EndReason::Timeout));
        assert_eq!(c.silence_ms(last + 12_000), 12_000);
        c.on_packet_received(last + 11_000);
        assert!(c.is_ended(), "too late");
    }

    #[test]
    fn a_clock_that_goes_backwards_changes_nothing() {
        let mut c = Session::client(cfg(1), "x", 1, 1_000);
        c.on_control(
            1_000,
            ControlMsg::Welcome {
                body: WelcomeBody {
                    nonce: 1,
                    session_id: 1,
                    server_time_ms: 0,
                    tick_rate_hz: 30,
                },
            },
        );
        c.poll(2_000);
        c.poll(500); // the clock steps back: nothing happens, and 2000 is still "now"
        c.on_packet_received(0); // a packet stamped from before: it is a sign of life now, at 2000
        assert_eq!(c.state(), SessionState::Connected);
        assert_eq!(c.silence_ms(2_000), 0);
        c.poll(3_400);
        assert_eq!(
            c.state(),
            SessionState::Connected,
            "heard at 2000, so 1400 ms of silence is not yet no signal"
        );
        c.poll(3_500);
        assert!(matches!(
            c.state(),
            SessionState::NoSignal { since_ms: 2_000 }
        ));
    }

    // ---- ending --------------------------------------------------------------------------------------------------------

    #[test]
    fn closing_says_goodbye_once_and_the_other_end_hears_why() {
        let (mut c, mut s) = pair(1, 1);
        let mut net = Net::new(20, 20);
        net.run(&mut c, &mut s, 300);
        c.take_events();
        s.take_events();
        c.close(DisconnectReason::Kicked);
        c.close(DisconnectReason::Normal);
        let out = c.take_outgoing();
        assert_eq!(
            out.iter()
                .filter(|o| matches!(o.msg, ControlMsg::Bye { .. }))
                .count(),
            1,
            "one goodbye, and it is the first reason"
        );
        assert_eq!(c.state(), SessionState::Ended(EndReason::LocalClose));
        net.route(out, true);
        net.run(&mut c, &mut s, 400);
        assert_eq!(
            s.state(),
            SessionState::Ended(EndReason::PeerClosed(DisconnectReason::Kicked))
        );
        assert_eq!(
            s.take_events(),
            vec![SessionEvent::Ended(EndReason::PeerClosed(
                DisconnectReason::Kicked
            ))]
        );
        assert_eq!(
            DisconnectReason::from_wire(200),
            DisconnectReason::Unknown(200)
        );
        for code in 0..=9u8 {
            assert_eq!(DisconnectReason::from_wire(code).to_wire(), code);
        }
    }

    // ---- the messages on the wire ----------------------------------------------------------------------------------------

    #[test]
    fn control_messages_are_small_and_round_trip() {
        let all = vec![
            ControlMsg::Hello {
                body: HelloBody {
                    protocol: 1,
                    schema: u64::MAX,
                    nonce: 7,
                    name: "Brou ✓".into(),
                },
            },
            ControlMsg::Welcome {
                body: WelcomeBody {
                    nonce: 7,
                    session_id: 9,
                    server_time_ms: 123_456_789,
                    tick_rate_hz: 128,
                },
            },
            ControlMsg::Reject {
                body: RejectBody {
                    nonce: 1,
                    reason: 2,
                },
            },
            ControlMsg::Ping {
                body: PingBody {
                    seq: 65535,
                    client_time_ms: 600_000,
                },
            },
            ControlMsg::Pong {
                body: PongBody {
                    seq: 1,
                    client_time_ms: 600_000,
                    server_time_ms: 7_000_000,
                },
            },
            ControlMsg::Bye {
                body: ByeBody { reason: 6 },
            },
        ];
        let mut sizes = Vec::new();
        for m in all {
            let mut buf = [0u8; 64];
            let n = net_write(&mut m.clone(), &mut buf).unwrap();
            sizes.push(n);
            assert_eq!(net_read::<ControlMsg>(&buf[..n]).unwrap(), m);
        }
        assert!(
            sizes[3] <= 8 && sizes[4] <= 12,
            "a ping is at most 8 bytes and a pong 12: {sizes:?}"
        );
        assert_eq!(
            net_read::<ControlMsg>(&[7]),
            Err(BitError::UnknownTag),
            "an unknown kind of control message"
        );
    }

    #[test]
    fn a_whole_connection_runs_through_the_packet_and_stream_framing() {
        let mut reg = MessageRegistry::new(4);
        reg.register::<ControlMsg>().unwrap();
        let mut ch = ChannelTable::new(1).unwrap();
        ch.define(0, ChannelKind::UnreliableUnordered).unwrap();
        let (mut c, mut s) = pair(11, 11);
        let (mut c_stream, mut s_stream) =
            (StreamWriter::new(4096, 256), StreamWriter::new(4096, 256));
        let (mut c_in, mut s_in) = (StreamReader::new(256), StreamReader::new(256));
        let (mut c_rx, mut s_rx) = (DatagramReceiver::new(), DatagramReceiver::new());
        let (mut c_seq, mut s_seq) = (Sequence(0), Sequence(0));
        for now in (0..3_000u64).step_by(10) {
            c.poll(now);
            s.poll(now + SERVER_CLOCK_OFFSET);
            let mut from_c = Vec::new();
            let mut from_s = Vec::new();
            for (sess, stream, seq, out) in [
                (&mut c, &mut c_stream, &mut c_seq, &mut from_c),
                (&mut s, &mut s_stream, &mut s_seq, &mut from_s),
            ] {
                for mut o in sess.take_outgoing() {
                    match o.delivery {
                        Delivery::Reliable => stream.push(&reg, &mut o.msg).unwrap(),
                        Delivery::Unreliable => {
                            let mut buf = [0u8; 100];
                            let mut b = DatagramBuilder::new(&mut buf, *seq, &reg, &ch).unwrap();
                            assert!(b.push(0, &mut o.msg).unwrap());
                            let n = b.finish();
                            *seq = seq.wrapping_next();
                            out.push(buf[..n].to_vec());
                        }
                    }
                }
            }
            // the streams' bytes arrive in pieces of 5
            for (stream, input, recv, sess, t, net) in [(
                &mut c_stream,
                &mut s_in,
                &mut s,
                now + SERVER_CLOCK_OFFSET,
                0,
                0,
            )]
            .map(|(a, b, c, d, _, _)| (a, b, c, d))
            .into_iter()
            .map(|(a, b, c, d)| (a, b, c, d, 0, 0))
            {
                let _ = (t, net);
                let mut piece = [0u8; 5];
                loop {
                    let n = stream.fill(&mut piece);
                    if n == 0 {
                        break;
                    }
                    input.feed(&piece[..n]).unwrap();
                    recv.on_packet_received(sess);
                }
            }
            let mut piece = [0u8; 5];
            loop {
                let n = s_stream.fill(&mut piece);
                if n == 0 {
                    break;
                }
                c_in.feed(&piece[..n]).unwrap();
                c.on_packet_received(now);
            }
            while let Some(m) = s_in.next_message(&reg).unwrap() {
                s.on_control(
                    now + SERVER_CLOCK_OFFSET,
                    *m.downcast::<ControlMsg>().ok().unwrap(),
                );
            }
            while let Some(m) = c_in.next_message(&reg).unwrap() {
                c.on_control(now, *m.downcast::<ControlMsg>().ok().unwrap());
            }
            for d in from_c {
                s_rx.receive(&d, &ch, &reg, |_, m| {
                    s.on_control(
                        now + SERVER_CLOCK_OFFSET,
                        *m.downcast::<ControlMsg>().ok().unwrap(),
                    )
                })
                .unwrap();
            }
            for d in from_s {
                c_rx.receive(&d, &ch, &reg, |_, m| {
                    c.on_control(now, *m.downcast::<ControlMsg>().ok().unwrap())
                })
                .unwrap();
            }
        }
        assert_eq!(
            (c.state(), s.state()),
            (SessionState::Connected, SessionState::Connected)
        );
        assert!(
            c.last_rtt_ms().is_some() && c.server_time_ms(3_000).is_some(),
            "pings and pongs crossed too"
        );
    }

    // ---- hostile and long -------------------------------------------------------------------------------------------------

    fn random_msg(rng: &mut Rng) -> ControlMsg {
        match rng.below(6) {
            0 => ControlMsg::Hello {
                body: HelloBody {
                    protocol: rng.below(3) as u16,
                    schema: rng.below(3),
                    nonce: rng.below(3) as u32,
                    name: "n".into(),
                },
            },
            1 => ControlMsg::Welcome {
                body: WelcomeBody {
                    nonce: rng.below(3) as u32,
                    session_id: 1,
                    server_time_ms: rng.next() >> 20,
                    tick_rate_hz: rng.below(200) as u16,
                },
            },
            2 => ControlMsg::Reject {
                body: RejectBody {
                    nonce: rng.below(3) as u32,
                    reason: rng.below(12) as u8,
                },
            },
            3 => ControlMsg::Ping {
                body: PingBody {
                    seq: rng.next() as u16,
                    client_time_ms: rng.next() >> 30,
                },
            },
            4 => ControlMsg::Pong {
                body: PongBody {
                    seq: 1,
                    client_time_ms: rng.next() >> 40,
                    server_time_ms: rng.next() >> 20,
                },
            },
            _ => ControlMsg::Bye {
                body: ByeBody {
                    reason: rng.below(12) as u8,
                },
            },
        }
    }

    #[test]
    fn random_messages_at_random_times_never_panic_and_an_ended_session_stays_ended() {
        let mut rng = Rng(0xC0FF_EE00_1234_5678);
        for round in 0..3_000 {
            let mut sess = if round % 2 == 0 {
                Session::client(cfg(rng.below(3)), "r", rng.below(3) as u32, 0)
            } else {
                Session::server(cfg(rng.below(3)), 1, 0)
            };
            let mut now = 0u64;
            let mut ended_at: Option<SessionState> = None;
            let mut ends = 0;
            for _ in 0..60 {
                now = if rng.below(10) == 0 {
                    now.saturating_sub(rng.below(500))
                } else {
                    now + rng.below(2_000)
                };
                match rng.below(4) {
                    0 => sess.poll(now),
                    1 => sess.on_packet_received(now),
                    2 => sess.close(DisconnectReason::Normal),
                    _ => sess.on_control(now, random_msg(&mut rng)),
                }
                ends += sess
                    .take_events()
                    .iter()
                    .filter(|e| matches!(e, SessionEvent::Ended(_)))
                    .count();
                if let Some(st) = ended_at {
                    assert_eq!(sess.state(), st, "round {round}: an ended session changed");
                    assert!(
                        sess.take_outgoing().is_empty(),
                        "round {round}: an ended session produced output"
                    );
                } else if sess.is_ended() {
                    ended_at = Some(sess.state());
                    sess.take_outgoing();
                }
            }
            assert!(
                ends <= 1,
                "round {round}: the end was announced {ends} times"
            );
        }
    }

    #[test]
    fn an_hour_with_losses_and_gaps_shorter_than_the_timeout_never_ends_and_a_long_one_does_once() {
        let (mut c, mut s) = pair(1, 1);
        let mut net = Net::new(25, 35);
        net.jitter_ms = 20;
        net.loss_pct = 3;
        let mut rng = Rng(77);
        let mut t = 2_000u64;
        while t < 3_600_000 {
            let len = 200 + rng.below(6_000); // up to 6.2 s: over no_signal, under the 10 s timeout
            net.down.push((t, t + len));
            t += len + 5_000 + rng.below(60_000);
        }
        let planned = net.down.len();
        let (mut shows, mut recoveries, mut worst) = (0, 0, 0i64);
        while net.now < 3_600_000 {
            net.step(&mut c, &mut s);
            for e in c.take_events() {
                match e {
                    SessionEvent::NoSignal { .. } => shows += 1,
                    SessionEvent::Recovered { .. } => recoveries += 1,
                    SessionEvent::Ended(r) => panic!("ended at {}: {r:?}", net.now),
                    SessionEvent::Connected { .. } => {}
                }
            }
            s.take_events();
            if net.now > 10_000 && net.is_down().is_none() {
                if let Some(est) = c.server_time_ms(net.now) {
                    worst = worst.max((est as i64 - (net.now + SERVER_CLOCK_OFFSET) as i64).abs());
                }
            }
        }
        assert!(
            planned > 50 && shows > 50,
            "the hour had {planned} gaps and the client showed NO SIGNAL {shows} times"
        );
        assert!(
            recoveries == shows || recoveries + 1 == shows,
            "every NO SIGNAL was recovered ({shows} / {recoveries})"
        );
        assert!(
            worst < 200,
            "the clock estimate stayed within 200 ms even across the gaps: {worst}"
        );
        net.down.push((net.now + 100, net.now + 60_000));
        net.run(&mut c, &mut s, net.now + 30_000);
        assert_eq!(
            (c.state(), s.state()),
            (
                SessionState::Ended(EndReason::Timeout),
                SessionState::Ended(EndReason::Timeout)
            )
        );
    }
}
