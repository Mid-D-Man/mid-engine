//! The driver: the piece that owns a [`Transport`] and runs the layers below against a clock.
//!
//! This is the glue of `docs/mid-net-foundation.md`. Underneath it are parts that each do one thing and
//! touch no socket: the bit codec and message registry, the datagram and stream framing, the session
//! (handshake, keepalive, "NO SIGNAL"). The driver holds one of each, plus the transport, and answers two
//! questions every frame: *what arrived* and *what must go out now*.
//!
//! ```text
//!   game ── send(channel, priority, msg) ──▶ [ datagram queue | stream writers ] ──▶ Transport
//!   game ◀── next_message() / next_event() ── [ inbox, events ] ◀── DatagramReceiver / StreamReader ◀──┘
//!                                                   Session (handshake, ping, NO SIGNAL) ◀─ control ─┘
//! ```
//!
//! # Using it
//!
//! Build a [`Schema`] once (the message registry and the channel table, which both ends must agree on),
//! share it between every connection, and call [`Driver::poll`] with a millisecond clock every frame:
//!
//! ```text
//! driver.poll(now_ms);                       // read the transport, advance the session, flush what is queued
//! while let Some(ev) = driver.next_event()   { /* Connected, NoSignal, Recovered, Ended, Fault ... */ }
//! while let Some(m)  = driver.next_message() { /* a decoded message and the channel it came on      */ }
//! driver.send_typed(CHANNEL_STATE, priority::NORMAL, my_state)?;   // anywhere between polls
//! ```
//!
//! The driver never blocks and starts no thread: it is deterministic, so a test replays an exact
//! network (see [`crate::sim`]). [`crate::threaded::NetThread`] runs the same driver on a thread of its own
//! for a program that would rather not call `poll`.
//!
//! # Channels, priorities and the budget
//!
//! A **channel** is a number with a promise (see [`ChannelKind`]). Unreliable channels travel in datagrams;
//! each reliable channel is **its own transport stream** (stream id = channel id), so a stalled message on
//! one cannot hold back another, and each keeps its own order.
//!
//! Datagram messages wait in a queue. At every flush they are put in order of **priority** (higher first,
//! then first-come first-served within a priority) and packed into datagrams of at most `mtu` bytes, until
//! `datagram_budget` bytes have gone out this flush; the rest wait for the next. So when the link cannot
//! carry everything, state goes before decoration and the decoration is what is late. A message that has
//! waited `datagram_ttl_ms` or more is dropped (a stale state update is worse than none), and when the
//! queue itself is full the lowest-priority, oldest message goes. The session's own pings are above every
//! priority the game can use ([`priority::CONTROL`] is reserved), ignore the budget, and so a saturated
//! link never looks like a dead one.
//!
//! # What ends a connection
//!
//! Besides what the session decides (timeout, goodbye, refusal), the driver ends a connection, and says why
//! in a [`Fault`] event, when the transport fails or closes, when the peer sends bytes that cannot be
//! parsed on a stream or on a stream that was never agreed, and when a reliable queue overflows (the peer
//! is not reading, and waiting would grow memory without bound). A garbage *datagram* is only counted: the
//! network can deliver those from anywhere, and they are not a sign of life from the peer.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use mid_net_transport::Transport;
use mid_net_wire::{
    BitError, ChannelError, ChannelKind, ChannelTable, DatagramBuilder, DatagramReceiver,
    MessageRegistry, NetMessage, NetMessageType, PushError, RawMessage, RegisterError, Sequence,
    StreamError, StreamReader, StreamWriter,
};

use crate::session::{
    ControlMsg, Delivery, DisconnectReason, EndReason, Session, SessionConfig, SessionEvent,
    SessionState, CONTROL_MESSAGE_ID,
};

/// Priorities of datagram messages: a number, higher goes first. Any `u8` up to [`priority::MAX`] will do;
/// these are the names to start with.
pub mod priority {
    /// Cosmetic: footsteps, particles, anything nobody would miss.
    pub const COSMETIC: u8 = 0;
    pub const LOW: u8 = 64;
    pub const NORMAL: u8 = 128;
    pub const HIGH: u8 = 192;
    /// State the game cannot do without (the owner's own state, hits).
    pub const CRITICAL: u8 = 224;
    /// The highest a game may use.
    pub const MAX: u8 = 254;
    /// Reserved for the session's own messages.
    pub const CONTROL: u8 = 255;
}

/// The channel the default table uses for the session's reliable messages (handshake, goodbye).
pub const CONTROL_RELIABLE_CHANNEL: u8 = 0;
/// The channel the default table uses for the session's datagrams (ping, pong).
pub const CONTROL_DATAGRAM_CHANNEL: u8 = 1;

// ---------------------------------------------------------------------------------------------------------
// Schema
// ---------------------------------------------------------------------------------------------------------

/// Why a [`Schema`] could not be made.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchemaError {
    /// The session's message could not be registered (id 0 is reserved for it and was already taken, or the
    /// registry's id width is invalid).
    Register(RegisterError),
    /// The channel for the session's reliable messages is not a defined [`ChannelKind::ReliableOrdered`] one.
    ControlReliable(u8),
    /// The channel for the session's datagrams is not a defined datagram channel.
    ControlDatagram(u8),
}

/// What both ends must agree on: which messages exist and which channels there are. Made once, shared by
/// every connection (it is read-only after this), and the thing a schema hash in the handshake stands for.
pub struct Schema {
    registry: MessageRegistry,
    channels: ChannelTable,
    control_reliable: u8,
    control_datagram: u8,
}

impl Schema {
    /// Seals a registry and a channel table, registering the session's [`ControlMsg`] under id 0 in the
    /// registry (so the application must not use id 0) and checking that the two control channels exist.
    pub fn new(
        mut registry: MessageRegistry,
        channels: ChannelTable,
        control_reliable: u8,
        control_datagram: u8,
    ) -> Result<Schema, SchemaError> {
        registry
            .register::<ControlMsg>()
            .map_err(SchemaError::Register)?;
        if channels.kind(control_reliable) != Some(ChannelKind::ReliableOrdered) {
            return Err(SchemaError::ControlReliable(control_reliable));
        }
        match channels.kind(control_datagram) {
            Some(ChannelKind::UnreliableSequenced) | Some(ChannelKind::UnreliableUnordered) => {}
            _ => return Err(SchemaError::ControlDatagram(control_datagram)),
        }
        Ok(Schema {
            registry,
            channels,
            control_reliable,
            control_datagram,
        })
    }

    /// A channel table with the two control channels defined ([`CONTROL_RELIABLE_CHANNEL`] reliable,
    /// [`CONTROL_DATAGRAM_CHANNEL`] unordered) and room for the game's own: a good start to add to.
    pub fn default_channels(bits: u32) -> Result<ChannelTable, ChannelError> {
        let mut t = ChannelTable::new(bits)?;
        t.define(CONTROL_RELIABLE_CHANNEL, ChannelKind::ReliableOrdered)?;
        t.define(CONTROL_DATAGRAM_CHANNEL, ChannelKind::UnreliableUnordered)?;
        Ok(t)
    }

    pub fn registry(&self) -> &MessageRegistry {
        &self.registry
    }
    pub fn channels(&self) -> &ChannelTable {
        &self.channels
    }
    pub fn control_reliable(&self) -> u8 {
        self.control_reliable
    }
    pub fn control_datagram(&self) -> u8 {
        self.control_datagram
    }
}

// ---------------------------------------------------------------------------------------------------------
// Configuration, errors, events
// ---------------------------------------------------------------------------------------------------------

/// How a driver behaves. [`Default`] is a sensible starting point for a game at 30 Hz.
#[derive(Debug, Clone, Copy)]
pub struct DriverConfig {
    /// The session's timings and the schema hash.
    pub session: SessionConfig,
    /// The largest datagram, in bytes.
    pub mtu: usize,
    /// The most datagram bytes sent in one flush (at least `mtu`). Session pings do not count.
    pub datagram_budget: usize,
    /// The most reliable stream bytes handed to the transport in one flush.
    pub reliable_budget: usize,
    /// How many datagram messages may wait; past it the lowest priority, oldest one is dropped.
    pub max_queued_datagrams: usize,
    /// A datagram message that has waited this long is dropped (0: never).
    pub datagram_ttl_ms: u64,
    /// How many bytes may wait on one reliable channel; past it the connection is ended.
    pub max_stream_pending: usize,
    /// The largest message on a reliable channel, in bytes.
    pub max_frame: usize,
    /// How many received messages may wait for the game; while it is full the driver stops reading the
    /// transport (which pushes back on the peer) instead of growing.
    pub max_inbox: usize,
    /// The most reads from the transport in one poll.
    pub max_receive_per_poll: usize,
}

impl Default for DriverConfig {
    fn default() -> Self {
        DriverConfig {
            session: SessionConfig::default(),
            mtu: 1200,
            datagram_budget: 8 * 1200,
            reliable_budget: 32 * 1024,
            max_queued_datagrams: 1024,
            datagram_ttl_ms: 500,
            max_stream_pending: 256 * 1024,
            max_frame: 16 * 1024,
            max_inbox: 4096,
            max_receive_per_poll: 1024,
        }
    }
}

/// The cap on messages that arrive before the handshake has finished (held, then released when it has).
const MAX_HELD: usize = 256;

impl DriverConfig {
    fn validate(&self) -> Result<(), DriverError> {
        if self.mtu < 64 {
            return Err(DriverError::BadConfig("mtu must be at least 64"));
        }
        if self.datagram_budget < self.mtu {
            return Err(DriverError::BadConfig("datagram_budget must be at least mtu"));
        }
        if self.reliable_budget == 0 {
            return Err(DriverError::BadConfig("reliable_budget must not be 0"));
        }
        if self.max_queued_datagrams == 0 || self.max_inbox == 0 || self.max_receive_per_poll == 0 {
            return Err(DriverError::BadConfig("queue and poll limits must not be 0"));
        }
        if self.max_frame < 16 || self.max_stream_pending < self.max_frame {
            return Err(DriverError::BadConfig(
                "max_frame must be at least 16 and max_stream_pending at least max_frame",
            ));
        }
        Ok(())
    }
}

/// Why a driver could not be made.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DriverError {
    BadConfig(&'static str),
}

/// Why a message was not accepted by [`Driver::send_boxed`] and friends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SendError {
    /// The handshake has not finished.
    NotConnected,
    /// The connection has ended.
    Ended,
    /// The channel is not defined.
    BadChannel,
    /// The message type (or id) is not in the schema.
    Unregistered,
    /// Message id 0 belongs to the session.
    Reserved,
    /// A reliable queue is full: the peer is not reading. The connection has been ended.
    QueueFull,
    /// The message is bigger than a frame may be.
    TooLarge,
    /// The message could not be encoded (a value that does not fit its field).
    Encode(BitError),
    /// (Servers) there is no such connection.
    NoSuchConnection,
}

/// Why a datagram message was dropped before it was sent (reported, because each of these is a bug in the
/// sender, unlike a message that merely expired or was crowded out, which are counted in [`DriverStats`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DropReason {
    /// It does not fit in a datagram even alone.
    Oversize,
    /// It could not be encoded.
    Encode(BitError),
    /// Its type or channel was not usable at send time.
    Unusable,
}

/// A reason the driver ended the connection by itself.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Fault {
    /// The transport returned an error (its `Debug` text).
    Transport(String),
    /// The transport was connected and no longer is.
    TransportClosed,
    /// Bytes arrived on a stream the schema does not have.
    UnknownStream(u32),
    /// A reliable stream sent more than a frame without completing it.
    StreamOverflow { channel: u8 },
    /// A reliable stream carried something that is not a message.
    MalformedStream { channel: u8, error: BitError },
    /// A reliable queue was full.
    SendQueueFull { channel: u8 },
}

/// What happened, for the game to show or log.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DriverEvent {
    /// The session's own events: connected, no signal, recovered, ended.
    Session(SessionEvent),
    /// The driver ended the connection by itself.
    Fault(Fault),
    /// A datagram message was dropped as unsendable.
    Dropped {
        channel: u8,
        id: u16,
        reason: DropReason,
    },
    /// (Threaded runner) a message handed to the thread was refused.
    SendRejected {
        channel: u8,
        id: u16,
        error: SendError,
    },
}

/// What the driver counted. Every field only grows.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DriverStats {
    pub datagrams_sent: u64,
    pub datagram_bytes_sent: u64,
    pub datagrams_received: u64,
    /// Datagrams that could not be parsed, and were ignored.
    pub bad_datagrams: u64,
    /// Datagrams whose packet number had been seen (the network delivered it twice).
    pub duplicate_datagrams: u64,
    /// Messages dropped on receive because a newer one of a sequenced channel had been delivered.
    pub stale_messages: u64,
    pub messages_sent: u64,
    pub messages_received: u64,
    pub reliable_bytes_sent: u64,
    pub reliable_bytes_received: u64,
    /// Datagram messages dropped for waiting longer than the ttl.
    pub dropped_expired: u64,
    /// Datagram messages dropped because the queue was full.
    pub dropped_queue_full: u64,
    /// Datagram messages dropped as unsendable (also reported as [`DriverEvent::Dropped`]).
    pub dropped_unsendable: u64,
    /// Messages that arrived before the handshake finished and could not be held.
    pub dropped_early: u64,
}

// ---------------------------------------------------------------------------------------------------------
// Messages in and out
// ---------------------------------------------------------------------------------------------------------

/// A received message and the channel it came on.
#[derive(Debug)]
pub struct Inbound {
    pub channel: u8,
    pub msg: Box<dyn NetMessage>,
}

impl Inbound {
    /// The message as a `T`, if that is what it is.
    pub fn downcast<T: NetMessageType>(self) -> Result<T, Inbound> {
        let channel = self.channel;
        self.msg
            .downcast::<T>()
            .map(|b| *b)
            .map_err(|msg| Inbound { channel, msg })
    }

    /// The message as opaque bytes, if its id was registered with
    /// [`MessageRegistry::register_raw`](mid_net_wire::MessageRegistry::register_raw).
    pub fn into_raw(self) -> Result<(u8, RawMessage), Inbound> {
        let channel = self.channel;
        self.msg
            .into_raw()
            .map(|b| (channel, *b))
            .map_err(|msg| Inbound { channel, msg })
    }
}

struct Pending {
    channel: u8,
    priority: u8,
    order: u64,
    queued_ms: u64,
    msg: Box<dyn NetMessage>,
}

struct StreamPair {
    channel: u8,
    writer: StreamWriter,
    reader: StreamReader,
}

// ---------------------------------------------------------------------------------------------------------
// The driver
// ---------------------------------------------------------------------------------------------------------

/// One end of one connection: a transport, a session and the queues between them. See the module docs.
pub struct Driver<T: Transport> {
    schema: Arc<Schema>,
    cfg: DriverConfig,
    transport: T,
    session: Session,
    receiver: DatagramReceiver,
    next_seq: Sequence,
    streams: Vec<StreamPair>,
    stream_of: [Option<usize>; 256],
    pending: Vec<Pending>,
    order: u64,
    inbox: VecDeque<Inbound>,
    held: VecDeque<Inbound>,
    events: VecDeque<DriverEvent>,
    stats: DriverStats,
    now_ms: u64,
    was_connected: bool,
    packet: Vec<u8>,
    scratch: Vec<u8>,
}

impl<T: Transport> Driver<T> {
    /// The client end: it says hello on the first poll and retries until the server answers.
    pub fn client(
        schema: Arc<Schema>,
        cfg: DriverConfig,
        transport: T,
        name: &str,
        nonce: u32,
        now_ms: u64,
    ) -> Result<Self, DriverError> {
        let session = Session::client(cfg.session, name, nonce, now_ms);
        Self::build(schema, cfg, transport, session, now_ms)
    }

    /// The server's end of one client's connection (the transport is already accepted).
    pub fn server_end(
        schema: Arc<Schema>,
        cfg: DriverConfig,
        transport: T,
        session_id: u32,
        now_ms: u64,
    ) -> Result<Self, DriverError> {
        let session = Session::server(cfg.session, session_id, now_ms);
        Self::build(schema, cfg, transport, session, now_ms)
    }

    fn build(
        schema: Arc<Schema>,
        cfg: DriverConfig,
        transport: T,
        session: Session,
        now_ms: u64,
    ) -> Result<Self, DriverError> {
        cfg.validate()?;
        let mut channels: Vec<u8> = (0..=255u8)
            .filter(|c| schema.channels.kind(*c) == Some(ChannelKind::ReliableOrdered))
            .collect();
        // the session's own stream is flushed first
        channels.sort_by_key(|c| (*c != schema.control_reliable, *c));
        let mut stream_of = [None; 256];
        let mut streams = Vec::with_capacity(channels.len());
        for (i, c) in channels.into_iter().enumerate() {
            stream_of[c as usize] = Some(i);
            streams.push(StreamPair {
                channel: c,
                writer: StreamWriter::new(cfg.max_stream_pending, cfg.max_frame),
                reader: StreamReader::new(cfg.max_frame),
            });
        }
        Ok(Driver {
            schema,
            cfg,
            transport,
            session,
            receiver: DatagramReceiver::new(),
            next_seq: Sequence(0),
            streams,
            stream_of,
            pending: Vec::new(),
            order: 0,
            inbox: VecDeque::new(),
            held: VecDeque::new(),
            events: VecDeque::new(),
            stats: DriverStats::default(),
            now_ms,
            was_connected: false,
            packet: vec![0; cfg.mtu],
            scratch: vec![0; 16 * 1024],
        })
    }

    // ----- the loop -------------------------------------------------------------------------------------

    /// Does everything due at `now_ms` (any monotonic millisecond clock): reads what the transport has,
    /// advances the session, and sends what is queued. Call it every frame; it never blocks.
    pub fn poll(&mut self, now_ms: u64) {
        self.now_ms = now_ms.max(self.now_ms);
        let now = self.now_ms;
        if !self.session.is_ended() {
            self.receive(now);
        }
        self.session.poll(now);
        self.flush(now);
        for e in self.session.take_events() {
            self.events.push_back(DriverEvent::Session(e));
        }
    }

    /// Ends the connection on purpose; the goodbye goes out with the next [`poll`](Self::poll).
    pub fn close(&mut self, reason: DisconnectReason) {
        self.session.close(reason);
    }

    // ----- what the game reads --------------------------------------------------------------------------

    /// The next received message, oldest first.
    pub fn next_message(&mut self) -> Option<Inbound> {
        self.inbox.pop_front()
    }

    /// The message [`next_message`](Self::next_message) would return, without taking it.
    pub fn peek_message(&self) -> Option<&Inbound> {
        self.inbox.front()
    }

    /// The next event, oldest first.
    pub fn next_event(&mut self) -> Option<DriverEvent> {
        self.events.pop_front()
    }

    pub fn messages_waiting(&self) -> usize {
        self.inbox.len()
    }

    pub fn state(&self) -> SessionState {
        self.session.state()
    }

    /// Connected, or connected and currently without signal (the game decides what to show).
    pub fn is_connected(&self) -> bool {
        matches!(
            self.session.state(),
            SessionState::Connected | SessionState::NoSignal { .. }
        )
    }

    pub fn is_ended(&self) -> bool {
        self.session.is_ended()
    }

    pub fn session(&self) -> &Session {
        &self.session
    }

    /// The server's clock, in milliseconds, as estimated from the keepalive (`None` before the first answer).
    pub fn server_time_ms(&self, now_ms: u64) -> Option<u64> {
        self.session.server_time_ms(now_ms)
    }

    pub fn rtt_ms(&self) -> f64 {
        self.session.rtt_ms()
    }

    pub fn stats(&self) -> DriverStats {
        self.stats
    }

    /// How many datagram messages are waiting to be sent.
    pub fn queued_datagrams(&self) -> usize {
        self.pending.len()
    }

    /// How many bytes wait unsent on a reliable channel (`None` if it is not one).
    pub fn queued_stream_bytes(&self, channel: u8) -> Option<usize> {
        self.stream_of[channel as usize].map(|i| self.streams[i].writer.pending_bytes())
    }

    pub fn schema(&self) -> &Arc<Schema> {
        &self.schema
    }

    pub fn transport(&self) -> &T {
        &self.transport
    }

    pub fn transport_mut(&mut self) -> &mut T {
        &mut self.transport
    }

    // ----- sending --------------------------------------------------------------------------------------

    /// Sends a message of a known type. See [`send_boxed`](Self::send_boxed).
    pub fn send_typed<M: NetMessageType>(
        &mut self,
        channel: u8,
        priority: u8,
        msg: M,
    ) -> Result<(), SendError> {
        self.send_boxed(channel, priority, Box::new(msg))
    }

    /// Sends opaque bytes under a message id registered with `register_raw`.
    pub fn send_raw(
        &mut self,
        channel: u8,
        priority: u8,
        id: u16,
        data: Vec<u8>,
    ) -> Result<(), SendError> {
        self.send_boxed(channel, priority, Box::new(RawMessage { id, data }))
    }

    /// Sends a message. On a datagram channel it is queued and goes out at the next flush in `priority`
    /// order (reliable channels ignore the priority: they are ordered streams). `Ok` means accepted, not
    /// delivered: an unreliable message may still be lost, expire, or be crowded out of a full queue.
    pub fn send_boxed(
        &mut self,
        channel: u8,
        priority: u8,
        msg: Box<dyn NetMessage>,
    ) -> Result<(), SendError> {
        match self.session.state() {
            SessionState::Connecting => return Err(SendError::NotConnected),
            SessionState::Ended(_) => return Err(SendError::Ended),
            SessionState::Connected | SessionState::NoSignal { .. } => {}
        }
        if msg.id() == CONTROL_MESSAGE_ID {
            return Err(SendError::Reserved);
        }
        self.enqueue(channel, priority.min(priority::MAX), msg)
    }

    fn enqueue(
        &mut self,
        channel: u8,
        priority: u8,
        mut msg: Box<dyn NetMessage>,
    ) -> Result<(), SendError> {
        if !self.schema.registry.is_registered(msg.id()) {
            return Err(SendError::Unregistered);
        }
        match self.schema.channels.kind(channel) {
            None => Err(SendError::BadChannel),
            Some(ChannelKind::ReliableOrdered) => {
                let ix = self.stream_of[channel as usize].ok_or(SendError::BadChannel)?;
                match self.streams[ix]
                    .writer
                    .push_dyn(&self.schema.registry, msg.as_mut())
                {
                    Ok(()) => {
                        self.stats.messages_sent += 1;
                        Ok(())
                    }
                    Err(StreamError::Full) => {
                        self.fault(Fault::SendQueueFull { channel }, DisconnectReason::ProtocolError);
                        Err(SendError::QueueFull)
                    }
                    Err(StreamError::TooLarge) => Err(SendError::TooLarge),
                    Err(StreamError::Unregistered) => Err(SendError::Unregistered),
                    Err(StreamError::Encode(e)) => Err(SendError::Encode(e)),
                }
            }
            Some(_) => {
                self.queue_datagram(channel, priority, msg);
                self.stats.messages_sent += 1;
                Ok(())
            }
        }
    }

    fn queue_datagram(&mut self, channel: u8, priority: u8, msg: Box<dyn NetMessage>) {
        if self.pending.len() >= self.cfg.max_queued_datagrams {
            // make room: the lowest priority goes, the oldest of those; unless the newcomer is lower still
            let victim = self
                .pending
                .iter()
                .enumerate()
                .min_by_key(|(_, p)| (p.priority, p.order))
                .map(|(i, p)| (i, p.priority));
            match victim {
                Some((i, vp)) if vp <= priority => {
                    self.pending.swap_remove(i);
                    self.stats.dropped_queue_full += 1;
                }
                _ => {
                    self.stats.dropped_queue_full += 1;
                    return;
                }
            }
        }
        self.order += 1;
        self.pending.push(Pending {
            channel,
            priority,
            order: self.order,
            queued_ms: self.now_ms,
            msg,
        });
    }

    // ----- faults ---------------------------------------------------------------------------------------

    /// Records why the driver is ending the connection, and ends it. Nothing is recorded for a connection
    /// that has already ended: what follows an end is noise.
    fn fault(&mut self, fault: Fault, reason: DisconnectReason) {
        if self.session.is_ended() {
            return;
        }
        self.events.push_back(DriverEvent::Fault(fault));
        self.session.close(reason);
    }

    fn transport_fault(&mut self, e: T::Error) {
        self.fault(
            Fault::Transport(format!("{e:?}")),
            DisconnectReason::Normal,
        );
    }

    // ----- receiving ------------------------------------------------------------------------------------

    fn receive(&mut self, now: u64) {
        if !self.transport.is_connected() {
            if self.was_connected {
                self.was_connected = false;
                self.fault(Fault::TransportClosed, DisconnectReason::Normal);
            }
            return;
        }
        self.was_connected = true;
        let mut budget = self.cfg.max_receive_per_poll;

        // The reliable side first: the handshake travels on it, and what is read after it can then be delivered.
        while budget > 0 && self.inbox.len() < self.cfg.max_inbox && !self.session.is_ended() {
            match self.transport.poll_reliable() {
                Ok(Some((stream_id, bytes))) => {
                    budget -= 1;
                    self.on_stream_bytes(stream_id, &bytes, now);
                }
                Ok(None) => break,
                Err(e) => {
                    self.transport_fault(e);
                    return;
                }
            }
        }
        while budget > 0 && self.inbox.len() < self.cfg.max_inbox && !self.session.is_ended() {
            match self.transport.poll_datagram() {
                Ok(Some(bytes)) => {
                    budget -= 1;
                    self.on_datagram(&bytes, now);
                }
                Ok(None) => break,
                Err(e) => {
                    self.transport_fault(e);
                    return;
                }
            }
        }
    }

    fn on_datagram(&mut self, bytes: &[u8], now: u64) {
        let schema = Arc::clone(&self.schema);
        let mut got: Vec<(u8, Box<dyn NetMessage>)> = Vec::new();
        let result = self
            .receiver
            .receive(bytes, &schema.channels, &schema.registry, |ch, m| {
                got.push((ch, m))
            });
        match result {
            Ok(report) => {
                self.stats.datagrams_received += 1;
                self.stats.stale_messages += report.stale as u64;
                if report.duplicate_packet {
                    self.stats.duplicate_datagrams += 1;
                }
                self.session.on_packet_received(now);
                for (channel, msg) in got {
                    self.route(channel, msg, now);
                }
            }
            Err(_) => self.stats.bad_datagrams += 1,
        }
    }

    fn on_stream_bytes(&mut self, stream_id: u32, bytes: &[u8], now: u64) {
        let ix = match u8::try_from(stream_id)
            .ok()
            .and_then(|c| self.stream_of[c as usize])
        {
            Some(ix) => ix,
            None => {
                self.fault(Fault::UnknownStream(stream_id), DisconnectReason::ProtocolError);
                return;
            }
        };
        let channel = self.streams[ix].channel;
        self.stats.reliable_bytes_received += bytes.len() as u64;
        self.session.on_packet_received(now);
        if self.streams[ix].reader.feed(bytes).is_err() {
            self.fault(Fault::StreamOverflow { channel }, DisconnectReason::ProtocolError);
            return;
        }
        let schema = Arc::clone(&self.schema);
        let mut got: Vec<Box<dyn NetMessage>> = Vec::new();
        let mut failed = None;
        loop {
            match self.streams[ix].reader.next_message(&schema.registry) {
                Ok(Some(m)) => got.push(m),
                Ok(None) => break,
                Err(e) => {
                    failed = Some(e);
                    break;
                }
            }
        }
        for msg in got {
            self.route(channel, msg, now);
        }
        if let Some(error) = failed {
            self.fault(
                Fault::MalformedStream { channel, error },
                DisconnectReason::ProtocolError,
            );
        }
    }

    /// Hands a decoded message to the session (if it is one of its own) or to the game.
    fn route(&mut self, channel: u8, msg: Box<dyn NetMessage>, now: u64) {
        if msg.id() == CONTROL_MESSAGE_ID {
            if let Ok(control) = msg.downcast::<ControlMsg>() {
                self.session.on_control(now, *control);
                self.release_held();
            }
            return;
        }
        self.stats.messages_received += 1;
        let inbound = Inbound { channel, msg };
        if self.is_connected() {
            self.inbox.push_back(inbound);
        } else if self.held.len() < MAX_HELD {
            // a message that got here ahead of the handshake (the streams are independent): keep it for later
            self.held.push_back(inbound);
        } else {
            self.stats.dropped_early += 1;
        }
    }

    fn release_held(&mut self) {
        if self.is_connected() {
            while let Some(m) = self.held.pop_front() {
                self.inbox.push_back(m);
            }
        }
    }

    // ----- sending side of the loop ---------------------------------------------------------------------

    fn flush(&mut self, now: u64) {
        // the session's own messages first: they become queue entries like any other
        let outgoing = self.session.take_outgoing();
        for o in outgoing {
            match o.delivery {
                Delivery::Reliable => {
                    let ix = self.stream_of[self.schema.control_reliable as usize]
                        .expect("the control channel is a defined reliable channel");
                    let mut msg = o.msg;
                    match self.streams[ix]
                        .writer
                        .push_dyn(&self.schema.registry, &mut msg)
                    {
                        Ok(()) => {}
                        Err(_) => self.fault(
                            Fault::SendQueueFull {
                                channel: self.schema.control_reliable,
                            },
                            DisconnectReason::ProtocolError,
                        ),
                    }
                }
                Delivery::Unreliable => {
                    self.order += 1;
                    self.pending.push(Pending {
                        channel: self.schema.control_datagram,
                        priority: priority::CONTROL,
                        order: self.order,
                        queued_ms: now,
                        msg: Box::new(o.msg),
                    });
                }
            }
        }
        // once the connection has ended, the game's messages are not sent; only the session's own may be
        if self.session.is_ended() {
            self.pending.retain(|p| p.priority == priority::CONTROL);
        }
        // waiting too long is the end of an unreliable message, connected or not
        let ttl = self.cfg.datagram_ttl_ms;
        if ttl > 0 {
            let before = self.pending.len();
            self.pending
                .retain(|p| p.priority == priority::CONTROL || now.saturating_sub(p.queued_ms) < ttl);
            self.stats.dropped_expired += (before - self.pending.len()) as u64;
        }
        if !self.transport.is_connected() {
            return;
        }
        self.flush_reliable();
        self.flush_datagrams();
    }

    fn flush_reliable(&mut self) {
        let mut budget = self.cfg.reliable_budget;
        for ix in 0..self.streams.len() {
            while budget > 0 {
                let want = budget.min(self.scratch.len());
                let n = self.streams[ix].writer.fill(&mut self.scratch[..want]);
                if n == 0 {
                    break;
                }
                budget -= n;
                let stream_id = self.streams[ix].channel as u32;
                if let Err(e) = self.transport.send_reliable(stream_id, &self.scratch[..n]) {
                    self.transport_fault(e);
                    return;
                }
                self.stats.reliable_bytes_sent += n as u64;
            }
        }
    }

    fn flush_datagrams(&mut self) {
        if self.pending.is_empty() {
            return;
        }
        self.pending
            .sort_by(|a, b| b.priority.cmp(&a.priority).then(a.order.cmp(&b.order)));
        let mut control: VecDeque<Pending> = std::mem::take(&mut self.pending).into();
        let n_control = control
            .iter()
            .take_while(|p| p.priority == priority::CONTROL)
            .count();
        let mut rest = control.split_off(n_control);
        // the session's pings ignore the budget; everything else shares it
        let ok = self.pack(&mut control, None) && self.pack(&mut rest, Some(self.cfg.datagram_budget));
        if ok {
            self.pending = rest.into_iter().collect();
        }
    }

    /// Packs `queue` front to back into datagrams and sends them, until it is empty or `budget` bytes
    /// have gone out. What is left stays in `queue`. `false` if the transport failed.
    fn pack(&mut self, queue: &mut VecDeque<Pending>, budget: Option<usize>) -> bool {
        let schema = Arc::clone(&self.schema);
        let mut used = 0usize;
        while !queue.is_empty() {
            let mut included: Vec<Pending> = Vec::new();
            let mut dropped: Vec<(u8, u16, DropReason)> = Vec::new();
            let len = {
                let mut b = match DatagramBuilder::new(
                    &mut self.packet,
                    self.next_seq,
                    &schema.registry,
                    &schema.channels,
                ) {
                    Ok(b) => b,
                    Err(_) => return false, // the buffer cannot hold a header: validate() rules this out
                };
                while let Some(front) = queue.front_mut() {
                    let outcome = b.push_dyn(front.channel, front.msg.as_mut());
                    let reason = match outcome {
                        Ok(true) => {
                            if let Some(p) = queue.pop_front() {
                                included.push(p);
                            }
                            continue;
                        }
                        Ok(false) if b.message_count() > 0 => break, // full: send this one, then go on
                        Ok(false) => DropReason::Oversize,
                        Err(PushError::Encode(e)) => DropReason::Encode(e),
                        Err(PushError::BadChannel) | Err(PushError::Unregistered) => {
                            DropReason::Unusable
                        }
                    };
                    if let Some(p) = queue.pop_front() {
                        dropped.push((p.channel, p.msg.id(), reason));
                    }
                }
                b.finish()
            };
            for (channel, id, reason) in dropped {
                self.stats.dropped_unsendable += 1;
                self.events.push_back(DriverEvent::Dropped {
                    channel,
                    id,
                    reason,
                });
            }
            if included.is_empty() {
                continue;
            }
            if let Some(limit) = budget {
                if used > 0 && used + len > limit {
                    // no room left this flush: they go back to the front, in order, for the next
                    for p in included.into_iter().rev() {
                        queue.push_front(p);
                    }
                    return true;
                }
            }
            if let Err(e) = self.transport.send_datagram(&self.packet[..len]) {
                self.transport_fault(e);
                return false;
            }
            used += len;
            self.next_seq = self.next_seq.wrapping_next();
            self.stats.datagrams_sent += 1;
            self.stats.datagram_bytes_sent += len as u64;
        }
        true
    }
}

// ---------------------------------------------------------------------------------------------------------
// Typed handlers
// ---------------------------------------------------------------------------------------------------------

type Handler<C> = Box<dyn FnMut(&mut C, u8, Box<dyn NetMessage>)>;

/// Calls a typed handler for each received message, by message id. It is a convenience over
/// [`Driver::next_message`]: the handlers work on a context `C` the caller owns, and a message nobody
/// handles is handed back.
pub struct Router<C> {
    handlers: HashMap<u16, Handler<C>>,
}

impl<C> Default for Router<C> {
    fn default() -> Self {
        Router::new()
    }
}

impl<C> Router<C> {
    pub fn new() -> Self {
        Router {
            handlers: HashMap::new(),
        }
    }

    /// Calls `f(ctx, channel, message)` for each message of type `T`. A second handler for the same type
    /// replaces the first.
    pub fn on<T: NetMessageType>(&mut self, mut f: impl FnMut(&mut C, u8, T) + 'static) {
        self.handlers.insert(
            T::ID,
            Box::new(move |ctx, channel, msg| {
                if let Ok(v) = msg.downcast::<T>() {
                    f(ctx, channel, *v);
                }
            }),
        );
    }

    /// The same for opaque bytes registered under `id`.
    pub fn on_raw(&mut self, id: u16, mut f: impl FnMut(&mut C, u8, RawMessage) + 'static) {
        self.handlers.insert(
            id,
            Box::new(move |ctx, channel, msg| {
                if let Ok(v) = msg.into_raw() {
                    f(ctx, channel, *v);
                }
            }),
        );
    }

    /// Runs the handler of `inbound`'s type; gives the message back if there is none.
    pub fn route(&mut self, ctx: &mut C, inbound: Inbound) -> Result<(), Inbound> {
        match self.handlers.get_mut(&inbound.msg.id()) {
            Some(h) => {
                h(ctx, inbound.channel, inbound.msg);
                Ok(())
            }
            None => Err(inbound),
        }
    }

    /// Routes every message the driver has waiting. Returns `(handled, unhandled)`.
    pub fn drain<T: Transport>(&mut self, driver: &mut Driver<T>, ctx: &mut C) -> (usize, usize) {
        let (mut handled, mut unhandled) = (0, 0);
        while let Some(m) = driver.next_message() {
            match self.route(ctx, m) {
                Ok(()) => handled += 1,
                Err(_) => unhandled += 1,
            }
        }
        (handled, unhandled)
    }
}

// ---------------------------------------------------------------------------------------------------------
// The server: many connections
// ---------------------------------------------------------------------------------------------------------

/// One client of a [`Server`]. Also the session id the client is told.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ConnId(pub u32);

/// Why a connection was not accepted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AcceptError {
    /// The server is full. The client was told so (`Rejected(Full)`) as far as its transport allowed.
    Full,
    Config(DriverError),
}

/// Many clients at once: one [`Driver`] per connection, a shared [`Schema`], and one place to read from.
/// *Accepting* a connection is the transport backend's job (a QUIC listener, a relay); the server takes the
/// [`Transport`] it produced and from then on runs it.
pub struct Server<T: Transport> {
    schema: Arc<Schema>,
    cfg: DriverConfig,
    max_connections: usize,
    conns: Vec<(ConnId, Driver<T>)>,
    next_id: u32,
    inbox: VecDeque<(ConnId, Inbound)>,
    events: VecDeque<(ConnId, DriverEvent)>,
}

impl<T: Transport> Server<T> {
    pub fn new(
        schema: Arc<Schema>,
        cfg: DriverConfig,
        max_connections: usize,
    ) -> Result<Self, DriverError> {
        cfg.validate()?;
        Ok(Server {
            schema,
            cfg,
            max_connections,
            conns: Vec::new(),
            next_id: 1,
            inbox: VecDeque::new(),
            events: VecDeque::new(),
        })
    }

    /// Takes over a newly accepted connection. When the server is full the client is sent a goodbye with
    /// the reason `Full` (so it can say "server full" instead of timing out) and the transport is dropped.
    pub fn accept(&mut self, transport: T, now_ms: u64) -> Result<ConnId, AcceptError> {
        let id = ConnId(self.next_id);
        let mut driver = Driver::server_end(
            Arc::clone(&self.schema),
            self.cfg,
            transport,
            id.0,
            now_ms,
        )
        .map_err(AcceptError::Config)?;
        if self.conns.len() >= self.max_connections {
            driver.close(DisconnectReason::Full);
            driver.poll(now_ms);
            return Err(AcceptError::Full);
        }
        self.next_id = self.next_id.wrapping_add(1).max(1);
        self.conns.push((id, driver));
        Ok(id)
    }

    /// Polls every connection, collects what they received and what happened, and forgets the ones that
    /// ended (their `Ended` event is in the queue).
    pub fn poll(&mut self, now_ms: u64) {
        for (id, driver) in self.conns.iter_mut() {
            driver.poll(now_ms);
            while let Some(m) = driver.next_message() {
                self.inbox.push_back((*id, m));
            }
            while let Some(e) = driver.next_event() {
                self.events.push_back((*id, e));
            }
        }
        self.conns.retain(|(_, d)| !d.is_ended());
    }

    pub fn next_message(&mut self) -> Option<(ConnId, Inbound)> {
        self.inbox.pop_front()
    }

    /// The message [`next_message`](Self::next_message) would return, without taking it.
    pub fn peek_message(&self) -> Option<&(ConnId, Inbound)> {
        self.inbox.front()
    }

    pub fn next_event(&mut self) -> Option<(ConnId, DriverEvent)> {
        self.events.pop_front()
    }

    pub fn len(&self) -> usize {
        self.conns.len()
    }

    pub fn is_empty(&self) -> bool {
        self.conns.is_empty()
    }

    pub fn connection_ids(&self) -> Vec<ConnId> {
        self.conns.iter().map(|(id, _)| *id).collect()
    }

    pub fn connection(&self, id: ConnId) -> Option<&Driver<T>> {
        self.conns.iter().find(|(c, _)| *c == id).map(|(_, d)| d)
    }

    pub fn connection_mut(&mut self, id: ConnId) -> Option<&mut Driver<T>> {
        self.conns
            .iter_mut()
            .find(|(c, _)| *c == id)
            .map(|(_, d)| d)
    }

    /// Sends to one client.
    pub fn send(
        &mut self,
        id: ConnId,
        channel: u8,
        priority: u8,
        msg: Box<dyn NetMessage>,
    ) -> Result<(), SendError> {
        self.connection_mut(id)
            .ok_or(SendError::NoSuchConnection)?
            .send_boxed(channel, priority, msg)
    }

    /// Sends a message made by `make` (called once per client) to every connected client except `except`.
    /// Returns how many accepted it.
    pub fn broadcast(
        &mut self,
        except: Option<ConnId>,
        channel: u8,
        priority: u8,
        mut make: impl FnMut() -> Box<dyn NetMessage>,
    ) -> usize {
        let mut sent = 0;
        for (id, d) in self.conns.iter_mut() {
            if Some(*id) == except || !d.is_connected() {
                continue;
            }
            if d.send_boxed(channel, priority, make()).is_ok() {
                sent += 1;
            }
        }
        sent
    }

    /// Ends one client's connection with `reason`; it goes out with the next poll.
    pub fn kick(&mut self, id: ConnId, reason: DisconnectReason) -> bool {
        match self.connection_mut(id) {
            Some(d) => {
                d.close(reason);
                true
            }
            None => false,
        }
    }
}

/// Why a connection ended, in words for a log line.
pub fn describe_end(reason: EndReason) -> String {
    match reason {
        EndReason::LocalClose => "closed here".to_string(),
        EndReason::PeerClosed(r) => format!("closed by the other side ({r:?})"),
        EndReason::Timeout => "timed out".to_string(),
        EndReason::ConnectTimeout => "could not connect in time".to_string(),
        EndReason::Rejected(r) => format!("refused ({r:?})"),
        EndReason::ProtocolError => "protocol error".to_string(),
    }
}
