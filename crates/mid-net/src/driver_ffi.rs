//! The C ABI of the driver and the server: what a program written in C (or anything that can call C)
//! uses to run `mid-net` without a line of Rust.
//!
//! * **Transport by callbacks.** The caller supplies a [`MidNetTransportVtable`]: five function pointers
//!   and a `user` pointer, shaped like the QUIC backend's own C functions (`mid_net_quic_send_datagram`
//!   and friends take a connection handle where these take `user`), so a backend plugs in with a thin
//!   adapter or none. Nothing here knows about sockets.
//! * **Opaque messages.** A C program has no Rust types to register, so its messages are an id and bytes:
//!   [`MidNetSchema`] registers ids as *raw* (see `RawMessage`), and a message is sent as
//!   `(channel, priority, id, bytes)` and received the same way. The bytes are the program's own
//!   packing; the driver adds a length (one to three bytes), which is what lets it find the end of a message
//!   it cannot read. Typed, bit-packed messages stay on the Rust side until generated code and a
//!   descriptor interpreter exist (see `docs/mid-net-foundation.md`).
//! * **Handles are opaque and single-threaded.** Create with `*_new` / `*_client`, destroy with `*_free`; call
//!   the functions of one handle from one thread at a time.
//!
//! ## Conventions (the same as `ffi.rs`)
//! Every function returns a status (`0` ok, negative an error, `1` "nothing waiting" where that is a
//! normal answer), never unwinds across the boundary (a panic becomes `MidNetDriverStatus::InternalPanic`),
//! checks its pointers for null, and is `unsafe` on the Rust side with its contract in `# Safety`.
//! Time is a millisecond clock of the caller's choosing that never goes backwards, passed to `poll`.

use std::ffi::CStr;
use std::os::raw::{c_char, c_void};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr;
use std::slice;
use std::sync::Arc;

use mid_net_transport::Transport;
use mid_net_wire::{ChannelError, ChannelKind, ChannelTable, MessageRegistry, RawMessage, RegisterError};

use crate::driver::{
    ConnId, Driver, DriverConfig, DriverEvent, Fault, Schema, SendError, Server, AcceptError,
};
use crate::session::{
    fnv1a64, DisconnectReason, EndReason, SchemaHash, SessionConfig, SessionEvent, SessionState,
    SESSION_PROTOCOL_VERSION,
};

/// Status codes. Success is `0`; `1` is the normal answer "nothing waiting".
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MidNetDriverStatus {
    Ok = 0,
    /// Nothing is waiting (`next_message`, `next_event`).
    None = 1,
    NullPointer = -1,
    /// The buffer is smaller than the message; the message is still queued and `*len` says how big it is.
    BufferTooSmall = -2,
    InternalPanic = -6,
    BadArgument = -7,
    BadConfig = -20,
    BadSchema = -21,
    BadTransport = -22,
    NotConnected = -30,
    Ended = -31,
    BadChannel = -32,
    Unregistered = -33,
    Reserved = -34,
    QueueFull = -35,
    TooLarge = -36,
    Encode = -37,
    NoSuchConnection = -38,
    ServerFull = -39,
    /// A channel or message id is already defined.
    Duplicate = -40,
    /// The schema is sealed: nothing can be added.
    SchemaSealed = -41,
    /// The schema must be sealed before a driver can use it.
    SchemaNotSealed = -42,
}

use MidNetDriverStatus as St;

fn guard(f: impl FnOnce() -> i32) -> i32 {
    catch_unwind(AssertUnwindSafe(f)).unwrap_or(St::InternalPanic as i32)
}

fn send_status(e: SendError) -> i32 {
    (match e {
        SendError::NotConnected => St::NotConnected,
        SendError::Ended => St::Ended,
        SendError::BadChannel => St::BadChannel,
        SendError::Unregistered => St::Unregistered,
        SendError::Reserved => St::Reserved,
        SendError::QueueFull => St::QueueFull,
        SendError::TooLarge => St::TooLarge,
        SendError::Encode(_) => St::Encode,
        SendError::NoSuchConnection => St::NoSuchConnection,
    }) as i32
}

// ---------------------------------------------------------------------------------------------------------
// The transport, by callbacks
// ---------------------------------------------------------------------------------------------------------

/// A transport made of C functions. All five must be set.
///
/// * `send_datagram(user, data, len)`: `0` ok, negative error.
/// * `recv_datagram(user, buf, cap)`: the datagram's length, `0` when none is waiting, negative error.
/// * `send_reliable(user, stream_id, data, len)`: `0` ok, negative error. The bytes are a piece of an ordered
///   stream; they may be as large as the driver's reliable budget.
/// * `recv_reliable(user, buf, cap, stream_id)`: the number of bytes (any piece of the stream: the driver does
///   not need messages to arrive whole), `0` when none, negative error; the stream id goes to `*stream_id`.
/// * `is_connected(user)`: non-zero once the transport can carry data.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct MidNetTransportVtable {
    pub user: *mut c_void,
    pub send_datagram: Option<unsafe extern "C" fn(*mut c_void, *const u8, usize) -> i32>,
    pub recv_datagram: Option<unsafe extern "C" fn(*mut c_void, *mut u8, usize) -> i32>,
    pub send_reliable: Option<unsafe extern "C" fn(*mut c_void, u32, *const u8, usize) -> i32>,
    pub recv_reliable: Option<unsafe extern "C" fn(*mut c_void, *mut u8, usize, *mut u32) -> i32>,
    pub is_connected: Option<unsafe extern "C" fn(*mut c_void) -> i32>,
}

const C_RECV_CAP: usize = 65_536;

struct CTransport {
    vt: MidNetTransportVtable,
    buf: Vec<u8>,
}

impl CTransport {
    fn new(vt: MidNetTransportVtable) -> Option<CTransport> {
        if vt.send_datagram.is_none()
            || vt.recv_datagram.is_none()
            || vt.send_reliable.is_none()
            || vt.recv_reliable.is_none()
            || vt.is_connected.is_none()
        {
            return None;
        }
        Some(CTransport {
            vt,
            buf: vec![0; C_RECV_CAP],
        })
    }
}

impl Transport for CTransport {
    type Error = i32;

    fn send_datagram(&mut self, bytes: &[u8]) -> Result<(), i32> {
        let f = self.vt.send_datagram.ok_or(St::BadTransport as i32)?;
        // SAFETY: the caller promised the function pointer and `user` are valid (see the vtable contract).
        let r = unsafe { f(self.vt.user, bytes.as_ptr(), bytes.len()) };
        if r < 0 {
            Err(r)
        } else {
            Ok(())
        }
    }

    fn poll_datagram(&mut self) -> Result<Option<Vec<u8>>, i32> {
        let f = self.vt.recv_datagram.ok_or(St::BadTransport as i32)?;
        // SAFETY: `buf` is valid for `C_RECV_CAP` bytes.
        let n = unsafe { f(self.vt.user, self.buf.as_mut_ptr(), C_RECV_CAP) };
        match n {
            n if n < 0 => Err(n),
            0 => Ok(None),
            n if n as usize > C_RECV_CAP => Err(St::BadTransport as i32),
            n => Ok(Some(self.buf[..n as usize].to_vec())),
        }
    }

    fn send_reliable(&mut self, stream_id: u32, bytes: &[u8]) -> Result<(), i32> {
        let f = self.vt.send_reliable.ok_or(St::BadTransport as i32)?;
        // SAFETY: as for `send_datagram`.
        let r = unsafe { f(self.vt.user, stream_id, bytes.as_ptr(), bytes.len()) };
        if r < 0 {
            Err(r)
        } else {
            Ok(())
        }
    }

    fn poll_reliable(&mut self) -> Result<Option<(u32, Vec<u8>)>, i32> {
        let f = self.vt.recv_reliable.ok_or(St::BadTransport as i32)?;
        let mut stream_id: u32 = 0;
        // SAFETY: `buf` is valid for `C_RECV_CAP` bytes and `stream_id` for one write.
        let n = unsafe { f(self.vt.user, self.buf.as_mut_ptr(), C_RECV_CAP, &mut stream_id) };
        match n {
            n if n < 0 => Err(n),
            0 => Ok(None),
            n if n as usize > C_RECV_CAP => Err(St::BadTransport as i32),
            n => Ok(Some((stream_id, self.buf[..n as usize].to_vec()))),
        }
    }

    fn is_connected(&self) -> bool {
        match self.vt.is_connected {
            // SAFETY: as for `send_datagram`.
            Some(f) => unsafe { f(self.vt.user) != 0 },
            None => false,
        }
    }
}

// ---------------------------------------------------------------------------------------------------------
// The schema
// ---------------------------------------------------------------------------------------------------------

/// What both ends must agree on, built up from C: channels and raw message ids, then sealed.
pub struct MidNetSchema {
    registry: Option<MessageRegistry>,
    channels: Option<ChannelTable>,
    sealed: Option<Arc<Schema>>,
}

/// Channel kinds for [`mid_net_schema_channel`].
pub const MID_NET_CHANNEL_UNRELIABLE_SEQUENCED: i32 = 0;
pub const MID_NET_CHANNEL_UNRELIABLE_UNORDERED: i32 = 1;
pub const MID_NET_CHANNEL_RELIABLE_ORDERED: i32 = 2;

/// Starts a schema. Message ids are `id_bits` wide on the wire (1 to 16: 8 allows 256 message types) and
/// channel ids `channel_bits` (1 to 8). Returns null for a bad width.
#[no_mangle]
pub extern "C" fn mid_net_schema_new(id_bits: u32, channel_bits: u32) -> *mut MidNetSchema {
    catch_unwind(|| {
        if !(1..=16).contains(&id_bits) {
            return ptr::null_mut();
        }
        let Ok(channels) = ChannelTable::new(channel_bits) else {
            return ptr::null_mut();
        };
        Box::into_raw(Box::new(MidNetSchema {
            registry: Some(MessageRegistry::new(id_bits)),
            channels: Some(channels),
            sealed: None,
        }))
    })
    .unwrap_or(ptr::null_mut())
}

/// Defines channel `id` (kind: one of the `MID_NET_CHANNEL_*` values).
///
/// # Safety
/// `schema` is a live handle from [`mid_net_schema_new`].
#[no_mangle]
pub unsafe extern "C" fn mid_net_schema_channel(schema: *mut MidNetSchema, id: u8, kind: i32) -> i32 {
    guard(|| {
        let Some(s) = (unsafe { schema.as_mut() }) else {
            return St::NullPointer as i32;
        };
        let kind = match kind {
            MID_NET_CHANNEL_UNRELIABLE_SEQUENCED => ChannelKind::UnreliableSequenced,
            MID_NET_CHANNEL_UNRELIABLE_UNORDERED => ChannelKind::UnreliableUnordered,
            MID_NET_CHANNEL_RELIABLE_ORDERED => ChannelKind::ReliableOrdered,
            _ => return St::BadArgument as i32,
        };
        let Some(channels) = s.channels.as_mut() else {
            return St::SchemaSealed as i32;
        };
        match channels.define(id, kind) {
            Ok(()) => St::Ok as i32,
            Err(ChannelError::Duplicate) => St::Duplicate as i32,
            Err(_) => St::BadArgument as i32,
        }
    })
}

/// Registers message id `id` as opaque bytes. Id 0 belongs to the session: do not use it.
///
/// # Safety
/// `schema` is a live handle from [`mid_net_schema_new`].
#[no_mangle]
pub unsafe extern "C" fn mid_net_schema_message(schema: *mut MidNetSchema, id: u16) -> i32 {
    guard(|| {
        let Some(s) = (unsafe { schema.as_mut() }) else {
            return St::NullPointer as i32;
        };
        let Some(reg) = s.registry.as_mut() else {
            return St::SchemaSealed as i32;
        };
        match reg.register_raw(id) {
            Ok(()) => St::Ok as i32,
            Err(RegisterError::Duplicate) => St::Duplicate as i32,
            Err(_) => St::BadArgument as i32,
        }
    })
}

/// Seals the schema: after this drivers can use it and nothing can be added. `control_reliable` and
/// `control_datagram` are the channels the session uses for its own messages (a reliable one and a
/// datagram one, defined above). On an error the schema is spent: free it and start again.
///
/// # Safety
/// `schema` is a live handle from [`mid_net_schema_new`].
#[no_mangle]
pub unsafe extern "C" fn mid_net_schema_seal(
    schema: *mut MidNetSchema,
    control_reliable: u8,
    control_datagram: u8,
) -> i32 {
    guard(|| {
        let Some(s) = (unsafe { schema.as_mut() }) else {
            return St::NullPointer as i32;
        };
        let (Some(reg), Some(ch)) = (s.registry.take(), s.channels.take()) else {
            return St::SchemaSealed as i32;
        };
        match Schema::new(reg, ch, control_reliable, control_datagram) {
            Ok(sch) => {
                s.sealed = Some(Arc::new(sch));
                St::Ok as i32
            }
            Err(_) => St::BadSchema as i32,
        }
    })
}

/// Frees a schema. Drivers made from it keep what they need: it may be freed while they live.
///
/// # Safety
/// `schema` is null or a live handle from [`mid_net_schema_new`] not freed before.
#[no_mangle]
pub unsafe extern "C" fn mid_net_schema_free(schema: *mut MidNetSchema) {
    if !schema.is_null() {
        let _ = catch_unwind(AssertUnwindSafe(|| drop(unsafe { Box::from_raw(schema) })));
    }
}

/// A 64-bit FNV-1a hash of `len` bytes: the usual way to make the schema hash of the handshake from the
/// text of the schema.
///
/// # Safety
/// `data` is valid for `len` bytes (or null with `len == 0`).
#[no_mangle]
pub unsafe extern "C" fn mid_net_fnv1a64(data: *const u8, len: usize) -> u64 {
    catch_unwind(|| {
        if data.is_null() {
            return fnv1a64(&[]).0;
        }
        fnv1a64(unsafe { slice::from_raw_parts(data, len) }).0
    })
    .unwrap_or(0)
}

// ---------------------------------------------------------------------------------------------------------
// Configuration, events, information
// ---------------------------------------------------------------------------------------------------------

/// How a driver behaves; fill it with [`mid_net_driver_config_default`] and change what you need.
/// Field meanings are those of `DriverConfig` and `SessionConfig`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MidNetDriverConfig {
    pub schema_hash: u64,
    pub ping_interval_ms: u64,
    pub no_signal_ms: u64,
    pub timeout_ms: u64,
    pub connect_timeout_ms: u64,
    pub hello_retry_ms: u64,
    pub datagram_ttl_ms: u64,
    pub mtu: u32,
    pub datagram_budget: u32,
    pub reliable_budget: u32,
    pub max_queued_datagrams: u32,
    pub max_stream_pending: u32,
    pub max_frame: u32,
    pub max_inbox: u32,
    pub max_receive_per_poll: u32,
    pub tick_rate_hz: u16,
    pub reserved: u16,
}

impl MidNetDriverConfig {
    fn to_rust(self) -> DriverConfig {
        DriverConfig {
            session: SessionConfig {
                protocol: SESSION_PROTOCOL_VERSION,
                schema: SchemaHash(self.schema_hash),
                ping_interval_ms: self.ping_interval_ms,
                no_signal_ms: self.no_signal_ms,
                timeout_ms: self.timeout_ms,
                connect_timeout_ms: self.connect_timeout_ms,
                hello_retry_ms: self.hello_retry_ms,
                tick_rate_hz: self.tick_rate_hz,
            },
            mtu: self.mtu as usize,
            datagram_budget: self.datagram_budget as usize,
            reliable_budget: self.reliable_budget as usize,
            max_queued_datagrams: self.max_queued_datagrams as usize,
            datagram_ttl_ms: self.datagram_ttl_ms,
            max_stream_pending: self.max_stream_pending as usize,
            max_frame: self.max_frame as usize,
            max_inbox: self.max_inbox as usize,
            max_receive_per_poll: self.max_receive_per_poll as usize,
        }
    }
}

/// Writes the default configuration (the schema hash is 0: set it).
///
/// # Safety
/// `out` is valid for one write.
#[no_mangle]
pub unsafe extern "C" fn mid_net_driver_config_default(out: *mut MidNetDriverConfig) -> i32 {
    guard(|| {
        if out.is_null() {
            return St::NullPointer as i32;
        }
        let d = DriverConfig::default();
        unsafe {
            *out = MidNetDriverConfig {
                schema_hash: d.session.schema.0,
                ping_interval_ms: d.session.ping_interval_ms,
                no_signal_ms: d.session.no_signal_ms,
                timeout_ms: d.session.timeout_ms,
                connect_timeout_ms: d.session.connect_timeout_ms,
                hello_retry_ms: d.session.hello_retry_ms,
                datagram_ttl_ms: d.datagram_ttl_ms,
                mtu: d.mtu as u32,
                datagram_budget: d.datagram_budget as u32,
                reliable_budget: d.reliable_budget as u32,
                max_queued_datagrams: d.max_queued_datagrams as u32,
                max_stream_pending: d.max_stream_pending as u32,
                max_frame: d.max_frame as u32,
                max_inbox: d.max_inbox as u32,
                max_receive_per_poll: d.max_receive_per_poll as u32,
                tick_rate_hz: d.session.tick_rate_hz,
                reserved: 0,
            };
        }
        St::Ok as i32
    })
}

/// Event kinds of [`MidNetEvent::kind`].
pub const MID_NET_EVENT_CONNECTED: i32 = 1; // a = session id, b = tick rate
pub const MID_NET_EVENT_NO_SIGNAL: i32 = 2; // a = silent for this many ms
pub const MID_NET_EVENT_RECOVERED: i32 = 3; // a = the gap in ms
pub const MID_NET_EVENT_ENDED: i32 = 4; // a = MID_NET_END_*, b = the peer's reason code where there is one
pub const MID_NET_EVENT_FAULT: i32 = 5; // a = MID_NET_FAULT_*, b = the stream or channel where there is one
pub const MID_NET_EVENT_DROPPED: i32 = 6; // a = channel << 16 | id, b = 1 oversize, 2 encode, 3 unusable
pub const MID_NET_EVENT_SEND_REJECTED: i32 = 7; // a = channel << 16 | id, b = status

/// `MidNetEvent::a` of an `Ended` event.
pub const MID_NET_END_LOCAL_CLOSE: u64 = 1;
pub const MID_NET_END_PEER_CLOSED: u64 = 2;
pub const MID_NET_END_TIMEOUT: u64 = 3;
pub const MID_NET_END_CONNECT_TIMEOUT: u64 = 4;
pub const MID_NET_END_REJECTED: u64 = 5;
pub const MID_NET_END_PROTOCOL_ERROR: u64 = 6;

/// `MidNetEvent::a` of a `Fault` event.
pub const MID_NET_FAULT_TRANSPORT: u64 = 1;
pub const MID_NET_FAULT_TRANSPORT_CLOSED: u64 = 2;
pub const MID_NET_FAULT_UNKNOWN_STREAM: u64 = 3;
pub const MID_NET_FAULT_STREAM_OVERFLOW: u64 = 4;
pub const MID_NET_FAULT_MALFORMED_STREAM: u64 = 5;
pub const MID_NET_FAULT_SEND_QUEUE_FULL: u64 = 6;

/// One event, flattened. `conn` is the connection for a server (0 for a single driver).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MidNetEvent {
    pub conn: u32,
    pub kind: i32,
    pub a: u64,
    pub b: u64,
}

fn end_codes(r: EndReason) -> (u64, u64) {
    match r {
        EndReason::LocalClose => (MID_NET_END_LOCAL_CLOSE, 0),
        EndReason::PeerClosed(d) => (MID_NET_END_PEER_CLOSED, d.to_wire() as u64),
        EndReason::Timeout => (MID_NET_END_TIMEOUT, 0),
        EndReason::ConnectTimeout => (MID_NET_END_CONNECT_TIMEOUT, 0),
        EndReason::Rejected(d) => (MID_NET_END_REJECTED, d.to_wire() as u64),
        EndReason::ProtocolError => (MID_NET_END_PROTOCOL_ERROR, 0),
    }
}

fn event_to_c(conn: u32, e: &DriverEvent) -> MidNetEvent {
    let (kind, a, b) = match e {
        DriverEvent::Session(SessionEvent::Connected {
            session_id,
            tick_rate_hz,
        }) => (MID_NET_EVENT_CONNECTED, *session_id as u64, *tick_rate_hz as u64),
        DriverEvent::Session(SessionEvent::NoSignal { silent_ms }) => {
            (MID_NET_EVENT_NO_SIGNAL, *silent_ms, 0)
        }
        DriverEvent::Session(SessionEvent::Recovered { gap_ms }) => {
            (MID_NET_EVENT_RECOVERED, *gap_ms, 0)
        }
        DriverEvent::Session(SessionEvent::Ended(r)) => {
            let (a, b) = end_codes(*r);
            (MID_NET_EVENT_ENDED, a, b)
        }
        DriverEvent::Fault(f) => {
            let (a, b) = match f {
                Fault::Transport(_) => (MID_NET_FAULT_TRANSPORT, 0),
                Fault::TransportClosed => (MID_NET_FAULT_TRANSPORT_CLOSED, 0),
                Fault::UnknownStream(s) => (MID_NET_FAULT_UNKNOWN_STREAM, *s as u64),
                Fault::StreamOverflow { channel } => (MID_NET_FAULT_STREAM_OVERFLOW, *channel as u64),
                Fault::MalformedStream { channel, .. } => {
                    (MID_NET_FAULT_MALFORMED_STREAM, *channel as u64)
                }
                Fault::SendQueueFull { channel } => (MID_NET_FAULT_SEND_QUEUE_FULL, *channel as u64),
            };
            (MID_NET_EVENT_FAULT, a, b)
        }
        DriverEvent::Dropped {
            channel,
            id,
            reason,
        } => {
            let why = match reason {
                crate::driver::DropReason::Oversize => 1,
                crate::driver::DropReason::Encode(_) => 2,
                crate::driver::DropReason::Unusable => 3,
            };
            (MID_NET_EVENT_DROPPED, ((*channel as u64) << 16) | *id as u64, why)
        }
        DriverEvent::SendRejected { channel, id, error } => (
            MID_NET_EVENT_SEND_REJECTED,
            ((*channel as u64) << 16) | *id as u64,
            (-send_status(*error)) as u64,
        ),
    };
    MidNetEvent { conn, kind, a, b }
}

/// `MidNetDriverInfo::state`.
pub const MID_NET_STATE_CONNECTING: i32 = 0;
pub const MID_NET_STATE_CONNECTED: i32 = 1;
pub const MID_NET_STATE_NO_SIGNAL: i32 = 2;
pub const MID_NET_STATE_ENDED: i32 = 3;

/// A snapshot of one connection. `end_kind` / `end_peer_reason` are those of the `Ended` event (0 while
/// the connection lives); `server_time_ms` is -1 before the first keepalive answer.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct MidNetDriverInfo {
    pub rtt_ms: f64,
    pub server_time_ms: i64,
    pub silent_ms: u64,
    pub session_id: u32,
    pub state: i32,
    pub end_kind: i32,
    pub end_peer_reason: i32,
    pub tick_rate_hz: u16,
    pub reserved: u16,
    pub messages_waiting: u32,
    pub queued_datagrams: u32,
}

fn info_of(d: &Driver<CTransport>, now_ms: u64) -> MidNetDriverInfo {
    let (state, end_kind, end_peer) = match d.state() {
        SessionState::Connecting => (MID_NET_STATE_CONNECTING, 0, 0),
        SessionState::Connected => (MID_NET_STATE_CONNECTED, 0, 0),
        SessionState::NoSignal { .. } => (MID_NET_STATE_NO_SIGNAL, 0, 0),
        SessionState::Ended(r) => {
            let (k, p) = end_codes(r);
            (MID_NET_STATE_ENDED, k as i32, p as i32)
        }
    };
    MidNetDriverInfo {
        rtt_ms: d.rtt_ms(),
        server_time_ms: d.server_time_ms(now_ms).map(|t| t as i64).unwrap_or(-1),
        silent_ms: d.session().silence_ms(now_ms),
        session_id: d.session().session_id(),
        state,
        end_kind,
        end_peer_reason: end_peer,
        tick_rate_hz: d.session().tick_rate_hz(),
        reserved: 0,
        messages_waiting: d.messages_waiting() as u32,
        queued_datagrams: d.queued_datagrams() as u32,
    }
}

/// The counters of a driver, in the order of `DriverStats`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MidNetDriverStats {
    pub datagrams_sent: u64,
    pub datagram_bytes_sent: u64,
    pub datagrams_received: u64,
    pub bad_datagrams: u64,
    pub duplicate_datagrams: u64,
    pub stale_messages: u64,
    pub messages_sent: u64,
    pub messages_received: u64,
    pub reliable_bytes_sent: u64,
    pub reliable_bytes_received: u64,
    pub dropped_expired: u64,
    pub dropped_queue_full: u64,
    pub dropped_unsendable: u64,
    pub dropped_early: u64,
}

fn stats_of(d: &Driver<CTransport>) -> MidNetDriverStats {
    let s = d.stats();
    MidNetDriverStats {
        datagrams_sent: s.datagrams_sent,
        datagram_bytes_sent: s.datagram_bytes_sent,
        datagrams_received: s.datagrams_received,
        bad_datagrams: s.bad_datagrams,
        duplicate_datagrams: s.duplicate_datagrams,
        stale_messages: s.stale_messages,
        messages_sent: s.messages_sent,
        messages_received: s.messages_received,
        reliable_bytes_sent: s.reliable_bytes_sent,
        reliable_bytes_received: s.reliable_bytes_received,
        dropped_expired: s.dropped_expired,
        dropped_queue_full: s.dropped_queue_full,
        dropped_unsendable: s.dropped_unsendable,
        dropped_early: s.dropped_early,
    }
}

// ---------------------------------------------------------------------------------------------------------
// One connection
// ---------------------------------------------------------------------------------------------------------

/// One end of one connection (a client, or a server's end of one client when there is no [`MidNetServer`]).
pub struct MidNetDriver {
    inner: Driver<CTransport>,
}

fn sealed<'a>(schema: *const MidNetSchema) -> Result<&'a Arc<Schema>, i32> {
    match unsafe { schema.as_ref() } {
        None => Err(St::NullPointer as i32),
        Some(s) => s.sealed.as_ref().ok_or(St::SchemaNotSealed as i32),
    }
}

unsafe fn name_of(name: *const c_char) -> String {
    if name.is_null() {
        String::new()
    } else {
        unsafe { CStr::from_ptr(name) }.to_string_lossy().into_owned()
    }
}

/// Makes the client end of a connection; it says hello on the first poll. `*out` is set to the new handle
/// (null on error).
///
/// # Safety
/// `schema` is a live sealed schema; `cfg` and `transport` point to valid structs (the vtable's functions and
/// `user` must stay valid for the life of the driver); `name` is null or a NUL-terminated string; `out` is valid
/// for one write.
#[no_mangle]
pub unsafe extern "C" fn mid_net_driver_client(
    schema: *const MidNetSchema,
    cfg: *const MidNetDriverConfig,
    transport: *const MidNetTransportVtable,
    name: *const c_char,
    nonce: u32,
    now_ms: u64,
    out: *mut *mut MidNetDriver,
) -> i32 {
    guard(|| {
        if out.is_null() || cfg.is_null() || transport.is_null() {
            return St::NullPointer as i32;
        }
        unsafe { *out = ptr::null_mut() };
        let sch = match sealed(schema) {
            Ok(s) => Arc::clone(s),
            Err(e) => return e,
        };
        let Some(t) = CTransport::new(unsafe { *transport }) else {
            return St::BadTransport as i32;
        };
        let cfg = unsafe { &*cfg }.to_rust();
        match Driver::client(sch, cfg, t, &unsafe { name_of(name) }, nonce, now_ms) {
            Ok(d) => {
                unsafe { *out = Box::into_raw(Box::new(MidNetDriver { inner: d })) };
                St::Ok as i32
            }
            Err(_) => St::BadConfig as i32,
        }
    })
}

/// Makes the server's end of one already-accepted connection, with `session_id` to tell the client. For
/// more than a client or two, use [`MidNetServer`].
///
/// # Safety
/// As [`mid_net_driver_client`].
#[no_mangle]
pub unsafe extern "C" fn mid_net_driver_server_end(
    schema: *const MidNetSchema,
    cfg: *const MidNetDriverConfig,
    transport: *const MidNetTransportVtable,
    session_id: u32,
    now_ms: u64,
    out: *mut *mut MidNetDriver,
) -> i32 {
    guard(|| {
        if out.is_null() || cfg.is_null() || transport.is_null() {
            return St::NullPointer as i32;
        }
        unsafe { *out = ptr::null_mut() };
        let sch = match sealed(schema) {
            Ok(s) => Arc::clone(s),
            Err(e) => return e,
        };
        let Some(t) = CTransport::new(unsafe { *transport }) else {
            return St::BadTransport as i32;
        };
        let cfg = unsafe { &*cfg }.to_rust();
        match Driver::server_end(sch, cfg, t, session_id, now_ms) {
            Ok(d) => {
                unsafe { *out = Box::into_raw(Box::new(MidNetDriver { inner: d })) };
                St::Ok as i32
            }
            Err(_) => St::BadConfig as i32,
        }
    })
}

/// Frees a driver (no goodbye is sent: call [`mid_net_driver_close`] and poll first for that).
///
/// # Safety
/// `driver` is null or a live handle not freed before.
#[no_mangle]
pub unsafe extern "C" fn mid_net_driver_free(driver: *mut MidNetDriver) {
    if !driver.is_null() {
        let _ = catch_unwind(AssertUnwindSafe(|| drop(unsafe { Box::from_raw(driver) })));
    }
}

macro_rules! with_driver {
    ($driver:expr, $d:ident, $body:block) => {{
        let Some(h) = (unsafe { $driver.as_mut() }) else {
            return St::NullPointer as i32;
        };
        let $d = &mut h.inner;
        $body
    }};
}

/// Reads the transport, advances the session and sends what is queued. Call every frame.
///
/// # Safety
/// `driver` is a live handle.
#[no_mangle]
pub unsafe extern "C" fn mid_net_driver_poll(driver: *mut MidNetDriver, now_ms: u64) -> i32 {
    guard(|| {
        with_driver!(driver, d, {
            d.poll(now_ms);
            St::Ok as i32
        })
    })
}

/// Ends the connection on purpose (`reason` is a `DisconnectReason` code: 0 normal, 6 kicked, ...); the
/// goodbye goes out with the next poll.
///
/// # Safety
/// `driver` is a live handle.
#[no_mangle]
pub unsafe extern "C" fn mid_net_driver_close(driver: *mut MidNetDriver, reason: u8) -> i32 {
    guard(|| {
        with_driver!(driver, d, {
            d.close(DisconnectReason::from_wire(reason));
            St::Ok as i32
        })
    })
}

/// Sends a message: `len` bytes under message `id` on `channel`. On a datagram channel it is queued and goes
/// in `priority` order (0 lowest, 254 highest; 255 is the session's); a reliable channel ignores the priority.
///
/// # Safety
/// `driver` is a live handle; `data` is valid for `len` bytes (or null with `len == 0`).
#[no_mangle]
pub unsafe extern "C" fn mid_net_driver_send(
    driver: *mut MidNetDriver,
    channel: u8,
    priority: u8,
    id: u16,
    data: *const u8,
    len: usize,
) -> i32 {
    guard(|| {
        if data.is_null() && len != 0 {
            return St::NullPointer as i32;
        }
        let bytes = if len == 0 {
            Vec::new()
        } else {
            unsafe { slice::from_raw_parts(data, len) }.to_vec()
        };
        with_driver!(driver, d, {
            match d.send_raw(channel, priority, id, bytes) {
                Ok(()) => St::Ok as i32,
                Err(e) => send_status(e),
            }
        })
    })
}

/// Takes the next received message: its channel and id go to `*channel` and `*id`, its bytes to `buf`
/// (`cap` bytes), their number to `*len`. Returns `0`, or `1` if nothing is waiting, or `-2`
/// (`BufferTooSmall`) if `cap` is too small, in which case the message stays queued and `*len` says how
/// large a buffer it needs.
///
/// # Safety
/// `driver` is a live handle; `channel`, `id` and `len` are valid for a write; `buf` is valid for `cap`
/// bytes (or null with `cap == 0`, to ask only how large the next message is).
#[no_mangle]
pub unsafe extern "C" fn mid_net_driver_next_message(
    driver: *mut MidNetDriver,
    channel: *mut u8,
    id: *mut u16,
    buf: *mut u8,
    cap: usize,
    len: *mut usize,
) -> i32 {
    guard(|| {
        if channel.is_null() || id.is_null() || len.is_null() || (buf.is_null() && cap != 0) {
            return St::NullPointer as i32;
        }
        with_driver!(driver, d, {
            loop {
                let Some(m) = d.peek_message() else {
                    return St::None as i32;
                };
                let Some(raw) = m.msg.as_raw() else {
                    d.next_message(); // not opaque bytes: nothing a C caller can read
                    continue;
                };
                let n = raw.data.len();
                unsafe { *len = n };
                if n > cap {
                    return St::BufferTooSmall as i32;
                }
                if n > 0 {
                    unsafe { ptr::copy_nonoverlapping(raw.data.as_ptr(), buf, n) };
                }
                unsafe {
                    *channel = m.channel;
                    *id = raw.id;
                }
                d.next_message();
                return St::Ok as i32;
            }
        })
    })
}

/// Takes the next event into `*out`. Returns `0`, or `1` if there is none.
///
/// # Safety
/// `driver` is a live handle; `out` is valid for one write.
#[no_mangle]
pub unsafe extern "C" fn mid_net_driver_next_event(driver: *mut MidNetDriver, out: *mut MidNetEvent) -> i32 {
    guard(|| {
        if out.is_null() {
            return St::NullPointer as i32;
        }
        with_driver!(driver, d, {
            match d.next_event() {
                Some(e) => {
                    unsafe { *out = event_to_c(0, &e) };
                    St::Ok as i32
                }
                None => St::None as i32,
            }
        })
    })
}

/// Fills `*out` with the state of the connection.
///
/// # Safety
/// `driver` is a live handle; `out` is valid for one write.
#[no_mangle]
pub unsafe extern "C" fn mid_net_driver_info(
    driver: *mut MidNetDriver,
    now_ms: u64,
    out: *mut MidNetDriverInfo,
) -> i32 {
    guard(|| {
        if out.is_null() {
            return St::NullPointer as i32;
        }
        with_driver!(driver, d, {
            unsafe { *out = info_of(d, now_ms) };
            St::Ok as i32
        })
    })
}

/// Fills `*out` with the counters.
///
/// # Safety
/// `driver` is a live handle; `out` is valid for one write.
#[no_mangle]
pub unsafe extern "C" fn mid_net_driver_stats(driver: *mut MidNetDriver, out: *mut MidNetDriverStats) -> i32 {
    guard(|| {
        if out.is_null() {
            return St::NullPointer as i32;
        }
        with_driver!(driver, d, {
            unsafe { *out = stats_of(d) };
            St::Ok as i32
        })
    })
}

// ---------------------------------------------------------------------------------------------------------
// Many connections
// ---------------------------------------------------------------------------------------------------------

/// A server: one connection per client, one place to read from.
pub struct MidNetServer {
    inner: Server<CTransport>,
}

/// Makes a server that takes at most `max_connections` clients.
///
/// # Safety
/// `schema` is a live sealed schema; `cfg` points to a valid struct; `out` is valid for one write.
#[no_mangle]
pub unsafe extern "C" fn mid_net_server_new(
    schema: *const MidNetSchema,
    cfg: *const MidNetDriverConfig,
    max_connections: u32,
    out: *mut *mut MidNetServer,
) -> i32 {
    guard(|| {
        if out.is_null() || cfg.is_null() {
            return St::NullPointer as i32;
        }
        unsafe { *out = ptr::null_mut() };
        let sch = match sealed(schema) {
            Ok(s) => Arc::clone(s),
            Err(e) => return e,
        };
        let cfg = unsafe { &*cfg }.to_rust();
        match Server::new(sch, cfg, max_connections as usize) {
            Ok(s) => {
                unsafe { *out = Box::into_raw(Box::new(MidNetServer { inner: s })) };
                St::Ok as i32
            }
            Err(_) => St::BadConfig as i32,
        }
    })
}

/// Frees a server and every connection in it.
///
/// # Safety
/// `server` is null or a live handle not freed before.
#[no_mangle]
pub unsafe extern "C" fn mid_net_server_free(server: *mut MidNetServer) {
    if !server.is_null() {
        let _ = catch_unwind(AssertUnwindSafe(|| drop(unsafe { Box::from_raw(server) })));
    }
}

macro_rules! with_server {
    ($server:expr, $s:ident, $body:block) => {{
        let Some(h) = (unsafe { $server.as_mut() }) else {
            return St::NullPointer as i32;
        };
        let $s = &mut h.inner;
        $body
    }};
}

/// Takes over a newly accepted connection; its id goes to `*conn`. When the server is full the client is told
/// (`Rejected`/`Full` on its side) and `-39` is returned.
///
/// # Safety
/// `server` is a live handle; `transport` points to a valid vtable that stays valid for the life of the
/// connection; `conn` is valid for one write.
#[no_mangle]
pub unsafe extern "C" fn mid_net_server_accept(
    server: *mut MidNetServer,
    transport: *const MidNetTransportVtable,
    now_ms: u64,
    conn: *mut u32,
) -> i32 {
    guard(|| {
        if transport.is_null() || conn.is_null() {
            return St::NullPointer as i32;
        }
        let Some(t) = CTransport::new(unsafe { *transport }) else {
            return St::BadTransport as i32;
        };
        with_server!(server, s, {
            match s.accept(t, now_ms) {
                Ok(id) => {
                    unsafe { *conn = id.0 };
                    St::Ok as i32
                }
                Err(AcceptError::Full) => St::ServerFull as i32,
                Err(AcceptError::Config(_)) => St::BadConfig as i32,
            }
        })
    })
}

/// Polls every connection and forgets the ones that ended (their `Ended` event is queued).
///
/// # Safety
/// `server` is a live handle.
#[no_mangle]
pub unsafe extern "C" fn mid_net_server_poll(server: *mut MidNetServer, now_ms: u64) -> i32 {
    guard(|| {
        with_server!(server, s, {
            s.poll(now_ms);
            St::Ok as i32
        })
    })
}

/// Sends to one client. See [`mid_net_driver_send`].
///
/// # Safety
/// `server` is a live handle; `data` is valid for `len` bytes (or null with `len == 0`).
#[no_mangle]
pub unsafe extern "C" fn mid_net_server_send(
    server: *mut MidNetServer,
    conn: u32,
    channel: u8,
    priority: u8,
    id: u16,
    data: *const u8,
    len: usize,
) -> i32 {
    guard(|| {
        if data.is_null() && len != 0 {
            return St::NullPointer as i32;
        }
        let bytes = if len == 0 {
            Vec::new()
        } else {
            unsafe { slice::from_raw_parts(data, len) }.to_vec()
        };
        with_server!(server, s, {
            match s.send(ConnId(conn), channel, priority, Box::new(RawMessage { id, data: bytes })) {
                Ok(()) => St::Ok as i32,
                Err(e) => send_status(e),
            }
        })
    })
}

/// Sends to every connected client except `except_conn` (0: none). The number that accepted it goes to
/// `*sent`.
///
/// # Safety
/// `server` is a live handle; `data` is valid for `len` bytes (or null with `len == 0`); `sent` is valid for a write.
#[no_mangle]
pub unsafe extern "C" fn mid_net_server_broadcast(
    server: *mut MidNetServer,
    except_conn: u32,
    channel: u8,
    priority: u8,
    id: u16,
    data: *const u8,
    len: usize,
    sent: *mut u32,
) -> i32 {
    guard(|| {
        if (data.is_null() && len != 0) || sent.is_null() {
            return St::NullPointer as i32;
        }
        let bytes = if len == 0 {
            Vec::new()
        } else {
            unsafe { slice::from_raw_parts(data, len) }.to_vec()
        };
        with_server!(server, s, {
            let except = if except_conn == 0 { None } else { Some(ConnId(except_conn)) };
            let n = s.broadcast(except, channel, priority, || {
                Box::new(RawMessage {
                    id,
                    data: bytes.clone(),
                })
            });
            unsafe { *sent = n as u32 };
            St::Ok as i32
        })
    })
}

/// Ends one client's connection with `reason` (a `DisconnectReason` code).
///
/// # Safety
/// `server` is a live handle.
#[no_mangle]
pub unsafe extern "C" fn mid_net_server_kick(server: *mut MidNetServer, conn: u32, reason: u8) -> i32 {
    guard(|| {
        with_server!(server, s, {
            if s.kick(ConnId(conn), DisconnectReason::from_wire(reason)) {
                St::Ok as i32
            } else {
                St::NoSuchConnection as i32
            }
        })
    })
}

/// Takes the next message from any client. As [`mid_net_driver_next_message`], plus `*conn`.
///
/// # Safety
/// `server` is a live handle; `conn`, `channel`, `id` and `len` are valid for a write; `buf` is valid for
/// `cap` bytes (or null with `cap == 0`).
#[no_mangle]
pub unsafe extern "C" fn mid_net_server_next_message(
    server: *mut MidNetServer,
    conn: *mut u32,
    channel: *mut u8,
    id: *mut u16,
    buf: *mut u8,
    cap: usize,
    len: *mut usize,
) -> i32 {
    guard(|| {
        if conn.is_null() || channel.is_null() || id.is_null() || len.is_null() || (buf.is_null() && cap != 0) {
            return St::NullPointer as i32;
        }
        with_server!(server, s, {
            loop {
                let Some((cid, m)) = s.peek_message() else {
                    return St::None as i32;
                };
                let Some(raw) = m.msg.as_raw() else {
                    s.next_message();
                    continue;
                };
                let n = raw.data.len();
                unsafe { *len = n };
                if n > cap {
                    return St::BufferTooSmall as i32;
                }
                if n > 0 {
                    unsafe { ptr::copy_nonoverlapping(raw.data.as_ptr(), buf, n) };
                }
                unsafe {
                    *conn = cid.0;
                    *channel = m.channel;
                    *id = raw.id;
                }
                s.next_message();
                return St::Ok as i32;
            }
        })
    })
}

/// Takes the next event of any client (`out->conn` says which). `0`, or `1` if there is none.
///
/// # Safety
/// `server` is a live handle; `out` is valid for one write.
#[no_mangle]
pub unsafe extern "C" fn mid_net_server_next_event(server: *mut MidNetServer, out: *mut MidNetEvent) -> i32 {
    guard(|| {
        if out.is_null() {
            return St::NullPointer as i32;
        }
        with_server!(server, s, {
            match s.next_event() {
                Some((id, e)) => {
                    unsafe { *out = event_to_c(id.0, &e) };
                    St::Ok as i32
                }
                None => St::None as i32,
            }
        })
    })
}

/// How many clients are connected (or connecting).
///
/// # Safety
/// `server` is a live handle.
#[no_mangle]
pub unsafe extern "C" fn mid_net_server_connection_count(server: *mut MidNetServer) -> i32 {
    guard(|| {
        with_server!(server, s, { s.len() as i32 })
    })
}

/// Fills `*out` with the state of one connection.
///
/// # Safety
/// `server` is a live handle; `out` is valid for one write.
#[no_mangle]
pub unsafe extern "C" fn mid_net_server_connection_info(
    server: *mut MidNetServer,
    conn: u32,
    now_ms: u64,
    out: *mut MidNetDriverInfo,
) -> i32 {
    guard(|| {
        if out.is_null() {
            return St::NullPointer as i32;
        }
        with_server!(server, s, {
            match s.connection(ConnId(conn)) {
                Some(d) => {
                    unsafe { *out = info_of(d, now_ms) };
                    St::Ok as i32
                }
                None => St::NoSuchConnection as i32,
            }
        })
    })
}

/// Fills `*out` with the counters of one connection.
///
/// # Safety
/// `server` is a live handle; `out` is valid for one write.
#[no_mangle]
pub unsafe extern "C" fn mid_net_server_connection_stats(
    server: *mut MidNetServer,
    conn: u32,
    out: *mut MidNetDriverStats,
) -> i32 {
    guard(|| {
        if out.is_null() {
            return St::NullPointer as i32;
        }
        with_server!(server, s, {
            match s.connection(ConnId(conn)) {
                Some(d) => {
                    unsafe { *out = stats_of(d) };
                    St::Ok as i32
                }
                None => St::NoSuchConnection as i32,
            }
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::driver::DropReason;

    const HEADER: &str = include_str!("../ffi-smoke-test/mid_net_driver.h");

    /// The value the C header gives `name` in a `#define`.
    fn header_value(name: &str) -> i64 {
        for line in HEADER.lines() {
            let mut it = line.split_whitespace();
            if it.next() == Some("#define") && it.next() == Some(name) {
                let v = it.next().unwrap_or_else(|| panic!("{name} has no value"));
                return v.parse().unwrap_or_else(|_| panic!("{name} = {v}"));
            }
        }
        panic!("{name} is not in the header");
    }

    #[test]
    fn the_header_and_the_rust_enum_agree_on_every_status() {
        let table: [(&str, MidNetDriverStatus); 21] = [
            ("MID_NET_ST_OK", St::Ok),
            ("MID_NET_ST_NONE", St::None),
            ("MID_NET_ST_NULL_POINTER", St::NullPointer),
            ("MID_NET_ST_BUFFER_TOO_SMALL", St::BufferTooSmall),
            ("MID_NET_ST_INTERNAL_PANIC", St::InternalPanic),
            ("MID_NET_ST_BAD_ARGUMENT", St::BadArgument),
            ("MID_NET_ST_BAD_CONFIG", St::BadConfig),
            ("MID_NET_ST_BAD_SCHEMA", St::BadSchema),
            ("MID_NET_ST_BAD_TRANSPORT", St::BadTransport),
            ("MID_NET_ST_NOT_CONNECTED", St::NotConnected),
            ("MID_NET_ST_ENDED", St::Ended),
            ("MID_NET_ST_BAD_CHANNEL", St::BadChannel),
            ("MID_NET_ST_UNREGISTERED", St::Unregistered),
            ("MID_NET_ST_RESERVED", St::Reserved),
            ("MID_NET_ST_QUEUE_FULL", St::QueueFull),
            ("MID_NET_ST_TOO_LARGE", St::TooLarge),
            ("MID_NET_ST_ENCODE", St::Encode),
            ("MID_NET_ST_NO_SUCH_CONNECTION", St::NoSuchConnection),
            ("MID_NET_ST_SERVER_FULL", St::ServerFull),
            ("MID_NET_ST_DUPLICATE", St::Duplicate),
            ("MID_NET_ST_SCHEMA_SEALED", St::SchemaSealed),
        ];
        for (name, st) in table {
            assert_eq!(header_value(name), st as i32 as i64, "{name}");
        }
        assert_eq!(header_value("MID_NET_ST_SCHEMA_NOT_SEALED"), St::SchemaNotSealed as i32 as i64);
    }

    #[test]
    fn the_header_and_the_rust_constants_agree_on_kinds_and_codes() {
        let table: &[(&str, i64)] = &[
            ("MID_NET_CHANNEL_UNRELIABLE_SEQUENCED", MID_NET_CHANNEL_UNRELIABLE_SEQUENCED as i64),
            ("MID_NET_CHANNEL_UNRELIABLE_UNORDERED", MID_NET_CHANNEL_UNRELIABLE_UNORDERED as i64),
            ("MID_NET_CHANNEL_RELIABLE_ORDERED", MID_NET_CHANNEL_RELIABLE_ORDERED as i64),
            ("MID_NET_EVENT_CONNECTED", MID_NET_EVENT_CONNECTED as i64),
            ("MID_NET_EVENT_NO_SIGNAL", MID_NET_EVENT_NO_SIGNAL as i64),
            ("MID_NET_EVENT_RECOVERED", MID_NET_EVENT_RECOVERED as i64),
            ("MID_NET_EVENT_ENDED", MID_NET_EVENT_ENDED as i64),
            ("MID_NET_EVENT_FAULT", MID_NET_EVENT_FAULT as i64),
            ("MID_NET_EVENT_DROPPED", MID_NET_EVENT_DROPPED as i64),
            ("MID_NET_EVENT_SEND_REJECTED", MID_NET_EVENT_SEND_REJECTED as i64),
            ("MID_NET_END_LOCAL_CLOSE", MID_NET_END_LOCAL_CLOSE as i64),
            ("MID_NET_END_PEER_CLOSED", MID_NET_END_PEER_CLOSED as i64),
            ("MID_NET_END_TIMEOUT", MID_NET_END_TIMEOUT as i64),
            ("MID_NET_END_CONNECT_TIMEOUT", MID_NET_END_CONNECT_TIMEOUT as i64),
            ("MID_NET_END_REJECTED", MID_NET_END_REJECTED as i64),
            ("MID_NET_END_PROTOCOL_ERROR", MID_NET_END_PROTOCOL_ERROR as i64),
            ("MID_NET_FAULT_TRANSPORT", MID_NET_FAULT_TRANSPORT as i64),
            ("MID_NET_FAULT_TRANSPORT_CLOSED", MID_NET_FAULT_TRANSPORT_CLOSED as i64),
            ("MID_NET_FAULT_UNKNOWN_STREAM", MID_NET_FAULT_UNKNOWN_STREAM as i64),
            ("MID_NET_FAULT_STREAM_OVERFLOW", MID_NET_FAULT_STREAM_OVERFLOW as i64),
            ("MID_NET_FAULT_MALFORMED_STREAM", MID_NET_FAULT_MALFORMED_STREAM as i64),
            ("MID_NET_FAULT_SEND_QUEUE_FULL", MID_NET_FAULT_SEND_QUEUE_FULL as i64),
            ("MID_NET_STATE_CONNECTING", MID_NET_STATE_CONNECTING as i64),
            ("MID_NET_STATE_CONNECTED", MID_NET_STATE_CONNECTED as i64),
            ("MID_NET_STATE_NO_SIGNAL", MID_NET_STATE_NO_SIGNAL as i64),
            ("MID_NET_STATE_ENDED", MID_NET_STATE_ENDED as i64),
        ];
        for (name, v) in table {
            assert_eq!(header_value(name), *v, "{name}");
        }
        // the codes are all different within a group
        for group in [&table[3..10], &table[10..16], &table[16..22], &table[22..26]] {
            let mut vs: Vec<i64> = group.iter().map(|(_, v)| *v).collect();
            vs.sort();
            vs.dedup();
            assert_eq!(vs.len(), group.len());
        }
    }

    #[test]
    fn the_c_structs_have_the_sizes_the_header_implies() {
        use std::mem::{align_of, size_of};
        assert_eq!(size_of::<MidNetEvent>(), 24);
        assert_eq!(size_of::<MidNetDriverInfo>(), 56); // 52 bytes of fields, padded to a multiple of 8
        assert_eq!(align_of::<MidNetDriverInfo>(), 8);
        assert_eq!(size_of::<MidNetDriverStats>(), 14 * 8);
        assert_eq!(size_of::<MidNetDriverConfig>(), 7 * 8 + 8 * 4 + 2 * 2 + 4); // 4 bytes of tail padding
        assert_eq!(size_of::<MidNetTransportVtable>(), 6 * size_of::<usize>());
    }

    #[test]
    fn the_default_c_configuration_is_the_default_rust_configuration() {
        let mut c: MidNetDriverConfig = unsafe { std::mem::zeroed() };
        assert_eq!(unsafe { mid_net_driver_config_default(&mut c) }, 0);
        assert_eq!(format!("{:?}", c.to_rust()), format!("{:?}", DriverConfig::default()));
        assert_eq!(c.reserved, 0);
        assert_eq!(unsafe { mid_net_driver_config_default(ptr::null_mut()) }, St::NullPointer as i32);
    }

    #[test]
    fn every_field_of_a_c_configuration_lands_in_its_own_field() {
        let c = MidNetDriverConfig {
            schema_hash: 11,
            ping_interval_ms: 12,
            no_signal_ms: 13,
            timeout_ms: 14,
            connect_timeout_ms: 15,
            hello_retry_ms: 16,
            datagram_ttl_ms: 17,
            mtu: 101,
            datagram_budget: 102,
            reliable_budget: 103,
            max_queued_datagrams: 104,
            max_stream_pending: 105,
            max_frame: 106,
            max_inbox: 107,
            max_receive_per_poll: 108,
            tick_rate_hz: 109,
            reserved: 0,
        };
        let r = c.to_rust();
        assert_eq!(r.session.schema.0, 11);
        assert_eq!(r.session.ping_interval_ms, 12);
        assert_eq!(r.session.no_signal_ms, 13);
        assert_eq!(r.session.timeout_ms, 14);
        assert_eq!(r.session.connect_timeout_ms, 15);
        assert_eq!(r.session.hello_retry_ms, 16);
        assert_eq!(r.datagram_ttl_ms, 17);
        assert_eq!(r.mtu, 101);
        assert_eq!(r.datagram_budget, 102);
        assert_eq!(r.reliable_budget, 103);
        assert_eq!(r.max_queued_datagrams, 104);
        assert_eq!(r.max_stream_pending, 105);
        assert_eq!(r.max_frame, 106);
        assert_eq!(r.max_inbox, 107);
        assert_eq!(r.max_receive_per_poll, 108);
        assert_eq!(r.session.tick_rate_hz, 109);
        assert_eq!(r.session.protocol, SESSION_PROTOCOL_VERSION);
    }

    #[test]
    fn every_send_error_has_its_own_negative_status() {
        use SendError::*;
        let all = [
            (NotConnected, St::NotConnected),
            (Ended, St::Ended),
            (BadChannel, St::BadChannel),
            (Unregistered, St::Unregistered),
            (Reserved, St::Reserved),
            (QueueFull, St::QueueFull),
            (TooLarge, St::TooLarge),
            (Encode(mid_net_wire::BitError::NotFinite), St::Encode),
            (NoSuchConnection, St::NoSuchConnection),
        ];
        for (e, st) in all {
            assert_eq!(send_status(e), st as i32, "{e:?}");
            assert!(send_status(e) < 0);
        }
    }

    #[test]
    fn events_are_flattened_with_their_codes() {
        let ev = |e: DriverEvent| event_to_c(7, &e);
        let c = |kind, a, b| MidNetEvent { conn: 7, kind, a, b };
        assert_eq!(
            ev(DriverEvent::Session(SessionEvent::Connected { session_id: 5, tick_rate_hz: 30 })),
            c(MID_NET_EVENT_CONNECTED, 5, 30)
        );
        assert_eq!(ev(DriverEvent::Session(SessionEvent::NoSignal { silent_ms: 1600 })), c(MID_NET_EVENT_NO_SIGNAL, 1600, 0));
        assert_eq!(ev(DriverEvent::Session(SessionEvent::Recovered { gap_ms: 2500 })), c(MID_NET_EVENT_RECOVERED, 2500, 0));
        assert_eq!(ev(DriverEvent::Session(SessionEvent::Ended(EndReason::Timeout))), c(MID_NET_EVENT_ENDED, MID_NET_END_TIMEOUT, 0));
        let kicked = DisconnectReason::Kicked.to_wire() as u64;
        assert_eq!(
            ev(DriverEvent::Session(SessionEvent::Ended(EndReason::PeerClosed(DisconnectReason::Kicked)))),
            c(MID_NET_EVENT_ENDED, MID_NET_END_PEER_CLOSED, kicked)
        );
        assert_eq!(
            ev(DriverEvent::Session(SessionEvent::Ended(EndReason::Rejected(DisconnectReason::SchemaMismatch)))),
            c(MID_NET_EVENT_ENDED, MID_NET_END_REJECTED, DisconnectReason::SchemaMismatch.to_wire() as u64)
        );
        for (r, code) in [
            (EndReason::LocalClose, MID_NET_END_LOCAL_CLOSE),
            (EndReason::ConnectTimeout, MID_NET_END_CONNECT_TIMEOUT),
            (EndReason::ProtocolError, MID_NET_END_PROTOCOL_ERROR),
        ] {
            assert_eq!(ev(DriverEvent::Session(SessionEvent::Ended(r))), c(MID_NET_EVENT_ENDED, code, 0));
        }
        let bit = mid_net_wire::BitError::NotFinite;
        let faults = [
            (Fault::Transport("x".into()), MID_NET_FAULT_TRANSPORT, 0),
            (Fault::TransportClosed, MID_NET_FAULT_TRANSPORT_CLOSED, 0),
            (Fault::UnknownStream(300), MID_NET_FAULT_UNKNOWN_STREAM, 300),
            (Fault::StreamOverflow { channel: 4 }, MID_NET_FAULT_STREAM_OVERFLOW, 4),
            (Fault::MalformedStream { channel: 5, error: bit }, MID_NET_FAULT_MALFORMED_STREAM, 5),
            (Fault::SendQueueFull { channel: 6 }, MID_NET_FAULT_SEND_QUEUE_FULL, 6),
        ];
        for (f, a, b) in faults {
            assert_eq!(ev(DriverEvent::Fault(f)), c(MID_NET_EVENT_FAULT, a, b));
        }
        for (reason, why) in [(DropReason::Oversize, 1), (DropReason::Encode(bit), 2), (DropReason::Unusable, 3)] {
            assert_eq!(
                ev(DriverEvent::Dropped { channel: 3, id: 12, reason }),
                c(MID_NET_EVENT_DROPPED, (3 << 16) | 12, why)
            );
        }
        assert_eq!(
            ev(DriverEvent::SendRejected { channel: 2, id: 9, error: SendError::Unregistered }),
            c(MID_NET_EVENT_SEND_REJECTED, (2 << 16) | 9, (-(St::Unregistered as i32)) as u64)
        );
    }

    // ---- the schema builder and a driver, through the C functions, with a transport made of Rust closures ----

    struct Pipe {
        sent_datagrams: Vec<Vec<u8>>,
        to_deliver: Vec<Vec<u8>>,
        connected: i32,
    }
    unsafe extern "C" fn p_send_dg(user: *mut c_void, d: *const u8, n: usize) -> i32 {
        let p = &mut *(user as *mut Pipe);
        p.sent_datagrams.push(slice::from_raw_parts(d, n).to_vec());
        0
    }
    unsafe extern "C" fn p_recv_dg(user: *mut c_void, buf: *mut u8, cap: usize) -> i32 {
        let p = &mut *(user as *mut Pipe);
        if p.to_deliver.is_empty() {
            return 0;
        }
        let d = p.to_deliver.remove(0);
        if d.len() > cap {
            return -2;
        }
        ptr::copy_nonoverlapping(d.as_ptr(), buf, d.len());
        d.len() as i32
    }
    unsafe extern "C" fn p_send_rel(_: *mut c_void, _: u32, _: *const u8, _: usize) -> i32 {
        0
    }
    unsafe extern "C" fn p_recv_rel(_: *mut c_void, _: *mut u8, _: usize, _: *mut u32) -> i32 {
        0
    }
    unsafe extern "C" fn p_connected(user: *mut c_void) -> i32 {
        (*(user as *mut Pipe)).connected
    }
    fn vtable(p: &mut Pipe) -> MidNetTransportVtable {
        MidNetTransportVtable {
            user: p as *mut Pipe as *mut c_void,
            send_datagram: Some(p_send_dg),
            recv_datagram: Some(p_recv_dg),
            send_reliable: Some(p_send_rel),
            recv_reliable: Some(p_recv_rel),
            is_connected: Some(p_connected),
        }
    }

    fn sealed_schema() -> *mut MidNetSchema {
        unsafe {
            let s = mid_net_schema_new(8, 4);
            assert!(!s.is_null());
            assert_eq!(mid_net_schema_channel(s, 0, MID_NET_CHANNEL_RELIABLE_ORDERED), 0);
            assert_eq!(mid_net_schema_channel(s, 1, MID_NET_CHANNEL_UNRELIABLE_UNORDERED), 0);
            assert_eq!(mid_net_schema_channel(s, 2, MID_NET_CHANNEL_UNRELIABLE_SEQUENCED), 0);
            assert_eq!(mid_net_schema_message(s, 10), 0);
            assert_eq!(mid_net_schema_seal(s, 0, 1), 0);
            s
        }
    }

    #[test]
    fn a_transport_error_from_c_ends_the_connection_and_a_panic_is_not_possible_across_the_boundary() {
        unsafe {
            let schema = sealed_schema();
            let mut pipe = Pipe { sent_datagrams: vec![], to_deliver: vec![], connected: 1 };
            let vt = vtable(&mut pipe);
            let mut cfg: MidNetDriverConfig = std::mem::zeroed();
            mid_net_driver_config_default(&mut cfg);
            let mut d: *mut MidNetDriver = ptr::null_mut();
            assert_eq!(mid_net_driver_client(schema, &cfg, &vt, ptr::null(), 1, 0, &mut d), 0);
            assert!(!d.is_null());
            // a datagram the driver cannot parse: counted, not fatal
            pipe.to_deliver.push(vec![1, 2, 3, 200, 9, 9]);
            assert_eq!(mid_net_driver_poll(d, 10), 0);
            let mut st: MidNetDriverStats = std::mem::zeroed();
            assert_eq!(mid_net_driver_stats(d, &mut st), 0);
            assert_eq!(st.bad_datagrams, 1);
            let mut info: MidNetDriverInfo = std::mem::zeroed();
            assert_eq!(mid_net_driver_info(d, 10, &mut info), 0);
            assert_eq!(info.state, MID_NET_STATE_CONNECTING);
            assert_eq!(info.server_time_ms, -1);
            assert_eq!(info.session_id, 0);
            // the transport reports an error: the connection ends with a fault
            pipe.connected = 0;
            assert_eq!(mid_net_driver_poll(d, 20), 0);
            // (it was never connected from the driver's point of view until the first poll saw it)
            mid_net_driver_free(d);
            mid_net_schema_free(schema);
        }
    }

    #[test]
    fn null_and_misuse_are_errors_not_crashes() {
        unsafe {
            assert_eq!(mid_net_driver_poll(ptr::null_mut(), 0), St::NullPointer as i32);
            assert_eq!(mid_net_driver_close(ptr::null_mut(), 0), St::NullPointer as i32);
            let mut ev = MidNetEvent::default();
            assert_eq!(mid_net_driver_next_event(ptr::null_mut(), &mut ev), St::NullPointer as i32);
            assert_eq!(mid_net_server_poll(ptr::null_mut(), 0), St::NullPointer as i32);
            assert_eq!(mid_net_server_connection_count(ptr::null_mut()), St::NullPointer as i32);
            assert_eq!(mid_net_schema_seal(ptr::null_mut(), 0, 1), St::NullPointer as i32);
            assert_eq!(mid_net_fnv1a64(ptr::null(), 0), fnv1a64(&[]).0);
            assert_eq!(mid_net_fnv1a64(b"abc".as_ptr(), 3), fnv1a64(b"abc").0);
            assert!(mid_net_schema_new(17, 4).is_null());
            assert!(mid_net_schema_new(8, 0).is_null());
            let s = mid_net_schema_new(8, 4);
            assert_eq!(mid_net_schema_seal(s, 0, 1), St::BadSchema as i32, "no channels defined");
            assert_eq!(mid_net_schema_seal(s, 0, 1), St::SchemaSealed as i32, "and the schema is spent");
            assert_eq!(mid_net_schema_channel(s, 0, 0), St::SchemaSealed as i32);
            mid_net_schema_free(s);
            let mut out: *mut MidNetDriver = ptr::NonNull::<MidNetDriver>::dangling().as_ptr();
            let vt = MidNetTransportVtable { user: ptr::null_mut(), send_datagram: None, recv_datagram: None, send_reliable: None, recv_reliable: None, is_connected: None };
            let mut cfg: MidNetDriverConfig = std::mem::zeroed();
            mid_net_driver_config_default(&mut cfg);
            let schema = sealed_schema();
            assert_eq!(mid_net_driver_client(schema, &cfg, &vt, ptr::null(), 1, 0, &mut out), St::BadTransport as i32);
            assert!(out.is_null(), "*out is cleared on an error");
            mid_net_schema_free(schema);
        }
    }
}
