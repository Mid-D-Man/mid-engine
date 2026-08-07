//! Native `Transport` backend over QUIC/WebTransport, via `web-transport-quinn`.
//!
//! **Verification status, read this first:** `web-transport-quinn`'s
//! dependency tree needs `edition2024` (confirmed by trying to resolve it
//! against this workspace's pinned rustc 1.75 -- `cpufeatures 0.3.0`, pulled
//! in transitively, requires a Cargo not yet stabilized on 1.75). That means
//! this file could not be compiled, type-checked, or test-run wherever it
//! was written. It was built by reading the real, current
//! `web-transport-quinn` 0.11.12 API on docs.rs (methods, signatures, and
//! async/sync-ness quoted in the comments below are copied from there, not
//! from memory) and reasoning through it by hand. The one piece of this
//! crate that COULD be compiled and tested locally -- the `framing` module
//! below -- was, and its tests pass. Everything touching `quinn` or
//! `web_transport_quinn` types is unverified pending a real toolchain
//! (needs ~1.85+). Run `cargo test -p mid-net-transport-quinn` on real CI
//! before trusting this beyond "the reasoning looks right on paper."
//!
//! ## Scope
//!
//! This crate implements [`mid_net_transport::Transport`] for an
//! **already-established** [`web_transport_quinn::Session`]. Dialing a
//! server, accepting connections, loading certs, and ALPN setup are all
//! out of scope here -- those are connection-*establishment* concerns,
//! genuinely separate from the wire contract `Transport` describes, and
//! building them now would mean guessing at how `headless-server` wants to
//! configure TLS before that's actually been decided. Establish a
//! `Session` however the caller needs to, then hand it to
//! [`QuinnTransport::new`].
//!
//! ## Runtime requirement
//!
//! `QuinnTransport::new` must be called from inside a running Tokio
//! runtime, and takes a [`tokio::runtime::Handle`] explicitly rather than
//! reaching for an ambient one. This is not a choice made for
//! convenience -- it's load-bearing. A `Session`'s underlying
//! `quinn::Connection` already required a Tokio runtime to exist at all
//! (QUIC's I/O driver has to be registered somewhere), and Tokio's
//! IO/timer resources are bound to the runtime that registered them.
//! Spinning up a second, independent runtime here and trying to drive the
//! same `Session`'s futures from it would poll resources from a reactor
//! that never registered them -- a well-known Tokio footgun, not a
//! guess specific to this crate. Passing the caller's own `Handle` and
//! calling `handle.spawn(..)` schedules onto the *same* runtime the
//! `Session` already belongs to, which sidesteps the problem entirely.
//! This reasoning is inferred from Tokio's documented runtime-resource
//! binding, not confirmed by running it -- flag for real-CI review
//! specifically if `QuinnTransport` ever panics with a "no reactor
//! running" or "I/O driver not registered" style error.
//!
//! ## Two channels, one wire format each
//!
//! - **Datagrams** (`PlayerState`): `Session::send_datagram`/`read_datagram`
//!   used directly, no framing needed -- WebTransport datagrams already
//!   arrive as discrete messages.
//! - **Reliable streams** (`PlayerEvent`): QUIC streams are ordered byte
//!   pipes with no message boundaries of their own, and a caller-chosen
//!   `stream_id: u32` (see `Transport::send_reliable`) has no relationship
//!   to QUIC's own protocol-level stream IDs. So each logical `stream_id`
//!   gets one dedicated uni stream, opened on first use: the first 4 bytes
//!   written to it are the `stream_id` itself (`framing::stream_header`),
//!   everything after that is a sequence of length-prefixed frames
//!   (`framing::encode_frame`), one per `send_reliable` call. Little-endian
//!   throughout, matching `mid-net-wire`/`mid-net-reliable`'s existing
//!   convention -- see `packet.rs`/`reliable.rs`.
//!
//! `web_transport_quinn::Session::open_uni`/`accept_uni` already write and
//! strip WebTransport's own internal stream-type/session-ID header
//! (confirmed from `session.rs`'s source on docs.rs); the framing here is
//! purely this crate's own, layered on top, not a reimplementation of
//! WebTransport's framing.

use bytes::Bytes;
use mid_net_transport::Transport;
use std::collections::HashMap;
use tokio::sync::mpsc;
use web_transport_quinn::{RecvStream, SendStream, Session};

/// Pure byte framing for the reliable-stream wire format described above.
/// No `quinn`, no `tokio`, no I/O -- deliberately kept free of both so it
/// can be unit tested without a live connection. This is the only part of
/// the crate whose tests could actually run in the sandbox this crate was
/// written in; see the crate-level doc comment.
pub(crate) mod framing {
    /// The first 4 bytes written to a freshly opened uni stream: which
    /// caller-chosen `stream_id` this stream carries.
    pub fn stream_header(stream_id: u32) -> [u8; 4] {
        stream_id.to_le_bytes()
    }

    pub fn decode_stream_header(buf: [u8; 4]) -> u32 {
        u32::from_le_bytes(buf)
    }

    /// One `send_reliable` payload on the wire: a 4-byte little-endian
    /// length prefix followed by the payload bytes.
    pub fn encode_frame(payload: &[u8]) -> Vec<u8> {
        let mut out = Vec::with_capacity(4 + payload.len());
        out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        out.extend_from_slice(payload);
        out
    }

    pub fn decode_frame_len(buf: [u8; 4]) -> u32 {
        u32::from_le_bytes(buf)
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn stream_header_round_trips() {
            assert_eq!(decode_stream_header(stream_header(7)), 7);
            assert_eq!(decode_stream_header(stream_header(u32::MAX)), u32::MAX);
            assert_eq!(decode_stream_header(stream_header(0)), 0);
        }

        #[test]
        fn stream_header_is_little_endian() {
            // mid-net-wire/mid-net-reliable are little-endian throughout;
            // this crate matches that convention rather than introducing
            // a second one for the transport layer.
            assert_eq!(stream_header(1), [1, 0, 0, 0]);
            assert_eq!(stream_header(256), [0, 1, 0, 0]);
        }

        #[test]
        fn encode_frame_prefixes_little_endian_length() {
            let frame = encode_frame(&[9, 9, 9]);
            assert_eq!(&frame[0..4], &[3, 0, 0, 0]);
            assert_eq!(&frame[4..], &[9, 9, 9]);
        }

        #[test]
        fn encode_frame_handles_empty_payload() {
            assert_eq!(encode_frame(&[]), vec![0, 0, 0, 0]);
        }

        #[test]
        fn frame_len_round_trips_through_the_prefix() {
            let frame = encode_frame(&[1, 2, 3, 4, 5]);
            let len_buf: [u8; 4] = frame[0..4].try_into().unwrap();
            assert_eq!(decode_frame_len(len_buf), 5);
        }

        #[test]
        fn distinct_payloads_do_not_collide() {
            assert_ne!(encode_frame(&[1, 2]), encode_frame(&[1, 2, 3]));
            assert_ne!(encode_frame(b"a"), encode_frame(b"b"));
        }
    }
}

/// Errors surfaced by [`QuinnTransport`]. Wraps the underlying
/// `web_transport_quinn::SessionError`, plus a case that crate has no
/// equivalent for: the background tasks this crate spawns to bridge
/// `Session`'s async API to `Transport`'s sync one can themselves end
/// (session closed, panic, runtime shutdown) independently of any single
/// call -- `WorkerGone` is what a subsequent `Transport` method call sees
/// when that's already happened.
#[derive(Debug)]
pub enum QuinnTransportError {
    Session(web_transport_quinn::SessionError),
    WorkerGone,
}

impl std::fmt::Display for QuinnTransportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            QuinnTransportError::Session(e) => write!(f, "webtransport session error: {e}"),
            QuinnTransportError::WorkerGone => {
                write!(f, "mid-net-transport-quinn background task ended unexpectedly")
            }
        }
    }
}

// web_transport_quinn::SessionError is built on `thiserror` (a listed
// dependency of that crate); assumed to implement std::error::Error on
// that basis, not individually confirmed against its source.
impl std::error::Error for QuinnTransportError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            QuinnTransportError::Session(e) => Some(e),
            QuinnTransportError::WorkerGone => None,
        }
    }
}

/// Wraps an established [`web_transport_quinn::Session`] as a
/// [`mid_net_transport::Transport`]. See the crate-level doc comment for
/// the runtime requirement and wire format.
pub struct QuinnTransport {
    session: Session,
    incoming_datagrams_rx: mpsc::UnboundedReceiver<Vec<u8>>,
    incoming_reliable_rx: mpsc::UnboundedReceiver<(u32, Vec<u8>)>,
    outgoing_reliable_tx: mpsc::UnboundedSender<(u32, Vec<u8>)>,
    /// Handles to the background tasks this struct owns, kept only so
    /// `Drop` can abort them -- never polled directly. Per-stream reader
    /// tasks (one spawned per incoming uni stream, from inside the accept
    /// loop below) are deliberately NOT tracked here: there can be
    /// arbitrarily many of them, and each one unwinds on its own shortly
    /// after `session.close()` runs in `Drop`, once its next read fails.
    /// That's a brief grace period, not instant cancellation -- an
    /// accepted trade-off against the bookkeeping cost of tracking an
    /// unbounded set of handles.
    background_tasks: Vec<tokio::task::JoinHandle<()>>,
}

impl QuinnTransport {
    /// Wrap an already-established WebTransport session. Must be called
    /// from inside the same Tokio runtime `session` belongs to -- see the
    /// crate-level doc comment's "Runtime requirement" section.
    pub fn new(session: Session, handle: &tokio::runtime::Handle) -> Self {
        let (datagram_tx, incoming_datagrams_rx) = mpsc::unbounded_channel::<Vec<u8>>();
        let (reliable_in_tx, incoming_reliable_rx) = mpsc::unbounded_channel::<(u32, Vec<u8>)>();
        let (outgoing_reliable_tx, mut reliable_out_rx) =
            mpsc::unbounded_channel::<(u32, Vec<u8>)>();

        let mut background_tasks = Vec::with_capacity(3);

        // 1. Datagram receive loop. `Session::read_datagram` is `pub async
        //    fn read_datagram(&self) -> Result<Bytes, SessionError>`
        //    (docs.rs, web-transport-quinn 0.11.12) -- awaits the next
        //    datagram, one at a time, forever until the session errors.
        {
            let session = session.clone();
            let datagram_tx = datagram_tx.clone();
            background_tasks.push(handle.spawn(async move {
                loop {
                    match session.read_datagram().await {
                        Ok(bytes) => {
                            if datagram_tx.send(bytes.to_vec()).is_err() {
                                break; // QuinnTransport (and its receiver) dropped.
                            }
                        }
                        Err(_) => break, // Session closed or errored.
                    }
                }
            }));
        }

        // 2. Incoming reliable streams: `Session::accept_uni` is `pub
        //    async fn accept_uni(&self) -> Result<RecvStream, SessionError>`
        //    -- one call per incoming stream. Each accepted stream gets
        //    its own reader task so a slow/stalled stream can't block
        //    accepting the next one.
        {
            let session = session.clone();
            let reliable_in_tx = reliable_in_tx.clone();
            let reader_handle = handle.clone();
            background_tasks.push(handle.spawn(async move {
                loop {
                    match session.accept_uni().await {
                        Ok(recv) => {
                            let reliable_in_tx = reliable_in_tx.clone();
                            reader_handle.spawn(read_stream_loop(recv, reliable_in_tx));
                        }
                        Err(_) => break, // Session closed or errored.
                    }
                }
            }));
        }

        // 3. Outgoing reliable writer. Owns the "open on first use, reuse
        //    after that" map described in the crate doc comment. Runs on
        //    its own task so `send_reliable` (sync, on the caller's
        //    thread) never blocks on `open_uni().await`/`write_all().await`
        //    -- it just hands the message to this task via the channel.
        {
            let session = session.clone();
            background_tasks.push(handle.spawn(async move {
                let mut open_streams: HashMap<u32, SendStream> = HashMap::new();
                while let Some((stream_id, payload)) = reliable_out_rx.recv().await {
                    if !open_streams.contains_key(&stream_id) {
                        // `Session::open_uni` is `pub async fn open_uni(&self)
                        // -> Result<SendStream, SessionError>` -- already
                        // writes WebTransport's own stream header
                        // internally (confirmed from session.rs source);
                        // what we write next is purely this crate's own
                        // framing on top of that.
                        match session.open_uni().await {
                            Ok(mut new_stream) => {
                                // SendStream::write_all mirrors
                                // quinn::SendStream::write_all -- this is
                                // literally the same call web-transport-quinn's
                                // own `Session::open_uni` uses internally
                                // for its header write (confirmed from
                                // session.rs source), not a guessed API.
                                if new_stream
                                    .write_all(&framing::stream_header(stream_id))
                                    .await
                                    .is_err()
                                {
                                    continue; // Couldn't even write the header; drop this message.
                                }
                                open_streams.insert(stream_id, new_stream);
                            }
                            Err(_) => continue, // Session closed; drop this message.
                        }
                    }

                    let stream = open_streams
                        .get_mut(&stream_id)
                        .expect("just inserted above, or already present");
                    let frame = framing::encode_frame(&payload);
                    if stream.write_all(&frame).await.is_err() {
                        // This stream is dead; drop it so the next message
                        // on this stream_id opens a fresh one.
                        open_streams.remove(&stream_id);
                    }
                }
            }));
        }

        Self {
            session,
            incoming_datagrams_rx,
            incoming_reliable_rx,
            outgoing_reliable_tx,
            background_tasks,
        }
    }
}

/// Reads one accepted stream: the 4-byte `stream_id` header, then a
/// length-prefixed frame per `send_reliable` call, forever until the
/// stream ends or errors.
async fn read_stream_loop(mut recv: RecvStream, tx: mpsc::UnboundedSender<(u32, Vec<u8>)>) {
    // RecvStream::read_exact is `pub async fn read_exact(&mut self, buf:
    // &mut [u8]) -> Result<(), ReadExactError>` (docs.rs) -- fills the
    // whole buffer or errors, which is exactly what a fixed-size header/
    // length-prefix read needs (unlike `read`, which may return short).
    let mut header_buf = [0u8; 4];
    if recv.read_exact(&mut header_buf).await.is_err() {
        return;
    }
    let stream_id = framing::decode_stream_header(header_buf);

    loop {
        let mut len_buf = [0u8; 4];
        if recv.read_exact(&mut len_buf).await.is_err() {
            return; // Stream ended -- not necessarily the whole session.
        }
        let len = framing::decode_frame_len(len_buf) as usize;

        let mut payload = vec![0u8; len];
        if recv.read_exact(&mut payload).await.is_err() {
            return;
        }

        if tx.send((stream_id, payload)).is_err() {
            return; // QuinnTransport (and its receiver) dropped.
        }
    }
}

impl Transport for QuinnTransport {
    type Error = QuinnTransportError;

    fn send_datagram(&mut self, bytes: &[u8]) -> Result<(), Self::Error> {
        // `Session::send_datagram` is `pub fn send_datagram(&self, data:
        // Bytes) -> Result<(), SessionError>` -- genuinely synchronous
        // (docs.rs confirms no `async`), so this needs no channel/task
        // round-trip, unlike every other Transport method here. The
        // `Bytes::copy_from_slice` is one unavoidable copy: `Transport`
        // hands us a borrowed `&[u8]`, `Session::send_datagram` needs an
        // owned `Bytes`.
        self.session
            .send_datagram(Bytes::copy_from_slice(bytes))
            .map_err(QuinnTransportError::Session)
    }

    fn poll_datagram(&mut self) -> Result<Option<Vec<u8>>, Self::Error> {
        match self.incoming_datagrams_rx.try_recv() {
            Ok(bytes) => Ok(Some(bytes)),
            Err(mpsc::error::TryRecvError::Empty) => Ok(None),
            Err(mpsc::error::TryRecvError::Disconnected) => Err(QuinnTransportError::WorkerGone),
        }
    }

    fn send_reliable(&mut self, stream_id: u32, bytes: &[u8]) -> Result<(), Self::Error> {
        // Enqueue-only: the actual `open_uni`/`write_all` calls happen on
        // the background writer task (see `QuinnTransport::new`), so this
        // returns immediately regardless of stream state. A successful
        // `Ok(())` here means "handed to the writer task", not "on the
        // wire" -- matching `Transport::send_reliable`'s doc, which
        // promises delivery, not synchronous confirmation.
        self.outgoing_reliable_tx
            .send((stream_id, bytes.to_vec()))
            .map_err(|_| QuinnTransportError::WorkerGone)
    }

    fn poll_reliable(&mut self) -> Result<Option<(u32, Vec<u8>)>, Self::Error> {
        match self.incoming_reliable_rx.try_recv() {
            Ok(msg) => Ok(Some(msg)),
            Err(mpsc::error::TryRecvError::Empty) => Ok(None),
            Err(mpsc::error::TryRecvError::Disconnected) => Err(QuinnTransportError::WorkerGone),
        }
    }

    fn is_connected(&self) -> bool {
        // QuinnTransport is only ever constructed from an already-fully-
        // established Session (see the crate doc comment's "Scope"
        // section) -- there is no "still handshaking" state representable
        // here, so "not yet closed" and "handshake complete" coincide by
        // construction. `Session::close_reason` is `pub fn close_reason(&self)
        // -> Option<SessionError>` -- `None` while open.
        self.session.close_reason().is_none()
    }
}

impl Drop for QuinnTransport {
    fn drop(&mut self) {
        // Close the underlying session explicitly rather than letting it
        // leak: the background tasks each hold their own `session.clone()`
        // (Session is cheap-clone, Arc-backed per its Send+Sync+Clone
        // impls on docs.rs), so dropping this struct alone would NOT
        // close the connection or stop those tasks' loops on its own.
        self.session.close(0, b"QuinnTransport dropped");
        for task in &self.background_tasks {
            task.abort();
        }
    }
}
