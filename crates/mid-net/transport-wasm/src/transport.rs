//! `WasmTransport` — only compiled for `--target wasm32-unknown-unknown`.
//! Everything above this module (`framing`, `queue`) compiles and is
//! tested on any host; this file is the part that can't be, for two
//! independent reasons: `web-sys`'s WebTransport bindings only exist
//! behind the wasm32 target at all, and even with the right target this
//! sandbox has no browser/JS runtime to actually execute a WebTransport
//! session against — real verification needs `wasm-bindgen-test` in an
//! actual browser (or headless Chrome/Firefox via `wasm-pack test`),
//! neither of which exists here. This is a stronger form of "unverified"
//! than `mid-net-transport-quinn`'s MSRV wall: that crate could at least
//! be reasoned about and partially syntax-checked against the real,
//! downloaded source; this one's syntax was only checked via an inline
//! module wrapper in a throwaway harness (see `lib.rs`'s doc comment for
//! why the obvious approach — cfg-gating the external `mod transport;`
//! itself — turned out not to work), and even that only proves syntax,
//! not that any referenced item actually exists. Treat this as a first
//! draft to be proven on real CI, and ideally a real `wasm-bindgen-test`
//! run, not as something with the same confidence level as the quinn
//! side.
//!
//! Every API call below is cited against the real, current
//! `web-transport-wasm` 0.5.10 source (downloaded from static.crates.io
//! and read directly, same discipline as the quinn crate). Three real
//! divergences from the native side, worth knowing before reading the
//! rest of this file:
//!
//! 1. **`Session::send_datagram` is `async` here**, not sync like
//!    `web_transport_quinn::Session::send_datagram`. So unlike
//!    `QuinnTransport`, datagram sends ALSO have to go through a
//!    queue-plus-background-task, not just reliable sends. `Ok(())` from
//!    `Transport::send_datagram` means "handed to the writer task", same
//!    as `send_reliable` already meant on both backends — now true for
//!    datagrams too, on this backend specifically.
//! 2. **No `read_exact` equivalent.** `RecvStream::read(max)` returns up
//!    to `max` bytes, not exactly `max` — `read_exact_n` below is a small
//!    hand-rolled loop over it.
//! 3. **Session clone handles can't both do the same *kind* of
//!    operation** — directly stated in the crate's own doc comment: "the
//!    session can be cloned to create multiple handles. However, handles
//!    cannot (currently) accept/open the same type of stream." Every
//!    clone below is used for exactly one operation kind (datagram recv,
//!    datagram send, accept_uni, open_uni) and never shared across
//!    tasks, specifically to stay clear of this.
//!
//! There's also no sync `is_connected()`-style getter at all —
//! `Session::closed()` is `async fn closed(&self) -> Error`, it *awaits*
//! until closed rather than reporting current state. Worked around with
//! a small `Rc<Cell<bool>>` flag flipped by a dedicated background task
//! that just awaits `closed()` once.
//!
//! wasm32 is single-threaded (in a browser tab, absent Web Workers +
//! `SharedArrayBuffer`, which is out of scope here) — every background
//! task below runs via `wasm_bindgen_futures::spawn_local` onto the
//! browser's own microtask queue, not an OS thread. That's also why
//! `queue::WakeQueue` is `Rc<RefCell<..>>`-based rather than needing
//! `Arc<Mutex<..>>`/atomics.

use std::cell::Cell;
use std::collections::HashMap;
use std::rc::Rc;

use bytes::Bytes;
use mid_net_transport::Transport;
use wasm_bindgen_futures::spawn_local;
use web_transport_wasm::{RecvStream, SendStream, Session};

use crate::framing;
use crate::queue::WakeQueue;

#[derive(Debug)]
pub enum WasmTransportError {
    Session(web_transport_wasm::Error),
}

impl std::fmt::Display for WasmTransportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WasmTransportError::Session(e) => write!(f, "webtransport session error: {e}"),
        }
    }
}

impl std::error::Error for WasmTransportError {}

/// Wraps an established `web_transport_wasm::Session` as a
/// [`mid_net_transport::Transport`]. See this module's doc comment for
/// the three real divergences from `mid-net-transport-quinn` and the
/// verification status.
pub struct WasmTransport {
    incoming_datagrams: WakeQueue<Vec<u8>>,
    incoming_reliable: WakeQueue<(u32, Vec<u8>)>,
    outgoing_datagrams: WakeQueue<Vec<u8>>,
    outgoing_reliable: WakeQueue<(u32, Vec<u8>)>,
    connected: Rc<Cell<bool>>,
}

impl WasmTransport {
    /// Wrap an already-established WebTransport session. Unlike
    /// `QuinnTransport::new`, this doesn't take a runtime handle — there
    /// is no runtime to hand in, `spawn_local` schedules onto whichever
    /// single JS event loop is already running, and that's the only one
    /// there ever is in a browser tab.
    pub fn new(session: Session) -> Self {
        let incoming_datagrams = WakeQueue::new();
        let incoming_reliable = WakeQueue::new();
        let outgoing_datagrams = WakeQueue::new();
        let outgoing_reliable = WakeQueue::new();
        let connected = Rc::new(Cell::new(true));

        // Five clones, five single-purpose tasks -- see this module's
        // doc comment point 3 on why each clone does exactly one kind of
        // operation and nothing else.
        spawn_local(datagram_reader_loop(session.clone(), incoming_datagrams.handle()));
        spawn_local(datagram_writer_loop(session.clone(), outgoing_datagrams.handle()));
        spawn_local(accept_uni_loop(session.clone(), incoming_reliable.handle()));
        spawn_local(reliable_writer_loop(session.clone(), outgoing_reliable.handle()));
        spawn_local(watch_closed(session, connected.clone()));

        Self {
            incoming_datagrams,
            incoming_reliable,
            outgoing_datagrams,
            outgoing_reliable,
            connected,
        }
    }
}

impl Transport for WasmTransport {
    type Error = WasmTransportError;

    fn send_datagram(&mut self, bytes: &[u8]) -> Result<(), Self::Error> {
        // Async on this backend (see module doc point 1) -- enqueue-only,
        // the real `session.send_datagram().await` call happens on
        // `datagram_writer_loop`. `Ok(())` means "handed off", not "on
        // the wire" -- matches what `send_reliable` already promised on
        // both backends, now also true for datagrams here specifically.
        self.outgoing_datagrams.push(bytes.to_vec());
        Ok(())
    }

    fn poll_datagram(&mut self) -> Result<Option<Vec<u8>>, Self::Error> {
        Ok(self.incoming_datagrams.try_pop())
    }

    fn send_reliable(&mut self, stream_id: u32, bytes: &[u8]) -> Result<(), Self::Error> {
        self.outgoing_reliable.push((stream_id, bytes.to_vec()));
        Ok(())
    }

    fn poll_reliable(&mut self) -> Result<Option<(u32, Vec<u8>)>, Self::Error> {
        Ok(self.incoming_reliable.try_pop())
    }

    fn is_connected(&self) -> bool {
        self.connected.get()
    }
}

/// `Session::closed` is `pub async fn closed(&self) -> Error` -- it
/// awaits until the session is closed and then returns why, there's no
/// sync "is this still open" getter at all. This task exists purely to
/// bridge that one async event into the sync `Cell<bool>` `is_connected()`
/// reads.
async fn watch_closed(session: Session, connected: Rc<Cell<bool>>) {
    session.closed().await;
    connected.set(false);
}

/// `Session::recv_datagram` is `pub async fn recv_datagram(&self) ->
/// Result<Bytes, Error>` -- one datagram per call, loop forever until it
/// errors (session closed).
async fn datagram_reader_loop(session: Session, incoming: WakeQueue<Vec<u8>>) {
    loop {
        match session.recv_datagram().await {
            Ok(bytes) => incoming.push(bytes.to_vec()),
            Err(_) => return,
        }
    }
}

/// Waits on `outgoing.next()` (see `queue.rs`) rather than busy-polling,
/// then does the real, async `session.send_datagram().await` per item,
/// in order, one at a time.
async fn datagram_writer_loop(session: Session, outgoing: WakeQueue<Vec<u8>>) {
    loop {
        let payload = outgoing.next().await;
        // Best-effort: a single dropped datagram on a send failure isn't
        // surfaced back to whichever `Transport::send_datagram` call
        // originally queued it (that call already returned `Ok(())`) --
        // same accepted trade-off `QuinnTransport`'s reliable writer
        // documents, now applying to datagrams here specifically.
        let _ = session.send_datagram(Bytes::from(payload)).await;
    }
}

/// `Session::accept_uni` is `pub async fn accept_uni(&self) ->
/// Result<RecvStream, Error>` -- one incoming stream per call. Each
/// accepted stream gets its own reader task so one slow/stalled stream
/// can't block accepting the next.
async fn accept_uni_loop(session: Session, incoming_reliable: WakeQueue<(u32, Vec<u8>)>) {
    loop {
        match session.accept_uni().await {
            Ok(recv) => spawn_local(read_stream_loop(recv, incoming_reliable.handle())),
            Err(_) => return,
        }
    }
}

/// `RecvStream::read(max)` returns *up to* `max` bytes per call, not
/// exactly `max` (see module doc point 2) -- loops until either exactly
/// `n` bytes have accumulated (`Some`) or the stream ends first (`None`).
async fn read_exact_n(recv: &mut RecvStream, n: usize) -> Result<Option<Vec<u8>>, web_transport_wasm::Error> {
    let mut buf = Vec::with_capacity(n);
    while buf.len() < n {
        match recv.read(n - buf.len()).await? {
            Some(chunk) => buf.extend_from_slice(&chunk),
            None => return Ok(None),
        }
    }
    Ok(Some(buf))
}

/// Reads one accepted stream: the 4-byte `stream_id` header, then a
/// length-prefixed frame per `send_reliable` call, forever until the
/// stream ends or errors.
async fn read_stream_loop(mut recv: RecvStream, incoming_reliable: WakeQueue<(u32, Vec<u8>)>) {
    let header = match read_exact_n(&mut recv, 4).await {
        Ok(Some(buf)) => buf,
        _ => return,
    };
    let stream_id = framing::decode_stream_header(header.try_into().expect("read_exact_n(4) returns exactly 4 bytes"));

    loop {
        let len_buf = match read_exact_n(&mut recv, 4).await {
            Ok(Some(buf)) => buf,
            _ => return, // stream ended -- not necessarily the whole session
        };
        let len = framing::decode_frame_len(len_buf.try_into().expect("read_exact_n(4) returns exactly 4 bytes")) as usize;

        let payload = match read_exact_n(&mut recv, len).await {
            Ok(Some(buf)) => buf,
            _ => return,
        };

        incoming_reliable.push((stream_id, payload));
    }
}

/// Owns the "open a stream on first use, reuse it after that" map
/// described in the crate's wire-format doc, symmetric with
/// `QuinnTransport`'s writer task. Waits on `outgoing.next()` rather
/// than busy-polling.
async fn reliable_writer_loop(session: Session, outgoing: WakeQueue<(u32, Vec<u8>)>) {
    let mut open_streams: HashMap<u32, SendStream> = HashMap::new();

    loop {
        let (stream_id, payload) = outgoing.next().await;

        if !open_streams.contains_key(&stream_id) {
            match session.open_uni().await {
                Ok(mut new_stream) => {
                    if new_stream.write(&framing::stream_header(stream_id)).await.is_err() {
                        continue; // couldn't even write the header; drop this message
                    }
                    open_streams.insert(stream_id, new_stream);
                }
                Err(_) => continue, // session closed; drop this message
            }
        }

        let stream = open_streams
            .get_mut(&stream_id)
            .expect("just inserted above, or already present");
        let frame = framing::encode_frame(&payload);
        if stream.write(&frame).await.is_err() {
            open_streams.remove(&stream_id);
        }
    }
}
