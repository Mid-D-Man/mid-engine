//! C ABI over the native QUIC/WebTransport transport (`mid-net-transport-quinn`).
//!
//! What this is for: a game written in C (Tower of Babylon is the first) that wants mid-net's
//! transport underneath its own protocol. The `Transport` trait is five calls -- send/poll a
//! datagram, send/poll a reliable message on a stream, ask whether the connection is up -- and
//! that is all this exposes, plus what the trait deliberately leaves out: *establishing* a
//! connection (bind and accept on the server, connect on the client). `mid-net`'s own
//! `PlayerState`/`PlayerEvent` codec is not involved: this moves bytes, the caller decides what
//! they mean. (The codec stays available through `mid_net.h`.)
//!
//! Shape: everything is non-blocking and poll-based, matching the trait. One process-wide Tokio
//! runtime (two worker threads) is created on first use and does all the I/O; the C side never
//! sees it. A server hands out one connection handle per client through `accept`; a client gets
//! its handle at once from `connect` and watches `state` until it is connected.
//!
//! ## Conventions (same as `mid-net`'s `ffi.rs`)
//! - Every function returns a status code or a non-negative count; `MidNetQuicStatus::Ok` is 0.
//!   The first six codes carry the same numbers as `MidNetStatus` (null pointer -1, buffer too
//!   small -2, internal panic -6).
//! - Every body runs inside `catch_unwind`: a panic becomes `InternalPanic`, never an unwind
//!   across the C boundary.
//! - Every function taking a raw pointer is `unsafe fn` with a `# Safety` contract and checks
//!   for null first.
//!
//! ## Things to know
//! - **IPv4 only, deliberately.** `web-transport-quinn`'s own client builder binds `[::]:0` and
//!   unwraps it, which panics where IPv6 sockets do not exist; this binds `0.0.0.0:0` (or
//!   `[::]:0` only when the host given is an IPv6 literal) and builds the client itself.
//! - **The client does not verify the server's certificate** (the `dangerous()` pattern the
//!   headless-server example uses): the connection is encrypted but the server is not
//!   authenticated. Right for a LAN game today, not for anything that matters. Pinned
//!   certificate hashes (`ClientBuilder::with_server_certificate_hashes`) are the next step.
//! - The server makes a fresh self-signed certificate for `localhost` each time it opens.
//! - An empty datagram is not delivered (it carries nothing); an oversized one is refused with
//!   `SendFailed`.
//! - `send_reliable` returns once the message is handed to the writer task, not once it is on
//!   the wire (see `mid-net-transport-quinn`). Closing a connection with a linger time lets
//!   queued messages (a last `Disconnect`) go out before the session is closed.

use std::ffi::CStr;
use std::net::SocketAddr;
use std::os::raw::c_char;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::slice;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use mid_net_transport::Transport;
use mid_net_transport_quinn::QuinnTransport;
use tokio::runtime::{Builder, Handle, Runtime};
use tokio::sync::mpsc;
use web_transport_quinn::proto::{ConnectRequest, ConnectResponse};
use web_transport_quinn::quinn;
use web_transport_quinn::{Client, ServerBuilder};

/// Status codes. Counts returned by `recv` functions are non-negative; everything else returns
/// one of these.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MidNetQuicStatus {
    Ok = 0,
    NullPointer = -1,
    BufferTooSmall = -2,
    /// Something inside this crate panicked. Exists so a caller gets a code, not UB.
    InternalPanic = -6,
    InvalidAddress = -10,
    BindFailed = -11,
    /// The connection is still being established.
    NotConnected = -12,
    /// The connection is closed, or failed to open (see `mid_net_quic_conn_error`).
    Closed = -13,
    /// The transport refused the message (too big for a datagram, or the writer is gone).
    SendFailed = -14,
    CertError = -15,
}

fn guard(f: impl FnOnce() -> i32) -> i32 {
    catch_unwind(AssertUnwindSafe(f)).unwrap_or(MidNetQuicStatus::InternalPanic as i32)
}

fn runtime() -> &'static Runtime {
    static RT: OnceLock<Runtime> = OnceLock::new();
    RT.get_or_init(|| {
        Builder::new_multi_thread()
            .worker_threads(2)
            .thread_name("mid-net-quic")
            .enable_all()
            .build()
            .expect("mid-net-quic: could not start the Tokio runtime")
    })
}

// ---------------------------------------------------------------------------------------------
// Connections
// ---------------------------------------------------------------------------------------------

enum ConnState {
    Connecting,
    Connected(QuinnTransport),
    Failed(String),
    Closed,
}

/// One connection to one peer: what a server's `accept` returns, and what a client's `connect`
/// returns at once (in the `Connecting` state until the handshake finishes).
pub struct MidNetQuicConn {
    state: Arc<Mutex<ConnState>>,
}

impl MidNetQuicConn {
    fn connected(t: QuinnTransport) -> Self {
        MidNetQuicConn { state: Arc::new(Mutex::new(ConnState::Connected(t))) }
    }
}

/// `0` connecting, `1` connected, `2` closed or failed.
pub const MID_NET_QUIC_CONNECTING: i32 = 0;
pub const MID_NET_QUIC_CONNECTED: i32 = 1;
pub const MID_NET_QUIC_CLOSED: i32 = 2;

fn conn_state_code(state: &mut ConnState) -> i32 {
    match state {
        ConnState::Connecting => MID_NET_QUIC_CONNECTING,
        ConnState::Connected(t) => {
            if t.is_connected() {
                MID_NET_QUIC_CONNECTED
            } else {
                MID_NET_QUIC_CLOSED
            }
        }
        ConnState::Failed(_) | ConnState::Closed => MID_NET_QUIC_CLOSED,
    }
}

// ---------------------------------------------------------------------------------------------
// Server
// ---------------------------------------------------------------------------------------------

/// A listening server. `accept` hands out one connection per client that has completed the
/// WebTransport handshake.
pub struct MidNetQuicServer {
    port: u16,
    accepted: Mutex<mpsc::UnboundedReceiver<MidNetQuicConn>>,
    task: tokio::task::JoinHandle<()>,
}

fn self_signed_cert() -> Result<
    (Vec<rustls_pki_types::CertificateDer<'static>>, rustls_pki_types::PrivateKeyDer<'static>),
    String,
> {
    let rcgen::CertifiedKey { cert, signing_key } =
        rcgen::generate_simple_self_signed(vec!["localhost".to_string()]).map_err(|e| e.to_string())?;
    let chain = vec![cert.der().clone()];
    let key = rustls_pki_types::PrivateKeyDer::Pkcs8(rustls_pki_types::PrivatePkcs8KeyDer::from(
        signing_key.serialize_der(),
    ));
    Ok((chain, key))
}

fn parse_host(host: *const c_char) -> Result<String, MidNetQuicStatus> {
    if host.is_null() {
        return Err(MidNetQuicStatus::NullPointer);
    }
    // SAFETY: the caller promises a valid NUL-terminated string (documented on each function).
    let s = unsafe { CStr::from_ptr(host) }.to_str().map_err(|_| MidNetQuicStatus::InvalidAddress)?;
    if s.is_empty() {
        return Err(MidNetQuicStatus::InvalidAddress);
    }
    Ok(s.to_string())
}

/// Opens a server on `bind_host:port` (`bind_host` NULL means every IPv4 address; `port` 0 picks
/// a free one, see `mid_net_quic_server_port`). Writes the handle to `*out`.
///
/// # Safety
/// `bind_host` is NULL or a valid NUL-terminated string; `out` is valid for one pointer write.
#[no_mangle]
pub unsafe extern "C" fn mid_net_quic_server_open(
    bind_host: *const c_char,
    port: u16,
    out: *mut *mut MidNetQuicServer,
) -> i32 {
    guard(|| {
        if out.is_null() {
            return MidNetQuicStatus::NullPointer as i32;
        }
        let host = if bind_host.is_null() {
            "0.0.0.0".to_string()
        } else {
            match parse_host(bind_host) {
                Ok(h) => h,
                Err(e) => return e as i32,
            }
        };
        let addr: SocketAddr = match format!("{host}:{port}").parse() {
            Ok(a) => a,
            Err(_) => return MidNetQuicStatus::InvalidAddress as i32,
        };
        let (chain, key) = match self_signed_cert() {
            Ok(c) => c,
            Err(_) => return MidNetQuicStatus::CertError as i32,
        };
        let rt = runtime();
        let _guard = rt.enter(); // quinn registers its socket with the runtime that is current
        let mut server = match ServerBuilder::new().with_addr(addr).with_certificate(chain, key) {
            Ok(s) => s,
            Err(_) => return MidNetQuicStatus::BindFailed as i32,
        };
        let bound = match server.local_addr() {
            Ok(a) => a.port(),
            Err(_) => return MidNetQuicStatus::BindFailed as i32,
        };
        let (tx, rx) = mpsc::unbounded_channel::<MidNetQuicConn>();
        let handle = rt.handle().clone();
        let task = rt.spawn(async move {
            while let Some(request) = server.accept().await {
                let tx = tx.clone();
                let handle = handle.clone();
                // Answering takes a round trip: do it off the accept loop so one slow client
                // cannot hold up the next.
                tokio::spawn(async move {
                    if let Ok(session) = request.respond(ConnectResponse::OK).await {
                        let _ = tx.send(MidNetQuicConn::connected(QuinnTransport::new(session, &handle)));
                    }
                });
            }
        });
        let boxed = Box::new(MidNetQuicServer { port: bound, accepted: Mutex::new(rx), task });
        // SAFETY: out was checked for null above.
        unsafe { *out = Box::into_raw(boxed) };
        MidNetQuicStatus::Ok as i32
    })
}

/// The port the server is bound to (useful after opening with port 0). 0 for a NULL handle.
///
/// # Safety
/// `server` is NULL or a handle from `mid_net_quic_server_open` not yet closed.
#[no_mangle]
pub unsafe extern "C" fn mid_net_quic_server_port(server: *const MidNetQuicServer) -> u16 {
    catch_unwind(AssertUnwindSafe(|| {
        if server.is_null() {
            0
        } else {
            // SAFETY: non-null, and valid per the contract above.
            unsafe { (*server).port }
        }
    }))
    .unwrap_or(0)
}

/// Takes the next client that has connected, if any. Returns `1` and writes the connection to
/// `*out`, or `0` when nobody is waiting, or a negative status.
///
/// # Safety
/// `server` is a live handle; `out` is valid for one pointer write.
#[no_mangle]
pub unsafe extern "C" fn mid_net_quic_server_accept(
    server: *mut MidNetQuicServer,
    out: *mut *mut MidNetQuicConn,
) -> i32 {
    guard(|| {
        if server.is_null() || out.is_null() {
            return MidNetQuicStatus::NullPointer as i32;
        }
        // SAFETY: non-null, valid per the contract.
        let server = unsafe { &*server };
        let mut rx = match server.accepted.lock() {
            Ok(g) => g,
            Err(_) => return MidNetQuicStatus::InternalPanic as i32,
        };
        match rx.try_recv() {
            Ok(conn) => {
                // SAFETY: out checked for null above.
                unsafe { *out = Box::into_raw(Box::new(conn)) };
                1
            }
            Err(_) => 0,
        }
    })
}

/// Stops listening and frees the handle. Connections already accepted are unaffected.
///
/// # Safety
/// `server` is NULL or a live handle; it is invalid afterwards.
#[no_mangle]
pub unsafe extern "C" fn mid_net_quic_server_close(server: *mut MidNetQuicServer) {
    let _ = catch_unwind(AssertUnwindSafe(|| {
        if server.is_null() {
            return;
        }
        // SAFETY: a handle made by Box::into_raw above, closed once.
        let server = unsafe { Box::from_raw(server) };
        server.task.abort();
    }));
}

// ---------------------------------------------------------------------------------------------
// Client
// ---------------------------------------------------------------------------------------------

#[derive(Debug)]
struct NoCertificateVerification(Arc<quinn::rustls::crypto::CryptoProvider>);

impl quinn::rustls::client::danger::ServerCertVerifier for NoCertificateVerification {
    fn verify_server_cert(
        &self,
        _end_entity: &quinn::rustls::pki_types::CertificateDer<'_>,
        _intermediates: &[quinn::rustls::pki_types::CertificateDer<'_>],
        _server_name: &quinn::rustls::pki_types::ServerName<'_>,
        _ocsp: &[u8],
        _now: quinn::rustls::pki_types::UnixTime,
    ) -> Result<quinn::rustls::client::danger::ServerCertVerified, quinn::rustls::Error> {
        Ok(quinn::rustls::client::danger::ServerCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        message: &[u8],
        cert: &quinn::rustls::pki_types::CertificateDer<'_>,
        dss: &quinn::rustls::DigitallySignedStruct,
    ) -> Result<quinn::rustls::client::danger::HandshakeSignatureValid, quinn::rustls::Error> {
        quinn::rustls::crypto::verify_tls12_signature(message, cert, dss, &self.0.signature_verification_algorithms)
    }

    fn verify_tls13_signature(
        &self,
        message: &[u8],
        cert: &quinn::rustls::pki_types::CertificateDer<'_>,
        dss: &quinn::rustls::DigitallySignedStruct,
    ) -> Result<quinn::rustls::client::danger::HandshakeSignatureValid, quinn::rustls::Error> {
        quinn::rustls::crypto::verify_tls13_signature(message, cert, dss, &self.0.signature_verification_algorithms)
    }

    fn supported_verify_schemes(&self) -> Vec<quinn::rustls::SignatureScheme> {
        self.0.signature_verification_algorithms.supported_schemes()
    }
}

fn build_client(ipv6: bool) -> Result<Client, String> {
    let provider = web_transport_quinn::crypto::default_provider();
    let mut crypto = quinn::rustls::ClientConfig::builder_with_provider(provider.clone())
        .with_protocol_versions(&[&quinn::rustls::version::TLS13])
        .map_err(|e| e.to_string())?
        .dangerous()
        .with_custom_certificate_verifier(Arc::new(NoCertificateVerification(provider)))
        .with_no_client_auth();
    crypto.alpn_protocols = vec![web_transport_quinn::ALPN.as_bytes().to_vec()];
    let quic = quinn::crypto::rustls::QuicClientConfig::try_from(crypto).map_err(|e| e.to_string())?;
    let mut config = quinn::ClientConfig::new(Arc::new(quic));
    let mut transport = quinn::TransportConfig::default();
    // A lobby can sit quiet for a while: keep the connection from timing out meanwhile.
    transport.keep_alive_interval(Some(Duration::from_secs(5)));
    config.transport_config(Arc::new(transport));
    let bind = if ipv6 { "[::]:0" } else { "0.0.0.0:0" };
    let endpoint = quinn::Endpoint::client(bind.parse().map_err(|_| "bad bind address".to_string())?)
        .map_err(|e| e.to_string())?;
    Ok(Client::new(endpoint, config))
}

/// Starts connecting to `host:port` and returns a handle at once, in the connecting state;
/// watch `mid_net_quic_conn_state`. `host` is an IPv4 address, an IPv6 literal or a name.
///
/// # Safety
/// `host` is a valid NUL-terminated string; `out` is valid for one pointer write.
#[no_mangle]
pub unsafe extern "C" fn mid_net_quic_connect(
    host: *const c_char,
    port: u16,
    out: *mut *mut MidNetQuicConn,
) -> i32 {
    guard(|| {
        if out.is_null() {
            return MidNetQuicStatus::NullPointer as i32;
        }
        let host = match parse_host(host) {
            Ok(h) => h,
            Err(e) => return e as i32,
        };
        let ipv6 = host.contains(':');
        let authority = if ipv6 { format!("[{host}]") } else { host.clone() };
        let url: url::Url = match format!("https://{authority}:{port}").parse() {
            Ok(u) => u,
            Err(_) => return MidNetQuicStatus::InvalidAddress as i32,
        };
        let rt = runtime();
        let _guard = rt.enter();
        let client = match build_client(ipv6) {
            Ok(c) => c,
            Err(_) => return MidNetQuicStatus::BindFailed as i32,
        };
        let state = Arc::new(Mutex::new(ConnState::Connecting));
        let task_state = state.clone();
        let handle = rt.handle().clone();
        rt.spawn(async move {
            let result = client.connect(ConnectRequest::new(url)).await;
            let next = match result {
                Ok(session) => ConnState::Connected(QuinnTransport::new(session, &handle)),
                Err(e) => ConnState::Failed(e.to_string()),
            };
            if let Ok(mut s) = task_state.lock() {
                // A close that came first stays closed.
                if matches!(*s, ConnState::Connecting) {
                    *s = next;
                }
            }
        });
        // SAFETY: out checked for null above.
        unsafe { *out = Box::into_raw(Box::new(MidNetQuicConn { state })) };
        MidNetQuicStatus::Ok as i32
    })
}

// ---------------------------------------------------------------------------------------------
// One connection
// ---------------------------------------------------------------------------------------------

fn with_transport(
    conn: *mut MidNetQuicConn,
    f: impl FnOnce(&mut QuinnTransport) -> i32,
) -> i32 {
    if conn.is_null() {
        return MidNetQuicStatus::NullPointer as i32;
    }
    // SAFETY: non-null; valid per each caller's documented contract.
    let conn = unsafe { &*conn };
    let mut state = match conn.state.lock() {
        Ok(g) => g,
        Err(_) => return MidNetQuicStatus::InternalPanic as i32,
    };
    match &mut *state {
        ConnState::Connecting => MidNetQuicStatus::NotConnected as i32,
        ConnState::Failed(_) | ConnState::Closed => MidNetQuicStatus::Closed as i32,
        ConnState::Connected(t) => {
            if !t.is_connected() {
                return MidNetQuicStatus::Closed as i32;
            }
            f(t)
        }
    }
}

/// `0` connecting, `1` connected, `2` closed or failed; a negative status for a NULL handle.
///
/// # Safety
/// `conn` is a live handle.
#[no_mangle]
pub unsafe extern "C" fn mid_net_quic_conn_state(conn: *mut MidNetQuicConn) -> i32 {
    guard(|| {
        if conn.is_null() {
            return MidNetQuicStatus::NullPointer as i32;
        }
        // SAFETY: non-null, valid per the contract.
        let conn = unsafe { &*conn };
        match conn.state.lock() {
            Ok(mut g) => conn_state_code(&mut g),
            Err(_) => MidNetQuicStatus::InternalPanic as i32,
        }
    })
}

/// Why a connection failed or closed, as text. Writes at most `cap - 1` bytes and a NUL, and
/// returns the length of the whole message (so `cap == 0` with a NULL `buf` asks the size).
/// Returns 0 when there is nothing to report.
///
/// # Safety
/// `conn` is a live handle; `buf` is NULL or valid for `cap` bytes.
#[no_mangle]
pub unsafe extern "C" fn mid_net_quic_conn_error(conn: *mut MidNetQuicConn, buf: *mut c_char, cap: usize) -> i32 {
    guard(|| {
        if conn.is_null() {
            return MidNetQuicStatus::NullPointer as i32;
        }
        // SAFETY: non-null, valid per the contract.
        let conn = unsafe { &*conn };
        let state = match conn.state.lock() {
            Ok(g) => g,
            Err(_) => return MidNetQuicStatus::InternalPanic as i32,
        };
        let text: &str = match &*state {
            ConnState::Failed(s) => s.as_str(),
            ConnState::Closed => "closed",
            _ => "",
        };
        if !buf.is_null() && cap > 0 {
            let n = text.len().min(cap - 1);
            // SAFETY: buf is valid for cap bytes and n + 1 <= cap.
            unsafe {
                std::ptr::copy_nonoverlapping(text.as_ptr(), buf as *mut u8, n);
                *buf.add(n) = 0;
            }
        }
        text.len() as i32
    })
}

/// Sends an unreliable datagram. `SendFailed` when it is too big for one or the link is gone.
///
/// # Safety
/// `conn` is a live handle; `data` is valid for `len` bytes (or NULL with `len == 0`).
#[no_mangle]
pub unsafe extern "C" fn mid_net_quic_send_datagram(conn: *mut MidNetQuicConn, data: *const u8, len: usize) -> i32 {
    guard(|| {
        if data.is_null() && len != 0 {
            return MidNetQuicStatus::NullPointer as i32;
        }
        // SAFETY: valid for len bytes per the contract.
        let bytes: &[u8] = if len == 0 { &[] } else { unsafe { slice::from_raw_parts(data, len) } };
        with_transport(conn, |t| match t.send_datagram(bytes) {
            Ok(()) => MidNetQuicStatus::Ok as i32,
            Err(_) => MidNetQuicStatus::SendFailed as i32,
        })
    })
}

/// Takes the next received datagram into `buf`. Returns its length, `0` when none is waiting,
/// `BufferTooSmall` when it did not fit in `cap` (it is dropped, as it would be on the wire).
///
/// # Safety
/// `conn` is a live handle; `buf` is valid for `cap` bytes.
#[no_mangle]
pub unsafe extern "C" fn mid_net_quic_recv_datagram(conn: *mut MidNetQuicConn, buf: *mut u8, cap: usize) -> i32 {
    guard(|| {
        if buf.is_null() && cap != 0 {
            return MidNetQuicStatus::NullPointer as i32;
        }
        with_transport(conn, |t| loop {
            match t.poll_datagram() {
                Ok(Some(d)) if d.is_empty() => continue,
                Ok(Some(d)) => {
                    if d.len() > cap {
                        return MidNetQuicStatus::BufferTooSmall as i32;
                    }
                    // SAFETY: buf is valid for cap bytes and d.len() <= cap.
                    unsafe { std::ptr::copy_nonoverlapping(d.as_ptr(), buf, d.len()) };
                    return d.len() as i32;
                }
                Ok(None) => return 0,
                Err(_) => return MidNetQuicStatus::Closed as i32,
            }
        })
    })
}

/// Sends a message on reliable stream `stream_id` (opened on first use); it arrives whole, in
/// order, once. An empty message is refused.
///
/// # Safety
/// `conn` is a live handle; `data` is valid for `len` bytes.
#[no_mangle]
pub unsafe extern "C" fn mid_net_quic_send_reliable(
    conn: *mut MidNetQuicConn,
    stream_id: u32,
    data: *const u8,
    len: usize,
) -> i32 {
    guard(|| {
        if data.is_null() {
            return MidNetQuicStatus::NullPointer as i32;
        }
        if len == 0 {
            return MidNetQuicStatus::SendFailed as i32;
        }
        // SAFETY: valid for len bytes per the contract.
        let bytes = unsafe { slice::from_raw_parts(data, len) };
        with_transport(conn, |t| match t.send_reliable(stream_id, bytes) {
            Ok(()) => MidNetQuicStatus::Ok as i32,
            Err(_) => MidNetQuicStatus::SendFailed as i32,
        })
    })
}

/// Takes the next reliable message into `buf` and writes its stream id to `*stream_id` (which
/// may be NULL). Returns its length, `0` when none is waiting, `BufferTooSmall` when it did not
/// fit (it is dropped).
///
/// # Safety
/// `conn` is a live handle; `buf` is valid for `cap` bytes; `stream_id` is NULL or valid for a write.
#[no_mangle]
pub unsafe extern "C" fn mid_net_quic_recv_reliable(
    conn: *mut MidNetQuicConn,
    stream_id: *mut u32,
    buf: *mut u8,
    cap: usize,
) -> i32 {
    guard(|| {
        if buf.is_null() && cap != 0 {
            return MidNetQuicStatus::NullPointer as i32;
        }
        with_transport(conn, |t| match t.poll_reliable() {
            Ok(Some((id, m))) => {
                if m.len() > cap {
                    return MidNetQuicStatus::BufferTooSmall as i32;
                }
                // SAFETY: buf valid for cap bytes, m.len() <= cap; stream_id checked for null.
                unsafe {
                    std::ptr::copy_nonoverlapping(m.as_ptr(), buf, m.len());
                    if !stream_id.is_null() {
                        *stream_id = id;
                    }
                }
                m.len() as i32
            }
            Ok(None) => 0,
            Err(_) => MidNetQuicStatus::Closed as i32,
        })
    })
}

/// Closes the connection and frees the handle. With `linger_ms > 0` the session stays open that
/// long in the background first, so messages already queued (a last `Disconnect`) are sent.
///
/// # Safety
/// `conn` is NULL or a live handle; it is invalid afterwards.
#[no_mangle]
pub unsafe extern "C" fn mid_net_quic_conn_close(conn: *mut MidNetQuicConn, linger_ms: u32) {
    let _ = catch_unwind(AssertUnwindSafe(|| {
        if conn.is_null() {
            return;
        }
        // SAFETY: a handle made by Box::into_raw above, closed once.
        let conn = unsafe { Box::from_raw(conn) };
        let taken = match conn.state.lock() {
            Ok(mut s) => std::mem::replace(&mut *s, ConnState::Closed),
            Err(_) => return,
        };
        if let ConnState::Connected(t) = taken {
            if linger_ms == 0 {
                drop(t); // closes the session (QuinnTransport's Drop)
            } else {
                let _ = Handle::try_current().ok().or_else(|| Some(runtime().handle().clone())).map(|h| {
                    h.spawn(async move {
                        tokio::time::sleep(Duration::from_millis(u64::from(linger_ms))).await;
                        drop(t);
                    })
                });
            }
        }
    }));
}
