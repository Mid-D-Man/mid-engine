//! Real, live integration test: two `QuinnTransport`s connected over an
//! actual loopback QUIC/WebTransport session -- not mocked, not simulated.
//! Self-signed cert via `rcgen`, client configured to skip certificate
//! verification (`ClientBuilder::dangerous().with_no_certificate_verification()`
//! -- a real, documented escape hatch in `web-transport-quinn` itself,
//! meant for exactly this: local testing, not a workaround this file
//! invented).
//!
//! **Verification status:** same situation as `src/lib.rs` -- could not be
//! compiled or run wherever this was written (the same `edition2024` MSRV
//! wall). Every API call here is copied from real, current source, not
//! memory: `web-transport-quinn` 0.11.12's own `examples/echo-client.rs`
//! and `examples/echo-server.rs` (downloaded from static.crates.io and
//! read directly -- this file's connection setup is structurally the same
//! shape as those examples, not a guess), and `rcgen` 0.14.8's own
//! `generate_simple_self_signed` doc example (same source). This is the
//! first time any test in this crate exercises the actual `quinn`/
//! `web_transport_quinn` code paths -- everything in `src/lib.rs` other
//! than the `framing` module has only ever been reasoned through by hand
//! before now.

use mid_net_transport::Transport;
use mid_net_transport_quinn::QuinnTransport;
use rustls_pki_types::{PrivateKeyDer, PrivatePkcs8KeyDer};
use web_transport_quinn::proto::{ConnectRequest, ConnectResponse};

/// Self-signed "localhost" cert, converted to the `rustls-pki-types` DER
/// shapes `ServerBuilder::with_certificate` expects. Verified against
/// rcgen 0.14.8's real source:
/// - `rcgen::Certificate::der(&self) -> &CertificateDer<'static>`
/// - `rcgen::KeyPair::serialize_der(&self) -> Vec<u8>` (PKCS#8), wrapped
///   as `PrivateKeyDer::Pkcs8` -- confirmed against rcgen's own
///   `KeyPair::from_pkcs8_der_and_sign_algo` doc, which round-trips
///   through exactly this type.
fn self_signed_localhost_cert() -> (
    Vec<rustls_pki_types::CertificateDer<'static>>,
    PrivateKeyDer<'static>,
) {
    let rcgen::CertifiedKey { cert, signing_key } =
        rcgen::generate_simple_self_signed(vec!["localhost".to_string()])
            .expect("rcgen: failed to generate the test's self-signed cert");
    let chain = vec![cert.der().clone()];
    let key = PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(signing_key.serialize_der()));
    (chain, key)
}

/// `poll_datagram`/`poll_reliable` are non-blocking by design (see
/// `src/lib.rs`'s doc comment) -- a message here travels through a real
/// background task and a real network round-trip on loopback, so the test
/// has to retry rather than expect it on the first call. Bounded at ~5s
/// (500 * 10ms) so a genuine regression fails the test instead of hanging
/// CI indefinitely.
async fn poll_until_some<T, R, F>(transport: &mut T, mut poll: F) -> Option<R>
where
    F: FnMut(&mut T) -> Option<R>,
{
    for _ in 0..500 {
        if let Some(v) = poll(transport) {
            return Some(v);
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    None
}

/// Sets up one server `Session` and one client `Session` connected to each
/// other over real loopback QUIC, on an OS-assigned port. Shared by both
/// tests below rather than duplicated.
async fn connected_session_pair() -> (web_transport_quinn::Session, web_transport_quinn::Session) {
    let (chain, key) = self_signed_localhost_cert();

    // `ServerBuilder`/`with_certificate` -- exact shape confirmed from
    // `examples/echo-server.rs`. Port 0 = OS-assigned, so parallel test
    // runs never collide on a fixed port.
    let mut server = web_transport_quinn::ServerBuilder::new()
        .with_addr("127.0.0.1:0".parse().unwrap())
        .with_certificate(chain, key)
        .expect("failed to build the WebTransport server");

    // `Server` derefs to `quinn::Endpoint` (confirmed from server.rs's own
    // `impl core::ops::Deref for Server`), so `local_addr()` is the real
    // bound port, not a guess at the port-0 we asked for.
    let server_addr = server
        .local_addr()
        .expect("server has no local address after binding");

    // Accept exactly one session in the background -- `Server::accept`
    // needs `&mut self` and loops internally (see server.rs source), so
    // this task owns `server` for the rest of its life.
    let server_session = tokio::spawn(async move {
        let request = server
            .accept()
            .await
            .expect("server's accept() returned None -- endpoint closed?");
        request
            .respond(ConnectResponse::OK)
            .await
            .expect("server failed to accept the WebTransport session")
    });

    // `dangerous().with_no_certificate_verification()` -- exact shape
    // confirmed from `examples/echo-client.rs`'s `--tls-disable-verify`
    // path. Appropriate here specifically because this is a self-signed
    // cert generated fresh per test run with nothing to verify it
    // against -- not a pattern to copy into anything that isn't a local
    // test.
    let client = web_transport_quinn::ClientBuilder::new()
        .dangerous()
        .with_no_certificate_verification()
        .expect("failed to build the certificate-skipping test client");

    let url: url::Url = format!("https://127.0.0.1:{}", server_addr.port())
        .parse()
        .expect("failed to build the loopback connect URL");

    let client_session = client
        .connect(ConnectRequest::new(url))
        .await
        .expect("client failed to connect over loopback");

    let server_session = server_session
        .await
        .expect("server's accept task panicked");

    (client_session, server_session)
}

#[tokio::test]
async fn datagram_round_trips_over_real_loopback_quic() {
    let handle = tokio::runtime::Handle::current();
    let (client_session, server_session) = connected_session_pair().await;

    let mut client_transport = QuinnTransport::new(client_session, &handle);
    let mut server_transport = QuinnTransport::new(server_session, &handle);

    client_transport
        .send_datagram(b"hello over real quic")
        .expect("send_datagram failed");

    let received = poll_until_some(&mut server_transport, |t| t.poll_datagram().unwrap())
        .await
        .expect("server never received the datagram within the timeout");
    assert_eq!(received, b"hello over real quic");

    assert!(client_transport.is_connected());
    assert!(server_transport.is_connected());
}

#[tokio::test]
async fn reliable_stream_round_trips_with_the_right_stream_id() {
    let handle = tokio::runtime::Handle::current();
    let (client_session, server_session) = connected_session_pair().await;

    let mut client_transport = QuinnTransport::new(client_session, &handle);
    let mut server_transport = QuinnTransport::new(server_session, &handle);

    // Sent from the server side deliberately (not just the client), to
    // prove `accept_uni`'s per-stream reader task path works too, not
    // only the writer path exercised by whichever side calls first in
    // the datagram test above.
    server_transport
        .send_reliable(7, b"reliable payload")
        .expect("send_reliable failed");

    let (stream_id, payload) =
        poll_until_some(&mut client_transport, |t| t.poll_reliable().unwrap())
            .await
            .expect("client never received the reliable message within the timeout");
    assert_eq!(stream_id, 7);
    assert_eq!(payload, b"reliable payload");

    // A second message on the SAME stream_id proves the "open on first
    // use, reuse after that" path (see src/lib.rs's writer task) actually
    // reuses the stream rather than erroring or opening a second one.
    server_transport
        .send_reliable(7, b"second message, same stream_id")
        .expect("send_reliable (second message) failed");

    let (stream_id_2, payload_2) =
        poll_until_some(&mut client_transport, |t| t.poll_reliable().unwrap())
            .await
            .expect("client never received the second reliable message within the timeout");
    assert_eq!(stream_id_2, 7);
    assert_eq!(payload_2, b"second message, same stream_id");
}
