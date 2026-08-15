//! Native test-fixture server for `mid-net-transport-wasm`'s browser
//! integration test. NOT part of `mid-net-transport-wasm`'s own
//! dependency tree, on purpose -- see this crate's `Cargo.toml` comment.
//! Exists purely to give the browser-side `wasm-bindgen-test` something
//! real to connect to, since browsers can't be WebTransport servers.
//!
//! Prints exactly two lines to stdout that the CI workflow parses --
//! this format is load-bearing, not just logging:
//!   PORT=<port>
//!   CERT_HASH_HEX=<hex-encoded sha256 of the cert this server presents>
//!
//! Accepts one session, echoes back anything received (datagram or
//! reliable) for a bounded window, then exits. A fixture, not a real
//! server.
//!
//! Verification status: same as everything else touching quinn --
//! reasoned through against real, downloaded source
//! (`web-transport-quinn`, `rcgen`, `ring`), never compiled or run
//! anywhere before real CI. The cert-validity-window reasoning
//! specifically (see `chrome_compatible_self_signed_cert` below) is
//! sourced from a Mozilla bug report on Firefox's own
//! `serverCertificateHashes` implementation noting THEIRS doesn't
//! enforce a 14-day cap, which is what points at Chrome's enforcing one
//! -- not from Chrome's own docs directly, since a precise authoritative
//! citation for the exact number wasn't found. If the browser test fails
//! specifically on certificate rejection, this window is the first thing
//! to reconsider.

use std::io::Write;
use std::time::Duration;

use mid_net_transport::Transport;
use mid_net_transport_quinn::QuinnTransport;
use rustls_pki_types::{PrivateKeyDer, PrivatePkcs8KeyDer};
use web_transport_quinn::proto::ConnectResponse;

/// ECDSA (rcgen's `KeyPair::generate()` default -- confirmed from source,
/// not assumed) with a validity window well under Chrome's 14-day cap
/// for `serverCertificateHashes`. `rcgen::generate_simple_self_signed`
/// (used everywhere else in this project) defaults to an absurd
/// 1975-4096 window -- fine for a native client that skips verification
/// entirely, but Chrome actually checks this one, so this fixture builds
/// `CertificateParams` by hand instead of using that helper.
fn chrome_compatible_self_signed_cert() -> (
    Vec<rustls_pki_types::CertificateDer<'static>>,
    PrivateKeyDer<'static>,
    Vec<u8>, // sha256 of the DER cert, for the browser to pin
) {
    let mut params = rcgen::CertificateParams::new(vec!["localhost".to_string()])
        .expect("rcgen: failed to build cert params");
    let now = time::OffsetDateTime::now_utc();
    params.not_before = now - time::Duration::hours(1); // small clock-skew buffer
    params.not_after = now + time::Duration::days(13); // safely under the 14-day cap

    let key_pair = rcgen::KeyPair::generate().expect("rcgen: failed to generate ECDSA key");
    let cert = params
        .self_signed(&key_pair)
        .expect("rcgen: failed to self-sign");

    let hash = ring::digest::digest(&ring::digest::SHA256, cert.der().as_ref());
    let chain = vec![cert.der().clone()];
    let key = PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(key_pair.serialize_der()));

    (chain, key, hash.as_ref().to_vec())
}

fn main() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("failed to start the tokio runtime");
    rt.block_on(async_main());
}

async fn async_main() {
    let (chain, key, hash) = chrome_compatible_self_signed_cert();
    let hash_hex: String = hash.iter().map(|b| format!("{b:02x}")).collect();

    let mut server = web_transport_quinn::ServerBuilder::new()
        .with_addr("127.0.0.1:0".parse().unwrap())
        .with_certificate(chain, key)
        .expect("failed to build the WebTransport server");

    let port = server
        .local_addr()
        .expect("server has no local address after binding")
        .port();

    // Load-bearing output format -- the CI workflow's readiness-poll
    // greps for exactly these two prefixes.
    println!("PORT={port}");
    println!("CERT_HASH_HEX={hash_hex}");
    std::io::stdout().flush().ok();

    let handle = tokio::runtime::Handle::current();

    let request = match tokio::time::timeout(Duration::from_secs(30), server.accept()).await {
        Ok(Some(request)) => request,
        Ok(None) => {
            eprintln!("FIXTURE: server's accept() returned None -- endpoint closed?");
            std::process::exit(1);
        }
        Err(_) => {
            eprintln!("FIXTURE: no incoming connection within 30s");
            std::process::exit(1);
        }
    };

    let session = request
        .respond(ConnectResponse::OK)
        .await
        .expect("failed to accept the WebTransport session");
    println!("FIXTURE: session accepted");
    std::io::stdout().flush().ok();

    let mut transport = QuinnTransport::new(session, &handle);

    // Bounded echo window -- ten seconds is comfortably enough for the
    // browser test's own handful of round trips, and bounds this
    // process's lifetime so the CI job doesn't need to hunt for it to
    // kill it (it exits on its own once the window closes).
    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    while tokio::time::Instant::now() < deadline {
        if let Ok(Some(bytes)) = transport.poll_datagram() {
            println!("FIXTURE: recv datagram, {} bytes, echoing", bytes.len());
            let _ = transport.send_datagram(&bytes);
        }
        if let Ok(Some((stream_id, bytes))) = transport.poll_reliable() {
            println!(
                "FIXTURE: recv reliable on stream {stream_id}, {} bytes, echoing",
                bytes.len()
            );
            let _ = transport.send_reliable(stream_id, &bytes);
        }
        std::io::stdout().flush().ok();
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    println!("FIXTURE: exchange window closed, exiting");
}
