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
//! Accepts sessions **in a loop**, echoing back anything received
//! (datagram or reliable) on each one, until an overall bounded window
//! closes, then exits. A fixture, not a real server.
//!
//! ## Why a loop, not a single `accept()`
//!
//! The first real run of this fixture (mid-net-transport-wasm-test.yml's
//! very first CI execution) accepted exactly one session, because that's
//! all the original version of this file ever did. `tests/web.rs` has
//! TWO `#[wasm_bindgen_test]` functions, and each one calls `connect()`
//! independently -- a fresh `WebTransport` session per test function, not
//! one shared session across both. Whichever test happened to connect
//! first (the reliable-stream one, empirically) got the one-and-only
//! accepted session and passed; the second (`datagram_...`) got
//! `WebTransportError: Opening handshake failed`, because nothing was
//! left calling `server.accept()` for its connection by the time it
//! tried -- the fixture had already moved into handling the first
//! session's bounded echo window on the only code path that existed.
//! This is now a loop that keeps calling `accept()` until an overall
//! deadline elapses, so it transparently handles however many
//! independent sessions `tests/web.rs` ends up needing -- one per
//! `#[wasm_bindgen_test]` function that calls `connect()`, without a
//! magic count to keep in sync with that file.
//!
//! Each accepted session is handled on its own spawned task rather than
//! inline in the accept loop, specifically so the loop returns to
//! `server.accept()` immediately after responding to a session instead
//! of blocking for that session's whole echo window first -- otherwise a
//! second test's `connect()` could stall for however long the first
//! session's window still had left to run, which is exactly the kind of
//! timing fragility real concurrent WebTransport sessions shouldn't
//! have to depend on.
//!
//! Verification status: same as everything else touching quinn --
//! reasoned through against real, downloaded source
//! (`web-transport-quinn`, `rcgen`, `ring`), and syntax-checked against
//! real rustc 1.75 (this crate's dependency tree hits the same
//! edition2024 MSRV wall documented elsewhere in this workspace, so it
//! cannot be fully compiled here -- see the root `Cargo.toml` comments).
//! The cert-validity-window reasoning specifically (see
//! `chrome_compatible_self_signed_cert` below) is sourced from a Mozilla
//! bug report on Firefox's own `serverCertificateHashes` implementation
//! noting THEIRS doesn't enforce a 14-day cap, which is what points at
//! Chrome's enforcing one -- not from Chrome's own docs directly, since a
//! precise authoritative citation for the exact number wasn't found. If
//! the browser test fails specifically on certificate rejection, this
//! window is the first thing to reconsider.

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

    // Overall process deadline covering every session this fixture will
    // ever accept, not just the first. Generous on purpose: it has to
    // cover however many test functions in tests/web.rs each
    // independently call connect(), plus wasm-pack's own build/startup
    // overhead before the first one even reaches the fixture.
    let overall_deadline = tokio::time::Instant::now() + Duration::from_secs(60);
    let mut sessions_accepted: u32 = 0;
    let mut session_tasks: Vec<tokio::task::JoinHandle<()>> = Vec::new();

    loop {
        let remaining = overall_deadline.saturating_duration_since(tokio::time::Instant::now());
        if remaining.is_zero() {
            println!("FIXTURE: overall window closed after {sessions_accepted} session(s)");
            break;
        }

        // The very first connection gets the full remaining budget (it
        // has to cover wasm-pack's own build/startup time before the
        // browser even starts dialing). Once at least one session has
        // been accepted, a short settle window is enough -- if another
        // test function were going to connect, it does so within a
        // couple of seconds of the previous one finishing, not tens of
        // seconds later, so there's no reason to keep waiting out the
        // full overall deadline just to notice nothing else is coming.
        let accept_timeout = if sessions_accepted == 0 {
            remaining
        } else {
            remaining.min(Duration::from_secs(5))
        };

        let request = match tokio::time::timeout(accept_timeout, server.accept()).await {
            Ok(Some(request)) => request,
            Ok(None) => {
                eprintln!("FIXTURE: server's accept() returned None -- endpoint closed?");
                if sessions_accepted == 0 {
                    std::process::exit(1);
                }
                break;
            }
            Err(_) => {
                // Nobody connected within the remaining time. That is a
                // real failure if literally nothing has ever connected;
                // once at least one session has been handled, it just
                // means no further test function is going to connect,
                // and the fixture can exit cleanly instead of treating
                // this as an error.
                if sessions_accepted == 0 {
                    eprintln!("FIXTURE: no incoming connection within 60s");
                    std::process::exit(1);
                }
                println!(
                    "FIXTURE: no further connections after {sessions_accepted} session(s), exiting"
                );
                break;
            }
        };

        let session = match request.respond(ConnectResponse::OK).await {
            Ok(session) => session,
            Err(e) => {
                eprintln!("FIXTURE: failed to accept a WebTransport session: {e}");
                continue;
            }
        };
        sessions_accepted += 1;
        let session_number = sessions_accepted;
        println!("FIXTURE: session {session_number} accepted");
        std::io::stdout().flush().ok();

        let task_handle = handle.clone();
        session_tasks.push(handle.spawn(async move {
            let mut transport = QuinnTransport::new(session, &task_handle);

            // Bounded per-session echo window -- ten seconds is
            // comfortably enough for one test function's handful of
            // round trips.
            let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
            while tokio::time::Instant::now() < deadline {
                if let Ok(Some(bytes)) = transport.poll_datagram() {
                    println!(
                        "FIXTURE: session {session_number} recv datagram, {} bytes, echoing",
                        bytes.len()
                    );
                    let _ = transport.send_datagram(&bytes);
                }
                if let Ok(Some((stream_id, bytes))) = transport.poll_reliable() {
                    println!(
                        "FIXTURE: session {session_number} recv reliable on stream {stream_id}, {} bytes, echoing",
                        bytes.len()
                    );
                    let _ = transport.send_reliable(stream_id, &bytes);
                }
                std::io::stdout().flush().ok();
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
            println!("FIXTURE: session {session_number} exchange window closed");
            std::io::stdout().flush().ok();
        }));
    }

    // Give any still-running session tasks a chance to finish their echo
    // window before the process exits and the endpoint/runtime drops out
    // from under them mid-exchange.
    for task in session_tasks {
        let _ = task.await;
    }

    println!("FIXTURE: exiting after {sessions_accepted} session(s) total");
    std::io::stdout().flush().ok();
            }
