//! Browser integration test, run via `wasm-pack test --chrome --headless`
//! against a real native `mid-net-transport-quinn`-backed server (see
//! `../wasm-test-server`). This is the ONE thing in this whole project
//! that proves native and browser backends can actually talk to each
//! other over the wire -- everything else only proved each side works
//! against itself.
//!
//! `WASM_TEST_SERVER_PORT`/`WASM_TEST_CERT_HASH_HEX` are baked in at
//! *compile* time via `env!()` -- the CI workflow starts the fixture
//! server first, parses its PORT/CERT_HASH_HEX stdout lines, and passes
//! them as environment variables to the `wasm-pack test` build step
//! specifically because a browser-run wasm test can't read a CI shell's
//! environment at runtime the normal native way.
//!
//! Verification status: never run, same as `src/transport.rs` -- no
//! wasm32 target or JS runtime anywhere in this project's tooling before
//! now. Every API call is cited against real source in `transport.rs`'s
//! own doc comment; this file adds only `ClientBuilder::with_server_certificate_hashes`
//! and the hex-decode helper below, both straightforward enough not to
//! need their own separate citation beyond `client.rs`'s real source
//! already checked for `lib.rs`'s doc comment.

#![cfg(target_arch = "wasm32")]

use mid_net_transport::Transport;
use mid_net_transport_wasm::WasmTransport;
use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

fn server_port() -> u16 {
    env!("WASM_TEST_SERVER_PORT")
        .parse()
        .expect("WASM_TEST_SERVER_PORT must be set at build time -- see the CI workflow")
}

fn cert_hash_bytes() -> Vec<u8> {
    let hex = env!("WASM_TEST_CERT_HASH_HEX");
    assert_eq!(hex.len(), 64, "expected a 32-byte sha256 hash as 64 hex chars");
    (0..hex.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).expect("WASM_TEST_CERT_HASH_HEX must be valid hex"))
        .collect()
}

async fn connect() -> WasmTransport {
    let port = server_port();
    let hash = cert_hash_bytes();

    // `with_server_certificate_hashes` -- the browser's own documented
    // mechanism for pinning a self-signed cert instead of validating
    // against a root CA (see lib.rs's doc comment for the citation and
    // the 14-day validity constraint this relies on the fixture server
    // satisfying).
    let client = web_transport_wasm::ClientBuilder::new().with_server_certificate_hashes(vec![hash]);

    let url: url::Url = format!("https://127.0.0.1:{port}/")
        .parse()
        .expect("failed to build the connect URL");

    let session = client
        .connect(url)
        .await
        .expect("wasm client failed to connect to the native quinn fixture server");

    WasmTransport::new(session)
}

#[wasm_bindgen_test]
async fn datagram_round_trips_against_a_real_native_quinn_server() {
    let mut transport = connect().await;

    transport
        .send_datagram(b"hello from the browser")
        .expect("send_datagram failed");

    let mut received = None;
    for _ in 0..250 {
        if let Ok(Some(bytes)) = transport.poll_datagram() {
            received = Some(bytes);
            break;
        }
        // wasm32 has no `tokio::time::sleep` -- `gloo-timers` wraps the
        // browser's own `setTimeout`, the standard way to yield/wait in
        // a wasm-bindgen-test.
        gloo_timers::future::TimeoutFuture::new(20).await;
    }

    // The fixture server echoes back whatever it receives.
    assert_eq!(received.as_deref(), Some(&b"hello from the browser"[..]));
}

#[wasm_bindgen_test]
async fn reliable_stream_round_trips_against_a_real_native_quinn_server() {
    let mut transport = connect().await;

    transport
        .send_reliable(3, b"reliable hello from the browser")
        .expect("send_reliable failed");

    let mut received = None;
    for _ in 0..250 {
        if let Ok(Some((stream_id, bytes))) = transport.poll_reliable() {
            received = Some((stream_id, bytes));
            break;
        }
        gloo_timers::future::TimeoutFuture::new(20).await;
    }

    let (stream_id, bytes) = received.expect("never received the echoed reliable message");
    assert_eq!(stream_id, 3);
    assert_eq!(bytes, b"reliable hello from the browser");
}
