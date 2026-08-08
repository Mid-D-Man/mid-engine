//! headless-server — integration smoke test
//!
//! Verifies mid-log + mid-net (now including mid-net-transport-quinn) work
//! together over a REAL QUIC/WebTransport connection -- not loopback-only
//! this time, a real bindable server and a real dialing client. mid-ecs is
//! still a stub (see crates/mid-ecs), so there's no world to tick -- that
//! TODO stays a TODO honestly rather than faking it with nothing behind it.
//!
//! Two modes, one binary. Kept as one rather than splitting into a second
//! example crate: both modes share all the connection-setup and tick-loop
//! code below, and splitting would mean maintaining that setup twice for
//! two thin CLI shells around it.
//!
//!   cargo run --bin headless-server -- server [bind-addr]
//!   cargo run --bin headless-server -- client <server-addr> [player-id]
//!
//! Server: binds (default 0.0.0.0:5000), generates a self-signed cert
//! (this is a demo -- a real deployment loads a real one), accepts
//! connections, and for each spawns a task that logs everything received
//! and sends a synthetic, slowly-moving PlayerState every tick.
//!
//! Client: connects, skipping certificate verification (trusting the demo
//! server's self-signed cert -- `ClientBuilder::dangerous()`'s documented
//! purpose, not a pattern for anything real), sends a synthetic
//! PlayerState every tick, logs everything received.
//!
//! Verification status: could not be run wherever this was written, same
//! `edition2024` MSRV wall as `mid-net-transport-quinn` itself (this
//! binary depends on it, and on `web-transport-quinn`, directly). Every
//! connection-setup call here is the same, already-source-verified shape
//! used in that crate's own `tests/loopback_integration.rs` -- see that
//! file's doc comment for exactly which real source was checked.

use std::{env, net::SocketAddr, time::Duration};

use mid_log::level::Tier;
use mid_log::{mid_info, mid_kvinfo, mid_warn};
use mid_net::{Connection, ConnectionEvent, PlayerEvent, PlayerId, PlayerState};
use mid_net_transport_quinn::QuinnTransport;
use rustls_pki_types::{PrivateKeyDer, PrivatePkcs8KeyDer};
use web_transport_quinn::proto::{ConnectRequest, ConnectResponse};

const DEFAULT_PORT: u16 = 5000;
// Demo tick rate. The original TODO's "128 Hz" figure is mid-net's own
// unreliable-channel design target (see docs/mid-net.md), not a promise
// this example shell keeps on its own -- 30 Hz here is plenty to prove
// the send/poll/echo path actually works end to end without spamming the
// log.
const TICK: Duration = Duration::from_millis(1000 / 30);

fn main() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("failed to start the tokio runtime");
    rt.block_on(async_main());
}

async fn async_main() {
    mid_log::logger::MidLogger::init();
    mid_info!(Tier::Mid, "Mid Engine headless server starting...");

    let args: Vec<String> = env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("server") => {
            let addr = args
                .get(2)
                .and_then(|s| s.parse().ok())
                .unwrap_or_else(|| SocketAddr::from(([0, 0, 0, 0], DEFAULT_PORT)));
            run_server(addr).await;
        }
        Some("client") => {
            let Some(addr) = args.get(2) else {
                eprintln!("usage: headless-server client <server-addr> [player-id]");
                std::process::exit(1);
            };
            let player_id: u32 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(1);
            run_client(addr, player_id).await;
        }
        _ => {
            eprintln!("usage:");
            eprintln!("  headless-server server [bind-addr]      (default 0.0.0.0:{DEFAULT_PORT})");
            eprintln!("  headless-server client <server-addr> [player-id]");
            std::process::exit(1);
        }
    }
}

/// Self-signed "localhost" cert -- same source-verified shape as
/// `mid-net-transport-quinn`'s own integration test (see that file for
/// the rcgen API citations); duplicated here rather than shared because
/// this is example code, not a library boundary worth introducing for
/// one helper function.
fn self_signed_localhost_cert() -> (
    Vec<rustls_pki_types::CertificateDer<'static>>,
    PrivateKeyDer<'static>,
) {
    let rcgen::CertifiedKey { cert, signing_key } =
        rcgen::generate_simple_self_signed(vec!["localhost".to_string()])
            .expect("rcgen: failed to generate the demo's self-signed cert");
    let chain = vec![cert.der().clone()];
    let key = PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(signing_key.serialize_der()));
    (chain, key)
}

/// A synthetic, slowly-orbiting position -- there's no real simulation to
/// drive this from (mid-ecs is still a stub), so it exists purely to give
/// the send/poll loop something non-constant to move through the pipe and
/// prove it round-trips, not to model anything.
fn synthetic_player_state(tick: u64) -> PlayerState {
    let t = tick as f32 * 0.1;
    PlayerState {
        x: t.sin() * 5.0,
        y: 0.0,
        z: t.cos() * 5.0,
        rot_x: 0.0,
        rot_y: t,
        rot_z: 0.0,
        rot_w: 1.0,
    }
}

async fn run_server(addr: SocketAddr) {
    let (chain, key) = self_signed_localhost_cert();

    // `ServerBuilder`/`with_certificate` -- exact shape confirmed from
    // web-transport-quinn's own `examples/echo-server.rs` (downloaded and
    // read directly, see the integration test's doc comment).
    let mut server = web_transport_quinn::ServerBuilder::new()
        .with_addr(addr)
        .with_certificate(chain, key)
        .expect("failed to build the WebTransport server");

    // `Server` derefs to `quinn::Endpoint`, so this is the real bound
    // address (relevant when `addr`'s port is 0).
    let bound = server
        .local_addr()
        .expect("server has no local address after binding");
    mid_info!(Tier::Mid, "listening on {}", bound);

    let handle = tokio::runtime::Handle::current();

    while let Some(request) = server.accept().await {
        let handle = handle.clone();
        tokio::spawn(async move {
            mid_info!(Tier::Mid, "incoming request: {}", request.url);

            let session = match request.respond(ConnectResponse::OK).await {
                Ok(session) => session,
                Err(err) => {
                    mid_warn!(Tier::Mid, "failed to accept session: {}", err);
                    return;
                }
            };
            mid_info!(Tier::Mid, "session accepted");

            let transport = QuinnTransport::new(session, &handle);
            run_connection(transport, "server", 0).await;
        });
    }
}

async fn run_client(addr: &str, player_id: u32) {
    // `dangerous().with_no_certificate_verification()` -- exact shape
    // confirmed from `examples/echo-client.rs`'s `--tls-disable-verify`
    // path. Appropriate here specifically because this demo client has no
    // real CA to check the demo server's self-signed cert against -- not
    // a pattern to copy into anything that isn't a local demo.
    let client = web_transport_quinn::ClientBuilder::new()
        .dangerous()
        .with_no_certificate_verification()
        .expect("failed to build the certificate-skipping demo client");

    let url: url::Url = format!("https://{addr}")
        .parse()
        .expect("server-addr must parse as host:port, e.g. 127.0.0.1:5000");

    mid_info!(Tier::Mid, "connecting to {}", url);
    let session = client
        .connect(ConnectRequest::new(url))
        .await
        .expect("failed to connect");
    mid_info!(Tier::Mid, "connected");

    let handle = tokio::runtime::Handle::current();
    let transport = QuinnTransport::new(session, &handle);
    run_connection(transport, "client", player_id).await;
}

/// Shared tick loop for both roles: announce with one `PlayerEvent`, then
/// every tick send a synthetic `PlayerState` and log anything received.
/// The two roles differ only in what they announce with and which log
/// prefix they use -- everything else about exercising the
/// `Connection<QuinnTransport>` API is identical, so it isn't duplicated.
async fn run_connection(transport: QuinnTransport, role: &'static str, player_id: u32) {
    let mut conn = Connection::new(transport);
    let mut interval = tokio::time::interval(TICK);
    let mut tick: u64 = 0;

    if let Err(err) = conn.send_player_event(&PlayerEvent {
        player_id: PlayerId(player_id),
        event: "hello".to_string(),
        payload: role.to_string(),
    }) {
        mid_warn!(Tier::Mid, "[{}] failed to send hello event: {:?}", role, err);
    }

    loop {
        interval.tick().await;

        if !conn.is_connected() {
            mid_info!(Tier::Mid, "[{}] connection closed", role);
            return;
        }

        if let Err(err) = conn.send_player_state(&synthetic_player_state(tick)) {
            mid_warn!(Tier::Mid, "[{}] send_player_state failed: {:?}", role, err);
        }
        tick += 1;

        match conn.poll() {
            Ok(events) => {
                for event in events {
                    match event {
                        ConnectionEvent::PlayerState(state) => {
                            mid_kvinfo!(Tier::Mid, "recv PlayerState";
                                "role" => role, "x" => state.x, "y" => state.y, "z" => state.z);
                        }
                        ConnectionEvent::PlayerEvent(event) => {
                            mid_info!(
                                Tier::Mid,
                                "[{}] recv PlayerEvent: {} {}",
                                role,
                                event.event,
                                event.payload
                            );
                        }
                    }
                }
            }
            Err(err) => {
                mid_warn!(Tier::Mid, "[{}] poll failed: {:?}", role, err);
                return;
            }
        }
    }
}
