//! mid-net — Reliable UDP-class netcode, hand-rolled wire format
//!
//! Restructured into subfolder crates (mirroring naia's own
//! socket/{client,server,shared} split, checked directly against its
//! Cargo.toml layout): each concern that could stand alone now does,
//! so a consumer only pulls in what it needs and a heavy
//! platform-specific transport backend never contaminates the
//! zero-dependency protocol layer's dependency tree.
//!
//! - `mid-net-wire`      — packet codec + sequence/ack arithmetic. Zero deps.
//! - `mid-net-transport` — the `Transport` trait + `LoopbackTransport`. Zero deps.
//! - `mid-net-reliable`  — frame headers, RTT estimator, retransmit buffer.
//!   Depends on `mid-net-wire` only. Not used by `connection.rs` below —
//!   `Transport::send_reliable` already guarantees real delivery, so
//!   there's no retransmit buffer for this layer to run. Kept as the
//!   implementation a future raw-UDP `Transport` impl would need
//!   internally to satisfy that guarantee itself; see `connection.rs`'s
//!   module doc for the full reasoning.
//! - `mid-net` (this crate) — facade: `connection.rs` composes the three
//!   crates above into `Connection<T: Transport>`, the actual
//!   `send_player_state`/`send_player_event`/`poll` API a game loop uses;
//!   also re-exports everything and will own `ffi.rs`'s C ABI surface.
//!   Concrete transport backends (`mid-net-transport-quinn`,
//!   `mid-net-transport-wasm`) are planned as sibling subfolder crates,
//!   not yet built — see docs/mid-net.md.

pub mod ffi;
pub mod connection;

pub use connection::{Connection, ConnectionEvent};
pub use mid_net_wire::{DecodeError, Packet, PacketKind, PlayerEvent, PlayerId, PlayerState, is_acked, AckTracker, Sequence};
pub use mid_net_reliable::{
    decode_reliable_frame, decode_unreliable_frame, encode_reliable_frame, encode_unreliable_frame,
    FrameError, ReliableHeader, RetransmitBuffer, RttEstimator, Timestamp, UnreliableHeader,
    RELIABLE_HEADER_SIZE, UNRELIABLE_HEADER_SIZE,
};
pub use mid_net_transport::{LoopbackTransport, Transport};
