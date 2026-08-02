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
//!   Depends on `mid-net-wire` only.
//! - `mid-net` (this crate) — facade: re-exports all three, will own
//!   `ffi.rs`'s C ABI surface. Concrete transport backends
//!   (`mid-net-transport-quinn`, `mid-net-transport-wasm`) are planned as
//!   sibling subfolder crates, not yet built — see docs/mid-net.md.

pub mod ffi;

pub use mid_net_wire::{DecodeError, Packet, PacketKind, PlayerEvent, PlayerId, PlayerState, is_acked, AckTracker, Sequence};
pub use mid_net_reliable::{
    decode_reliable_frame, decode_unreliable_frame, encode_reliable_frame, encode_unreliable_frame,
    FrameError, ReliableHeader, RetransmitBuffer, RttEstimator, Timestamp, UnreliableHeader,
    RELIABLE_HEADER_SIZE, UNRELIABLE_HEADER_SIZE,
};
pub use mid_net_transport::{LoopbackTransport, Transport};
