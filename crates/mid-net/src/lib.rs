//! mid-net — Reliable UDP-class netcode, hand-rolled wire format
//!
//! Unreliable channel : position, rotation, animation state (128 Hz)
//! Reliable channel   : discrete events — join, pickup, damage
//!
//! Packet *shapes* are authored as reference schema in `.mdix` files
//! under `packets/` (human-readable only — nothing in this crate parses
//! them; DixScript is intentionally not a dependency here, core crates
//! never carry it, see docs/architecture.md "Dependency philosophy").
//! Wire encoding is hand-rolled (`packet.rs`) — explicit little-endian,
//! zero external dependencies.
//!
//! Sequence numbers and ack-bitfield tracking live in `sequence.rs`,
//! tested against the gafferongames.com reference design docs/mid-net.md
//! cites. `reliable.rs` builds frame headers, an RTT estimator, and a
//! retransmit buffer on top — kept as the correct, tested implementation
//! for any transport without native reliability, though the chosen
//! transport (QUIC) means `PlayerEvent` doesn't need it in practice; see
//! docs/mid-net.md "Reliability mechanism".
//!
//! `transport.rs` is the pluggable-backend boundary (a `Transport` trait,
//! verified against Unity Netcode for Entities' `INetworkInterface` +
//! `DefaultDriverConstructor.cs` pattern) that `packet.rs`/`sequence.rs`/
//! `reliable.rs` never bypass to talk to a socket directly.
//! `LoopbackTransport` is real and tested today; native (`quinn`) and
//! browser (`web-transport-wasm`) backends belong in `socket.rs`, not
//! yet built — quinn needs a newer Rust than this sandbox can compile,
//! so that part is static-analysis-only for now. See docs/mid-net.md
//! "Transport" and "Mobile" for the full picture.
//!
//! 7.8ms budget per tick at 128 Hz. Design the packet budget early.

pub mod socket;
pub mod packet;
pub mod reliable;
pub mod sequence;
pub mod transport;
pub mod ffi;

pub use packet::{DecodeError, Packet, PacketKind, PlayerEvent, PlayerId, PlayerState};
pub use sequence::{is_acked, AckTracker, Sequence};
pub use reliable::{
    decode_reliable_frame, decode_unreliable_frame, encode_reliable_frame, encode_unreliable_frame,
    FrameError, ReliableHeader, RetransmitBuffer, RttEstimator, Timestamp, UnreliableHeader,
    RELIABLE_HEADER_SIZE, UNRELIABLE_HEADER_SIZE,
};
pub use transport::{LoopbackTransport, Transport};
