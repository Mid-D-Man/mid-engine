//! mid-net — Reliable UDP netcode, hand-rolled wire format
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
//! cites. `reliable.rs` builds the actual send/receive protocol on top:
//! frame headers (kind + sequence, or kind + sequence + piggybacked ack
//! for the reliable channel), an RTT-based retransmit buffer, and
//! nothing else — no sockets, no wall clock, so it runs identically on
//! native and `wasm32` and stays FFI-safe (plain data in, plain data
//! out). `socket.rs` (still a stub) is where the actual transport
//! per-platform lives — UDP natively, WebTransport datagrams in-browser.
//!
//! 7.8ms budget per tick at 128 Hz. Design the packet budget early.

pub mod socket;
pub mod packet;
pub mod reliable;
pub mod sequence;
pub mod ffi;

// socket.rs is still a 2-line stub — this re-export doesn't resolve yet
// and the crate won't build as a whole until MidSocket exists. Left in
// deliberately (not commented out) so it stays visible as the next real
// gap rather than getting quietly forgotten; packet.rs/sequence.rs/
// reliable.rs have no dependency on socket and compile/test fine
// standalone (verified as a group, not just individually).
pub use socket::MidSocket;

pub use packet::{DecodeError, Packet, PacketKind, PlayerEvent, PlayerId, PlayerState};
pub use sequence::{is_acked, AckTracker, Sequence};
pub use reliable::{
    decode_reliable_frame, decode_unreliable_frame, encode_reliable_frame, encode_unreliable_frame,
    FrameError, ReliableHeader, RetransmitBuffer, RttEstimator, Timestamp, UnreliableHeader,
    RELIABLE_HEADER_SIZE, UNRELIABLE_HEADER_SIZE,
};
