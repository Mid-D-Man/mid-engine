//! mid-net — Reliable UDP + DixScript packet serialization
//!
//! Unreliable channel : position, rotation, animation state (128 Hz)
//! Reliable channel   : discrete events — join, pickup, damage
//!
//! Packet *shapes* are defined in .mdix files under `packets/`.
//! Wire encoding is hand-rolled (`packet.rs`) — explicit little-endian,
//! zero external dependencies. Not bincode/serde: see docs/mid-net.md,
//! "Dependency philosophy", for why.
//!
//! Reliable-channel sequence numbers and ack-bitfield tracking live in
//! `sequence.rs`, tested against the gafferongames.com reference design
//! docs/mid-net.md cites. `reliable.rs` (still a stub) is the layer that
//! turns this into an actual send/receive protocol — retransmit buffer,
//! RTT-based timeout, framing `packet.rs` payloads with a kind tag and
//! sequence number.
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
// gap rather than getting quietly forgotten; packet.rs itself has no
// dependency on socket and compiles/tests fine standalone.
pub use socket::MidSocket;

pub use packet::{DecodeError, Packet, PacketKind, PlayerEvent, PlayerId, PlayerState};
pub use sequence::{is_acked, AckTracker, Sequence};
