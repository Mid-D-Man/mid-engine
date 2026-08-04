//! mid-net-wire — packet payload codec and sequence/ack arithmetic.
//!
//! The lowest layer: no transport, no reliability protocol, no platform
//! assumptions at all -- just "how do these bytes turn into a
//! PlayerState/PlayerEvent and back" (packet.rs) and "which of two
//! sequence numbers is more recent, accounting for wraparound"
//! (sequence.rs). Zero dependencies, deliberately reusable on its own --
//! e.g. mid-ecs's replication/sync module may want just the wire codec
//! without pulling in reliable.rs's retransmit machinery.

pub mod packet;
pub mod sequence;

pub use packet::{DecodeError, Packet, PacketKind, PlayerEvent, PlayerId, PlayerState, PLAYER_STATE_WIRE_SIZE};
pub use sequence::{is_acked, AckTracker, Sequence};
