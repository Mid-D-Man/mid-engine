//! mid-net-wire — packet payload codec and sequence/ack arithmetic.
//!
//! The lowest layer: no transport, no reliability protocol, no platform
//! assumptions at all -- just "how do these bytes turn into a
//! PlayerState/PlayerEvent and back" (packet.rs) and "which of two
//! sequence numbers is more recent, accounting for wraparound"
//! (sequence.rs) and "how do fields pack into bits and back, with floats
//! quantised to a range" (bits.rs: the base of every typed message) and
//! "how does a struct describe its own wire form once, for writing and
//! reading alike" (stream.rs: `NetSerialize`, `NetPod`).
//! Zero dependencies, deliberately reusable on its own --
//! e.g. mid-ecs's replication/sync module may want just the wire codec
//! without pulling in reliable.rs's retransmit machinery.

pub mod bits;
pub mod frame;
pub mod packet;
pub mod poly;
pub mod sequence;
pub mod stream;

pub use bits::{quantization_step, BitError, BitReader, BitWriter};
pub use frame::{
    ChannelError, ChannelKind, ChannelTable, DatagramBuilder, DatagramReceiver, PushError,
    ReceiveReport, StreamError, StreamReader, StreamWriter, MAX_MESSAGES_PER_DATAGRAM,
};
pub use packet::{
    DecodeError, Packet, PacketKind, PlayerEvent, PlayerId, PlayerState, PLAYER_STATE_WIRE_SIZE,
};
pub use poly::{
    Dispatcher, MessageRegistry, NetMessage, NetMessageType, RawMessage, RegisterError,
    RAW_MESSAGE_MAX,
};
pub use sequence::{is_acked, AckTracker, Sequence};
pub use stream::{
    net_read, net_round_trip, net_vec, net_write, net_write_ref, read_pod, write_pod, BitStream,
    BitStreamExt, NetPod, NetSerialize, SInt, UInt,
};
