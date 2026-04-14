//! mid-net — Reliable UDP + DixScript packet serialization
//!
//! Unreliable channel : position, rotation, animation state (128 Hz)
//! Reliable channel   : discrete events — join, pickup, damage
//!
//! Packet *shapes* are defined in .mdix files.
//! Wire encoding uses bincode for zero-copy byte layout.
//!
//! 7.8ms budget per tick at 128 Hz. Design the packet budget early.

pub mod socket;
pub mod packet;
pub mod reliable;
pub mod sequence;
pub mod ffi;

pub use socket::MidSocket;
pub use packet::Packet;
