//! Selective-reliability layer on top of UDP.
//!
//! Unreliable: position / rotation / animation (can drop freely)
//! Reliable  : join, pickup, damage (sequence numbers + ACK + retransmit)
//!
//! No head-of-line blocking — a lost position packet is just stale.
// Auto-generated stub
