//! The pluggable-transport boundary — `packet.rs`/`sequence.rs`/`reliable.rs`
//! never depend on a concrete transport, only on this trait. Same idea as
//! Unity Netcode's `NetworkTransport` abstraction (UTP vs. WebSocket vs.
//! third-party swapped underneath one `NetworkManager`), for the same
//! reason: we already need two backends no matter what (native QUIC via
//! `quinn`/`web-transport-quinn`, browser via `web-transport-wasm`), so a
//! third — Steam Sockets, a custom relay, whatever — is just another impl
//! of this trait, not a redesign.
//!
//! **Deliberately synchronous and poll-based, not `async fn`.** An async
//! trait spanning native and `wasm32` hits Rust's `!Send`-on-wasm wall —
//! checked against how the `web-transport` crate itself handles exactly
//! this: it doesn't unify native/wasm behind one trait either, it swaps
//! concrete types per target. Each implementation is free to use async,
//! threads, or JS callbacks internally; this trait only asks for a
//! queue-drain once a tick, matching the same "no runtime baked into the
//! protocol logic" principle `reliable.rs` already follows.
//!
//! Two logical channels, matching `docs/mid-net.md`: unreliable datagrams
//! (`PlayerState`, loss is fine) and reliable streams (`PlayerEvent` —
//! QUIC's own stream reliability now owns this, not `reliable.rs`'s
//! `RetransmitBuffer`; see docs/mid-net.md "Reliability mechanism").
//! `stream_id` lets a caller run more than one logical reliable stream
//! over one transport without them head-of-line-blocking each other.

use std::collections::VecDeque;

pub trait Transport {
    type Error: std::fmt::Debug;

    /// Send an unreliable, unordered datagram. Best-effort — matches
    /// `PlayerState`'s channel; loss is expected and fine.
    fn send_datagram(&mut self, bytes: &[u8]) -> Result<(), Self::Error>;

    /// Drain one received datagram, if any are queued. Non-blocking —
    /// returns `Ok(None)` rather than waiting when nothing's arrived.
    fn poll_datagram(&mut self) -> Result<Option<Vec<u8>>, Self::Error>;

    /// Write `bytes` to the reliable, ordered stream identified by
    /// `stream_id` (opened on first use). Matches `PlayerEvent`'s channel.
    fn send_reliable(&mut self, stream_id: u32, bytes: &[u8]) -> Result<(), Self::Error>;

    /// Drain one received reliable message, if any are queued. Non-blocking.
    fn poll_reliable(&mut self) -> Result<Option<(u32, Vec<u8>)>, Self::Error>;

    /// True once the underlying connection handshake has completed.
    fn is_connected(&self) -> bool;
}

/// In-memory loopback pair — no sockets, no async, no platform code.
/// Two of these, cross-wired via `pump`, run a client+server in one
/// process for tests. Also doubles as a sanity check that `Transport`
/// doesn't secretly assume a real socket underneath — if a trait method
/// can't be satisfied by "push onto a `VecDeque`", it's the wrong method.
pub struct LoopbackTransport {
    outgoing_datagrams: VecDeque<Vec<u8>>,
    outgoing_reliable: VecDeque<(u32, Vec<u8>)>,
    incoming_datagrams: VecDeque<Vec<u8>>,
    incoming_reliable: VecDeque<(u32, Vec<u8>)>,
}

impl LoopbackTransport {
    pub fn new_pair() -> (Self, Self) {
        (
            LoopbackTransport {
                outgoing_datagrams: VecDeque::new(),
                outgoing_reliable: VecDeque::new(),
                incoming_datagrams: VecDeque::new(),
                incoming_reliable: VecDeque::new(),
            },
            LoopbackTransport {
                outgoing_datagrams: VecDeque::new(),
                outgoing_reliable: VecDeque::new(),
                incoming_datagrams: VecDeque::new(),
                incoming_reliable: VecDeque::new(),
            },
        )
    }

    /// Test-only: move each side's outgoing queue into the other's
    /// incoming queue. Stands in for "a tick of real network I/O".
    pub fn pump(a: &mut Self, b: &mut Self) {
        while let Some(d) = a.outgoing_datagrams.pop_front() {
            b.incoming_datagrams.push_back(d);
        }
        while let Some(d) = b.outgoing_datagrams.pop_front() {
            a.incoming_datagrams.push_back(d);
        }
        while let Some(r) = a.outgoing_reliable.pop_front() {
            b.incoming_reliable.push_back(r);
        }
        while let Some(r) = b.outgoing_reliable.pop_front() {
            a.incoming_reliable.push_back(r);
        }
    }
}

impl Transport for LoopbackTransport {
    type Error = std::convert::Infallible;

    fn send_datagram(&mut self, bytes: &[u8]) -> Result<(), Self::Error> {
        self.outgoing_datagrams.push_back(bytes.to_vec());
        Ok(())
    }

    fn poll_datagram(&mut self) -> Result<Option<Vec<u8>>, Self::Error> {
        Ok(self.incoming_datagrams.pop_front())
    }

    fn send_reliable(&mut self, stream_id: u32, bytes: &[u8]) -> Result<(), Self::Error> {
        self.outgoing_reliable.push_back((stream_id, bytes.to_vec()));
        Ok(())
    }

    fn poll_reliable(&mut self) -> Result<Option<(u32, Vec<u8>)>, Self::Error> {
        Ok(self.incoming_reliable.pop_front())
    }

    fn is_connected(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loopback_pair_delivers_datagrams_after_pump() {
        let (mut client, mut server) = LoopbackTransport::new_pair();
        client.send_datagram(&[1, 2, 3]).unwrap();
        assert_eq!(server.poll_datagram().unwrap(), None, "not delivered until pumped");
        LoopbackTransport::pump(&mut client, &mut server);
        assert_eq!(server.poll_datagram().unwrap(), Some(vec![1, 2, 3]));
        assert_eq!(server.poll_datagram().unwrap(), None, "drained, nothing left");
    }

    #[test]
    fn loopback_pair_delivers_reliable_streams_with_id() {
        let (mut client, mut server) = LoopbackTransport::new_pair();
        client.send_reliable(7, b"pickup:item42").unwrap();
        LoopbackTransport::pump(&mut client, &mut server);
        assert_eq!(server.poll_reliable().unwrap(), Some((7, b"pickup:item42".to_vec())));
    }

    #[test]
    fn generic_fn_over_the_trait_compiles_and_works() {
        // Proves the trait is genuinely usable as an abstraction boundary,
        // not just implementable -- real socket.rs/reliable.rs code would
        // be written against `T: Transport`, same as this.
        fn send_position<T: Transport>(t: &mut T, bytes: &[u8]) -> Result<(), T::Error> {
            t.send_datagram(bytes)
        }
        let (mut client, mut server) = LoopbackTransport::new_pair();
        send_position(&mut client, &[9, 9]).unwrap();
        LoopbackTransport::pump(&mut client, &mut server);
        assert_eq!(server.poll_datagram().unwrap(), Some(vec![9, 9]));
    }

    #[test]
    fn both_channels_independent() {
        let (mut client, mut server) = LoopbackTransport::new_pair();
        client.send_datagram(&[1]).unwrap();
        client.send_reliable(1, &[2]).unwrap();
        LoopbackTransport::pump(&mut client, &mut server);
        // Order of draining one channel doesn't affect the other.
        assert_eq!(server.poll_reliable().unwrap(), Some((1, vec![2])));
        assert_eq!(server.poll_datagram().unwrap(), Some(vec![1]));
    }
}
