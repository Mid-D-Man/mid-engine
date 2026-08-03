//! `Connection` — the composed, usable API: `PlayerState`/`PlayerEvent`
//! in and out of a `Transport`, nothing more.
//!
//! Deliberately does NOT use `mid-net-reliable`'s `RetransmitBuffer`/
//! `RttEstimator`/reliable-frame functions. Worth being explicit about
//! why, since it looks like they should be used here and aren't: the
//! real `Transport` trait (`mid-net-transport`) bakes real reliability
//! into `send_reliable`/`poll_reliable` themselves — there's no
//! `has_native_reliability()` escape hatch, every impl is required to
//! actually deliver reliably. So there's nothing here for a retransmit
//! buffer to do. Those pieces stay correct and tested; they're the
//! implementation a *future raw-UDP* `Transport` impl would use
//! internally to satisfy `send_reliable`'s contract itself, one layer
//! down from here, not something this layer calls.
//!
//! What this layer still needs from `mid-net-reliable`: just
//! `encode_unreliable_frame`/`decode_unreliable_frame` — the kind+sequence
//! framing `PlayerState` needs for staleness detection on the datagram
//! channel. The reliable channel needs no sequence number at all
//! (the transport already guarantees order), just a one-byte kind tag
//! for symmetry / future packet kinds.

use mid_net_reliable::{decode_unreliable_frame, encode_unreliable_frame};
use mid_net_transport::Transport;
use mid_net_wire::{Packet, PacketKind, PlayerEvent, PlayerState, Sequence};

/// Single logical reliable stream for now -- every `PlayerEvent` goes
/// through this one. Per-entity/per-event-type streams (avoiding
/// head-of-line blocking between unrelated events, per the
/// Unity/Unreal research in docs/mid-net.md) is real future work, not
/// done here -- `Transport::send_reliable`'s `stream_id` parameter
/// exists specifically so that's an additive change later, not a
/// redesign.
const EVENT_STREAM_ID: u32 = 0;

#[derive(Debug, Clone, PartialEq)]
pub enum ConnectionEvent {
    PlayerState(PlayerState),
    PlayerEvent(PlayerEvent),
}

/// Wraps one `Transport` with the `PlayerState`/`PlayerEvent` framing
/// and staleness bookkeeping. One `Connection` per remote peer.
pub struct Connection<T: Transport> {
    transport: T,
    next_unreliable_seq: Sequence,
    last_seen_unreliable_seq: Option<Sequence>,
}

impl<T: Transport> Connection<T> {
    pub fn new(transport: T) -> Self {
        Connection { transport, next_unreliable_seq: Sequence(0), last_seen_unreliable_seq: None }
    }

    /// Escape hatch to the wrapped transport. Needed by anyone
    /// implementing their own `Transport` whose I/O needs an explicit
    /// external step to actually move bytes — a loopback-style test
    /// double being the obvious case (real socket-backed transports
    /// don't need this; the OS delivers on its own). Found this was
    /// missing by actually trying to write a third-party transport and
    /// hitting the private field, not by reasoning about it in the
    /// abstract — fixed rather than left as a known gap.
    pub fn transport_mut(&mut self) -> &mut T {
        &mut self.transport
    }

    pub fn transport(&self) -> &T {
        &self.transport
    }

    pub fn is_connected(&self) -> bool {
        self.transport.is_connected()
    }

    /// Unreliable, 128 Hz channel. Loss is fine; framed with a sequence
    /// number purely so the receiving side can drop anything that
    /// arrives out of order relative to what it's already applied.
    pub fn send_player_state(&mut self, state: &PlayerState) -> Result<(), T::Error> {
        let seq = self.next_unreliable_seq;
        self.next_unreliable_seq = seq.wrapping_next();

        let mut payload = Vec::new();
        state.encode(&mut payload);

        let mut framed = Vec::new();
        encode_unreliable_frame(PacketKind::PlayerState, seq, &payload, &mut framed);

        self.transport.send_datagram(&framed)
    }

    /// Reliable channel. No sequence number needed here -- ordering and
    /// delivery are the transport's job, not this layer's.
    pub fn send_player_event(&mut self, event: &PlayerEvent) -> Result<(), T::Error> {
        let mut framed = vec![PacketKind::PlayerEvent as u8];
        event.encode(&mut framed);
        self.transport.send_reliable(EVENT_STREAM_ID, &framed)
    }

    /// Drains everything currently buffered on both channels and returns
    /// fully-decoded events. Malformed or stale bytes are silently
    /// dropped rather than surfaced as an error -- on the unreliable
    /// channel that's exactly the behavior the channel is supposed to
    /// have (loss is fine); on the reliable channel it would mean either
    /// a version mismatch or real corruption, neither of which this
    /// layer has enough context to usefully report on, so it drops
    /// rather than guesses.
    pub fn poll(&mut self) -> Result<Vec<ConnectionEvent>, T::Error> {
        let mut events = Vec::new();

        while let Some(bytes) = self.transport.poll_datagram()? {
            let Ok((header, payload)) = decode_unreliable_frame(&bytes) else { continue };
            if header.kind != PacketKind::PlayerState {
                continue;
            }
            let is_stale = self
                .last_seen_unreliable_seq
                .is_some_and(|last| !header.sequence.is_more_recent_than(last));
            if is_stale {
                continue;
            }
            self.last_seen_unreliable_seq = Some(header.sequence);
            if let Ok(state) = PlayerState::decode(payload) {
                events.push(ConnectionEvent::PlayerState(state));
            }
        }

        while let Some((_stream_id, bytes)) = self.transport.poll_reliable()? {
            let Some((&kind_byte, payload)) = bytes.split_first() else { continue };
            if PacketKind::from_u8(kind_byte) != Some(PacketKind::PlayerEvent) {
                continue;
            }
            if let Ok(event) = PlayerEvent::decode(payload) {
                events.push(ConnectionEvent::PlayerEvent(event));
            }
        }

        Ok(events)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mid_net_transport::LoopbackTransport;

    #[test]
    fn player_state_round_trips_through_a_connection_pair() {
        let (t_a, t_b) = LoopbackTransport::new_pair();
        let mut a = Connection::new(t_a);
        let mut b = Connection::new(t_b);

        let state = PlayerState { x: 1.0, y: 2.0, z: 3.0, rot_x: 0.0, rot_y: 0.0, rot_z: 0.0, rot_w: 1.0 };
        a.send_player_state(&state).unwrap();

        assert!(b.poll().unwrap().is_empty(), "nothing delivered before a pump");
        // reach into the transports directly to pump -- Connection doesn't
        // own tick/pump, that's the transport's concern
        pump(&mut a, &mut b);

        assert_eq!(b.poll().unwrap(), vec![ConnectionEvent::PlayerState(state)]);
    }

    #[test]
    fn player_event_round_trips_through_a_connection_pair() {
        let (t_a, t_b) = LoopbackTransport::new_pair();
        let mut a = Connection::new(t_a);
        let mut b = Connection::new(t_b);

        let event = PlayerEvent { player_id: mid_net_wire::PlayerId(7), event: "pickup".into(), payload: "item_id=3".into() };
        a.send_player_event(&event).unwrap();
        pump(&mut a, &mut b);

        assert_eq!(b.poll().unwrap(), vec![ConnectionEvent::PlayerEvent(event)]);
    }

    #[test]
    fn both_channels_arrive_together() {
        let (t_a, t_b) = LoopbackTransport::new_pair();
        let mut a = Connection::new(t_a);
        let mut b = Connection::new(t_b);

        a.send_player_state(&PlayerState::default()).unwrap();
        a.send_player_event(&PlayerEvent::default()).unwrap();
        pump(&mut a, &mut b);

        let events = b.poll().unwrap();
        assert_eq!(events.len(), 2);
        assert!(events.contains(&ConnectionEvent::PlayerState(PlayerState::default())));
        assert!(events.contains(&ConnectionEvent::PlayerEvent(PlayerEvent::default())));
    }

    #[test]
    fn stale_player_state_is_dropped_not_applied() {
        // Bypasses Connection::send_player_state to inject frames with an
        // explicit, out-of-natural-order sequence -- LoopbackTransport's
        // FIFO pump can't produce real reordering on its own, so this is
        // the only way to actually exercise the staleness path. Uses
        // `transport_mut()` rather than reaching into the private field
        // directly -- same accessor any real external Transport
        // implementer would need, exercised here for consistency.
        let (mut raw_a, t_b) = LoopbackTransport::new_pair();
        let mut b = Connection::new(t_b);

        let newer = PlayerState { x: 2.0, ..PlayerState::default() };
        let older = PlayerState { x: 1.0, ..PlayerState::default() };

        let mut payload = Vec::new();
        newer.encode(&mut payload);
        let mut frame = Vec::new();
        encode_unreliable_frame(PacketKind::PlayerState, Sequence(10), &payload, &mut frame);
        raw_a.send_datagram(&frame).unwrap();

        let mut payload2 = Vec::new();
        older.encode(&mut payload2);
        let mut frame2 = Vec::new();
        encode_unreliable_frame(PacketKind::PlayerState, Sequence(3), &payload2, &mut frame2);
        raw_a.send_datagram(&frame2).unwrap();

        LoopbackTransport::pump(&mut raw_a, b.transport_mut());

        // Both frames arrive in the same poll batch (10 first, then the
        // stale 3) -- only the newer one should survive.
        assert_eq!(b.poll().unwrap(), vec![ConnectionEvent::PlayerState(newer)]);
    }

    #[test]
    fn malformed_bytes_on_either_channel_are_dropped_not_panicking() {
        let (mut raw_a, t_b) = LoopbackTransport::new_pair();
        let mut b = Connection::new(t_b);

        raw_a.send_datagram(&[0xFF, 0xFF]).unwrap(); // too short to even have a valid header
        raw_a.send_reliable(EVENT_STREAM_ID, &[]).unwrap(); // empty -- no kind byte at all
        LoopbackTransport::pump(&mut raw_a, b.transport_mut());

        assert_eq!(b.poll().unwrap(), Vec::new());
    }

    fn pump(a: &mut Connection<LoopbackTransport>, b: &mut Connection<LoopbackTransport>) {
        LoopbackTransport::pump(a.transport_mut(), b.transport_mut());
    }
            }
