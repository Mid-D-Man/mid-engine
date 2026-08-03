//! Stands in for "a studio using mid-engine writes their own transport."
//! Doesn't touch mid-net's source at all -- only depends on
//! `mid_net_transport::Transport` (the trait) and `mid_net::Connection`
//! (the composed API), same as any real downstream consumer would.
//!
//! Deliberately different in behavior from `LoopbackTransport`, not just
//! a rename of it: simulates datagram loss (a deterministic drop
//! pattern, not real randomness, so the test stays reproducible) while
//! still guaranteeing delivery on the reliable channel, same as the
//! trait contract requires.

use mid_net_transport::Transport;
use std::collections::VecDeque;

pub struct FlakyTransport {
    outgoing_datagrams: VecDeque<Vec<u8>>,
    outgoing_reliable: VecDeque<(u32, Vec<u8>)>,
    incoming_datagrams: VecDeque<Vec<u8>>,
    incoming_reliable: VecDeque<(u32, Vec<u8>)>,
    drop_every_nth: u32,
    datagram_counter: u32,
}

impl FlakyTransport {
    pub fn new_pair(drop_every_nth: u32) -> (Self, Self) {
        let make = || FlakyTransport {
            outgoing_datagrams: VecDeque::new(),
            outgoing_reliable: VecDeque::new(),
            incoming_datagrams: VecDeque::new(),
            incoming_reliable: VecDeque::new(),
            drop_every_nth,
            datagram_counter: 0,
        };
        (make(), make())
    }

    /// Moves everything queued on each side into the other's inbox,
    /// dropping datagrams (never reliable traffic) per the configured
    /// rate. Called on the raw transports via `Connection::transport_mut()`
    /// -- the accessor that turned out to be necessary once this was
    /// actually written as a real external crate.
    pub fn pump(a: &mut Self, b: &mut Self) {
        Self::pump_one_direction(&mut a.outgoing_datagrams, &mut a.outgoing_reliable, &mut b.incoming_datagrams, &mut b.incoming_reliable, &mut a.datagram_counter, a.drop_every_nth);
        Self::pump_one_direction(&mut b.outgoing_datagrams, &mut b.outgoing_reliable, &mut a.incoming_datagrams, &mut a.incoming_reliable, &mut b.datagram_counter, b.drop_every_nth);
    }

    fn pump_one_direction(
        out_dg: &mut VecDeque<Vec<u8>>,
        out_rel: &mut VecDeque<(u32, Vec<u8>)>,
        in_dg: &mut VecDeque<Vec<u8>>,
        in_rel: &mut VecDeque<(u32, Vec<u8>)>,
        counter: &mut u32,
        drop_every_nth: u32,
    ) {
        while let Some(d) = out_dg.pop_front() {
            *counter += 1;
            let dropped = drop_every_nth != 0 && *counter % drop_every_nth == 0;
            if !dropped {
                in_dg.push_back(d);
            }
        }
        while let Some(r) = out_rel.pop_front() {
            in_rel.push_back(r);
        }
    }
}

impl Transport for FlakyTransport {
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
    use mid_net::{Connection, ConnectionEvent, PlayerEvent, PlayerId, PlayerState};

    #[test]
    fn custom_transport_works_through_connection_no_mid_net_edits_needed() {
        let (t_a, t_b) = FlakyTransport::new_pair(0); // never drop, for this test
        let mut a = Connection::new(t_a);
        let mut b = Connection::new(t_b);

        let state = PlayerState { x: 5.0, ..PlayerState::default() };
        a.send_player_state(&state).unwrap();
        assert!(b.poll().unwrap().is_empty(), "nothing delivered before a pump");

        FlakyTransport::pump(a.transport_mut(), b.transport_mut());

        assert_eq!(b.poll().unwrap(), vec![ConnectionEvent::PlayerState(state)]);
    }

    #[test]
    fn simulated_loss_drops_some_datagrams_reliable_channel_unaffected() {
        let (t_a, t_b) = FlakyTransport::new_pair(2); // drop every 2nd datagram
        let mut a = Connection::new(t_a);
        let mut b = Connection::new(t_b);

        for i in 0..10 {
            a.send_player_state(&PlayerState { x: i as f32, ..PlayerState::default() }).unwrap();
        }
        a.send_player_event(&PlayerEvent { player_id: PlayerId(1), event: "join".into(), payload: String::new() }).unwrap();

        FlakyTransport::pump(a.transport_mut(), b.transport_mut());
        let events = b.poll().unwrap();

        let state_count = events.iter().filter(|e| matches!(e, ConnectionEvent::PlayerState(_))).count();
        let event_count = events.iter().filter(|e| matches!(e, ConnectionEvent::PlayerEvent(_))).count();

        assert_eq!(state_count, 5, "half of 10 datagrams should have been dropped");
        assert_eq!(event_count, 1, "reliable channel must never drop, regardless of the flaky datagram setting");
    }

    #[test]
    fn bidirectional_traffic_through_custom_transport() {
        let (t_a, t_b) = FlakyTransport::new_pair(0);
        let mut a = Connection::new(t_a);
        let mut b = Connection::new(t_b);

        a.send_player_state(&PlayerState { x: 1.0, ..PlayerState::default() }).unwrap();
        b.send_player_state(&PlayerState { x: 2.0, ..PlayerState::default() }).unwrap();
        FlakyTransport::pump(a.transport_mut(), b.transport_mut());

        assert_eq!(a.poll().unwrap(), vec![ConnectionEvent::PlayerState(PlayerState { x: 2.0, ..PlayerState::default() })]);
        assert_eq!(b.poll().unwrap(), vec![ConnectionEvent::PlayerState(PlayerState { x: 1.0, ..PlayerState::default() })]);
    }
  }
