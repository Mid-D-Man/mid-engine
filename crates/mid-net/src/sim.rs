//! A simulated network: a pair of [`Transport`]s joined by a link that loses, duplicates, delays and
//! reorders datagrams, cuts reliable bytes into pieces, and goes silent on command. Everything is driven
//! by a clock the test sets and a seeded random generator, so a run is the same run every time.
//!
//! It exists so that the driver and everything above it can be tested against a bad network without a
//! socket, a thread or real time, and it is public because a game can use it the same way (a "lag
//! switch" for a test build costs a few lines).
//!
//! What it models, matching what the real transports promise:
//! * **Datagrams** may be lost, duplicated, delayed (a base latency plus jitter, so two sent in order can
//!   arrive out of order) and are dropped when bigger than the MTU. Nothing arrives during an outage.
//! * **Reliable bytes** are never lost and never reordered, but are delayed, are cut into pieces of random
//!   size (a stream is a byte pipe: the reader must not depend on how it is cut), and are held back, not
//!   dropped, by an outage.
//! * [`SimNet::cut`] closes the link: every call then fails and `is_connected` is false.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use mid_net_transport::Transport;

/// How bad the network is. Probabilities are per thousand so that the type stays integer and exact.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SimConfig {
    /// Fixed one-way delay in milliseconds.
    pub latency_ms: u64,
    /// A random extra delay of `0..=jitter_ms` per datagram (this is what reorders them).
    pub jitter_ms: u64,
    /// Chance in 1/1000 that a datagram is lost.
    pub loss_per_mille: u32,
    /// Chance in 1/1000 that a datagram arrives twice.
    pub duplicate_per_mille: u32,
    /// A datagram longer than this is dropped (0: no limit).
    pub mtu: usize,
    /// Reliable bytes are cut into pieces of `1..=reliable_piece_max` bytes (0: delivered as sent).
    pub reliable_piece_max: usize,
}

impl Default for SimConfig {
    /// A perfect link with no delay.
    fn default() -> Self {
        SimConfig {
            latency_ms: 0,
            jitter_ms: 0,
            loss_per_mille: 0,
            duplicate_per_mille: 0,
            mtu: 0,
            reliable_piece_max: 0,
        }
    }
}

/// What the link did, for assertions.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SimStats {
    pub datagrams_sent: u64,
    pub datagrams_lost: u64,
    pub datagrams_duplicated: u64,
    pub datagrams_too_big: u64,
    pub datagrams_delivered: u64,
    pub reliable_bytes_sent: u64,
    pub reliable_bytes_delivered: u64,
    /// The longest datagram handed to the link, in bytes (whether or not it was then dropped).
    pub max_datagram_len: usize,
}

/// Why a simulated call failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimError {
    /// The link was cut with [`SimNet::cut`].
    Closed,
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    /// `0..n` (n > 0).
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

struct Datagram {
    due_ms: u64,
    order: u64,
    bytes: Vec<u8>,
}

struct Piece {
    due_ms: u64,
    stream_id: u32,
    bytes: Vec<u8>,
}

/// What travels one way.
#[derive(Default)]
struct Lane {
    datagrams: Vec<Datagram>,
    reliable: VecDeque<Piece>,
}

struct Inner {
    now_ms: u64,
    cfg: SimConfig,
    rng: Rng,
    order: u64,
    lanes: [Lane; 2],
    outages: Vec<(u64, u64)>,
    cut: bool,
    connected: bool,
    stats: SimStats,
}

impl Inner {
    fn in_outage(&self, t: u64) -> Option<u64> {
        self.outages
            .iter()
            .find(|(from, to)| t >= *from && t < *to)
            .map(|(_, to)| *to)
    }

    fn send_datagram(&mut self, from: usize, bytes: &[u8]) {
        self.stats.datagrams_sent += 1;
        self.stats.max_datagram_len = self.stats.max_datagram_len.max(bytes.len());
        if self.cfg.mtu != 0 && bytes.len() > self.cfg.mtu {
            self.stats.datagrams_too_big += 1;
            return;
        }
        if self.in_outage(self.now_ms).is_some() {
            self.stats.datagrams_lost += 1;
            return;
        }
        if self.rng.below(1000) < self.cfg.loss_per_mille as u64 {
            self.stats.datagrams_lost += 1;
            return;
        }
        let copies = if self.rng.below(1000) < self.cfg.duplicate_per_mille as u64 {
            self.stats.datagrams_duplicated += 1;
            2
        } else {
            1
        };
        for _ in 0..copies {
            let jitter = if self.cfg.jitter_ms > 0 {
                self.rng.below(self.cfg.jitter_ms + 1)
            } else {
                0
            };
            self.order += 1;
            let d = Datagram {
                due_ms: self.now_ms + self.cfg.latency_ms + jitter,
                order: self.order,
                bytes: bytes.to_vec(),
            };
            self.lanes[from].datagrams.push(d);
        }
    }

    fn send_reliable(&mut self, from: usize, stream_id: u32, bytes: &[u8]) {
        self.stats.reliable_bytes_sent += bytes.len() as u64;
        let mut due = self.now_ms + self.cfg.latency_ms;
        if let Some(end) = self.in_outage(self.now_ms) {
            due = due.max(end);
        }
        // never earlier than what is already queued: a stream does not overtake itself
        if let Some(last) = self.lanes[from].reliable.back() {
            due = due.max(last.due_ms);
        }
        let mut rest = bytes;
        loop {
            let take = if self.cfg.reliable_piece_max == 0 {
                rest.len()
            } else {
                (1 + self.rng.below(self.cfg.reliable_piece_max as u64) as usize).min(rest.len())
            };
            let (head, tail) = rest.split_at(take);
            self.lanes[from].reliable.push_back(Piece {
                due_ms: due,
                stream_id,
                bytes: head.to_vec(),
            });
            rest = tail;
            if rest.is_empty() {
                break;
            }
        }
    }

    fn poll_datagram(&mut self, to: usize) -> Option<Vec<u8>> {
        let from = 1 - to;
        loop {
            if self.in_outage(self.now_ms).is_some() {
                return None;
            }
            let now = self.now_ms;
            let best = self.lanes[from]
                .datagrams
                .iter()
                .enumerate()
                .filter(|(_, d)| d.due_ms <= now)
                .min_by_key(|(_, d)| (d.due_ms, d.order))
                .map(|(i, _)| i)?;
            let d = self.lanes[from].datagrams.swap_remove(best);
            // a datagram that would have arrived during a blackout never arrives
            if self.in_outage(d.due_ms).is_some() {
                self.stats.datagrams_lost += 1;
                continue;
            }
            self.stats.datagrams_delivered += 1;
            return Some(d.bytes);
        }
    }

    fn poll_reliable(&mut self, to: usize) -> Option<(u32, Vec<u8>)> {
        let from = 1 - to;
        if self.in_outage(self.now_ms).is_some() {
            return None;
        }
        let due = self.lanes[from].reliable.front()?.due_ms;
        if due > self.now_ms {
            return None;
        }
        let p = self.lanes[from].reliable.pop_front()?;
        self.stats.reliable_bytes_delivered += p.bytes.len() as u64;
        Some((p.stream_id, p.bytes))
    }
}

/// The controls of a simulated link. Cloning gives another handle to the same link.
#[derive(Clone)]
pub struct SimNet(Arc<Mutex<Inner>>);

/// One end of a simulated link.
pub struct SimTransport {
    net: SimNet,
    side: usize,
}

impl SimNet {
    /// A link and its two ends. `seed` fixes every random choice (it must not be 0).
    pub fn pair(cfg: SimConfig, seed: u64) -> (SimNet, SimTransport, SimTransport) {
        let net = SimNet(Arc::new(Mutex::new(Inner {
            now_ms: 0,
            cfg,
            rng: Rng(if seed == 0 { 0x9E37_79B9_7F4A_7C15 } else { seed }),
            order: 0,
            lanes: [Lane::default(), Lane::default()],
            outages: Vec::new(),
            cut: false,
            connected: true,
            stats: SimStats::default(),
        })));
        let a = SimTransport {
            net: net.clone(),
            side: 0,
        };
        let b = SimTransport {
            net: net.clone(),
            side: 1,
        };
        (net, a, b)
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Inner> {
        self.0.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Sets the clock (milliseconds; it only matters that both ends and the test agree).
    pub fn set_time(&self, now_ms: u64) {
        self.lock().now_ms = now_ms;
    }

    pub fn now(&self) -> u64 {
        self.lock().now_ms
    }

    /// Changes how bad the link is from now on (what is already in flight keeps its delay).
    pub fn set_config(&self, cfg: SimConfig) {
        self.lock().cfg = cfg;
    }

    /// Nothing is delivered between `from_ms` and `to_ms`; datagrams sent or due then are lost, reliable bytes wait.
    pub fn outage(&self, from_ms: u64, to_ms: u64) {
        self.lock().outages.push((from_ms, to_ms));
    }

    /// Whether the transports say they are connected (a transport that connects late).
    pub fn set_connected(&self, connected: bool) {
        self.lock().connected = connected;
    }

    /// Closes the link for good: every call fails.
    pub fn cut(&self) {
        self.lock().cut = true;
    }

    pub fn stats(&self) -> SimStats {
        self.lock().stats
    }
}

impl Transport for SimTransport {
    type Error = SimError;

    fn send_datagram(&mut self, bytes: &[u8]) -> Result<(), SimError> {
        let mut g = self.net.lock();
        if g.cut {
            return Err(SimError::Closed);
        }
        g.send_datagram(self.side, bytes);
        Ok(())
    }

    fn poll_datagram(&mut self) -> Result<Option<Vec<u8>>, SimError> {
        let mut g = self.net.lock();
        if g.cut {
            return Err(SimError::Closed);
        }
        Ok(g.poll_datagram(self.side))
    }

    fn send_reliable(&mut self, stream_id: u32, bytes: &[u8]) -> Result<(), SimError> {
        let mut g = self.net.lock();
        if g.cut {
            return Err(SimError::Closed);
        }
        g.send_reliable(self.side, stream_id, bytes);
        Ok(())
    }

    fn poll_reliable(&mut self) -> Result<Option<(u32, Vec<u8>)>, SimError> {
        let mut g = self.net.lock();
        if g.cut {
            return Err(SimError::Closed);
        }
        Ok(g.poll_reliable(self.side))
    }

    fn is_connected(&self) -> bool {
        let g = self.net.lock();
        g.connected && !g.cut
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn perfect() -> (SimNet, SimTransport, SimTransport) {
        SimNet::pair(SimConfig::default(), 7)
    }

    #[test]
    fn a_perfect_link_delivers_in_order_both_ways() {
        let (_n, mut a, mut b) = perfect();
        a.send_datagram(&[1]).unwrap();
        a.send_datagram(&[2]).unwrap();
        b.send_datagram(&[9]).unwrap();
        assert_eq!(b.poll_datagram().unwrap(), Some(vec![1]));
        assert_eq!(b.poll_datagram().unwrap(), Some(vec![2]));
        assert_eq!(b.poll_datagram().unwrap(), None);
        assert_eq!(a.poll_datagram().unwrap(), Some(vec![9]));
        a.send_reliable(3, &[5, 6]).unwrap();
        assert_eq!(b.poll_reliable().unwrap(), Some((3, vec![5, 6])));
        assert_eq!(b.poll_reliable().unwrap(), None);
        assert!(a.is_connected() && b.is_connected());
    }

    #[test]
    fn latency_holds_a_datagram_until_its_time() {
        let (n, mut a, mut b) = SimNet::pair(
            SimConfig {
                latency_ms: 50,
                ..SimConfig::default()
            },
            7,
        );
        a.send_datagram(&[1]).unwrap();
        n.set_time(49);
        assert_eq!(b.poll_datagram().unwrap(), None);
        n.set_time(50);
        assert_eq!(b.poll_datagram().unwrap(), Some(vec![1]));
    }

    #[test]
    fn loss_and_duplication_follow_the_configured_rates() {
        let (n, mut a, mut b) = SimNet::pair(
            SimConfig {
                loss_per_mille: 300,
                duplicate_per_mille: 100,
                ..SimConfig::default()
            },
            99,
        );
        for i in 0..10_000u32 {
            a.send_datagram(&i.to_le_bytes()).unwrap();
        }
        let mut got = 0;
        while b.poll_datagram().unwrap().is_some() {
            got += 1;
        }
        let s = n.stats();
        assert_eq!(s.datagrams_sent, 10_000);
        assert!((2_700..3_300).contains(&s.datagrams_lost), "lost {}", s.datagrams_lost);
        assert!(
            (500..900).contains(&s.datagrams_duplicated),
            "duplicated {}",
            s.datagrams_duplicated
        );
        assert_eq!(got as u64, s.datagrams_delivered);
        assert_eq!(s.datagrams_delivered, 10_000 - s.datagrams_lost + s.datagrams_duplicated);
    }

    #[test]
    fn jitter_reorders_datagrams_but_a_zero_jitter_link_never_does() {
        let reordered = |jitter| {
            let (n, mut a, mut b) = SimNet::pair(
                SimConfig {
                    latency_ms: 10,
                    jitter_ms: jitter,
                    ..SimConfig::default()
                },
                5,
            );
            for i in 0..200u8 {
                a.send_datagram(&[i]).unwrap();
            }
            n.set_time(1000);
            let mut seen = Vec::new();
            while let Some(d) = b.poll_datagram().unwrap() {
                seen.push(d[0]);
            }
            assert_eq!(seen.len(), 200);
            seen.windows(2).any(|w| w[1] < w[0])
        };
        assert!(reordered(30));
        assert!(!reordered(0));
    }

    #[test]
    fn a_datagram_over_the_mtu_is_dropped() {
        let (n, mut a, mut b) = SimNet::pair(
            SimConfig {
                mtu: 4,
                ..SimConfig::default()
            },
            5,
        );
        a.send_datagram(&[0; 5]).unwrap();
        a.send_datagram(&[0; 4]).unwrap();
        assert_eq!(b.poll_datagram().unwrap(), Some(vec![0; 4]));
        assert_eq!(b.poll_datagram().unwrap(), None);
        assert_eq!(n.stats().datagrams_too_big, 1);
    }

    #[test]
    fn reliable_bytes_are_cut_into_pieces_but_keep_their_order() {
        let (_n, mut a, mut b) = SimNet::pair(
            SimConfig {
                reliable_piece_max: 5,
                ..SimConfig::default()
            },
            11,
        );
        let data: Vec<u8> = (0..200u8).collect();
        a.send_reliable(1, &data[..100]).unwrap();
        a.send_reliable(1, &data[100..]).unwrap();
        let mut got = Vec::new();
        let mut pieces = 0;
        while let Some((id, p)) = b.poll_reliable().unwrap() {
            assert_eq!(id, 1);
            assert!(p.len() <= 5 && !p.is_empty());
            got.extend(p);
            pieces += 1;
        }
        assert_eq!(got, data);
        assert!(pieces >= 40);
    }

    #[test]
    fn an_outage_loses_datagrams_and_holds_reliable_bytes() {
        let (n, mut a, mut b) = SimNet::pair(
            SimConfig {
                latency_ms: 10,
                ..SimConfig::default()
            },
            5,
        );
        n.outage(100, 200);
        n.set_time(150);
        a.send_datagram(&[1]).unwrap(); // sent during the outage: lost
        a.send_reliable(0, &[2]).unwrap(); // held until it ends
        n.set_time(199);
        assert_eq!(b.poll_datagram().unwrap(), None);
        assert_eq!(b.poll_reliable().unwrap(), None);
        n.set_time(200);
        assert_eq!(b.poll_datagram().unwrap(), None);
        assert_eq!(b.poll_reliable().unwrap(), Some((0, vec![2]))); // released the moment the outage ends
        assert_eq!(n.stats().datagrams_lost, 1);
    }

    #[test]
    fn a_datagram_that_falls_due_during_an_outage_is_lost() {
        let (n, mut a, mut b) = SimNet::pair(
            SimConfig {
                latency_ms: 30,
                ..SimConfig::default()
            },
            5,
        );
        n.set_time(90);
        a.send_datagram(&[1]).unwrap(); // due at 120, inside the outage below
        n.outage(100, 200);
        n.set_time(250);
        assert_eq!(b.poll_datagram().unwrap(), None);
        assert_eq!(n.stats().datagrams_lost, 1);
    }

    #[test]
    fn a_cut_link_fails_every_call_and_is_not_connected() {
        let (n, mut a, mut b) = perfect();
        n.cut();
        assert_eq!(a.send_datagram(&[1]), Err(SimError::Closed));
        assert_eq!(a.send_reliable(0, &[1]), Err(SimError::Closed));
        assert_eq!(b.poll_datagram(), Err(SimError::Closed));
        assert_eq!(b.poll_reliable(), Err(SimError::Closed));
        assert!(!a.is_connected() && !b.is_connected());
    }

    #[test]
    fn a_link_can_connect_late() {
        let (n, a, _b) = perfect();
        n.set_connected(false);
        assert!(!a.is_connected());
        n.set_connected(true);
        assert!(a.is_connected());
    }

    #[test]
    fn the_same_seed_gives_the_same_run() {
        let run = |seed| {
            let (n, mut a, mut b) = SimNet::pair(
                SimConfig {
                    loss_per_mille: 400,
                    jitter_ms: 9,
                    ..SimConfig::default()
                },
                seed,
            );
            for i in 0..100u8 {
                a.send_datagram(&[i]).unwrap();
            }
            n.set_time(100);
            let mut v = Vec::new();
            while let Some(d) = b.poll_datagram().unwrap() {
                v.push(d[0]);
            }
            v
        };
        assert_eq!(run(3), run(3));
        assert_ne!(run(3), run(4));
    }
}
