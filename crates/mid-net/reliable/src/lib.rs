//! Reliability layer: frame headers, retransmit buffer, RTT-based
//! timeout. Built on `sequence.rs` (sequence/ack arithmetic) and
//! `packet.rs` (payload encode/decode); ties them together with the
//! actual send/receive protocol.
//!
//! ## Two hard constraints this module is built around
//!
//! **No sockets, no wall clock, no async runtime — anywhere in this
//! file.** Two independent reasons landed on the same design:
//!
//! 1. *Platform.* `std::net::UdpSocket` doesn't exist on `wasm32-unknown-unknown`
//!    — no raw UDP in a browser sandbox. The real transport there is
//!    WebTransport datagrams (baseline-available across browsers as of
//!    March 2026) or, for P2P, a WebRTC `RTCDataChannel` in unreliable/
//!    unordered mode — checked current browser support before committing
//!    to this rather than assuming. Both are still "send/receive a byte
//!    buffer, no delivery guarantee, no ordering guarantee, MTU-sized" —
//!    the same shape UDP presents — so this module talks only in raw
//!    bytes in and bytes out. Which concrete transport moves those bytes
//!    is entirely `socket.rs`'s problem, picked per-target with `cfg`,
//!    and this file never needs to change when that gets built.
//! 2. *FFI.* `std::time::Instant` has no defined layout and can't cross a
//!    C ABI. `Timestamp` below is a plain `u64` millisecond count instead
//!    — the caller (native: `Instant`-based clock; wasm: `performance.now()`
//!    via `web_sys`) converts to that at the FFI boundary, and everything
//!    inside this module is plain data. Also means every test below runs
//!    against a manually-advanced fake clock — no real sleeping, fully
//!    deterministic, no flaky timing-dependent tests.
//!
//! Wire format follows the ack-piggyback convention from
//! docs/mid-net.md: every reliable send carries `sequence` (this
//! packet's own number) plus the sender's current `ack`/`ack_bits` for
//! the *other* direction, so acks don't need their own dedicated
//! packets or their own reliability.

use mid_net_wire::PacketKind;
use mid_net_wire::{is_acked, Sequence};

/// Milliseconds on a caller-supplied monotonic clock. Deliberately plain
/// data (not `std::time::Instant`) — see the module doc. Unlike
/// `Sequence`, this is genuinely linear (a real clock doesn't wrap in
/// the lifetime of a connection), so `Ord` is safe and given here on
/// purpose, unlike `Sequence`'s deliberate omission of it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Timestamp(pub u64);

impl Timestamp {
    pub fn elapsed_ms_since(self, earlier: Timestamp) -> u64 {
        self.0.saturating_sub(earlier.0)
    }
}

// ---------------------------------------------------------------------
// Frame headers
// ---------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameError {
    UnexpectedEnd,
    UnknownPacketKind(u8),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UnreliableHeader {
    pub kind: PacketKind,
    pub sequence: Sequence,
}

pub const UNRELIABLE_HEADER_SIZE: usize = 3; // kind(1) + sequence(2)

/// Unreliable channel framing: just enough to identify the packet type
/// and let the receiver drop stale/out-of-order arrivals. No ack, no
/// retransmit — loss is fine on this channel by design.
pub fn encode_unreliable_frame(kind: PacketKind, sequence: Sequence, payload: &[u8], buf: &mut Vec<u8>) {
    buf.reserve(UNRELIABLE_HEADER_SIZE + payload.len());
    buf.push(kind as u8);
    buf.extend_from_slice(&sequence.0.to_le_bytes());
    buf.extend_from_slice(payload);
}

pub fn decode_unreliable_frame(buf: &[u8]) -> Result<(UnreliableHeader, &[u8]), FrameError> {
    if buf.len() < UNRELIABLE_HEADER_SIZE {
        return Err(FrameError::UnexpectedEnd);
    }
    let kind = PacketKind::from_u8(buf[0]).ok_or(FrameError::UnknownPacketKind(buf[0]))?;
    let sequence = Sequence(u16::from_le_bytes([buf[1], buf[2]]));
    Ok((UnreliableHeader { kind, sequence }, &buf[UNRELIABLE_HEADER_SIZE..]))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReliableHeader {
    pub kind: PacketKind,
    pub sequence: Sequence,
    pub ack: Sequence,
    pub ack_bits: u32,
}

pub const RELIABLE_HEADER_SIZE: usize = 9; // kind(1) + sequence(2) + ack(2) + ack_bits(4)

/// Reliable channel framing: `sequence` is this packet's own number
/// (what the peer's `AckTracker` will record); `ack`/`ack_bits` piggyback
/// this side's receive state for the *other* direction, per
/// docs/mid-net.md's "Reliability mechanism".
pub fn encode_reliable_frame(kind: PacketKind, sequence: Sequence, ack: Sequence, ack_bits: u32, payload: &[u8], buf: &mut Vec<u8>) {
    buf.reserve(RELIABLE_HEADER_SIZE + payload.len());
    buf.push(kind as u8);
    buf.extend_from_slice(&sequence.0.to_le_bytes());
    buf.extend_from_slice(&ack.0.to_le_bytes());
    buf.extend_from_slice(&ack_bits.to_le_bytes());
    buf.extend_from_slice(payload);
}

pub fn decode_reliable_frame(buf: &[u8]) -> Result<(ReliableHeader, &[u8]), FrameError> {
    if buf.len() < RELIABLE_HEADER_SIZE {
        return Err(FrameError::UnexpectedEnd);
    }
    let kind = PacketKind::from_u8(buf[0]).ok_or(FrameError::UnknownPacketKind(buf[0]))?;
    let sequence = Sequence(u16::from_le_bytes([buf[1], buf[2]]));
    let ack = Sequence(u16::from_le_bytes([buf[3], buf[4]]));
    let ack_bits = u32::from_le_bytes([buf[5], buf[6], buf[7], buf[8]]);
    Ok((ReliableHeader { kind, sequence, ack, ack_bits }, &buf[RELIABLE_HEADER_SIZE..]))
}

// ---------------------------------------------------------------------
// RTT estimation
// ---------------------------------------------------------------------

const DEFAULT_RTT_MS: f64 = 200.0;
const MIN_TIMEOUT_MS: f64 = 50.0;
const MAX_TIMEOUT_MS: f64 = 3000.0;
// Same smoothing constants as TCP's classic RTO estimator (Jacobson/Karels):
// SRTT tracks the mean, RTTVAR tracks mean deviation, RTO = SRTT + 4*RTTVAR.
const EWMA_ALPHA: f64 = 0.125;
const VARIANCE_BETA: f64 = 0.25;

/// Exponentially-smoothed RTT estimate, used to size the retransmit
/// timeout instead of a fixed guess (fixed timeouts either resend too
/// eagerly on a slow connection or too late on a fast one).
#[derive(Debug, Clone, Copy)]
pub struct RttEstimator {
    smoothed_ms: Option<f64>,
    variance_ms: f64,
}

impl RttEstimator {
    pub fn new() -> Self {
        RttEstimator { smoothed_ms: None, variance_ms: 0.0 }
    }

    pub fn on_sample(&mut self, sample_ms: f64) {
        match self.smoothed_ms {
            None => {
                self.smoothed_ms = Some(sample_ms);
                self.variance_ms = sample_ms / 2.0;
            }
            Some(prev) => {
                let delta = sample_ms - prev;
                self.variance_ms += VARIANCE_BETA * (delta.abs() - self.variance_ms);
                self.smoothed_ms = Some(prev + EWMA_ALPHA * delta);
            }
        }
    }

    pub fn smoothed_ms(&self) -> f64 {
        self.smoothed_ms.unwrap_or(DEFAULT_RTT_MS)
    }

    /// Retransmit timeout: smoothed RTT plus a variance margin, clamped
    /// so one lucky or unlucky sample can't produce a timeout that
    /// never fires or fires every tick.
    pub fn timeout_ms(&self) -> f64 {
        let variance = if self.smoothed_ms.is_some() { self.variance_ms } else { self.smoothed_ms() / 2.0 };
        (self.smoothed_ms() + 4.0 * variance).clamp(MIN_TIMEOUT_MS, MAX_TIMEOUT_MS)
    }
}

impl Default for RttEstimator {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------
// Retransmit buffer
// ---------------------------------------------------------------------

struct SentPacket {
    sequence: Sequence,
    first_sent_at: Timestamp,
    last_sent_at: Timestamp,
    retransmit_count: u32,
    payload: Vec<u8>,
}

/// Sender-side reliability state for one direction of one connection.
/// Owns the RTT estimate and the set of sent-but-unacknowledged packets.
/// Deliberately plain owned data (`Vec`s, no borrows, no lifetimes) so
/// `ffi.rs` can later wrap this behind an opaque handle without any
/// redesign here.
pub struct RetransmitBuffer {
    unacked: Vec<SentPacket>,
    rtt: RttEstimator,
}

impl RetransmitBuffer {
    pub fn new() -> Self {
        RetransmitBuffer { unacked: Vec::new(), rtt: RttEstimator::new() }
    }

    pub fn rtt_estimate_ms(&self) -> f64 {
        self.rtt.smoothed_ms()
    }

    /// Record a freshly-sent reliable packet so it's tracked for
    /// ack/retransmit. `payload` is the exact bytes to resend verbatim
    /// if it's lost (already-encoded packet payload, pre-framing).
    pub fn on_sent(&mut self, sequence: Sequence, now: Timestamp, payload: Vec<u8>) {
        self.unacked.push(SentPacket {
            sequence,
            first_sent_at: now,
            last_sent_at: now,
            retransmit_count: 0,
            payload,
        });
    }

    /// Process an ack header received from the peer: drop every packet
    /// it confirms, and feed an RTT sample for each one that was never
    /// retransmitted. (Retransmitted packets are excluded from RTT
    /// sampling on purpose — Karn's algorithm: once a packet's been
    /// resent, an incoming ack is ambiguous about which transmission it's
    /// actually acknowledging, so trusting its timing would poison the
    /// RTT estimate.)
    pub fn on_ack_received(&mut self, ack: Sequence, ack_bits: u32, now: Timestamp) {
        let mut i = 0;
        while i < self.unacked.len() {
            if is_acked(ack, ack_bits, self.unacked[i].sequence) {
                let p = self.unacked.swap_remove(i);
                if p.retransmit_count == 0 {
                    self.rtt.on_sample(now.elapsed_ms_since(p.first_sent_at) as f64);
                }
                // don't advance i -- swap_remove moved a new element into position i
            } else {
                i += 1;
            }
        }
    }

    /// Packets whose retransmit timeout has elapsed as of `now`. Marks
    /// each one as resent (updates `last_sent_at`, bumps
    /// `retransmit_count`) as a side effect, so the caller can hand the
    /// returned bytes straight to the transport without any further
    /// bookkeeping.
    pub fn collect_due_for_retransmit(&mut self, now: Timestamp) -> Vec<(Sequence, Vec<u8>)> {
        let timeout = self.rtt.timeout_ms();
        let mut due = Vec::new();
        for p in self.unacked.iter_mut() {
            if now.elapsed_ms_since(p.last_sent_at) as f64 >= timeout {
                due.push((p.sequence, p.payload.clone()));
                p.last_sent_at = now;
                p.retransmit_count += 1;
            }
        }
        due
    }

    pub fn unacked_count(&self) -> usize {
        self.unacked.len()
    }
}

impl Default for RetransmitBuffer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mid_net_wire::PacketKind;

    // --- frame headers ---

    #[test]
    fn unreliable_frame_round_trips() {
        let mut buf = Vec::new();
        encode_unreliable_frame(PacketKind::PlayerState, Sequence(42), &[1, 2, 3], &mut buf);
        let (header, payload) = decode_unreliable_frame(&buf).unwrap();
        assert_eq!(header, UnreliableHeader { kind: PacketKind::PlayerState, sequence: Sequence(42) });
        assert_eq!(payload, &[1, 2, 3]);
    }

    #[test]
    fn unreliable_frame_rejects_short_buffer() {
        assert_eq!(decode_unreliable_frame(&[0, 1]), Err(FrameError::UnexpectedEnd));
    }

    #[test]
    fn unreliable_frame_rejects_unknown_kind() {
        let buf = [7u8, 0, 0]; // kind byte 7 doesn't exist
        assert_eq!(decode_unreliable_frame(&buf), Err(FrameError::UnknownPacketKind(7)));
    }

    #[test]
    fn reliable_frame_round_trips() {
        let mut buf = Vec::new();
        encode_reliable_frame(PacketKind::PlayerEvent, Sequence(9), Sequence(100), 0b1011, &[9, 9], &mut buf);
        let (header, payload) = decode_reliable_frame(&buf).unwrap();
        assert_eq!(header, ReliableHeader { kind: PacketKind::PlayerEvent, sequence: Sequence(9), ack: Sequence(100), ack_bits: 0b1011 });
        assert_eq!(payload, &[9, 9]);
    }

    #[test]
    fn reliable_frame_rejects_short_buffer() {
        assert_eq!(decode_reliable_frame(&[0u8; 8]), Err(FrameError::UnexpectedEnd));
    }

    #[test]
    fn empty_payload_round_trips_on_both_frame_kinds() {
        let mut buf = Vec::new();
        encode_unreliable_frame(PacketKind::PlayerState, Sequence(1), &[], &mut buf);
        let (_, payload) = decode_unreliable_frame(&buf).unwrap();
        assert!(payload.is_empty());

        let mut buf2 = Vec::new();
        encode_reliable_frame(PacketKind::PlayerEvent, Sequence(1), Sequence(1), 0, &[], &mut buf2);
        let (_, payload2) = decode_reliable_frame(&buf2).unwrap();
        assert!(payload2.is_empty());
    }

    // --- RttEstimator ---

    #[test]
    fn rtt_estimator_starts_at_conservative_default() {
        let rtt = RttEstimator::new();
        assert_eq!(rtt.smoothed_ms(), DEFAULT_RTT_MS);
        assert!(rtt.timeout_ms() >= DEFAULT_RTT_MS); // includes variance margin
    }

    #[test]
    fn rtt_estimator_converges_toward_stable_samples() {
        let mut rtt = RttEstimator::new();
        for _ in 0..50 {
            rtt.on_sample(100.0);
        }
        assert!((rtt.smoothed_ms() - 100.0).abs() < 1.0, "should converge close to a stable 100ms RTT, got {}", rtt.smoothed_ms());
        // Stable samples -> low variance -> timeout should settle close to the RTT itself.
        assert!(rtt.timeout_ms() < 150.0, "timeout should tighten once RTT is stable, got {}", rtt.timeout_ms());
    }

    #[test]
    fn rtt_estimator_widens_timeout_under_jitter() {
        let mut stable = RttEstimator::new();
        let mut jittery = RttEstimator::new();
        for i in 0..20 {
            stable.on_sample(100.0);
            jittery.on_sample(if i % 2 == 0 { 50.0 } else { 150.0 });
        }
        assert!(jittery.timeout_ms() > stable.timeout_ms(), "a jittery connection should get a wider timeout than a stable one");
    }

    #[test]
    fn rtt_estimator_timeout_is_clamped() {
        let mut rtt = RttEstimator::new();
        rtt.on_sample(0.001); // absurdly fast
        assert!(rtt.timeout_ms() >= MIN_TIMEOUT_MS);
        for _ in 0..10 {
            rtt.on_sample(100_000.0); // absurdly slow
        }
        assert!(rtt.timeout_ms() <= MAX_TIMEOUT_MS);
    }

    // --- RetransmitBuffer ---

    #[test]
    fn acked_packet_is_removed_and_unacked_is_not() {
        let mut buf = RetransmitBuffer::new();
        buf.on_sent(Sequence(1), Timestamp(0), vec![1]);
        buf.on_sent(Sequence(2), Timestamp(0), vec![2]);
        assert_eq!(buf.unacked_count(), 2);

        // Peer reports it received seq 1 (as its `ack`) but not seq 2.
        buf.on_ack_received(Sequence(1), 0, Timestamp(20));
        assert_eq!(buf.unacked_count(), 1);
    }

    #[test]
    fn ack_bitfield_can_ack_older_packets_too() {
        let mut buf = RetransmitBuffer::new();
        buf.on_sent(Sequence(1), Timestamp(0), vec![1]);
        buf.on_sent(Sequence(2), Timestamp(0), vec![2]);
        // ack=2, bit0 set -> seq 1 (2-1) also acked.
        buf.on_ack_received(Sequence(2), 0b1, Timestamp(20));
        assert_eq!(buf.unacked_count(), 0);
    }

    #[test]
    fn nothing_retransmitted_before_timeout_elapses() {
        let mut buf = RetransmitBuffer::new();
        buf.on_sent(Sequence(1), Timestamp(0), vec![1, 2, 3]);
        // Default timeout starts well above 10ms (DEFAULT_RTT_MS = 200ms + margin).
        let due = buf.collect_due_for_retransmit(Timestamp(10));
        assert!(due.is_empty());
    }

    #[test]
    fn retransmits_after_timeout_and_updates_bookkeeping() {
        let mut buf = RetransmitBuffer::new();
        buf.on_sent(Sequence(1), Timestamp(0), vec![1, 2, 3]);
        let timeout = buf.rtt.timeout_ms() as u64;

        let due = buf.collect_due_for_retransmit(Timestamp(timeout + 1));
        assert_eq!(due, vec![(Sequence(1), vec![1, 2, 3])]);

        // Immediately after a retransmit, it should NOT be due again right away.
        let due_again = buf.collect_due_for_retransmit(Timestamp(timeout + 2));
        assert!(due_again.is_empty());
    }

    #[test]
    fn retransmitted_packet_does_not_pollute_rtt_sample() {
        let mut buf = RetransmitBuffer::new();
        buf.on_sent(Sequence(1), Timestamp(0), vec![1]);
        let timeout = buf.rtt.timeout_ms() as u64;
        // Force a retransmit before any ack arrives.
        let due = buf.collect_due_for_retransmit(Timestamp(timeout + 1));
        assert_eq!(due.len(), 1);

        let rtt_before = buf.rtt_estimate_ms();
        // Ack finally arrives, long after the original send -- if this were
        // wrongly sampled as an RTT it would massively skew the estimate.
        buf.on_ack_received(Sequence(1), 0, Timestamp(timeout + 5000));
        assert_eq!(buf.rtt_estimate_ms(), rtt_before, "ack for a retransmitted packet must not feed an RTT sample");
    }

    #[test]
    fn clean_ack_without_retransmit_does_feed_rtt_sample() {
        let mut buf = RetransmitBuffer::new();
        buf.on_sent(Sequence(1), Timestamp(1000), vec![1]);
        buf.on_ack_received(Sequence(1), 0, Timestamp(1075)); // 75ms RTT, no retransmit involved
        assert_eq!(buf.rtt_estimate_ms(), 75.0);
    }

    #[test]
    fn duplicate_ack_of_already_removed_packet_is_harmless() {
        let mut buf = RetransmitBuffer::new();
        buf.on_sent(Sequence(1), Timestamp(0), vec![1]);
        buf.on_ack_received(Sequence(1), 0, Timestamp(50));
        assert_eq!(buf.unacked_count(), 0);
        // Same ack arrives again (redundant, per the ack-piggyback design) -- must not panic or misbehave.
        buf.on_ack_received(Sequence(1), 0, Timestamp(60));
        assert_eq!(buf.unacked_count(), 0);
    }
}
