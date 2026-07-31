//! Sequence number arithmetic and ack-bitfield tracking for the reliable
//! channel.
//!
//! This module is deliberately narrow: wraparound-aware comparison plus
//! "what have I received" bookkeeping, no framing, no sockets, no retry
//! timing. `reliable.rs` builds the actual send/receive protocol
//! (retransmit buffer, RTT-based timeout, wrapping `packet.rs` payloads
//! with a frame header) on top of what's here. Kept separate and tested
//! hard on its own because a subtle off-by-one or wraparound bug here
//! silently corrupts delivery guarantees rather than panicking — see
//! docs/mid-net.md's "Reliability mechanism" section.
//!
//! Convention (checked against gafferongames.com's reference design, the
//! same one docs/mid-net.md cites): `ack` is the highest sequence number
//! received so far; `ack_bits` is a 32-bit window of what came before it.
//! Bit `i` (0-indexed, `i` in `0..32`) set means `ack - (i + 1)` was also
//! received. `ack` itself is never stored in the bitfield — being the
//! latest received sequence number, it's implicitly known.

/// A 16-bit wire sequence number. Comparisons are cyclic (see
/// `is_more_recent_than`), so this deliberately does NOT implement
/// `PartialOrd`/`Ord` — `<`/`>` would silently do plain integer
/// comparison, which is wrong the moment a sequence number wraps past
/// `u16::MAX`. Use `is_more_recent_than` explicitly instead.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Sequence(pub u16);

impl Sequence {
    pub fn wrapping_next(self) -> Self {
        Sequence(self.0.wrapping_add(1))
    }

    /// True if `self` is more recent than `other`, accounting for
    /// wraparound. Standard technique: the wrapped difference, reinterpreted
    /// as a signed value of the same width, is positive iff `self` is ahead
    /// of `other` within half the sequence space — which is exactly the
    /// "closer forward than backward" rule wraparound comparison needs.
    pub fn is_more_recent_than(self, other: Self) -> bool {
        let diff = self.0.wrapping_sub(other.0) as i16;
        diff > 0
    }
}

/// Receiver-side bookkeeping: tracks which sequence numbers have arrived
/// so far, in the form the wire ack header wants (`ack` + `ack_bits`).
/// One of these belongs to whichever side is *receiving* reliable
/// packets, per direction of a connection (i.e. a full duplex link needs
/// two — one per direction — same as it needs two independent sequence
/// counters).
#[derive(Debug, Clone, Copy, Default)]
pub struct AckTracker {
    latest: Option<Sequence>,
    bits: u32,
}

impl AckTracker {
    pub fn new() -> Self {
        AckTracker { latest: None, bits: 0 }
    }

    /// Record that `seq` was received. Safe to call with packets arriving
    /// out of order, duplicated, or arbitrarily late — each case is
    /// handled without panicking; packets older than the 32-slot window
    /// are silently untracked (nothing to represent them with) rather
    /// than treated as an error.
    pub fn record_received(&mut self, seq: Sequence) {
        let Some(latest) = self.latest else {
            self.latest = Some(seq);
            self.bits = 0;
            return;
        };

        if seq.is_more_recent_than(latest) {
            // seq becomes the new ack. The old ack `latest` is now `shift`
            // positions behind it, landing on bit `shift - 1`; everything
            // already in `bits` moves left by `shift` to re-base onto the
            // new ack. `shift` is derived from a wrapping subtraction of
            // two already-validated-forward sequence numbers, so it's
            // always in 1..=32767 here -- never 0, never negative.
            let shift = seq.0.wrapping_sub(latest.0) as u32;
            self.bits = if shift > 32 {
                0
            } else {
                let shifted_old = self.bits.checked_shl(shift).unwrap_or(0);
                shifted_old | (1u32 << (shift - 1))
            };
            self.latest = Some(seq);
        } else if latest.is_more_recent_than(seq) {
            // An older packet arriving late, or a duplicate. Mark it if
            // it still fits the window; otherwise there's nothing to do.
            let shift = latest.0.wrapping_sub(seq.0) as u32;
            if shift >= 1 && shift <= 32 {
                self.bits |= 1u32 << (shift - 1);
            }
        }
        // else: seq == latest, an exact duplicate of the current ack --
        // already known received, no-op.
    }

    /// The `(ack, ack_bits)` pair to put on the wire, if anything has
    /// been received yet.
    pub fn snapshot(&self) -> Option<(Sequence, u32)> {
        self.latest.map(|ack| (ack, self.bits))
    }
}

/// Sender-side query: given an `(ack, ack_bits)` pair reported by the
/// peer, was `seq` (a sequence number this side previously sent)
/// acknowledged? Stateless and symmetric with `AckTracker::record_received`
/// — `reliable.rs`'s retransmit buffer calls this per pending outbound
/// packet against the most recent ack it's heard from the peer.
pub fn is_acked(ack: Sequence, ack_bits: u32, seq: Sequence) -> bool {
    if seq == ack {
        return true;
    }
    if !ack.is_more_recent_than(seq) {
        // seq is the same age or newer than ack -- the peer can't have
        // acked something it hasn't reported as received yet.
        return false;
    }
    let shift = ack.0.wrapping_sub(seq.0) as u32;
    shift <= 32 && (ack_bits & (1u32 << (shift - 1))) != 0
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Sequence::is_more_recent_than ---

    #[test]
    fn more_recent_normal_case() {
        assert!(Sequence(5).is_more_recent_than(Sequence(3)));
        assert!(!Sequence(3).is_more_recent_than(Sequence(5)));
        assert!(!Sequence(5).is_more_recent_than(Sequence(5)));
    }

    #[test]
    fn more_recent_handles_wraparound() {
        // 0 comes right after 65535 in the cyclic sequence, so 0 is more
        // recent even though 0 < 65535 as a plain integer.
        assert!(Sequence(0).is_more_recent_than(Sequence(65535)));
        assert!(!Sequence(65535).is_more_recent_than(Sequence(0)));
        assert!(Sequence(1).is_more_recent_than(Sequence(65535)));
    }

    // --- AckTracker::record_received / snapshot ---

    #[test]
    fn sequential_no_loss() {
        let mut t = AckTracker::new();
        for seq in 0..=3u16 {
            t.record_received(Sequence(seq));
        }
        // Received 0,1,2: bit0=2, bit1=1, bit2=0, all present.
        assert_eq!(t.snapshot(), Some((Sequence(3), 0b111)));
    }

    #[test]
    fn single_gap_is_reflected_in_bits() {
        // Receive 0, 1, then 3 -- 2 is lost.
        let mut t = AckTracker::new();
        t.record_received(Sequence(0));
        t.record_received(Sequence(1));
        t.record_received(Sequence(3));
        // ack=3. bit0 -> 2 (NOT received). bit1 -> 1 (received). bit2 -> 0 (received).
        assert_eq!(t.snapshot(), Some((Sequence(3), 0b110)));
        assert!(!is_acked(Sequence(3), 0b110, Sequence(2)));
        assert!(is_acked(Sequence(3), 0b110, Sequence(1)));
        assert!(is_acked(Sequence(3), 0b110, Sequence(0)));
    }

    #[test]
    fn matches_gafferongames_reference_example() {
        // From the reference article: receives 1,2,4,5,9,10 (3,6,7,8 lost).
        let mut t = AckTracker::new();
        for seq in [1u16, 2, 4, 5, 9, 10] {
            t.record_received(Sequence(seq));
        }
        assert_eq!(t.snapshot().unwrap().0, Sequence(10));
        for lost in [3u16, 6, 7, 8] {
            assert!(!is_acked(Sequence(10), t.snapshot().unwrap().1, Sequence(lost)), "seq {lost} should not be acked");
        }
        for received in [9u16, 5, 4, 2, 1] {
            assert!(is_acked(Sequence(10), t.snapshot().unwrap().1, Sequence(received)), "seq {received} should be acked");
        }
    }

    #[test]
    fn out_of_order_arrival_within_window() {
        let mut t = AckTracker::new();
        t.record_received(Sequence(5));
        t.record_received(Sequence(3)); // arrives late, still within window
        assert_eq!(t.snapshot().unwrap().0, Sequence(5)); // ack doesn't move backward
        assert!(is_acked(Sequence(5), t.snapshot().unwrap().1, Sequence(3)));
        assert!(!is_acked(Sequence(5), t.snapshot().unwrap().1, Sequence(4))); // never received
    }

    #[test]
    fn duplicate_packet_is_a_no_op() {
        let mut t = AckTracker::new();
        t.record_received(Sequence(5));
        let before = t.snapshot();
        t.record_received(Sequence(5));
        assert_eq!(t.snapshot(), before);
    }

    #[test]
    fn stale_packet_beyond_window_is_ignored_not_panicking() {
        let mut t = AckTracker::new();
        t.record_received(Sequence(100));
        let before = t.snapshot();
        t.record_received(Sequence(60)); // 40 behind -- outside the 32-slot window
        assert_eq!(t.snapshot(), before, "packet older than the window must not perturb state");
    }

    #[test]
    fn forward_jump_beyond_window_drops_old_history_without_panicking() {
        let mut t = AckTracker::new();
        t.record_received(Sequence(5));
        t.record_received(Sequence(4));
        t.record_received(Sequence(3));
        // Jump forward by 40 -- everything before this is outside the new window.
        t.record_received(Sequence(45));
        assert_eq!(t.snapshot(), Some((Sequence(45), 0)));
        assert!(!is_acked(Sequence(45), 0, Sequence(5)));
    }

    #[test]
    fn exact_window_boundary_shift_of_32_is_representable() {
        let mut t = AckTracker::new();
        t.record_received(Sequence(0));
        t.record_received(Sequence(32)); // shift == 32, the edge case
        assert_eq!(t.snapshot(), Some((Sequence(32), 1u32 << 31)));
        assert!(is_acked(Sequence(32), 1u32 << 31, Sequence(0)));
    }

    #[test]
    fn record_received_across_u16_wraparound_does_not_panic() {
        let mut t = AckTracker::new();
        for seq in [65533u16, 65534, 65535, 0, 1, 2] {
            t.record_received(Sequence(seq));
        }
        let (ack, bits) = t.snapshot().unwrap();
        assert_eq!(ack, Sequence(2));
        // 1, 0, 65535 immediately precede 2 in cyclic order and were all received.
        assert!(is_acked(ack, bits, Sequence(1)));
        assert!(is_acked(ack, bits, Sequence(0)));
        assert!(is_acked(ack, bits, Sequence(65535)));
    }

    // --- is_acked ---

    #[test]
    fn is_acked_true_for_the_ack_itself() {
        assert!(is_acked(Sequence(10), 0, Sequence(10)));
    }

    #[test]
    fn is_acked_false_for_anything_newer_than_ack() {
        // The peer can't have acked a packet it hasn't reported receiving.
        assert!(!is_acked(Sequence(10), 0xFFFF_FFFF, Sequence(11)));
    }
}
