//! Putting messages into packets and taking them out again: channels, datagram packets, and reliable
//! streams.
//!
//! This is layer 2 of `docs/mid-net-foundation.md`. It does not touch a socket (that is the
//! facade's job, over a `Transport`); it is the bytes on the two kinds of wire:
//!
//! * **Datagrams** (unreliable): a packet is `seq:16 | count:8 | message*`, a message is
//!   `channel | message id | fields` ([`MessageRegistry`] / [`Dispatcher`] say how the last two are
//!   written). Many messages share one packet, up to the buffer the caller gives ([`DatagramBuilder`]
//!   backs a message out cleanly if it does not fit). Each channel has its own meaning
//!   ([`ChannelKind`]): a **sequenced** channel (state: only the newest matters) drops a message that
//!   arrives after a newer packet's, an **unordered** channel (cosmetic) delivers each message once, in
//!   whatever order, and a packet seen twice is dropped whole ([`DatagramReceiver`]).
//! * **Reliable streams** (ordered, a byte pipe): a message is `length | message id | fields`, length in
//!   bytes as a varint, so the receiver can cut messages out of a stream that arrives in arbitrary pieces
//!   ([`StreamReader`]); [`StreamWriter`] queues them with a **hard limit**, because a reliable queue that
//!   may grow without bound is how a slow peer takes a server down (`docs/mid-net.md`): past the limit a
//!   push fails and the caller must drop the connection.
//!
//! Every reader treats its input as hostile: nothing here can panic, an unknown channel or a reliable
//! channel inside a datagram is an error, and a stream frame over the limit is refused before it is
//! buffered.

use std::collections::VecDeque;
use std::fmt;

use crate::bits::{BitError, BitReader, BitWriter};
use crate::poly::{Dispatcher, MessageRegistry, NetMessage, NetMessageType};
use crate::sequence::Sequence;

/// What a channel promises.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChannelKind {
    /// Datagrams; a message older than one already delivered on this channel is dropped (state updates).
    UnreliableSequenced,
    /// Datagrams; every message is delivered at most once, in any order (cosmetic events).
    UnreliableUnordered,
    /// A stream: every message, once, in order. Only on a [`StreamWriter`]/[`StreamReader`], never in a datagram.
    ReliableOrdered,
}

/// Why a channel could not be defined.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChannelError {
    /// The id does not fit the table's width.
    IdTooLarge,
    Duplicate,
    /// The width is 0 or more than 8 bits.
    BadWidth,
}

impl fmt::Display for ChannelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            ChannelError::IdTooLarge => "channel id does not fit the width",
            ChannelError::Duplicate => "channel already defined",
            ChannelError::BadWidth => "channel width must be 1 to 8 bits",
        })
    }
}

/// The channels a connection has, by id, with `bits` bits to write an id in.
pub struct ChannelTable {
    bits: u32,
    kinds: Vec<Option<ChannelKind>>,
}

impl ChannelTable {
    /// A table with room for `1 << bits` channels (`bits` 1 to 8; 2 bits are enough for 4 channels).
    pub fn new(bits: u32) -> Result<Self, ChannelError> {
        if !(1..=8).contains(&bits) {
            return Err(ChannelError::BadWidth);
        }
        Ok(ChannelTable {
            bits,
            kinds: vec![None; 1usize << bits],
        })
    }

    pub fn bits(&self) -> u32 {
        self.bits
    }

    pub fn define(&mut self, id: u8, kind: ChannelKind) -> Result<(), ChannelError> {
        let slot = self
            .kinds
            .get_mut(id as usize)
            .ok_or(ChannelError::IdTooLarge)?;
        if slot.is_some() {
            return Err(ChannelError::Duplicate);
        }
        *slot = Some(kind);
        Ok(())
    }

    pub fn kind(&self, id: u8) -> Option<ChannelKind> {
        self.kinds.get(id as usize).copied().flatten()
    }
}

/// Why a message could not be added to a packet.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PushError {
    /// The channel is not defined, or is a reliable one (those go on a stream).
    BadChannel,
    /// The message's type is not in the registry.
    Unregistered,
    /// The message could not be encoded (a value that does not fit its field, ...).
    Encode(BitError),
}

impl fmt::Display for PushError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PushError::BadChannel => f.write_str("channel not defined or not a datagram channel"),
            PushError::Unregistered => f.write_str("message type not registered"),
            PushError::Encode(e) => write!(f, "message could not be encoded: {e}"),
        }
    }
}

/// A packet carries at most this many messages (the count is a byte).
pub const MAX_MESSAGES_PER_DATAGRAM: usize = 255;

/// Builds one datagram in a buffer the caller owns (its length is the packet budget: an MTU, about 1200 bytes).
pub struct DatagramBuilder<'a, 'r> {
    w: BitWriter<'a>,
    reg: &'r MessageRegistry,
    channels: &'r ChannelTable,
    count: usize,
}

impl<'a, 'r> DatagramBuilder<'a, 'r> {
    /// Starts a packet with sequence number `seq` (the caller counts them: [`Sequence::wrapping_next`]).
    /// `Err(Overflow)` if the buffer cannot even hold the header.
    pub fn new(
        buf: &'a mut [u8],
        seq: Sequence,
        reg: &'r MessageRegistry,
        channels: &'r ChannelTable,
    ) -> Result<Self, BitError> {
        let mut w = BitWriter::new(buf);
        w.write_u16(seq.0)?;
        w.write_u8(0)?; // the count, patched in finish()
        Ok(DatagramBuilder {
            w,
            reg,
            channels,
            count: 0,
        })
    }

    pub fn message_count(&self) -> usize {
        self.count
    }

    pub fn bytes_so_far(&self) -> usize {
        self.w.bytes_written()
    }

    /// Adds a message. `Ok(true)` added; `Ok(false)` it does not fit (or the packet has 255 messages): the
    /// packet is exactly as it was, send it and start the next with this message. `Err` the message can never
    /// go in a datagram on this channel; the packet is as it was.
    pub fn push_dyn(&mut self, channel: u8, msg: &mut dyn NetMessage) -> Result<bool, PushError> {
        match self.channels.kind(channel) {
            Some(ChannelKind::UnreliableSequenced) | Some(ChannelKind::UnreliableUnordered) => {}
            _ => return Err(PushError::BadChannel),
        }
        if self.count >= MAX_MESSAGES_PER_DATAGRAM {
            return Ok(false);
        }
        let mark = self.w.bits_written();
        let attempt = self
            .w
            .write_bits(channel as u64, self.channels.bits())
            .and_then(|_| self.reg.write(msg, &mut self.w));
        match attempt {
            Ok(()) => {
                self.count += 1;
                Ok(true)
            }
            Err(e) => {
                self.w.truncate_to(mark);
                match e {
                    BitError::Overflow => Ok(false),
                    BitError::UnknownTag => Err(PushError::Unregistered),
                    other => Err(PushError::Encode(other)),
                }
            }
        }
    }

    /// [`push_dyn`](Self::push_dyn) for a value of a known type.
    pub fn push<T: NetMessageType>(&mut self, channel: u8, msg: &mut T) -> Result<bool, PushError> {
        self.push_dyn(channel, msg)
    }

    /// Finishes the packet; returns how many bytes of the buffer are the packet.
    pub fn finish(mut self) -> usize {
        self.w
            .set_byte(2, self.count as u8)
            .expect("the header is always written");
        self.w.bytes_written()
    }
}

/// What came of one received datagram.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ReceiveReport {
    /// Messages handed to the game.
    pub delivered: usize,
    /// Messages read and dropped as stale (a newer one on a sequenced channel had already been delivered).
    pub stale: usize,
    /// The whole packet was dropped: its sequence number was seen before, or is too old to tell.
    pub duplicate_packet: bool,
}

/// What a receiver remembers: which packet sequence numbers have been seen (the last 64) and the newest packet that carried each channel.
#[derive(Clone)]
struct RxState {
    latest: Option<Sequence>,
    window: u64, // bit i set: latest - (i + 1) was seen
    last_on_channel: [Option<Sequence>; 256],
}

impl RxState {
    /// True the first time a sequence number is seen (and it is not too old to know); false for a repeat or a packet more than 64 behind.
    fn accept(&mut self, seq: Sequence) -> bool {
        let Some(latest) = self.latest else {
            self.latest = Some(seq);
            self.window = 0;
            return true;
        };
        if seq.is_more_recent_than(latest) {
            let shift = seq.0.wrapping_sub(latest.0) as u32;
            self.window = if shift > 64 {
                0
            } else {
                self.window.checked_shl(shift).unwrap_or(0) | (1u64 << (shift - 1))
            };
            self.latest = Some(seq);
            true
        } else if latest.is_more_recent_than(seq) {
            let behind = latest.0.wrapping_sub(seq.0) as u32;
            if !(1..=64).contains(&behind) || self.window >> (behind - 1) & 1 == 1 {
                return false;
            }
            self.window |= 1u64 << (behind - 1);
            true
        } else {
            false
        }
    }

    /// Whether a message of a sequenced channel in packet `seq` is current; remembers it if so.
    fn current(&mut self, channel: u8, seq: Sequence) -> bool {
        let slot = &mut self.last_on_channel[channel as usize];
        match *slot {
            Some(last) if last != seq && !seq.is_more_recent_than(last) => false,
            _ => {
                *slot = Some(seq);
                true
            }
        }
    }
}

/// Takes datagrams apart: drops a packet seen before, drops a message of a sequenced channel that a newer packet has already replaced, and
/// hands over the rest.
pub struct DatagramReceiver {
    st: RxState,
}

impl DatagramReceiver {
    pub fn new() -> Self {
        DatagramReceiver {
            st: RxState {
                latest: None,
                window: 0,
                last_on_channel: [None; 256],
            },
        }
    }

    /// Reads a datagram and gives every message that should be delivered to `on_message` **after the whole packet has been read without
    /// error**. A malformed packet delivers nothing and leaves the receiver exactly as it was (the same packet, made right, is still new).
    /// Costs one allocation per message (see [`receive_dispatch`](Self::receive_dispatch)).
    pub fn receive(
        &mut self,
        bytes: &[u8],
        channels: &ChannelTable,
        reg: &MessageRegistry,
        mut on_message: impl FnMut(u8, Box<dyn NetMessage>),
    ) -> Result<ReceiveReport, BitError> {
        let mut r = BitReader::new(bytes);
        let seq = Sequence(r.read_u16()?);
        let count = r.read_u8()? as usize;
        let mut report = ReceiveReport::default();
        let mut st = self.st.clone(); // all the bookkeeping happens on a copy, kept only if the whole packet reads
        if !st.accept(seq) {
            report.duplicate_packet = true;
            return Ok(report);
        }
        let mut staged: Vec<(u8, Box<dyn NetMessage>)> = Vec::new();
        for _ in 0..count {
            let channel = r.read_bits(channels.bits())? as u8;
            let deliver = match channels.kind(channel) {
                Some(ChannelKind::UnreliableSequenced) => {
                    let fresh = st.current(channel, seq);
                    if !fresh {
                        report.stale += 1;
                    }
                    fresh
                }
                Some(ChannelKind::UnreliableUnordered) => true,
                _ => return Err(BitError::UnknownTag),
            };
            let msg = reg.read(&mut r)?;
            if deliver {
                staged.push((channel, msg));
            }
        }
        self.st = st;
        report.delivered = staged.len();
        for (channel, msg) in staged {
            on_message(channel, msg);
        }
        Ok(report)
    }

    /// Reads a datagram and runs the [`Dispatcher`]'s typed handler for each message that should be delivered, with no allocation. Unlike
    /// [`receive`](Self::receive) the handlers run as the packet is read, so a malformed packet may have delivered the messages before the
    /// malformed one: on `Err` the sender is broken or hostile, and the connection should be dropped.
    pub fn receive_dispatch<C>(
        &mut self,
        bytes: &[u8],
        channels: &ChannelTable,
        dispatcher: &mut Dispatcher<C>,
        ctx: &mut C,
    ) -> Result<ReceiveReport, BitError> {
        let mut r = BitReader::new(bytes);
        let seq = Sequence(r.read_u16()?);
        let count = r.read_u8()? as usize;
        let mut report = ReceiveReport::default();
        if !self.st.accept(seq) {
            report.duplicate_packet = true;
            return Ok(report);
        }
        for _ in 0..count {
            let channel = r.read_bits(channels.bits())? as u8;
            let deliver = match channels.kind(channel) {
                Some(ChannelKind::UnreliableSequenced) => self.st.current(channel, seq),
                Some(ChannelKind::UnreliableUnordered) => true,
                _ => return Err(BitError::UnknownTag),
            };
            dispatcher.dispatch_if(ctx, &mut r, deliver)?;
            if deliver {
                report.delivered += 1;
            } else {
                report.stale += 1;
            }
        }
        Ok(report)
    }
}

impl Default for DatagramReceiver {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Reliable streams
// ---------------------------------------------------------------------------------------------------------------

/// Why a stream message could not be queued.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamError {
    /// The queue is at its limit. **Drop the connection**: the peer is not reading fast enough, and waiting grows memory without bound.
    Full,
    /// The message is bigger than a frame may be.
    TooLarge,
    Unregistered,
    Encode(BitError),
}

impl fmt::Display for StreamError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StreamError::Full => f.write_str("reliable queue full"),
            StreamError::TooLarge => f.write_str("message larger than a frame"),
            StreamError::Unregistered => f.write_str("message type not registered"),
            StreamError::Encode(e) => write!(f, "message could not be encoded: {e}"),
        }
    }
}

/// The reliable side of a connection, writing: messages queued as `length | message`, handed to the transport in pieces.
pub struct StreamWriter {
    queue: VecDeque<Vec<u8>>,
    front_sent: usize,
    pending_bytes: usize,
    max_bytes: usize,
    scratch: Vec<u8>,
}

impl StreamWriter {
    /// `max_bytes`: the most that may wait unsent (past it a push is `Full`); `max_frame`: the biggest message in bytes.
    pub fn new(max_bytes: usize, max_frame: usize) -> Self {
        StreamWriter {
            queue: VecDeque::new(),
            front_sent: 0,
            pending_bytes: 0,
            max_bytes,
            scratch: vec![0; max_frame],
        }
    }

    pub fn pending_bytes(&self) -> usize {
        self.pending_bytes
    }

    pub fn pending_messages(&self) -> usize {
        self.queue.len()
    }

    /// Queues a message. On `Err` nothing was queued.
    pub fn push_dyn(
        &mut self,
        reg: &MessageRegistry,
        msg: &mut dyn NetMessage,
    ) -> Result<(), StreamError> {
        let mut w = BitWriter::new(&mut self.scratch);
        match reg.write(msg, &mut w) {
            Ok(()) => {}
            Err(BitError::Overflow) => return Err(StreamError::TooLarge),
            Err(BitError::UnknownTag) => return Err(StreamError::Unregistered),
            Err(e) => return Err(StreamError::Encode(e)),
        }
        let len = w.bytes_written();
        let mut head = [0u8; 10];
        let mut hw = BitWriter::new(&mut head);
        hw.write_varint(len as u64).map_err(StreamError::Encode)?;
        let head_len = hw.bytes_written();
        if self.pending_bytes + head_len + len > self.max_bytes {
            return Err(StreamError::Full);
        }
        let mut frame = Vec::with_capacity(head_len + len);
        frame.extend_from_slice(&head[..head_len]);
        frame.extend_from_slice(&self.scratch[..len]);
        self.pending_bytes += frame.len();
        self.queue.push_back(frame);
        Ok(())
    }

    pub fn push<T: NetMessageType>(
        &mut self,
        reg: &MessageRegistry,
        msg: &mut T,
    ) -> Result<(), StreamError> {
        self.push_dyn(reg, msg)
    }

    /// Copies up to `out.len()` bytes of what is waiting into `out` and forgets them; returns how many. The transport's
    /// "send these bytes on the stream" takes whatever it can (a frame may be cut across two calls: the stream is a byte pipe).
    pub fn fill(&mut self, out: &mut [u8]) -> usize {
        let mut n = 0;
        while n < out.len() {
            let Some(front) = self.queue.front() else {
                break;
            };
            let rest = &front[self.front_sent..];
            let take = rest.len().min(out.len() - n);
            out[n..n + take].copy_from_slice(&rest[..take]);
            n += take;
            self.front_sent += take;
            self.pending_bytes -= take;
            if self.front_sent == front.len() {
                self.queue.pop_front();
                self.front_sent = 0;
            }
        }
        n
    }
}

/// The reliable side of a connection, reading: bytes arrive in any pieces, whole messages come out.
pub struct StreamReader {
    buf: Vec<u8>,
    max_frame: usize,
}

impl StreamReader {
    /// `max_frame`: the biggest message in bytes; a frame that claims more is refused before it is buffered.
    pub fn new(max_frame: usize) -> Self {
        StreamReader {
            buf: Vec::new(),
            max_frame,
        }
    }

    pub fn buffered(&self) -> usize {
        self.buf.len()
    }

    /// Adds bytes from the stream. `Err(OutOfRange)` if they would make the buffer bigger than one frame (plus its length) can be:
    /// a peer that sends more than it ever completes.
    pub fn feed(&mut self, bytes: &[u8]) -> Result<(), BitError> {
        if self.buf.len().saturating_add(bytes.len()) > self.max_frame.saturating_add(10) + 4096 {
            return Err(BitError::OutOfRange);
        }
        self.buf.extend_from_slice(bytes);
        Ok(())
    }

    /// The length prefix and the frame it announces, if the whole frame is here: `(prefix bytes, frame bytes)`.
    fn next_frame_len(&self) -> Result<Option<(usize, usize)>, BitError> {
        let mut r = BitReader::new(&self.buf[..self.buf.len().min(10)]);
        let len = match r.read_varint() {
            Ok(l) => l,
            Err(BitError::Overflow) => return Ok(None), // the length itself is not all here yet
            Err(e) => return Err(e),
        };
        if len > self.max_frame as u64 {
            return Err(BitError::OutOfRange);
        }
        let head = r.bits_read() / 8;
        if self.buf.len() < head + len as usize {
            return Ok(None);
        }
        Ok(Some((head, len as usize)))
    }

    /// [`next_frame_len`](Self::next_frame_len), and on an error forget what is buffered (a length that is wrong means the stream cannot be cut up any more).
    fn frame_or_reset(&mut self) -> Result<Option<(usize, usize)>, BitError> {
        match self.next_frame_len() {
            Err(e) => {
                self.buf.clear();
                Err(e)
            }
            ok => ok,
        }
    }

    /// The next whole message, if one has arrived. `Err` for a frame that is too large or malformed (the frame is skipped; a peer that sends
    /// these is broken or hostile: drop the connection).
    pub fn next_message(
        &mut self,
        reg: &MessageRegistry,
    ) -> Result<Option<Box<dyn NetMessage>>, BitError> {
        let Some((head, len)) = self.frame_or_reset()? else {
            return Ok(None);
        };
        let result = {
            let mut r = BitReader::new(&self.buf[head..head + len]);
            reg.read(&mut r).and_then(|m| {
                if r.bits_remaining() >= 8 {
                    Err(BitError::OutOfRange)
                } else {
                    Ok(m)
                }
            })
        };
        self.buf.drain(..head + len);
        result.map(Some)
    }

    /// The same for a [`Dispatcher`]: runs the handler of the next whole message and returns its id.
    pub fn next_dispatch<C>(
        &mut self,
        dispatcher: &mut Dispatcher<C>,
        ctx: &mut C,
    ) -> Result<Option<u16>, BitError> {
        let Some((head, len)) = self.frame_or_reset()? else {
            return Ok(None);
        };
        let result = {
            let mut r = BitReader::new(&self.buf[head..head + len]);
            dispatcher.dispatch(ctx, &mut r).and_then(|id| {
                if r.bits_remaining() >= 8 {
                    Err(BitError::OutOfRange)
                } else {
                    Ok(id)
                }
            })
        };
        self.buf.drain(..head + len);
        result.map(Some)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{impl_net_serialize, Sequence};

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn below(&mut self, n: u64) -> u64 {
            self.next() % n
        }
    }

    #[derive(Debug, Default, Clone, PartialEq)]
    struct Pos {
        n: u32,
        x: f32,
    }
    impl_net_serialize!(Pos {
        n: raw(),
        x: quant(-100.0, 100.0, 13)
    });
    impl NetMessageType for Pos {
        const ID: u16 = 1;
    }

    #[derive(Debug, Default, Clone, PartialEq)]
    struct Event {
        kind: u8,
        id: u32,
    }
    impl_net_serialize!(Event {
        kind: bits(3),
        id: varint()
    });
    impl NetMessageType for Event {
        const ID: u16 = 2;
    }

    #[derive(Debug, Default, Clone, PartialEq)]
    struct Chat {
        text: String,
    }
    impl_net_serialize!(Chat { text: string(200) });
    impl NetMessageType for Chat {
        const ID: u16 = 3;
    }

    #[derive(Debug, Default, Clone, PartialEq)]
    struct Unlisted {
        a: u8,
    }
    impl_net_serialize!(Unlisted { a: raw() });
    impl NetMessageType for Unlisted {
        const ID: u16 = 9;
    }

    const SEQUENCED: u8 = 0;
    const UNORDERED: u8 = 1;
    const RELIABLE: u8 = 2;

    fn setup() -> (MessageRegistry, ChannelTable) {
        let mut reg = MessageRegistry::new(4);
        reg.register::<Pos>().unwrap();
        reg.register::<Event>().unwrap();
        reg.register::<Chat>().unwrap();
        let mut ch = ChannelTable::new(2).unwrap();
        ch.define(SEQUENCED, ChannelKind::UnreliableSequenced)
            .unwrap();
        ch.define(UNORDERED, ChannelKind::UnreliableUnordered)
            .unwrap();
        ch.define(RELIABLE, ChannelKind::ReliableOrdered).unwrap();
        (reg, ch)
    }

    /// One packet carrying `pos` on the sequenced channel and `events` on the unordered one.
    fn packet(seq: u16, pos: Option<u32>, events: &[u32]) -> Vec<u8> {
        let (reg, ch) = setup();
        let mut buf = vec![0u8; 400];
        let mut b = DatagramBuilder::new(&mut buf, Sequence(seq), &reg, &ch).unwrap();
        if let Some(n) = pos {
            assert!(b.push(SEQUENCED, &mut Pos { n, x: 1.0 }).unwrap());
        }
        for &id in events {
            assert!(b.push(UNORDERED, &mut Event { kind: 1, id }).unwrap());
        }
        let n = b.finish();
        buf.truncate(n);
        buf
    }

    fn collect(
        rx: &mut DatagramReceiver,
        bytes: &[u8],
    ) -> (Result<ReceiveReport, BitError>, Vec<(u8, u32)>) {
        let (reg, ch) = setup();
        let mut got = Vec::new();
        let rep = rx.receive(bytes, &ch, &reg, |c, m| {
            let v = if let Some(p) = m.downcast_ref::<Pos>() {
                p.n
            } else {
                m.downcast_ref::<Event>().unwrap().id
            };
            got.push((c, v));
        });
        (rep, got)
    }

    // ---- building a datagram ------------------------------------------------------------------------------------------

    #[test]
    fn a_packet_is_its_sequence_its_count_and_its_messages_and_comes_back_whole() {
        let bytes = packet(0x0102, Some(77), &[5, 6]);
        assert_eq!(
            &bytes[..3],
            &[0x02, 0x01, 3],
            "sequence little-endian, then the count"
        );
        let mut rx = DatagramReceiver::new();
        let (rep, got) = collect(&mut rx, &bytes);
        assert_eq!(
            rep.unwrap(),
            ReceiveReport {
                delivered: 3,
                stale: 0,
                duplicate_packet: false
            }
        );
        assert_eq!(got, vec![(SEQUENCED, 77), (UNORDERED, 5), (UNORDERED, 6)]);
    }

    #[test]
    fn a_message_that_does_not_fit_leaves_the_packet_exactly_as_it_was() {
        let (reg, ch) = setup();
        // Pos is 2 (channel) + 4 (id) + 32 + 13 = 51 bits; Event is 2 + 4 + 3 + a varint byte = 17 bits and up.
        let mut with_try = [0u8; 8];
        let mut without = [0u8; 8];
        let (a, b) = {
            let mut x = DatagramBuilder::new(&mut with_try, Sequence(9), &reg, &ch).unwrap();
            let mut y = DatagramBuilder::new(&mut without, Sequence(9), &reg, &ch).unwrap();
            assert!(x.push(UNORDERED, &mut Event { kind: 3, id: 1 }).unwrap());
            assert!(y.push(UNORDERED, &mut Event { kind: 3, id: 1 }).unwrap());
            for _ in 0..5 {
                assert_eq!(
                    x.push(
                        SEQUENCED,
                        &mut Pos {
                            n: u32::MAX,
                            x: 99.0
                        }
                    ),
                    Ok(false),
                    "23 bits left, a Pos needs 51"
                );
            }
            // After failures at odd bit positions a smaller message still fits, and lands in the same place.
            assert!(x.push(UNORDERED, &mut Event { kind: 7, id: 2 }).unwrap());
            assert!(y.push(UNORDERED, &mut Event { kind: 7, id: 2 }).unwrap());
            assert_eq!(
                (x.message_count(), x.bytes_so_far()),
                (y.message_count(), y.bytes_so_far())
            );
            (x.finish(), y.finish())
        };
        assert_eq!(a, b);
        assert_eq!(with_try[..a], without[..b], "byte for byte the same packet");
        let mut rx = DatagramReceiver::new();
        let (rep, got) = collect(&mut rx, &with_try[..a]);
        assert_eq!(rep.unwrap().delivered, 2);
        assert_eq!(got, vec![(UNORDERED, 1), (UNORDERED, 2)]);
    }

    #[test]
    fn what_can_never_go_in_a_datagram_is_refused_and_changes_nothing() {
        let (reg, ch) = setup();
        let mut buf = [0u8; 64];
        let mut b = DatagramBuilder::new(&mut buf, Sequence(1), &reg, &ch).unwrap();
        assert!(b.push(SEQUENCED, &mut Pos::default()).unwrap());
        let before = (b.message_count(), b.bytes_so_far());
        assert_eq!(
            b.push(RELIABLE, &mut Chat::default()),
            Err(PushError::BadChannel),
            "reliable messages go on a stream"
        );
        assert_eq!(
            b.push(3, &mut Pos::default()),
            Err(PushError::BadChannel),
            "a channel that is not defined"
        );
        assert_eq!(
            b.push(UNORDERED, &mut Unlisted { a: 1 }),
            Err(PushError::Unregistered)
        );
        assert_eq!(
            b.push(UNORDERED, &mut Event { kind: 9, id: 1 }),
            Err(PushError::Encode(BitError::OutOfRange)),
            "kind 9 does not fit 3 bits"
        );
        assert_eq!((b.message_count(), b.bytes_so_far()), before);
        assert!(PushError::BadChannel.to_string().contains("channel"));
        let mut tiny = [0u8; 2];
        assert!(
            DatagramBuilder::new(&mut tiny, Sequence(1), &reg, &ch).is_err(),
            "no room even for the header"
        );
    }

    #[test]
    fn a_packet_holds_at_most_255_messages() {
        let (reg, ch) = setup();
        let mut buf = vec![0u8; 4096];
        let mut b = DatagramBuilder::new(&mut buf, Sequence(1), &reg, &ch).unwrap();
        for i in 0..255 {
            assert!(
                b.push(UNORDERED, &mut Event { kind: 0, id: 0 }).unwrap(),
                "message {i}"
            );
        }
        assert_eq!(b.push(UNORDERED, &mut Event::default()), Ok(false));
        let n = b.finish();
        assert_eq!(buf[2], 255);
        let mut rx = DatagramReceiver::new();
        let (rep, got) = collect(&mut rx, &buf[..n]);
        assert_eq!((rep.unwrap().delivered, got.len()), (255, 255));
    }

    #[test]
    fn channel_tables_refuse_what_does_not_fit() {
        assert_eq!(ChannelTable::new(0).err(), Some(ChannelError::BadWidth));
        assert_eq!(ChannelTable::new(9).err(), Some(ChannelError::BadWidth));
        let mut t = ChannelTable::new(2).unwrap();
        t.define(3, ChannelKind::ReliableOrdered).unwrap();
        assert_eq!(
            t.define(4, ChannelKind::ReliableOrdered),
            Err(ChannelError::IdTooLarge)
        );
        assert_eq!(
            t.define(3, ChannelKind::UnreliableSequenced),
            Err(ChannelError::Duplicate)
        );
        assert_eq!(
            (t.kind(3), t.kind(0), t.kind(200), t.bits()),
            (Some(ChannelKind::ReliableOrdered), None, None, 2)
        );
    }

    // ---- receiving: stale, duplicate, reordered ---------------------------------------------------------------------------

    #[test]
    fn an_older_state_never_overwrites_a_newer_one_but_unordered_events_still_arrive() {
        let mut rx = DatagramReceiver::new();
        let (_, got1) = collect(&mut rx, &packet(1, Some(10), &[100]));
        let (_, got3) = collect(&mut rx, &packet(3, Some(30), &[300]));
        let (rep2, got2) = collect(&mut rx, &packet(2, Some(20), &[200])); // late
        assert_eq!(got1, vec![(SEQUENCED, 10), (UNORDERED, 100)]);
        assert_eq!(got3, vec![(SEQUENCED, 30), (UNORDERED, 300)]);
        assert_eq!(
            got2,
            vec![(UNORDERED, 200)],
            "the late state is dropped, the late event is not"
        );
        assert_eq!(
            rep2.unwrap(),
            ReceiveReport {
                delivered: 1,
                stale: 1,
                duplicate_packet: false
            }
        );
    }

    #[test]
    fn a_packet_seen_twice_is_dropped_whole_and_one_too_old_to_know_is_too() {
        let mut rx = DatagramReceiver::new();
        let p = packet(5, Some(1), &[1]);
        assert_eq!(collect(&mut rx, &p).1.len(), 2);
        let (rep, got) = collect(&mut rx, &p);
        assert!(rep.unwrap().duplicate_packet && got.is_empty());
        let (rep, got) = collect(&mut rx, &packet(5 + 64, Some(2), &[2]));
        assert!(!rep.unwrap().duplicate_packet && got.len() == 2);
        let (rep, _) = collect(&mut rx, &packet(5, Some(1), &[1]));
        assert!(
            rep.unwrap().duplicate_packet,
            "still the repeat of 5, now 64 behind"
        );
        let (rep, got) = collect(&mut rx, &packet(5 + 64 + 100, Some(3), &[]));
        assert!(!rep.unwrap().duplicate_packet && got.len() == 1);
        let (rep, got) = collect(&mut rx, &packet(5 + 64 + 100 - 65, Some(4), &[4]));
        assert!(
            rep.unwrap().duplicate_packet && got.is_empty(),
            "65 behind: too old to tell a repeat from a new one, so dropped"
        );
        let (rep, got) = collect(&mut rx, &packet(5 + 64 + 100 - 64, Some(4), &[4]));
        assert!(
            !rep.unwrap().duplicate_packet && got == vec![(UNORDERED, 4)],
            "64 behind is still known: new, so its event arrives (its state is stale)"
        );
    }

    #[test]
    fn sequence_numbers_wrap_around_without_losing_or_repeating_anything() {
        let mut rx = DatagramReceiver::new();
        let mut all = Vec::new();
        for (i, seq) in [65534u16, 65535, 0, 1].into_iter().enumerate() {
            all.extend(collect(&mut rx, &packet(seq, Some(i as u32), &[i as u32 + 10])).1);
        }
        assert_eq!(
            all.iter()
                .filter(|m| m.0 == SEQUENCED)
                .map(|m| m.1)
                .collect::<Vec<_>>(),
            vec![0, 1, 2, 3]
        );
        assert_eq!(all.len(), 8);
        let (rep, got) = collect(&mut rx, &packet(65535, Some(99), &[]));
        assert!(
            rep.unwrap().duplicate_packet && got.is_empty(),
            "a repeat from before the wrap"
        );
    }

    #[test]
    fn a_malformed_packet_delivers_nothing_and_leaves_the_receiver_as_it_was() {
        let (reg, ch) = setup();
        let good = packet(7, Some(1), &[1, 2]);
        let mut rx = DatagramReceiver::new();
        // cut short, in every place
        for cut in 0..good.len() {
            let (rep, got) = collect(&mut rx, &good[..cut]);
            assert!(rep.is_err() && got.is_empty(), "cut at {cut}");
        }
        // after all those failures the whole packet is still new
        let (rep, got) = collect(&mut rx, &good);
        assert_eq!((rep.unwrap().delivered, got.len()), (3, 3));
        // a channel that is not defined, a reliable channel, an unknown message id
        let mut buf = vec![0u8; 64];
        for (label, chan, msg_id) in [
            ("undefined channel", 3u64, 1u64),
            ("reliable channel", RELIABLE as u64, 1),
            ("unknown message id", UNORDERED as u64, 12),
        ] {
            let mut w = BitWriter::new(&mut buf);
            w.write_u16(100).unwrap();
            w.write_u8(2).unwrap();
            // a good message first, so that something could have been delivered before the bad one
            w.write_bits(UNORDERED as u64, 2).unwrap();
            reg.write(&mut Event { kind: 1, id: 1 }, &mut w).unwrap();
            w.write_bits(chan, 2).unwrap();
            if msg_id == 1 {
                reg.write(&mut Pos::default(), &mut w).unwrap(); // a whole valid message: only the channel is wrong
            } else {
                w.write_bits(msg_id, 4).unwrap();
                w.write_bits(0, 40).unwrap();
            }
            let bytes = w.written().to_vec();
            let mut rx2 = DatagramReceiver::new();
            let (rep, got) = collect(&mut rx2, &bytes);
            assert!(rep.is_err() && got.is_empty(), "{label}: nothing delivered");
            let _ = &ch;
        }
        // a count that promises more than there is
        let mut lying = packet(9, Some(1), &[]);
        lying[2] = 9;
        let (rep, got) = collect(&mut DatagramReceiver::new(), &lying);
        assert!(rep.is_err() && got.is_empty());
    }

    // ---- receiving with typed handlers ---------------------------------------------------------------------------------------

    #[derive(Default)]
    struct World {
        states: Vec<u32>,
        events: Vec<u32>,
    }

    fn dispatcher() -> Dispatcher<World> {
        let mut d = Dispatcher::<World>::new(4);
        d.on::<Pos>(|w, p| w.states.push(p.n)).unwrap();
        d.on::<Event>(|w, e| w.events.push(e.id)).unwrap();
        d
    }

    #[test]
    fn typed_handlers_get_the_current_state_and_every_event_and_a_skipped_message_still_moves_the_reader_on(
    ) {
        let (_, ch) = setup();
        let mut rx = DatagramReceiver::new();
        let mut d = dispatcher();
        let mut world = World::default();
        let r6 = rx
            .receive_dispatch(&packet(6, Some(60), &[]), &ch, &mut d, &mut world)
            .unwrap();
        // Packet 5 arrives late: its state is skipped (the reader must still get past it) and the event after it is delivered.
        let r5 = rx
            .receive_dispatch(&packet(5, Some(50), &[500, 501]), &ch, &mut d, &mut world)
            .unwrap();
        assert_eq!((r6.delivered, r5.delivered, r5.stale), (1, 2, 1));
        assert_eq!(
            (world.states.clone(), world.events.clone()),
            (vec![60], vec![500, 501])
        );
        let dup = rx
            .receive_dispatch(&packet(6, Some(61), &[]), &ch, &mut d, &mut world)
            .unwrap();
        assert!(dup.duplicate_packet);
        assert_eq!(world.states, vec![60]);
        // Two states in one packet on the same channel are both current (the same packet).
        let (reg, chans) = setup();
        let mut buf = [0u8; 100];
        let mut b = DatagramBuilder::new(&mut buf, Sequence(7), &reg, &chans).unwrap();
        b.push(SEQUENCED, &mut Pos { n: 70, x: 0.0 }).unwrap();
        b.push(SEQUENCED, &mut Pos { n: 71, x: 0.0 }).unwrap();
        let n = b.finish();
        rx.receive_dispatch(&buf[..n], &ch, &mut d, &mut world)
            .unwrap();
        assert_eq!(world.states, vec![60, 70, 71]);
    }

    // ---- an unreliable network ------------------------------------------------------------------------------------------------

    #[test]
    fn over_a_network_that_loses_repeats_and_reorders_nothing_old_ever_overwrites_new_and_no_event_arrives_twice(
    ) {
        for (seed, loss, dup, jitter) in [
            (1u64, 0u64, 0u64, 0u64),
            (2, 20, 0, 0),
            (3, 0, 30, 0),
            (4, 0, 0, 8),
            (5, 25, 20, 10),
            (6, 40, 40, 20),
        ] {
            let mut rng = Rng(0x9E37_79B9_7F4A_7C15 ^ seed);
            let (reg, ch) = setup();
            let mut rx = DatagramReceiver::new();
            // in flight: (arrival tick, bytes)
            let mut flight: Vec<(u64, Vec<u8>)> = Vec::new();
            let mut seq = Sequence(65500); // across the wrap
            let (mut states, mut events): (Vec<u32>, Vec<u32>) = (Vec::new(), Vec::new());
            let mut sent_events = 0u32;
            for tick in 0..6000u64 {
                if tick < 5000 {
                    let mut buf = [0u8; 200];
                    let mut b = DatagramBuilder::new(&mut buf, seq, &reg, &ch).unwrap();
                    b.push(
                        SEQUENCED,
                        &mut Pos {
                            n: tick as u32,
                            x: 0.0,
                        },
                    )
                    .unwrap();
                    b.push(
                        UNORDERED,
                        &mut Event {
                            kind: 1,
                            id: sent_events,
                        },
                    )
                    .unwrap();
                    sent_events += 1;
                    let n = b.finish();
                    seq = seq.wrapping_next();
                    if rng.below(100) >= loss {
                        flight.push((tick + 1 + rng.below(jitter + 1), buf[..n].to_vec()));
                        if rng.below(100) < dup {
                            flight.push((tick + 1 + rng.below(jitter + 3), buf[..n].to_vec()));
                        }
                    }
                }
                let mut i = 0;
                while i < flight.len() {
                    if flight[i].0 <= tick {
                        let (_, bytes) = flight.swap_remove(i);
                        let mut world = World::default();
                        let mut d = dispatcher();
                        rx.receive_dispatch(&bytes, &ch, &mut d, &mut world)
                            .unwrap();
                        states.extend(world.states);
                        events.extend(world.events);
                    } else {
                        i += 1;
                    }
                }
            }
            assert!(
                states.windows(2).all(|w| w[0] < w[1]),
                "seed {seed}: a state older than one already delivered got through"
            );
            let mut sorted = events.clone();
            sorted.sort_unstable();
            sorted.dedup();
            assert_eq!(
                sorted.len(),
                events.len(),
                "seed {seed}: an event was delivered twice"
            );
            if loss == 0 && jitter == 0 {
                assert_eq!(
                    (states.len(), events.len()),
                    (5000, 5000),
                    "seed {seed}: a clean network loses nothing"
                );
            } else {
                assert!(!states.is_empty() && events.len() > 1000, "seed {seed}");
            }
        }
    }

    #[test]
    fn random_bytes_into_a_receiver_never_panic() {
        let (reg, ch) = setup();
        let mut rng = Rng(0xABCD_EF01_2345_6789);
        let mut rx = DatagramReceiver::new();
        let mut rx2 = DatagramReceiver::new();
        let mut d = dispatcher();
        for _ in 0..30000 {
            let len = rng.below(60) as usize;
            let bytes: Vec<u8> = (0..len).map(|_| rng.next() as u8).collect();
            let _ = rx.receive(&bytes, &ch, &reg, |_, _| {});
            let _ = rx2.receive_dispatch(&bytes, &ch, &mut d, &mut World::default());
        }
    }

    // ---- reliable streams -----------------------------------------------------------------------------------------------------

    #[test]
    fn messages_queued_on_a_stream_come_out_in_order_however_the_bytes_are_cut_up() {
        let (reg, _) = setup();
        let mut w = StreamWriter::new(100_000, 1000);
        let texts: Vec<String> = (0..60).map(|i| "x".repeat(i * 3)).collect();
        for t in &texts {
            w.push(&reg, &mut Chat { text: t.clone() }).unwrap();
        }
        w.push(
            &reg,
            &mut Event {
                kind: 5,
                id: 123_456,
            },
        )
        .unwrap();
        let total = w.pending_bytes();
        assert_eq!(w.pending_messages(), 61);
        let mut rng = Rng(42);
        let mut wire = Vec::new();
        loop {
            let mut chunk = vec![0u8; 1 + rng.below(9) as usize];
            let n = w.fill(&mut chunk);
            if n == 0 {
                break;
            }
            wire.extend_from_slice(&chunk[..n]);
        }
        assert_eq!(
            (wire.len(), w.pending_bytes(), w.pending_messages()),
            (total, 0, 0)
        );
        let mut r = StreamReader::new(1000);
        let mut got: Vec<Box<dyn NetMessage>> = Vec::new();
        let mut at = 0;
        while at < wire.len() {
            let take = (1 + rng.below(11) as usize).min(wire.len() - at);
            r.feed(&wire[at..at + take]).unwrap();
            at += take;
            while let Some(m) = r.next_message(&reg).unwrap() {
                got.push(m);
            }
        }
        assert_eq!(got.len(), 61);
        for (i, t) in texts.iter().enumerate() {
            assert_eq!(&got[i].downcast_ref::<Chat>().unwrap().text, t);
        }
        assert_eq!(
            got[60].downcast_ref::<Event>().unwrap(),
            &Event {
                kind: 5,
                id: 123_456
            }
        );
        assert_eq!(r.buffered(), 0);
        // byte by byte
        let mut r = StreamReader::new(1000);
        let mut n = 0;
        for b in &wire {
            r.feed(&[*b]).unwrap();
            while r.next_message(&reg).unwrap().is_some() {
                n += 1;
            }
        }
        assert_eq!(n, 61);
    }

    #[test]
    fn a_full_queue_says_so_queues_nothing_and_takes_messages_again_once_it_drains() {
        let (reg, _) = setup();
        let mut w = StreamWriter::new(100, 64);
        let mut accepted = 0;
        loop {
            let before = (w.pending_bytes(), w.pending_messages());
            match w.push(
                &reg,
                &mut Chat {
                    text: "0123456789".into(),
                },
            ) {
                Ok(()) => accepted += 1,
                Err(e) => {
                    assert_eq!(e, StreamError::Full);
                    assert_eq!(
                        (w.pending_bytes(), w.pending_messages()),
                        before,
                        "nothing was queued"
                    );
                    break;
                }
            }
        }
        assert!(accepted > 3 && w.pending_bytes() <= 100);
        let mut out = [0u8; 40];
        assert_eq!(w.fill(&mut out), 40);
        assert!(
            w.push(
                &reg,
                &mut Chat {
                    text: "0123456789".into()
                }
            )
            .is_ok(),
            "room again"
        );
        assert!(StreamError::Full.to_string().contains("full"));
    }

    #[test]
    fn what_cannot_be_queued_is_refused_and_changes_nothing() {
        let (reg, _) = setup();
        let mut w = StreamWriter::new(10_000, 16);
        assert_eq!(
            w.push(
                &reg,
                &mut Chat {
                    text: "this text is far too long for a 16 byte frame".into()
                }
            ),
            Err(StreamError::TooLarge)
        );
        assert_eq!(
            w.push(&reg, &mut Unlisted { a: 1 }),
            Err(StreamError::Unregistered)
        );
        assert_eq!(
            w.push(&reg, &mut Event { kind: 9, id: 1 }),
            Err(StreamError::Encode(BitError::OutOfRange))
        );
        assert_eq!((w.pending_bytes(), w.pending_messages()), (0, 0));
        let mut none = [0u8; 0];
        assert_eq!(w.fill(&mut none), 0);
    }

    #[test]
    fn a_stream_reader_refuses_a_frame_that_claims_too_much_before_buffering_it() {
        let (reg, _) = setup();
        let mut r = StreamReader::new(100);
        r.feed(&[0xE8, 0x07]).unwrap(); // a frame of 1000 bytes
        assert_eq!(r.next_message(&reg).err(), Some(BitError::OutOfRange));
        assert_eq!(
            r.buffered(),
            0,
            "what was buffered is forgotten: the stream cannot be cut up any more"
        );
        // more bytes than one frame can need, without a complete frame
        let mut r = StreamReader::new(100);
        assert!(r.feed(&vec![0x80u8; 5000]).is_err());
        // a length that is cut short waits, a frame that is not all here waits
        let mut r = StreamReader::new(100);
        r.feed(&[0x80]).unwrap();
        assert!(r.next_message(&reg).unwrap().is_none());
        let mut r = StreamReader::new(100);
        r.feed(&[10, 1, 2]).unwrap();
        assert!(r.next_message(&reg).unwrap().is_none());
        assert_eq!(r.buffered(), 3);
        // an over-long length (more than 64 bits) is an error too
        let mut r = StreamReader::new(100);
        r.feed(&[0xFF; 10]).unwrap();
        assert!(r.next_message(&reg).is_err());
    }

    #[test]
    fn a_malformed_frame_is_skipped_and_the_next_one_still_reads() {
        let (reg, _) = setup();
        let mut w = StreamWriter::new(1000, 64);
        let mut wire = Vec::new();
        // frame 1: a good Event with a junk byte after it (the length is one too long)
        w.push(&reg, &mut Event { kind: 1, id: 1 }).unwrap();
        let mut good = [0u8; 64];
        let n = w.fill(&mut good);
        wire.push(good[0] + 2);
        wire.extend_from_slice(&good[1..n]);
        wire.extend_from_slice(&[0xAA, 0xBB]);
        // frame 2: an unknown message id
        wire.extend_from_slice(&[2, 0x0F, 0]);
        // frame 3: good
        w.push(&reg, &mut Chat { text: "ok".into() }).unwrap();
        let n = w.fill(&mut good);
        wire.extend_from_slice(&good[..n]);
        let mut r = StreamReader::new(64);
        r.feed(&wire).unwrap();
        assert!(r.next_message(&reg).is_err(), "trailing bytes in a frame");
        assert!(r.next_message(&reg).is_err(), "unknown id");
        let m = r.next_message(&reg).unwrap().unwrap();
        assert_eq!(m.downcast_ref::<Chat>().unwrap().text, "ok");
        assert!(r.next_message(&reg).unwrap().is_none());
    }

    #[test]
    fn stream_messages_can_go_to_typed_handlers_and_a_bad_frame_runs_none() {
        let (reg, _) = setup();
        let mut w = StreamWriter::new(1000, 64);
        w.push(&reg, &mut Event { kind: 2, id: 9 }).unwrap();
        w.push(&reg, &mut Event { kind: 2, id: 10 }).unwrap();
        let mut wire = vec![0u8; 100];
        let n = w.fill(&mut wire);
        let mut r = StreamReader::new(64);
        r.feed(&wire[..n]).unwrap();
        let mut d = dispatcher();
        let mut world = World::default();
        assert_eq!(r.next_dispatch(&mut d, &mut world), Ok(Some(2)));
        assert_eq!(r.next_dispatch(&mut d, &mut world), Ok(Some(2)));
        assert_eq!(r.next_dispatch(&mut d, &mut world), Ok(None));
        assert_eq!(world.events, vec![9, 10]);
        r.feed(&[3, 0x0F, 0, 0]).unwrap(); // an unknown id
        assert!(r.next_dispatch(&mut d, &mut world).is_err());
        r.feed(&[2, 0x1F, 0]).unwrap(); // Pos (id 1) with its fields cut short inside the frame
        assert!(r.next_dispatch(&mut d, &mut world).is_err());
        assert_eq!(
            world.states.len(),
            0,
            "no handler ran for a frame that did not read"
        );
    }

    #[test]
    fn random_streams_never_panic_and_never_buffer_more_than_a_frame_can_need() {
        let (reg, _) = setup();
        let mut rng = Rng(0x5151_5151_7777_1234);
        for _ in 0..3000 {
            let mut r = StreamReader::new(50);
            for _ in 0..20 {
                let len = rng.below(30) as usize;
                let bytes: Vec<u8> = (0..len).map(|_| rng.next() as u8).collect();
                if r.feed(&bytes).is_err() {
                    break;
                }
                for _ in 0..4 {
                    let _ = r.next_message(&reg);
                }
                assert!(r.buffered() <= 50 + 10 + 4096);
            }
        }
    }
}
