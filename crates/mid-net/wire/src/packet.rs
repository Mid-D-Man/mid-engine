//! Packet shapes and wire codec for mid-net.
//!
//! Shapes are defined in `.mdix` under `packets/` (`player-state.mdix`,
//! `player-event.mdix`); the Rust types below are a hand-written mirror
//! of those shapes. Checked DixScript's actual Rust API (now published,
//! `dixscript` 1.0.0) — it's a dynamic accessor over parsed `.mdix`
//! (`data.get::<T>("path")`), not Rust struct codegen, so there's no
//! DixScript step this mirror is standing in for. The `.mdix` files
//! stay the authored source of truth for the shape; keep this file in
//! sync by hand if they change.
//!
//! Wire encoding is hand-rolled, explicit little-endian, zero external
//! dependencies — no `bincode`/`serde`. Two reasons: (1) mid-net's own
//! dependency mandate rules out pulling in a serialization crate when the
//! wire shapes are this small and fixed; (2) Ubel Stratum's LOW tier
//! (manual memory, FFI) is the eventual consumer of these bytes, and a
//! crate-specific reflection-based format doesn't give a non-Rust caller
//! anything to bind against — a flat byte layout does.
//!
//! Scope boundary: this module only turns a packet *payload* into bytes
//! and back. It does not write a kind tag, sequence number, or ack
//! bitfield — that framing is `reliable.rs`'s job, one layer up. By the
//! time `decode` is called here, the caller already knows which type
//! they're decoding (they read it off the frame header first).

use std::fmt;

/// One byte on the wire identifying which packet shape follows a frame
/// header. Owned here because it's intrinsic to "which packet shapes
/// exist", same as the `.mdix` files are.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum PacketKind {
    PlayerState = 0,
    PlayerEvent = 1,
}

impl PacketKind {
    pub fn from_u8(byte: u8) -> Option<Self> {
        match byte {
            0 => Some(PacketKind::PlayerState),
            1 => Some(PacketKind::PlayerEvent),
            _ => None,
        }
    }
}

/// Decode failures. Kept local to mid-net for now — mid-common's
/// `error.rs` is still an empty stub, so this isn't wired into a
/// workspace-wide error type yet. Small enough to migrate later without
/// touching call sites (just re-export from mid-common once it exists).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeError {
    /// Buffer ran out while a fixed-size or length-prefixed field was
    /// still being read.
    UnexpectedEnd,
    /// A length-prefixed string field wasn't valid UTF-8.
    InvalidUtf8,
    /// Buffer had bytes left over after every field was read. For a
    /// strict per-packet payload this means a framing bug upstream, not
    /// a recoverable condition — surfaced rather than silently ignored.
    TrailingBytes,
}

impl fmt::Display for DecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DecodeError::UnexpectedEnd => write!(f, "buffer ended before packet was fully read"),
            DecodeError::InvalidUtf8 => write!(f, "string field was not valid UTF-8"),
            DecodeError::TrailingBytes => write!(f, "buffer had unread bytes after decoding"),
        }
    }
}

impl std::error::Error for DecodeError {}

/// Read cursor over a byte slice. Private to this module — exists purely
/// to keep each packet's `decode` from hand-rolling offset arithmetic,
/// since an off-by-one here silently corrupts a field rather than
/// panicking (same class of bug the sequence-number wraparound logic
/// has to be careful about).
struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Cursor { buf, pos: 0 }
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], DecodeError> {
        let end = self.pos.checked_add(n).ok_or(DecodeError::UnexpectedEnd)?;
        let slice = self.buf.get(self.pos..end).ok_or(DecodeError::UnexpectedEnd)?;
        self.pos = end;
        Ok(slice)
    }

    fn read_u16_le(&mut self) -> Result<u16, DecodeError> {
        let b = self.take(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }

    fn read_u32_le(&mut self) -> Result<u32, DecodeError> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn read_f32_le(&mut self) -> Result<f32, DecodeError> {
        let b = self.take(4)?;
        Ok(f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    /// u16-length-prefixed UTF-8 string (max 65535 bytes — well past
    /// anything sane for an event name or payload at 128 Hz).
    fn read_string(&mut self) -> Result<String, DecodeError> {
        let len = self.read_u16_le()? as usize;
        let bytes = self.take(len)?;
        std::str::from_utf8(bytes)
            .map(str::to_owned)
            .map_err(|_| DecodeError::InvalidUtf8)
    }

    /// Must be called at the end of every `decode` — turns "silently
    /// ignored trailing bytes" into a surfaced error.
    fn finish(self) -> Result<(), DecodeError> {
        if self.pos == self.buf.len() {
            Ok(())
        } else {
            Err(DecodeError::TrailingBytes)
        }
    }
}

fn write_string(buf: &mut Vec<u8>, s: &str) {
    debug_assert!(s.len() <= u16::MAX as usize, "string field exceeds u16 length prefix");
    buf.extend_from_slice(&(s.len() as u16).to_le_bytes());
    buf.extend_from_slice(s.as_bytes());
}

/// Common interface for every wire packet type. `encode`/`decode` handle
/// only this packet's own fields — see the module doc for the framing
/// boundary.
pub trait Packet: Sized {
    const KIND: PacketKind;

    /// Append this packet's encoded fields to `buf`.
    fn encode(&self, buf: &mut Vec<u8>);

    /// Decode a payload previously produced by `encode`. `buf` must
    /// contain exactly one payload — leftover bytes are an error.
    fn decode(buf: &[u8]) -> Result<Self, DecodeError>;
}

/// Local stand-in for a shared player identifier. mid-common's
/// `types.rs` stub already documents wanting a `PlayerId` — this can
/// move there and get re-exported once that crate has real content;
/// not doing that migration today since it'd add a cross-crate
/// dependency edge that's a separate decision from "write packet.rs".
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PlayerId(pub u32);

/// Unreliable channel, sent every tick at 128 Hz (`packets/player-state.mdix`).
/// Loss is fine — the next tick's packet supersedes it. Fixed wire size,
/// no length-prefixed fields, so encode/decode cost is just 7 stores/loads.
///
/// `repr(C)`: all fields are plain `f32`, already C-compatible, so this
/// crosses the FFI boundary by value directly — `ffi.rs` doesn't need a
/// separate mirror struct kept in sync by hand. `PlayerEvent` below can't
/// do this (it owns `String`s, not C-representable), so it gets an
/// opaque-handle FFI wrapper instead; see `ffi.rs`.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct PlayerState {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub rot_x: f32,
    pub rot_y: f32,
    pub rot_z: f32,
    pub rot_w: f32,
}

/// Exact wire size in bytes: 7 × f32. Asserted by a test below — if this
/// ever drifts from the struct's actual encode output, that's a bug, not
/// a spec change.
pub const PLAYER_STATE_WIRE_SIZE: usize = 28;

impl Default for PlayerState {
    fn default() -> Self {
        // Matches packets/player-state.mdix: position at origin, identity
        // quaternion (w = 1).
        PlayerState { x: 0.0, y: 0.0, z: 0.0, rot_x: 0.0, rot_y: 0.0, rot_z: 0.0, rot_w: 1.0 }
    }
}

impl Packet for PlayerState {
    const KIND: PacketKind = PacketKind::PlayerState;

    fn encode(&self, buf: &mut Vec<u8>) {
        buf.extend_from_slice(&self.x.to_le_bytes());
        buf.extend_from_slice(&self.y.to_le_bytes());
        buf.extend_from_slice(&self.z.to_le_bytes());
        buf.extend_from_slice(&self.rot_x.to_le_bytes());
        buf.extend_from_slice(&self.rot_y.to_le_bytes());
        buf.extend_from_slice(&self.rot_z.to_le_bytes());
        buf.extend_from_slice(&self.rot_w.to_le_bytes());
    }

    fn decode(buf: &[u8]) -> Result<Self, DecodeError> {
        let mut c = Cursor::new(buf);
        let state = PlayerState {
            x: c.read_f32_le()?,
            y: c.read_f32_le()?,
            z: c.read_f32_le()?,
            rot_x: c.read_f32_le()?,
            rot_y: c.read_f32_le()?,
            rot_z: c.read_f32_le()?,
            rot_w: c.read_f32_le()?,
        };
        c.finish()?;
        Ok(state)
    }
}

/// Reliable channel — join, pickup, damage, etc. (`packets/player-event.mdix`).
/// Must arrive; sequence number + ack/retransmit is added by `reliable.rs`,
/// not here.
///
/// `event` and `payload` are both plain UTF-8 strings, matching the
/// literal `.mdix` defaults (`event = "unknown"`, `payload = ""`) — the
/// `.mdix` format has no separate bytes/string type to infer from, it
/// just has the literal. Worth confirming: a lot of real event payloads
/// (damage amount, item id, etc.) are more naturally opaque bytes than
/// UTF-8 text, so `payload: Vec<u8>` may be the better long-term shape.
/// Kept as `String` for now to match the schema as written; this is a
/// cheap field-type change later if the answer is "bytes".
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlayerEvent {
    pub player_id: PlayerId,
    pub event: String,
    pub payload: String,
}

impl Default for PlayerEvent {
    fn default() -> Self {
        PlayerEvent { player_id: PlayerId(0), event: "unknown".to_string(), payload: String::new() }
    }
}

impl Packet for PlayerEvent {
    const KIND: PacketKind = PacketKind::PlayerEvent;

    fn encode(&self, buf: &mut Vec<u8>) {
        buf.extend_from_slice(&self.player_id.0.to_le_bytes());
        write_string(buf, &self.event);
        write_string(buf, &self.payload);
    }

    fn decode(buf: &[u8]) -> Result<Self, DecodeError> {
        let mut c = Cursor::new(buf);
        let event = PlayerEvent {
            player_id: PlayerId(c.read_u32_le()?),
            event: c.read_string()?,
            payload: c.read_string()?,
        };
        c.finish()?;
        Ok(event)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn player_state_round_trips() {
        let s = PlayerState {
            x: 1.5,
            y: -2.25,
            z: 100.0,
            rot_x: 0.0,
            rot_y: 0.707,
            rot_z: 0.0,
            rot_w: 0.707,
        };
        let mut buf = Vec::new();
        s.encode(&mut buf);
        assert_eq!(buf.len(), PLAYER_STATE_WIRE_SIZE);
        let decoded = PlayerState::decode(&buf).expect("decode should succeed");
        assert_eq!(s, decoded);
    }

    #[test]
    fn player_state_default_round_trips() {
        let s = PlayerState::default();
        let mut buf = Vec::new();
        s.encode(&mut buf);
        assert_eq!(buf.len(), PLAYER_STATE_WIRE_SIZE);
        assert_eq!(PlayerState::decode(&buf).unwrap(), s);
    }

    #[test]
    fn player_state_wire_size_is_exact() {
        let mut buf = Vec::new();
        PlayerState::default().encode(&mut buf);
        assert_eq!(buf.len(), PLAYER_STATE_WIRE_SIZE, "PLAYER_STATE_WIRE_SIZE drifted from actual encode output");
    }

    #[test]
    fn player_state_decode_rejects_truncated_buffer() {
        let mut buf = Vec::new();
        PlayerState::default().encode(&mut buf);
        buf.truncate(buf.len() - 1);
        assert_eq!(PlayerState::decode(&buf), Err(DecodeError::UnexpectedEnd));
    }

    #[test]
    fn player_state_decode_rejects_trailing_bytes() {
        let mut buf = Vec::new();
        PlayerState::default().encode(&mut buf);
        buf.push(0xFF);
        assert_eq!(PlayerState::decode(&buf), Err(DecodeError::TrailingBytes));
    }

    #[test]
    fn player_event_round_trips() {
        let e = PlayerEvent {
            player_id: PlayerId(42),
            event: "pickup".to_string(),
            payload: "item_id=17".to_string(),
        };
        let mut buf = Vec::new();
        e.encode(&mut buf);
        let decoded = PlayerEvent::decode(&buf).expect("decode should succeed");
        assert_eq!(e, decoded);
    }

    #[test]
    fn player_event_default_round_trips() {
        let e = PlayerEvent::default();
        let mut buf = Vec::new();
        e.encode(&mut buf);
        assert_eq!(PlayerEvent::decode(&buf).unwrap(), e);
    }

    #[test]
    fn player_event_handles_empty_strings() {
        let e = PlayerEvent { player_id: PlayerId(1), event: String::new(), payload: String::new() };
        let mut buf = Vec::new();
        e.encode(&mut buf);
        // 4 (player_id) + 2 (event len=0) + 2 (payload len=0) = 8 bytes, no string bytes.
        assert_eq!(buf.len(), 8);
        assert_eq!(PlayerEvent::decode(&buf).unwrap(), e);
    }

    #[test]
    fn player_event_decode_rejects_bad_utf8() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&1u32.to_le_bytes()); // player_id
        buf.extend_from_slice(&1u16.to_le_bytes()); // event len = 1
        buf.push(0xFF); // invalid utf8 byte
        buf.extend_from_slice(&0u16.to_le_bytes()); // payload len = 0
        assert_eq!(PlayerEvent::decode(&buf), Err(DecodeError::InvalidUtf8));
    }

    #[test]
    fn player_event_decode_rejects_truncated_string() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&10u16.to_le_bytes()); // claims 10 bytes follow
        buf.extend_from_slice(b"short"); // only 5 provided
        assert_eq!(PlayerEvent::decode(&buf), Err(DecodeError::UnexpectedEnd));
    }

    #[test]
    fn packet_kind_round_trips_through_u8() {
        assert_eq!(PacketKind::from_u8(0), Some(PacketKind::PlayerState));
        assert_eq!(PacketKind::from_u8(1), Some(PacketKind::PlayerEvent));
        assert_eq!(PacketKind::from_u8(2), None);
        assert_eq!(PlayerState::KIND as u8, 0);
        assert_eq!(PlayerEvent::KIND as u8, 1);
    }

    #[test]
    fn distinct_player_states_do_not_collide() {
        // Sanity check against a silent field-order bug: two different
        // states must not encode to the same bytes.
        let a = PlayerState { x: 1.0, ..PlayerState::default() };
        let b = PlayerState { y: 1.0, ..PlayerState::default() };
        let mut ba = Vec::new();
        let mut bb = Vec::new();
        a.encode(&mut ba);
        b.encode(&mut bb);
        assert_ne!(ba, bb);
    }
    }
