//! Pure byte framing for the reliable-stream wire format (see the
//! crate-level doc comment in `lib.rs`). No `web-sys`, no `wasm-bindgen`,
//! no I/O -- kept free of both so this compiles and is genuinely tested
//! on any host, not just a real `wasm32-unknown-unknown` build.
//!
//! **Deliberately byte-identical to `mid-net-transport-quinn`'s own
//! `framing` module**, not just similar: a real deployment will have
//! native (`quinn`) peers and browser (`wasm`) peers talking to each
//! other over the same protocol, so the stream_id header + length-prefix
//! shape has to match exactly on both sides. Duplicated rather than
//! shared via a common crate — this is four small functions, and pulling
//! a shared `mid-net-framing` crate into existence for them would add a
//! workspace member for less code than the crate boilerplate itself,
//! not worth it at this size. If it ever drifts, the tests in both
//! crates (little-endian assertions included) are what would catch it.

/// The first 4 bytes written to a freshly opened uni stream: which
/// caller-chosen `stream_id` this stream carries.
pub fn stream_header(stream_id: u32) -> [u8; 4] {
    stream_id.to_le_bytes()
}

pub fn decode_stream_header(buf: [u8; 4]) -> u32 {
    u32::from_le_bytes(buf)
}

/// One `send_reliable` payload on the wire: a 4-byte little-endian length
/// prefix followed by the payload bytes.
pub fn encode_frame(payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(4 + payload.len());
    out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    out.extend_from_slice(payload);
    out
}

pub fn decode_frame_len(buf: [u8; 4]) -> u32 {
    u32::from_le_bytes(buf)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stream_header_round_trips() {
        assert_eq!(decode_stream_header(stream_header(7)), 7);
        assert_eq!(decode_stream_header(stream_header(u32::MAX)), u32::MAX);
        assert_eq!(decode_stream_header(stream_header(0)), 0);
    }

    #[test]
    fn stream_header_is_little_endian() {
        assert_eq!(stream_header(1), [1, 0, 0, 0]);
        assert_eq!(stream_header(256), [0, 1, 0, 0]);
    }

    #[test]
    fn encode_frame_prefixes_little_endian_length() {
        let frame = encode_frame(&[9, 9, 9]);
        assert_eq!(&frame[0..4], &[3, 0, 0, 0]);
        assert_eq!(&frame[4..], &[9, 9, 9]);
    }

    #[test]
    fn encode_frame_handles_empty_payload() {
        assert_eq!(encode_frame(&[]), vec![0, 0, 0, 0]);
    }

    #[test]
    fn frame_len_round_trips_through_the_prefix() {
        let frame = encode_frame(&[1, 2, 3, 4, 5]);
        let len_buf: [u8; 4] = frame[0..4].try_into().unwrap();
        assert_eq!(decode_frame_len(len_buf), 5);
    }

    #[test]
    fn matches_mid_net_transport_quinn_s_wire_shape() {
        // Not importing that crate (it's native-only, pulling it in here
        // would defeat the point of this crate's own wasm32 gating) --
        // this test instead pins the exact byte layout by hand, so a
        // change to either crate's framing that breaks cross-backend
        // compatibility fails a test in both places, not silently.
        assert_eq!(stream_header(42), [42, 0, 0, 0]);
        assert_eq!(encode_frame(b"hi"), vec![2, 0, 0, 0, b'h', b'i']);
    }
}
