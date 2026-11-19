//! Bit-level writer and reader for packed network messages.
//!
//! This is the base of every typed message above the wire: a message says
//! how many bits each field needs (a bool is 1, a health value in
//! 0..=100 is 7, a position quantised to 1 cm over 1 km is 17) and the
//! writer packs them with no padding between fields, so a packet carries
//! what it must and nothing else. At 128 ticks a second (`docs/mid-net.md`,
//! "Packet Budget") the saving is the whole point.
//!
//! # The contract (pinned by golden-vector tests, because other languages
//! must reproduce it exactly)
//! * Bits are packed **least significant bit first**: the first bit written
//!   is bit 0 of byte 0, and a multi-bit value is written low bits first.
//!   `write_bits(0b101, 3)` then `write_bool(true)` is byte `0b0000_1101`.
//! * Varints are 7 data bits plus a continuation bit per group, low group
//!   first (LEB128), written bit-wise, so they need not be byte aligned.
//!   A `u64` takes at most 10 groups; the reader rejects an 11th group and
//!   any bits that do not fit 64.
//! * Signed varints are zigzag (`0, -1, 1, -2` are `0, 1, 2, 3`).
//! * A quantised float is `round((v - min) / (max - min) * (2^bits - 1))`
//!   (round half away from zero, in `f64`), and reads back as
//!   `min + q / (2^bits - 1) * (max - min)`. The error is at most half a
//!   step ([`quantization_step`]). NaN and infinities are refused, never
//!   silently turned into numbers; a finite value outside the range is
//!   clamped to it (a wall of a map is the edge of the range).
//! * A unit vector is written octahedrally: two quantised components in
//!   `-1..=1` (`bits` each). It reads back normalised.
//! * No allocation: the writer fills a slice the caller owns (an MTU-sized
//!   array on the stack), and the reader borrows the bytes it is given.
//!
//! Nothing here can panic on bad input: every write that does not fit, and
//! every read past the end, is an `Err` (a datagram from the network is
//! untrusted, so a reader is run over arbitrary bytes in the tests).

use core::fmt;

/// Why a write or a read did not happen. Nothing is written or consumed
/// past the point of failure that could be mistaken for success: a failed
/// call leaves the writer or reader usable but the caller should drop the
/// packet.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BitError {
    /// The writer's buffer is full, or the reader ran out of bits.
    Overflow,
    /// The value does not fit the number of bits asked for, or the number of
    /// bits is not allowed (0, or more than the type holds).
    OutOfRange,
    /// A float that is NaN or infinite, or a range with `min >= max`, or a zero vector.
    NotFinite,
    /// A varint longer than a `u64` can be, or with bits past the 64th.
    BadVarint,
    /// A string that is not UTF-8, or longer than the limit the caller gave.
    BadString,
}

impl fmt::Display for BitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            BitError::Overflow => "out of space",
            BitError::OutOfRange => "value or bit count out of range",
            BitError::NotFinite => "float not finite, bad range or zero vector",
            BitError::BadVarint => "malformed varint",
            BitError::BadString => "malformed string",
        })
    }
}

/// How coarse a quantised float is: the distance between two neighbouring values it can hold.
/// The most a value can be off after a round trip is half of this.
pub fn quantization_step(min: f32, max: f32, bits: u32) -> f64 {
    let levels = ((1u64 << bits.clamp(1, 32)) - 1) as f64;
    (max as f64 - min as f64) / levels
}

/// Packs bits into a caller-owned buffer, first bit into bit 0 of byte 0.
pub struct BitWriter<'a> {
    buf: &'a mut [u8],
    pos: usize, // bits written
}

impl<'a> BitWriter<'a> {
    /// Starts at the front of `buf`; the bytes in it are overwritten as they are reached.
    pub fn new(buf: &'a mut [u8]) -> Self {
        BitWriter { buf, pos: 0 }
    }

    /// Bits written so far.
    pub fn bits_written(&self) -> usize {
        self.pos
    }

    /// Whole bytes the packet is so far (a partly used last byte counts).
    pub fn bytes_written(&self) -> usize {
        self.pos.div_ceil(8)
    }

    /// Bits that can still be written.
    pub fn bits_free(&self) -> usize {
        self.buf.len() * 8 - self.pos
    }

    /// The bytes written so far (the last one zero-padded).
    pub fn written(&self) -> &[u8] {
        &self.buf[..self.bytes_written()]
    }

    /// Writes the low `bits` (1 to 64) of `value`. A value with set bits beyond `bits` is `OutOfRange`.
    pub fn write_bits(&mut self, value: u64, bits: u32) -> Result<(), BitError> {
        if bits == 0 || bits > 64 || (bits < 64 && value >> bits != 0) {
            return Err(BitError::OutOfRange);
        }
        if self.bits_free() < bits as usize {
            return Err(BitError::Overflow);
        }
        let mut left = bits as usize;
        let mut v = value;
        while left > 0 {
            let byte = self.pos / 8;
            let off = self.pos % 8;
            let take = (8 - off).min(left);
            let chunk = (v & ((1u64 << take) - 1)) as u8;
            if off == 0 {
                self.buf[byte] = chunk; // a fresh byte: clears whatever was there
            } else {
                self.buf[byte] |= chunk << off;
            }
            v >>= take;
            self.pos += take;
            left -= take;
        }
        Ok(())
    }

    pub fn write_bool(&mut self, v: bool) -> Result<(), BitError> {
        self.write_bits(v as u64, 1)
    }
    pub fn write_u8(&mut self, v: u8) -> Result<(), BitError> {
        self.write_bits(v as u64, 8)
    }
    pub fn write_u16(&mut self, v: u16) -> Result<(), BitError> {
        self.write_bits(v as u64, 16)
    }
    pub fn write_u32(&mut self, v: u32) -> Result<(), BitError> {
        self.write_bits(v as u64, 32)
    }
    pub fn write_u64(&mut self, v: u64) -> Result<(), BitError> {
        self.write_bits(v, 64)
    }

    /// A float as its 32 raw bits (for values that must come back exactly).
    pub fn write_f32(&mut self, v: f32) -> Result<(), BitError> {
        self.write_bits(v.to_bits() as u64, 32)
    }

    /// An unsigned integer in as few 7-bit groups as it needs.
    pub fn write_varint(&mut self, mut v: u64) -> Result<(), BitError> {
        // Check the space first so a failure leaves nothing half written.
        if self.bits_free() < varint_groups(v) * 8 {
            return Err(BitError::Overflow);
        }
        loop {
            let low = (v & 0x7f) as u8;
            v >>= 7;
            if v == 0 {
                return self.write_bits(low as u64, 8);
            }
            self.write_bits((low | 0x80) as u64, 8)?;
        }
    }

    /// A signed integer, zigzag-coded, as a varint (small magnitudes of either sign are short).
    pub fn write_zigzag(&mut self, v: i64) -> Result<(), BitError> {
        self.write_varint(((v << 1) ^ (v >> 63)) as u64)
    }

    /// A float in `min..=max` in `bits` bits (1 to 32). Finite values outside the range are clamped to it.
    pub fn write_f32_quantized(
        &mut self,
        v: f32,
        min: f32,
        max: f32,
        bits: u32,
    ) -> Result<(), BitError> {
        if bits == 0 || bits > 32 {
            return Err(BitError::OutOfRange);
        }
        if !v.is_finite() || !min.is_finite() || !max.is_finite() || min >= max {
            return Err(BitError::NotFinite);
        }
        let q = quantize(v as f64, min as f64, max as f64, bits);
        self.write_bits(q, bits)
    }

    /// A unit vector in `2 * bits` bits (octahedral), `bits` 2 to 16. The vector need not be exactly
    /// normalised (it is normalised first); a zero or non-finite vector is `NotFinite`.
    pub fn write_unit_vec3(&mut self, x: f32, y: f32, z: f32, bits: u32) -> Result<(), BitError> {
        if !(2..=16).contains(&bits) {
            return Err(BitError::OutOfRange);
        }
        let (u, v) = oct_encode(x, y, z).ok_or(BitError::NotFinite)?;
        if self.bits_free() < 2 * bits as usize {
            return Err(BitError::Overflow);
        }
        self.write_bits(quantize(u, -1.0, 1.0, bits), bits)?;
        self.write_bits(quantize(v, -1.0, 1.0, bits), bits)
    }

    /// Skips to the next byte boundary (the skipped bits are zero).
    pub fn align_to_byte(&mut self) -> Result<(), BitError> {
        let pad = (8 - self.pos % 8) % 8;
        if pad == 0 {
            return Ok(());
        }
        self.write_bits(0, pad as u32)
    }

    /// Whole bytes, after aligning to a byte boundary.
    pub fn write_bytes(&mut self, bytes: &[u8]) -> Result<(), BitError> {
        let pad = (8 - self.pos % 8) % 8;
        if self.bits_free() < pad + bytes.len() * 8 {
            return Err(BitError::Overflow);
        }
        self.align_to_byte()?;
        let at = self.pos / 8;
        self.buf[at..at + bytes.len()].copy_from_slice(bytes);
        self.pos += bytes.len() * 8;
        Ok(())
    }

    /// A UTF-8 string: its byte length as a varint, then the bytes (aligned). Either all of it is written or none.
    pub fn write_str(&mut self, s: &str) -> Result<(), BitError> {
        let pad = (8 - self.pos % 8) % 8; // the varint is whole bytes, so the padding after it is the same
        let need = (varint_groups(s.len() as u64) * 8)
            .saturating_add(pad)
            .saturating_add(s.len().saturating_mul(8));
        if self.bits_free() < need {
            return Err(BitError::Overflow);
        }
        self.write_varint(s.len() as u64)?;
        self.write_bytes(s.as_bytes())
    }
}

/// Reads what a [`BitWriter`] wrote. Borrows the bytes; never panics on any input.
pub struct BitReader<'a> {
    buf: &'a [u8],
    pos: usize, // bits read
}

impl<'a> BitReader<'a> {
    pub fn new(buf: &'a [u8]) -> Self {
        BitReader { buf, pos: 0 }
    }

    pub fn bits_read(&self) -> usize {
        self.pos
    }

    /// Bits left (counting the zero padding of a last partly used byte).
    pub fn bits_remaining(&self) -> usize {
        self.buf.len() * 8 - self.pos
    }

    pub fn read_bits(&mut self, bits: u32) -> Result<u64, BitError> {
        if bits == 0 || bits > 64 {
            return Err(BitError::OutOfRange);
        }
        if self.bits_remaining() < bits as usize {
            return Err(BitError::Overflow);
        }
        let mut out = 0u64;
        let mut got = 0usize;
        let mut left = bits as usize;
        while left > 0 {
            let byte = self.pos / 8;
            let off = self.pos % 8;
            let take = (8 - off).min(left);
            let chunk = ((self.buf[byte] >> off) as u64) & ((1u64 << take) - 1);
            out |= chunk << got;
            got += take;
            self.pos += take;
            left -= take;
        }
        Ok(out)
    }

    pub fn read_bool(&mut self) -> Result<bool, BitError> {
        Ok(self.read_bits(1)? == 1)
    }
    pub fn read_u8(&mut self) -> Result<u8, BitError> {
        Ok(self.read_bits(8)? as u8)
    }
    pub fn read_u16(&mut self) -> Result<u16, BitError> {
        Ok(self.read_bits(16)? as u16)
    }
    pub fn read_u32(&mut self) -> Result<u32, BitError> {
        Ok(self.read_bits(32)? as u32)
    }
    pub fn read_u64(&mut self) -> Result<u64, BitError> {
        self.read_bits(64)
    }

    /// A float from its 32 raw bits. Any bit pattern is accepted (a NaN from the network is the caller's to refuse).
    pub fn read_f32(&mut self) -> Result<f32, BitError> {
        Ok(f32::from_bits(self.read_bits(32)? as u32))
    }

    pub fn read_varint(&mut self) -> Result<u64, BitError> {
        let mut out = 0u64;
        for group in 0..10u32 {
            let b = self.read_bits(8)?;
            let data = b & 0x7f;
            let shift = group * 7;
            if shift == 63 && data > 1 {
                return Err(BitError::BadVarint); // bits past the 64th
            }
            out |= data << shift;
            if b & 0x80 == 0 {
                return Ok(out);
            }
        }
        Err(BitError::BadVarint) // an 11th group
    }

    pub fn read_zigzag(&mut self) -> Result<i64, BitError> {
        let u = self.read_varint()?;
        Ok(((u >> 1) as i64) ^ -((u & 1) as i64))
    }

    /// A float read back from `bits` bits in `min..=max` (the same arguments the writer used).
    pub fn read_f32_quantized(&mut self, min: f32, max: f32, bits: u32) -> Result<f32, BitError> {
        if bits == 0 || bits > 32 {
            return Err(BitError::OutOfRange);
        }
        if !min.is_finite() || !max.is_finite() || min >= max {
            return Err(BitError::NotFinite);
        }
        let q = self.read_bits(bits)?;
        Ok(dequantize(q, min as f64, max as f64, bits) as f32)
    }

    /// A unit vector written by [`BitWriter::write_unit_vec3`] with the same `bits`.
    pub fn read_unit_vec3(&mut self, bits: u32) -> Result<(f32, f32, f32), BitError> {
        if !(2..=16).contains(&bits) {
            return Err(BitError::OutOfRange);
        }
        let u = dequantize(self.read_bits(bits)?, -1.0, 1.0, bits);
        let v = dequantize(self.read_bits(bits)?, -1.0, 1.0, bits);
        Ok(oct_decode(u, v))
    }

    pub fn align_to_byte(&mut self) -> Result<(), BitError> {
        let pad = (8 - self.pos % 8) % 8;
        if pad == 0 {
            return Ok(());
        }
        self.read_bits(pad as u32).map(|_| ())
    }

    /// `n` whole bytes after aligning to a byte boundary, borrowed from the packet.
    pub fn read_bytes(&mut self, n: usize) -> Result<&'a [u8], BitError> {
        let pad = (8 - self.pos % 8) % 8;
        let need = pad
            .checked_add(n.checked_mul(8).ok_or(BitError::Overflow)?)
            .ok_or(BitError::Overflow)?;
        if self.bits_remaining() < need {
            return Err(BitError::Overflow);
        }
        self.align_to_byte()?;
        let at = self.pos / 8;
        self.pos += n * 8;
        Ok(&self.buf[at..at + n])
    }

    /// A string written by [`BitWriter::write_str`], at most `max_len` bytes long.
    pub fn read_str(&mut self, max_len: usize) -> Result<&'a str, BitError> {
        let len = self.read_varint()?;
        if len > max_len as u64 {
            return Err(BitError::BadString);
        }
        let bytes = self.read_bytes(len as usize)?;
        core::str::from_utf8(bytes).map_err(|_| BitError::BadString)
    }
}

fn varint_groups(mut v: u64) -> usize {
    let mut groups = 1usize;
    v >>= 7;
    while v != 0 {
        groups += 1;
        v >>= 7;
    }
    groups
}

fn quantize(v: f64, min: f64, max: f64, bits: u32) -> u64 {
    let levels = ((1u64 << bits) - 1) as f64;
    let t = ((v - min) / (max - min)).clamp(0.0, 1.0);
    (t * levels).round() as u64
}

fn dequantize(q: u64, min: f64, max: f64, bits: u32) -> f64 {
    let levels = ((1u64 << bits) - 1) as f64;
    min + (q as f64 / levels) * (max - min)
}

// Octahedral mapping: the unit sphere folded onto a square. (Meyer et al., "On Floating-Point Normal Vectors".)
fn oct_encode(x: f32, y: f32, z: f32) -> Option<(f64, f64)> {
    if !x.is_finite() || !y.is_finite() || !z.is_finite() {
        return None;
    }
    let (x, y, z) = (x as f64, y as f64, z as f64);
    let l1 = x.abs() + y.abs() + z.abs();
    if l1 < 1e-30 {
        return None;
    }
    let (mut u, mut v) = (x / l1, y / l1);
    if z < 0.0 {
        let (ou, ov) = (u, v);
        u = (1.0 - ov.abs()) * if ou >= 0.0 { 1.0 } else { -1.0 };
        v = (1.0 - ou.abs()) * if ov >= 0.0 { 1.0 } else { -1.0 };
    }
    Some((u, v))
}

fn oct_decode(u: f64, v: f64) -> (f32, f32, f32) {
    let mut x = u;
    let mut y = v;
    let z = 1.0 - u.abs() - v.abs();
    let mut zz = z;
    if z < 0.0 {
        x = (1.0 - v.abs()) * if u >= 0.0 { 1.0 } else { -1.0 };
        y = (1.0 - u.abs()) * if v >= 0.0 { 1.0 } else { -1.0 };
        zz = z;
    }
    let len = (x * x + y * y + zz * zz).sqrt();
    if len < 1e-30 {
        return (0.0, 0.0, 1.0);
    }
    ((x / len) as f32, (y / len) as f32, (zz / len) as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    // A small deterministic generator, so a failing case can be rerun.
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
        fn unit(&mut self) -> f64 {
            (self.next() >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    // The angle between two vectors, by atan2 of the cross product and the dot (acos is badly conditioned for nearly equal directions).
    fn angle(a: (f32, f32, f32), b: (f32, f32, f32)) -> f64 {
        let (ax, ay, az) = (a.0 as f64, a.1 as f64, a.2 as f64);
        let (bx, by, bz) = (b.0 as f64, b.1 as f64, b.2 as f64);
        let cx = ay * bz - az * by;
        let cy = az * bx - ax * bz;
        let cz = ax * by - ay * bx;
        (cx * cx + cy * cy + cz * cz)
            .sqrt()
            .atan2(ax * bx + ay * by + az * bz)
    }

    fn mask(bits: u32) -> u64 {
        if bits == 64 {
            u64::MAX
        } else {
            (1u64 << bits) - 1
        }
    }

    // ---- golden vectors: the contract other languages must reproduce ------------------------------------------

    #[test]
    fn golden_bit_packing_is_least_significant_bit_first() {
        let mut b = [0u8; 8];
        let mut w = BitWriter::new(&mut b);
        w.write_bits(0b101, 3).unwrap();
        w.write_bool(true).unwrap();
        assert_eq!(w.written(), &[0b0000_1101]);

        let mut b = [0xffu8; 8]; // garbage in the buffer must not leak into what is written
        let mut w = BitWriter::new(&mut b);
        w.write_bits(0xABC, 12).unwrap();
        w.write_bits(0x5, 3).unwrap();
        assert_eq!(w.written(), &[0xBC, 0x5A]);
        assert_eq!(w.bits_written(), 15);

        let mut b = [0u8; 8];
        let mut w = BitWriter::new(&mut b);
        w.write_u16(0x1234).unwrap();
        w.write_u32(0xDEADBEEF).unwrap();
        assert_eq!(w.written(), &[0x34, 0x12, 0xEF, 0xBE, 0xAD, 0xDE]);
    }

    #[test]
    fn golden_varints_and_zigzag() {
        let cases: &[(u64, &[u8])] = &[
            (0, &[0x00]),
            (127, &[0x7F]),
            (128, &[0x80, 0x01]),
            (300, &[0xAC, 0x02]),
            (
                u64::MAX,
                &[0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x01],
            ),
        ];
        for (v, bytes) in cases {
            let mut b = [0u8; 16];
            let mut w = BitWriter::new(&mut b);
            w.write_varint(*v).unwrap();
            assert_eq!(w.written(), *bytes, "varint {v}");
            let mut r = BitReader::new(bytes);
            assert_eq!(r.read_varint().unwrap(), *v);
        }
        for (v, z) in [
            (0i64, 0u8),
            (-1, 1),
            (1, 2),
            (-2, 3),
            (2, 4),
            (63, 126),
            (-64, 127),
        ] {
            let mut b = [0u8; 4];
            let mut w = BitWriter::new(&mut b);
            w.write_zigzag(v).unwrap();
            assert_eq!(w.written(), &[z], "zigzag {v}");
        }
        let mut b = [0u8; 32];
        let mut w = BitWriter::new(&mut b);
        w.write_zigzag(i64::MIN).unwrap();
        w.write_zigzag(i64::MAX).unwrap();
        let mut r = BitReader::new(w.written());
        assert_eq!(r.read_zigzag().unwrap(), i64::MIN);
        assert_eq!(r.read_zigzag().unwrap(), i64::MAX);
    }

    #[test]
    fn golden_quantised_values() {
        let mut b = [0u8; 8];
        let mut w = BitWriter::new(&mut b);
        w.write_f32_quantized(0.5, 0.0, 1.0, 8).unwrap(); // 127.5 rounds half away from zero: 128
        w.write_f32_quantized(0.0, 0.0, 1.0, 8).unwrap();
        w.write_f32_quantized(1.0, 0.0, 1.0, 8).unwrap();
        w.write_f32_quantized(-5.0, -10.0, 10.0, 4).unwrap(); // a quarter of the way: 3.75 -> 4
        assert_eq!(w.written(), &[0x80, 0x00, 0xFF, 0x04]);
        let mut w2 = BitWriter::new(&mut b);
        w2.write_unit_vec3(0.0, 0.0, 1.0, 8).unwrap();
        assert_eq!(w2.written(), &[0x80, 0x80]);
    }

    // ---- round trips -----------------------------------------------------------------------------------------

    #[test]
    fn every_width_round_trips_at_every_bit_offset() {
        for offset in 0..8u32 {
            for bits in 1..=64u32 {
                let values = [
                    0u64,
                    1,
                    mask(bits),
                    mask(bits) >> 1,
                    0xAAAA_AAAA_AAAA_AAAA & mask(bits),
                    0x5555_5555_5555_5555 & mask(bits),
                ];
                for v in values {
                    let mut b = [0u8; 24];
                    let mut w = BitWriter::new(&mut b);
                    if offset > 0 {
                        w.write_bits(mask(offset), offset).unwrap();
                    }
                    w.write_bits(v, bits).unwrap();
                    w.write_bits(0b10, 2).unwrap(); // and what follows is undisturbed
                    let mut r = BitReader::new(w.written());
                    if offset > 0 {
                        assert_eq!(r.read_bits(offset).unwrap(), mask(offset));
                    }
                    assert_eq!(
                        r.read_bits(bits).unwrap(),
                        v,
                        "bits {bits} offset {offset} value {v:#x}"
                    );
                    assert_eq!(r.read_bits(2).unwrap(), 0b10);
                }
            }
        }
    }

    #[test]
    fn a_long_random_mix_of_every_kind_of_field_comes_back() {
        #[derive(Debug)]
        enum Op {
            Bits(u64, u32),
            Bool(bool),
            Var(u64),
            Zig(i64),
            F32(f32),
            Quant(f32, f32, f32, u32),
            Vec3(f32, f32, f32, u32),
            Str(String),
            Bytes(Vec<u8>),
        }
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        for round in 0..3000 {
            let mut ops = Vec::new();
            for _ in 0..(1 + rng.below(12)) {
                ops.push(match rng.below(9) {
                    0 => {
                        let bits = 1 + rng.below(64) as u32;
                        Op::Bits(rng.next() & mask(bits), bits)
                    }
                    1 => Op::Bool(rng.below(2) == 1),
                    2 => Op::Var(rng.next() >> rng.below(64)),
                    3 => Op::Zig((rng.next() as i64) >> rng.below(64)),
                    4 => Op::F32(f32::from_bits(rng.next() as u32)),
                    5 => {
                        let bits = 1 + rng.below(32) as u32;
                        let min = (rng.unit() * 200.0 - 100.0) as f32;
                        let max = min + 0.5 + (rng.unit() * 500.0) as f32;
                        Op::Quant(min + (max - min) * rng.unit() as f32, min, max, bits)
                    }
                    6 => {
                        let bits = 2 + rng.below(15) as u32;
                        Op::Vec3(
                            (rng.unit() * 2.0 - 1.0) as f32,
                            (rng.unit() * 2.0 - 1.0) as f32,
                            (rng.unit() * 2.0 - 1.0) as f32 + 0.01,
                            bits,
                        )
                    }
                    7 => {
                        let n = rng.below(20) as usize;
                        Op::Str("aé✓x".chars().cycle().take(n).collect())
                    }
                    _ => {
                        let n = rng.below(16) as usize;
                        Op::Bytes((0..n).map(|_| rng.next() as u8).collect())
                    }
                });
            }
            let mut buf = [0xA5u8; 1024];
            let mut w = BitWriter::new(&mut buf);
            for op in &ops {
                match op {
                    Op::Bits(v, b) => w.write_bits(*v, *b).unwrap(),
                    Op::Bool(v) => w.write_bool(*v).unwrap(),
                    Op::Var(v) => w.write_varint(*v).unwrap(),
                    Op::Zig(v) => w.write_zigzag(*v).unwrap(),
                    Op::F32(v) => w.write_f32(*v).unwrap(),
                    Op::Quant(v, lo, hi, b) => w.write_f32_quantized(*v, *lo, *hi, *b).unwrap(),
                    Op::Vec3(x, y, z, b) => w.write_unit_vec3(*x, *y, *z, *b).unwrap(),
                    Op::Str(s) => w.write_str(s).unwrap(),
                    Op::Bytes(v) => w.write_bytes(v).unwrap(),
                }
            }
            let mut r = BitReader::new(w.written());
            for op in &ops {
                match op {
                    Op::Bits(v, b) => {
                        assert_eq!(r.read_bits(*b).unwrap(), *v, "round {round} {op:?}")
                    }
                    Op::Bool(v) => assert_eq!(r.read_bool().unwrap(), *v),
                    Op::Var(v) => assert_eq!(r.read_varint().unwrap(), *v),
                    Op::Zig(v) => assert_eq!(r.read_zigzag().unwrap(), *v),
                    Op::F32(v) => assert_eq!(r.read_f32().unwrap().to_bits(), v.to_bits()),
                    Op::Quant(v, lo, hi, b) => {
                        let got = r.read_f32_quantized(*lo, *hi, *b).unwrap();
                        let step = quantization_step(*lo, *hi, *b);
                        assert!(
                            ((got as f64) - (*v as f64)).abs()
                                <= step / 2.0 + (hi - lo) as f64 * 1e-6,
                            "round {round} {op:?} got {got}"
                        );
                    }
                    Op::Vec3(x, y, z, b) => {
                        let (gx, gy, gz) = r.read_unit_vec3(*b).unwrap();
                        let ang = angle((gx, gy, gz), (*x, *y, *z));
                        let step = 2.0 / (((1u64 << *b) - 1) as f64);
                        assert!(
                            ang <= step * 2.0 + 1e-6,
                            "round {round} {op:?} angle {ang} step {step}"
                        );
                        assert!(((gx * gx + gy * gy + gz * gz) as f64 - 1.0).abs() < 1e-4);
                    }
                    Op::Str(s) => assert_eq!(r.read_str(64).unwrap(), s),
                    Op::Bytes(v) => assert_eq!(r.read_bytes(v.len()).unwrap(), &v[..]),
                }
            }
        }
    }

    // ---- limits and refusals ------------------------------------------------------------------------------------

    #[test]
    fn a_full_buffer_is_exactly_full_and_one_bit_more_is_refused_without_a_trace() {
        let mut b = [0u8; 2];
        let mut w = BitWriter::new(&mut b);
        w.write_bits(0x7FFF, 15).unwrap();
        assert_eq!(w.bits_free(), 1);
        w.write_bool(true).unwrap();
        assert_eq!(w.bits_free(), 0);
        assert_eq!(w.write_bool(true), Err(BitError::Overflow));
        assert_eq!(w.bits_written(), 16);
        assert_eq!(w.written(), &[0xFF, 0xFF]);

        // A write that does not fit changes nothing, for every kind of write.
        let mut b = [0u8; 3];
        let mut w = BitWriter::new(&mut b);
        w.write_bits(1, 20).unwrap();
        let before = (w.bits_written(), w.written().to_vec());
        assert_eq!(w.write_bits(1, 5), Err(BitError::Overflow));
        assert_eq!(w.write_varint(1 << 40), Err(BitError::Overflow));
        assert_eq!(w.write_u32(1), Err(BitError::Overflow));
        assert_eq!(
            w.write_unit_vec3(0.0, 0.0, 1.0, 16),
            Err(BitError::Overflow)
        );
        assert_eq!(w.write_str("hello"), Err(BitError::Overflow));
        assert_eq!(w.write_bytes(&[1, 2]), Err(BitError::Overflow));
        assert_eq!((w.bits_written(), w.written().to_vec()), before);

        // The first half of a two-part write fits and the second does not: still nothing is written.
        let mut b = [0u8; 3];
        let mut w = BitWriter::new(&mut b);
        w.write_bits(0b1010, 4).unwrap(); // 20 bits free: one component of a 16-bit vector fits, two do not
        let before = (w.bits_written(), w.written().to_vec());
        assert_eq!(w.write_unit_vec3(0.0, 0.0, 1.0, 16), Err(BitError::Overflow));
        assert_eq!((w.bits_written(), w.written().to_vec()), before);
        let mut b = [0u8; 4];
        let mut w = BitWriter::new(&mut b);
        w.write_bits(1, 1).unwrap(); // pad 7 + a one-byte length + 3 bytes of text = 7 + 8 + 24 = 39 bits; 31 are free
        let before = (w.bits_written(), w.written().to_vec());
        assert_eq!(w.write_str("abc"), Err(BitError::Overflow));
        assert_eq!((w.bits_written(), w.written().to_vec()), before);
    }

    #[test]
    fn values_and_widths_that_do_not_fit_are_refused() {
        let mut b = [0u8; 16];
        let mut w = BitWriter::new(&mut b);
        assert_eq!(w.write_bits(8, 3), Err(BitError::OutOfRange));
        assert_eq!(w.write_bits(0, 0), Err(BitError::OutOfRange));
        assert_eq!(w.write_bits(0, 65), Err(BitError::OutOfRange));
        assert_eq!(
            w.write_f32_quantized(0.5, 0.0, 1.0, 0),
            Err(BitError::OutOfRange)
        );
        assert_eq!(
            w.write_f32_quantized(0.5, 0.0, 1.0, 33),
            Err(BitError::OutOfRange)
        );
        assert_eq!(
            w.write_unit_vec3(0.0, 0.0, 1.0, 1),
            Err(BitError::OutOfRange)
        );
        assert_eq!(
            w.write_unit_vec3(0.0, 0.0, 1.0, 17),
            Err(BitError::OutOfRange)
        );
        assert_eq!(w.bits_written(), 0);
        let mut r = BitReader::new(&[0xFF; 16]);
        assert_eq!(r.read_bits(0), Err(BitError::OutOfRange));
        assert_eq!(r.read_bits(65), Err(BitError::OutOfRange));
        assert_eq!(r.read_unit_vec3(1), Err(BitError::OutOfRange));
        assert_eq!(r.bits_read(), 0);
    }

    #[test]
    fn floats_that_cannot_be_trusted_are_refused_and_the_rest_are_clamped() {
        let mut b = [0u8; 16];
        let mut w = BitWriter::new(&mut b);
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert_eq!(
                w.write_f32_quantized(bad, 0.0, 1.0, 8),
                Err(BitError::NotFinite)
            );
            assert_eq!(
                w.write_unit_vec3(bad, 0.0, 1.0, 8),
                Err(BitError::NotFinite)
            );
        }
        assert_eq!(
            w.write_f32_quantized(0.5, 1.0, 1.0, 8),
            Err(BitError::NotFinite)
        );
        assert_eq!(
            w.write_f32_quantized(0.5, 2.0, 1.0, 8),
            Err(BitError::NotFinite)
        );
        assert_eq!(
            w.write_f32_quantized(0.5, f32::NAN, 1.0, 8),
            Err(BitError::NotFinite)
        );
        assert_eq!(
            w.write_unit_vec3(0.0, 0.0, 0.0, 8),
            Err(BitError::NotFinite)
        );
        assert_eq!(w.bits_written(), 0);
        w.write_f32_quantized(99.0, 0.0, 1.0, 8).unwrap();
        w.write_f32_quantized(-99.0, 0.0, 1.0, 8).unwrap();
        let mut r = BitReader::new(w.written());
        assert_eq!(r.read_f32_quantized(0.0, 1.0, 8).unwrap(), 1.0);
        assert_eq!(r.read_f32_quantized(0.0, 1.0, 8).unwrap(), 0.0);
        assert_eq!(
            BitReader::new(&[0]).read_f32_quantized(1.0, 1.0, 8),
            Err(BitError::NotFinite)
        );
    }

    #[test]
    fn quantisation_hits_the_ends_exactly_and_never_misses_by_more_than_half_a_step() {
        for bits in 1..=32u32 {
            let (lo, hi) = (-37.5f32, 812.25f32);
            let mut b = [0u8; 16];
            let mut w = BitWriter::new(&mut b);
            w.write_f32_quantized(lo, lo, hi, bits).unwrap();
            w.write_f32_quantized(hi, lo, hi, bits).unwrap();
            let mut r = BitReader::new(w.written());
            assert_eq!(
                r.read_f32_quantized(lo, hi, bits).unwrap(),
                lo,
                "min at {bits} bits"
            );
            let top = r.read_f32_quantized(lo, hi, bits).unwrap();
            assert!(
                (top - hi).abs() <= (hi - lo) * 1e-6,
                "max at {bits} bits: {top}"
            );
        }
        let mut rng = Rng(7);
        for bits in [1u32, 2, 3, 7, 8, 12, 16, 17, 24] {
            let step = quantization_step(-1.0, 1.0, bits);
            for _ in 0..400 {
                let v = (rng.unit() * 2.0 - 1.0) as f32;
                let mut b = [0u8; 8];
                let mut w = BitWriter::new(&mut b);
                w.write_f32_quantized(v, -1.0, 1.0, bits).unwrap();
                let got = BitReader::new(w.written())
                    .read_f32_quantized(-1.0, 1.0, bits)
                    .unwrap();
                assert!(
                    ((got - v) as f64).abs() <= step / 2.0 + 1e-6,
                    "{v} at {bits} bits came back {got}"
                );
            }
        }
    }

    #[test]
    fn the_unit_vector_keeps_its_direction_on_every_axis_and_in_every_octant() {
        let mut dirs = vec![
            (1.0, 0.0, 0.0),
            (-1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, -1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, 0.0, -1.0),
        ];
        for sx in [-1.0f32, 1.0] {
            for sy in [-1.0f32, 1.0] {
                for sz in [-1.0f32, 1.0] {
                    dirs.push((sx, sy * 2.0, sz * 3.0));
                    dirs.push((sx * 10.0, sy * 0.001, sz));
                }
            }
        }
        for (x, y, z) in dirs {
            for bits in [8u32, 12, 16] {
                let mut b = [0u8; 8];
                let mut w = BitWriter::new(&mut b);
                w.write_unit_vec3(x, y, z, bits).unwrap();
                let (gx, gy, gz) = BitReader::new(w.written()).read_unit_vec3(bits).unwrap();
                let step = 2.0 / (((1u64 << bits) - 1) as f64);
                assert!(
                    angle((gx, gy, gz), (x, y, z)) <= step * 2.0 + 1e-6,
                    "({x},{y},{z}) at {bits} bits came back ({gx},{gy},{gz})"
                );
            }
        }
        // A scaled vector is the same direction (it is normalised first).
        let mut b1 = [0u8; 8];
        let mut b2 = [0u8; 8];
        let mut w1 = BitWriter::new(&mut b1);
        w1.write_unit_vec3(1.0, 2.0, 3.0, 12).unwrap();
        let mut w2 = BitWriter::new(&mut b2);
        w2.write_unit_vec3(100.0, 200.0, 300.0, 12).unwrap();
        assert_eq!(w1.written(), w2.written());
    }

    #[test]
    fn varints_that_cannot_be_a_u64_are_refused_and_the_odd_but_harmless_are_not() {
        let eleven = [0x80u8; 10]
            .iter()
            .copied()
            .chain([0x00])
            .collect::<Vec<_>>();
        assert_eq!(
            BitReader::new(&eleven).read_varint(),
            Err(BitError::BadVarint)
        );
        let tenth_too_big = [0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x02];
        assert_eq!(
            BitReader::new(&tenth_too_big).read_varint(),
            Err(BitError::BadVarint)
        );
        assert_eq!(
            BitReader::new(&[0x80]).read_varint(),
            Err(BitError::Overflow)
        ); // cut short
        assert_eq!(BitReader::new(&[]).read_varint(), Err(BitError::Overflow));
        assert_eq!(BitReader::new(&[0x80, 0x00]).read_varint(), Ok(0)); // not the shortest form, but the same number
    }

    #[test]
    fn reads_past_the_end_fail_without_consuming_anything() {
        let mut r = BitReader::new(&[0xAB, 0xCD]);
        assert_eq!(r.read_bits(12).unwrap(), 0xDAB & 0xFFF);
        assert_eq!(r.read_bits(5), Err(BitError::Overflow));
        assert_eq!(r.bits_read(), 12);
        assert_eq!(r.read_bits(4).unwrap(), 0xC);
        assert_eq!(r.read_bool(), Err(BitError::Overflow));
        assert_eq!(r.read_bytes(1), Err(BitError::Overflow));
        assert_eq!(r.read_bytes(usize::MAX / 2), Err(BitError::Overflow)); // no arithmetic overflow, no panic
        assert_eq!(r.read_bytes(usize::MAX), Err(BitError::Overflow));
        assert_eq!(r.bits_read(), 16);
    }

    #[test]
    fn strings_and_bytes() {
        let mut b = [0u8; 64];
        let mut w = BitWriter::new(&mut b);
        w.write_bits(0b1, 1).unwrap(); // start unaligned: the string is aligned for itself
        w.write_str("").unwrap();
        w.write_str("Brou's Tower ✓").unwrap();
        w.write_bytes(&[]).unwrap();
        let mut r = BitReader::new(w.written());
        assert_eq!(r.read_bits(1).unwrap(), 1);
        assert_eq!(r.read_str(32).unwrap(), "");
        assert_eq!(r.read_str(32).unwrap(), "Brou's Tower ✓");
        assert_eq!(r.read_bytes(0).unwrap(), &[] as &[u8]);
        let mut r = BitReader::new(w.written());
        r.read_bits(1).unwrap();
        r.read_str(32).unwrap();
        assert_eq!(r.read_str(5), Err(BitError::BadString)); // longer than the caller allows
                                                             // Not UTF-8, and a length that claims more than is there.
        assert_eq!(
            BitReader::new(&[2, 0xC3, 0x28]).read_str(8),
            Err(BitError::BadString)
        );
        assert_eq!(
            BitReader::new(&[5, b'a', b'b']).read_str(8),
            Err(BitError::Overflow)
        );
        // The padding bits are zero (what is on the wire does not depend on what was in the buffer).
        let mut b = [0xFFu8; 4];
        let mut w = BitWriter::new(&mut b);
        w.write_bits(1, 3).unwrap();
        w.align_to_byte().unwrap();
        assert_eq!(w.written(), &[0b0000_0001]);
    }

    // ---- hostile input -------------------------------------------------------------------------------------------

    #[test]
    fn a_reader_run_over_random_bytes_with_random_reads_never_panics() {
        let mut rng = Rng(0xDEAD_BEEF_CAFE_F00D);
        for _ in 0..20000 {
            let len = rng.below(40) as usize;
            let bytes: Vec<u8> = (0..len).map(|_| rng.next() as u8).collect();
            let mut r = BitReader::new(&bytes);
            for _ in 0..30 {
                let _ = match rng.below(12) {
                    0 => r.read_bits(rng.below(70) as u32).map(|_| ()),
                    1 => r.read_bool().map(|_| ()),
                    2 => r.read_varint().map(|_| ()),
                    3 => r.read_zigzag().map(|_| ()),
                    4 => r.read_f32().map(|_| ()),
                    5 => r
                        .read_f32_quantized(-1.0, 1.0, rng.below(40) as u32)
                        .map(|_| ()),
                    6 => r.read_unit_vec3(rng.below(20) as u32).map(|_| ()),
                    7 => r.read_str(rng.below(64) as usize).map(|_| ()),
                    8 => r.read_bytes(rng.below(50) as usize).map(|_| ()),
                    9 => r.align_to_byte(),
                    10 => r.read_f32_quantized(f32::NAN, 1.0, 8).map(|_| ()),
                    _ => r.read_u64().map(|_| ()),
                };
                assert!(r.bits_read() <= bytes.len() * 8);
            }
        }
    }

    #[test]
    fn a_decoded_unit_vector_is_always_a_unit_vector_whatever_the_bits_were() {
        for bits in [2u32, 5, 8, 16] {
            let max = (1u64 << bits) - 1;
            let step = (max / 40).max(1);
            let mut u = 0;
            while u <= max {
                let mut v = 0;
                while v <= max {
                    let mut b = [0u8; 8];
                    let mut w = BitWriter::new(&mut b);
                    w.write_bits(u, bits).unwrap();
                    w.write_bits(v, bits).unwrap();
                    let (x, y, z) = BitReader::new(w.written()).read_unit_vec3(bits).unwrap();
                    assert!(
                        ((x * x + y * y + z * z) as f64 - 1.0).abs() < 1e-4,
                        "{u},{v} at {bits} bits: ({x},{y},{z})"
                    );
                    v += step;
                }
                u += step;
            }
        }
    }
}
