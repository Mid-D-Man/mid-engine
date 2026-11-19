//! One description of a type's wire form that works for both directions, and the two ways a struct
//! crosses the network.
//!
//! This is the answer to "how does a struct go over the net" that Unity's netcode gives with
//! `INetworkSerializable` and `INetworkSerializeByMemcpy`, in the shape this engine uses everywhere
//! (declarative macros, no procedural macro, no dependency, Rust 1.75):
//!
//! * [`NetSerialize`] -- a type writes **one** function, `net_serialize`, that visits its fields
//!   through a [`BitStream`]. The same function is the writer and the reader (a [`BitWriter`] reads
//!   the value out of the fields, a [`BitReader`] puts the value into them), so a field cannot be
//!   written in one order and read in another, which is the most common netcode bug. Fields choose
//!   their own cost: a bool is 1 bit, a health in `0..=100` is 7, a position is quantised.
//!   [`impl_net_serialize!`] writes the function from a field list.
//! * [`NetPod`] -- the by-memcpy path, for `#[repr(C)]` plain data with no padding and no field that
//!   has invalid bit patterns (integers and floats): the bytes are copied as they are in memory,
//!   little-endian only. It is bigger than the packed form and never needs a description.
//!   [`impl_net_pod!`] checks at compile time that there is no padding.
//!
//! What a trait *object* cannot do is cross a network, because the other end has no idea what type
//! it is. The two honest ways are an `enum` (a tag, then the variant) and a registry from a message
//! id to a decoder (`mid-net`'s message layer, next); both are built from these traits.
//!
//! Untrusted input: everything read is checked. A length that claims more elements than there are
//! bits left in the packet is refused before anything is allocated ([`net_vec`]), and no input can panic.

use crate::bits::{BitError, BitReader, BitWriter};

#[cfg(target_endian = "big")]
compile_error!("NetPod copies bytes as they are in memory and the wire is little-endian: big-endian targets are not supported");

/// A bit stream that is either writing or reading. Every method takes the value by `&mut`: a writer
/// reads it, a reader overwrites it, and the code that calls it is the same.
pub trait BitStream {
    /// `true` for a reader (values are being filled in), `false` for a writer.
    const READING: bool;

    /// Bits that can still be written, or read.
    fn bits_left(&self) -> usize;
    fn bits(&mut self, value: &mut u64, bits: u32) -> Result<(), BitError>;
    fn flag(&mut self, value: &mut bool) -> Result<(), BitError>;
    fn varint(&mut self, value: &mut u64) -> Result<(), BitError>;
    fn zigzag(&mut self, value: &mut i64) -> Result<(), BitError>;
    fn f32_raw(&mut self, value: &mut f32) -> Result<(), BitError>;
    fn f32_quantized(
        &mut self,
        value: &mut f32,
        min: f32,
        max: f32,
        bits: u32,
    ) -> Result<(), BitError>;
    fn unit_vec3(&mut self, value: &mut [f32; 3], bits: u32) -> Result<(), BitError>;
    /// A fixed number of whole bytes (aligned): the length is the slice's, so it is not sent.
    fn bytes(&mut self, value: &mut [u8]) -> Result<(), BitError>;
    /// A UTF-8 string of at most `max_len` bytes (length as a varint, then the bytes).
    fn string(&mut self, value: &mut String, max_len: usize) -> Result<(), BitError>;
}

impl BitStream for BitWriter<'_> {
    const READING: bool = false;
    fn bits_left(&self) -> usize {
        self.bits_free()
    }
    fn bits(&mut self, value: &mut u64, bits: u32) -> Result<(), BitError> {
        self.write_bits(*value, bits)
    }
    fn flag(&mut self, value: &mut bool) -> Result<(), BitError> {
        self.write_bool(*value)
    }
    fn varint(&mut self, value: &mut u64) -> Result<(), BitError> {
        self.write_varint(*value)
    }
    fn zigzag(&mut self, value: &mut i64) -> Result<(), BitError> {
        self.write_zigzag(*value)
    }
    fn f32_raw(&mut self, value: &mut f32) -> Result<(), BitError> {
        self.write_f32(*value)
    }
    fn f32_quantized(
        &mut self,
        value: &mut f32,
        min: f32,
        max: f32,
        bits: u32,
    ) -> Result<(), BitError> {
        self.write_f32_quantized(*value, min, max, bits)
    }
    fn unit_vec3(&mut self, value: &mut [f32; 3], bits: u32) -> Result<(), BitError> {
        self.write_unit_vec3(value[0], value[1], value[2], bits)
    }
    fn bytes(&mut self, value: &mut [u8]) -> Result<(), BitError> {
        self.write_bytes(value)
    }
    fn string(&mut self, value: &mut String, max_len: usize) -> Result<(), BitError> {
        if value.len() > max_len {
            return Err(BitError::BadString);
        }
        self.write_str(value)
    }
}

impl BitStream for BitReader<'_> {
    const READING: bool = true;
    fn bits_left(&self) -> usize {
        self.bits_remaining()
    }
    fn bits(&mut self, value: &mut u64, bits: u32) -> Result<(), BitError> {
        *value = self.read_bits(bits)?;
        Ok(())
    }
    fn flag(&mut self, value: &mut bool) -> Result<(), BitError> {
        *value = self.read_bool()?;
        Ok(())
    }
    fn varint(&mut self, value: &mut u64) -> Result<(), BitError> {
        *value = self.read_varint()?;
        Ok(())
    }
    fn zigzag(&mut self, value: &mut i64) -> Result<(), BitError> {
        *value = self.read_zigzag()?;
        Ok(())
    }
    fn f32_raw(&mut self, value: &mut f32) -> Result<(), BitError> {
        *value = self.read_f32()?;
        Ok(())
    }
    fn f32_quantized(
        &mut self,
        value: &mut f32,
        min: f32,
        max: f32,
        bits: u32,
    ) -> Result<(), BitError> {
        *value = self.read_f32_quantized(min, max, bits)?;
        Ok(())
    }
    fn unit_vec3(&mut self, value: &mut [f32; 3], bits: u32) -> Result<(), BitError> {
        let (x, y, z) = self.read_unit_vec3(bits)?;
        *value = [x, y, z];
        Ok(())
    }
    fn bytes(&mut self, value: &mut [u8]) -> Result<(), BitError> {
        let got = self.read_bytes(value.len())?;
        value.copy_from_slice(got);
        Ok(())
    }
    fn string(&mut self, value: &mut String, max_len: usize) -> Result<(), BitError> {
        *value = self.read_str(max_len)?.to_owned();
        Ok(())
    }
}

/// An unsigned integer type a field can be stored in at fewer bits than it has.
pub trait UInt: Copy {
    const BITS: u32;
    fn to_u64(self) -> u64;
    fn from_u64(v: u64) -> Option<Self>;
}
/// A signed integer type, stored in two's complement at fewer bits than it has.
pub trait SInt: Copy {
    const BITS: u32;
    fn to_i64(self) -> i64;
    fn from_i64(v: i64) -> Option<Self>;
}
macro_rules! impl_ints {
    ($($u:ty, $i:ty, $bits:expr);*) => {$(
        impl UInt for $u {
            const BITS: u32 = $bits;
            fn to_u64(self) -> u64 { self as u64 }
            fn from_u64(v: u64) -> Option<Self> { <$u>::try_from(v).ok() }
        }
        impl SInt for $i {
            const BITS: u32 = $bits;
            fn to_i64(self) -> i64 { self as i64 }
            fn from_i64(v: i64) -> Option<Self> { <$i>::try_from(v).ok() }
        }
    )*};
}
impl_ints!(u8, i8, 8; u16, i16, 16; u32, i32, 32; u64, i64, 64);

/// Typed helpers over any [`BitStream`] (what the field kinds of [`impl_net_serialize!`] call).
pub trait BitStreamExt: BitStream {
    /// An unsigned field in `bits` bits (at most the type's width).
    fn uint<T: UInt>(&mut self, value: &mut T, bits: u32) -> Result<(), BitError> {
        if bits == 0 || bits > T::BITS {
            return Err(BitError::OutOfRange);
        }
        let mut v = value.to_u64();
        self.bits(&mut v, bits)?;
        if Self::READING {
            *value = T::from_u64(v).ok_or(BitError::OutOfRange)?;
        }
        Ok(())
    }

    /// A signed field in `bits` bits, two's complement (at most the type's width). A value that does not fit is refused.
    fn sint<T: SInt>(&mut self, value: &mut T, bits: u32) -> Result<(), BitError> {
        if bits == 0 || bits > T::BITS {
            return Err(BitError::OutOfRange);
        }
        let mask = if bits == 64 {
            u64::MAX
        } else {
            (1u64 << bits) - 1
        };
        let mut raw = 0u64;
        if !Self::READING {
            let v = value.to_i64();
            if bits < 64 {
                let half = 1i64 << (bits - 1);
                if v < -half || v >= half {
                    return Err(BitError::OutOfRange);
                }
            }
            raw = (v as u64) & mask;
        }
        self.bits(&mut raw, bits)?;
        if Self::READING {
            let v = if bits < 64 && raw >> (bits - 1) & 1 == 1 {
                (raw | !mask) as i64
            } else {
                raw as i64
            };
            *value = T::from_i64(v).ok_or(BitError::OutOfRange)?;
        }
        Ok(())
    }

    /// An unsigned field as a varint (small values are short); a value that does not fit the type is refused.
    fn varuint<T: UInt>(&mut self, value: &mut T) -> Result<(), BitError> {
        let mut v = value.to_u64();
        self.varint(&mut v)?;
        if Self::READING {
            *value = T::from_u64(v).ok_or(BitError::OutOfRange)?;
        }
        Ok(())
    }
}
impl<S: BitStream + ?Sized> BitStreamExt for S {}

/// A type that describes its own wire form once, for writing and reading alike.
///
/// Writing a value takes `&mut self` only because the same function reads: use [`net_write_ref`] for a
/// shared reference to a `Clone` type.
pub trait NetSerialize {
    fn net_serialize<S: BitStream>(&mut self, s: &mut S) -> Result<(), BitError>;
}

macro_rules! impl_net_prim {
    ($($t:ty, $bits:expr);*) => {$(
        impl NetSerialize for $t {
            fn net_serialize<S: BitStream>(&mut self, s: &mut S) -> Result<(), BitError> {
                let mut v = *self as u64 & (if $bits == 64 { u64::MAX } else { (1u64 << $bits) - 1 });
                s.bits(&mut v, $bits)?;
                if S::READING { *self = v as $t; }
                Ok(())
            }
        }
    )*};
}
impl_net_prim!(u8, 8; u16, 16; u32, 32; u64, 64; i8, 8; i16, 16; i32, 32; i64, 64);

impl NetSerialize for bool {
    fn net_serialize<S: BitStream>(&mut self, s: &mut S) -> Result<(), BitError> {
        s.flag(self)
    }
}
impl NetSerialize for f32 {
    fn net_serialize<S: BitStream>(&mut self, s: &mut S) -> Result<(), BitError> {
        s.f32_raw(self)
    }
}
impl NetSerialize for f64 {
    fn net_serialize<S: BitStream>(&mut self, s: &mut S) -> Result<(), BitError> {
        let mut v = self.to_bits();
        s.bits(&mut v, 64)?;
        if S::READING {
            *self = f64::from_bits(v);
        }
        Ok(())
    }
}
impl<T: NetSerialize, const N: usize> NetSerialize for [T; N] {
    fn net_serialize<S: BitStream>(&mut self, s: &mut S) -> Result<(), BitError> {
        for item in self.iter_mut() {
            item.net_serialize(s)?;
        }
        Ok(())
    }
}
impl<T: NetSerialize + Default> NetSerialize for Option<T> {
    fn net_serialize<S: BitStream>(&mut self, s: &mut S) -> Result<(), BitError> {
        let mut present = self.is_some();
        s.flag(&mut present)?;
        if S::READING {
            *self = if present { Some(T::default()) } else { None };
        }
        if let Some(v) = self {
            v.net_serialize(s)?;
        }
        Ok(())
    }
}

/// A list of at most `max_len` items (its length as a varint, then each item). The limit is the
/// caller's and is also checked on writing. **Reading never allocates more than the packet can hold:**
/// a claimed length larger than the bits left is refused before anything is allocated (every item is at
/// least one bit), so a four-byte packet cannot ask for a gigabyte.
pub fn net_vec<S: BitStream, T: NetSerialize + Default>(
    s: &mut S,
    list: &mut Vec<T>,
    max_len: usize,
) -> Result<(), BitError> {
    let mut len = list.len() as u64;
    if !S::READING && list.len() > max_len {
        return Err(BitError::OutOfRange);
    }
    s.varint(&mut len)?;
    if S::READING {
        if len > max_len as u64 || len > s.bits_left() as u64 {
            return Err(BitError::OutOfRange);
        }
        list.clear();
        list.resize_with(len as usize, T::default);
    }
    for item in list.iter_mut() {
        item.net_serialize(s)?;
    }
    Ok(())
}

/// Writes `value` at the start of `buf`; returns the bytes used. On `Err` the packet must be dropped.
pub fn net_write<T: NetSerialize>(value: &mut T, buf: &mut [u8]) -> Result<usize, BitError> {
    let mut w = BitWriter::new(buf);
    value.net_serialize(&mut w)?;
    Ok(w.bytes_written())
}

/// Like [`net_write`] for a shared reference to a type that is `Clone`.
pub fn net_write_ref<T: NetSerialize + Clone>(
    value: &T,
    buf: &mut [u8],
) -> Result<usize, BitError> {
    net_write(&mut value.clone(), buf)
}

/// Reads a `T` from the start of `bytes`.
pub fn net_read<T: NetSerialize + Default>(bytes: &[u8]) -> Result<T, BitError> {
    let mut r = BitReader::new(bytes);
    let mut v = T::default();
    v.net_serialize(&mut r)?;
    Ok(v)
}

/// Makes `value` exactly what the other end will read: written and read back through `scratch`. Quantised
/// fields lose their extra precision here, so a sender that runs its own game from the rounded value (as the
/// game's prediction does with its commands) agrees with the receiver to the bit.
pub fn net_round_trip<T: NetSerialize + Default>(
    value: &mut T,
    scratch: &mut [u8],
) -> Result<(), BitError> {
    let n = net_write(value, scratch)?;
    *value = net_read(&scratch[..n])?;
    Ok(())
}

/// Describes a struct's wire form from a field list; the fields' kinds say what each costs.
///
/// ```
/// use mid_net_wire::{impl_net_serialize, net_read, net_write};
///
/// #[derive(Default, Debug, PartialEq, Clone)]
/// struct Hit { target: u16, damage: f32, dir: [f32; 3], crit: bool, note: String }
/// impl_net_serialize!(Hit {
///     target: bits(10),               // an id below 1024: 10 bits
///     damage: quant(0.0, 500.0, 12), // 0 to 500 in 12 bits
///     dir:    unit(12),              // a direction: 24 bits
///     crit:   flag(),
///     note:   string(16),
/// });
///
/// let mut buf = [0u8; 64];
/// let mut hit = Hit { target: 7, damage: 120.0, dir: [0.0, 0.0, 1.0], crit: true, note: "x".into() };
/// let n = net_write(&mut hit, &mut buf).unwrap();
/// let back: Hit = net_read(&buf[..n]).unwrap();
/// assert_eq!(back.target, 7);
/// assert!(back.crit);
/// ```
///
/// Field kinds: `flag()` (a `bool`), `bits(n)` (an unsigned integer in `n` bits), `sbits(n)` (a signed one),
/// `varint()` (an unsigned integer, short when small), `quant(min, max, bits)` (an `f32`), `unit(bits)` (an
/// `[f32; 3]` direction), `string(max)` (a `String`), `list(max)` (a `Vec` of anything that is `NetSerialize + Default`),
/// `raw()` (anything that is `NetSerialize`: a full-width number, a nested struct, an array, an `Option`).
#[macro_export]
macro_rules! impl_net_serialize {
    ($ty:ty { $($field:ident : $kind:ident ( $($arg:expr),* $(,)? )),* $(,)? }) => {
        impl $crate::NetSerialize for $ty {
            #[allow(unused_variables)]
            fn net_serialize<S: $crate::BitStream>(&mut self, s: &mut S) -> ::core::result::Result<(), $crate::BitError> {
                $( $crate::impl_net_serialize!(@field s, self.$field, $kind ( $($arg),* )); )*
                Ok(())
            }
        }
    };
    (@field $s:ident, $p:expr, flag ()) => { $crate::BitStream::flag($s, &mut $p)? };
    (@field $s:ident, $p:expr, bits ($n:expr)) => { $crate::BitStreamExt::uint($s, &mut $p, $n)? };
    (@field $s:ident, $p:expr, sbits ($n:expr)) => { $crate::BitStreamExt::sint($s, &mut $p, $n)? };
    (@field $s:ident, $p:expr, varint ()) => { $crate::BitStreamExt::varuint($s, &mut $p)? };
    (@field $s:ident, $p:expr, quant ($min:expr, $max:expr, $n:expr)) => { $crate::BitStream::f32_quantized($s, &mut $p, $min, $max, $n)? };
    (@field $s:ident, $p:expr, unit ($n:expr)) => { $crate::BitStream::unit_vec3($s, &mut $p, $n)? };
    (@field $s:ident, $p:expr, string ($n:expr)) => { $crate::BitStream::string($s, &mut $p, $n)? };
    (@field $s:ident, $p:expr, list ($n:expr)) => { $crate::net_vec($s, &mut $p, $n)? };
    (@field $s:ident, $p:expr, raw ()) => { $crate::NetSerialize::net_serialize(&mut $p, $s)? };
}

/// Plain data that can be copied to and from the wire as it is in memory.
///
/// # Safety
/// The implementing type must be `#[repr(C)]`, contain only integers, floats and arrays of them (so that
/// **every bit pattern is a valid value**: no `bool`, no `char`, no enum, no reference), and have **no
/// padding**. [`impl_net_pod!`] checks the last at compile time. The wire form is little-endian
/// (big-endian targets do not compile).
pub unsafe trait NetPod: Copy + 'static {}

macro_rules! impl_pod_prims { ($($t:ty),*) => {$( unsafe impl NetPod for $t {} )*}; }
impl_pod_prims!(u8, u16, u32, u64, i8, i16, i32, i64, f32, f64);
unsafe impl<T: NetPod, const N: usize> NetPod for [T; N] {}

/// Marks a `#[repr(C)]` struct as [`NetPod`] and fails to compile if it has padding.
///
/// ```
/// use mid_net_wire::impl_net_pod;
/// #[repr(C)] #[derive(Clone, Copy)]
/// struct State { x: f32, y: f32, z: f32, tick: u32 }
/// impl_net_pod!(State { x: f32, y: f32, z: f32, tick: u32 });
/// ```
/// ```compile_fail
/// use mid_net_wire::impl_net_pod;
/// #[repr(C)] #[derive(Clone, Copy)]
/// struct Padded { a: u8, b: u32 } // three bytes of padding after `a`
/// impl_net_pod!(Padded { a: u8, b: u32 });
/// ```
#[macro_export]
macro_rules! impl_net_pod {
    ($ty:ty { $($field:ident : $ft:ty),* $(,)? }) => {
        const _: () = ::core::assert!(
            ::core::mem::size_of::<$ty>() == 0 $(+ ::core::mem::size_of::<$ft>())*,
            "a NetPod struct must have no padding"
        );
        unsafe impl $crate::NetPod for $ty {}
    };
}

/// Writes a [`NetPod`] value's bytes (aligned to a byte boundary first).
pub fn write_pod<T: NetPod>(w: &mut BitWriter, value: &T) -> Result<(), BitError> {
    // SAFETY: a NetPod has no padding and no uninitialised bytes, so all `size_of::<T>()` bytes are initialised.
    let bytes = unsafe {
        core::slice::from_raw_parts((value as *const T).cast::<u8>(), core::mem::size_of::<T>())
    };
    w.write_bytes(bytes)
}

/// Reads a [`NetPod`] value (after aligning to a byte boundary).
pub fn read_pod<T: NetPod>(r: &mut BitReader) -> Result<T, BitError> {
    let bytes = r.read_bytes(core::mem::size_of::<T>())?;
    let mut out = core::mem::MaybeUninit::<T>::uninit();
    // SAFETY: `bytes` is exactly `size_of::<T>()` long, and every bit pattern is a valid NetPod.
    unsafe {
        core::ptr::copy_nonoverlapping(bytes.as_ptr(), out.as_mut_ptr().cast::<u8>(), bytes.len());
        Ok(out.assume_init())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[derive(Default, Debug, Clone, PartialEq)]
    struct Vec3 {
        x: f32,
        y: f32,
        z: f32,
    }
    impl_net_serialize!(Vec3 {
        x: quant(-512.0, 512.0, 20),
        y: quant(-512.0, 512.0, 20),
        z: quant(-512.0, 512.0, 20),
    });

    #[derive(Default, Debug, Clone, PartialEq)]
    struct Item {
        kind: u8,
        count: u16,
    }
    impl_net_serialize!(Item {
        kind: bits(6),
        count: varint()
    });

    #[derive(Default, Debug, Clone, PartialEq)]
    struct Everything {
        alive: bool,
        team: u8,
        health: u16,
        delta: i16,
        big: u64,
        tag: u32,
        pos: Vec3,
        aim: [f32; 3],
        name: String,
        items: Vec<Item>,
        held: Option<Item>,
        slots: [u8; 4],
        raw_float: f32,
        raw_double: f64,
        signed_full: i64,
    }
    impl_net_serialize!(Everything {
        alive: flag(),
        team: bits(3),
        health: bits(7),
        delta: sbits(9),
        big: varint(),
        tag: raw(),
        pos: raw(),
        aim: unit(12),
        name: string(24),
        items: list(8),
        held: raw(),
        slots: raw(),
        raw_float: raw(),
        raw_double: raw(),
        signed_full: raw(),
    });

    fn sample() -> Everything {
        Everything {
            alive: true,
            team: 5,
            health: 93,
            delta: -200,
            big: 1 << 50,
            tag: 0xDEAD_BEEF,
            pos: Vec3 {
                x: 100.25,
                y: -3.5,
                z: 511.0,
            },
            aim: [0.0, 0.6, 0.8],
            name: "Brou ✓".to_string(),
            items: vec![
                Item { kind: 3, count: 1 },
                Item {
                    kind: 63,
                    count: 40000,
                },
            ],
            held: Some(Item { kind: 9, count: 2 }),
            slots: [1, 2, 3, 255],
            raw_float: -0.0,
            raw_double: 1.0e300,
            signed_full: i64::MIN,
        }
    }

    #[test]
    fn one_description_is_the_writer_and_the_reader_of_a_struct_with_every_kind_of_field() {
        let mut buf = [0u8; 256];
        let mut e = sample();
        let n = net_write(&mut e, &mut buf).unwrap();
        let back: Everything = net_read(&buf[..n]).unwrap();
        assert!(back.alive);
        assert_eq!(
            (back.team, back.health, back.delta, back.big, back.tag),
            (5, 93, -200, 1 << 50, 0xDEAD_BEEF)
        );
        assert!(
            (back.pos.x - 100.25).abs() < 0.001
                && (back.pos.y + 3.5).abs() < 0.001
                && (back.pos.z - 511.0).abs() < 0.001
        );
        assert!((back.aim[1] - 0.6).abs() < 0.002 && (back.aim[2] - 0.8).abs() < 0.002);
        assert_eq!(back.name, "Brou ✓");
        assert_eq!(back.items, e.items);
        assert_eq!(back.held, e.held);
        assert_eq!(back.slots, [1, 2, 3, 255]);
        assert_eq!(back.raw_float.to_bits(), (-0.0f32).to_bits());
        assert_eq!(back.raw_double, 1.0e300);
        assert_eq!(back.signed_full, i64::MIN);
    }

    #[test]
    fn the_bytes_of_a_small_struct_are_pinned() {
        #[derive(Default)]
        struct Small {
            alive: bool,
            team: u8,
            hp: u16,
            delta: i16,
        }
        impl_net_serialize!(Small {
            alive: flag(),
            team: bits(3),
            hp: bits(7),
            delta: sbits(9)
        });
        let mut buf = [0u8; 8];
        let mut s = Small {
            alive: true,
            team: 5,
            hp: 100,
            delta: -3,
        };
        let n = net_write(&mut s, &mut buf).unwrap();
        // The 20 bits as one number, first field in the lowest bits: 1 | 5 << 1 | 100 << 4 | 0x1FD << 11 (-3 in 9 bits) = 0x0FEE4B, so 4B EE 0F.
        assert_eq!(n, 3);
        assert_eq!(&buf[..n], &[0x4B, 0xEE, 0x0F]);
        let back: Small = net_read(&buf[..n]).unwrap();
        assert_eq!(
            (back.alive, back.team, back.hp, back.delta),
            (true, 5, 100, -3)
        );
    }

    #[test]
    fn reading_overwrites_every_field_of_whatever_was_there() {
        let mut buf = [0u8; 256];
        let mut e = sample();
        e.held = None;
        e.items.clear();
        e.name.clear();
        let n = net_write(&mut e, &mut buf).unwrap();
        let mut dirty = sample(); // all fields set to something else, an Option that is Some, a list that is not empty
        let mut r = BitReader::new(&buf[..n]);
        dirty.net_serialize(&mut r).unwrap();
        assert!(dirty.held.is_none() && dirty.items.is_empty() && dirty.name.is_empty());
    }

    #[test]
    fn signed_fields_are_sign_extended_and_a_value_that_does_not_fit_is_refused() {
        for bits in [2u32, 5, 9, 16, 31, 32] {
            let lo = -(1i64 << (bits - 1));
            let hi = (1i64 << (bits - 1)) - 1;
            for v in [lo, lo + 1, -1, 0, 1, hi - 1, hi] {
                let mut buf = [0u8; 16];
                let mut w = BitWriter::new(&mut buf);
                let mut x = v as i32;
                w.sint(&mut x, bits).unwrap();
                let mut out = 12345i32;
                BitReader::new(w.written()).sint(&mut out, bits).unwrap();
                assert_eq!(out as i64, v, "{v} in {bits} bits");
            }
            if bits < 32 {
                let mut buf = [0u8; 16];
                let mut w = BitWriter::new(&mut buf);
                let mut over = (hi + 1) as i32;
                let mut under = (lo - 1) as i32;
                assert_eq!(
                    w.sint(&mut over, bits),
                    Err(BitError::OutOfRange),
                    "{} in {bits} bits",
                    hi + 1
                );
                assert_eq!(
                    w.sint(&mut under, bits),
                    Err(BitError::OutOfRange),
                    "{} in {bits} bits",
                    lo - 1
                );
                assert_eq!(w.bits_written(), 0);
            }
        }
        let mut buf = [0u8; 16];
        let mut w = BitWriter::new(&mut buf);
        let mut i = i64::MIN;
        w.sint(&mut i, 64).unwrap();
        let mut o = 0i64;
        BitReader::new(w.written()).sint(&mut o, 64).unwrap();
        assert_eq!(o, i64::MIN);
        let mut small = 1i8;
        assert_eq!(w.sint(&mut small, 9), Err(BitError::OutOfRange)); // more bits than the type has
        assert_eq!(w.sint(&mut small, 0), Err(BitError::OutOfRange));
    }

    #[test]
    fn unsigned_and_varint_fields_refuse_what_does_not_fit_the_type() {
        let mut buf = [0u8; 16];
        let mut w = BitWriter::new(&mut buf);
        let mut b = 255u8;
        assert_eq!(w.uint(&mut b, 9), Err(BitError::OutOfRange)); // more bits than a u8 has
        let mut big = 70u8;
        assert_eq!(w.uint(&mut big, 6), Err(BitError::OutOfRange)); // 70 needs 7 bits
        assert_eq!(w.bits_written(), 0);
        let mut v = 300u32;
        w.varuint(&mut v).unwrap();
        let mut as_u8 = 0u8;
        assert_eq!(
            BitReader::new(w.written()).varuint(&mut as_u8),
            Err(BitError::OutOfRange)
        ); // 300 is not a u8
        let mut as_u16 = 0u16;
        BitReader::new(w.written()).varuint(&mut as_u16).unwrap();
        assert_eq!(as_u16, 300);
    }

    #[test]
    fn a_list_is_limited_on_writing_and_on_reading_and_never_allocates_what_the_packet_cannot_hold()
    {
        let mut buf = [0u8; 64];
        let mut too_long = vec![Item::default(); 9];
        let mut w = BitWriter::new(&mut buf);
        assert_eq!(net_vec(&mut w, &mut too_long, 8), Err(BitError::OutOfRange));

        // A packet that claims 4 billion items in 5 bytes.
        let mut claim = [0u8; 5];
        let mut w = BitWriter::new(&mut claim);
        w.write_varint(4_000_000_000).unwrap();
        let mut list: Vec<Item> = Vec::new();
        let mut r = BitReader::new(&claim);
        assert_eq!(
            net_vec(&mut r, &mut list, usize::MAX),
            Err(BitError::OutOfRange)
        );
        assert_eq!(
            list.capacity(),
            0,
            "nothing was allocated for a claim the packet cannot back"
        );

        // A claim within the limit but more than the bits left.
        let mut few = [0u8; 2];
        let mut w = BitWriter::new(&mut few);
        w.write_varint(100).unwrap();
        let mut r = BitReader::new(&few);
        assert_eq!(net_vec(&mut r, &mut list, 1000), Err(BitError::OutOfRange));

        // A claim over the caller's limit.
        let mut ok = [0u8; 4];
        let mut w = BitWriter::new(&mut ok);
        w.write_varint(9).unwrap();
        w.write_bits(0, 20).unwrap();
        assert_eq!(
            net_vec(&mut BitReader::new(w.written()), &mut list, 8),
            Err(BitError::OutOfRange)
        );

        // An empty list, and one of exactly the limit, round trip.
        for n in [0usize, 1, 8] {
            let mut items: Vec<Item> = (0..n)
                .map(|i| Item {
                    kind: i as u8,
                    count: i as u16 * 100,
                })
                .collect();
            let mut buf = [0u8; 64];
            let mut w = BitWriter::new(&mut buf);
            net_vec(&mut w, &mut items, 8).unwrap();
            let mut back: Vec<Item> = vec![Item::default(); 3];
            net_vec(&mut BitReader::new(w.written()), &mut back, 8).unwrap();
            assert_eq!(back, items);
        }
    }

    #[test]
    fn reading_a_list_into_an_old_one_starts_every_item_clean() {
        // `local` is not on the wire: a reused element must not keep what an older packet left in it.
        #[derive(Default, Clone, Debug, PartialEq)]
        struct Tagged {
            v: u8,
            local: u32,
        }
        impl_net_serialize!(Tagged { v: raw() });
        let mut buf = [0u8; 16];
        let mut src = vec![Tagged { v: 1, local: 5 }, Tagged { v: 2, local: 6 }];
        let mut w = BitWriter::new(&mut buf);
        net_vec(&mut w, &mut src, 8).unwrap();
        let mut old = vec![Tagged { v: 9, local: 9 }; 5];
        net_vec(&mut BitReader::new(w.written()), &mut old, 8).unwrap();
        assert_eq!(
            old,
            vec![Tagged { v: 1, local: 0 }, Tagged { v: 2, local: 0 }]
        );
    }

    #[test]
    fn an_option_costs_one_bit_when_empty() {
        let mut buf = [0u8; 8];
        let mut none: Option<u32> = None;
        let mut w = BitWriter::new(&mut buf);
        none.net_serialize(&mut w).unwrap();
        assert_eq!(w.bits_written(), 1);
        let mut some = Some(7u32);
        let mut w = BitWriter::new(&mut buf);
        some.net_serialize(&mut w).unwrap();
        assert_eq!(w.bits_written(), 33);
        let mut back: Option<u32> = Some(99);
        back.net_serialize(&mut BitReader::new(w.written()))
            .unwrap();
        assert_eq!(back, Some(7));
    }

    #[test]
    fn a_value_can_be_made_exactly_what_the_other_end_will_read() {
        let mut scratch = [0u8; 64];
        let mut v = Vec3 {
            x: 100.123_46,
            y: -0.000_01,
            z: 7.25,
        };
        let original = v.clone();
        net_round_trip(&mut v, &mut scratch).unwrap();
        assert_ne!(v, original, "the extra precision is gone");
        assert!((v.x - original.x).abs() < 0.001);
        let once = v.clone();
        net_round_trip(&mut v, &mut scratch).unwrap();
        assert_eq!(v, once, "and doing it again changes nothing");
        let mut tiny = [0u8; 2];
        assert_eq!(net_round_trip(&mut v, &mut tiny), Err(BitError::Overflow));
        let n = net_write_ref(&original, &mut scratch).unwrap();
        assert_eq!(n, 8); // 3 x 20 bits = 60 bits
    }

    #[test]
    fn strings_are_limited_on_both_sides() {
        #[derive(Default)]
        struct Name {
            s: String,
        }
        impl_net_serialize!(Name { s: string(4) });
        let mut buf = [0u8; 16];
        assert_eq!(
            net_write(
                &mut Name {
                    s: "toolong".into()
                },
                &mut buf
            ),
            Err(BitError::BadString)
        );
        let n = net_write(&mut Name { s: "abcd".into() }, &mut buf).unwrap();
        assert_eq!(net_read::<Name>(&buf[..n]).unwrap().s, "abcd");
        let mut long = [0u8; 16];
        let mut w = BitWriter::new(&mut long);
        w.write_str("abcdef").unwrap();
        assert!(net_read::<Name>(w.written()).is_err());
    }

    // ---- the by-memcpy path -------------------------------------------------------------------------------------

    #[repr(C)]
    #[derive(Clone, Copy, Debug, PartialEq)]
    struct PlayerStateLike {
        x: f32,
        y: f32,
        z: f32,
        rx: f32,
        ry: f32,
        rz: f32,
        rw: f32,
    }
    impl_net_pod!(PlayerStateLike {
        x: f32,
        y: f32,
        z: f32,
        rx: f32,
        ry: f32,
        rz: f32,
        rw: f32
    });

    #[repr(C)]
    #[derive(Clone, Copy, Debug, PartialEq)]
    struct Mixed {
        stamp: u64,
        origin: [f32; 3],
        id: u32,
        extra: u32,
        hp: u16,
        team: u8,
        flags: u8,
    }
    impl_net_pod!(Mixed {
        stamp: u64,
        origin: [f32; 3],
        id: u32,
        extra: u32,
        hp: u16,
        team: u8,
        flags: u8
    });

    #[test]
    fn plain_data_crosses_as_its_bytes_little_endian_and_unpadded() {
        let s = PlayerStateLike {
            x: 1.0,
            y: -2.5,
            z: 0.0,
            rx: 0.0,
            ry: 0.0,
            rz: 0.0,
            rw: 1.0,
        };
        let mut buf = [0u8; 64];
        let mut w = BitWriter::new(&mut buf);
        write_pod(&mut w, &s).unwrap();
        assert_eq!(
            w.bytes_written(),
            28,
            "the same 28 bytes PlayerState takes on the wire today"
        );
        assert_eq!(
            &w.written()[..8],
            &[0x00, 0x00, 0x80, 0x3F, 0x00, 0x00, 0x20, 0xC0]
        ); // 1.0f32, -2.5f32 little-endian
        let back: PlayerStateLike = read_pod(&mut BitReader::new(w.written())).unwrap();
        assert_eq!(back, s);

        let m = Mixed {
            id: 0xAABBCCDD,
            extra: 7,
            hp: 77,
            team: 2,
            flags: 0x81,
            origin: [1.0, 2.0, 3.0],
            stamp: u64::MAX - 5,
        };
        let mut w = BitWriter::new(&mut buf);
        w.write_bits(0b101, 3).unwrap(); // not aligned: the pod aligns itself
        write_pod(&mut w, &m).unwrap();
        assert_eq!(w.bytes_written(), 1 + core::mem::size_of::<Mixed>());
        let mut r = BitReader::new(w.written());
        assert_eq!(r.read_bits(3).unwrap(), 0b101);
        assert_eq!(read_pod::<Mixed>(&mut r).unwrap(), m);
        assert_eq!(
            &w.written()[1..9],
            &[0xFA, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF]
        ); // the u64 first, little-endian
    }

    #[test]
    fn every_bit_pattern_is_a_valid_pod_and_a_short_packet_is_refused() {
        let nan_bits = [
            0x00u8, 0x00, 0xC0, 0x7F, 0xFF, 0xFF, 0xFF, 0xFF, 0x01, 0x00, 0x80, 0x7F,
        ]; // NaNs of several kinds, an infinity pattern
        let mut b = [0u8; 28];
        b[..12].copy_from_slice(&nan_bits);
        let got: PlayerStateLike = read_pod(&mut BitReader::new(&b)).unwrap();
        assert!(got.x.is_nan() && got.y.is_nan() && got.z.is_nan());
        assert_eq!(
            read_pod::<PlayerStateLike>(&mut BitReader::new(&b[..27])),
            Err(BitError::Overflow)
        );
        let mut w = BitWriter::new(&mut b[..20]);
        assert_eq!(write_pod(&mut w, &got), Err(BitError::Overflow));
        assert_eq!(w.bits_written(), 0);
    }

    // ---- hostile input ---------------------------------------------------------------------------------------------

    #[test]
    fn a_struct_read_from_random_bytes_never_panics_and_never_allocates_wildly() {
        let mut rng = Rng(0x1234_5678_9ABC_DEF1);
        let mut ok = 0;
        for _ in 0..30000 {
            let len = rng.below(120) as usize;
            let bytes: Vec<u8> = (0..len).map(|_| rng.next() as u8).collect();
            if let Ok(e) = net_read::<Everything>(&bytes) {
                ok += 1;
                assert!(e.items.len() <= 8 && e.name.len() <= 24);
            }
            let _ = read_pod::<Mixed>(&mut BitReader::new(&bytes));
        }
        assert!(
            ok > 0,
            "some random packets are valid, which is fine: what matters is that none panicked"
        );
    }
}
