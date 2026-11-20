//! Sending "one of several things": the two ways a trait's worth of types can cross a network.
//!
//! A trait object cannot be sent: the other end has no idea what it is. What can be sent is **which
//! one it is, then its fields**, and there are two honest ways to say which:
//!
//! 1. **An `enum`** ([`impl_net_enum!`]): the set of variants is closed and known to both ends at
//!    compile time. The variant is a small tag (2 bits for 4 variants) and then the variant's fields.
//!    Use it when the alternatives are a fixed list (the commands a player can issue, the kinds of
//!    event). A new variant is a new protocol version.
//! 2. **A message id and a decoder table** ([`MessageRegistry`], [`Dispatcher`]): the set is open;
//!    each type is registered under a number, a message is that number then the type's
//!    [`NetSerialize`] fields, and the receiver looks the number up. Use it for the messages and RPCs of
//!    a whole protocol, where each module registers its own. [`MessageRegistry`] gives back a
//!    `Box<dyn NetMessage>` (to inspect, log or forward without knowing the type; `downcast` gets the
//!    type back), and [`Dispatcher`] calls a typed handler per id with no allocation.
//!
//! Both read untrusted bytes: an unknown tag or id is [`BitError::UnknownTag`], never a panic, and a
//! handler runs only after its whole message has been read without error.

use std::any::Any;
use std::collections::HashMap;
use std::fmt;

use crate::bits::{BitError, BitReader, BitWriter};
use crate::stream::NetSerialize;

/// Describes an enum's wire form: a tag of `tag_bits` bits (each variant has a literal tag number), then
/// the fields of that variant, with the same field kinds as [`impl_net_serialize!`](crate::impl_net_serialize).
/// Variants are written struct-style (`Stop {}` for one with no fields); the field types and the enum
/// itself need `Default` for reading (the variant is built, then its fields are read into it).
///
/// ```
/// use mid_net_wire::{impl_net_enum, net_read, net_write};
///
/// #[derive(Debug, PartialEq)]
/// enum Command {
///     Move { dir: [f32; 3], sprint: bool },
///     Fire { weapon: u8 },
///     Stop {},
/// }
/// impl Default for Command { fn default() -> Self { Command::Stop {} } }
/// impl_net_enum!(Command, 2, {
///     Move { dir: unit(12), sprint: flag() } = 0,
///     Fire { weapon: bits(4) } = 1,
///     Stop {} = 2,
/// });
///
/// let mut buf = [0u8; 8];
/// let n = net_write(&mut Command::Fire { weapon: 9 }, &mut buf).unwrap();
/// assert_eq!(n, 1); // 2 bits of tag + 4 bits of weapon
/// assert_eq!(net_read::<Command>(&buf[..n]).unwrap(), Command::Fire { weapon: 9 });
/// ```
#[macro_export]
macro_rules! impl_net_enum {
    ($ty:ident, $tag_bits:expr, {
        $( $variant:ident { $($field:ident : $kind:ident ( $($arg:expr),* $(,)? )),* $(,)? } = $tag:literal ),+ $(,)?
    }) => {
        impl $crate::NetSerialize for $ty {
            #[allow(unused_variables)]
            fn net_serialize<S: $crate::BitStream>(&mut self, s: &mut S) -> ::core::result::Result<(), $crate::BitError> {
                let mut tag: u64 = match self { $( $ty::$variant { .. } => $tag, )+ };
                $crate::BitStream::bits(s, &mut tag, $tag_bits)?;
                if S::READING {
                    *self = match tag {
                        $( $tag => $ty::$variant { $( $field: ::core::default::Default::default() ),* }, )+
                        _ => return ::core::result::Result::Err($crate::BitError::UnknownTag),
                    };
                }
                match self {
                    $( $ty::$variant { $($field),* } => { $( $crate::impl_net_serialize!(@field s, *$field, $kind ( $($arg),* )); )* } )+
                }
                Ok(())
            }
        }
    };
}

/// A type that is sent as a message: it describes its fields ([`NetSerialize`]), can be built empty to be
/// read into (`Default`), and has an id that is the same on both ends.
pub trait NetMessageType: NetSerialize + Default + Any {
    const ID: u16;
}

/// A message whose type is not known where it is held (the object-safe side of [`NetMessageType`]).
pub trait NetMessage: Any {
    fn id(&self) -> u16;
    fn write_fields(&mut self, w: &mut BitWriter) -> Result<(), BitError>;
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
    fn into_any(self: Box<Self>) -> Box<dyn Any>;
}

impl<T: NetMessageType> NetMessage for T {
    fn id(&self) -> u16 {
        T::ID
    }
    fn write_fields(&mut self, w: &mut BitWriter) -> Result<(), BitError> {
        self.net_serialize(w)
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn into_any(self: Box<Self>) -> Box<dyn Any> {
        self
    }
}

impl fmt::Debug for dyn NetMessage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "NetMessage(id {})", self.id())
    }
}

impl dyn NetMessage {
    /// The message as a `T`, if that is what it is.
    pub fn downcast_ref<T: NetMessageType>(&self) -> Option<&T> {
        self.as_any().downcast_ref::<T>()
    }
    pub fn downcast_mut<T: NetMessageType>(&mut self) -> Option<&mut T> {
        self.as_any_mut().downcast_mut::<T>()
    }
    /// The message as an owned `T`; the box back if it is another type.
    pub fn downcast<T: NetMessageType>(self: Box<Self>) -> Result<Box<T>, Box<dyn NetMessage>> {
        if self.as_any().is::<T>() {
            Ok(self
                .into_any()
                .downcast::<T>()
                .expect("checked with is::<T>()"))
        } else {
            Err(self)
        }
    }
}

/// Why a type could not be registered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegisterError {
    /// The id does not fit in the registry's id width.
    IdTooLarge,
    /// Something is already registered under that id.
    Duplicate,
    /// The id width is 0 or more than 16 bits.
    BadIdBits,
}

impl fmt::Display for RegisterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            RegisterError::IdTooLarge => "message id does not fit the id width",
            RegisterError::Duplicate => "message id already registered",
            RegisterError::BadIdBits => "id width must be 1 to 16 bits",
        })
    }
}

fn check_id(id_bits: u32, id: u16, taken: bool) -> Result<(), RegisterError> {
    if !(1..=16).contains(&id_bits) {
        return Err(RegisterError::BadIdBits);
    }
    if id_bits < 16 && (id as u32) >> id_bits != 0 {
        return Err(RegisterError::IdTooLarge);
    }
    if taken {
        return Err(RegisterError::Duplicate);
    }
    Ok(())
}

type Decoder = fn(&mut BitReader) -> Result<Box<dyn NetMessage>, BitError>;

/// Message ids to decoders. A message on the wire is its id (`id_bits` bits) then its fields.
pub struct MessageRegistry {
    id_bits: u32,
    decoders: HashMap<u16, Decoder>,
}

impl MessageRegistry {
    /// A registry whose ids are written in `id_bits` bits (1 to 16; 8 allows 256 message types in a byte).
    pub fn new(id_bits: u32) -> Self {
        MessageRegistry {
            id_bits,
            decoders: HashMap::new(),
        }
    }

    pub fn id_bits(&self) -> u32 {
        self.id_bits
    }

    pub fn len(&self) -> usize {
        self.decoders.len()
    }

    pub fn is_empty(&self) -> bool {
        self.decoders.is_empty()
    }

    pub fn register<T: NetMessageType>(&mut self) -> Result<(), RegisterError> {
        check_id(self.id_bits, T::ID, self.decoders.contains_key(&T::ID))?;
        self.decoders.insert(T::ID, |r| {
            let mut v = T::default();
            v.net_serialize(r)?;
            Ok(Box::new(v))
        });
        Ok(())
    }

    /// Writes the id then the fields. A message whose id is not registered is refused (`UnknownTag`): the other end could not read it.
    pub fn write(&self, msg: &mut dyn NetMessage, w: &mut BitWriter) -> Result<(), BitError> {
        let id = msg.id();
        if !self.decoders.contains_key(&id) {
            return Err(BitError::UnknownTag);
        }
        w.write_bits(id as u64, self.id_bits)?;
        msg.write_fields(w)
    }

    /// Reads an id and then that type's fields.
    pub fn read(&self, r: &mut BitReader) -> Result<Box<dyn NetMessage>, BitError> {
        let id = r.read_bits(self.id_bits)? as u16;
        let decode = self.decoders.get(&id).ok_or(BitError::UnknownTag)?;
        decode(r)
    }
}

type Handler<C> = Box<dyn FnMut(&mut C, &mut BitReader) -> Result<(), BitError>>;

/// Message ids to typed handlers, with a context the handlers work on. Decoding is into a value on the
/// stack (no allocation per message), and a handler is called only once its message has been read in full.
pub struct Dispatcher<C> {
    id_bits: u32,
    handlers: HashMap<u16, Handler<C>>,
}

impl<C> Dispatcher<C> {
    pub fn new(id_bits: u32) -> Self {
        Dispatcher {
            id_bits,
            handlers: HashMap::new(),
        }
    }

    /// Calls `handler` with each message of type `T` that [`dispatch`](Self::dispatch) reads.
    pub fn on<T: NetMessageType>(
        &mut self,
        mut handler: impl FnMut(&mut C, T) + 'static,
    ) -> Result<(), RegisterError> {
        check_id(self.id_bits, T::ID, self.handlers.contains_key(&T::ID))?;
        self.handlers.insert(
            T::ID,
            Box::new(move |ctx, r| {
                let mut v = T::default();
                v.net_serialize(r)?;
                handler(ctx, v);
                Ok(())
            }),
        );
        Ok(())
    }

    /// Reads one message and runs its handler; returns the id handled. On `Err` no handler has run.
    pub fn dispatch(&mut self, ctx: &mut C, r: &mut BitReader) -> Result<u16, BitError> {
        let id = r.read_bits(self.id_bits)? as u16;
        let handler = self.handlers.get_mut(&id).ok_or(BitError::UnknownTag)?;
        handler(ctx, r)?;
        Ok(id)
    }
}

/// Writes one message the way [`Dispatcher`] reads it: its id in `id_bits` bits, then its fields.
pub fn write_message<T: NetMessageType>(
    msg: &mut T,
    id_bits: u32,
    w: &mut BitWriter,
) -> Result<(), BitError> {
    w.write_bits(T::ID as u64, id_bits)?;
    msg.net_serialize(w)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{impl_net_serialize, net_read, net_write};
    use std::cell::RefCell;
    use std::rc::Rc;

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
    struct Hit {
        target: u16,
        damage: f32,
        crit: bool,
    }
    impl_net_serialize!(Hit {
        target: bits(10),
        damage: quant(0.0, 500.0, 12),
        crit: flag()
    });
    impl NetMessageType for Hit {
        const ID: u16 = 3;
    }

    #[derive(Debug, Default, Clone, PartialEq)]
    struct Chat {
        text: String,
    }
    impl_net_serialize!(Chat { text: string(32) });
    impl NetMessageType for Chat {
        const ID: u16 = 7;
    }

    #[derive(Debug, Default, Clone, PartialEq)]
    struct Ping {
        seq: u32,
    }
    impl_net_serialize!(Ping { seq: raw() });
    impl NetMessageType for Ping {
        const ID: u16 = 1;
    }

    #[derive(Debug, Default, Clone, PartialEq)]
    struct TooBig {
        a: u8,
    }
    impl_net_serialize!(TooBig { a: raw() });
    impl NetMessageType for TooBig {
        const ID: u16 = 300;
    }

    // ---- the enum way ---------------------------------------------------------------------------------------------

    #[derive(Debug, PartialEq, Clone)]
    enum Command {
        Move { dir: [f32; 3], sprint: bool },
        Fire { weapon: u8, hit: Hit },
        Stop {},
        Say { text: String, volume: i16 },
    }
    impl Default for Command {
        fn default() -> Self {
            Command::Stop {}
        }
    }
    impl_net_enum!(Command, 3, {
        Move { dir: unit(12), sprint: flag() } = 0,
        Fire { weapon: bits(4), hit: raw() } = 1,
        Stop {} = 2,
        Say { text: string(8), volume: sbits(6) } = 5,
    });

    fn roundtrip(c: &Command) -> Command {
        let mut buf = [0u8; 64];
        let n = net_write(&mut c.clone(), &mut buf).unwrap();
        net_read::<Command>(&buf[..n]).unwrap()
    }

    #[test]
    fn every_variant_of_an_enum_comes_back_as_itself() {
        let hit = Hit {
            target: 12,
            damage: 250.0,
            crit: true,
        };
        match roundtrip(&Command::Move {
            dir: [0.0, 0.0, 1.0],
            sprint: true,
        }) {
            Command::Move { dir, sprint } => {
                assert!(sprint && (dir[2] - 1.0).abs() < 0.01);
            }
            other => panic!("{other:?}"),
        }
        match roundtrip(&Command::Fire {
            weapon: 9,
            hit: hit.clone(),
        }) {
            Command::Fire { weapon, hit: h } => {
                assert!(weapon == 9 && h.target == 12 && h.crit && (h.damage - 250.0).abs() < 0.2)
            }
            other => panic!("{other:?}"),
        }
        assert_eq!(roundtrip(&Command::Stop {}), Command::Stop {});
        assert_eq!(
            roundtrip(&Command::Say {
                text: "hi ✓".into(),
                volume: -17
            }),
            Command::Say {
                text: "hi ✓".into(),
                volume: -17
            }
        );
    }

    #[test]
    fn the_tag_costs_its_bits_and_the_bytes_are_pinned() {
        let mut buf = [0u8; 8];
        let n = net_write(&mut Command::Stop {}, &mut buf).unwrap();
        assert_eq!((n, buf[0]), (1, 0b010)); // tag 2 in 3 bits, nothing else
                                             // Say: tag 5 in 3 bits (101); the string's length (1) as a varint right after it, at bit 3 (0000_0001 -> 0x0D, 0x00 over two bytes);
                                             // padding up to the byte boundary; 'A' (0x41); then volume -2 in 6 bits (111110 = 0x3E).
        let mut w = BitWriter::new(&mut buf);
        Command::Say {
            text: "A".into(),
            volume: -2,
        }
        .net_serialize(&mut w)
        .unwrap();
        assert_eq!(w.written(), &[0x0D, 0x00, 0x41, 0x3E]);
    }

    #[test]
    fn reading_replaces_whatever_variant_was_there_and_an_unknown_tag_is_an_error() {
        let mut buf = [0u8; 16];
        let n = net_write(&mut Command::Stop {}, &mut buf).unwrap();
        let mut c = Command::Say {
            text: "old".into(),
            volume: 5,
        };
        c.net_serialize(&mut BitReader::new(&buf[..n])).unwrap();
        assert_eq!(c, Command::Stop {});
        for bad in [3u8, 4, 6, 7] {
            assert_eq!(
                net_read::<Command>(&[bad]),
                Err(BitError::UnknownTag),
                "tag {bad}"
            );
        }
        assert_eq!(net_read::<Command>(&[]), Err(BitError::Overflow));
    }

    #[test]
    fn a_tag_that_does_not_fit_its_width_is_refused_on_writing() {
        #[derive(Debug, PartialEq)]
        enum Wide {
            A {},
            B {},
        }
        impl Default for Wide {
            fn default() -> Self {
                Wide::A {}
            }
        }
        impl_net_enum!(Wide, 2, { A {} = 0, B {} = 4 }); // 4 needs 3 bits
        let mut buf = [0u8; 4];
        assert_eq!(
            net_write(&mut Wide::B {}, &mut buf),
            Err(BitError::OutOfRange)
        );
        assert_eq!(net_write(&mut Wide::A {}, &mut buf), Ok(1));
    }

    #[test]
    fn enums_nest_in_structs_and_lists() {
        #[derive(Debug, Default, PartialEq)]
        struct Frame {
            seq: u8,
            cmds: Vec<Command>,
        }
        impl_net_serialize!(Frame {
            seq: raw(),
            cmds: list(4)
        });
        let mut f = Frame {
            seq: 9,
            cmds: vec![
                Command::Stop {},
                Command::Say {
                    text: "x".into(),
                    volume: 3,
                },
                Command::Stop {},
            ],
        };
        let mut buf = [0u8; 64];
        let n = net_write(&mut f, &mut buf).unwrap();
        assert_eq!(net_read::<Frame>(&buf[..n]).unwrap(), f);
    }

    // ---- the registry way -----------------------------------------------------------------------------------------

    fn registry() -> MessageRegistry {
        let mut r = MessageRegistry::new(8);
        r.register::<Ping>().unwrap();
        r.register::<Hit>().unwrap();
        r.register::<Chat>().unwrap();
        r
    }

    #[test]
    fn a_registry_sends_a_value_without_the_receiver_knowing_its_type() {
        let reg = registry();
        let mut buf = [0u8; 64];
        let mut w = BitWriter::new(&mut buf);
        reg.write(
            &mut Chat {
                text: "hello".into(),
            },
            &mut w,
        )
        .unwrap();
        reg.write(
            &mut Hit {
                target: 5,
                damage: 100.0,
                crit: false,
            },
            &mut w,
        )
        .unwrap(); // two messages in one packet: the second starts where the first ended
        let mut r = BitReader::new(w.written());
        let first = reg.read(&mut r).unwrap();
        assert_eq!(first.id(), 7);
        assert_eq!(first.downcast_ref::<Chat>().unwrap().text, "hello");
        assert!(
            first.downcast_ref::<Hit>().is_none(),
            "asked for the wrong type, it says so"
        );
        let second = reg.read(&mut r).unwrap();
        let hit = second.downcast::<Hit>().unwrap();
        assert_eq!((hit.target, hit.crit), (5, false));
        assert!((hit.damage - 100.0).abs() < 0.2);
        // downcast to the wrong type hands the message back.
        let back = reg
            .read(&mut BitReader::new(w.written()))
            .unwrap()
            .downcast::<Hit>()
            .unwrap_err();
        assert_eq!(back.id(), 7);
    }

    #[test]
    fn a_message_is_its_id_in_the_registrys_width_then_its_fields() {
        let reg = registry();
        let mut buf = [0u8; 16];
        let mut w = BitWriter::new(&mut buf);
        reg.write(&mut Ping { seq: 0x0102_0304 }, &mut w).unwrap();
        assert_eq!(w.written(), &[1, 0x04, 0x03, 0x02, 0x01]); // id 1 in 8 bits, then the u32 little-endian
        let mut reg4 = MessageRegistry::new(4);
        reg4.register::<Ping>().unwrap();
        let mut w = BitWriter::new(&mut buf);
        reg4.write(&mut Ping { seq: 0xFFFF_FFFF }, &mut w).unwrap();
        assert_eq!(w.bits_written(), 4 + 32);
        assert_eq!(w.written()[0], 0xF1); // id 1 in the low 4 bits, the first 4 bits of the number above them
    }

    #[test]
    fn the_id_width_is_the_same_on_writing_reading_and_dispatching() {
        for bits in [4u32, 8, 16] {
            let mut reg = MessageRegistry::new(bits);
            reg.register::<Ping>().unwrap();
            reg.register::<Hit>().unwrap();
            let mut buf = [0u8; 32];
            let mut w = BitWriter::new(&mut buf);
            reg.write(
                &mut Hit {
                    target: 9,
                    damage: 10.0,
                    crit: true,
                },
                &mut w,
            )
            .unwrap();
            reg.write(&mut Ping { seq: 77 }, &mut w).unwrap();
            let mut r = BitReader::new(w.written());
            assert_eq!(
                reg.read(&mut r)
                    .unwrap()
                    .downcast_ref::<Hit>()
                    .unwrap()
                    .target,
                9,
                "{bits} bits"
            );
            assert_eq!(
                reg.read(&mut r)
                    .unwrap()
                    .downcast_ref::<Ping>()
                    .unwrap()
                    .seq,
                77,
                "{bits} bits"
            );

            // What write_message writes, a dispatcher of the same width reads, and a registry too.
            let mut d = Dispatcher::<Vec<u32>>::new(bits);
            d.on::<Ping>(|v, p| v.push(p.seq)).unwrap();
            let mut w = BitWriter::new(&mut buf);
            write_message(&mut Ping { seq: 1234 }, bits, &mut w).unwrap();
            assert_eq!(w.bits_written() as u32, bits + 32);
            let mut got = Vec::new();
            d.dispatch(&mut got, &mut BitReader::new(w.written()))
                .unwrap();
            assert_eq!(got, vec![1234], "{bits} bits");
            assert_eq!(
                reg.read(&mut BitReader::new(w.written()))
                    .unwrap()
                    .downcast_ref::<Ping>()
                    .unwrap()
                    .seq,
                1234
            );
        }
        let mut reg = MessageRegistry::new(16);
        reg.register::<TooBig>().unwrap(); // id 300 needs more than 8 bits
        let mut buf = [0u8; 8];
        let mut w = BitWriter::new(&mut buf);
        reg.write(&mut TooBig { a: 5 }, &mut w).unwrap();
        assert_eq!(
            reg.read(&mut BitReader::new(w.written())).unwrap().id(),
            300
        );
    }

    #[test]
    fn registering_and_sending_what_cannot_work_is_refused() {
        let mut reg = MessageRegistry::new(8);
        assert!(reg.is_empty());
        reg.register::<Ping>().unwrap();
        assert_eq!(reg.register::<Ping>(), Err(RegisterError::Duplicate));
        assert_eq!(reg.register::<TooBig>(), Err(RegisterError::IdTooLarge)); // 300 does not fit 8 bits
        assert_eq!(
            MessageRegistry::new(0).register::<Ping>(),
            Err(RegisterError::BadIdBits)
        );
        assert_eq!(
            MessageRegistry::new(17).register::<Ping>(),
            Err(RegisterError::BadIdBits)
        );
        assert_eq!(MessageRegistry::new(16).register::<TooBig>(), Ok(()));
        assert_eq!((reg.len(), reg.id_bits()), (1, 8));
        let mut buf = [0u8; 16];
        assert_eq!(
            reg.write(&mut Hit::default(), &mut BitWriter::new(&mut buf)),
            Err(BitError::UnknownTag),
            "not registered: the receiver could not read it"
        );
        let mut w = BitWriter::new(&mut buf);
        assert_eq!(w.bits_written(), 0);
        let _ = reg.write(&mut Hit::default(), &mut w);
        assert_eq!(w.bits_written(), 0, "and nothing was written");
        // The messages of a second registry are not understood by the first.
        assert_eq!(
            reg.read(&mut BitReader::new(&[7, 0])).err(),
            Some(BitError::UnknownTag)
        );
        assert_eq!(
            reg.read(&mut BitReader::new(&[])).err(),
            Some(BitError::Overflow)
        );
        assert_eq!(
            reg.read(&mut BitReader::new(&[1, 0, 0])).err(),
            Some(BitError::Overflow)
        ); // id 1 (Ping) but the u32 is cut short
    }

    // ---- typed handlers -------------------------------------------------------------------------------------------

    #[derive(Default)]
    struct World {
        hits: Vec<Hit>,
        chats: Vec<String>,
        pings: u32,
    }

    fn dispatcher() -> Dispatcher<World> {
        let mut d = Dispatcher::<World>::new(8);
        d.on::<Hit>(|w, h| w.hits.push(h)).unwrap();
        d.on::<Chat>(|w, c| w.chats.push(c.text)).unwrap();
        d.on::<Ping>(|w, p| w.pings += p.seq).unwrap();
        d
    }

    #[test]
    fn a_dispatcher_runs_the_handler_of_the_type_that_arrived() {
        let mut d = dispatcher();
        let mut world = World::default();
        let mut buf = [0u8; 64];
        let mut w = BitWriter::new(&mut buf);
        write_message(
            &mut Hit {
                target: 3,
                damage: 50.0,
                crit: true,
            },
            8,
            &mut w,
        )
        .unwrap();
        write_message(&mut Chat { text: "gg".into() }, 8, &mut w).unwrap();
        write_message(&mut Ping { seq: 40 }, 8, &mut w).unwrap();
        write_message(&mut Ping { seq: 2 }, 8, &mut w).unwrap();
        let mut r = BitReader::new(w.written());
        assert_eq!(d.dispatch(&mut world, &mut r), Ok(3));
        assert_eq!(d.dispatch(&mut world, &mut r), Ok(7));
        assert_eq!(d.dispatch(&mut world, &mut r), Ok(1));
        assert_eq!(d.dispatch(&mut world, &mut r), Ok(1));
        assert_eq!(
            (
                world.hits.len(),
                world.hits[0].target,
                world.chats.clone(),
                world.pings
            ),
            (1, 3, vec!["gg".to_string()], 42)
        );
    }

    #[test]
    fn a_handler_runs_only_for_a_message_that_was_read_in_full() {
        let mut d = dispatcher();
        let mut world = World::default();
        // An unknown id.
        assert_eq!(
            d.dispatch(&mut world, &mut BitReader::new(&[9, 1, 2, 3])),
            Err(BitError::UnknownTag)
        );
        // A Chat whose text claims more bytes than there are, and one whose text is over the limit, and one that is not UTF-8.
        assert!(d
            .dispatch(&mut world, &mut BitReader::new(&[7, 5, b'a']))
            .is_err());
        assert!(d
            .dispatch(&mut world, &mut BitReader::new(&[7, 40]))
            .is_err());
        assert!(d
            .dispatch(&mut world, &mut BitReader::new(&[7, 2, 0xC3, 0x28]))
            .is_err());
        // A Ping cut short.
        assert!(d
            .dispatch(&mut world, &mut BitReader::new(&[1, 0, 0]))
            .is_err());
        assert_eq!(
            (world.hits.len(), world.chats.len(), world.pings),
            (0, 0, 0),
            "no handler ran for any of them"
        );
        // Registering twice, or an id that does not fit.
        assert_eq!(d.on::<Ping>(|_, _| {}), Err(RegisterError::Duplicate));
        assert_eq!(d.on::<TooBig>(|_, _| {}), Err(RegisterError::IdTooLarge));
    }

    #[test]
    fn handlers_may_keep_their_own_state() {
        let seen = Rc::new(RefCell::new(Vec::new()));
        let mut d = Dispatcher::<()>::new(8);
        let log = Rc::clone(&seen);
        d.on::<Ping>(move |_, p| log.borrow_mut().push(p.seq))
            .unwrap();
        let mut buf = [0u8; 16];
        for seq in [5u32, 6] {
            let mut w = BitWriter::new(&mut buf);
            write_message(&mut Ping { seq }, 8, &mut w).unwrap();
            d.dispatch(&mut (), &mut BitReader::new(w.written()))
                .unwrap();
        }
        assert_eq!(*seen.borrow(), vec![5, 6]);
    }

    // ---- hostile input ----------------------------------------------------------------------------------------------

    #[test]
    fn random_bytes_into_every_reader_never_panic_and_never_run_a_handler_on_garbage() {
        let reg = registry();
        let mut d = dispatcher();
        let mut rng = Rng(0xFEED_FACE_1234_5678);
        let mut accepted = 0;
        for _ in 0..30000 {
            let len = rng.below(40) as usize;
            let bytes: Vec<u8> = (0..len).map(|_| rng.next() as u8).collect();
            let _ = net_read::<Command>(&bytes);
            if reg.read(&mut BitReader::new(&bytes)).is_ok() {
                accepted += 1;
            }
            let mut world = World::default();
            let ok = d.dispatch(&mut world, &mut BitReader::new(&bytes)).is_ok();
            let ran = world.hits.len() + world.chats.len() + (world.pings != 0) as usize;
            assert!(
                ran <= 1 && (!ok || ran == 1 || world.pings == 0),
                "a handler ran at most once, and only for a message that read"
            );
        }
        assert!(accepted > 0);
    }
}
