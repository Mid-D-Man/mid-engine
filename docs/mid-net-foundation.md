# mid-net foundation: from two packets to a game-ready netcode layer

**Status:** design + the first piece built (`bits.rs`, below). Written to be argued with: the decisions that need the owner are at the end.

## Where mid-net is, and what "game ready" means here
Today mid-net moves two hand-written packets (`PlayerState`, `PlayerEvent`) over a `Transport` (a datagram channel and reliable streams, over QUIC/WebTransport). The game that uses it first, Tower of Babylon (C99), has already built its own protocol in C on top of the byte transport: `Hello/Welcome/Input/Snapshot/Disconnect`, a 30 Hz server, client prediction with server reconciliation for movement, and now lag compensation. That work is the proof of what the layer must do; this document is about moving the general parts into mid-net so every game (the Rust engine's, and any C game over the FFI) gets them, instead of every game re-writing them.

"Game ready" is these seven things, in the order they are needed:

| | What | Today |
|---|---|---|
| 1 | Pack fields into bits (bool = 1 bit, health = 7, a position = 17 quantised): the budget is 7.8 ms and about a kilobyte per tick; and let a struct describe its own wire form once (Unity's `INetworkSerializable` / `INetworkSerializeByMemcpy`) | **done: `mid-net-wire::bits` and `::stream`** |
| 2 | Typed **messages** with an id, a channel and a codec; many per packet; a dispatcher | two fixed packets |
| 3 | **RPCs**: a message with a target (server, one client, the owner, all, all but one, a team), a reliability, an authority check, a rate limit, optionally a reply | none |
| 4 | **Network variables / replicated state**: entities with typed fields; the server sends only what changed since what each client last acknowledged; spawn and despawn; interest filtering | none (the game's C snapshot is hand-written) |
| 5 | **Prediction toolkit**: command history, replay from the server's answer, a fading correction; a **rewind buffer** for lag compensation | in the game's C only |
| 6 | **Session**: handshake with a schema check, keepalive, time sync, timeouts that can be told apart ("no signal" before "lost"), clear disconnect reasons | in the game's C only; a 5 s silence drops everyone |
| 7 | **Schema + codegen** from `.mdix`: one description, generated Rust and C with identical bytes, and a schema hash in the handshake so two builds that disagree refuse each other instead of misreading | hand-kept `packets/*.mdix` |

## The shape
Everything above the transport is built from the bottom up, each layer usable alone (the `wire` crate's rule: zero dependencies, no platform assumptions).

* **`mid-net-wire`** (exists): `Sequence`/`AckTracker`, the two packets, and now `bits` (see below).
* **message layer** (new, 2): `MsgId(u16)`, a `Channel` (`UnreliableSequenced`: state, stale ones dropped; `UnreliableUnordered`: cosmetic; `ReliableOrdered`: events; each channel its own sequence space, never one global one, as `mid-net.md` already decided), a `Message` trait (`ID`, `CHANNEL`, `encode(&self, &mut BitWriter)`, `decode(&mut BitReader)`), packets that batch messages up to the MTU, a dispatcher that is a table from id to handler. A backpressure rule on the reliable queue (a cap, and what happens past it: drop the connection, never grow without limit: Unreal's documented failure).
* **RPC layer** (3): an RPC is a message plus `Target`, `Delivery`, an `allow(sender, rpc) -> bool` hook the server runs before anything executes, a per-rpc and per-sender rate limit, and execution **queued to the start of the next tick** (never inline from the packet handler) so the order is the same on every machine. Request/response is a pair of messages with a correlation id and a timeout. A client can only call RPCs marked client-callable, and a server-to-client RPC that arrives at the server is dropped and counted.
* **replication layer** (4): `EntityId`, an entity *kind* (a list of fields: type, bits, range, interpolation mode, who may write it), per-client baselines (the last server tick the client acknowledged), a **dirty mask** per entity (a bit per field changed since the baseline) written ahead of only the changed fields, spawn and despawn on the reliable channel with the first full state, and an **interest filter** (a predicate per client: team, room, distance) deciding who gets which entity. The client keeps a short interpolation buffer per entity and fires an `on_change` per field. A baseline older than the server keeps (about 2 s) falls back to a full state.
* **prediction toolkit** (5): generic `CommandHistory<Cmd, State>` and `Reconciler` (re-run from the server's state with the unacknowledged commands, report the error, feed a fading offset) and `RewindBuffer<T>` (the last N ticks of poses, `at(tick)`, `at_f(tick)`, a clamp on how far back anyone may ask). Ported from the game's tested C (`net_client.c`, `net_history.c`), with the same tests.
* **session layer** (6): version + **schema hash** in the handshake, a keepalive so an idle link is told from a dead one, round-trip and server-tick estimation, a state machine `Connecting → Connected → NoSignal(since) → Lost(reason)` so a game can show "NO SIGNAL 3 s" and reconnect instead of dropping at the first 5 s gap.
* **codegen** (7): `tools/mdix-compiler` (the only place `dixscript` is allowed, by policy) reads `.mdix` schema files and writes plain Rust and plain C: message structs, their `encode/decode`, RPC send/dispatch stubs, replicated-entity structs with the dirty-mask delta codec, and the schema hash. The generated code is checked in, so the core crates stay free of `dixscript` and of reflection, and a C game compiles the same bytes with no Rust in its build.

## What exists now: `bits.rs` (layer 1)
`BitWriter` over a caller-owned slice (no allocation), `BitReader` over borrowed bytes. Fields: `bits(1..=64)`, `bool`, `u8/u16/u32/u64`, raw `f32`, `varint` (LEB128, bit-wise), `zigzag`, **quantised float** (`min..=max` in 1 to 32 bits, error at most half a step), **octahedral unit vector** (`2 * bits`, 2 to 16), byte-aligned `bytes`, `str` with a length limit. **The packing is a contract** other languages must reproduce: least significant bit first, golden vectors pinned in the tests (`write_bits(0b101,3)` then `write_bool(true)` is `0b0000_1101`; varint 300 is `AC 02`; a quantised 0.5 in `0..1` at 8 bits is `0x80`, half rounded away from zero). **No input can panic it** (20 000 random buffers with random reads in the tests); every write that does not fit, and every read past the end, is an `Err`, and a failed write changes nothing. NaN and infinity are refused rather than turned into a number; a finite value outside a range is clamped to it. **Tests:** 15 new (40 in the crate; the 15 include a 3000-round random mix of every kind of field, every width at every bit offset, and the hostile-input run), 16 mutations caught (packing order, continuation bit, zigzag, rounding, clamping, varint limits, octahedral fold, range checks, atomic writes and reads, length overflow, string limits, NaN).

## Structs and traits over the net (layer 1, second part: `stream.rs`, built)
Primitives are easy; the question is a struct (a `Hit`, a `PlayerState`, an inventory) and a trait. Unity answers with two interfaces; this layer answers with the same two and one more, in the engine's usual shape (declarative macros, no procedural macro, no dependency, Rust 1.75):

| Unity netcode | here | what it is |
|---|---|---|
| `INetworkSerializable` + `BufferSerializer<T>` | **`NetSerialize` + `BitStream`** | a type writes **one** function that visits its fields through a stream that is either writing or reading, so the writer and the reader are the same code and a field cannot be written in one order and read in another |
| `INetworkSerializeByMemcpy` | **`NetPod`** + `write_pod`/`read_pod` | `#[repr(C)]` plain data copied as its bytes (little-endian only); `impl_net_pod!` fails to compile if the struct has padding (a doc test shows it); bigger than the packed form, no description needed |
| `NetworkVariable<T>` | layer 4 (replication) | later |
| a polymorphic message / `INetworkMessage` | an `enum` (a tag, then the variant), or a message id with a decoder table (layer 2) | **a trait object cannot cross a network**: the other end has no idea what it is; both honest ways are built from `NetSerialize` |

`impl_net_serialize!(Hit { target: bits(10), damage: quant(0.0, 500.0, 12), dir: unit(12), crit: flag(), note: string(16) })` writes the function from a field list; the field kinds are `flag`, `bits(n)`, `sbits(n)`, `varint`, `quant(min,max,bits)`, `unit(bits)`, `string(max)`, `list(max)`, and `raw` (a full-width number, a nested struct, an array, an `Option`). Lists and strings carry a limit the caller chooses, checked on both sides, and **reading a list never allocates more than the packet can hold** (a claimed length bigger than the bits left is refused before anything is allocated: a 5-byte packet cannot ask for 4 GB). `net_round_trip` makes a value exactly what the other end will read (quantised fields lose their extra precision), which is what a sender needs to run its own simulation from the rounded value, as the game's prediction does with its commands.
**Tests:** 13 unit tests and 3 doc tests (one is the `compile_fail` for padding): one description writing and reading a struct with every kind of field, pinned bytes, reading over a dirty struct, signed fields at their edges, widths and varints that do not fit the type, list and string limits, the allocation claim, `Option`, the round trip, plain data (little-endian bytes, unaligned start, every bit pattern valid, short packets), and 30 000 random packets that must neither panic nor allocate wildly; 14 mutations all caught (13 by tests, 1 by the compiler). **Compiled and tested on Rust 1.75 (the workspace floor) and on current stable; clippy is clean.**

## Independence, and what mid-math's FFI teaches
mid-net must stand on its own: it serves this engine and any other game, in any language. So the core crates keep their rule (zero dependencies, no platform assumptions, nothing from the engine), and anything that joins mid-net to another engine crate is a **bridge** the user opts into, as `mid-reflect` already does with its `math` feature and `mid-math` does with `mint` and `ffi`. Neither side depends on the other.
`mid-math`'s C ABI (`ffi/`, off by default, 1 300+ `extern "C"` functions) is the pattern to follow, and it shows three things that carry over:
* **Small plain data crosses as a `#[repr(C)]` value struct** (`CVec3`, `CQuat`), and **compressed storage types cross as their raw bits**: `f16`/`bf16` as `u16`, `F8Norm` and the 8-bit floats as `u8`, `BitMask8..64` as the integer, `BitMask128/256` as `[u64; 2]` / `[u64; 4]`. Those storage types are exactly what a network field is ("store compressed, unpack only to compute"): `f16`/`bf16` for velocities and angles, `F8Norm` for 0..1 values, the deterministic `Fixed8/12/16` for positions, and **`BitMask`s are the natural dirty masks and interest sets** of the replication layer.
* **Arrays cross as pointer + length** (its `wide_batch` functions): the same shape as a packet buffer.
* **Stateful things need an opaque handle** (mid-math has none because it is stateless; mid-net's QUIC FFI already uses handles for connections), and everything a C caller must free has one named free function.
So the plan is: a small `mid-net-math` **bridge crate** (depends on both) that implements `NetSerialize` for the mid-math types that make sense (vectors, quaternions, the storage types, bit masks), so the Rust engine writes `pos: raw()` and gets a quantised or packed form; mid-net itself defines the half-float and fixed-point wire forms only as IEEE-754 binary16 bits and plain integers, which any language can read without mid-math.

## What a C (or any other) game uses
The contract is **data, not Rust**: a *schema* (messages, RPCs, entity kinds, with every field's kind, bits and range) written once in `.mdix`. From it the generator writes: (a) plain Rust and plain C source with the typed structs and their encode/decode, byte-identical by the packing contract in `bits.rs`, and (b) the same schema as a small **descriptor table** (an array of field descriptors: kind, bits, min, max) plus its **hash**, which the handshake compares so two builds that disagree refuse each other. A game in another language can read the descriptor table with a few hundred lines of its own, or call the Rust interpreter through the C ABI; nothing requires Rust at run time. That settles decision 1 below.

## How this meets Tower of Babylon (the use cases that drive the API)
The game's next features are exactly the layers above, so they are the first customers and decide the shape:
* **Pick up, drop, swap, use an item** (players start with only hands; items are carried and dropped by everyone) = **RPCs** from client to server (`PickUp(item)`, `Drop(slot)`, `Use(slot)`), with an authority check (reach, ownership, is it still there) and a reply or a correction.
* **An item lying on the floor, a monster, a door** = **replicated entities**: spawned by the server with their full state, then only their changes.
* **Health, ammo, the held slot, the team** = **network variables**: owner-only or everyone, quantised (health in 7 bits, ammo in 6), with `on_change` to drive the HUD.
* **Teams** (players hurt each other except teammates) = a field on the player entity, an **interest and target filter** (`ToTeam`), and a rule in the server's damage code; mid-net provides the group and the filter, the game decides what friendly fire means.
* **Shooting** = the fire bit in the command, the seed derived on both ends, the rewind buffer for the hit check (`net_shots.md` in the game repo).

## Order of work (each step ends with tests, mutations, and a drop the owner can apply)
1. `bits` (applied) and `stream` (**this drop**).
2. Messages + channels + the dispatcher + the reliable-queue backpressure, with a loopback harness that can lose, reorder, duplicate and delay (the existing `LoopbackTransport` plus an impairment wrapper).
3. Session layer: keepalive, NO SIGNAL, schema hash, time sync (this also fixes the drop-after-a-while the owner saw on two phones).
4. RPCs.
5. Replication: entities, dirty masks, baselines, interest, spawn/despawn, interpolation; a soak test (an hour of simulated play with loss and jitter, as the game's net layer already has).
6. The prediction toolkit and rewind buffer, ported with their tests.
7. `mdix-compiler` codegen for Rust and C; the C ABI; then the game moves its messages onto it one at a time (items and teams first, movement last, because movement already works).

## Decisions for the owner
1. **Settled by the independence rule, pending the owner's word:** the contract is the **schema** (`.mdix`) and its **descriptor table + hash**; generated Rust and C is the default path (no registry, no reflection, a C game links only the transport and compiles the same bytes); the Rust interpreter behind the C ABI is an optional convenience for a language that would rather not generate code. Nothing at run time requires Rust.
2. **Tick rate:** the doc's budget says 128 Hz, the game runs 30 Hz. The layer should be tick-agnostic (the tick is a number the caller advances); is 128 Hz a real target for the Rust engine or a ceiling?
3. **ECS:** should replication in the Rust engine be defined on `mid-ecs` components (the schema's entity kind maps to a component set) from the start, or stay a plain entity table the ECS adapts to later?
4. **Teams:** a generic `Group` (id, members) in mid-net with filters on it, and teams as one use of it (recommended), or teams only in the game?
5. **Authority model:** server-authoritative for everything except what an owner predicts (their own movement, their own held item), as in the game now; is there any case you want client-authority (cosmetic physics, a car)?
