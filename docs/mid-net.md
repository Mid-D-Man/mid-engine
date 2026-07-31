# mid-net

Reliable UDP with DixScript (.mdix) packet definitions.

**Status:** in progress (build order: math → common → geom → **net** → ecs → physics). `packet.rs` (hand-rolled codec, `PlayerState`/`PlayerEvent`, 12 tests) and `sequence.rs` (wraparound-aware sequence comparison + ack-bitfield tracking, 13 tests including the gafferongames.com reference scenario) have real implementations, 25 tests passing total. `socket.rs`, `reliable.rs`, `ffi.rs` are still skeletons — `reliable.rs` is next: retransmit buffer, RTT-based timeout, and framing `packet.rs` payloads with `sequence.rs`'s sequence numbers.

## Dependency philosophy

Same mandate as mid-math: **zero external dependencies where at all possible, minimal where not, works on every target with no platform-specific runtime requirement.** No assumption of a particular OS network stack beyond standard UDP sockets — no io_uring-only paths, no eBPF/XDP, nothing that only runs on Linux. This rules out reaching for existing Rust netcode crates (laminar, renet, etc.) as dependencies; they're useful as *reference reading*, not as things to pull in.

**Resolved:** "wire encoder uses bincode" meant the literal `bincode` crate (confirmed against the actual stub doc comments in `lib.rs`/`packet.rs`, which said so explicitly). That contradicted the zero-dependency mandate, so it's replaced with a hand-rolled encoder — explicit little-endian, fixed layout for `PlayerState`, length-prefixed fields for `PlayerEvent`'s strings. `bincode` is removed from `Cargo.toml`. Same spirit as mid-math's SIMD work: hand-rolled over dependency, in this case with an extra motivation — Ubel Stratum's LOW tier (manual memory, FFI) is a plausible future consumer of these bytes, and a Rust-only reflection-based format like bincode gives a non-Rust caller nothing to bind against, where a flat byte layout does.

`tokio`/`bytes` in `Cargo.toml` are a **separate, still-open** question — async runtime and buffer-type choice for `socket.rs`, not resolved by the wire-encoding decision above. Revisit before building `socket.rs`.

## Why UDP

TCP head-of-line blocking stalls all packets when one is lost.
For position updates at 128 Hz, a dropped packet is just stale — skip it.
UDP delivers the next packet immediately.

## Two Channels

| Channel | Content | Loss behaviour |
|---|---|---|
| Unreliable | position, rotation, animation | drop freely |
| Reliable | join, pickup, damage, events | ACK + retransmit |

### Validated against Unity and Unreal (checked their actual docs/source, not just recalled)

Both engines converge on exactly this split, which is a good sign this is the right shape:

- **Unreal**: RPCs are Server/Client/NetMulticast, each either Reliable or Unreliable. Their own guidance: unreliable for anything called every tick or non-critical (movement, cosmetic effects); reliable for infrequent-but-critical (spawning, state changes). Explicit warning: overusing Reliable can overflow its queue and force a disconnect — worth having an explicit cap/backpressure policy on the reliable channel rather than an unbounded resend queue.
- **Unity (Netcode for GameObjects)**: same split via `NetworkDelivery`/QoS channels (Reliable vs Unreliable), configurable per message. Same guidance: high-frequency (multiple-times-a-second) → unreliable.
- **Both**: reliable delivery ordering is guaranteed **per-object/per-channel, not globally.** Don't build a single global sequence number across all reliable traffic — that reintroduces TCP-style head-of-line blocking between unrelated entities, which is the exact problem UDP was chosen to avoid. Keep reliable and unreliable streams on independent sequence spaces (confirmed against the general reliable-UDP literature too — mixing them reintroduces the stall you're trying to avoid, since a lost reliable packet's resend-wait shouldn't block delivery of unrelated unreliable packets that were sent after it).

### Reliability mechanism (the concrete "ACK + retransmit")

Standard pattern used across the reliable-UDP-for-games space (this is the same approach behind e.g. Glenn Fiedler's `reliable` library, cited widely as the reference design):

- Each reliable packet carries a monotonically increasing sequence number.
- Receiver tracks the highest sequence number seen plus a bitfield of the last N (typically 32) packets received, and sends that back as the ack.
- Sender keeps unacked packets in a small buffer, resends on timeout (RTT-based, not fixed) or when a gap is confirmed via the ack bitfield.
- Sequence numbers wrap — comparisons need wraparound-aware logic (`(a - b) as i16 > 0` style), not raw `>`.
- Keep this as its own small, self-contained module — it's the one part of mid-net worth writing tests against known packet-loss sequences early, since a subtle wraparound or off-by-one bug here silently corrupts delivery guarantees rather than crashing.

## DixScript Integration

Packet shapes are defined in `.mdix` files under `packets/`. `dixscript`
1.0.0 published to crates.io 2026-07-27 — checked its actual Rust API
and Cargo.toml (not assumed) before deciding whether mid-net should
depend on it:

- **Not a fit for the wire format.** Its Rust API is a dynamic accessor
  over parsed `.mdix` (`data.get::<T>("path")`), not struct codegen —
  there's no DixScript step `packet.rs`'s hand-written types are
  standing in for waiting to be automated away. It also ships a real
  binary Packer/Unpacker (the "DLM pipeline"), but that's a
  general-purpose encode path for arbitrary DixScript ASTs — encryption
  and compression built in, self-describing — not tuned for a fixed
  28-byte per-tick struct. Wrong shape for the 128 Hz hot path.
- **Not added as a dependency, mandatory or not.** Even with
  `default-features = false`, `dixscript` pulls in 23 mandatory
  transitive crates (serde, regex, chrono, aes-gcm, chacha20poly1305,
  argon2, uuid, phf, …) — a reasonable budget for a general config/data
  format, but it directly conflicts with mid-net's own zero-dependency
  mandate above. Not reaching for it here for the same reason `bincode`
  got removed.
- **Where it might still fit, later, deliberately:** authoring/
  validating the `.mdix` schema files themselves at dev time (e.g. via
  `mdix-cli`, build-from-source only right now, not on crates.io yet),
  or as a save-file/non-hot-path format elsewhere in the engine where
  its encryption and compression are actually wanted. Not a mid-net
  runtime dependency either way.

`packet.rs` stays the fast path for in-flight bytes; the `.mdix` files
stay the human-authored source of truth for packet shape, kept in sync
by hand for now.

## Packet Budget

7.8 ms per tick at 128 Hz. Design entity delta budgets early.
This constrains everything else in the networking system.

## Sibling project check

`midn` (the LTE/5G core project) was checked for reusable code — nothing transfers. It's a 3GPP telecom stack (S1AP/NGAP/NAS signaling, GTP-U tunneling, Milenage/TUAK SIM auth), a different protocol domain with different constraints, and its userplane layer is Linux-eBPF/XDP-specific, which conflicts with mid-net's cross-platform requirement. Not worth revisiting for mid-net specifically. (`midn-ecs` is worth a glance later, for mid-ecs — see that crate's notes.)
