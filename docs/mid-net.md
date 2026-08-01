# mid-net

Reliable UDP netcode, hand-rolled wire format. Packet shapes are
documented as `.mdix` reference schema, but DixScript itself is not a
dependency of this crate — see "Dependency philosophy" below and
`docs/architecture.md`.

**Status:** in progress (build order: math → common → geom → **net** → ecs → physics). `packet.rs` (hand-rolled codec, `PlayerState`/`PlayerEvent`, 12 tests), `sequence.rs` (wraparound-aware sequence/ack arithmetic, 13 tests including the gafferongames.com reference scenario), and `reliable.rs` (frame headers, RTT estimator, retransmit buffer, 17 tests) have real implementations — 42 tests passing total. Not yet built: a single "connection" object composing `AckTracker` + `RetransmitBuffer` + frame (de)coding into one `send`/`poll_received`/`update` API — right now these are tested building blocks, not glued into one type yet. `socket.rs` (actual transport) and `ffi.rs` are still skeletons.

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

Standard pattern used across the reliable-UDP-for-games space (this is the same approach behind e.g. Glenn Fiedler's `reliable` library, cited widely as the reference design). Implemented in `sequence.rs` + `reliable.rs`, tested against the reference design's own worked example:

- Each reliable packet carries a monotonically increasing sequence number (`sequence.rs::Sequence`).
- Receiver tracks the highest sequence number seen plus a 32-bit bitfield of what came before it, and sends that back as the ack (`sequence.rs::AckTracker`) — piggybacked on the header of every outgoing reliable packet, not sent as separate ack packets.
- Sender keeps unacked packets in a small buffer (`reliable.rs::RetransmitBuffer`), resends on an RTT-based timeout (`RttEstimator`, same smoothing constants as TCP's classic Jacobson/Karels RTO estimator) rather than a fixed guess.
- **Karn's algorithm applied:** a packet's RTT is only sampled the first time it's acked with zero retransmits behind it. Once a packet's been resent, an ack for it is ambiguous — no way to tell which transmission it's acknowledging — so counting it would poison the RTT estimate. Tested explicitly (`retransmitted_packet_does_not_pollute_rtt_sample`).
- Sequence numbers wrap — comparisons need wraparound-aware logic, not raw `>`. `Sequence` deliberately has no `PartialOrd`/`Ord` impl so `<`/`>` can't be reached for by habit; `is_more_recent_than` is explicit about it.

## Platform & FFI

Two hard constraints, both load-bearing in `reliable.rs`'s design, not just aspirational:

- **No `std::net`, no `std::time::Instant`, anywhere in `packet.rs`/`sequence.rs`/`reliable.rs`.** `std::net::UdpSocket` doesn't exist on `wasm32-unknown-unknown` — checked current browser transport options rather than assuming: WebTransport datagrams (baseline-available across browsers as of March 2026, UDP-like: unreliable, unordered, MTU-sized) are the client-server fit; WebRTC `RTCDataChannel` unreliable mode is the P2P equivalent but needs ICE/STUN/TURN for client-server, more complexity than this needs. Either way, both present the same shape as UDP — a byte buffer in, a byte buffer out, no delivery guarantee — which is exactly the interface `reliable.rs` and `packet.rs` are written against. `socket.rs` is where the actual per-platform transport gets picked (`cfg`-gated); nothing above it needs to change when that happens.
- **Time is a caller-supplied `Timestamp(u64)` (milliseconds), never queried internally.** `std::time::Instant` has no defined layout and isn't meaningful across a C ABI, and it panics on `wasm32-unknown-unknown` without a shim. Pushing "what does now mean" up to the caller sidesteps both problems at once, and as a side effect makes every `reliable.rs` test deterministic — a manually-advanced fake clock, no real sleeping, no timing flakiness.

## DixScript Integration

Packet shapes are documented as `.mdix` reference schema under
`packets/`, but this is **not** a mid-net dependency — decided this pass
and then generalized into a repo-wide policy (see
`docs/architecture.md`, "Technical Mandates"): `dixscript` is not a
dependency of any core crate, not just mid-net, not just "for now."

- **Not a fit for the wire format even setting the dependency question
  aside.** Checked its actual Rust API: it's a dynamic accessor over
  parsed `.mdix` (`data.get::<T>("path")`), not struct codegen — there
  was never a DixScript step `packet.rs`'s hand-written types were
  standing in for. It also ships a real binary Packer/Unpacker (the
  "DLM pipeline"), but that's a general-purpose encode path for
  arbitrary DixScript ASTs — encryption and compression built in,
  self-describing — not tuned for a fixed 28-byte per-tick struct.
- **Not added as a dependency, mandatory or not, by policy now — not a
  case-by-case call.** Even with `default-features = false`, `dixscript`
  pulls in 23 mandatory transitive crates (serde, regex, chrono,
  aes-gcm, chacha20poly1305, argon2, uuid, phf, …). Right budget for a
  general config/data format, wrong one for a core systems crate.
- **Where it actually lives:** `tools/mdix-compiler` — a separate
  binary, not linked into any core crate — now depends on `dixscript`
  for real, since compiling `.mdix` files is exactly its job. That's
  the concrete instance of "DixScript as the engine's convenient data
  format" the policy above points to. mid-net's own `.mdix` files stay
  human-authored reference schema only.

`packet.rs` stays the fast path for in-flight bytes; the `.mdix` files
stay the human-authored source of truth for packet shape, kept in sync
by hand for now.

## Packet Budget

7.8 ms per tick at 128 Hz. Design entity delta budgets early.
This constrains everything else in the networking system.

## Sibling project check

`midn` (the LTE/5G core project) was checked for reusable code — nothing transfers. It's a 3GPP telecom stack (S1AP/NGAP/NAS signaling, GTP-U tunneling, Milenage/TUAK SIM auth), a different protocol domain with different constraints, and its userplane layer is Linux-eBPF/XDP-specific, which conflicts with mid-net's cross-platform requirement. Not worth revisiting for mid-net specifically. (`midn-ecs` is worth a glance later, for mid-ecs — see that crate's notes.)
